"""Collect full MPM states, train a fixed-topology ROM, and evaluate free rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def collect_demo(args):
    from experiments.differentiable_mpm.runtime import init_runtime
    from experiments.differentiable_mpm.reduced_order_data import collect_trajectory
    from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, ParticleState, SimulationConfig, SDFData, ToolControl
    init_runtime("cpu", "f64")
    config = SimulationConfig(n_particles=27, grid=12, dt=2e-4, precision="f64",
                              particle_mass=0.001, particle_volume=1e-6, gravity=-2,
                              plasticity="stretch-clamp", use_jp=True, tool_collision="sdf",
                              tool_contact_padding=0.004, p2g_mode="serial")
    axis = np.linspace(.40, .46, 3)
    reference = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    grid_axis = np.linspace(-.18, .18, 13)
    coordinates = np.stack(np.meshgrid(grid_axis, grid_axis, grid_axis, indexing="ij"), -1)
    radius = np.linalg.norm(coordinates, axis=-1)
    sdf = SDFData(np.stack([radius - .065] * 2),
                  np.stack([coordinates / np.maximum(radius[..., None], 1e-12)] * 2),
                  np.full((2, 3), -.18), np.full((2, 3), .03))
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(args.trajectories):
        state = ParticleState.initial(reference, np.float64)
        state.v[:] = rng.normal(0, .02, (1, 3))
        state.C[:] = np.diag(rng.uniform(-.15, .15, 3))
        state.F[:] = np.diag(rng.uniform(.96, 1.04, 3))
        parameters = dict(DEFAULT_PARAMETERS, youngs_modulus=4200.0, viscosity=8.0)
        controls = []
        linear = rng.uniform(-.15, .15, (2, 3))
        origins = np.array([[.43, .40, .36], [.43, .43, .52]])
        origins += rng.uniform(-.006, .006, origins.shape)
        for k in range(args.steps):
            control = ToolControl.stationary(k * config.dt)
            control.poses[:, :3] = origins + linear * control.time
            control.velocities[:, :3] = linear
            controls.append(control)
        trajectory = collect_trajectory(config, state, parameters, controls, sdf,
                                        trajectory_id=f"contact-{i:03d}", reference_x=reference)
        trajectory.save(args.output / f"contact-{i:03d}.npz")
        print(f"Collected {trajectory.trajectory_id}: {trajectory.n_steps} steps", flush=True)


def collect_config(args):
    from experiments.differentiable_mpm.config import load_config
    from experiments.differentiable_mpm.data import prepare_experiment
    from experiments.differentiable_mpm.runtime import init_runtime
    from experiments.differentiable_mpm.reduced_order_data import collect_trajectory
    experiment = prepare_experiment(load_config(args.config))
    init_runtime(args.backend, experiment.simulation_config.precision)
    count = min(args.steps, experiment.total_steps) if args.steps else experiment.total_steps
    trajectory = collect_trajectory(experiment.simulation_config, experiment.initial_state,
                                    experiment.parameters, [experiment.controls[k] for k in range(count)],
                                    experiment.sdf, trajectory_id=args.trajectory_id)
    trajectory.metadata['experiment_fingerprint'] = experiment.fingerprint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save(args.output)
    print(f"Collected {count} steps to {args.output}")


def fit(args):
    import torch
    from experiments.differentiable_mpm.reduced_order_data import Trajectory
    from experiments.differentiable_mpm.train_reduced_order import split_trajectories, train, evaluate
    torch.set_num_threads(args.threads)
    trajectories = [Trajectory.load(path) for path in sorted(args.data.glob('*.npz'))]
    training, validation, test = split_trajectories(trajectories, args.seed)
    model, report = train(training, validation, rank=args.rank, hidden_dim=args.width,
                          epochs=args.epochs, horizon=args.horizon, seed=args.seed)
    report.update({"test_ids": [t.trajectory_id for t in test], "test": evaluate(model, test),
                   "latent_dim": model.codec.latent_dim,
                   "full_state_dim": 25 * model.codec.n_particles,
                   "rank_per_field": model.codec.ranks, "seed": args.seed})
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / 'model.pt')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in report.items() if key != 'history'}, indent=2))


def evaluate_file(args):
    import torch
    from experiments.differentiable_mpm.reduced_order import ReducedDynamics
    from experiments.differentiable_mpm.reduced_order_data import Trajectory
    from experiments.differentiable_mpm.train_reduced_order import evaluate
    torch.set_num_threads(args.threads)
    model = ReducedDynamics.load(args.model)
    report = evaluate(model, [Trajectory.load(path) for path in args.trajectories])
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('collect-demo', help='Tiny synthetic two-sphere contact trajectories')
    demo.add_argument('--output', type=Path, required=True)
    demo.add_argument('--trajectories', type=int, default=10)
    demo.add_argument('--steps', type=int, default=48)
    demo.add_argument('--seed', type=int, default=7)
    demo.set_defaults(run=collect_demo)
    real = commands.add_parser('collect', help='Replay an existing calibrated experiment')
    real.add_argument('--config', type=Path, required=True)
    real.add_argument('--output', type=Path, required=True)
    real.add_argument('--trajectory-id', required=True)
    real.add_argument('--steps', type=int)
    real.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    real.set_defaults(run=collect_config)
    training = commands.add_parser('train')
    training.add_argument('--data', type=Path, required=True)
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--rank', type=int, default=8)
    training.add_argument('--width', type=int, default=128)
    training.add_argument('--epochs', type=int, default=100)
    training.add_argument('--horizon', type=int, default=8)
    training.add_argument('--seed', type=int, default=7)
    training.add_argument('--threads', type=int, default=1)
    training.set_defaults(run=fit)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--model', type=Path, required=True)
    evaluation.add_argument('--trajectories', type=Path, nargs='+', required=True)
    evaluation.add_argument('--output', type=Path)
    evaluation.add_argument('--threads', type=int, default=1)
    evaluation.set_defaults(run=evaluate_file)
    args = parser.parse_args()
    for name in ('steps', 'trajectories', 'rank', 'width', 'epochs', 'horizon', 'threads'):
        value = getattr(args, name, None)
        if isinstance(value, int) and value < 1:
            parser.error(f'--{name} must be positive')
    args.run(args)


if __name__ == '__main__':
    main()

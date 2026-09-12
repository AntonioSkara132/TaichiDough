"""Small same-model inverse problem using observed visible geometry, not particle IDs.

Run with ``python -m experiments.differentiable_mpm.synthetic --backend cpu
--precision f64 --iterations 12 --output-dir experiments/differentiable_mpm/runs/synthetic``.
The known initial deformation/velocity are prescribed excitations, not parameters
inferred from the observations. This checks differentiation and optimization; it
is not independent evidence that the material model fits real dough.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .checkpoint import CheckpointedRollout
from .loss import LOSS_VERSION, SUPPORTED_LOSS_VERSIONS, LossConfig, Observation, ObservationLoss
from .optimize import AdamOptions, ObjectiveValue, ProjectedAdam
from .parameters import PhysicalParameterSpace
from .renderer import Camera
from .results import RUN_ROOT, RunStore, canonical_hash, source_identity
from .runtime import init_runtime
from .solver import Stepper
from .state import DEFAULT_PARAMETERS, PHYSICS_VERSIONS, ParticleState, SimulationConfig, ToolControl


SYNTHETIC_VERSION = "prestrained-visible-patch-v1"
PREDICTION_TEMPERATURE_M = 0.01
TARGET_TEMPERATURE_M = 0.002
FIT_PARAMETERS = ("youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max")
TRUTH = {**DEFAULT_PARAMETERS, "youngs_modulus": 12000.0, "poisson_ratio": 0.28,
         "viscosity": 12.0, "plastic_min": 0.88, "plastic_max": 1.12}
INITIAL_PARAMETERS = {**DEFAULT_PARAMETERS, "youngs_modulus": 18000.0, "poisson_ratio": 0.22,
                      "viscosity": 5.0, "plastic_min": 0.93, "plastic_max": 1.05}


@dataclass(frozen=True)
class SyntheticConfig:
    steps: int = 32
    segment_length: int = 8
    particles_per_axis: int = 4
    grid: int = 16
    dt: float = 0.001
    precision: str = "f64"
    seed: int = 7
    observation_count: int = 4
    loss_version: str = LOSS_VERSION
    physics_version: str = "corrected-v1"
    visibility_temperature_m: float = PREDICTION_TEMPERATURE_M
    target_visibility_temperature_m: float = TARGET_TEMPERATURE_M

    def __post_init__(self):
        if self.physics_version not in PHYSICS_VERSIONS:
            raise ValueError("physics_version must be corrected-v1 or legacy-v1")
        if (not np.isfinite([self.visibility_temperature_m, self.target_visibility_temperature_m]).all()
                or min(self.visibility_temperature_m, self.target_visibility_temperature_m) <= 0):
            raise ValueError("Prediction and target visibility temperatures must be finite and positive")
        if self.loss_version not in SUPPORTED_LOSS_VERSIONS:
            raise ValueError("Unknown synthetic observation loss version")
        if type(self.steps) is not int or self.steps < 2 or self.steps > 256:
            raise ValueError("Synthetic steps must be an integer in [2,256]")
        if type(self.segment_length) is not int or not 1 <= self.segment_length <= 64:
            raise ValueError("Synthetic segment_length must be an integer in [1,64]")
        if type(self.particles_per_axis) is not int or not 3 <= self.particles_per_axis <= 6:
            raise ValueError("particles_per_axis must be an integer in [3,6]")
        if self.grid not in {12, 16}:
            raise ValueError("This small synthetic example supports grid 12 or 16")
        if not np.isfinite(self.dt) or not 0 < self.dt <= 0.002:
            raise ValueError("Synthetic dt must be positive and at most 0.002 s")
        if self.precision not in {"f32", "f64"}:
            raise ValueError("precision must be f32 or f64")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.observation_count) is not int or not 1 <= self.observation_count <= self.steps:
            raise ValueError("Require observation_count between one and steps")

    @property
    def observation_steps(self):
        return tuple(np.linspace(1, self.steps, self.observation_count + 1, dtype=int)[1:].tolist())


def array_hash(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def synthetic_camera():
    transform = np.eye(4)
    transform[:3, 3] = [-0.5, -0.5, 0.5]
    return Camera(64, 56, 160.0, 160.0, 32.0, 28.0, transform)


def synthetic_loss_config(version=LOSS_VERSION, visibility_temperature_m=PREDICTION_TEMPERATURE_M):
    return LossConfig(version=version, footprint_radius=2, visibility_temperature_m=visibility_temperature_m,
                      opacity_gain=12.0, depth_scale_m=0.0005, distance_scale_m=0.002,
                      depth_weight=1.0, coverage_weight=0.03, distance_weight=0.5,
                      max_observed_points=512)


def make_initial_state(config, heldout=False):
    dtype = np.float64 if config.precision == "f64" else np.float32
    axis = np.linspace(-0.075, 0.075, config.particles_per_axis)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    # Small deterministic offsets avoid stencil and nearest-association ties.
    rng = np.random.default_rng(config.seed)
    offsets += rng.uniform(-0.0003, 0.0003, offsets.shape)
    state = ParticleState.initial(offsets + 0.5, dtype=dtype)
    velocity_gradient = np.array([[1.5, 1.8, 0.2], [0.4, -2.0, 1.0], [0.1, -0.6, 0.5]])
    angle = 0.57 if heldout else 0.17
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0],
                         [-np.sin(angle), 0.0, np.cos(angle)]])
    if heldout:
        velocity_gradient = np.array([[-1.0, -0.8, 2.1], [1.7, 1.2, -0.4], [-0.6, 0.3, -0.7]])
    for p, offset in enumerate(offsets):
        stretches = np.array([0.79 + 0.1 * offset[1], 1.23 + 0.1 * offset[0], 0.98 + 0.05 * offset[2]])
        state.F[p] = rotation @ np.diag(stretches) @ rotation.T
        state.C[p] = velocity_gradient
    state.v[:] = offsets @ velocity_gradient.T
    state.validate()
    return state


def simulation_config(config):
    n = config.particles_per_axis ** 3
    spacing = 0.15 / (config.particles_per_axis - 1)
    particle_volume = spacing ** 3
    return SimulationConfig(n_particles=n, grid=config.grid, dt=config.dt,
                            particle_mass=800.0 * particle_volume, particle_volume=particle_volume,
                            gravity=0.0, floor_y=0.0, plasticity="stretch-clamp", use_jp=False,
                            tool_collision="none", tool_contact_padding=0.0, precision=config.precision,
                            physics_version=config.physics_version)


def parameter_space():
    return PhysicalParameterSpace(INITIAL_PARAMETERS, FIT_PARAMETERS, "stretch-clamp",
                                  bounds={"youngs_modulus": (4000.0, 40000.0), "poisson_ratio": (0.1, 0.4),
                                          "viscosity": (0.0, 80.0), "plastic_min": (0.75, 0.98),
                                          "plastic_max": (1.02, 1.3)},
                                  scales={"viscosity": 20.0, "plastic_min": 0.1, "plastic_max": 0.1,
                                          "poisson_ratio": 0.1})


def excitation_diagnostics(state, config):
    trial = (np.eye(3)[None] + config.dt * state.C) @ state.F
    singular_values = np.linalg.svd(trial.astype(np.float64), compute_uv=False)
    return {"initial_trial_min_singular_value": float(singular_values.min()),
            "initial_trial_max_singular_value": float(singular_values.max()),
            "truth_lower_active_particles": int(np.any(singular_values < TRUTH["plastic_min"], axis=1).sum()),
            "truth_upper_active_particles": int(np.any(singular_values > TRUTH["plastic_max"], axis=1).sum()),
            "initial_lower_active_particles": int(np.any(singular_values < INITIAL_PARAMETERS["plastic_min"], axis=1).sum()),
            "initial_upper_active_particles": int(np.any(singular_values > INITIAL_PARAMETERS["plastic_max"], axis=1).sum())}


def declared_identity(config, runtime):
    training = make_initial_state(config)
    heldout = make_initial_state(config, heldout=True)
    return {"kind": SYNTHETIC_VERSION, "settings": asdict(config), "runtime": runtime,
            "source": source_identity(), "simulation": asdict(simulation_config(config)),
            "camera": synthetic_camera().as_dict(), "loss": synthetic_loss_config(config.loss_version, config.visibility_temperature_m).as_dict(),
            "target_loss": synthetic_loss_config(config.loss_version, config.target_visibility_temperature_m).as_dict(),
            "truth": TRUTH, "parameter_space": parameter_space().settings(),
            "training_initial_state": {name: array_hash(array) for name, array in training.arrays().items()},
            "heldout_initial_state": {name: array_hash(array) for name, array in heldout.arrays().items()},
            "observations_at_steps": list(config.observation_steps),
            "target_generation": "same-model forward MPM; visible depth, foreground and optical exterior points only",
            "heldout_selection": "unused excitation evaluated only after training candidate is frozen"}


def generate_observations(stepper, initial_state, controls, observation_steps, objective, seed):
    """Forward truth, then discard particle correspondence from target observations."""
    stepper.set_parameters(TRUTH)
    stepper.load_state(0, initial_state)
    observations = []
    slot = 0
    rng = np.random.default_rng(seed)
    camera = objective.camera
    initial = objective.initial_render
    initial_mask = initial.valid.copy()
    initial_mask &= rng.random(initial_mask.shape) > 0.07
    for completed, control in enumerate(controls, start=1):
        stepper.advance(slot, control)
        slot += 1
        if completed in observation_steps:
            rendered = objective.renderer.forward(stepper.state(slot).x)
            # A partial camera view plus deterministic missing samples; no negative
            # labels are assigned to missing pixels and no particle IDs are stored.
            mask = rendered.valid & (rng.random(rendered.valid.shape) > 0.07)
            visible = objective._visible_particle_ids(rendered)
            pixels = np.floor(rendered.projected_positions[visible]).astype(int)
            visible = visible[mask[pixels[:, 1], pixels[:, 0]]]
            points = rendered.optical_positions[visible].copy()
            observations.append(Observation(len(observations) + 1, completed,
                                            rendered.depth.copy(), mask, initial.depth.copy(), initial_mask,
                                            points, timestamp=completed * stepper.config.dt))
        if slot == stepper.capacity - 1 and completed < len(controls):
            stepper.load_state(0, stepper.state(slot))
            slot = 0
    if len(observations) != len(observation_steps):
        raise RuntimeError("Synthetic truth replay missed an observation state")
    return observations


@dataclass
class SyntheticProblem:
    training: CheckpointedRollout
    heldout: CheckpointedRollout
    initial_state: ParticleState
    heldout_state: ParticleState
    target_summary: dict


def build_problem(config):
    initial = make_initial_state(config)
    heldout = make_initial_state(config, heldout=True)
    segment = min(config.segment_length, config.steps)
    stepper = Stepper(simulation_config(config), TRUTH, capacity=segment + 1)
    controls = [ToolControl.stationary(index * config.dt) for index in range(config.steps)]
    training_loss = ObservationLoss(synthetic_camera(), synthetic_loss_config(config.loss_version, config.visibility_temperature_m), initial.x, config.precision)
    heldout_loss = ObservationLoss(synthetic_camera(), synthetic_loss_config(config.loss_version, config.visibility_temperature_m), heldout.x, config.precision)
    training_target_loss, heldout_target_loss = training_loss, heldout_loss
    if config.target_visibility_temperature_m != config.visibility_temperature_m:
        target_settings = synthetic_loss_config(config.loss_version, config.target_visibility_temperature_m)
        training_target_loss = ObservationLoss(synthetic_camera(), target_settings, initial.x, config.precision)
        heldout_target_loss = ObservationLoss(synthetic_camera(), target_settings, heldout.x, config.precision)
    observations = generate_observations(stepper, initial, controls, config.observation_steps, training_target_loss, config.seed + 1)
    heldout_observations = generate_observations(stepper, heldout, controls, config.observation_steps, heldout_target_loss, config.seed + 2)
    training_rollout = CheckpointedRollout(stepper, initial, controls, observations, training_loss, segment_length=segment)
    heldout_rollout = CheckpointedRollout(stepper, heldout, controls, heldout_observations, heldout_loss, segment_length=segment)

    def summaries(records):
        return [{"frame_index": record.frame_index, "step": record.step,
                 "observed_pixels": int(record.observed_valid.sum()),
                 "observed_points": int(len(record.observed_points)),
                 "depth_sha256": array_hash(record.observed_depth),
                 "mask_sha256": array_hash(record.observed_valid),
                 "points_sha256": array_hash(record.observed_points)} for record in records]

    return SyntheticProblem(training_rollout, heldout_rollout, initial, heldout,
                            {"physics_version": config.physics_version,
                             "training": summaries(observations), "heldout": summaries(heldout_observations),
                             "training_excitation": excitation_diagnostics(initial, config),
                             "heldout_excitation": excitation_diagnostics(heldout, config),
                             "particle_correspondence_used": False})


def objective_from_rollout(rollout):
    def objective(parameters):
        evaluation = rollout.value_and_gradient(parameters)
        return ObjectiveValue(evaluation.value, evaluation.gradient,
                              {**evaluation.diagnostics, "frames": evaluation.frames})
    return objective


def run_calibration(config, iterations, output_dir, runtime, *, learning_rate=0.06, resume=False,
                    learning_rate_policy="persistent-v1", learning_rate_growth=1.25):
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    options = AdamOptions(learning_rate=learning_rate, max_backtracks=8,
                          gradient_tolerance=1e-9, loss_tolerance=1e-12,
                          learning_rate_policy=learning_rate_policy, learning_rate_growth=learning_rate_growth)
    identity = declared_identity(config, runtime)
    identity["optimizer"] = asdict(options)
    started = time.perf_counter()
    with RunStore(output_dir, identity, resume=resume) as store:
        problem = build_problem(config)
        if resume:
            if canonical_hash(store.read_json("synthetic_targets.json")) != canonical_hash(problem.target_summary):
                raise ValueError("Regenerated synthetic observations differ from this run; resume rejected")
        else:
            store.write_json("synthetic_targets.json", problem.target_summary)

        def callback(event, state):
            store.save_optimizer(event, state)
            message = {key: event[key] for key in ("type", "iteration", "status", "value", "value_before",
                                                   "value_after", "accepted", "learning_rate", "gradient_norm",
                                                   "learning_rate_policy", "proposal_learning_rate", "accepted_learning_rate") if key in event}
            print(json.dumps(message, sort_keys=True), flush=True)

        optimizer = ProjectedAdam(parameter_space(), objective_from_rollout(problem.training), options,
                                  callback=callback, objective_id=canonical_hash(identity))
        if resume:
            optimizer.load_state_dict(store.optimizer_state())
        initial_evaluation = problem.training.value_and_gradient(INITIAL_PARAMETERS)
        optimum_reference = problem.training.value_and_gradient(TRUTH, compute_grad=False)
        result = optimizer.run(iterations)
        selected = dict(result.best_parameters)
        # The selected candidate is now fixed; held-out measurements cannot choose updates.
        heldout_initial = problem.heldout.value_and_gradient(INITIAL_PARAMETERS, compute_grad=False)
        heldout_selected = problem.heldout.value_and_gradient(selected, compute_grad=False)
        heldout_truth = problem.heldout.value_and_gradient(TRUTH, compute_grad=False)
        initial_coordinate_gradient = parameter_space().pullback(parameter_space().coordinates(), initial_evaluation.gradient)
        report = {
            "schema": SYNTHETIC_VERSION, "status": result.status,
            "physics_version": config.physics_version,
            "runtime": {**runtime, "forward_verified": True, "backward_verified": True},
            "loss": synthetic_loss_config(config.loss_version, config.visibility_temperature_m).as_dict(), "optimizer": asdict(options),
            "target_loss": synthetic_loss_config(config.loss_version, config.target_visibility_temperature_m).as_dict(),
            "fitted_parameters": list(FIT_PARAMETERS), "truth": TRUTH, "initial_parameters": INITIAL_PARAMETERS,
            "selected_parameters": selected, "iterations": result.iterations,
            "accepted_updates": result.accepted_updates, "optimizer_evaluations": result.evaluations,
            "training": {"initial_loss": initial_evaluation.value, "selected_loss": result.best_value,
                         "truth_loss": optimum_reference.value,
                         "relative_loss_reduction": (initial_evaluation.value - result.best_value) / max(abs(initial_evaluation.value), 1e-15),
                         "initial_physical_gradient": initial_evaluation.gradient,
                         "initial_coordinate_gradient": dict(zip(FIT_PARAMETERS, initial_coordinate_gradient.tolist())),
                         "initial_gradient_diagnostics": initial_evaluation.diagnostics},
            "heldout": {"initial_loss": heldout_initial.value, "selected_loss": heldout_selected.value,
                        "truth_loss": heldout_truth.value, "used_for_selection": False},
            "relative_parameter_errors": {name: abs(selected[name] - TRUTH[name]) / abs(TRUTH[name]) for name in FIT_PARAMETERS},
            "targets": problem.target_summary, "elapsed_seconds": time.perf_counter() - started,
            "interpretation": "The synthetic forward material law is shared, but prediction and target visibility temperatures may differ. Loss reduction tests gradients and optimization, not physical identification. Recovery may be non-unique, and fitted parameters can compensate for observation-loss smoothing. This is not real-dough calibration.",
        }
        store.write_json("result.json", report)
        store.write_json("optimization_history.json", result.history)
        store.append_event({"event": "completed", "status": result.status,
                            "training_initial_loss": initial_evaluation.value, "training_selected_loss": result.best_value})
        return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--precision", choices=("f32", "f64"), required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--loss-version", choices=SUPPORTED_LOSS_VERSIONS, default=LOSS_VERSION)
    parser.add_argument("--physics-version", choices=PHYSICS_VERSIONS, default="corrected-v1",
                        help="Forward physics for both targets and predictions; legacy-v1 preserves prior transfer behavior")
    parser.add_argument("--visibility-temperature-m", type=float, default=PREDICTION_TEMPERATURE_M,
                        help="Prediction-renderer depth-softmax temperature in meters")
    parser.add_argument("--target-visibility-temperature-m", type=float, default=TARGET_TEMPERATURE_M,
                        help="Synthetic observation-generator temperature; keep fixed for loss comparisons")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--particles-per-axis", type=int, default=4)
    parser.add_argument("--grid", type=int, choices=(12, 16), default=16)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--observation-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--learning-rate-policy", choices=("persistent-v1", "recover-v1"), default="persistent-v1")
    parser.add_argument("--learning-rate-growth", type=float, default=1.25)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = SyntheticConfig(steps=args.steps, segment_length=args.segment_length,
                             particles_per_axis=args.particles_per_axis, grid=args.grid,
                             dt=args.dt, precision=args.precision, seed=args.seed,
                             observation_count=args.observation_count, loss_version=args.loss_version,
                             physics_version=args.physics_version,
                             visibility_temperature_m=args.visibility_temperature_m,
                             target_visibility_temperature_m=args.target_visibility_temperature_m)
    output = args.output_dir.resolve()
    if output == RUN_ROOT.resolve() or not output.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError(f"output-dir must be a new child directory of {RUN_ROOT}")
    runtime = init_runtime(args.backend, args.precision, args.cpu_threads, seed=args.seed)
    report = run_calibration(config, args.iterations, output, runtime,
                             learning_rate=args.learning_rate, resume=args.resume,
                             learning_rate_policy=args.learning_rate_policy,
                             learning_rate_growth=args.learning_rate_growth)
    print(json.dumps({"result": str(output / "result.json"), "status": report["status"],
                      "training": report["training"]["relative_loss_reduction"],
                      "heldout": report["heldout"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

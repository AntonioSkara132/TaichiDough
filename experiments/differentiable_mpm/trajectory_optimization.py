"""Small synthetic tool-path optimization using control gradients from checkpointed MPM.

This is a contact-gradient demonstration, not an Episode 14 fit or a general planner.
The first tool has a fixed orientation. Its linear velocities are the decision
variables; each substep uses the velocity-integrated midpoint tool position.
"""
from dataclasses import dataclass
from pathlib import Path
import json
from time import perf_counter
import zipfile

import numpy as np
import taichi as ti

from .checkpoint import CheckpointedRollout
from .solver import Stepper
from .state import DEFAULT_PARAMETERS, ParticleState, SDFData, SimulationConfig, ToolControl


@dataclass
class PositionTarget:
    frame_index: int
    step: int
    positions: np.ndarray


class PositionLoss:
    def __init__(self, scale=0.003):
        self.scale = scale

    def value_and_grad_positions(self, positions, observation, compute_grad=True):
        difference = (positions - observation.positions) / self.scale
        value = 0.5 * float(np.mean(np.sum(difference * difference, axis=1)))
        gradient = difference / (self.scale * len(positions)) if compute_grad else None
        return type("LossRecord", (), {"value": value, "gradient": gradient,
                                        "components": {}, "diagnostics": {}})()


def controls_from_velocities(base_control, velocities, dt):
    """For v[t], use center[t] + dt*v[t]/2 and center[t+1] = center[t] + dt*v[t]."""
    velocities = np.asarray(velocities, dtype=np.float64)
    if velocities.ndim != 2 or velocities.shape[1] != 3 or not np.isfinite(velocities).all():
        raise ValueError("velocities must be finite [steps,3]")
    center = np.asarray(base_control.poses[0, :3], dtype=np.float64).copy()
    result = []
    for t, velocity in enumerate(velocities):
        poses = base_control.poses.copy()
        twists = base_control.velocities.copy()
        poses[0, :3] = center + 0.5 * dt * velocity
        twists[0, :3] = velocity
        result.append(ToolControl(poses, twists, base_control.time + t * dt))
        center += dt * velocity
    return result


def velocity_path_gradient(control_gradient, dt):
    """Chain dL/d(tool centers and linear velocities) through midpoint integration."""
    pose = np.asarray(control_gradient["poses"], dtype=np.float64)[:, 0, :3]
    direct = np.asarray(control_gradient["velocities"], dtype=np.float64)[:, 0, :3]
    if pose.shape != direct.shape or pose.ndim != 2 or pose.shape[1] != 3:
        raise ValueError("Expected matching [steps,2,7] poses and [steps,2,6] velocities")
    later = np.cumsum(pose[::-1], axis=0)[::-1] - pose
    return direct + dt * (0.5 * pose + later)


def smoothness(velocities, weight):
    padded = np.vstack((np.zeros((1, 3)), velocities, np.zeros((1, 3))))
    differences = np.diff(padded, axis=0)
    value = weight * float(np.mean(differences ** 2))
    gradient = 2 * weight / differences.size * (differences[:-1] - differences[1:])
    return value, gradient


def project_speed(velocities, max_speed):
    speed = np.linalg.norm(velocities, axis=1, keepdims=True)
    return velocities * np.minimum(1.0, max_speed / np.maximum(speed, 1e-12))


def demo_fixture(precision="f32"):
    dtype = np.float32 if precision == "f32" else np.float64
    config = SimulationConfig(n_particles=4, grid=12, precision=precision, dt=2e-4,
                              particle_mass=0.001, particle_volume=1e-6, gravity=-2.0,
                              tool_collision="sdf", tool_contact_padding=0.004,
                              tool_contact_model="coulomb-bounded-adhesive-v2")
    state = ParticleState.initial([[.417, .40, .405], [.441, .415, .409],
                                   [.456, .428, .432], [.431, .421, .445]], dtype)
    state.v[:] = [[.1, -.2, .04], [-.05, -.1, .08], [.02, -.3, -.03], [.01, -.1, .07]]
    state.C[:] = [[.2, .3, .02], [-.08, -.1, .07], [.03, -.01, .15]]
    state.F[:] = [[.94, .03, .01], [0, 1.04, -.02], [.01, .015, 1.09]]
    params = dict(DEFAULT_PARAMETERS, youngs_modulus=4200.0, viscosity=8.0,
                  tool_friction_coefficient=0.6, tool_adhesion=5.0)
    resolution = 13
    axis = np.linspace(-.18, .18, resolution, dtype=dtype)
    coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    radius = np.linalg.norm(coordinates, axis=-1)
    distance = radius - .08
    normals = coordinates / np.maximum(radius[..., None], 1e-8)
    sdf = SDFData(np.stack((distance, distance)), np.stack((normals, normals)),
                  np.full((2, 3), -.18, dtype), np.full((2, 3), .36 / (resolution - 1), dtype))
    base = ToolControl.stationary()
    base.poses[0, :3] = [.43, .40, .37]
    return config, state, params, base, sdf


def run_demo(output_dir, *, iterations=5, steps=4, precision="f32", arch=ti.cuda,
             max_speed=0.35, smooth_weight=1e-4):
    """Generate a synthetic target, check one action derivative, optimize and save results.

    The loss is differentiable through tool contact within its active branch.
    Backtracking only accepts an objective improvement; velocity norms are capped.
    """
    if steps < 2 or iterations < 1 or max_speed <= 0 or smooth_weight < 0:
        raise ValueError("Require steps >= 2, iterations >= 1 and valid constraints")
    ti.init(arch=arch, default_fp=ti.f32 if precision == "f32" else ti.f64, offline_cache=True)
    if arch == ti.cuda and ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("CUDA initialization failed")
    config, state, params, base, sdf = demo_fixture(precision)
    stepper = Stepper(config, params, capacity=steps + 1, sdf=sdf, control_gradients=True)
    true_velocities = np.tile(np.array([.19, -.07, .04]), (steps, 1))
    true_velocities[:, 0] *= np.linspace(1.0, .6, steps)
    target_controls = controls_from_velocities(base, true_velocities, config.dt)
    stepper.load_state(0, state)
    target_states = {}
    for t, control in enumerate(target_controls):
        stepper.advance(t, control)
        if (t + 1) in (steps // 2, steps):
            target_states[t + 1] = stepper.state(t + 1).x.copy()
    targets = [PositionTarget(i, t, target_states[t]) for i, t in enumerate(sorted(target_states))]
    loss = PositionLoss()

    def evaluate(actions, compute_grad=False, capture=False):
        controls = controls_from_velocities(base, actions, config.dt)
        rollout = CheckpointedRollout(stepper, state, controls, targets, loss, segment_length=2)
        frames = {}
        t0 = perf_counter()
        result = rollout.value_and_gradient(params, compute_grad=compute_grad,
                                             compute_control_grad=compute_grad,
                                             frame_callback=(lambda t, s: frames.setdefault(t, s.x.copy())) if capture else None,
                                             frame_steps={0, steps // 2, steps} if capture else None)
        elapsed = perf_counter() - t0
        regularizer, regularizer_gradient = smoothness(actions, smooth_weight)
        total = result.value + regularizer
        if not compute_grad:
            return total, result, frames, elapsed
        gradient = velocity_path_gradient(result.control_gradient, config.dt) + regularizer_gradient
        return total, gradient, result, frames, elapsed

    zero = np.zeros((steps, 3))
    initial_loss, initial_grad, initial_result, initial_frames, compile_seconds = evaluate(zero, compute_grad=True, capture=True)
    full_replay = CheckpointedRollout(stepper, state, controls_from_velocities(base, zero, config.dt),
                                      targets, loss, segment_length=steps)
    full_result = full_replay.value_and_gradient(params, compute_control_grad=True)
    segment_difference = max(float(np.max(np.abs(initial_result.control_gradient[name] -
                                                 full_result.control_gradient[name])))
                             for name in ("poses", "velocities", "rotation"))
    segment_scale = max(float(np.max(np.abs(full_result.control_gradient[name])))
                        for name in ("poses", "velocities", "rotation"))
    if segment_difference > .05 * max(1.0, segment_scale):
        raise RuntimeError(f"Segmented control gradients disagree: {segment_difference}")

    def independent_control_result(kind, h):
        controls = controls_from_velocities(base, zero, config.dt)
        if kind == "position":
            controls[0].poses[0, 0] += h
        else:
            controls[0].poses[0, 3:] = [np.sin(h / 2), 0, 0, np.cos(h / 2)]
        return CheckpointedRollout(stepper, state, controls, targets, loss, segment_length=2).value_and_gradient(
            params, compute_grad=False)

    pose_h, rotation_h = (5e-5, .005) if precision == "f32" else (2e-5, .0005)
    pose_plus, pose_minus = (independent_control_result("position", pose_h),
                             independent_control_result("position", -pose_h))
    pose_branches_match = (pose_plus.diagnostics["branch_counts_total"] ==
                           initial_result.diagnostics["branch_counts_total"] ==
                           pose_minus.diagnostics["branch_counts_total"])
    if not pose_branches_match:
        raise RuntimeError("Pose finite difference changed tool-contact branch counts")
    pose_fd = (pose_plus.value - pose_minus.value) / (2 * pose_h)
    rotation_fd = ((independent_control_result("rotation", rotation_h).value -
                    independent_control_result("rotation", -rotation_h).value) / (2 * rotation_h))
    pose_ad = float(initial_result.control_gradient["poses"][0, 0, 0])
    rotation_ad = float(initial_result.control_gradient["rotation"][0, 0, 0])
    pose_error = abs(pose_fd - pose_ad) / max(abs(pose_fd), abs(pose_ad), 1e-8)
    rotation_error = abs(rotation_fd - rotation_ad) / max(abs(rotation_fd), abs(rotation_ad), 1e-8)
    if max(pose_error, rotation_error) > (.12 if precision == "f32" else .05):
        raise RuntimeError(f"Tool pose gradient check failed: position error={pose_error}, rotation error={rotation_error}")
    direction = np.random.default_rng(14).normal(size=zero.shape)
    direction /= np.linalg.norm(direction)
    fd_step = .02 if precision == "f32" else .002
    plus, _, _, _ = evaluate(fd_step * direction)
    minus, _, _, _ = evaluate(-fd_step * direction)
    finite_difference = (plus - minus) / (2 * fd_step)
    analytic = float(np.sum(initial_grad * direction))
    relative_error = abs(analytic - finite_difference) / max(abs(analytic), abs(finite_difference), 1e-8)
    if not np.isfinite(relative_error) or relative_error > (.25 if precision == "f32" else .05):
        raise RuntimeError(f"Tool-path gradient check failed: analytic={analytic}, FD={finite_difference}, error={relative_error}")

    actions = zero.copy()
    loss_value = initial_loss
    history = [{"iteration": 0, "total_loss": float(loss_value), "observation_loss": float(initial_result.value),
                "max_speed": 0.0}]
    iteration_times = []
    warm_phase_times = []
    for iteration in range(1, iterations + 1):
        start = perf_counter()
        value, gradient, result, _, _ = evaluate(actions, compute_grad=True)
        warm_phase_times.append({"forward_s": result.diagnostics["forward_seconds"],
                                 "backward_s": result.diagnostics["backward_seconds"]})
        if not np.isfinite(gradient).all():
            raise RuntimeError("Nonfinite velocity gradient")
        scale = max(float(np.linalg.norm(gradient)), 1e-15)
        accepted = False
        for line_step in (.12, .06, .03, .015, .0075):
            candidate = project_speed(actions - line_step * gradient / scale, max_speed)
            candidate_loss, candidate_result, _, _ = evaluate(candidate)
            if candidate_loss < value - 1e-10:
                actions = candidate
                loss_value = candidate_loss
                accepted = True
                break
        iteration_times.append(perf_counter() - start)
        history.append({"iteration": iteration, "total_loss": float(loss_value),
                        "observation_loss": float(candidate_result.value if accepted else result.value),
                        "max_speed": float(np.max(np.linalg.norm(actions, axis=1))), "accepted": accepted})
        if not accepted:
            break
    final_loss, final_result, final_frames, _ = evaluate(actions, capture=True)
    if final_loss > initial_loss + 1e-9:
        raise RuntimeError("Final path is worse than the starting path")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"gpu": str(ti.lang.impl.current_cfg().arch), "precision": precision, "steps": steps,
               "dt_s": config.dt, "iterations_requested": iterations, "iterations_run": len(history) - 1,
               "initial_loss": float(initial_loss), "optimized_loss": float(final_loss),
               "initial_observation_loss": float(initial_result.value),
               "optimized_observation_loss": float(final_result.value),
               "loss_reduction_fraction": float(1 - final_loss / initial_loss),
               "gradient_check": {"analytic": analytic, "finite_difference": float(finite_difference),
                                  "relative_error": float(relative_error), "fd_step_m_per_s": fd_step,
                                  "pose_x_analytic": pose_ad, "pose_x_finite_difference": float(pose_fd),
                                  "pose_x_fd_step_m": pose_h, "pose_x_relative_error": float(pose_error),
                                  "pose_x_branch_counts_unchanged": pose_branches_match,
                                  "rotation_x_analytic": rotation_ad, "rotation_x_finite_difference": float(rotation_fd),
                                  "rotation_x_relative_error": float(rotation_error),
                                  "segment_gradient_max_abs_difference": segment_difference},
               "first_gradient_wall_s_including_compilation": compile_seconds,
               "first_gradient_forward_s": initial_result.diagnostics["forward_seconds"],
               "first_gradient_backward_s": initial_result.diagnostics["backward_seconds"],
               "later_iteration_wall_s": iteration_times, "warm_gradient_phase_s": warm_phase_times,
               "max_speed_m_per_s": max_speed,
               "smoothness_weight": smooth_weight, "history": history,
               "note": "Synthetic 4-particle sphere-contact test; not Episode 14 calibration or a long-horizon planner."}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output_dir / "trajectories.npz", initial_velocities=zero,
                        optimized_velocities=actions, target_velocities=true_velocities,
                        initial_positions=initial_frames[steps], optimized_positions=final_frames[steps],
                        target_positions=target_states[steps], initial_tool_centers=np.array([c.poses[0, :3] for c in controls_from_velocities(base, zero, config.dt)]),
                        optimized_tool_centers=np.array([c.poses[0, :3] for c in controls_from_velocities(base, actions, config.dt)]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    target = target_states[steps]
    axes[0].scatter(target[:, 0] * 1e3, target[:, 1] * 1e3, color="black", s=24, label="target")
    for label, positions, color in (("initial", initial_frames[steps], "C1"),
                                    ("optimized", final_frames[steps], "C0")):
        displaced = target + 20 * (positions - target)
        axes[0].scatter(displaced[:, 0] * 1e3, displaced[:, 1] * 1e3, s=22, color=color,
                        label=f"{label} (offset ×20)")
        for original, point in zip(target, displaced):
            axes[0].plot([original[0] * 1e3, point[0] * 1e3], [original[1] * 1e3, point[1] * 1e3],
                         color=color, alpha=.6)
        error_um = np.linalg.norm(positions - target, axis=1) * 1e6
        axes[1].plot(np.arange(len(target)), error_um, marker="o", color=color, label=label)
    axes[0].set(xlabel="particle x [mm]", ylabel="particle y [mm]", title="Final particle positions")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="particle index", ylabel="3D distance to target [µm]", title="Final position errors")
    axes[1].legend()
    for label, values in (("target", true_velocities), ("optimized", actions)):
        axes[2].plot(np.arange(steps), values[:, 0], marker="o", label=f"{label} vx")
    axes[2].set(xlabel="substep", ylabel="tool vx [m/s]", title="Integrated velocity actions")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(fig)
    with zipfile.ZipFile(output_dir / "results.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("summary.json", "trajectories.npz", "comparison.png"):
            archive.write(output_dir / name, name)
    return summary

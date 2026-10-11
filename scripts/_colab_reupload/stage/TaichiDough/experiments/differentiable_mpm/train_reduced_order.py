"""Train and evaluate a POD/latent model with whole-trajectory splits."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .reduced_order import PODCodec, ReducedDynamics, field_slices
from .state import STATE_NAMES


def validate_trajectories(trajectories):
    if not trajectories:
        raise ValueError("No trajectories")
    first = trajectories[0]
    seen = set()
    for trajectory in trajectories:
        trajectory.validate()
        if trajectory.trajectory_id in seen:
            raise ValueError("Trajectory IDs must be unique")
        seen.add(trajectory.trajectory_id)
        if (trajectory.identity != first.identity or trajectory.dt != first.dt
                or trajectory.n_particles != first.n_particles):
            raise ValueError("Training trajectories have incompatible scenes or timesteps")


def split_trajectories(trajectories, seed=7):
    """Keep at least one complete trajectory each for validation and test."""
    validate_trajectories(trajectories)
    if len(trajectories) < 3:
        raise ValueError("Require at least three independent trajectories")
    order = np.random.default_rng(seed).permutation(len(trajectories))
    count = max(1, len(trajectories) // 5)
    test = [trajectories[i] for i in order[:count]]
    validation = [trajectories[i] for i in order[count:2 * count]]
    train = [trajectories[i] for i in order[2 * count:]]
    return train, validation, test


def _tensors(trajectory, model):
    model.check_scene(trajectory.identity, trajectory.dt, trajectory.n_particles)
    device = next(model.parameters()).device
    return (torch.as_tensor(trajectory.states, dtype=torch.float32, device=device),
            torch.as_tensor(trajectory.controls, dtype=torch.float32, device=device),
            torch.as_tensor(trajectory.parameters, dtype=torch.float32, device=device))


def _field_errors(prediction, truth, n_particles):
    difference = prediction - truth
    return {name: float(difference[..., section].square().mean().sqrt())
            for name, section in field_slices(n_particles).items()}


@torch.no_grad()
def evaluate(model, trajectories):
    """Report projected-state, one-step and recursive full-state errors separately."""
    model.eval()
    results = []
    for trajectory in trajectories:
        trajectory.validate()
        states, controls, parameters = _tensors(trajectory, model)
        latent = model.codec.encode(states)
        reconstructed = model.codec.decode(latent)
        one_step = model.codec.decode(model.step(latent[:-1], controls, parameters))
        start = time.perf_counter()
        predicted_latent = model.rollout(latent[0], controls, parameters)
        latent_seconds = time.perf_counter() - start
        decoded = model.codec.decode(predicted_latent)
        persistence = states[0].expand_as(states[1:])
        velocity = persistence.clone()
        sections = field_slices(trajectory.n_particles)
        elapsed = torch.arange(1, len(states), device=states.device)[:, None] * model.dt
        velocity[:, sections['x']] += elapsed * states[0, sections['v']]
        fields = {
            "compression": _field_errors(reconstructed, states, trajectory.n_particles),
            "one_step": _field_errors(one_step, states[1:], trajectory.n_particles),
            "rollout": _field_errors(decoded[1:], states[1:], trajectory.n_particles),
            "persistence": _field_errors(persistence, states[1:], trajectory.n_particles),
            "constant_velocity": _field_errors(velocity, states[1:], trajectory.n_particles),
        }
        x = (decoded[1:, sections['x']] - states[1:, sections['x']]).reshape(-1, trajectory.n_particles, 3)
        final_x = x[-1]
        f = decoded[1:, sections['F']].reshape(-1, 3, 3)
        jp = decoded[1:, sections['Jp']]
        results.append({"trajectory_id": trajectory.trajectory_id, "steps": trajectory.n_steps,
                        "simulated_seconds": trajectory.n_steps * trajectory.dt,
                        "errors_rms": fields,
                        "position_vector_rms_m": float(x.square().sum(-1).mean().sqrt()),
                        "final_position_vector_rms_m": float(final_x.square().sum(-1).mean().sqrt()),
                        "nonpositive_F_determinants": int((torch.linalg.det(f) <= 0).sum()),
                        "nonpositive_Jp": int((jp <= 0).sum()),
                        "latent_rollout_seconds": latent_seconds,
                        "taichi_collection_seconds": trajectory.metadata.get("collection_seconds"),
                        "finite": bool(torch.isfinite(decoded).all())})
    return results


@torch.no_grad()
def _validation_loss(model, trajectories):
    model.eval()
    values = []
    for trajectory in trajectories:
        states, controls, parameters = _tensors(trajectory, model)
        z = model.codec.encode(states)
        prediction = model.rollout(z[0], controls, parameters)
        values.append(float((prediction[1:] - z[1:]).square().mean()))
    return float(np.mean(values))


def train(train_trajectories, validation_trajectories, *, rank=8, hidden_dim=128,
          epochs=100, horizon=8, batch_size=64, learning_rate=1e-3, seed=7, device="cpu"):
    """Fit bases and normalizers only on train; select model on recursive validation."""
    validate_trajectories(train_trajectories + validation_trajectories)
    if not train_trajectories or not validation_trajectories:
        raise ValueError("Require nonempty train and validation trajectories")
    if min(epochs, horizon, batch_size) < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Positive epochs, horizon, batch size and learning rate required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    snapshots = np.concatenate([t.states for t in train_trajectories])
    first = train_trajectories[0]
    codec = PODCodec.fit(snapshots, first.n_particles, rank)
    model = ReducedDynamics(codec, first.identity, first.dt, hidden_dim).to(device)
    prepared = [_tensors(t, model) for t in train_trajectories]
    model.fit_condition_normalization(torch.cat([c for _, c, _ in prepared]),
                                      torch.cat([p.expand(len(c), -1) for _, c, p in prepared]))
    windows = []
    for states, controls, parameters in prepared:
        length = min(horizon, len(controls))
        with torch.no_grad():
            z = model.codec.encode(states)
        for start in range(len(controls) - length + 1):
            windows.append((z[start:start + length + 1], controls[start:start + length], parameters))
    # Group equal-length windows so different trajectory lengths remain supported.
    lengths = sorted(set(len(c) for _, c, _ in windows))
    optimizer = torch.optim.Adam(model.network.parameters(), lr=learning_rate)
    best = _validation_loss(model, validation_trajectories)
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "validation_latent_mse": best}]
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for length in lengths:
            indices = [i for i, (_, c, _) in enumerate(windows) if len(c) == length]
            rng.shuffle(indices)
            for offset in range(0, len(indices), batch_size):
                batch = [windows[i] for i in indices[offset:offset + batch_size]]
                z, c, p = (torch.stack([row[i] for row in batch]) for i in range(3))
                prediction = model.rollout(z[:, 0], c, p)
                loss = (prediction[:, 1:] - z[:, 1:]).square().mean()
                # Also fit every intervening transition from its true projected state.
                one = model.step(z[:, :-1], c, p[:, None, :])
                loss = loss + (one - z[:, 1:]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite reduced-model training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
                count += len(batch)
        value = _validation_loss(model, validation_trajectories)
        history.append({"epoch": epoch, "train_latent_mse": total / count,
                        "validation_latent_mse": value})
        if value < best:
            best = value
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history,
                   "best_epoch": min(history, key=lambda row: row['validation_latent_mse'])['epoch'],
                   "train_ids": [t.trajectory_id for t in train_trajectories],
                   "validation_ids": [t.trajectory_id for t in validation_trajectories]}

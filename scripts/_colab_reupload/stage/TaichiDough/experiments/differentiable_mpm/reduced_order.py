"""Fixed-particle POD compression and differentiable learned MPM dynamics.

This is a data-trained approximation, not a constitutive law. Decoding does not
ensure positive deformation determinants or plastic volumes.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .state import PARAMETER_NAMES, STATE_NAMES

VERSION = 1
CONTROL_DIM = 36
FIELD_WIDTHS = (3, 3, 9, 9, 1)


def control_features(poses, velocities):
    """Differentiable [..,2,7] xyzw poses + [..,2,6] velocities to [..,36].

    Normalize the quaternion before converting to a matrix. q and -q give the
    same features; gradients remain available for tool-path optimization.
    """
    if poses.shape[-2:] != (2, 7) or velocities.shape != (*poses.shape[:-1], 6):
        raise ValueError("Require two xyzw poses and two six-component velocities")
    if not torch.isfinite(poses).all() or not torch.isfinite(velocities).all():
        raise ValueError("Nonfinite tool controls")
    q = poses[..., 3:]
    norm = q.norm(dim=-1, keepdim=True)
    if (norm < 1e-12).any():
        raise ValueError("Zero tool quaternion")
    x, y, z, w = (q / norm).unbind(-1)
    matrix = torch.stack((1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
                          2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
                          2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)), dim=-1)
    return torch.cat((poses[..., :3], matrix, velocities), dim=-1).flatten(-2)


def field_slices(n_particles):
    offset = 0
    result = {}
    for name, width in zip(STATE_NAMES, FIELD_WIDTHS):
        result[name] = slice(offset, offset + width * n_particles)
        offset += width * n_particles
    return result


class PODCodec(nn.Module):
    """Independent field bases fitted on training snapshots only.

    Each field uses one RMS scale, preserving spatial correlations. Constant
    fields retain their training mean and use a unit scale. Bases are buffers,
    so gradients pass through encoding/decoding without updating the bases.
    """

    def __init__(self, n_particles, ranks):
        super().__init__()
        self.n_particles = int(n_particles)
        self.ranks = tuple(int(r) for r in ranks)
        if self.n_particles < 1 or len(self.ranks) != 5:
            raise ValueError("Require positive particle count and five field ranks")
        for name, width, rank in zip(STATE_NAMES, FIELD_WIDTHS, self.ranks):
            size = width * self.n_particles
            if not 1 <= rank <= size:
                raise ValueError(f"Invalid POD rank for {name}")
            self.register_buffer(f"mean_{name}", torch.zeros(size))
            self.register_buffer(f"scale_{name}", torch.ones(()))
            self.register_buffer(f"basis_{name}", torch.zeros(size, rank))
        self.register_buffer("latent_mean", torch.zeros(sum(self.ranks)))
        self.register_buffer("latent_scale", torch.ones(sum(self.ranks)))
        self.register_buffer("latent_active", torch.zeros(sum(self.ranks)))

    @property
    def latent_dim(self):
        return sum(self.ranks)

    @classmethod
    def fit(cls, snapshots, n_particles, rank=8):
        x = torch.as_tensor(snapshots, dtype=torch.float64)
        if x.ndim != 2 or len(x) < 2 or x.shape[1] != 25 * n_particles or not torch.isfinite(x).all():
            raise ValueError("Require finite [snapshots,25*N] training states, at least two")
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer per field")
        ranks = [min(rank, len(x) - 1, w * n_particles) for w in FIELD_WIDTHS]
        codec = cls(n_particles, ranks).double()
        with torch.no_grad():
            for name, section in field_slices(n_particles).items():
                block = x[:, section]
                mean = block.mean(0)
                centered = block - mean
                scale = centered.square().mean().sqrt()
                if scale < 1e-12:
                    scale = scale.new_tensor(1.0)
                _, singular, vh = torch.linalg.svd(centered / scale, full_matrices=False)
                r = ranks[STATE_NAMES.index(name)]
                active = singular[:r] > max(1e-10, float(singular[0]) * 1e-10)
                basis = vh[:r].T * active
                getattr(codec, f"mean_{name}").copy_(mean)
                getattr(codec, f"scale_{name}").copy_(scale)
                getattr(codec, f"basis_{name}").copy_(basis)
            codec.latent_active.copy_(torch.cat([
                (getattr(codec, f"basis_{name}").square().sum(0) > 0).double()
                for name in STATE_NAMES]))
            raw = codec.raw_encode(x)
            codec.latent_mean.copy_(raw.mean(0))
            scale = raw.std(0, unbiased=False)
            codec.latent_scale.copy_(torch.where(scale > 1e-8, scale, torch.ones_like(scale)))
        return codec.float()

    def raw_encode(self, states):
        if states.shape[-1] != 25 * self.n_particles:
            raise ValueError("State width does not match particle count")
        blocks = []
        for name, section in field_slices(self.n_particles).items():
            blocks.append(((states[..., section] - getattr(self, f"mean_{name}"))
                           / getattr(self, f"scale_{name}")) @ getattr(self, f"basis_{name}"))
        return torch.cat(blocks, dim=-1)

    def encode(self, states):
        return (self.raw_encode(states) - self.latent_mean) / self.latent_scale

    def decode(self, latent):
        if latent.shape[-1] != self.latent_dim:
            raise ValueError("Latent width does not match POD bases")
        raw = latent * self.latent_scale + self.latent_mean
        blocks, offset = [], 0
        for name, rank in zip(STATE_NAMES, self.ranks):
            blocks.append((raw[..., offset:offset + rank] @ getattr(self, f"basis_{name}").T)
                          * getattr(self, f"scale_{name}") + getattr(self, f"mean_{name}"))
            offset += rank
        return torch.cat(blocks, dim=-1)


class ReducedDynamics(nn.Module):
    """Residual latent update at one fixed physics timestep and scene identity."""

    def __init__(self, codec, identity, dt, hidden_dim=128):
        super().__init__()
        if not isinstance(identity, str) or not identity or not 0 < dt < float("inf") or hidden_dim < 1:
            raise ValueError("Require scene identity, positive finite dt and network width")
        self.codec = codec
        self.identity = identity
        self.dt = float(dt)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("condition_mean", torch.zeros(CONTROL_DIM + len(PARAMETER_NAMES)))
        self.register_buffer("condition_scale", torch.ones(CONTROL_DIM + len(PARAMETER_NAMES)))
        self.network = nn.Sequential(
            nn.Linear(codec.latent_dim + len(self.condition_mean), hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, codec.latent_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def fit_condition_normalization(self, controls, parameters):
        values = torch.cat((controls, parameters), dim=-1)
        if values.ndim != 2 or values.shape[-1] != len(self.condition_mean) or not torch.isfinite(values).all():
            raise ValueError("Invalid training controls or parameters")
        with torch.no_grad():
            self.condition_mean.copy_(values.mean(0))
            scale = values.std(0, unbiased=False)
            # Never magnify floating-point noise in constant conditioning fields.
            floor = 1e-6 * values.abs().mean(0).clamp_min(1)
            self.condition_scale.copy_(torch.where(scale > floor, scale, torch.ones_like(scale)))

    def step(self, latent, control, parameters):
        if control.shape[-1] != CONTROL_DIM or parameters.shape[-1] != len(PARAMETER_NAMES):
            raise ValueError("Invalid control or parameter width")
        parameters = torch.broadcast_to(parameters, (*control.shape[:-1], len(PARAMETER_NAMES)))
        condition = (torch.cat((control, parameters), dim=-1) - self.condition_mean) / self.condition_scale
        increment = self.network(torch.cat((latent, condition), dim=-1))
        return latent + increment * self.codec.latent_active

    def rollout(self, initial_latent, controls, parameters):
        """Controls [...,T,36]; return [...,T+1,latent_dim] without detaching."""
        latent = initial_latent
        states = [latent]
        for k in range(controls.shape[-2]):
            latent = self.step(latent, controls[..., k, :], parameters)
            states.append(latent)
        return torch.stack(states, dim=-2)

    def check_scene(self, identity, dt, n_particles):
        if identity != self.identity or dt != self.dt or n_particles != self.codec.n_particles:
            raise ValueError("Trajectory scene, timestep or particle identity differs from the model")

    def save(self, path):
        torch.save({"version": VERSION, "identity": self.identity, "dt": self.dt,
                    "n_particles": self.codec.n_particles, "ranks": self.codec.ranks,
                    "hidden_dim": self.hidden_dim, "state_dict": self.state_dict()}, Path(path))

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        if checkpoint["version"] != VERSION:
            raise ValueError("Unsupported reduced model checkpoint version")
        codec = PODCodec(checkpoint["n_particles"], checkpoint["ranks"])
        model = cls(codec, checkpoint["identity"], checkpoint["dt"], checkpoint["hidden_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()

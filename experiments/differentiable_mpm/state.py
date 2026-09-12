"""Shared NumPy state and fixed numerical settings for the experimental solver."""
from dataclasses import dataclass, fields
from typing import Mapping

import numpy as np
from numpy.typing import DTypeLike


PARAMETER_NAMES = (
    "youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max",
    "tool_retention", "floor_retention",
)
DEFAULT_PARAMETERS = {
    "youngs_modulus": 130579.320726, "poisson_ratio": 0.3, "viscosity": 0.0,
    "plastic_min": 0.9, "plastic_max": 1.1,
    "tool_retention": 0.2, "floor_retention": 0.4,
}
STATE_NAMES = ("x", "v", "C", "F", "Jp")
P2G_MODES = ("atomic", "serial")
PHYSICS_VERSIONS = ("corrected-v1", "legacy-v1")


class InvalidStateError(RuntimeError):
    """The requested forward or backward evaluation is not valid."""


@dataclass
class ParticleState:
    x: np.ndarray
    v: np.ndarray
    C: np.ndarray
    F: np.ndarray
    Jp: np.ndarray

    @classmethod
    def initial(cls, positions, dtype: DTypeLike = np.float32):
        x = np.asarray(positions, dtype=dtype).copy()
        if x.ndim != 2 or x.shape[1] != 3 or len(x) == 0:
            raise ValueError("Particle positions must be a nonempty [N,3] array")
        n = len(x)
        return cls(x, np.zeros_like(x), np.zeros((n, 3, 3), dtype=dtype),
                   np.repeat(np.eye(3, dtype=dtype)[None], n, axis=0), np.ones(n, dtype=dtype))

    def copy(self):
        return ParticleState(**{name: getattr(self, name).copy() for name in STATE_NAMES})

    def zeros_like(self):
        return ParticleState(**{name: np.zeros_like(getattr(self, name)) for name in STATE_NAMES})

    def arrays(self):
        return {name: getattr(self, name) for name in STATE_NAMES}

    def validate(self):
        n = len(self.x)
        expected = {"x": (n, 3), "v": (n, 3), "C": (n, 3, 3), "F": (n, 3, 3), "Jp": (n,)}
        for name, dimensions in expected.items():
            a = np.asarray(getattr(self, name))
            if a.shape != dimensions or not np.isfinite(a).all():
                raise InvalidStateError(f"Invalid particle state array {name}")
        if n == 0:
            raise InvalidStateError("No particles")


@dataclass(frozen=True)
class SimulationConfig:
    n_particles: int
    grid: int = 48
    dt: float = 0.0002
    particle_mass: float = 0.25 / 24000
    particle_volume: float = (0.25 / 283.071209393435) / 24000
    gravity: float = -9.81
    floor_y: float = 0.0
    plasticity: str = "none"
    use_jp: bool = False
    jp_hardening: float = 0.0
    jp_min: float = 0.5
    jp_max: float = 2.0
    tool_collision: str = "none"
    tool_contact_padding: float = 1.0 / 384.0
    tool_contact_absorption: float = 0.0
    tool_stickiness: float = 0.0
    floor_absorption: float = 0.0
    floor_stickiness: float = 0.0
    floor_plastic_damping_band: float = 0.0
    velocity_damping: float = 1.0
    plastic_velocity_damping: float = 1.0
    plastic_affine_damping: float = 1.0
    precision: str = "f32"
    min_singular_value: float = 1e-6
    p2g_mode: str = "atomic"
    physics_version: str = "corrected-v1"

    def __post_init__(self):
        if self.n_particles < 1 or self.grid < 8:
            raise ValueError("Require particles > 0 and grid >= 8")
        if self.plasticity not in {"none", "stretch-clamp"}:
            raise ValueError("plasticity must be none or stretch-clamp")
        if self.tool_collision not in {"none", "sdf"}:
            raise ValueError("The experiment supports recorded SDF tools or no tools")
        if self.precision not in {"f32", "f64"}:
            raise ValueError("precision must be f32 or f64")
        if not isinstance(self.p2g_mode, str) or self.p2g_mode not in P2G_MODES:
            raise ValueError("p2g_mode must be atomic or serial")
        if not isinstance(self.physics_version, str) or self.physics_version not in PHYSICS_VERSIONS:
            raise ValueError("physics_version must be corrected-v1 or legacy-v1")
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"Nonfinite setting: {f.name}")
        if min(self.dt, self.particle_mass, self.particle_volume, self.min_singular_value) <= 0:
            raise ValueError("Time step, mass, volume and singular-value threshold must be positive")
        if not (0 < self.jp_min <= self.jp_max):
            raise ValueError("Invalid Jp limits")
        for name in ("tool_contact_absorption", "tool_stickiness", "floor_absorption",
                     "floor_stickiness", "velocity_damping", "plastic_velocity_damping",
                     "plastic_affine_damping"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if min(self.tool_contact_padding, self.floor_plastic_damping_band) < 0:
            raise ValueError("Contact padding and floor damping band must be nonnegative")

    @property
    def allocated_grid(self) -> int:
        return self.grid + (self.physics_version == "corrected-v1")

    @property
    def numpy_dtype(self) -> DTypeLike:
        return np.float32 if self.precision == "f32" else np.float64


@dataclass
class ToolControl:
    """Fixed recorded controls: poses [2,7] use xyzw quaternions; velocities [2,6]."""
    poses: np.ndarray
    velocities: np.ndarray
    time: float = 0.0

    @classmethod
    def stationary(cls, time=0.0):
        poses = np.zeros((2, 7), dtype=np.float64)
        poses[:, :3] = 10.0
        poses[:, 6] = 1.0
        return cls(poses, np.zeros((2, 6), dtype=np.float64), time)

    def validate(self):
        if self.poses.shape != (2, 7) or self.velocities.shape != (2, 6):
            raise ValueError("Tool controls require poses [2,7] and velocities [2,6]")
        if not (np.isfinite(self.poses).all() and np.isfinite(self.velocities).all() and np.isfinite(self.time)):
            raise ValueError("Nonfinite tool controls")
        if not np.allclose(np.linalg.norm(self.poses[:, 3:], axis=1), 1, atol=1e-5):
            raise ValueError("Tool quaternions must be unit length")


@dataclass
class SDFData:
    distances: np.ndarray
    gradients: np.ndarray
    minimums: np.ndarray
    spacings: np.ndarray

    @property
    def resolution(self):
        return int(self.distances.shape[1])

    def validate(self):
        r = self.resolution
        if self.distances.shape != (2, r, r, r) or self.gradients.shape != (2, r, r, r, 3):
            raise ValueError("Require two cubic SDF distance and normal volumes")
        if self.minimums.shape != (2, 3) or self.spacings.shape != (2, 3) or r < 2:
            raise ValueError("Invalid SDF bounds")
        if not all(np.isfinite(a).all() for a in (self.distances, self.gradients, self.minimums, self.spacings)):
            raise ValueError("Nonfinite SDF data")
        if not np.all(self.spacings > 0):
            raise ValueError("SDF spacing must be positive")


def validate_parameters(values: Mapping[str, float]):
    if set(values) != set(PARAMETER_NAMES):
        raise ValueError(f"Physical parameters must be exactly {PARAMETER_NAMES}")
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite material/contact parameters")
    if values["youngs_modulus"] <= 0 or not -1 < values["poisson_ratio"] < 0.5:
        raise ValueError("Require E > 0 and -1 < nu < 0.5")
    if values["viscosity"] < 0 or not 0 < values["plastic_min"] <= 1 <= values["plastic_max"]:
        raise ValueError("Invalid viscosity or stretch limits")
    if not all(0 <= values[n] <= 1 for n in ("tool_retention", "floor_retention")):
        raise ValueError("Retention multipliers must be in [0,1]")

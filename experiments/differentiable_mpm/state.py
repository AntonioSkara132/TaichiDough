"""Shared NumPy state and fixed numerical settings for the experimental solver."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from typing import Mapping

import numpy as np
from numpy.typing import DTypeLike


PARAMETER_NAMES = (
    "youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max",
    "tool_retention", "floor_retention", "tool_friction_coefficient", "tool_stickiness",
    "tool_adhesion", "floor_friction_coefficient", "von_mises_yield_stress_pa",
)
LEGACY_PARAMETER_NAMES = PARAMETER_NAMES[:7]
# Contact parameters that SimulationConfig also stores, so old seven-value records
# take their values from the simulation settings they were run with.
TUNABLE_TOOL_PARAMETERS = PARAMETER_NAMES[7:11]
# The von Mises yield stress is also a SimulationConfig setting. It is absent from
# DEFAULT_PARAMETERS so that spreading the defaults never replaces a configured
# value; normalize_parameters fills it from the simulation, else the inactive value.
YIELD_STRESS_PARAMETER = "von_mises_yield_stress_pa"
INACTIVE_YIELD_STRESS_PA = 1000.0
SIMULATION_MIRRORED_PARAMETERS = (*TUNABLE_TOOL_PARAMETERS, YIELD_STRESS_PARAMETER)
DEFAULT_PARAMETERS = {
    "youngs_modulus": 130579.320726, "poisson_ratio": 0.3, "viscosity": 0.0,
    "plastic_min": 0.9, "plastic_max": 1.1,
    "tool_retention": 0.2, "floor_retention": 0.4,
    "tool_friction_coefficient": 0.5, "tool_stickiness": 0.0,
    "tool_adhesion": 0.0, "floor_friction_coefficient": 0.5,
}
STATE_NAMES = ("x", "v", "C", "F", "Jp")
P2G_MODES = ("atomic", "serial")
PHYSICS_VERSIONS = ("corrected-v1", "legacy-v1")
BOUNDED_ADHESIVE_MODELS = ("coulomb-bounded-adhesive-v2", "coulomb-bounded-shear-adhesive-v3")
TOOL_CONTACT_MODELS = ("retention-v1", "coulomb-v1", "coulomb-adhesive-v1", *BOUNDED_ADHESIVE_MODELS)
TOOL_GRID_CONTACT_MODES = ("inherit", "dirichlet-v1")
FLOOR_CONTACT_MODELS = ("retention-v1", "coulomb-v1")
# Relative tangential speed (m/s) added in quadrature before dividing by it, so the
# Coulomb scale and its derivative stay finite when tool or floor and dough move together.
SLIP_SPEED_EPSILON = 1e-8
# Optional sub-box of the 1/grid lattice; omitted from records when unset so that
# full-cube configurations keep their earlier fingerprints.
GRID_BOX_SETTINGS = ("grid_box_lower", "grid_box_cells")


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
    von_mises_yield_stress_pa: float | None = None
    use_jp: bool = False
    jp_hardening: float = 0.0
    jp_min: float = 0.5
    jp_max: float = 2.0
    tool_collision: str = "none"
    tool_contact_padding: float = 1.0 / 384.0
    tool_contact_model: str = "retention-v1"
    tool_grid_contact_mode: str = "inherit"
    tool_grid_contact_padding: float | None = None
    tool_friction_coefficient: float = 0.5
    tool_contact_absorption: float = 0.0
    tool_stickiness: float = 0.0
    tool_adhesion: float = 0.0
    floor_contact_model: str = "retention-v1"
    floor_friction_coefficient: float = 0.5
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
    # The grid covers [0,1]^3 with spacing 1/grid unless a box is given. A box keeps
    # spacing 1/grid and global node indices, and allocates only nodes
    # grid_box_lower .. grid_box_lower + grid_box_cells - 1 on each axis.
    grid_box_lower: tuple[int, int, int] | None = None
    grid_box_cells: tuple[int, int, int] | None = None

    def __post_init__(self):
        if self.n_particles < 1 or self.grid < 8:
            raise ValueError("Require particles > 0 and grid >= 8")
        self._validate_grid_box()
        if self.plasticity not in {"none", "stretch-clamp", "von-mises"}:
            raise ValueError("plasticity must be none, stretch-clamp or von-mises")
        yield_stress = self.von_mises_yield_stress_pa
        if yield_stress is not None and (
                isinstance(yield_stress, (bool, np.bool_))
                or not isinstance(yield_stress, (int, float))
                or not np.isfinite(yield_stress)
                or yield_stress <= 0):
            raise ValueError("von_mises_yield_stress_pa must be a finite positive number")
        if self.plasticity == "von-mises":
            if yield_stress is None:
                raise ValueError("von-mises plasticity requires von_mises_yield_stress_pa")
            if self.use_jp or self.jp_hardening != 0:
                raise ValueError("von-mises plasticity does not support Jp evolution or hardening")
            if self.plastic_velocity_damping != 1 or self.plastic_affine_damping != 1:
                raise ValueError("von-mises plasticity requires unit plastic damping")
        if self.tool_collision not in {"none", "sdf"}:
            raise ValueError("The experiment supports recorded SDF tools or no tools")
        if not isinstance(self.tool_contact_model, str) or self.tool_contact_model not in TOOL_CONTACT_MODELS:
            raise ValueError(f"tool_contact_model must be one of {TOOL_CONTACT_MODELS}")
        if not isinstance(self.tool_grid_contact_mode, str) or self.tool_grid_contact_mode not in TOOL_GRID_CONTACT_MODES:
            raise ValueError(f"tool_grid_contact_mode must be one of {TOOL_GRID_CONTACT_MODES}")
        if self.tool_grid_contact_mode == "dirichlet-v1" and self.tool_collision != "sdf":
            raise ValueError("dirichlet-v1 grid contact requires SDF tool collision")
        grid_padding = self.tool_grid_contact_padding
        if grid_padding is not None and (
                isinstance(grid_padding, (bool, np.bool_)) or not isinstance(grid_padding, (int, float))
                or not np.isfinite(grid_padding) or grid_padding < 0):
            raise ValueError("tool_grid_contact_padding must be None or a finite nonnegative number")
        for name in ("tool_friction_coefficient", "tool_adhesion", "floor_friction_coefficient"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.floor_contact_model, str) or self.floor_contact_model not in FLOOR_CONTACT_MODELS:
            raise ValueError(f"floor_contact_model must be one of {FLOOR_CONTACT_MODELS}")
        if self.tool_contact_model == "coulomb-v1" and (self.tool_contact_absorption != 0 or self.tool_stickiness != 0):
            raise ValueError("coulomb-v1 requires zero tool contact absorption and stickiness")
        if self.tool_contact_model == "coulomb-adhesive-v1" and self.tool_contact_absorption != 0:
            raise ValueError("coulomb-adhesive-v1 requires zero tool contact absorption")
        if self.tool_contact_model in BOUNDED_ADHESIVE_MODELS and (
                self.tool_contact_absorption != 0 or self.tool_stickiness != 0):
            raise ValueError(f"{self.tool_contact_model} requires zero tool contact absorption and stickiness")
        if self.tool_contact_model not in BOUNDED_ADHESIVE_MODELS and self.tool_adhesion != 0:
            raise ValueError(f"tool_adhesion requires one of {BOUNDED_ADHESIVE_MODELS}")
        if self.floor_contact_model == "coulomb-v1" and (self.floor_absorption != 0 or self.floor_stickiness != 0):
            raise ValueError("Floor coulomb-v1 requires zero floor absorption and stickiness")
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

    def _validate_grid_box(self):
        lower, cells = self.grid_box_lower, self.grid_box_cells
        if lower is None and cells is None:
            return
        if lower is None or cells is None:
            raise ValueError("grid_box_lower and grid_box_cells must be given together")
        values = []
        for name, value in (("grid_box_lower", lower), ("grid_box_cells", cells)):
            if (isinstance(value, (str, bytes)) or len(value) != 3
                    or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in value)):
                raise ValueError(f"{name} must be three integers")
            values.append(tuple(int(v) for v in value))
        lower, cells = values
        if min(cells) < 8:
            raise ValueError("grid_box_cells must be at least 8 on every axis")
        if min(lower) < 0 or any(a + b > self.grid for a, b in zip(lower, cells)):
            raise ValueError("the grid box must lie inside the full cube of grid cells")
        # The floor and the walls below it act at global node 0 as in the full cube.
        if lower[1] != 0:
            raise ValueError("grid_box_lower[1] must be 0 so the box contains the floor nodes")
        object.__setattr__(self, "grid_box_lower", lower)
        object.__setattr__(self, "grid_box_cells", cells)

    @property
    def box_lower(self) -> tuple[int, int, int]:
        return (0, 0, 0) if self.grid_box_lower is None else self.grid_box_lower

    @property
    def box_cells(self) -> tuple[int, int, int]:
        return (self.grid,) * 3 if self.grid_box_cells is None else self.grid_box_cells

    @property
    def effective_tool_grid_contact_padding(self) -> float:
        return self.tool_contact_padding if self.tool_grid_contact_padding is None else self.tool_grid_contact_padding

    @property
    def allocated_grid(self) -> int:
        return self.grid + (self.physics_version == "corrected-v1")

    @property
    def numpy_dtype(self) -> DTypeLike:
        return np.float32 if self.precision == "f32" else np.float64

    def record(self) -> dict:
        """asdict without unset grid-box settings."""
        values = asdict(self)
        for name in GRID_BOX_SETTINGS:
            if values[name] is None:
                values.pop(name)
            else:
                values[name] = list(values[name])
        return values


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


def quaternion_matrix(quaternions):
    """Rotation matrices [...,3,3] from xyzw quaternions [...,4]; the formula the solver uploads."""
    x, y, z, w = np.moveaxis(np.asarray(quaternions, dtype=np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def quaternion_matrix_vjp(quaternions, matrix_gradient):
    """dL/dq [...,4] for the raw (not renormalized) xyzw components, given dL/dR [...,3,3]."""
    x, y, z, w = np.moveaxis(np.asarray(quaternions, dtype=np.float64), -1, 0)
    G = np.asarray(matrix_gradient, dtype=np.float64)
    g = lambda i, j: G[..., i, j]
    return 2 * np.stack([
        y * (g(0, 1) + g(1, 0)) + z * (g(0, 2) + g(2, 0)) - w * (g(1, 2) - g(2, 1)) - 2 * x * (g(1, 1) + g(2, 2)),
        x * (g(0, 1) + g(1, 0)) + z * (g(1, 2) + g(2, 1)) + w * (g(0, 2) - g(2, 0)) - 2 * y * (g(0, 0) + g(2, 2)),
        x * (g(0, 2) + g(2, 0)) + y * (g(1, 2) + g(2, 1)) - w * (g(0, 1) - g(1, 0)) - 2 * z * (g(0, 0) + g(1, 1)),
        -z * (g(0, 1) - g(1, 0)) + y * (g(0, 2) - g(2, 0)) - x * (g(1, 2) - g(2, 1)),
    ], -1)


def rotation_vector_vjp(rotations, matrix_gradient):
    """dL/dθ [...,3] for a world-frame rotation R -> exp([θ]x) R at θ = 0, given dL/dR."""
    M = np.asarray(matrix_gradient, dtype=np.float64) @ np.swapaxes(np.asarray(rotations, dtype=np.float64), -1, -2)
    return np.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]], -1)


@dataclass
class ToolControlGradient:
    """Loss derivatives for one ToolControl, laid out like it.

    poses [2,7] holds dL/d(center xyz) and dL/d(raw quaternion xyzw); the quaternion
    part includes a component along q that a unit-norm update removes. rotation [2,3]
    is dL/dθ for a world-frame rotation exp([θ]x) R, the usual direction for
    updating orientation. velocities [2,6] holds dL/d(linear, angular velocity).
    """
    poses: np.ndarray
    velocities: np.ndarray
    rotation: np.ndarray

    @classmethod
    def zeros(cls):
        return cls(np.zeros((2, 7)), np.zeros((2, 6)), np.zeros((2, 3)))

    def __iadd__(self, other):
        self.poses += other.poses
        self.velocities += other.velocities
        self.rotation += other.rotation
        return self


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
    missing = set(LEGACY_PARAMETER_NAMES) - set(values)
    unknown = set(values) - set(PARAMETER_NAMES)
    if missing or unknown:
        raise ValueError(f"Invalid physical parameters: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite material/contact parameters")
    if values["youngs_modulus"] <= 0 or not -1 < values["poisson_ratio"] < 0.5:
        raise ValueError("Require E > 0 and -1 < nu < 0.5")
    if values["viscosity"] < 0 or not 0 < values["plastic_min"] <= 1 <= values["plastic_max"]:
        raise ValueError("Invalid viscosity or stretch limits")
    if not all(0 <= values[n] <= 1 for n in ("tool_retention", "floor_retention")):
        raise ValueError("Retention multipliers must be in [0,1]")
    if "tool_friction_coefficient" in values and (isinstance(values["tool_friction_coefficient"], (bool, np.bool_)) or
                                                 values["tool_friction_coefficient"] < 0):
        raise ValueError("tool_friction_coefficient must be finite and nonnegative")
    if "tool_stickiness" in values and (isinstance(values["tool_stickiness"], (bool, np.bool_)) or
                                      not 0 <= values["tool_stickiness"] <= 1):
        raise ValueError("tool_stickiness must be in [0,1]")
    for name in ("tool_adhesion", "floor_friction_coefficient"):
        if name in values and (isinstance(values[name], (bool, np.bool_)) or values[name] < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if YIELD_STRESS_PARAMETER in values and (isinstance(values[YIELD_STRESS_PARAMETER], (bool, np.bool_))
                                            or values[YIELD_STRESS_PARAMETER] <= 0):
        raise ValueError(f"{YIELD_STRESS_PARAMETER} must be finite and positive")


def normalize_parameters(values: Mapping[str, float], simulation=None):
    """Extend old seven-value records using their fixed simulation contact settings.

    Explicit physical contact values take precedence. Callers loading old files
    must supply their simulation settings rather than replace recorded friction
    with a new default.
    """
    validate_parameters(values)
    defaults = {}
    for name in SIMULATION_MIRRORED_PARAMETERS:
        if isinstance(simulation, Mapping):
            value = simulation.get(name)
        else:
            value = getattr(simulation, name, None)
        if value is None:
            value = INACTIVE_YIELD_STRESS_PA if name == YIELD_STRESS_PARAMETER else DEFAULT_PARAMETERS[name]
        defaults[name] = value
    result = {**defaults, **values}
    validate_parameters(result)
    return {name: float(result[name]) for name in PARAMETER_NAMES}


def validate_tool_parameters(values, config, fit=()):
    """Reject unsupported contact fits instead of returning identically zero gradients."""
    model = config.tool_contact_model
    fitted = set(fit)
    if model in ("coulomb-v1", *BOUNDED_ADHESIVE_MODELS) and values["tool_stickiness"] != 0:
        raise ValueError("Nonzero tool_stickiness requires retention-v1 or coulomb-adhesive-v1")
    if model not in BOUNDED_ADHESIVE_MODELS and values["tool_adhesion"] != 0:
        raise ValueError(f"Nonzero tool_adhesion requires one of {BOUNDED_ADHESIVE_MODELS}")
    if model not in BOUNDED_ADHESIVE_MODELS and "tool_adhesion" in fitted:
        raise ValueError(f"tool_adhesion is inactive outside {BOUNDED_ADHESIVE_MODELS}")
    if model in BOUNDED_ADHESIVE_MODELS and "tool_stickiness" in fitted:
        raise ValueError(f"{model} uses tool_adhesion instead of tool_stickiness")
    floor = getattr(config, "floor_contact_model", "retention-v1")
    if floor == "coulomb-v1" and "floor_retention" in fitted:
        raise ValueError("floor_retention is inactive with floor_contact_model='coulomb-v1'")
    if floor != "coulomb-v1" and "floor_friction_coefficient" in fitted:
        raise ValueError("floor_friction_coefficient requires floor_contact_model='coulomb-v1'")
    if fitted.intersection(set(TUNABLE_TOOL_PARAMETERS) - {"floor_friction_coefficient"}) and config.tool_collision != "sdf":
        raise ValueError("Fitting tool contact parameters requires tool_collision='sdf'")
    if YIELD_STRESS_PARAMETER in fitted and getattr(config, "plasticity", "none") != "von-mises":
        raise ValueError(f"{YIELD_STRESS_PARAMETER} is inactive without von-mises plasticity")
    if model == "retention-v1" and "tool_friction_coefficient" in fitted:
        raise ValueError("tool_friction_coefficient is inactive in retention-v1")
    if model == "coulomb-v1" and "tool_stickiness" in fitted:
        raise ValueError("Fitting tool_stickiness with Coulomb friction requires coulomb-adhesive-v1")
    if model != "retention-v1" and "tool_retention" in fitted:
        raise ValueError("tool_retention is inactive in Coulomb contact models")
    if model == "retention-v1" and "tool_stickiness" in fitted:
        if "tool_retention" in fitted:
            raise ValueError("Fit either tool_retention or tool_stickiness, not their redundant product")
        if values["tool_retention"] == 0 or config.tool_contact_absorption == 1:
            raise ValueError("Fixed retention/absorption makes tool_stickiness inactive")

"""Full-state, single-physics-step trajectories for a fixed-topology reduced model.

Every transition retains the control actually passed to MPM. Collection uses two
Taichi state slots, but the returned NumPy archive stores all states (25N values
per frame); use short trajectories when collecting large particle sets. Particle
order is fixed by a shared reference array, not inferred from the current cloud.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping

import numpy as np

from .state import (PARAMETER_NAMES, STATE_NAMES, ParticleState, SDFData,
                    SimulationConfig, ToolControl, normalize_parameters,
                    validate_parameters)

ARCHIVE_VERSION = 1
STATE_VALUES_PER_PARTICLE = 25
CONTROL_DIM = 36
CONTROL_ENCODING = "two-tools-position-rotation-matrix-linear-angular-velocity-v1"


def _real_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "fi" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def pack_state(state: ParticleState) -> np.ndarray:
    """Flatten x, v, C, F and Jp, with each whole field preceding the next."""
    state.validate()
    arrays = [_real_array(getattr(state, name), name) for name in STATE_NAMES]
    return np.concatenate([array.reshape(-1) for array in arrays])


def unpack_state(values, n_particles: int | None = None) -> ParticleState:
    values = _real_array(values, "Packed state")
    if values.ndim != 1 or not len(values) or len(values) % STATE_VALUES_PER_PARTICLE:
        raise ValueError("Packed state must be a nonempty vector of length 25N")
    n = len(values) // STATE_VALUES_PER_PARTICLE
    if n_particles is not None and n_particles != n:
        raise ValueError("Packed state particle count does not match")
    dimensions = ((n, 3), (n, 3), (n, 3, 3), (n, 3, 3), (n,))
    arrays = {}
    offset = 0
    for name, dimensions_for_field in zip(STATE_NAMES, dimensions):
        size = int(np.prod(dimensions_for_field))
        arrays[name] = values[offset:offset + size].reshape(dimensions_for_field).copy()
        offset += size
    return ParticleState(**arrays)


def encode_control(control: ToolControl) -> np.ndarray:
    """Encode xyzw poses without the quaternion q/-q ambiguity."""
    control.validate()
    poses = _real_array(control.poses, "Tool poses").astype(np.float64)
    velocities = _real_array(control.velocities, "Tool velocities").astype(np.float64)
    quaternion = poses[:, 3:]
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    x, y, z, w = quaternion.T
    rotation = np.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), axis=1)
    return np.concatenate((poses[:, :3], rotation, velocities), axis=1).reshape(CONTROL_DIM)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Metadata must be finite JSON-compatible values") from error


def _array_hash(values):
    values = np.ascontiguousarray(_real_array(values, "Scene array"), dtype="<f8")
    digest = hashlib.sha256()
    digest.update(_json(list(values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _scene_metadata(config, reference_x, sdf):
    reference_x = _real_array(reference_x, "Reference particle positions")
    if reference_x.shape != (config.n_particles, 3):
        raise ValueError("Reference particle positions must match configuration [N,3]")
    if config.tool_collision == "sdf":
        if sdf is None:
            raise ValueError("SDF collision requires SDF geometry")
        sdf.validate()
    elif sdf is not None:
        raise ValueError("SDF geometry supplied with collision disabled")
    geometry = None if sdf is None else {
        name: _array_hash(getattr(sdf, name))
        for name in ("distances", "gradients", "minimums", "spacings")
    }
    return {"simulation_config": asdict(config),
            "reference_positions_sha256": _array_hash(reference_x),
            "sdf_sha256": geometry,
            "state_names": list(STATE_NAMES),
            "parameter_names": list(PARAMETER_NAMES),
            "control_encoding": CONTROL_ENCODING,
            "coordinate_frame": "simulation-world"}


def scene_identity(config: SimulationConfig, reference_x, sdf: SDFData | None = None) -> str:
    """Hash numerical settings, ordered reference particles and fixed tool geometry.

    Material/contact parameters are separate transition inputs. All trajectories
    for one model must pass the same reference positions, even if their initial
    positions, velocities or internal deformations differ.
    """
    return hashlib.sha256(_json(_scene_metadata(config, reference_x, sdf)).encode()).hexdigest()


@dataclass
class Trajectory:
    states: np.ndarray
    controls: np.ndarray
    parameters: np.ndarray
    dt: float
    trajectory_id: str
    identity: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_particles(self):
        return self.states.shape[1] // STATE_VALUES_PER_PARTICLE

    @property
    def n_steps(self):
        return len(self.controls)

    def validate(self):
        states = _real_array(self.states, "Trajectory states")
        controls = _real_array(self.controls, "Trajectory controls")
        parameters = _real_array(self.parameters, "Trajectory parameters")
        if (states.ndim != 2 or states.shape[1] == 0 or
                states.shape[1] % STATE_VALUES_PER_PARTICLE):
            raise ValueError("Trajectory states must have dimensions [T+1,25N]")
        if controls.ndim != 2 or controls.shape[1] != CONTROL_DIM or len(controls) < 1:
            raise ValueError("Trajectory controls must have dimensions [T,36] with T > 0")
        if len(states) != len(controls) + 1:
            raise ValueError("Each control requires its before and after state")
        if parameters.shape != (len(PARAMETER_NAMES),):
            raise ValueError("Trajectory parameters require all nine ordered values")
        validate_parameters(dict(zip(PARAMETER_NAMES, parameters.tolist())))
        if isinstance(self.dt, (bool, np.bool_)) or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Trajectory dt must be finite and positive")
        for name in ("trajectory_id", "identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Trajectory {name} must be a nonempty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("Trajectory metadata must be a dictionary")
        _json(self.metadata)
        # Archive controls are rotation matrices, not arbitrary nine-value features.
        rotations = controls.reshape(-1, 2, 18)[:, :, 3:12].reshape(-1, 3, 3)
        if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5) or
                not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
            raise ValueError("Encoded control rotations must be proper rotation matrices")
        if "simulation_config" in self.metadata:
            config = SimulationConfig(**self.metadata["simulation_config"])
            if config.n_particles != self.n_particles or config.dt != self.dt:
                raise ValueError("Trajectory numerical settings do not match its state or dt")
            scene_keys = ("simulation_config", "reference_positions_sha256", "sdf_sha256",
                          "state_names", "parameter_names", "control_encoding", "coordinate_frame")
            if not all(key in self.metadata for key in scene_keys):
                raise ValueError("Incomplete scene identity metadata")
            scene_record = {key: self.metadata[key] for key in scene_keys}
            expected = hashlib.sha256(_json(scene_record).encode()).hexdigest()
            if self.identity != expected:
                raise ValueError("Trajectory scene metadata does not match its identity")
        if "state_names" in self.metadata and self.metadata["state_names"] != list(STATE_NAMES):
            raise ValueError("Unsupported state ordering")
        if ("parameter_names" in self.metadata and
                self.metadata["parameter_names"] != list(PARAMETER_NAMES)):
            raise ValueError("Unsupported parameter ordering")
        if ("control_encoding" in self.metadata and
                self.metadata["control_encoding"] != CONTROL_ENCODING):
            raise ValueError("Unsupported control encoding")
        return self

    def save(self, path):
        save_trajectory(path, self)

    @classmethod
    def load(cls, path, expected_identity=None):
        return load_trajectory(path, expected_identity=expected_identity)


def save_trajectory(path, trajectory: Trajectory):
    trajectory.validate()
    record = {"version": ARCHIVE_VERSION, "dt": float(trajectory.dt),
              "trajectory_id": trajectory.trajectory_id, "identity": trajectory.identity,
              "metadata": trajectory.metadata, "state_names": list(STATE_NAMES),
              "parameter_names": list(PARAMETER_NAMES), "control_encoding": CONTROL_ENCODING}
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, states=trajectory.states, controls=trajectory.controls,
                            parameters=trajectory.parameters, record=np.asarray(_json(record)))


def load_trajectory(path, expected_identity=None) -> Trajectory:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"states", "controls", "parameters", "record"}:
            raise ValueError("Invalid reduced-order trajectory archive fields")
        record_value = archive["record"]
        if record_value.shape != () or record_value.dtype.kind not in "US":
            raise ValueError("Trajectory record must be a scalar JSON string")
        try:
            record = json.loads(str(record_value.item()))
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid trajectory JSON record") from error
        if not isinstance(record, dict) or record.get("version") != ARCHIVE_VERSION:
            raise ValueError("Unsupported reduced-order trajectory archive version")
        if (record.get("state_names") != list(STATE_NAMES) or
                record.get("parameter_names") != list(PARAMETER_NAMES) or
                record.get("control_encoding") != CONTROL_ENCODING):
            raise ValueError("Unsupported trajectory field ordering or control encoding")
        required = {"dt", "trajectory_id", "identity", "metadata"}
        if not required.issubset(record):
            raise ValueError("Incomplete trajectory record")
        trajectory = Trajectory(archive["states"].copy(), archive["controls"].copy(),
                                archive["parameters"].copy(), record["dt"],
                                record["trajectory_id"], record["identity"], record["metadata"])
    trajectory.validate()
    if expected_identity is not None and trajectory.identity != expected_identity:
        raise ValueError("Trajectory belongs to a different scene or particle topology")
    return trajectory


def collect_trajectory(config: SimulationConfig, initial_state: ParticleState,
                       parameters: Mapping[str, float], controls: Iterable[ToolControl],
                       sdf: SDFData | None = None, trajectory_id: str = "trajectory",
                       reference_x=None) -> Trajectory:
    """Collect every exact MPM step; caller initializes Taichi with matching precision."""
    from .solver import Stepper

    initial_state.validate()
    if len(initial_state.x) != config.n_particles:
        raise ValueError("Initial state particle count does not match configuration")
    reference_x = initial_state.x if reference_x is None else reference_x
    metadata = _scene_metadata(config, reference_x, sdf)
    identity = hashlib.sha256(_json(metadata).encode()).hexdigest()
    values = normalize_parameters(parameters, config)
    controls = tuple(controls)
    if not controls:
        raise ValueError("Collection requires at least one physics step")
    features = np.stack([encode_control(control) for control in controls])
    origin = float(controls[0].time)
    times = np.asarray([control.time for control in controls], dtype=np.float64)
    if not np.allclose(times, origin + np.arange(len(controls)) * config.dt,
                       rtol=0, atol=max(1e-12, config.dt * 1e-6)):
        raise ValueError("Control timestamps must be consecutive physics-step times")
    metadata["initial_time_s"] = origin
    stepper = Stepper(config, values, capacity=2, sdf=sdf)
    stepper.load_state(0, initial_state)
    started = perf_counter()
    states = [pack_state(stepper.state(0))]
    for control in controls:
        stepper.advance(0, control)
        state = stepper.state(1)
        states.append(pack_state(state))
        stepper.load_state(0, state)
    metadata["collection_seconds"] = perf_counter() - started
    # First-use Taichi kernels can compile during the timed stepping/export loop.
    metadata["collection_timing"] = "stepping-export-reload; includes first-use kernel compilation"
    trajectory = Trajectory(np.stack(states), features,
                            np.asarray([values[name] for name in PARAMETER_NAMES]),
                            config.dt, trajectory_id, identity, metadata)
    return trajectory.validate()

"""Explicit, portable experiment configuration without modifying saved inputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .state import DEFAULT_PARAMETERS, PARAMETER_NAMES, SimulationConfig, validate_parameters


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SCHEMA = "taichidough/differentiable-mpm-experiment/v1"
REQUIRED_PATHS = ("episode", "calibration", "initial_particles", "reconstruction_metadata", "tool_geometry")
SDF_PATHS = ("collision_manifest", "ur_collision_mesh", "kinova_collision_mesh")
OPTIONAL_PATHS = ("source_manifest",)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrameWindow:
    start_frame: int
    end_frame: int
    stride: int = 1

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (self.start_frame, self.end_frame, self.stride)):
            raise ValueError("Window indices and stride must be integers")
        if not 1 <= self.start_frame <= self.end_frame or self.stride < 1:
            raise ValueError("Scored windows require 1 <= start <= end and stride >= 1")

    def indices(self):
        result = list(range(self.start_frame, self.end_frame + 1, self.stride))
        if result[-1] != self.end_frame:
            result.append(self.end_frame)
        return result


@dataclass(frozen=True)
class ObservationSettings:
    width: int = 160
    height: int = 120
    splat_radius: int = 1
    trim_quantile: float = 0.005
    strict_splat_radius: int = 1

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (self.width, self.height, self.splat_radius, self.strict_splat_radius)):
            raise ValueError("Observation dimensions and splat radii must be integers")
        if self.width < 1 or self.height < 1 or min(self.splat_radius, self.strict_splat_radius) < 0:
            raise ValueError("Invalid observation dimensions or splat radius")
        if not np.isfinite(self.trim_quantile) or not 0 <= self.trim_quantile < 0.5:
            raise ValueError("trim_quantile must be in [0,0.5)")


@dataclass
class ExperimentConfig:
    name: str
    paths: dict[str, Path]
    expected_sha256: dict[str, str]
    simulation: dict[str, Any]
    parameters: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMETERS))
    fit_parameters: list[str] = field(default_factory=lambda: ["youngs_modulus", "poisson_ratio", "viscosity"])
    parameter_bounds: dict[str, list[float]] = field(default_factory=dict)
    training: FrameWindow = field(default_factory=lambda: FrameWindow(1, 60))
    validation: FrameWindow = field(default_factory=lambda: FrameWindow(61, 97))
    observation: ObservationSettings = field(default_factory=ObservationSettings)
    loss: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    mass_kg: float = 0.25
    density_kg_m3: float = 283.071209393435
    replay_max_gap_s: float = 0.1
    tool_sdf_resolution: int = 64
    tool_mesh_scale: float = 0.001
    backend: str = "cpu"
    segment_length: int = 64
    seed: int = 0
    expected_sequence_fingerprint: str | None = None
    strict_loss: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None
    source_document_sha256: str | None = None
    path_overrides: dict[str, str] = field(default_factory=dict)

    def validate(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Experiment name is required")
        missing = set(REQUIRED_PATHS) - self.paths.keys()
        if self.simulation.get("tool_collision", "none") == "sdf":
            missing |= set(SDF_PATHS) - self.paths.keys()
        if missing:
            raise ValueError("Missing explicit input paths: " + ", ".join(sorted(missing)))
        unknown_paths = set(self.paths) - set(REQUIRED_PATHS + SDF_PATHS + OPTIONAL_PATHS)
        if unknown_paths:
            raise ValueError("Unknown input paths: " + ", ".join(sorted(unknown_paths)))
        for name, expected in self.expected_sha256.items():
            if name not in self.paths or name == "episode":
                raise ValueError(f"SHA-256 refers to an unknown file input {name}")
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise ValueError(f"Invalid expected SHA-256 for {name}")
        if self.expected_sequence_fingerprint is not None:
            h = self.expected_sequence_fingerprint
            if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("Invalid expected sequence fingerprint")
        known_sim = {f.name for f in fields(SimulationConfig)} - {"particle_mass", "particle_volume"}
        unknown = set(self.simulation) - known_sim
        if unknown:
            raise ValueError("Unknown/fitted mass settings in simulation: " + ", ".join(sorted(unknown)))
        for name in ("n_particles", "grid"):
            value = self.simulation.get(name, 48 if name == "grid" else None)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        SimulationConfig(**self.simulation)
        validate_parameters(self.parameters)
        if not self.fit_parameters or len(set(self.fit_parameters)) != len(self.fit_parameters):
            raise ValueError("fit_parameters must contain distinct parameter names")
        if set(self.fit_parameters) - set(PARAMETER_NAMES):
            raise ValueError("Unknown fitted parameter")
        if self.simulation.get("plasticity", "none") != "stretch-clamp" and set(self.fit_parameters) & {"plastic_min", "plastic_max"}:
            raise ValueError("Fitting plastic bounds requires explicit plasticity='stretch-clamp'")
        if set(self.parameter_bounds) - set(self.fit_parameters):
            raise ValueError("Parameter bounds must refer to fitted parameters")
        for name, bound in self.parameter_bounds.items():
            if len(bound) != 2 or not np.isfinite(bound).all() or not bound[0] < bound[1]:
                raise ValueError(f"Invalid bounds for {name}")
            if not bound[0] <= self.parameters[name] <= bound[1]:
                raise ValueError(f"Initial {name} lies outside its bounds")
            for endpoint in bound:
                candidate = {**self.parameters, name: endpoint}
                validate_parameters(candidate)
        if self.strict_loss.get("require_all_frames", True) is not True:
            raise ValueError("Strict evaluation must require all selected frames")
        if self.training.end_frame >= self.validation.start_frame:
            raise ValueError("Training and held-out windows must be ordered and disjoint")
        if self.backend not in {"cpu", "cuda", "vulkan"}:
            raise ValueError("backend must explicitly select cpu, cuda or vulkan")
        if isinstance(self.segment_length, bool) or not isinstance(self.segment_length, int) or self.segment_length < 1:
            raise ValueError("segment_length must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("mass_kg", "density_kg_m3", "replay_max_gap_s", "tool_mesh_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if isinstance(self.tool_sdf_resolution, bool) or not isinstance(self.tool_sdf_resolution, int) or self.tool_sdf_resolution < 16:
            raise ValueError("tool_sdf_resolution must be an integer >= 16")
        return self

    def window(self, split: str) -> FrameWindow:
        if split == "training":
            return self.training
        if split in {"validation", "held-out"}:
            return self.validation
        raise ValueError("split must be training or validation")

    def as_dict(self):
        result = asdict(self)
        result["schema"] = SCHEMA
        result["paths"] = {key: str(path) for key, path in self.paths.items()}
        result["config_path"] = None if self.config_path is None else str(self.config_path)
        return result

    @property
    def fingerprint(self):
        return canonical_hash(self.as_dict())


def load_config(path: str | Path, path_overrides: Mapping[str, str | Path] | None = None) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"Experiment configuration must use {SCHEMA}")
    value = copy.deepcopy(value)
    value.pop("schema")
    allowed = {f.name for f in fields(ExperimentConfig)} - {"config_path", "source_document_sha256", "path_overrides"}
    if set(value) - allowed:
        raise ValueError("Unknown experiment settings: " + ", ".join(sorted(set(value) - allowed)))
    raw_paths = value.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ValueError("paths must be an object of explicit input paths")
    overrides = dict(path_overrides or {})
    if set(overrides) - set(REQUIRED_PATHS + SDF_PATHS + OPTIONAL_PATHS):
        raise ValueError("Unknown path override")
    raw_paths.update(overrides)
    resolved = {}
    for key, text in raw_paths.items():
        p = Path(text).expanduser()
        resolved[key] = (p if p.is_absolute() else REPOSITORY_ROOT / p).resolve()
    value["paths"] = resolved
    value.setdefault("expected_sha256", {})
    for name in ("training", "validation"):
        if name in value:
            value[name] = FrameWindow(**value[name])
    if "observation" in value:
        value["observation"] = ObservationSettings(**value["observation"])
    value["parameters"] = {**DEFAULT_PARAMETERS, **value.get("parameters", {})}
    config = ExperimentConfig(**value, config_path=path, source_document_sha256=hashlib.sha256(raw).hexdigest(),
                              path_overrides={k: str(v) for k, v in overrides.items()})
    return config.validate()


def verify_input_paths(config: ExperimentConfig) -> dict[str, Any]:
    """Check exactly supplied paths; never search/rebase to a different input."""
    config.validate()
    records = {}
    for name, path in config.paths.items():
        if name == "episode":
            if not path.is_dir():
                raise FileNotFoundError(f"Explicit episode directory does not exist: {path}")
            records[name] = {"path": str(path)}
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Explicit {name} input does not exist: {path}")
        actual = file_sha256(path)
        expected = config.expected_sha256.get(name)
        if expected is not None and actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
        records[name] = {"path": str(path), "sha256": actual, "expected_sha256": expected}
    for filename in ("pointclouds_interpolated.pt", "paths_interpolated.pt"):
        path = config.paths["episode"] / filename
        if not path.is_file():
            raise FileNotFoundError(f"Recorded episode input does not exist: {path}")
        records[filename] = {"path": str(path), "sha256": file_sha256(path)}
    metadata = config.paths["episode"] / "sequence_metadata.json"
    if metadata.is_file():
        records["sequence_metadata"] = {"path": str(metadata), "sha256": file_sha256(metadata)}
    return records

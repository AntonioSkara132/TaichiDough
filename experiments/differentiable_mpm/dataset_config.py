"""Explicit shared-material datasets built from existing episode configurations."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from .config import ExperimentConfig, FrameWindow, canonical_hash, load_config
from .loss import LossConfig
from .optimize import AdamOptions
from .parameters import PhysicalParameterSpace
from .state import SimulationConfig, validate_parameters


SCHEMA = "taichidough/differentiable-dataset/v1"
MATERIAL_NAMES = ("youngs_modulus", "poisson_ratio", "viscosity", "plastic_min", "plastic_max")
TOOL_CONTACT = {"retention": 1.0, "absorption": 0.0, "stickiness": 0.0}
_EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _object(value, name, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"Invalid {name} fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _material_values(values):
    if not isinstance(values, Mapping) or set(values) != set(MATERIAL_NAMES):
        raise ValueError("Shared material values must specify exactly " + ", ".join(MATERIAL_NAMES))
    return {name: _number(values[name], name) for name in MATERIAL_NAMES}


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _resolved_path(value, base, name):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class DatasetEpisode:
    id: str
    config: ExperimentConfig
    config_path: Path
    membership: str
    weight: float
    scored_window: FrameWindow

    def parameters_for(self, shared_values):
        """Keep this episode's fixed floor parameter when inserting shared material."""
        values = {**self.config.parameters, **_material_values(shared_values), "tool_retention": 1.0}
        validate_parameters(values)
        return values

    def as_dict(self):
        return {"id": self.id, "config_path": str(self.config_path), "config": self.config.as_dict(),
                "membership": self.membership, "weight": self.weight,
                "scored_window": asdict(self.scored_window)}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    episodes: tuple[DatasetEpisode, ...]
    shared_initial: dict[str, float]
    fit_parameters: tuple[str, ...]
    parameter_bounds: dict[str, list[float]]
    config_path: Path
    source_document_sha256: str

    def parameter_space(self):
        first = self.episodes[0]
        return PhysicalParameterSpace(first.parameters_for(self.shared_initial), self.fit_parameters,
                                      first.config.simulation.get("plasticity", "none"),
                                      bounds=self.parameter_bounds)

    def as_dict(self):
        return {"schema": SCHEMA, "name": self.name,
                "shared_parameters": {"initial": dict(self.shared_initial), "fit": list(self.fit_parameters),
                                      "bounds": {key: list(value) for key, value in self.parameter_bounds.items()}},
                "tool_contact": dict(TOOL_CONTACT), "episodes": [episode.as_dict() for episode in self.episodes],
                "config_path": str(self.config_path), "source_document_sha256": self.source_document_sha256}

    @property
    def fingerprint(self):
        return canonical_hash(self.as_dict())


def _common_settings(config):
    numerical = asdict(SimulationConfig(**config.simulation))
    # These values describe an episode's discretization, reconstruction or floor.
    for name in tuple(numerical):
        if name in {"n_particles", "grid", "dt", "particle_mass", "particle_volume"} or name.startswith("floor_"):
            numerical.pop(name)
    return {"simulation": numerical, "backend": config.backend,
            "observation": asdict(config.observation), "loss": LossConfig(**config.loss).as_dict(),
            "strict_loss": config.strict_loss, "optimizer": asdict(AdamOptions(**config.optimizer))}


def load_dataset(path, *, path_overrides=None, backend=None, precision=None, p2g_mode=None,
                 physics_version=None, segment_length=None):
    """Resolve and validate a manifest without reading recordings or changing files.

    Episode-config references and input overrides are relative to the dataset
    manifest. Inputs inside an episode config retain load_config's existing path
    rules. Runtime overrides apply to every episode before compatibility checks.
    """
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    document = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    _object(document, "dataset", {"schema", "name", "shared_parameters", "tool_contact", "episodes"},
            {"schema", "name", "shared_parameters", "episodes"})
    if document["schema"] != SCHEMA:
        raise ValueError(f"Dataset manifest must use {SCHEMA}")
    if not isinstance(document["name"], str) or not document["name"].strip():
        raise ValueError("Dataset name is required")
    shared = _object(document["shared_parameters"], "shared_parameters", {"initial", "fit", "bounds"}, {"initial", "fit"})
    initial = _material_values(shared["initial"])
    fit = shared["fit"]
    if (not isinstance(fit, list) or not fit or not all(isinstance(name, str) for name in fit)
            or len(set(fit)) != len(fit) or set(fit) - set(MATERIAL_NAMES)):
        raise ValueError("Shared fit must list distinct material parameter names only")
    raw_bounds = _object(shared.get("bounds", {}), "shared bounds", fit)
    bounds = {}
    for name, values in raw_bounds.items():
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"Shared {name} bounds must be two finite numbers")
        bounds[name] = [_number(value, f"{name} bound") for value in values]
    contact = _object(document.get("tool_contact", {}), "tool_contact", TOOL_CONTACT)
    for name, expected in TOOL_CONTACT.items():
        if _number(contact.get(name, expected), f"tool_contact.{name}") != expected:
            raise ValueError("The first dataset setup requires retention=1, absorption=0 and stickiness=0")
    rows = document["episodes"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset episodes must be a nonempty list")
    if path_overrides is None:
        path_overrides = {}
    if not isinstance(path_overrides, Mapping) or not all(isinstance(key, str) for key in path_overrides):
        raise ValueError("Dataset path overrides must map episode IDs to input-path mappings")
    for name, value in (("backend", backend), ("precision", precision), ("p2g_mode", p2g_mode),
                        ("physics_version", physics_version)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} override must be a string")
    episodes = []
    ids = set()
    recordings = set()
    fingerprints = set()
    common = None
    for row in rows:
        _object(row, "episode", {"id", "config", "membership", "weight", "scored_window", "path_overrides"},
                {"id", "config", "membership", "scored_window"})
        episode_id = row["id"]
        if not isinstance(episode_id, str) or _EPISODE_ID.fullmatch(episode_id) is None:
            raise ValueError("Episode IDs must be safe nonempty names containing letters, digits, _, - or .")
        if episode_id in ids:
            raise ValueError(f"Duplicate episode ID: {episode_id}")
        ids.add(episode_id)
        membership = row["membership"]
        if not isinstance(membership, str) or membership not in {"training", "validation"}:
            raise ValueError(f"Episode {episode_id} membership must be training or validation")
        weight = _number(row.get("weight", 1.0), f"Episode {episode_id} weight")
        if weight <= 0:
            raise ValueError(f"Episode {episode_id} weight must be positive")
        window = _object(row["scored_window"], "scored_window", {"start_frame", "end_frame", "stride"},
                         {"start_frame", "end_frame"})
        scored_window = FrameWindow(**window)
        config_path = _resolved_path(row["config"], path.parent, "Episode config")
        local_overrides = row.get("path_overrides", {})
        external_overrides = path_overrides.get(episode_id, {})
        if not isinstance(local_overrides, dict) or not isinstance(external_overrides, Mapping):
            raise ValueError(f"Episode {episode_id} path overrides must be input-path mappings")
        overrides = {**local_overrides, **external_overrides}
        if not all(isinstance(name, str) for name in overrides):
            raise ValueError(f"Episode {episode_id} input-path names must be strings")
        overrides = {name: _resolved_path(value, path.parent, f"{episode_id}.{name}")
                     for name, value in overrides.items()}
        config = load_config(config_path, overrides)
        config.parameters = {**config.parameters, **initial, "tool_retention": 1.0}
        config.fit_parameters = list(fit)
        config.parameter_bounds = {name: list(values) for name, values in bounds.items()}
        config.simulation.update(tool_contact_absorption=0.0, tool_stickiness=0.0)
        for name, value in (("precision", precision), ("p2g_mode", p2g_mode), ("physics_version", physics_version)):
            if value is not None:
                config.simulation[name] = value
        config.simulation.setdefault("physics_version", "corrected-v1")
        if backend is not None:
            config.backend = backend
        if segment_length is not None:
            config.segment_length = segment_length
        config.validate()
        PhysicalParameterSpace(config.parameters, fit, config.simulation.get("plasticity", "none"), bounds=bounds)
        recording = config.paths["episode"]
        if recording in recordings:
            raise ValueError(f"Duplicate recording path for episode {episode_id}: {recording}")
        recordings.add(recording)
        fingerprint = config.expected_sequence_fingerprint
        if fingerprint is not None:
            if fingerprint in fingerprints:
                raise ValueError(f"Duplicate expected recording fingerprint for episode {episode_id}")
            fingerprints.add(fingerprint)
        current = _common_settings(config)
        if common is None:
            common = current
        elif current != common:
            differing = [name for name in common if current[name] != common[name]]
            raise ValueError(f"Episode {episode_id} has incompatible shared settings: {', '.join(differing)}")
        episodes.append(DatasetEpisode(episode_id, config, config_path, membership, weight, scored_window))
    unknown_overrides = set(path_overrides) - ids
    if unknown_overrides:
        raise ValueError("Path overrides name unknown episodes: " + ", ".join(sorted(unknown_overrides)))
    if not any(episode.membership == "training" for episode in episodes):
        raise ValueError("Dataset requires at least one training episode")
    return DatasetConfig(document["name"], tuple(episodes), initial, tuple(fit), bounds, path,
                         hashlib.sha256(raw).hexdigest())

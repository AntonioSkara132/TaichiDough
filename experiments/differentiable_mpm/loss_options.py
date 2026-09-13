"""Explicit configuration and construction of optional paper-style losses."""
from dataclasses import dataclass, fields
import hashlib
import json
from numbers import Integral, Real
from typing import Mapping

import numpy as np

from .loss import LOSS_VERSION, SUPPORTED_LOSS_VERSIONS, LossConfig


DPSI_LOSS_VERSIONS = (
    "dpsi-pcd-cd-v1", "dpsi-prt-cd-v1",
    "dpsi-pcd-emd-v1", "dpsi-prt-emd-v1",
)
EMPM_LOSS_VERSIONS = ("empm-offline-v1", "empm-mask-inspired-v1")
PAPER_LOSS_VERSIONS = DPSI_LOSS_VERSIONS + EMPM_LOSS_VERSIONS


def _allowed_fields(version):
    common = {"version", "target_source", "target_sample_count",
              "predicted_sample_count", "sampling_seed", "min_target_points"}
    if "-pcd-" in version or version in EMPM_LOSS_VERSIONS:
        common.add("target_voxel_size_m")
    if "-emd-" in version:
        common.update(("max_assignment_pairs", "max_assignment_bytes"))
    if version == "empm-offline-v1":
        common.update(("geometric_weight", "tracking_weight", "empty_track_policy"))
    if version == "empm-mask-inspired-v1":
        common.update(("geometric_weight", "mask_weight", "mask_epsilon",
                       "footprint_radius", "visibility_temperature_m", "opacity_gain"))
    return common


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class PaperLossConfig:
    version: str
    target_source: str | None = None
    target_voxel_size_m: float = 0.0
    target_sample_count: int | None = None
    predicted_sample_count: int | None = None
    sampling_seed: int = 0
    max_assignment_pairs: int = 4_000_000
    max_assignment_bytes: int = 256 * 1024 * 1024
    min_target_points: int = 1
    geometric_weight: float = 1.0
    tracking_weight: float | None = None
    empty_track_policy: str = "error"
    mask_weight: float | None = None
    mask_epsilon: float = 1e-8
    footprint_radius: int = 2
    visibility_temperature_m: float = 0.005
    opacity_gain: float = 8.0

    def __post_init__(self):
        if self.version not in PAPER_LOSS_VERSIONS:
            raise ValueError(f"Unsupported paper loss version: {self.version}")
        volume = "-prt-" in self.version
        if self.target_source is None:
            object.__setattr__(self, "target_source", "external" if volume else "recorded_cloud")
        if self.target_source not in {"recorded_cloud", "external"}:
            raise ValueError("target_source must be recorded_cloud or external")
        if volume and self.target_source != "external":
            raise ValueError("DPSI PRT requires an external reconstructed-volume target")
        if self.tracking_weight is None:
            object.__setattr__(self, "tracking_weight", 1.0 if self.version == "empm-offline-v1" else 0.0)
        if self.mask_weight is None:
            object.__setattr__(self, "mask_weight", 1.0 if self.version == "empm-mask-inspired-v1" else 0.0)
        for name in ("target_sample_count", "predicted_sample_count"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(value, name))
        for name in ("max_assignment_pairs", "max_assignment_bytes", "min_target_points", "footprint_radius"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "sampling_seed", _integer(self.sampling_seed, "sampling_seed", 0))
        if self.footprint_radius > 8:
            raise ValueError("footprint_radius must be in [1,8]")
        for name in ("target_voxel_size_m", "geometric_weight", "tracking_weight", "mask_weight"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        for name in ("mask_epsilon", "visibility_temperature_m", "opacity_gain"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        if self.empty_track_policy not in {"error", "skip"}:
            raise ValueError("empty_track_policy must be error or skip")
        allowed = _allowed_fields(self.version)
        defaults = {
            "target_voxel_size_m": 0.0, "max_assignment_pairs": 4_000_000,
            "max_assignment_bytes": 256 * 1024 * 1024, "geometric_weight": 1.0,
            "tracking_weight": 0.0, "empty_track_policy": "error", "mask_weight": 0.0,
            "mask_epsilon": 1e-8, "footprint_radius": 2,
            "visibility_temperature_m": 0.005, "opacity_gain": 8.0,
        }
        for name, default in defaults.items():
            if name not in allowed and getattr(self, name) != default:
                raise ValueError(f"{name} is inactive for {self.version}")
        if self.version == "empm-offline-v1" and self.geometric_weight + self.tracking_weight <= 0:
            raise ValueError("At least one EMPM geometric/tracking weight must be positive")
        if self.version == "empm-mask-inspired-v1" and self.mask_weight <= 0:
            raise ValueError("The mask-inspired mode requires positive mask_weight")

    def as_dict(self):
        allowed = _allowed_fields(self.version)
        return {field.name: getattr(self, field.name) for field in fields(self) if field.name in allowed}

    def fingerprint(self, camera):
        record = {"loss": self.as_dict(), "camera": camera.as_dict()}
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()


def parse_loss_config(mapping):
    """Keep existing partial configurations exact; validate new modes separately."""
    if not isinstance(mapping, Mapping):
        raise ValueError("loss must be a mapping")
    values = dict(mapping)
    version = values.get("version", LOSS_VERSION)
    if version in SUPPORTED_LOSS_VERSIONS:
        return LossConfig(**values)
    if version not in PAPER_LOSS_VERSIONS:
        raise ValueError(f"Unsupported training loss version: {version}")
    extra = set(values) - _allowed_fields(version)
    if extra:
        raise ValueError(f"Unsupported settings for {version}: {sorted(extra)}")
    return PaperLossConfig(**values)


def is_paper_loss(config):
    return isinstance(config, PaperLossConfig)


def loss_temporal_reduction(config):
    return "sum" if is_paper_loss(config) and config.version in EMPM_LOSS_VERSIONS else "mean"


def make_observation_loss(camera, config, initial_positions, precision="f32"):
    if is_paper_loss(config):
        from .point_set_loss import PointSetLoss
        return PointSetLoss(camera, config, initial_positions, precision=precision)
    from .loss import ObservationLoss
    return ObservationLoss(camera, config, initial_positions, precision=precision)

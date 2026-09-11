"""Partial-observation training loss with explicit Taichi position adjoints.

This versioned objective is separate from the unchanged strict evaluator.
Pixel support and nearest visible-particle associations are fixed during each
local derivative evaluation. Missing observation pixels are unknown. Current
predicted non-overlap keeps all observed coverage and depth penalties, while a
directed distance term supplies derivatives of continuous predicted coordinates.

Version 2 blends each supported depth residual with its missing-depth penalty
using predicted coverage. Thus the depth contribution approaches the declared
missing penalty when the last compact splat fades out. Version 1 retains the
unweighted supported-depth residual for reproduction of earlier experiments.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree
import taichi as ti

from .renderer import Camera, LocalSplatRenderer
from .state import InvalidStateError


LOSS_VERSION = "partial-visible-splats-v2"
SUPPORTED_LOSS_VERSIONS = ("partial-visible-splats-v1", LOSS_VERSION)


@dataclass(frozen=True)
class LossConfig:
    footprint_radius: int = 2
    visibility_temperature_m: float = 0.005
    opacity_gain: float = 8.0
    visible_cutoff_temperatures: float = 2.0
    depth_scale_m: float = 0.005
    distance_scale_m: float = 0.005
    huber_delta: float = 1.0
    depth_weight: float = 1.0
    coverage_weight: float = 0.5
    distance_weight: float = 0.5
    min_observed_pixels: int = 4
    min_depth_pixels: int = 1
    min_predicted_pixels: int = 1
    max_observed_points: int = 1024
    missing_depth_residual: float = 4.0
    version: str = LOSS_VERSION

    def __post_init__(self):
        if self.version not in SUPPORTED_LOSS_VERSIONS:
            raise ValueError(f"Unsupported training loss version: {self.version}")
        positive = ("visibility_temperature_m", "opacity_gain", "visible_cutoff_temperatures",
                    "depth_scale_m", "distance_scale_m", "huber_delta", "missing_depth_residual")
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        weights = (self.depth_weight, self.coverage_weight, self.distance_weight)
        if not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("Loss weights must be finite, nonnegative and not all zero")
        for name in ("min_observed_pixels", "min_depth_pixels", "min_predicted_pixels", "max_observed_points"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.footprint_radius, int) or not 1 <= self.footprint_radius <= 8:
            raise ValueError("footprint_radius must be an integer in [1,8]")

    def as_dict(self):
        return asdict(self)

    def fingerprint(self, camera):
        record = {"loss": self.as_dict(), "camera": camera.as_dict()}
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass
class Observation:
    frame_index: int
    step: int
    observed_depth: np.ndarray
    observed_valid: np.ndarray
    observed_initial_depth: np.ndarray
    observed_initial_valid: np.ndarray
    observed_points: np.ndarray | None = None
    timestamp: float = 0.0

    @property
    def source_frame(self):
        return self.frame_index

    @property
    def completed_substeps(self):
        return self.step

    @property
    def depth(self):
        return self.observed_depth

    @property
    def valid(self):
        return self.observed_valid


@dataclass
class LossRecord:
    value: float
    gradient: np.ndarray | None
    components: dict
    diagnostics: dict


@ti.data_oriented
class ObservationLoss:
    def __init__(self, camera: Camera, config: LossConfig, initial_positions, precision="f32"):
        self.camera = camera
        self.config = config
        self.renderer = LocalSplatRenderer(camera, len(initial_positions), config.footprint_radius,
                                           config.visibility_temperature_m, config.opacity_gain, precision)
        self.fp = self.renderer.fp
        self.numpy_dtype = self.renderer.numpy_dtype
        initial = self.renderer.forward(initial_positions)
        if int(initial.valid.sum()) < config.min_predicted_pixels:
            raise InvalidStateError("Initial simulation has insufficient predicted image support")
        self.initial_render = initial
        dimensions = (camera.height, camera.width)
        self.observed_depth = ti.field(self.fp, dimensions)
        self.observed_initial_depth = ti.field(self.fp, dimensions)
        self.initial_simulation_depth = ti.field(self.fp, dimensions)
        self.observed_mask = ti.field(ti.i32, dimensions)
        self.depth_mask = ti.field(ti.i32, dimensions)
        self.initial_simulation_depth.from_numpy(np.ascontiguousarray(initial.depth))
        self.point_targets = ti.Vector.field(3, self.fp, config.max_observed_points)
        self.point_particle_ids = ti.field(ti.i32, config.max_observed_points)
        self.components_field = ti.field(self.fp, shape=3, needs_grad=True)
        self.total = ti.field(self.fp, shape=(), needs_grad=True)
        weights = np.array([config.depth_weight, config.coverage_weight, config.distance_weight], dtype=np.float64)
        self.weights = tuple((weights / weights.sum()).tolist())

    @ti.func
    def _huber(self, residual):
        value = 0.5 * residual * residual
        if ti.abs(residual) > self.config.huber_delta:
            value = self.config.huber_delta * (ti.abs(residual) - 0.5 * self.config.huber_delta)
        return value

    @ti.kernel
    def _pixel_loss(self, observed_count: ti.i32, depth_count: ti.i32):
        for row, column in self.observed_mask:
            if self.observed_mask[row, column] == 1:
                self.components_field[1] += (1.0 - self.renderer.coverage[row, column]) / ti.cast(observed_count, self.fp)
            if self.depth_mask[row, column] == 1:
                term = self._huber(ti.cast(self.config.missing_depth_residual, self.fp))
                if self.renderer.valid[row, column] == 1:
                    observed_change = self.observed_depth[row, column] - self.observed_initial_depth[row, column]
                    simulation_change = self.renderer.depth[row, column] - self.initial_simulation_depth[row, column]
                    term = self._huber((simulation_change - observed_change) / self.config.depth_scale_m)
                if ti.static(self.config.version == "partial-visible-splats-v2"):
                    coverage = self.renderer.coverage[row, column]
                    missing = self._huber(ti.cast(self.config.missing_depth_residual, self.fp))
                    term = coverage * term + (1.0 - coverage) * missing
                self.components_field[0] += term / ti.cast(depth_count, self.fp)

    @ti.kernel
    def _point_loss(self, point_count: ti.i32):
        for index in range(point_count):
            particle = self.point_particle_ids[index]
            residual = (self.renderer.optical[particle] - self.point_targets[index]) / self.config.distance_scale_m
            squared = residual.dot(residual)
            term = 0.5 * squared
            if squared > self.config.huber_delta ** 2:
                term = self.config.huber_delta * (ti.sqrt(squared) - 0.5 * self.config.huber_delta)
            self.components_field[2] += term / ti.cast(point_count, self.fp)

    @ti.kernel
    def _combine(self):
        for index in ti.static(range(3)):
            self.total[None] += ti.static(self.weights[index]) * self.components_field[index]

    def _validated_observation(self, observation):
        expected = (self.camera.height, self.camera.width)
        arrays = [np.asarray(observation.observed_depth, dtype=self.numpy_dtype),
                  np.asarray(observation.observed_initial_depth, dtype=self.numpy_dtype)]
        masks = [np.asarray(observation.observed_valid, dtype=bool),
                 np.asarray(observation.observed_initial_valid, dtype=bool)]
        if any(a.shape != expected for a in arrays + masks):
            raise InvalidStateError("Observation arrays must match the training camera dimensions")
        if observation.frame_index < 0 or observation.step < 0 or not np.isfinite(observation.timestamp):
            raise InvalidStateError("Invalid observation frame, integration step or timestamp")
        for depth, mask in zip(arrays, masks):
            if (not np.isfinite(depth[mask]).all() or np.any(depth[mask] <= self.camera.near_m)
                    or np.any(depth[mask] >= self.camera.far_m)):
                raise InvalidStateError("Observed valid depth must be finite and strictly between camera near/far planes")
        observed_count = int(masks[0].sum())
        if observed_count < self.config.min_observed_pixels:
            raise InvalidStateError(f"Observed support {observed_count} is below {self.config.min_observed_pixels}")
        depth_mask = masks[0] & masks[1] & self.initial_render.valid
        depth_count = int(depth_mask.sum())
        if self.config.depth_weight > 0 and depth_count < self.config.min_depth_pixels:
            raise InvalidStateError(f"Fixed initial-referenced depth support {depth_count} is below {self.config.min_depth_pixels}")
        points = observation.observed_points
        if points is None:
            points = self.camera.optical_points(arrays[0], masks[0])
        points = np.asarray(points, dtype=self.numpy_dtype)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
            raise InvalidStateError("Observed optical points must be a nonempty [M,3] array")
        if (not np.isfinite(points).all() or np.any(points[:, 2] <= self.camera.near_m)
                or np.any(points[:, 2] >= self.camera.far_m)):
            raise InvalidStateError("Observed optical points must be finite and strictly between camera near/far planes")
        if len(points) > self.config.max_observed_points:
            samples = np.linspace(0, len(points) - 1, self.config.max_observed_points, dtype=np.int64)
            points = points[samples]
        return arrays, masks, depth_mask, observed_count, depth_count, points

    def _visible_particle_ids(self, rendered):
        active_ids = np.flatnonzero(rendered.active_particles)
        if len(active_ids) == 0:
            raise InvalidStateError("All predicted particles are offscreen or outside camera near/far planes")
        pixels = np.floor(rendered.projected_positions[active_ids]).astype(np.int64)
        front = rendered.front_depth[pixels[:, 1], pixels[:, 0]]
        z = rendered.optical_positions[active_ids, 2]
        cutoff = self.config.visible_cutoff_temperatures * self.config.visibility_temperature_m
        visible = active_ids[z - front <= cutoff]
        if len(visible) == 0:
            raise InvalidStateError("No visible predicted particles for directed observation loss")
        return visible

    def value_and_grad_positions(self, positions, observation, compute_grad=True):
        arrays, masks, depth_mask, observed_count, depth_count, points = self._validated_observation(observation)
        rendered = self.renderer.forward(positions)
        predicted_count = int(rendered.valid.sum())
        if predicted_count < self.config.min_predicted_pixels:
            raise InvalidStateError(f"Predicted support {predicted_count} is below {self.config.min_predicted_pixels}")
        if not np.isfinite(rendered.depth[rendered.valid]).all() or not np.isfinite(rendered.coverage).all():
            raise InvalidStateError("Nonfinite differentiated render")
        visible = self._visible_particle_ids(rendered)
        # Associations are discrete data; residual coordinates and their derivatives
        # are evaluated below in Taichi, not in this nearest-neighbor query.
        tree = cKDTree(rendered.optical_positions[visible].astype(np.float64))
        _, nearest = tree.query(points.astype(np.float64), k=1)
        particle_ids = visible[np.asarray(nearest, dtype=np.int64)]
        target_buffer = np.zeros((self.config.max_observed_points, 3), dtype=self.numpy_dtype)
        id_buffer = np.zeros(self.config.max_observed_points, dtype=np.int32)
        target_buffer[:len(points)] = points
        id_buffer[:len(points)] = particle_ids
        self.point_targets.from_numpy(target_buffer)
        self.point_particle_ids.from_numpy(id_buffer)
        self.observed_depth.from_numpy(np.ascontiguousarray(np.where(masks[0], arrays[0], 0), dtype=self.numpy_dtype))
        self.observed_initial_depth.from_numpy(np.ascontiguousarray(np.where(masks[1], arrays[1], 0), dtype=self.numpy_dtype))
        self.observed_mask.from_numpy(np.ascontiguousarray(masks[0], dtype=np.int32))
        self.depth_mask.from_numpy(np.ascontiguousarray(depth_mask, dtype=np.int32))
        self.components_field.fill(0)
        self.total[None] = 0
        self._pixel_loss(observed_count, max(depth_count, 1))
        self._point_loss(len(points))
        self._combine()
        value = float(self.total[None])
        components = dict(zip(("depth_change", "positive_coverage", "observed_to_visible_distance"),
                              map(float, self.components_field.to_numpy())))
        if not np.isfinite(value) or not all(np.isfinite(list(components.values()))):
            raise InvalidStateError("Nonfinite observation objective")
        gradient = None
        if compute_grad:
            self.renderer.clear_gradients()
            self.components_field.grad.fill(0)
            self.total.grad[None] = 1
            self._combine.grad()
            self._point_loss.grad(len(points))
            self._pixel_loss.grad(observed_count, max(depth_count, 1))
            gradient = self.renderer.reverse()
            if not np.isfinite(gradient).all():
                raise InvalidStateError("Nonfinite observation position derivative")
        diagnostics = {
            "loss_version": self.config.version, "frame_index": int(observation.frame_index),
            "step": int(observation.step), "observed_pixels": observed_count,
            "fixed_depth_pixels": depth_count, "predicted_pixels": predicted_count,
            "current_overlap_pixels": int(np.count_nonzero(masks[0] & rendered.valid)),
            "visible_predicted_particles": int(len(visible)), "directed_points": int(len(points)),
            "active_predicted_particles": int(rendered.active_particles.sum()),
            "association_derivative": "piecewise selected visible particle; continuous coordinates",
        }
        return LossRecord(value, gradient, components, diagnostics)

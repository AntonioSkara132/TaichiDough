"""Paper-style point objectives with explicit piecewise position derivatives.

DPSI uses endpoint, unsquared sums. EMPM's squared mean Chamfer and soft-IoU
mask here are documented choices where the paper does not define the operator.
Discrete nearest-neighbor/assignment choices are recomputed at each evaluation.
"""
from dataclasses import dataclass, field
import hashlib
from numbers import Integral

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .loss import LossRecord
from .loss_options import PaperLossConfig, DPSI_LOSS_VERSIONS
from .state import InvalidStateError


@dataclass
class PointSetObservation:
    frame_index: int
    step: int
    points_scene: np.ndarray
    timestamp: float = 0.0
    target_representation: str = "partial_observed"
    track_particle_ids: np.ndarray | None = None
    track_positions_scene: np.ndarray | None = None
    track_valid: np.ndarray | None = None
    foreground_mask: np.ndarray | None = None
    known_mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def source_frame(self):
        return self.frame_index

    @property
    def completed_substeps(self):
        return self.step


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _points(array, name, *, allow_empty=False):
    try:
        points = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain metric coordinates") from error
    if points.ndim != 2 or points.shape[1] != 3 or (not allow_empty and not len(points)):
        raise ValueError(f"{name} must be {'possibly empty ' if allow_empty else 'nonempty '}[N,3] coordinates")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} has nonfinite coordinates")
    return points


def _binary_mask(array, expected, name):
    values = np.asarray(array)
    if values.shape != expected or values.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a binary array with dimensions {expected}")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must contain only zero or one")
    return values.astype(bool, copy=False)


def sampled_indices(size, count, seed):
    """A supplied count is a cap; retain all indices if the set is smaller."""
    if count is None or count >= size:
        return np.arange(size, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(size, count, replace=False)).astype(np.int64)


def voxel_centroids(points, voxel_size):
    """Use lexicographically ordered, globally anchored voxel centroids."""
    if voxel_size == 0 or not len(points):
        return points.copy()
    scaled = points / voxel_size
    if not np.isfinite(scaled).all() or np.max(np.abs(scaled)) >= np.iinfo(np.int64).max // 2:
        raise ValueError("Target voxel coordinates exceed the integer range")
    cells = np.floor(scaled).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    result = sums / counts[:, None]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite target voxel centroids")
    return result


def _unit_vectors(delta):
    distance = np.linalg.norm(delta, axis=1)
    unit = np.zeros_like(delta)
    np.divide(delta, distance[:, None], out=unit, where=distance[:, None] > 0)
    return distance, unit


def chamfer_value_and_grad(predicted, target, *, squared=False, compute_grad=True):
    """Bidirectional unsquared sums, or bidirectional squared means."""
    predicted = _points(predicted, "predicted points")
    target = _points(target, "target points")
    _, target_to_predicted = cKDTree(predicted).query(target, k=1, workers=1)
    _, predicted_to_target = cKDTree(target).query(predicted, k=1, workers=1)
    target_to_predicted = np.asarray(target_to_predicted, dtype=np.int64)
    predicted_to_target = np.asarray(predicted_to_target, dtype=np.int64)
    target_delta = predicted[target_to_predicted] - target
    predicted_delta = predicted - target[predicted_to_target]
    if squared:
        toward_target = float(np.mean(np.einsum("ij,ij->i", target_delta, target_delta)))
        toward_prediction = float(np.mean(np.einsum("ij,ij->i", predicted_delta, predicted_delta)))
        target_derivative = 2.0 * target_delta / len(target)
        predicted_derivative = 2.0 * predicted_delta / len(predicted)
    else:
        target_distance, target_derivative = _unit_vectors(target_delta)
        predicted_distance, predicted_derivative = _unit_vectors(predicted_delta)
        toward_target = float(np.sum(target_distance))
        toward_prediction = float(np.sum(predicted_distance))
    gradient = None
    if compute_grad:
        gradient = predicted_derivative.copy()
        np.add.at(gradient, target_to_predicted, target_derivative)
    components = {"chamfer_target_to_simulation": toward_target,
                  "chamfer_simulation_to_target": toward_prediction}
    diagnostics = {"target_to_simulation_ids_sha256": _array_hash(target_to_predicted),
                   "simulation_to_target_ids_sha256": _array_hash(predicted_to_target)}
    return toward_target + toward_prediction, gradient, components, diagnostics


def assignment_memory_estimate(n_target, n_predicted):
    # The estimate includes four float64 cost-sized buffers plus linear work.
    return int(n_target) * int(n_predicted) * 8 * 4 + 64 * (int(n_target) + int(n_predicted))


def validate_assignment_size(n_target, n_predicted, *, max_pairs, max_bytes):
    """Validate cardinality and estimated workspace without allocating costs."""
    if n_target > n_predicted:
        raise ValueError("DPSI assignment requires n_target <= n_simulated; sets are not swapped")
    pairs = int(n_target) * int(n_predicted)
    estimate = assignment_memory_estimate(n_target, n_predicted)
    if pairs > max_pairs or estimate > max_bytes:
        raise ValueError(f"Assignment allocation refused: {pairs} pairs, estimated {estimate} bytes; "
                         f"limits are {max_pairs} pairs and {max_bytes} bytes. "
                         "Choose explicit target/predicted sampling or different limits.")
    return {"assignment_pairs": pairs, "assignment_estimated_bytes": estimate,
            "unmatched_simulated_points": int(n_predicted) - int(n_target)}


def assignment_value_and_grad(predicted, target, *, max_pairs, max_bytes, compute_grad=True):
    """Exact rectangular target-to-simulation assignment on the supplied sets."""
    predicted = _points(predicted, "predicted points")
    target = _points(target, "target points")
    n_target, n_predicted = len(target), len(predicted)
    diagnostics = validate_assignment_size(n_target, n_predicted, max_pairs=max_pairs, max_bytes=max_bytes)
    cost = cdist(target, predicted, metric="euclidean")
    if not np.isfinite(cost).all():
        raise InvalidStateError("Nonfinite assignment distances")
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != n_target or not np.array_equal(rows, np.arange(n_target)):
        raise RuntimeError("Linear assignment did not assign every target point")
    value = float(np.sum(cost[rows, columns]))
    gradient = None
    if compute_grad:
        _, derivative = _unit_vectors(predicted[columns] - target[rows])
        gradient = np.zeros_like(predicted)
        np.add.at(gradient, columns, derivative)
    components = {"target_to_simulation_assignment": value}
    diagnostics.update(assignment_cost_bytes=int(cost.nbytes),
                       assignment_ids_sha256=_array_hash(columns.astype(np.int64)))
    return value, gradient, components, diagnostics


def soft_iou_value_and_grad(coverage, foreground, known, epsilon=1e-8):
    """Return soft-IoU loss and coverage adjoint on explicitly known pixels."""
    coverage = np.asarray(coverage, dtype=np.float64)
    if coverage.ndim != 2 or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("Coverage must be a finite two-dimensional array in [0,1]")
    foreground = _binary_mask(foreground, coverage.shape, "foreground_mask")
    known = _binary_mask(known, coverage.shape, "known_mask")
    if not known.any():
        raise ValueError("Segmentation has no known pixels")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Mask epsilon must be positive and finite")
    c, m = coverage[known], foreground[known].astype(np.float64)
    intersection = float(np.sum(c * m)) + epsilon
    union = float(np.sum(c + m - c * m)) + epsilon
    gradient = np.zeros_like(coverage)
    gradient[known] = (intersection * (1.0 - m) - m * union) / (union * union)
    return 1.0 - intersection / union, gradient


class PointSetLoss:
    def __init__(self, camera, config: PaperLossConfig, initial_positions, precision="f32"):
        if not isinstance(config, PaperLossConfig):
            raise ValueError("PointSetLoss requires a PaperLossConfig")
        if precision not in {"f32", "f64"}:
            raise ValueError("precision must be f32 or f64")
        initial = _points(initial_positions, "initial positions")
        self.camera = camera
        self.config = config
        self.precision = precision
        self.numpy_dtype = np.float32 if precision == "f32" else np.float64
        self.n_particles = len(initial)
        self.predicted_ids = sampled_indices(self.n_particles, config.predicted_sample_count, config.sampling_seed)
        self.predicted_ids.setflags(write=False)
        self.renderer = None

    def _validate_observation(self, observation):
        for name in ("frame_index", "step"):
            value = getattr(observation, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"Observation {name} must be a nonnegative integer")
        if not np.isfinite(observation.timestamp):
            raise ValueError("Observation timestamp must be finite")
        representation = observation.target_representation
        if representation not in {"partial_observed", "full_observed", "inferred_volume"}:
            raise ValueError("Unknown target representation")
        volume = "-prt-" in self.config.version
        if volume and representation != "inferred_volume":
            raise ValueError("DPSI PRT requires an inferred_volume target")
        if not volume and representation == "inferred_volume":
            raise ValueError("This point-cloud objective requires observed targets, not inferred volume")
        if not isinstance(observation.metadata, dict):
            raise ValueError("Observation metadata must be a mapping")

    def _target(self, observation):
        points = _points(observation.points_scene, "observed scene points",
                         allow_empty=self.config.geometric_weight == 0)
        original_count = len(points)
        if self.config.geometric_weight == 0:
            return points, {"target_input_points": original_count, "target_used_points": 0}
        points = voxel_centroids(points, self.config.target_voxel_size_m)
        voxel_count = len(points)
        indices = sampled_indices(voxel_count, self.config.target_sample_count, self.config.sampling_seed)
        points = points[indices]
        if len(points) < self.config.min_target_points:
            raise ValueError("Target has fewer points than min_target_points after explicit preprocessing")
        return points, {"target_input_points": original_count, "target_voxel_points": voxel_count,
                        "target_used_points": len(points), "target_indices_sha256": _array_hash(indices),
                        "target_used_points_sha256": _array_hash(points)}

    def _track_data(self, observation):
        ids = np.asarray(observation.track_particle_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
            raise ValueError("Positive tracking_weight requires nonempty integer track_particle_ids")
        if np.any(ids < 0) or np.any(ids >= self.n_particles) or len(np.unique(ids)) != len(ids):
            raise ValueError("Tracked particle IDs must be unique and in range")
        ids = ids.astype(np.int64, copy=False)
        targets = np.asarray(observation.track_positions_scene, dtype=np.float64)
        if targets.shape != (len(ids), 3):
            raise ValueError("track_positions_scene must match tracked IDs")
        valid = _binary_mask(observation.track_valid, (len(ids),), "track_valid")
        if not np.isfinite(targets[valid]).all():
            raise ValueError("Valid tracked positions must be finite")
        count = int(valid.sum())
        if count == 0 and self.config.empty_track_policy == "error":
            raise ValueError("Scored frame has no valid tracks; empty_track_policy is error")
        diagnostics = {"tracked_points": len(ids), "valid_tracked_points": count,
                       "tracking_skipped": count == 0,
                       "tracked_particle_ids_sha256": _array_hash(ids),
                       "track_valid_sha256": _array_hash(valid)}
        return ids, targets, valid, diagnostics

    def _tracking(self, positions, observation, compute_grad):
        ids, targets, valid, diagnostics = self._track_data(observation)
        delta = positions[ids[valid]] - targets[valid]
        value = float(np.sum(delta * delta))
        gradient = None
        if compute_grad:
            gradient = np.zeros_like(positions)
            np.add.at(gradient, ids[valid], 2.0 * delta)
        return value, gradient, diagnostics

    def _mask_data(self, observation):
        expected = (self.camera.height, self.camera.width)
        foreground = _binary_mask(observation.foreground_mask, expected, "foreground_mask")
        known = _binary_mask(observation.known_mask, expected, "known_mask")
        if not known.any():
            raise ValueError("Segmentation has no known pixels")
        diagnostics = {"known_mask_pixels": int(known.sum()),
                       "known_foreground_pixels": int(np.sum(known & foreground)),
                       "mask_operator": "known-pixel-soft-iou-v1",
                       "coverage_operator": "normalized-front-footprint-v1"}
        return foreground, known, diagnostics

    def _mask(self, positions, observation, compute_grad):
        foreground, known, diagnostics = self._mask_data(observation)
        if self.renderer is None:
            from .renderer import LocalSplatRenderer
            self.renderer = LocalSplatRenderer(self.camera, self.n_particles,
                                               self.config.footprint_radius,
                                               self.config.visibility_temperature_m,
                                               self.config.opacity_gain, self.precision)
        rendered = self.renderer.forward(positions)
        value, coverage_gradient = soft_iou_value_and_grad(rendered.coverage, foreground, known,
                                                          self.config.mask_epsilon)
        gradient = None
        if compute_grad:
            self.renderer.clear_gradients()
            self.renderer.coverage.grad.from_numpy(np.ascontiguousarray(coverage_gradient, dtype=self.numpy_dtype))
            gradient = self.renderer.reverse().astype(np.float64)
        return value, gradient, diagnostics

    def _base_diagnostics(self, observation, target_diagnostics):
        dpsi = self.config.version in DPSI_LOSS_VERSIONS
        diagnostics = {"loss_version": self.config.version,
                       "target_representation": observation.target_representation,
                       "target_source": self.config.target_source,
                       "prediction_support": "all-configured-particles",
                       "prediction_input_points": self.n_particles,
                       "prediction_used_points": len(self.predicted_ids),
                       "prediction_indices_sha256": _array_hash(self.predicted_ids),
                       "sampling_seed": self.config.sampling_seed,
                       "sampling_algorithm": "numpy-default-rng-choice-sorted-v1",
                       "sampling_count_policy": "explicit-maximum-no-upsampling",
                       "target_voxel_size_m": self.config.target_voxel_size_m,
                       "point_reduction": "unsquared-sums" if dpsi else "squared-directional-means",
                       "temporal_reduction": "mean-single-endpoint" if dpsi else "sum",
                       "units": "m" if dpsi else ("weighted-m2-plus-dimensionless" if self.config.mask_weight else "m2"),
                       "correspondence_derivative": "fixed-local-associations",
                       "tie_policy": "scipy-choice-on-fixed-input-order",
                       "zero_distance_subgradient": "zero",
                       "partial_target_warning": observation.target_representation == "partial_observed",
                       "paper_operator_adaptation": not dpsi,
                       "sampling_adaptation": (len(self.predicted_ids) != self.n_particles
                                               or target_diagnostics.get("target_used_points", 0)
                                               != target_diagnostics.get("target_voxel_points", 0)),
                       **target_diagnostics}
        if self.config.version == "empm-offline-v1":
            diagnostics["geometry_only_ablation"] = self.config.tracking_weight == 0
            diagnostics["empty_track_policy"] = self.config.empty_track_policy
        return diagnostics

    def prepare_observation(self, observation):
        """Validate support and preprocessing without renderer/runtime allocation.

        This does not modify the observation or cache sampled coordinates. Later
        value evaluations repeat exactly the same preprocessing on its originals.
        """
        self._validate_observation(observation)
        target, target_diagnostics = self._target(observation)
        diagnostics = self._base_diagnostics(observation, target_diagnostics)
        if "-emd-" in self.config.version:
            diagnostics.update(validate_assignment_size(
                len(target), len(self.predicted_ids), max_pairs=self.config.max_assignment_pairs,
                max_bytes=self.config.max_assignment_bytes))
        if self.config.version == "empm-offline-v1" and self.config.tracking_weight > 0:
            _, _, _, details = self._track_data(observation)
            diagnostics.update(details)
        if self.config.version == "empm-mask-inspired-v1":
            _, _, details = self._mask_data(observation)
            diagnostics.update(details)
        return diagnostics

    def value_and_grad_positions(self, positions, observation, compute_grad=True):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (self.n_particles, 3) or not np.isfinite(positions).all():
            raise InvalidStateError("Predicted positions must be finite and match the initial particle count")
        self._validate_observation(observation)
        target, target_diagnostics = self._target(observation)
        gradient = np.zeros_like(positions) if compute_grad else None
        components, weights = {}, {}
        dpsi = self.config.version in DPSI_LOSS_VERSIONS
        diagnostics = self._base_diagnostics(observation, target_diagnostics)
        value = 0.0
        if self.config.geometric_weight > 0:
            predicted = positions[self.predicted_ids]
            if "-emd-" in self.config.version:
                geometric, local_gradient, part, details = assignment_value_and_grad(
                    predicted, target, max_pairs=self.config.max_assignment_pairs,
                    max_bytes=self.config.max_assignment_bytes, compute_grad=compute_grad)
            else:
                geometric, local_gradient, part, details = chamfer_value_and_grad(
                    predicted, target, squared=not dpsi, compute_grad=compute_grad)
            value += self.config.geometric_weight * geometric
            components.update(part)
            weights.update({name: self.config.geometric_weight for name in part})
            diagnostics.update(details)
            if compute_grad:
                np.add.at(gradient, self.predicted_ids, self.config.geometric_weight * local_gradient)
        if self.config.version == "empm-offline-v1":
            if self.config.tracking_weight > 0:
                tracked, track_gradient, details = self._tracking(positions, observation, compute_grad)
                value += self.config.tracking_weight * tracked
                components["tracked_squared_distance"] = tracked
                weights["tracked_squared_distance"] = self.config.tracking_weight
                diagnostics.update(details)
                if compute_grad:
                    gradient += self.config.tracking_weight * track_gradient
        if self.config.version == "empm-mask-inspired-v1":
            mask, mask_gradient, details = self._mask(positions, observation, compute_grad)
            value += self.config.mask_weight * mask
            components["segmentation_soft_iou"] = mask
            weights["segmentation_soft_iou"] = self.config.mask_weight
            diagnostics.update(details)
            if compute_grad:
                gradient += self.config.mask_weight * mask_gradient
        diagnostics["component_weights"] = weights
        if not np.isfinite(value) or not all(np.isfinite(list(components.values()))):
            raise InvalidStateError("Nonfinite paper objective")
        if compute_grad:
            gradient = np.ascontiguousarray(gradient, dtype=self.numpy_dtype)
            if not np.isfinite(gradient).all():
                raise InvalidStateError("Nonfinite paper position gradient")
        return LossRecord(float(value), gradient, components, diagnostics)

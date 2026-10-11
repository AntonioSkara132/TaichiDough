#!/usr/bin/env python3
"""Reusable losses, deterministic search helpers, and cache checks for material calibration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


LOSS_COMPONENTS = (
    "depth_change",
    "mask_iou",
    "observed_to_simulation_distance",
    "real_coverage",
)
DEFAULT_LOSS_WEIGHTS = {
    "depth_change": 1.0,
    "mask_iou": 1.0,
    "observed_to_simulation_distance": 1.0,
    "real_coverage": 1.0,
}


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON data deterministically for fingerprints and cache keys."""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value is not canonical JSON data: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_fingerprint(episode_dir: Path) -> str:
    """Match the sequence fingerprint used by deformpath_dynamics."""
    episode_dir = Path(episode_dir)
    pointclouds = episode_dir / "pointclouds_interpolated.pt"
    paths = episode_dir / "paths_interpolated.pt"
    missing = [str(path) for path in (pointclouds, paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Sequence files are missing: " + ", ".join(missing))
    return hashlib.sha256((file_sha256(pointclouds) + file_sha256(paths)).encode()).hexdigest()


def visible_optical_points(
    depth: np.ndarray,
    valid: np.ndarray,
    camera_or_field_of_view: Mapping[str, Any] | float,
    max_points: int = 0,
) -> np.ndarray:
    """Back-project visible pixels into the camera optical frame."""
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if depth.ndim != 2 or depth.shape != valid.shape:
        raise ValueError("Depth and validity mask must be matching two-dimensional arrays")
    if max_points < 0:
        raise ValueError("max_points cannot be negative")
    if not np.isfinite(depth[valid]).all():
        raise ValueError("Visible depth values must be finite")
    pixel_y, pixel_x = np.nonzero(valid)
    if len(pixel_x) == 0:
        return np.empty((0, 3), dtype=np.float64)
    height, width = depth.shape
    z = depth[pixel_y, pixel_x]
    if isinstance(camera_or_field_of_view, Mapping):
        camera = camera_or_field_of_view
        exact_keys = ("width", "height", "fx", "fy", "cx", "cy")
        if all(key in camera for key in exact_keys):
            calibrated_width = int(camera["width"])
            calibrated_height = int(camera["height"])
            if (width, height) != (calibrated_width, calibrated_height):
                raise ValueError("Depth dimensions do not match the exact camera intrinsics")
            fx = _finite_float(camera["fx"], "camera fx")
            fy = _finite_float(camera["fy"], "camera fy")
            cx = _finite_float(camera["cx"], "camera cx")
            cy = _finite_float(camera["cy"], "camera cy")
            if fx <= 0 or fy <= 0:
                raise ValueError("Camera focal lengths must be positive")
            x = (pixel_x.astype(np.float64) + 0.5 - cx) * z / fx
            optical_y = (pixel_y.astype(np.float64) + 0.5 - cy) * z / fy
        else:
            fov = _finite_float(camera.get("fieldOfView"), "field of view")
            if not 0.0 < fov < 180.0:
                raise ValueError("Field of view must be between 0 and 180 degrees")
            focal = height / (2.0 * np.tan(np.radians(fov) / 2.0))
            x = (pixel_x.astype(np.float64) + 0.5 - width * 0.5) * z / focal
            optical_y = (pixel_y.astype(np.float64) + 0.5 - height * 0.5) * z / focal
    else:
        fov = _finite_float(camera_or_field_of_view, "field of view")
        if not 0.0 < fov < 180.0:
            raise ValueError("Field of view must be between 0 and 180 degrees")
        focal = height / (2.0 * np.tan(np.radians(fov) / 2.0))
        x = (pixel_x.astype(np.float64) + 0.5 - width * 0.5) * z / focal
        optical_y = (pixel_y.astype(np.float64) + 0.5 - height * 0.5) * z / focal
    points = np.stack([x, optical_y, z], axis=1)
    if max_points and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).round().astype(np.int64)
        points = points[indices]
    return points


def huber_loss(values: np.ndarray, delta: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    delta = _finite_float(delta, "Huber delta")
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    absolute = np.abs(values)
    return np.where(absolute <= delta, 0.5 * values * values, delta * (absolute - 0.5 * delta))


def one_sided_nearest_distances(
    observed_points: np.ndarray,
    visible_simulation_points: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Return observed-to-visible-simulation distances; never compare hidden particles."""
    observed = np.asarray(observed_points, dtype=np.float64)
    simulation = np.asarray(visible_simulation_points, dtype=np.float64)
    if observed.ndim != 2 or simulation.ndim != 2 or observed.shape[1:] != (3,) or simulation.shape[1:] != (3,):
        raise ValueError("Observed and simulation point clouds must have shape [N, 3]")
    if not np.isfinite(observed).all() or not np.isfinite(simulation).all():
        raise ValueError("Visible point clouds must contain only finite coordinates")
    if chunk_size < 1:
        raise ValueError("Nearest-distance chunk size must be positive")
    if len(observed) == 0 or len(simulation) == 0:
        return np.empty((0,), dtype=np.float64)
    result = np.empty(len(observed), dtype=np.float64)
    for start in range(0, len(observed), chunk_size):
        source = observed[start:start + chunk_size]
        squared = np.sum((source[:, None, :] - simulation[None, :, :]) ** 2, axis=2)
        result[start:start + len(source)] = np.sqrt(np.min(squared, axis=1))
    return result


def partial_view_loss(
    observed_depth: np.ndarray,
    observed_valid: np.ndarray,
    simulation_depth: np.ndarray,
    simulation_valid: np.ndarray,
    observed_initial_depth: np.ndarray,
    observed_initial_valid: np.ndarray,
    simulation_initial_depth: np.ndarray,
    simulation_initial_valid: np.ndarray,
    observed_points: np.ndarray,
    visible_simulation_points: np.ndarray,
    *,
    weights: Mapping[str, float] | None = None,
    depth_scale_m: float = 0.005,
    distance_scale_m: float = 0.005,
    huber_delta: float = 1.0,
    min_common_pixels: int = 16,
    min_observed_pixels: int = 16,
    min_simulation_pixels: int = 16,
    min_observed_points: int = 16,
    min_simulation_points: int = 16,
    nearest_chunk_size: int = 1024,
) -> dict[str, Any]:
    """Score one partial-view frame using current and initial visible surfaces.

    Depth-change residuals use support visible in both initial states and both current
    states. The point distance is directed from the observed cloud to the visible
    simulated cloud. Missing or insufficient support makes the result invalid.
    """
    arrays = [
        np.asarray(observed_depth, dtype=np.float64),
        np.asarray(simulation_depth, dtype=np.float64),
        np.asarray(observed_initial_depth, dtype=np.float64),
        np.asarray(simulation_initial_depth, dtype=np.float64),
    ]
    masks = [
        np.asarray(observed_valid, dtype=bool),
        np.asarray(simulation_valid, dtype=bool),
        np.asarray(observed_initial_valid, dtype=bool),
        np.asarray(simulation_initial_valid, dtype=bool),
    ]
    shape = arrays[0].shape
    if len(shape) != 2 or any(array.shape != shape for array in arrays + masks):
        raise ValueError("All depth and validity arrays must share one two-dimensional shape")
    for name, array, mask in zip(
        ("observed", "simulation", "observed initial", "simulation initial"), arrays, masks
    ):
        if not np.isfinite(array[mask]).all():
            raise ValueError(f"{name.capitalize()} visible depth values must be finite")
    thresholds = {
        "min_common_pixels": min_common_pixels,
        "min_observed_pixels": min_observed_pixels,
        "min_simulation_pixels": min_simulation_pixels,
        "min_observed_points": min_observed_points,
        "min_simulation_points": min_simulation_points,
    }
    if any(not isinstance(value, int) or value < 1 for value in thresholds.values()):
        raise ValueError("All support thresholds must be positive integers")
    depth_scale = _finite_float(depth_scale_m, "depth scale")
    distance_scale = _finite_float(distance_scale_m, "distance scale")
    huber_delta_value = _finite_float(huber_delta, "Huber delta")
    if depth_scale <= 0 or distance_scale <= 0 or huber_delta_value <= 0:
        raise ValueError("Depth scale, point-distance scale, and Huber delta must be positive")

    observed_points = np.asarray(observed_points, dtype=np.float64)
    simulation_points = np.asarray(visible_simulation_points, dtype=np.float64)
    if observed_points.ndim != 2 or observed_points.shape[1:] != (3,):
        raise ValueError("Observed optical point cloud must have shape [N, 3]")
    if simulation_points.ndim != 2 or simulation_points.shape[1:] != (3,):
        raise ValueError("Visible simulation optical point cloud must have shape [N, 3]")
    if not np.isfinite(observed_points).all() or not np.isfinite(simulation_points).all():
        raise ValueError("Optical point clouds must contain only finite coordinates")

    observed_count = int(masks[0].sum())
    simulation_count = int(masks[1].sum())
    common_current = masks[0] & masks[1]
    union_current = masks[0] | masks[1]
    depth_support = masks[0] & masks[1] & masks[2] & masks[3]
    support = {
        "observed_pixels": observed_count,
        "simulation_pixels": simulation_count,
        "common_current_pixels": int(common_current.sum()),
        "union_current_pixels": int(union_current.sum()),
        "depth_change_pixels": int(depth_support.sum()),
        "observed_points": int(len(observed_points)),
        "visible_simulation_points": int(len(simulation_points)),
    }
    components: dict[str, float | None] = {name: None for name in LOSS_COMPONENTS}
    reasons: list[str] = []

    if support["depth_change_pixels"] < min_common_pixels:
        reasons.append(
            f"depth-change common support {support['depth_change_pixels']} is below {min_common_pixels} pixels"
        )
    else:
        observed_change = arrays[0] - arrays[2]
        simulation_change = arrays[1] - arrays[3]
        normalized_residual = (simulation_change - observed_change)[depth_support] / depth_scale
        components["depth_change"] = float(np.mean(huber_loss(normalized_residual, huber_delta_value)))

    if observed_count < min_observed_pixels:
        reasons.append(f"observed support {observed_count} is below {min_observed_pixels} pixels")
    if simulation_count < min_simulation_pixels:
        reasons.append(f"visible simulation support {simulation_count} is below {min_simulation_pixels} pixels")
    if int(union_current.sum()) == 0:
        reasons.append("current mask union is empty")
    else:
        components["mask_iou"] = 1.0 - float(common_current.sum() / union_current.sum())
    if observed_count:
        components["real_coverage"] = 1.0 - float(common_current.sum() / observed_count)

    if len(observed_points) < min_observed_points:
        reasons.append(f"observed point support {len(observed_points)} is below {min_observed_points}")
    if len(simulation_points) < min_simulation_points:
        reasons.append(
            f"visible simulation point support {len(simulation_points)} is below {min_simulation_points}"
        )
    if len(observed_points) >= min_observed_points and len(simulation_points) >= min_simulation_points:
        distances = one_sided_nearest_distances(observed_points, simulation_points, nearest_chunk_size)
        components["observed_to_simulation_distance"] = float(
            np.mean(huber_loss(distances / distance_scale, huber_delta_value))
        )
        support["nearest_distance_samples"] = int(len(distances))
    else:
        support["nearest_distance_samples"] = 0

    selected_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - set(LOSS_COMPONENTS)
        if unknown:
            raise ValueError("Unknown loss weights: " + ", ".join(sorted(unknown)))
        selected_weights.update(weights)
    selected_weights = {
        name: _finite_float(selected_weights[name], f"weight {name}") for name in LOSS_COMPONENTS
    }
    if any(value < 0 for value in selected_weights.values()) or not any(selected_weights.values()):
        raise ValueError("Loss weights must be non-negative and at least one must be positive")
    for name, weight in selected_weights.items():
        if weight > 0 and components[name] is None:
            reasons.append(f"weighted component {name} is unavailable")

    valid = not reasons
    weighted_total = None
    if valid:
        component_values = [components[name] for name in LOSS_COMPONENTS]
        finite_component_values = [float(value) for value in component_values if value is not None]
        if len(finite_component_values) != len(LOSS_COMPONENTS):
            raise RuntimeError("Valid loss has an unavailable component")
        weight_sum = sum(selected_weights.values())
        weighted_total = float(
            sum(
                selected_weights[name] * value
                for name, value in zip(LOSS_COMPONENTS, finite_component_values)
            )
            / weight_sum
        )
    return {
        "valid": valid,
        "failure_reason": None if valid else "; ".join(dict.fromkeys(reasons)),
        "components": components,
        "weighted_total": weighted_total,
        "weights": selected_weights,
        "support": support,
        "normalization": {
            "depth_scale_m": depth_scale,
            "distance_scale_m": distance_scale,
            "huber_delta": huber_delta_value,
        },
    }


def aggregate_loss_results(
    results: Sequence[Mapping[str, Any]],
    *,
    result_weights: Sequence[float] | None = None,
    require_all: bool = True,
    minimum_valid: int = 1,
) -> dict[str, Any]:
    """Aggregate frame or window results without replacing invalid entries by zero."""
    if minimum_valid < 1:
        raise ValueError("minimum_valid must be positive")
    if result_weights is None:
        weights = np.ones(len(results), dtype=np.float64)
    else:
        weights = np.asarray(result_weights, dtype=np.float64)
        if weights.shape != (len(results),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Aggregation weights must be finite, positive, and match the result count")
    invalid = [index for index, result in enumerate(results) if not result.get("valid", False)]
    valid_indices = [index for index in range(len(results)) if index not in invalid]
    reasons: list[str] = []
    if len(valid_indices) < minimum_valid:
        reasons.append(f"only {len(valid_indices)} valid results; {minimum_valid} required")
    if require_all and invalid:
        reasons.append(f"{len(invalid)} of {len(results)} results are invalid")
    valid = not reasons
    components: dict[str, float | None] = {name: None for name in LOSS_COMPONENTS}
    total = None
    if valid_indices:
        selected = weights[valid_indices]
        total = float(
            np.average([float(results[index]["weighted_total"]) for index in valid_indices], weights=selected)
        )
        for name in LOSS_COMPONENTS:
            values = [results[index]["components"].get(name) for index in valid_indices]
            finite_values = [float(value) for value in values if value is not None]
            if len(finite_values) == len(values):
                components[name] = float(np.average(finite_values, weights=selected))
    if not valid:
        total = None
    return {
        "valid": valid,
        "failure_reason": None if valid else "; ".join(reasons),
        "components": components,
        "weighted_total": total,
        "counts": {
            "total": len(results),
            "valid": len(valid_indices),
            "invalid": len(invalid),
        },
        "invalid_indices": invalid,
    }


def aggregate_named_windows(
    window_results: Mapping[str, Mapping[str, Any]],
    window_specs: Sequence[Mapping[str, Any]],
    split: str,
    *,
    require_all: bool = True,
) -> dict[str, Any]:
    """Aggregate one named manifest split while retaining per-window identity."""
    selected_specs = [spec for spec in window_specs if spec.get("split") == split]
    names = [str(spec.get("name", "")) for spec in selected_specs]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        return {
            "valid": False,
            "failure_reason": f"split {split!r} has no unique named windows",
            "components": {name: None for name in LOSS_COMPONENTS},
            "weighted_total": None,
            "window_names": names,
        }
    missing = [name for name in names if name not in window_results]
    if missing:
        return {
            "valid": False,
            "failure_reason": "missing window results: " + ", ".join(missing),
            "components": {name: None for name in LOSS_COMPONENTS},
            "weighted_total": None,
            "window_names": names,
        }
    weights = [_finite_float(spec.get("weight", 1.0), f"window {spec['name']} weight") for spec in selected_specs]
    aggregate = aggregate_loss_results(
        [window_results[name] for name in names],
        result_weights=weights,
        require_all=require_all,
        minimum_valid=len(names) if require_all else 1,
    )
    aggregate.update({"split": split, "window_names": names, "window_weights": weights})
    return aggregate


def _stable_candidate(value: float) -> float:
    return float(f"{_finite_float(value, 'candidate'):.12g}")


def logarithmic_candidates(
    lower: float,
    upper: float,
    count: int,
    *,
    include: Iterable[float] = (),
) -> list[float]:
    lower = _finite_float(lower, "lower candidate bound")
    upper = _finite_float(upper, "upper candidate bound")
    if lower <= 0 or upper <= lower or count < 2:
        raise ValueError("Logarithmic candidate bounds must satisfy 0 < lower < upper and count >= 2")
    values = np.geomspace(lower, upper, int(count)).tolist() + [float(value) for value in include]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("All included candidates must be finite and positive")
    return sorted({_stable_candidate(value) for value in values})


def refinement_candidates(
    evaluated_candidates: Iterable[float],
    best_candidate: float,
    subdivisions: int = 4,
) -> list[float]:
    """Subdivide the logarithmic intervals adjacent to the current best."""
    values = sorted({_stable_candidate(value) for value in evaluated_candidates})
    best = _stable_candidate(best_candidate)
    if best not in values:
        raise ValueError("Best candidate must already be in the evaluated set")
    if subdivisions < 2:
        raise ValueError("Refinement subdivisions must be at least two")
    index = values.index(best)
    intervals = []
    if index > 0:
        intervals.append((values[index - 1], best))
    if index + 1 < len(values):
        intervals.append((best, values[index + 1]))
    existing = set(values)
    proposed: set[float] = set()
    for lower, upper in intervals:
        for value in np.geomspace(lower, upper, subdivisions + 1)[1:-1]:
            candidate = _stable_candidate(float(value))
            if candidate not in existing:
                proposed.add(candidate)
    return sorted(proposed)


def boundary_expansion(
    current_bounds: Sequence[float],
    best_candidate: float,
    hard_bounds: Sequence[float],
    expansion_factor: float = 10.0,
) -> dict[str, Any]:
    lower, upper = map(float, current_bounds)
    hard_lower, hard_upper = map(float, hard_bounds)
    best = float(best_candidate)
    factor = _finite_float(expansion_factor, "expansion factor")
    if not 0 < hard_lower <= lower < upper <= hard_upper or factor <= 1:
        raise ValueError("Bounds must satisfy 0 < hard lower <= lower < upper <= hard upper")
    lower_boundary = math.isclose(best, lower, rel_tol=1e-10, abs_tol=0.0)
    upper_boundary = math.isclose(best, upper, rel_tol=1e-10, abs_tol=0.0)
    if not lower_boundary and not upper_boundary:
        return {
            "boundary": None,
            "expanded": False,
            "hit_hard_limit": False,
            "bounds": [lower, upper],
            "reason": None,
        }
    if lower_boundary:
        new_lower = max(hard_lower, lower / factor)
        expanded = new_lower < lower
        return {
            "boundary": "lower",
            "expanded": expanded,
            "hit_hard_limit": not expanded,
            "bounds": [new_lower, upper],
            "reason": None if expanded else "best candidate is at the hard lower safety limit",
        }
    new_upper = min(hard_upper, upper * factor)
    expanded = new_upper > upper
    return {
        "boundary": "upper",
        "expanded": expanded,
        "hit_hard_limit": not expanded,
        "bounds": [lower, new_upper],
        "reason": None if expanded else "best candidate is at the hard upper safety limit",
    }


def select_best_candidate(candidate_totals: Mapping[float, float | None]) -> dict[str, Any]:
    valid = []
    for candidate, total in candidate_totals.items():
        if total is not None and math.isfinite(float(total)):
            valid.append((_stable_candidate(float(candidate)), float(total)))
    if not valid:
        return {"valid": False, "youngs_modulus_pa": None, "loss": None, "failure_reason": "no valid candidates"}
    candidate, loss = min(valid, key=lambda item: (item[1], item[0]))
    return {"valid": True, "youngs_modulus_pa": candidate, "loss": loss, "failure_reason": None}


def flat_minimum_diagnostics(
    candidate_totals: Mapping[float, float | None],
    *,
    relative_tolerance: float = 0.02,
    absolute_tolerance: float = 1e-6,
    minimum_log10_span: float = 0.15,
) -> dict[str, Any]:
    relative = _finite_float(relative_tolerance, "flat relative tolerance")
    absolute = _finite_float(absolute_tolerance, "flat absolute tolerance")
    span_threshold = _finite_float(minimum_log10_span, "flat log span")
    if relative < 0 or absolute < 0 or span_threshold < 0:
        raise ValueError("Flat-minimum tolerances cannot be negative")
    valid = sorted(
        (_stable_candidate(candidate), float(total))
        for candidate, total in candidate_totals.items()
        if total is not None and math.isfinite(float(total))
    )
    if not valid:
        return {"flat": False, "reason": "no valid candidates", "near_minimum_candidates_pa": []}
    best_candidate, best_loss = min(valid, key=lambda item: (item[1], item[0]))
    threshold = best_loss + max(absolute, relative * max(abs(best_loss), 1e-12))
    near = [candidate for candidate, total in valid if total <= threshold]
    log_span = 0.0 if len(near) < 2 else math.log10(max(near) / min(near))
    flat = len(near) >= 2 and log_span >= span_threshold
    return {
        "flat": flat,
        "best_candidate_pa": best_candidate,
        "best_loss": best_loss,
        "loss_threshold": threshold,
        "near_minimum_candidates_pa": near,
        "log10_span": log_span,
        "minimum_log10_span": span_threshold,
        "reason": "multiple materially separated candidates are indistinguishable at the configured loss tolerance" if flat else None,
    }


def whole_window_bootstrap_interval(
    candidate_window_losses: Mapping[float, Sequence[float]],
    *,
    window_weights: Sequence[float] | None = None,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Resample complete windows and report a log-space interval of selected moduli."""
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("Bootstrap samples must be positive and confidence must lie between zero and one")
    candidate_items = sorted(
        (_stable_candidate(float(candidate)), candidate) for candidate in candidate_window_losses
    )
    candidates = [stable for stable, _ in candidate_items]
    if not candidates:
        raise ValueError("Bootstrap requires at least one candidate")
    if len(candidates) != len(set(candidates)):
        raise ValueError("Bootstrap candidates must remain unique at 12 significant digits")
    arrays = [
        np.asarray(candidate_window_losses[original], dtype=np.float64)
        for _, original in candidate_items
    ]
    window_count = len(arrays[0])
    if window_count < 1 or any(array.shape != (window_count,) for array in arrays):
        raise ValueError("Every bootstrap candidate must contain the same nonempty window list")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Bootstrap window losses must be finite")
    matrix = np.stack(arrays, axis=0)
    if window_weights is None:
        weights = np.ones(window_count, dtype=np.float64)
    else:
        weights = np.asarray(window_weights, dtype=np.float64)
        if weights.shape != (window_count,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Bootstrap window weights must be finite, positive, and match the window count")
    generator = np.random.default_rng(int(seed))
    selected = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = generator.integers(0, window_count, size=window_count)
        means = np.average(matrix[:, indices], axis=1, weights=weights[indices])
        selected[sample] = candidates[int(np.argmin(means))]
    alpha = (1.0 - confidence) / 2.0
    log_selected = np.log(selected)
    lower, median, upper = np.exp(np.quantile(log_selected, [alpha, 0.5, 1.0 - alpha]))
    counts = {str(candidate): int(np.sum(selected == candidate)) for candidate in candidates}
    return {
        "method": "whole-window nonparametric bootstrap with configured window weights; best discrete candidate per replicate",
        "samples": int(samples),
        "confidence": float(confidence),
        "seed": int(seed),
        "window_count": window_count,
        "lower_pa": float(lower),
        "median_pa": float(median),
        "upper_pa": float(upper),
        "selection_counts": counts,
    }


def cache_completion_status(
    expected_metadata: Mapping[str, Any],
    actual_metadata: Mapping[str, Any] | None,
    present_relative_paths: Iterable[str],
    required_relative_paths: Iterable[str],
) -> dict[str, Any]:
    """Validate a cache record from already-read metadata and a path listing."""
    present = {str(Path(path).as_posix()) for path in present_relative_paths}
    required = [str(Path(path).as_posix()) for path in required_relative_paths]
    missing = [path for path in required if path not in present]
    mismatches = []
    if actual_metadata is None:
        mismatches.append("cache metadata is missing or unreadable")
    else:
        for key, expected in expected_metadata.items():
            if actual_metadata.get(key) != expected:
                mismatches.append(f"cache metadata field {key!r} does not match")
        if actual_metadata.get("status") != "complete":
            mismatches.append("cache status is not complete")
    reasons = mismatches + (["missing required files: " + ", ".join(missing)] if missing else [])
    return {
        "complete": not reasons,
        "failure_reason": None if not reasons else "; ".join(reasons),
        "missing_files": missing,
        "metadata_mismatches": mismatches,
    }


def validate_cache_directory(
    cache_dir: Path,
    expected_metadata: Mapping[str, Any],
    required_relative_paths: Iterable[str],
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "cache_metadata.json"
    actual = None
    if metadata_path.is_file():
        try:
            actual = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            actual = None
    present = [str(path.relative_to(cache_dir)) for path in cache_dir.rglob("*") if path.is_file()] if cache_dir.is_dir() else []
    return cache_completion_status(expected_metadata, actual, present, required_relative_paths)


def evaluate_scalar_search(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    coarse_count: int,
    *,
    include: Iterable[float] = (),
    refinement_rounds: int = 2,
    subdivisions: int = 4,
) -> dict[str, Any]:
    """Small deterministic search runner used by tests and inexpensive scalar objectives."""
    candidates = logarithmic_candidates(lower, upper, coarse_count, include=include)
    totals = {candidate: _finite_float(objective(candidate), "objective result") for candidate in candidates}
    rounds = []
    for _ in range(refinement_rounds):
        best = select_best_candidate(totals)
        proposed = refinement_candidates(totals, best["youngs_modulus_pa"], subdivisions)
        rounds.append(proposed)
        if not proposed:
            break
        totals.update({candidate: _finite_float(objective(candidate), "objective result") for candidate in proposed})
    ordered_totals = dict(sorted(totals.items()))
    return {
        "best": select_best_candidate(totals),
        "candidates": list(ordered_totals),
        "candidate_totals": ordered_totals,
        "refinement_rounds": rounds,
    }

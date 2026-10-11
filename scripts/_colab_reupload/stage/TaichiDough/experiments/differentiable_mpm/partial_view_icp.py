"""Partial-view point-to-plane ICP between sampled tool meshes and RGB-D tool points.

The fitted transform is ``marker_from_mesh``: mesh samples ``q`` map to marker coordinates
with ``q @ R.T + t``. Each update is applied on the left, in marker coordinates
(``candidate = delta @ correction``), where ``delta = [Exp(rotation_step), translation_step]``.

This module depends only on NumPy and SciPy so that dataset tools outside the experiment
package can import it without the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Rotation columns of the solver Jacobian are multiplied by 1 / ROTATION_LEVER_M so that
# rotation and translation have comparable magnitudes in the scaled normal equations.
ROTATION_LEVER_M = 0.1


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def sample_mesh(triangles, count, seed):
    edges = triangles[:, 1:] - triangles[:, :1]
    crosses = np.cross(edges[:, 0], edges[:, 1])
    twice_area = np.linalg.norm(crosses, axis=1)
    keep = twice_area > 1e-14
    triangles, crosses, twice_area = triangles[keep], crosses[keep], twice_area[keep]
    rng = np.random.default_rng(seed)
    index = rng.choice(len(triangles), count, p=twice_area / twice_area.sum())
    uv = rng.random((count, 2))
    over = uv.sum(axis=1) > 1
    uv[over] = 1 - uv[over]
    selected = triangles[index]
    points = selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    normals = crosses[index] / twice_area[index, None]
    return points, normals


def visible_model(points, normals, correction, camera_from_marker, camera, min_visible=100):
    q = transform(points, correction)
    n = normals @ correction[:3, :3].T
    optical = transform(q, camera_from_marker)
    z = optical[:, 2]
    good = z > .01
    u = np.floor((camera['fx'] * optical[:, 0] / np.maximum(z, .01) + camera['cx']) / 2).astype(int)
    v = np.floor((camera['fy'] * optical[:, 1] / np.maximum(z, .01) + camera['cy']) / 2).astype(int)
    width, height = int(np.ceil(camera['width'] / 2)), int(np.ceil(camera['height'] / 2))
    good &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    index = np.flatnonzero(good)
    bins = v[index] * width + u[index]
    depth = np.full(width * height, np.inf)
    np.minimum.at(depth, bins, z[index])
    index = index[z[index] <= depth[bins] + .0015]
    if len(index) < min_visible:
        raise ValueError(f'Fewer than {min_visible} camera-visible mesh samples')
    return q[index], n[index]


def correspondences(frame, model, correction, camera, limit, min_visible=100):
    q, n = visible_model(*model, correction, frame['camera_from_marker'], camera, min_visible)
    distance, index = cKDTree(q).query(frame['points'], workers=1)
    keep = distance < limit
    return frame['points'][keep], q[index[keep]], n[index[keep]], distance[keep], distance


@dataclass(frozen=True)
class IcpConfig:
    """Solver settings. The defaults reproduce the original multi-frame static fit."""

    iterations: int = 35
    # (first_iteration, correspondence distance limit in m); the last entry with
    # first_iteration <= iteration applies.
    distance_schedule: tuple[tuple[int, float], ...] = ((0, .08), (8, .04), (18, .025))
    trim_quantile: float = .85
    residual_scale_m: float = .004
    distance_scale_m: float = .01
    tangential_weight: float = .12
    max_step_rotation_deg: float = 6.0
    max_step_translation_m: float = .008
    # Total-motion bounds are measured on candidate @ inverse(bound_reference). The
    # translation bound is the displacement of bound_pivot_marker (marker coordinates).
    max_total_translation_m: float = .15
    max_total_rotation_deg: float = 70.0
    bound_reference: np.ndarray | None = None
    bound_pivot_marker: np.ndarray | None = None
    min_frame_correspondences: int = 50
    min_total_correspondences: int = 100
    min_visible: int = 100
    convergence_min_iteration: int = 21
    convergence_step: float = 2e-6
    damping: float = 1e-7


@dataclass
class IcpResult:
    transform: np.ndarray
    status: str
    history: list[dict[str, Any]]
    # Plane-only Hessian of the last accepted iteration in the solver's scaled units, with
    # per-frame 1/n weight normalization. Kept for conditioning reports.
    scaled_plane_hessian: np.ndarray
    final: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.status == 'converged'


def _distance_limit(config: IcpConfig, iteration: int) -> float:
    limit = config.distance_schedule[0][1]
    for start, value in config.distance_schedule:
        if iteration >= start:
            limit = value
    return float(limit)


def _robust_weights(residual, distance, config: IcpConfig):
    weight = np.minimum(1, config.residual_scale_m / np.maximum(np.abs(residual), 1e-12))
    weight *= np.minimum(1, config.distance_scale_m / np.maximum(distance, 1e-12))
    return weight


def _trimmed_correspondences(frame, model, correction, camera, limit, config: IcpConfig):
    p, q, n, distance, all_distance = correspondences(frame, model, correction, camera, limit, config.min_visible)
    if len(p) < config.min_frame_correspondences:
        return None, all_distance
    cutoff = np.quantile(distance, config.trim_quantile)
    keep = distance <= cutoff
    return (p[keep], q[keep], n[keep], distance[keep]), all_distance


def total_motion(candidate, config: IcpConfig) -> tuple[float, float]:
    """Return (pivot displacement in m, rotation angle in rad) of candidate relative to the bound reference."""
    reference = np.eye(4) if config.bound_reference is None else config.bound_reference
    relative = candidate @ np.linalg.inv(reference)
    pivot = np.zeros(3) if config.bound_pivot_marker is None else np.asarray(config.bound_pivot_marker, dtype=float)
    displacement = float(np.linalg.norm(relative[:3, :3] @ pivot + relative[:3, 3] - pivot))
    angle = float(Rotation.from_matrix(relative[:3, :3]).magnitude())
    return displacement, angle


def fit_icp_detailed(frames, model, camera, initial=None, config: IcpConfig | None = None) -> IcpResult:
    config = IcpConfig() if config is None else config
    correction = np.eye(4) if initial is None else np.array(initial, dtype=float)
    history: list[dict[str, Any]] = []
    last_hessian = np.zeros((6, 6))
    status = 'max_iterations'
    failure = ''
    for iteration in range(config.iterations):
        limit = _distance_limit(config, iteration)
        hessian, gradient = np.zeros((6, 6)), np.zeros(6)
        plane_hessian = np.zeros((6, 6))
        errors, used = [], 0
        try:
            for frame in frames:
                selected, all_distance = _trimmed_correspondences(frame, model, correction, camera, limit, config)
                if selected is None:
                    continue
                p, q, n, distance = selected
                residual = np.sum((p - q) * n, axis=1)
                jacobian = np.column_stack((np.cross(q, n) / ROTATION_LEVER_M, n))
                weight = _robust_weights(residual, distance, config)
                weight /= len(p)
                h = jacobian.T @ (weight[:, None] * jacobian)
                plane_hessian += h
                hessian += h
                gradient += jacobian.T @ (weight * residual)
                # Small tangential term retains edge information; rank uses plane terms only.
                if config.tangential_weight:
                    for axis in range(3):
                        unit = np.zeros(3); unit[axis] = 1
                        j = np.column_stack((np.cross(q, unit) / ROTATION_LEVER_M,
                                             np.broadcast_to(unit, (len(q), 3))))
                        w = config.tangential_weight * weight
                        hessian += j.T @ (w[:, None] * j)
                        gradient += j.T @ (w * (p[:, axis] - q[:, axis]))
                errors.extend(all_distance.tolist())
                used += len(p)
        except ValueError as error:
            history.append({'iteration': iteration, 'visibility_failure': str(error)})
            status = 'insufficient_visible_model'
            failure = str(error)
            break
        if used < config.min_total_correspondences:
            history.append({'iteration': iteration, 'correspondences': used})
            status = 'insufficient_correspondences'
            failure = 'Insufficient registration correspondences'
            break
        step = np.linalg.lstsq(hessian + np.eye(6) * config.damping, gradient, rcond=1e-8)[0]
        rotation_step = step[:3] / ROTATION_LEVER_M
        translation_step = step[3:]
        fraction = min(1, np.deg2rad(config.max_step_rotation_deg) / max(np.linalg.norm(rotation_step), 1e-12),
                       config.max_step_translation_m / max(np.linalg.norm(translation_step), 1e-12))
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec(rotation_step * fraction).as_matrix()
        delta[:3, 3] = translation_step * fraction
        candidate = delta @ correction
        displacement, angle = total_motion(candidate, config)
        if displacement > config.max_total_translation_m or angle > np.deg2rad(config.max_total_rotation_deg):
            history.append({'iteration': iteration, 'bounded_stop': True})
            status = 'bounded_stop'
            break
        correction = candidate
        last_hessian = plane_hessian
        history.append({'iteration': iteration, 'median_distance_m': float(np.median(errors)),
                        'correspondences': used, 'step_translation_m': float(np.linalg.norm(translation_step * fraction)),
                        'step_rotation_deg': float(np.rad2deg(np.linalg.norm(rotation_step * fraction)))})
        if iteration >= config.convergence_min_iteration and np.linalg.norm(step) * fraction < config.convergence_step:
            status = 'converged'
            break
    return IcpResult(transform=correction, status=status, history=history, scaled_plane_hessian=last_hessian,
                     final={'failure': failure} if failure else {})


def fit_icp(frames, model, camera, initial=None, iterations=35, tangential_weight=.12, config: IcpConfig | None = None):
    """Original interface: returns (marker_from_mesh, history, scaled plane Hessian)."""
    config = IcpConfig() if config is None else config
    config = replace(config, iterations=iterations, tangential_weight=tangential_weight)
    result = fit_icp_detailed(frames, model, camera, initial, config)
    if result.status in ('insufficient_correspondences', 'insufficient_visible_model'):
        raise ValueError(result.final['failure'])
    return result.transform, result.history, result.scaled_plane_hessian


def point_to_plane_information(frame, model, correction, camera, limit, config: IcpConfig | None = None) -> dict[str, Any]:
    """Physical-unit information and covariance of a single-frame fit at ``correction``.

    The perturbation is left-multiplied in marker coordinates, ordered
    (rotation x, y, z in rad, translation x, y, z in m). Weights are the robust IRLS weights
    without per-frame normalization, so ``covariance = sigma^2 * inverse(information)``,
    where sigma^2 is the weighted plane-residual variance with six degrees of freedom removed.
    Neighbouring depth points are correlated, so callers should inflate this covariance.
    """
    config = IcpConfig() if config is None else config
    selected, all_distance = _trimmed_correspondences(frame, model, correction, camera, limit, config)
    if selected is None:
        return {'valid': False, 'reason': 'insufficient_correspondences',
                'correspondences': 0, 'points': int(len(all_distance))}
    p, q, n, distance = selected
    residual = np.sum((p - q) * n, axis=1)
    weight = _robust_weights(residual, distance, config)
    jacobian = np.column_stack((np.cross(q, n), n))
    information = jacobian.T @ (weight[:, None] * jacobian)
    dof = float(weight.sum()) - 6.0
    sigma2 = float(np.sum(weight * residual ** 2) / dof) if dof > 0 else float('nan')
    eigenvalues = np.linalg.eigvalsh(information)
    result = {
        'valid': bool(dof > 0 and np.isfinite(sigma2) and eigenvalues[0] > 0),
        'points': int(len(all_distance)),
        'correspondences': int(len(p)),
        'weight_sum': float(weight.sum()),
        'residual_sigma_m': float(np.sqrt(sigma2)) if np.isfinite(sigma2) else float('nan'),
        'median_distance_m': float(np.median(all_distance)),
        'p90_distance_m': float(np.quantile(all_distance, .9)),
        'within_3mm': float(np.mean(all_distance < .003)),
        'within_5mm': float(np.mean(all_distance < .005)),
        'inlier_fraction': float(len(p) / max(len(all_distance), 1)),
        'median_abs_plane_m': float(np.median(np.abs(residual))),
        'information': information,
        'information_eigenvalues': eigenvalues,
    }
    if result['valid']:
        result['covariance'] = sigma2 * np.linalg.inv(information)
    else:
        result['reason'] = 'degenerate_information'
    return result


def evaluate(frames, model, correction, camera):
    rows = []
    for frame in frames:
        p, q, n, d, all_distance = correspondences(frame, model, correction, camera, .08)
        residual = np.sum((p - q) * n, axis=1)
        rows.append({'processed_frame': frame['processed'], 'raw_frame': frame['raw'],
                     'points': len(frame['points']), 'median_distance_mm': float(np.median(all_distance) * 1000),
                     'p90_distance_mm': float(np.quantile(all_distance, .9) * 1000),
                     'within_3mm': float(np.mean(all_distance < .003)),
                     'within_5mm': float(np.mean(all_distance < .005)),
                     'median_abs_plane_mm': float(np.median(np.abs(residual)) * 1000),
                     'mean_residual_marker_xyz_mm': ((p - q).mean(axis=0) * 1000).tolist()})
    return {'equal_frame_median_distance_mm': float(np.mean([row['median_distance_mm'] for row in rows])),
            'equal_frame_p90_distance_mm': float(np.mean([row['p90_distance_mm'] for row in rows])),
            'equal_frame_within_3mm': float(np.mean([row['within_3mm'] for row in rows])),
            'equal_frame_within_5mm': float(np.mean([row['within_5mm'] for row in rows])), 'frames': rows}


def registration_passes(before: Mapping[str, Any], after: Mapping[str, Any], relative_information: Sequence[float]) -> tuple[bool, list[str]]:
    reasons = []
    baseline = float(before["equal_frame_median_distance_mm"])
    fitted = float(after["equal_frame_median_distance_mm"])
    if not fitted <= 0.70 * baseline:
        reasons.append("heldout_median_improvement_below_30_percent")
    if not fitted < 5.0:
        reasons.append("heldout_equal_frame_median_not_below_5_mm")
    if not float(after["equal_frame_within_5mm"]) > 0.75:
        reasons.append("heldout_within_5mm_not_above_75_percent")
    if not float(relative_information[0]) > 1e-4:
        reasons.append("point_to_plane_information_ratio_not_above_1e-4")
    before_frames = {int(row["raw_frame"]): row for row in before["frames"]}
    for row in after["frames"]:
        base = before_frames[int(row["raw_frame"])]
        if float(row["median_distance_mm"]) > max(5.0, 1.25 * float(base["median_distance_mm"])):
            reasons.append(f"heldout_frame_{row['raw_frame']}_regressed")
    return not reasons, reasons

"""Qualify interpolated tool distances and normals without running a simulation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np


QUALIFICATION_VERSION = "triangle-interpolation-v2"
OPPOSING_FACE_ANGLE_DEGREES = 5.
REGULAR_SUPPORT_ANGLE_DEGREES = 1.
NORMAL_ACCEPTANCE = ("all-evaluated", "stable-support")
CROSSING_ACCEPTANCE = ("exact", "below-tolerance")
# strict-v2: fixed 0.1 mm distance limit, and every evaluated normal probe counts.
# grid-relative-v1: distance limit is a fraction of the MPM grid spacing, and the
# normal limit applies where all eight interpolation corners see the same face
# (exact normals within REGULAR_SUPPORT_ANGLE_DEGREES); probes whose corners span
# an edge, corner or hole rim are still reported but do not fail the field. A
# missing interior or crossing-count mismatch is accepted only where the deepest
# material on the profile lies within the distance limit of the surface (and,
# on a thickness profile, within half the thickness tolerance of its wall), so
# the field misses nothing larger than the error it already allows. A lost
# plate is still rejected: its depth is half its thickness.
QUALIFICATION_PROFILES = {"strict-v2": {"distance_tolerance_m": 1e-4, "normal_acceptance": "all-evaluated",
                                        "crossing_acceptance": "exact"},
                          "grid-relative-v1": {"distance_grid_fraction": .05, "normal_acceptance": "stable-support",
                                               "crossing_acceptance": "below-tolerance"}}


def qualification_settings(profile="strict-v2", grid_spacing_m=None):
    """Return the qualify_* keyword settings for a named profile."""
    if profile not in QUALIFICATION_PROFILES:
        raise ValueError(f"Unknown tool SDF qualification profile: {profile}")
    rule = QUALIFICATION_PROFILES[profile]
    if "distance_grid_fraction" in rule:
        if grid_spacing_m is None or not np.isfinite(grid_spacing_m) or grid_spacing_m <= 0:
            raise ValueError(f"{profile} requires a positive MPM grid spacing")
        distance = rule["distance_grid_fraction"] * float(grid_spacing_m)
    else:
        distance = rule["distance_tolerance_m"]
    return {"distance_tolerance": distance, "normal_acceptance": rule["normal_acceptance"],
            "crossing_acceptance": rule["crossing_acceptance"]}


def qualification_identity(profile="strict-v2", grid_spacing_m=None):
    settings = qualification_settings(profile, grid_spacing_m)
    return {"version": QUALIFICATION_VERSION, "profile": profile,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "distance_tolerance_m": settings["distance_tolerance"], "thickness_relative_tolerance": .1,
            "normal_tolerance_degrees": 10., "max_faces": 128, "precisions": ["f32", "f64"],
            "opposing_face_angle_degrees": OPPOSING_FACE_ANGLE_DEGREES,
            "regular_support_angle_degrees": REGULAR_SUPPORT_ANGLE_DEGREES,
            "normal_acceptance": settings["normal_acceptance"],
            "crossing_acceptance": settings["crossing_acceptance"],
            "boundary_metric": "mesh distance at refined field zero crossings"}


def array_identity(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def interpolate_field(volume, points, *, dtype=np.float64, _validated=False):
    """Match Stepper._sdf_at, including its domain and normal threshold.

    A volume is (distances, normals, minimum, spacing) for one tool. Arithmetic
    uses the requested solver precision; source arrays retain their stored values.
    """
    distances, normals, minimum, spacing = (np.asarray(v) for v in volume)
    minimum, spacing = minimum.astype(dtype), spacing.astype(dtype)
    points = np.asarray(points, dtype=dtype)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (N,3) array")
    if not _validated and (distances.ndim != 3 or len(set(distances.shape)) != 1 or distances.shape[0] < 2
            or normals.shape != distances.shape + (3,) or minimum.shape != (3,)
            or spacing.shape != (3,) or np.any(spacing <= 0)
            or not all(np.isfinite(v).all() for v in (distances, normals, minimum, spacing))):
        raise ValueError("invalid tool field arrays")
    def lookup(indices):
        index = tuple(np.moveaxis(indices, -1, 0))
        return distances[index], normals[index]

    return _interpolate_grid(distances.shape[0], minimum, spacing, points, lookup, dtype)


def _interpolate_grid(resolution, minimum, spacing, points, lookup, dtype):
    """Shared interpolation for stored arrays and sparsely queried grid corners."""
    minimum, spacing = np.asarray(minimum, dtype=dtype), np.asarray(spacing, dtype=dtype)
    points = np.asarray(points, dtype=dtype)
    gp = (points - minimum) / spacing
    valid = np.all((gp >= 0) & (gp <= resolution - 1), axis=1)
    result = np.full(len(points), 1e6, dtype=dtype)
    gradient = np.zeros((len(points), 3), dtype=dtype)
    g = gp[valid]
    base = np.clip(np.floor(g).astype(np.int64), 0, resolution - 2)
    fraction = np.clip(g - base.astype(dtype), 0, 1)
    values = np.zeros(len(g), dtype=dtype)
    vectors = np.zeros((len(g), 3), dtype=dtype)
    offsets = tuple(itertools.product(range(2), repeat=3))
    indices = base[None, :, :] + np.asarray(offsets)[:, None, :]
    corner_distances, corner_normals = lookup(indices)
    for corner, offset in enumerate(offsets):
        weight = np.ones(len(g), dtype=dtype)
        for axis in range(3):
            weight *= fraction[:, axis] if offset[axis] else 1 - fraction[:, axis]
        values += weight * corner_distances[corner].astype(dtype)
        vectors += weight[:, None] * corner_normals[corner].astype(dtype)
    squared = np.sum(vectors * vectors, axis=1)
    denominator = np.full(len(g), 1e-6, dtype=dtype)
    selected = squared > 1e-12
    denominator[selected] = np.sqrt(squared[selected])
    vectors /= denominator[:, None]
    result[valid], gradient[valid] = values, vectors
    return valid, result, gradient


class _DenseSamples:
    def __init__(self, volume):
        interpolate_field(volume, np.empty((0, 3)))
        self.volume = volume
        self.resolution = np.asarray(volume[0]).shape[0]
        self.minimum = np.asarray(volume[2])
        self.spacing = np.asarray(volume[3])
        self.arrays = {name: array_identity(value) for name, value in zip(
            ("distances", "normals", "minimum", "spacing"), volume)}

    def sample(self, points, dtype):
        return interpolate_field(self.volume, points, dtype=dtype, _validated=True)


class _SparseSamples:
    def __init__(self, triangles, resolution, padding, query):
        from .triangle_sdf import grid_bounds

        self.minimum, self.spacing = grid_bounds(triangles, resolution, padding)
        self.resolution = int(resolution)
        self.query = query
        self.arrays = {}
        self.queried_corners = 0
        self.largest_query = 0

    def _lookup(self, indices):
        flat = indices.reshape(-1, 3)
        unique, inverse = np.unique(flat, axis=0, return_inverse=True)
        # This expression and TriangleDistanceQuery's final float32 conversion
        # are exactly the grid-point construction used by build_tool_sdf.
        points = self.minimum.astype(np.float64) + unique * self.spacing.astype(np.float64)
        if len(points):
            distances, normals = self.query.query(points)
        else:
            distances, normals = np.empty(0, np.float32), np.empty((0, 3), np.float32)
        self.queried_corners += len(unique)
        self.largest_query = max(self.largest_query, len(unique))
        return distances[inverse].reshape(indices.shape[:-1]), normals[inverse].reshape(indices.shape)

    def sample(self, points, dtype):
        return _interpolate_grid(self.resolution, self.minimum, self.spacing, points, self._lookup, dtype)


def _ray_exit(origin, direction, triangles, epsilon):
    """Independent double-precision ray/triangle exit distance (Moller-Trumbore)."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    usable = np.abs(determinant) > 1e-12 * np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1)
    inverse = np.zeros(len(triangles))
    inverse[usable] = 1 / determinant[usable]
    s = origin - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * (q @ direction)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    hit = usable & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > epsilon)
    return float(t[hit].min()) if np.any(hit) else None


def _line_hits(origin, direction, triangles):
    """All line/triangle intersections, retaining incident faces at shared edges."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    usable = np.abs(determinant) > 1e-12 * np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1)
    inverse = np.zeros(len(triangles))
    inverse[usable] = 1 / determinant[usable]
    s = origin - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * (q @ direction)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    hit = usable & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    groups = []
    for index in np.flatnonzero(hit)[np.argsort(t[hit])]:
        value = float(t[index])
        if groups and abs(value - groups[-1][0]) <= 1e-12:
            groups[-1][1].append(int(index))
        else:
            groups.append((value, [int(index)]))
    return groups


def _reference_crossings(hits, lower, upper, origin, direction, query):
    """Discard line touches which do not separate an inside/outside interval."""
    selected = [(t, faces) for t, faces in hits if lower < t < upper]
    if not selected:
        return []
    roots = np.asarray([row[0] for row in selected])
    boundaries = np.concatenate(([lower], roots, [upper]))
    midpoints = .5 * (boundaries[:-1] + boundaries[1:])
    signs = np.asarray(query.query(origin + midpoints[:, None] * direction)[0]) < 0
    return [row for i, row in enumerate(selected) if signs[i] != signs[i + 1]]


def _refined_crossings(coordinates, distances, field, origin, direction, dtype, tolerance):
    roots = []
    zeros = np.flatnonzero(distances == 0)
    for run in np.split(zeros, np.flatnonzero(np.diff(zeros) > 1) + 1):
        if not len(run):
            continue
        before, after = run[0] - 1, run[-1] + 1
        if before >= 0 and after < len(distances) and distances[before] * distances[after] < 0:
            roots.append(float(.5 * (coordinates[run[0]] + coordinates[run[-1]])))
    indices = np.flatnonzero(distances[:-1] * distances[1:] < 0)
    left, right = coordinates[indices].copy(), coordinates[indices + 1].copy()
    left_values = distances[indices].copy()
    for _ in range(40):
        active = right - left > tolerance
        if not np.any(active):
            break
        mid = .5 * (left[active] + right[active])
        valid, values, _ = field.sample(origin + mid[:, None] * direction, dtype)
        if not np.all(valid):
            raise ValueError("Root refinement left the field domain")
        rows = np.flatnonzero(active)
        zero = values == 0
        left[rows[zero]] = mid[zero]
        right[rows[zero]] = mid[zero]
        same = ~zero & ((values < 0) == (left_values[rows] < 0))
        left[rows[same]] = mid[same]
        left_values[rows[same]] = values[same]
        other = ~zero & ~same
        right[rows[other]] = mid[other]
    roots.extend(.5 * (left + right))
    roots = np.asarray(sorted(roots))
    return roots[np.concatenate(([True], np.diff(roots) > 1e-12))] if len(roots) else roots


def _normal_support_spread(field, points, exact_normals, query, dtype):
    gp = (np.asarray(points, dtype=dtype) - field.minimum.astype(dtype)) / field.spacing.astype(dtype)
    base = np.clip(np.floor(gp).astype(np.int64), 0, field.resolution - 2)
    corners = base[:, None, :] + np.asarray(tuple(itertools.product(range(2), repeat=3)))[None]
    unique, inverse = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    positions = field.minimum.astype(np.float64) + unique * field.spacing.astype(np.float64)
    _, normals = query.query(positions)
    normals = normals[inverse].reshape(len(points), 8, 3)
    dots = np.einsum("nij,nj->ni", normals, exact_normals)
    return np.rad2deg(np.arccos(np.clip(dots, -1, 1))).max(axis=1)


def _face_probes(triangles, maximum):
    """Include large faces and opposite directions, then spread remaining indices."""
    count = len(triangles)
    if count <= maximum:
        return np.arange(count)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)
    normals = cross / area[:, None]
    centers = triangles.mean(axis=1)
    chosen = []
    for coordinates in (normals, centers):
        for axis in range(3):
            for index in (np.argmin(coordinates[:, axis]), np.argmax(coordinates[:, axis])):
                if int(index) not in chosen:
                    chosen.append(int(index))
    for index in np.argsort(-area, kind="stable")[:max(1, maximum // 4)]:
        if int(index) not in chosen:
            chosen.append(int(index))
    remaining = max(0, maximum - len(chosen))
    for index in np.linspace(0, count - 1, remaining, dtype=int):
        if int(index) not in chosen:
            chosen.append(int(index))
    return np.asarray(sorted(chosen[:maximum]))


def _crossings(coordinates, distances):
    roots = list(coordinates[distances == 0])
    indices = np.flatnonzero(distances[:-1] * distances[1:] < 0)
    for index in indices:
        a, b = distances[index:index + 2]
        roots.append(float(coordinates[index] - a * (coordinates[index + 1] - coordinates[index]) / (b - a)))
    return np.asarray(sorted(set(roots)))


def _qualify_precision(triangles, field, padding, *, distance_tolerance=1e-4,
                       thickness_relative_tolerance=.1, normal_tolerance_degrees=10.,
                       max_faces=128, query=None, dtype=np.float64, normal_acceptance="all-evaluated",
                       crossing_acceptance="exact"):
    """Check deterministic, bounded probes; passing is not a proof between probes.

    With normal_acceptance="all-evaluated" every evaluated normal error counts,
    including stencils with varying exact normals; "stable-support" applies the
    normal limit only to stencils whose exact corner normals agree. With
    crossing_acceptance="below-tolerance", a missing interior or crossing is
    accepted on a profile whose deepest material is within the distance limit
    (see QUALIFICATION_PROFILES). Physical thickness is measured only between
    opposing faces within five degrees; other chord lengths remain diagnostics.
    Boundary errors use geometric distance, not displacement along an oblique ray.
    """
    from .triangle_sdf import TriangleDistanceQuery

    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    if (not np.isfinite([distance_tolerance, thickness_relative_tolerance, normal_tolerance_degrees]).all()
            or distance_tolerance <= 0 or not 0 < thickness_relative_tolerance < 1
            or not 0 < normal_tolerance_degrees < 90
            or isinstance(max_faces, bool) or not isinstance(max_faces, int) or max_faces < 12
            or normal_acceptance not in NORMAL_ACCEPTANCE or crossing_acceptance not in CROSSING_ACCEPTANCE):
        raise ValueError("invalid qualification settings")
    query = query if query is not None else TriangleDistanceQuery(triangles)
    spacing = field.spacing
    extent = np.ptp(triangles.reshape(-1, 3), axis=0)
    epsilon = max(float(extent.max()) * 1e-7, 1e-10)
    ids = _face_probes(triangles, max_faces)
    failures = []
    profiles = []
    total_probes = regular_probes = ambiguous_probes = 0
    worst_distance = worst_normal = worst_crossing = worst_thickness = worst_boundary = 0.
    support_regular = support_varying = eligible_thickness = 0
    shallow_profiles = profiles_without_normal_probes = 0
    regular_normal_passed = all_normal_passed = True
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]
    for face_id in ids:
        triangle = triangles[face_id]
        origin = triangle.mean(axis=0)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        hits = _line_hits(origin, -normal, triangles)
        exterior = max(float(padding), float(np.min(spacing)) * .25, epsilon * 4)
        all_reference = (_reference_crossings(hits, hits[0][0] - exterior, hits[-1][0] + exterior,
                                              origin, -normal, query) if hits else [])
        exits = [row for row in all_reference if row[0] > epsilon]
        if not exits:
            failures.append({"face": int(face_id), "reason": "no opposite boundary intersection"})
            continue
        thickness, exit_faces = exits[0]
        opposing_angle = float(np.rad2deg(np.arccos(np.clip(face_normals[exit_faces] @ -normal, -1, 1))).max())
        thickness_eligible = opposing_angle <= OPPOSING_FACE_ANGLE_DEGREES
        eligible_thickness += int(thickness_eligible)
        # Include every crossing in the padded interval, including true reentry
        # into another part of a concave solid. Tangential touches do not define
        # an interior interval or a thickness endpoint.
        reference = [row for row in all_reference if -exterior < row[0] < thickness + exterior]
        reference_roots = np.asarray([row[0] for row in reference])
        extra = []
        boundaries = np.concatenate(([-exterior], reference_roots, [thickness + exterior]))
        for i, root in enumerate(reference_roots):
            delta = min(epsilon, (root - boundaries[i]) * .25, (boundaries[i + 2] - root) * .25)
            extra.extend((root - delta, root, root + delta))
        coordinates = np.unique(np.concatenate((
            np.linspace(-exterior, 0., 9), np.linspace(0., thickness, 65),
            np.linspace(thickness, thickness + exterior, 9),
            [-epsilon, epsilon, thickness * .1, thickness * .9, thickness - epsilon, thickness + epsilon], extra)))
        points = origin - coordinates[:, None] * normal
        exact_distance, exact_normals = query.query(points)
        exact_distance = np.asarray(exact_distance).reshape(-1)
        exact_normals = np.asarray(exact_normals).reshape(-1, 3)
        valid, interpolated, sampled_normals = field.sample(points, dtype)
        total_probes += len(points)
        reasons = []
        if not np.all(valid):
            reasons.append("profile extends outside field")
        error = np.abs(interpolated - exact_distance)
        max_error = float(error.max())
        worst_distance = max(worst_distance, max_error)
        if max_error > distance_tolerance:
            reasons.append("distance error exceeds tolerance")
        # Only near-parallel opposing faces define a physical thickness here.
        # A grazing or corner-cutting chord is still reported, but is not used
        # as a blade-thickness measurement.
        deepest = float(max(0., -exact_distance.min()))
        shallow = (crossing_acceptance == "below-tolerance" and max_error <= distance_tolerance
                   and deepest <= distance_tolerance
                   and (not thickness_eligible or deepest <= .5 * thickness_relative_tolerance * thickness))
        center = (coordinates >= .25 * thickness) & (coordinates <= .75 * thickness)
        interior_missing = thickness_eligible and (not np.any(center) or np.any(interpolated[center] >= 0))
        if interior_missing and not shallow:
            reasons.append("negative interior is missing")
        roots = (_refined_crossings(coordinates, interpolated, field, origin, -normal, dtype,
                                   min(1e-8, .01 * distance_tolerance)) if np.all(valid)
                 else _crossings(coordinates, interpolated))
        crossing_error = represented = relative_error = boundary_error = None
        if len(roots):
            root_distances, _ = query.query(origin - roots[:, None] * normal)
            boundary_error = float(np.abs(root_distances).max())
            worst_boundary = max(worst_boundary, boundary_error)
            if boundary_error > distance_tolerance:
                reasons.append("boundary distance error exceeds tolerance")
        crossing_mismatch = len(roots) != len(reference_roots) or not len(reference_roots)
        if crossing_mismatch and not (shallow and len(reference_roots)):
            reasons.append("field crossing count differs from reference solid")
        if (crossing_mismatch or interior_missing) and shallow and len(reference_roots):
            shallow_profiles += 1
        if not crossing_mismatch:
            crossing_error = float(np.abs(roots - reference_roots).max())
            worst_crossing = max(worst_crossing, crossing_error)
            entry_index = int(np.argmin(np.abs(reference_roots)))
            exit_index = int(np.argmin(np.abs(reference_roots - thickness)))
            represented = float(roots[exit_index] - roots[entry_index])
            relative_error = abs(represented - thickness) / thickness
            if thickness_eligible:
                worst_thickness = max(worst_thickness, relative_error)
                if relative_error > thickness_relative_tolerance:
                    reasons.append("represented thickness error exceeds tolerance")
        # On the entry face and its nearby inside/outside band the expected
        # outward direction is this face's normal. Check the opposite face via
        # its own selected triangle, not by assuming parallel blade faces.
        near_entry = (coordinates <= .2 * thickness) & (np.abs(exact_distance) > epsilon)
        regular = near_entry & ((exact_normals @ normal) >= np.cos(np.deg2rad(1.)))
        ambiguous_probes += int(np.count_nonzero(near_entry & ~regular))
        regular_probes += int(np.count_nonzero(regular))
        normal_error = None
        normal_rows = []
        if np.any(regular):
            dots = np.einsum("ij,ij->i", sampled_normals[regular], exact_normals[regular])
            angles = np.rad2deg(np.arccos(np.clip(dots, -1, 1)))
            spread = _normal_support_spread(field, points[regular], exact_normals[regular], query, dtype)
            stable_support = (spread <= REGULAR_SUPPORT_ANGLE_DEGREES) & valid[regular]
            support_regular += int(np.count_nonzero(stable_support))
            support_varying += int(np.count_nonzero(~stable_support))
            regular_normal_passed &= bool(np.all(angles[stable_support] <= normal_tolerance_degrees))
            all_normal_passed &= bool(np.all(angles <= normal_tolerance_degrees))
            normal_error = float(angles.max())
            worst_normal = max(worst_normal, normal_error)
            checked = angles if normal_acceptance == "all-evaluated" else angles[stable_support]
            for index, angle, variation, stable in zip(np.flatnonzero(regular), angles, spread, stable_support):
                normal_rows.append({"point_m": points[index].tolist(),
                                    "exact_distance_m": float(exact_distance[index]),
                                    "angular_error_degrees": float(angle),
                                    "reference_corner_spread_degrees": float(variation),
                                    "support": "regular" if stable else "varying",
                                    "inside_field": bool(valid[index]),
                                    "interpolated_normal_length": float(np.linalg.norm(sampled_normals[index]))})
            if checked.size and checked.max() > normal_tolerance_degrees:
                reasons.append("regular-face normal error exceeds tolerance")
        else:
            all_normal_passed = False
            profiles_without_normal_probes += 1
            if normal_acceptance == "all-evaluated":
                reasons.append("no regular-face normal probes")
        row = {"face": int(face_id), "chord_length_m": thickness, "represented_chord_length_m": represented,
               "chord_relative_error": relative_error, "thickness_eligible": thickness_eligible,
               "thickness_eligibility_reason": "opposing faces within five degrees" if thickness_eligible
                   else "exit incident normals are not all opposite within five degrees",
               "opposing_face_angle_degrees": opposing_angle,
               "thickness_m": thickness if thickness_eligible else None,
               "represented_thickness_m": represented if thickness_eligible else None,
               "distance_error_m": max_error, "boundary_distance_error_m": boundary_error,
               "zero_crossing_error_m": crossing_error, "ray_crossing_displacement_m": crossing_error,
               "thickness_relative_error": relative_error if thickness_eligible else None,
               "normal_error_degrees": normal_error, "normal_probes": normal_rows,
               "zero_crossings": len(roots), "expected_crossings": len(reference_roots),
               "deepest_material_m": deepest,
               "reference_crossings_m": reference_roots.tolist(), "field_crossings_m": roots.tolist(),
               "passed": not reasons}
        profiles.append(row)
        if reasons:
            failures.append({"face": int(face_id), "reasons": reasons})
    normal_passed = (bool(regular_probes) and all_normal_passed if normal_acceptance == "all-evaluated"
                     else bool(support_regular) and regular_normal_passed)
    return {"version": QUALIFICATION_VERSION, "passed": bool(profiles) and not failures and normal_passed,
            "scope": "Sampled distances, all reference crossing counts, geometric boundary offsets, near-parallel thicknesses, and "
                     + ("ALL evaluated normal probes" if normal_acceptance == "all-evaluated" else "stable-support normal probes")
                     + " must pass; no guarantee between probes.",
            "normal_acceptance": normal_acceptance, "crossing_acceptance": crossing_acceptance,
            "metric_scope": {"zero_crossing_error_m": "ray displacement diagnostic, not an acceptance threshold",
                             "boundary_distance_error_m": "mesh distance at refined field roots",
                             "thickness_relative_error": "only entry/exit normals opposite within five degrees",
                             "normal_support": "at most one degree exact-normal spread across interpolation corners; diagnostic only"},
            "regular_support_normal_passed": bool(support_regular) and regular_normal_passed,
            "all_probe_normal_passed": bool(regular_probes) and all_normal_passed,
            "tolerances": {"distance_m": distance_tolerance, "thickness_relative": thickness_relative_tolerance,
                           "normal_degrees": normal_tolerance_degrees, "normal_acceptance": normal_acceptance,
                           "crossing_acceptance": crossing_acceptance},
            "padding_m": float(padding), "resolution": int(field.resolution),
            "spacing_m": np.asarray(spacing).tolist(),
            "coverage": {"mesh_faces": len(triangles), "selected_faces": len(ids), "profiles": len(profiles),
                         "point_probes": total_probes, "regular_normal_probes": regular_probes,
                         "excluded_ambiguous_normal_probes": ambiguous_probes,
                         "regular_support_normal_probes": support_regular,
                         "varying_support_normal_probes": support_varying,
                         "physical_thickness_profiles": eligible_thickness,
                         "accepted_shallow_profiles": shallow_profiles,
                         "profiles_without_normal_probes": profiles_without_normal_probes},
            "worst": {"distance_error_m": worst_distance, "normal_error_degrees": worst_normal,
                      "boundary_distance_error_m": worst_boundary,
                      "zero_crossing_error_m": worst_crossing, "ray_crossing_displacement_m": worst_crossing,
                      "thickness_relative_error": worst_thickness},
            "arrays": field.arrays,
            "profiles": profiles, "failures": failures}


def _qualify_samples(triangles, field, padding, *, distance_tolerance=1e-4,
                  thickness_relative_tolerance=.1, normal_tolerance_degrees=10.,
                  max_faces=128, query=None, normal_acceptance="all-evaluated",
                  crossing_acceptance="exact"):
    """Require the sampled profiles to pass in both solver precisions."""
    from .triangle_sdf import TriangleDistanceQuery

    query = query if query is not None else TriangleDistanceQuery(triangles)
    reports = {}
    for name, dtype in (("f32", np.float32), ("f64", np.float64)):
        reports[name] = _qualify_precision(
            triangles, field, padding, distance_tolerance=distance_tolerance,
            thickness_relative_tolerance=thickness_relative_tolerance,
            normal_tolerance_degrees=normal_tolerance_degrees,
            max_faces=max_faces, query=query, dtype=dtype, normal_acceptance=normal_acceptance,
            crossing_acceptance=crossing_acceptance)
    report = dict(reports["f64"])
    report["passed"] = all(value["passed"] for value in reports.values())
    report["precisions"] = list(reports)
    report["coverage"] = {**report["coverage"], "counts_per_precision": True}
    report["coverage_by_precision"] = {name: value["coverage"] for name, value in reports.items()}
    for key in ("regular_support_normal_passed", "all_probe_normal_passed"):
        report[key] = all(value[key] for value in reports.values())
    report["worst"] = {key: max(value["worst"][key] for value in reports.values())
                       for key in report["worst"]}
    for key in ("profiles", "failures"):
        report[key] = [{"precision": name, **row} for name, value in reports.items() for row in value[key]]
    return report


def qualify_field(triangles, volume, padding, *, distance_tolerance=1e-4,
                  thickness_relative_tolerance=.1, normal_tolerance_degrees=10.,
                  max_faces=128, query=None, normal_acceptance="all-evaluated",
                  crossing_acceptance="exact"):
    """Qualify stored field arrays in both solver precisions."""
    return _qualify_samples(
        triangles, _DenseSamples(volume), padding, distance_tolerance=distance_tolerance,
        thickness_relative_tolerance=thickness_relative_tolerance,
        normal_tolerance_degrees=normal_tolerance_degrees, max_faces=max_faces, query=query,
        normal_acceptance=normal_acceptance, crossing_acceptance=crossing_acceptance)


def qualify_resolution(triangles, resolution, padding, *, distance_tolerance=1e-4,
                       thickness_relative_tolerance=.1, normal_tolerance_degrees=10.,
                       max_faces=128, query=None, normal_acceptance="all-evaluated",
                  crossing_acceptance="exact"):
    """Preflight the same profiles using only required corners, never an R-cubed array.

    A pass predicts the sampled profiles only. Build and qualify the complete
    stored arrays afterwards; this report has no full-array hashes and cannot
    substitute for a qualified dense artifact.
    """
    from .triangle_sdf import TriangleDistanceQuery

    query = query if query is not None else TriangleDistanceQuery(triangles)
    field = _SparseSamples(triangles, resolution, padding, query)
    report = _qualify_samples(
        triangles, field, padding, distance_tolerance=distance_tolerance,
        thickness_relative_tolerance=thickness_relative_tolerance,
        normal_tolerance_degrees=normal_tolerance_degrees, max_faces=max_faces, query=query,
        normal_acceptance=normal_acceptance, crossing_acceptance=crossing_acceptance)
    report["preflight"] = True
    report["dense_qualification_required"] = True
    report["memory"] = {"dense_array_bytes": 16 * field.resolution ** 3,
                        "estimated_dense_working_bytes": 16 * field.resolution ** 3 * 6 + 64 * 1024 ** 2,
                        "queried_corners_including_repeats": field.queried_corners,
                        "largest_unique_corner_batch": field.largest_query,
                        "largest_corner_value_batch_bytes": field.largest_query * 16}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="One binary STL; never modified")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--scale", type=float, default=.001, help="STL units to metres")
    parser.add_argument("--translation", type=float, nargs=3, default=[0., 0., 0.], metavar=("X", "Y", "Z"))
    parser.add_argument("--rotation", type=float, nargs=9, default=np.eye(3).ravel().tolist(),
                        help="Row-major proper rotation applied after scale, before translation")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--padding", type=float, required=True, help="Maximum particle/grid contact band in metres")
    parser.add_argument("--max-memory-mib", type=float, default=2048.)
    parser.add_argument("--max-faces", type=int, default=128)
    parser.add_argument("--profile", choices=sorted(QUALIFICATION_PROFILES), default="strict-v2")
    parser.add_argument("--grid-spacing", type=float, help="MPM grid spacing in metres, for grid-relative profiles")
    args = parser.parse_args(argv)
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("scale must be positive and finite")
    if args.resolution < 8 or not np.isfinite(args.max_memory_mib) or args.max_memory_mib <= 0:
        parser.error("invalid resolution or memory budget")
    if args.max_faces < 12:
        parser.error("max-faces must be at least 12")
    # Arrays, construction copies, and archive buffers; backend overhead remains
    # separately bounded by mesh size and query chunks, not all voxel/face pairs.
    estimate = 4 * args.resolution ** 3 * 4 * 6 + 64 * 1024 ** 2
    if estimate > args.max_memory_mib * 1024 ** 2:
        parser.error(f"estimated field working memory {estimate / 1024 ** 2:.1f} MiB exceeds budget")
    rotation = np.asarray(args.rotation).reshape(3, 3)
    if (not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-8)):
        parser.error("rotation must be a proper orthonormal matrix")
    if not np.isfinite(args.translation).all() or not np.isfinite(args.padding) or args.padding < 0:
        parser.error("translation/padding must be finite and padding nonnegative")
    try:
        settings = qualification_settings(args.profile, args.grid_spacing)
    except ValueError as error:
        parser.error(str(error))
    from .prepare_blade_collision_meshes import read_binary_stl
    from .triangle_sdf import build_tool_sdf, builder_identity

    raw = args.mesh.read_bytes()
    triangles = read_binary_stl(raw).astype(np.float64) * args.scale
    triangles = triangles @ rotation.T + np.asarray(args.translation)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"mesh": str(args.mesh.resolve()), "mesh_sha256": hashlib.sha256(raw).hexdigest(),
                "scale": args.scale, "rotation": rotation.tolist(), "translation_m": args.translation,
                "builder": builder_identity(),
                "qualification": {**qualification_identity(args.profile, args.grid_spacing), "max_faces": args.max_faces},
                "estimated_working_memory_bytes": estimate,
                "frame": "explicit input transform; no implicit tracking transform"}
    try:
        volume = build_tool_sdf(triangles, args.resolution, args.padding)
        report = qualify_field(triangles, volume, args.padding, max_faces=args.max_faces, **settings)
    except ValueError as error:
        report = {"version": QUALIFICATION_VERSION, "passed": False, "error": str(error)}
        volume = None
    report["construction"] = metadata
    if volume is not None and report["passed"]:
        temporary = args.output_dir / "qualified_field.npz.tmp"
        with temporary.open("xb") as stream:
            np.savez(stream, **dict(zip(("distances", "normals", "minimum", "spacing"), volume)))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(args.output_dir / "qualified_field.npz")
    temporary_report = args.output_dir / "qualification.json.tmp"
    with temporary_report.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_report.rename(args.output_dir / "qualification.json")
    print(json.dumps({"passed": report["passed"], "report": str(args.output_dir / "qualification.json")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

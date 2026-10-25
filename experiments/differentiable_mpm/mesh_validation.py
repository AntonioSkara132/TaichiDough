"""Exhaustive mesh self-intersection checks on the represented binary coordinates.

AABB comparisons exclude only strictly disjoint boxes. All subsequent geometric
predicates use integers or rational numbers, without a geometric tolerance.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
import time

import numpy as np


VALIDATION_VERSION = "exact-dyadic-aabb-v1"


def validation_identity():
    return {"version": VALIDATION_VERSION,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__,
            "arithmetic": "exact dyadic integers and rational intersections",
            "candidate_search": "all overlapping closed triangle AABBs",
            "imported_geometry_helpers": []}


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _side(normal, origin, point):
    return _dot(normal, _sub(point, origin))


def _separated(distances):
    return all(d > 0 for d in distances) or all(d < 0 for d in distances)


def _orient2(a, b, c, axes):
    x, y = axes
    return (b[x] - a[x]) * (c[y] - a[y]) - (b[y] - a[y]) * (c[x] - a[x])


def _inside(point, triangle, normal):
    return all(_dot(_cross(_sub(triangle[(i + 1) % 3], triangle[i]), _sub(point, triangle[i])), normal) >= 0
               for i in range(3))


def _interpolate(a, b, numerator, denominator):
    fraction = Fraction(numerator) / Fraction(denominator)
    return tuple(x + fraction * (y - x) for x, y in zip(a, b))


def _coplanar_disjoint(a, b, axes):
    for first, second in ((a, b), (b, a)):
        orientation = 1 if _orient2(*first, axes) > 0 else -1
        for i in range(3):
            if all(orientation * _orient2(first[i], first[(i + 1) % 3], p, axes) < 0 for p in second):
                return True
    return False


def _coplanar_intersection(a, b, axes):
    polygon = list(a)
    orientation = 1 if _orient2(*b, axes) > 0 else -1
    for i in range(3):
        if not polygon:
            break
        previous = polygon[-1]
        previous_side = orientation * _orient2(b[i], b[(i + 1) % 3], previous, axes)
        clipped = []
        for point in polygon:
            side = orientation * _orient2(b[i], b[(i + 1) % 3], point, axes)
            if (side >= 0) != (previous_side >= 0):
                clipped.append(_interpolate(previous, point, previous_side, previous_side - side))
            if side >= 0:
                clipped.append(point)
            previous, previous_side = point, side
        polygon = clipped
    return set(polygon)


def _intersection(a, b, na, nb, da, db, axes=None):
    if axes is not None:
        return _coplanar_intersection(a, b, axes)
    points = set()
    for triangle, distances, other, normal in ((a, da, b, nb), (b, db, a, na)):
        for i in range(3):
            point, following = triangle[i], triangle[(i + 1) % 3]
            d0, d1 = distances[i], distances[(i + 1) % 3]
            if d0 == 0 and _inside(point, other, normal):
                points.add(point)
            if d0 * d1 < 0:
                hit = _interpolate(point, following, d0, d0 - d1)
                if _inside(hit, other, normal):
                    points.add(hit)
    return points


def _within_shared(points, shared):
    if not points:
        return True
    if len(shared) == 1:
        return points == {shared[0]}
    if len(shared) == 2:
        start, end = shared
        edge = _sub(end, start)
        length_squared = _dot(edge, edge)
        return all(not any(_cross(_sub(point, start), edge))
                   and 0 <= _dot(_sub(point, start), edge) <= length_squared for point in points)
    # Duplicate faces are invalid, even with opposite winding.
    return False


def _integer_vertices(vertices):
    ratios = [[float(x).as_integer_ratio() for x in vertex] for vertex in vertices]
    denominator = max(d for vertex in ratios for _, d in vertex)
    # Float denominators are powers of two, so the conversion is exact.
    return [tuple(n * (denominator // d) for n, d in vertex) for vertex in ratios], denominator


def validate_mesh_intersections(triangles, *, max_reported_pairs=16):
    """Check every overlapping triangle AABB, including adjacent triangle pairs.

    Exactly shared vertices and edges are permitted. Any additional point,
    segment or area of intersection is rejected. This checks intersections and
    degeneracy, not closedness, winding, connectedness or signed volume.
    """
    started = time.perf_counter()
    values = np.asarray(triangles, dtype=np.float64)
    if values.ndim == 2 and values.shape[1:] == (3,) and len(values) % 3 == 0:
        values = values.reshape(-1, 3, 3)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Require nonempty finite triangles with shape (N,3,3) or (3N,3)")
    if isinstance(max_reported_pairs, bool) or not isinstance(max_reported_pairs, int) or max_reported_pairs < 0:
        raise ValueError("max_reported_pairs must be a nonnegative integer")
    vertices, inverse = np.unique(values.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    integer_vertices, denominator = _integer_vertices(vertices)
    integer_triangles = [tuple(integer_vertices[i] for i in face) for face in faces]
    normals = [_cross(_sub(t[1], t[0]), _sub(t[2], t[0])) for t in integer_triangles]
    degenerate = [i for i, normal in enumerate(normals) if not any(normal)]
    report = {"version": VALIDATION_VERSION, "valid": False, "complete": False,
              "faces": len(faces), "vertices": len(vertices), "degenerate_face_count": len(degenerate),
              "degenerate_face_indices": degenerate, "candidate_pair_count": 0,
              "disjoint_pair_count": 0, "allowed_adjacent_pair_count": 0,
              "unexpected_intersection_count": 0, "rational_pair_count": 0, "examples": []}
    if degenerate:
        report["seconds"] = time.perf_counter() - started
        return report
    lower, upper = values.min(axis=1), values.max(axis=1)
    face_sets = [set(map(int, face)) for face in faces]
    for i in range(len(faces) - 1):
        candidates = np.flatnonzero(np.all(upper[i] >= lower[i + 1:], axis=1)
                                    & np.all(upper[i + 1:] >= lower[i], axis=1)) + i + 1
        report["candidate_pair_count"] += len(candidates)
        a, na = integer_triangles[i], normals[i]
        for j in candidates:
            j = int(j)
            b, nb = integer_triangles[j], normals[j]
            shared_ids = face_sets[i] & face_sets[j]
            db = tuple(_side(na, a[0], p) for p in b)
            if _separated(db):
                report["disjoint_pair_count"] += 1
                continue
            da = tuple(_side(nb, b[0], p) for p in a)
            if _separated(da):
                report["disjoint_pair_count"] += 1
                continue
            coplanar = not any(da) and not any(db)
            shared = [integer_vertices[k] for k in sorted(shared_ids)]
            axes = [k for k in range(3) if k != max(range(3), key=lambda k: abs(na[k]))] if coplanar else None
            if len(shared) == 2:
                # Noncoplanar planes meet only along the common edge.
                allowed = not coplanar
                if coplanar:
                    other_a = next(p for p in a if p not in shared)
                    other_b = next(p for p in b if p not in shared)
                    allowed = _orient2(*shared, other_a, axes) * _orient2(*shared, other_b, axes) < 0
                if allowed:
                    report["allowed_adjacent_pair_count"] += 1
                    continue
            if len(shared) == 1 and not coplanar:
                # If a triangle meets the other plane only at the shared vertex,
                # their intersection cannot extend beyond that vertex.
                if any(sum(d == 0 for d in ds) == 1 and _separated(tuple(d for d in ds if d != 0))
                       for ds in (da, db)):
                    report["allowed_adjacent_pair_count"] += 1
                    continue
            if coplanar and _coplanar_disjoint(a, b, axes):
                report["disjoint_pair_count"] += 1
                continue
            report["rational_pair_count"] += 1
            points = _intersection(a, b, na, nb, da, db, axes)
            if not points:
                report["disjoint_pair_count"] += 1
            elif _within_shared(points, shared):
                report["allowed_adjacent_pair_count"] += 1
            else:
                report["unexpected_intersection_count"] += 1
                if len(report["examples"]) < max_reported_pairs:
                    ordered = sorted(points)
                    report["examples"].append({"triangle_indices": [i, j],
                                               "shared_vertex_count": len(shared), "coplanar": coplanar,
                                               "intersection_points": [[float(Fraction(x, denominator)) for x in p] for p in ordered],
                                               "intersection_points_rational": [[str(Fraction(x, denominator)) for x in p] for p in ordered]})
    report["complete"] = True
    report["valid"] = report["unexpected_intersection_count"] == 0
    report["seconds"] = time.perf_counter() - started
    return report

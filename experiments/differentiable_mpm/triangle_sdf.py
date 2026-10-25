"""Triangle-distance samples for a single validated closed solid, without dilation.

This module builds samples, not a certificate of interpolated thin-part accuracy.
The caller must qualify the runtime interpolation before using a field for contact.
All coordinates and distances use metres; geometry is fixed, not differentiated.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


BUILDER_VERSION = "triangle-distance-v1"
DEFAULT_NSAMPLES = 5


def _open3d():
    try:
        import open3d
    except ImportError as error:
        raise ImportError("triangle-distance-v1 requires Open3D") from error
    return open3d


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _triangles(value):
    triangles = np.asarray(value, dtype=np.float64)
    if triangles.ndim == 2 and triangles.shape[1:] == (3,) and len(triangles) % 3 == 0:
        triangles = triangles.reshape(-1, 3, 3)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("Require nonempty triangle coordinates shaped (N,3,3) or (3N,3)")
    if not np.isfinite(triangles).all():
        raise ValueError("Triangle coordinates must be finite")
    return triangles


def builder_identity(*, nsamples=DEFAULT_NSAMPLES):
    from .mesh_validation import validation_identity

    nsamples = _positive_integer(nsamples, "nsamples")
    if nsamples % 2 == 0:
        raise ValueError("nsamples must be odd")
    return {
        "builder": BUILDER_VERSION,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "open3d_version": _open3d().__version__,
        "numpy_version": np.__version__,
        "sign": "single-solid-ray-parity",
        "nsamples": nsamples,
        "normal": "signed-closest-point-direction; oriented-face-at-zero",
        "bounds": "contact-padding-plus-two-final-intervals",
        "geometry_precision": "float32; validated-before-and-after-conversion",
        "intersection_validation": validation_identity(),
        "qualification": "required-separately",
    }


def _validate_mesh(triangles, label):
    o3d = _open3d()
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) == 0):
        raise ValueError(f"{label}: degenerate triangle")
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    if np.any(counts != 2):
        raise ValueError(f"{label}: require closed two-face manifold edges")
    directed_sorted = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    reversed_edges = directed[:, ::-1]
    reversed_sorted = reversed_edges[np.lexsort((reversed_edges[:, 1], reversed_edges[:, 0]))]
    if not np.array_equal(directed_sorted, reversed_sorted):
        raise ValueError(f"{label}: inconsistent triangle orientation")
    # The tensor conversion avoids Open3D 0.17's legacy NumPy-2 constructor crash.
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.ascontiguousarray(vertices)),
        o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.int64))).to_legacy()
    if not mesh.is_vertex_manifold():
        raise ValueError(f"{label}: nonmanifold vertex")
    _, component_counts, _ = mesh.cluster_connected_triangles()
    if len(component_counts) != 1:
        raise ValueError(f"{label}: require one connected solid; component/cavity semantics are unsupported")
    from .mesh_validation import validate_mesh_intersections

    intersections = validate_mesh_intersections(triangles)
    if not intersections["complete"] or not intersections["valid"]:
        raise ValueError(f"{label}: self-intersecting or invalid triangles: {intersections}")
    relative = triangles - vertices[0]
    volume = np.einsum("ij,ij->i", relative[:, 0], np.cross(relative[:, 1], relative[:, 2])).sum() / 6
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"{label}: require positive volume and outward orientation")
    return mesh, {"vertices": len(vertices), "faces": len(faces), "edges": len(edges),
                  "components": 1, "signed_volume_m3": float(volume), "self_intersections": 0,
                  "intersection_validation": intersections}


class TriangleDistanceQuery:
    """Validated Open3D queries; normals at equidistant features are not unique."""

    def __init__(self, triangles, *, nsamples=DEFAULT_NSAMPLES, nthreads=1):
        self.nsamples = _positive_integer(nsamples, "nsamples")
        self.nthreads = _positive_integer(nthreads, "nthreads")
        if self.nsamples % 2 == 0:
            raise ValueError("nsamples must be odd")
        self.triangles = _triangles(triangles).copy()
        source_mesh, original = _validate_mesh(self.triangles, "source mesh")
        with np.errstate(over="ignore"):
            effective = self.triangles.astype(np.float32).astype(np.float64)
        if not np.isfinite(effective).all():
            raise ValueError("Mesh is not representable in float32")
        if np.array_equal(effective, self.triangles):
            mesh, converted = source_mesh, original
        else:
            mesh, converted = _validate_mesh(effective, "float32 query mesh")
        self.metadata = {"source": original, "query_mesh": converted,
                         "maximum_coordinate_rounding_m": float(np.max(np.abs(effective - self.triangles)))}
        self.effective_triangles = effective
        self.o3d = _open3d()
        self.scene = self.o3d.t.geometry.RaycastingScene(nthreads=self.nthreads)
        self.scene.add_triangles(self.o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def query(self, points):
        """Return float32 signed distances and outward unit normals for (...,3).

        Negative means interior. At exactly zero distance the backend's selected
        oriented face supplies the normal. At edges/medial axes the selected
        closest feature is arbitrary; qualification must identify those queries.
        """
        points = np.asarray(points)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("Query points must be finite with final dimension 3")
        with np.errstate(over="ignore"):
            flat = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if not np.isfinite(flat).all():
            raise ValueError("Query points are not representable in float32")
        if len(flat) == 0:
            return np.empty(points.shape[:-1], np.float32), np.empty(points.shape, np.float32)
        tensor = self.o3d.core.Tensor(np.ascontiguousarray(flat))
        closest = self.scene.compute_closest_points(tensor, nthreads=self.nthreads)
        signed = self.scene.compute_signed_distance(tensor, nthreads=self.nthreads,
                                                   nsamples=self.nsamples).numpy().copy()
        delta = flat.astype(np.float64) - closest["points"].numpy().astype(np.float64)
        distance = np.linalg.norm(delta, axis=1)
        normals = closest["primitive_normals"].numpy().copy().astype(np.float64)
        away = distance > 0
        normals[away] = delta[away] / distance[away, None]
        normals[away & (signed < 0)] *= -1
        lengths = np.linalg.norm(normals, axis=1)
        if (not np.isfinite(signed).all() or not np.isfinite(lengths).all()
                or np.any(lengths <= 0)):
            raise ValueError("Triangle query returned a nonfinite distance or invalid normal")
        normals /= lengths[:, None]
        return signed.reshape(points.shape[:-1]), normals.astype(np.float32).reshape(points.shape)


def grid_bounds(triangles, resolution, padding):
    """Return float32 minimum/spacing with padding plus two intervals per side."""
    resolution = _positive_integer(resolution, "resolution", minimum=6)
    if isinstance(padding, (bool, np.bool_)) or not np.isscalar(padding):
        raise ValueError("padding must be a finite nonnegative number")
    try:
        padding = float(padding)
    except (TypeError, ValueError) as error:
        raise ValueError("padding must be a finite nonnegative number") from error
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be a finite nonnegative number")
    points = _triangles(triangles).reshape(-1, 3)
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = upper - lower
    if np.any(span <= 0):
        raise ValueError("Mesh bounds must have positive extent on every axis")
    safety = 16 * np.finfo(np.float32).eps * np.maximum.reduce((np.abs(lower), np.abs(upper), span,
                                                              np.full(3, padding)))
    with np.errstate(over="ignore", under="ignore"):
        spacing = np.nextafter(((span + 2 * (padding + safety)) / (resolution - 5)).astype(np.float32),
                               np.float32(np.inf))
        minimum = np.nextafter((lower - padding - safety - 2 * spacing.astype(np.float64)).astype(np.float32),
                               np.float32(-np.inf))
    if not np.isfinite(spacing).all() or not np.isfinite(minimum).all() or np.any(spacing <= 0):
        raise ValueError("SDF bounds are not representable in float32")
    margin = padding + 2 * spacing.astype(np.float64)
    last = minimum.astype(np.float64) + (resolution - 1) * spacing.astype(np.float64)
    last_f32 = minimum + np.float32(resolution - 1) * spacing
    if np.any(lower - minimum < margin) or np.any(np.minimum(last, last_f32) - upper < margin):
        raise ValueError("Float32 bounds do not cover the contact padding plus interpolation margin")
    return minimum, spacing


def build_tool_sdf(triangles, resolution, padding, *, chunk_size=65536,
                   nsamples=DEFAULT_NSAMPLES, nthreads=1):
    """Return distance, normal, minimum and spacing arrays in legacy cache order.

    Triangle input accepts (N,3,3) or (3N,3). Output is float32, with shapes
    (R,R,R), (R,R,R,3), (3,), (3,). This rejects a completely missing interior,
    but thin-part and interpolated-normal qualification remains the caller's job.
    """
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    minimum, spacing = grid_bounds(triangles, resolution, padding)
    resolution = int(resolution)
    query = TriangleDistanceQuery(triangles, nsamples=nsamples, nthreads=nthreads)
    total = resolution ** 3
    distances = np.empty(total, dtype=np.float32)
    normals = np.empty((total, 3), dtype=np.float32)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        indices = np.arange(start, stop, dtype=np.int64)
        coordinates = np.column_stack((indices // resolution ** 2, indices // resolution % resolution,
                                       indices % resolution))
        points = minimum.astype(np.float64) + coordinates * spacing.astype(np.float64)
        distances[start:stop], normals[start:stop] = query.query(points)
    if not np.any(distances < 0):
        raise ValueError("Sampled SDF has no negative interior; increase resolution and qualify thin parts")
    return distances.reshape((resolution,) * 3), normals.reshape((resolution,) * 3 + (3,)), minimum, spacing

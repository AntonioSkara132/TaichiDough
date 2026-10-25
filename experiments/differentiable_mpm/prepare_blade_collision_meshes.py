"""Extract unchanged blade triangles into new, explicitly qualified collision assets."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


SCHEMA = "taichidough/tool-collision-meshes/v1"
DERIVATION = "exact-connected-blade-extraction-v1"
STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
DEFAULT_SOURCE = Path(__file__).resolve().parent / "data/episode18_dynamics_icp_final_v1"


@dataclass(frozen=True)
class BladeSource:
    name: str
    filename: str
    sha256: str
    bounds_mm: tuple
    faces: int = 3522


SOURCES = (
    BladeSource("UR5e_spathla", "ur_spathla_x10_closed.stl",
                "5d1330c411cbfa27316d16df592ab8327c57f44f2d2a6792fffd9a41d0e5f806",
                ((-38.14160919189453, -52.70305633544922, 31.702747344970703),
                 (119.89225769042969, -50.044944763183594, 148.33230590820312))),
    BladeSource("gen3_spathla", "gen3_fixed_spathla_closed.stl",
                "c5b630244460ef5f61d7128716e7a109a608cf344d962cb578dc6540e30e9f5f",
                ((-37.44306182861328, 47.64384841918945, 34.80012130737305),
                 (120.60326385498047, 52.073829650878906, 151.46646118164062))),
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_binary_stl(data):
    if len(data) < 84:
        raise ValueError("Binary STL is shorter than its header")
    count = struct.unpack_from("<I", data, 80)[0]
    if count == 0 or len(data) != 84 + 50 * count:
        raise ValueError("Binary STL triangle count does not match its length")
    triangles = np.frombuffer(data, dtype=STL_RECORD, offset=84)["vertices"].copy()
    if not np.isfinite(triangles).all():
        raise ValueError("Binary STL has nonfinite vertices")
    return triangles


def connected_components(triangles):
    """Return original face indices connected through exactly equal vertices."""
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    parents = np.arange(len(faces))
    first_face = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    for index, face in enumerate(faces):
        for vertex in face:
            previous = first_face.setdefault(int(vertex), index)
            left, right = root(index), root(previous)
            if left != right:
                parents[max(left, right)] = min(left, right)
    groups = {}
    for index in range(len(faces)):
        groups.setdefault(root(index), []).append(index)
    return vertices, faces, [np.asarray(group, dtype=np.int64) for group in groups.values()]


def extract_blade(data, source):
    """Identify a pinned component and preserve its full 50-byte STL records."""
    if _sha256(data) != source.sha256:
        raise ValueError(f"Source SHA-256 mismatch for {source.name}")
    triangles = read_binary_stl(data)
    _, _, components = connected_components(triangles)
    candidates = []
    summaries = []
    expected = np.asarray(source.bounds_mm, dtype=np.float64)
    for indices in components:
        points = triangles[indices].reshape(-1, 3)
        bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        summaries.append({"faces": len(indices), "bounds_mm": bounds.tolist()})
        if len(indices) == source.faces and np.array_equal(bounds, expected):
            candidates.append(indices)
    if len(candidates) != 1:
        raise ValueError(f"Expected one pinned blade component for {source.name}, found {len(candidates)}")
    indices = candidates[0]
    records = b"".join(data[84 + 50 * int(i):84 + 50 * (int(i) + 1)] for i in indices)
    extracted = data[:80] + struct.pack("<I", len(indices)) + records
    return extracted, triangles, indices, summaries


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normal(triangle):
    return _cross(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))


def _inside(point, triangle, normal):
    signs = [_dot(_cross(_sub(triangle[(i + 1) % 3], triangle[i]), _sub(point, triangle[i])), normal)
             for i in range(3)]
    return all(value >= 0 for value in signs)


def _interpolate(a, b, fraction):
    return tuple(x + fraction * (y - x) for x, y in zip(a, b))


def _orient2(a, b, c, axes):
    x, y = axes
    return (b[x] - a[x]) * (c[y] - a[y]) - (b[y] - a[y]) * (c[x] - a[x])


def _coplanar_points(a, b, normal):
    axes = [i for i in range(3) if i != max(range(3), key=lambda i: abs(normal[i]))]
    points = {p for p in a if _inside(p, b, normal)}
    # The second triangle may have the opposite winding.
    points |= {p for p in b if _inside(p, a, _normal(a))}
    for i in range(3):
        p, q = a[i], a[(i + 1) % 3]
        for j in range(3):
            r, s = b[j], b[(j + 1) % 3]
            d0, d1 = _orient2(r, s, p, axes), _orient2(r, s, q, axes)
            if d0 != d1 and min(d0, d1) <= 0 <= max(d0, d1):
                point = _interpolate(p, q, d0 / (d0 - d1))
                if all(min(r[k], s[k]) <= point[k] <= max(r[k], s[k]) for k in axes):
                    points.add(point)
    return points


def exact_triangle_intersection(first, second):
    """Check two triangles using exact rationals of their stored float coordinates.

    This is an independent check of reported pairs, not a mesh-wide search.
    The returned coordinates are rounded only when serializing the evidence.
    """
    a, b = [tuple(tuple(Fraction(float(x)) for x in p) for p in triangle)
            for triangle in (first, second)]
    na, nb = _normal(a), _normal(b)
    if not any(na) or not any(nb):
        return {"kind": "degenerate", "intersects": None, "points_mm": []}
    da = tuple(_dot(_sub(p, b[0]), nb) for p in a)
    db = tuple(_dot(_sub(p, a[0]), na) for p in b)
    evidence = {"arithmetic": "exact rational predicates on binary STL float32 coordinates",
                "first_vertices_against_second_plane": [int(d > 0) - int(d < 0) for d in da],
                "second_vertices_against_first_plane": [int(d > 0) - int(d < 0) for d in db]}
    if any(all(d > 0 for d in ds) or all(d < 0 for d in ds) for ds in (da, db)):
        return {"kind": "disjoint", "intersects": False, "points_mm": [], **evidence}
    coplanar = not any(da) and not any(db)
    points = set()
    if coplanar:
        points = _coplanar_points(a, b, nb)
    else:
        for triangle, distances, other, normal in ((a, da, b, nb), (b, db, a, na)):
            for i in range(3):
                p, q = triangle[i], triangle[(i + 1) % 3]
                d0, d1 = distances[i], distances[(i + 1) % 3]
                if d0 == 0 and _inside(p, other, normal):
                    points.add(p)
                if d0 * d1 < 0:
                    point = _interpolate(p, q, d0 / (d0 - d1))
                    if _inside(point, other, normal):
                        points.add(point)
    ordered = sorted(points)
    kind = "disjoint"
    if len(points) == 1:
        kind = "point-contact"
    elif len(points) >= 2:
        kind = "coplanar-contact" if coplanar else "intersection-segment"
        if coplanar and any(any(_cross(_sub(p, ordered[0]), _sub(q, ordered[0])))
                            for p in ordered[1:] for q in ordered[1:]):
            kind = "coplanar-overlap"
    points_float = np.asarray([[float(x) for x in p] for p in ordered], dtype=np.float64)
    extent = 0.0 if len(points_float) < 2 else max(float(np.linalg.norm(p - q)) for p in points_float for q in points_float)
    return {"kind": kind, "intersects": bool(points), "points_mm": points_float.tolist(),
            "points_rational_mm": [[str(x) for x in p] for p in ordered],
            "maximum_point_separation_mm": extent,
            "strict_plane_straddle": all(min(ds) < 0 < max(ds) for ds in (da, db)), **evidence}


def mesh_topology(triangles):
    vertices, faces, components = connected_components(triangles)
    edges = Counter()
    orientations = Counter()
    for face in faces:
        for left, right in zip(face, np.roll(face, -1)):
            left, right = int(left), int(right)
            edge = (min(left, right), max(left, right))
            edges[edge] += 1
            orientations[edge] += 1 if left < right else -1
    triangle64 = triangles.astype(np.float64)
    cross = np.cross(triangle64[:, 1] - triangle64[:, 0], triangle64[:, 2] - triangle64[:, 0])
    volume = np.einsum("ij,ij->i", triangle64[:, 0], np.cross(triangle64[:, 1], triangle64[:, 2])).sum() / 6
    return {"faces": len(faces), "vertices": len(vertices), "connected_components": len(components),
            "boundary_edges": sum(count == 1 for count in edges.values()),
            "nonmanifold_edges": sum(count > 2 for count in edges.values()),
            "inconsistent_winding_edges": sum(count == 2 and orientations[edge] != 0 for edge, count in edges.items()),
            "degenerate_faces": int(np.count_nonzero(np.all(cross == 0, axis=1))),
            "signed_volume_mm3": float(volume)}


def audit_blade(path, triangles, original_indices):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.remove_duplicated_vertices()
    backend_triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    if not np.array_equal(backend_triangles, triangles):
        raise ValueError("Geometry backend changed triangle coordinates or order")
    pairs = np.asarray(mesh.get_self_intersecting_triangles(), dtype=np.int64).reshape(-1, 2)
    findings = []
    for left, right in pairs:
        result = exact_triangle_intersection(triangles[left], triangles[right])
        findings.append({"blade_triangle_indices": [int(left), int(right)],
                         "source_triangle_indices": [int(original_indices[left]), int(original_indices[right])],
                         "triangles_mm": triangles[[left, right]].tolist(), **result})
    topology = mesh_topology(triangles)
    checks = {"closed_edge_manifold": mesh.is_edge_manifold(allow_boundary_edges=False),
              "vertex_manifold": mesh.is_vertex_manifold(), "orientable": mesh.is_orientable(),
              "watertight": mesh.is_watertight()}
    accepted = (all(checks.values()) and not len(pairs) and topology["degenerate_faces"] == 0
                and topology["inconsistent_winding_edges"] == 0 and topology["signed_volume_mm3"] > 0)
    return {"accepted": bool(accepted), "status": "qualified" if accepted else "not-qualified",
            "backend": "open3d", "backend_version": o3d.__version__, "checks": checks,
            "reported_intersection_pairs": len(pairs), "exact_pair_checks": findings,
            "exact_check_scope": "Only backend-reported pairs; not an independent exhaustive search",
            "topology": topology}


def _write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_views(path, triangles, indices, findings):
    """Draw orthographic before/after projections without modifying geometry."""
    width, height = 1080, 700
    retained = set(map(int, indices))
    suspect = {i for finding in findings for i in finding["source_triangle_indices"]}
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font:14px sans-serif;fill:#202020}polygon{fill:none;stroke-width:.35}</style>',
                '<text x="16" y="22">Gray: omitted mount. Blue: unchanged blade. Red: reported intersection pairs.</text>']
    for row in range(2):
        for column, axes in enumerate(((0, 1), (0, 2), (1, 2))):
            xy = triangles[:, :, axes]
            lower = xy.min(axis=(0, 1))
            extent = xy.max(axis=(0, 1)) - lower
            scale = min(320 / max(float(extent[0]), 1e-9), 260 / max(float(extent[1]), 1e-9))
            offset = np.array([column * 360 + 20, row * 330 + 65])
            projected = (xy - lower) * scale
            projected[:, :, 1] = 260 - projected[:, :, 1]
            projected += offset
            label = "Source" if row == 0 else "Retained blade"
            elements.append(f'<text x="{offset[0]}" y="{offset[1] - 12}">{label}: {"xyz"[axes[0]]}/{"xyz"[axes[1]]}, raw STL mm</text>')
            # Draw marked triangles last so their outlines remain visible.
            for index in sorted(range(len(triangles)), key=lambda i: i in suspect):
                if row and index not in retained:
                    continue
                color = "#c02828" if index in suspect else "#1864ab" if index in retained else "#b0b0b0"
                points = " ".join(f"{p[0]:.4f},{p[1]:.4f}" for p in projected[index])
                elements.append(f'<polygon points="{points}" stroke="{color}"/>')
    elements.append("</svg>")
    with path.open("x") as stream:
        stream.write("\n".join(elements))


def prepare_blade_collision_meshes(source_directory, output_directory, *, write_views=True):
    """Write only a fresh directory; qualification failure remains explicit.

    Assets with failed geometry checks are diagnostic outputs, not approved SDF
    inputs. This function never repairs, recenters or otherwise changes a blade.
    """
    source_directory = Path(source_directory).resolve()
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_directory}")
    prepared = []
    for source in SOURCES:
        path = source_directory / source.filename
        data = path.read_bytes()
        prepared.append((source, path, *extract_blade(data, source)))
    # Check the optional geometry dependency before writing any output.
    import open3d as o3d

    output_directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "derivation": DERIVATION,
                "generator_sha256": _sha256(Path(__file__).read_bytes()),
                "numpy_version": np.__version__, "open3d_version": o3d.__version__,
                "retained_triangle_geometry_modified": False, "tracking_transforms_modified": False,
                "qualification": "pending", "tools": []}
    for source, source_path, extracted, all_triangles, indices, components in prepared:
        filename = source.filename.removesuffix(".stl") + "_blade_only.stl"
        destination = output_directory / filename
        with destination.open("xb") as stream:
            stream.write(extracted)
        triangles = all_triangles[indices]
        audit = audit_blade(destination, triangles, indices)
        evidence_name = filename.removesuffix(".stl") + "_geometry.json"
        _write_json(output_directory / evidence_name, audit)
        views = None
        if write_views:
            views = filename.removesuffix(".stl") + "_views.svg"
            _write_views(output_directory / views, all_triangles, indices, audit["exact_pair_checks"])
        manifest["tools"].append({"name": source.name, "collision_mesh": filename,
                                  "collision_sha256": _sha256(extracted), "source_mesh": str(source_path),
                                  "source_sha256": source.sha256, "units": "millimetres",
                                  "coordinate_frame": "raw_stl_visual", "topology": audit["topology"],
                                  "bounds_mm": {"min": list(source.bounds_mm[0]), "max": list(source.bounds_mm[1])},
                                  "source_components": components, "source_triangle_indices": indices.tolist(),
                                  "triangle_records_preserved": True, "geometry_accepted": audit["accepted"],
                                  "qualification_report": evidence_name,
                                  "qualification_report_sha256": _sha256((output_directory / evidence_name).read_bytes()),
                                  "views": views})
    accepted = all(row["geometry_accepted"] for row in manifest["tools"])
    manifest["qualification"] = "qualified" if accepted else "not-qualified"
    manifest["qualification_note"] = ("Blade geometry passes the recorded mesh checks; sampled fields still require qualification."
                                       if accepted else "Do not use as qualified SDF inputs. Review the reported geometry defects; no repairs were performed.")
    manifest_path = output_directory / "collision_manifest.json"
    _write_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path.read_bytes()),
            "accepted": accepted, "tools": manifest["tools"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-views", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_blade_collision_meshes(args.source_dir, args.output_dir, write_views=not args.no_views)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

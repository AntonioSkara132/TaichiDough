"""Prepare new blade meshes by bounded local edge flips, without moving vertices."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from . import mesh_validation, prepare_blade_collision_meshes as extraction


DERIVATION = "pinned-blade-local-edge-flips-v1"
MAX_SLIVER_ALTITUDE_MM = 0.0001
SOURCE_FACE_PAIRS = {
    "UR5e_spathla": ((5030, 5031), (3819, 3822), (6534, 6533), (3954, 3953),
                     (5000, 5002), (3938, 3939), (6940, 6934), (7082, 7083),
                     (4057, 4059), (5057, 5054), (3699, 3701), (6188, 6190),
                     (7033, 7035), (5830, 5829), (7120, 3775), (4390, 4389)),
    "gen3_spathla": ((5685, 5687), (7008, 7010), (5901, 5903), (7176, 7178),
                     (5737, 5739), (5809, 5811), (5802, 5804), (5768, 5770),
                     (5784, 5786), (5710, 5712), (5876, 5878), (5382, 5384),
                     (5698, 5700), (7312, 7314), (7293, 7295)),
}
BOUND_JUSTIFICATION = (
    "For each flip, project sliver vertex c onto its longest edge ab at q. "
    "Exact rational arithmetic verifies q lies on segment ab. Replacing c by q "
    "maps both triangulations onto triangle abd, with displacement at most h=|c-q| "
    "in either direction. Their Hausdorff distance is therefore at most 2h. "
    "The squared bound is checked exactly against its upward-rounded float value. "
    "No face participates in two flips, so the whole-mesh bound is the maximum "
    "patch bound. All vertices and bounding extrema are unchanged."
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _altitude(triangle):
    return float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
                 / max(np.linalg.norm(triangle[i] - triangle[(i + 1) % 3]) for i in range(3)))


def patch_distance_bound(a, b, c):
    """Return a certified upper bound on 2*distance(c, segment(a,b)), in mm."""
    a, b, c = [tuple(Fraction(float(x)) for x in point) for point in (a, b, c)]
    edge = tuple(y - x for x, y in zip(a, b))
    offset = tuple(y - x for x, y in zip(a, c))
    length_squared = sum(x * x for x in edge)
    if length_squared == 0:
        raise ValueError("Cannot flip a zero-length edge")
    dot = sum(x * y for x, y in zip(edge, offset))
    fraction = dot / length_squared
    if not 0 <= fraction <= 1:
        raise ValueError("Sliver projection falls outside its longest edge")
    height_squared = sum(x * x for x in offset) - dot * dot / length_squared
    squared_bound = 4 * height_squared
    bound = math.sqrt(float(squared_bound))
    while Fraction(bound) ** 2 < squared_bound:
        bound = math.nextafter(bound, math.inf)
    return {"hausdorff_upper_bound_mm": bound, "squared_bound_rational_mm2": str(squared_bound),
            "edge_projection_fraction": str(fraction)}


def apply_pinned_flips(extracted_data, source_triangle_indices, source):
    """Change only the specified triangle records; every vertex stays unchanged."""
    triangles = extraction.read_binary_stl(extracted_data)
    vertices, faces, _ = extraction.connected_components(triangles)
    vertices = vertices.astype(np.float64)
    original_faces = faces.copy()
    positions = {int(original): i for i, original in enumerate(source_triangle_indices)}
    used = set()
    changes = []
    for source_left, source_right in SOURCE_FACE_PAIRS[source.name]:
        left, right = positions[source_left], positions[source_right]
        if left in used or right in used:
            raise ValueError("Pinned flips must have disjoint face pairs")
        used.update((left, right))
        face = faces[left].copy()
        longest = int(np.argmax([np.linalg.norm(vertices[face[(i + 1) % 3]] - vertices[face[i]]) for i in range(3)]))
        a, b, c = map(int, (face[longest], face[(longest + 1) % 3], face[(longest + 2) % 3]))
        if set(map(int, faces[left])) & set(map(int, faces[right])) != {a, b}:
            raise ValueError("Pinned neighbor must share the longest sliver edge")
        d = int(next(v for v in faces[right] if v not in (a, b)))
        if any(c in other and d in other for i, other in enumerate(faces) if i not in (left, right)):
            raise ValueError("Replacement diagonal already belongs to another face")
        before = faces[[left, right]].copy()
        after = np.asarray(((c, a, d), (c, d, b)), dtype=np.int64)
        old_altitude = _altitude(vertices[face])
        new_altitude = min(_altitude(vertices[f]) for f in after)
        if not 0 < old_altitude < MAX_SLIVER_ALTITUDE_MM:
            raise ValueError("Pinned face is not a nondegenerate numerical sliver")
        if new_altitude <= MAX_SLIVER_ALTITUDE_MM or new_altitude <= 10 * old_altitude:
            raise ValueError("Flip does not sufficiently improve triangle altitude")
        neighbor = vertices[faces[right]]
        normal = np.cross(neighbor[1] - neighbor[0], neighbor[2] - neighbor[0])
        normal /= np.linalg.norm(normal)
        planarity = float(abs(np.dot(vertices[c] - neighbor[0], normal)))
        if planarity > MAX_SLIVER_ALTITUDE_MM:
            raise ValueError("Flip leaves the nearly coplanar patch")
        for f in after:
            new_normal = np.cross(vertices[f[1]] - vertices[f[0]], vertices[f[2]] - vertices[f[0]])
            if np.dot(new_normal, normal) / np.linalg.norm(new_normal) < 0.999:
                raise ValueError("Flip changes the patch orientation")
        bound = patch_distance_bound(vertices[a], vertices[b], vertices[c])
        faces[[left, right]] = after
        changes.append({"source_triangle_indices": [source_left, source_right],
                        "blade_triangle_indices": [left, right],
                        "before_vertices_mm": vertices[before].tolist(), "after_vertices_mm": vertices[after].tolist(),
                        "old_sliver_altitude_mm": old_altitude, "new_minimum_altitude_mm": new_altitude,
                        "quad_planarity_mm": planarity, **bound})
    changed_indices = sorted(used)
    records = np.frombuffer(extracted_data, dtype=extraction.STL_RECORD, offset=84).copy()
    records["vertices"][changed_indices] = vertices[faces[changed_indices]].astype(np.float32)
    changed_triangles = vertices[faces[changed_indices]]
    normals = np.cross(changed_triangles[:, 1] - changed_triangles[:, 0], changed_triangles[:, 2] - changed_triangles[:, 0])
    records["normal"][changed_indices] = (normals / np.linalg.norm(normals, axis=1)[:, None]).astype(np.float32)
    output = extracted_data[:80] + struct.pack("<I", len(records)) + records.tobytes()
    final = extraction.read_binary_stl(output)
    if not np.array_equal(np.unique(final.reshape(-1, 3), axis=0), vertices.astype(np.float32)):
        raise ValueError("Retriangulation changed the vertex set")
    if not np.array_equal(final.min(axis=(0, 1)), triangles.min(axis=(0, 1))) or not np.array_equal(final.max(axis=(0, 1)), triangles.max(axis=(0, 1))):
        raise ValueError("Retriangulation changed mesh bounds")
    actual_changed = np.flatnonzero(np.any(faces != original_faces, axis=1)).tolist()
    if actual_changed != changed_indices:
        raise ValueError("Changed faces do not match the pinned flip pairs")
    unchanged = set(range(len(records))) - used
    if any(output[84 + 50 * i:84 + 50 * (i + 1)] != extracted_data[84 + 50 * i:84 + 50 * (i + 1)] for i in unchanged):
        raise ValueError("An unlisted triangle record changed")
    return output, {"source_triangle_indices": source_triangle_indices.tolist(), "flips": changes,
                    "changed_blade_triangle_indices": changed_indices,
                    "vertex_displacement_mm": 0.0, "vertex_set_unchanged": True, "bounds_unchanged": True,
                    "unchanged_triangle_records_preserved": True,
                    "hausdorff_upper_bound_mm": max(row["hausdorff_upper_bound_mm"] for row in changes),
                    "geometry_bound_justification": BOUND_JUSTIFICATION}


def production_transform(triangles, name):
    """Call the preserved loading/transform arithmetic used by data.py."""
    from .reference_adapter import get_reference_modules, reference_policy

    # Verify the immutable snapshots; unrelated edits to live helpers do not
    # change the recorded coordinate transforms used for these assets.
    with reference_policy("frozen"):
        simulator = get_reference_modules().simulator
    if name == "UR5e_spathla":
        origin, rpy = simulator.UR_TOOL_VISUAL_ORIGIN, simulator.UR_TOOL_VISUAL_RPY
    elif name == "gen3_spathla":
        origin, rpy = simulator.KINOVA_TOOL_VISUAL_ORIGIN, simulator.KINOVA_TOOL_VISUAL_RPY
    else:
        raise ValueError(f"Unknown tool {name}")
    # Match load_binary_stl exactly: multiplication is performed on float32 data.
    vertices = np.array(np.asarray(triangles, dtype=np.float32).reshape(-1, 3) * 0.001, dtype=np.float32)
    rotation = simulator.rpy_to_matrix(rpy)
    transformed = simulator.mesh_in_tool_frame(vertices, origin, rotation).reshape(-1, 3, 3)
    return transformed, {"mesh_scale": 0.001, "visual_origin_m": origin.tolist(), "visual_rpy_rad": rpy.tolist(),
                         "rotation": rotation.tolist(), "arithmetic": "existing float32 STL scale and tool-frame transform",
                         "reference_policy": "frozen", "simulator_source_sha256": _sha256(Path(simulator.__file__).read_bytes())}


def validate_geometry(triangles):
    import open3d as o3d

    values = np.asarray(triangles, dtype=np.float64)
    topology = extraction.mesh_topology(values)
    if (topology["boundary_edges"] or topology["nonmanifold_edges"] or topology["inconsistent_winding_edges"]
            or topology["degenerate_faces"] or topology["connected_components"] != 1
            or not np.isfinite(topology["signed_volume_mm3"]) or topology["signed_volume_mm3"] <= 0):
        raise ValueError(f"Invalid blade topology: {topology}")
    vertices, inverse = np.unique(values.reshape(-1, 3), axis=0, return_inverse=True)
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.ascontiguousarray(vertices)),
                                     o3d.core.Tensor(np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int64))).to_legacy()
    if not mesh.is_vertex_manifold() or not mesh.is_orientable():
        raise ValueError("Blade has nonmanifold vertices or is not orientable")
    intersections = mesh_validation.validate_mesh_intersections(values)
    if not intersections["valid"] or not intersections["complete"]:
        raise ValueError(f"Blade intersections failed exhaustive validation: {intersections}")
    # The topology utility's unit suffix is for raw STL data. Name it explicitly
    # here because this same audit also accepts tool-frame metre coordinates.
    volume = topology.pop("signed_volume_mm3")
    return {"topology": topology, "signed_volume_in_coordinate_units_cubed": volume,
            "topology_backend": {"name": "open3d", "version": o3d.__version__},
            "vertex_manifold": True, "orientable": True, "intersection_validation": intersections}


def prepare_valid_blade_meshes(source_directory, output_directory):
    """Write a fresh directory only after both tools pass all geometry checks."""
    source_directory, output_directory = Path(source_directory).resolve(), Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_directory}")
    prepared = []
    for source in extraction.SOURCES:
        source_path = source_directory / source.filename
        extracted, original, indices, components = extraction.extract_blade(source_path.read_bytes(), source)
        data, derivation = apply_pinned_flips(extracted, indices, source)
        triangles = extraction.read_binary_stl(data)
        raw_audit = validate_geometry(triangles)
        transformed, transform_record = production_transform(triangles, source.name)
        transformed_audit = validate_geometry(transformed)
        report = {"schema": "taichidough/blade-local-retriangulation/v1", "derivation": DERIVATION,
                  "name": source.name, "source_mesh": str(source_path), "source_sha256": source.sha256,
                  "collision_sha256": _sha256(data), **derivation,
                  "raw_mm_validation": raw_audit, "production_tool_frame_m_validation": transformed_audit,
                  "production_transform": transform_record, "mesh_geometry_valid": True,
                  "field_qualification": "pending"}
        prepared.append((source, source_path, data, report, components))
    output_directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": extraction.SCHEMA, "derivation": DERIVATION,
                "mesh_qualification": "qualified", "field_qualification": "pending",
                "tracking_transforms_modified": False, "vertex_positions_modified": False,
                "generator_sha256": _sha256(Path(__file__).read_bytes()),
                "extraction_source_sha256": _sha256(Path(extraction.__file__).read_bytes()),
                "validation_identity": mesh_validation.validation_identity(),
                "test_modules": ["tests/test_prepare_valid_blade_meshes.py", "tests/test_mesh_validation.py"],
                "numpy_version": np.__version__, "tools": []}
    for source, source_path, data, report, components in prepared:
        filename = source.filename.removesuffix(".stl") + "_blade_retriangulated.stl"
        with (output_directory / filename).open("xb") as stream:
            stream.write(data)
        report_name = filename.removesuffix(".stl") + "_geometry.json"
        extraction._write_json(output_directory / report_name, report)
        topology = {**report["raw_mm_validation"]["topology"],
                    "signed_volume_mm3": report["raw_mm_validation"]["signed_volume_in_coordinate_units_cubed"]}
        manifest["tools"].append({"name": source.name, "collision_mesh": filename, "collision_sha256": _sha256(data),
                                  "source_mesh": str(source_path), "source_sha256": source.sha256,
                                  "units": "millimetres", "coordinate_frame": "raw_stl_visual", "topology": topology,
                                  "bounds_mm": {"min": list(source.bounds_mm[0]), "max": list(source.bounds_mm[1])},
                                  "source_components": components, "geometry_accepted": True,
                                  "mesh_qualification": "qualified", "field_qualification": "pending",
                                  "geometry_report": report_name,
                                  "geometry_report_sha256": _sha256((output_directory / report_name).read_bytes()),
                                  "hausdorff_upper_bound_mm": report["hausdorff_upper_bound_mm"]})
    manifest_path = output_directory / "collision_manifest.json"
    extraction._write_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path.read_bytes()),
            "mesh_geometry_valid": True, "field_qualification": "pending", "tools": manifest["tools"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=extraction.DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_valid_blade_meshes(args.source_dir, args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

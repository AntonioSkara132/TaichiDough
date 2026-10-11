#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build or inspect the one-frame Episode3 manual tool-pose Blender scene."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Sequence

import bpy  # pyright: ignore[reportMissingImports]
from mathutils import Matrix, Vector  # pyright: ignore[reportMissingImports]


COLLECTION_NAME = "Episode3_Manual_Tool_Pose"
CONTEXT_NAME = "Episode3_Context_Points"
TOOL_POINTS_NAME = "Episode3_Tool_HSV_Points"
TOOLS = ("UR5e_spathla", "gen3_spathla")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix(value: Sequence[Sequence[float]]) -> Matrix:
    result = Matrix(value)
    if len(result) != 4:
        raise ValueError("Expected a 4x4 matrix")
    return result


def validate_rigid(value: Matrix, name: str) -> None:
    if len(value) != 4 or any(not math.isfinite(number) for row in value for number in row):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if max(abs(value[3][index] - expected) for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))) > 1e-6:
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = value.to_3x3()
    columns = [Vector((rotation[0][index], rotation[1][index], rotation[2][index])) for index in range(3)]
    if max(abs(column.length - 1.0) for column in columns) > 1e-5:
        raise ValueError(f"{name} contains scale")
    if max(abs(columns[a].dot(columns[b])) for a in range(3) for b in range(a + 1, 3)) > 1e-5:
        raise ValueError(f"{name} contains shear")
    if abs(rotation.determinant() - 1.0) > 1e-5:
        raise ValueError(f"{name} rotation determinant is not +1")


def read_binary_ply(path: Path) -> list[tuple[float, float, float]]:
    raw = path.read_bytes()
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"PLY header is incomplete: {path}")
    header_end = end + len(b"end_header\n")
    header = raw[:header_end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"PLY must be binary little-endian: {path}")
    count_rows = [row for row in header if row.startswith("element vertex ")]
    if len(count_rows) != 1:
        raise ValueError(f"PLY vertex count is missing: {path}")
    count = int(count_rows[0].split()[-1])
    payload = raw[header_end:]
    if len(payload) != count * 12:
        raise ValueError(f"PLY payload size differs: {path}")
    return list(struct.iter_unpack("<fff", payload))


def read_binary_stl(path: Path, visual: Matrix, scale: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"STL is too short: {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Unsupported STL encoding: {path}")
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for triangle in range(count):
        values = struct.unpack_from("<12fH", raw, 84 + triangle * 50)
        base = len(vertices)
        for offset in (3, 6, 9):
            point = visual @ Vector((values[offset] * scale, values[offset + 1] * scale, values[offset + 2] * scale, 1.0))
            vertices.append((point.x, point.y, point.z))
        faces.append((base, base + 1, base + 2))
    return vertices, faces


def owned_collection() -> bpy.types.Collection | None:
    collection = bpy.data.collections.get(COLLECTION_NAME)
    if collection is not None and collection.get("episode3_manual_pose_owner") is True:
        return collection
    return None


def remove_owned_collection() -> None:
    collection = owned_collection()
    if collection is None:
        return
    for obj in list(collection.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and getattr(data, "users", 1) == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Camera):
                bpy.data.cameras.remove(data)
    bpy.data.collections.remove(collection)


def material(name: str, color: tuple[float, float, float, float], *, metallic: float = 0.0) -> bpy.types.Material:
    value = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    value.diffuse_color = color
    value.use_nodes = True
    principled = value.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Alpha"].default_value = color[3]
        principled.inputs["Metallic"].default_value = metallic
        principled.inputs["Roughness"].default_value = 0.35
    value.blend_method = "BLEND" if color[3] < 1.0 else "OPAQUE"
    value.show_transparent_back = True
    return value


def point_node_group(name: str, radius: float, point_material: bpy.types.Material) -> bpy.types.NodeTree:
    group = bpy.data.node_groups.get(name)
    if group is not None:
        bpy.data.node_groups.remove(group, do_unlink=True)
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.inputs.new("NodeSocketGeometry", "Geometry")
    group.outputs.new("NodeSocketGeometry", "Geometry")
    nodes = group.nodes
    links = group.links
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    to_points = nodes.new("GeometryNodeMeshToPoints")
    to_points.mode = "VERTICES"
    to_points.inputs["Radius"].default_value = radius
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = point_material
    links.new(input_node.outputs["Geometry"], to_points.inputs["Mesh"])
    links.new(to_points.outputs["Points"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])
    return group


def create_point_object(
    collection: bpy.types.Collection,
    name: str,
    points: list[tuple[float, float, float]],
    color: tuple[float, float, float, float],
    radius: float,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(points, [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    point_material = material(f"{name}_Material", color)
    modifier = obj.modifiers.new("Display as points", "NODES")
    modifier.node_group = point_node_group(f"{name}_Geometry", radius, point_material)
    obj["episode3_manual_pose_owner"] = True
    return obj


def create_tool(
    collection: bpy.types.Collection,
    record: dict[str, Any],
    color: tuple[float, float, float, float],
) -> tuple[bpy.types.Object, bpy.types.Object]:
    name = record["name"]
    marker = bpy.data.objects.new(f"Marker_{name}", None)
    marker.empty_display_type = "ARROWS"
    marker.empty_display_size = 0.08
    marker.matrix_world = matrix(record["blender_from_marker"])
    marker["episode3_manual_pose_owner"] = True
    collection.objects.link(marker)

    stl = Path(record["collision_stl"]["path"])
    if sha256(stl) != record["collision_stl"]["sha256"]:
        raise ValueError(f"STL hash differs for {name}")
    vertices, faces = read_binary_stl(
        stl,
        matrix(record["tool_link_from_scaled_raw_visual"]),
        float(record["stl_scale_to_metres"]),
    )
    mesh = bpy.data.meshes.new(f"Registration_{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"Registration_{name}", mesh)
    collection.objects.link(obj)
    obj.parent = marker
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = matrix(record["initial_marker_from_mesh"])
    obj.rotation_mode = "XYZ"
    obj.lock_scale = (True, True, True)
    obj.data.materials.append(material(f"Registration_{name}_Material", color, metallic=0.15))
    obj.show_in_front = True
    obj["episode3_manual_pose_owner"] = True
    obj["tool_name"] = name
    obj["initial_marker_from_mesh"] = json.dumps(record["initial_marker_from_mesh"])
    return marker, obj


def load_manifest(bundle: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = bundle / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "taichidough/episode3-blender-pose-frame/v1":
        raise ValueError("Frame manifest schema is not supported")
    if manifest.get("validated_geometry") is not False or manifest.get("episode18_registration_used") is not False:
        raise ValueError("Frame manifest incorrectly claims accepted geometry")
    for filename, record in manifest["point_files"].items():
        path = bundle / filename
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Point file hash differs: {filename}")
    return manifest, manifest_path


def build(bundle: Path, blend_path: Path) -> None:
    manifest, manifest_path = load_manifest(bundle)
    context = read_binary_ply(bundle / "context_points.ply")
    tool_points = read_binary_ply(bundle / "tool_hsv_points.ply")
    remove_owned_collection()
    collection = bpy.data.collections.new(COLLECTION_NAME)
    collection["episode3_manual_pose_owner"] = True
    bpy.context.scene.collection.children.link(collection)

    create_point_object(collection, CONTEXT_NAME, context, (0.20, 0.22, 0.24, 0.65), 0.0014)
    create_point_object(collection, TOOL_POINTS_NAME, tool_points, (1.0, 0.03, 0.01, 1.0), 0.0024)
    tool_records = {record["name"]: record for record in manifest["tools"]}
    created = []
    created.append(create_tool(collection, tool_records["UR5e_spathla"], (0.02, 0.35, 1.0, 0.62)))
    created.append(create_tool(collection, tool_records["gen3_spathla"], (0.05, 1.0, 0.24, 0.62)))

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene["episode3_manual_pose_bundle"] = str(bundle)
    scene["episode3_manual_pose_manifest"] = str(manifest_path)
    scene["episode3_manual_pose_manifest_sha256"] = sha256(manifest_path)
    scene["episode3_manual_pose_frame"] = int(manifest["frame"]["raw_pointcloud_ordinal"])
    scene["instructions"] = "Move and rotate Registration_UR5e_spathla and Registration_gen3_spathla; do not scale; save the .blend file."
    scene.world.color = (0.025, 0.025, 0.025)

    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    created[0][1].select_set(True)
    bpy.context.view_layer.objects.active = created[0][1]
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))


def matrix_rows(value: Matrix) -> list[list[float]]:
    return [[float(value[row][column]) for column in range(4)] for row in range(4)]


def tool_result(name: str) -> dict[str, Any]:
    marker = bpy.data.objects.get(f"Marker_{name}")
    obj = bpy.data.objects.get(f"Registration_{name}")
    if marker is None or obj is None:
        raise ValueError(f"Blender scene is missing {name}")
    candidate = marker.matrix_world.inverted_safe() @ obj.matrix_world
    validate_rigid(candidate, f"marker_from_mesh for {name}")
    initial = matrix(json.loads(obj["initial_marker_from_mesh"]))
    delta = candidate @ initial.inverted_safe()
    validate_rigid(delta, f"delta for {name}")
    rotation = delta.to_3x3().to_quaternion().normalized()
    if rotation.w < 0.0:
        rotation.negate()
    angle = float(rotation.angle)
    if abs(angle) < 1e-12:
        rotvec = [0.0, 0.0, 0.0]
    else:
        rotvec = [math.degrees(angle) * float(value) for value in rotation.axis]
    candidate_quaternion = candidate.to_3x3().to_quaternion().normalized()
    if candidate_quaternion.w < 0.0:
        candidate_quaternion.negate()
    return {
        "name": name,
        "initial_marker_from_mesh": matrix_rows(initial),
        "candidate_marker_from_mesh": matrix_rows(candidate),
        "candidate_translation_m": [float(candidate[index][3]) for index in range(3)],
        "candidate_quaternion_xyzw": [
            float(candidate_quaternion.x), float(candidate_quaternion.y),
            float(candidate_quaternion.z), float(candidate_quaternion.w),
        ],
        "delta_marker_local": matrix_rows(delta),
        "translation_delta_mm": [float(delta[index][3]) * 1000.0 for index in range(3)],
        "translation_norm_mm": float(delta.to_translation().length * 1000.0),
        "rotation_delta_rotvec_deg": rotvec,
        "rotation_angle_deg": math.degrees(angle),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def extract(bundle: Path, output: Path) -> dict[str, Any]:
    manifest, manifest_path = load_manifest(bundle)
    results = [tool_result(name) for name in TOOLS]
    report = {
        "schema": "taichidough/episode3-manual-tool-registration/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "manual_candidate",
        "validated_geometry": False,
        "episode18_registration_used": False,
        "blend_file": str(Path(bpy.data.filepath).resolve()),
        "blender_version": bpy.app.version_string,
        "bundle_manifest": str(manifest_path),
        "bundle_manifest_sha256": sha256(manifest_path),
        "frame": manifest["frame"],
        "matrix_convention": "marker_from_mesh maps mesh/tool-link coordinates into recorded marker coordinates",
        "composition": manifest["composition"],
        "source_hashes": manifest["source_hashes"],
        "results": results,
        "production_configs_modified": False,
        "operations_not_run": ["calibration", "automatic_registration", "MPM", "gradient", "material_fit", "dataset_creation"],
    }
    atomic_json(output, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv if argv is None else argv)
    values = values[values.index("--") + 1:] if "--" in values else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "extract"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--blend", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = args.bundle.resolve()
    if args.mode == "build":
        if args.blend is None:
            raise ValueError("--blend is required in build mode")
        build(bundle, args.blend.resolve())
        print(json.dumps({"mode": "build", "blend": str(args.blend.resolve())}))
    else:
        if args.output is None:
            raise ValueError("--output is required in extract mode")
        report = extract(bundle, args.output.resolve())
        print(json.dumps({"mode": "extract", "output": str(args.output.resolve()), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

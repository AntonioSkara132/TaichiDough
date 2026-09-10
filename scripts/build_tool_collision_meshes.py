#!/usr/bin/env python3
"""Build validated watertight collision solids from the bundled visual STL meshes.

The outputs stay in the raw visual-STL millimetre coordinate frame.  The simulator
applies its existing scale and URDF visual-to-link transform exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import struct
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_closing
from skimage.measure import marching_cubes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLS = (
    ("UR5e_spathla", "ur_spathla.stl", "ur_spathla_collision_solid.stl"),
    ("gen3_spathla", "gen3_spathla.stl", "gen3_spathla_collision_solid.stl"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is too short for binary STL")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + triangle_count * 50:
        raise ValueError(f"{path} is not a supported binary STL")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.array(np.frombuffer(raw, dtype=dtype, count=triangle_count, offset=84)["vertices"], dtype=np.float64)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edges_a, edges_b)
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 1e-12] /= lengths[lengths > 1e-12, None]
    records = np.empty(len(triangles), dtype=np.dtype([
        ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
    ]))
    records["normal"] = normals.astype(np.float32)
    records["vertices"] = triangles.astype(np.float32)
    records["attribute"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"TaichiDough watertight collision solid v1".ljust(80, b"\0")
    path.write_bytes(header + struct.pack("<I", len(records)) + records.tobytes())


def ball(radius: int) -> np.ndarray:
    coordinates = np.ogrid[tuple(slice(-radius, radius + 1) for _ in range(3))]
    squared_distance = np.zeros((radius * 2 + 1,) * 3, dtype=np.int64)
    for component in coordinates:
        squared_distance += component * component
    return squared_distance <= radius * radius


def rasterize_surface(triangles: np.ndarray, minimum: np.ndarray, pitch: float, dimensions: np.ndarray) -> np.ndarray:
    surface = np.zeros(tuple(dimensions), dtype=bool)
    for triangle in triangles:
        longest_edge = max(np.linalg.norm(triangle[1] - triangle[0]), np.linalg.norm(triangle[2] - triangle[1]), np.linalg.norm(triangle[0] - triangle[2]))
        subdivisions = max(1, int(np.ceil(longest_edge / pitch)))
        samples = []
        for row in range(subdivisions + 1):
            for column in range(subdivisions + 1 - row):
                u, v = row / subdivisions, column / subdivisions
                samples.append(triangle[0] + u * (triangle[1] - triangle[0]) + v * (triangle[2] - triangle[0]))
        indices = np.rint((np.asarray(samples) - minimum) / pitch).astype(np.int64)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    marked = np.clip(indices + (ox, oy, oz), 0, dimensions - 1)
                    surface[marked[:, 0], marked[:, 1], marked[:, 2]] = True
    return surface


def exterior(empty: np.ndarray) -> np.ndarray:
    result = np.zeros_like(empty, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque()
    nx, ny, nz = empty.shape
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if x not in (0, nx - 1) and y not in (0, ny - 1) and z not in (0, nz - 1):
                    continue
                if empty[x, y, z]:
                    result[x, y, z] = True
                    queue.append((x, y, z))
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            xn, yn, zn = x + dx, y + dy, z + dz
            if 0 <= xn < nx and 0 <= yn < ny and 0 <= zn < nz and empty[xn, yn, zn] and not result[xn, yn, zn]:
                result[xn, yn, zn] = True
                queue.append((xn, yn, zn))
    return result


def topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    edges = Counter()
    for face in np.asarray(faces, dtype=np.int64):
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges[tuple(sorted((int(left), int(right))))] += 1
    incidences = Counter(edges.values())
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "boundary_edges": int(sum(count for incidence, count in incidences.items() if incidence == 1)),
        "nonmanifold_edges": int(sum(count for incidence, count in incidences.items() if incidence > 2)),
    }


def build_solid(triangles: np.ndarray, pitch_mm: float, closing_radius_voxels: int, padding_voxels: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lower = triangles.reshape(-1, 3).min(axis=0) - padding_voxels * pitch_mm
    upper = triangles.reshape(-1, 3).max(axis=0) + padding_voxels * pitch_mm
    dimensions = np.ceil((upper - lower) / pitch_mm).astype(np.int64) + 1
    surface = rasterize_surface(triangles, lower, pitch_mm, dimensions)
    closed = binary_closing(surface, structure=ball(closing_radius_voxels))
    solid = ~exterior(~closed)
    if not solid.any() or solid.all():
        raise ValueError("solidification produced an empty or full volume")
    vertices, faces, _normals, _values = marching_cubes(solid.astype(np.float32), level=0.5, spacing=(pitch_mm,) * 3)
    vertices += lower
    stats = {
        "voxel_origin_mm": lower.tolist(),
        "voxel_pitch_mm": float(pitch_mm),
        "voxel_dimensions": dimensions.tolist(),
        "padding_voxels": int(padding_voxels),
        "closing_radius_voxels": int(closing_radius_voxels),
        "surface_voxels": int(surface.sum()),
        "closed_surface_voxels": int(closed.sum()),
        "solid_voxels": int(solid.sum()),
        "solid_volume_mm3": float(solid.sum() * pitch_mm ** 3),
        "marching_cubes_level": 0.5,
    }
    return vertices, faces, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=PROJECT_ROOT / "meshes")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "meshes")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--voxel-pitch-mm", type=float, default=1.0)
    parser.add_argument("--closing-radius-voxels", type=int, default=1)
    parser.add_argument("--padding-voxels", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.voxel_pitch_mm <= 0 or args.closing_radius_voxels < 0 or args.padding_voxels < 2:
        raise ValueError("voxel pitch must be positive, closing radius nonnegative, and padding at least two voxels")
    manifest_path = args.manifest or args.output_dir / "tool_collision_meshes_v1.json"
    tools = []
    for name, source_name, collision_name in DEFAULT_TOOLS:
        source_path = args.mesh_dir / source_name
        collision_path = args.output_dir / collision_name
        triangles = load_binary_stl(source_path)
        vertices, faces, build = build_solid(triangles, args.voxel_pitch_mm, args.closing_radius_voxels, args.padding_voxels)
        checks = topology(vertices, faces)
        if checks["boundary_edges"] or checks["nonmanifold_edges"]:
            raise ValueError(f"{name} collision mesh is not watertight: {checks}")
        write_binary_stl(collision_path, vertices, faces)
        tools.append({
            "name": name,
            "source_mesh": source_name,
            "source_sha256": sha256(source_path),
            "collision_mesh": collision_name,
            "collision_sha256": sha256(collision_path),
            "coordinate_frame": "raw_stl_visual",
            "units": "millimetres",
            "topology": checks,
            "bounds_mm": {"minimum": vertices.min(axis=0).tolist(), "maximum": vertices.max(axis=0).tolist()},
            "build": build,
        })
        print(f"Wrote {collision_path}: {checks['faces']} faces, {build['solid_voxels']} solid voxels")
    document = {
        "schema": "taichidough/tool-collision-meshes/v1",
        "generator": {"script": "scripts/build_tool_collision_meshes.py", "algorithm": "triangle voxelization + closing + exterior fill + marching cubes", "scipy": importlib.metadata.version("scipy"), "scikit_image": importlib.metadata.version("scikit-image")},
        "tools": tools,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

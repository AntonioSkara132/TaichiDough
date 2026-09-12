"""Freeze approved Episode 18 registrations into new, hash-checked inputs (no simulation)."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/differentiable_mpm"
DEST = EXP / "data/episode18_registered_tools_v1"
CONFIG = EXP / "configs/episode18_table_aligned_registered_tools.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def topology(path):
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    raw = path.read_bytes()
    triangles = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].astype(np.float64)
    assert len(triangles) == int.from_bytes(raw[80:84], "little")
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, ids, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    balance = np.bincount(ids, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    result = dict(faces=len(faces), vertices=len(vertices), boundary_edges=int(np.sum(counts == 1)),
                  nonmanifold_edges=int(np.sum(counts > 2)), inconsistent_winding_edges=int(np.sum(balance != 0)),
                  degenerate_faces=int(np.sum(np.linalg.norm(normals, axis=1) == 0)),
                  signed_volume_mm3=float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6))
    assert result["boundary_edges"] == result["nonmanifold_edges"] == result["inconsistent_winding_edges"] == result["degenerate_faces"] == 0, result
    return result, {"min": vertices.min(0).tolist(), "max": vertices.max(0).tolist()}


def main():
    if DEST.exists() or CONFIG.exists():
        raise FileExistsError("New input directory/config already exists; refusing to overwrite")
    source_config = EXP / "data/episode18_table_aligned_v1/episode18_table_aligned_coulomb.json"
    config = json.loads(source_config.read_text())
    geometry_path = ROOT / "configs/tool_geometry_episode18_sdf.json"
    geometry = json.loads(geometry_path.read_text())
    registrations = [EXP / "runs/ur5e_mirrored_collision_registration_20260912_v1/tool_0_result.json",
                     EXP / "runs/tool_registration_20260912_shared_v1/tool_1_result.json"]
    meshes = [ROOT / "meshes/ur_spathla_collision_solid.stl", ROOT / "meshes/gen3_spathla_collision_solid.stl"]
    preserved = {str(p): sha(p) for p in [source_config, geometry_path, *registrations, *meshes]}
    assert sha(meshes[0]) == "65e2afa80894c45dde9ef1791786c8652d1de22a0a6b39de098fb2567552b251"
    gen3 = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:meshes/gen3_spathla_collision_solid.stl"])
    assert hashlib.sha256(gen3).hexdigest() == "c4d7a910e928d265f387003f33638a3ac123f7c8472af5d691882a099fb4867a"
    DEST.mkdir(parents=True)
    rows = []
    for j, (tool, result_path, source_mesh, key) in enumerate(zip(geometry["tools"], registrations, meshes, ["ur_collision_mesh", "kinova_collision_mesh"])):
        result = json.loads(result_path.read_text())
        assert tool["name"] == result["name"]
        transform = np.asarray(result["candidate_marker_from_mesh"])
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-12)
        np.testing.assert_array_equal(transform[3], [0, 0, 0, 1])
        tool["marker_from_mesh"] = transform.tolist()
        target = DEST / source_mesh.name
        if j == 0:
            shutil.copyfile(source_mesh, target)
        else:
            with target.open("xb") as stream:
                stream.write(gen3)
        audit, bounds = topology(target)
        rows.append(dict(name=tool["name"], collision_mesh=target.name, collision_sha256=sha(target),
                         units="millimetres", coordinate_frame="raw_stl_visual", topology=audit, bounds_mm=bounds,
                         registration_file=f"tool_{j}_registration.json", registration_sha256=sha(result_path)))
        shutil.copyfile(result_path, DEST / f"tool_{j}_registration.json")
        config["paths"][key] = str(target.relative_to(ROOT))
    for key in ["calibration", "initial_particles", "reconstruction_metadata"]:
        source = ROOT / config["paths"][key]
        assert sha(source) == config["expected_sha256"][key]
        preserved[str(source)] = sha(source)
        target = DEST / source.name
        shutil.copyfile(source, target)
        config["paths"][key] = str(target.relative_to(ROOT))
    write_json(DEST / "tool_geometry.json", geometry)
    write_json(DEST / "collision_manifest.json", {"schema": "taichidough/tool-collision-meshes/v1", "tools": rows})
    config["name"] = "episode18-table-aligned-registered-tools"
    for key, filename in [("tool_geometry", "tool_geometry.json"), ("collision_manifest", "collision_manifest.json")]:
        config["paths"][key] = str((DEST / filename).relative_to(ROOT))
    config["expected_sha256"] = {key: sha(ROOT / value) for key, value in config["paths"].items() if key != "episode"}
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG, config)
    assert all(sha(path) == digest for path, digest in preserved.items())
    write_json(DEST / "preparation_provenance.json", dict(
        sources_unchanged=True, source_hashes=preserved, config_sha256=sha(CONFIG),
        gen3_asset="Original registered collision solid recovered read-only from Git; current working file adds a separate 2 mm origin cube (12 triangles).",
        ur_asset="Exact mirrored collision solid used in approved new registration, including its separate origin cube; no further reflection or geometry edits.",
        winding="Input winding preserved. Frozen SDF uses exterior flood fill, not face normals; normals are signed-distance gradients.",
        transform="scene_from_source @ source_from_marker(t) @ candidate_marker_from_mesh @ visual_origin_RPY @ scale; full candidate applied once.",
        box_proxy="Inherited marker_from_collider and half extents are inactive in SDF mode and are not registered box proxies.",
        simulation_run=False, calibration_run=False))
    print(CONFIG, flush=True)


if __name__ == "__main__":
    main()

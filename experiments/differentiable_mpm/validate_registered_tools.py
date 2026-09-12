"""Host-only registered-tool input validation. Does not initialize Taichi or run MPM."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .data import prepare_experiment
from .prepare_registered_tools import CONFIG, DEST, topology, write_json
from .prepare_table_aligned import mesh_corners, tool_support
from .reference_adapter import get_reference_modules, reference_policy


def winding(points, triangles):
    results = []
    for point in points:
        a, b, c = (triangles - point).transpose(1, 0, 2)
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = la*lb*lc + np.einsum("ij,ij->i", a, b)*lc + np.einsum("ij,ij->i", b, c)*la + np.einsum("ij,ij->i", c, a)*lb
        results.append(float(np.sum(2*np.arctan2(numerator, denominator))/(4*np.pi)))
    return np.asarray(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DEST / "validation.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_config(args.config)
    report = {"simulation_run": False, "gradient_run": False, "cuda_qualified": False}
    with reference_policy("frozen"):
        print("Preparing training inputs and both SDFs (no simulation)", flush=True)
        train = prepare_experiment(config, split="training", build_sdf=True)
        print("Preparing validation inputs", flush=True)
        validation = prepare_experiment(config, split="validation", build_sdf=False)
        report["training"] = train.summary()
        report["validation"] = validation.summary()
        helpers = get_reference_modules()
        dyn, sim = helpers.dynamics, helpers.simulator
        full = dyn.ToolReplay(train.sequence, train.calibration, 0, len(train.sequence.times)-1,
                              max_gap_s=config.replay_max_gap_s,
                              marker_from_tool_frames=train.tool_geometry.marker_from_mesh)
        position_error, rotation_error, camera_error = 0., 0., 0.
        settings = [("ur_collision_mesh", sim.UR_TOOL_VISUAL_ORIGIN, sim.UR_TOOL_VISUAL_RPY),
                    ("kinova_collision_mesh", sim.KINOVA_TOOL_VISUAL_ORIGIN, sim.KINOVA_TOOL_VISUAL_RPY)]
        by_name = dict(zip(["UR5e_spathla", "gen3_spathla"], settings))
        sdf_reports = []
        for j, name in enumerate(train.sequence.names):
            key, origin, rpy = by_name[name]
            vertices, _ = sim.load_binary_stl(config.paths[key], config.tool_mesh_scale)
            mesh = sim.mesh_in_tool_frame(vertices, origin, sim.rpy_to_matrix(rpy)).astype(np.float64)
            triangles = mesh.reshape(-1, 3, 3)
            distance, gradient = train.sdf.distances[j], train.sdf.gradients[j]
            lower, spacing = train.sdf.minimums[j], train.sdf.spacings[j]
            interior = np.argwhere(distance < -float(max(spacing)))
            assert len(interior) > 0
            interior = interior[np.linspace(0, len(interior)-1, min(16, len(interior))).astype(int)]
            inside_points = lower + interior*spacing
            inside_winding = winding(inside_points, triangles)
            assert np.all(np.abs(inside_winding) > .99), inside_winding
            face_checks = []
            for axis, side in itertools.product(range(3), [0, -1]):
                index = [distance.shape[0]//2]*3
                index[axis] = side
                assert distance[tuple(index)] > 0
                outward = float(gradient[tuple(index)][axis]) * (-1 if side == 0 else 1)
                assert outward > 0, (axis, side, outward)
                face_checks.append(outward)
            exterior = np.array(list(itertools.product([0, distance.shape[0]-1], repeat=3)))
            outside_winding = winding(lower + exterior*spacing, triangles)
            assert np.max(np.abs(outside_winding)) < 1e-5
            sdf_reports.append(dict(name=name, topology=topology(config.paths[key])[0],
                                    negative_voxels=int(np.sum(distance < 0)), positive_voxels=int(np.sum(distance > 0)),
                                    minimum_distance_m=float(distance.min()), maximum_distance_m=float(distance.max()),
                                    interior_winding_abs_range=[float(abs(inside_winding).min()), float(abs(inside_winding).max())],
                                    exterior_winding_abs_max=float(abs(outside_winding).max()),
                                    boundary_face_gradient_outward_components=face_checks))
            for i in range(len(full.times)):
                source_from_marker = np.eye(4)
                source_from_marker[:3, :3] = dyn.quaternion_matrix(train.sequence.poses[i, j, 3:7])
                source_from_marker[:3, 3] = train.sequence.poses[i, j, :3]
                expected = train.calibration.scene_from_source @ source_from_marker @ train.tool_geometry.marker_from_mesh[j]
                actual = np.eye(4)
                actual[:3, :3] = dyn.quaternion_matrix(full.poses[i, j, 3:7])
                actual[:3, 3] = full.poses[i, j, :3]
                position_error = max(position_error, float(abs(expected[:3, 3]-actual[:3, 3]).max()))
                rotation_error = max(rotation_error, float(abs(expected[:3, :3]-actual[:3, :3]).max()))
                world = mesh[::1000] @ actual[:3, :3].T + actual[:3, 3]
                camera_from_scene = np.linalg.inv(train.calibration.scene_from_camera)
                camera_from_tool = camera_from_scene @ expected
                camera_a = world @ camera_from_scene[:3, :3].T + camera_from_scene[:3, 3]
                camera_b = mesh[::1000] @ camera_from_tool[:3, :3].T + camera_from_tool[:3, 3]
                camera_error = max(camera_error, float(abs(camera_a-camera_b).max()))
        assert position_error < 1e-7 and rotation_error < 1e-12 and camera_error < 1e-7
        support = tool_support(full, mesh_corners(config, sim, train.sequence.names), dyn)
        assert min(support["min"][0], support["min"][2]) >= 0 and max(support["max"]) < 1, support
        report.update(sdf_checks=sdf_reports, full_episode_tool_support=support,
                      transform_composition_max_errors={"position_m": position_error, "rotation": rotation_error, "mesh_camera_m": camera_error},
                      full_episode_frames=len(full.times), passed=True)
        write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ["sdf_checks", "full_episode_tool_support", "transform_composition_max_errors", "passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_replay_sdf_collision import (
    MANIFEST_SCHEMA,
    ToolAsset,
    create_report,
    parse_args,
    rebuild_tool_sdf,
    selected_contact_rows,
    validate_collision_manifest,
    validate_contact_debug,
    validate_sdf_replay_metadata,
)


def write_binary_stl(path, size=40.0):
    vertices = np.array([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size],
    ], dtype=np.float32)
    faces = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
    data = bytearray(b"synthetic cube".ljust(80, b"\0"))
    data.extend(struct.pack("<I", len(faces)))
    for face in faces:
        data.extend(struct.pack("<12fH", 0, 0, 0, *vertices[face[0]], *vertices[face[1]], *vertices[face[2]], 0))
    path.write_bytes(data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplaySdfCollisionDiagnosticTests(unittest.TestCase):
    def replay_metadata(self):
        return {
            "replay": {"tool_collision": "sdf", "tool_pose_frame": "mesh_tool_link", "tool_names": ["UR5e_spathla", "gen3_spathla"]},
            "parameters": {"tool_mesh_scale": .001, "tool_sdf_resolution": 16},
            "frames": [{"frame": 0, "source_frame": 2, "sim_time_s": 0.0}, {"frame": 1, "source_frame": 3, "sim_time_s": .02}],
        }

    def test_replay_validation_rejects_non_sdf_and_duplicate_names(self):
        metadata = self.replay_metadata()
        metadata["replay"]["tool_collision"] = "box"
        with self.assertRaisesRegex(ValueError, "tool_collision"):
            validate_sdf_replay_metadata(metadata)
        metadata = self.replay_metadata()
        metadata["replay"]["tool_names"] = ["same", "same"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_sdf_replay_metadata(metadata)

    def test_manifest_validation_rejects_unmatched_data(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_collision_manifest({"schema": "wrong", "tools": []})
        manifest = {"schema": MANIFEST_SCHEMA, "tools": [{"name": "first", "source_mesh": "a", "collision_mesh": "b", "coordinate_frame": "raw_stl_visual"}, {"name": "first", "source_mesh": "a", "collision_mesh": "b", "coordinate_frame": "raw_stl_visual"}]}
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_collision_manifest(manifest)

    def test_rebuild_has_inside_and_exterior_sdf_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); mesh = root / "cube.stl"; write_binary_stl(mesh)
            asset = ToolAsset("UR5e_spathla", mesh, mesh, digest(mesh), digest(mesh), {})
            inspection = rebuild_tool_sdf(asset, 16, 0.0, .001)
            self.assertGreater(inspection.statistics["negative_voxels"], 0)
            self.assertGreater(inspection.statistics["positive_voxels"], 0)
            self.assertEqual(inspection.statistics["nonfinite_voxels"], 0)
            self.assertTrue(np.all(np.asarray(inspection.statistics["voxel_spacing_scene_m"]) > 0))

    def test_contact_debug_preserves_proximity_and_response_distinction(self):
        replay = validate_sdf_replay_metadata(self.replay_metadata())
        debug = {
            "schema": "taichidough/replay-sdf-contact-debug/v1", "tool_collision": "sdf", "tool_names": replay["tool_names"],
            "frames": [{"frame": 0, "snapshot_without_substep_evaluations": True, "tools": [
                {"grid_nodes": {"contact_candidates": 8, "applied_responses": 0, "inward_normal_velocity_removed": 0}, "particles": {"contact_candidates": 2, "applied_responses": 1, "inward_normal_velocity_removed": 1}},
                {"grid_nodes": {"contact_candidates": 0, "applied_responses": 0, "inward_normal_velocity_removed": 0}, "particles": {"contact_candidates": 0, "applied_responses": 0, "inward_normal_velocity_removed": 0}},
            ]}]}
        indexed = validate_contact_debug(debug, replay)
        rows = selected_contact_rows(replay, indexed, 6)
        self.assertEqual(rows[0]["tools"][0]["grid_nodes"]["contact_candidates"], 8)
        self.assertEqual(rows[0]["tools"][0]["grid_nodes"]["applied_responses"], 0)
        self.assertTrue(rows[0]["snapshot_without_substep_evaluations"])

    def test_report_outputs_are_written_and_nonempty_destination_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); replay_dir = root / "replay"; replay_dir.mkdir()
            replay_dir.joinpath("camera_parameters.json").write_text(json.dumps(self.replay_metadata()))
            manifest_tools = []
            for name, prefix in (("UR5e_spathla", "ur"), ("gen3_spathla", "kinova")):
                source, collision = root / f"{prefix}.stl", root / f"{prefix}_solid.stl"
                write_binary_stl(source); write_binary_stl(collision, 42.0)
                manifest_tools.append({"name": name, "source_mesh": source.name, "collision_mesh": collision.name, "source_sha256": digest(source), "collision_sha256": digest(collision), "coordinate_frame": "raw_stl_visual", "build": {}, "topology": {}})
            manifest = root / "collision.json"; manifest.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "tools": manifest_tools}))
            output = root / "report"
            args = parse_args(["--replay-dir", str(replay_dir), "--collision-manifest", str(manifest), "--output-dir", str(output), "--sdf-resolution", "16"])
            create_report(args)
            self.assertTrue((output / "index.html").is_file())
            self.assertTrue((output / "sdf_collision_contact_sheet.png").is_file())
            self.assertTrue((output / "visualization_manifest.json").is_file())
            text = (output / "index.html").read_text()
            self.assertIn("data:image/png;base64", text)
            self.assertIn("No replay SDF contact-debug", text)
            with self.assertRaisesRegex(ValueError, "must be empty"):
                create_report(args)


if __name__ == "__main__":
    unittest.main()

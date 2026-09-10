import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import taichi_viscoelastic_mpm_scene as mpm


class MpmMassTests(unittest.TestCase):
    def test_tool_geometry_parses_two_rigid_box_colliders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tools.json"
            document = {
                "schema": "taichidough/tool-geometry/v1",
                "tools": [
                    {
                        "name": "left",
                        "half_extents_m": [0.04, 0.01, 0.08],
                        "marker_from_collider": [
                            [1, 0, 0, 0.01],
                            [0, 1, 0, 0.02],
                            [0, 0, 1, 0.03],
                            [0, 0, 0, 1],
                        ],
                        "marker_from_mesh": [
                            [0, -1, 0, 0.04],
                            [1, 0, 0, -0.01],
                            [0, 0, 1, 0.02],
                            [0, 0, 0, 1],
                        ],
                    },
                    {
                        "name": "right",
                        "half_extents_m": [0.05, 0.02, 0.09],
                        "marker_from_collider": [
                            [0, -1, 0, -0.01],
                            [1, 0, 0, 0.0],
                            [0, 0, 1, 0.04],
                            [0, 0, 0, 1],
                        ],
                    },
                ],
            }
            path.write_text(json.dumps(document), encoding="utf-8")
            parsed = mpm.resolve_tool_geometry(path)
            self.assertEqual(parsed["schema"], "taichidough/tool-geometry/v1")
            self.assertEqual(parsed["names"], ["left", "right"])
            self.assertEqual(parsed["half_extents_m"][1], [0.05, 0.02, 0.09])
            self.assertEqual(parsed["marker_from_collider"][0][0][3], 0.01)
            self.assertEqual(parsed["marker_from_mesh"][0][0][3], 0.04)
            self.assertEqual(parsed["marker_from_mesh"][1], np.eye(4).tolist())
            self.assertEqual(parsed["source"], str(path.resolve()))
            self.assertFalse(parsed["proxy"])
            self.assertEqual(len(parsed["fingerprint"]), 64)
            aligned = mpm.align_tool_geometry(parsed, ["right", "left"])
            self.assertEqual(aligned["names"], ["right", "left"])
            self.assertEqual(aligned["half_extents_m"][0], [0.05, 0.02, 0.09])
            self.assertEqual(aligned["marker_from_collider"][1][0][3], 0.01)
            self.assertEqual(aligned["marker_from_mesh"][1][0][3], 0.04)

    def test_tool_geometry_rejects_invalid_schema_count_extents_and_transform(self):
        valid_tool = {
            "name": "left",
            "half_extents_m": [0.04, 0.01, 0.08],
            "marker_from_collider": np.eye(4).tolist(),
        }
        documents = [
            {"schema": "wrong", "tools": [valid_tool, {**valid_tool, "name": "right"}]},
            {"schema": "taichidough/tool-geometry/v1", "tools": [valid_tool]},
            {
                "schema": "taichidough/tool-geometry/v1",
                "tools": [valid_tool, {**valid_tool, "name": "right", "half_extents_m": [0.1, -0.1, 0.1]}],
            },
            {
                "schema": "taichidough/tool-geometry/v1",
                "tools": [
                    valid_tool,
                    {
                        **valid_tool,
                        "name": "right",
                        "marker_from_collider": [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    },
                ],
            },
            {
                "schema": "taichidough/tool-geometry/v1",
                "tools": [
                    valid_tool,
                    {
                        **valid_tool,
                        "name": "right",
                        "marker_from_mesh": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    },
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tools.json"
            for document in documents:
                with self.subTest(document=document), self.assertRaises(ValueError):
                    path.write_text(json.dumps(document), encoding="utf-8")
                    mpm.load_tool_geometry(path)

    def test_legacy_tool_geometry_and_argument_ambiguity(self):
        legacy = mpm.resolve_tool_geometry(None, [0.05, 0.06, 0.07], [0.01, 0.02, 0.03])
        self.assertEqual(legacy["schema"], "taichidough/tool-geometry/v1")
        self.assertEqual(legacy["source"], "legacy_cli")
        self.assertTrue(legacy["proxy"])
        self.assertEqual(len(legacy["fingerprint"]), 64)
        self.assertEqual(legacy["half_extents_m"], [[0.05, 0.06, 0.07]] * 2)
        self.assertEqual(legacy["marker_from_collider"][0][0][3], 0.01)
        self.assertEqual(legacy["marker_from_mesh"], [np.eye(4).tolist(), np.eye(4).tolist()])
        aligned = mpm.align_tool_geometry(legacy, ["gripper_a", "gripper_b"])
        self.assertEqual(aligned["names"], ["gripper_a", "gripper_b"])
        self.assertNotEqual(aligned["fingerprint"], legacy["fingerprint"])
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            mpm.resolve_tool_geometry("tools.json", [0.05, 0.05, 0.05], None)

    def test_replay_marker_transforms_follow_collision_mode(self):
        geometry = {
            "marker_from_collider": np.broadcast_to(np.eye(4), (2, 4, 4)).copy().tolist(),
            "marker_from_mesh": np.broadcast_to(np.eye(4), (2, 4, 4)).copy().tolist(),
        }
        geometry["marker_from_collider"][0][0][3] = .01
        geometry["marker_from_mesh"][0][0][3] = .04
        self.assertAlmostEqual(mpm.replay_marker_transforms(geometry, "box")[0, 0, 3], .01)
        self.assertAlmostEqual(mpm.replay_marker_transforms(geometry, "none")[0, 0, 3], .01)
        self.assertAlmostEqual(mpm.replay_marker_transforms(geometry, "sdf")[0, 0, 3], .04)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            mpm.replay_marker_transforms(geometry, "capsule")

    def test_simulator_tool_extents_support_udp_sender_arguments(self):
        defaults = mpm.resolve_sim_tool_half_extents(SimpleNamespace())
        self.assertEqual(defaults, ((0.05, 0.05, 0.05),) * 2)
        per_tool = mpm.resolve_sim_tool_half_extents(
            SimpleNamespace(
                tool_half_extents_by_tool=[
                    [0.07, 0.018, 0.07],
                    [0.03, 0.03, 0.03],
                ]
            )
        )
        self.assertEqual(per_tool[0], (0.07, 0.018, 0.07))
        self.assertEqual(per_tool[1], (0.03, 0.03, 0.03))
        with self.assertRaises(ValueError):
            mpm.resolve_sim_tool_half_extents(
                SimpleNamespace(tool_half_extents_by_tool=[[0.1, 0.1, 0.1]])
            )

    def test_mpm_grid_stencil_safety_bounds(self):
        grid = 16
        self.assertTrue(mpm.mpm_grid_stencil_is_safe([0.5 / grid, 0.5 / grid, 0.5 / grid], grid))
        self.assertTrue(mpm.mpm_grid_stencil_is_safe([(0.5 - 1e-6) / grid] * 3, grid))
        self.assertTrue(mpm.mpm_grid_stencil_is_safe([(grid - 1.5001) / grid] * 3, grid))
        self.assertFalse(mpm.mpm_grid_stencil_is_safe([(-0.6) / grid] * 3, grid))
        self.assertFalse(mpm.mpm_grid_stencil_is_safe([(grid - 1.5) / grid] * 3, grid))
        self.assertFalse(mpm.mpm_grid_stencil_is_safe([float("nan"), 0.5, 0.5], grid))

    def test_unsafe_particle_exits_with_diagnostic_instead_of_signal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            particle_path = root / "unsafe.npy"
            output_dir = root / "output"
            np.save(particle_path, np.array([[0.999, 0.5, 0.5]], dtype=np.float32))
            command = [
                sys.executable,
                str(Path(mpm.__file__).resolve()),
                "--cpu",
                "--particles", "1",
                "--grid", "16",
                "--steps", "1",
                "--dt", "0.001",
                "--substeps-per-frame", "1",
                "--initial-particles", str(particle_path),
                "--initial-particles-raw-scene-coordinates",
                "--initial-particles-axis-map", "xyz",
                "--initial-particles-fit", "none",
                "--tool-collision", "none",
                "--no-publish-dough-center",
                "--output-dir", str(output_dir),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
            self.assertNotEqual(completed.returncode, 0)
            self.assertGreaterEqual(completed.returncode, 0)
            diagnostic_path = output_dir / "invalid_state.json"
            self.assertTrue(diagnostic_path.is_file(), completed.stderr)
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["schema"], "taichidough/mpm-invalid-state/v1")
            self.assertEqual(diagnostic["failure_kind"], "pre_p2g_stencil_out_of_bounds")
            self.assertEqual(diagnostic["particle_index"], 0)
            self.assertEqual(diagnostic["substep"], 0)
            self.assertFalse((output_dir / "camera_parameters.json").exists())

    def test_floor_contact_keeps_soft_particles_at_metric_zero(self):
        simulator = Path(mpm.__file__).resolve()
        initial_particles = np.array([[0.5, 0.001, 0.5]], dtype=np.float32)
        for youngs_modulus in (1000.0, 2000.0):
            with self.subTest(youngs_modulus=youngs_modulus), tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / "output"
                particle_path = Path(temp_dir) / "initial_particles.npy"
                np.save(particle_path, initial_particles)
                process = subprocess.run(
                    [
                        sys.executable,
                        str(simulator),
                        "--cpu",
                        "--particles", "1",
                        "--grid", "16",
                        "--steps", "100",
                        "--substeps-per-frame", "1",
                        "--save-every", "100",
                        "--dt", "0.0002",
                        "--youngs-modulus", str(youngs_modulus),
                        "--viscosity", "0",
                        "--floor-y", "0",
                        "--floor-friction", "0.7",
                        "--floor-absorption", "0",
                        "--floor-stickiness", "0",
                        "--floor-plastic-damping-band", "0",
                        "--velocity-damping", "1",
                        "--plastic-velocity-damping", "1",
                        "--plastic-affine-damping", "1",
                        "--tool-collision", "none",
                        "--initial-particles", str(particle_path),
                        "--initial-particles-raw-scene-coordinates",
                        "--initial-particles-axis-map", "xyz",
                        "--initial-particles-fit", "none",
                        "--no-publish-dough-center",
                        "--output-dir", str(output_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                final_particles = np.load(output_dir / "particles_000000.npy")
                self.assertGreaterEqual(float(final_particles[:, 1].min()), 0.0)

    def test_reconstructed_volume_and_mass_survive_resize_and_grid_change(self):
        first = mpm.compute_mass_properties(100, 32, 1100.0, object_volume_m3=0.002)
        resized = mpm.compute_mass_properties(250, 96, 1100.0, object_volume_m3=0.002)
        self.assertEqual(first["mass_source"], "reconstructed_volume_and_density")
        self.assertAlmostEqual(first["object_volume_m3"], resized["object_volume_m3"])
        self.assertAlmostEqual(first["total_mass_kg"], resized["total_mass_kg"])
        self.assertAlmostEqual(first["particle_volume_m3"] * 100, 0.002)
        self.assertAlmostEqual(resized["particle_mass_kg"] * 250, 2.2)

    def test_measured_mass_derives_density(self):
        properties = mpm.compute_mass_properties(
            200,
            48,
            900.0,
            object_volume_m3=0.0015,
            object_mass_kg=1.8,
        )
        self.assertEqual(properties["mass_source"], "reconstructed_volume_and_measured_mass")
        self.assertAlmostEqual(properties["density_kg_m3"], 1200.0)
        self.assertAlmostEqual(properties["total_mass_kg"], 1.8)

    def test_procedural_mode_retains_grid_derived_particle_volume(self):
        properties = mpm.compute_mass_properties(10, 50, 1000.0)
        expected_particle_volume = (0.5 / 50.0) ** 3
        self.assertAlmostEqual(properties["particle_volume_m3"], expected_particle_volume)
        self.assertAlmostEqual(properties["total_mass_kg"], expected_particle_volume * 10 * 1000.0)

    def test_invalid_physical_values_are_rejected(self):
        for kwargs in (
            {"particle_count": 0, "grid_size": 32, "density": 1000.0},
            {"particle_count": 10, "grid_size": 0, "density": 1000.0},
            {"particle_count": 10, "grid_size": 32, "density": float("nan")},
            {"particle_count": 10, "grid_size": 32, "density": 1000.0, "object_volume_m3": -1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mpm.compute_mass_properties(**kwargs)
        with self.assertRaisesRegex(ValueError, "requires reconstructed"):
            mpm.compute_mass_properties(10, 32, 1000.0, object_mass_kg=1.0)

    def test_floor_height_accepts_horizontal_and_rejects_tilted_plane(self):
        self.assertAlmostEqual(mpm.floor_y_from_plane([0.0, 2.0, 0.0, -0.4]), 0.2)
        self.assertAlmostEqual(mpm.floor_y_from_plane([0.0, -1.0, 0.0, 0.2]), 0.2)
        with self.assertRaisesRegex(ValueError, "tilted"):
            mpm.floor_y_from_plane([0.01, 1.0, 0.0, -0.2])

    def test_reconstructed_particles_reject_additional_transforms(self):
        mpm.validate_reconstructed_particle_options("xyz", "none", 1.0, (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "axis-map"):
            mpm.validate_reconstructed_particle_options("xzy", "none", 1.0, (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "fitting"):
            mpm.validate_reconstructed_particle_options("xyz", "anisotropic", 1.0, (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "additional scale"):
            mpm.validate_reconstructed_particle_options("xyz", "none", 2.0, (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "additional offset"):
            mpm.validate_reconstructed_particle_options("xyz", "none", 1.0, (0.1, 0.0, 0.0))

    def test_v2_depth_export_preserves_exact_camera_metadata(self):
        camera = {
            "width": 8,
            "height": 6,
            "fx": 10.0,
            "fy": 11.0,
            "cx": 3.5,
            "cy": 2.5,
            "zNear": 0.05,
            "zFar": 2.0,
            "scene_from_camera": np.eye(4).tolist(),
        }
        points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            rendered = mpm.render_particle_depth(
                points,
                camera["width"],
                camera["height"],
                "metric",
                camera,
                Path(temp_dir),
                0,
            )

            for field in (
                "width",
                "height",
                "fx",
                "fy",
                "cx",
                "cy",
                "zNear",
                "zFar",
                "scene_from_camera",
            ):
                self.assertEqual(rendered[field], camera[field])
            self.assertNotIn("position", rendered)
            self.assertNotIn("lookAt", rendered)
            self.assertNotIn("fieldOfView", rendered)
            self.assertTrue(Path(rendered["depth_array"]).is_file())
            self.assertTrue(Path(rendered["depth_image"]).is_file())

    def test_metadata_checks_path_hash_calibration_and_floor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            particles_path = directory / "sampled_particles_xyz.npy"
            particles = np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], dtype=np.float32)
            np.save(particles_path, particles)
            metadata_path = directory / "reconstruction_metadata.json"
            metadata = {
                "schema": "voxel_dough_reconstruction/v2",
                "object_volume_m3": 0.001,
                "voxel_size": 0.01,
                "voxel_count": 1000,
                "sampled_particles": 2,
                "sampled_particles_sha256": mpm.particle_array_sha256(particles),
                "calibration_schema": "taichidough/scene-calibration/v2",
                "calibration_fingerprint": "calibration-a",
                "calibration": {
                    "schema": "taichidough/scene-calibration/v2",
                    "is_metric": True,
                    "scene_frame": "taichi_scene",
                    "fingerprint": "calibration-a",
                },
                "array_frames": {"sampled_particles_xyz": "taichi_scene"},
                "fill": {"mode": "floor", "floor_plane_scene": [0.0, 1.0, 0.0, -0.2]},
                "outputs": {"sampled_particles_xyz": str(particles_path.resolve())},
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            calibration = SimpleNamespace(fingerprint="calibration-a")
            result = mpm.load_reconstruction_metadata(metadata_path, particles_path, particles, calibration)
            self.assertAlmostEqual(result["floor_y"], 0.2)
            self.assertEqual(result["particle_sha256"], metadata["sampled_particles_sha256"])
            self.assertEqual(len(result["metadata_sha256"]), 64)

            with self.assertRaisesRegex(ValueError, "particle path"):
                mpm.load_reconstruction_metadata(metadata_path, directory / "other.npy", particles, calibration)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                mpm.load_reconstruction_metadata(
                    metadata_path, particles_path, particles + 0.01, calibration
                )
            with self.assertRaisesRegex(ValueError, "calibration fingerprint"):
                mpm.load_reconstruction_metadata(
                    metadata_path,
                    particles_path,
                    particles,
                    SimpleNamespace(fingerprint="calibration-b"),
                )


if __name__ == "__main__":
    unittest.main()

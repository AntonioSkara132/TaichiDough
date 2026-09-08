"""Check cloud/mesh registration without starting ROS or a Taichi window."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dough_scene", PACKAGE / "scripts/taichi_viscoelastic_mpm_scene.py"
)
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)


def urdf_matrix(xyz, rpy):
    """Independent homogeneous URDF transform: translation * Rz * Ry * Rx."""
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = xyz
    return transform


class ToolCameraMappingTest(unittest.TestCase):
    def test_top_view_landmarks(self):
        rotation, origin = scene.resolve_tool_mapping(
            scene.CAMERA_VIEWS["top_dough"], ["auto"] * 9, ["auto"] * 3, [1, 1, 1]
        )
        np.testing.assert_allclose(origin, [0.5, 0.75, 0.5])
        # ROS optical camera: right, down, forward, and the dough centre.
        camera_points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0.45]])
        expected = [[0.5, 0.75, 0.5], [0.4, 0.75, 0.5], [0.5, 0.75, 0.4], [0.5, 0.65, 0.5], scene.SCENE_CENTER]
        np.testing.assert_allclose(camera_points @ rotation.T + origin, expected, atol=1e-7)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_depth_backprojection_returns_same_scene_surface(self):
        # Check actual depth projection/backprojection as used by the publisher,
        # with a sphere on the optical axis so its surface is known analytically.
        for name, config in scene.CAMERA_VIEWS.items():
            with self.subTest(view=name):
                rotation, origin = scene.optical_to_scene_transform(config)
                centre = origin + rotation[:, 2] * 0.4
                depth = scene.compute_particle_depth(centre[None, :], 80, 60, config)
                cloud = scene.depth_to_xyz(depth, config["fieldOfView"], config["zFar"])
                self.assertGreater(len(cloud), 0)
                in_scene = cloud @ rotation.T + origin
                radii = np.linalg.norm(in_scene - centre, axis=1)
                # The splatter approximates rays using the sphere centre depth.
                np.testing.assert_allclose(radii, scene.PARTICLE_RENDER_RADIUS, atol=0.0002)
                np.testing.assert_allclose((in_scene - origin) @ rotation, cloud, atol=1e-7)

    def test_mesh_vertices_match_rviz_urdf_in_optical_frame(self):
        urdf = ET.parse(PACKAGE / "test/fixtures/rf_lab_tool_visuals.urdf").getroot()
        mesh_dir = PACKAGE / "meshes"
        # Pose includes a nontrivial rotation so rotating translations in the
        # wrong axes or applying the fixed tool joint twice is observable.
        axis = np.array([1.0, 2.0, -3.0]) / np.sqrt(14.0)
        quaternion = np.r_[axis * np.sin(0.7), np.cos(0.7)]
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        tool_matrix = np.eye(4)
        tool_matrix[:3, :3] = np.eye(3) + np.sin(1.4) * cross + (1 - np.cos(1.4)) * cross @ cross
        tool_matrix[:3, 3] = [0.08, -0.06, 0.42]
        pose = np.r_[tool_matrix[:3, 3], quaternion][None, :]
        for link, xyz, rpy in (
            ("ur5e_spathla_frame", scene.UR_TOOL_VISUAL_ORIGIN, scene.UR_TOOL_VISUAL_RPY),
            ("kinova_spathla_frame", scene.KINOVA_TOOL_VISUAL_ORIGIN, scene.KINOVA_TOOL_VISUAL_RPY),
        ):
            visual = urdf.find(f".//link[@name='{link}']/visual")
            mesh = visual.find("geometry/mesh")
            origin = visual.find("origin")
            urdf_xyz = np.fromstring(origin.get("xyz"), sep=" ")
            urdf_rpy = np.fromstring(origin.get("rpy"), sep=" ")
            np.testing.assert_allclose(xyz, urdf_xyz)
            np.testing.assert_allclose(rpy, urdf_rpy)
            mesh_scale = np.fromstring(mesh.get("scale"), sep=" ")
            np.testing.assert_allclose(mesh_scale, [0.001] * 3)
            vertices, _ = scene.load_binary_stl(mesh_dir / Path(mesh.get("filename")).name, mesh_scale[0])
            homogeneous = np.c_[vertices, np.ones(len(vertices))]
            expected_optical = (homogeneous @ (tool_matrix @ urdf_matrix(urdf_xyz, urdf_rpy)).T)[:, :3]
            for view, config in scene.CAMERA_VIEWS.items():
                for scale in (1.0, 1.5):
                    with self.subTest(link=link, view=view, scale=scale):
                        rotation, offset = scene.optical_to_scene_transform(config)
                        mapped = scene.map_tool_poses(pose, rotation, offset, np.full(3, scale))[0]
                        actual = scene.transformed_tool_mesh(
                            vertices * scale, mapped, xyz * scale, scene.rpy_to_matrix(rpy)
                        )
                        recovered_optical = (actual - offset) @ rotation / scale
                        np.testing.assert_allclose(recovered_optical, expected_optical, atol=3e-7)

    def test_reflections_and_nonrigid_scale_are_rejected(self):
        config = scene.CAMERA_VIEWS["top_dough"]
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            scene.resolve_tool_mapping(config, [-1, 0, 0, 0, 0, -1, 0, 1, 0], [0.5, 0.95, 0.5], [1, 1, 1])
        with self.assertRaisesRegex(ValueError, "uniform"):
            scene.resolve_tool_mapping(config, ["auto"] * 9, ["auto"] * 3, [1, 2, 1])

    def test_explicit_calibration_overrides(self):
        rotation, offset = scene.resolve_tool_mapping(
            scene.CAMERA_VIEWS["top_dough"], np.eye(3).ravel(), ["1", "2", "3"], [1, 1, 1]
        )
        np.testing.assert_allclose(rotation, np.eye(3))
        np.testing.assert_allclose(offset, [1, 2, 3])

    def test_subscriber_uses_cloud_frame_and_rejects_other_frames(self):
        # Exercise the callback with real message types without opening DDS.
        try:
            from dual_description.msg import PoseStampedArray
            from geometry_msgs.msg import PoseStamped
        except ImportError:
            self.skipTest("Source the ROS workspace to test the optional ROS subscriber")

        subscriber = scene.RosToolPoseSubscriber.__new__(scene.RosToolPoseSubscriber)
        subscriber.rotation, subscriber.offset = scene.optical_to_scene_transform(scene.CAMERA_VIEWS["top_dough"])
        subscriber.scale = np.ones(3)
        subscriber.expected_frame = "camera_depth_optical_frame"
        subscriber._last_positions = subscriber._last_time = None
        subscriber.velocities = np.zeros((2, 3))
        subscriber.node = SimpleNamespace(get_logger=lambda: Mock())
        message = PoseStampedArray()
        message.header.frame_id = subscriber.expected_frame
        message.header.stamp.sec = 1
        for _ in range(2):
            stamped = PoseStamped()
            stamped.header.frame_id = subscriber.expected_frame
            stamped.pose.position.z = 0.45
            stamped.pose.orientation.w = 1.0
            message.poses.append(stamped)
        subscriber._callback(message)
        np.testing.assert_allclose(subscriber.poses[:, :3], [scene.SCENE_CENTER] * 2, atol=1e-7)
        previous = subscriber.poses.copy()
        message.header.frame_id = "world"
        message.poses[0].pose.position.z = 0.9
        subscriber._callback(message)
        np.testing.assert_array_equal(subscriber.poses, previous)


if __name__ == "__main__":
    unittest.main()

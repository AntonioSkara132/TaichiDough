"""Keep the standalone scene usable without the optional ROS packages."""

import builtins
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


PACKAGE = Path(__file__).resolve().parents[1]


def import_scene():
    spec = importlib.util.spec_from_file_location(
        "standalone_scene", PACKAGE / "scripts/taichi_viscoelastic_mpm_scene.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandaloneTest(unittest.TestCase):
    def test_import_and_bundled_mesh_lookup_without_ros(self):
        original_import = builtins.__import__

        def without_ros(name, *args, **kwargs):
            if name.split(".")[0] in {"ament_index_python", "rclpy", "dual_description", "sensor_msgs"}:
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_ros):
            scene = import_scene()
            for filename in ("ur_spathla.stl", "gen3_spathla.stl"):
                path = scene.find_tool_mesh(filename)
                self.assertEqual(path, PACKAGE / "meshes" / filename)
                vertices, indices = scene.load_binary_stl(path, 0.001)
                self.assertGreater(len(vertices), 0)
                self.assertEqual(len(vertices), len(indices))

    def test_sourced_package_and_explicit_mesh_selection(self):
        scene = import_scene()
        fake_package = types.ModuleType("ament_index_python.packages")
        fake_package.PackageNotFoundError = type("PackageNotFoundError", (LookupError,), {})
        with tempfile.TemporaryDirectory() as folder:
            share = Path(folder)
            (share / "meshes").mkdir()
            installed = share / "meshes/ur_spathla.stl"
            installed.touch()
            fake_package.get_package_share_directory = lambda name: str(share)
            with patch.dict(sys.modules, {"ament_index_python.packages": fake_package}):
                self.assertEqual(scene.find_tool_mesh("ur_spathla.stl"), installed)
                explicit = PACKAGE / "meshes/ur_spathla.stl"
                self.assertEqual(scene.find_tool_mesh("ur_spathla.stl", explicit), explicit)
                # A package without this particular mesh falls back to bundled.
                self.assertEqual(scene.find_tool_mesh("gen3_spathla.stl"), PACKAGE / "meshes/gen3_spathla.stl")

                def missing_package(name):
                    raise fake_package.PackageNotFoundError(name)

                fake_package.get_package_share_directory = missing_package
                self.assertEqual(scene.find_tool_mesh("ur_spathla.stl"), explicit)

    def test_missing_override_does_not_silently_select_another_mesh(self):
        scene = import_scene()
        with self.assertRaises(FileNotFoundError):
            scene.find_tool_mesh("ur_spathla.stl", PACKAGE / "meshes/does_not_exist.stl")


if __name__ == "__main__":
    unittest.main()

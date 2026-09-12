"""Preservation and import-binding checks that do not initialize Taichi."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from experiments.differentiable_mpm import reference_adapter as adapter
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, SimulationConfig
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class PreservationTests(unittest.TestCase):
    def setUp(self):
        # Strict assertions remain strict even when check.py wraps this suite in frozen.
        context = adapter.reference_policy("strict")
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def test_live_source_status_and_snapshot_hashes_are_reported(self):
        # A later live edit must be reported, not mistaken for a preserved original.
        with adapter.reference_policy("frozen"):
            result = adapter.verify_reference()
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["files"]), 9)
        self.assertEqual(sum("snapshot" in row for row in result["files"]), 9)
        self.assertEqual(sum(row["snapshot_origin"] == "supplement" for row in result["files"]), 6)
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(
            (adapter.EXPERIMENT_ROOT / "reference_manifest.json").read_bytes()).hexdigest())
        self.assertEqual(result["supplemental_snapshots_sha256"], hashlib.sha256(
            (adapter.EXPERIMENT_ROOT / "reference_snapshots.json").read_bytes()).hexdigest())
        changed = []
        for row in result["files"]:
            source = adapter.REPOSITORY_ROOT / row["source"]
            actual = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
            self.assertEqual(row["actual_sha256"], actual)
            self.assertEqual(row["original_unchanged"], actual == row["expected_sha256"])
            if not row["original_unchanged"]:
                changed.append(row["source"])
            if "snapshot" in row:
                snapshot_hash = hashlib.sha256((adapter.EXPERIMENT_ROOT / row["snapshot"]).read_bytes()).hexdigest()
                self.assertEqual(snapshot_hash, row["expected_sha256"])
                self.assertEqual(snapshot_hash, row["snapshot_sha256"])
        self.assertEqual(result["originals_unchanged"], not changed)
        self.assertEqual([row["source"] for row in result["changed_originals"]], changed)
        if changed:
            with self.assertRaisesRegex(ValueError, "Working source"):
                adapter.verify_reference()
        else:
            self.assertTrue(adapter.verify_reference()["originals_unchanged"])

    def test_frozen_imports_restore_unrelated_aliases(self):
        unrelated_topview = ModuleType("deformpath_topview")
        unrelated_dynamics = ModuleType("deformpath_dynamics")
        aliases = {"deformpath_topview": unrelated_topview, "deformpath_dynamics": unrelated_dynamics}
        previous = {name: sys.modules.get(name) for name in aliases}
        sys.modules.update(aliases)
        try:
            adapter._MODULES.clear()
            with adapter.reference_policy("frozen"):
                modules = adapter.get_reference_modules()
            self.assertIs(sys.modules["deformpath_topview"], unrelated_topview)
            self.assertIs(sys.modules["deformpath_dynamics"], unrelated_dynamics)
            self.assertIs(modules.simulator.apply_calibration, modules.topview.apply_calibration)
            self.assertIs(modules.dynamics.apply_calibration, modules.topview.apply_calibration)
            self.assertEqual(modules.simulator.PROJECT_ROOT, adapter.REPOSITORY_ROOT)
            self.assertTrue(modules.simulator.OUTPUT_DIR.is_relative_to(adapter.EXPERIMENT_ROOT))
            for module in (modules.simulator, modules.dynamics, modules.topview):
                self.assertEqual(Path(module.__file__).parent, adapter.EXPERIMENT_ROOT / "reference")
            with adapter.reference_policy("frozen"):
                self.assertIs(adapter.load_reference(), modules.simulator)
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_physics_selects_distinct_verified_simulators_and_preserved_helpers(self):
        with adapter.reference_policy("frozen"):
            old = adapter.reference_identity("legacy-v1")
            new = adapter.reference_identity("corrected-v1")
            self.assertEqual(old["simulator_sha256"],
                             "6653543ac16c8fcbdc111c73ebaa2c5e2d8c1cdc899e3750dce539b2730a2f07")
            self.assertEqual(new["simulator_sha256"],
                             "d33f0aec3952fa72282cd181757f678b2e2a48e5b51016c23d40fd05d6a3ac3e")
            legacy = adapter.get_reference_modules(physics_version="legacy-v1")
            corrected = adapter.get_reference_modules(physics_version="corrected-v1")
            self.assertIsNot(legacy.simulator, corrected.simulator)
            self.assertIs(adapter.load_reference(), legacy.simulator)
            self.assertIs(corrected.simulator.apply_calibration, corrected.topview.apply_calibration)
            self.assertEqual(Path(corrected.simulator.__file__), adapter.EXPERIMENT_ROOT / new["simulator_snapshot"])
            for name, path in legacy.source_paths.items():
                if name != "scripts/taichi_viscoelastic_mpm_scene.py":
                    self.assertEqual(corrected.source_paths[name], path)
            with self.assertRaisesRegex(ValueError, "Unknown reference physics"):
                adapter.reference_identity("unknown")

    def test_corrected_reference_rejects_snapshot_and_manifest_changes(self):
        with temporary_directory() as temporary:
            root, experiment, _ = self._copy_recorded_files(temporary)
            manifest_path = adapter.EXPERIMENT_ROOT / "reference_corrected_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            snapshot = experiment / manifest["snapshot"]
            snapshot.parent.mkdir()
            original = (adapter.EXPERIMENT_ROOT / manifest["snapshot"]).read_bytes()
            snapshot.write_bytes(original)
            target_manifest = experiment / manifest_path.name
            shutil.copy2(manifest_path, target_manifest)
            adapter.reference_identity("corrected-v1", root, experiment)
            snapshot.write_bytes(original + b"\n# changed\n")
            with self.assertRaisesRegex(ValueError, "Corrected snapshot"):
                adapter.reference_identity("corrected-v1", root, experiment)
            manifest["sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            target_manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Corrected reference manifest"):
                adapter.reference_identity("corrected-v1", root, experiment)
            snapshot.write_bytes(original)
            shutil.copy2(manifest_path, target_manifest)
            (root / "scripts/taichi_viscoelastic_mpm_scene.py").write_text("# live drift\n")
            with self.assertRaisesRegex(ValueError, "Working source"):
                adapter.reference_identity("corrected-v1", root, experiment)
            with adapter.reference_policy("frozen"):
                selected = adapter.reference_identity("corrected-v1", root, experiment)
                self.assertEqual(selected["simulator_sha256"], hashlib.sha256(original).hexdigest())

    def _copy_recorded_files(self, temporary, include_supplement=True):
        root = Path(temporary) / "repo"
        experiment = root / "experiments" / "differentiable_mpm"
        experiment.mkdir(parents=True)
        manifest_path = adapter.EXPERIMENT_ROOT / "reference_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        supplement_path = adapter.EXPERIMENT_ROOT / "reference_snapshots.json"
        supplement = json.loads(supplement_path.read_text())
        for row in manifest["files"]:
            snapshot = row.get("snapshot", supplement["snapshots"].get(row["source"]))
            baseline = adapter.EXPERIMENT_ROOT / snapshot
            self.assertEqual(hashlib.sha256(baseline.read_bytes()).hexdigest(), row["sha256"])
            target = root / row["source"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline, target)
            if "snapshot" in row or include_supplement:
                target = experiment / snapshot
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(baseline, target)
        shutil.copy2(manifest_path, experiment / manifest_path.name)
        if include_supplement:
            shutil.copy2(supplement_path, experiment / supplement_path.name)
        return root, experiment, manifest

    def _rewrite_supplement(self, experiment, transform):
        path = experiment / "reference_snapshots.json"
        supplement = json.loads(path.read_text())
        transform(supplement)
        path.write_text(json.dumps(supplement))

    def test_verification_uses_relocated_checkout_not_historical_root(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, _ = self._copy_recorded_files(temporary)
            self.assertEqual(adapter.verify_reference(root, experiment)["repository_root"], str(root))

    def test_verification_rejects_changed_working_source(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary)
            source = root / manifest["files"][0]["source"]
            source.write_bytes(source.read_bytes() + b"\n# test mutation\n")
            with self.assertRaisesRegex(ValueError, "Working source differs"):
                adapter.verify_reference(root, experiment)

    def test_verification_rejects_changed_snapshot(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary)
            snapshot = experiment / manifest["files"][0]["snapshot"]
            snapshot.write_bytes(snapshot.read_bytes() + b"\n# test mutation\n")
            with self.assertRaisesRegex(ValueError, "Frozen snapshot differs"):
                adapter.verify_reference(root, experiment)

    def test_verification_rejects_path_traversal(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary)
            manifest["files"][0]["source"] = "../outside.py"
            (experiment / "reference_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "relative file path"):
                adapter.verify_reference(root, experiment)

    def test_frozen_policy_reports_changed_snapshotted_source(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary)
            source = root / manifest["files"][0]["source"]
            source.write_bytes(source.read_bytes() + b"\n# later live edit\n")
            with self.assertRaisesRegex(ValueError, "Working source differs"):
                adapter.verify_reference(root, experiment)
            with adapter.reference_policy("frozen"):
                result = adapter.verify_reference(root, experiment)
            self.assertEqual(result["policy"], "frozen")
            self.assertTrue(result["valid"])
            self.assertFalse(result["originals_unchanged"])
            self.assertEqual(len(result["changed_originals"]), 1)
            changed = result["changed_originals"][0]
            self.assertEqual(changed["source"], manifest["files"][0]["source"])
            self.assertEqual(changed["expected_sha256"], manifest["files"][0]["sha256"])
            self.assertEqual(changed["snapshot_sha256"], changed["expected_sha256"])
            self.assertEqual(changed["actual_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertNotEqual(changed["actual_sha256"], changed["expected_sha256"])
            self.assertTrue(changed["original_exists"])

    def test_frozen_policy_reports_missing_snapshotted_source(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary)
            (root / manifest["files"][0]["source"]).unlink()
            with self.assertRaisesRegex(ValueError, "Working source is missing"):
                adapter.verify_reference(root, experiment)
            with adapter.reference_policy("frozen"):
                result = adapter.verify_reference(root, experiment)
            self.assertFalse(result["originals_unchanged"])
            self.assertIsNone(result["changed_originals"][0]["actual_sha256"])
            self.assertFalse(result["changed_originals"][0]["original_exists"])

    def test_frozen_policy_rejects_changed_unsnapshotted_helper(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary, include_supplement=False)
            record = next(row for row in manifest["files"] if "snapshot" not in row)
            source = root / record["source"]
            source.write_bytes(source.read_bytes() + b"\n# changed helper\n")
            with adapter.reference_policy("frozen"):
                with self.assertRaisesRegex(ValueError, "Working source differs"):
                    adapter.verify_reference(root, experiment)

    def test_frozen_policy_rejects_missing_unsnapshotted_helper(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary, include_supplement=False)
            record = next(row for row in manifest["files"] if "snapshot" not in row)
            (root / record["source"]).unlink()
            with adapter.reference_policy("frozen"):
                with self.assertRaisesRegex(ValueError, "Working source is missing"):
                    adapter.verify_reference(root, experiment)

    def test_frozen_policy_rejects_tampered_snapshot(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary)
            record = manifest["files"][0]
            snapshot = experiment / record["snapshot"]
            snapshot.write_bytes(snapshot.read_bytes() + b"\n# corrupted snapshot\n")
            (root / record["source"]).unlink()
            with adapter.reference_policy("frozen"):
                with self.assertRaisesRegex(ValueError, "Frozen snapshot differs"):
                    adapter.verify_reference(root, experiment)

    def test_policy_is_restored_after_nesting_and_exceptions(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, _ = self._copy_recorded_files(temporary)
            self.assertEqual(adapter.verify_reference(root, experiment)["policy"], "strict")
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with adapter.reference_policy("frozen"):
                    self.assertEqual(adapter.verify_reference(root, experiment)["policy"], "frozen")
                    with adapter.reference_policy("strict"):
                        self.assertEqual(adapter.verify_reference(root, experiment)["policy"], "strict")
                    self.assertEqual(adapter.verify_reference(root, experiment)["policy"], "frozen")
                    raise RuntimeError("intentional")
            self.assertEqual(adapter.verify_reference(root, experiment)["policy"], "strict")
            with self.assertRaisesRegex(ValueError, "Reference policy"):
                with adapter.reference_policy("ignore"):
                    self.fail("Invalid policy was accepted")
            self.assertEqual(adapter.verify_reference(root, experiment)["policy"], "strict")

    def test_cached_modules_still_recheck_active_policy(self):
        with temporary_directory(prefix="reference-preservation-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary)
            source = root / manifest["files"][0]["source"]
            source.write_bytes(source.read_bytes() + b"\n# later live edit\n")
            with adapter.reference_policy("frozen"):
                modules = adapter.get_reference_modules(root, experiment)
                self.assertIs(adapter.load_reference(root, experiment), modules.simulator)
            with self.assertRaisesRegex(ValueError, "Working source differs"):
                adapter.get_reference_modules(root, experiment)

    def test_supplemented_helper_drift_uses_original_verified_bytes(self):
        with temporary_directory(prefix="reference-supplement-") as temporary:
            root, experiment, manifest = self._copy_recorded_files(temporary)
            source_name = "scripts/material_calibration.py"
            source = root / source_name
            source.write_bytes(source.read_bytes() + b"\n# later helper edit\n")
            with self.assertRaisesRegex(ValueError, "Working source differs"):
                adapter.verify_reference(root, experiment)
            with adapter.reference_policy("frozen"):
                selected = adapter.verified_reference_path(source_name, root, experiment)
                modules = adapter.get_reference_modules(root, experiment)
            expected = next(row["sha256"] for row in manifest["files"] if row["source"] == source_name)
            self.assertEqual(hashlib.sha256(selected.read_bytes()).hexdigest(), expected)
            self.assertEqual(Path(modules.material_calibration.__file__), selected)
            self.assertIs(modules.calibrate_youngs_modulus.partial_view_loss,
                          modules.material_calibration.partial_view_loss)
            self.assertIs(modules.evaluate_dynamic_topview_match.ToolReplay, modules.dynamics.ToolReplay)

    def test_supplement_corruption_and_missing_bytes_are_always_rejected(self):
        for missing in (False, True):
            with self.subTest(missing=missing), temporary_directory(prefix="reference-supplement-") as temporary:
                root, experiment, _ = self._copy_recorded_files(temporary)
                snapshot = experiment / "reference/material_calibration.py"
                if missing:
                    snapshot.unlink()
                else:
                    snapshot.write_bytes(snapshot.read_bytes() + b"\n# corrupted copy\n")
                for policy in ("strict", "frozen"):
                    with adapter.reference_policy(policy), self.assertRaisesRegex(ValueError, "Frozen snapshot"):
                        adapter.verify_reference(root, experiment)

    def test_supplement_rejects_unknown_duplicate_and_unsafe_records(self):
        mutations = [
            lambda doc: doc["snapshots"].update({"scripts/unlisted.py": "reference/unlisted.py"}),
            lambda doc: doc["snapshots"].update({"scripts/deformpath_topview.py": "reference/other.py"}),
            lambda doc: doc["snapshots"].update({"scripts/material_calibration.py": "../outside.py"}),
            lambda doc: doc["snapshots"].update({"scripts/material_calibration.py": "runs/unverified.py"}),
            lambda doc: doc["snapshots"].update({"scripts/material_calibration.py": "reference/deformpath_topview.py"}),
            lambda doc: doc["snapshots"].update({"scripts/material_calibration.py": {"sha256": "0" * 64}}),
            lambda doc: doc.update({"expected_sha256": {"scripts/material_calibration.py": "0" * 64}}),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), temporary_directory(prefix="reference-supplement-") as temporary:
                root, experiment, _ = self._copy_recorded_files(temporary)
                self._rewrite_supplement(experiment, mutation)
                with adapter.reference_policy("frozen"), self.assertRaises(ValueError):
                    adapter.verify_reference(root, experiment)
        with temporary_directory(prefix="reference-supplement-") as temporary:
            root, experiment, _ = self._copy_recorded_files(temporary)
            (experiment / "reference_snapshots.json").write_text(
                '{"schema_version":1,"snapshots":{"scripts/material_calibration.py":"reference/a.py",'
                '"scripts/material_calibration.py":"reference/b.py"}}')
            with adapter.reference_policy("frozen"), self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                adapter.verify_reference(root, experiment)

    def test_evaluator_keeps_baseline_aliases_for_lazy_imports(self):
        with adapter.reference_policy("frozen"):
            modules = adapter.get_reference_modules()
            def check_lazy_imports():
                import material_calibration
                import taichi_viscoelastic_mpm_scene
                self.assertIs(material_calibration, modules.material_calibration)
                self.assertIs(taichi_viscoelastic_mpm_scene, modules.simulator)
                self.assertEqual(sys.argv[1:], ["--test-argument"])
                return 19
            previous_argv = sys.argv
            with mock.patch.object(modules.evaluate_dynamic_topview_match, "main", side_effect=check_lazy_imports):
                self.assertEqual(adapter.run_reference_evaluator(["--test-argument"]), 19)
            self.assertIs(sys.argv, previous_argv)

    def test_launched_evaluator_records_baseline_imports_without_simulating(self):
        with temporary_directory(prefix="reference-evaluator-") as temporary:
            report_path = Path(temporary) / "imports.json"
            completed = subprocess.run(
                [sys.executable, "-m", "experiments.differentiable_mpm.reference_adapter",
                 "--reference-policy", "frozen", "--import-report", str(report_path), "--", "--help"],
                cwd=adapter.REPOSITORY_ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["reference_policy"], "frozen")
            self.assertEqual(report["simulator_project_root"], str(adapter.REPOSITORY_ROOT))
            with adapter.reference_policy("frozen"):
                expected = {row["source"]: row["expected_sha256"] for row in adapter.verify_reference()["files"]}
            for name, record in report["modules"].items():
                self.assertEqual(Path(record["path"]).parent, adapter.EXPERIMENT_ROOT / "reference")
                self.assertEqual(record["sha256"], expected[f"scripts/{name}.py"])

    def test_parity_worker_commands_propagate_the_explicit_policy(self):
        from experiments.differentiable_mpm import real_parity
        from experiments.differentiable_mpm.tests import test_forward_parity
        for policy in ("strict", "frozen"):
            with adapter.reference_policy(policy):
                commands = [real_parity.worker_command(Path("run"), "short", "reference"),
                            test_forward_parity.worker_command("reference", "elastic", 1, Path("out.npz"))]
            for command in commands:
                self.assertEqual(command[command.index("--reference-policy") + 1], policy)
        def check_worker(*args):
            self.assertEqual(adapter.current_reference_policy(), "frozen")
        with mock.patch.object(real_parity, "run_worker", side_effect=check_worker) as worker:
            self.assertEqual(real_parity.main(["--reference-policy", "frozen", "--worker", "reference",
                                              "--stage", "short", "--run-dir", "test-run"]), 0)
            worker.assert_called_once()
        self.assertEqual(adapter.current_reference_policy(), "strict")

    def test_parity_worker_commands_propagate_physics_independently(self):
        from experiments.differentiable_mpm.tests import test_forward_parity
        for policy in ("strict", "frozen"):
            for version in ("corrected-v1", "legacy-v1"):
                with adapter.reference_policy(policy):
                    command = test_forward_parity.worker_command("reference", "elastic", 1, Path("out.npz"), version)
                self.assertEqual(command[command.index("--reference-policy") + 1], policy)
                self.assertEqual(command[command.index("--physics-version") + 1], version)
                self.assertEqual(test_forward_parity.make_fixture("elastic", 1, version)[0].physics_version, version)

    def test_real_parity_rejects_mixed_reference_identity_without_running(self):
        from experiments.differentiable_mpm import real_parity
        with temporary_directory() as temporary, adapter.reference_policy("frozen"):
            root = Path(temporary)
            (root / "inputs.npz").write_bytes(b"fixture")
            source = root / "candidate_source"
            source.mkdir()
            hashes = {}
            for name in real_parity.PINNED_FILES:
                (source / name).write_text("# synthetic source\n")
                hashes[name] = real_parity._hash_file(source / name)
            metadata = {"schema": "taichidough/real-forward-parity-inputs/v1",
                        "simulation": {"physics_version": "corrected-v1"},
                        "candidate_source_sha256": hashes,
                        "inputs_npz_sha256": real_parity._hash_file(root / "inputs.npz"),
                        "position_acceptance_limits": real_parity.POSITION_LIMITS,
                        "reference_identity": adapter.reference_identity("corrected-v1")}
            path = root / "inputs.json"
            path.write_text(json.dumps(metadata))
            self.assertEqual(real_parity._verified_metadata(root), metadata)
            metadata["reference_identity"] = adapter.reference_identity("legacy-v1")
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "reference identity"):
                real_parity._verified_metadata(root)
            metadata.pop("reference_identity")
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "reference identity"):
                real_parity._verified_metadata(root)
            # Historical runs remain legacy records, without assigning them a new identity.
            metadata["simulation"].pop("physics_version")
            path.write_text(json.dumps(metadata))
            self.assertNotIn("reference_identity", real_parity._verified_metadata(root))

    def test_arguments_preserve_physics_and_disable_scripted_tools(self):
        config = SimulationConfig(n_particles=8, plasticity="stretch-clamp", use_jp=True,
                                  jp_hardening=2.0, tool_collision="sdf",
                                  plastic_velocity_damping=0.91, plastic_affine_damping=0.87,
                                  tool_contact_absorption=0.13, tool_stickiness=0.21)
        args = adapter.reference_arguments(config, DEFAULT_PARAMETERS)
        self.assertFalse(args.pure_viscoelastic)
        self.assertTrue(args.replay_episode)
        self.assertFalse(args.ros_control)
        self.assertEqual(args.tool_contact_friction, DEFAULT_PARAMETERS["tool_retention"])
        self.assertEqual(args.floor_friction, DEFAULT_PARAMETERS["floor_retention"])
        self.assertEqual(args.mass_properties["particle_mass_kg"], config.particle_mass)
        self.assertEqual(args.mass_properties["particle_volume_m3"], config.particle_volume)
        self.assertEqual(args.tool_contact_absorption, config.tool_contact_absorption)
        self.assertEqual(args.tool_stickiness, config.tool_stickiness)
        self.assertTrue(args.use_jp)
        self.assertEqual(args.jp_hardening, 2.0)
        self.assertEqual(args.plastic_affine_damping, 0.87)

    def test_reference_rejects_claimed_float64_parity(self):
        with self.assertRaisesRegex(ValueError, "float32"):
            adapter.reference_arguments(SimulationConfig(n_particles=8, precision="f64"), DEFAULT_PARAMETERS)


if __name__ == "__main__":
    unittest.main()

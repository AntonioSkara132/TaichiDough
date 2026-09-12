"""Inventory discovery is cheap, deterministic, and never implies readiness."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.differentiable_mpm.dataset_inventory import (
    MAX_METADATA_BYTES, inventory_draft, scan_dataset,
)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def episode(self, variant="raw", session="snimanje_23_10", name="episode18_kugla", processed=True):
        path = self.base / variant / session / name
        path.mkdir(parents=True)
        for filename in ("pointclouds.pt", "paths.pt"):
            (path / filename).write_bytes(b"not a tensor")
        if processed:
            for filename in ("pointclouds_interpolated.pt", "paths_interpolated.pt"):
                (path / filename).write_bytes(b"not a tensor")
            (path / "sequence_metadata.json").write_text("{}")
        return path

    def test_processed_presence_is_not_readiness(self):
        episode = self.episode()
        report = scan_dataset([self.base / "raw"])
        row = report["episodes"][0]
        self.assertTrue(row["raw_pair_present"])
        self.assertTrue(row["processed_inputs_present"])
        self.assertFalse(row["fully_validated"])
        self.assertEqual(row["path"], str(episode))
        self.assertIn("missing_reconstruction_candidate", row["blockers"])
        self.assertIn("missing_calibration_candidate", row["blockers"])
        self.assertEqual(report["counts"]["fully_validated"], 0)

    def test_raw_only(self):
        episode = self.episode(processed=False)
        (episode / "recording.db3").write_bytes(b"bag")
        report = scan_dataset([self.base / "raw"])
        row = report["episodes"][0]
        self.assertTrue(row["raw_bags_present"])
        self.assertFalse(row["processed_inputs_present"])
        self.assertIn("missing_processed_inputs", row["blockers"])

    def test_duplicate_variants_are_flagged_not_combined(self):
        self.episode("v1")
        self.episode("v2")
        self.episode("v2", session="snimanje_19_6")
        report = scan_dataset([self.base / "v2", self.base / "v1"])
        self.assertEqual(len(report["episodes"]), 3)
        self.assertEqual(len(report["duplicate_recording_groups"]), 1)
        self.assertEqual(report["duplicate_recording_groups"][0]["recording_id"],
                         "snimanje_23_10/episode18_kugla")
        duplicates = [e for e in report["episodes"] if "duplicate_recording_variants" in e["blockers"]]
        self.assertEqual(len(duplicates), 2)

    def test_config_exact_and_session_candidates_are_distinct(self):
        episode = self.episode()
        config = self.base / "experiment.json"
        config.write_text(json.dumps({"paths": {"episode": str(episode)}, "mass_kg": 0.25}))
        alternate = self.base / "alternate.json"
        alternate.write_text(json.dumps({"paths": {"episode": "/remote/other/snimanje_23_10/episode18_kugla"}}))
        other = self.base / "other.json"
        other.write_text(json.dumps({"paths": {"episode": "/remote/other/snimanje_19_6/episode18_kugla"}}))
        row = scan_dataset([self.base / "raw"], config_paths=[other, config, alternate])["episodes"][0]
        self.assertEqual(len(row["config_candidates"]), 2)
        self.assertEqual({c["match"] for c in row["config_candidates"]},
                         {"resolved_path", "session_candidate_only"})
        self.assertIn("ambiguous_config_candidates", row["blockers"])
        self.assertFalse(row["fully_validated"])

    def test_reconstruction_ambiguity_and_particle_presence(self):
        episode = self.episode()
        recon = self.base / "recon"
        for name in ("first", "second"):
            directory = recon / name
            directory.mkdir(parents=True)
            (directory / "reconstruction_metadata.json").write_text(json.dumps({
                "episode_dir": str(episode), "frame": 0}))
        (recon / "first" / "sampled_particles_xyz.npy").write_bytes(b"particles")
        row = scan_dataset([self.base / "raw"], reconstruction_roots=[recon])["episodes"][0]
        self.assertEqual(len(row["reconstruction_candidates"]), 2)
        self.assertEqual([c["particles"]["exists"] for c in row["reconstruction_candidates"]], [True, False])
        self.assertIn("ambiguous_reconstruction_candidates", row["blockers"])

    def test_calibration_candidates_and_external_reference_not_opened(self):
        episode = self.episode()
        for name in ("scene_calibration_v1.json", "scene_calibration_v2.json"):
            (episode / name).write_text("{}")
        (episode / "sequence_metadata.json").write_text(json.dumps({
            "calibration": {"status": "available", "path": "/missing/external.json"}}))
        row = scan_dataset([self.base / "raw"])["episodes"][0]
        self.assertEqual(len(row["calibration_candidates"]), 3)
        self.assertIn("ambiguous_calibration_candidates", row["blockers"])
        self.assertEqual(row["calibration_record_status"], "available")

    def test_malformed_and_oversized_metadata_reported(self):
        for i, text in enumerate(("{", "[]", '{"x": 1, "x": 2}', '{"x": NaN}', " " * (MAX_METADATA_BYTES + 1))):
            episode = self.episode(name=f"episode{i}_kugla")
            (episode / "sequence_metadata.json").write_text(text)
        report = scan_dataset([self.base / "raw"])
        self.assertEqual(len(report["errors"]), 5)
        self.assertTrue(all(not e["sequence_metadata_parsed"] for e in report["episodes"]))

    def test_symlinks_are_not_traversed_or_parsed(self):
        episode = self.episode()
        external = self.base / "external"
        external.mkdir()
        (external / "episode9_kugla").mkdir()
        (self.base / "raw" / "escape").symlink_to(external, target_is_directory=True)
        (episode / "conversion_metadata.json").symlink_to(external / "missing.json")
        root_link = self.base / "root_link"
        root_link.symlink_to(self.base / "raw", target_is_directory=True)
        report = scan_dataset([self.base / "raw", root_link])
        self.assertEqual(len(report["episodes"]), 1)
        self.assertEqual(sum(e["kind"] == "symlink_skipped" for e in report["errors"]), 3)

    def test_deterministic_no_tensor_loading(self):
        self.episode("v1")
        self.episode("v2")
        original = Path.open
        def only_metadata(path, *args, **kwargs):
            if path.suffix in (".pt", ".npy", ".db3"):
                raise AssertionError("Attempted tensor or recording read")
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "open", only_metadata):
            first = scan_dataset([self.base / "v1", self.base / "v2"])
            second = scan_dataset([self.base / "v2", self.base / "v1", self.base / "v1"])
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)

    def test_unqualified_basename_is_not_cross_matched(self):
        episode = self.base / "raw" / "episode18_kugla"
        episode.mkdir(parents=True)
        config = self.base / "config.json"
        config.write_text(json.dumps({"paths": {"episode": "/remote/episode18_kugla"}}))
        row = scan_dataset([self.base / "raw"], config_paths=[config])["episodes"][0]
        self.assertIsNone(row["recording_id"])
        self.assertEqual(row["config_candidates"], [])
        self.assertIn("unresolved_session_identity", row["blockers"])

    def test_missing_roots_and_invalid_configs_are_errors(self):
        config = self.base / "config.json"
        config.write_text('{"paths": []}')
        report = scan_dataset([self.base / "missing"], config_paths=[config])
        self.assertEqual({e["kind"] for e in report["errors"]}, {"missing_root", "config_error"})

    def test_draft_is_unrunnable_and_does_not_invent_values(self):
        self.episode()
        draft = inventory_draft(scan_dataset([self.base / "raw"]))
        self.assertFalse(draft["runnable"])
        self.assertIn("draft", draft["schema"])
        row = draft["episodes"][0]
        self.assertIsNone(row["config"])
        self.assertIsNone(row["split"])
        self.assertIsNone(row["weight"])
        self.assertNotIn("mass_kg", row)
        with self.assertRaises(ValueError):
            inventory_draft({})


if __name__ == "__main__":
    unittest.main()

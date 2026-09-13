"""Host-only validation of optional loss targets; no Taichi initialization."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from experiments.differentiable_mpm import loss_targets as targets
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class TargetCamera:
    width = 4
    height = 3

    def __init__(self):
        self.camera_from_scene = np.array([[0., -1., 0., 0.1], [1., 0., 0., 0.2],
                                          [0., 0., 1., 0.3], [0., 0., 0., 1.]])

    def as_dict(self):
        return {"width": self.width, "height": self.height,
                "camera_from_scene": self.camera_from_scene.tolist()}


class LossTargetTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory("loss-targets-")
        self.directory = Path(self.temp.name)
        self.camera = TargetCamera()
        self.config = SimpleNamespace(paths={}, expected_sha256={})
        self.loss = SimpleNamespace(version="dpsi-pcd-cd-v1", target_source="recorded_cloud",
                                    tracking_weight=0., mask_weight=0., empty_track_policy="error")
        self.kwargs = dict(frame_indices=(1, 3), sequence_fingerprint="1" * 64,
                           calibration_sha256="2" * 64, initial_particles_sha256="3" * 64,
                           n_particles=5, camera=self.camera, frame_times={0: 10., 1: 10.1, 2: 10.2, 3: 10.3})

    def tearDown(self):
        self.temp.cleanup()

    def metadata(self, kind):
        result = {"schema": targets.TARGET_SCHEMAS[kind], "units": "pixel" if kind == "masks" else "m",
                  "coordinate_frame": "image" if kind == "masks" else "scene",
                  "timestamp_reference": "sequence", "sequence_fingerprint": "1" * 64,
                  "calibration_sha256": "2" * 64,
                  "provenance": {"kind": "synthetic_fixture", "description": "Synthetic loader fixture"}}
        if kind == "points":
            result["target_representation"] = "partial_observed"
        elif kind == "tracks":
            result["initial_particles_sha256"] = "3" * 64
        else:
            result["camera_fingerprint"] = targets.camera_fingerprint(self.camera)
        return result

    def arrays(self, kind):
        result = {"frame_indices": np.array([1, 3]), "timestamps": np.array([10.1, 10.3])}
        if kind == "points":
            result.update(offsets=np.array([0, 2, 3]), points=np.array([[.1, .2, .3], [.2, .3, .4], [.4, .5, .6]]))
        elif kind == "tracks":
            result.update(track_ids=np.array([10, 20]), particle_ids=np.array([1, 4]),
                          positions=np.arange(12, dtype=float).reshape(2, 2, 3) / 10,
                          valid=np.array([[1, 0], [1, 1]], dtype=bool))
        else:
            result.update(foreground=np.zeros((2, 3, 4), dtype=bool), known=np.ones((2, 3, 4), dtype=bool))
            result["foreground"][:, 1, 1] = True
        return result

    def write_pair(self, kind, metadata=None, arrays=None):
        archive_name, meta_name = targets.TARGET_PATH_PAIRS[kind]
        arrays = self.arrays(kind) if arrays is None else arrays
        metadata = self.metadata(kind) if metadata is None else metadata
        path = self.directory / (archive_name + ".npz")
        meta_path = self.directory / (meta_name + ".json")
        np.savez_compressed(path, **arrays)
        meta_path.write_text(json.dumps(metadata, allow_nan=False))
        for name, target in ((archive_name, path), (meta_name, meta_path)):
            self.config.paths[name] = target
            self.config.expected_sha256[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        return path, meta_path

    def load(self):
        return targets.load_loss_targets(self.config, self.loss, **self.kwargs)

    def test_no_files_needed_for_recorded_cloud_or_explicit_ablation(self):
        result = self.load()
        self.assertEqual(result.by_frame, {1: {}, 3: {}})
        self.assertEqual(result.provenance["inputs"], {})
        self.loss.version = "empm-offline-v1"
        self.assertEqual(targets.required_target_paths(self.loss), ())
        self.assertEqual(self.load().by_frame, {1: {}, 3: {}})

    def test_external_point_loading_conversion_and_immutable_copies(self):
        self.loss.target_source = "external"
        arrays = self.arrays("points")
        expected = arrays["points"].copy()
        metadata = self.metadata("points")
        metadata["coordinate_frame"] = "camera_optical"
        matrix = self.camera.camera_from_scene
        arrays["points"] = expected @ matrix[:3, :3].T + matrix[:3, 3]
        paths = self.write_pair("points", metadata, arrays)
        before = [path.read_bytes() for path in paths]
        result = self.load()
        np.testing.assert_allclose(result.by_frame[1]["points_scene"], expected[:2], atol=1e-15)
        np.testing.assert_allclose(result.by_frame[3]["points_scene"], expected[2:], atol=1e-15)
        self.assertFalse(result.by_frame[1]["points_scene"].flags.writeable)
        self.assertEqual(result.by_frame[1]["metadata"]["source_kind"], "synthetic_fixture")
        self.assertEqual(before, [path.read_bytes() for path in paths])
        self.assertEqual(result.provenance["inputs"]["points"]["scored_frames"], [1, 3])
        self.assertNotIn("path", result.provenance["inputs"]["points"]["archive"])

    def test_required_pairs_and_hashes(self):
        self.loss.target_source = "external"
        self.assertEqual(targets.required_target_paths(self.loss), targets.TARGET_PATH_PAIRS["points"])
        with self.assertRaisesRegex(ValueError, "requires inputs"):
            self.load()
        path, _ = self.write_pair("points")
        self.config.paths.pop("loss_point_targets_metadata")
        with self.assertRaisesRegex(ValueError, "provided together"):
            self.load()
        self.write_pair("points")
        self.config.expected_sha256.pop("loss_point_targets")
        with self.assertRaisesRegex(ValueError, "Expected hash"):
            self.load()
        self.write_pair("points")
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.load()

    def test_missing_frames_and_changed_timestamps_fail(self):
        self.loss.target_source = "external"
        arrays = self.arrays("points")
        arrays["frame_indices"] = np.array([1, 2])
        arrays["timestamps"] = np.array([10.1, 10.2])
        self.write_pair("points", arrays=arrays)
        with self.assertRaisesRegex(ValueError, "missing scored frames"):
            self.load()
        arrays = self.arrays("points")
        arrays["timestamps"][1] += .001
        self.write_pair("points", arrays=arrays)
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            self.load()
        arrays["timestamps"][1] = 10.3
        arrays["frame_indices"] = np.array([1, 1])
        self.write_pair("points", arrays=arrays)
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            self.load()

    def test_metadata_identity_units_and_provenance(self):
        self.loss.target_source = "external"
        for key, value, message in (("units", "mm", "units"), ("coordinate_frame", "world", "coordinate_frame"),
                                    ("sequence_fingerprint", "4" * 64, "sequence_fingerprint differs"),
                                    ("calibration_sha256", "4" * 64, "calibration_sha256 differs"),
                                    ("timestamp_reference", "simulated", "timestamp_reference"),
                                    ("provenance", {"kind": "simulation", "description": "not observed"}, "provenance")):
            with self.subTest(key=key):
                metadata = self.metadata("points")
                metadata[key] = value
                self.write_pair("points", metadata)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_volume_target_required_for_prt_and_rejected_for_pcd(self):
        self.loss.target_source = "external"
        self.loss.version = "dpsi-prt-emd-v1"
        self.write_pair("points")
        with self.assertRaisesRegex(ValueError, "PRT requires volume"):
            self.load()
        metadata = self.metadata("points")
        metadata["target_representation"] = "inferred_volume"
        self.write_pair("points", metadata)
        self.assertEqual(self.load().by_frame[3]["target_representation"], "inferred_volume")
        self.loss.version = "dpsi-pcd-emd-v1"
        with self.assertRaisesRegex(ValueError, "other modes require observed cloud"):
            self.load()

    def test_point_offsets_nonfinite_and_malformed_array_types(self):
        self.loss.target_source = "external"
        for field, value in (("offsets", np.array([0, 3, 2])), ("points", np.array([[np.nan, 0, 0]])),
                             ("offsets", np.array([0., 2., 3.])), ("points", np.zeros((3, 2))),
                             ("points", np.zeros((3, 3), dtype=complex))):
            with self.subTest(field=field):
                arrays = self.arrays("points")
                arrays[field] = value
                self.write_pair("points", arrays=arrays)
                with self.assertRaises(ValueError):
                    self.load()

    def test_extra_frames_checked_even_when_not_scored(self):
        self.loss.target_source = "external"
        arrays = self.arrays("points")
        arrays.update(frame_indices=np.array([0, 1, 3]), timestamps=np.array([10., 10.1, 10.3]),
                      offsets=np.array([0, 1, 2, 3]))
        self.write_pair("points", arrays=arrays)
        result = self.load()
        self.assertEqual(set(result.by_frame), {1, 3})
        self.assertEqual(result.provenance["inputs"]["points"]["available_frames"], [0, 1, 3])
        arrays["timestamps"][0] = 9.5
        self.write_pair("points", arrays=arrays)
        with self.assertRaisesRegex(ValueError, "timestamp mismatch at frame 0"):
            self.load()

    def test_track_ids_fixed_mapping_masks_and_empty_policy(self):
        self.loss.version = "empm-offline-v1"
        self.loss.tracking_weight = 1.
        arrays = self.arrays("tracks")
        arrays["positions"][0, 1] = np.nan
        self.write_pair("tracks", arrays=arrays)
        result = self.load()
        np.testing.assert_array_equal(result.by_frame[1]["track_particle_ids"], [1, 4])
        np.testing.assert_array_equal(result.by_frame[1]["track_valid"], [True, False])
        np.testing.assert_array_equal(result.by_frame[1]["track_positions_scene"][1], [0., 0., 0.])
        self.assertFalse(result.by_frame[1]["track_particle_ids"].flags.writeable)
        arrays["valid"][1] = False
        self.write_pair("tracks", arrays=arrays)
        with self.assertRaisesRegex(ValueError, "no valid tracks"):
            self.load()
        self.loss.empty_track_policy = "skip"
        self.assertTrue(self.load().by_frame[3]["metadata"]["tracking_skipped_empty"])
        self.assertEqual(set(self.load().by_frame), {1, 3})

    def test_track_mapping_hash_duplicates_range_and_nonfinite(self):
        self.loss.version = "empm-offline-v1"
        self.loss.tracking_weight = 1.
        metadata = self.metadata("tracks")
        metadata["initial_particles_sha256"] = "4" * 64
        self.write_pair("tracks", metadata)
        with self.assertRaisesRegex(ValueError, "initial_particles_sha256 differs"):
            self.load()
        for field, value in (("particle_ids", np.array([1, 1])), ("track_ids", np.array([10, 10])),
                             ("particle_ids", np.array([1, 5])), ("particle_ids", np.array([1., 4.])),
                             ("positions", np.full((2, 2, 3), np.nan)), ("valid", np.ones((2, 2)) * .5)):
            with self.subTest(field=field):
                arrays = self.arrays("tracks")
                arrays[field] = value
                self.write_pair("tracks", arrays=arrays)
                with self.assertRaises(ValueError):
                    self.load()

    def test_masks_require_camera_known_pixels_and_boolean_values(self):
        self.loss.version = "empm-mask-inspired-v1"
        self.loss.mask_weight = 1.
        with self.assertRaisesRegex(ValueError, "requires inputs"):
            self.load()
        arrays = self.arrays("masks")
        arrays["known"][0, 1, 1] = False
        self.write_pair("masks", arrays=arrays)
        result = self.load()
        self.assertFalse(result.by_frame[1]["known_mask"][1, 1])
        self.assertTrue(result.by_frame[1]["foreground_mask"][1, 1])
        self.assertFalse(result.by_frame[1]["known_mask"].flags.writeable)
        metadata = self.metadata("masks")
        metadata["camera_fingerprint"] = "5" * 64
        self.write_pair("masks", metadata, arrays)
        with self.assertRaisesRegex(ValueError, "camera_fingerprint differs"):
            self.load()
        arrays["known"][0] = False
        self.write_pair("masks", arrays=arrays)
        with self.assertRaisesRegex(ValueError, "no known segmentation pixels"):
            self.load()
        arrays = self.arrays("masks")
        arrays["foreground"] = np.zeros((2, 3, 5), dtype=bool)
        self.write_pair("masks", arrays=arrays)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.load()

    def test_external_cloud_and_tracks_merge(self):
        self.loss.version = "empm-offline-v1"
        self.loss.target_source = "external"
        self.loss.tracking_weight = 1.
        self.write_pair("points")
        self.write_pair("tracks")
        result = self.load()
        self.assertIn("points_scene", result.by_frame[1])
        self.assertIn("track_positions_scene", result.by_frame[1])
        self.assertEqual(set(result.by_frame[1]["metadata"]["target_sources"]), {"points", "tracks"})
        self.assertEqual(len(targets.required_target_paths(self.loss)), 4)

    def test_irrelevant_inputs_and_nonfinite_requested_times_rejected(self):
        self.write_pair("points")
        with self.assertRaisesRegex(ValueError, "target_source='external'"):
            self.load()
        self.loss.target_source = "external"
        self.kwargs["frame_times"][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            self.load()
        self.kwargs["frame_times"][1] = 10.1
        self.kwargs["frame_indices"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()

    def test_relocation_keeps_identity_and_metadata_hash_is_checked(self):
        self.loss.target_source = "external"
        _, meta = self.write_pair("points")
        first = self.load()
        relocated = self.directory / "relocated"
        relocated.mkdir()
        for name, path in list(self.config.paths.items()):
            target = relocated / path.name
            target.write_bytes(path.read_bytes())
            self.config.paths[name] = target
        second = self.load()
        self.assertEqual(first.provenance, second.provenance)
        np.testing.assert_array_equal(first.by_frame[1]["points_scene"], second.by_frame[1]["points_scene"])
        meta = self.config.paths["loss_point_targets_metadata"]
        meta.write_text(meta.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.load()

    def test_archive_limits_and_pickle_refusal(self):
        self.loss.target_source = "external"
        self.write_pair("points")
        with patch.object(targets, "MAX_ARCHIVE_BYTES", 10), self.assertRaisesRegex(ValueError, "input limit"):
            self.load()
        with patch.object(targets, "MAX_UNCOMPRESSED_BYTES", 10), self.assertRaisesRegex(ValueError, "uncompressed"):
            self.load()
        arrays = self.arrays("points")
        arrays["points"] = np.array([["not", "numeric", "points"]], dtype=object)
        self.write_pair("points", arrays=arrays)
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            self.load()

    def test_declared_array_dimensions_checked_before_loading(self):
        self.loss.target_source = "external"
        path, _ = self.write_pair("points")
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False,
                                                    "shape": (10**12, 3)})
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("points.npy", header.getvalue())
        self.config.expected_sha256["loss_point_targets"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(targets.np, "load") as load:
            with self.assertRaisesRegex(ValueError, "dimensions do not match stored bytes"):
                self.load()
            load.assert_not_called()

    def test_unknown_and_duplicate_metadata_fields_rejected(self):
        self.loss.target_source = "external"
        metadata = self.metadata("points")
        metadata["typo"] = True
        self.write_pair("points", metadata)
        with self.assertRaisesRegex(ValueError, "metadata fields"):
            self.load()
        _, path = self.write_pair("points")
        path.write_text('{"schema":"a", "schema":"b"}')
        self.config.expected_sha256["loss_point_targets_metadata"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.load()


if __name__ == "__main__":
    unittest.main()

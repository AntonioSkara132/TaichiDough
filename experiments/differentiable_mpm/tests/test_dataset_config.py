"""Host-only validation for shared-material dataset manifests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.differentiable_mpm.config import SCHEMA as EPISODE_SCHEMA
from experiments.differentiable_mpm.dataset_config import MATERIAL_NAMES, SCHEMA, load_dataset
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class DatasetConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory(prefix="dataset-config-")
        self.root = Path(self.temporary.name)
        self.path = self.root / "dataset.json"
        self.initial = dict(zip(MATERIAL_NAMES, (12000.0, 0.28, 12.0, 0.88, 1.12)))
        self.configs = []
        self.config_paths = []
        for index in range(2):
            config = {
                "schema": EPISODE_SCHEMA, "name": f"episode{index}",
                "paths": {name: str(self.root / f"episode{index}" / name) for name in
                          ("episode", "calibration", "initial_particles", "reconstruction_metadata", "tool_geometry")},
                "expected_sha256": {"initial_particles": "a" * 64},
                "simulation": {"n_particles": 8, "grid": 16, "precision": "f64", "plasticity": "stretch-clamp",
                               "tool_contact_absorption": 0.3, "tool_stickiness": 0.6},
                "parameters": {**DEFAULT_PARAMETERS, "floor_retention": 0.2 + 0.5 * index},
                "fit_parameters": ["youngs_modulus", "poisson_ratio", "viscosity"],
                "training": {"start_frame": 1, "end_frame": 2},
                "validation": {"start_frame": 3, "end_frame": 4},
            }
            self.configs.append(config)
            self.config_paths.append(self.root / f"episode{index}.json")
        self.document = {
            "schema": SCHEMA, "name": "shared-material",
            "shared_parameters": {"initial": self.initial, "fit": list(MATERIAL_NAMES), "bounds": {
                "youngs_modulus": [4000, 40000], "poisson_ratio": [0.1, 0.4], "viscosity": [0, 80],
                "plastic_min": [0.75, 0.98], "plastic_max": [1.02, 1.3]}},
            "episodes": [{"id": f"episode{index}", "config": f"episode{index}.json",
                          "membership": "training" if index == 0 else "validation", "weight": index + 1,
                          "scored_window": {"start_frame": 1, "end_frame": 4, "stride": 2}}
                         for index in range(2)],
        }
        self.write()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, document=None):
        for path, config in zip(self.config_paths, self.configs):
            path.write_text(json.dumps(config))
        self.path.write_text(json.dumps(self.document if document is None else document))

    def test_effective_parameters_and_nonsticky_tools_preserve_each_floor(self):
        dataset = load_dataset(self.path)
        self.assertEqual(dataset.fit_parameters, MATERIAL_NAMES)
        self.assertEqual(dataset.shared_initial, self.initial)
        for index, episode in enumerate(dataset.episodes):
            self.assertEqual(episode.config.fit_parameters, list(MATERIAL_NAMES))
            self.assertEqual(episode.config.parameter_bounds, dataset.parameter_bounds)
            self.assertEqual(episode.config.simulation["tool_contact_absorption"], 0)
            self.assertEqual(episode.config.simulation["tool_stickiness"], 0)
            self.assertEqual(episode.config.parameters["tool_retention"], 1)
            self.assertEqual(episode.config.parameters["floor_retention"], 0.2 + 0.5 * index)
            self.assertEqual(episode.scored_window.indices(), [1, 3, 4])
            self.assertEqual({name: episode.config.parameters[name] for name in MATERIAL_NAMES}, self.initial)

    def test_parameter_merge_never_uses_another_episodes_floor(self):
        dataset = load_dataset(self.path)
        shared = {**self.initial, "viscosity": 0.0}
        for index, episode in enumerate(dataset.episodes):
            effective = episode.parameters_for(shared)
            self.assertEqual(effective["viscosity"], 0)
            self.assertEqual(effective["floor_retention"], 0.2 + 0.5 * index)
            self.assertEqual(effective["tool_retention"], 1)
        self.assertEqual(dataset.shared_initial["viscosity"], 12)

    def test_parameter_merge_rejects_missing_extra_and_invalid_values(self):
        episode = load_dataset(self.path).episodes[0]
        invalid = [{name: value for name, value in self.initial.items() if name != "viscosity"},
                   {**self.initial, "floor_retention": 0.5}, {**self.initial, "tool_retention": 0.5},
                   {**self.initial, "viscosity": -1}, {**self.initial, "viscosity": float("nan")},
                   {**self.initial, "plastic_min": True}]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                episode.parameters_for(values)

    def test_parameter_space_uses_existing_transforms_and_fitted_material_only(self):
        dataset = load_dataset(self.path)
        space = dataset.parameter_space()
        self.assertEqual(space.fit, MATERIAL_NAMES)
        effective = space.physical(space.coordinates())
        for name in MATERIAL_NAMES:
            self.assertAlmostEqual(effective[name], self.initial[name])
        self.assertEqual(effective["tool_retention"], 1)
        self.assertEqual(effective["floor_retention"], 0.2)
        gradient = {name: 1.0 for name in MATERIAL_NAMES}
        np.testing.assert_allclose(space.pullback(space.coordinates(), gradient), [12000, 0.1, 100, -0.1, 0.1])

    def test_sources_hashes_and_original_bytes_are_preserved(self):
        paths = [self.path, *self.config_paths]
        before = {path: path.read_bytes() for path in paths}
        dataset = load_dataset(self.path)
        self.assertEqual(dataset.config_path, self.path.resolve())
        self.assertEqual(dataset.source_document_sha256, hashlib.sha256(before[self.path]).hexdigest())
        for episode, path in zip(dataset.episodes, self.config_paths):
            self.assertEqual(episode.config_path, path)
            self.assertEqual(episode.config.config_path, path)
            self.assertEqual(episode.config.source_document_sha256, hashlib.sha256(before[path]).hexdigest())
            self.assertEqual(episode.config.expected_sha256["initial_particles"], "a" * 64)
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.assertEqual(dataset.fingerprint, load_dataset(self.path).fingerprint)

    def test_manifest_relative_paths_and_explicit_override_precedence(self):
        self.document["episodes"][0]["path_overrides"] = {"episode": "recordings/one", "calibration": "calibration/one.json"}
        self.write()
        dataset = load_dataset(self.path, path_overrides={"episode0": {"episode": Path("remote/one")}})
        config = dataset.episodes[0].config
        self.assertEqual(config.paths["episode"], self.root / "remote" / "one")
        self.assertEqual(config.paths["calibration"], self.root / "calibration" / "one.json")
        self.assertEqual(config.paths["initial_particles"], Path(self.configs[0]["paths"]["initial_particles"]))
        self.assertEqual(config.path_overrides["episode"], str(self.root / "remote" / "one"))

    def test_path_overrides_reject_unknown_episodes_inputs_and_malformed_values(self):
        invalid = [{"unknown": {"episode": "/a"}}, {"episode0": {"unknown_input": "/a"}},
                   {"episode0": "not a mapping"}, {"episode0": {"episode": None}},
                   {"episode0": {1: "/a"}}, {1: {}}, []]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_dataset(self.path, path_overrides=overrides)

    def test_runtime_overrides_apply_before_common_settings_checks(self):
        self.configs[1]["backend"] = "vulkan"
        self.configs[1]["simulation"].update(precision="f32", p2g_mode="serial", physics_version="legacy-v1")
        self.write()
        with self.assertRaisesRegex(ValueError, "incompatible shared"):
            load_dataset(self.path)
        dataset = load_dataset(self.path, backend="cuda", precision="f64", p2g_mode="atomic",
                               physics_version="corrected-v1", segment_length=3)
        for episode in dataset.episodes:
            self.assertEqual(episode.config.backend, "cuda")
            self.assertEqual(episode.config.simulation["precision"], "f64")
            self.assertEqual(episode.config.simulation["p2g_mode"], "atomic")
            self.assertEqual(episode.config.simulation["physics_version"], "corrected-v1")
            self.assertEqual(episode.config.segment_length, 3)

    def test_invalid_runtime_overrides_are_rejected(self):
        for kwargs in ({"backend": "auto"}, {"backend": []}, {"physics_version": "unknown"},
                       {"physics_version": True}, {"precision": "f16"}, {"p2g_mode": "other"},
                       {"segment_length": 0}, {"segment_length": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                load_dataset(self.path, **kwargs)

    def test_default_corrected_and_explicit_legacy_are_distinct(self):
        corrected = load_dataset(self.path)
        legacy = load_dataset(self.path, physics_version="legacy-v1")
        self.assertEqual(corrected.episodes[0].config.simulation["physics_version"], "corrected-v1")
        self.assertEqual(legacy.episodes[0].config.simulation["physics_version"], "legacy-v1")
        self.assertNotEqual(corrected.fingerprint, legacy.fingerprint)

    def test_config_declared_legacy_is_not_silently_replaced(self):
        for config in self.configs:
            config["simulation"]["physics_version"] = "legacy-v1"
        self.write()
        self.assertEqual(load_dataset(self.path).episodes[0].config.simulation["physics_version"], "legacy-v1")

    def test_different_mass_floor_grid_dt_and_camera_inputs_are_allowed(self):
        self.configs[1]["simulation"].update(n_particles=27, grid=24, dt=0.0001, floor_y=0.02,
                                              floor_absorption=0.2, floor_stickiness=0.1,
                                              floor_plastic_damping_band=0.01)
        self.configs[1].update(mass_kg=0.5, density_kg_m3=400, segment_length=8)
        self.write()
        dataset = load_dataset(self.path)
        self.assertEqual(dataset.episodes[1].config.mass_kg, 0.5)
        self.assertEqual(dataset.episodes[1].config.simulation["grid"], 24)
        self.assertEqual(dataset.episodes[1].config.simulation["floor_y"], 0.02)
        self.assertNotEqual(dataset.episodes[0].config.paths["calibration"], dataset.episodes[1].config.paths["calibration"])

    def test_incompatible_material_numerical_and_loss_settings_are_rejected(self):
        original = deepcopy(self.configs[1])
        variants = [("simulation", {"use_jp": True}), ("simulation", {"jp_hardening": 2}),
                    ("simulation", {"velocity_damping": 0.9}), ("simulation", {"precision": "f32"}),
                    ("simulation", {"p2g_mode": "serial"}), ("simulation", {"physics_version": "legacy-v1"}),
                    ("loss", {"version": "partial-visible-splats-v1"}), ("loss", {"depth_weight": 2}),
                    ("loss", {"visibility_temperature_m": 0.01}), ("optimizer", {"learning_rate": 0.1}),
                    ("observation", {"width": 80}), ("strict_loss", {"depth_weight": 2})]
        for field, values in variants:
            self.configs[1] = deepcopy(original)
            self.configs[1].setdefault(field, {}).update(values)
            self.write()
            with self.subTest(field=field, values=values), self.assertRaisesRegex(ValueError, "incompatible shared"):
                load_dataset(self.path)

    def test_implicit_and_explicit_common_defaults_compare_equally(self):
        self.configs[1]["simulation"].update(use_jp=False, physics_version="corrected-v1", p2g_mode="atomic")
        self.configs[1]["loss"] = {"version": "partial-visible-splats-v2", "depth_weight": 1}
        self.configs[1]["optimizer"] = {"learning_rate": 0.05, "learning_rate_policy": "persistent-v1"}
        self.write()
        self.assertEqual(len(load_dataset(self.path).episodes), 2)

    def test_shared_fit_subset_and_bounds_replace_source_fit_settings(self):
        self.document["shared_parameters"]["fit"] = ["youngs_modulus", "viscosity"]
        self.document["shared_parameters"]["bounds"] = {"youngs_modulus": [4000, 40000]}
        self.configs[1]["fit_parameters"] = ["tool_retention"]
        self.configs[1]["parameter_bounds"] = {"tool_retention": [0, 1]}
        self.write()
        dataset = load_dataset(self.path)
        self.assertEqual(dataset.parameter_space().fit, ("youngs_modulus", "viscosity"))
        self.assertEqual(dataset.episodes[1].config.parameter_bounds, {"youngs_modulus": [4000, 40000]})

    def test_fit_rejects_contact_duplicates_unknowns_and_empty(self):
        for fit in (["tool_retention"], ["floor_retention"], ["unknown"], ["viscosity", "viscosity"],
                    [], "viscosity", [True], [{}]):
            document = deepcopy(self.document)
            document["shared_parameters"]["fit"] = fit
            self.write(document)
            with self.subTest(fit=fit), self.assertRaises(ValueError):
                load_dataset(self.path)

    def test_plastic_fitting_requires_active_model_for_every_episode(self):
        self.configs[1]["simulation"]["plasticity"] = "none"
        self.write()
        with self.assertRaisesRegex(ValueError, "plastic bounds"):
            load_dataset(self.path)

    def test_material_initial_and_bounds_must_be_valid(self):
        cases = [({"viscosity": -1}, None), ({"poisson_ratio": 0.5}, None), ({"youngs_modulus": True}, None),
                 ({"youngs_modulus": 10 ** 400}, None), ({}, {"viscosity": [10, 5]}),
                 ({}, {"viscosity": [20, 30]}), ({}, {"plastic_min": [0.5, 1.1]}),
                 ({}, {"viscosity": [0]}), ({}, {"viscosity": [False, 20]})]
        for initial, bounds in cases:
            document = deepcopy(self.document)
            document["shared_parameters"]["initial"].update(initial)
            if bounds is not None:
                document["shared_parameters"]["bounds"] = bounds
            self.write(document)
            with self.subTest(initial=initial, bounds=bounds), self.assertRaises(ValueError):
                load_dataset(self.path)

    def test_exact_five_initial_values_are_required(self):
        for values in ({k: v for k, v in self.initial.items() if k != "viscosity"},
                       {**self.initial, "tool_retention": 1}, [], None):
            document = deepcopy(self.document)
            document["shared_parameters"]["initial"] = values
            self.write(document)
            with self.subTest(values=values), self.assertRaises(ValueError):
                load_dataset(self.path)

    def test_nonsticky_contact_defaults_and_explicit_values(self):
        self.document["tool_contact"] = {"retention": 1, "absorption": 0, "stickiness": 0}
        self.write()
        self.assertEqual(load_dataset(self.path).as_dict()["tool_contact"], self.document["tool_contact"])
        for values in ({"retention": 0.2}, {"absorption": 0.1}, {"stickiness": 0.1}, {"retention": True}):
            document = deepcopy(self.document)
            document["tool_contact"] = values
            self.write(document)
            with self.subTest(values=values), self.assertRaises(ValueError):
                load_dataset(self.path)

    def test_duplicate_episode_id_and_recording_are_rejected(self):
        document = deepcopy(self.document)
        document["episodes"][1]["id"] = "episode0"
        self.write(document)
        with self.assertRaisesRegex(ValueError, "Duplicate episode ID"):
            load_dataset(self.path)
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate recording path"):
            load_dataset(self.path, path_overrides={"episode1": {"episode": self.configs[0]["paths"]["episode"]}})

    def test_duplicate_expected_recording_fingerprint_is_rejected(self):
        for config in self.configs:
            config["expected_sequence_fingerprint"] = "c" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate expected recording fingerprint"):
            load_dataset(self.path)

    def test_unsafe_ids_membership_and_empty_training_are_rejected(self):
        for value in ("", "../episode", "x/y", ".", "..", "has space", "a" * 129, True):
            document = deepcopy(self.document)
            document["episodes"][0]["id"] = value
            self.write(document)
            with self.subTest(id=value), self.assertRaises(ValueError):
                load_dataset(self.path)
        for value in ("held-out", "train", "", True, []):
            document = deepcopy(self.document)
            document["episodes"][0]["membership"] = value
            self.write(document)
            with self.subTest(membership=value), self.assertRaises(ValueError):
                load_dataset(self.path)
        document = deepcopy(self.document)
        document["episodes"][0]["membership"] = "validation"
        self.write(document)
        with self.assertRaisesRegex(ValueError, "at least one training"):
            load_dataset(self.path)

    def test_positive_finite_weights_and_explicit_windows_are_required(self):
        for value in (0, -1, float("nan"), float("inf"), True, "1", [], None):
            document = deepcopy(self.document)
            document["episodes"][0]["weight"] = value
            self.write(document)
            with self.subTest(weight=value), self.assertRaises(ValueError):
                load_dataset(self.path)
        for window in ({"start_frame": 0, "end_frame": 4}, {"start_frame": 2, "end_frame": 1},
                       {"start_frame": True, "end_frame": 4}, {"start_frame": 1, "end_frame": 4, "stride": 0}):
            document = deepcopy(self.document)
            document["episodes"][0]["scored_window"] = window
            self.write(document)
            with self.subTest(window=window), self.assertRaises(ValueError):
                load_dataset(self.path)

    def test_all_training_and_default_unit_weights_are_allowed(self):
        for row in self.document["episodes"]:
            row["membership"] = "training"
            row.pop("weight")
        self.write()
        self.assertTrue(all(episode.weight == 1 for episode in load_dataset(self.path).episodes))

    def test_unknown_and_missing_manifest_fields_are_rejected(self):
        documents = []
        for location in ((), ("shared_parameters",), ("tool_contact",), ("episodes", 0), ("episodes", 0, "scored_window")):
            document = deepcopy(self.document)
            target = document
            for key in location:
                if key == "tool_contact":
                    target.setdefault(key, {})
                target = target[key]
            target["unknown"] = 1
            documents.append(document)
        for key in ("schema", "name", "shared_parameters", "episodes"):
            document = deepcopy(self.document)
            document.pop(key)
            documents.append(document)
        for document in documents:
            self.write(document)
            with self.subTest(document=document), self.assertRaises(ValueError):
                load_dataset(self.path)

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        for text in ('{"schema":"a","schema":"b"}', '{"name":NaN}', '{"name":Infinity}'):
            self.path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_dataset(self.path)

    def test_fingerprint_records_membership_order_window_weight_and_overrides(self):
        baseline = load_dataset(self.path).fingerprint
        variants = []
        for key, value in (("weight", 2), ("scored_window", {"start_frame": 2, "end_frame": 4}),
                           ("membership", "training")):
            document = deepcopy(self.document)
            document["episodes"][1][key] = value if key != "weight" else 3
            variants.append(document)
        document = deepcopy(self.document)
        document["episodes"].reverse()
        variants.append(document)
        for document in variants:
            self.write(document)
            self.assertNotEqual(load_dataset(self.path).fingerprint, baseline)
        self.write()
        self.assertNotEqual(load_dataset(self.path, segment_length=8).fingerprint, baseline)
        self.assertNotEqual(load_dataset(self.path, path_overrides={"episode0": {"episode": "other"}}).fingerprint, baseline)

    def test_loader_does_not_initialize_runtime_or_require_recordings(self):
        import taichi as ti
        self.assertIsNone(ti.lang.impl.get_runtime().prog)
        self.assertFalse(Path(self.configs[0]["paths"]["episode"]).exists())
        load_dataset(self.path)
        self.assertIsNone(ti.lang.impl.get_runtime().prog)


if __name__ == "__main__":
    unittest.main()

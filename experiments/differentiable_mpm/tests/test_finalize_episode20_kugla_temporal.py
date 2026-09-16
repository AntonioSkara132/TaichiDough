from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import experiments.differentiable_mpm.finalize_episode20_kugla_temporal as module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_manifest(root: Path) -> None:
    rows = {
        str(path.relative_to(root)): module.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "SHA256SUMS"
        and path.relative_to(root).parts[0]
        not in {"reconstructions", module.FINALIZATION_DIR}
    }
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in rows.items()),
        encoding="utf-8",
    )


def table_calibration() -> dict[str, Any]:
    return {
        "schema": "taichidough/scene-calibration/v2",
        "source_frame": "mocap",
        "scene_frame": "table-aligned",
        "scene_from_source": [
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, -0.05],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "floor_plane_scene": [0.0, 1.0, 0.0, 0.0],
        "provenance": {
            "table_alignment": {
                "previous_scene_frame": "mocap",
                "source_tensors_transformed": False,
            }
        },
    }


def make_temporal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    temporal = tmp_path / "temporal"
    temporal.mkdir()
    ranges = [
        {"output_name": f"episode20_kugla_chunk{index:02d}"}
        for index in range(1, 11)
    ]
    write_json(
        temporal / "range_selection.json",
        {
            "source": {"split_group": "episode20"},
            "ranges": ranges,
        },
    )
    write_json(
        temporal / "assumptions.json",
        {
            "schema": module.TEMPORAL_SCHEMA + "/assumptions",
            "physical": {
                "nominal_density_kg_m3": module.NOMINAL_DENSITY_KG_M3,
                "reference_volume_m3": module.REFERENCE_VOLUME_M3,
                "eventual_mass_kg": module.CONSTANT_MASS_KG,
                "mass_policy": "constant across all chunks, inherited from the Episode18 reference volume and nominal density",
                "measurement_status": "inherited_assumption_not_episode20_measurement",
            },
            "coordinate_frames": {
                "pointclouds": "mocap",
                "tool_paths": "mocap",
                "scene": "table-aligned",
                "scene_from_source": "rigid transform from the independently estimated Episode20 table plane",
                "table_plane_validation": "passed",
            },
            "inherited_geometry": {
                "physical_setup_status": "assumed_same_UR5e_and_Gen3_spatula_marker_assemblies",
                "sha256": {},
            },
        },
    )
    write_json(
        temporal / "validation.json",
        {
            "status": "valid",
            "checks": {"expected_frame_counts": list(module.EXPECTED_FRAME_COUNTS)},
        },
    )
    calibration_path = temporal / "calibration/scene_calibration_v2.json"
    write_json(calibration_path, table_calibration())
    write_json(
        temporal / "calibration/episode20_table_plane_validation.json",
        {"passed": True, "calibration_sha256": module.sha256_file(calibration_path)},
    )

    reconstruction_rows = []
    for row, count in zip(ranges, module.EXPECTED_FRAME_COUNTS):
        name = row["output_name"]
        recording = temporal / "recordings" / name
        recording.mkdir(parents=True)
        particles_dir = temporal / "reconstructions" / name / "frame_0000"
        particles_dir.mkdir(parents=True)
        particles = np.full((24000, 3), [0.5, 0.01, 0.5], dtype=np.float32)
        np.save(particles_dir / "sampled_particles_xyz.npy", particles)
        write_json(
            particles_dir / "reconstruction_metadata.json",
            {
                "object_volume_m3": 0.0001,
                "array_frames": {"sampled_particles_xyz": "table-aligned"},
            },
        )
        write_json(recording / "sequence_metadata.json", {"frame_count": count})
        reconstruction_rows.append({"episode_id": name})
    write_json(
        temporal / "reconstructions/reconstructions_manifest.json",
        {"status": "valid", "chunks": reconstruction_rows},
    )
    (temporal / "reconstructions/SHA256SUMS").write_text("synthetic\n", encoding="utf-8")

    inherited = tmp_path / "inherited"
    inherited.mkdir()
    inherited_hashes = {}
    for name in module.INHERITED_FILES:
        path = inherited / name
        path.write_bytes(name.encode())
        inherited_hashes[name] = module.sha256_file(path)
    assumptions = json.loads((temporal / "assumptions.json").read_text())
    assumptions["inherited_geometry"]["sha256"] = inherited_hashes
    write_json(temporal / "assumptions.json", assumptions)
    monkeypatch.setattr(module, "INHERITED_BUNDLE", inherited)

    preserved = tmp_path / "v1"
    preserved.mkdir()
    (preserved / "kept.txt").write_text("unchanged\n", encoding="utf-8")
    base = tmp_path / "base.json"
    write_json(
        base,
        {
            "name": "base",
            "paths": {},
            "expected_sha256": {},
            "simulation": {},
        },
    )
    write_manifest(temporal)
    return temporal, preserved, base


class Policy:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        del args


def test_validate_calibration_requires_measured_table_alignment(tmp_path: Path):
    path = tmp_path / "calibration.json"
    calibration = table_calibration()
    write_json(path, calibration)
    assert module.validate_calibration(path)["scene_frame"] == "table-aligned"
    calibration["scene_from_source"] = np.eye(4).tolist()
    write_json(path, calibration)
    with pytest.raises(ValueError, match="measured table alignment"):
        module.validate_calibration(path)


def test_finalize_publishes_ten_training_members_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    temporal, preserved, base = make_temporal(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "load_config", lambda path: path)

    def verify_input_paths(config: Any) -> None:
        del config

    def reference_policy(policy: str) -> Policy:
        del policy
        return Policy()

    monkeypatch.setattr(module, "verify_input_paths", verify_input_paths)
    monkeypatch.setattr(module, "reference_policy", reference_policy)

    def load_sequence(recording: Path) -> SimpleNamespace:
        index = int(recording.name[-2:]) - 1
        count = module.EXPECTED_FRAME_COUNTS[index]
        return SimpleNamespace(times=list(range(count)), fingerprint=f"{index + 1:064x}")

    def get_reference_modules(**kwargs: Any) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(
            dynamics=SimpleNamespace(load_observation_sequence=load_sequence)
        )

    def load_dataset(path: Path) -> SimpleNamespace:
        del path
        return SimpleNamespace(
            episodes=[SimpleNamespace(membership="training") for _ in range(10)],
            fingerprint="dataset-fingerprint",
        )

    monkeypatch.setattr(module, "get_reference_modules", get_reference_modules)
    monkeypatch.setattr(module, "load_dataset", load_dataset)

    dataset_path = module.finalize(temporal, base, preserved)

    assert dataset_path.is_file()
    output = temporal / module.FINALIZATION_DIR
    dataset = json.loads(dataset_path.read_text())
    assert len(dataset["episodes"]) == 10
    assert {row["membership"] for row in dataset["episodes"]} == {"training"}
    assert all(row["scored_window"]["start_frame"] == 1 for row in dataset["episodes"])
    validation = json.loads((output / "preparation_validation.json").read_text())
    assert validation["checks"]["source_coordinate_frame"] == "mocap"
    assert validation["checks"]["scene_coordinate_frame"] == "table-aligned"
    assert validation["checks"]["measured_nonidentity_scene_from_source"] is True
    assert (preserved / "kept.txt").read_text() == "unchanged\n"
    assert not list(temporal.glob(f".{module.FINALIZATION_DIR}.*"))
    with pytest.raises(FileExistsError, match="overwrite"):
        module.finalize(temporal, base, preserved)

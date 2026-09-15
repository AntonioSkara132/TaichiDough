from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import importlib

subject = importlib.import_module(
    "experiments.differentiable_mpm.episode18_temporal_rgb_topview"
)


PARAMETERS = {
    "youngs_modulus": 8800.0, "poisson_ratio": 0.49, "viscosity": 5.0,
    "plastic_min": 0.86857, "plastic_max": 1.06446, "floor_retention": 0.99,
    "tool_friction_coefficient": 0.99, "tool_stickiness": 0.1,
}


def fake_dataset(ids=subject.EXPECTED_IDS):
    episodes = []
    for episode_id in ids:
        episodes.append(SimpleNamespace(
            id=episode_id,
            scored_window=SimpleNamespace(start_frame=1, end_frame=10),
            parameters_for=lambda values: {**values, "tool_retention": 1.0},
        ))
    return SimpleNamespace(episodes=episodes, fingerprint="fingerprint")


def test_requires_exact_ordered_thirteen_chunks():
    subject.validate_episode_ids(fake_dataset())
    with pytest.raises(ValueError, match="exactly"):
        subject.validate_episode_ids(fake_dataset(subject.EXPECTED_IDS[::-1]))
    with pytest.raises(ValueError, match="exactly"):
        subject.validate_episode_ids(fake_dataset(subject.EXPECTED_IDS[:-1]))


def test_parameters_require_all_finite_and_simulator_valid_not_optimizer_bounded():
    assert subject.validate_explicit_parameters(PARAMETERS, fake_dataset()) == PARAMETERS
    # floor_retention 0.99 is outside the optimizer bound used by the dataset but valid for replay.
    with pytest.raises(ValueError, match="eight"):
        subject.validate_explicit_parameters({k: v for k, v in PARAMETERS.items() if k != "viscosity"}, fake_dataset())
    with pytest.raises(ValueError, match="finite"):
        subject.validate_explicit_parameters({**PARAMETERS, "viscosity": float("nan")}, fake_dataset())
    with pytest.raises(ValueError, match="stretch"):
        subject.validate_explicit_parameters({**PARAMETERS, "plastic_min": 1.1}, fake_dataset())


def test_midpoint_excludes_zero_and_prefers_earlier_tie_or_nearest():
    records = [{"source_frame": value} for value in (0, 2, 4, 8)]
    assert subject.choose_midpoint(records, 1, 7)["source_frame"] == 4
    assert subject.choose_midpoint(records, 5, 9)["source_frame"] == 8
    with pytest.raises(ValueError):
        subject.choose_midpoint([{"source_frame": 0}], 1, 2)


def test_local_to_original_ordinal_mapping():
    row = {"start_pointcloud_ordinal": 46, "end_pointcloud_ordinal_exclusive": 65}
    assert subject.original_ordinal(9, row) == 55
    with pytest.raises(ValueError):
        subject.original_ordinal(19, row)


@pytest.mark.parametrize(("encoding", "pixel", "expected"), [
    ("rgb8", [1, 2, 3], [1, 2, 3]), ("bgr8", [1, 2, 3], [3, 2, 1]),
    ("rgba8", [1, 2, 3, 44], [1, 2, 3]), ("bgra8", [1, 2, 3, 44], [3, 2, 1]),
])
def test_decode_rgb_variants_and_padding(encoding, pixel, expected):
    channels = len(pixel); padding = [99, 98]
    message = SimpleNamespace(encoding=encoding, width=1, height=2, step=channels + 2,
                              data=bytes(pixel + padding + pixel + padding))
    image = subject.decode_image(message)
    assert image.shape == (2, 1, 3)
    assert image[0, 0].tolist() == expected


def test_timestamp_matching_prefers_earlier_on_tie_and_exact_replaces_nearest():
    targets = {4: 100}
    candidates = {}
    subject.update_nearest(candidates, targets, 110, "late")
    subject.update_nearest(candidates, targets, 90, "early")
    assert candidates[4] == (10, 90, "early")
    subject.update_nearest(candidates, targets, 100, "exact")
    assert candidates[4] == (0, 100, "exact")


def test_panel_layout_is_four_by_seven_with_last_blank():
    occupied = {subject.panel_position(index, kind) for index in range(13) for kind in ("rgb", "sim")}
    assert len(occupied) == 26
    assert max(row for row, _ in occupied) == 3
    assert max(column for _, column in occupied) == 6
    assert (2, 6) not in occupied and (3, 6) not in occupied


def identity_args(tmp_path: Path):
    return SimpleNamespace(output_dir=tmp_path / "run", resume=False)


def test_fresh_output_and_exact_resume_identity(tmp_path: Path):
    args = identity_args(tmp_path)
    identity = {"dataset_fingerprint": "a", "parameters": PARAMETERS}
    manifest = subject.prepare_root(args, identity)
    assert manifest["identity"] == identity
    with pytest.raises(FileExistsError):
        subject.prepare_root(args, identity)
    args.resume = True
    assert subject.prepare_root(args, identity)["identity"] == identity
    with pytest.raises(ValueError, match="identity"):
        subject.prepare_root(args, {**identity, "dataset_fingerprint": "b"})


def test_thirteen_forward_commands_have_all_explicit_parameters_and_no_calibration(tmp_path: Path):
    python = tmp_path / "python"; python.write_text("#!/bin/sh\n"); python.chmod(0o755)
    args = SimpleNamespace(simulation_python=str(python), render_python=str(python), dataset=tmp_path / "dataset.json",
                           backend="cpu", precision="f32", cpu_threads=1, reference_policy="frozen",
                           output_dir=tmp_path / "output")
    commands = subject.forward_commands(args, PARAMETERS)
    assert len(commands) == 13
    for command, episode_id in zip(commands, subject.EXPECTED_IDS, strict=True):
        assert command[command.index("--episode-id") + 1] == episode_id
        assert "calibrat" not in " ".join(command).lower()
        for flag in subject.PARAMETER_FLAGS.values():
            assert flag in command


def test_snapshot_hashes_detect_change(tmp_path: Path):
    path = tmp_path / "particles.npy"; np.save(path, np.zeros((2, 3)))
    rows = [{"episode_id": "chunk", "snapshot": str(path)}]
    before = subject.snapshot_hashes(rows)
    np.save(path, np.ones((2, 3)))
    assert subject.snapshot_hashes(rows) != before


def test_manifest_json_rejects_nonfinite_provenance(tmp_path: Path):
    path = tmp_path / "manifest.json"
    subject.write_json(path, {"parameters": PARAMETERS, "complete": True})
    assert json.loads(path.read_text())["complete"] is True
    with pytest.raises(ValueError):
        subject.write_json(path, {"bad": float("nan")})

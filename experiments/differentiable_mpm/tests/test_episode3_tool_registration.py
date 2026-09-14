from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.differentiable_mpm.register_episode3_tools import (
    require_identity,
    registration_passes,
    select_training_candidate,
)


def write_identity(path: Path, status="accepted") -> str:
    value = {
        "schema": "taichidough/episode3-tool-identity/v1",
        "status": status,
        "mapping": {"UR5e_spathla": "track-ur", "gen3_spathla": "track-gen"} if status == "accepted" else None,
    }
    path.write_text(json.dumps(value) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_requires_pinned_accepted_distinct_identity(tmp_path):
    path = tmp_path / "identity.json"
    digest = write_identity(path)
    assert require_identity(path, digest)["mapping"]["UR5e_spathla"] == "track-ur"
    with pytest.raises(ValueError, match="SHA-256"):
        require_identity(path, "0" * 64)
    digest = write_identity(path, "identity_unresolved")
    with pytest.raises(ValueError, match="unresolved"):
        require_identity(path, digest)
    digest = write_identity(path)
    value = json.loads(path.read_text())
    value["mapping"]["gen3_spathla"] = "track-ur"
    path.write_text(json.dumps(value) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="distinct"):
        require_identity(path, digest)


def test_seed_selection_uses_training_score_only():
    candidates = [
        {"seed_index": 0, "train_score_mm": 4.0, "heldout_score_mm": 0.1},
        {"seed_index": 1, "train_score_mm": 2.0, "heldout_score_mm": 10.0},
        {"seed_index": 2, "train_score_mm": 3.0, "heldout_score_mm": 1.0},
        {"seed_index": 3, "train_score_mm": 2.5, "heldout_score_mm": 0.5},
    ]
    assert select_training_candidate(candidates)["seed_index"] == 1
    with pytest.raises(ValueError, match="four"):
        select_training_candidate(candidates[:3])


def metrics(median, within, frames=None):
    return {
        "equal_frame_median_distance_mm": median,
        "equal_frame_within_5mm": within,
        "frames": frames or [{"raw_frame": 1, "median_distance_mm": median}],
    }


def test_visual_acceptance_uses_heldout_thresholds():
    accepted, reasons = registration_passes(
        metrics(8.0, 0.3), metrics(4.0, 0.8), [2e-4, 1, 1, 1, 1, 1]
    )
    assert accepted
    assert reasons == []
    accepted, reasons = registration_passes(
        metrics(6.0, 0.4), metrics(4.9, 0.9), [1e-3, 1, 1, 1, 1, 1]
    )
    assert not accepted
    assert "heldout_median_improvement_below_30_percent" in reasons


def test_collision_metrics_cannot_change_visual_rejection():
    before = metrics(8.0, 0.2)
    visual_after = metrics(6.0, 0.70)
    accepted, _ = registration_passes(before, visual_after, [1e-6, 1, 1, 1, 1, 1])
    assert not accepted
    # There is intentionally no collision argument: collision results are reporting only.

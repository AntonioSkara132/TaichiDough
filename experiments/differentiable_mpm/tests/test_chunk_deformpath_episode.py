from __future__ import annotations

import json

import pytest

from scripts.chunk_deformpath_episode import parse_ranges


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def test_ranges_can_intentionally_omit_only_outer_checkpoint_segments(tmp_path):
    path = tmp_path / "ranges.json"
    write_json(path, {"allow_omitted_edges": True, "ranges": [
        {"start": 2, "end": 5},
        {"start": 5, "end": 8},
    ]})
    assert parse_ranges(path, list(range(10)), 10) == [(2, 5, 2, 5), (5, 8, 5, 8)]


def test_ranges_still_reject_middle_holes_when_edge_omission_is_enabled(tmp_path):
    path = tmp_path / "ranges.json"
    write_json(path, {"allow_omitted_edges": True, "ranges": [
        {"start": 2, "end": 5},
        {"start": 6, "end": 8},
    ]})
    with pytest.raises(ValueError, match="contiguous half-open"):
        parse_ranges(path, list(range(10)), 10)

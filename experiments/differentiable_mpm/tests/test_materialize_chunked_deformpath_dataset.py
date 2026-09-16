from __future__ import annotations

import pytest

from scripts.materialize_chunked_deformpath_dataset import validation_chunk_names


def rows(*names):
    return [{"directory": name} for name in names]


def test_validation_chunk_names_accepts_repeated_and_comma_separated_names():
    selected = validation_chunk_names(rows("chunk01", "chunk02", "chunk03"),
                                      ["chunk01,chunk03", "chunk03"])
    assert selected == {"chunk01", "chunk03"}


def test_validation_chunk_names_rejects_unknown_or_all_chunks():
    with pytest.raises(ValueError, match="Unknown validation chunks"):
        validation_chunk_names(rows("chunk01"), ["chunk02"])
    with pytest.raises(ValueError, match="At least one chunk"):
        validation_chunk_names(rows("chunk01", "chunk02"), ["chunk01,chunk02"])

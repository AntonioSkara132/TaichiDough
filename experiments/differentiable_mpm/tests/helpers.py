"""Portable, private temporary directories for experiment tests."""
import os
from pathlib import Path
import tempfile


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def scratch_root():
    """Create scratch storage only when a test requests it."""
    job = os.environ.get("CLAUDE_JOB_DIR")
    root = Path(job).expanduser().resolve() / "tmp" if job else EXPERIMENT_ROOT / "runs" / "test_tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def temporary_directory(prefix="differentiable-mpm-test-"):
    """Return a TemporaryDirectory with a unique child name in selected scratch."""
    return tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_root())

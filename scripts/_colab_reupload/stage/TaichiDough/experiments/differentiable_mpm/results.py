"""Experiment-owned run records with explicit identity checks and atomic snapshots."""
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
RUN_ROOT = EXPERIMENT_ROOT / "runs"
SCHEMA = "taichidough-differentiable-run-v1"


def json_value(value) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_identity():
    sources = {}
    for path in sorted(EXPERIMENT_ROOT.rglob("*.py")):
        if "runs" not in path.relative_to(EXPERIMENT_ROOT).parts and "__pycache__" not in path.parts:
            sources[str(path.relative_to(EXPERIMENT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for package in ("taichi", "numpy", "scipy", "torch", "scikit-image", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"sources": sources, "dependencies": versions, "python": sys.version}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunStore(AbstractContextManager):
    """Only resume a run whose complete supplied identity matches this execution.

    The OS lock is released on process exit, including a crash. The lock file stays
    in the run directory. Historical result directories are never deleted.
    """

    def __init__(self, path, identity, *, resume=False, allowed_root=RUN_ROOT):
        self.path = Path(path).resolve()
        allowed = Path(allowed_root).resolve()
        if self.path == allowed or not self.path.is_relative_to(allowed):
            raise ValueError(f"Run directory must be a child of {allowed}")
        self.identity = json_value(identity)
        self.identity_hash = canonical_hash(self.identity)
        self.lock = None
        manifest_path = self.path / "run_manifest.json"
        if resume:
            if not manifest_path.is_file():
                raise ValueError("Resume requires this experiment's existing run_manifest.json")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema") != SCHEMA or manifest.get("identity_sha256") != self.identity_hash:
                raise ValueError("Run identity differs; create a new run instead of reusing old results")
            if canonical_hash(manifest.get("identity")) != self.identity_hash:
                raise ValueError("Stored run identity has been modified or is corrupt")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.mkdir(exist_ok=False)
        try:
            self.lock = (self.path / "run.lock").open("a+")
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not resume:
                self.write_json("run_manifest.json", {
                    "schema": SCHEMA, "created_at": utc_now(),
                    "identity_sha256": self.identity_hash, "identity": self.identity,
                })
            self.append_event({"event": "resume" if resume else "start", "pid": os.getpid()})
        except BaseException:
            if self.lock is not None:
                self.lock.close()
                self.lock = None
            raise

    def write_json(self, name, data):
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Result snapshots must be local JSON filenames")
        content = json.dumps(json_value(data), sort_keys=True, indent=2, allow_nan=False) + "\n"
        target = self.path / name
        temporary = self.path / ("." + name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    def read_json(self, name):
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Result snapshots must be local JSON filenames")
        return json.loads((self.path / name).read_text())

    def append_event(self, event):
        record = {"time": utc_now(), **json_value(event)}
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with (self.path / "events.jsonl").open("a") as stream:
            stream.write(line)
            stream.flush()

    def save_optimizer(self, event, state):
        self.append_event(event)
        self.write_json("optimizer_state.json", {
            "identity_sha256": self.identity_hash, "saved_at": utc_now(), "state": state,
        })

    def optimizer_state(self):
        record = self.read_json("optimizer_state.json")
        if record.get("identity_sha256") != self.identity_hash:
            raise ValueError("Optimizer state does not belong to the verified run identity")
        return record["state"]

    def __exit__(self, exc_type, exc_value, traceback):
        if self.lock is not None:
            if exc_value is not None:
                self.append_event({"event": "failure", "exception": type(exc_value).__name__,
                                   "message": str(exc_value)})
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None
        return False

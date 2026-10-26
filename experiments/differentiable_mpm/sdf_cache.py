"""Content-addressed tool SDF files shared by independent Linux workers."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


ARRAY_NAMES = ("distances", "gradients", "minimums", "spacings")
SCHEMA = "taichidough/tool-sdf-cache/v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(_json(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _pin_versions(value, numpy_version, open3d_version=None):
    """bag0 worktree addition: report the numpy / Open3D versions the shipped cache was built with (version strings
    only label the identity; the arrays themselves are checksum-verified on load)."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "numpy_version": out[k] = numpy_version
            elif k == "open3d_version" and open3d_version: out[k] = open3d_version
            else: out[k] = _pin_versions(v, numpy_version, open3d_version)
        return out
    if isinstance(value, list):
        return [_pin_versions(v, numpy_version, open3d_version) for v in value]
    return value


def cache_identity(vertices, resolution, padding, source):
    """Include actual transformed vertices, builder source and numerical settings."""
    description = {"schema": SCHEMA, "source": source, "numpy_version": np.__version__,
                   "vertices_sha256": _array_hash(vertices), "resolution": int(resolution),
                   "padding": float(padding)}
    pinned = os.environ.get("TAICHIDOUGH_SDF_CACHE_NUMPY_VERSION")
    if pinned:
        description = _pin_versions(description, pinned, os.environ.get("TAICHIDOUGH_SDF_CACHE_OPEN3D_VERSION"))
    return hashlib.sha256(_json(description).encode()).hexdigest(), description


def _validate(arrays, resolution):
    expected = ((resolution,) * 3, (resolution,) * 3 + (3,), (3,), (3,))
    if len(arrays) != 4:
        raise ValueError("Require four SDF arrays")
    for name, value, dimensions in zip(ARRAY_NAMES, arrays, expected):
        if value.shape != dimensions or value.dtype != np.dtype("float32"):
            raise ValueError(f"Invalid cached SDF dimensions or dtype: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite cached SDF: {name}")
    if not np.any(arrays[0] < 0) or not np.all(arrays[3] > 0):
        raise ValueError("Cached SDF must have interior voxels and positive spacing")


def _read(path, description):
    try:
        with path.open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAY_NAMES) | {"metadata"}:
                raise ValueError("Unexpected SDF cache entries")
            metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
            if metadata["inputs"] != description:
                raise ValueError("SDF cache input identity differs")
            arrays = tuple(archive[name].copy() for name in ARRAY_NAMES)
        _validate(arrays, description["resolution"])
        actual = {name: _array_hash(array) for name, array in zip(ARRAY_NAMES, arrays)}
        if metadata["arrays"] != actual:
            raise ValueError("SDF cache array checksum differs")
        return arrays
    except Exception as error:
        raise ValueError(f"Invalid SDF cache file {path}: {error}") from error


def cached_mesh_sdf(cache_dir, vertices, resolution, padding, source, builder, *, read_only=False):
    """Build once under a process lock; publish a complete, checked file atomically.

    Invalid existing files are rejected rather than silently replaced. Read-only
    workers require a prebuilt entry and never invoke the builder.
    """
    key, description = cache_identity(vertices, resolution, padding, source)
    directory = Path(cache_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.npz"
    with (directory / f"{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            arrays = _read(path, description)
            print(f"SDF cache hit: {key}", flush=True)
            return arrays
        if read_only:
            raise FileNotFoundError(f"Prebuilt SDF cache entry missing: {path}")
        arrays = tuple(builder(vertices, resolution, padding))
        _validate(arrays, int(resolution))
        metadata = {"inputs": description,
                    "arrays": {name: _array_hash(array) for name, array in zip(ARRAY_NAMES, arrays)}}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{key}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, **dict(zip(ARRAY_NAMES, arrays)),
                         metadata=np.frombuffer(_json(metadata).encode(), dtype=np.uint8))
                stream.flush()
                os.fsync(stream.fileno())
            checked = _read(temporary, description)
            os.replace(temporary, path)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"SDF cache built: {key}", flush=True)
        return checked

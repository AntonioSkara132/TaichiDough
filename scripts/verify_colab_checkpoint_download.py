"""Verify checkpoint files downloaded independently from Google Drive to the local host.

The download directory is untrusted. Run this trusted script with Python -I and
pass the directory as an argument; do not execute code from the download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON token: {value}")


def _read(path):
    return json.loads(path.read_bytes(), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    args = parser.parse_args(argv)
    root = args.download_dir.resolve()
    checkpoint_path = root / "optimizer_state.json"
    marker_path = root / "checkpoint_sha256.json"
    if not checkpoint_path.is_file() or not marker_path.is_file():
        parser.error("Both independently downloaded files must be present")
    actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = _read(checkpoint_path)
    marker = _read(marker_path)
    iteration = checkpoint.get("state", {}).get("iterations")
    if (checkpoint.get("schema") != "taichidough/minibatch-adam-continuation/v1"
            or type(iteration) is not int or iteration != args.expected_iteration
            or checkpoint.get("state", {}).get("accepted_updates") != iteration
            or checkpoint.get("identity_sha256") != args.expected_identity
            or marker.get("identity_sha256") != args.expected_identity
            or marker.get("iterations") != iteration
            or marker.get("optimizer_state_sha256") != actual_hash):
        raise ValueError("Downloaded checkpoint, digest marker, update count or run identity differs")
    vectors = (checkpoint["state"].get(name) for name in
               ("coordinates", "first_moment", "second_moment"))
    if any(not isinstance(vector, list) or len(vector) != 8
           or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                  for x in vector) for vector in vectors):
        raise ValueError("Downloaded optimizer vectors are not finite eight-parameter states")
    if len(checkpoint["state"].get("history", [])) != iteration:
        raise ValueError("Downloaded optimizer history has the wrong length")
    print(f"Verified independent Drive download: update {iteration}, "
          f"identity {args.expected_identity}, optimizer_state SHA-256 {actual_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

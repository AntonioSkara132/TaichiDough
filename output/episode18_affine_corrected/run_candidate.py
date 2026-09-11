#!/usr/bin/env python3
"""Run one fingerprinted calibration candidate, optionally over a shorter smoke window."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# pyright: reportMissingImports=false
from calibrate_youngs_modulus import _fingerprint_record, execute_candidate_window, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--youngs-modulus", type=float, default=2000.0)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    path = args.manifest.resolve()
    manifest = validate_manifest(json.loads(path.read_text()), path)
    window = next(dict(item) for item in manifest["windows"] if item["split"] == args.split)
    if args.end_frame is not None:
        if not window["start_frame"] < args.end_frame <= window["end_frame"]:
            raise ValueError("Smoke end frame must lie after the scoring start and within the original window")
        window["end_frame"] = args.end_frame
        window["name"] += f"-smoke-to-{args.end_frame}"
    started = time.monotonic()
    result = execute_candidate_window(manifest, _fingerprint_record(manifest), window, args.youngs_modulus)
    result["total_wall_s"] = time.monotonic() - started
    summary = {
        key: result.get(key)
        for key in ("status", "window", "split", "youngs_modulus_pa", "cache_dir", "failure_reason", "total_wall_s")
    }
    summary["weighted_total"] = (result.get("loss") or {}).get("weighted_total")
    filename = f"candidate_e{args.youngs_modulus:g}_{window['name']}.json"
    result_path = path.parent / filename
    if result_path.exists():
        raise ValueError(f"Refusing to overwrite existing candidate report: {result_path}")
    with result_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

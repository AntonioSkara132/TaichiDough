"""Prepare a fresh mocap-coordinate Episode20 export without replacing existing data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import torch

from experiments.differentiable_mpm.table_alignment import compose_table_calibration


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
DATA_TOOLS = WORKSPACE / "data" / "deformpath_training"
DEFAULT_BAG = DATA_TOOLS / "DeformPath2" / "snimanje_23_10" / "episode20_kugla"
DEFAULT_OUTPUT = DATA_TOOLS / "DeformPath3" / "snimanje_23_10" / "episode20_kugla"
REFERENCE_EXPORT = DATA_TOOLS / "DeformPath3" / "snimanje_23_10" / "episode18_kugla"
EXPORTER = DATA_TOOLS / "export_deformpath2_offline.py"
INTERPOLATOR = DATA_TOOLS / "interpolate_deformpath_sequence.py"
TABLE_ESTIMATOR = Path(__file__).with_name("estimate_episode20_table_plane.py")
TOOL_GEOMETRY = Path(__file__).parent / "data" / "episode18_registered_tools_v1" / "tool_geometry.json"
POSE_FRAMES = ("UR5e_spathla", "gen3_spathla")
REQUIRED_TAG_PARENT = "tag16h5:3"
REQUIRED_TAG_CHILD = "tag3_real"
OBSERVED_TAG_TOKENS = ("tag16h5:3", "tag16h5:4")
SPACE_RESERVE_BYTES = 1 << 30
SCHEMA = "taichidough/episode20-mocap-export-validation/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def disk_preflight(parent: Path, reference: Path, reserve_bytes: int = SPACE_RESERVE_BYTES) -> dict[str, int | bool]:
    if reserve_bytes < 0:
        raise ValueError("reserve_bytes must be non-negative")
    reference_bytes = tree_size(reference)
    required_bytes = reference_bytes + reserve_bytes
    available_bytes = shutil.disk_usage(parent).free
    result: dict[str, int | bool] = {
        "reference_export_bytes": reference_bytes,
        "reserve_bytes": reserve_bytes,
        "required_available_bytes": required_bytes,
        "available_bytes": available_bytes,
        "passed": available_bytes >= required_bytes,
    }
    if not result["passed"]:
        missing = required_bytes - available_bytes
        raise OSError(
            f"Insufficient free space for a fresh Episode20 export: available={available_bytes}, "
            f"required={required_bytes}, additional_required={missing} bytes"
        )
    return result


def scan_tokens(paths: Iterable[Path], tokens: Iterable[str]) -> dict[str, bool]:
    encoded = {token: token.encode("utf-8") for token in tokens}
    found = {token: False for token in encoded}
    overlap = max((len(value) for value in encoded.values()), default=1) - 1
    for path in paths:
        previous = b""
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                data = previous + block
                for token, value in encoded.items():
                    found[token] = found[token] or value in data
                if all(found.values()):
                    return found
                previous = data[-overlap:] if overlap else b""
    return found


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def torch_load(path: Path, *, mmap: bool = False) -> Any:
    options: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    if mmap:
        options["mmap"] = True
    try:
        return torch.load(path, **options)
    except TypeError:
        options.pop("weights_only", None)
        options.pop("mmap", None)
        return torch.load(path, **options)


def finite_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.isfinite().all().item():
        raise ValueError(f"{label} must be a finite tensor")
    return value


def calibration_fingerprint(document: dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_table_aligned_calibration(staging: Path) -> None:
    source_path = staging / "scene_calibration_v2.json"
    source = json_object(source_path)
    report_path = staging / "episode20_table_plane_validation.json"
    report = json_object(report_path)
    source_hash = sha256_file(source_path)
    if report.get("passed") is not True:
        raise ValueError("Independent Episode20 table-plane validation failed")
    if report.get("source_mocap_calibration_sha256") != source_hash:
        raise ValueError("Table-plane report belongs to another mocap calibration")
    plane = report.get("estimated_plane_scene")
    translation_xz = report.get("recommended_translation_xz_m")
    derived = compose_table_calibration(
        source,
        plane,
        translation_xz=translation_xz,
        scene_frame="table-aligned",
        provenance={
            "table_plane_report": "episode20_table_plane_validation.json",
            "table_plane_report_sha256_before_final_calibration": sha256_file(report_path),
            "source_mocap_calibration": "scene_calibration_mocap_v2.json",
            "source_mocap_calibration_sha256": source_hash,
        },
    )
    raw_path = staging / "scene_calibration_mocap_v2.json"
    os.rename(source_path, raw_path)
    source_path.write_text(json.dumps(derived, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report.update(
        calibration_sha256=sha256_file(source_path),
        final_scene_frame="table-aligned",
        final_scene_from_source=derived["scene_from_source"],
        final_floor_plane_scene=derived["floor_plane_scene"],
    )
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_calibration(calibration: dict[str, Any], conversion: dict[str, Any]) -> dict[str, Any]:
    if calibration.get("schema") != "taichidough/scene-calibration/v2":
        raise ValueError("Episode20 export must contain a v2 scene calibration")
    if calibration.get("source_frame") != "mocap" or calibration.get("scene_frame") != "table-aligned":
        raise ValueError("Episode20 calibration must map mocap inputs into table-aligned coordinates")
    transform = np.asarray(calibration.get("scene_from_source"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Episode20 table alignment must be a finite 4x4 transform")
    if np.allclose(transform, np.eye(4), rtol=0, atol=1e-12):
        raise ValueError("Episode20 table alignment must record the measured nonidentity transform")
    floor = np.asarray(calibration.get("floor_plane_scene"), dtype=np.float64)
    if floor.shape != (4,) or not np.isfinite(floor).all() or not np.allclose(floor, [0, 1, 0, 0], rtol=0, atol=1e-12):
        raise ValueError("Mocap export requires the declared mocap table plane")
    provenance = calibration.get("provenance", {})
    source_provenance = provenance.get("source_calibration_provenance", {})
    if source_provenance.get("camera_tag_parent_frame") != REQUIRED_TAG_PARENT or source_provenance.get("camera_tag_frame") != REQUIRED_TAG_CHILD:
        raise ValueError("Calibration uses the wrong camera AprilTag parent or child")
    alignment = provenance.get("table_alignment", {})
    if alignment.get("previous_scene_frame") != "mocap" or alignment.get("source_tensors_transformed") is not False:
        raise ValueError("Episode20 calibration lacks the expected rigid table-alignment provenance")
    record = conversion.get("calibration", {})
    if record.get("status") != "available" or record.get("source_frame") != "mocap" or record.get("scene_frame") != "table-aligned":
        raise ValueError("Conversion metadata does not declare the table-aligned calibration")
    diagnostics = calibration.get("diagnostics", {}).get("mocap_from_camera_aggregation", {})
    inliers = diagnostics.get("inlier_count")
    samples = diagnostics.get("sample_count")
    if not isinstance(inliers, int) or not isinstance(samples, int) or inliers < 10 or samples < inliers:
        raise ValueError("Moving-AprilTag calibration has insufficient inliers")
    translation = np.asarray(diagnostics.get("translation_residual_m", []), dtype=np.float64)
    rotation = np.asarray(diagnostics.get("rotation_residual_deg", []), dtype=np.float64)
    if translation.size != samples or rotation.size != samples or not np.isfinite(translation).all() or not np.isfinite(rotation).all():
        raise ValueError("Moving-AprilTag calibration residuals are incomplete or nonfinite")
    return {
        "sample_count": samples,
        "inlier_count": inliers,
        "translation_limit_m": float(diagnostics["translation_limit_m"]),
        "rotation_limit_deg": float(diagnostics["rotation_limit_deg"]),
        "maximum_translation_residual_m": float(translation.max()),
        "maximum_rotation_residual_deg": float(rotation.max()),
    }


def validate_tensors(directory: Path, conversion: dict[str, Any], sequence: dict[str, Any]) -> dict[str, Any]:
    raw_clouds = torch_load(directory / "pointclouds.pt", mmap=True)
    if not (isinstance(raw_clouds, list) and len(raw_clouds) == 1 and isinstance(raw_clouds[0], list)):
        raise ValueError("Raw pointcloud payload must contain one episode list")
    raw_frames = raw_clouds[0]
    if len(raw_frames) != 250:
        raise ValueError(f"Episode20 raw export must contain 250 pointcloud frames, got {len(raw_frames)}")
    for index, frame in enumerate(raw_frames):
        tensor = finite_tensor(frame, f"raw pointcloud frame {index}")
        if tensor.ndim != 2 or tensor.shape[1] != 10:
            raise ValueError("Raw pointcloud frames must use [N,10] layout")

    raw_paths = torch_load(directory / "paths.pt")
    if not isinstance(raw_paths, dict) or raw_paths.get("pose_frames") != list(POSE_FRAMES):
        raise ValueError("Raw path payload must name the two expected tool streams")
    streams = raw_paths.get("streams")
    if not isinstance(streams, dict) or set(streams) != set(POSE_FRAMES):
        raise ValueError("Raw path payload is missing an expected tool stream")
    for name in POSE_FRAMES:
        tensor = finite_tensor(streams[name], f"raw path stream {name}")
        if tensor.ndim != 2 or tensor.shape[1] != 14 or len(tensor) < 2:
            raise ValueError("Raw path streams must use nonempty [T,14] layout")

    processed_clouds = torch_load(directory / "pointclouds_interpolated.pt")
    processed_paths = torch_load(directory / "paths_interpolated.pt")
    if not (isinstance(processed_clouds, list) and len(processed_clouds) == 1 and isinstance(processed_clouds[0], list)):
        raise ValueError("Interpolated pointcloud payload must contain one episode list")
    frames = processed_clouds[0]
    if not (isinstance(processed_paths, list) and len(processed_paths) == 1 and isinstance(processed_paths[0], dict)):
        raise ValueError("Interpolated path payload must contain one episode dictionary")
    paths = processed_paths[0]
    path = finite_tensor(paths.get("path"), "interpolated paths")
    validity = paths.get("stream_validity")
    if not isinstance(validity, torch.Tensor) or tuple(validity.shape) != (len(frames), 2) or not validity.bool().all().item():
        raise ValueError("Every retained Episode20 sample must contain both tool streams")
    if paths.get("pose_frames") != list(POSE_FRAMES) or tuple(path.shape) != (len(frames), 2, 14):
        raise ValueError("Interpolated paths must use [T,2,14] with the expected stream order")
    for index, frame in enumerate(frames):
        tensor = finite_tensor(frame, f"interpolated pointcloud frame {index}")
        if tensor.ndim != 2 or tensor.shape[1] != 7:
            raise ValueError("Interpolated pointcloud frames must use [N,7] layout")
    times = path[:, :, 13].detach().cpu().numpy().astype(np.float64)
    if not np.allclose(times[:, 0], times[:, 1], rtol=0, atol=1e-6) or np.any(np.diff(times[:, 0]) <= 0):
        raise ValueError("Interpolated tool timestamps must agree and increase strictly")
    indices = sequence.get("pointcloud_indices")
    if not isinstance(indices, list) or len(indices) != len(frames) or any(type(value) is not int for value in indices):
        raise ValueError("sequence_metadata pointcloud_indices do not match interpolated frames")
    if sequence.get("pose_frames") != list(POSE_FRAMES) or sequence.get("calibration", {}).get("source_frame") != "mocap":
        raise ValueError("sequence_metadata does not preserve the mocap frame and tool order")

    details = conversion.get("episode", {}).get("pointcloud_transform_details")
    if not isinstance(details, list) or len(details) != 250:
        raise ValueError("Conversion metadata must record all pointcloud transforms")
    source_frames: list[str] = []
    for row in details:
        value = row.get("source_frame")
        if not isinstance(value, str) or not value:
            raise ValueError("Raw PointCloud2 header.frame_id must be constant and recorded")
        if value not in source_frames:
            source_frames.append(value)
    source_frames.sort()
    if len(source_frames) != 1:
        raise ValueError("Raw PointCloud2 header.frame_id must be constant and recorded")
    if any(row.get("output_frame") != "mocap" for row in details):
        raise ValueError("Every pointcloud transform must output mocap coordinates")
    return {
        "raw_pointcloud_frames": len(raw_frames),
        "interpolated_frames": len(frames),
        "raw_pointcloud_header_frame_ids": source_frames,
        "pose_frames": list(POSE_FRAMES),
        "all_interpolated_streams_valid": True,
        "time_range_s": [float(times[0, 0]), float(times[-1, 0])],
    }


def replace_paths(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: replace_paths(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_paths(item, old, new) for item in value]
    if isinstance(value, str) and value.startswith(old):
        return new + value[len(old):]
    return value


def rewrite_generated_metadata(staging: Path, destination: Path) -> None:
    old, new = str(staging.resolve()), str(destination.resolve())
    calibration_path = staging / "scene_calibration_v2.json"
    calibration_document = json_object(calibration_path)
    calibration_record = {
        "status": "available",
        "path": str(destination / "scene_calibration_v2.json"),
        "sha256": sha256_file(calibration_path),
        "fingerprint": calibration_fingerprint(calibration_document),
        "source_frame": calibration_document["source_frame"],
        "scene_frame": calibration_document["scene_frame"],
        "floor_plane_scene": calibration_document["floor_plane_scene"],
    }
    conversion_path = staging / "conversion_metadata.json"
    conversion = replace_paths(json_object(conversion_path), old, new)
    conversion["calibration"] = dict(calibration_record)
    conversion_path.write_text(json.dumps(conversion, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    conversion_hash = sha256_file(conversion_path)
    sequence_path = staging / "sequence_metadata.json"
    sequence = replace_paths(json_object(sequence_path), old, new)
    sequence["calibration"] = {
        **calibration_record,
        "source_conversion_metadata_path": str(destination / "conversion_metadata.json"),
        "source_conversion_metadata_sha256": conversion_hash,
    }
    sequence_path.write_text(json.dumps(sequence, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with status {completed.returncode}; see {log_path}")


def build_commands(bag: Path, staging: Path, tool_geometry: Path) -> list[list[str]]:
    return [
        [
            sys.executable, str(EXPORTER), "--bag-dir", str(bag), "--output-dir", str(staging),
            "--output-frame", "mocap", "--pose-reference-frame", "pointcloud",
            "--pose-frames", *POSE_FRAMES, "--max-points", "0",
            "--camera-tag-parent-frame", REQUIRED_TAG_PARENT, "--camera-tag-frame", REQUIRED_TAG_CHILD,
            "--mocap-tag-frame", "apriltag3", "--mocap-frame", "mocap", "--camera-frame", "camera_link",
        ],
        [
            sys.executable, str(INTERPOLATOR), "--input-dir", str(staging), "--output-dir", str(staging),
            "--timeline", "pointcloud", "--paths-format", "episode-tensor", "--require-all-streams",
            "--max-points", "4096", "--seed", "0",
        ],
        [
            sys.executable, str(TABLE_ESTIMATOR), "--episode", str(staging),
            "--calibration", str(staging / "scene_calibration_v2.json"),
            "--tool-geometry", str(tool_geometry),
            "--output", str(staging / "episode20_table_plane_validation.json"),
        ],
    ]


def source_files(bag: Path, tool_geometry: Path) -> list[Path]:
    databases = sorted(bag.glob("*.db3"))
    files = [bag / "metadata.yaml", *databases, EXPORTER, INTERPOLATOR, TABLE_ESTIMATOR, tool_geometry]
    missing = [path for path in files if not path.is_file()]
    if missing or not databases:
        raise FileNotFoundError("Missing Episode20 preparation input: " + ", ".join(map(str, missing or [bag / "*.db3"])))
    return files


def prepare(bag: Path, output: Path, *, reference: Path = REFERENCE_EXPORT,
            tool_geometry: Path = TOOL_GEOMETRY, reserve_bytes: int = SPACE_RESERVE_BYTES) -> Path:
    bag, output, reference, tool_geometry = (Path(value).expanduser().resolve() for value in (bag, output, reference, tool_geometry))
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing Episode20 export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not reference.is_dir():
        raise FileNotFoundError(f"Reference export does not exist: {reference}")
    inputs = source_files(bag, tool_geometry)
    input_hashes = {str(path): sha256_file(path) for path in inputs}
    database_paths = [path for path in inputs if path.suffix == ".db3"]
    observed_tags = scan_tokens(database_paths, OBSERVED_TAG_TOKENS)
    if not observed_tags[REQUIRED_TAG_PARENT]:
        raise ValueError(f"Raw bag does not contain required parent frame {REQUIRED_TAG_PARENT}")
    space = disk_preflight(output.parent, reference, reserve_bytes)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        commands = build_commands(bag, staging, tool_geometry)
        for index, command in enumerate(commands, 1):
            run_command(command, REPO, staging / f"stage_{index}.log")
        derive_table_aligned_calibration(staging)
        rewrite_generated_metadata(staging, output)
        conversion = json_object(staging / "conversion_metadata.json")
        sequence = json_object(staging / "sequence_metadata.json")
        calibration = json_object(staging / "scene_calibration_v2.json")
        table = json_object(staging / "episode20_table_plane_validation.json")
        calibration_checks = validate_calibration(calibration, conversion)
        tensor_checks = validate_tensors(staging, conversion, sequence)
        if table.get("passed") is not True:
            raise ValueError("Independent Episode20 table-plane validation failed")
        if table.get("calibration_sha256") != sha256_file(staging / "scene_calibration_v2.json"):
            raise ValueError("Table-plane report does not name the derived Episode20 calibration")
        input_hashes_end = {str(path): sha256_file(path) for path in inputs}
        if input_hashes_end != input_hashes:
            raise RuntimeError("An Episode20 preparation input changed during export")
        generated_names = (
            "pointclouds.pt", "paths.pt", "conversion_metadata.json", "scene_calibration_v2.json",
            "scene_calibration_mocap_v2.json", "pointclouds_interpolated.pt", "paths_interpolated.pt", "sequence_metadata.json",
            "episode20_table_plane_validation.json", "stage_1.log", "stage_2.log", "stage_3.log",
        )
        generated_hashes = {name: sha256_file(staging / name) for name in generated_names}
        validation = {
            "schema": SCHEMA,
            "passed": True,
            "source_bag": str(bag),
            "destination": str(output),
            "tag_frames_observed_in_bag": observed_tags,
            "selected_tag_parent": REQUIRED_TAG_PARENT,
            "selected_tag_child": REQUIRED_TAG_CHILD,
            "disk_preflight": space,
            "calibration": calibration_checks,
            "tensors": tensor_checks,
            "table_plane_report": {
                "path": str(output / "episode20_table_plane_validation.json"),
                "sha256": generated_hashes["episode20_table_plane_validation.json"],
                "passed": True,
            },
            "input_sha256_start": input_hashes,
            "input_sha256_end": input_hashes_end,
            "generated_sha256": generated_hashes,
            "commands": commands,
            "scene_calibration_estimated": True,
            "material_calibration_optimization_run": False,
            "fit_run": False,
        }
        validation_path = staging / "episode20_export_validation.json"
        validation_path.write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest_names = [*generated_names, validation_path.name]
        manifest = "".join(f"{sha256_file(staging / name)}  {name}\n" for name in sorted(manifest_names))
        (staging / "SHA256SUMS").write_text(manifest, encoding="utf-8")
        os.rename(staging, output)
        return output
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-export", type=Path, default=REFERENCE_EXPORT)
    parser.add_argument("--tool-geometry", type=Path, default=TOOL_GEOMETRY)
    parser.add_argument("--space-reserve-bytes", type=int, default=SPACE_RESERVE_BYTES)
    args = parser.parse_args(argv)
    result = prepare(args.bag, args.output, reference=args.reference_export,
                     tool_geometry=args.tool_geometry, reserve_bytes=args.space_reserve_bytes)
    print(result)
    print("Prepared mocap-coordinate Episode20 inputs; no material calibration was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

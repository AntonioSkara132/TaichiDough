"""Verified optional observations for the paper-inspired loss modes.

Each archive has a separate JSON metadata file. Both files need expected SHA256
entries in ExperimentConfig. NPZ files are numeric arrays loaded without pickle.

Common metadata fields:
  schema: taichidough/loss-{point-targets,tracks,masks}/v1
  units: m (coordinates); pixel (segmentation masks)
  coordinate_frame: scene | camera_optical (coordinates); image (masks)
  timestamp_reference: sequence
  sequence_fingerprint, calibration_sha256: exact prepared-input identities
  provenance: {kind: observed | inferred_volume | synthetic_fixture,
               description: nonempty text, ...optional source details...}

Point targets additionally declare target_representation as partial_observed,
 full_observed, or inferred_volume. NPZ: frame_indices[F], timestamps[F], offsets[F+1],
 points[N,3]. DPSI PRT modes require volume; other modes require a cloud.
Tracks additionally declare initial_particles_sha256. NPZ: frame_indices[F],
 timestamps[F], track_ids[K], particle_ids[K], positions[F,K,3], valid[F,K].
Particle IDs are a fixed mapping to the exact initial particle file, not nearest
neighbors reassigned at each frame. Invalid track coordinates may be nonfinite;
returned coordinates for invalid entries are zero and the validity is retained.
Masks additionally declare camera_fingerprint, the canonical JSON SHA256 of
camera.as_dict(). NPZ: frame_indices[F], timestamps[F], foreground[F,H,W],
known[F,H,W]. Unknown pixels are neither foreground nor background evidence.

All arrays have explicit processed-frame indices and recorded sequence times in
seconds. Every scored frame must be present. Extra frames are allowed and still
validated. Time matching uses absolute tolerance 1e-6 s and no relative tolerance.
Returned coordinate arrays are immutable scene-coordinate float64 copies.
The loader does not reconstruct geometry, infer tracking, or initialize Taichi.
"""
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Mapping
import zipfile

import numpy as np


TARGET_PATH_PAIRS = {
    "points": ("loss_point_targets", "loss_point_targets_metadata"),
    "tracks": ("loss_tracks", "loss_tracks_metadata"),
    "masks": ("loss_masks", "loss_masks_metadata"),
}
TARGET_PATH_NAMES = tuple(name for pair in TARGET_PATH_PAIRS.values() for name in pair)
TARGET_SCHEMAS = {
    "points": "taichidough/loss-point-targets/v1",
    "tracks": "taichidough/loss-tracks/v1",
    "masks": "taichidough/loss-masks/v1",
}
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
TIMESTAMP_ATOL_S = 1e-6


@dataclass(frozen=True)
class TargetBundle:
    by_frame: dict
    provenance: dict


def camera_fingerprint(camera):
    """Camera identity used by mask metadata, including image size and transform."""
    return hashlib.sha256(json.dumps(camera.as_dict(), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha256(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in value)):
        raise ValueError(f"{label} must be a SHA-256 hexadecimal string")
    return value.lower()


def _verified_bytes(config, name, max_bytes):
    expected = _sha256(config.expected_sha256.get(name), f"Expected hash for {name}")
    path = Path(config.paths[name])
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{name} exceeds its {max_bytes}-byte input limit")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    return data, {"sha256": actual, "size_bytes": len(data)}


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON constant {value}")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key}")
        result[key] = value
    return result


def _read_pair(config, kind):
    archive_name, metadata_name = TARGET_PATH_PAIRS[kind]
    archive_bytes, archive_record = _verified_bytes(config, archive_name, MAX_ARCHIVE_BYTES)
    metadata_bytes, metadata_record = _verified_bytes(config, metadata_name, MAX_METADATA_BYTES)
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"), parse_constant=_reject_json_constant,
                              object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {metadata_name} JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_name} must contain a JSON object")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zipped:
            members = zipped.infolist()
            names = [entry.filename for entry in members]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate array entries in {archive_name}")
            if (not members or any(not name.endswith(".npy") or "/" in name or "\\" in name for name in names)):
                raise ValueError(f"{archive_name} must contain only named NPY arrays")
            if sum(entry.file_size for entry in members) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(f"{archive_name} exceeds the uncompressed input limit")
            for entry in members:
                with zipped.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        dimensions, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        dimensions, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"Unsupported NPY version in {archive_name}")
                    if dtype.hasobject:
                        raise ValueError("Object arrays cannot be loaded when allow_pickle=False")
                    if dtype.kind not in "buif" or any(size < 0 for size in dimensions):
                        raise ValueError(f"{archive_name} arrays must be real numeric or boolean")
                    if math.prod(dimensions) * dtype.itemsize != entry.file_size - stream.tell():
                        raise ValueError(f"NPY dimensions do not match stored bytes in {archive_name}")
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, zipfile.BadZipFile, EOFError, TypeError) as error:
        raise ValueError(f"Invalid {archive_name} numeric NPZ archive: {error}") from error
    if any(array.dtype.kind not in "buif" for array in arrays.values()):
        raise ValueError(f"{archive_name} arrays must be real numeric or boolean")
    return metadata, arrays, {"archive": archive_record, "metadata": metadata_record}


def _integer_vector(value, label, *, unique=False):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or not len(array):
        raise ValueError(f"{label} must be a nonempty integer vector")
    if np.any(array < 0) or np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{label} contains out-of-range integer values")
    result = array.astype(np.int64)
    if unique and len(np.unique(result)) != len(result):
        raise ValueError(f"{label} contains duplicate IDs")
    return result


def _numeric(value, label):
    array = np.asarray(value)
    if array.dtype.kind not in "uif":
        raise ValueError(f"{label} must contain real numeric values")
    return array.astype(np.float64)


def _boolean(value, dimensions, label):
    array = np.asarray(value)
    if array.shape != dimensions or array.dtype.kind not in "bui" or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{label} must have dimensions {dimensions} and boolean or integer 0/1 values")
    return array.astype(bool)


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _metadata(metadata, kind, sequence_fingerprint, calibration_sha256):
    common = {"schema", "units", "coordinate_frame", "timestamp_reference", "sequence_fingerprint",
              "calibration_sha256", "provenance"}
    extra = {"points": {"target_representation"}, "tracks": {"initial_particles_sha256"},
             "masks": {"camera_fingerprint"}}[kind]
    if set(metadata) != common | extra:
        raise ValueError(f"{kind} metadata fields must be exactly {sorted(common | extra)}")
    if metadata["schema"] != TARGET_SCHEMAS[kind]:
        raise ValueError(f"Unsupported {kind} metadata schema")
    expected_units = "pixel" if kind == "masks" else "m"
    if metadata["units"] != expected_units:
        raise ValueError(f"{kind} metadata units must be {expected_units}")
    frames = {"image"} if kind == "masks" else {"scene", "camera_optical"}
    if metadata["coordinate_frame"] not in frames:
        raise ValueError(f"Unsupported {kind} coordinate_frame")
    if metadata["timestamp_reference"] != "sequence":
        raise ValueError("Target timestamp_reference must be sequence")
    for name, expected in (("sequence_fingerprint", sequence_fingerprint),
                           ("calibration_sha256", calibration_sha256)):
        if _sha256(metadata[name], name) != _sha256(expected, f"Prepared {name}"):
            raise ValueError(f"Target {name} differs from the prepared input")
    source = metadata["provenance"]
    kinds = {"observed", "synthetic_fixture"}
    if kind == "points":
        kinds.add("inferred_volume")
    if (not isinstance(source, dict) or source.get("kind") not in kinds
            or not isinstance(source.get("description"), str) or not source["description"].strip()):
        raise ValueError(f"{kind} provenance needs an allowed kind and nonempty description")
    if kind == "points":
        representation = metadata["target_representation"]
        if representation not in {"partial_observed", "full_observed", "inferred_volume"}:
            raise ValueError("Unsupported point target_representation")
        if representation == "inferred_volume" and source["kind"] not in {"inferred_volume", "synthetic_fixture"}:
            raise ValueError("Volume targets require inferred_volume or synthetic_fixture provenance")
        if representation != "inferred_volume" and source["kind"] == "inferred_volume":
            raise ValueError("Cloud targets cannot declare inferred_volume provenance")


def _frames(arrays, requested, frame_times):
    indices = _integer_vector(arrays["frame_indices"], "frame_indices", unique=True)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("frame_indices must be strictly increasing")
    times = _numeric(arrays["timestamps"], "timestamps")
    if times.shape != indices.shape or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite, increasing and match frame_indices")
    missing = sorted(set(requested) - set(map(int, indices)))
    if missing:
        raise ValueError(f"Target archive is missing scored frames {missing}")
    for frame, timestamp in zip(indices, times):
        if int(frame) not in frame_times:
            raise ValueError(f"Target frame {frame} has no recorded sequence timestamp")
        expected = frame_times[int(frame)]
        if (isinstance(expected, (bool, np.bool_))
                or not isinstance(expected, (int, float, np.integer, np.floating))
                or not np.isfinite(expected)
                or not np.isclose(timestamp, expected, rtol=0, atol=TIMESTAMP_ATOL_S)):
            raise ValueError(f"Target timestamp mismatch at frame {frame}")
    return indices


def _scene_positions(points, coordinate_frame, camera):
    if coordinate_frame == "scene":
        return points
    transform = np.asarray(camera.camera_from_scene, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, rtol=0, atol=1e-5)):
        raise ValueError("Camera optical targets require a proper rigid camera_from_scene transform")
    converted = (points - transform[:3, 3]) @ transform[:3, :3]
    if not np.isfinite(converted).all():
        raise ValueError("Nonfinite coordinates after camera-to-scene conversion")
    return converted


def _point_payload(metadata, arrays, indices, requested, camera, version):
    offsets = _integer_vector(arrays["offsets"], "offsets")
    points = _numeric(arrays["points"], "points")
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or len(offsets) != len(indices) + 1 or offsets[0] != 0
            or offsets[-1] != len(points) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Point targets require finite [N,3] coordinates and nonempty packed frame intervals")
    representation = metadata["target_representation"]
    is_volume = version.startswith("dpsi-prt-")
    if is_volume != (representation == "inferred_volume"):
        raise ValueError("DPSI PRT requires volume targets; other modes require observed cloud targets")
    points = _scene_positions(points, metadata["coordinate_frame"], camera)
    by_frame = {}
    for index, frame in enumerate(indices):
        if int(frame) in requested:
            by_frame[int(frame)] = {"points_scene": _readonly(points[offsets[index]:offsets[index + 1]]),
                                   "target_representation": representation}
    return by_frame


def _track_payload(metadata, arrays, indices, requested, camera, n_particles, loss_config,
                   initial_particles_sha256):
    if (_sha256(metadata["initial_particles_sha256"], "initial_particles_sha256")
            != _sha256(initial_particles_sha256, "Prepared initial_particles_sha256")):
        raise ValueError("Track mapping initial_particles_sha256 differs from the prepared particles")
    track_ids = _integer_vector(arrays["track_ids"], "track_ids", unique=True)
    particle_ids = _integer_vector(arrays["particle_ids"], "particle_ids", unique=True)
    if len(track_ids) != len(particle_ids) or np.any(particle_ids >= n_particles):
        raise ValueError("Track particle_ids must match track_ids and reference existing particles")
    positions = _numeric(arrays["positions"], "track positions")
    dimensions = (len(indices), len(track_ids))
    if positions.shape != dimensions + (3,):
        raise ValueError("Track positions must have dimensions [frames, tracks, 3]")
    valid = _boolean(arrays["valid"], dimensions, "track valid")
    if not np.isfinite(positions[valid]).all():
        raise ValueError("Valid track positions must be finite")
    positions = positions.copy()
    positions[~valid] = 0
    positions = _scene_positions(positions, metadata["coordinate_frame"], camera)
    positions[~valid] = 0
    by_frame = {}
    for index, frame in enumerate(indices):
        if int(frame) in requested:
            count = int(valid[index].sum())
            if loss_config.tracking_weight > 0 and not count and loss_config.empty_track_policy == "error":
                raise ValueError(f"Scored frame {frame} has no valid tracks")
            by_frame[int(frame)] = {
                "track_particle_ids": _readonly(particle_ids),
                "track_positions_scene": _readonly(positions[index]), "track_valid": _readonly(valid[index]),
                "metadata": {"track_ids": track_ids.tolist(), "valid_track_count": count,
                             "tracking_skipped_empty": bool(loss_config.tracking_weight > 0 and not count)},
            }
    return by_frame


def _mask_payload(metadata, arrays, indices, requested, camera):
    if _sha256(metadata["camera_fingerprint"], "camera_fingerprint") != camera_fingerprint(camera):
        raise ValueError("Mask camera_fingerprint differs from the prepared camera")
    dimensions = (len(indices), camera.height, camera.width)
    foreground = _boolean(arrays["foreground"], dimensions, "foreground mask")
    known = _boolean(arrays["known"], dimensions, "known-pixel mask")
    by_frame = {}
    for index, frame in enumerate(indices):
        if int(frame) in requested:
            if not known[index].any():
                raise ValueError(f"Scored frame {frame} has no known segmentation pixels")
            by_frame[int(frame)] = {"foreground_mask": _readonly(foreground[index]),
                                   "known_mask": _readonly(known[index])}
    return by_frame


def required_target_paths(loss_config):
    """Named input files required by the selected objective's active terms."""
    version = loss_config.version
    kinds = []
    if getattr(loss_config, "target_source", "recorded_cloud") == "external" or version.startswith("dpsi-prt-"):
        kinds.append("points")
    if version == "empm-offline-v1" and getattr(loss_config, "tracking_weight", 0.0) > 0:
        kinds.append("tracks")
    if version == "empm-mask-inspired-v1" and getattr(loss_config, "mask_weight", 0.0) > 0:
        kinds.append("masks")
    return tuple(name for kind in kinds for name in TARGET_PATH_PAIRS[kind])


def load_loss_targets(config, loss_config, *, frame_indices, sequence_fingerprint,
                      calibration_sha256, initial_particles_sha256, n_particles, camera, frame_times):
    """Load required supplemental inputs; return immutable arrays keyed by scored frame.

    frame_times is a mapping from every available processed frame ID to its actual
    sequence timestamp. Metadata contains no paths: relocation uses the existing
    named ExperimentConfig inputs. Reading and hashing the same bytes prevents a
    file change between verification and decoding from replacing the checked data.
    """
    requested = _integer_vector(np.asarray(frame_indices), "Requested frame_indices", unique=True)
    requested = set(map(int, requested))
    if not isinstance(frame_times, Mapping):
        raise ValueError("frame_times must map processed frame IDs to sequence timestamps")
    if isinstance(n_particles, bool) or not isinstance(n_particles, (int, np.integer)) or n_particles < 1:
        raise ValueError("n_particles must be a positive integer")
    version = str(loss_config.version)
    target_source = getattr(loss_config, "target_source", "recorded_cloud")
    required_names = set(required_target_paths(loss_config))
    required = {kind: names[0] in required_names for kind, names in TARGET_PATH_PAIRS.items()}
    by_frame = {frame: {} for frame in sorted(requested)}
    records = {}
    keys = {
        "points": {"frame_indices", "timestamps", "offsets", "points"},
        "tracks": {"frame_indices", "timestamps", "track_ids", "particle_ids", "positions", "valid"},
        "masks": {"frame_indices", "timestamps", "foreground", "known"},
    }
    for kind, names in TARGET_PATH_PAIRS.items():
        present = [name in config.paths for name in names]
        if any(present) and not all(present):
            raise ValueError(f"{kind} target archive and metadata must be provided together")
        if required[kind] and not all(present):
            raise ValueError(f"Loss {version} requires inputs {names}")
        if not any(present):
            continue
        if kind == "points" and target_source != "external":
            raise ValueError("Point target files require target_source='external'")
        if kind == "tracks" and version != "empm-offline-v1":
            raise ValueError("Track files are only supported by empm-offline-v1")
        if kind == "masks" and version != "empm-mask-inspired-v1":
            raise ValueError("Mask files are only supported by empm-mask-inspired-v1")
        metadata, arrays, record = _read_pair(config, kind)
        _metadata(metadata, kind, sequence_fingerprint, calibration_sha256)
        if set(arrays) != keys[kind]:
            raise ValueError(f"{kind} NPZ arrays must be exactly {sorted(keys[kind])}")
        indices = _frames(arrays, requested, frame_times)
        if kind == "points":
            payload = _point_payload(metadata, arrays, indices, requested, camera, version)
        elif kind == "tracks":
            if getattr(loss_config, "empty_track_policy", "error") not in {"error", "skip"}:
                raise ValueError("empty_track_policy must be error or skip")
            payload = _track_payload(metadata, arrays, indices, requested, camera, n_particles,
                                     loss_config, initial_particles_sha256)
        else:
            payload = _mask_payload(metadata, arrays, indices, requested, camera)
        for frame, values in payload.items():
            extra_metadata = values.pop("metadata", {})
            by_frame[frame].update(values)
            by_frame[frame].setdefault("metadata", {}).update(extra_metadata)
            by_frame[frame]["metadata"].setdefault("target_sources", {})[kind] = metadata["provenance"]
            if kind == "points":
                by_frame[frame]["metadata"]["source_kind"] = metadata["provenance"]["kind"]
        records[kind] = {**record, "definition": metadata, "available_frames": indices.tolist(),
                         "scored_frames": sorted(requested), "required_by_loss": required[kind]}
    return TargetBundle(by_frame, {"schema": "taichidough/prepared-loss-targets/v1", "inputs": records,
                                   "scored_frames": sorted(requested), "coordinates": "scene",
                                   "timestamp_atol_s": TIMESTAMP_ATOL_S})

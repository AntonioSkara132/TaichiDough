"""Bounded metadata-only discovery; file presence never implies validated inputs."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import stat


SCHEMA = "taichidough/differentiable-mpm-inventory/v1"
DRAFT_SCHEMA = "taichidough/differentiable-mpm-inventory-draft/v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAX_METADATA_BYTES = 2 * 1024 * 1024
EPISODE_NAME = re.compile(r"episode\d+(?:[_-].*)?$")
SESSION_NAME = re.compile(r"(?:snimanje_)?\d{1,2}_\d{1,2}$")


def _error(errors, path, kind, message):
    errors.append({"path": str(path), "kind": kind, "message": str(message)})


def _document(path, errors):
    try:
        if path.is_symlink():
            raise ValueError("Symlink metadata is not followed")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Metadata must be a regular file")
        if info.st_size > MAX_METADATA_BYTES:
            raise ValueError(f"Metadata exceeds {MAX_METADATA_BYTES} byte limit")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result
        def invalid(value):
            raise ValueError(f"Nonfinite JSON constant: {value}")
        def finite_float(value):
            result = float(value)
            if not math.isfinite(result):
                invalid(value)
            return result
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                            parse_constant=invalid, parse_float=finite_float)
        if not isinstance(result, dict):
            raise ValueError("Metadata must be a JSON object")
        return result
    except (OSError, ValueError, UnicodeError) as exc:
        _error(errors, path, "metadata_error", exc)
        return None


def _walk(root, errors):
    """Visit only the explicitly supplied directory; do not follow symlinks."""
    if root.is_symlink() or root.resolve() != root:
        _error(errors, root, "symlink_skipped", "Roots reached through symlinks are not traversed")
        return
    if not root.is_dir():
        _error(errors, root, "missing_root", "Root is not an existing directory")
        return
    def onerror(exc):
        _error(errors, exc.filename or root, "scan_error", exc)
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(directory)
        kept = []
        for name in sorted(dirs):
            child = parent / name
            if child.is_symlink():
                _error(errors, child, "symlink_skipped", "Directory symlink is not traversed")
            elif not name.startswith(".") and name != "__pycache__":
                kept.append(name)
        dirs[:] = kept
        regular = []
        for name in sorted(files):
            child = parent / name
            if child.is_symlink():
                _error(errors, child, "symlink_skipped", "File symlink is not inspected")
            else:
                regular.append(name)
        yield parent, regular


def _absolute(path):
    # Preserve the last path component so explicit root symlinks can be rejected.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _recording_id(path):
    """Only date/session-qualified names are matched across dataset variants."""
    if not EPISODE_NAME.fullmatch(path.name):
        return None
    for parent in path.parents:
        if SESSION_NAME.fullmatch(parent.name):
            between = path.relative_to(parent).as_posix()
            return parent.name + "/" + between
    return None


def _reference_path(value, base):
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _file_candidate(path):
    return {"path": str(path), "exists": path.is_file() and not path.is_symlink()}


def scan_dataset(roots: list[Path], *, config_paths=(), reconstruction_roots=()):
    """Return deterministic discovery data without tensor loads or input validation.

    Experiment config relative paths follow the existing repository-root convention.
    Session-name matches are candidates only: they never establish data equivalence.
    """
    errors, episodes, reconstructions, configs = [], [], [], []
    roots = sorted({_absolute(p) for p in roots}, key=str)
    root_names = Counter(root.name for root in roots)
    root_records = []
    for number, root in enumerate(roots):
        root_id = root.name if root_names[root.name] == 1 else f"{root.name}@{number + 1}"
        root_records.append({"id": root_id, "path": str(root)})
        for directory, names in _walk(root, errors):
            if not EPISODE_NAME.fullmatch(directory.name):
                continue
            relative = directory.relative_to(root).as_posix()
            metadata = {}
            for name in ("sequence_metadata.json", "conversion_metadata.json"):
                if name in names:
                    metadata[name] = _document(directory / name, errors)
            sequence = metadata.get("sequence_metadata.json")
            calibration = sequence.get("calibration") if isinstance(sequence, dict) else None
            calibration_candidates = {directory / name for name in names
                                      if name.startswith("scene_calibration") and name.endswith(".json")}
            if isinstance(calibration, dict):
                linked = _reference_path(calibration.get("path"), directory)
                if linked is not None:
                    # Record external references but never open them during discovery.
                    calibration_candidates.add(linked)
            raw = all(name in names for name in ("pointclouds.pt", "paths.pt"))
            processed = all(name in names for name in
                            ("pointclouds_interpolated.pt", "paths_interpolated.pt", "sequence_metadata.json"))
            episodes.append({"id": root_id + "/" + relative, "root_id": root_id,
                             "relative_path": relative, "path": str(directory),
                             "recording_id": _recording_id(directory),
                             "raw_pair_present": raw,
                             "raw_bags_present": any(name.endswith((".db3", ".mcap")) for name in names),
                             "processed_inputs_present": processed,
                             "sequence_metadata_parsed": isinstance(sequence, dict),
                             "calibration_record_status": calibration.get("status") if isinstance(calibration, dict) else None,
                             "calibration_candidates": [_file_candidate(p) for p in sorted(calibration_candidates, key=str)],
                             "config_candidates": [], "reconstruction_candidates": [],
                             "fully_validated": False, "status": "discovered_not_validated"})
    for root in sorted({_absolute(p) for p in reconstruction_roots}, key=str):
        for directory, names in _walk(root, errors):
            if "reconstruction_metadata.json" not in names:
                continue
            path = directory / "reconstruction_metadata.json"
            document = _document(path, errors)
            if document is None:
                continue
            episode = _reference_path(document.get("episode_dir"), directory)
            reconstructions.append({"path": str(path), "episode_path": str(episode) if episode else None,
                                    "recording_id": _recording_id(episode) if episode else None,
                                    "source_frame": document.get("frame"),
                                    "particles": _file_candidate(directory / "sampled_particles_xyz.npy")})
    for path in sorted({_absolute(p) for p in config_paths}, key=str):
        document = _document(path, errors)
        if document is None:
            continue
        paths = document.get("paths")
        if not isinstance(paths, dict):
            _error(errors, path, "config_error", "Expected paths object")
            continue
        episode = _reference_path(paths.get("episode"), REPOSITORY_ROOT)
        if episode is None:
            _error(errors, path, "config_error", "Expected paths.episode string")
            continue
        inputs = {name: str(value) if value else None for name in
                  ("calibration", "initial_particles", "reconstruction_metadata", "tool_geometry")
                  for value in [_reference_path(paths.get(name), REPOSITORY_ROOT)]}
        configs.append({"path": str(path), "episode_path": str(episode),
                        "recording_id": _recording_id(episode), "input_paths": inputs,
                        "mass_kg_declared": document.get("mass_kg"),
                        "density_kg_m3_declared": document.get("density_kg_m3"),
                        "validated": False})
    groups = defaultdict(list)
    for episode in episodes:
        if episode["recording_id"]:
            groups[episode["recording_id"]].append(episode["id"])
        for field, candidates in (("config_candidates", configs), ("reconstruction_candidates", reconstructions)):
            for candidate in candidates:
                exact = candidate["episode_path"] == str(Path(episode["path"]).resolve())
                session = episode["recording_id"] and candidate["recording_id"] == episode["recording_id"]
                if exact or session:
                    episode[field].append({**candidate, "match": "resolved_path" if exact else "session_candidate_only"})
        blockers = []
        if not episode["processed_inputs_present"]:
            blockers.append("missing_processed_inputs")
        if not episode["sequence_metadata_parsed"]:
            blockers.append("missing_or_invalid_sequence_metadata")
        existing_calibrations = [c for c in episode["calibration_candidates"] if c["exists"]]
        if not existing_calibrations:
            blockers.append("missing_calibration_candidate")
        elif len(existing_calibrations) > 1:
            blockers.append("ambiguous_calibration_candidates")
        if not episode["reconstruction_candidates"]:
            blockers.append("missing_reconstruction_candidate")
        elif len(episode["reconstruction_candidates"]) > 1:
            blockers.append("ambiguous_reconstruction_candidates")
        if any(not c["particles"]["exists"] for c in episode["reconstruction_candidates"]):
            blockers.append("reconstruction_candidate_missing_particles")
        if any(c["source_frame"] != 0 for c in episode["reconstruction_candidates"]):
            blockers.append("reconstruction_candidate_not_frame_zero")
        if not episode["config_candidates"]:
            blockers.append("missing_config_candidate")
        elif len(episode["config_candidates"]) > 1:
            blockers.append("ambiguous_config_candidates")
        if not any(isinstance(c["mass_kg_declared"], (int, float))
                   and not isinstance(c["mass_kg_declared"], bool)
                   and c["mass_kg_declared"] > 0 for c in episode["config_candidates"]):
            blockers.append("missing_explicit_positive_mass")
        if any(c["match"] == "session_candidate_only" for c in
               episode["config_candidates"] + episode["reconstruction_candidates"]):
            blockers.append("session_candidate_requires_source_verification")
        if not episode["recording_id"]:
            blockers.append("unresolved_session_identity")
        if len(groups.get(episode["recording_id"], [])) > 1:
            blockers.append("duplicate_recording_variants")
        # Candidate settings remain declarations, not verified physical measurements.
        blockers.append("requires_full_input_validation")
        episode["blockers"] = blockers
    # Groups are complete only after all episodes have been visited.
    duplicates = [{"recording_id": key, "episode_ids": sorted(ids)}
                  for key, ids in sorted(groups.items()) if len(ids) > 1]
    duplicate_ids = {item for group in duplicates for item in group["episode_ids"]}
    for episode in episodes:
        if episode["id"] in duplicate_ids and "duplicate_recording_variants" not in episode["blockers"]:
            episode["blockers"].append("duplicate_recording_variants")
        episode["blockers"].sort()
    return {"schema": SCHEMA, "roots": root_records, "episodes": episodes,
            "duplicate_recording_groups": duplicates,
            "reconstruction_candidates": reconstructions, "config_candidates": configs,
            "errors": sorted(errors, key=lambda row: (row["path"], row["kind"], row["message"])),
            "counts": {"episodes": len(episodes),
                       "raw_pairs_present": sum(e["raw_pair_present"] for e in episodes),
                       "raw_bags_present": sum(e["raw_bags_present"] for e in episodes),
                       "processed_inputs_present": sum(e["processed_inputs_present"] for e in episodes),
                       "episodes_with_config_candidates": sum(bool(e["config_candidates"]) for e in episodes),
                       "fully_validated": 0},
            "notes": ["File presence and metadata declarations are not input validation.",
                      "Dataset variants are never combined automatically.",
                      "Session matches do not verify relocated inputs or identical recordings.",
                      "No tensor contents or large-file hashes were read."]}


def inventory_draft(report):
    """Produce an explicitly incomplete document, never a runnable dataset config."""
    if report.get("schema") != SCHEMA:
        raise ValueError("Expected a dataset inventory report")
    return {"schema": DRAFT_SCHEMA, "runnable": False,
            "episodes": [{"id": e["id"], "recording_id": e["recording_id"],
                          "config": None, "split": None, "weight": None,
                          "candidate_configs": [c["path"] for c in e["config_candidates"]],
                          "blockers": list(e["blockers"])} for e in report["episodes"]],
            "notes": ["Choose one input variant per recording and supply verified episode configs.",
                      "Mass, calibration, material grouping and frame windows are not inferred."]}

#!/usr/bin/env python3
"""Identify an effective Young's modulus for one recorded TaichiDough model setup."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

try:
    from material_calibration import (
        LOSS_COMPONENTS,
        aggregate_loss_results,
        aggregate_named_windows,
        boundary_expansion,
        canonical_json_hash,
        file_sha256,
        flat_minimum_diagnostics,
        logarithmic_candidates,
        partial_view_loss,
        refinement_candidates,
        select_best_candidate,
        sequence_fingerprint,
        validate_cache_directory,
        visible_optical_points,
        whole_window_bootstrap_interval,
    )
except ImportError:
    from .material_calibration import (
        LOSS_COMPONENTS,
        aggregate_loss_results,
        aggregate_named_windows,
        boundary_expansion,
        canonical_json_hash,
        file_sha256,
        flat_minimum_diagnostics,
        logarithmic_candidates,
        partial_view_loss,
        refinement_candidates,
        select_best_candidate,
        sequence_fingerprint,
        validate_cache_directory,
        visible_optical_points,
        whole_window_bootstrap_interval,
    )


MANIFEST_SCHEMA = "taichidough/material-calibration-manifest/v1"
RESULT_SCHEMA = "taichidough/material-calibration/v1"
CACHE_SCHEMA = "taichidough/material-calibration-cache/v1"
DEFAULT_YOUNGS_MODULUS_PA = 2000.0
REQUIRED_CACHE_FILES = (
    "simulation/camera_parameters.json",
    "evaluation/dynamic_topview_metrics.json",
    "loss.json",
)
PLACEHOLDER_PATTERN = re.compile(r"^<[^>]+>$")
TOKEN_PATTERN = re.compile(r"^\{([A-Za-z0-9_.]+)\}$")


REQUIRED_SIMULATOR_ARGUMENTS = {
    "--particles",
    "--grid",
    "--dt",
    "--substeps-per-frame",
    "--replay-stride",
    "--replay-max-gap",
    "--depth-width",
    "--depth-height",
    "--depth-splat-radius",
    "--depth-pointcloud-max-points",
    "--poisson-ratio",
    "--viscosity",
    "--density",
    "--object-mass-kg",
    "--gravity",
    "--floor-y",
    "--floor-friction",
    "--floor-absorption",
    "--tool-contact-padding",
    "--tool-contact-friction",
    "--tool-contact-absorption",
    "--tool-stickiness",
    "--floor-stickiness",
    "--floor-plastic-damping-band",
    "--velocity-damping",
    "--pure-viscoelastic",
    "--plastic-min",
    "--plastic-max",
    "--plastic-velocity-damping",
    "--plastic-affine-damping",
    "--use-jp",
    "--jp-hardening",
    "--jp-min",
    "--jp-max",
    "--cpu",
    "--initial-particles-fit",
    "--initial-particles-axis-map",
    "--initial-particles-scale",
    "--initial-particles-offset",
    "--initial-particles-raw-scene-coordinates",
    "--initial-particles-seed",
    "--no-publish-dough-center",
}
REQUIRED_EVALUATOR_ARGUMENTS = {
    "--view",
    "--frame-stride",
    "--pair-tolerance",
    "--cell-size",
    "--trim-quantile",
}
RESERVED_SIMULATOR_ARGUMENTS = {
    "--youngs-modulus",
    "--output-dir",
    "--replay-episode",
    "--replay-start-frame",
    "--replay-end-frame",
    "--initial-particles",
    "--initial-particles-metadata",
    "--initial-particles-calibration",
    "--tool-geometry",
}
RESERVED_EVALUATOR_ARGUMENTS = {
    "--episode-dir",
    "--taichi-metadata",
    "--calibration",
    "--output-dir",
}


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required manifest field {location}.{key}")
    value = mapping[key]
    if value is None or value == "" or (isinstance(value, str) and PLACEHOLDER_PATTERN.match(value)):
        raise ValueError(f"Manifest field {location}.{key} still contains a placeholder")
    return value


def _number(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result


def _integer(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _resolve(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _validate_tool_geometry_document(document: Any) -> None:
    if not isinstance(document, dict) or document.get("schema") != "taichidough/tool-geometry/v1":
        raise ValueError("Geometry input must use taichidough/tool-geometry/v1")
    tools = document.get("tools")
    if not isinstance(tools, list) or len(tools) != 2:
        raise ValueError("Tool geometry must describe exactly two tools")
    names = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"].strip():
            raise ValueError(f"Tool geometry tools[{index}].name must be a nonempty string")
        names.append(tool["name"].strip())
        extents = np.asarray(tool.get("half_extents_m"), dtype=np.float64)
        if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(extents <= 0):
            raise ValueError(f"Tool geometry tools[{index}].half_extents_m must contain three positive values")
        transform = np.asarray(tool.get("marker_from_collider"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"Tool geometry tools[{index}].marker_from_collider must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError(f"Tool geometry tools[{index}].marker_from_collider has an invalid last row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError(f"Tool geometry tools[{index}].marker_from_collider rotation must be proper and orthonormal")
    if len(set(names)) != 2:
        raise ValueError("Tool geometry names must be unique")


def _floor_y_from_plane(plane: Any) -> float:
    if isinstance(plane, dict):
        if "coefficients" in plane:
            values = plane["coefficients"]
        elif "normal" in plane:
            normal = np.asarray(plane["normal"], dtype=np.float64)
            if "point" in plane:
                point = np.asarray(plane["point"], dtype=np.float64)
                values = [*normal, -float(np.dot(normal, point))]
            else:
                offset = plane.get("offset", plane.get("d", plane.get("distance")))
                values = [*normal, offset]
        else:
            values = [plane.get(key) for key in ("a", "b", "c", "d")]
    else:
        values = plane
    coefficients = np.asarray(values, dtype=np.float64)
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise ValueError("Reconstruction floor plane must contain four finite coefficients")
    normal_norm = float(np.linalg.norm(coefficients[:3]))
    if normal_norm <= 1e-12:
        raise ValueError("Reconstruction floor plane normal must be nonzero")
    coefficients /= normal_norm
    if abs(coefficients[0]) > 1e-6 or abs(coefficients[2]) > 1e-6 or abs(abs(coefficients[1]) - 1.0) > 1e-6:
        raise ValueError("Reconstruction floor plane is incompatible with the simulator constant-Y floor")
    return float(-coefficients[3] / coefficients[1])


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal fingerprint")
    return value.lower()


def _validate_argument_map(arguments: Any, required: set[str], reserved: set[str], name: str) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError(f"{name} must be a JSON object mapping CLI flags to fixed values")
    invalid_names = [flag for flag in arguments if not isinstance(flag, str) or not flag.startswith("--")]
    if invalid_names:
        raise ValueError(f"{name} contains invalid CLI flag names")
    missing = sorted(required - set(arguments))
    if missing:
        raise ValueError(f"{name} is missing fixed arguments: " + ", ".join(missing))
    conflicts = sorted(reserved & set(arguments))
    if conflicts:
        raise ValueError(f"{name} must not set candidate/window arguments: " + ", ".join(conflicts))
    for flag, value in arguments.items():
        if isinstance(value, str) and PLACEHOLDER_PATTERN.match(value):
            raise ValueError(f"{name}.{flag} still contains a placeholder")
        if isinstance(value, list):
            if not value or any(isinstance(item, (dict, list)) for item in value):
                raise ValueError(f"{name}.{flag} must be a nonempty flat list")
        elif isinstance(value, dict):
            raise ValueError(f"{name}.{flag} cannot be a JSON object")
        elif value is None and flag != "--pair-tolerance":
            raise ValueError(f"Only evaluator --pair-tolerance may be null; {name}.{flag} is null")
    return dict(arguments)


def compile_cli_arguments(arguments: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for flag in sorted(arguments):
        value = arguments[flag]
        if value is None or value is False:
            continue
        result.append(flag)
        if value is True:
            continue
        values = value if isinstance(value, list) else [value]
        result.extend(str(item) for item in values)
    return result


def _validate_physical_and_numerical_parameters(simulator: Mapping[str, Any], evaluator: Mapping[str, Any]) -> None:
    for flag in (
        "--particles", "--grid", "--substeps-per-frame", "--replay-stride", "--depth-width", "--depth-height"
    ):
        _integer(simulator[flag], f"fixed_parameters.simulator_arguments.{flag}", positive=True)
    for flag in ("--depth-splat-radius", "--depth-pointcloud-max-points", "--initial-particles-seed"):
        _integer(simulator[flag], f"fixed_parameters.simulator_arguments.{flag}", nonnegative=True)
    for flag in ("--dt", "--replay-max-gap", "--density", "--object-mass-kg", "--initial-particles-scale"):
        _number(simulator[flag], f"fixed_parameters.simulator_arguments.{flag}", positive=True)
    for flag in ("--viscosity", "--tool-contact-friction"):
        _number(simulator[flag], f"fixed_parameters.simulator_arguments.{flag}", nonnegative=True)
    velocity_damping = _number(simulator["--velocity-damping"], "velocity damping", positive=True)
    if velocity_damping > 1.0:
        raise ValueError("Velocity damping cannot exceed one")
    poisson = _number(simulator["--poisson-ratio"], "poisson ratio")
    if not -1.0 < poisson < 0.5:
        raise ValueError("Poisson ratio must lie between -1 and 0.5")
    for flag in ("--floor-friction", "--tool-contact-padding", "--floor-plastic-damping-band"):
        _number(simulator[flag], f"fixed_parameters.simulator_arguments.{flag}", nonnegative=True)
    for flag in ("--plastic-min", "--plastic-max", "--jp-min", "--jp-max"):
        _number(simulator[flag], f"fixed_parameters.simulator_arguments.{flag}", positive=True)
    for flag in (
        "--floor-absorption", "--tool-contact-absorption", "--tool-stickiness", "--floor-stickiness",
        "--plastic-velocity-damping", "--plastic-affine-damping",
    ):
        value = _number(simulator[flag], f"fixed_parameters.simulator_arguments.{flag}", nonnegative=True)
        if value > 1.0:
            raise ValueError(f"{flag} cannot exceed one")
    if float(simulator["--plastic-min"]) > float(simulator["--plastic-max"]):
        raise ValueError("Plastic minimum cannot exceed plastic maximum")
    if float(simulator["--jp-min"]) > float(simulator["--jp-max"]):
        raise ValueError("Jp minimum cannot exceed Jp maximum")
    for flag in ("--gravity", "--floor-y", "--jp-hardening"):
        _number(simulator[flag], f"fixed_parameters.simulator_arguments.{flag}")
    for flag in ("--initial-particles-offset",):
        value = simulator[flag]
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{flag} must contain three coordinates")
        for coordinate in value:
            _number(coordinate, flag)
    for flag in ("--pure-viscoelastic", "--use-jp", "--cpu", "--initial-particles-raw-scene-coordinates", "--no-publish-dough-center"):
        if not isinstance(simulator[flag], bool):
            raise ValueError(f"{flag} must be explicitly true or false")
    if simulator["--initial-particles-fit"] != "none":
        raise ValueError("Reconstructed scene-coordinate particles require --initial-particles-fit none")
    if simulator["--initial-particles-axis-map"] != "xyz":
        raise ValueError("Reconstructed scene-coordinate particles require --initial-particles-axis-map xyz")
    if not math.isclose(float(simulator["--initial-particles-scale"]), 1.0):
        raise ValueError("Reconstructed scene-coordinate particles require unit initial-particle scale")
    if any(not math.isclose(float(value), 0.0, abs_tol=1e-12) for value in simulator["--initial-particles-offset"]):
        raise ValueError("Reconstructed scene-coordinate particles cannot use an initial-particle offset")
    if simulator["--initial-particles-raw-scene-coordinates"]:
        raise ValueError("The simulator derives scene-coordinate initialization from reconstruction metadata; do not pass --initial-particles-raw-scene-coordinates")
    _integer(evaluator["--frame-stride"], "evaluator frame stride", positive=True)
    if evaluator["--pair-tolerance"] is not None:
        _number(evaluator["--pair-tolerance"], "evaluator pair tolerance", nonnegative=True)
    _number(evaluator["--cell-size"], "evaluator cell size", positive=True)
    trim = _number(evaluator["--trim-quantile"], "evaluator trim quantile", nonnegative=True)
    if trim >= 0.5:
        raise ValueError("Evaluator trim quantile must be less than 0.5")


def _validate_command_template(template: Any, required_tokens: Iterable[str], name: str) -> list[str]:
    if not isinstance(template, list) or not template or any(not isinstance(item, str) or not item for item in template):
        raise ValueError(f"commands.{name} must be a nonempty JSON string array")
    tokens = {match.group(1) for item in template if (match := TOKEN_PATTERN.match(item))}
    missing = sorted(set(required_tokens) - tokens)
    if missing:
        raise ValueError(f"commands.{name} is missing substitution tokens: " + ", ".join("{" + token + "}" for token in missing))
    return list(template)


def validate_manifest(manifest: Any, manifest_path: Path) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object")
    if _required(manifest, "schema", "manifest") != MANIFEST_SCHEMA:
        raise ValueError(f"Manifest schema must be {MANIFEST_SCHEMA!r}")
    _required(manifest, "name", "manifest")
    inputs = _required(manifest, "inputs", "manifest")
    if not isinstance(inputs, dict):
        raise ValueError("manifest.inputs must be a JSON object")
    resolved_inputs: dict[str, dict[str, Any]] = {}
    for name in ("geometry", "reconstruction_metadata", "initial_particles", "calibration", "simulator", "evaluator"):
        spec = _required(inputs, name, "manifest.inputs")
        if not isinstance(spec, dict):
            raise ValueError(f"manifest.inputs.{name} must be a JSON object")
        path = _resolve(str(_required(spec, "path", f"manifest.inputs.{name}")), manifest_path)
        expected = _validate_sha256(_required(spec, "sha256", f"manifest.inputs.{name}"), f"inputs.{name}.sha256")
        if not path.is_file():
            raise ValueError(f"Input file does not exist: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Fingerprint mismatch for inputs.{name}: expected {expected}, found {actual}")
        resolved_inputs[name] = {"path": str(path), "sha256": actual}
    geometry_document = json.loads(Path(resolved_inputs["geometry"]["path"]).read_text())
    _validate_tool_geometry_document(geometry_document)

    sequence = _required(inputs, "sequence", "manifest.inputs")
    if not isinstance(sequence, dict):
        raise ValueError("manifest.inputs.sequence must be a JSON object")
    episode_dir = _resolve(str(_required(sequence, "episode_dir", "manifest.inputs.sequence")), manifest_path)
    if not episode_dir.is_dir():
        raise ValueError(f"Sequence episode directory does not exist: {episode_dir}")
    expected_sequence = _validate_sha256(
        _required(sequence, "fingerprint", "manifest.inputs.sequence"), "inputs.sequence.fingerprint"
    )
    actual_sequence = sequence_fingerprint(episode_dir)
    if actual_sequence != expected_sequence:
        raise ValueError(f"Sequence fingerprint mismatch: expected {expected_sequence}, found {actual_sequence}")
    resolved_inputs["sequence"] = {"episode_dir": str(episode_dir), "fingerprint": actual_sequence}

    reconstruction = json.loads(Path(resolved_inputs["reconstruction_metadata"]["path"]).read_text())
    if not isinstance(reconstruction, dict) or reconstruction.get("schema") != "voxel_dough_reconstruction/v2":
        raise ValueError("Reconstruction metadata must use voxel_dough_reconstruction/v2")
    reconstructed_volume = _number(
        reconstruction.get("object_volume_m3"), "reconstruction object volume", positive=True
    )
    reconstruction_frame = _integer(
        reconstruction.get("frame"), "reconstruction source frame", nonnegative=True
    )
    voxel_size = _number(reconstruction.get("voxel_size"), "reconstruction voxel size", positive=True)
    voxel_count = _integer(reconstruction.get("voxel_count"), "reconstruction voxel count", positive=True)
    if not math.isclose(reconstructed_volume, voxel_count * voxel_size ** 3, rel_tol=1e-9, abs_tol=1e-15):
        raise ValueError("Reconstruction object volume is inconsistent with voxel count and voxel size")
    calibration_info = reconstruction.get("calibration") or {}
    calibration_schema = reconstruction.get("calibration_schema") or calibration_info.get("schema")
    if calibration_info.get("is_metric") is not True or not str(calibration_schema).lower().endswith("/v2"):
        raise ValueError("Reconstruction metadata must identify a metric v2 calibration")
    scene_frame = calibration_info.get("scene_frame")
    particle_frame = (reconstruction.get("array_frames") or {}).get("sampled_particles_xyz")
    if not scene_frame:
        scene_frame = particle_frame
    if not particle_frame or particle_frame != scene_frame:
        raise ValueError("Reconstructed particles must be identified as scene-coordinate data")
    reconstruction_fill = reconstruction.get("fill") or {}
    if reconstruction_fill.get("mode") != "floor":
        raise ValueError("Material calibration requires floor-mode reconstruction metadata")
    reconstructed_floor_y = _floor_y_from_plane(reconstruction_fill.get("floor_plane_scene"))
    reconstruction_output = reconstruction.get("outputs", {}).get("sampled_particles_xyz")
    if reconstruction_output:
        output_path = Path(reconstruction_output)
        if not output_path.is_absolute():
            metadata_parent = Path(resolved_inputs["reconstruction_metadata"]["path"]).parent
            candidates = (metadata_parent / output_path, metadata_parent / output_path.name)
            output_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if output_path.resolve() != Path(resolved_inputs["initial_particles"]["path"]):
            raise ValueError("Initial particle file does not match reconstruction metadata output")

    fixed = _required(manifest, "fixed_parameters", "manifest")
    if not isinstance(fixed, dict):
        raise ValueError("manifest.fixed_parameters must be a JSON object")
    simulator_arguments = _validate_argument_map(
        _required(fixed, "simulator_arguments", "manifest.fixed_parameters"),
        REQUIRED_SIMULATOR_ARGUMENTS,
        RESERVED_SIMULATOR_ARGUMENTS,
        "fixed_parameters.simulator_arguments",
    )
    evaluator_arguments = _validate_argument_map(
        _required(fixed, "evaluator_arguments", "manifest.fixed_parameters"),
        REQUIRED_EVALUATOR_ARGUMENTS,
        RESERVED_EVALUATOR_ARGUMENTS,
        "fixed_parameters.evaluator_arguments",
    )
    recorded_setup = _required(fixed, "recorded_setup", "manifest.fixed_parameters")
    if not isinstance(recorded_setup, dict):
        raise ValueError("fixed_parameters.recorded_setup must be a JSON object")
    recorded_mass = _number(
        _required(recorded_setup, "dough_mass_kg", "fixed_parameters.recorded_setup"),
        "recorded dough mass",
        positive=True,
    )
    simulator_mass = _number(simulator_arguments["--object-mass-kg"], "simulator object mass", positive=True)
    if not math.isclose(recorded_mass, simulator_mass, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("Recorded dough mass must match simulator --object-mass-kg")
    derived_density = recorded_mass / reconstructed_volume
    if not math.isclose(float(simulator_arguments["--density"]), derived_density, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError("Simulator --density must equal recorded dough mass divided by reconstructed volume")
    if not math.isclose(float(simulator_arguments["--floor-y"]), reconstructed_floor_y, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Simulator --floor-y must match the floor height in reconstruction metadata")
    _required(recorded_setup, "geometry_description", "fixed_parameters.recorded_setup")
    declared_tool_fingerprint = _validate_sha256(
        _required(recorded_setup, "tool_geometry_fingerprint", "fixed_parameters.recorded_setup"),
        "fixed_parameters.recorded_setup.tool_geometry_fingerprint",
    )
    if declared_tool_fingerprint != resolved_inputs["geometry"]["sha256"]:
        raise ValueError("Recorded tool geometry fingerprint must match inputs.geometry.sha256")
    _validate_physical_and_numerical_parameters(simulator_arguments, evaluator_arguments)

    windows = _required(manifest, "windows", "manifest")
    if not isinstance(windows, list) or not windows:
        raise ValueError("manifest.windows must be a nonempty array")
    names: set[str] = set()
    validated_windows = []
    for index, raw in enumerate(windows):
        if not isinstance(raw, dict):
            raise ValueError(f"windows[{index}] must be a JSON object")
        name = str(_required(raw, "name", f"windows[{index}]"))
        if name in names:
            raise ValueError(f"Duplicate window name {name!r}")
        names.add(name)
        split = _required(raw, "split", f"windows[{index}]")
        if split not in ("training", "validation"):
            raise ValueError(f"Window {name!r} split must be training or validation")
        start = _integer(_required(raw, "start_frame", f"windows[{index}]"), f"window {name} start", nonnegative=True)
        end = _integer(_required(raw, "end_frame", f"windows[{index}]"), f"window {name} end", nonnegative=True)
        if start < reconstruction_frame:
            raise ValueError(
                f"Window {name!r} starts before reconstruction frame {reconstruction_frame}"
            )
        if end <= start:
            raise ValueError(f"Window {name!r} must contain at least one post-initial frame")
        weight = _number(raw.get("weight", 1.0), f"window {name} weight", positive=True)
        validated_windows.append(
            {
                "name": name,
                "split": split,
                "start_frame": start,
                "end_frame": end,
                "replay_start_frame": reconstruction_frame,
                "weight": weight,
            }
        )
    if not any(window["split"] == "training" for window in validated_windows):
        raise ValueError("At least one training window is required")
    if not any(window["split"] == "validation" for window in validated_windows):
        raise ValueError("At least one held-out validation window is required")
    training_ranges = [window for window in validated_windows if window["split"] == "training"]
    validation_ranges = [window for window in validated_windows if window["split"] == "validation"]
    for training in training_ranges:
        for validation in validation_ranges:
            if max(training["start_frame"], validation["start_frame"]) <= min(training["end_frame"], validation["end_frame"]):
                raise ValueError(
                    f"Held-out window {validation['name']!r} overlaps training window {training['name']!r}"
                )

    loss = _required(manifest, "loss", "manifest")
    if not isinstance(loss, dict):
        raise ValueError("manifest.loss must be a JSON object")
    weights = _required(loss, "weights", "manifest.loss")
    if not isinstance(weights, dict) or set(weights) != set(LOSS_COMPONENTS):
        raise ValueError("loss.weights must specify exactly: " + ", ".join(LOSS_COMPONENTS))
    for key, value in weights.items():
        _number(value, f"loss.weights.{key}", nonnegative=True)
    for key in ("depth_scale_m", "distance_scale_m", "huber_delta"):
        _number(_required(loss, key, "manifest.loss"), f"loss.{key}", positive=True)
    for key in ("min_common_pixels", "min_observed_pixels", "min_simulation_pixels", "min_observed_points", "min_simulation_points"):
        _integer(_required(loss, key, "manifest.loss"), f"loss.{key}", positive=True)
    _integer(loss.get("nearest_chunk_size", 1024), "loss.nearest_chunk_size", positive=True)
    if not isinstance(loss.get("require_all_frames", True), bool):
        raise ValueError("loss.require_all_frames must be boolean")

    search = _required(manifest, "search", "manifest")
    if not isinstance(search, dict):
        raise ValueError("manifest.search must be a JSON object")
    default = _number(search.get("default_youngs_modulus_pa", DEFAULT_YOUNGS_MODULUS_PA), "search default modulus", positive=True)
    if not math.isclose(default, DEFAULT_YOUNGS_MODULUS_PA):
        raise ValueError("The required comparison default is current E=2000 Pa")
    lower = _number(_required(search, "initial_min_pa", "manifest.search"), "initial minimum", positive=True)
    upper = _number(_required(search, "initial_max_pa", "manifest.search"), "initial maximum", positive=True)
    hard_lower = _number(_required(search, "hard_min_pa", "manifest.search"), "hard minimum", positive=True)
    hard_upper = _number(_required(search, "hard_max_pa", "manifest.search"), "hard maximum", positive=True)
    if not hard_lower <= lower < default < upper <= hard_upper:
        raise ValueError("Search bounds must satisfy hard_min <= initial_min < 2000 < initial_max <= hard_max")
    _integer(_required(search, "coarse_count", "manifest.search"), "search coarse count", positive=True)
    if int(search["coarse_count"]) < 3:
        raise ValueError("Search coarse_count must be at least 3")
    _integer(search.get("refinement_rounds", 2), "search refinement rounds", nonnegative=True)
    if _integer(search.get("refinement_subdivisions", 4), "search refinement subdivisions", positive=True) < 2:
        raise ValueError("Search refinement_subdivisions must be at least 2")
    _number(search.get("boundary_expansion_factor", 10.0), "boundary expansion factor", positive=True)
    if float(search.get("boundary_expansion_factor", 10.0)) <= 1:
        raise ValueError("Boundary expansion factor must exceed one")
    _integer(search.get("max_boundary_expansions", 4), "maximum boundary expansions", nonnegative=True)
    for key in ("flat_relative_tolerance", "flat_absolute_tolerance", "flat_minimum_log10_span"):
        _number(search.get(key, {"flat_relative_tolerance": 0.02, "flat_absolute_tolerance": 1e-6, "flat_minimum_log10_span": 0.15}[key]), f"search.{key}", nonnegative=True)
    _integer(search.get("bootstrap_samples", 2000), "bootstrap samples", positive=True)
    confidence = _number(search.get("bootstrap_confidence", 0.95), "bootstrap confidence", positive=True)
    if confidence >= 1:
        raise ValueError("Bootstrap confidence must be less than one")
    _integer(search.get("bootstrap_seed", 0), "bootstrap seed", nonnegative=True)

    commands = _required(manifest, "commands", "manifest")
    if not isinstance(commands, dict):
        raise ValueError("manifest.commands must be a JSON object")
    python_command = str(_required(commands, "python", "manifest.commands"))
    simulator_template = _validate_command_template(
        _required(commands, "simulator", "manifest.commands"),
        (
            "python", "inputs.simulator.path", "simulator_arguments", "simulation_dir",
            "youngs_modulus_pa", "inputs.sequence.episode_dir", "window.replay_start_frame",
            "window.end_frame", "inputs.initial_particles.path", "inputs.reconstruction_metadata.path",
            "inputs.calibration.path", "inputs.geometry.path",
        ),
        "simulator",
    )
    evaluator_template = _validate_command_template(
        _required(commands, "evaluator", "manifest.commands"),
        (
            "python", "inputs.evaluator.path", "evaluator_arguments", "inputs.sequence.episode_dir",
            "simulation_metadata", "inputs.calibration.path", "evaluation_dir",
        ),
        "evaluator",
    )

    execution = _required(manifest, "execution", "manifest")
    if not isinstance(execution, dict):
        raise ValueError("manifest.execution must be a JSON object")
    cache_dir = _resolve(str(_required(execution, "cache_dir", "manifest.execution")), manifest_path)
    result_path = _resolve(str(_required(execution, "result_path", "manifest.execution")), manifest_path)
    timeout = _number(_required(execution, "subprocess_timeout_s", "manifest.execution"), "subprocess timeout", positive=True)

    normalized = json.loads(json.dumps(manifest))
    normalized["inputs"] = resolved_inputs
    normalized["windows"] = validated_windows
    normalized["fixed_parameters"]["simulator_arguments"] = simulator_arguments
    normalized["fixed_parameters"]["evaluator_arguments"] = evaluator_arguments
    normalized["commands"] = {
        "python": python_command,
        "simulator": simulator_template,
        "evaluator": evaluator_template,
    }
    normalized["execution"] = {
        **execution,
        "cache_dir": str(cache_dir),
        "result_path": str(result_path),
        "subprocess_timeout_s": timeout,
    }
    normalized["search"] = {**search, "default_youngs_modulus_pa": default}
    return normalized


def _lookup(context: Mapping[str, Any], dotted: str) -> Any:
    value: Any = context
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Command template token {{{dotted}}} has no value")
        value = value[part]
    return value


def render_command(template: Iterable[str], context: Mapping[str, Any]) -> list[str]:
    argv: list[str] = []
    for item in template:
        match = TOKEN_PATTERN.match(item)
        if not match:
            if "{" in item or "}" in item:
                raise ValueError(f"Command template substitutions must occupy a complete argument: {item!r}")
            argv.append(item)
            continue
        value = _lookup(context, match.group(1))
        if isinstance(value, list):
            argv.extend(str(element) for element in value)
        else:
            argv.append(str(value))
    return argv


def _run_command(argv: list[str], stdout_path: Path, stderr_path: Path, timeout_s: float) -> dict[str, Any]:
    start = time.monotonic()
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=timeout_s, check=False)
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "duration_s": time.monotonic() - start,
            "failure_reason": None if completed.returncode == 0 else f"command exited with status {completed.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {
            "argv": argv,
            "returncode": None,
            "duration_s": time.monotonic() - start,
            "failure_reason": f"command exceeded timeout of {timeout_s:g} seconds",
        }
    except OSError as exc:
        return {
            "argv": argv,
            "returncode": None,
            "duration_s": time.monotonic() - start,
            "failure_reason": f"could not execute command: {exc}",
        }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Missing {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description.capitalize()} must contain a JSON object")
    return value


def _resolve_evaluation_array(value: str, evaluation_path: Path) -> Path:
    path = Path(value)
    candidates = (path, evaluation_path.parent / path, evaluation_path.parent / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"Cannot resolve evaluator array {value!r} from {evaluation_path}")


def _loss_kwargs(loss_settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "weights": loss_settings["weights"],
        "depth_scale_m": loss_settings["depth_scale_m"],
        "distance_scale_m": loss_settings["distance_scale_m"],
        "huber_delta": loss_settings["huber_delta"],
        "min_common_pixels": loss_settings["min_common_pixels"],
        "min_observed_pixels": loss_settings["min_observed_pixels"],
        "min_simulation_pixels": loss_settings["min_simulation_pixels"],
        "min_observed_points": loss_settings["min_observed_points"],
        "min_simulation_points": loss_settings["min_simulation_points"],
        "nearest_chunk_size": loss_settings.get("nearest_chunk_size", 1024),
    }


def score_evaluation_artifact(
    evaluation_path: Path,
    loss_settings: Mapping[str, Any],
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict[str, Any]:
    if start_frame is None and end_frame is None:
        scoring_start = None
        scoring_end = None
    elif start_frame is None or end_frame is None:
        raise ValueError("Both scoring frame bounds must be provided together")
    else:
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError("Scoring frame bounds must satisfy 0 <= start_frame < end_frame")
        scoring_start = start_frame
        scoring_end = end_frame
    report = _load_json(evaluation_path, "dynamic evaluation")
    if report.get("benchmark") != "dynamic-topview-proxy-replay/v1":
        raise ValueError("Dynamic evaluation benchmark version is not supported")
    frames = report.get("frames")
    camera = report.get("camera", {})
    if not isinstance(frames, list) or not frames:
        raise ValueError("Dynamic evaluation contains no frames")
    if not isinstance(camera, dict):
        raise ValueError("Dynamic evaluation camera metadata must be an object")
    initial_row = frames[0]
    if float(initial_row.get("time_s", -1)) != 0 or "arrays" not in initial_row:
        raise ValueError("First evaluation frame must be the recorded and simulated initial state")
    initial_npz_path = _resolve_evaluation_array(initial_row["arrays"], evaluation_path)
    with np.load(initial_npz_path) as initial_arrays:
        observed_initial_depth = initial_arrays["real_depth"].copy()
        observed_initial_valid = initial_arrays["real_valid"].astype(bool)
        simulation_initial_depth = initial_arrays["sim_depth"].copy()
        simulation_initial_valid = initial_arrays["sim_valid"].astype(bool)
    initial_simulation_points = visible_optical_points(simulation_initial_depth, simulation_initial_valid, camera)
    frame_results = []
    frozen_results = []
    frame_records = []
    for frame in frames[1:]:
        source_frame = frame.get("source_frame")
        if scoring_start is not None and scoring_end is not None:
            if isinstance(source_frame, bool) or not isinstance(source_frame, int):
                raise ValueError("Every scored evaluator frame must have an integer source_frame")
            if source_frame < scoring_start or source_frame > scoring_end:
                continue
        if frame.get("status") != "paired":
            invalid = {
                "valid": False,
                "failure_reason": f"evaluator frame status is {frame.get('status')!r}",
                "components": {name: None for name in LOSS_COMPONENTS},
                "weighted_total": None,
            }
            frame_results.append(invalid)
            frozen_results.append(invalid)
            frame_records.append({"source_frame": frame.get("source_frame"), **invalid})
            continue
        arrays_path = _resolve_evaluation_array(frame["arrays"], evaluation_path)
        with np.load(arrays_path) as arrays:
            observed_depth = arrays["real_depth"].copy()
            observed_valid = arrays["real_valid"].astype(bool)
            simulation_depth = arrays["sim_depth"].copy()
            simulation_valid = arrays["sim_valid"].astype(bool)
        observed_points = visible_optical_points(observed_depth, observed_valid, camera)
        simulation_points = visible_optical_points(simulation_depth, simulation_valid, camera)
        kwargs = _loss_kwargs(loss_settings)
        loss = partial_view_loss(
            observed_depth,
            observed_valid,
            simulation_depth,
            simulation_valid,
            observed_initial_depth,
            observed_initial_valid,
            simulation_initial_depth,
            simulation_initial_valid,
            observed_points,
            simulation_points,
            **kwargs,
        )
        frozen = partial_view_loss(
            observed_depth,
            observed_valid,
            simulation_initial_depth,
            simulation_initial_valid,
            observed_initial_depth,
            observed_initial_valid,
            simulation_initial_depth,
            simulation_initial_valid,
            observed_points,
            initial_simulation_points,
            **kwargs,
        )
        frame_results.append(loss)
        frozen_results.append(frozen)
        frame_records.append(
            {
                "source_frame": frame.get("source_frame"),
                "time_s": frame.get("time_s"),
                "status": frame.get("status"),
                **loss,
            }
        )
    require_all = bool(loss_settings.get("require_all_frames", True))
    aggregate = aggregate_loss_results(frame_results, require_all=require_all, minimum_valid=1)
    frozen_aggregate = aggregate_loss_results(frozen_results, require_all=require_all, minimum_valid=1)
    return {
        **aggregate,
        "frames": frame_records,
        "scored_source_frame_range": (
            None if start_frame is None else [start_frame, end_frame]
        ),
        "frozen_baseline": frozen_aggregate,
        "source_evaluation": str(evaluation_path.resolve()),
    }


def _metadata_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            _metadata_value_matches(found, wanted) for found, wanted in zip(actual, expected)
        )
    return actual == expected


def _verify_simulation_metadata(
    metadata_path: Path,
    candidate_pa: float,
    window: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    metadata = _load_json(metadata_path, "simulation metadata")
    parameters = metadata.get("parameters", {})
    actual_candidate = _number(parameters.get("youngs_modulus"), "simulation Young's modulus", positive=True)
    if not math.isclose(actual_candidate, candidate_pa, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Simulation metadata Young's modulus {actual_candidate} does not match candidate {candidate_pa}")
    replay = metadata.get("replay", {})
    expected = (
        window["replay_start_frame"],
        window["end_frame"],
        fingerprints["sequence"]["fingerprint"],
    )
    actual = (replay.get("source_start_frame"), replay.get("source_end_frame"), replay.get("sequence_fingerprint"))
    if actual != expected:
        raise ValueError("Simulation replay metadata does not match the requested window and sequence")
    for flag, configured in manifest["fixed_parameters"]["simulator_arguments"].items():
        parameter_name = flag[2:].replace("-", "_")
        expected_value = configured
        if flag == "--no-publish-dough-center":
            parameter_name = "publish_dough_center"
            expected_value = not bool(configured)
        elif flag == "--initial-particles-raw-scene-coordinates":
            expected_value = True
        actual_value = parameters.get(parameter_name)
        if not _metadata_value_matches(actual_value, expected_value):
            raise ValueError(
                f"Simulation metadata parameter {parameter_name!r} does not match fixed argument {flag}"
            )
    input_parameters = {
        "initial_particles": manifest["inputs"]["initial_particles"]["path"],
        "initial_particles_metadata": manifest["inputs"]["reconstruction_metadata"]["path"],
        "initial_particles_calibration": manifest["inputs"]["calibration"]["path"],
        "tool_geometry": manifest["inputs"]["geometry"]["path"],
        "replay_episode": manifest["inputs"]["sequence"]["episode_dir"],
    }
    for parameter_name, expected_path in input_parameters.items():
        actual_path = parameters.get(parameter_name)
        if actual_path is None or Path(str(actual_path)).resolve() != Path(expected_path).resolve():
            raise ValueError(f"Simulation metadata input {parameter_name!r} does not match the manifest")
    geometry_metadata = metadata.get("tool_geometry") or {}
    if geometry_metadata.get("fingerprint") != fingerprints["geometry"]["sha256"]:
        raise ValueError("Simulation tool geometry fingerprint does not match the manifest")
    recorded_mass = float(manifest["fixed_parameters"]["recorded_setup"]["dough_mass_kg"])
    if not _metadata_value_matches(metadata.get("total_mass_kg"), recorded_mass):
        raise ValueError("Simulation total mass does not match the recorded dough mass")


def _cache_payload(manifest: Mapping[str, Any], fingerprints: Mapping[str, Any], window: Mapping[str, Any], candidate_pa: float) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA,
        "manifest_schema": manifest["schema"],
        "manifest_name": manifest["name"],
        "fingerprints": fingerprints,
        "fixed_parameters": manifest["fixed_parameters"],
        "commands": manifest["commands"],
        "loss": manifest["loss"],
        "window": dict(window),
        "youngs_modulus_pa": candidate_pa,
    }


def execute_candidate_window(
    manifest: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    window: Mapping[str, Any],
    candidate_pa: float,
    *,
    retry_failures: bool = False,
) -> dict[str, Any]:
    payload = _cache_payload(manifest, fingerprints, window, candidate_pa)
    cache_key = canonical_json_hash(payload)
    cache_dir = Path(manifest["execution"]["cache_dir"]) / cache_key
    expected_metadata = {"schema": CACHE_SCHEMA, "cache_key": cache_key, "payload_hash": cache_key}
    completion = validate_cache_directory(cache_dir, expected_metadata, REQUIRED_CACHE_FILES)
    if completion["complete"]:
        loss = _load_json(cache_dir / "loss.json", "cached loss")
        return {
            "window": window["name"],
            "split": window["split"],
            "youngs_modulus_pa": candidate_pa,
            "status": "complete",
            "cached": True,
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
            "loss": loss,
            "failure_reason": None,
        }
    previous_metadata = None
    metadata_path = cache_dir / "cache_metadata.json"
    if metadata_path.is_file():
        try:
            previous_metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            previous_metadata = None
    if previous_metadata and previous_metadata.get("status") == "failed" and not retry_failures:
        return {
            "window": window["name"],
            "split": window["split"],
            "youngs_modulus_pa": candidate_pa,
            "status": "failed",
            "cached": True,
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
            "loss": None,
            "failure_reason": previous_metadata.get("failure_reason", "previous cached execution failed"),
        }
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    simulation_dir = cache_dir / "simulation"
    evaluation_dir = cache_dir / "evaluation"
    simulation_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)
    context = {
        "python": manifest["commands"]["python"],
        "inputs": manifest["inputs"],
        "simulator_arguments": compile_cli_arguments(manifest["fixed_parameters"]["simulator_arguments"]),
        "evaluator_arguments": compile_cli_arguments(manifest["fixed_parameters"]["evaluator_arguments"]),
        "simulation_dir": str(simulation_dir),
        "simulation_metadata": str(simulation_dir / "camera_parameters.json"),
        "evaluation_dir": str(evaluation_dir),
        "youngs_modulus_pa": f"{candidate_pa:.12g}",
        "window": window,
    }
    simulator_argv = render_command(manifest["commands"]["simulator"], context)
    evaluator_argv = render_command(manifest["commands"]["evaluator"], context)
    metadata: dict[str, Any] = {
        **expected_metadata,
        "status": "running",
        "payload": payload,
        "candidate": {"youngs_modulus_pa": candidate_pa, "window": dict(window)},
        "commands": {},
        "required_files": list(REQUIRED_CACHE_FILES),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    timeout = float(manifest["execution"]["subprocess_timeout_s"])
    simulator_run = _run_command(simulator_argv, cache_dir / "simulator.stdout.log", cache_dir / "simulator.stderr.log", timeout)
    metadata["commands"]["simulator"] = simulator_run
    failure = simulator_run["failure_reason"]
    if failure is None:
        try:
            _verify_simulation_metadata(
                simulation_dir / "camera_parameters.json", candidate_pa, window, fingerprints, manifest
            )
        except (OSError, ValueError) as exc:
            failure = str(exc)
    if failure is None:
        evaluator_run = _run_command(evaluator_argv, cache_dir / "evaluator.stdout.log", cache_dir / "evaluator.stderr.log", timeout)
        metadata["commands"]["evaluator"] = evaluator_run
        failure = evaluator_run["failure_reason"]
    if failure is None:
        try:
            loss = score_evaluation_artifact(
                evaluation_dir / "dynamic_topview_metrics.json",
                manifest["loss"],
                start_frame=window["start_frame"],
                end_frame=window["end_frame"],
            )
            if not loss["valid"]:
                failure = "partial-view loss is invalid: " + str(loss["failure_reason"])
            (cache_dir / "loss.json").write_text(json.dumps(loss, indent=2, allow_nan=False))
        except (OSError, ValueError, KeyError) as exc:
            loss = None
            failure = f"could not score dynamic evaluation: {exc}"
    else:
        loss = None
    metadata["status"] = "complete" if failure is None else "failed"
    metadata["failure_reason"] = failure
    metadata["completed_at_unix_s"] = time.time()
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    if failure is None:
        completion = validate_cache_directory(cache_dir, expected_metadata, REQUIRED_CACHE_FILES)
        if not completion["complete"]:
            failure = "cache did not pass completion validation: " + str(completion["failure_reason"])
            metadata["status"] = "failed"
            metadata["failure_reason"] = failure
            metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return {
        "window": window["name"],
        "split": window["split"],
        "youngs_modulus_pa": candidate_pa,
        "status": "complete" if failure is None else "failed",
        "cached": False,
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "loss": loss if failure is None else None,
        "failure_reason": failure,
    }


def _fingerprint_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(json.dumps(manifest["inputs"]))
    tool_arguments = {
        key: value
        for key, value in manifest["fixed_parameters"]["simulator_arguments"].items()
        if key.startswith("--tool-")
    }
    record["tool"] = {
        "fingerprint": canonical_json_hash(
            {
                "arguments": tool_arguments,
                "declared_geometry_fingerprint": manifest["fixed_parameters"]["recorded_setup"]["tool_geometry_fingerprint"],
            }
        ),
        "arguments": tool_arguments,
    }
    record["fixed_parameters"] = {"fingerprint": canonical_json_hash(manifest["fixed_parameters"])}
    return record


def _candidate_record(
    candidate_pa: float,
    manifest: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    cache: dict[tuple[float, str], dict[str, Any]],
    windows: list[Mapping[str, Any]],
    retry_failures: bool,
) -> dict[str, Any]:
    window_executions = {}
    window_losses = {}
    failures = []
    for window in windows:
        key = (candidate_pa, window["name"])
        if key not in cache:
            cache[key] = execute_candidate_window(
                manifest, fingerprints, window, candidate_pa, retry_failures=retry_failures
            )
        execution = cache[key]
        window_executions[window["name"]] = execution
        if execution["status"] == "complete":
            window_losses[window["name"]] = execution["loss"]
        else:
            failures.append(f"{window['name']}: {execution['failure_reason']}")
            window_losses[window["name"]] = {
                "valid": False,
                "failure_reason": execution["failure_reason"],
                "components": {name: None for name in LOSS_COMPONENTS},
                "weighted_total": None,
            }
    split = windows[0]["split"] if windows else "training"
    aggregate = aggregate_named_windows(window_losses, manifest["windows"], split)
    return {
        "youngs_modulus_pa": candidate_pa,
        "status": "complete" if aggregate.get("valid") else "failed",
        "aggregate": aggregate,
        "windows": window_losses,
        "executions": window_executions,
        "failure_reason": None if aggregate.get("valid") else "; ".join(failures) or aggregate.get("failure_reason"),
    }


def run_calibration(manifest: Mapping[str, Any], *, retry_failures: bool = False) -> tuple[dict[str, Any], int]:
    fingerprints = _fingerprint_record(manifest)
    manifest_hash = canonical_json_hash({key: value for key, value in manifest.items() if key != "execution"})
    training_windows = [window for window in manifest["windows"] if window["split"] == "training"]
    validation_windows = [window for window in manifest["windows"] if window["split"] == "validation"]
    search = manifest["search"]
    bounds = [float(search["initial_min_pa"]), float(search["initial_max_pa"])]
    hard_bounds = [float(search["hard_min_pa"]), float(search["hard_max_pa"])]
    default = float(search["default_youngs_modulus_pa"])
    execution_cache: dict[tuple[float, str], dict[str, Any]] = {}
    records: dict[float, dict[str, Any]] = {}

    def evaluate_candidates(candidates: Iterable[float]) -> None:
        for candidate in sorted(set(float(value) for value in candidates)):
            if candidate not in records:
                print(f"Evaluate training candidate E={candidate:.12g} Pa", flush=True)
                records[candidate] = _candidate_record(
                    candidate, manifest, fingerprints, execution_cache, training_windows, retry_failures
                )

    evaluate_candidates(logarithmic_candidates(bounds[0], bounds[1], int(search["coarse_count"]), include=[default]))
    boundary_history = []
    boundary_unresolved = False
    for _ in range(int(search.get("max_boundary_expansions", 4)) + 1):
        totals = {
            candidate: record["aggregate"]["weighted_total"] if record["aggregate"].get("valid") else None
            for candidate, record in records.items()
        }
        best = select_best_candidate(totals)
        if not best["valid"]:
            break
        expansion = boundary_expansion(
            bounds,
            best["youngs_modulus_pa"],
            hard_bounds,
            float(search.get("boundary_expansion_factor", 10.0)),
        )
        boundary_history.append(expansion)
        if expansion["boundary"] is None:
            break
        if not expansion["expanded"]:
            boundary_unresolved = True
            break
        if len(boundary_history) > int(search.get("max_boundary_expansions", 4)):
            boundary_unresolved = True
            break
        bounds = expansion["bounds"]
        evaluate_candidates(logarithmic_candidates(bounds[0], bounds[1], int(search["coarse_count"]), include=[default]))

    refinement_history = []
    for _ in range(int(search.get("refinement_rounds", 2))):
        totals = {
            candidate: record["aggregate"]["weighted_total"] if record["aggregate"].get("valid") else None
            for candidate, record in records.items()
        }
        best = select_best_candidate(totals)
        if not best["valid"]:
            break
        proposed = refinement_candidates(
            records.keys(), best["youngs_modulus_pa"], int(search.get("refinement_subdivisions", 4))
        )
        refinement_history.append(proposed)
        if not proposed:
            break
        evaluate_candidates(proposed)

    totals = {
        candidate: record["aggregate"]["weighted_total"] if record["aggregate"].get("valid") else None
        for candidate, record in records.items()
    }
    best = select_best_candidate(totals)
    flat = flat_minimum_diagnostics(
        totals,
        relative_tolerance=float(search.get("flat_relative_tolerance", 0.02)),
        absolute_tolerance=float(search.get("flat_absolute_tolerance", 1e-6)),
        minimum_log10_span=float(search.get("flat_minimum_log10_span", 0.15)),
    )
    final_boundary = None
    if best["valid"]:
        final_boundary = boundary_expansion(
            bounds, best["youngs_modulus_pa"], hard_bounds, float(search.get("boundary_expansion_factor", 10.0))
        )
        boundary_unresolved = boundary_unresolved or final_boundary["boundary"] is not None
    rejection_reasons = []
    if not best["valid"]:
        rejection_reasons.append(best["failure_reason"])
    if boundary_unresolved:
        rejection_reasons.append("best training loss remains on a searched range boundary")
    if flat.get("flat"):
        rejection_reasons.append(str(flat["reason"]))
    accepted = not rejection_reasons

    bootstrap = None
    if best["valid"]:
        candidate_window_losses = {}
        training_names = [window["name"] for window in training_windows]
        for candidate, record in records.items():
            if record["aggregate"].get("valid"):
                candidate_window_losses[candidate] = [
                    float(record["windows"][name]["weighted_total"]) for name in training_names
                ]
        if candidate_window_losses:
            bootstrap = whole_window_bootstrap_interval(
                candidate_window_losses,
                window_weights=[float(window["weight"]) for window in training_windows],
                samples=int(search.get("bootstrap_samples", 2000)),
                confidence=float(search.get("bootstrap_confidence", 0.95)),
                seed=int(search.get("bootstrap_seed", 0)),
            )

    validation_record = None
    if best["valid"]:
        print(f"Evaluate held-out windows at E={best['youngs_modulus_pa']:.12g} Pa without refitting", flush=True)
        validation_record = _candidate_record(
            best["youngs_modulus_pa"], manifest, fingerprints, execution_cache, validation_windows, retry_failures
        )

    valid_candidates = sorted(candidate for candidate, record in records.items() if record["aggregate"].get("valid"))
    soft = records[valid_candidates[0]] if valid_candidates else None
    stiff = records[valid_candidates[-1]] if valid_candidates else None
    default_record = records.get(default)
    best_candidate_pa = float(best["youngs_modulus_pa"]) if best["valid"] else None
    best_record = records.get(best_candidate_pa) if best_candidate_pa is not None else None
    frozen_windows = {}
    if default_record:
        for name, window_loss in default_record["windows"].items():
            frozen = window_loss.get("frozen_baseline")
            if frozen is not None:
                frozen_windows[name] = frozen
    frozen_training = aggregate_named_windows(frozen_windows, manifest["windows"], "training") if frozen_windows else None

    validation_valid = bool(
        validation_record is not None and validation_record["aggregate"].get("valid")
    )
    validation_failure_reason = None
    if best["valid"] and not validation_valid:
        validation_failure_reason = (
            validation_record.get("failure_reason") if validation_record is not None
            else "held-out validation was not evaluated"
        )
    status = (
        "accepted" if accepted and validation_valid
        else ("needs_review" if best["valid"] else "failed")
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "interpretation": "Effective Young's modulus for this recorded geometry, reconstruction, tool replay, simulator model, and fixed parameter set; not a universal dough constant.",
        "manifest": {
            "name": manifest["name"],
            "schema": manifest["schema"],
            "canonical_hash": manifest_hash,
        },
        "fingerprints": fingerprints,
        "fixed_parameters": manifest["fixed_parameters"],
        "windows": manifest["windows"],
        "loss": manifest["loss"],
        "search": {
            "configured": search,
            "final_bounds_pa": bounds,
            "hard_bounds_pa": hard_bounds,
            "boundary_history": boundary_history,
            "final_boundary_diagnostic": final_boundary,
            "refinement_candidates_pa": refinement_history,
            "flat_minimum_diagnostic": flat,
            "selection": {
                **best,
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
            },
            "bootstrap_interval": bootstrap,
            "candidates": [records[candidate] for candidate in sorted(records)],
        },
        "validation": {
            "fit_performed": False,
            "valid": validation_valid,
            "failure_reason": validation_failure_reason,
            "candidate_youngs_modulus_pa": best.get("youngs_modulus_pa"),
            "result": validation_record,
        },
        "baselines": {
            "best": best_record,
            "default_e_2000_pa": default_record,
            "softest_successful": soft,
            "stiffest_successful": stiff,
            "frozen_training": frozen_training,
        },
        "failures": [
            {
                "youngs_modulus_pa": candidate,
                "reason": record["failure_reason"],
            }
            for candidate, record in sorted(records.items())
            if not record["aggregate"].get("valid")
        ],
    }
    exit_code = 0 if status == "accepted" else (2 if best["valid"] else 1)
    return result, exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Versioned material-calibration manifest JSON")
    parser.add_argument("--output", type=Path, help="Override execution.result_path")
    parser.add_argument("--retry-failures", action="store_true", help="Re-run failed cache entries")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs and print the manifest hash without running Taichi")
    args = parser.parse_args()
    try:
        manifest_path = args.manifest.resolve()
        manifest = validate_manifest(_load_json(manifest_path, "manifest"), manifest_path)
        if args.output:
            manifest["execution"]["result_path"] = str(args.output.resolve())
        if args.validate_only:
            print(canonical_json_hash({key: value for key, value in manifest.items() if key != "execution"}))
            return
        result, exit_code = run_calibration(manifest, retry_failures=args.retry_failures)
        output_path = Path(manifest["execution"]["result_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(f"Wrote {output_path}")
        selection = result["search"]["selection"]
        if result["status"] == "accepted":
            print(f"Selected effective Young's modulus: {selection['youngs_modulus_pa']:.12g} Pa")
        elif not selection["accepted"]:
            print("Selection not accepted: " + "; ".join(selection["rejection_reasons"]), file=sys.stderr)
        else:
            print(
                "Training selection completed, but held-out validation is invalid: "
                + str(result["validation"]["failure_reason"]),
                file=sys.stderr,
            )
        raise SystemExit(exit_code)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

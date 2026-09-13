"""Verified recorded inputs for the isolated differentiable experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import EXPERIMENT_ROOT, ExperimentConfig, FrameWindow, canonical_hash, file_sha256, verify_input_paths
from .reference_adapter import get_reference_modules
from .replay import RecordedControls, ReplayFrame, observation_schedule
from .state import ParticleState, SDFData, SimulationConfig


def validate_initial_stencils(particles, config: SimulationConfig):
    """Use the selected solver's base rounding and logical grid bounds."""
    positions = np.asarray(particles, dtype=config.numpy_dtype)
    scaled = positions * config.grid - 0.5
    corrected = config.physics_version == "corrected-v1"
    base = (np.floor(scaled) if corrected else np.trunc(scaled)).astype(np.int64)
    if np.any(base < (-1 if corrected else 0)) or np.any(base + 2 >= config.grid):
        raise ValueError("Initial particle has an unsafe MPM grid stencil")


def array_fingerprint(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _new_or_identical(path: Path, content: bytes):
    """Generated input copies may be reused only when their bytes match."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"Existing generated input differs; refusing to replace {path}")


def verify_source_manifest(config: ExperimentConfig, records: dict, sequence_fingerprint: str):
    path = config.paths.get("source_manifest")
    if path is None:
        return None
    document = json.loads(path.read_text())
    if document.get("schema") != "taichidough/material-calibration-manifest/v1":
        raise ValueError("Unsupported source material manifest")
    supplied = document.get("inputs", {})
    for original, current in (("geometry", "tool_geometry"), ("initial_particles", "initial_particles"),
                              ("reconstruction_metadata", "reconstruction_metadata"), ("calibration", "calibration")):
        expected = supplied.get(original, {}).get("sha256")
        if not expected or expected != records[current]["sha256"]:
            raise ValueError(f"Explicit {current} does not match source-manifest content hash")
    if supplied.get("sequence", {}).get("fingerprint") != sequence_fingerprint:
        raise ValueError("Recorded sequence content differs from the source manifest")
    old_mass = document.get("fixed_parameters", {}).get("recorded_setup", {}).get("dough_mass_kg")
    if old_mass is None or not np.isclose(float(old_mass), config.mass_kg, rtol=1e-12, atol=0):
        raise ValueError("Measured mass does not match the source manifest")
    return {"path": str(path), "sha256": records["source_manifest"]["sha256"],
            "recorded_paths": {k: supplied.get(k) for k in ("geometry", "initial_particles", "reconstruction_metadata", "calibration", "sequence")}}


def resolved_reconstruction(config: ExperimentConfig, records: dict, sequence_fingerprint: str):
    """Make a path-only metadata copy after content has been verified."""
    original = config.paths["reconstruction_metadata"]
    metadata = json.loads(original.read_text())
    if metadata.get("frame") != 0 or metadata.get("pointclouds_name") != "pointclouds_interpolated.pt":
        raise ValueError("Initial reconstruction must describe source frame 0 of pointclouds_interpolated.pt")
    original_episode = Path(str(metadata.get("episode_dir", ""))).expanduser().resolve()
    target_episode = config.paths["episode"]
    recorded_particle = (metadata.get("outputs") or {}).get("sampled_particles_xyz")
    target_particle = config.paths["initial_particles"]
    if recorded_particle:
        recorded_path = Path(recorded_particle).expanduser()
        if not recorded_path.is_absolute():
            recorded_path = original.parent / recorded_path
        particle_moved = recorded_path.resolve() != target_particle
    else:
        particle_moved = False
    moved = original_episode != target_episode or particle_moved
    if moved and config.paths.get("source_manifest") is None:
        if not config.expected_sequence_fingerprint or "reconstruction_metadata" not in config.expected_sha256 or "initial_particles" not in config.expected_sha256:
            raise ValueError("Relocated reconstruction requires a verified source manifest or explicit sequence, metadata and particle hashes")
        if config.expected_sequence_fingerprint != sequence_fingerprint:
            raise ValueError("Relocated episode fingerprint mismatch")
    if not moved:
        return original, {"original_path": str(original), "original_sha256": records["reconstruction_metadata"]["sha256"],
                          "resolved_path": str(original), "resolved_sha256": records["reconstruction_metadata"]["sha256"], "path_changes": {}}
    previous = {"episode_dir": metadata.get("episode_dir"), "sampled_particles_xyz": recorded_particle}
    metadata["episode_dir"] = str(target_episode)
    metadata.setdefault("outputs", {})["sampled_particles_xyz"] = str(target_particle)
    content = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode()
    generated_hash = hashlib.sha256(content).hexdigest()
    path = EXPERIMENT_ROOT / "runs" / "prepared_inputs" / generated_hash / "reconstruction_metadata.json"
    _new_or_identical(path, content)
    return path, {"original_path": str(original), "original_sha256": records["reconstruction_metadata"]["sha256"],
                  "resolved_path": str(path), "resolved_sha256": generated_hash,
                  "path_changes": {"before": previous, "after": {"episode_dir": str(target_episode), "sampled_particles_xyz": str(target_particle)}}}


def scaled_camera(calibration, width: int, height: int):
    """Scale the calibrated pixel coordinate system, preserving metric extrinsics."""
    original = calibration.camera
    values = dict(original)
    scale_x, scale_y = width / original["width"], height / original["height"]
    for key in ("fx", "cx"):
        values[key] = float(original[key]) * scale_x
    for key in ("fy", "cy"):
        values[key] = float(original[key]) * scale_y
    values["width"], values["height"] = int(width), int(height)
    values["scene_from_camera"] = np.asarray(calibration.scene_from_camera, dtype=np.float64).tolist()
    return values


def camera_from_calibration(calibration, width: int, height: int):
    """Retain both calibrated clipping planes in the training renderer."""
    from .renderer import Camera

    values = scaled_camera(calibration, width, height)
    topview = get_reference_modules().topview
    return Camera(width, height, values["fx"], values["fy"], values["cx"], values["cy"],
                  topview.rigid_inverse(calibration.scene_from_camera),
                  near_m=values["zNear"], far_m=values["zFar"])


def visible_optical_points(depth, valid, camera):
    """Reconstruct pixel centers using the strict evaluator's +0.5 convention."""
    rows, columns = np.nonzero(valid)
    z = np.asarray(depth)[rows, columns].astype(np.float64)
    x = (columns + 0.5 - camera["cx"]) * z / camera["fx"]
    y = (rows + 0.5 - camera["cy"]) * z / camera["fy"]
    return np.column_stack((x, y, z)).astype(np.float32)


def collision_inputs(config, records, sequence_names, simulator, build_sdf):
    if config.simulation.get("tool_collision", "none") != "sdf":
        return None, None
    manifest = json.loads(config.paths["collision_manifest"].read_text())
    if manifest.get("schema") != "taichidough/tool-collision-meshes/v1":
        raise ValueError("Unsupported collision-solid manifest schema")
    rows = manifest.get("tools")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Collision-solid manifest must describe exactly two tools")
    by_name = {row.get("name"): row for row in rows}
    asset_settings = {
        "UR5e_spathla": ("ur_collision_mesh", simulator.UR_TOOL_VISUAL_ORIGIN, simulator.UR_TOOL_VISUAL_RPY),
        "gen3_spathla": ("kinova_collision_mesh", simulator.KINOVA_TOOL_VISUAL_ORIGIN, simulator.KINOVA_TOOL_VISUAL_RPY),
    }
    if set(by_name) != set(asset_settings) or set(sequence_names) != set(asset_settings):
        raise ValueError("Collision solids and recorded tool stream names differ")
    volumes = []
    asset_records = []
    for name in sequence_names:
        key, origin, rpy = asset_settings[name]
        row = by_name[name]
        if row.get("collision_sha256") != records[key]["sha256"]:
            raise ValueError(f"Collision mesh content for {name} does not match its manifest")
        if row.get("units") != "millimetres" or row.get("coordinate_frame") != "raw_stl_visual":
            raise ValueError("Collision manifest must declare millimetres in raw_stl_visual coordinates")
        topology = row.get("topology", {})
        if topology.get("boundary_edges") != 0 or topology.get("nonmanifold_edges") != 0:
            raise ValueError(f"Collision mesh for {name} is not declared closed and manifold")
        asset_records.append({"name": name, **records[key], "visual_origin_m": np.asarray(origin).tolist(),
                              "visual_rpy_rad": np.asarray(rpy).tolist()})
        if build_sdf:
            vertices, _ = simulator.load_binary_stl(config.paths[key], config.tool_mesh_scale)
            transformed = simulator.mesh_in_tool_frame(vertices, origin, simulator.rpy_to_matrix(rpy))
            volumes.append(simulator.build_mesh_sdf(transformed, config.tool_sdf_resolution,
                                                    config.simulation.get("tool_contact_padding", 1 / 384)))
    sdf = None
    derived = None
    if build_sdf:
        sdf = SDFData(np.stack([v[0] for v in volumes]), np.stack([v[1] for v in volumes]),
                      np.stack([v[2] for v in volumes]), np.stack([v[3] for v in volumes]))
        sdf.validate()
        derived = {name: array_fingerprint(getattr(sdf, name)) for name in ("distances", "gradients", "minimums", "spacings")}
    return sdf, {"manifest": records["collision_manifest"], "assets_in_stream_order": asset_records,
                 "resolution": config.tool_sdf_resolution, "mesh_scale": config.tool_mesh_scale,
                 "padding_m": config.simulation.get("tool_contact_padding", 1 / 384), "derived_arrays": derived}


@dataclass
class PreparedExperiment:
    config: ExperimentConfig
    simulation_config: SimulationConfig
    initial_state: ParticleState
    parameters: dict[str, float]
    sdf: SDFData | None
    controls: RecordedControls
    observations: list[Any]
    camera: Any
    loss_config: Any
    provenance: dict[str, Any]
    total_steps: int
    frames: tuple[ReplayFrame, ...]
    split: str
    scored_frames: tuple[int, ...]
    calibration: Any
    sequence: Any
    tool_geometry: Any
    reconstruction_info: dict[str, Any]
    mass_properties: dict[str, Any]

    @property
    def fingerprint(self):
        return canonical_hash(self.provenance)

    @property
    def end_frame(self):
        return self.frames[-1].source_frame

    @property
    def objective_frames(self):
        return tuple(int(observation.frame_index) for observation in self.observations)

    def summary(self):
        result = {"name": self.config.name, "split": self.split, "particle_count": self.simulation_config.n_particles,
                "total_steps": self.total_steps, "replay_source_frames": [0, self.end_frame],
                "scored_frames": list(self.scored_frames), "observation_count": len(self.observations),
                "parameters": dict(self.parameters), "fit_parameters": list(self.config.fit_parameters),
                "mass": self.mass_properties, "training_camera": {"width": self.camera.width, "height": self.camera.height},
                "sdf_built": self.sdf is not None, "fingerprint": self.fingerprint, "provenance": self.provenance}
        if "objective_frames" in self.provenance:
            result.update(objective_frames=list(self.objective_frames), loss_version=self.loss_config.version,
                          observation_reduction=self.provenance["observation_reduction"])
        return result


def prepare_paper_observations(config, loss_config, *, sequence, calibration, point_filter,
                               camera, frames, scored_frames, records, helpers, initial_positions):
    """Prepare objective targets without changing export frames or simulated state."""
    from .loss_options import loss_temporal_reduction
    from .loss_targets import load_loss_targets
    from .point_set_loss import PointSetLoss, PointSetObservation

    endpoint = frames[-1].source_frame
    objective_frames = (endpoint,) if loss_config.version.startswith("dpsi-") else tuple(scored_frames)
    targets = load_loss_targets(
        config, loss_config, frame_indices=objective_frames,
        sequence_fingerprint=sequence.fingerprint, calibration_sha256=records["calibration"]["sha256"],
        initial_particles_sha256=records["initial_particles"]["sha256"],
        n_particles=config.simulation["n_particles"], camera=camera,
        frame_times={i: float(time) for i, time in enumerate(sequence.times)})
    objective_loss = PointSetLoss(camera, loss_config, initial_positions,
                                  precision=config.simulation.get("precision", "f32"))
    observations, hashes, filter_counts, preflight = [], {}, {}, {}
    for index in objective_frames:
        payload = dict(targets.by_frame[index])
        if loss_config.target_source == "recorded_cloud" and loss_config.geometric_weight == 0:
            payload.update(points_scene=np.empty((0, 3), dtype=np.float64),
                           target_representation="partial_observed")
        elif loss_config.target_source == "recorded_cloud":
            points = sequence.points[index]
            points = points[np.isfinite(points[:, :3]).all(axis=1)]
            if not len(points):
                raise ValueError(f"Observation {index} has no finite points")
            points = helpers.topview.apply_calibration(
                helpers.topview.filter_xyz(points, config.observation.trim_quantile), calibration)
            points, counts = helpers.dynamics.filter_scene_points(points, point_filter)
            points = np.ascontiguousarray(points[:, :3], dtype=np.float64)
            if len(points) < loss_config.min_target_points or not np.isfinite(points).all():
                raise ValueError(f"Observation {index} has insufficient finite target points after filtering")
            payload.update(points_scene=points, target_representation="partial_observed")
            payload.setdefault("metadata", {}).update(
                source_kind="observed", target_source="recorded_cloud",
                observation_coverage="partial single-view cloud; hidden geometry is not observed")
            filter_counts[str(index)] = counts
        frame = frames[index]
        observation = PointSetObservation(frame_index=index, step=frame.completed_substeps,
                                          timestamp=frame.target_time_s, **payload)
        preflight[str(index)] = objective_loss.prepare_observation(observation)
        observations.append(observation)
        hashes[str(index)] = {
            name: array_fingerprint(np.asarray(getattr(observation, name)))
            for name in ("points_scene", "track_particle_ids", "track_positions_scene", "track_valid",
                         "foreground_mask", "known_mask") if getattr(observation, name) is not None}
        hashes[str(index)]["metadata"] = canonical_hash(observation.metadata)
        hashes[str(index)]["target_representation"] = observation.target_representation
    objective = {"objective_frames": list(objective_frames),
                 "observation_reduction": loss_temporal_reduction(loss_config),
                 "loss_targets": targets.provenance, "objective_input_checks": preflight,
                 "target_coordinates": "scene", "target_units": "m"}
    return observations, hashes, filter_counts, objective


def prepare_experiment(config: ExperimentConfig, split="training", end_frame=None, build_sdf=True,
                       *, scored_window: FrameWindow | None = None) -> PreparedExperiment:
    """Verify inputs and replay from state zero, optionally scoring an explicit window."""
    from .loss import Observation
    from .loss_options import is_paper_loss, parse_loss_config

    if scored_window is not None and not isinstance(scored_window, FrameWindow):
        raise ValueError("scored_window must be a FrameWindow or None")
    if split not in {"training", "validation", "held-out"}:
        raise ValueError("split must be training or validation")
    window = config.window(split) if scored_window is None else scored_window
    records = verify_input_paths(config)
    helpers = get_reference_modules()
    simulator, dynamics, topview = helpers.simulator, helpers.dynamics, helpers.topview
    calibration = topview.load_calibration(config.paths["calibration"])
    if not calibration.is_metric or calibration.schema != "taichidough/scene-calibration/v2":
        raise ValueError("Differentiable training requires a metric v2 camera calibration")
    if calibration.source_frame != "mocap" or calibration.scene_frame not in {"mocap", "table-aligned"}:
        raise ValueError("Recorded dough replay requires mocap source and mocap or table-aligned scene coordinates")
    distortion = np.asarray(calibration.camera.get("d", []), dtype=float)
    if distortion.size and not np.allclose(distortion, 0, rtol=0, atol=1e-12):
        raise ValueError("Nonzero camera distortion is not supported by the differentiable pinhole renderer")
    sequence = dynamics.load_observation_sequence(config.paths["episode"])
    if config.expected_sequence_fingerprint and sequence.fingerprint != config.expected_sequence_fingerprint:
        raise ValueError("Recorded sequence fingerprint does not match the explicit expected value")
    source = verify_source_manifest(config, records, sequence.fingerprint)
    if end_frame is not None and (isinstance(end_frame, (bool, np.bool_)) or not isinstance(end_frame, (int, np.integer))):
        raise ValueError("Replay endpoint must be an integer source-frame index")
    selected_end = window.end_frame if end_frame is None else int(end_frame)
    if not window.start_frame <= selected_end <= window.end_frame:
        raise ValueError("Requested endpoint must be inside the selected scoring window")
    frames = observation_schedule(sequence.times, sequence.original_indices, selected_end,
                                  config.simulation.get("dt", 0.0002))
    scored_frames = tuple(i for i in window.indices() if i <= selected_end)
    if selected_end not in scored_frames:
        scored_frames += (selected_end,)
    metadata_path, metadata_record = resolved_reconstruction(config, records, sequence.fingerprint)
    raw_particles = np.load(config.paths["initial_particles"], allow_pickle=False)
    if raw_particles.ndim != 2 or raw_particles.shape[1] != 3 or not np.isfinite(raw_particles).all():
        raise ValueError("Reconstruction particles must be finite [N,3] values; filtering changes initialization")
    particles = simulator.load_initial_particles(config.paths["initial_particles"])
    reconstruction = simulator.load_reconstruction_metadata(metadata_path, config.paths["initial_particles"], particles, calibration)
    if reconstruction["scene_frame"] != calibration.scene_frame:
        raise ValueError("Initial particle scene frame differs from the camera calibration")
    expected_count = config.simulation.get("n_particles")
    if len(particles) != expected_count:
        raise ValueError(f"Particle count {len(particles)} differs from fixed n_particles={expected_count}; no resampling is performed")
    mass = simulator.compute_mass_properties(len(particles), config.simulation.get("grid", 48),
                                             config.density_kg_m3, reconstruction["object_volume_m3"], config.mass_kg)
    if not np.isclose(config.density_kg_m3, mass["density_kg_m3"], rtol=1e-8, atol=0):
        raise ValueError("Configured density disagrees with measured mass / reconstructed volume")
    numerical = dict(config.simulation)
    declared_floor = numerical.get("floor_y", reconstruction["floor_y"])
    if not np.isclose(declared_floor, reconstruction["floor_y"], rtol=0, atol=1e-8):
        raise ValueError("Configured floor differs from reconstruction floor")
    numerical.update(floor_y=reconstruction["floor_y"], particle_mass=mass["particle_mass_kg"], particle_volume=mass["particle_volume_m3"])
    sim_config = SimulationConfig(**numerical)
    validate_initial_stencils(particles, sim_config)
    state = ParticleState.initial(particles, sim_config.numpy_dtype)
    state.validate()
    geometry = dynamics.load_tool_geometry(config.paths["tool_geometry"], sequence.names)
    transforms = geometry.marker_from_mesh if sim_config.tool_collision == "sdf" else geometry.marker_from_collider
    replay = dynamics.ToolReplay(sequence, calibration, 0, selected_end,
                                 max_gap_s=config.replay_max_gap_s, marker_from_tool_frames=transforms)
    total_steps = frames[-1].completed_substeps
    controls = RecordedControls(replay, total_steps, sim_config.dt)
    sdf, collision_record = collision_inputs(config, records, sequence.names, simulator, build_sdf)
    filter_metadata = {"initialization": {"metadata_path": str(metadata_path), "metadata_sha256": reconstruction["metadata_sha256"]}}
    point_filter = dynamics.load_scene_point_filter(filter_metadata, metadata_path, calibration)
    settings = config.observation
    training_camera = scaled_camera(calibration, settings.width, settings.height)
    camera = camera_from_calibration(calibration, settings.width, settings.height)
    loss_config = parse_loss_config(config.loss)
    objective_metadata = {}
    if is_paper_loss(loss_config):
        observations, observation_hashes, filter_counts, objective_metadata = prepare_paper_observations(
            config, loss_config, sequence=sequence, calibration=calibration, point_filter=point_filter,
            camera=camera, frames=frames, scored_frames=scored_frames, records=records,
            helpers=helpers, initial_positions=state.x)
    else:
        processed = {}
        filter_counts = {}
        for index in (0, *scored_frames):
            points = sequence.points[index]
            points = points[np.isfinite(points[:, :3]).all(axis=1)]
            if not len(points):
                raise ValueError(f"Observation {index} has no finite points")
            points = topview.apply_calibration(topview.filter_xyz(points, settings.trim_quantile), calibration)
            points, counts = dynamics.filter_scene_points(points, point_filter)
            depth, nearest, *_ = topview.rasterize_depth(points, settings.width, settings.height,
                                                        training_camera, settings.splat_radius)
            valid = nearest >= 0
            if int(valid.sum()) < loss_config.min_observed_pixels:
                raise ValueError(f"Observation {index} has only {int(valid.sum())} valid pixels after filtering")
            processed[index] = (depth, valid, visible_optical_points(depth, valid, training_camera))
            filter_counts[str(index)] = counts
        depth0, valid0, _ = processed[0]
        observations = []
        for index in scored_frames:
            depth, valid, optical = processed[index]
            frame = frames[index]
            observations.append(Observation(frame_index=index, step=frame.completed_substeps,
                                            observed_depth=depth, observed_valid=valid,
                                            observed_initial_depth=depth0, observed_initial_valid=valid0,
                                            observed_points=optical, timestamp=frame.target_time_s))
        observation_hashes = {str(i): {"depth": array_fingerprint(processed[i][0]), "valid": array_fingerprint(processed[i][1]),
                                     "points": array_fingerprint(processed[i][2])} for i in (0, *scored_frames)}
    frozen = json.loads((EXPERIMENT_ROOT / "reference_manifest.json").read_text())
    provenance = {"schema": "taichidough/differentiable-prepared-inputs/v1", "configuration": config.as_dict(),
                  "input_files": records, "source_manifest": source, "sequence_fingerprint": sequence.fingerprint,
                  "reconstruction": metadata_record, "particle_array_content_sha256_f32": simulator.particle_array_sha256(particles),
                  "mass": mass, "simulation": asdict(sim_config), "calibration": topview.calibration_metadata(calibration),
                  "tool_geometry": dynamics.tool_geometry_metadata(geometry), "collision": collision_record,
                  "split": split, "scored_frames": list(scored_frames), "frames": [asdict(f) for f in frames],
                  "training_camera": training_camera, "observation_processing": asdict(settings),
                  "loss": loss_config.as_dict(), "observation_array_hashes": observation_hashes,
                  "observation_filter_counts": filter_counts,
                  "control_arrays": {"poses": array_fingerprint(controls.poses), "velocities": array_fingerprint(controls.velocities)},
                  "reference_sources": frozen["files"]}
    provenance.update(objective_metadata)
    if scored_window is not None:
        provenance["scored_window_override"] = asdict(scored_window)
    return PreparedExperiment(config, sim_config, state, dict(config.parameters), sdf, controls,
                              observations, camera, loss_config, provenance, total_steps, frames,
                              split, scored_frames, calibration, sequence, geometry, reconstruction, mass)

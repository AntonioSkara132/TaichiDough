"""Generate new table-aligned inputs without changing recordings or running calibration."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .config import (EXPERIMENT_ROOT, REPOSITORY_ROOT, file_sha256, load_config,
                     verify_input_paths)
from .data import prepare_experiment, validate_initial_stencils
from .evaluate import fresh_directory, require_owned_output
from .reference_adapter import get_reference_modules, reference_policy
from .table_alignment import (GRAVITY_ASSUMPTION, compose_table_calibration,
                              table_alignment_transform)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def move(points, transform):
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def bounds(points):
    points = np.asarray(points)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Bounds require nonempty finite points')
    return {'min': points.min(axis=0).tolist(), 'max': points.max(axis=0).tolist()}


def mesh_corners(config, simulator, names):
    settings = {
        'UR5e_spathla': ('ur_collision_mesh', simulator.UR_TOOL_VISUAL_ORIGIN,
                        simulator.UR_TOOL_VISUAL_RPY),
        'gen3_spathla': ('kinova_collision_mesh', simulator.KINOVA_TOOL_VISUAL_ORIGIN,
                       simulator.KINOVA_TOOL_VISUAL_RPY),
    }
    result = []
    padding = config.simulation.get('tool_contact_padding', 0)
    for name in names:
        key, origin, rpy = settings[name]
        vertices, _ = simulator.load_binary_stl(config.paths[key], config.tool_mesh_scale)
        vertices = simulator.mesh_in_tool_frame(vertices, origin, simulator.rpy_to_matrix(rpy))
        lower, upper = vertices.min(axis=0) - padding, vertices.max(axis=0) + padding
        result.append(np.array(list(itertools.product(*zip(lower, upper)))))
    return result


def tool_support(replay, corners, dynamics):
    """Bound every SLERP interval using its endpoint support plus a rotation bound."""
    poses = replay.poses
    rotated = np.stack([
        np.stack([corners[j] @ dynamics.quaternion_matrix(p[j, 3:7]).T + p[j, :3]
                  for j in range(2)]) for p in poses
    ])
    lower, upper = rotated.min(axis=(0, 1, 2)), rotated.max(axis=(0, 1, 2))
    for i in range(len(poses) - 1):
        for j in range(2):
            a = dynamics.quaternion_matrix(poses[i, j, 3:7])
            b = dynamics.quaternion_matrix(poses[i + 1, j, 3:7])
            angle = np.arccos(np.clip((np.trace(b @ a.T) - 1) / 2, -1, 1))
            radius = np.linalg.norm(corners[j], axis=1).max()
            displacement = 2 * radius * np.sin(angle / 2)
            centers = poses[i:i + 2, j, :3]
            start_offsets = corners[j] @ a.T
            lower = np.minimum(lower, centers.min(axis=0) + start_offsets.min(axis=0) - displacement)
            upper = np.maximum(upper, centers.max(axis=0) + start_offsets.max(axis=0) + displacement)
    return {'min': lower.tolist(), 'max': upper.tolist(),
            'method': 'continuous piecewise-linear centers and conservative SLERP rotation displacement'}


def run(args):
    base = load_config(args.config)
    records = verify_input_paths(base)
    summary = json.loads(args.plane_summary.read_text())
    verification_path = args.plane_summary.with_name('verification.json')
    verification = json.loads(verification_path.read_text())
    if verification.get('passed') is not True:
        raise ValueError('Table-plane estimate has not passed verification')
    if verification.get('output_sha256', {}).get('summary.json') != file_sha256(args.plane_summary):
        raise ValueError('Plane summary differs from its verified output hash')
    if Path(summary['episode']).resolve() != base.paths['episode']:
        raise ValueError('Plane estimate describes a different recording')
    source_paths = {row['path'] for row in records.values() if 'sha256' in row}
    source_paths.update(summary['input_hashes'])
    source_paths.update(str(path) for path in (args.config, args.plane_summary, verification_path,
                                               Path(__file__), EXPERIMENT_ROOT / 'table_alignment.py',
                                               EXPERIMENT_ROOT / 'data.py'))
    reconstruction_script = REPOSITORY_ROOT / 'scripts/reconstruct_voxel_dough_from_deformpath.py'
    source_paths.update(str(path) for path in (reconstruction_script, REPOSITORY_ROOT / 'scripts/deformpath_topview.py'))
    source_hashes = {str(path): file_sha256(Path(path)) for path in sorted(source_paths)}
    for path, expected in summary['input_hashes'].items():
        if source_hashes[path] != expected:
            raise ValueError(f'Plane-estimation input changed: {path}')
    output = fresh_directory(args.output_dir)
    helpers = get_reference_modules()
    topview, dynamics, simulator = helpers.topview, helpers.dynamics, helpers.simulator
    old_calibration = topview.load_calibration(base.paths['calibration'])
    old_document = json.loads(base.paths['calibration'].read_text())
    sequence = dynamics.load_observation_sequence(base.paths['episode'])
    if base.expected_sequence_fingerprint != sequence.fingerprint:
        raise ValueError('Base configuration sequence fingerprint differs')
    geometry = dynamics.load_tool_geometry(base.paths['tool_geometry'], sequence.names)
    old_replay = dynamics.ToolReplay(sequence, old_calibration, 0, len(sequence.times) - 1,
                                    max_gap_s=base.replay_max_gap_s,
                                    marker_from_tool_frames=geometry.marker_from_mesh)
    plane = np.asarray(summary['estimated_plane_scene'], dtype=float)
    transform = table_alignment_transform(plane)
    old_clouds = [topview.apply_calibration(topview.filter_xyz(points, base.observation.trim_quantile),
                                           old_calibration) for points in sequence.points]
    rotated_clouds = [move(points, transform) for points in old_clouds]
    corners = mesh_corners(base, simulator, sequence.names)
    # Center the full recorded motion horizontally without scaling it or moving Y=0.
    old_boxes = np.concatenate([
        corners[j] @ dynamics.quaternion_matrix(p[j, 3:7]).T + p[j, :3]
        for p in old_replay.poses for j in range(2)
    ])
    combined = np.concatenate([*rotated_clouds, move(old_boxes, transform)])
    translation_xz = 0.5 - (combined.min(axis=0) + combined.max(axis=0))[[0, 2]] / 2
    transform = table_alignment_transform(plane, translation_xz=translation_xz)
    derived_document = compose_table_calibration(
        old_document, plane, translation_xz=translation_xz,
        provenance={'plane_summary': str(args.plane_summary),
                    'plane_summary_sha256': file_sha256(args.plane_summary),
                    'source_calibration_sha256': records['calibration']['sha256'],
                    'horizontal_placement': 'center complete observed motion and padded tool mesh bounds at X=Z=0.5'})
    calibration_path = output / 'scene_calibration_v2.json'
    write_json(calibration_path, derived_document)
    calibration = topview.load_calibration(calibration_path)
    replay = dynamics.ToolReplay(sequence, calibration, 0, len(sequence.times) - 1,
                                 max_gap_s=base.replay_max_gap_s,
                                 marker_from_tool_frames=geometry.marker_from_mesh)
    clouds = [topview.apply_calibration(topview.filter_xyz(points, base.observation.trim_quantile), calibration)
              for points in sequence.points]
    optical_before = np.linalg.inv(old_calibration.scene_from_camera) @ old_calibration.scene_from_source
    optical_after = np.linalg.inv(calibration.scene_from_camera) @ calibration.scene_from_source
    np.testing.assert_allclose(optical_after, optical_before, atol=1e-12, rtol=0)
    max_point_error = max(float(np.max(np.abs(new - move(old, transform))))
                          for new, old in zip(clouds, old_clouds))
    if max_point_error > 3e-7:
        raise ValueError('Observation composition differs from the common transform')
    pose_error = float(np.max(np.abs(replay.poses[:, :, :3] - move(old_replay.poses[:, :, :3], transform))))
    orientation_error = max(float(np.max(np.abs(
        dynamics.quaternion_matrix(new[j, 3:7]) -
        transform[:3, :3] @ dynamics.quaternion_matrix(old[j, 3:7]))))
        for new, old in zip(replay.poses, old_replay.poses) for j in range(2))
    if pose_error > 3e-7 or orientation_error > 1e-7:
        raise ValueError('Tool position/orientation composition differs from the common transform')
    midpoint_times = (replay.times[:-1] + replay.times[1:]) / 2
    velocities = []
    velocity_error = 0.0
    for time in midpoint_times:
        _, new_v = replay.at(float(time))
        _, old_v = old_replay.at(float(time))
        expected = old_v.reshape(2, 2, 3) @ transform[:3, :3].T
        velocity_error = max(velocity_error, float(np.max(np.abs(new_v.reshape(2, 2, 3) - expected))))
        velocities.append(new_v)
    if velocity_error > 1e-4:
        raise ValueError('Transformed tool velocities differ from rotated recorded velocities')
    offsets = np.r_[0, np.cumsum([len(points) for points in clouds])]
    np.savez_compressed(output / 'observed_points_scene.npz', points=np.concatenate(clouds),
                        offsets=offsets, times=sequence.times, original_indices=sequence.original_indices)
    np.savez_compressed(output / 'tool_trajectories_scene.npz', poses=replay.poses,
                        times=replay.times, interval_midpoint_times=midpoint_times,
                        interval_velocities=np.asarray(velocities), names=np.asarray(sequence.names))
    print(f'Table transform and all {len(sequence.times)} recorded frames exported.', flush=True)

    old_metadata = json.loads(base.paths['reconstruction_metadata'].read_text())
    fill = old_metadata['fill']
    command = [sys.executable, str(reconstruction_script), '--episode-dir', str(base.paths['episode']),
               '--pointclouds-name', 'pointclouds_interpolated.pt', '--frame', '0',
               '--calibration', str(calibration_path), '--output-dir', str(output / 'reconstruction'),
               '--num-particles', str(base.simulation['n_particles']),
               '--voxel-size', str(old_metadata['voxel_size']), '--fill-mode', 'floor',
               '--fill-axis', 'y', '--fill-direction', 'negative',
               '--floor-clearance', str(fill['floor_clearance_m']),
               '--floor-min-thickness', str(fill['floor_min_thickness_m']),
               '--floor-max-thickness', str(fill['floor_max_thickness_m']),
               '--bbox-padding', str(old_metadata['bbox_padding']),
               '--trim-quantile', str(old_metadata['trim_quantile']),
               '--footprint-dilate', str(old_metadata['footprint_dilate']), '--seed', str(base.seed)]
    write_json(output / 'reconstruction_command.json', {'argv': command})
    with (output / 'reconstruction.log').open('x') as log:
        subprocess.run(command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    frame_dir = output / 'reconstruction' / base.paths['episode'].name / 'frame_0000'
    particle_path = frame_dir / 'sampled_particles_xyz.npy'
    metadata_path = frame_dir / 'reconstruction_metadata.json'
    particles = np.load(particle_path, allow_pickle=False)
    reconstruction = simulator.load_reconstruction_metadata(metadata_path, particle_path, particles, calibration)
    mass = simulator.compute_mass_properties(len(particles), base.simulation['grid'], base.density_kg_m3,
                                             reconstruction['object_volume_m3'], base.mass_kg)
    write_json(output / 'mass_properties.json', mass)
    config_document = json.loads(args.config.read_text())
    config_document['name'] = 'episode18-table-aligned-coulomb'
    config_document['paths'].pop('source_manifest', None)
    config_document['expected_sha256'].pop('source_manifest', None)
    for name, path in [('calibration', calibration_path), ('initial_particles', particle_path),
                       ('reconstruction_metadata', metadata_path)]:
        config_document['paths'][name] = str(path.relative_to(REPOSITORY_ROOT))
        config_document['expected_sha256'][name] = file_sha256(path)
    config_document['density_kg_m3'] = mass['density_kg_m3']
    config_document['expected_sequence_fingerprint'] = sequence.fingerprint
    config_document['simulation'].update(physics_version='corrected-v1', floor_y=0.0,
        tool_contact_model='coulomb-v1', tool_friction_coefficient=0.5,
        tool_contact_absorption=0.0, tool_stickiness=0.0)
    config_document['parameters']['tool_retention'] = 1.0
    config_document['fit_parameters'] = ['youngs_modulus', 'poisson_ratio', 'viscosity', 'plastic_min', 'plastic_max']
    config_document['parameter_bounds'] = {k: v for k, v in config_document['parameter_bounds'].items()
                                           if k in config_document['fit_parameters']}
    config_document['backend'] = 'cuda'
    config_path = output / 'episode18_table_aligned_coulomb.json'
    write_json(config_path, config_document)
    config = load_config(config_path)
    point_filter = dynamics.load_scene_point_filter(
        {'initialization': {'metadata_path': str(metadata_path), 'metadata_sha256': file_sha256(metadata_path)}},
        metadata_path, calibration)
    retained, counts = [], []
    for points in clouds:
        selected, count = dynamics.filter_scene_points(points, point_filter)
        retained.append(selected)
        counts.append(count)
    selected_points = np.concatenate(retained)
    from .state import SimulationConfig
    sim = SimulationConfig(**config.simulation)
    validate_initial_stencils(particles, sim)
    validate_initial_stencils(selected_points, sim)
    if (particles < 0).any() or (particles >= 1).any() or (selected_points < 0).any() or (selected_points >= 1).any():
        raise ValueError('Particles or retained observations leave the simulation domain')
    support = tool_support(replay, corners, dynamics)
    lower, upper = np.array(support['min']), np.array(support['max'])
    if np.any(lower[[0, 2]] < 0) or np.any(upper[[0, 2]] >= 1) or upper[1] >= 1:
        raise ValueError(f'Complete tool motion exceeds the simulation domain: {support}')
    print(f'Reconstruction: {mass["object_volume_m3"]:.9g} m^3; density={mass["density_kg_m3"]:.6f} kg/m^3.', flush=True)
    # Exercise the actual loader, observations and SDF assets, but no Taichi kernels.
    prepared_reports = {}
    for split in ('training', 'validation'):
        prepared = prepare_experiment(config, split=split, build_sdf=(split == 'training'))
        prepared_reports[split] = {'fingerprint': prepared.fingerprint,
            'total_steps': prepared.total_steps, 'scored_frames': list(prepared.scored_frames),
            'observation_count': len(prepared.observations), 'sdf_built': prepared.sdf is not None,
            'controls_finite': bool(np.isfinite(prepared.controls.poses).all() and
                                    np.isfinite(prepared.controls.velocities).all())}
        del prepared
    source_after = {path: file_sha256(Path(path)) for path in source_hashes}
    if source_after != source_hashes:
        raise ValueError('An input or generation source changed during reconstruction/validation')
    outputs = {str(path.relative_to(output)): file_sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}
    report = {'schema': 'taichidough/table-aligned-inputs/v1', 'status': 'inputs_valid',
        'config': str(config_path), 'source_hashes': source_hashes, 'sources_unchanged': True,
        'output_sha256': outputs, 'transform': transform.tolist(), 'gravity_assumption': GRAVITY_ASSUMPTION,
        'recorded_frame_count': len(sequence.times), 'full_recording_poses_transformed': True,
        'raw_tensors_modified': False, 'exported_scene_arrays_used_as_source_inputs': False,
        'mass': mass, 'particle_bounds': bounds(particles),
        'observed_bounds_before_clearance': bounds(np.concatenate(clouds)),
        'observed_bounds_after_clearance': bounds(selected_points), 'observation_filter_counts': counts,
        'tool_continuous_bounds': support, 'tools_below_floor_allowed': True,
        'initial_particle_stencils_valid': True, 'retained_observation_stencils_valid': True,
        'camera_optical_transform_max_error': float(np.max(np.abs(optical_after - optical_before))),
        'observation_transform_max_error_m': max_point_error, 'tool_position_max_error_m': pose_error,
        'tool_rotation_matrix_max_error': orientation_error, 'tool_velocity_max_error': velocity_error,
        'prepared_splits': prepared_reports, 'simulation_run': False, 'gradient_test_run': False,
        'calibration_run': False,
        'warnings': ['Density uses recorded 0.25 kg mass; verify the mass and visible-top-to-floor volume assumption.',
                     'Observed geometry fits the domain; future simulated trajectories have not been checked.',
                     'Initial material values are inherited guesses, not a fit of this corrected geometry.']}
    write_json(output / 'validation.json', report)
    print(f'Input validation passed. Configuration: {config_path}', flush=True)
    print(f'Validation and hashes: {output / "validation.json"}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT_ROOT / 'configs/episode18_stretch_clamp.json')
    parser.add_argument('--plane-summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-policy', choices=('strict', 'frozen'), default='strict')
    args = parser.parse_args(argv)
    args.config = args.config.expanduser().resolve()
    args.plane_summary = args.plane_summary.expanduser().resolve()
    args.output_dir = require_owned_output(args.output_dir)
    if os.environ.get('CLAUDE_JOB_DIR'):
        os.environ['TMPDIR'] = str(Path(os.environ['CLAUDE_JOB_DIR']) / 'tmp')
    with reference_policy(args.reference_policy):
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())

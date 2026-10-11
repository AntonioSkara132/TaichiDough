"""Fit fixed marker-to-tool corrections to partial HSV-selected recorded surfaces."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import uuid

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .config import REPOSITORY_ROOT, EXPERIMENT_ROOT, file_sha256
from .estimate_table_plane import hsv_mask
from .reference_adapter import get_reference_modules, reference_policy
from .partial_view_icp import (  # noqa: F401  (re-exported for existing callers)
    correspondences, evaluate, fit_icp, pose_matrix, sample_mesh, transform, visible_model,
)

HSV = (8, 33, 123, 255, 180, 255)
NAMES = ('UR5e_spathla', 'gen3_spathla')


def save_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(args):
    import torch
    episode = args.episode.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(EXPERIMENT_ROOT) or output.exists():
        raise ValueError('Choose a new output directory inside the experiment')
    output.mkdir(parents=True)
    geometry_path = REPOSITORY_ROOT / 'configs/tool_geometry_episode18_sdf.json'
    calibration_path = EXPERIMENT_ROOT / 'data/episode18_table_aligned_v1/scene_calibration_v2.json'
    paths = [episode / n for n in ('pointclouds.pt', 'paths_interpolated.pt', 'sequence_metadata.json', 'conversion_metadata.json')]
    paths += [geometry_path, calibration_path, Path(__file__), EXPERIMENT_ROOT / 'estimate_table_plane.py']
    mesh_paths = [REPOSITORY_ROOT / 'meshes' / n for n in ('ur_spathla.stl','gen3_spathla.stl',
                  'ur_spathla_collision_solid.stl','gen3_spathla_collision_solid.stl')]
    paths += mesh_paths
    hashes = {str(path): file_sha256(path) for path in paths}
    conversion = json.loads((episode / 'conversion_metadata.json').read_text())
    assert conversion['pointcloud_layout']['axis_1_features'] == ['x','y','z','normalized_frame_time',
        'frame_number','valid_flag','relative_timestamp_s','r','g','b']
    assert conversion['output_frame'] == 'mocap'
    metadata = json.loads((episode / 'sequence_metadata.json').read_text())
    calibration = json.loads(calibration_path.read_text())
    scene_from_source = np.array(calibration['scene_from_source'])
    camera_from_source = np.linalg.inv(calibration['scene_from_camera']) @ scene_from_source
    camera = calibration['camera']
    raw_frames = torch.load(episode / 'pointclouds.pt', map_location='cpu', weights_only=True, mmap=True)[0]
    record = torch.load(episode / 'paths_interpolated.pt', map_location='cpu', weights_only=True)[0]
    assert tuple(record['pose_frames']) == NAMES
    poses = record['path'].numpy()
    valid = record['stream_validity'].numpy().astype(bool)
    raw_indices = metadata['pointcloud_indices']
    assert len(raw_indices) == len(poses)
    geometry = json.loads(geometry_path.read_text())
    with reference_policy('frozen'):
        simulator = get_reference_modules().simulator
    specifications = [(simulator.UR_TOOL_VISUAL_ORIGIN, simulator.UR_TOOL_VISUAL_RPY),
                      (simulator.KINOVA_TOOL_VISUAL_ORIGIN, simulator.KINOVA_TOOL_VISUAL_RPY)]
    triangles, models, collision_models = [], [], []
    for j, (origin, rpy) in enumerate(specifications):
        v, _ = simulator.load_binary_stl(mesh_paths[j], .001)
        tri = simulator.mesh_in_tool_frame(v, origin, simulator.rpy_to_matrix(rpy)).reshape(-1, 3, 3).astype(float)
        triangles.append(tri)
        models.append(sample_mesh(tri, 24000, args.seed + j))
        v, _ = simulator.load_binary_stl(mesh_paths[j + 2], .001)
        tri_collision = simulator.mesh_in_tool_frame(v, origin, simulator.rpy_to_matrix(rpy)).reshape(-1, 3, 3)
        collision_models.append(sample_mesh(tri_collision, 24000, args.seed + j))
        np.savez_compressed(output / f'mesh_{j}.npz', triangles=tri, samples=models[j][0], normals=models[j][1])
    train_indices = list(range(0, 353, 32))
    heldout_indices = list(range(16, 369, 32)) + [387]
    selected_indices = sorted(set(train_indices + heldout_indices))
    frames = [[], []]
    extraction = []
    for processed in selected_indices:
        raw_index = int(raw_indices[processed])
        if not valid[processed].all():
            raise ValueError(f'Invalid recorded pose at processed frame {processed}')
        values = raw_frames[raw_index].numpy()
        if not np.allclose(values[:, 4], raw_index):
            raise ValueError('Raw frame-number column disagrees with metadata mapping')
        raw_time = float(np.median(values[:, 6]))
        pose_time = float(poses[processed, 0, 13])
        if abs(raw_time - pose_time) > 1e-5:
            raise ValueError('Raw and interpolated pose timestamps differ')
        rgb_values = values[:, 7:10]
        if np.nanmin(rgb_values) < 0 or np.nanmax(rgb_values) > 255:
            raise ValueError('Raw RGB values are not byte-range channels')
        rgb = np.clip(np.rint(rgb_values), 0, 255).astype(np.uint8)
        selection = hsv_mask(rgb, HSV) & np.isfinite(values[:, :3]).all(axis=1) & (values[:, 5] > 0)
        source = values[selection, :3].astype(float)
        # HSV already excludes gray table and pale dough. Do not discard points based on table height.
        transforms = [pose_matrix(poses[processed, j]) for j in range(2)]
        local = [transform(source, np.linalg.inv(matrix)) for matrix in transforms]
        initial = [np.asarray(geometry['tools'][j]['marker_from_mesh']) for j in range(2)]
        distance = np.column_stack([cKDTree(transform(models[j][0], initial[j])).query(local[j])[0] for j in range(2)])
        label = distance.argmin(axis=1)
        row = {'processed_frame': processed, 'raw_frame': raw_index, 'raw_timestamp_s': raw_time,
               'pose_timestamp_s': pose_time, 'hsv_points': len(source), 'split': 'train' if processed in train_indices else 'heldout', 'tools': []}
        for j in range(2):
            # Broad 100 mm model-distance ROI; leave a 10 mm ambiguity gap between tools.
            keep = (label == j) & (distance[:, j] < .10) & ((distance[:, 1-j] - distance[:, j]) > .01)
            selected = local[j][keep]
            count_before = len(selected)
            if len(selected):
                neighbors = cKDTree(selected).query_ball_point(selected, .004, return_length=True)
                selected = selected[neighbors >= 5]
            if len(selected) < 500:
                raise ValueError(f'Insufficient HSV tool support: frame {processed}, tool {j}, n={len(selected)}')
            rng = np.random.default_rng(args.seed + processed * 11 + j)
            take = rng.choice(len(selected), min(args.points_per_frame, len(selected)), replace=False)
            sample = selected[take]
            frame = {'processed': processed, 'raw': raw_index, 'points': sample,
                     'camera_from_marker': camera_from_source @ transforms[j], 'source_from_marker': transforms[j],
                     'split': row['split']}
            frames[j].append(frame)
            row['tools'].append({'name': NAMES[j], 'roi_points': count_before, 'dense_points': len(selected),
                                 'fit_points': len(sample)})
            np.savez_compressed(output / f'frame_{processed:04d}_tool_{j}.npz', points_marker=sample,
                points_source=transform(sample, transforms[j]), source_from_marker=transforms[j],
                camera_from_marker=frame['camera_from_marker'], scene_from_source=scene_from_source)
        extraction.append(row)
    save_json(output / 'extraction.json', {'hsv_opencv': HSV, 'rgb_layout_verified': True, 'frames': extraction,
        'table_height_filter': False, 'roi_distance_m': .10, 'association_ambiguity_margin_m': .01})
    print(f'Extracted {len(selected_indices)} frames; {len(train_indices)} train and {len(heldout_indices)} heldout.', flush=True)
    results = []
    corrected_geometry = copy.deepcopy(geometry)
    for j in range(2):
        train = [frame for frame in frames[j] if frame['split'] == 'train']
        heldout = [frame for frame in frames[j] if frame['split'] == 'heldout']
        initial = np.asarray(geometry['tools'][j]['marker_from_mesh'], dtype=float)
        baseline_train = evaluate(train, models[j], initial, camera)
        baseline_heldout = evaluate(heldout, models[j], initial, camera)
        offsets = []
        for frame in train:
            visible, _ = visible_model(*models[j], initial, frame['camera_from_marker'], camera)
            offsets.append(np.median(frame['points'], axis=0) - np.median(visible, axis=0))
        shifted = initial.copy(); shifted[:3, 3] += np.median(offsets, axis=0)
        seeds = [initial, shifted]
        for angle in (-20, 20):
            seed = shifted.copy()
            seed[:3, :3] = Rotation.from_rotvec(np.deg2rad(angle) * np.array([0, 1, 0])).as_matrix() @ initial[:3, :3]
            seeds.append(seed)
        candidates = []
        for index, seed in enumerate(seeds):
            fitted, history, hessian = fit_icp(train, models[j], camera, seed, iterations=args.iterations)
            score = evaluate(train, models[j], fitted, camera)
            candidates.append((score['equal_frame_median_distance_mm'], fitted, history, hessian, score))
            print(f'{NAMES[j]} initialization {index}: train median distance {candidates[-1][0]:.3f} mm', flush=True)
        candidates.sort(key=lambda item: item[0])
        _, fitted, history, hessian, after_train = candidates[0]
        after_heldout = evaluate(heldout, models[j], fitted, camera)
        collision_before = evaluate(heldout, collision_models[j], initial, camera)
        collision_after = evaluate(heldout, collision_models[j], fitted, camera)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        relative = eigenvalues / max(eigenvalues[-1], 1e-20)
        delta = fitted @ np.linalg.inv(initial)
        translations = [np.linalg.norm(item[1][:3, 3] - fitted[:3, 3]) for item in candidates]
        rotations = [np.rad2deg(Rotation.from_matrix(item[1][:3, :3] @ fitted[:3, :3].T).magnitude()) for item in candidates]
        baseline = baseline_heldout['equal_frame_median_distance_mm']
        improved = after_heldout['equal_frame_median_distance_mm'] < baseline * .7
        supported = bool(improved and after_heldout['equal_frame_median_distance_mm'] < 5 and
                         after_heldout['equal_frame_within_5mm'] > .75 and relative[0] > 1e-4)
        corrected_geometry['tools'][j]['marker_from_mesh'] = fitted.tolist()
        result = {'name': NAMES[j], 'initial_marker_from_mesh': initial.tolist(),
                  'candidate_marker_from_mesh': fitted.tolist(), 'delta_marker_local': delta.tolist(),
                  'translation_delta_mm': (delta[:3, 3] * 1000).tolist(),
                  'translation_norm_mm': float(np.linalg.norm(delta[:3, 3]) * 1000),
                  'rotation_delta_rotvec_deg': np.rad2deg(Rotation.from_matrix(delta[:3, :3]).as_rotvec()).tolist(),
                  'rotation_angle_deg': float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude())),
                  'train_before': baseline_train, 'train_after': after_train,
                  'heldout_before': baseline_heldout, 'heldout_after': after_heldout,
                  'collision_heldout_before': collision_before, 'collision_heldout_after': collision_after,
                  'point_to_plane_scaled_information_eigenvalues': eigenvalues.tolist(),
                  'relative_information_eigenvalues': relative.tolist(),
                  'weakest_scaled_direction_rot_then_translation': eigenvectors[:, 0].tolist(),
                  'multistart_translation_spread_mm': float(max(translations) * 1000),
                  'multistart_rotation_spread_deg': float(max(rotations)),
                  'candidate_scores_train_mm': [item[0] for item in candidates],
                  'supported_by_heldout': supported, 'history': history}
        results.append(result)
        save_json(output / f'tool_{j}_result.json', result)
        print(f'{NAMES[j]} heldout: {baseline:.3f} -> {after_heldout["equal_frame_median_distance_mm"]:.3f} mm; supported={supported}', flush=True)
    all_supported = all(row['supported_by_heldout'] for row in results)
    if all_supported:
        corrected_geometry['registration_provenance'] = {'method': 'shared-marker-local-partial-surface-ICP',
            'source_geometry_sha256': hashes[str(geometry_path)], 'hsv_opencv': list(HSV),
            'note': 'Only marker_from_mesh corrected; box proxy marker_from_collider is unchanged. Camera and recorded poses fixed.'}
        save_json(output / 'tool_geometry_episode18_registered_candidate.json', corrected_geometry)
    source_after = {path: file_sha256(Path(path)) for path in hashes}
    if source_after != hashes:
        raise ValueError('Registration source/input changed during run')
    report = {'schema': 'taichidough/tool-cloud-registration/v1', 'status': 'completed',
              'method': 'fixed marker_from_mesh per tool, robust visible-surface point-to-plane ICP with weak edge term',
              'hsv_opencv': list(HSV), 'source_hashes': hashes, 'sources_unchanged': True,
              'calibration_fixed': True, 'poses_fixed': True, 'fit_coordinate_frame': 'marker-local',
              'train_processed_frames': train_indices, 'heldout_processed_frames': heldout_indices,
              'settings': {'mesh_samples': 24000, 'points_per_frame': args.points_per_frame,
                           'iterations': args.iterations, 'seed': args.seed, 'visible_depth_bin_px': 2,
                           'visible_depth_tolerance_m': .0015, 'maximum_translation_m': .15,
                           'maximum_rotation_deg': 70, 'tangential_weight': .12},
              'candidate_geometry_written': all_supported, 'existing_configs_modified': False,
              'results': results,
              'limitations': ['ICP is local and partial planar surfaces can constrain some rigid components weakly.',
                  'Mesh visibility is approximated by sampled-surface z-buffering, not exact ray tracing.',
                  'HSV-selected parts need not cover the whole physical tool.',
                  'Camera extrinsic, mesh dimensions and timing are fixed; their errors can bias marker-local corrections.',
                  'No material calibration or simulation was run.']}
    report['output_sha256'] = {str(p.relative_to(output)): file_sha256(p) for p in output.iterdir() if p.is_file()}
    save_json(output / 'summary.json', report)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=REPOSITORY_ROOT.parent / 'data/deformpath_training/DeformPath3/snimanje_23_10/episode18_kugla')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1818)
    parser.add_argument('--points-per-frame', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=35)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

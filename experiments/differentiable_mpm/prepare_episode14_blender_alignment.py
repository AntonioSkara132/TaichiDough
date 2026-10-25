"""Export episode14 multi-frame observations for manual shared mounting alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO.parent / 'data/deformpath_training'
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(REPO))
import refine_spatula_trajectories as runner
import spatula_trajectory_refinement as refinement
from experiments.differentiable_mpm import render_episode14_rgb_paths as render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=TRAIN/'configs/episode14_dynamics_spatula_icp.json')
    parser.add_argument('--frames', type=int, nargs='+', default=[868], help='Raw pointcloud ordinals; one reference frame by default')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    data = runner.load_inputs(config)
    segmentation, _, _, _ = runner.build_configs(config)
    seeds = runner.baseline_and_seeds(config, data)
    posed = [runner.posed_seed_samples(data['models'][n], seeds[n]['seeds']) for n in refinement.TOOL_NAMES]
    options = argparse.Namespace(bag=data['bag_dir'], sequence=data['sequence_dir'],
                                export=runner.resolve(config['inputs']['sequence_export_metadata']).parent,
                                paths=None, geometry=None)
    context = render.load_context(options)
    selected = args.frames
    indices = {int(raw): k for k, raw in enumerate(data['indices'])}
    geometry_path = runner.resolve(config['inputs']['baseline_geometry'])
    document = {'schema': 'taichidough/manual-mounting-workspace/v1', 'units': 'metres',
                'matrix_convention': 'marker_from_mesh maps visual-transformed tool-link coordinates into mocap marker coordinates',
                'mesh_coordinates': 'STL millimetres converted to metres then frozen corrected-v1 visual transform applied exactly once',
                'path_quaternion_order': 'xyzw', 'display_coordinates': 'marker coordinates without axis remapping; display offsets are not exported',
                'baseline_geometry': json.loads(geometry_path.read_text()),
                'baseline_geometry_sha256': hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
                'camera': context['camera'], 'camera_from_mocap': context['camera_from_world'].tolist(),
                'tools': [], 'frames': [], 'input_hashes': data['hashes']}
    for j, name in enumerate(refinement.TOOL_NAMES):
        vertices, faces = context['meshes'][j]
        document['tools'].append({'name': name, 'vertices_link_m': vertices.tolist(), 'triangles': faces.tolist(),
                                  'baseline_marker_from_mesh': context['marker_from_tool'][j].tolist()})
    with render.RgbFrameReader(context['bag'], cache_frames=4) as reader:
        for raw in selected:
            k = indices[raw]
            associations = runner.associate_frame(data, k, 'moving_tag', segmentation.hsv, posed, segmentation)
            rgb_ordinal = context['bag'].rgb_mappings[raw].rgb_ordinal
            image = cv2.cvtColor(reader.get(rgb_ordinal), cv2.COLOR_RGB2BGR)
            image_name = f'rgb_pc{raw:04d}.png'
            cv2.imwrite(str(args.output/image_name), image)
            document['frames'].append({'raw': raw, 'rgb_ordinal': rgb_ordinal, 'rgb_file': image_name,
                                       'timestamp_ns': str(context['bag'].pointcloud.frames[raw].header_stamp_ns),
                                       'tools': [{'name': name, 'points_marker_m': associations[j]['points'].tolist(),
                                                  'counts': associations[j]['counts'],
                                                  'mocap_from_marker': runner.marker_poses(data, k)[j].tolist()}
                                                 for j, name in enumerate(refinement.TOOL_NAMES)]})
    (args.output/'alignment_inputs.json').write_text(json.dumps(document)+'\n')
    print('Prepared', args.output, flush=True)


if __name__ == '__main__':
    main()

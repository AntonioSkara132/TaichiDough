"""Prepare one full, unfiltered RGB-D frame in original mocap scene coordinates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO.parent/'data/deformpath_training'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mounting-inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-export', type=Path, default=TRAIN/'episode14_dynamics_table_rgb32768_export_v1')
    args = parser.parse_args()
    document = json.loads(args.mounting_inputs.read_text())
    if len(document['frames']) != 1:
        raise ValueError('Exactly one reference frame is required')
    frame = document['frames'][0]
    metadata_path = args.full_export/'conversion_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    calibration = json.loads((args.full_export/'scene_calibration_v2.json').read_text())
    if metadata['output_frame'] != 'mocap' or metadata['pointcloud_export']['color_filter_applied']:
        raise ValueError('Expected an unfiltered mocap-frame RGB-D export')
    if not np.allclose(np.linalg.inv(calibration['scene_from_camera']), document['camera_from_mocap'], atol=1e-10):
        raise ValueError('Full-cloud camera calibration differs from mounting reference')
    raw = frame['raw']
    if str(metadata['episode']['timestamps_ns'][raw]) != frame['timestamp_ns']:
        raise ValueError('Full cloud timestamp does not match reference frame')
    payload = torch.load(args.full_export/'pointclouds.pt', map_location='cpu', weights_only=True, mmap=True)
    cloud = payload[0][raw].numpy()
    valid = np.isfinite(cloud[:, :3]).all(axis=1) & (cloud[:, 5] > .5)
    cloud = cloud[valid]
    document['workspace_layout'] = 'recorded_scene'
    document['display_coordinates'] = 'Original calibrated mocap coordinates; no display offsets or axis remapping'
    document['full_cloud_source'] = {
        'path': str(args.full_export/'pointclouds.pt'), 'raw_frame': raw,
        'color_filter_applied': False, 'sample_cap': 32768,
        'count': len(cloud), 'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        'frame_array_sha256': hashlib.sha256(cloud.tobytes()).hexdigest(),
    }
    frame['points_mocap_m'] = cloud[:, :3].tolist()
    frame['rgb'] = cloud[:, 7:10].tolist()
    # The full scene replaces per-tool filtered point observations.
    for tool in frame['tools']:
        tool.pop('points_marker_m', None)
        tool.pop('counts', None)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.mounting_inputs.parent/frame['rgb_file'], args.output/frame['rgb_file'])
    (args.output/'alignment_inputs.json').write_text(json.dumps(document)+'\n')
    print('Prepared', args.output, 'with', len(cloud), 'full-scene colored points')


if __name__ == '__main__':
    main()

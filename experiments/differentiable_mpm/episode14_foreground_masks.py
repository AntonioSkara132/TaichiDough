"""Export observed Episode 14 tool points as camera-aligned masks outside chunk data.

Hand polygons are edited separately. These are foreground labels, not a depth-tested
claim that dough is hidden behind each marked pixel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
DATASET = REPO / 'experiments/differentiable_mpm/data/episode14_dynamics_density1200_roi_temporal_v3'
BAG = PROJECT / ('data/deformpath_training/DeformPath2_release/AltDynamics/'
                 'AlternativeDynamicsBag_verified_11_12_20_20261007_125929/'
                 'snimanje_16_6/episode14_dynamics')
TOOL_EXPORT = PROJECT / 'data/deformpath_training/episode14_dynamics_tools_orange_nocap_export_v1'
CALIBRATION = PROJECT / 'data/deformpath_training/episode14_dynamics_sequence_v4/scene_calibration_v2.json'
OUTPUT = REPO / 'experiments/differentiable_mpm/data/episode14_tool_hand_masks_v3_manual_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def project_points(points, camera_from_mocap, camera):
    """Return in-image integer pixels for mocap XYZ with positive optical depth."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected [N,3] mocap XYZ')
    optical = points @ camera_from_mocap[:3, :3].T + camera_from_mocap[:3, 3]
    valid = np.isfinite(optical).all(axis=1) & (optical[:, 2] > .01)
    optical = optical[valid]
    u = camera['fx'] * optical[:, 0] / optical[:, 2] + camera['cx']
    v = camera['fy'] * optical[:, 1] / optical[:, 2] + camera['cy']
    inside = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < camera['width'])
              & (v >= 0) & (v < camera['height']))
    pixels = np.rint(np.column_stack((u[inside], v[inside]))).astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, camera['width'] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, camera['height'] - 1)
    return pixels


def point_mask(pixels, shape=(480, 640), radius=1):
    """Mark observed point pixels; radius expands support, without filling holes."""
    if radius < 0 or radius > 3:
        raise ValueError('Point radius must be between zero and three pixels')
    mask = np.zeros(shape, np.uint8)
    pixels = np.asarray(pixels, dtype=np.int32).reshape(-1, 2)
    if len(pixels):
        if ((pixels[:, 0] < 0) | (pixels[:, 0] >= shape[1]) |
                (pixels[:, 1] < 0) | (pixels[:, 1] >= shape[0])).any():
            raise ValueError('Point pixel outside image')
        mask[pixels[:, 1], pixels[:, 0]] = 255
        if radius:
            size = 2 * radius + 1
            mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    return mask


def rasterize_polygons(polygons, shape=(480, 640)):
    """Rasterize ordered add/erase polygons drawn in recorded RGB pixel coordinates."""
    mask = np.zeros(shape, np.uint8)
    for row in polygons:
        if row.get('mode') not in ('add', 'erase'):
            raise ValueError('Polygon mode must be add or erase')
        vertices = np.asarray(row.get('vertices'), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError('Polygon requires at least three 2D vertices')
        if (not np.isfinite(vertices).all() or (vertices[:, 0] < 0).any()
                or (vertices[:, 0] >= shape[1]).any() or (vertices[:, 1] < 0).any()
                or (vertices[:, 1] >= shape[0]).any()):
            raise ValueError('Polygon vertex outside recorded RGB')
        polygon = np.rint(vertices).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255 if row['mode'] == 'add' else 0)
    return mask


def frame_rows(chunk_metadata, bag_index):
    ordinals = chunk_metadata['raw_pointcloud_ordinals']
    if len(ordinals) != chunk_metadata['frame_count'] or len(set(ordinals)) != len(ordinals):
        raise ValueError('Invalid chunk frame list')
    rows = []
    for local, raw in enumerate(ordinals):
        if not isinstance(raw, int) or not 0 <= raw < bag_index.pointcloud_count:
            raise ValueError('Invalid raw pointcloud ordinal')
        stamp = bag_index.pointcloud.frames[raw].header_stamp_ns
        mapping = bag_index.rgb_mappings[raw]
        rgb = bag_index.rgb.frames[mapping.rgb_ordinal] if mapping.exact else None
        rows.append({'local_frame': local, 'raw_pointcloud_ordinal': raw,
                     'pointcloud_header_stamp_ns': str(stamp),
                     'rgb_ordinal': rgb.ordinal if rgb else None,
                     'rgb_header_stamp_ns': str(rgb.header_stamp_ns) if rgb else None,
                     'rgb_match': 'exact' if rgb else 'missing_exact_rgb',
                     'tool_status': 'pending',
                     'hand_status': 'unreviewed' if rgb else 'missing_exact_rgb',
                     'tool_mask': None, 'hand_mask': None})
    if (str(bag_index.pointcloud.frames[ordinals[0]].header_stamp_ns)
            != chunk_metadata['first_retained_header_stamp_ns'] or
            str(bag_index.pointcloud.frames[ordinals[-1]].header_stamp_ns)
            != chunk_metadata['last_retained_header_stamp_ns']):
        raise ValueError('Chunk timestamps disagree with recorded bag')
    return rows


def build_manifest(dataset, bag_index, calibration_path, tool_path):
    dataset = Path(dataset)
    chunks = {}
    for directory in sorted((dataset / 'recordings').glob('episode14_dynamics_chunk*')):
        metadata_path = directory / 'sequence_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        chunks[directory.name] = {
            'metadata_sha256': sha256(metadata_path),
            'sequence_fingerprint': metadata['parent']['sequence_fingerprint'],
            'frames': frame_rows(metadata, bag_index),
        }
    if not chunks:
        raise ValueError('No Episode 14 chunks found')
    return {'schema_version': 1, 'semantics': 'observed tool point support and manually reviewed visible hands',
            'source': {'dataset_dir': str(dataset), 'dataset_selection_sha256': sha256(dataset / 'range_selection.json'),
                       'bag_dir': str(bag_index.bag_dir), 'bag_fingerprint': bag_index.source_fingerprint,
                       'calibration_path': str(calibration_path), 'calibration_sha256': sha256(calibration_path),
                       'tool_points_path': str(tool_path), 'tool_points_sha256': sha256(tool_path)},
            'camera_resolution': [640, 480], 'tool_point_radius_px': 1, 'chunks': chunks}


def export_tool_masks(output=OUTPUT, dataset=DATASET, bag=BAG, calibration=CALIBRATION,
                      tool_export=TOOL_EXPORT):
    """Create a new sidecar; never change a chunk recording or an existing output."""
    import torch

    sys.path.insert(0, str(PROJECT / 'scripts'))
    from rosbag_temporal_annotations import index_rosbag

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    bag_index = index_rosbag(Path(bag))
    calibration = Path(calibration)
    tool_export = Path(tool_export)
    camera_record = json.loads(calibration.read_text())
    camera = camera_record['camera']
    if (camera_record['source_frame'] != 'mocap' or camera_record['scene_frame'] != 'mocap'
            or (camera['width'], camera['height']) != (640, 480)
            or not np.allclose(camera['d'], 0)):
        raise ValueError('Unexpected tool-point camera calibration')
    camera_from_mocap = np.linalg.inv(np.asarray(camera_record['scene_from_camera'], dtype=np.float64))
    manifest = build_manifest(dataset, bag_index, calibration, tool_export / 'pointclouds.pt')
    conversion = json.loads((tool_export / 'conversion_metadata.json').read_text())
    stamps = np.asarray(conversion['episode']['timestamps_ns'], dtype=np.int64)
    if not np.array_equal(stamps, np.asarray(bag_index.pointcloud.header_stamps_ns, dtype=np.int64)):
        raise ValueError('Orange export timestamps differ from raw pointcloud headers')
    frames = torch.load(tool_export / 'pointclouds.pt', map_location='cpu', weights_only=True, mmap=True)[0]
    if len(frames) != len(stamps):
        raise ValueError('Tool export frame count differs from bag')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    for chunk, details in manifest['chunks'].items():
        tool_dir = temp / 'chunks' / chunk / 'tool'
        tool_dir.mkdir(parents=True)
        for row in details['frames']:
            raw = row['raw_pointcloud_ordinal']
            cloud = frames[raw].detach().cpu().numpy()
            if cloud.ndim != 2 or cloud.shape[1] != 10 or not np.isfinite(cloud).all():
                raise ValueError(f'Invalid orange cloud at raw {raw}; unfinished output: {temp}')
            if len(cloud) and (not np.all(cloud[:, 4] == raw) or not np.all(cloud[:, 5] == 1)):
                raise ValueError(f'Orange cloud has wrong frame/validity at raw {raw}')
            if len(cloud) and np.max(np.abs(cloud[:, 6] - (stamps[raw] - stamps[0]) * 1e-9)) > 1e-3:
                raise ValueError(f'Orange cloud timestamp differs at raw {raw}')
            pixels = project_points(cloud[:, :3], camera_from_mocap, camera)
            mask = point_mask(pixels, (camera['height'], camera['width']))
            name = f'pc{raw:04d}.png'
            if not cv2.imwrite(str(tool_dir / name), mask):
                raise OSError(f'Failed to write tool mask {tool_dir / name}')
            row['tool_mask'] = f'chunks/{chunk}/tool/{name}'
            row['tool_status'] = 'ready' if len(pixels) else 'no_observed_points'
            row['tool_source_points'] = len(cloud)
            row['tool_projected_points'] = len(pixels)
        print(f'{chunk}: {len(details["frames"])} tool frames', flush=True)
    (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if output.exists():
        raise FileExistsError(output)
    temp.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = export_tool_masks(args.output)
    print('Saved', args.output, 'with', sum(len(c['frames']) for c in manifest['chunks'].values()), 'frames')


if __name__ == '__main__':
    main()

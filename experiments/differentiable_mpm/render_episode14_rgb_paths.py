"""Project episode14 exported marker/tool paths onto timestamp-matched bag RGB.

No simulation or calibration fitting is performed. Tool geometry uses the same
frozen simulator visual transforms and saved marker-to-tool transforms as the
prepared episode14 dataset; the overlay is a check of those assumptions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
TRAIN = PROJECT / 'data/deformpath_training'
DATASET = REPO / 'experiments/differentiable_mpm/data/episode14_dynamics_density1200_roi_temporal_v2'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PROJECT / 'scripts'))
from rosbag_temporal_annotations import index_rosbag, RgbFrameReader
from experiments.differentiable_mpm.reference_adapter import reference_policy, get_reference_modules

# Fixed blue/orange identities; direct text labels and distinct markers supplement color.
COLORS = [(214, 120, 42), (52, 104, 234)]  # BGR: #2a78d6, #ea6834
NAMES = ['UR5e', 'gen3']


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project(points, camera_from_world, camera):
    optical = transform(points, camera_from_world)
    pixels = np.column_stack((camera['fx'] * optical[:, 0] / optical[:, 2] + camera['cx'],
                              camera['fy'] * optical[:, 1] / optical[:, 2] + camera['cy']))
    valid = np.isfinite(pixels).all(1) & (optical[:, 2] > .01)
    return pixels, valid


def triangle_silhouette(pixels, valid, triangles, shape):
    """Rasterize the union; a single fillPoly call cancels overlapping faces."""
    mask = np.zeros(shape, np.uint8)
    valid_triangles = triangles[valid[triangles].all(axis=1)]
    polygons = pixels[valid_triangles]
    polygons = polygons[(np.abs(polygons) < 10000).all(axis=(1, 2))]
    for polygon in np.rint(polygons * 8).astype(np.int32):
        cv2.fillConvexPoly(mask, polygon, 255, shift=3)
    return mask


def text(image, value, position, size=.5, color=(245, 245, 245), thickness=1):
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, size, (15, 15, 15), thickness+2, cv2.LINE_AA)
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def current_marker(image, pixel, color, tool, radius=5):
    p = tuple(np.rint(pixel).astype(int))
    if tool == 0:
        cv2.circle(image, p, radius+2, (18, 18, 18), -1, cv2.LINE_AA)
        cv2.circle(image, p, radius, color, -1, cv2.LINE_AA)
    else:
        vertices = np.asarray([[p[0], p[1]-radius-1], [p[0]+radius+1,p[1]],
                               [p[0],p[1]+radius+1], [p[0]-radius-1,p[1]]], np.int32)
        cv2.polylines(image, [vertices], True, (18,18,18), 5, cv2.LINE_AA)
        cv2.fillPoly(image, [vertices], color, cv2.LINE_AA)


def load_mounts(geometry_path):
    geometry = json.loads(geometry_path.read_text())['tools']
    if [row['name'] for row in geometry] != ['UR5e_spathla', 'gen3_spathla']:
        raise ValueError('Unexpected geometry tool order')
    mounts = [np.asarray(row['marker_from_mesh'], float) for row in geometry]
    for matrix in mounts:
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
            raise ValueError('Geometry mounting must be a rigid marker_from_mesh transform')
    return mounts


def load_context(args):
    bag = index_rosbag(args.bag)
    conversion = json.loads((args.export / 'conversion_metadata.json').read_text())
    source_stamps = np.asarray(conversion['episode']['timestamps_ns'], dtype=np.int64)
    bag_stamps = np.asarray(bag.pointcloud.header_stamps_ns, dtype=np.int64)
    if not np.array_equal(source_stamps, bag_stamps):
        raise ValueError('Exported timestamps do not match bag pointcloud headers')
    sequence_meta = json.loads((args.sequence / 'sequence_metadata.json').read_text())
    indices = np.asarray(sequence_meta['pointcloud_indices'], dtype=int)
    paths_path = args.paths or args.sequence / 'paths_interpolated.pt'
    payload = torch.load(paths_path, map_location='cpu', weights_only=True)[0]
    paths = payload['path'].numpy().astype(np.float64)
    if payload['pose_frames'] != ['UR5e_spathla', 'gen3_spathla']:
        raise ValueError('Unexpected tool order')
    if len(paths) != len(indices) or not np.isfinite(paths).all():
        raise ValueError('Invalid path data')
    origin_ns = int(bag_stamps[0])
    times = (bag_stamps[indices] - origin_ns).astype(np.float64) * 1e-9
    time_rounding = float(np.max(np.abs(paths[:, :, 13] - times[:, None])))
    if time_rounding > 1e-5:
        raise ValueError(f'Export path times differ from exact bag times: {time_rounding}')
    calibration_path = args.sequence / 'scene_calibration_v2.json'
    calibration = json.loads(calibration_path.read_text())
    if calibration['source_frame'] != 'mocap' or calibration['scene_frame'] != 'mocap':
        raise ValueError('Expected untransformed mocap sequence')
    world_from_camera = np.asarray(calibration['scene_from_camera'], float)
    camera_from_world = np.linalg.inv(world_from_camera)
    table_calibration_path = REPO/'experiments/differentiable_mpm/data/episode14_dynamics_table_aligned_v1/scene_calibration_v2.json'
    table_calibration = json.loads(table_calibration_path.read_text())
    equivalent = np.linalg.inv(np.asarray(table_calibration['scene_from_camera'])) @ np.asarray(table_calibration['scene_from_source'])
    if not np.allclose(equivalent, camera_from_world, atol=1e-12):
        raise ValueError('Table-aligned simulator camera projection differs from exported mocap camera')
    # Read real RGB camera intrinsics from the bag, not a guessed/render camera.
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo
    with sqlite3.connect(f'file:{bag.database_path}?mode=ro', uri=True) as conn:
        row = conn.execute('SELECT m.data FROM messages m JOIN topics t ON m.topic_id=t.id WHERE t.name=? ORDER BY m.timestamp LIMIT 1',
                           ('/camera/camera/color/camera_info',)).fetchone()
    message = deserialize_message(row[0], CameraInfo)
    camera = calibration['camera']
    for key, actual in [('fx', message.k[0]), ('fy', message.k[4]), ('cx', message.k[2]), ('cy', message.k[5]),
                        ('width', message.width), ('height', message.height)]:
        if not np.isclose(camera[key], actual):
            raise ValueError(f'Camera calibration disagrees with bag CameraInfo: {key}')
    if not np.allclose(message.d, 0) or not np.allclose(camera.get('d', []), 0):
        raise ValueError('Nonzero distortion requires distortion-aware projection')
    assets = REPO/'experiments/differentiable_mpm/data/episode18_dynamics_icp_final_v1'
    geometry_path = args.geometry or assets/'tool_geometry.json'
    manifest = json.loads((assets/'collision_manifest.json').read_text())['tools']
    marker_from_tool = load_mounts(geometry_path)
    with reference_policy('frozen'):
        simulator = get_reference_modules(physics_version='corrected-v1').simulator
    visual = [(simulator.UR_TOOL_VISUAL_ORIGIN, simulator.UR_TOOL_VISUAL_RPY),
              (simulator.KINOVA_TOOL_VISUAL_ORIGIN, simulator.KINOVA_TOOL_VISUAL_RPY)]
    meshes = []
    for row, (offset, rpy) in zip(manifest, visual):
        file = assets/row['collision_mesh']
        if hashlib.sha256(file.read_bytes()).hexdigest() != row['collision_sha256']:
            raise ValueError('Collision mesh hash mismatch')
        vertices, triangles = simulator.load_binary_stl(file, .001)
        link_vertices = simulator.mesh_in_tool_frame(vertices, offset, simulator.rpy_to_matrix(rpy))
        meshes.append((link_vertices.astype(float), np.asarray(triangles).reshape(-1,3)))
    annotations = json.loads((args.bag/'temporal_annotations.json').read_text())
    boundaries = [b['pointcloud_ordinal'] for b in annotations['boundaries']]
    slerps = [Slerp(times, Rotation.from_quat(paths[:, i, 3:7])) for i in range(2)]
    return locals()


def pose_at(time, tool, context):
    path = context['paths'][:, tool]
    result = np.eye(4)
    result[:3, 3] = [np.interp(time, context['times'], path[:, i]) for i in range(3)]
    result[:3, :3] = context['slerps'][tool](time).as_matrix()
    return result


def draw_frame(rgb_ordinal, context, args):
    bag, camera = context['bag'], context['camera']
    stamp = bag.rgb.frames[rgb_ordinal].header_stamp_ns
    time = (stamp-context['origin_ns'])*1e-9
    rgb = context['reader'].get(rgb_ordinal)
    raw = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if raw.shape[:2] != (camera['height'], camera['width']):
        raise ValueError('RGB image size differs from camera intrinsics')
    overlay = raw.copy()
    camera_from_world = context['camera_from_world']
    projected = []
    for tool in range(2):
        marker_pose = pose_at(time, tool, context)
        tool_pose = marker_pose @ context['marker_from_tool'][tool]
        vertices, triangles = context['meshes'][tool]
        pixels, valid = project(transform(vertices, tool_pose), camera_from_world, camera)
        mask = triangle_silhouette(pixels, valid, triangles, raw.shape[:2])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colored = overlay.copy()
        colored[mask > 0] = COLORS[tool]
        overlay = cv2.addWeighted(overlay, .92, colored, .08, 0)
        cv2.drawContours(overlay, contours, -1, (15,15,15), 3, cv2.LINE_AA)
        cv2.drawContours(overlay, contours, -1, COLORS[tool], 1, cv2.LINE_AA)
        past = np.linspace(max(context['times'][0], time-.6), time, 25)
        trail_world = np.stack([(pose_at(t, tool, context) @ context['marker_from_tool'][tool])[:3,3] for t in past])
        trail, trail_valid = project(trail_world, camera_from_world, camera)
        if trail_valid.all():
            poly = np.rint(trail).astype(np.int32)
            cv2.polylines(overlay, [poly], False, (10,10,10), 5, cv2.LINE_AA)
            cv2.polylines(overlay, [poly], False, COLORS[tool], 2, cv2.LINE_AA)
        locations, visible = project(np.stack([tool_pose[:3,3],marker_pose[:3,3]]), camera_from_world, camera)
        if visible[0]:
            current_marker(overlay, locations[0], COLORS[tool], tool)
            label = tuple(np.rint(locations[0]+[9,-9]).astype(int))
            text(overlay, NAMES[tool], label, .45)
        if visible[1]:
            p = tuple(np.rint(locations[1]).astype(int))
            cv2.drawMarker(overlay, p, (15,15,15), cv2.MARKER_CROSS, 12, 3, cv2.LINE_AA)
            cv2.drawMarker(overlay, p, COLORS[tool], cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        projected.append({'tool':NAMES[tool], 'link_origin_uv':locations[0].tolist(), 'marker_origin_uv':locations[1].tolist()})
    nearest = int(np.argmin(np.abs(context['bag_stamps'] - stamp)))
    delta_ms = (int(context['bag_stamps'][nearest])-stamp)*1e-6
    chunk = int(np.searchsorted(context['boundaries'], nearest, side='right'))
    width, height = raw.shape[1], raw.shape[0]
    frame = np.full((height+132, width*2,3), (25,25,24), np.uint8)
    frame[48:48+height,:width] = raw
    frame[48:48+height,width:] = overlay
    text(frame, 'EPISODE 14  |  recorded RGB', (15,30), .65)
    text(frame, 'Exported paths + simulator tool geometry', (width+15,30), .60)
    y = height+72
    text(frame, f'pc {nearest:04d}  |  RGB {rgb_ordinal:04d}  |  t={time:.3f}s  |  chunk {chunk:02d}  |  RGB-PC dt {delta_ms:+.2f}ms', (15,y), .52)
    for tool,x in [(0,15),(1,135)]:
        current_marker(frame, np.array([x+5,y+29]), COLORS[tool], tool)
        text(frame,NAMES[tool],(x+19,y+34),.48)
    text(frame, '0.6s trail: tool-link origin, NOT tip   |   +: mocap marker   |   outline: simulator STL', (270,y+34), .46)
    text(frame, args.label, (15,y+56), .43, (192,192,187))
    return frame, {'rgb_ordinal':rgb_ordinal, 'timestamp_ns':str(stamp), 'time_s':time, 'nearest_pointcloud_ordinal':nearest,
                   'pointcloud_minus_rgb_ms':delta_ms, 'chunk':chunk, 'projected_origins':projected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, default=TRAIN/'DeformPath2_release/AltDynamics/AlternativeDynamicsBag_verified_11_12_20_20261007_125929/snimanje_16_6/episode14_dynamics')
    parser.add_argument('--sequence', type=Path, default=TRAIN/'episode14_dynamics_sequence_v4')
    parser.add_argument('--export', type=Path, default=TRAIN/'episode14_dynamics_hsv_roi4096_export_v4')
    parser.add_argument('--output', type=Path, default=DATASET/'visualization')
    parser.add_argument('--start', type=int, default=700)
    parser.add_argument('--end', type=int, default=1345)
    parser.add_argument('--stills-only', action='store_true')
    parser.add_argument('--still-frames', type=int, nargs='+', default=[732,858,1021,1257,1345], help='Raw pointcloud ordinals for still images')
    parser.add_argument('--geometry', type=Path, help='Override tool_geometry.json; marker_from_mesh follows simulator visual transform')
    parser.add_argument('--paths', type=Path, help='Override path tensor with identical frame indices and timestamps')
    parser.add_argument('--label', default='Saved tool-mount transforms reused; this visualization checks alignment, not a fitted correction.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    context = load_context(args)
    bag = context['bag']
    start_ns = bag.pointcloud.frames[args.start].header_stamp_ns
    end_ns = bag.pointcloud.frames[args.end].header_stamp_ns
    selected = [f.ordinal for f in bag.rgb.frames if start_ns <= f.header_stamp_ns <= end_ns]
    still_raw = args.still_frames
    stills = {bag.rgb_mappings[k].rgb_ordinal:k for k in still_raw if args.start <= k <= args.end}
    fps = (len(selected)-1)/((bag.rgb.frames[selected[-1]].header_stamp_ns-bag.rgb.frames[selected[0]].header_stamp_ns)*1e-9)
    output = args.output/'episode14_paths_vs_rgb.mp4'
    if not args.stills_only and output.exists():
        raise FileExistsError(output)
    rows = []
    with RgbFrameReader(bag, cache_frames=4) as reader:
        context['reader'] = reader
        encoder = None
        try:
            for k,rgb_ordinal in enumerate(selected):
                if args.stills_only and rgb_ordinal not in stills:
                    continue
                frame, row = draw_frame(rgb_ordinal, context, args)
                rows.append(row)
                if rgb_ordinal in stills:
                    cv2.imwrite(str(args.output/f'overlay_pc{stills[rgb_ordinal]:04d}.png'), frame)
                if not args.stills_only:
                    if encoder is None:
                        encoder = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','bgr24',
                                                    '-s',f'{frame.shape[1]}x{frame.shape[0]}','-r',str(fps),'-i','-',
                                                    '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
                    encoder.stdin.write(frame.tobytes())
                if k%100 == 0:
                    print(f'{k+1}/{len(selected)} RGB frames',flush=True)
        finally:
            if encoder is not None:
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('Video encoder failed')
    report = {
        'bag':str(args.bag),'sequence':str(args.sequence),'output_video':str(output),
        'range_pointcloud_inclusive':[args.start,args.end],'rendered_rgb_frames':len(rows),'fps':fps,
        'duration_between_first_last_headers_s':(end_ns-start_ns)*1e-9,
        'max_export_path_time_rounding_s':context['time_rounding'],
        'all_source_pointcloud_headers_exactly_match_export':True,
        'intrinsics_verified_against_bag_camera_info':True,
        'table_and_mocap_projection_equivalent':True,
        'pose_sampling':'Linear position + SLERP orientation at each RGB header timestamp, using exact associated pointcloud header times',
        'playback':'Constant frame rate preserves endpoint duration; tiny original frame-interval variations are not represented',
        'tool_geometry':'Frozen corrected-v1 simulator STL visual transforms, saved marker_from_mesh; no new fitting or guessed tip offset',
        'mounting_status':args.label,
        'geometry_path':str(context['geometry_path']),
        'geometry_sha256':hashlib.sha256(context['geometry_path'].read_bytes()).hexdigest(),
        'paths_path':str(context['paths_path']),
        'paths_sha256':hashlib.sha256(context['paths_path'].read_bytes()).hexdigest(),
        'silhouette_rasterization':'Union of independently filled triangles, 1/8 pixel coordinates',
        'tool_colors':{'UR5e':'#2a78d6','gen3':'#ea6834'},
        'frames':rows}
    report_path=args.output/('stills_alignment.json' if args.stills_only else 'alignment.json')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print('Wrote',report_path,flush=True)


if __name__ == '__main__':
    main()

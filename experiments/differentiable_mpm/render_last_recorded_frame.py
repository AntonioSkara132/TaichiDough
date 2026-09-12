"""Render the final recorded observation and tool poses; no simulation is run."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
INPUT = ROOT / 'data/episode18_table_aligned_v1'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
BG, INK = '#fcfcfb', '#242522'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mesh_transform_constants(path):
    constants = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    'UR_TOOL_VISUAL_ORIGIN', 'UR_TOOL_VISUAL_RPY',
                    'KINOVA_TOOL_VISUAL_ORIGIN', 'KINOVA_TOOL_VISUAL_RPY'}:
                    constants[target.id] = np.asarray(ast.literal_eval(node.value.args[0]), dtype=np.float32)
    if len(constants) != 4:
        raise ValueError('Expected four preserved mesh transform constants')
    return constants


def load_mesh(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], 'little')
    if len(raw) != 84 + 50 * count:
        raise ValueError('Invalid binary STL size')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    return np.frombuffer(raw, dtype=dtype, offset=84, count=count)['vertices'].reshape(-1, 3).astype(float) * .001


def format_axis(ax, lower, upper):
    # Matplotlib plots X,Z horizontally and the simulation's Y vertically.
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[2], upper[2]), zlim=(lower[1], upper[1]),
           xlabel='X offset (mm)', ylabel='Z offset (mm)', zlabel='Height above table (mm)')
    extent = upper - lower
    ax.set_box_aspect(extent[[0, 2, 1]])
    ax.set_proj_type('ortho')
    ax.view_init(elev=30, azim=-58)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)


def table(ax, lower, upper, alpha=1):
    x0, z0, x1, z1 = lower[0], lower[2], upper[0], upper[2]
    ax.add_collection3d(Poly3DCollection([[[x0, z0, 0], [x1, z0, 0], [x1, z1, 0], [x0, z1, 0]]],
                         facecolors='#e5e5df', edgecolors='#bfc1b8', linewidth=.7, alpha=alpha, zorder=0))


def main():
    paths = [INPUT / 'observed_points_scene.npz', INPUT / 'tool_trajectories_scene.npz',
             INPUT / 'scene_calibration_v2.json',
             ROOT / 'reference/taichi_viscoelastic_mpm_scene.py',
             REPO / 'meshes/ur_spathla_collision_solid.stl',
             REPO / 'meshes/gen3_spathla_collision_solid.stl']
    input_hashes = {str(p): sha(p) for p in paths}
    with np.load(paths[0], allow_pickle=False) as data:
        index = len(data['times']) - 1
        original = int(data['original_indices'][index])
        points = data['points'][data['offsets'][index]:data['offsets'][index + 1]].copy()
        timestamp = float(data['times'][index])
        elapsed = timestamp - float(data['times'][0])
    with np.load(paths[1], allow_pickle=False) as data:
        poses, names = data['poses'][index], data['names'].tolist()
        if len(data['times']) != index + 1 or not np.isclose(data['times'][index], elapsed, atol=1e-8):
            raise ValueError('Observation and tool timing differ')
    finite_count = len(points)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 1] >= .003)]
    origin = np.array([points[:, 0].mean(), 0, points[:, 2].mean()])
    local = (points - origin) * 1000
    # Interpolate only nearby observed samples; never fill volume or bridge large holes.
    faces = Delaunay(local[:, [0, 2]]).simplices
    triangles = local[faces]
    edges = np.stack([triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 1],
                      triangles[:, 0] - triangles[:, 2]], axis=1)
    faces = faces[np.linalg.norm(edges, axis=2).max(axis=1) <= 8.0]
    constants = mesh_transform_constants(paths[3])
    mesh_by_name = {'UR5e_spathla': (paths[4], 'UR'), 'gen3_spathla': (paths[5], 'KINOVA')}
    meshes = []
    for pose, name in zip(poses, names):
        path, prefix = mesh_by_name[name]
        vertices = load_mesh(path)
        vertices = vertices @ Rotation.from_euler('xyz', constants[prefix + '_TOOL_VISUAL_RPY']).as_matrix().T
        vertices += constants[prefix + '_TOOL_VISUAL_ORIGIN']
        vertices = vertices @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
        meshes.append((vertices - origin) * 1000)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    output = ROOT / 'runs' / ('episode18_last_recorded_frame_' + stamp + '_' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': '#60615c', 'ytick.color': '#60615c',
                         'axes.edgecolor': '#aaa9a2', 'figure.facecolor': BG, 'axes.facecolor': BG,
                         'savefig.facecolor': BG})
    subtitle = f'Processed frame {index} / {index} · original raw frame {original} · t = {elapsed:.3f} s · recorded data, not simulation'
    lower = np.array([-65., 0., -65.])
    upper = np.array([65., 45., 65.])
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    table(ax, lower, upper)
    ax.add_collection3d(Poly3DCollection(local[:, [0, 2, 1]][faces], facecolors=BLUE,
                         shade=True, lightsource=LightSource(azdeg=310, altdeg=50), linewidths=0, zorder=2))
    ax.scatter(local[:, 0], local[:, 2], local[:, 1], color=BLUE, s=2, linewidths=0, alpha=.55, zorder=3)
    format_axis(ax, lower, upper)
    fig.suptitle('Episode 18 · final recorded dough', x=.08, ha='left', fontsize=22, fontweight='bold')
    fig.text(.08, .918, subtitle, fontsize=11, color='#60615c')
    fig.text(.08, .055, f'{len(points):,} visible points · local surface interpolation ≤ 8 mm edges · no hidden-volume reconstruction', fontsize=10)
    fig.subplots_adjust(left=.02, right=.95, bottom=.1, top=.9)
    fig.savefig(output / '01_final_observed_dough.png', dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')
    all_points = np.concatenate([local, *meshes])
    lower, upper = all_points.min(axis=0) - 12, all_points.max(axis=0) + 12
    lower[1], upper[1] = min(0, lower[1]), max(45, upper[1])
    table(ax, lower, upper, alpha=.3)
    ax.scatter(local[:, 0], local[:, 2], local[:, 1], color=BLUE, s=7, linewidths=0, depthshade=False)
    for mesh, color, label in zip(meshes, (ORANGE, AQUA), ('UR5e tool', 'Kinova tool')):
        ax.add_collection3d(Poly3DCollection(mesh[:, [0, 2, 1]].reshape(-1, 3, 3), facecolors=color,
                             alpha=.75, edgecolors='none', linewidths=0))
        center = mesh.mean(axis=0)
        ax.text(center[0], center[2], center[1] + 12, label, color=INK, fontsize=10)
    format_axis(ax, lower, upper)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=c, markeredgecolor='none', markersize=9, label=l)
               for c, l in [(BLUE, 'Recorded dough points'), (ORANGE, 'UR5e tool'), (AQUA, 'Kinova tool')]]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.08, .015), ncol=3, frameon=False)
    fig.suptitle('Episode 18 · final frame with recorded tools', x=.08, ha='left', fontsize=21, fontweight='bold')
    fig.text(.08, .925, subtitle, fontsize=11, color='#60615c')
    fig.text(.08, .08, 'Actual collision meshes and final recorded poses · equal metric axes · transparent table at Y = 0', fontsize=10)
    fig.subplots_adjust(left=.02, right=.93, bottom=.14, top=.91)
    fig.savefig(output / '02_final_frame_with_tools.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9))
    order = np.argsort(local[:, 1])
    scatter = ax.scatter(local[order, 0], local[order, 2], c=local[order, 1], cmap='Blues', vmin=0, vmax=40, s=9, linewidths=0)
    ax.set(xlim=(-65, 65), ylim=(-65, 65), xlabel='X offset (mm)', ylabel='Z offset (mm)')
    ax.set_aspect('equal')
    ax.grid(color='#e8e8e2', linewidth=.6)
    ax.set_axisbelow(True)
    fig.colorbar(scatter, ax=ax, shrink=.7, pad=.04, label='Observed height above table (mm)')
    fig.suptitle('Episode 18 · final observed top view', x=.1, ha='left', fontsize=20, fontweight='bold')
    fig.text(.1, .923, f'Processed frame {index} · raw frame {original} · t = {elapsed:.3f} s · no simulation', color='#60615c')
    fig.subplots_adjust(left=.12, right=.88, top=.88, bottom=.11)
    fig.savefig(output / '03_final_observed_top_down.png', dpi=180)
    plt.close(fig)
    if {str(p): sha(p) for p in paths} != input_hashes:
        raise ValueError('Rendering inputs changed')
    manifest = {'schema': 'taichidough/final-recorded-frame-render/v1', 'processed_frame': index,
                'processed_frame_count': index + 1, 'original_raw_frame': original,
                'recorded_timestamp_s': timestamp, 'elapsed_s': elapsed, 'simulation_run': False,
                'calibration_run': False, 'geometry': 'final recorded visible points, not simulated particles',
                'points_before_clearance': finite_count, 'rendered_points': len(points),
                'floor_clearance_m': .003, 'max_observed_height_mm': float(local[:, 1].max()),
                'surface_interpolation': {'method': 'XZ Delaunay of observed samples', 'maximum_3d_edge_mm': 8.0,
                                          'triangles_retained': len(faces), 'hidden_volume_filled': False},
                'equal_metric_axes': True, 'axis_origin_scene_m': origin.tolist(),
                'tool_names': names, 'tool_poses_scene': poses.tolist(), 'input_sha256': input_hashes,
                'files': {p.name: sha(p) for p in output.glob('*.png')}, 'script_sha256': sha(Path(__file__))}
    (output / 'render_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output)
    print(json.dumps({key: manifest[key] for key in ['processed_frame', 'original_raw_frame', 'elapsed_s', 'rendered_points', 'max_observed_height_mm']}))


if __name__ == '__main__':
    main()

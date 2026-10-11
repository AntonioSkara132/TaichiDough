"""Project final recorded dough and collision meshes along +X, without simulation."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial.transform import Rotation

from render_last_recorded_frame import load_mesh, mesh_transform_constants

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
INPUT = ROOT / 'data/episode18_table_aligned_v1'
OUTPUT = ROOT / 'runs/episode18_last_recorded_frame_20260912T185532_f8ebbe'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = OUTPUT / '05_side_looking_plus_x_with_tools.png'
    if target.exists():
        raise FileExistsError(f'Refusing to replace {target}')
    previous = json.loads((OUTPUT / 'render_manifest.json').read_text())
    paths = [INPUT / 'observed_points_scene.npz', INPUT / 'tool_trajectories_scene.npz',
             ROOT / 'reference/taichi_viscoelastic_mpm_scene.py',
             REPO / 'meshes/ur_spathla_collision_solid.stl',
             REPO / 'meshes/gen3_spathla_collision_solid.stl']
    hashes = {str(path): sha(path) for path in paths}
    for path, actual in hashes.items():
        if actual != previous['input_sha256'][path]:
            raise ValueError(f'Render input differs from previous final frame: {path}')
    with np.load(paths[0], allow_pickle=False) as data:
        index = len(data['times']) - 1
        raw = int(data['original_indices'][index])
        elapsed = float(data['times'][index] - data['times'][0])
        points = data['points'][data['offsets'][index]:data['offsets'][index + 1]].copy()
    points = points[np.isfinite(points).all(axis=1) & (points[:, 1] >= previous['floor_clearance_m'])]
    with np.load(paths[1], allow_pickle=False) as data:
        poses, names = data['poses'][index], data['names'].tolist()
        if not np.isclose(data['times'][index], elapsed, atol=1e-8):
            raise ValueError('Observation and tool timestamps differ')
    origin = np.asarray(previous['axis_origin_scene_m'])
    local = (points - origin) * 1000
    constants = mesh_transform_constants(paths[2])
    mapping = {'UR5e_spathla': (paths[3], 'UR', '#eb6834', 'UR5e tool'),
               'gen3_spathla': (paths[4], 'KINOVA', '#1baf7a', 'Kinova tool')}
    meshes, labels, colors = [], [], []
    for pose, name in zip(poses, names):
        path, prefix, color, label = mapping[name]
        vertices = load_mesh(path)
        vertices = vertices @ Rotation.from_euler('xyz', constants[prefix + '_TOOL_VISUAL_RPY']).as_matrix().T
        vertices += constants[prefix + '_TOOL_VISUAL_ORIGIN']
        vertices = vertices @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
        meshes.append((vertices - origin) * 1000)
        labels.append(label)
        colors.append(color)
    all_geometry = np.concatenate([local, *meshes])
    low, high = all_geometry.min(axis=0), all_geometry.max(axis=0)
    low[1], high[1] = min(low[1], 0), max(high[1], 0)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'text.color': '#242522', 'axes.labelcolor': '#242522',
                         'xtick.color': '#60615c', 'ytick.color': '#60615c',
                         'axes.edgecolor': '#aaa9a2', 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    fig, ax = plt.subplots(figsize=(12, 8))
    for mesh, color, label in zip(meshes, colors, labels):
        triangles = mesh.reshape(-1, 3, 3)
        collection = PolyCollection(triangles[:, :, [2, 1]], facecolors=color,
                                    edgecolors='none', linewidths=0, alpha=.18)
        ax.add_collection(collection)
        center = mesh.mean(axis=0)
        ax.annotate(label, xy=(center[2], center[1]), xytext=(15, -22),
                    textcoords='offset points', color='#242522', fontsize=11,
                    arrowprops={'arrowstyle': '-', 'color': '#60615c', 'linewidth': .8})
    ax.scatter(local[:, 2], local[:, 1], color='#2a78d6', s=7, linewidths=0, alpha=.8, zorder=4)
    ax.axhline(0, color='#77796f', linewidth=1.3, zorder=3)
    ax.set(xlim=(low[2] - 16, high[2] + 16), ylim=(low[1] - 16, high[1] + 16),
           xlabel='Z offset (mm)', ylabel='Height above table, Y (mm)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(color='#e8e8e2', linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(high[2] + 12, 2, 'Table: Y = 0', ha='right', va='bottom', fontsize=10, color='#60615c')
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#2a78d6',
                      markeredgecolor='none', markersize=7, label='Recorded dough')]
    handles.extend(Patch(facecolor=color, alpha=.65, label=label) for color, label in zip(colors, labels))
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.08, .055), ncol=3, frameon=False)
    fig.suptitle('Episode 18 · final dough and tools, looking along +X',
                 x=.08, ha='left', fontsize=19, fontweight='bold')
    fig.text(.08, .923, f'Camera on −X side · frame {index} / raw {raw} · t = {elapsed:.3f} s · recorded data, not simulation',
             fontsize=11, color='#60615c')
    fig.text(.08, .03, 'Equal metric scale · transparent collision-mesh projection · full below-floor geometry retained', fontsize=10)
    fig.subplots_adjust(left=.1, right=.96, top=.87, bottom=.18)
    with target.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=180)
    plt.close(fig)
    if {str(path): sha(path) for path in paths} != hashes:
        raise ValueError('Inputs changed during rendering')
    manifest = {'schema': 'taichidough/final-recorded-side-tools/v1', 'processed_frame': index,
                'original_raw_frame': raw, 'elapsed_s': elapsed, 'view_direction': '+X',
                'camera_side': '-X', 'projection': 'orthographic Z-Y', 'equal_metric_scale': True,
                'tools_repositioned': False, 'below_floor_mesh_clipped': False, 'simulation_run': False,
                'tool_names': names, 'tool_poses_scene': poses.tolist(),
                'tool_bounds_scene_m': [{'min': (mesh / 1000 + origin).min(axis=0).tolist(),
                                         'max': (mesh / 1000 + origin).max(axis=0).tolist()} for mesh in meshes],
                'input_sha256': hashes, 'image_sha256': sha(target)}
    with (OUTPUT / '05_side_looking_plus_x_with_tools_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(target)


if __name__ == '__main__':
    main()

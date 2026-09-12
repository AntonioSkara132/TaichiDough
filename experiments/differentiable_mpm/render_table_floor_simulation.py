"""Render saved forward-simulation particles and prescribed tools, without rerunning physics."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

from render_last_recorded_frame import mesh_transform_constants, load_mesh, sha, table, format_axis

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT):
        raise ValueError('Run must be inside the experiment')
    result_path = run / 'simulation_result.json'
    result = json.loads(result_path.read_text())
    output = run / 'renders'
    output.mkdir(exist_ok=False)
    records = list(result['frames'])
    if result['completed_steps'] != records[-1]['step']:
        records.append({'source_frame': None, 'original_source_frame': None,
                        'step': result['completed_steps'], 'sim_time_s': result['sim_time_s'],
                        'particles': 'last_valid_particles.npy', 'tool_poses': result['last_valid_tool_poses']})
    initial = np.load(run / records[0]['particles'])
    origin = np.array([initial[:, 0].mean(), 0, initial[:, 2].mean()])
    constants = mesh_transform_constants(ROOT / 'reference/taichi_viscoelastic_mpm_scene.py')
    meshes = []
    mesh_paths = [REPO / 'meshes/ur_spathla_collision_solid.stl', REPO / 'meshes/gen3_spathla_collision_solid.stl']
    for path, prefix in zip(mesh_paths, ('UR', 'KINOVA')):
        vertices = load_mesh(path)
        vertices = vertices @ Rotation.from_euler('xyz', constants[prefix + '_TOOL_VISUAL_RPY']).as_matrix().T
        vertices += constants[prefix + '_TOOL_VISUAL_ORIGIN']
        meshes.append(vertices)

    def load_record(record):
        points = (np.load(run / record['particles']) - origin) * 1000
        tools = [(v @ Rotation.from_quat(np.asarray(p)[3:]).as_matrix().T + np.asarray(p)[:3] - origin) * 1000
                 for v, p in zip(meshes, record['tool_poses'])]
        return points, tools

    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for record in records:
        points, tools = load_record(record)
        all_points = np.concatenate([points, *tools])
        lower = np.minimum(lower, all_points.min(axis=0))
        upper = np.maximum(upper, all_points.max(axis=0))
    lower -= 15
    upper += 15
    lower[1] = min(lower[1], -10)
    upper[1] = max(upper[1], 50)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': '#242522',
                         'axes.labelcolor': '#242522', 'xtick.color': '#60615c', 'ytick.color': '#60615c',
                         'axes.edgecolor': '#aaa9a2', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                         'savefig.facecolor': '#fcfcfb'})
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=c, markersize=8,
                      markeredgecolor='none', label=l) for c, l in
               ((BLUE, 'Simulated dough'), (ORANGE, 'UR5e tool'), (AQUA, 'Kinova tool'))]
    stop_note = 'Full episode completed' if result['status'] == 'completed' else 'Stopped before target: last valid state only'

    def side(record, path, small=False):
        points, tools = load_record(record)
        fig, ax = plt.subplots(figsize=(12, 6.5))
        for vertices, color in zip(tools, (ORANGE, AQUA)):
            ax.add_collection(PolyCollection(vertices[:, [2, 1]].reshape(-1, 3, 2),
                              facecolors=color, edgecolors='none', alpha=.32))
        ax.scatter(points[:, 2], points[:, 1], s=.6 if small else 1.2, alpha=.45, color=BLUE, linewidths=0)
        ax.axhline(0, color='#555750', linewidth=1.2)
        ax.set(xlim=(lower[2], upper[2]), ylim=(lower[1], upper[1]),
               xlabel='Z offset (mm)', ylabel='Height above table Y (mm)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color='#e8e8e2', linewidth=.6)
        ax.set_axisbelow(True)
        frame = f'processed frame {record["source_frame"]}' if record['source_frame'] is not None else 'between recorded frames'
        fig.suptitle('Simulated Episode 18 · corrected table floor · looking along +X', x=.075, ha='left', fontsize=17, fontweight='bold')
        fig.text(.075, .91, f't = {record["sim_time_s"]:.4f} s · step {record["step"]} · {frame} · table Y = 0', fontsize=11)
        fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.075, .055), ncol=3, frameon=False)
        fig.text(.075, .025, 'Forward simulation · inherited material guesses · original tool geometry · equal metric scale', fontsize=9)
        fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.18)
        fig.savefig(path, dpi=100 if small else 180)
        plt.close(fig)

    chosen = sorted(set([0, len(records) // 2, len(records) - 1]))
    for i, index in enumerate(chosen):
        side(records[index], output / f'{i + 1:02d}_simulated_side_step_{records[index]["step"]:06d}.png')
    final = records[-1]
    points, tools = load_record(final)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')
    table(ax, lower, upper, alpha=.25)
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=1.8, color=BLUE, alpha=.65, linewidths=0, depthshade=False)
    for vertices, color in zip(tools, (ORANGE, AQUA)):
        ax.add_collection3d(Poly3DCollection(vertices[:, [0, 2, 1]].reshape(-1, 3, 3),
                            facecolors=color, edgecolors='none', alpha=.38))
    format_axis(ax, lower, upper)
    fig.suptitle('Simulated dough and tools · last valid state', x=.075, ha='left', fontsize=21, fontweight='bold')
    fig.text(.075, .91, f't = {final["sim_time_s"]:.4f} s · step {final["step"]} · {stop_note}', fontsize=11)
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.075, .015), ncol=3, frameon=False)
    fig.subplots_adjust(left=.03, right=.92, bottom=.11, top=.86)
    fig.savefig(output / '04_simulated_last_valid_perspective.png', dpi=180)
    plt.close(fig)

    movie_dir = output / 'video_frames'
    movie_dir.mkdir()
    indices = sorted(set([*range(0, len(records), 3), len(records) - 1]))
    for number, index in enumerate(indices):
        side(records[index], movie_dir / f'frame_{number:05d}.png', small=True)
    video = output / 'simulated_table_floor.mp4'
    with (output / 'ffmpeg.log').open('x') as stream:
        subprocess.run(['ffmpeg', '-nostdin', '-y', '-framerate', '10', '-i', str(movie_dir / 'frame_%05d.png'),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(video)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    manifest = {'simulation_result_sha256': sha(result_path), 'script_sha256': sha(Path(__file__)),
                'status': result['status'], 'last_simulation_time_s': result['sim_time_s'],
                'last_valid_step': result['completed_steps'], 'axis_origin_scene_m': origin.tolist(),
                'video_frame_count': len(indices), 'video_fps': 10,
                'video_sampling': 'Every third saved recorded-frame state, plus last valid state; displayed timestamp is authoritative.',
                'geometry': 'Actual simulated particles and original prescribed collision meshes, no surface smoothing.',
                'outputs': {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}}
    (output / 'render_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()

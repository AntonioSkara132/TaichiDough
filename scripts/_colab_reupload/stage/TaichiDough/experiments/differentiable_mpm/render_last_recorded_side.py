"""Orthographic final recorded dough view from negative X toward positive X."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runs/episode18_last_recorded_frame_20260912T185532_f8ebbe'
SOURCE = ROOT / 'data/episode18_table_aligned_v1/observed_points_scene.npz'


def main():
    path = OUTPUT / '04_side_looking_plus_x.png'
    if path.exists():
        raise FileExistsError(f'Refusing to replace {path}')
    previous = json.loads((OUTPUT / 'render_manifest.json').read_text())
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_hash != previous['input_sha256'][str(SOURCE)]:
        raise ValueError('Observed points differ from the previous final-frame renders')
    with np.load(SOURCE, allow_pickle=False) as data:
        index = len(data['times']) - 1
        raw = int(data['original_indices'][index])
        elapsed = float(data['times'][index] - data['times'][0])
        points = data['points'][data['offsets'][index]:data['offsets'][index + 1]].copy()
    points = points[np.isfinite(points).all(axis=1) & (points[:, 1] >= previous['floor_clearance_m'])]
    origin = np.asarray(previous['axis_origin_scene_m'])
    local = (points - origin) * 1000
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'text.color': '#242522', 'axes.labelcolor': '#242522',
                         'xtick.color': '#60615c', 'ytick.color': '#60615c',
                         'axes.edgecolor': '#aaa9a2', 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.scatter(local[:, 2], local[:, 1], color='#2a78d6', s=10, linewidths=0, alpha=.7)
    ax.axhline(0, color='#77796f', linewidth=1.3)
    ax.set(xlim=(-65, 65), ylim=(-3, 43), xlabel='Z offset (mm)', ylabel='Height above table, Y (mm)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.grid(color='#e8e8e2', linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(63, .9, 'Table: Y = 0', ha='right', va='bottom', fontsize=10, color='#60615c')
    fig.suptitle('Episode 18 · final dough, looking along +X', x=.08, ha='left', fontsize=21, fontweight='bold')
    fig.text(.08, .9, f'Camera on −X side · Z horizontal / Y vertical · frame {index} / raw {raw} · t = {elapsed:.3f} s',
             fontsize=11, color='#60615c')
    fig.text(.08, .045, f'{len(points):,} recorded visible points · equal metric scale · orthographic projection · not simulation', fontsize=10)
    fig.subplots_adjust(left=.1, right=.96, top=.84, bottom=.19)
    with path.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=180)
    plt.close(fig)
    manifest = {'schema': 'taichidough/final-recorded-side-render/v1', 'processed_frame': index,
                'original_raw_frame': raw, 'elapsed_s': elapsed, 'rendered_points': len(points),
                'view_direction': '+X', 'camera_side': '-X', 'horizontal_axis': '+Z',
                'vertical_axis': '+Y', 'equal_metric_axes': True, 'projection': 'orthographic',
                'simulation_run': False, 'input_sha256': source_hash,
                'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with (OUTPUT / '04_side_looking_plus_x_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(path)


if __name__ == '__main__':
    main()

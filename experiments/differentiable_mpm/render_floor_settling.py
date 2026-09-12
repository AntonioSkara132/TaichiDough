"""Render recorded snapshots from the tools-disabled floor simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BLUE = '#2a78d6'
ORANGE = '#eb6834'
BACKGROUND = '#fcfcfb'
INK = '#242622'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    result_path = root / 'simulation_result.json'
    result = json.loads(result_path.read_text())
    out = root / 'renders'
    out.mkdir(exist_ok=False)
    frames_dir = out / 'video_frames'
    frames_dir.mkdir()
    frames = result['frames']
    points = [np.load(root / row['particles'], allow_pickle=False) for row in frames]
    center_z = float(np.mean(points[0][:, 2]))
    z_min = min(float(p[:, 2].min()) for p in points)
    z_max = max(float(p[:, 2].max()) for p in points)
    y_min = min(0, min(float(p[:, 1].min()) for p in points))
    y_max = max(float(p[:, 1].max()) for p in points)
    xlim = ((z_min - center_z) * 1000 - 8, (z_max - center_z) * 1000 + 8)
    ylim = (y_min * 1000 - 3, max(25, y_max * 1000 + 8))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': '#60625e', 'ytick.color': '#60625e'})

    def side(p, row, path, dpi=100):
        fig, ax = plt.subplots(figsize=(12, 5), facecolor=BACKGROUND)
        ax.set_facecolor(BACKGROUND)
        ax.scatter((p[:, 2] - center_z) * 1000, p[:, 1] * 1000, s=1.8,
                   color=BLUE, alpha=.35, linewidths=0, rasterized=True)
        ax.axhline(0, color='#6c7066', lw=1.5)
        ax.set(xlim=xlim, ylim=ylim, xlabel='Z offset from initial center (mm)',
               ylabel='Height above table, Y (mm)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color='#e5e6e2', linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle('Simulated dough settling · tools disabled · looking along +X',
                     x=.075, ha='left', fontsize=17, fontweight='bold', y=.97)
        fig.text(.075, .88, f't = {row["time_s"]:.2f} s   |   minimum Y = {row["min_y_m"] * 1000:.3f} mm   |   grid 48, dx = 20.833 mm', fontsize=12)
        fig.text(.075, .04, '24,000 simulated particles · gravity only plus material/floor response · equal metric scale · table at Y = 0', fontsize=10)
        fig.subplots_adjust(left=.09, right=.98, top=.79, bottom=.17)
        fig.savefig(path, dpi=dpi, facecolor=BACKGROUND)
        plt.close(fig)

    side(points[0], frames[0], out / '01_initial_side.png', dpi=180)
    side(points[-1], frames[-1], out / '02_final_side.png', dpi=180)
    for i, (p, row) in enumerate(zip(points, frames)):
        side(p, row, frames_dir / f'frame_{i:05d}.png')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-framerate', '50',
               '-i', str(frames_dir / 'frame_%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
               '-crf', '19', '-movflags', '+faststart', str(out / 'floor_settling_no_tools.mp4')]
    subprocess.run(command, check=True)
    times = np.array([row['time_s'] for row in frames])
    minimum = np.array([row['min_y_m'] * 1000 for row in frames])
    com = np.array([row['com_y_m'] * 1000 for row in frames])
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.plot(times, minimum, color=BLUE, linewidth=2, label='Lowest particle')
    ax.plot(times, com, color=ORANGE, linewidth=2, linestyle='--', label='Center of mass')
    ax.axhline(0, color='#6c7066', linewidth=1)
    ax.set(xlabel='Simulated time (s)', ylabel='Height above table (mm)', xlim=(0, times[-1]))
    ax.legend(frameon=False)
    ax.grid(color='#e5e6e2', linewidth=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.annotate(f'Lowest: {minimum[-1]:.3f} mm', (times[-1], minimum[-1]),
                xytext=(-8, 14), textcoords='offset points', ha='right')
    ax.annotate(f'Center of mass: {com[-1]:.3f} mm', (times[-1], com[-1]),
                xytext=(-8, 14), textcoords='offset points', ha='right')
    fig.suptitle('Tools-disabled settling · particle heights', fontweight='bold')
    fig.text(.125, .01, 'Snapshots every 0.02 s; tabular values in settling_metrics.csv. No vertical repositioning.', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(out / '03_height_history.png', dpi=180, facecolor=BACKGROUND)
    plt.close(fig)
    manifest = {'schema': 'taichidough/floor-settling-render/v1', 'simulation_result_sha256': sha(result_path),
                'script_sha256': sha(__file__), 'view': '+X orthographic; Z horizontal, Y vertical; equal metric scale',
                'simulation_time_s': result['time_s'], 'particle_count': len(points[0]),
                'video_frames': len(frames), 'video_fps': 50, 'video_command': command,
                'palette_validation': 'blue #2a78d6 and orange #eb6834 passed light-mode validator',
                'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    with (out / 'render_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print('Rendered: ' + str(out))


if __name__ == '__main__':
    main()

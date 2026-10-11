"""Render recorded tool clouds against original and candidate rigid mesh transforms."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, SURFACE = '#20211f', '#666761', '#fcfcfb'


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def optical(points, camera_from_marker, camera):
    p = transform(points, camera_from_marker)
    return np.stack([camera['fx'] * p[..., 0] / p[..., 2] + camera['cx'],
                     camera['fy'] * p[..., 1] / p[..., 2] + camera['cy']], axis=-1)


def save(fig, path):
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def axes_style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#a5a69f')
    ax.grid(color='#e2e3de', linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_aspect('equal', adjustable='box')


def run(directory):
    plt.rcParams.update({'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK,
                         'axes.titlecolor': INK, 'font.family': 'DejaVu Sans'})
    report = json.loads((directory / 'summary.json').read_text())
    calibration_path = next(p for p in report['source_hashes'] if p.endswith('scene_calibration_v2.json'))
    calibration = json.loads(Path(calibration_path).read_text())
    camera = calibration['camera']
    scene_from_source = np.asarray(calibration['scene_from_source'])
    for frame in (176, 387):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
        fig.suptitle(f'Episode 18 · partial tool registration · held-out frame {frame}', fontsize=19, fontweight='bold')
        for tool, result in enumerate(report['results']):
            data = np.load(directory / f'frame_{frame:04d}_tool_{tool}.npz')
            mesh = np.load(directory / f'mesh_{tool}.npz')['triangles']
            points = optical(data['points_marker'], data['camera_from_marker'], camera)
            matrices = [np.asarray(result['initial_marker_from_mesh']), np.asarray(result['candidate_marker_from_mesh'])]
            meshes = [optical(transform(mesh, matrix), data['camera_from_marker'], camera) for matrix in matrices]
            combined = np.concatenate([points, *[tri.reshape(-1, 2) for tri in meshes]])
            lower, upper = combined.min(0) - 8, combined.max(0) + 8
            for column, (triangles, label, color) in enumerate(zip(meshes, ['Original mesh', 'Candidate correction'], [ORANGE, GREEN])):
                ax = axes[tool, column]
                ax.add_collection(PolyCollection(triangles, facecolors=color, edgecolors='none', alpha=.12))
                # A small mesh sample makes the underlying geometry visible without dense triangle edges.
                samples = np.load(directory / f'mesh_{tool}.npz')['samples'][::25]
                projected = optical(transform(samples, matrices[column]), data['camera_from_marker'], camera)
                ax.scatter(projected[:, 0], projected[:, 1], s=2, c=color, alpha=.5, label=label)
                ax.scatter(points[:, 0], points[:, 1], s=5, c=BLUE, alpha=.7, label='HSV tool observations')
                ax.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]), xlabel='Camera u (pixels)', ylabel='Camera v (pixels)')
                ax.set_title(f'{result["name"]} · {label}')
                axes_style(ax)
                ax.legend(loc='best', fontsize=9, framealpha=.9)
        fig.text(.04, .015, 'Camera and recorded poses fixed · 1,000 observed points/tool · candidate is not automatically applied', fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, .95))
        save(fig, directory / f'camera_overlay_heldout_{frame:04d}.png')
    frame = 387
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=SURFACE)
    fig.suptitle('Final recorded frame · +X side view · before and candidate correction', fontsize=18, fontweight='bold')
    for tool, result in enumerate(report['results']):
        data = np.load(directory / f'frame_{frame:04d}_tool_{tool}.npz')
        mesh = np.load(directory / f'mesh_{tool}.npz')['triangles']
        scene_from_marker = scene_from_source @ data['source_from_marker']
        points = transform(data['points_marker'], scene_from_marker)
        matrices = [np.asarray(result['initial_marker_from_mesh']), np.asarray(result['candidate_marker_from_mesh'])]
        meshes = [transform(mesh, scene_from_marker @ matrix) for matrix in matrices]
        z_origin = float(np.median(points[:, 2]))
        combined = np.concatenate([points, *[tri.reshape(-1, 3) for tri in meshes]])
        lower, upper = combined.min(0), combined.max(0)
        for column, (triangles, label, color) in enumerate(zip(meshes, ['Original mesh', 'Candidate correction'], [ORANGE, GREEN])):
            ax = axes[tool, column]
            projected = np.stack([(triangles[..., 2] - z_origin) * 1000, triangles[..., 1] * 1000], axis=-1)
            ax.add_collection(PolyCollection(projected, facecolors=color, edgecolors='none', alpha=.13))
            ax.scatter((points[:, 2] - z_origin) * 1000, points[:, 1] * 1000,
                       s=7, c=BLUE, alpha=.75, label='HSV tool observations')
            ax.plot([], [], color=color, linewidth=5, label=label)
            ax.axhline(0, color=MUTED, linewidth=1.2, label='Table Y=0')
            ax.set(xlim=((lower[2] - z_origin) * 1000 - 8, (upper[2] - z_origin) * 1000 + 8),
                   ylim=(min(lower[1] * 1000 - 8, -8), upper[1] * 1000 + 8),
                   xlabel='Z offset (mm)', ylabel='Height above table Y (mm)')
            ax.set_title(f'{result["name"]} · {label}')
            axes_style(ax)
            ax.legend(loc='best', fontsize=9, framealpha=.9)
    fig.text(.04, .015, 'Equal metric scale · full tool extents retained · no floor-based clipping or pose adjustment', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    save(fig, directory / 'side_view_heldout_0387.png')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    records = []
    for ax, result in zip(axes, report['results']):
        for key, label, color, marker in [('heldout_before', 'Original', ORANGE, 's'), ('heldout_after', 'Candidate', GREEN, 'o')]:
            rows = result[key]['frames']
            x = [row['processed_frame'] for row in rows]
            y = [row['median_distance_mm'] for row in rows]
            ax.plot(x, y, color=color, marker=marker, markersize=5, linewidth=1.7, label=label)
            ax.annotate(label, (x[-1], y[-1]), xytext=(5, 0), textcoords='offset points', fontsize=9)
            for row in rows:
                records.append({'tool': result['name'], 'transform': label, **{k:v for k,v in row.items() if not isinstance(v,list)}})
        ax.set_title(result['name'])
        ax.set(xlabel='Held-out processed frame', ylabel='Median observation-to-visible-mesh distance (mm)', ylim=(0, None))
        ax.grid(color='#e2e3de', linewidth=.7)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper right')
        ax.margins(x=.18)
    fig.suptitle('Independent held-out registration residuals', fontsize=18, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, directory / 'heldout_residuals.png')
    with (directory / 'heldout_residuals.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)

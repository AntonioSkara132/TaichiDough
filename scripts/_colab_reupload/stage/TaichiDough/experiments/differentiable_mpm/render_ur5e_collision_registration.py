"""Render the updated UR5e collision-mesh registration without changing its fit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from .render_tool_registration import transform, optical, save, axes_style, BLUE, ORANGE, GREEN, INK, MUTED, SURFACE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(directory):
    plt.rcParams.update({'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK,
                         'axes.titlecolor': INK, 'font.family': 'DejaVu Sans'})
    report = json.loads((directory / 'summary.json').read_text())
    result = report['results'][0]
    calibration_path = next(p for p in report['source_hashes'] if p.endswith('scene_calibration_v2.json'))
    calibration = json.loads(Path(calibration_path).read_text())
    scene_from_source = np.asarray(calibration['scene_from_source'])
    camera = calibration['camera']
    model = np.load(directory / 'mesh_0.npz')
    mesh, samples = model['triangles'], model['samples'][::20]
    matrices = [np.asarray(result['initial_marker_from_mesh']), np.asarray(result['candidate_marker_from_mesh'])]
    labels = ['Updated mesh · original transform', 'Updated mesh · fitted transform']
    outputs = []
    for frame in (176, 387):
        data = np.load(directory / f'frame_{frame:04d}_tool_0.npz')
        points = optical(data['points_marker'], data['camera_from_marker'], camera)
        meshes = [optical(transform(mesh, matrix), data['camera_from_marker'], camera) for matrix in matrices]
        combined = np.concatenate([points, *[tri.reshape(-1, 2) for tri in meshes]])
        lower, upper = combined.min(0) - 10, combined.max(0) + 10
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), facecolor=SURFACE)
        fig.suptitle(f'UR5e mirrored collision mesh · held-out frame {frame}', fontsize=19, fontweight='bold')
        for column, (triangles, label, color) in enumerate(zip(meshes, labels, [ORANGE, GREEN])):
            ax = axes[column]
            ax.add_collection(PolyCollection(triangles, facecolors=color, edgecolors='none', alpha=.12))
            projected = optical(transform(samples, matrices[column]), data['camera_from_marker'], camera)
            ax.scatter(projected[:, 0], projected[:, 1], s=3, c=color, alpha=.55, marker='x', label='Mesh samples')
            ax.scatter(points[:, 0], points[:, 1], s=6, c=BLUE, alpha=.7, label='Measured tool points')
            ax.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]), xlabel='Camera u (pixels)', ylabel='Camera v (pixels)')
            ax.set_title(label)
            axes_style(ax)
            ax.legend(loc='best', fontsize=9, framealpha=.95)
        fig.text(.04, .02, 'Same updated STL in both panels · camera and recorded motion fixed · no simulation settings changed', fontsize=10)
        fig.tight_layout(rect=(0, .06, 1, .92))
        path = directory / f'camera_overlay_heldout_{frame:04d}.png'
        save(fig, path); outputs.append(path)

    frame = 387
    data = np.load(directory / f'frame_{frame:04d}_tool_0.npz')
    scene_from_marker = scene_from_source @ data['source_from_marker']
    points = transform(data['points_marker'], scene_from_marker)
    meshes = [transform(mesh, scene_from_marker @ matrix) for matrix in matrices]
    z_origin = float(np.median(points[:, 2]))
    combined = np.concatenate([points, *[tri.reshape(-1, 3) for tri in meshes]])
    lower, upper = combined.min(0), combined.max(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
    fig.suptitle('UR5e final frame · +X side view · updated collision mesh', fontsize=18, fontweight='bold')
    for ax, triangles, label, color in zip(axes, meshes, labels, [ORANGE, GREEN]):
        projected = np.stack([(triangles[..., 2] - z_origin) * 1000, triangles[..., 1] * 1000], axis=-1)
        ax.add_collection(PolyCollection(projected, facecolors=color, edgecolors='none', alpha=.14))
        ax.scatter((points[:, 2] - z_origin) * 1000, points[:, 1] * 1000,
                   s=8, c=BLUE, alpha=.8, label='Measured tool points')
        ax.plot([], [], color=color, linewidth=5, label='Updated collision mesh')
        ax.axhline(0, color=MUTED, linewidth=1.3, label='Table Y = 0')
        ax.set(xlim=((lower[2] - z_origin) * 1000 - 8, (upper[2] - z_origin) * 1000 + 8),
               ylim=(min(lower[1] * 1000 - 8, -8), upper[1] * 1000 + 8),
               xlabel='Z offset (mm)', ylabel='Height above table Y (mm)')
        ax.set_title(label)
        axes_style(ax)
        ax.legend(loc='best', fontsize=9, framealpha=.95)
    fig.text(.04, .02, 'Equal metric scale · all tool extents retained · no table-based clipping or repositioning', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .92))
    path = directory / 'side_view_heldout_0387.png'
    save(fig, path); outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    records = []
    for ax, metric, title in zip(axes, ['median_distance_mm', 'p90_distance_mm'], ['Median distance', '90th-percentile distance']):
        for key, label, color, marker in [('heldout_before', 'Original transform', ORANGE, 's'),
                                          ('heldout_after', 'Fitted transform', GREEN, 'o')]:
            rows = result[key]['frames']
            x, y = [row['processed_frame'] for row in rows], [row[metric] for row in rows]
            ax.plot(x, y, color=color, marker=marker, markersize=5, linewidth=1.7, label=label)
            ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points', fontsize=9)
            if metric == 'median_distance_mm':
                records.extend({'transform': label, **{k:v for k,v in row.items() if not isinstance(v,list)}} for row in rows)
        ax.set_title(title)
        ax.set(xlabel='Held-out processed frame', ylabel='Observation-to-visible-mesh distance (mm)', ylim=(0, None))
        ax.grid(color='#e2e3de', linewidth=.7)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper left', fontsize=9)
        ax.margins(x=.30)
    fig.suptitle('UR5e · independent held-out residuals · updated collision STL', fontsize=17, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .9))
    path = directory / 'heldout_residuals.png'
    save(fig, path); outputs.append(path)
    csv_path = directory / 'heldout_residuals.csv'
    with csv_path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    outputs.append(csv_path)
    manifest = {'schema': 'taichidough/ur5e-registration-renders/v1',
        'summary_sha256': sha(directory / 'summary.json'), 'renderer_sha256': sha(__file__),
        'input_mesh_sha256': report['source_hashes'][report['fitting_target']],
        'palette': [BLUE, ORANGE, GREEN], 'palette_validation': 'all-pairs light passed; labels provide contrast relief',
        'final_mesh_minimum_y_m': {'original_transform': float(meshes[0][..., 1].min()),
                                  'fitted_transform': float(meshes[1][..., 1].min())},
        'output_sha256': {p.name: sha(p) for p in outputs}}
    with (directory / 'render_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')
    print(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)

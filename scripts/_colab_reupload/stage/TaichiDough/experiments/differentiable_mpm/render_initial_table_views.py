"""Render actual table-aligned initial geometry without running the simulator."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parent
BLUE = '#2a78d6'
ORANGE = '#eb6834'
BG = '#fcfcfb'
INK = '#242522'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def depth_image(points, calibration):
    camera = calibration['camera']
    inverse = np.linalg.inv(np.asarray(calibration['scene_from_camera']))
    optical = points @ inverse[:3, :3].T + inverse[:3, 3]
    valid = (optical[:, 2] > camera.get('zNear', .01)) & (optical[:, 2] < camera.get('zFar', 5))
    optical = optical[valid]
    u = np.floor(camera['fx'] * optical[:, 0] / optical[:, 2] + camera['cx']).astype(int)
    v = np.floor(camera['fy'] * optical[:, 1] / optical[:, 2] + camera['cy']).astype(int)
    width, height = camera['width'], camera['height']
    result = np.full((height, width), np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = u + dx, v + dy
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            np.minimum.at(result, (y[inside], x[inside]), optical[inside, 2])
    result[~np.isfinite(result)] = np.nan
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/episode18_table_aligned_v1')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    output = (args.output_dir or ROOT / 'runs' / ('episode18_table_aligned_initial_views_' + stamp + '_' + uuid.uuid4().hex[:6])).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Choose a fresh output directory within the experiment')
    output.mkdir(parents=True)
    frame = args.input_dir / 'reconstruction/episode18_kugla/frame_0000'
    paths = {name: frame / (name + '.npy') for name in ('sampled_particles_xyz', 'voxel_centers_xyz', 'real_points_xyz', 'surface_particles_xyz')}
    particles, voxels, observed, surface = (np.load(paths[name], allow_pickle=False) for name in paths)
    metadata_path = frame / 'reconstruction_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    calibration_path = args.input_dir / 'scene_calibration_v2.json'
    calibration = json.loads(calibration_path.read_text())
    origin = np.array([particles[:, 0].mean(), 0, particles[:, 2].mean()])
    local = (particles - origin) * 1000
    surface_local = (surface - origin) * 1000
    spacing = metadata['voxel_size']
    indices = np.rint((voxels - voxels.min(axis=0)) / spacing).astype(int) + 1
    occupancy = np.zeros(indices.max(axis=0) + 2, dtype=np.float32)
    occupancy[tuple(indices.T)] = 1
    vertices, faces, _, _ = marching_cubes(occupancy, .5, spacing=(spacing,) * 3)
    vertices += voxels.min(axis=0) - spacing
    mesh_local = (vertices - origin) * 1000
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': '#666762', 'ytick.color': '#666762',
                         'axes.edgecolor': '#aaa9a2', 'figure.facecolor': BG, 'axes.facecolor': BG,
                         'savefig.facecolor': BG})

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    horizontal = 100
    plane = Poly3DCollection([[[-horizontal, -horizontal, 0], [horizontal, -horizontal, 0],
                               [horizontal, horizontal, 0], [-horizontal, horizontal, 0]]],
                             facecolors='#e5e5df', edgecolors='#cccfc6', linewidth=.7, zorder=0)
    ax.add_collection3d(plane)
    triangles = mesh_local[:, [0, 2, 1]][faces]
    mesh = Poly3DCollection(triangles, facecolors=BLUE, linewidths=0, shade=True,
                           lightsource=LightSource(azdeg=310, altdeg=45), zorder=2)
    ax.add_collection3d(mesh)
    ax.set(xlim=(-100, 100), ylim=(-100, 100), zlim=(0, 55),
           xlabel='X offset (mm)', ylabel='Z offset (mm)', zlabel='Height above table (mm)')
    ax.set_box_aspect((200, 200, 55))
    ax.view_init(elev=26, azim=-58)
    ax.set_proj_type('ortho')
    ax.set_xticks([-75, -25, 25, 75])
    ax.set_yticks([-75, -25, 25, 75])
    ax.set_zticks([0, 20, 40])
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)
    fig.suptitle('New dough reconstruction · perspective', x=.08, ha='left', fontsize=21, fontweight='bold')
    fig.text(.08, .915, 'Initial state · occupied-voxel boundary · table at Y = 0 · equal metric scale', color='#60615c')
    fig.text(.08, .055, '24,000 particles   |   113.832 mL   |   maximum sampled height 15.945 mm', fontsize=11)
    fig.subplots_adjust(left=.04, right=.94, bottom=.09, top=.9)
    fig.savefig(output / '01_perspective.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9))
    order = np.argsort(local[:, 1])
    points = ax.scatter(local[order, 0], local[order, 2], c=local[order, 1], s=5,
                        cmap='Blues', vmin=0, vmax=16, linewidths=0)
    ax.set(xlabel='X offset (mm)', ylabel='Z offset (mm)', xlim=(-95, 95), ylim=(-95, 95))
    ax.set_aspect('equal')
    ax.grid(color='#e8e8e2', linewidth=.6)
    ax.set_axisbelow(True)
    fig.colorbar(points, ax=ax, shrink=.72, pad=.045, label='Particle height above table (mm)')
    fig.suptitle('New dough reconstruction · top view', x=.1, ha='left', fontsize=20, fontweight='bold')
    fig.text(.1, .925, 'Initial state · all 24,000 sampled particles · equal X/Z scale', color='#60615c')
    fig.subplots_adjust(left=.12, right=.87, bottom=.1, top=.88)
    fig.savefig(output / '02_top_down.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.scatter(local[:, 0], local[:, 1], s=2, color=BLUE, alpha=.12, linewidths=0, label='Sampled volume particles')
    ax.scatter(surface_local[:, 0], surface_local[:, 1], s=5, marker='x', linewidths=.5,
               color=ORANGE, alpha=.65, label='Reconstructed top samples')
    ax.axhline(0, color='#555750', linewidth=1.3)
    ax.text(82, 1.0, 'Table · Y = 0', ha='right', fontsize=10)
    ax.set(xlim=(-85, 85), ylim=(-3, 24), xlabel='X offset (mm)', ylabel='Height (mm)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.grid(color='#e8e8e2', linewidth=.6)
    ax.set_axisbelow(True)
    fig.suptitle('New dough reconstruction · side profile', x=.09, ha='left', fontsize=20, fontweight='bold')
    fig.text(.09, .865, 'Initial state · looking along Z · height is not exaggerated', color='#60615c')
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.09, .025), ncol=2, frameon=False,
                        markerscale=2)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    fig.subplots_adjust(left=.09, right=.97, bottom=.22, top=.78)
    fig.savefig(output / '03_side_profile.png', dpi=180)
    plt.close(fig)

    real_depth = depth_image(observed, calibration)
    initial_depth = depth_image(particles, calibration)
    union = np.isfinite(real_depth) | np.isfinite(initial_depth)
    rows, cols = np.nonzero(union)
    crop = [max(0, cols.min() - 12), min(union.shape[1], cols.max() + 13),
            max(0, rows.min() - 12), min(union.shape[0], rows.max() + 13)]
    finite = np.r_[real_depth[np.isfinite(real_depth)], initial_depth[np.isfinite(initial_depth)]] * 1000
    lower, upper = float(finite.min()), float(finite.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#ededE7')
    for ax, depth, title in zip(axes, (real_depth, initial_depth),
                                 ('Observed visible points · frame 0', 'Reconstructed particles · frame 0')):
        image = ax.imshow(depth * 1000, cmap=cmap, vmin=lower, vmax=upper, interpolation='nearest')
        ax.set(xlim=crop[:2], ylim=(crop[3], crop[2]), xlabel='Calibrated camera pixel U', ylabel='Camera pixel V')
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_aspect('equal')
    fig.suptitle('Recorded-camera view · observation and reconstruction', x=.07, ha='left', fontsize=20, fontweight='bold')
    fig.text(.07, .91, 'Identical calibrated camera and crop · nearest-depth 3 × 3 point splats · no simulation', color='#60615c')
    fig.subplots_adjust(left=.07, right=.86, top=.82, bottom=.16, wspace=.24)
    cax = fig.add_axes([.9, .21, .018, .52])
    fig.colorbar(image, cax=cax, label='Optical depth from camera (mm)')
    fig.text(.07, .055, 'Observed points retain the reconstruction’s 3 mm floor-clearance filter; gray pixels have no point support.', fontsize=10)
    fig.savefig(output / '04_calibrated_camera_comparison.png', dpi=180)
    plt.close(fig)

    manifest = {'initial_state_only': True, 'simulation_run': False, 'calibration_run': False,
                'input_directory': str(args.input_dir.resolve()), 'input_sha256': {str(p): sha(p) for p in [*paths.values(), metadata_path, calibration_path]},
                'rendering': 'Perspective uses marching-cubes boundary of actual occupied voxels, without smoothing. Other geometry views use actual sampled particles.',
                'axis_origin_scene_m': origin.tolist(), 'equal_metric_axes': True,
                'particle_count': len(particles), 'particle_max_height_mm': float(local[:, 1].max()),
                'files': {p.name: sha(p) for p in output.glob('*.png')}, 'script_sha256': sha(Path(__file__))}
    (output / 'render_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()

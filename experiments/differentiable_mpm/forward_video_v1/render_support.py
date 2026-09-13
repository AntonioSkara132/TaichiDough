"""Bounded-grid density extraction and recoverable video encoding; no simulation imports."""
from pathlib import Path
import json
import os
import shutil
import subprocess


def density_boundary(points, particle_volume, voxel=.0015, sigma=.8, level=.20,
                     tile_cells=64, max_triangles=2_000_000):
    """March disjoint cell tiles with Gaussian halos, without a full bounding-box grid.

    All particle contributions are included. The density threshold can omit sparse
    particles, as in the original renderer. Only grid memory is bounded per tile;
    the returned triangle array still requires memory proportional to mesh size.
    """
    import itertools
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected nonempty finite Nx3 particle positions')
    if not np.isfinite(particle_volume) or particle_volume <= 0:
        raise ValueError('Particle volume must be finite and positive')
    if not (np.isfinite(voxel) and voxel > 0 and np.isfinite(sigma) and sigma > 0
            and np.isfinite(level) and level > 0):
        raise ValueError('Voxel size, Gaussian sigma, and density level must be positive')
    if not isinstance(tile_cells, int) or not 4 <= tile_cells <= 128:
        raise ValueError('tile_cells must be an integer in [4, 128]')
    # Same truncate=3 kernel as the original renderer, with an extra halo sample.
    halo = int(3 * sigma + .5) + 1
    if halo > tile_cells:
        raise ValueError('Gaussian halo exceeds tile size')
    scaled = np.floor(points / voxel)
    if np.abs(scaled).max() > 2**50:
        raise ValueError('Particle coordinates exceed reliable voxel indexing range')
    ids = scaled.astype(np.int64)
    occupied = np.unique(ids, axis=0)
    tiles = set()
    for index in occupied:
        low = (index - halo - 1) // tile_cells
        high = (index + halo) // tile_cells
        tiles.update(itertools.product(*(range(int(a), int(b) + 1) for a, b in zip(low, high))))
    triangles = []
    total = 0
    peak = 0.0
    side = tile_cells + 1 + 2 * halo
    for tile in sorted(tiles):
        base = np.asarray(tile, dtype=np.int64) * tile_cells
        origin = base - halo
        local = ids - origin
        mask = ((local >= 0) & (local < side)).all(axis=1)
        density = np.zeros((side,) * 3, dtype=np.float32)
        np.add.at(density, tuple(local[mask].T), particle_volume / voxel**3)
        gaussian_filter(density, sigma, output=density, mode='constant', cval=0, truncate=3)
        core = density[halo:halo + tile_cells + 1, halo:halo + tile_cells + 1,
                       halo:halo + tile_cells + 1]
        peak = max(peak, float(core.max()))
        if not float(core.min()) < level < float(core.max()):
            continue
        vertices, faces, _, _ = marching_cubes(core, level=level, spacing=(voxel,) * 3)
        total += len(faces)
        if total > max_triangles:
            raise ValueError('Density mesh exceeds triangle budget; no particles were removed')
        vertices += (base + .5) * voxel
        triangles.append(vertices[faces])
    if not triangles:
        raise ValueError('No density isosurface at requested display threshold')
    mesh = np.concatenate(triangles)
    return mesh, {
        'vertices': total * 3, 'triangles': total, 'density_max': peak,
        'boundary_min_m': mesh.min(axis=(0, 1)).tolist(),
        'boundary_max_m': mesh.max(axis=(0, 1)).tolist(),
        'extraction': 'disjoint cell tiles with Gaussian halos; all particles included',
        'tile_cells': tile_cells, 'tiles_checked': len(tiles), 'max_grid_voxels': side**3,
        'vertices_note': 'Triangle vertices counted with repetitions, not welded vertices',
    }


def camera_zoom(value):
    """Validate a display-only zoom, leaving full-scene framing available at one."""
    import argparse
    import math
    try:
        zoom = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('Camera zoom must be a number in [0.25, 4]') from exc
    if not math.isfinite(zoom) or not .25 <= zoom <= 4:
        raise argparse.ArgumentTypeError('Camera zoom must be finite and in [0.25, 4]')
    return zoom


def executable(name, explicit=None):
    """Resolve an encoder without changing installation or filesystem permissions."""
    selected = str(explicit or os.environ.get(name.upper()) or shutil.which(name) or '')
    if not selected:
        raise RuntimeError(f'{name} is missing; specify --{name} with an installed executable')
    resolved = shutil.which(selected)
    if not resolved:
        raise RuntimeError(f'{name} executable not found: {selected}')
    # An existing non-Snap executable avoids Snap restrictions on mounted inputs.
    if not explicit and not os.environ.get(name.upper()) and '/snap/' in resolved:
        system = Path('/usr/bin') / name
        if system.is_file() and os.access(system, os.X_OK) and '/snap/' not in str(system.resolve()):
            resolved = str(system)
    return resolved


def video_tools(ffmpeg=None, ffprobe=None):
    return executable('ffmpeg', ffmpeg), executable('ffprobe', ffprobe)


def encode_video(concat, video, ffmpeg=None, ffprobe=None, minimum_frames=1):
    """Encode saved images only. Refuse overwrite and retain error logs on failure."""
    ffmpeg, ffprobe = video_tools(ffmpeg, ffprobe)
    concat, video = Path(concat).resolve(), Path(video).resolve()
    if not concat.is_file():
        raise FileNotFoundError(concat)
    if video.exists():
        raise FileExistsError(video)
    log = video.with_suffix('.ffmpeg.log')
    command = [ffmpeg, '-nostdin', '-n', '-f', 'concat', '-safe', '0', '-i', str(concat),
               '-vf', 'fps=30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19',
               '-movflags', '+faststart', str(video)]
    with log.open('x') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        detail = log.read_text(errors='replace')[-6000:]
        raise RuntimeError(f'FFmpeg exited {process.returncode}. Saved states and images remain intact. '
                           f'Log: {log}\n{detail}\n'
                           'If Snap denies /mnt access, select an already installed non-Snap '
                           'FFmpeg with --ffmpeg and --ffprobe. No permissions were changed.')
    probe_process = subprocess.run([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration:format=duration',
        '-of', 'json', str(video)], capture_output=True, text=True)
    if probe_process.returncode:
        raise RuntimeError(f'Video was encoded but ffprobe failed: {probe_process.stderr}\nVideo: {video}')
    probe = json.loads(probe_process.stdout)
    if not probe.get('streams') or int(probe['streams'][0]['nb_read_frames']) < minimum_frames:
        raise RuntimeError(f'Encoded video failed frame-count verification: {video}')
    return command, probe

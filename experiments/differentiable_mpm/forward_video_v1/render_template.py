"""Perspective movie of saved particles using a documented density isosurface."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont


def perspective_image(triangles, meshes, record, points, lower, upper, floor, status):
    plot = pv.Plotter(off_screen=True, window_size=(1200, 600))
    plot.set_background('#fcfcfb')
    def add(tris, color):
        flat = tris.reshape(-1, 3)
        faces = np.column_stack((np.full(len(tris), 3), np.arange(len(flat)).reshape(-1, 3)))
        mesh = pv.PolyData(flat, faces.ravel())
        plot.add_mesh(mesh, color=color, smooth_shading=False, ambient=.3, diffuse=.7,
                      specular=.15, specular_power=18, show_edges=False)
    add(triangles, COLORS[0])
    for vertices, pose, color, label in zip(meshes, record['tool_poses'], COLORS[1:], ('UR5e','Gen3')):
        transformed = vertices @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
        add(transformed, color)
        center = transformed.reshape(-1,3).mean(axis=0)
        center[1] = transformed[:,:,1].max()+.012
        plot.add_point_labels([center], [label], font_size=15, text_color='#242522',
                              show_points=False, shape=None, always_visible=True)
    table = np.array([[lower[0], floor, lower[2]], [upper[0], floor, lower[2]],
                      [upper[0], floor, upper[2]], [lower[0], floor, upper[2]]])
    plot.add_mesh(pv.PolyData(table, [4,0,1,2,3]), color='#e1e0d9', lighting=False)
    origin = np.array([lower[0]+.01, floor+.0002, lower[2]+.01])
    plot.add_mesh(pv.Line(origin, origin+[.05,0,0]), color='#242522', line_width=3)
    plot.add_point_labels([origin+[.025,0,.007]], ['50 mm'], font_size=14,
                          text_color='#242522', show_points=False, shape=None, always_visible=True)
    target = (lower+upper)/2
    direction = np.array([1.05,.85,-1.65]); direction /= np.linalg.norm(direction)
    right = np.cross([0,1,0], direction); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    corners = np.array([[x,y,z] for x in (lower[0],upper[0]) for y in (lower[1],upper[1]) for z in (lower[2],upper[2])]) - target
    tangent = np.tan(np.deg2rad(17))
    distance = 1.10 * max(np.max(np.abs(corners@up)/tangent + corners@direction),
                          np.max(np.abs(corners@right)/(2*tangent) + corners@direction))
    position = target + direction*distance
    plot.camera_position = [position, target, [0,1,0]]
    plot.camera.view_angle = 34
    plot.camera.clipping_range = (.001,10)
    plot.enable_anti_aliasing('ssaa')
    rgb = plot.screenshot(return_img=True)
    plot.close()
    image = Image.new('RGB', (1200,800), '#fcfcfb')
    image.paste(Image.fromarray(rgb), (0,100))
    draw = ImageDraw.Draw(image)
    regular = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    def text(x,y,value,size=17,strong=False):
        draw.text((x,y),value,font=ImageFont.truetype(bold if strong else regular,size),fill='#242522')
    text(32,18,'Episode 18 | requested-material forward simulation'+(' | PARTIAL RUN' if status!='completed' else ''),24,True)
    config = json.loads((SIM/'forward_demo_config.json').read_text())
    p = config['parameters']
    text(32,55,f"E = {p['youngs_modulus']/1000:g} kPa | viscosity = {p['viscosity']:g} Pa s | plastic stretch = {p['plastic_min']:g}–{p['plastic_max']:g} | {config['backend']} f32",18)
    frame = record['source_frame'] if record['source_frame'] is not None else 'between samples'
    text(32,80,f'Frame {frame}/387 | t = {record["sim_time_s"]:.4f} s | lowest particle = {(points[:,1].min()-floor)*1000:.3f} mm above table',16)
    for x,color,label in zip((32,300,645),COLORS,('Simulated dough','UR5e registered tool','Gen3 registered tool')):
        draw.rectangle((x,707,x+18,725),fill=color)
        text(x+28,705,label,18)
    text(32,740,'Density boundary: 1.5 mm voxels, Gaussian width 1.2 mm, level 0.20. No particle motion or floor filling added.',13)
    text(32,764,'Depth-tested opaque meshes | fixed perspective camera | Y is physical up | SDF padding is not drawn.',13)
    return image

ROOT = Path(__file__).resolve().parent
SIM = ROOT / 'snapshot/experiments/differentiable_mpm/runs/forward'
OUTPUT = ROOT / 'perspective'
VOXEL = 0.0015
SIGMA = 0.8
LEVEL = 0.20
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def stl(path):
    raw = path.read_bytes()
    n = int.from_bytes(raw[80:84], 'little')
    assert len(raw) == 84 + 50 * n
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    return np.frombuffer(raw, dtype=dtype, offset=84, count=n)['vertices'].astype(float)


def boundary(points, particle_volume):
    # Globally anchored voxel centers; zero outside finite particle support.
    origin = np.floor(points.min(axis=0) / VOXEL) * VOXEL - 4 * VOXEL
    size = np.ceil((points.max(axis=0) - origin) / VOXEL).astype(int) + 5
    if np.prod(size) > 30_000_000:
        raise ValueError('Particle extent exceeds documented rendering grid budget')
    ids = np.floor((points - origin) / VOXEL).astype(int)
    density = np.zeros(tuple(size), dtype=np.float32)
    np.add.at(density, tuple(ids.T), particle_volume / VOXEL ** 3)
    density = gaussian_filter(density, SIGMA, mode='constant', cval=0, truncate=3)
    if float(density.max()) <= LEVEL:
        raise ValueError('No density isosurface at requested display threshold')
    vertices, faces, _, _ = marching_cubes(density, level=LEVEL, spacing=(VOXEL,) * 3)
    vertices += origin + VOXEL / 2
    return vertices[faces], {'vertices': len(vertices), 'triangles': len(faces),
        'density_max': float(density.max()), 'boundary_min_m': vertices.min(axis=0).tolist(),
        'boundary_max_m': vertices.max(axis=0).tolist()}


def main():
    start = time.perf_counter()
    result = json.loads((SIM / 'simulation_result.json').read_text())
    prepared = json.loads((SIM / 'prepared_inputs.json').read_text())
    config = json.loads((SIM / 'forward_demo_config.json').read_text())
    collision = prepared['provenance']['collision']
    inputs = [SIM / 'simulation_result.json', SIM / 'prepared_inputs.json', SIM / 'forward_demo_config.json']
    for record in prepared['provenance']['input_files'].values():
        if 'sha256' in record:
            path = Path(record['path'])
            assert sha(path) == record['sha256'], 'Prepared input changed: ' + str(path)
            inputs.append(path)
    records = list(result['frames'])
    if result['completed_steps'] != records[-1]['step']:
        records.append({'source_frame': None, 'step': result['completed_steps'],
            'sim_time_s': result['sim_time_s'], 'particles': 'last_valid_particles.npy',
            'tool_poses': result['last_valid_tool_poses']})
    meshes = []
    for asset in collision['assets_in_stream_order']:
        path = Path(asset['path'])
        assert sha(path) == asset['sha256']
        inputs.append(path)
        vertices = stl(path) * collision['mesh_scale']
        vertices = vertices @ Rotation.from_euler('xyz', asset['visual_rpy_rad']).as_matrix().T
        vertices += np.asarray(asset['visual_origin_m'])
        meshes.append(vertices)
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    heights = []
    for record in records:
        path = SIM / record['particles']; inputs.append(path)
        points = np.load(path, allow_pickle=False)
        assert points.shape == (24000, 3) and np.isfinite(points).all()
        values = [points]
        for vertices, pose in zip(meshes, record['tool_poses']):
            values.append((vertices @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]).reshape(-1, 3))
        for values in values:
            lower = np.minimum(lower, values.min(axis=0))
            upper = np.maximum(upper, values.max(axis=0))
        heights.append({'step': record['step'], 'source_frame': record['source_frame'],
            'time_s': record['sim_time_s'], 'min_y_m': float(points[:, 1].min()),
            'max_y_m': float(points[:, 1].max())})
    before = {str(p): sha(p) for p in dict.fromkeys(inputs)}
    OUTPUT.mkdir()
    frames_dir = OUTPUT / 'frames'; frames_dir.mkdir()
    save_json(OUTPUT / 'input_hashes.json', before)
    save_json(OUTPUT / 'particle_heights.json', heights)
    lower -= .015; upper += .015; lower[1] = min(lower[1], -.005)
    floor = float(config['simulation']['floor_y'])
    particle_volume = float(prepared['mass']['particle_volume_m3'])
    indices = sorted(set([*range(0, len(records), 3), len(records) // 2, len(records) - 1]))
    frame_details = []
    for number, index in enumerate(indices):
        record = records[index]
        points = np.load(SIM / record['particles'], allow_pickle=False)
        triangles, info = boundary(points, particle_volume)
        image = perspective_image(triangles, meshes, record, points, lower, upper, floor, result['status'])
        image.save(frames_dir / f'frame_{number:05d}.png')
        still = {0:'01_initial_perspective.png', len(records)//2:'02_midpoint_perspective.png', len(records)-1:'03_final_perspective.png'}.get(index)
        if still:
            image.save(OUTPUT / still)
        frame_details.append({'movie_frame':number, 'source_frame':record['source_frame'], 'time_s':record['sim_time_s'], **info})
        if number % 10 == 0 or number == len(indices)-1:
            print(f'Rendered perspective {number+1}/{len(indices)}', flush=True)
    # Timestamp-based frame holds preserve the saved motion duration despite subsampling.
    times = [records[i]['sim_time_s'] for i in indices]
    tail = float(np.median(np.diff(times))) if len(times)>1 else .1
    concat = OUTPUT / 'frame_timing.txt'
    with concat.open('x') as f:
        for number, t in enumerate(times):
            duration = times[number+1]-t if number+1<len(times) else tail
            f.write(f"file 'frames/frame_{number:05d}.png'\nduration {duration:.9f}\n")
        f.write(f"file 'frames/frame_{len(times)-1:05d}.png'\n")
    video = OUTPUT / 'requested_material_perspective.mp4'
    command = ['ffmpeg','-nostdin','-n','-f','concat','-safe','0','-i',str(concat),
               '-vf','fps=30','-c:v','libx264','-pix_fmt','yuv420p','-crf','19','-movflags','+faststart',str(video)]
    with (OUTPUT/'ffmpeg.log').open('x') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration:format=duration','-of','json',str(video)],text=True))
    assert int(probe['streams'][0]['nb_read_frames']) >= len(indices)
    after = {p:sha(p) for p in before}
    assert before == after, 'Simulation inputs changed during rendering'
    snapshot_before = json.loads((ROOT/'snapshot_hashes_before.json').read_text())
    snapshot_after = {p:sha(ROOT/p) for p in snapshot_before}
    assert snapshot_before == snapshot_after, 'An extracted snapshot input changed'
    save_json(OUTPUT/'render_manifest.json', {'simulation_status':result['status'], 'completed_steps':result['completed_steps'],
        'last_simulation_time_s':result['sim_time_s'], 'last_saved_source_frame':result['last_saved_source_frame'],
        'sampled_snapshots':len(indices), 'render_engine':'VTK/PyVista opaque depth-tested offscreen rendering',
        'pyvista_version':pv.__version__, 'renderer_sha256':sha(Path(__file__)), 'forward_driver_sha256':sha(ROOT/'forward.py'),
        'timing':'Recorded timestamps determine frame holds, converted to 30 fps; final sample held for median interval.',
        'voxel_m':VOXEL,'gaussian_sigma_voxels':SIGMA,'density_level':LEVEL,'particle_volume_m3':particle_volume,
        'boundary_note':'Rendering-only density isosurface can extend roughly 1–3 mm beyond particle centers; it is not a new simulated state or closed observed volume.',
        'last_particle_height_m':heights[-1], 'parameters':config['parameters'],'simulation_settings':config['simulation'],
        'runtime':result['runtime'], 'raw_inputs_unchanged':before==after, 'snapshot_103_files_unchanged':snapshot_before==snapshot_after,
        'fixed_scene_bounds_m':{'min':lower.tolist(),'max':upper.tolist()},'frames':frame_details,
        'ffmpeg_command':command,'ffprobe':probe,'video_sha256':sha(video),
        'stills_sha256':{p.name:sha(p) for p in OUTPUT.glob('*.png')},'elapsed_s':time.perf_counter()-start})
    print('Finished perspective: '+str(video),flush=True)


if __name__ == '__main__':
    main()

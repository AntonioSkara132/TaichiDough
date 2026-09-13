"""Run a parameterized Episode18 forward replay and render its perspective movie."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import uuid

from render_support import camera_zoom

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ARCHIVE_SHA256 = '740658218a57a15f62ac2184c45d4a286ad8e31524b6687e48e6bf124f2621d0'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def execute(command, log_path):
    print('Running: ' + ' '.join(map(str, command)), flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with log_path.open('x') as log:
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--youngs-modulus', type=float, default=100000, help='Pa; default 100000')
    parser.add_argument('--viscosity', type=float, default=10, help='Pa s; default 10')
    parser.add_argument('--plastic-min', type=float, default=.09, help='Stretch ratio; default 0.09, not 0.9')
    parser.add_argument('--plastic-max', type=float, default=1.2)
    parser.add_argument('--poisson-ratio', type=float, default=.4047302679702308)
    parser.add_argument('--tool-friction', type=float, default=.3)
    parser.add_argument('--floor-retention', type=float, default=.4)
    parser.add_argument('--backend', choices=['vulkan','cuda','cpu'], default='vulkan')
    parser.add_argument('--camera-zoom', type=camera_zoom, default=1.0,
                        help='Display-only zoom; try 1.8 for a closer view (objects may leave view)')
    parser.add_argument('--episode', type=Path, default=EXPERIMENT.parents[2]/'data/deformpath_training/DeformPath3/snimanje_23_10/episode18_kugla', help='Local directory of the verified Episode18 recording')
    parser.add_argument('--bundle', type=Path, default=EXPERIMENT/'bundles/episode18_registered_tools_v1.tar.gz')
    parser.add_argument('--output-dir', type=Path, help='Fresh directory inside the experiment; default runs/episode18_forward_video_TIMESTAMP_ID')
    parser.add_argument('--simulation-python', default='/usr/bin/python3', help='Interpreter with Taichi and input dependencies')
    default_render = Path.home()/'miniconda3/envs/prancer/bin/python'
    parser.add_argument('--render-python', default=str(default_render) if default_render.is_file() else sys.executable, help='Interpreter with NumPy, SciPy, scikit-image, PyVista/VTK and Pillow')
    parser.add_argument('--prepare-only', action='store_true', help='Verify and prepare a fresh run, without simulation or rendering')
    args = parser.parse_args()
    requested = {'youngs_modulus':args.youngs_modulus, 'viscosity':args.viscosity,
        'plastic_min':args.plastic_min, 'plastic_max':args.plastic_max,
        'poisson_ratio':args.poisson_ratio, 'floor_retention':args.floor_retention}
    if not all(math.isfinite(v) for v in (*requested.values(), args.tool_friction)):
        parser.error('Parameters must be finite')
    if not args.episode.is_dir():
        parser.error('Episode directory does not exist; pass --episode with the local Episode18 directory')
    bundle = args.bundle.expanduser().resolve()
    if sha(bundle) != ARCHIVE_SHA256:
        raise ValueError('Bundle SHA256 differs from the verified original; no extraction or simulation performed')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    output = (args.output_dir or EXPERIMENT/'runs'/('episode18_forward_video_'+stamp+'_'+uuid.uuid4().hex[:6])).expanduser().resolve()
    if not output.is_relative_to(EXPERIMENT.resolve()):
        raise ValueError('Output must stay inside the experimental directory')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/'snapshot'; snapshot.mkdir()
    with tarfile.open(bundle) as archive:
        archive.extractall(snapshot, filter='data')
    isolated = snapshot/'experiments/differentiable_mpm'
    for line in (isolated/'data/episode18_registered_tools_v1/BUNDLE_SHA256SUMS').read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (snapshot/relative.lstrip('*')).resolve()
        if not path.is_relative_to(snapshot) or sha(path) != expected:
            raise ValueError('Bundled file checksum mismatch: '+relative)
    snapshot_hashes = {str(p.relative_to(output)):sha(p) for p in snapshot.rglob('*') if p.is_file()}
    save(output/'snapshot_hashes_before.json', snapshot_hashes)
    config = json.loads((isolated/'configs/episode18_table_aligned_registered_tools.json').read_text())
    config['paths']['episode'] = str(args.episode.expanduser().resolve())
    config['parameters'].update(requested)
    config['simulation']['tool_friction_coefficient'] = args.tool_friction
    config['backend'] = args.backend
    # These are optimizer intervals, not physical-validity limits. No fit is run.
    for name, value in requested.items():
        if name in config['parameter_bounds']:
            lo, hi = config['parameter_bounds'][name]
            config['parameter_bounds'][name] = [min(lo,value), max(hi,value)]
    save(output/'requested_config.json', config)
    shutil.copyfile(HERE/'forward_template.py', output/'forward.py')
    shutil.copyfile(HERE/'render_template.py', output/'render_perspective.py')
    shutil.copyfile(HERE/'render_support.py', output/'render_support.py')
    simulation = isolated/'runs/forward'
    forward = [args.simulation_python, output/'forward.py', '--config', output/'requested_config.json',
               '--output-dir', simulation, '--backend', args.backend]
    render = [args.render_python, output/'render_perspective.py', '--camera-zoom', str(args.camera_zoom)]
    save(output/'launcher_manifest.json', {'archive':str(bundle), 'archive_sha256':ARCHIVE_SHA256,
        'requested_parameters':requested,'tool_friction':args.tool_friction,'backend':args.backend,
        'simulation_command':list(map(str,forward)), 'render_command':list(map(str,render)),
        'templates_sha256':{p.name:sha(p) for p in (HERE/'forward_template.py',HERE/'render_template.py',HERE/'render_support.py')},
        'simulation_requested':not args.prepare_only, 'calibration_requested':False})
    print('Output: '+str(output), flush=True)
    if args.prepare_only:
        print('Prepared only. Commands are recorded in launcher_manifest.json; simulation was not run.', flush=True)
        return 0
    dependency_check = [args.render_python, '-c',
        'import sys; sys.path.insert(0, ' + repr(str(output)) + '); '
        'import numpy, scipy, skimage, pyvista; from PIL import Image; '
        'from render_support import video_tools; print("Render dependencies OK:", video_tools())']
    if execute(dependency_check, output/'render_dependencies.log'):
        raise RuntimeError('Rendering dependency check failed before simulation. '
                           'Choose --render-python with the required packages and installed FFmpeg/ffprobe. '
                           'See render_dependencies.log; no simulation was started.')
    code = execute(forward, output/'forward_stdout.log')
    result_path = simulation/'simulation_result.json'
    if not result_path.is_file():
        raise RuntimeError(f'Forward process exited {code} without a valid completion record; see forward_stdout.log')
    result = json.loads(result_path.read_text())
    if code != 0:
        print('Forward stopped; rendering only saved valid states. Failure: '+json.dumps(result.get('failure')), flush=True)
    render_code = execute(render, output/'render_stdout.log')
    if render_code:
        raise RuntimeError('Rendering failed; raw simulation states remain intact. See render_stdout.log')
    print('VIDEO: '+str(output/'perspective/requested_material_perspective.mp4'), flush=True)
    print('SIMULATION STATUS: '+result['status'], flush=True)
    return 0 if result['status']=='completed' and code==0 else 2


if __name__ == '__main__':
    raise SystemExit(main())

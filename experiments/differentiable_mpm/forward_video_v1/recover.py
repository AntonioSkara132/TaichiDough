"""Render an existing forward run, or encode existing images, without running physics."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import uuid

from render_support import camera_zoom, encode_video, video_tools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--encode-only', action='store_true', help='Use existing frame_timing.txt; no graphics dependencies needed')
    parser.add_argument('--camera-zoom', type=camera_zoom, default=1.0,
                        help='1 fits the complete motion; 1.8 gives a closer view, possibly cropping objects')
    parser.add_argument('--frames-dir', type=Path, help='Directory containing frame_timing.txt; default RUN/perspective')
    parser.add_argument('--output-dir', type=Path, help='Fresh directory inside RUN; default perspective_recovery_TIMESTAMP_ID')
    parser.add_argument('--ffmpeg', help='Installed encoder path; does not install or change permissions')
    parser.add_argument('--ffprobe', help='Installed probe path')
    args = parser.parse_args()
    if args.encode_only and args.camera_zoom != 1.0:
        parser.error('Camera zoom requires rendering saved states; it cannot change existing images')
    root = args.run_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error('Run directory does not exist')
    tools = video_tools(args.ffmpeg, args.ffprobe)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    output = (args.output_dir or root / f'perspective_recovery_{stamp}_{uuid.uuid4().hex[:6]}').resolve()
    if not output.is_relative_to(root) or output == root:
        parser.error('Output must be a fresh directory inside the run directory')
    if output.exists():
        parser.error('Output already exists; choose a fresh directory to preserve previous files')
    video = output / 'requested_material_perspective.mp4'
    if args.encode_only:
        frames = (args.frames_dir or root / 'perspective').expanduser().resolve()
        concat = frames / 'frame_timing.txt'
        if not concat.is_file():
            parser.error(f'Missing {concat}; use full recovery to render saved states')
        output.mkdir()
        command, probe = encode_video(concat, video, *tools)
        with (output / 'encoding_manifest.json').open('x') as stream:
            json.dump({'mode': 'encode-only', 'simulation_run': str(root),
                       'timing_file': str(concat), 'ffmpeg_command': command, 'ffprobe': probe,
                       'simulation_executed': False}, stream, indent=2)
    else:
        if args.frames_dir:
            parser.error('--frames-dir is only used with --encode-only')
        sim = root / 'snapshot/experiments/differentiable_mpm/runs/forward'
        if not (sim / 'simulation_result.json').is_file():
            parser.error('No saved simulation_result.json; this tool cannot run a simulation')
        template = Path(__file__).with_name('render_template.py')
        spec = importlib.util.spec_from_file_location('saved_forward_renderer', template)
        renderer = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(renderer)
        except ImportError as exc:
            raise RuntimeError('Rendering dependencies are missing or incompatible. Use a Python '
                               'environment with NumPy, SciPy, scikit-image, PyVista/VTK and Pillow. '
                               'The saved simulation is unchanged.') from exc
        renderer.ROOT, renderer.SIM, renderer.OUTPUT = root, sim, output
        renderer.FFMPEG, renderer.FFPROBE = tools
        renderer.CAMERA_ZOOM = args.camera_zoom
        renderer.main()
    print('VIDEO: ' + str(video), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

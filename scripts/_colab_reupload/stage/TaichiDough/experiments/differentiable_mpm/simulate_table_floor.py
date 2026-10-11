"""Forward-only recorded replay of the table-aligned floor, with streamed snapshots."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

from .config import EXPERIMENT_ROOT, file_sha256, load_config
from .data import prepare_experiment
from .evaluate import fresh_directory
from .reference_adapter import reference_identity, reference_policy
from .runtime import init_runtime
from .solver import Stepper
from .state import InvalidStateError


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT_ROOT / 'data/episode18_table_aligned_v1/episode18_table_aligned_coulomb.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--backend', choices=['vulkan', 'cpu', 'cuda'], default='vulkan')
    parser.add_argument('--cpu-threads', type=int, default=8)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    output = fresh_directory(args.output_dir or EXPERIMENT_ROOT / 'runs' / ('episode18_table_floor_simulation_' + stamp + '_' + uuid.uuid4().hex[:6]))
    print('Output: ' + str(output), flush=True)
    raw = json.loads(args.config.read_text())
    raw['name'] = 'episode18-table-floor-full-forward-demo'
    raw['backend'] = args.backend
    raw['simulation']['precision'] = 'f32'
    raw['simulation']['p2g_mode'] = 'atomic'
    raw['training'] = {'start_frame': 1, 'end_frame': 386, 'stride': 1}
    raw['validation'] = {'start_frame': 387, 'end_frame': 387, 'stride': 1}
    config_path = output / 'forward_demo_config.json'
    write_json(config_path, raw)
    config = load_config(config_path)
    source_paths = [EXPERIMENT_ROOT / name for name in ('simulate_table_floor.py', 'solver.py', 'spectral.py', 'state.py', 'runtime.py', 'data.py', 'replay.py', 'config.py')]
    hashes = {str(path): file_sha256(path) for path in source_paths}
    start = time.perf_counter()
    with reference_policy('frozen'):
        reference = reference_identity('corrected-v1')
        runtime = init_runtime(args.backend, 'f32', cpu_threads=args.cpu_threads, seed=config.seed)
        print('Runtime: ' + json.dumps(runtime), flush=True)
        prepared = prepare_experiment(config, split='validation', build_sdf=True)
        write_json(output / 'prepared_inputs.json', prepared.summary())
        frames = {f.completed_substeps: f for f in prepared.frames}
        snapshots = output / 'snapshots'
        snapshots.mkdir()
        records = []
        stepper = Stepper(prepared.simulation_config, prepared.parameters, capacity=65, sdf=prepared.sdf)
        stepper.load_state(0, prepared.initial_state)
        np.save(snapshots / 'particles_000000.npy', prepared.initial_state.x)
        records.append({'source_frame': 0, 'original_source_frame': prepared.frames[0].original_source_frame,
                        'step': 0, 'sim_time_s': 0.0, 'particles': 'snapshots/particles_000000.npy',
                        'tool_poses': prepared.controls.at_completed_step(0).poses.tolist()})
        manifest = {'schema': 'taichidough/table-floor-forward/v1', 'requested_backend': args.backend,
                    'runtime': runtime, 'parameters': prepared.parameters, 'reference': reference,
                    'config_sha256': file_sha256(config_path), 'source_sha256': hashes,
                    'prepared_fingerprint': prepared.fingerprint, 'target_steps': prepared.total_steps,
                    'target_source_frame': prepared.end_frame, 'target_time_s': prepared.frames[-1].target_time_s,
                    'simulation_run': True, 'calibration_run': False, 'backward_run': False,
                    'limitations': ['Material parameters are inherited guesses, not fitted to this geometry.',
                                   'Original tool geometry is used; registration corrections are not approved.',
                                   'Density is2196.218989kg/m3 from recorded250g mass and113.832mL volume.']}
        write_json(output / 'run_manifest.json', manifest)
        completed, slot = 0, 0
        failure = None
        simulation_start = time.perf_counter()
        with (output / 'progress.jsonl').open('x') as progress:
            try:
                for step in range(prepared.total_steps):
                    stepper.advance(slot, prepared.controls[step])
                    completed = step + 1
                    slot += 1
                    if completed in frames:
                        frame = frames[completed]
                        state = stepper.state(slot)
                        path = snapshots / f'particles_{frame.source_frame:06d}.npy'
                        np.save(path, state.x)
                        records.append({'source_frame': frame.source_frame, 'original_source_frame': frame.original_source_frame,
                                        'step': completed, 'sim_time_s': frame.sim_time_s,
                                        'particles': str(path.relative_to(output)),
                                        'tool_poses': prepared.controls.at_completed_step(completed).poses.tolist(),
                                        'bounds_min': state.x.min(axis=0).tolist(), 'bounds_max': state.x.max(axis=0).tolist()})
                    if completed == 1 or completed % 1000 == 0 or completed == prepared.total_steps:
                        elapsed = time.perf_counter() - simulation_start
                        event = {'step': completed, 'sim_time_s': completed * prepared.simulation_config.dt,
                                 'target_steps': prepared.total_steps, 'elapsed_s': elapsed,
                                 'steps_per_s': completed / elapsed, 'diagnostics': stepper.diagnostics()}
                        progress.write(json.dumps(event, allow_nan=False) + '\n')
                        progress.flush()
                        print(json.dumps(event), flush=True)
                    if slot == stepper.capacity - 1 and completed < prepared.total_steps:
                        stepper.load_state(0, stepper.state(slot))
                        slot = 0
            except InvalidStateError as exc:
                failure = {'type': type(exc).__name__, 'message': str(exc), 'attempted_step': completed + 1,
                           'diagnostics': stepper.diagnostics()}
                print('SIMULATION STOPPED: ' + json.dumps(failure), flush=True)
            except Exception as exc:
                failure = {'type': type(exc).__name__, 'message': str(exc), 'attempted_step': completed + 1,
                           'traceback': traceback.format_exc()}
                print('SIMULATION ERROR: ' + json.dumps(failure), flush=True)
        last_state = stepper.state(slot)
        last_state.validate()
        np.savez_compressed(output / 'last_valid_state.npz', **last_state.arrays())
        np.save(output / 'last_valid_particles.npy', last_state.x)
        source_after = {str(path): file_sha256(path) for path in source_paths}
        result = {'schema': manifest['schema'], 'status': 'completed' if failure is None and completed == prepared.total_steps else 'stopped',
                  'completed_steps': completed, 'sim_time_s': completed * prepared.simulation_config.dt,
                  'target_steps': prepared.total_steps, 'last_saved_source_frame': records[-1]['source_frame'],
                  'elapsed_s': time.perf_counter() - start, 'failure': failure,
                  'source_unchanged': source_after == hashes, 'source_sha256_after': source_after,
                  'frames': records, 'runtime': {**runtime, 'forward_verified': completed > 0},
                  'last_valid_bounds': {'min': last_state.x.min(axis=0).tolist(), 'max': last_state.x.max(axis=0).tolist()},
                  'last_valid_tool_poses': prepared.controls.at_completed_step(completed).poses.tolist()}
        write_json(output / 'simulation_result.json', result)
        print('RESULT: ' + json.dumps({k: result[k] for k in ('status', 'completed_steps', 'sim_time_s', 'last_saved_source_frame', 'failure', 'source_unchanged')}), flush=True)
        print('Output: ' + str(output), flush=True)
        return 0 if result['status'] == 'completed' and result['source_unchanged'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Forward-only floor settling with verified reconstruction and explicitly unused tools."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

from .config import EXPERIMENT_ROOT, file_sha256, load_config
from .data import validate_initial_stencils
from .evaluate import fresh_directory
from .reference_adapter import get_reference_modules, reference_identity, reference_policy
from .runtime import init_runtime
from .solver import Stepper
from .state import ParticleState, SimulationConfig, ToolControl


def write_json(path, document):
    with Path(path).open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT_ROOT / 'data/episode18_table_aligned_v1/episode18_table_aligned_coulomb.json')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = fresh_directory(args.output_dir or EXPERIMENT_ROOT / 'runs' /
                             ('floor_settling_no_tools_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]))
    config = load_config(args.config)
    used = ('calibration', 'initial_particles', 'reconstruction_metadata')
    inputs = {}
    for name in used:
        actual = file_sha256(config.paths[name])
        expected = config.expected_sha256[name]
        if actual != expected:
            raise ValueError(f'Used input hash mismatch: {name}')
        inputs[name] = {'path': str(config.paths[name]), 'sha256': actual}
    source_paths = [EXPERIMENT_ROOT / name for name in
                    ('simulate_floor_settling.py', 'solver.py', 'state.py', 'spectral.py',
                     'runtime.py', 'config.py', 'data.py', 'reference_adapter.py')]
    source_hashes = {str(path): file_sha256(path) for path in source_paths}
    snapshots = output / 'snapshots'
    snapshots.mkdir()
    with reference_policy('frozen'):
        reference = reference_identity('corrected-v1')
        helpers = get_reference_modules()
        calibration = helpers.topview.load_calibration(config.paths['calibration'])
        particles = np.load(config.paths['initial_particles'], allow_pickle=False)
        metadata = helpers.simulator.load_reconstruction_metadata(
            config.paths['reconstruction_metadata'], config.paths['initial_particles'], particles, calibration)
        if len(particles) != config.simulation['n_particles']:
            raise ValueError('Particle count mismatch')
        if not np.isclose(metadata['floor_y'], config.simulation['floor_y'], atol=1e-8, rtol=0):
            raise ValueError('Reconstruction and configured floor differ')
        mass = helpers.simulator.compute_mass_properties(len(particles), config.simulation['grid'],
            config.density_kg_m3, metadata['object_volume_m3'], config.mass_kg)
        if not np.isclose(mass['density_kg_m3'], config.density_kg_m3, rtol=1e-8, atol=0):
            raise ValueError('Density differs from measured mass / reconstructed volume')
        numerical = dict(config.simulation)
        numerical.update(tool_collision='none', physics_version='corrected-v1', precision='f32',
                         p2g_mode='atomic', particle_mass=mass['particle_mass_kg'],
                         particle_volume=mass['particle_volume_m3'])
        sim = SimulationConfig(**numerical)
        validate_initial_stencils(particles, sim)
        state = ParticleState.initial(particles, sim.numpy_dtype)
        state.validate()
        runtime = init_runtime('vulkan', 'f32', cpu_threads=8, seed=config.seed)
        print('Runtime: ' + json.dumps(runtime), flush=True)
        manifest = {'schema': 'taichidough/floor-settling/v1', 'source_config': str(args.config.resolve()),
                    'source_config_sha256': file_sha256(args.config), 'inputs': inputs,
                    'unused_inputs': {name: str(path) for name, path in config.paths.items() if name not in used},
                    'unused_input_note': 'Tool assets, geometry, collision manifest and recorded sequence are not loaded or verified because tools are disabled.',
                    'simulation': asdict(sim), 'parameters': config.parameters, 'mass': mass,
                    'runtime': runtime, 'reference': reference, 'source_sha256': source_hashes,
                    'target_steps': 10000, 'target_time_s': 2.0, 'snapshot_interval_steps': 100,
                    'backward_run': False, 'calibration_run': False, 'tool_collision': 'none'}
        write_json(output / 'run_manifest.json', manifest)
        stepper = Stepper(sim, config.parameters, capacity=65)
        stepper.load_state(0, state)
        control = ToolControl.stationary()
        records = []
        started = time.perf_counter()
        failure = None
        completed = 0
        slot = 0
        with (output / 'progress.jsonl').open('x') as stream:
            def save(step, current, diagnostics):
                current.validate()
                y, vy = current.x[:, 1], current.v[:, 1]
                path = snapshots / f'particles_{step:06d}.npy'
                np.save(path, current.x)
                row = {'step': step, 'time_s': step * sim.dt,
                       'min_y_m': float(y.min()), 'median_y_m': float(np.median(y)),
                       'max_y_m': float(y.max()), 'com_y_m': float(y.astype(np.float64).mean()),
                       'min_vy_m_s': float(vy.min()), 'mean_vy_m_s': float(vy.astype(np.float64).mean()),
                       'max_vy_m_s': float(vy.max()),
                       'within_1mm_count': int(np.count_nonzero(y - sim.floor_y <= .001)),
                       'grid_floor': diagnostics.get('grid_floor', 0),
                       'particle_floor': diagnostics.get('particle_floor', 0),
                       'yielded': diagnostics.get('yielded', 0),
                       'diagnostics_are_last_step_not_interval_sum': True,
                       'particles': str(path.relative_to(output)),
                       'elapsed_s': time.perf_counter() - started}
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                if step % 1000 == 0:
                    print(json.dumps(row), flush=True)
            save(0, state, {})
            try:
                for step in range(10000):
                    stepper.advance(slot, control)
                    slot += 1
                    completed = step + 1
                    if completed % 100 == 0:
                        state = stepper.state(slot)
                        save(completed, state, stepper.diagnostics())
                    if slot == stepper.capacity - 1 and completed < 10000:
                        stepper.load_state(0, stepper.state(slot))
                        slot = 0
            except Exception as exc:
                failure = {'type': type(exc).__name__, 'message': str(exc),
                           'attempted_step': completed + 1, 'traceback': traceback.format_exc(),
                           'diagnostics': stepper.diagnostics()}
            state = stepper.state(slot)
            state.validate()
            if records[-1]['step'] != completed:
                save(completed, state, stepper.diagnostics())
        np.savez_compressed(output / 'last_valid_state.npz', **state.arrays())
        with (output / 'settling_metrics.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        unchanged_sources = source_hashes == {str(path): file_sha256(path) for path in source_paths}
        unchanged_inputs = all(record['sha256'] == file_sha256(Path(record['path'])) for record in inputs.values())
        onset = {str(mm) + '_mm': next((row['time_s'] for row in records if row['min_y_m'] > mm / 1000), None)
                 for mm in (1, 5, 10, 15)}
        result = {'schema': manifest['schema'], 'status': 'completed' if completed == 10000 and failure is None else 'stopped',
                  'completed_steps': completed, 'time_s': completed * sim.dt, 'elapsed_s': time.perf_counter() - started,
                  'failure': failure, 'initial': records[0], 'final': records[-1],
                  'first_sample_all_particles_above': onset, 'source_unchanged': unchanged_sources,
                  'used_inputs_unchanged': unchanged_inputs, 'runtime': {**runtime, 'forward_verified': completed > 0},
                  'frames': records, 'limits': ['One grid48 settling run; no numerical floor fix or convergence study.',
                                              'No tool geometry or trajectories used.',
                                              'Rendered particles are not repositioned or filled again.']}
        write_json(output / 'simulation_result.json', result)
        print('RESULT: ' + json.dumps({k: result[k] for k in ('status', 'completed_steps', 'time_s', 'first_sample_all_particles_above', 'source_unchanged', 'used_inputs_unchanged')}), flush=True)
        print('Output: ' + str(output), flush=True)
        return 0 if result['status'] == 'completed' and unchanged_sources and unchanged_inputs else 2


if __name__ == '__main__':
    raise SystemExit(main())

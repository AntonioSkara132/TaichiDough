"""CLI for verified replay, reverse-mode gradients and joint material calibration."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np

from .checkpoint import CheckpointedRollout, estimate_memory
from .config import load_config, verify_input_paths
from .data import prepare_experiment
from .loss import ObservationLoss
from .optimize import AdamOptions, ObjectiveValue, ProjectedAdam
from .parameters import PhysicalParameterSpace
from .reference_adapter import reference_identity, reference_policy, verify_reference
from .results import EXPERIMENT_ROOT, RUN_ROOT, RunStore, json_value, source_identity
from .runtime import init_runtime
from .solver import Stepper
from .state import PARAMETER_NAMES, P2G_MODES, PHYSICS_VERSIONS, InvalidStateError, validate_parameters


DEFAULT_CONFIG = EXPERIMENT_ROOT / 'configs' / 'episode18_viscoelastic.json'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'gradient', 'fit', 'replay', 'evaluate'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--path', action='append', default=[], metavar='NAME=PATH',
                        help='Explicit input path override; content hashes are still checked')
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'vulkan'])
    parser.add_argument('--precision', choices=['f32', 'f64'])
    parser.add_argument('--p2g-mode', choices=P2G_MODES,
                        help='Override particle-to-grid transfer: atomic (default) or fixed-order serial; serial can be slower')
    parser.add_argument('--physics-version', choices=PHYSICS_VERSIONS,
                        help='Override solver physics: corrected-v1 (default) or legacy-v1 for old-run reproduction')
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reference-policy', choices=['strict', 'frozen'], default='strict',
                        help='strict checks live originals; frozen explicitly targets verified snapshots and records live changes')
    parser.add_argument('--segment-length', type=int)
    parser.add_argument('--ignore-recompute-mismatch', action='store_true',
                        help='Continue through finite replay mismatches with warnings; gradients may be approximate')
    parser.add_argument('--end-frame', type=int, help='Shorter endpoint inside the configured scoring window')
    parser.add_argument('--split', choices=['training', 'validation'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--parameters', type=Path, help='Frozen best_parameters result, or complete physical parameter JSON')
    parser.add_argument('--iterations', type=int, default=20, help='Additional optimizer attempts, not objective calls')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--learning-rate-policy', choices=['persistent-v1', 'recover-v1'],
                        help='Optional bounded recovery of proposal rates; accepted trials still require backtracking checks')
    parser.add_argument('--learning-rate-growth', type=float,
                        help='Proposal-rate growth factor for recover-v1; capped at the initial learning rate')
    parser.add_argument('--max-evaluations', type=int)
    parser.add_argument('--resume', action='store_true', help='Resume an exactly matching fit run')
    parser.add_argument('--no-evaluate', action='store_true', help='Skip final independent training/held-out evaluation')
    parser.add_argument('--no-runtime', action='store_true', help='Validate inputs only; no backend qualification')
    parser.add_argument('--finite-difference', action='store_true', help='Check gradient against coordinate perturbations')
    args = parser.parse_args(argv)
    if args.iterations < 0:
        parser.error('--iterations must be nonnegative')
    if args.resume and (args.action != 'fit' or args.output_dir is None):
        parser.error('--resume requires fit and an explicit --output-dir')
    if args.no_runtime and args.action != 'validate':
        parser.error('--no-runtime is only meaningful for validate')
    if args.finite_difference and args.action != 'gradient':
        parser.error('--finite-difference is only meaningful for gradient')
    if args.parameters is not None and args.action == 'fit':
        parser.error('Set the initial fit parameters in the experiment config, not --parameters')
    if args.action == 'evaluate' and args.parameters is None:
        parser.error('evaluate requires frozen --parameters selected using training data')
    if args.action in {'fit', 'gradient'} and args.split == 'validation':
        parser.error('Calibration and gradient checks use training data only')
    return args


def read_parameters(path):
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    if set(record) == set(PARAMETER_NAMES):
        values = record
    elif 'best_parameters' in record:
        values = record['best_parameters']
    elif isinstance(record.get('optimization'), dict) and 'best_parameters' in record['optimization']:
        values = record['optimization']['best_parameters']
    else:
        raise ValueError('Parameter file needs complete physical values or a best_parameters record')
    values = {k: float(v) for k, v in values.items()}
    validate_parameters(values)
    return values, {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}


def evaluation_record(result):
    record = {'value': result.value, 'gradient': result.gradient, 'frames': result.frames,
              'diagnostics': result.diagnostics}
    if result.initial_gradient is not None:
        record['initial_state_gradient_norms'] = {
            name: float(np.linalg.norm(np.asarray(array, dtype=np.float64)))
            for name, array in result.initial_gradient.arrays().items()}
    return record


def progress_printer(event):
    phase, step, total = event['phase'], event['step'], event['total_steps']
    if phase == 'observation':
        frame = event['frame_index']
        if frame <= 3 or frame % 10 == 0 or step == total:
            print(f'frame={frame} step={step}/{total} loss={event["value"]:.8g}', flush=True)
    elif phase == 'backward_start':
        print(f'Forward complete; starting backward replay over {total} steps (first call compiles adjoints)', flush=True)
    elif phase == 'backward_segment' and (step == 0 or step % 1024 == 0):
        print(f'backward step={step}/{total}', flush=True)
    elif phase == 'recompute_mismatch':
        ignored = event.get('ignored', False)
        number = event.get('mismatch_number', 1)
        if ignored and number > 3 and number % 100 != 0:
            return
        label = 'WARNING: ignoring replay mismatch' if ignored else 'Replay mismatch'
        segment = (f', segment={event["segment_start_step"]}–{event["segment_end_step"]}'
                   if 'segment_start_step' in event else '')
        detail = (json.dumps(json_value(event['differing_counts']), sort_keys=True)
                  if 'differing_counts' in event else event.get('message', event.get('mismatch_kind', '')))
        print(f'{label} at step={step}/{total}{segment}: {detail}', flush=True)


def stored_progress(store):
    def callback(event):
        progress_printer(event)
        if event['phase'] == 'recompute_mismatch':
            filename = 'last_recompute_warning.json' if event.get('ignored', False) else 'last_recompute_failure.json'
            store.write_json(filename, event)
        if (event['phase'] in {'observation', 'invalid_adjoint', 'recompute_mismatch'}
                or (event['phase'] == 'backward_segment' and event['step'] % 1024 == 0)
                or event['step'] == event['total_steps']):
            store.append_event({'event': 'replay_progress', **event})
    return callback


def make_rollout(prepared, stepper=None, progress=progress_printer, *, ignore_recompute_mismatch=False):
    length = min(prepared.config.segment_length, max(1, prepared.total_steps))
    if stepper is None:
        stepper = Stepper(prepared.simulation_config, prepared.parameters, capacity=length + 1, sdf=prepared.sdf)
    elif asdict(stepper.config) != asdict(prepared.simulation_config) or stepper.capacity < length + 1:
        raise ValueError('Existing stepper does not match the fixed replay configuration')
    loss = ObservationLoss(prepared.camera, prepared.loss_config, prepared.initial_state.x,
                           precision=prepared.simulation_config.precision)
    rollout = CheckpointedRollout(stepper, prepared.initial_state, prepared.controls,
                                  prepared.observations, loss, length, progress=progress,
                                  ignore_recompute_mismatch=ignore_recompute_mismatch)
    return stepper, rollout


def finite_difference_report(rollout, space, parameters, physical_gradient):
    u = space.coordinates()
    # The initial physical values belong to this space; all checks perturb its own coordinates.
    if any(not np.isclose(space.physical(u)[k], parameters[k], rtol=1e-12, atol=1e-12) for k in parameters):
        raise ValueError('Finite-difference space does not describe the evaluated parameters')
    ad = space.pullback(u, physical_gradient)
    names = list(space.fit)
    rows = []
    for index, name in enumerate(names):
        checks = []
        for h in (1e-3, 3e-4, 1e-4):
            plus, minus = u.copy(), u.copy()
            plus[index] += h
            minus[index] -= h
            plus, minus = space.project(plus), space.project(minus)
            denominator = plus[index] - minus[index]
            if denominator <= 0:
                raise ValueError(f'No finite-difference interval for {name}')
            a = rollout.value_and_gradient(space.physical(plus), compute_grad=False).value
            b = rollout.value_and_gradient(space.physical(minus), compute_grad=False).value
            fd = (a - b) / denominator
            error = abs(fd - ad[index]) / max(abs(fd), abs(ad[index]), 1e-8)
            checks.append({'step': h, 'coordinate_span': float(denominator), 'finite_difference': fd,
                           'relative_error': float(error),
                           'scheme': 'central' if np.isclose(plus[index] + minus[index], 2 * u[index], rtol=0, atol=1e-14) else 'one-sided at bound'})
        rows.append({'parameter': name, 'coordinate_gradient': float(ad[index]), 'checks': checks,
                     'passed': any(c['relative_error'] < 1e-2 for c in checks)})
    return {'rows': rows, 'passed': all(r['passed'] for r in rows),
            'note': 'Piecewise contact/visibility can switch under perturbation; inspect all step sizes.'}


def export_and_evaluate(prepared, stepper, parameters, store, runtime, *, strict=True):
    from .evaluate import export_replay, run_strict_evaluation
    _, rollout = make_rollout(prepared, stepper, progress=stored_progress(store),
                              ignore_recompute_mismatch=runtime.get('ignore_recompute_mismatch', False))
    positions = {}
    frame_steps = {f.completed_substeps for f in prepared.frames}
    def collect(step, state):
        positions[step] = state.x.copy()
    evaluation = rollout.value_and_gradient(parameters, compute_grad=False,
                                           frame_callback=collect, frame_steps=frame_steps)
    if set(positions) != frame_steps:
        raise InvalidStateError('Replay export is missing an expected frame state')
    suffix = prepared.split + '_' + uuid.uuid4().hex[:8]
    metadata = export_replay(prepared, positions, store.path / (suffix + '_simulation'),
                             parameters=parameters, runtime_metadata=runtime)
    result = {'split': prepared.split, 'simulation_metadata': str(metadata),
              'differentiable_objective': evaluation_record(evaluation), 'strict_evaluation': None}
    if strict:
        result['strict_evaluation'] = run_strict_evaluation(prepared, metadata, store.path / (suffix + '_strict'))
    store.write_json(prepared.split + '_evaluation.json', result)
    return result


def run(args):
    overrides = {}
    for item in args.path:
        if '=' not in item:
            raise ValueError('--path requires NAME=PATH')
        name, value = item.split('=', 1)
        if name in overrides:
            raise ValueError(f'Duplicate path override: {name}')
        overrides[name] = value
    config = load_config(args.config, overrides)
    if args.backend is not None:
        config.backend = args.backend
    if args.precision is not None:
        config.simulation['precision'] = args.precision
    if args.p2g_mode is not None:
        config.simulation['p2g_mode'] = args.p2g_mode
    config.simulation['physics_version'] = args.physics_version or config.simulation.get('physics_version', 'corrected-v1')
    if args.segment_length is not None:
        config.segment_length = args.segment_length
    frozen_source = None
    if args.parameters is not None:
        config.parameters, frozen_source = read_parameters(args.parameters)
    config.validate()
    reference_verification = verify_reference()
    physics_reference = reference_identity(config.simulation['physics_version'])
    if reference_verification.get('originals_unchanged') is False:
        print('WARNING: working helper sources differ; using verified frozen helpers. Solver physics is selected separately. '
              + json.dumps(reference_verification.get('changed_originals', [])), flush=True)
    verify_input_paths(config)
    runtime = ({'initialization_verified': False, 'backend': config.backend}
               if args.no_runtime else init_runtime(config.backend, config.simulation.get('precision', 'f32'),
                                                    args.cpu_threads, args.debug, config.seed))
    runtime = {**runtime, 'ignore_recompute_mismatch': args.ignore_recompute_mismatch,
               'physics_version': config.simulation['physics_version'], 'physics_reference': physics_reference}
    if args.ignore_recompute_mismatch:
        print('WARNING: finite replay mismatches will be logged and ignored; gradients may be approximate. '
              'Invalid states and nonfinite losses/gradients still reject an evaluation.', flush=True)
    split = args.split or ('validation' if args.action == 'evaluate' else 'training')
    prepared = prepare_experiment(config, split=split, end_frame=args.end_frame, build_sdf=True)
    memory = estimate_memory(prepared.simulation_config.n_particles, prepared.total_steps, config.segment_length,
                             prepared.simulation_config.precision, prepared.simulation_config.grid, config.tool_sdf_resolution,
                             physics_version=prepared.simulation_config.physics_version)
    print(f'Prepared {prepared.simulation_config.n_particles} particles; replay0–{prepared.end_frame}; '
          f'{prepared.total_steps} steps; {len(prepared.observations)} observations; backend={config.backend}', flush=True)
    print('Memory estimate: ' + json.dumps(memory), flush=True)
    print(f'Physics version: {prepared.simulation_config.physics_version}; reference={physics_reference["simulator_sha256"]}', flush=True)
    print(f'P2G mode: {prepared.simulation_config.p2g_mode} '
          '(transfer mode only; other reductions are unchanged)', flush=True)
    options_dict = dict(config.optimizer)
    if args.learning_rate is not None:
        options_dict['learning_rate'] = args.learning_rate
    if args.learning_rate_policy is not None:
        options_dict['learning_rate_policy'] = args.learning_rate_policy
    if args.learning_rate_growth is not None:
        options_dict['learning_rate_growth'] = args.learning_rate_growth
    if args.max_evaluations is not None:
        options_dict['max_evaluations'] = args.max_evaluations
    options = AdamOptions(**options_dict)
    space = PhysicalParameterSpace(prepared.parameters, config.fit_parameters,
                                   prepared.simulation_config.plasticity, bounds=config.parameter_bounds)
    identity = {'schema': 'differentiable-calibration-identity-v1', 'action': args.action,
                'prepared': prepared.provenance, 'source': source_identity(), 'runtime': runtime,
                'parameter_space': space.settings(), 'optimizer': asdict(options),
                'frozen_parameter_source': frozen_source, 'finite_difference': args.finite_difference,
                'reference_policy': args.reference_policy, 'reference_verification': reference_verification}
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        output = RUN_ROOT / (config.name + '_' + args.action + '_' + stamp + '_' + uuid.uuid4().hex[:6])
    else:
        output = args.output_dir
    with RunStore(output, identity, resume=args.resume) as store:
        store.write_json('resolved_inputs.json', prepared.summary())
        store.write_json('memory_estimate.json', memory)
        if args.action == 'validate':
            result = {'status': 'inputs_valid', 'runtime': runtime, 'summary': prepared.summary(), 'memory_estimate': memory}
            store.write_json('result.json', result)
            print(f'Input validation written to {store.path / "result.json"}', flush=True)
            return 0
        stepper, rollout = make_rollout(prepared, progress=stored_progress(store),
                                        ignore_recompute_mismatch=args.ignore_recompute_mismatch)
        if args.action == 'gradient':
            try:
                evaluation = rollout.value_and_gradient(prepared.parameters)
            except InvalidStateError as error:
                store.write_json('result.json', {'status': 'gradient_failed', 'error': str(error),
                                                 'parameters': prepared.parameters, 'runtime': runtime})
                store.append_event({'event': 'gradient_failed', 'error': str(error)})
                raise
            result = {'status': 'gradient_computed', 'parameters': prepared.parameters,
                      'runtime': {**runtime, 'forward_verified': True, 'backward_verified': True,
                                  'backward_replay_consistent': evaluation.diagnostics.get('replay_consistent', True)},
                      'evaluation': evaluation_record(evaluation),
                      'coordinate_gradient': space.pullback(space.coordinates(), evaluation.gradient).tolist()}
            if args.finite_difference:
                result['finite_difference'] = finite_difference_report(rollout, space, prepared.parameters, evaluation.gradient)
                if not result['finite_difference']['passed']:
                    result['status'] = 'gradient_check_failed'
            store.write_json('result.json', result)
            print(f'Loss={evaluation.value:.9g}; gradients={evaluation.gradient}', flush=True)
            print(f'Gradient result: {store.path / "result.json"}', flush=True)
            return 2 if result['status'] == 'gradient_check_failed' else 0
        if args.action in {'replay', 'evaluate'}:
            result = export_and_evaluate(prepared, stepper, prepared.parameters, store, runtime,
                                         strict=args.action == 'evaluate')
            valid = args.action == 'replay' or result.get('strict_evaluation', {}).get('valid') is True
            status = 'replay_complete' if args.action == 'replay' else ('evaluation_complete' if valid else 'evaluation_failed')
            store.write_json('result.json', {'status': status, **result,
                                             'runtime': {**runtime, 'forward_verified': True}})
            print(f'Replay result: {store.path / "result.json"}', flush=True)
            return 0 if valid else 2

        count = [0]
        successful_objectives = [0]
        objectives_with_mismatches = [0]
        def objective(parameters):
            count[0] += 1
            store.append_event({'event': 'objective_started', 'call_in_process': count[0], 'parameters': parameters})
            print(f'Objective call {count[0]}: {parameters}', flush=True)
            evaluation = rollout.value_and_gradient(parameters)
            successful_objectives[0] += 1
            if not evaluation.diagnostics.get('replay_consistent', True):
                objectives_with_mismatches[0] += 1
                print('WARNING: objective gradient used mismatched replay; counts='
                      + json.dumps(evaluation.diagnostics.get('recompute_mismatch_counts', {})), flush=True)
            record = evaluation_record(evaluation)
            store.write_json('last_objective.json', {'parameters': parameters, **record})
            print(f'Objective loss={evaluation.value:.9g}; physical_gradient={evaluation.gradient}', flush=True)
            return ObjectiveValue(evaluation.value, evaluation.gradient, evaluation.diagnostics)
        def callback(event, state):
            store.save_optimizer(event, state)
            print('Optimizer: ' + json.dumps(json_value(event), sort_keys=True), flush=True)
        optimizer = ProjectedAdam(space, objective, options, callback, objective_id=store.identity_hash)
        if args.resume:
            optimizer.load_state_dict(store.optimizer_state())
        try:
            optimization = optimizer.run(args.iterations)
        except InvalidStateError as error:
            failure = {
                'status': optimizer.status if optimizer.status == 'invalid_initial' else 'fit_failed',
                'error': str(error), 'optimizer_evaluations': optimizer.evaluations,
                'accepted_updates': optimizer.accepted_updates,
                'runtime': {**runtime, 'backward_verified': successful_objectives[0] > 0,
                            'successful_objectives_in_process': successful_objectives[0],
                            'objectives_with_replay_mismatch': objectives_with_mismatches[0],
                            'execution_scope': 'this process; rejected evaluations are not verified gradients'},
            }
            store.write_json('result.json', failure)
            store.append_event({'event': 'fit_failed', **failure})
            raise
        selected = {'best_parameters': optimization.best_parameters, 'training_value': optimization.best_value,
                    'selection_frames': list(prepared.scored_frames), 'selection_split': 'training',
                    'identity_sha256': store.identity_hash,
                    'ignore_recompute_mismatch': args.ignore_recompute_mismatch,
                    'physics_version': prepared.simulation_config.physics_version,
                    'physics_reference': physics_reference,
                    'simulator_reference_sha256': physics_reference['simulator_sha256'],
                    'baseline_note': f'These parameters target {prepared.simulation_config.physics_version}; '
                                     'parameters fitted with another physics version require a new fit, not a constant rescaling.'}
        store.write_json('selected_parameters.json', selected)
        result = {'status': optimization.status, 'optimization': asdict(optimization),
                  'best_parameters': optimization.best_parameters, 'evaluations': {}}
        store.write_json('result.json', result)
        if not args.no_evaluate:
            result['evaluations']['training'] = export_and_evaluate(prepared, stepper, optimization.best_parameters, store, runtime)
            validation = prepare_experiment(config, split='validation', build_sdf=False)
            result['evaluations']['validation'] = export_and_evaluate(validation, stepper, optimization.best_parameters, store, runtime)
            store.write_json('result.json', result)
        evaluation_failed = any(value.get('strict_evaluation', {}).get('valid') is not True
                                for value in result['evaluations'].values())
        optimizer_failed = optimization.status in {'invalid_initial', 'stalled_invalid', 'stalled_descent'}
        if evaluation_failed:
            result['status'] = 'independent_evaluation_failed'
        result['runtime'] = {**runtime,
                             'forward_verified': successful_objectives[0] > 0 or bool(result['evaluations']),
                             'backward_verified': successful_objectives[0] > 0,
                             'successful_objectives_in_process': successful_objectives[0],
                             'objectives_with_replay_mismatch': objectives_with_mismatches[0],
                             'backward_replay_consistent': successful_objectives[0] > 0 and objectives_with_mismatches[0] == 0,
                             'execution_scope': 'this process; restored optimizer records are not new execution'}
        store.write_json('result.json', result)
        print(f'Calibration result: {store.path / "result.json"}', flush=True)
        return 2 if evaluation_failed or optimizer_failed else 0


def main(argv=None):
    args = parse_args(argv)
    try:
        with reference_policy(args.reference_policy):
            return run(args)
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    raise SystemExit(main())

"""CLI integration using analytic mock rollouts; these tests do not execute MPM."""
from contextlib import contextmanager, ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.differentiable_mpm import calibrate
from experiments.differentiable_mpm.config import ExperimentConfig, FrameWindow, REQUIRED_PATHS
from experiments.differentiable_mpm.optimize import AdamOptions, ProjectedAdam
from experiments.differentiable_mpm.parameters import PhysicalParameterSpace
from experiments.differentiable_mpm.results import RunStore
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, PARAMETER_NAMES, InvalidStateError, SimulationConfig
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class AnalyticRollout:
    """A supplied-gradient objective replacing all simulator/renderer execution."""

    def __init__(self, target=0.5, invalid_proposals=False):
        self.target = target
        self.invalid_proposals = invalid_proposals
        self.calls = []

    def value_and_gradient(self, parameters, compute_grad=True, **kwargs):
        self.calls.append((dict(parameters), compute_grad))
        if self.invalid_proposals and parameters['viscosity'] != 0.1:
            raise InvalidStateError('Mock unstable proposal')
        delta = parameters['viscosity'] - self.target
        gradient = dict.fromkeys(PARAMETER_NAMES, 0.0)
        gradient['viscosity'] = delta
        return SimpleNamespace(value=0.5 * delta ** 2, gradient=gradient if compute_grad else None,
                               frames=[], diagnostics={'mock_rollout': True}, initial_gradient=None)


class CalibrateCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory(prefix='calibrate-cli-')
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.config = ExperimentConfig(
            name='cli_fixture', paths={name: self.root / name for name in REQUIRED_PATHS},
            expected_sha256={}, simulation={'n_particles': 1, 'grid': 8, 'dt': 0.001,
                                           'precision': 'f64', 'plasticity': 'none', 'tool_collision': 'none'},
            parameters=dict(DEFAULT_PARAMETERS, viscosity=0.1), fit_parameters=['viscosity'],
            parameter_bounds={'viscosity': [0.0, 1.0]},
            training=FrameWindow(1, 3), validation=FrameWindow(4, 5), segment_length=2,
            optimizer={'learning_rate': 0.0003}, tool_sdf_resolution=16,
        )

    def parameters_file(self, record, name='parameters.json'):
        path = self.root / name
        path.write_text(json.dumps(record) + '\n')
        return path

    @contextmanager
    def environment(self, *, config=None, source='source-a', rollout=None, strict_valid=True):
        template = deepcopy(config or self.config)
        rollout = rollout or AnalyticRollout()
        prepared_calls = []
        stores = []
        exports = []

        def load(path, overrides):
            result = deepcopy(template)
            result.paths.update({key: Path(value) for key, value in overrides.items()})
            result.path_overrides = dict(overrides)
            return result

        def prepare(current, split='training', end_frame=None, build_sdf=True):
            window = current.window(split)
            end = window.end_frame if end_frame is None else end_frame
            if not window.start_frame <= end <= window.end_frame:
                raise ValueError('Requested endpoint must be inside the selected scoring window')
            scored = tuple(range(window.start_frame, end + 1))
            result = SimpleNamespace(
                config=current, simulation_config=SimulationConfig(**current.simulation),
                parameters=dict(current.parameters), sdf=None, observations=list(scored),
                total_steps=end, end_frame=end, split=split, scored_frames=scored,
                frames=tuple(SimpleNamespace(source_frame=i, completed_substeps=i) for i in range(end + 1)),
                provenance={'mock_prepared': True, 'config': current.as_dict(), 'split': split,
                            'scored_frames': list(scored), 'input_content_hash': 'fixture-input-v1'},
            )
            result.summary = lambda: {'mock_prepared': True, 'split': split, 'scored_frames': list(scored)}
            prepared_calls.append((split, end, build_sdf))
            return result

        def runtime(backend, precision, threads, debug, seed):
            return {'backend': backend, 'precision': precision, 'threads': threads,
                    'debug': debug, 'seed': seed, 'initialization_verified': True, 'mock_runtime': True}

        def store(path, identity, resume=False):
            result = RunStore(path, identity, resume=resume, allowed_root=self.root)
            stores.append(result)
            return result

        def export(prepared, stepper, parameters, store, runtime, strict=True):
            exports.append((prepared.split, dict(parameters), strict))
            return {'split': prepared.split, 'strict_evaluation': {
                'valid': strict_valid, 'status': 'completed' if strict_valid else 'invalid',
                'failure_reason': None if strict_valid else 'Mock insufficient observed support',
            } if strict else None}

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            mocks = {}
            for name, replacement in (
                ('load_config', Mock(side_effect=load)),
                ('verify_reference', Mock(return_value={})),
                ('verify_input_paths', Mock(return_value={'mock_verified': True})),
                ('prepare_experiment', Mock(side_effect=prepare)),
                ('init_runtime', Mock(side_effect=runtime)),
                ('source_identity', Mock(return_value={'source': source})),
                ('estimate_memory', Mock(return_value={'mock_estimate': True})),
                ('make_rollout', Mock(return_value=(SimpleNamespace(), rollout))),
                ('export_and_evaluate', Mock(side_effect=export)),
                ('RunStore', store), ('RUN_ROOT', self.root),
            ):
                stack.enter_context(patch.object(calibrate, name, replacement))
                mocks[name] = replacement
            yield SimpleNamespace(mocks=mocks, prepared_calls=prepared_calls, stores=stores,
                                  exports=exports, rollout=rollout)

    def run_cli(self, arguments):
        return calibrate.run(calibrate.parse_args(arguments))

    def test_action_and_option_parsing(self):
        args = calibrate.parse_args(['gradient', '--config', 'sample.json', '--path', 'calibration=/x=a.json',
                                     '--backend', 'cpu', '--precision', 'f64', '--segment-length', '8',
                                     '--end-frame', '2', '--finite-difference'])
        self.assertEqual(args.action, 'gradient')
        self.assertEqual(args.config, Path('sample.json'))
        self.assertEqual(args.path, ['calibration=/x=a.json'])
        self.assertEqual(args.segment_length, 8)
        self.assertEqual(args.end_frame, 2)
        self.assertTrue(args.finite_difference)

    def test_main_applies_and_restores_explicit_reference_policy(self):
        from experiments.differentiable_mpm.reference_adapter import _REFERENCE_POLICY
        original = _REFERENCE_POLICY.get()
        def inspect_policy(args):
            self.assertEqual(_REFERENCE_POLICY.get(), args.reference_policy)
            return 0
        with patch.object(calibrate, 'run', side_effect=inspect_policy):
            self.assertEqual(calibrate.main(['validate', '--reference-policy', 'frozen']), 0)
            self.assertEqual(calibrate.main(['validate']), 0)
        self.assertEqual(_REFERENCE_POLICY.get(), original)
        with patch.object(calibrate, 'run', side_effect=ValueError('probe')), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'probe'):
                calibrate.main(['validate', '--reference-policy', 'frozen'])
        self.assertEqual(_REFERENCE_POLICY.get(), original)

    def test_forbidden_action_combinations_fail_before_execution(self):
        cases = [
            ['fit', '--split', 'validation'], ['gradient', '--split', 'validation'],
            ['evaluate'], ['fit', '--parameters', 'parameters.json'], ['fit', '--resume'],
            ['replay', '--resume', '--output-dir', 'x'], ['fit', '--no-runtime'],
            ['fit', '--finite-difference'], ['fit', '--iterations', '-1'], ['unknown'],
        ]
        for argv in cases:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    calibrate.parse_args(argv)
                self.assertNotEqual(result.exception.code, 0)

    def test_path_override_parsing_rejects_duplicates_and_missing_separator(self):
        for paths in (['calibration'], ['calibration=a', 'calibration=b']):
            with self.subTest(paths=paths), self.environment() as env:
                arguments = ['validate', '--no-runtime']
                for value in paths:
                    arguments += ['--path', value]
                with self.assertRaises(ValueError):
                    self.run_cli(arguments)
                env.mocks['load_config'].assert_not_called()
        with self.environment() as env:
            path = self.root / 'calibration=relocated.json'
            self.assertEqual(self.run_cli(['validate', '--no-runtime', '--path', f'calibration={path}']), 0)
            self.assertEqual(env.mocks['load_config'].call_args.args[1], {'calibration': str(path)})
            env.mocks['verify_input_paths'].assert_called_once()

    def test_parameter_files_extract_complete_values_and_fingerprint_original_bytes(self):
        for i, record in enumerate((DEFAULT_PARAMETERS, {'best_parameters': DEFAULT_PARAMETERS},
                                    {'optimization': {'best_parameters': DEFAULT_PARAMETERS}})):
            path = self.parameters_file(record, f'parameters-{i}.json')
            values, source = calibrate.read_parameters(path)
            self.assertEqual(values, DEFAULT_PARAMETERS)
            self.assertEqual(source['path'], str(path.resolve()))
            self.assertEqual(source['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        for i, record in enumerate(({'youngs_modulus': 100}, {'best_parameters': {'viscosity': 1}},
                                    dict(DEFAULT_PARAMETERS, viscosity=float('nan')))):
            with self.subTest(record=record), self.assertRaises(ValueError):
                calibrate.read_parameters(self.parameters_file(record, f'invalid-{i}.json'))

    def test_inactive_plastic_parameters_rejected_before_runtime_or_preparation(self):
        config = deepcopy(self.config)
        config.fit_parameters = ['viscosity', 'plastic_min']
        with self.environment(config=config) as env:
            with self.assertRaisesRegex(ValueError, 'explicit plasticity'):
                self.run_cli(['fit', '--iterations', '0', '--no-evaluate'])
            env.mocks['init_runtime'].assert_not_called()
            env.mocks['prepare_experiment'].assert_not_called()
            env.mocks['make_rollout'].assert_not_called()

    def test_validate_no_runtime_does_not_claim_backend_execution(self):
        output = self.root / 'validate'
        with self.environment() as env:
            self.assertEqual(self.run_cli(['validate', '--no-runtime', '--output-dir', str(output)]), 0)
            env.mocks['init_runtime'].assert_not_called()
            env.mocks['make_rollout'].assert_not_called()
        result = json.loads((output / 'result.json').read_text())
        self.assertFalse(result['runtime']['initialization_verified'])
        self.assertEqual(result['status'], 'inputs_valid')

    def test_mock_gradient_action_uses_real_parameter_transform_api(self):
        output = self.root / 'gradient'
        with self.environment() as env:
            code = self.run_cli(['gradient', '--finite-difference', '--output-dir', str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(env.prepared_calls, [('training', 3, True)])
            self.assertTrue(env.rollout.calls[0][1])
            self.assertTrue(all(not compute_grad for _, compute_grad in env.rollout.calls[1:]))
        result = json.loads((output / 'result.json').read_text())
        self.assertEqual(result['status'], 'gradient_computed')
        self.assertTrue(result['finite_difference']['passed'])
        self.assertAlmostEqual(result['coordinate_gradient'][0], -40.0)

    def test_finite_difference_detects_wrong_gradient_and_parameter_space(self):
        space = PhysicalParameterSpace(self.config.parameters, ['viscosity'], 'none',
                                       bounds={'viscosity': [0, 1]}, scales={'viscosity': 1})
        wrong = calibrate.finite_difference_report(AnalyticRollout(), space, self.config.parameters, {'viscosity': 2})
        self.assertFalse(wrong['passed'])
        with self.assertRaisesRegex(ValueError, 'does not describe'):
            calibrate.finite_difference_report(AnalyticRollout(), space,
                                               dict(self.config.parameters, viscosity=0.2), {'viscosity': -0.3})

    def test_finite_difference_reports_one_sided_stencils_at_large_coordinate_bounds(self):
        parameters = dict(DEFAULT_PARAMETERS, viscosity=1000.0)
        space = PhysicalParameterSpace(parameters, ['viscosity'], 'none')
        rollout = AnalyticRollout(target=0.0)
        report = calibrate.finite_difference_report(rollout, space, parameters, {'viscosity': 1000.0})
        self.assertTrue(report['passed'])
        for check in report['rows'][0]['checks']:
            self.assertEqual(check['scheme'], 'one-sided at bound')

    def test_optimizer_callback_integration_and_exact_resume(self):
        output = self.root / 'fit'
        with self.environment() as env:
            self.assertEqual(self.run_cli(['fit', '--iterations', '1', '--no-evaluate', '--output-dir', str(output)]), 0)
            self.assertEqual(env.prepared_calls, [('training', 3, True)])
        first = json.loads((output / 'optimizer_state.json').read_text())
        self.assertEqual(first['state']['iterations'], 1)
        self.assertEqual(first['state']['accepted_updates'], 1)
        with self.environment() as env:
            self.assertEqual(self.run_cli(['fit', '--iterations', '2', '--resume', '--no-evaluate', '--output-dir', str(output)]), 0)
            self.assertEqual(len(env.rollout.calls), 2)
        saved = json.loads((output / 'optimizer_state.json').read_text())
        state = saved['state']
        self.assertEqual(state['iterations'], 3)
        self.assertEqual(state['accepted_updates'], 3)
        self.assertEqual(state['objective_id'], saved['identity_sha256'])
        space = PhysicalParameterSpace(state['space']['initial'], state['space']['fit'], state['space']['plasticity'],
                                       bounds=state['space']['bounds'], scales=state['space']['scales'])
        restored = ProjectedAdam(space, lambda parameters: None, AdamOptions(**state['options']),
                                 objective_id=saved['identity_sha256'])
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        selected = json.loads((output / 'selected_parameters.json').read_text())
        self.assertEqual(selected['selection_split'], 'training')
        self.assertEqual(selected['selection_frames'], [1, 2, 3])
        result = json.loads((output / 'result.json').read_text())
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['optimization']['iterations'], 3)

    def test_fit_execution_flags_count_actual_rollouts_in_this_process(self):
        output = self.root / 'execution_flags'
        base = ['fit', '--iterations', '0', '--no-evaluate', '--output-dir', str(output)]
        with self.environment() as env:
            self.assertEqual(self.run_cli(base), 0)
            self.assertEqual(len(env.rollout.calls), 1)
        first = json.loads((output / 'result.json').read_text())['runtime']
        self.assertTrue(first['forward_verified'])
        self.assertTrue(first['backward_verified'])
        self.assertEqual(first['successful_objectives_in_process'], 1)
        with self.environment() as env:
            self.assertEqual(self.run_cli(base + ['--resume']), 0)
            self.assertEqual(len(env.rollout.calls), 0)
        resumed = json.loads((output / 'result.json').read_text())['runtime']
        self.assertFalse(resumed['forward_verified'])
        self.assertFalse(resumed['backward_verified'])
        self.assertEqual(resumed['successful_objectives_in_process'], 0)

    def test_resume_rejects_changed_runtime_inputs_sources_or_optimizer(self):
        mutations = [
            ('backend', None, 'source-a', ['--backend', 'vulkan']),
            ('precision', None, 'source-a', ['--precision', 'f32']),
            ('path', None, 'source-a', ['--path', 'calibration=/different/calibration.json']),
            ('source', None, 'source-b', []),
            ('optimizer', None, 'source-a', ['--learning-rate', '0.001']),
            ('rate_policy', None, 'source-a', ['--learning-rate-policy', 'recover-v1']),
            ('rate_growth', None, 'source-a', ['--learning-rate-growth', '1.5']),
            ('horizon', None, 'source-a', ['--end-frame', '2']),
        ]
        changed = deepcopy(self.config)
        changed.parameters['youngs_modulus'] += 1
        mutations.append(('fixed_parameter', changed, 'source-a', []))
        for name, config, source, extra in mutations:
            with self.subTest(name=name):
                output = self.root / ('resume_' + name)
                base = ['fit', '--iterations', '0', '--no-evaluate', '--output-dir', str(output)]
                with self.environment():
                    self.assertEqual(self.run_cli(base), 0)
                with self.environment(config=config, source=source) as env:
                    with self.assertRaisesRegex(ValueError, 'Run identity differs'):
                        self.run_cli(base + ['--resume'] + extra)
                    env.mocks['make_rollout'].assert_not_called()

    def test_training_selection_is_frozen_before_held_out_evaluation(self):
        output = self.root / 'fit_with_validation'
        with self.environment() as env:
            self.assertEqual(self.run_cli(['fit', '--iterations', '1', '--output-dir', str(output)]), 0)
            self.assertEqual(env.prepared_calls, [('training', 3, True), ('validation', 5, False)])
            self.assertEqual([split for split, _, _ in env.exports], ['training', 'validation'])
            self.assertEqual(env.exports[0][1], env.exports[1][1])
            self.assertTrue(all(strict for _, _, strict in env.exports))
        selected = json.loads((output / 'selected_parameters.json').read_text())
        self.assertEqual(env.exports[1][1], selected['best_parameters'])
        self.assertEqual(selected['selection_frames'], [1, 2, 3])

    def test_evaluate_defaults_to_held_out_and_records_frozen_parameter_source(self):
        output = self.root / 'evaluate'
        source = self.parameters_file({'best_parameters': self.config.parameters})
        with self.environment() as env:
            self.assertEqual(self.run_cli(['evaluate', '--parameters', str(source), '--output-dir', str(output)]), 0)
            self.assertEqual(env.prepared_calls, [('validation', 5, True)])
            self.assertEqual(env.exports[0][0], 'validation')
            self.assertTrue(env.exports[0][2])
        manifest = json.loads((output / 'run_manifest.json').read_text())
        self.assertEqual(manifest['identity']['frozen_parameter_source']['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_failed_gradient_check_is_nonzero(self):
        output = self.root / 'failed_gradient'
        rollout = AnalyticRollout()
        original = rollout.value_and_gradient

        def wrong_gradient(parameters, compute_grad=True, **kwargs):
            result = original(parameters, compute_grad, **kwargs)
            if compute_grad:
                result.gradient['viscosity'] = 2.0
            return result

        rollout.value_and_gradient = wrong_gradient
        with self.environment(rollout=rollout):
            self.assertNotEqual(self.run_cli(['gradient', '--finite-difference', '--output-dir', str(output)]), 0)
        self.assertEqual(json.loads((output / 'result.json').read_text())['status'], 'gradient_check_failed')

    def test_invalid_gradient_writes_failure_without_claiming_execution(self):
        output = self.root / 'invalid_gradient'
        rollout = AnalyticRollout()
        rollout.value_and_gradient = Mock(side_effect=InvalidStateError('nonfinite adjoint at step 64'))
        with self.environment(rollout=rollout):
            with self.assertRaisesRegex(InvalidStateError, 'nonfinite adjoint'):
                self.run_cli(['gradient', '--output-dir', str(output)])
        result = json.loads((output / 'result.json').read_text())
        self.assertEqual(result['status'], 'gradient_failed')
        self.assertIn('step 64', result['error'])
        self.assertFalse(result['runtime'].get('backward_verified', False))
        self.assertNotIn('gradient', result)

    def test_invalid_initial_fit_persists_failure_without_selected_parameters(self):
        output = self.root / 'invalid_initial_fit'
        rollout = AnalyticRollout()
        rollout.value_and_gradient = Mock(side_effect=InvalidStateError(
            'Contact/yield summaries differ on recomputation at step 9774'))
        with self.environment(rollout=rollout):
            with self.assertRaisesRegex(InvalidStateError, 'step 9774'):
                self.run_cli(['fit', '--iterations', '20', '--no-evaluate', '--output-dir', str(output)])
        result = json.loads((output / 'result.json').read_text())
        self.assertEqual(result['status'], 'invalid_initial')
        self.assertEqual(result['accepted_updates'], 0)
        self.assertEqual(result['optimizer_evaluations'], 1)
        self.assertFalse(result['runtime']['backward_verified'])
        self.assertEqual(result['runtime']['successful_objectives_in_process'], 0)
        self.assertFalse((output / 'selected_parameters.json').exists())
        saved = json.loads((output / 'optimizer_state.json').read_text())
        self.assertEqual(saved['state']['status'], 'invalid_initial')
        events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
        self.assertTrue(any(event.get('event') == 'fit_failed' for event in events))

    def test_recompute_failure_is_printed_and_persisted(self):
        event = {'phase': 'recompute_mismatch', 'step': 9774, 'total_steps': 10006,
                 'stage': 'segment_forward_recompute', 'segment_start_step': 9728,
                 'segment_end_step': 9792, 'expected_counts': {'particle_tool0': 18},
                 'recomputed_counts': {'particle_tool0': 19},
                 'differing_counts': {'particle_tool0': {'forward': 18, 'recomputed': 19}}}
        store = Mock()
        stream = io.StringIO()
        with redirect_stdout(stream):
            calibrate.stored_progress(store)(event)
        store.write_json.assert_called_once_with('last_recompute_failure.json', event)
        store.append_event.assert_called_once_with({'event': 'replay_progress', **event})
        self.assertIn('9774/10006', stream.getvalue())
        self.assertIn('9728–9792', stream.getvalue())
        self.assertIn('particle_tool0', stream.getvalue())

    def test_failed_required_strict_evaluation_is_not_reported_as_complete(self):
        output = self.root / 'failed_evaluate'
        source = self.parameters_file(self.config.parameters)
        with self.environment(strict_valid=False):
            code = self.run_cli(['evaluate', '--parameters', str(source), '--output-dir', str(output)])
        result = json.loads((output / 'result.json').read_text())
        self.assertFalse(result['strict_evaluation']['valid'])
        self.assertNotEqual(code, 0)
        self.assertNotEqual(result['status'], 'replay_complete')

    def test_fit_reports_required_validation_failure(self):
        output = self.root / 'fit_failed_evaluation'
        with self.environment(strict_valid=False):
            code = self.run_cli(['fit', '--iterations', '0', '--output-dir', str(output)])
        result = json.loads((output / 'result.json').read_text())
        self.assertFalse(result['evaluations']['validation']['strict_evaluation']['valid'])
        self.assertNotEqual(code, 0)
        self.assertNotEqual(result['status'], 'budget_exhausted')

    def test_optimizer_with_only_invalid_proposals_returns_nonzero(self):
        output = self.root / 'fit_stalled_invalid'
        with self.environment(rollout=AnalyticRollout(invalid_proposals=True)):
            code = self.run_cli(['fit', '--iterations', '3', '--no-evaluate', '--output-dir', str(output)])
        result = json.loads((output / 'result.json').read_text())
        self.assertEqual(result['optimization']['accepted_updates'], 0)
        self.assertEqual(result['optimization']['status'], 'stalled_invalid')
        self.assertNotEqual(code, 0)


if __name__ == '__main__':
    unittest.main()

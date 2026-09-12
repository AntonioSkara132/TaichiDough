"""Host-only dataset CLI tests with actual persistence, aggregation and optimizer."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.differentiable_mpm import calibrate_dataset as cli
from experiments.differentiable_mpm.config import REQUIRED_PATHS, SCHEMA as EPISODE_SCHEMA
from experiments.differentiable_mpm.dataset_config import MATERIAL_NAMES, SCHEMA
from experiments.differentiable_mpm.multi_episode import EpisodeExecutionError
from experiments.differentiable_mpm.results import RunStore
from experiments.differentiable_mpm.state import DEFAULT_PARAMETERS, PARAMETER_NAMES
from experiments.differentiable_mpm.tests.helpers import temporary_directory


class AnalyticEpisodeRunner:
    def __init__(self, owner, dataset):
        self.owner, self.dataset = owner, dataset

    def __call__(self, episode, shared, compute_grad=True):
        return self.run(episode, 'objective', shared, compute_grad=compute_grad)['evaluation']

    def run(self, episode, action, shared, *, compute_grad=False, no_runtime=False):
        effective = episode.parameters_for(shared)
        call = {'episode_id': episode.id, 'action': action, 'shared': dict(shared),
                'effective': effective, 'compute_grad': compute_grad, 'no_runtime': no_runtime,
                'scored_window': episode.scored_window}
        self.owner.calls.append(call)
        if action == 'validate':
            if episode.id in self.owner.preflight_failures:
                raise EpisodeExecutionError('Mock missing reconstruction')
            summary = self.owner.summary(episode, no_runtime=no_runtime)
            return {'prepared_summary': summary['summary'], 'prepared_fingerprint': summary['prepared_fingerprint'],
                    'runtime': summary['runtime']}
        if action == 'objective':
            if episode.membership != 'training':
                raise AssertionError('Held-out episode entered optimizer objective')
            delta = shared['viscosity'] - self.owner.targets[episode.id]
            gradient = dict.fromkeys(PARAMETER_NAMES, 0.0)
            gradient['viscosity'] = delta
            return {'evaluation': {'value': 0.5 * delta ** 2,
                    'gradient': gradient if compute_grad else None,
                    'frames': [], 'diagnostics': {'replay_consistent': True}}}
        if action in {'replay', 'evaluate'}:
            return {'export': {'strict_evaluation': {'valid': self.owner.strict_valid},
                               'effective_parameters': effective,
                               'scored_frames': list(episode.scored_window.indices())}}
        raise AssertionError(action)


class DatasetCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory(prefix='dataset-cli-')
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.calls, self.datasets = [], []
        self.targets = {'train_a': 0.8, 'train_b': 0.0, 'holdout': 1000.0}
        self.floors = {'train_a': 0.2, 'train_b': 0.6, 'holdout': 0.9}
        self.strict_valid = True
        self.preflight_failures = set()
        self.same_sequence = False
        self.prepared_revision = 'prepared-v1'
        self.dataset_path = self.root / 'dataset.json'
        self.shared = {name: DEFAULT_PARAMETERS[name] for name in MATERIAL_NAMES}
        self.shared['viscosity'] = 0.1
        rows = []
        for index, name in enumerate(self.floors):
            config_path = self.root / (name + '.json')
            config = {'schema': EPISODE_SCHEMA, 'name': name,
                      'paths': {key: str(self.root / name / key) for key in REQUIRED_PATHS},
                      'expected_sha256': {},
                      'simulation': {'n_particles': 1, 'grid': 8, 'precision': 'f64',
                                     'plasticity': 'stretch-clamp', 'physics_version': 'corrected-v1'},
                      'parameters': dict(DEFAULT_PARAMETERS, floor_retention=self.floors[name]),
                      'fit_parameters': ['viscosity'], 'parameter_bounds': {'viscosity': [0.0, 1.0]},
                      'training': {'start_frame': 1, 'end_frame': 3},
                      'validation': {'start_frame': 4, 'end_frame': 5},
                      'optimizer': {'learning_rate': 0.0001}, 'segment_length': 2}
            config_path.write_text(json.dumps(config))
            rows.append({'id': name, 'config': str(config_path),
                         'membership': 'validation' if name == 'holdout' else 'training',
                         'weight': 3.0 if name == 'train_b' else 1.0,
                         'scored_window': {'start_frame': 1, 'end_frame': 7 + index, 'stride': 1}})
        self.document = {'schema': SCHEMA, 'name': 'host-fixture',
                         'shared_parameters': {'initial': self.shared, 'fit': ['viscosity'],
                                               'bounds': {'viscosity': [0.0, 1.0]}},
                         'episodes': rows}
        self.write_dataset()

    def write_dataset(self):
        self.dataset_path.write_text(json.dumps(self.document))

    def summary(self, episode, *, no_runtime=False):
        sequence = 'shared-sequence' if self.same_sequence else episode.id
        return {'prepared_fingerprint': hashlib.sha256((episode.id + self.prepared_revision).encode()).hexdigest(),
                'runtime': {'backend': episode.config.backend,
                            'precision': episode.config.simulation.get('precision', 'f32'),
                            'actual_arch': None if no_runtime else {'cpu': 'Arch.x64', 'cuda': 'Arch.cuda', 'vulkan': 'Arch.vulkan'}[episode.config.backend],
                            'initialization_verified': not no_runtime, 'seed': 0, 'cpu_threads': 1},
                'summary': {'provenance': {'sequence_fingerprint': hashlib.sha256(sequence.encode()).hexdigest()},
                            'scored_frames': list(episode.scored_window.indices())}}

    @contextmanager
    def environment(self, *, actual_preflight=False, source='source-v1'):
        def runner(dataset, *args, **kwargs):
            self.datasets.append(dataset)
            return AnalyticEpisodeRunner(self, dataset)
        def preflight(dataset, args, options, source):
            summaries = {ep.id: self.summary(ep, no_runtime=args.no_runtime) for ep in dataset.episodes}
            return summaries, {name: {'error': 'Mock preflight failure'} for name in self.preflight_failures}, self.root / 'preflight'
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(patch.object(cli, 'RUN_ROOT', self.root))
            stack.enter_context(patch.object(cli, 'source_identity', return_value={'fixture_source': source}))
            stack.enter_context(patch('experiments.differentiable_mpm.reference_adapter.reference_identity',
                side_effect=lambda version: {'physics_version': version, 'simulator_sha256': version + '-mock-sha',
                                            'manifest_sha256': 'mock-manifest', 'simulator_snapshot': version}))
            stack.enter_context(patch.object(cli, 'RunStore', side_effect=lambda path, identity, resume=False:
                RunStore(path, identity, resume=resume, allowed_root=self.root)))
            mocked_runner = stack.enter_context(patch.object(cli, '_runner', side_effect=runner))
            if not actual_preflight:
                stack.enter_context(patch.object(cli, 'preflight', side_effect=preflight))
            yield mocked_runner

    def run_cli(self, action, output, *extra):
        return cli.run(cli.parse_args([action, '--dataset', str(self.dataset_path),
                                      '--output-dir', str(output), *extra]))

    def json(self, output, name='result.json'):
        return json.loads((output / name).read_text())

    def fit(self, output, *extra):
        return self.run_cli('fit', output, '--iterations', '1', '--no-evaluate', *extra)

    def test_parser_rejects_invalid_combinations_before_execution(self):
        d = ['--dataset', str(self.dataset_path)]
        cases = [['fit'], ['inventory'], ['fit', *d, '--no-runtime'],
                 ['validate', *d, '--finite-difference'], ['gradient', *d, '--no-evaluate'],
                 ['fit', *d, '--resume'], ['evaluate', *d], ['fit', *d, '--parameters', 'x'],
                 ['fit', *d, '--split', 'validation'], ['gradient', *d, '--split', 'training'],
                 ['fit', *d, '--root', '/x'], ['fit', *d, '--segment-length', '0'],
                 ['fit', *d, '--worker-timeout-s', 'nan'], ['fit', *d, '--iterations', '-1'],
                 ['fit', *d, '--physics-version', 'unknown'], ['fit', *d, '--cpu-threads', '0']]
        for args in cases:
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(args)

    def test_options_reach_all_episodes_and_explicit_path_only(self):
        output = self.root / 'overrides'
        with self.environment():
            self.assertEqual(self.run_cli('validate', output, '--no-runtime', '--backend', 'cuda',
                '--precision', 'f32', '--physics-version', 'legacy-v1', '--p2g-mode', 'serial',
                '--segment-length', '5', '--path', 'train_a.episode=/mnt/episode=a'), 0)
        for ep in self.datasets[-1].episodes:
            self.assertEqual(ep.config.backend, 'cuda')
            self.assertEqual(ep.config.simulation['precision'], 'f32')
            self.assertEqual(ep.config.simulation['physics_version'], 'legacy-v1')
            self.assertEqual(ep.config.simulation['p2g_mode'], 'serial')
            self.assertEqual(ep.config.segment_length, 5)
            expected = Path('/mnt/episode=a') if ep.id == 'train_a' else self.root / ep.id / 'episode'
            self.assertEqual(ep.config.paths['episode'], expected)
        self.assertEqual(self.calls, [])
        result = self.json(output)
        self.assertTrue(result['no_runtime'])

    def test_path_overrides_reject_missing_empty_duplicate_and_unknown_ids(self):
        for flags in (['--path', 'episode=x'], ['--path', 'train_a.episode='],
                      ['--path', 'train_a.episode=x', '--path', 'train_a.episode=y'],
                      ['--path', 'unknown.episode=x']):
            with self.subTest(flags=flags), self.environment() as runner, self.assertRaises(ValueError):
                self.run_cli('validate', self.root / 'invalid', '--no-runtime', *flags)
            runner.assert_not_called()

    def test_full_explicit_windows_are_not_clipped_to_episode_training_split(self):
        output = self.root / 'windows'
        with self.environment():
            self.assertEqual(self.run_cli('replay', output), 0)
        self.assertEqual([call['scored_window'].end_frame for call in self.calls], [7, 8, 9])
        self.assertEqual(self.json(output)['evaluations']['holdout']['scored_frames'], list(range(1, 10)))
        del self.document['episodes'][0]['scored_window']
        self.write_dataset()
        with self.environment(), self.assertRaises(ValueError):
            self.run_cli('validate', self.root / 'missing-window', '--no-runtime')

    def test_actual_optimizer_acceptance_uses_complete_weighted_training_set(self):
        output = self.root / 'fit'
        with self.environment():
            self.assertEqual(self.fit(output), 0)
        objective_calls = [call for call in self.calls if call['action'] == 'objective']
        self.assertGreaterEqual(len(objective_calls), 4)
        self.assertEqual(len(objective_calls) % 2, 0)
        for i in range(0, len(objective_calls), 2):
            a, b = objective_calls[i:i+2]
            self.assertEqual([a['episode_id'], b['episode_id']], ['train_a', 'train_b'])
            self.assertEqual(a['shared'], b['shared'])
        result = self.json(output)
        self.assertEqual(result['optimization']['accepted_updates'], 1)
        eta = result['shared_material_parameters']['viscosity']
        self.assertGreater(eta, 0.1)  # train_b worsens, but the complete weighted objective decreases.
        expected = 0.25 * 0.5 * (eta - 0.8) ** 2 + 0.75 * 0.5 * eta ** 2
        selected = self.json(output, 'selected_parameters.json')
        self.assertAlmostEqual(selected['training_value'], expected)
        self.assertLess(expected, 0.25 * 0.5 * 0.7 ** 2 + 0.75 * 0.5 * 0.1 ** 2)

    def test_selection_contains_shared_only_and_actual_per_episode_floors(self):
        output = self.root / 'selection'
        with self.environment():
            self.assertEqual(self.fit(output), 0)
        selected = self.json(output, 'selected_parameters.json')
        self.assertEqual(selected['schema'], cli.SELECTION_SCHEMA)
        self.assertEqual(selected['physics_version'], 'corrected-v1')
        self.assertEqual(selected['physics_reference']['simulator_sha256'], 'corrected-v1-mock-sha')
        self.assertEqual(set(selected['shared_material_parameters']), set(MATERIAL_NAMES))
        self.assertEqual(selected['selection_membership'], 'training')
        self.assertTrue(selected['independent_heldout_episodes_declared'])
        for name, floor in self.floors.items():
            parameters = selected['episodes'][name]['effective_parameters']
            self.assertEqual(parameters['floor_retention'], floor)
            self.assertEqual(parameters['tool_retention'], 1.0)
        for call in self.calls:
            self.assertEqual(call['effective']['floor_retention'], self.floors[call['episode_id']])
            self.assertEqual(call['effective']['tool_retention'], 1.0)

    def test_holdout_evaluation_uses_frozen_selection_after_training(self):
        output = self.root / 'holdout'
        with self.environment():
            self.assertEqual(self.run_cli('fit', output, '--iterations', '1'), 0)
        objectives = [c for c in self.calls if c['action'] == 'objective']
        exports = [c for c in self.calls if c['action'] == 'evaluate']
        self.assertTrue(all(c['episode_id'] != 'holdout' for c in objectives))
        self.assertEqual([c['episode_id'] for c in exports], ['train_a', 'train_b', 'holdout'])
        selected = self.json(output, 'selected_parameters.json')['shared_material_parameters']
        self.assertTrue(all(c['shared'] == selected for c in exports))
        first_export = next(i for i, call in enumerate(self.calls) if call['action'] == 'evaluate')
        self.assertTrue(all(call['action'] != 'objective' for call in self.calls[first_export:]))

    def test_exact_resume_preserves_optimizer_and_new_objectives_are_complete(self):
        output = self.root / 'resume'
        with self.environment():
            self.assertEqual(self.fit(output), 0)
        first = self.json(output, 'optimizer_state.json')
        self.calls.clear()
        with self.environment():
            self.assertEqual(self.fit(output, '--resume'), 0)
        second = self.json(output, 'optimizer_state.json')
        self.assertEqual(first['identity_sha256'], second['identity_sha256'])
        self.assertEqual(second['state']['iterations'], 2)
        self.assertEqual(second['state']['accepted_updates'], 2)
        self.assertEqual([c['episode_id'] for c in self.calls], ['train_a', 'train_b'])

    def test_resume_rejects_membership_weight_window_physics_policy_and_prepared_changes(self):
        original = deepcopy(self.document)
        mutations = ('membership', 'weight', 'window', 'physics', 'policy', 'prepared', 'source', 'recompute_policy')
        for name in mutations:
            with self.subTest(name=name):
                self.document = deepcopy(original)
                self.prepared_revision = 'prepared-v1'
                self.write_dataset()
                output = self.root / ('resume-' + name)
                with self.environment():
                    self.assertEqual(self.fit(output), 0)
                extra, source = [], 'source-v1'
                if name == 'membership':
                    self.document['episodes'][1]['membership'] = 'validation'
                elif name == 'weight':
                    self.document['episodes'][1]['weight'] = 4.0
                elif name == 'window':
                    self.document['episodes'][0]['scored_window']['end_frame'] = 6
                elif name == 'physics':
                    extra = ['--physics-version', 'legacy-v1']
                elif name == 'policy':
                    extra = ['--reference-policy', 'frozen']
                elif name == 'prepared':
                    self.prepared_revision = 'prepared-v2'
                elif name == 'source':
                    source = 'source-v2'
                else:
                    extra = ['--ignore-recompute-mismatch']
                self.write_dataset()
                self.calls.clear()
                with self.environment(source=source), self.assertRaisesRegex(ValueError, 'Run identity differs'):
                    self.fit(output, '--resume', *extra)
                self.assertEqual(self.calls, [])

    def test_failed_preflight_never_starts_objective_or_writes_selection(self):
        output = self.root / 'preflight-failed'
        self.preflight_failures = {'train_b'}
        with self.environment(actual_preflight=True):
            self.assertEqual(self.fit(output), 2)
        self.assertEqual([c['action'] for c in self.calls], ['validate'] * 3)
        self.assertFalse((output / 'selected_parameters.json').exists())
        preflight = list(self.root.glob('dataset_preflight_*/result.json'))
        self.assertEqual(len(preflight), 1)
        self.assertFalse(json.loads(preflight[0].read_text())['passed'])

    def test_duplicate_actual_sequence_is_rejected_before_objective(self):
        self.same_sequence = True
        with self.environment(actual_preflight=True):
            self.assertEqual(self.fit(self.root / 'duplicate'), 2)
        self.assertEqual([c['action'] for c in self.calls], ['validate'] * 3)
        result = json.loads(next(self.root.glob('dataset_preflight_*/result.json')).read_text())
        self.assertFalse(result['passed'])
        self.assertIn('same verified recording', result['failures']['train_b']['error'])

    def test_no_runtime_actual_preflight_is_validation_only(self):
        output = self.root / 'no-runtime'
        with self.environment(actual_preflight=True):
            self.assertEqual(self.run_cli('validate', output, '--no-runtime'), 0)
        self.assertEqual([c['action'] for c in self.calls], ['validate'] * 3)
        self.assertTrue(all(c['no_runtime'] and not c['compute_grad'] for c in self.calls))
        inputs = self.json(output, 'validated_inputs.json')
        self.assertTrue(all(not row['runtime']['initialization_verified'] for row in inputs['episodes'].values()))

    def test_frozen_evaluate_and_split_do_not_recompute_training_gradients(self):
        fit = self.root / 'fit-for-evaluate'
        with self.environment():
            self.assertEqual(self.fit(fit), 0)
        self.calls.clear()
        with self.environment():
            self.assertEqual(self.run_cli('evaluate', self.root / 'evaluate', '--parameters',
                str(fit / 'selected_parameters.json'), '--split', 'validation'), 0)
        self.assertEqual([(c['episode_id'], c['action']) for c in self.calls], [('holdout', 'evaluate')])
        self.assertFalse(self.calls[0]['compute_grad'])

    def test_corrupted_and_mismatched_selection_rejected_before_execution(self):
        fit = self.root / 'fit-for-corruption'
        with self.environment():
            self.assertEqual(self.fit(fit), 0)
        good = self.json(fit, 'selected_parameters.json')
        variants = []
        for key, value in (('schema', 'single-episode'), ('dataset_identity', {}), ('dataset_fingerprint', 'wrong')):
            row = deepcopy(good)
            row[key] = value
            variants.append(json.dumps(row))
        for material in ({'viscosity': 1.0}, dict(good['shared_material_parameters'], tool_retention=1),
                         dict(good['shared_material_parameters'], viscosity=-1.0)):
            row = deepcopy(good)
            row['shared_material_parameters'] = material
            variants.append(json.dumps(row))
        variants.extend(['{broken', '{"schema": "a", "schema": "b"}', '{"x": NaN}'])
        for index, content in enumerate(variants):
            path = self.root / f'corrupt-{index}.json'
            path.write_text(content)
            self.calls.clear()
            with self.subTest(index=index), self.environment(), self.assertRaises((ValueError, EpisodeExecutionError)):
                self.run_cli('evaluate', self.root / f'eval-corrupt-{index}', '--parameters', str(path))
            self.assertEqual(self.calls, [])

    def test_gradient_finite_differences_use_all_training_episodes(self):
        output = self.root / 'gradient'
        with self.environment():
            self.assertEqual(self.run_cli('gradient', output, '--finite-difference'), 0)
        result = self.json(output)
        self.assertTrue(result['finite_difference']['passed'])
        self.assertEqual([c['episode_id'] for c in self.calls], ['train_a', 'train_b'] * 7)
        self.assertTrue(all(not c['compute_grad'] for c in self.calls[2:]))

    def test_failed_independent_evaluation_is_nonzero_without_changing_selection(self):
        self.strict_valid = False
        output = self.root / 'invalid-heldout'
        with self.environment():
            self.assertEqual(self.run_cli('fit', output, '--iterations', '1'), 2)
        self.assertEqual(self.json(output)['status'], 'independent_evaluation_failed')
        self.assertEqual(self.json(output, 'selected_parameters.json')['selection_membership'], 'training')


if __name__ == '__main__':
    unittest.main()

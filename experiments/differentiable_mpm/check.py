"""Run experiment tests in separate processes so Taichi runtimes do not interfere."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .results import EXPERIMENT_ROOT, RUN_ROOT, RunStore, source_identity


STANDARD_TESTS = (
    'test_preservation', 'test_checkpoint', 'test_results', 'test_optimizer',
    'test_inputs', 'test_calibrate_cli', 'test_recompute_check', 'test_backend', 'test_spectral', 'test_loss',
    'test_solver', 'test_trajectory', 'test_forward_parity', 'test_synthetic',
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=STANDARD_TESTS)
    parser.add_argument('--quick', action='store_true', help='Only non-kernel orchestration/input tests')
    parser.add_argument('--regressions', action='store_true', help='Also run the existing lightweight sweep tests')
    parser.add_argument('--reference-policy', choices=['strict', 'frozen'], default='strict',
                        help='Explicit baseline policy for experiment tests; production regressions use live files')
    parser.add_argument('--timeout-s', type=float, default=1800)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.timeout_s <= 0:
        parser.error('--timeout-s must be positive')
    names = list(args.only or (STANDARD_TESTS[:7] if args.quick else STANDARD_TESTS))
    repo = EXPERIMENT_ROOT.parents[1]
    commands = []
    for name in names:
        path = EXPERIMENT_ROOT / 'tests' / (name + '.py')
        if not path.is_file():
            raise FileNotFoundError(f'Test has not been implemented: {path}')
        wrapper = ('import runpy, sys\n'
                   'from experiments.differentiable_mpm.reference_adapter import reference_policy\n'
                   'policy, path = sys.argv[1:3]\n'
                   'sys.argv = [path, *sys.argv[3:]]\n'
                   'with reference_policy(policy):\n'
                   '    runpy.run_path(path, run_name="__main__")\n')
        commands.append((name, [sys.executable, '-c', wrapper, args.reference_policy, str(path), '-v']))
    if args.regressions:
        for name in ('test_episode18_tool_friction_sweep', 'test_episode18_viscosity_sweep', 'test_replay_contact_padding_sweep'):
            path = repo / 'test' / (name + '.py')
            if not path.is_file():
                raise FileNotFoundError(path)
            commands.append((name, [sys.executable, str(path), '-v']))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    output = args.output_dir or RUN_ROOT / ('checks_' + stamp + '_' + uuid.uuid4().hex[:6])
    identity = {'action': 'isolated-test-suite', 'commands': commands, 'source': source_identity(),
                'reference_policy': args.reference_policy,
                'regression_scope': 'current production files, not the preserved experiment baseline'}
    results = []
    with RunStore(output, identity) as store:
        env = dict(os.environ)
        env['PYTHONPATH'] = str(repo) + os.pathsep + str(repo / 'scripts') + os.pathsep + env.get('PYTHONPATH', '')
        scratch = Path(env['CLAUDE_JOB_DIR']) / 'tmp' if env.get('CLAUDE_JOB_DIR') else store.path / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        env['TMPDIR'] = str(scratch)
        for name, command in commands:
            print('Running ' + name, flush=True)
            log = store.path / (name + '.log')
            status = 'failed'
            code = None
            with log.open('x') as stream:
                try:
                    process = subprocess.run(command, cwd=repo, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, timeout=args.timeout_s)
                    code = process.returncode
                    status = 'passed' if code == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    status = 'timeout'
            record = {'test': name, 'status': status, 'returncode': code, 'log': str(log)}
            results.append(record)
            store.append_event(record)
            store.write_json('result.json', {'tests': results, 'complete': False})
            print(f'{name}: {status}', flush=True)
        passed = all(item['status'] == 'passed' for item in results)
        store.write_json('result.json', {'tests': results, 'complete': True, 'passed': passed})
        print(f'Test report: {store.path / "result.json"}', flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Joint shared-material calibration over explicitly selected recorded episodes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

import numpy as np

from .dataset_config import MATERIAL_NAMES, load_dataset
from .dataset_inventory import inventory_draft, scan_dataset
from .multi_episode import (DatasetObjective, EpisodeExecutionError, EpisodeProcessEvaluator,
                            read_json, stable_runtime)
from .optimize import AdamOptions, ProjectedAdam
from .results import RUN_ROOT, RunStore, canonical_hash, source_identity
from .state import P2G_MODES, PHYSICS_VERSIONS

SELECTION_SCHEMA = "taichidough/dataset-material-selection/v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "validate", "gradient", "fit", "replay", "evaluate"))
    parser.add_argument("--dataset", type=Path, help="Explicit shared-material dataset manifest")
    parser.add_argument("--root", action="append", type=Path, default=[], help="Inventory recording root (repeatable)")
    parser.add_argument("--episode-config", action="append", type=Path, default=[], help="Inventory config candidate")
    parser.add_argument("--reconstruction-root", action="append", type=Path, default=[])
    parser.add_argument("--path", action="append", default=[], metavar="EPISODE.INPUT=PATH")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("--precision", choices=("f32", "f64"))
    parser.add_argument("--physics-version", choices=PHYSICS_VERSIONS)
    parser.add_argument("--p2g-mode", choices=P2G_MODES)
    parser.add_argument("--segment-length", type=int)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    parser.add_argument("--ignore-recompute-mismatch", action="store_true")
    parser.add_argument("--worker-timeout-s", type=float, help="Optional per-episode timeout; no timeout by default")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--learning-rate-policy", choices=("persistent-v1", "recover-v1"))
    parser.add_argument("--learning-rate-growth", type=float)
    parser.add_argument("--max-backtracks", type=int)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-runtime", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--finite-difference", action="store_true")
    parser.add_argument("--parameters", type=Path, help="Frozen dataset selected_parameters.json")
    parser.add_argument("--split", choices=("all", "training", "validation"), default="all")
    args = parser.parse_args(argv)
    if args.action == "inventory":
        if not args.root:
            parser.error("inventory requires at least one --root")
    elif args.dataset is None:
        parser.error("This action requires --dataset")
    if args.action != "inventory" and (args.root or args.episode_config or args.reconstruction_root):
        parser.error("Discovery roots/config candidates are only for inventory")
    if args.iterations < 0 or args.cpu_threads < 1:
        parser.error("iterations must be nonnegative and cpu-threads positive")
    if args.segment_length is not None and args.segment_length < 1:
        parser.error("segment-length must be positive")
    if args.worker_timeout_s is not None and (not np.isfinite(args.worker_timeout_s) or args.worker_timeout_s <= 0):
        parser.error("worker-timeout-s must be positive and finite")
    if args.resume and (args.action != "fit" or args.output_dir is None):
        parser.error("resume requires fit and an explicit output-dir")
    if args.no_runtime and args.action != "validate":
        parser.error("no-runtime is only for validate")
    if args.finite_difference and args.action != "gradient":
        parser.error("finite-difference is only for gradient")
    if args.no_evaluate and args.action != "fit":
        parser.error("no-evaluate is only for fit")
    if args.parameters is not None and args.action not in {"gradient", "replay", "evaluate"}:
        parser.error("parameters is only for gradient, replay or evaluate")
    if args.action == "evaluate" and args.parameters is None:
        parser.error("evaluate requires frozen dataset parameters")
    if args.split != "all" and args.action not in {"replay", "evaluate"}:
        parser.error("split selection is only for replay/evaluate; fitting always uses all training episodes")
    return args


def dataset_options(args):
    overrides = {}
    for text in args.path:
        try:
            key, path = text.split("=", 1)
            episode, name = key.rsplit(".", 1)
        except ValueError as error:
            raise ValueError("Path overrides require EPISODE.INPUT=PATH") from error
        if not episode or not name or not path or name in overrides.get(episode, {}):
            raise ValueError("Empty or duplicate episode path override")
        overrides.setdefault(episode, {})[name] = str(Path(path).expanduser().resolve())
    result = {key: getattr(args, key) for key in ("backend", "precision", "p2g_mode", "physics_version", "segment_length")
              if getattr(args, key) is not None}
    if overrides:
        result["path_overrides"] = overrides
    return result


def fresh_name(prefix):
    return prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


def progress(event):
    kind = event.get("event")
    if kind in {"worker_started", "episode_started"}:
        print(f"{kind}: {event['episode_id']} {event.get('action', '')}", flush=True)
        if event.get("directory"):
            print(f"  logs: {event['directory']}", flush=True)
    elif kind == "episode_finished":
        print(f"Episode {event['episode_id']}: loss={event['value']:.9g}; "
              f"weight={event['normalized_weight']:.6g}; elapsed={event['elapsed_s']:.2f}s", flush=True)
    elif kind == "dataset_objective_finished":
        print(f"Dataset loss={event['value']:.9g}; gradient={event['gradient']}", flush=True)
        if not event["replay_consistent"]:
            print("WARNING: this combined gradient includes finite replay mismatches.", flush=True)
    elif kind.endswith("failed"):
        print(f"{kind}: {event.get('error', '')}", file=sys.stderr, flush=True)


def _runner(dataset, args, options, output_root, source, *, event=progress, prepared=None, runtimes=None):
    return EpisodeProcessEvaluator(dataset, args.dataset, options, output_root,
                                   reference_policy=args.reference_policy, cpu_threads=args.cpu_threads,
                                   ignore_recompute_mismatch=args.ignore_recompute_mismatch,
                                   expected_source=source, expected_prepared=prepared,
                                   expected_runtimes=runtimes, timeout_s=args.worker_timeout_s, event=event)


def preflight(dataset, args, options, source):
    """Verify episodes sequentially, retaining summaries rather than numerical arrays."""
    root = RUN_ROOT / fresh_name("dataset_preflight")
    identity = {"action": "dataset-preflight", "dataset": dataset.as_dict(), "source": source,
                "reference_policy": args.reference_policy, "cpu_threads": args.cpu_threads,
                "ignore_recompute_mismatch": args.ignore_recompute_mismatch, "no_runtime": args.no_runtime}
    summaries = {}
    failures = {}
    sequences = {}
    with RunStore(root, identity) as store:
        def event(record):
            store.append_event(record)
            progress(record)
        runner = _runner(dataset, args, options, store.path / "episodes", source, event=event)
        for ep in dataset.episodes:
            try:
                record = runner.run(ep, "validate", dataset.shared_initial, no_runtime=args.no_runtime)
                summary = record.get("prepared_summary")
                if not isinstance(summary, dict) or not isinstance(summary.get("provenance"), dict):
                    raise EpisodeExecutionError("Preflight did not return prepared provenance")
                sequence = summary["provenance"].get("sequence_fingerprint")
                if not isinstance(sequence, str) or len(sequence) != 64:
                    raise EpisodeExecutionError("Preflight did not verify sequence fingerprint")
                if sequence in sequences:
                    raise ValueError(f"Episodes {sequences[sequence]} and {ep.id} use the same verified recording")
                sequences[sequence] = ep.id
                summaries[ep.id] = {"prepared_fingerprint": record["prepared_fingerprint"],
                                    "runtime": record["runtime"], "summary": summary}
            except Exception as error:
                failures[ep.id] = {"error_type": type(error).__name__, "error": str(error)}
                store.append_event({"event": "episode_preflight_failed", "episode_id": ep.id, **failures[ep.id]})
            store.write_json("result.json", {"complete": False, "episodes": summaries, "failures": failures})
        passed = not failures and len(summaries) == len(dataset.episodes)
        store.write_json("result.json", {"complete": True, "passed": passed, "episodes": summaries,
                                         "failures": failures, "no_runtime": args.no_runtime})
    return summaries, failures, root


def _dataset_identity(dataset, args, summaries, source):
    from .reference_adapter import reference_identity, reference_policy
    version = dataset.episodes[0].config.simulation.get("physics_version", "corrected-v1")
    with reference_policy(args.reference_policy):
        physics_reference = reference_identity(version)
    return {"schema": "taichidough/validated-dataset/v1", "dataset": dataset.as_dict(),
            "physics_version": version, "physics_reference": physics_reference,
            "dataset_fingerprint": dataset.fingerprint, "source": source,
            "reference_policy": args.reference_policy, "cpu_threads": args.cpu_threads,
            "ignore_recompute_mismatch": args.ignore_recompute_mismatch,
            "episodes": [{"episode_id": ep.id,
                          "prepared_fingerprint": summaries[ep.id]["prepared_fingerprint"],
                          "runtime": stable_runtime(summaries[ep.id]["runtime"])} for ep in dataset.episodes]}


def load_selection(path, dataset, dataset_identity):
    record = read_json(path)
    if not isinstance(record, dict) or record.get("schema") != SELECTION_SCHEMA:
        raise ValueError("Expected a dataset material selection, not a single-episode parameter file")
    if record.get("dataset_identity") != dataset_identity:
        raise ValueError("Selected material belongs to different episode inputs, model or runtime settings")
    if record.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Selected material belongs to another dataset configuration")
    shared = record.get("shared_material_parameters")
    if not isinstance(shared, dict) or set(shared) != set(MATERIAL_NAMES):
        raise ValueError("Selected material must supply exactly the five shared physical values")
    for ep in dataset.episodes:
        ep.parameters_for(shared)
    return dict(shared), {"path": str(Path(path).resolve()), "content_sha256": canonical_hash(record)}


def _evaluate_selected(dataset, runner, parameters, split, *, strict=True):
    episodes = [ep for ep in dataset.episodes if split == "all" or ep.membership == split]
    if not episodes:
        raise ValueError(f"Dataset has no {split} episodes")
    results = {}
    for ep in episodes:
        response = runner.run(ep, "evaluate" if strict else "replay", parameters)
        exported = response.get("export")
        if not isinstance(exported, dict):
            raise EpisodeExecutionError("Worker did not return replay/evaluation records")
        results[ep.id] = {"membership": ep.membership, **exported}
    passed = (not strict or all(row.get("strict_evaluation", {}).get("valid") is True for row in results.values()))
    return results, passed


def finite_difference_report(objective, dataset, shared, gradient):
    from .parameters import PhysicalParameterSpace
    base = dataset.episodes[0].parameters_for(shared)
    space = PhysicalParameterSpace(base, dataset.fit_parameters,
                                   dataset.episodes[0].config.simulation.get("plasticity", "none"),
                                   bounds=dataset.parameter_bounds)
    u = space.coordinates()
    ad = space.pullback(u, gradient)
    rows = []
    for i, name in enumerate(space.fit):
        checks = []
        for step in (1e-3, 3e-4, 1e-4):
            plus, minus = u.copy(), u.copy()
            plus[i] += step
            minus[i] -= step
            plus, minus = space.project(plus), space.project(minus)
            span = plus[i] - minus[i]
            if span <= 0:
                raise ValueError("Finite difference has no interval within parameter bounds")
            a = objective.value_and_gradient(space.physical(plus), compute_grad=False).value
            b = objective.value_and_gradient(space.physical(minus), compute_grad=False).value
            fd = (a - b) / span
            relative = abs(fd - ad[i]) / max(abs(fd), abs(ad[i]), 1e-8)
            checks.append({"step": step, "coordinate_span": float(span), "finite_difference": float(fd),
                           "relative_error": float(relative),
                           "scheme": "central" if np.isclose(plus[i] + minus[i], 2 * u[i], rtol=0, atol=1e-14) else "one-sided at bound"})
        rows.append({"parameter": name, "coordinate_gradient": float(ad[i]), "checks": checks,
                     "passed": any(row["relative_error"] < 1e-2 for row in checks)})
    return {"rows": rows, "passed": all(row["passed"] for row in rows),
            "note": "Every perturbation evaluates all training episodes; piecewise contact may switch."}


def run(args):
    source = source_identity()
    output = args.output_dir or RUN_ROOT / fresh_name("dataset_" + args.action)
    if args.action == "inventory":
        identity = {"action": "dataset-inventory", "roots": [str(p.resolve()) for p in args.root],
                    "config_paths": [str(p.resolve()) for p in args.episode_config],
                    "reconstruction_roots": [str(p.resolve()) for p in args.reconstruction_root], "source": source}
        with RunStore(output, identity) as store:
            report = scan_dataset(args.root, config_paths=args.episode_config,
                                  reconstruction_roots=args.reconstruction_root)
            store.write_json("inventory.json", report)
            store.write_json("draft.json", inventory_draft(report))
            store.write_json("result.json", {"status": "inventory_complete", "counts": report["counts"],
                                             "errors": report["errors"], "fully_validated": False})
            print(f"Inventory: {store.path / 'inventory.json'}", flush=True)
        return 0 if not report["errors"] else 2
    options = dataset_options(args)
    dataset = load_dataset(args.dataset, **options)
    print(f"Dataset {dataset.name}: {len(dataset.episodes)} episodes; shared material={dataset.fit_parameters}", flush=True)
    print("Execution: one subprocess at a time; input preparation and compilation repeat per episode/candidate.", flush=True)
    if args.ignore_recompute_mismatch:
        print("WARNING: finite replay mismatches are allowed; resulting gradients may be approximate.", flush=True)
    optimizer_options = dict(dataset.episodes[0].config.optimizer)
    for key in ("learning_rate", "learning_rate_policy", "learning_rate_growth", "max_backtracks", "max_evaluations"):
        if getattr(args, key) is not None:
            optimizer_options[key] = getattr(args, key)
    adam_options = AdamOptions(**optimizer_options)
    summaries, failures, preflight_root = preflight(dataset, args, options, source)
    if failures:
        print(f"Dataset validation failed; no fit started. Details: {preflight_root / 'result.json'}", file=sys.stderr, flush=True)
        return 2
    verified = _dataset_identity(dataset, args, summaries, source)
    identity = {"action": "dataset-" + args.action, "dataset_identity": verified}
    if args.action == "fit":
        identity["optimizer"] = asdict(adam_options)
    shared = dict(dataset.shared_initial)
    if args.parameters is not None:
        shared, selection_record = load_selection(args.parameters, dataset, verified)
        identity["selection"] = selection_record
    if args.action in {"replay", "evaluate"}:
        identity["evaluation_split"] = args.split
    with RunStore(output, identity, resume=args.resume) as store:
        store.write_json("validated_inputs.json", {"preflight_directory": str(preflight_root), "episodes": summaries})
        def event(record):
            store.append_event(record)
            progress(record)
        runner = _runner(dataset, args, options, store.path / "episodes", source, event=event,
                         prepared={key: value["prepared_fingerprint"] for key, value in summaries.items()},
                         runtimes={key: stable_runtime(value["runtime"]) for key, value in summaries.items()})
        objective = DatasetObjective(dataset, runner, event=event)
        if args.action == "validate":
            store.write_json("result.json", {"status": "validated", "no_runtime": args.no_runtime,
                                             "episodes": list(summaries), "dataset_identity": verified})
            print(f"Validation: {store.path / 'result.json'}", flush=True)
            return 0
        if args.action == "gradient":
            evaluation = objective(shared)
            result = {"status": "gradient_complete", "evaluation": asdict(evaluation), "shared_material_parameters": shared}
            if args.finite_difference:
                result["finite_difference"] = finite_difference_report(objective, dataset, shared, evaluation.gradient)
                if not result["finite_difference"]["passed"]:
                    result["status"] = "gradient_check_failed"
            store.write_json("result.json", result)
            return 2 if result["status"] == "gradient_check_failed" else 0
        if args.action in {"replay", "evaluate"}:
            evaluations, passed = _evaluate_selected(dataset, runner, shared, args.split, strict=args.action == "evaluate")
            store.write_json("result.json", {"status": "evaluation_complete" if passed else "evaluation_failed",
                                             "evaluations": evaluations, "shared_material_parameters": shared})
            return 0 if passed else 2
        def callback(event_record, state):
            store.save_optimizer(event_record, state)
            print("Optimizer: " + json.dumps(event_record, sort_keys=True), flush=True)
        optimizer = ProjectedAdam(dataset.parameter_space(), objective, adam_options, callback, objective_id=store.identity_hash)
        if args.resume:
            optimizer.load_state_dict(store.optimizer_state())
        try:
            optimization = optimizer.run(args.iterations)
        except BaseException as error:
            store.write_json("result.json", {"status": optimizer.status if optimizer.status == "invalid_initial" else "fit_failed",
                                             "error_type": type(error).__name__, "error": str(error),
                                             "accepted_updates": optimizer.accepted_updates,
                                             "evaluations": optimizer.evaluations})
            raise
        selected = {name: optimization.best_parameters[name] for name in MATERIAL_NAMES}
        selections = {ep.id: {"membership": ep.membership, "weight": ep.weight,
                               "scored_frames": list(ep.scored_window.indices()),
                               "effective_parameters": ep.parameters_for(selected)} for ep in dataset.episodes}
        has_holdout = any(ep.membership == "validation" for ep in dataset.episodes)
        selection = {"schema": SELECTION_SCHEMA, "shared_material_parameters": selected,
                     "fitted_parameters": list(dataset.fit_parameters), "training_value": optimization.best_value,
                     "selection_membership": "training", "dataset_fingerprint": dataset.fingerprint,
                     "dataset_identity": verified, "identity_sha256": store.identity_hash,
                     "physics_version": verified["physics_version"], "physics_reference": verified["physics_reference"],
                     "episodes": selections, "independent_heldout_episodes_declared": has_holdout,
                     "ignore_recompute_mismatch": args.ignore_recompute_mismatch,
                     "note": "Shared material values only; contacts and geometry remain episode-specific. No parameter conversion between physics versions."}
        store.write_json("selected_parameters.json", selection)
        result = {"status": optimization.status, "optimization": asdict(optimization),
                  "shared_material_parameters": selected, "evaluations": {},
                  "independent_heldout_episodes_declared": has_holdout,
                  "successful_objectives_this_process": objective.successful_calls,
                  "objectives_with_replay_mismatch": objective.calls_with_mismatches,
                  "backward_replay_consistent": objective.successful_calls > 0 and objective.calls_with_mismatches == 0}
        store.write_json("result.json", result)
        passed = True
        if not args.no_evaluate:
            evaluations, passed = _evaluate_selected(dataset, runner, selected, "all")
            result["evaluations"] = evaluations
            if not passed:
                result["status"] = "independent_evaluation_failed"
        result["independent_evaluation_skipped"] = args.no_evaluate
        store.write_json("result.json", result)
        print(f"Dataset result: {store.path / 'result.json'}", flush=True)
        return 2 if not passed or optimization.status in {"invalid_initial", "stalled_invalid", "stalled_descent"} else 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

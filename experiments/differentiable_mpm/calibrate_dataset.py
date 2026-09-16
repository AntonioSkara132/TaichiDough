"""Joint shared-material calibration over explicitly selected recorded episodes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
import sys
import uuid

import numpy as np

from .dataset_config import MATERIAL_NAMES, SCHEMA_V2, load_dataset
from .dataset_inventory import inventory_draft, scan_dataset
from .loss_options import is_paper_loss, loss_temporal_reduction, parse_loss_config
from .multi_episode import (DatasetObjective, EpisodeExecutionError, EpisodeProcessEvaluator,
                            normalized_weights, read_json, stable_runtime)
from .optimize import AdamOptions, ObjectiveValue, ProjectedAdam
from .results import RUN_ROOT, RunStore, canonical_hash, source_identity
from .run_logging import (format_optimizer_boundary, format_optimizer_evaluation,
                          stability_settings)
from .state import InvalidStateError, P2G_MODES, PHYSICS_VERSIONS

SELECTION_SCHEMA = "taichidough/dataset-material-selection/v1"
SELECTION_SCHEMA_V2 = "taichidough/dataset-physical-selection/v2"


def episode_selection_record(episode, parameters):
    record = {"membership": episode.membership, "weight": episode.weight,
              "scored_frames": list(episode.scored_window.indices()),
              "effective_parameters": episode.parameters_for(parameters)}
    loss_config = parse_loss_config(episode.config.loss)
    if is_paper_loss(loss_config):
        objective_frames = ([episode.scored_window.end_frame] if loss_config.version.startswith("dpsi-")
                            else list(episode.scored_window.indices()))
        record.update(objective_frames=objective_frames, loss=loss_config.as_dict(),
                      observation_reduction=loss_temporal_reduction(loss_config))
    return record


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
    parser.add_argument("--episode-batch-size", type=int, default=0,
                        help="Fit with independent training-episode minibatches of this size; also limits parallel workers. "
                             "Default 0 keeps the exact full training objective.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--learning-rate-policy", choices=("persistent-v1", "recover-v1"))
    parser.add_argument("--learning-rate-growth", type=float)
    parser.add_argument("--max-backtracks", type=int)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--parameter-stability-updates", type=int)
    parser.add_argument("--parameter-stability-rtol", type=float)
    parser.add_argument("--parameter-stability-atol", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--initial-youngs-modulus", type=float)
    parser.add_argument("--initial-viscosity", type=float)
    parser.add_argument("--fit-log", choices=("concise", "detailed", "quiet"), default="concise")
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
    if args.episode_batch_size < 0:
        parser.error("episode-batch-size must be nonnegative")
    if args.parameter_stability_updates is not None and args.parameter_stability_updates < 0:
        parser.error("parameter-stability-updates must be nonnegative")
    if args.parameter_stability_rtol is not None and (not np.isfinite(args.parameter_stability_rtol)
                                                       or args.parameter_stability_rtol < 0):
        parser.error("parameter-stability-rtol must be finite and nonnegative")
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
    if args.episode_batch_size and args.action != "fit":
        parser.error("episode-batch-size is only for fit")
    if args.episode_batch_size and args.resume:
        parser.error("resume is not supported for minibatch dataset fits")
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
    initial_overrides = {}
    if args.initial_youngs_modulus is not None:
        initial_overrides["youngs_modulus"] = args.initial_youngs_modulus
    if args.initial_viscosity is not None:
        initial_overrides["viscosity"] = args.initial_viscosity
    if initial_overrides:
        result["initial_overrides"] = initial_overrides
    return result


def fresh_name(prefix):
    return prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


def progress(event, mode="detailed"):
    kind = event.get("event", "")
    if kind.endswith("failed"):
        print(f"{kind}: {event.get('error', '')}", file=sys.stderr, flush=True)
    elif kind == "dataset_objective_finished" and not event["replay_consistent"]:
        print("WARNING: this combined gradient includes finite replay mismatches.", flush=True)
    elif mode != "detailed":
        return
    elif kind in {"worker_started", "episode_started"}:
        print(f"{kind}: {event['episode_id']} {event.get('action', '')}", flush=True)
        if event.get("directory"):
            print(f"  logs: {event['directory']}", flush=True)
    elif kind == "episode_finished":
        print(f"Episode {event['episode_id']}: loss={event['value']:.9g}; "
              f"weight={event['normalized_weight']:.6g}; elapsed={event['elapsed_s']:.2f}s", flush=True)
    elif kind == "dataset_objective_finished":
        print(f"Dataset loss={event['value']:.9g}; gradient={event['gradient']}", flush=True)


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
            progress(record, args.fit_log if args.action == "fit" else "detailed")
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
    expected_schema = SELECTION_SCHEMA_V2 if dataset.schema == SCHEMA_V2 else SELECTION_SCHEMA
    if not isinstance(record, dict) or record.get("schema") != expected_schema:
        raise ValueError("Expected a compatible dataset parameter selection")
    if record.get("dataset_identity") != dataset_identity:
        raise ValueError("Selected material belongs to different episode inputs, model or runtime settings")
    if record.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Selected material belongs to another dataset configuration")
    parameter_key = "shared_physical_parameters" if dataset.schema == SCHEMA_V2 else "shared_material_parameters"
    shared = record.get(parameter_key)
    if not isinstance(shared, dict) or set(shared) != set(dataset.shared_parameter_names):
        raise ValueError("Selected parameters must supply exactly the dataset shared physical values")
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


def _episode_batches(episodes, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [tuple(episodes[index:index + batch_size]) for index in range(0, len(episodes), batch_size)]


def _evaluate_episode_batch(dataset, episodes, parameters, *, compute_grad, runner_factory,
                            max_workers, event):
    """Evaluate independent episodes concurrently and combine them as one batch."""
    if not episodes:
        raise ValueError("Cannot evaluate an empty episode batch")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    shared = {}
    for name in dataset.shared_parameter_names:
        if name not in parameters:
            raise ValueError(f"Missing shared parameter {name}")
        value = float(parameters[name])
        if not math.isfinite(value):
            raise InvalidStateError(f"Shared parameter {name} is not finite")
        shared[name] = value
    weights = normalized_weights(episodes)
    started = datetime.now(timezone.utc)
    rows = []
    mismatch_counts = {}
    replay_consistent = True
    derivatives = {name: [] for name in dataset.fit_parameters}
    contributions = []

    def run_one(ep, weight):
        runner = runner_factory()
        response = runner.run(ep, "objective", shared, compute_grad=compute_grad)
        record = response.get("evaluation")
        if not isinstance(record, dict):
            raise EpisodeExecutionError(f"Episode {ep.id} did not return an evaluation")
        value = float(record["value"])
        if not math.isfinite(value):
            raise InvalidStateError(f"Episode {ep.id} returned a nonfinite loss")
        gradient = {}
        if compute_grad:
            supplied = record.get("gradient")
            if not isinstance(supplied, dict):
                raise EpisodeExecutionError(f"Episode {ep.id} did not return gradients")
            for name in dataset.fit_parameters:
                gradient[name] = float(supplied[name])
                if not math.isfinite(gradient[name]):
                    raise InvalidStateError(f"Episode {ep.id} returned a nonfinite {name} gradient")
        diagnostics = record.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            raise EpisodeExecutionError(f"Episode {ep.id} returned invalid diagnostics")
        return ep, float(weight), value, gradient, diagnostics

    event({"event": "episode_batch_started", "episode_ids": [ep.id for ep in episodes],
           "compute_grad": compute_grad, "max_workers": max_workers})
    with ThreadPoolExecutor(max_workers=min(max_workers, len(episodes))) as executor:
        futures = [executor.submit(run_one, ep, weight) for ep, weight in zip(episodes, weights)]
        for future in as_completed(futures):
            ep, weight, value, gradient, diagnostics = future.result()
            weighted = weight * value
            contributions.append(weighted)
            consistent = diagnostics.get("replay_consistent", True)
            if type(consistent) is not bool:
                raise EpisodeExecutionError("replay_consistent must be boolean")
            replay_consistent = replay_consistent and consistent
            counts = diagnostics.get("recompute_mismatch_counts", {})
            if not isinstance(counts, dict):
                raise EpisodeExecutionError("Mismatch counts must be a mapping")
            for name, count in counts.items():
                if type(count) is not int or count < 0:
                    raise EpisodeExecutionError("Mismatch counts must be nonnegative integers")
                mismatch_counts[name] = mismatch_counts.get(name, 0) + count
            if compute_grad:
                for name in dataset.fit_parameters:
                    derivatives[name].append(weight * gradient[name])
            row = {"episode_id": ep.id, "membership": ep.membership, "value": value,
                   "normalized_weight": weight, "weighted_value": weighted,
                   "gradient": gradient, "diagnostics": diagnostics}
            rows.append(row)
            event({"event": "episode_batch_member_finished", **row})
    order = {ep.id: index for index, ep in enumerate(episodes)}
    rows.sort(key=lambda row: order[row["episode_id"]])
    value = math.fsum(contributions)
    gradient = {name: math.fsum(values) for name, values in derivatives.items()} if compute_grad else {}
    if not math.isfinite(value) or not all(math.isfinite(v) for v in gradient.values()):
        raise InvalidStateError("Combined batch loss or gradient is nonfinite")
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    diagnostics = {"objective_version": "independent-episode-minibatch-v1",
                   "episodes": rows, "batch_episode_count": len(rows),
                   "elapsed_s": elapsed, "replay_consistency_checked": compute_grad,
                   "replay_consistent": replay_consistent,
                   "recompute_mismatch_counts": mismatch_counts,
                   "recompute_mismatch_count": sum(mismatch_counts.values())}
    event({"event": "episode_batch_finished", "episode_ids": [ep.id for ep in episodes],
           "value": value, "gradient": gradient, "replay_consistent": replay_consistent,
           "elapsed_s": elapsed})
    return ObjectiveValue(value, gradient, diagnostics)


def _evaluate_all_training_batches(dataset, batches, shared, *, runner_factory, max_workers, event):
    rows = []
    weighted_values = []
    training = tuple(ep for batch in batches for ep in batch)
    weights = dict(zip([ep.id for ep in training], normalized_weights(training)))
    for batch in batches:
        result = _evaluate_episode_batch(dataset, batch, shared, compute_grad=False,
                                         runner_factory=runner_factory, max_workers=max_workers,
                                         event=event)
        for row in result.diagnostics["episodes"]:
            global_weight = weights[row["episode_id"]]
            row = dict(row)
            row["dataset_normalized_weight"] = global_weight
            row["dataset_weighted_value"] = global_weight * row["value"]
            rows.append(row)
            weighted_values.append(row["dataset_weighted_value"])
    return {"value": math.fsum(weighted_values), "episodes": rows}


def _run_minibatch_fit(dataset, args, adam_options, store, runner_factory, event):
    """Projected Adam updates after each independent episode batch."""
    training = [ep for ep in dataset.episodes if ep.membership == "training"]
    if not training:
        raise ValueError("Dataset has no training episodes")
    batch_size = min(args.episode_batch_size, len(training))
    batches = _episode_batches(training, batch_size)
    space = dataset.parameter_space()
    u = space.coordinates()
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    learning_rate = adam_options.learning_rate
    best_u = u.copy()
    best_value = math.inf
    history = []
    successful = 0
    mismatches = 0
    for iteration in range(1, args.iterations + 1):
        batch = batches[(iteration - 1) % len(batches)]
        shared = space.physical(u)
        result = _evaluate_episode_batch(dataset, batch, shared, compute_grad=True,
                                         runner_factory=runner_factory, max_workers=batch_size,
                                         event=event)
        successful += 1
        mismatches += int(not result.diagnostics.get("replay_consistent", True))
        gradient = space.pullback(u, result.gradient)
        if not np.isfinite(gradient).all():
            raise InvalidStateError("Batch coordinate gradient is not finite")
        b1, b2 = adam_options.beta1, adam_options.beta2
        m = b1 * m + (1.0 - b1) * gradient
        v = b2 * v + (1.0 - b2) * gradient * gradient
        m_hat = m / (1.0 - b1 ** iteration)
        v_hat = v / (1.0 - b2 ** iteration)
        step = -learning_rate * m_hat / (np.sqrt(v_hat) + adam_options.epsilon)
        old_u = u.copy()
        u = space.project(u + step)
        step_norm = float(np.linalg.norm(u - old_u, ord=np.inf))
        record = {"type": "minibatch_step", "iteration": iteration,
                  "batch_index": (iteration - 1) % len(batches),
                  "episode_ids": [ep.id for ep in batch],
                  "value_before_update": result.value,
                  "physical_gradient": dict(result.gradient),
                  "coordinate_gradient": gradient.tolist(),
                  "gradient_norm": float(np.linalg.norm(gradient)),
                  "step_norm": step_norm,
                  "learning_rate": learning_rate,
                  "parameters_before": shared,
                  "parameters_after": space.physical(u),
                  "diagnostics": result.diagnostics}
        history.append(record)
        store.append_event({"event": "optimizer_minibatch_step", **record})
        store.write_json("optimizer_state.json", {"schema": "taichidough/minibatch-adam-state/v1",
                                                  "identity_sha256": store.identity_hash,
                                                  "state": {"iterations": iteration,
                                                            "accepted_updates": iteration,
                                                            "coordinates": u.tolist(),
                                                            "first_moment": m.tolist(),
                                                            "second_moment": v.tolist(),
                                                            "physical_parameters": space.physical(u),
                                                            "history": history}})
        print(f"batch update {iteration}: loss={result.value:.9g}; episodes={','.join(ep.id for ep in batch)}; "
              f"step={step_norm:.4g}; params={space.physical(u)}", flush=True)
    final = _evaluate_all_training_batches(dataset, batches, space.physical(u),
                                           runner_factory=runner_factory, max_workers=batch_size,
                                           event=event)
    best_u = u.copy()
    best_value = final["value"]
    return {"best_parameters": space.physical(best_u), "best_value": best_value,
            "current_parameters": space.physical(u), "current_value": best_value,
            "status": "budget_exhausted", "iterations": args.iterations,
            "accepted_updates": args.iterations,
            "evaluations": successful, "history": history,
            "final_training_evaluation": final,
            "successful_objectives": successful,
            "objectives_with_replay_mismatch": mismatches,
            "batch_size": batch_size}


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
    parameter_key = "shared_physical_parameters" if dataset.schema == SCHEMA_V2 else "shared_material_parameters"
    if args.action != "fit" or args.fit_log == "detailed":
        print(f"Dataset {dataset.name}: {len(dataset.episodes)} episodes; shared material={dataset.fit_parameters}", flush=True)
        print("Execution: one subprocess at a time; input preparation and compilation repeat per episode/candidate.", flush=True)
    if args.ignore_recompute_mismatch:
        print("WARNING: finite replay mismatches are allowed; resulting gradients may be approximate.", flush=True)
    optimizer_options = dict(dataset.episodes[0].config.optimizer)
    for key in ("learning_rate", "learning_rate_policy", "learning_rate_growth", "max_backtracks", "max_evaluations"):
        if getattr(args, key) is not None:
            optimizer_options[key] = getattr(args, key)
    optimizer_options = stability_settings(
        optimizer_options, dataset.fit_parameters,
        updates=args.parameter_stability_updates,
        rtol=args.parameter_stability_rtol,
        atol_items=args.parameter_stability_atol,
    )
    adam_options = AdamOptions(**optimizer_options)
    summaries, failures, preflight_root = preflight(dataset, args, options, source)
    if failures:
        print(f"Dataset validation failed; no fit started. Details: {preflight_root / 'result.json'}", file=sys.stderr, flush=True)
        return 2
    verified = _dataset_identity(dataset, args, summaries, source)
    identity = {"action": "dataset-" + args.action, "dataset_identity": verified}
    if args.action == "fit":
        identity["optimizer"] = asdict(adam_options)
        if args.episode_batch_size:
            identity["optimizer"]["episode_batch_size"] = args.episode_batch_size
            identity["optimizer"]["objective"] = "independent-episode-minibatch-v1"
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
            progress(record, args.fit_log if args.action == "fit" else "detailed")
        runner = _runner(dataset, args, options, store.path / "episodes", source, event=event,
                         prepared={key: value["prepared_fingerprint"] for key, value in summaries.items()},
                         runtimes={key: stable_runtime(value["runtime"]) for key, value in summaries.items()})
        def runner_factory():
            return _runner(dataset, args, options, store.path / "episodes", source, event=event,
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
            result = {"status": "gradient_complete", "evaluation": asdict(evaluation), parameter_key: shared}
            if args.finite_difference:
                result["finite_difference"] = finite_difference_report(objective, dataset, shared, evaluation.gradient)
                if not result["finite_difference"]["passed"]:
                    result["status"] = "gradient_check_failed"
            store.write_json("result.json", result)
            return 2 if result["status"] == "gradient_check_failed" else 0
        if args.action in {"replay", "evaluate"}:
            evaluations, passed = _evaluate_selected(dataset, runner, shared, args.split, strict=args.action == "evaluate")
            store.write_json("result.json", {"status": "evaluation_complete" if passed else "evaluation_failed",
                                             "evaluations": evaluations, parameter_key: shared})
            return 0 if passed else 2
        fit_names = dataset.fit_parameters
        if args.episode_batch_size:
            optimization = _run_minibatch_fit(dataset, args, adam_options, store, runner_factory, event)
            selected = {name: optimization["best_parameters"][name] for name in dataset.shared_parameter_names}
            selections = {ep.id: episode_selection_record(ep, selected) for ep in dataset.episodes}
            has_holdout = any(ep.membership == "validation" for ep in dataset.episodes)
            selection_key = "shared_physical_parameters" if dataset.schema == SCHEMA_V2 else "shared_material_parameters"
            selection_schema = SELECTION_SCHEMA_V2 if dataset.schema == SCHEMA_V2 else SELECTION_SCHEMA
            selection = {"schema": selection_schema, selection_key: selected,
                         "fitted_parameters": list(dataset.fit_parameters), "training_value": optimization["best_value"],
                         "selection_membership": "training", "dataset_fingerprint": dataset.fingerprint,
                         "dataset_identity": verified, "identity_sha256": store.identity_hash,
                         "physics_version": verified["physics_version"], "physics_reference": verified["physics_reference"],
                         "episodes": selections, "independent_heldout_episodes_declared": has_holdout,
                         "ignore_recompute_mismatch": args.ignore_recompute_mismatch,
                         "objective": {"version": "independent-episode-minibatch-v1",
                                       "episode_batch_size": optimization["batch_size"],
                                       "final_training_evaluation": optimization["final_training_evaluation"]},
                         "note": ("Shared material and selected contact values; each chunk is treated as an independent relaxed-state episode. "
                                  "No parameter conversion between physics versions." if dataset.schema == SCHEMA_V2 else
                                  "Shared material values only; each chunk is treated as an independent relaxed-state episode. "
                                  "No parameter conversion between physics versions.")}
            store.write_json("selected_parameters.json", selection)
            result = {"status": optimization["status"], "optimization": optimization,
                      selection_key: selected, "evaluations": {},
                      "independent_heldout_episodes_declared": has_holdout,
                      "successful_objectives_this_process": optimization["successful_objectives"],
                      "objectives_with_replay_mismatch": optimization["objectives_with_replay_mismatch"],
                      "backward_replay_consistent": (optimization["successful_objectives"] > 0
                                                     and optimization["objectives_with_replay_mismatch"] == 0),
                      "objective": {"version": "independent-episode-minibatch-v1",
                                    "episode_batch_size": optimization["batch_size"]}}
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
            return 2 if not passed or optimization["status"] in {"invalid_initial", "stalled_invalid", "stalled_descent"} else 0

        def evaluation_observer(event_record):
            store.append_event({"event": "optimizer_" + event_record["type"], **event_record})
            message = format_optimizer_evaluation(event_record, fit_names)
            if args.fit_log != "quiet" or event_record.get("valid") is False:
                print(message, file=sys.stderr if event_record.get("valid") is False else sys.stdout, flush=True)
        def callback(event_record, state):
            store.save_optimizer(event_record, state)
            message = format_optimizer_boundary(event_record, state, fit_names)
            terminal = event_record.get("status") in ProjectedAdam._TERMINAL
            if message is not None and (args.fit_log != "quiet" or event_record.get("type") == "status" or terminal):
                print(message, flush=True)
        optimizer = ProjectedAdam(dataset.parameter_space(), objective, adam_options, callback,
                                  objective_id=store.identity_hash,
                                  evaluation_observer=evaluation_observer)
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
        selected = {name: optimization.best_parameters[name] for name in dataset.shared_parameter_names}
        selections = {ep.id: episode_selection_record(ep, selected) for ep in dataset.episodes}
        has_holdout = any(ep.membership == "validation" for ep in dataset.episodes)
        selection_key = "shared_physical_parameters" if dataset.schema == SCHEMA_V2 else "shared_material_parameters"
        selection_schema = SELECTION_SCHEMA_V2 if dataset.schema == SCHEMA_V2 else SELECTION_SCHEMA
        selection = {"schema": selection_schema, selection_key: selected,
                     "fitted_parameters": list(dataset.fit_parameters), "training_value": optimization.best_value,
                     "selection_membership": "training", "dataset_fingerprint": dataset.fingerprint,
                     "dataset_identity": verified, "identity_sha256": store.identity_hash,
                     "physics_version": verified["physics_version"], "physics_reference": verified["physics_reference"],
                     "episodes": selections, "independent_heldout_episodes_declared": has_holdout,
                     "ignore_recompute_mismatch": args.ignore_recompute_mismatch,
                     "note": ("Shared material and selected contact values; remaining contacts and geometry stay episode-specific. "
                              "No parameter conversion between physics versions." if dataset.schema == SCHEMA_V2 else
                              "Shared material values only; contacts and geometry remain episode-specific. No parameter conversion between physics versions.")}
        store.write_json("selected_parameters.json", selection)
        result = {"status": optimization.status, "optimization": asdict(optimization),
                  selection_key: selected, "evaluations": {},
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

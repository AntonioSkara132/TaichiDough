"""Continue an accepted independent-episode minibatch Adam fit without modifying its source tree.

Run this trusted script from outside experiments/differentiable_mpm: the experiment's
source identity includes every Python file below that directory. The original run
must have finished its independent evaluations before a continuation is created.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import uuid
import warnings

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.differentiable_mpm import calibrate_dataset as calibration  # noqa: E402
from experiments.differentiable_mpm.dataset_config import SCHEMA_V2, load_dataset  # noqa: E402
from experiments.differentiable_mpm.multi_episode import read_json, stable_runtime  # noqa: E402
from experiments.differentiable_mpm.optimize import AdamOptions  # noqa: E402
from experiments.differentiable_mpm.results import RunStore, canonical_hash, source_identity  # noqa: E402
from experiments.differentiable_mpm.run_logging import number, parameter_text  # noqa: E402
from experiments.differentiable_mpm.state import InvalidStateError  # noqa: E402

CHECKPOINT_SCHEMA = "taichidough/minibatch-adam-state/v1"
CONTINUATION_SCHEMA = "taichidough/minibatch-adam-continuation/v1"


def _partial_evaluation_inventory(original_run, identity, checkpoint):
    expected = {ep["episode_id"] for ep in identity["dataset_identity"]["episodes"]}
    source = identity["dataset_identity"]["source"]
    original_dataset = identity["dataset_identity"]["dataset"]
    shared_names = set(original_dataset["shared_parameters"]["initial"])
    fit_names = set(original_dataset["shared_parameters"]["fit"])
    selected = checkpoint["state"]["physical_parameters"]
    if (shared_names != fit_names or set(selected) != shared_names | {"tool_retention"}
            or selected["tool_retention"] != 1.0
            or any(ep["config"]["parameters"].get("tool_retention") != 1.0
                   for ep in original_dataset["episodes"])):
        raise ValueError("Original fit and fixed tool retention differ from selected physical parameters")
    rows = []
    for request_path in sorted((original_run / "episodes").glob("*/request.json")):
        request = read_json(request_path)
        if request.get("action") != "evaluate":
            continue
        result_path = request_path.parent / "worker" / "result.json"
        if not result_path.is_file():
            continue  # The interrupted worker never completed a strict evaluation.
        output = read_json(result_path)
        episode_id = request.get("episode_id")
        if (episode_id not in expected or output.get("episode_id") != episode_id
                or request.get("expected_source") != source or output.get("source") != source
                or output.get("source_after") != source or output.get("status") != "ok"
                or output.get("request_id") != request.get("request_id")
                or output.get("request_sha256") != canonical_hash(request)
                or not isinstance(request.get("parameters"), dict)
                or set(request["parameters"]) != shared_names
                or request["parameters"] != {name: selected[name] for name in shared_names}
                or output.get("export", {}).get("strict_evaluation", {}).get("valid") is not True):
            raise ValueError(f"Interrupted evaluation has invalid or mismatched completed result: {result_path}")
        rows.append({"episode_id": episode_id, "membership": next(ep["membership"] for ep in identity["dataset_identity"]["dataset"]["episodes"] if ep["id"] == episode_id),
                     "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                     "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()})
    ids = [row["episode_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Partial evaluation archive has no completed unique strict episodes")
    return {"schema": "taichidough/partial-strict-evaluation-inventory/v1",
            "original_identity_sha256": canonical_hash(identity),
            "original_optimizer_checkpoint_sha256": canonical_hash(checkpoint),
            "completed_count": len(rows), "evaluations": rows}


def _read_origin(original_run, *, allow_partial_evaluation=False):
    original_run = Path(original_run).resolve()
    with (original_run / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Original run is still active; wait for its evaluations to finish") from error
        try:
            manifest = read_json(original_run / "run_manifest.json")
            result = read_json(original_run / "result.json")
            checkpoint = read_json(original_run / "optimizer_state.json")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    identity = manifest.get("identity")
    digest = manifest.get("identity_sha256")
    if manifest.get("schema") != "taichidough-differentiable-run-v1" or not isinstance(identity, dict):
        raise ValueError("Original run manifest has the wrong schema")
    if canonical_hash(identity) != digest or checkpoint.get("identity_sha256") != digest:
        raise ValueError("Original run identity and optimizer checkpoint disagree")
    if identity.get("action") != "dataset-fit" or identity.get("optimizer", {}).get("objective") != "independent-episode-minibatch-v1":
        raise ValueError("Original run is not a minibatch dataset fit")
    snapshot = checkpoint.get("state")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or not isinstance(snapshot, dict):
        raise ValueError("Original optimizer checkpoint has the wrong schema")
    if (result.get("status") != "budget_exhausted"
            or result.get("optimization", {}).get("accepted_updates") != snapshot.get("accepted_updates")):
        raise ValueError("Original optimizer has not completed its accepted-update budget")
    evaluations = result.get("evaluations")
    episodes = identity["dataset_identity"]["episodes"]
    complete = (result.get("independent_evaluation_skipped") is False
                and isinstance(evaluations, dict)
                and set(evaluations) == {ep["episode_id"] for ep in episodes}
                and all(row.get("strict_evaluation", {}).get("valid") is True
                        for row in evaluations.values()))
    if complete:
        inventory = None
    elif allow_partial_evaluation:
        inventory_path = original_run / "partial_evaluation_inventory.json"
        if (original_run / "episodes").is_dir():
            inventory = _partial_evaluation_inventory(original_run, identity, checkpoint)
            if inventory_path.is_file() and read_json(inventory_path) != inventory:
                raise ValueError("Recorded partial evaluations differ from completed worker results")
        elif inventory_path.is_file():
            inventory = read_json(inventory_path)
            if (inventory.get("schema") != "taichidough/partial-strict-evaluation-inventory/v1"
                    or inventory.get("original_identity_sha256") != digest
                    or inventory.get("original_optimizer_checkpoint_sha256") != canonical_hash(checkpoint)
                    or not 0 < inventory.get("completed_count", 0) < len(episodes)):
                raise ValueError("Archived partial evaluation inventory is incompatible")
        else:
            raise ValueError("Partial evaluation override requires completed strict worker results")
        if inventory["completed_count"] >= len(episodes):
            raise ValueError("All strict evaluations have completed; use the normal origin check")
    else:
        raise ValueError("Original independent evaluations are incomplete; explicit partial-evaluation override required")
    return manifest, result, checkpoint, inventory


def verify_snapshot(snapshot, *, dataset, batch_size, origin_iteration, target_iteration,
                    adam_options=None, original_history=None):
    """Reject incomplete, reordered, nonfinite or inconsistent optimizer snapshots."""
    space = dataset.parameter_space()
    episodes = [ep for ep in dataset.episodes if ep.membership == "training"]
    batches = calibration._episode_batches(episodes, min(batch_size, len(episodes)))
    iteration = snapshot.get("iterations")
    if (type(iteration) is not int or not origin_iteration <= iteration <= target_iteration
            or snapshot.get("accepted_updates") != iteration):
        raise ValueError("Optimizer update count is inconsistent with continuation target")
    history = snapshot.get("history")
    if not isinstance(history, list) or len(history) != iteration:
        raise ValueError("Optimizer history is incomplete")
    for index, row in enumerate(history, 1):
        expected_batch = batches[(index - 1) % len(batches)]
        if (row.get("iteration") != index or row.get("batch_index") != (index - 1) % len(batches)
                or row.get("episode_ids") != [ep.id for ep in expected_batch]):
            raise ValueError(f"Optimizer history differs from the fixed batch order at update {index}")
    vectors = []
    for name in ("coordinates", "first_moment", "second_moment"):
        values = np.asarray(snapshot.get(name), dtype=np.float64)
        if values.shape != (len(dataset.fit_parameters),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid optimizer {name}")
        vectors.append(values)
    u, m, v = vectors
    if np.any(v < 0) or not np.array_equal(space.project(u), u):
        raise ValueError("Optimizer moments or coordinates are outside valid bounds")
    physical = space.physical(u)
    if snapshot.get("physical_parameters") != physical or history[-1]["parameters_after"] != physical:
        raise ValueError("Optimizer coordinates do not reproduce their recorded physical parameters")
    if original_history is not None and history[:origin_iteration] != original_history:
        raise ValueError("Continuation history does not preserve the original accepted updates")
    if adam_options is not None:
        check_u = space.coordinates()
        check_m, check_v = np.zeros_like(check_u), np.zeros_like(check_u)
        b1, b2 = adam_options.beta1, adam_options.beta2
        for index, row in enumerate(history, 1):
            if row["parameters_before"] != space.physical(check_u):
                raise ValueError(f"Parameters before update {index} do not match Adam history")
            gradient = space.pullback(check_u, row["physical_gradient"])
            if row["coordinate_gradient"] != gradient.tolist() or row["learning_rate"] != adam_options.learning_rate:
                raise ValueError(f"Gradient or learning rate differs at update {index}")
            check_m = b1 * check_m + (1.0 - b1) * gradient
            check_v = b2 * check_v + (1.0 - b2) * gradient * gradient
            m_hat = check_m / (1.0 - b1 ** index)
            v_hat = check_v / (1.0 - b2 ** index)
            step = -adam_options.learning_rate * m_hat / (np.sqrt(v_hat) + adam_options.epsilon)
            old_u = check_u.copy()
            check_u = space.project(check_u + step)
            if (row["parameters_after"] != space.physical(check_u)
                    or row["step_norm"] != float(np.linalg.norm(check_u - old_u, ord=np.inf))):
                raise ValueError(f"Parameters after update {index} do not match Adam history")
        if not (np.array_equal(u, check_u) and np.array_equal(m, check_m) and np.array_equal(v, check_v)):
            raise ValueError("Optimizer coordinates or moments do not match accepted-update history")
    return u, m, v, history, batches


def continue_updates(dataset, state, *, batch_size, target_iteration, max_updates, adam_options,
                     runner_factory, store, event, on_checkpoint=None):
    """Use the original minibatch ordering, transforms, moments and bias correction."""
    space = dataset.parameter_space()
    u, m, v, history, batches = verify_snapshot(
        state, dataset=dataset, batch_size=batch_size,
        origin_iteration=state["origin_iteration"], target_iteration=target_iteration,
        adam_options=adam_options,
    )
    fit_names = tuple(dataset.fit_parameters)
    successful = 0
    mismatches = 0
    first = state["iterations"] + 1
    last = min(target_iteration, state["iterations"] + max_updates)
    for iteration in range(first, last + 1):
        batch_index = (iteration - 1) % len(batches)
        batch = batches[batch_index]
        shared = space.physical(u)
        print(f"batch start {iteration}: batch={batch_index + 1}/{len(batches)}; "
              f"episodes={','.join(ep.id for ep in batch)}; params[{parameter_text({name: shared[name] for name in fit_names})}]",
              flush=True)
        result = calibration._evaluate_episode_batch(dataset, batch, shared, compute_grad=True,
                                                     runner_factory=runner_factory, max_workers=batch_size,
                                                     event=event)
        successful += 1
        mismatches += int(not result.diagnostics.get("replay_consistent", True))
        if mismatches:
            raise InvalidStateError("Batch has inconsistent backward replay; no update accepted")
        gradient = space.pullback(u, result.gradient)
        if not np.isfinite(gradient).all():
            raise InvalidStateError("Batch coordinate gradient is not finite")
        b1, b2 = adam_options.beta1, adam_options.beta2
        m = b1 * m + (1.0 - b1) * gradient
        v = b2 * v + (1.0 - b2) * gradient * gradient
        m_hat = m / (1.0 - b1 ** iteration)
        v_hat = v / (1.0 - b2 ** iteration)
        step = -adam_options.learning_rate * m_hat / (np.sqrt(v_hat) + adam_options.epsilon)
        old_u = u.copy()
        u = space.project(u + step)
        step_norm = float(np.linalg.norm(u - old_u, ord=np.inf))
        parameters_after = space.physical(u)
        record = {"type": "minibatch_step", "iteration": iteration, "batch_index": batch_index,
                  "episode_ids": [ep.id for ep in batch], "value_before_update": result.value,
                  "physical_gradient": dict(result.gradient), "coordinate_gradient": gradient.tolist(),
                  "gradient_norm": float(np.linalg.norm(gradient)), "step_norm": step_norm,
                  "learning_rate": adam_options.learning_rate, "parameters_before": shared,
                  "parameters_after": parameters_after, "diagnostics": result.diagnostics}
        history.append(record)
        store.append_event({"event": "optimizer_minibatch_step", **record})
        state.update(iterations=iteration, accepted_updates=iteration,
                     coordinates=u.tolist(), first_moment=m.tolist(), second_moment=v.tolist(),
                     physical_parameters=parameters_after, history=history,
                     continuation_objectives=state.get("continuation_objectives", 0) + 1,
                     continuation_mismatches=state.get("continuation_mismatches", 0)
                     + int(not result.diagnostics.get("replay_consistent", True)))
        store.write_json("optimizer_state.json", {"schema": CONTINUATION_SCHEMA,
                                                  "identity_sha256": store.identity_hash, "state": state})
        if on_checkpoint is not None:
            on_checkpoint()
        print(f"batch update {iteration}: batch={batch_index + 1}/{len(batches)}; "
              f"loss={number(result.value)}; step={number(step_norm)}; "
              f"params[{parameter_text({name: parameters_after[name] for name in fit_names})}]; "
              f"replay_consistent={result.diagnostics.get('replay_consistent', True)}", flush=True)
        youngs_bounds = getattr(dataset, "parameter_bounds", {}).get("youngs_modulus")
        if youngs_bounds is not None:
            lower, upper = youngs_bounds
            modulus = parameters_after["youngs_modulus"]
            if modulus <= lower + 0.1 * (upper - lower) or modulus >= upper - 0.1 * (upper - lower):
                warning = {"event": "youngs_modulus_near_bound", "iteration": iteration,
                           "youngs_modulus": modulus, "bounds": youngs_bounds}
                store.append_event(warning)
                print(f"WARNING: E={modulus:.6g} Pa near {youngs_bounds} Pa bound at update {iteration}",
                      flush=True)
    return {"updates_this_process": successful, "replay_mismatches_this_process": mismatches,
            "iterations": state["iterations"], "target_iterations": target_iteration,
            "batch_size": batch_size}


def _fsync_directory(path):
    handle = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.fsync(handle)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
            warnings.warn("Mirror filesystem does not support directory fsync; file writes are still flushed",
                          RuntimeWarning, stacklevel=2)
    finally:
        os.close(handle)


def _copy_snapshot(source, destination):
    destination = Path(destination)
    temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with Path(source).open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise OSError(f"Mirrored file differs from source: {destination}")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _write_mirror_hash(mirror, snapshot_hash, iteration, identity_hash):
    marker = mirror / "checkpoint_sha256.json"
    temporary = marker.with_name("." + marker.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump({"optimizer_state_sha256": snapshot_hash, "iterations": iteration,
                       "identity_sha256": identity_hash}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        _fsync_directory(mirror)
    finally:
        temporary.unlink(missing_ok=True)


def _origin_args(dataset_path, original_identity):
    options = original_identity["optimizer"]
    source_settings = original_identity["dataset_identity"]
    return argparse.Namespace(dataset=dataset_path, reference_policy=source_settings["reference_policy"],
                              cpu_threads=source_settings["cpu_threads"],
                              ignore_recompute_mismatch=source_settings["ignore_recompute_mismatch"],
                              worker_timeout_s=None, fit_log="quiet", no_runtime=False)


def _check_preflight_identity(dataset, run_args, summaries, source, original_identity, *, allow_host_change):
    current = calibration._dataset_identity(dataset, run_args, summaries, source)
    if current == original_identity:
        return []
    if not allow_host_change:
        raise ValueError("Current preflight does not reproduce the original validated dataset and runtime")
    previous = deepcopy(original_identity)
    checked = deepcopy(current)
    changes = []
    if len(previous["episodes"]) != len(checked["episodes"]):
        raise ValueError("Preflight episode count changed")
    for old, new in zip(previous["episodes"], checked["episodes"], strict=True):
        old_host = old["runtime"].pop("host")
        new_host = new["runtime"].pop("host")
        if old_host != new_host:
            changes.append({"episode_id": old["episode_id"], "previous_host": old_host,
                            "current_host": new_host})
    if previous != checked:
        raise ValueError("Preflight changed more than the runtime host; refusing continuation")
    if not changes:
        raise ValueError("Preflight changed without a recorded host difference")
    return changes


def run(args):
    original_run = args.original_run.resolve()
    manifest, original_result, checkpoint, partial_inventory = _read_origin(
        original_run, allow_partial_evaluation=args.allow_partial_evaluation_after_user_request)
    original_identity = manifest["identity"]
    original_dataset_identity = original_identity["dataset_identity"]
    if source_identity() != original_dataset_identity["source"]:
        raise ValueError("Current experiment source or dependencies differ from original run")
    original_dataset = original_dataset_identity["dataset"]
    original_config = original_dataset["episodes"][0]["config"]
    simulation = original_config["simulation"]
    options = {"backend": original_config["backend"], "precision": simulation["precision"],
               "p2g_mode": simulation.get("p2g_mode"),
               "physics_version": simulation.get("physics_version"),
               "segment_length": original_config["segment_length"],
               "initial_overrides": {name: original_dataset["shared_parameters"]["initial"][name]
                                     for name in ("youngs_modulus", "viscosity")
                                     if name in original_dataset["shared_parameters"]["initial"]}}
    dataset = load_dataset(args.dataset, **options)
    if (dataset.as_dict() != original_dataset_identity["dataset"]
            or dataset.fingerprint != original_dataset_identity["dataset_fingerprint"]):
        raise ValueError("Dataset manifest, episode configurations or fitted parameter settings changed")
    original_options = dict(original_identity["optimizer"])
    batch_size = original_options.pop("episode_batch_size")
    if original_options.pop("objective") != "independent-episode-minibatch-v1":
        raise ValueError("Original objective differs")
    adam_options = AdamOptions(**original_options)
    if asdict(adam_options) != original_options:
        raise ValueError("Original Adam options cannot be reproduced")
    origin_iteration = checkpoint["state"]["iterations"]
    if args.target_updates < origin_iteration + args.min_additional_updates:
        raise ValueError("Target does not provide the requested minimum number of additional updates")
    verify_snapshot(checkpoint["state"], dataset=dataset, batch_size=batch_size,
                    origin_iteration=origin_iteration, target_iteration=args.target_updates,
                    adam_options=adam_options)
    original_selection = original_run / "selected_parameters.json"
    if (read_json(original_selection).get("identity_sha256") != manifest["identity_sha256"]
            or original_result.get("optimization", {}).get("current_parameters")
            != checkpoint["state"]["physical_parameters"]):
        raise ValueError("Original selected parameters and checkpoint do not match")
    source = original_dataset_identity["source"]
    origin_manifest_hash = canonical_hash(manifest)
    origin_checkpoint_hash = canonical_hash(checkpoint)
    origin_result_hash = canonical_hash(original_result)
    continuation_identity = {"action": "dataset-fit-minibatch-continuation", "schema": CONTINUATION_SCHEMA,
                             "original_identity_sha256": manifest["identity_sha256"],
                             "original_run_manifest_sha256": origin_manifest_hash,
                             "original_optimizer_checkpoint_sha256": origin_checkpoint_hash,
                             "original_result_sha256": origin_result_hash,
                             "partial_evaluation_after_user_request": partial_inventory is not None,
                             "partial_evaluation_inventory_sha256": (canonical_hash(partial_inventory)
                                                                     if partial_inventory is not None else None),
                             "dataset_identity": original_dataset_identity,
                             "optimizer": original_identity["optimizer"],
                             "origin_iteration": origin_iteration,
                             "target_iteration": args.target_updates}
    run_args = _origin_args(args.dataset, original_identity)
    with RunStore(args.output_dir, continuation_identity, resume=args.resume) as store:
        if not args.resume and partial_inventory is not None:
            store.write_json("original_partial_evaluation_inventory.json", partial_inventory)
            print(f"Original strict evaluation stopped after {partial_inventory['completed_count']}/"
                  f"{len(dataset.episodes)} completed episodes; continuing by explicit user request", flush=True)
        if not args.resume:
            validated_inputs_path = original_run / "validated_inputs.json"
            original_summaries = None
            if validated_inputs_path.is_file():
                previous_validation = read_json(validated_inputs_path)
                original_summaries = previous_validation["episodes"]
                same_host = all(value["runtime"]["host"] == socket.gethostname()
                                for value in original_summaries.values())
            else:
                previous_validation = None
                same_host = False
            if (original_summaries is not None and same_host
                    and calibration._dataset_identity(dataset, run_args, original_summaries, source)
                    == original_dataset_identity):
                summaries = original_summaries
                preflight_root = Path(previous_validation["preflight_directory"])
                changes = []
                store.append_event({"event": "original_preflight_reused",
                                    "original_identity_sha256": manifest["identity_sha256"]})
            else:
                summaries, failures, preflight_root = calibration.preflight(dataset, run_args, options, source)
                if failures:
                    raise ValueError(f"Dataset preflight failed: {preflight_root / 'result.json'}")
                changes = _check_preflight_identity(dataset, run_args, summaries, source,
                                                    original_dataset_identity,
                                                    allow_host_change=args.allow_host_change)
                if original_summaries is None:
                    store.append_event({"event": "original_validated_inputs_absent_fresh_preflight_performed",
                                        "original_identity_sha256": manifest["identity_sha256"]})
            store.write_json("validated_inputs.json", {"episodes": summaries,
                                                        "original_identity_sha256": manifest["identity_sha256"],
                                                        "preflight_directory": str(preflight_root),
                                                        "host_changes": changes})
            state = deepcopy(checkpoint["state"])
            state.update(origin_iteration=origin_iteration, continuation_objectives=0,
                         continuation_mismatches=0)
            store.write_json("optimizer_state.json", {"schema": CONTINUATION_SCHEMA,
                                                      "identity_sha256": store.identity_hash, "state": state})
        else:
            validated = store.read_json("validated_inputs.json")
            if validated.get("original_identity_sha256") != manifest["identity_sha256"]:
                raise ValueError("Continuation preflight record belongs to a different original run")
            summaries = validated["episodes"]
            _check_preflight_identity(dataset, run_args, summaries, source,
                                      original_dataset_identity, allow_host_change=True)
            if (not args.allow_host_change
                    and any(value["runtime"]["host"] != socket.gethostname()
                            for value in summaries.values())):
                raise ValueError("Runtime host changed; rerun with --allow-host-change for fresh preflight")
            if args.allow_host_change:
                new_summaries, failures, preflight_root = calibration.preflight(dataset, run_args, options, source)
                if failures:
                    raise ValueError(f"Resume preflight failed: {preflight_root / 'result.json'}")
                changes = _check_preflight_identity(dataset, run_args, new_summaries, source,
                                                    original_dataset_identity, allow_host_change=True)
                if changes != validated.get("host_changes", []):
                    store.append_event({"event": "runtime_host_change", "changes": changes,
                                        "preflight_directory": str(preflight_root)})
                summaries = new_summaries
                store.write_json("validated_inputs.json", {"episodes": summaries,
                                                            "original_identity_sha256": manifest["identity_sha256"],
                                                            "preflight_directory": str(preflight_root),
                                                            "host_changes": changes})
            saved = store.read_json("optimizer_state.json")
            if saved.get("schema") != CONTINUATION_SCHEMA or saved.get("identity_sha256") != store.identity_hash:
                raise ValueError("Continuation checkpoint identity differs from run manifest")
            state = saved["state"]
        if state["origin_iteration"] != origin_iteration:
            raise ValueError("Continuation optimizer state uses a different original iteration")
        verify_snapshot(state, dataset=dataset, batch_size=batch_size,
                        origin_iteration=origin_iteration, target_iteration=args.target_updates,
                        adam_options=adam_options, original_history=checkpoint["state"]["history"])

        mirror = args.mirror_dir.resolve()
        content_root = Path("/content")
        if (content_root.exists() and mirror.is_relative_to(content_root)
                and not mirror.is_relative_to(content_root / "drive")):
            raise ValueError("Colab mirror must be outside /content or on mounted /content/drive")
        if not args.resume:
            mirror.mkdir(parents=True, exist_ok=False)
            origin_copy = mirror / "original_run"
            origin_copy.mkdir()
            for name in ("run_manifest.json", "optimizer_state.json", "result.json",
                         "selected_parameters.json", "validated_inputs.json"):
                if (original_run / name).is_file():
                    _copy_snapshot(original_run / name, origin_copy / name)
            if partial_inventory is not None:
                _copy_snapshot(store.path / "original_partial_evaluation_inventory.json",
                               origin_copy / "partial_evaluation_inventory.json")
        else:
            if read_json(mirror / "run_manifest.json").get("identity_sha256") != store.identity_hash:
                raise ValueError("Mirror directory belongs to a different run")
            marker = read_json(mirror / "checkpoint_sha256.json")
            mirrored_checkpoint = read_json(mirror / "optimizer_state.json")
            mirrored_hash = hashlib.sha256((mirror / "optimizer_state.json").read_bytes()).hexdigest()
            mirrored_iteration = mirrored_checkpoint["state"]["iterations"]
            if (marker.get("identity_sha256") != store.identity_hash
                    or mirrored_checkpoint.get("identity_sha256") != store.identity_hash):
                raise ValueError("Off-runtime checkpoint mirror has a different run identity")
            if (marker.get("optimizer_state_sha256") != mirrored_hash
                    or marker.get("iterations") != mirrored_iteration):
                if marker.get("iterations") != mirrored_iteration - 1:
                    raise ValueError("Off-runtime checkpoint hash does not match the checkpoint")
                verify_snapshot(mirrored_checkpoint["state"], dataset=dataset, batch_size=batch_size,
                                origin_iteration=origin_iteration, target_iteration=args.target_updates,
                                adam_options=adam_options, original_history=checkpoint["state"]["history"])
                _write_mirror_hash(mirror, mirrored_hash, mirrored_iteration, store.identity_hash)
                store.append_event({"event": "mirror_hash_recovered", "iteration": mirrored_iteration})
            if mirrored_checkpoint["state"]["iterations"] > state["iterations"]:
                raise ValueError("Mirror checkpoint is newer than local state; restore from mirror first")
            if (mirrored_checkpoint["state"]["iterations"] == state["iterations"]
                    and canonical_hash(mirrored_checkpoint) != canonical_hash(saved)):
                raise ValueError("Local and off-runtime checkpoints differ at the same update")
        _copy_snapshot(store.path / "run_manifest.json", mirror / "run_manifest.json")
        _copy_snapshot(store.path / "validated_inputs.json", mirror / "validated_inputs.json")

        def on_checkpoint():
            digest = _copy_snapshot(store.path / "optimizer_state.json", mirror / "optimizer_state.json")
            _write_mirror_hash(mirror, digest, state["iterations"], store.identity_hash)

        on_checkpoint()
        if state["iterations"] == args.target_updates and (store.path / "result.json").is_file():
            finished = store.read_json("result.json")
            if finished.get("status") == "budget_exhausted" and finished.get("iterations") == args.target_updates:
                if not (store.path / "selected_parameters.json").is_file():
                    raise ValueError("Completed result has no parameter selection")
                _copy_snapshot(store.path / "result.json", mirror / "result.json")
                _copy_snapshot(store.path / "selected_parameters.json", mirror / "selected_parameters.json")
                print(f"Continuation already completed: {args.target_updates} updates", flush=True)
                return 0

        def event(record):
            store.append_event(record)
            calibration.progress(record, "quiet")

        def runner_factory():
            return calibration._runner(dataset, run_args, options, store.path / "episodes", source,
                                       event=event,
                                       prepared={key: value["prepared_fingerprint"] for key, value in summaries.items()},
                                       runtimes={key: stable_runtime(value["runtime"]) for key, value in summaries.items()})

        report = continue_updates(dataset, state, batch_size=batch_size, target_iteration=args.target_updates,
                                  max_updates=args.max_updates_this_run, adam_options=adam_options,
                                  runner_factory=runner_factory, store=store, event=event,
                                  on_checkpoint=on_checkpoint)
        if state["iterations"] == args.target_updates:
            training = [ep for ep in dataset.episodes if ep.membership == "training"]
            batches = calibration._episode_batches(training, min(batch_size, len(training)))
            final = calibration._evaluate_all_training_batches(dataset, batches,
                                                               dataset.parameter_space().physical(state["coordinates"]),
                                                               runner_factory=runner_factory, max_workers=batch_size,
                                                               event=event)
            selected = state["physical_parameters"]
            selection_key = "shared_physical_parameters" if dataset.schema == SCHEMA_V2 else "shared_material_parameters"
            selection = {"schema": (calibration.SELECTION_SCHEMA_V2 if dataset.schema == SCHEMA_V2
                                     else calibration.SELECTION_SCHEMA),
                         selection_key: selected, "fitted_parameters": list(dataset.fit_parameters),
                         "training_value": final["value"], "selection_membership": "training",
                         "dataset_fingerprint": dataset.fingerprint, "dataset_identity": original_dataset_identity,
                         "identity_sha256": store.identity_hash,
                         "physics_version": original_dataset_identity["physics_version"],
                         "physics_reference": original_dataset_identity["physics_reference"],
                         "episodes": {ep.id: calibration.episode_selection_record(ep, selected)
                                      for ep in dataset.episodes},
                         "independent_heldout_episodes_declared": any(ep.membership == "validation" for ep in dataset.episodes),
                         "ignore_recompute_mismatch": run_args.ignore_recompute_mismatch,
                         "objective": {"version": "independent-episode-minibatch-v1", "episode_batch_size": batch_size,
                                       "final_training_evaluation": final},
                         "continuation": {"original_identity_sha256": manifest["identity_sha256"],
                                          "origin_iteration": origin_iteration, "target_iteration": args.target_updates}}
            store.write_json("selected_parameters.json", selection)
            report.update(status="budget_exhausted", final_training_evaluation=final,
                          current_parameters=selected, current_value=final["value"],
                          independent_evaluation_skipped=True)
        else:
            report.update(status="paused", current_parameters=state["physical_parameters"])
        store.write_json("result.json", report)
        if mirror is not None:
            _copy_snapshot(store.path / "result.json", mirror / "result.json")
            if state["iterations"] == args.target_updates:
                _copy_snapshot(store.path / "selected_parameters.json", mirror / "selected_parameters.json")
        print(f"Continuation: {state['iterations']}/{args.target_updates} updates; "
              f"state={store.path / 'optimizer_state.json'}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-updates", type=int, required=True)
    parser.add_argument("--min-additional-updates", type=int, default=150,
                        help="Refuse fewer than 30 more five-batch epochs (150 updates)")
    parser.add_argument("--max-updates-this-run", type=int, default=1,
                        help="Exactly one accepted update per invocation so the host can verify its off-runtime checkpoint")
    parser.add_argument("--mirror-dir", type=Path, required=True,
                        help="Durable checkpoint mirror outside Colab's ephemeral runtime; new empty directory on first run")
    parser.add_argument("--allow-partial-evaluation-after-user-request", action="store_true",
                        dest="allow_partial_evaluation_after_user_request",
                        help="Continue after explicitly requested stop of original strict evaluation; pin completed worker results")
    parser.add_argument("--allow-host-change", action="store_true",
                        help="After a full preflight, allow only a Colab hostname change; optimizer state is retained")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.target_updates < 1 or args.min_additional_updates < 1:
        parser.error("All update counts must be positive")
    if args.max_updates_this_run != 1:
        parser.error("Run one update at a time; verify the off-runtime checkpoint before each resume")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

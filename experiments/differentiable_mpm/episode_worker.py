"""Evaluate one dataset episode in a fresh process with explicit result identity."""
import argparse
from pathlib import Path
import json
import sys
from time import perf_counter

from .calibrate import evaluation_record, export_and_evaluate, make_rollout, stored_progress
from .checkpoint import estimate_memory
from .data import prepare_experiment
from .reference_adapter import reference_identity, reference_policy, verify_reference
from .results import RunStore, canonical_hash, source_identity
from .runtime import init_runtime
from .state import InvalidStateError


REQUEST_SCHEMA = "taichidough/episode-request/v1"
RESULT_SCHEMA = "taichidough/episode-result/v1"
ACTIONS = ("validate", "objective", "replay", "evaluate")


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate request key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def load_request(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def validate_request(request):
    required = {"schema", "request_id", "episode_id", "action", "dataset_path", "dataset_options",
                "parameters", "compute_grad", "reference_policy", "cpu_threads",
                "ignore_recompute_mismatch", "expected_source", "expected_dataset_fingerprint",
                "expected_prepared_fingerprint"}
    if not isinstance(request, dict) or not required <= request.keys() or set(request) - required - {"no_runtime"}:
        raise ValueError("Episode request has missing or unknown fields")
    if request["schema"] != REQUEST_SCHEMA or request["action"] not in ACTIONS:
        raise ValueError("Unsupported episode request schema or action")
    for name in ("request_id", "episode_id", "dataset_path"):
        if not isinstance(request[name], str) or not request[name]:
            raise ValueError(f"{name} must be a nonempty string")
    if request["reference_policy"] not in {"strict", "frozen"}:
        raise ValueError("reference_policy must be strict or frozen")
    for name in ("compute_grad", "ignore_recompute_mismatch"):
        if type(request[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if type(request.get("no_runtime", False)) is not bool:
        raise ValueError("no_runtime must be a boolean")
    if request.get("no_runtime", False) and request["action"] != "validate":
        raise ValueError("no_runtime is only valid for input validation")
    if request["action"] != "objective" and request["compute_grad"]:
        raise ValueError("Only objective requests compute gradients")
    if type(request["cpu_threads"]) is not int or request["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be a positive integer")
    for name in ("dataset_options", "parameters", "expected_source"):
        if not isinstance(request[name], dict):
            raise ValueError(f"{name} must be an object")
    for name in ("expected_dataset_fingerprint", "expected_prepared_fingerprint"):
        value = request[name]
        if name == "expected_prepared_fingerprint" and value is None and request["action"] == "validate":
            continue
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be a SHA-256 digest")
    canonical_hash(request)


def _load_dataset(request):
    from .dataset_config import load_dataset
    return load_dataset(request["dataset_path"], **request["dataset_options"])


def _evaluate(request, store, response):
    dataset = _load_dataset(request)
    response["dataset_fingerprint"] = dataset.fingerprint
    if dataset.fingerprint != request["expected_dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint differs from the parent request")
    matches = [episode for episode in dataset.episodes if episode.id == request["episode_id"]]
    if len(matches) != 1:
        raise ValueError("Episode ID must identify exactly one dataset entry")
    episode = matches[0]
    config = episode.config
    parameters = episode.parameters_for(request["parameters"])
    response["parameters"] = parameters
    # Candidate parameters never alter preparation identity or the declared initialization.
    physics = config.simulation.get("physics_version", "corrected-v1")
    verify_reference()
    physics_reference = reference_identity(physics)
    runtime = ({"initialization_verified": False, "backend": config.backend,
                "precision": config.simulation.get("precision", "f32")}
               if request.get("no_runtime", False) else
               init_runtime(config.backend, config.simulation.get("precision", "f32"),
                            cpu_threads=request["cpu_threads"], seed=config.seed))
    runtime = {**runtime, "physics_version": physics, "physics_reference": physics_reference,
               "ignore_recompute_mismatch": request["ignore_recompute_mismatch"]}
    response["runtime"] = runtime
    prepared = prepare_experiment(config, split=episode.membership,
                                  build_sdf=True, scored_window=episode.scored_window)
    response["prepared_fingerprint"] = prepared.fingerprint
    expected = request["expected_prepared_fingerprint"]
    if expected is not None and prepared.fingerprint != expected:
        raise ValueError("Prepared episode fingerprint differs from validated inputs")
    sim = prepared.simulation_config
    memory = estimate_memory(sim.n_particles, prepared.total_steps, config.segment_length,
                             sim.precision, sim.grid, config.tool_sdf_resolution,
                             physics_version=sim.physics_version)
    response["prepared_summary"] = {**prepared.summary(), "memory_estimate": memory,
                                    "n_particles": sim.n_particles}
    if request["action"] == "validate":
        return
    if request["action"] == "objective":
        _, rollout = make_rollout(prepared, progress=stored_progress(store),
                                  ignore_recompute_mismatch=request["ignore_recompute_mismatch"])
        evaluation = rollout.value_and_gradient(parameters, compute_grad=request["compute_grad"])
        response["evaluation"] = evaluation_record(evaluation)
        runtime["forward_verified"] = True
        runtime["backward_verified"] = request["compute_grad"]
        runtime["backward_replay_consistent"] = (request["compute_grad"] and
                                                   evaluation.diagnostics.get("replay_consistent", True))
    else:
        response["export"] = export_and_evaluate(prepared, None, parameters, store, runtime,
                                                 strict=request["action"] == "evaluate")
        runtime["forward_verified"] = True


def run_request(request, output_dir):
    """Write one checked response; parameter invalidity is distinct from worker errors."""
    validate_request(request)
    requested_hash = canonical_hash(request)
    actual_source = source_identity()
    response = {"schema": RESULT_SCHEMA, "request_id": request["request_id"],
                "request_sha256": requested_hash, "episode_id": request["episode_id"],
                "status": "error", "dataset_fingerprint": None, "source": actual_source,
                "prepared_fingerprint": None, "runtime": None, "parameters": None,
                "compute_grad": request["compute_grad"]}
    identity = {"action": "isolated-episode-worker", "request": request, "source": actual_source}
    started = perf_counter()
    with RunStore(output_dir, identity) as store:
        try:
            if actual_source != request["expected_source"]:
                raise ValueError("Worker source or dependencies differ from the parent request")
            with reference_policy(request["reference_policy"]):
                _evaluate(request, store, response)
            response["status"] = "ok"
        except InvalidStateError as error:
            response.update(status="invalid", error_type=type(error).__name__, error=str(error))
        except Exception as error:
            response.update(status="error", error_type=type(error).__name__, error=str(error))
        final_source = source_identity()
        response["source_after"] = final_source
        if final_source != actual_source:
            response.update(status="error", error_type="SourceChangedError",
                            error="Worker source or dependencies changed during episode execution")
        response["elapsed_s"] = perf_counter() - started
        store.write_json("result.json", response)
        store.append_event({"event": "episode_finished", "episode_id": request["episode_id"],
                            "status": response["status"], "elapsed_s": response["elapsed_s"]})
    return {"ok": 0, "invalid": 2, "error": 1}[response["status"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run_request(load_request(args.request), args.output_dir)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

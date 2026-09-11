"""Readable application logs alongside the existing machine-readable run records.

This module observes Python events only: no Taichi calls, field downloads, timers
in background threads, or optimizer-state checkpoints during candidate trials.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback
import uuid

from .results import RUN_ROOT, json_value


SCHEMA = "taichidough-calibration-log-v1"
BANNER = "# " + SCHEMA


def safe_value(value):
    value = json_value(value)
    if isinstance(value, dict):
        return {key: safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def number(value):
    return "n/a" if value is None else f"{float(value):.6g}"


def named_values(values):
    return " ".join(f"{key}={number(value)}" for key, value in values.items())


def parameter_text(values):
    labels = {"youngs_modulus": "E(Pa)", "poisson_ratio": "nu", "viscosity": "viscosity(Pa.s)",
              "plastic_min": "plastic_min", "plastic_max": "plastic_max",
              "tool_retention": "tool_retention", "floor_retention": "floor_retention"}
    return named_values({labels.get(key, key): value for key, value in values.items()})


class RunLogger:
    def __init__(self, *, root=RUN_ROOT, interval_s=5.0, level="info", stream=None, clock=perf_counter):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("log interval must be positive and finite")
        if level not in {"info", "debug"}:
            raise ValueError("log level must be info or debug")
        self.clock, self.started = clock, clock()
        self.interval_s, self.level = interval_s, level
        self.stream = sys.stdout if stream is None else stream
        self.session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}_" + uuid.uuid4().hex[:8]
        directory = Path(root) / "attempt_logs"
        directory.mkdir(parents=True, exist_ok=True)
        self.attempt_path = directory / (self.session + ".log")
        self.attempt = self.attempt_path.open("x+", encoding="utf-8")
        self.attempt.write(BANNER + "\n")
        self.attempt.flush()
        self.store = self.run_stream = None
        self.pending = []
        self.context = {}
        self.fit_names = []
        self.sequence = 0
        self.failed = False
        self.closed = False
        self.last_print = -math.inf
        self.phase_started = self.started
        self.sample = None
        self.phase_samples = 0
        self.local_evaluations = 0
        self.emit("attempt_started", f"Application log: {self.attempt_path}", log_level=level,
                  log_interval_s=interval_s)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, tb):
        try:
            if error is not None:
                self.failure(error)
        finally:
            self.close()
        return False

    def _record(self, kind, **fields):
        self.sequence += 1
        return safe_value({"schema": SCHEMA, "event": kind, "session": self.session,
                           "sequence": self.sequence, "time": datetime.now(timezone.utc).isoformat(),
                           "elapsed_s": self.clock() - self.started, **self.context, **fields})

    def _write(self, record, message, console=True):
        if not message:
            return
        context = " ".join(f"{key}={record[key]}" for key in ("split", "evaluation", "iteration", "backtrack")
                           if record.get(key) is not None)
        prefix = f'[{record["time"][11:19]} +{record["elapsed_s"]:.1f}s]'
        for text in str(message).splitlines():
            line = f"{prefix} {context + ' ' if context else ''}{text}\n"
            self.attempt.write(line)
            if self.run_stream is not None:
                self.run_stream.write(line)
            if console:
                self.stream.write(line)
        self.attempt.flush()
        if self.run_stream is not None:
            self.run_stream.flush()
        if console:
            self.stream.flush()

    def emit(self, kind, message="", *, console=True, **fields):
        record = self._record(kind, **fields)
        if self.store is None:
            self.pending.append(record)
        else:
            self.store.append_event(record)
        self._write(record, message, console)
        return record

    @contextmanager
    def bind_store(self, store):
        path = store.path / "run.log"
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Refusing non-regular application log: {path}")
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                if stream.readline().rstrip("\n") != BANNER:
                    raise ValueError(f"Existing run.log is not this application's log: {path}")
        self.run_stream = path.open("a", encoding="utf-8")
        self.attempt.flush()
        self.attempt.seek(0)
        self.run_stream.write(self.attempt.read())
        self.attempt.seek(0, 2)
        self.run_stream.flush()
        self.store = store
        try:
            for record in self.pending:
                store.append_event(record)
            self.pending.clear()
            self.emit("run_opened", f"Verified run: {store.path}\nReadable log: {path}\nMachine events: {store.path / 'events.jsonl'}",
                      identity_sha256=store.identity_hash)
            yield self
        except BaseException as error:
            self.failure(error)
            raise
        finally:
            self.run_stream.close()
            self.run_stream = None
            self.store = None

    def phase(self, name, **fields):
        self.context["phase"] = name
        if "split" in fields:
            self.context["split"] = fields["split"]
        self.phase_started = self.clock()
        self.sample = None
        self.phase_samples = 0
        self.emit("phase_started", f"Starting {name}", **fields)

    def progress(self, event):
        phase = event["phase"]
        total, step = int(event["total_steps"]), int(event["step"])
        backward = phase.startswith("backward")
        completed = int(event.get("completed_steps", total - step if backward else step))
        if phase in {"forward_start", "backward_start"}:
            self.phase("backward" if backward else "forward")
        elapsed = float(event.get("phase_elapsed_s", self.clock() - self.phase_started))
        rate = None
        if phase.endswith("_segment"):
            if self.sample is not None and completed > self.sample[0] and elapsed > self.sample[1]:
                rate = (completed - self.sample[0]) / (elapsed - self.sample[1])
            self.sample = (completed, elapsed)
            self.phase_samples += 1
        eta = (total - completed) / rate if rate is not None and rate > 0 else None
        critical = any(word in phase for word in ("invalid", "failed", "mismatch"))
        now = self.clock()
        show = (self.level == "debug" or critical or phase.endswith(("_start", "_complete"))
                or completed == total or (phase == "observation" and event.get("frame_index", 99) <= 3)
                or now - self.last_print >= self.interval_s)
        message = ""
        if show:
            self.last_print = now
            if critical:
                message = f"ERROR {phase}: {json.dumps(safe_value(event), sort_keys=True)}"
            elif phase == "observation":
                support = event.get("diagnostics", {})
                message = (f'frame={event["frame_index"]} step={step}/{total} loss={number(event["value"])} '
                           + named_values(event.get("components", {})))
                selected = {key: support[key] for key in ("observed_pixels", "current_overlap_pixels", "fixed_depth_pixels",
                                                           "visible_predicted_particles") if key in support}
                if selected:
                    message += " support: " + named_values(selected)
            else:
                message = f"{phase} completed={completed}/{total} global_state={step} phase_elapsed={elapsed:.1f}s"
                if rate is not None:
                    message += f" recent_wall_rate={rate:.1f}steps/s phase_ETA≈{eta:.1f}s"
                elif phase.endswith("_start"):
                    message += "; first execution may include compilation; no completion verified yet"
        self.emit("replay_progress", message, **event, phase_elapsed_s=elapsed,
                  recent_wall_steps_per_s=rate, phase_eta_s=eta)

    def objective_result(self, result):
        frames = result.frames
        component_names = sorted({key for frame in frames for key in frame.get("components", {})})
        means = {key: sum(float(frame["components"][key]) for frame in frames if key in frame.get("components", {})) /
                 sum(key in frame.get("components", {}) for frame in frames) for key in component_names}
        support = {}
        for key in ("observed_pixels", "current_overlap_pixels", "fixed_depth_pixels", "visible_predicted_particles", "directed_points"):
            values = [frame.get("diagnostics", {}).get(key) for frame in frames]
            values = [value for value in values if value is not None]
            if values:
                support[key] = [min(values), max(values)]
        diagnostics = result.diagnostics
        message = (f"Rollout loss={number(result.value)} forward={number(diagnostics.get('forward_seconds'))}s "
                   f"backward={number(diagnostics.get('backward_seconds'))}s; raw component means: {named_values(means)}")
        if support:
            message += "\nSupport ranges: " + json.dumps(support, sort_keys=True)
        self.emit("rollout_completed", message, value=result.value, component_means=means,
                  support_ranges=support, diagnostics=diagnostics)

    def optimizer_event(self, event):
        kind = event["type"]
        self.context.update({key: event.get(key) for key in ("iteration", "evaluation", "backtrack")})
        if kind == "evaluation_started":
            self.local_evaluations += 1
            message = f'Candidate start ({event["stage"]}, process_call={self.local_evaluations}): ' + parameter_text(event["parameters"])
        elif kind == "evaluation":
            if event["valid"]:
                named_gradient = dict(zip(self.fit_names, event["coordinate_gradient"]))
                message = (f'Candidate complete loss={number(event["value"])} elapsed={event["elapsed_s"]:.2f}s\n'
                           + "dL/du: " + named_values(named_gradient)
                           + "\nPhysical derivatives: " + named_values(event["physical_gradient"]))
            else:
                message = f'Candidate FAILED after {event["elapsed_s"]:.2f}s: {event["error_type"]}: {event["error"]}'
        else:
            message = (f'{"ACCEPT" if event["accepted"] else "REJECT"} trial rate={number(event["learning_rate"])} '
                       f'value={number(event.get("value"))} reason={event.get("reason", "sufficient_decrease")}')
            if "acceptance_limit" in event:
                message += f' limit={number(event["acceptance_limit"])}'
        self.emit("optimizer_" + kind, message, payload=event)

    def optimizer_checkpoint(self, event, state):
        record = self._record("optimizer_checkpoint", payload=event)
        self.store.save_optimizer(record, state)
        message = (f'Optimizer status={state["status"]} accepted_updates={state["accepted_updates"]} '
                   f'evaluations={state["evaluations"]} checkpoint_saved')
        if event["type"] == "step":
            message += (f' proposal_rate={number(event.get("proposal_learning_rate"))} '
                        f'accepted_rate={number(event.get("accepted_learning_rate"))} '
                        f'retained_rate={number(event["learning_rate"])} '
                        f'gradient_norm={number(event["gradient_norm"])} '
                        f'projected_gradient_norm={number(event["projected_gradient_norm"])}')
        self._write(record, message)

    def resumed(self, state):
        self.emit("optimizer_restored", f'Resumed iteration={state["iterations"]}, accepted_updates={state["accepted_updates"]}, '
                  f'evaluations={state["evaluations"]}, retained_rate={number(state["learning_rate"])}; '
                  "restored records are not new forward/backward execution",
                  restored_iteration=state["iterations"], restored_evaluations=state["evaluations"],
                  best=state["best"])

    def failure(self, error):
        if self.failed:
            return
        self.failed = True
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record = self.emit("attempt_" + status, f'{status.upper()}: {type(error).__name__}: {error}\n'
                           f'Application log: {self.attempt_path}; last completed optimizer checkpoint is retained',
                           error_type=type(error).__name__, error=str(error))
        self._write(record, "".join(traceback.format_exception(type(error), error, error.__traceback__)), console=False)

    def close(self):
        if self.closed:
            return
        if self.pending:
            path = self.attempt_path.with_suffix(".jsonl")
            with path.open("x", encoding="utf-8") as stream:
                for record in self.pending:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.attempt.close()
        self.closed = True

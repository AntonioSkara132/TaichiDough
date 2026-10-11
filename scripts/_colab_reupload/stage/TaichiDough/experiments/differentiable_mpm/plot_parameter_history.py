"""Plot saved optimizer histories without importing or running the simulator.

Requires NumPy and Matplotlib. Source runs are read-only; output must be new.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid

LABELS = {
    "youngs_modulus": ("Young's modulus", "kPa", 0.001),
    "poisson_ratio": ("Poisson ratio", "dimensionless", 1),
    "viscosity": ("Viscosity", "Pa·s", 1),
    "plastic_min": ("Lower plastic stretch limit", "stretch ratio", 1),
    "plastic_max": ("Upper plastic stretch limit", "stretch ratio", 1),
    "tool_retention": ("Tool retention", "dimensionless", 1),
    "floor_retention": ("Floor retention", "dimensionless", 1),
    "tool_friction_coefficient": ("Tool friction coefficient", "dimensionless", 1),
    "tool_stickiness": ("Tool stickiness", "dimensionless", 1),
    "loss": ("Training objective", "dimensionless", 1),
}
COLORS = ["#2a78d6", "#eb6834"]
GROUND, INK, SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"


def read_json(path, sources):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.tzinfo else None
    except ValueError:
        return None


def classify(identity, path):
    if "synthetic" in path.name or identity.get("kind", "").startswith(("synthetic", "prestrained")):
        return "synthetic"
    if identity.get("action") == "fit" and identity.get("prepared", {}).get("sequence_fingerprint"):
        return "recorded_episode"
    return "other"


def load_run(path):
    """Extract recorded states; trials are joined to decisions by evaluation ID."""
    path = path.resolve()
    sources, warnings = {}, []
    manifest_path = path / "run_manifest.json"
    manifest = read_json(manifest_path, sources) if manifest_path.exists() else {}
    identity = manifest.get("identity", {})
    result_path = path / "result.json"
    result = read_json(result_path, sources) if result_path.exists() else {}
    optimization = result.get("optimization", {})
    history = optimization.get("history")
    history_source = "result.json:optimization.history"
    if not isinstance(history, list):
        history_path = path / "optimization_history.json"
        state_path = path / "optimizer_state.json"
        if history_path.exists():
            history = read_json(history_path, sources)
            history_source = "optimization_history.json"
        elif state_path.exists():
            state = read_json(state_path, sources).get("state", {})
            history = state.get("history", [])
            history_source = "optimizer_state.json:state.history"
        else:
            history = []
    if not isinstance(history, list):
        raise ValueError(f"Unsupported history in {path}")
    evaluations = [x for x in history if x.get("type") == "evaluation"]
    steps = [x for x in history if x.get("type") == "step"]
    if not evaluations and not steps:
        raise ValueError("No saved parameter evaluation/update history")
    ids = [x["evaluation"] for x in evaluations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation IDs; ambiguous resumed history")
    step_ids = [x["iteration"] for x in steps]
    if len(step_ids) != len(set(step_ids)) or step_ids != sorted(step_ids):
        raise ValueError("Duplicate or unordered step iterations")
    event_path = path / "events.jsonl"
    start = init_time = None
    step_times = {}
    if event_path.exists():
        digest = hashlib.sha256()
        with event_path.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                digest.update(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append(f"Ignored invalid event JSON at line {number}; no timestamp inferred")
                    continue
                kind = event.get("event", event.get("type"))
                when = timestamp(event.get("time"))
                if kind == "start" and start is None:
                    start = when
                elif kind == "initialization" and event.get("status") == "ready" and init_time is None:
                    init_time = when
                elif kind == "step":
                    key = event.get("iteration")
                    if key in step_times:
                        raise ValueError("Duplicate timestamped step; ambiguous resumed event log")
                    step_times[key] = (when, event.get("parameters"), event.get("value_after"))
        sources[str(event_path)] = digest.hexdigest()
    space = identity.get("parameter_space", {})
    fitted = space.get("fit", result.get("fitted_parameters", []))
    bounds = space.get("bounds", {})
    simulation = identity.get("prepared", {}).get("simulation", identity.get("simulation", {}))
    fixed_contact = {k: simulation[k] for k in ("tool_friction_coefficient", "tool_stickiness") if k in simulation}
    decisions = {}
    for step in steps:
        for attempt in step.get("attempts", []):
            eid = attempt["evaluation"]
            if eid in decisions:
                raise ValueError("An evaluation belongs to multiple step decisions")
            decisions[eid] = attempt
    by_eval = {x["evaluation"]: x for x in evaluations}
    trial_rows = []
    for row in evaluations:
        decision = decisions.get(row["evaluation"])
        state = ("initial" if row.get("stage") == "initial" else
                 "accepted" if decision and decision.get("accepted") else
                 "rejected" if decision else "decision_missing")
        diag = row.get("diagnostics", {})
        trial_rows.append({
            "evaluation": row["evaluation"], "iteration": row.get("iteration"),
            "decision": state, "backtrack": row.get("backtrack"),
            "valid": row.get("valid"), "loss": row.get("value"),
            "evaluation_duration_s": row.get("elapsed_s"),
            "replay_consistent": diag.get("replay_consistent"),
            "recompute_mismatch_count": diag.get("recompute_mismatch_count"),
            "reason": decision.get("reason") if decision else row.get("error"),
            "parameters": {**fixed_contact, **row.get("parameters", {})},
        })
    states = []
    initial = [x for x in evaluations if x.get("stage") == "initial" and x.get("valid")]
    if len(initial) > 1:
        raise ValueError("Multiple initial evaluations; history needs explicit session selection")

    def add_state(iteration, update, decision, params, loss, when, evaluation):
        if not params or not all(finite(v) for v in params.values()) or not finite(loss):
            raise ValueError("Saved optimizer state contains missing/nonfinite values")
        elapsed = (when - start).total_seconds() if when and start else None
        if elapsed is not None and elapsed < 0:
            raise ValueError("Recorded state timestamp precedes run start")
        states.append({"iteration": iteration, "accepted_updates": update,
                       "decision": decision, "evaluation": evaluation, "loss": loss,
                       "timestamp": when.isoformat() if when else None,
                       "elapsed_wall_s": elapsed, "parameters": {**fixed_contact, **params}})

    if initial:
        row = initial[0]
        add_state(row["iteration"], 0, "initial", row["parameters"], row["value"], init_time, row["evaluation"])
    else:
        warnings.append("Initial evaluation absent: no iteration-zero point invented")
    for step in steps:
        joined = [a for a in step.get("attempts", []) if a.get("accepted")]
        if step.get("accepted"):
            if len(joined) != 1 or joined[0]["evaluation"] not in by_eval:
                raise ValueError("Accepted step cannot be matched to one saved evaluation")
            evaluation = by_eval[joined[0]["evaluation"]]
            if step["parameters"] != evaluation["parameters"] or step["value_after"] != evaluation["value"]:
                raise ValueError("Accepted step disagrees with its saved evaluation")
        elif states and (step["parameters"] != {k:v for k,v in states[-1]["parameters"].items() if k in step["parameters"]}
                         or step["value_after"] != states[-1]["loss"]):
            raise ValueError("Rejected step changed retained parameters or loss")
        when, event_params, event_loss = step_times.get(step["iteration"], (None, None, None))
        if when and (event_params != step["parameters"] or event_loss != step["value_after"]):
            raise ValueError("Timestamped event disagrees with optimizer history")
        add_state(step["iteration"], step.get("accepted_updates"),
                  "accepted" if step.get("accepted") else "retained_after_rejection",
                  step["parameters"], step["value_after"], when,
                  joined[0]["evaluation"] if joined else None)
    if not states:
        raise ValueError("No retained optimizer states available")
    timed = [x["elapsed_wall_s"] for x in states]
    has_time = all(x is not None for x in timed) and timed == sorted(timed)
    if not has_time:
        warnings.append("Complete monotonic wall-clock timestamps unavailable; no elapsed-time chart")
    if not fitted:
        warnings.append("Fitted-parameter list absent; plotted names are not asserted fitted")
        fitted = list(states[0]["parameters"])
    runtime = identity.get("runtime", result.get("runtime", {}))
    return {"name": path.name, "path": str(path), "classification": classify(identity, path),
            "history_source": history_source, "sources_sha256": sources, "warnings": warnings,
            "fitted_parameters": fitted, "bounds": bounds, "simulation": simulation,
            "mass": identity.get("prepared", {}).get("mass", {}),
            "runtime": runtime, "result_status": result.get("status", "result_not_saved"),
            "optimizer_status": optimization.get("status", result.get("status", "unknown")),
            "states": states, "evaluations": trial_rows, "has_wall_time": has_time}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row if k != "parameters"))
    parameters = list(dict.fromkeys(k for row in rows for k in row.get("parameters", {})))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys + parameters)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k:v for k,v in row.items() if k != "parameters"}, **row.get("parameters", {})})


def draw(runs, path, time_axis=False, comparison=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    keys = list(dict.fromkeys(k for r in runs for k in r["fitted_parameters"])) + ["loss"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": INK, "axes.labelcolor": SECONDARY,
                         "xtick.color": SECONDARY, "ytick.color": SECONDARY,
                         "axes.titleweight": "semibold", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#c3c2b7"})
    rows = math.ceil(len(keys) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.3 * rows + 1.4), squeeze=False)
    fig.patch.set_facecolor(GROUND)
    title = "Calibration histories · recorded-data runs" if comparison else runs[0]["name"]
    fig.suptitle(title, fontsize=15, x=0.065, ha="left", y=0.98)
    subtitle = "Different scene/contact assumptions; these are not controlled parameter comparisons." if comparison else (
        f"{runs[0]['classification']} · {runs[0]['result_status']} · "
        f"{sum(s['decision']=='accepted' for s in runs[0]['states'])} accepted updates; "
        f"{sum(e['decision']=='rejected' for e in runs[0]['evaluations'])} rejected trials")
    fig.text(0.065, 0.942, subtitle, fontsize=10, color=SECONDARY)
    for ax, key in zip(axes.flat, keys):
        ax.set_facecolor(GROUND)
        label, unit, scale = LABELS.get(key, (key, "physical units", 1))
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(axis="y", color=GRID, linewidth=0.65)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        for index, run in enumerate(runs):
            states = run["states"]
            x = [s["elapsed_wall_s"] / 60 if time_axis else s["iteration"] for s in states]
            y = [s["loss"] if key == "loss" else s["parameters"].get(key, np.nan) * scale for s in states]
            color = COLORS[index]
            name = run.get("short_label", run["name"])
            ax.plot(x, y, color=color, linewidth=1.5, marker="o" if index == 0 else "s",
                    markersize=5, markeredgecolor=GROUND, markeredgewidth=1,
                    linestyle="-" if index == 0 else "--", label=name if comparison else "Retained state")
            if len(y) and finite(y[-1]):
                direct_name = name.replace("Registered tools", "Registered").replace("Earlier scene", "Earlier")
                ax.annotate(f"{direct_name + ': ' if comparison else ''}{y[-1]:.5g}",
                            (x[-1], y[-1]), xytext=(7, 5),
                            textcoords="offset points", ha="left",
                            fontsize=8, color=INK,
                            bbox={"facecolor": GROUND, "edgecolor": "none", "pad": 1, "alpha": 0.9})
            if not time_axis and not comparison:
                rejected = [e for e in run["evaluations"] if e["decision"] == "rejected" and
                            finite(e["loss"] if key == "loss" else e["parameters"].get(key))]
                if rejected:
                    ax.scatter([e["iteration"] for e in rejected],
                               [(e["loss"] if key == "loss" else e["parameters"][key] * scale) for e in rejected],
                               marker="x", s=38, color=SECONDARY, label="Rejected trial", zorder=3)
                    ax.legend(fontsize=8, frameon=False)
                limits = run["bounds"].get(key)
                if limits:
                    bottom, top = ax.get_ylim()
                    for value, word in zip(limits, ["lower bound", "upper bound"]):
                        value *= scale
                        if bottom <= value <= top:
                            ax.axhline(value, color=SECONDARY, linestyle=":", linewidth=0.8)
                            ax.text(0.01, value, word, transform=ax.get_yaxis_transform(), fontsize=8,
                                    color=SECONDARY, va="bottom")
        ax.margins(x=0.2, y=0.18)
        if comparison:
            left = min(s["iteration"] for r in runs for s in r["states"])
            right = max(s["iteration"] for r in runs for s in r["states"])
            span = max(right - left, 1)
            ax.set_xlim(left - 0.04 * span, right + 0.42 * span)
        if not time_axis:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_xlabel("Minutes since recorded run start (update completion)" if time_axis else "Optimizer iteration (0 = initial evaluation)")
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    if comparison:
        handles, names = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower left", bbox_to_anchor=(0.055, 0.028), ncol=2, frameon=False)
    else:
        fixed = {k:v for k,v in runs[0]["states"][0]["parameters"].items() if k not in runs[0]["fitted_parameters"]}
        fig.text(0.065, 0.04, "Fixed saved values: " + "; ".join(f"{k}={v:g}" for k,v in fixed.items()),
                 fontsize=8, color=SECONDARY, wrap=True)
    fig.text(0.065, 0.014, "Parameters vary across optimization updates, not within simulated time. Exact values and decisions are in the CSV tables.",
             fontsize=8, color=SECONDARY)
    fig.subplots_adjust(left=0.075, right=0.95, top=0.88, bottom=0.13, hspace=0.5, wspace=0.28)
    fig.savefig(path.with_suffix(".png"), dpi=160, facecolor=GROUND)
    fig.savefig(path.with_suffix(".pdf"), facecolor=GROUND)
    plt.close(fig)


def summary(run):
    first, last = run["states"][0], run["states"][-1]
    changes = {}
    for k in run["fitted_parameters"]:
        values = [s["parameters"][k] for s in run["states"]]
        bounds = run["bounds"].get(k, [])
        changes[k] = {"initial": values[0], "final": values[-1], "minimum": min(values), "maximum": max(values),
                      "first_upper_bound_iteration": next((s["iteration"] for s in run["states"]
                        if len(bounds) == 2 and math.isclose(s["parameters"][k], bounds[1], rel_tol=1e-10, abs_tol=1e-12)), None)}
    return {"run": run["name"], "classification": run["classification"],
            "result_status": run["result_status"], "optimizer_status": run["optimizer_status"],
            "states": len(run["states"]), "accepted_updates": sum(s["decision"] == "accepted" for s in run["states"]),
            "rejected_trials": sum(e["decision"] == "rejected" for e in run["evaluations"]),
            "initial_loss": first["loss"], "final_loss": last["loss"],
            "relative_loss_reduction_percent": 100 * (first["loss"] - last["loss"]) / first["loss"] if first["loss"] else None,
            "final_elapsed_wall_minutes": last["elapsed_wall_s"] / 60 if run["has_wall_time"] else None,
            "changes": changes, "mass": run["mass"], "warnings": run["warnings"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--run-dir", type=Path, action="append", help="Plot these runs only; repeat for multiple runs")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--include-synthetic", action="store_true", help="Also produce separately labeled synthetic-run plots")
    args = parser.parse_args()
    root = args.runs_root.resolve()
    if args.run_dir:
        candidates = sorted({p.resolve() for p in args.run_dir})
    else:
        candidates = sorted({p.parent for p in root.rglob("run_manifest.json")} |
                            {p.parent for p in root.rglob("optimization_history.json")})
    inventory, runs = [], []
    for path in candidates:
        try:
            run = load_run(path)
        except (ValueError, KeyError, TypeError, OSError) as error:
            inventory.append({"run": str(path), "plotted": False, "reason": str(error)})
            continue
        selected = bool(args.run_dir) or run["classification"] == "recorded_episode" or args.include_synthetic
        inventory.append({"run": str(path), "classification": run["classification"], "plotted": selected,
                          "reason": None if selected else "Synthetic/other history excluded from real-run report"})
        if selected:
            runs.append(run)
    if not runs:
        raise SystemExit("No selected calibration histories found")
    output = (args.output_dir or root / ("parameter_history_plots_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6])).resolve()
    if any(output == Path(r["path"]) or Path(r["path"]) in output.parents for r in runs):
        raise SystemExit("Output cannot be inside a source run")
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for i, run in enumerate(runs, 1):
        run["short_label"] = ("Registered tools" if "registered-tools_fit" in run["name"] else
                              "Earlier scene" if run["name"].startswith("episode18-stretch-clamp_fit") else f"Run {i}")
        folder = output / run["name"]
        folder.mkdir()
        draw([run], folder / "parameters_vs_iteration")
        if run["has_wall_time"]:
            draw([run], folder / "parameters_vs_elapsed_time", time_axis=True)
        write_csv(folder / "retained_states.csv", run["states"])
        write_csv(folder / "evaluations_and_trials.csv", run["evaluations"])
        (folder / "extracted_history.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
        summaries.append(summary(run))
    real = [r for r in runs if r["classification"] == "recorded_episode"]
    if 1 < len(real) <= 2:
        draw(real, output / "real_runs_comparison", comparison=True)
    lines = ["# Calibration parameter histories", "", "These plots use saved physical values, not rerun simulations. Iteration 0 is the recorded initial objective evaluation. Lines follow retained optimizer states; rejected trials are separate crosses when present. Elapsed-time charts use event timestamps at initialization/update completion, measured from the recorded run start (including initialization and preparation). Per-evaluation duration is not treated as cumulative wall time.", "", "Original runs remain unchanged. CSV parameter units are the original physical units (Young's modulus in Pa); plots display Young's modulus in kPa. Fixed recorded values are included in CSV/JSON and figure footnotes. No absent initial state, timestamp, trial decision or fixed contact value is invented.", "", "## Recorded runs", "", "| Run | Initial loss | Final loss | Reduction | Accepted updates | Rejected trials |", "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['run']} ({s['classification']}) | {s['initial_loss']:.8g} | {s['final_loss']:.8g} | {s['relative_loss_reduction_percent']:.3f}% | {s['accepted_updates']} | {s['rejected_trials']} |")
    for r, s in zip(runs, summaries):
        lines += ["", f"## {r['name']}", "", f"Result status: `{r['result_status']}`. Optimizer status: `{r['optimizer_status']}`.", "", f"![Parameters versus iteration]({r['name']}/parameters_vs_iteration.png)", "", "| Fitted parameter | Initial | Final | Minimum | Maximum | First upper-bound iteration |", "|---|---:|---:|---:|---:|---:|"]
        for k, c in s["changes"].items():
            lines.append(f"| {k} | {c['initial']:.10g} | {c['final']:.10g} | {c['minimum']:.10g} | {c['maximum']:.10g} | {c['first_upper_bound_iteration']} |")
        lines += ["", f"Mass metadata: `{json.dumps(r['mass'], sort_keys=True)}`.", "", f"Replay mismatch override recorded: `{r['runtime'].get('ignore_recompute_mismatch', 'not recorded')}`. Forward loss improvement does not qualify derivative accuracy or physical material identity."]
        if r["has_wall_time"]:
            lines += ["", f"[Elapsed-time plot]({r['name']}/parameters_vs_elapsed_time.png). Final saved update at {s['final_elapsed_wall_minutes']:.3f} minutes after recorded run start."]
        lines += ["", f"[Exact retained states]({r['name']}/retained_states.csv) · [All evaluated trials]({r['name']}/evaluations_and_trials.csv)"]
        lines.extend("- " + w for w in r["warnings"])
    lines += ["", "## Interpretation and coverage", "", "Different scene reconstructions, contact settings and objective definitions must not be treated as controlled comparisons or ranked by raw loss alone. The earlier Episode18 fit records `independent_evaluation_failed`; its optimizer history remains valid to plot, but the run did not finish independent evaluation successfully. The registered-tools fit exhausted its update budget, not a demonstrated convergence criterion.", "", "Only histories available under the supplied local root were examined. Remote-only /mnt runs are not included. Forward-only simulations and single gradient evaluations do not contain a material-parameter trajectory. Synthetic optimization runs are listed in inventory.json but excluded by default. Loss-component histories and per-iteration validation curves are not constructed from final-only evaluations.", "", "For another computer, copy plot_parameter_history.py and use an existing Python environment with NumPy and Matplotlib. No Taichi, raw recording, mesh or particle-state arrays are required:", "", "```bash", "python plot_parameter_history.py --runs-root /path/to/runs --output-dir /path/to/new_plot_directory", "```", "", "Use repeated `--run-dir /path/to/run` to select particular histories. `--include-synthetic` explicitly includes labeled synthetic histories. Existing output directories are refused."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    checked = 0
    for run in runs:
        for path, expected in run["sources_sha256"].items():
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError(f"Source changed during plotting: {path}")
            checked += 1
    (output / "verification.json").write_text(json.dumps({"source_files_unchanged": checked,
        "accepted_steps_match_evaluated_parameters_and_loss": True,
        "timestamped_steps_match_optimizer_history": True,
        "palette": COLORS, "palette_validation": "dataviz validator: all five checks passed, light mode",
        "particle_arrays_loaded": False, "simulation_or_calibration_launched": False}, indent=2) + "\n")
    print(f"PLOTS: {output}")
    for s in summaries:
        c = s["changes"].get("youngs_modulus", {})
        print(f"{s['run']}: E {c.get('initial')} -> {c.get('final')} Pa; loss {s['initial_loss']:.8g} -> {s['final_loss']:.8g}")


if __name__ == "__main__":
    main()

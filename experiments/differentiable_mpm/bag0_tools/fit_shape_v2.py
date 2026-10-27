"""Calibration fit with the combined loss (shape descriptors relative to first frame, late-weighted, + original loss
with frame-referenced depth) using Adam on normalised parameters (log scale for E and yield), both parts per update.
usage: fit_shape.py ROOT DATASET OUT_DIR --parts P1 P2 --start '{"youngs_modulus":..,...}' [--updates 40] [--lr 0.05]
       [--orig-weight 1.0] [--fix viscosity,tool_adhesion] [--validate P3 P4] [--resume]"""
import json, os, sys, time, math, argparse, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("dataset"); ap.add_argument("out")
ap.add_argument("--parts", nargs="+", required=True); ap.add_argument("--start", required=True); ap.add_argument("--updates", type=int, default=40); ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--orig-weight", type=float, default=1.0); ap.add_argument("--fix", default="viscosity,tool_adhesion"); ap.add_argument("--validate", nargs="*", default=[]); ap.add_argument("--validate-every", type=int, default=10); ap.add_argument("--resume", action="store_true"); ap.add_argument("--precision", default="f64")
a = ap.parse_args(); os.chdir(a.root); sys.path.insert(0, a.root); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.makedirs(a.out, exist_ok=True)
from experiments.differentiable_mpm.calibrate import make_rollout
from experiments.differentiable_mpm.data import prepare_experiment
from experiments.differentiable_mpm.dataset_config import load_dataset
from experiments.differentiable_mpm.reference_adapter import reference_policy
from experiments.differentiable_mpm.runtime import init_runtime
from experiments.differentiable_mpm.loss_options import parse_loss_config
import dataclasses
from shape_loss import ShapeDescriptorLoss, descriptors
D = os.path.dirname(a.dataset) + "/"; T = np.asarray(json.load(open(D + "scene_calibration_v2.json"))["scene_from_source"])
LOGP = ("youngs_modulus", "von_mises_yield_stress_pa")
def ext(p): return (np.percentile(p[:, 0], 95) - np.percentile(p[:, 0], 5), np.percentile(p[:, 2], 95) - np.percentile(p[:, 2], 5), np.percentile(p[:, 1], 99))

with reference_policy("frozen"):
    ds = load_dataset(a.dataset, backend="cuda", precision=a.precision, p2g_mode="atomic"); eps = {e.id: e for e in ds.episodes}
    init_runtime("cuda", a.precision, cpu_threads=1, seed=eps[a.parts[0]].config.seed)
    bounds = dict(ds.parameter_bounds) if hasattr(ds, "parameter_bounds") else {}
    bounds.setdefault("von_mises_yield_stress_pa", [100.0, 20000.0]); bounds.setdefault("youngs_modulus", [2000.0, 60000.0])
    if os.environ.get("FIT_POISSON_MAX"): bounds["poisson_ratio"] = [bounds["poisson_ratio"][0], float(os.environ["FIT_POISSON_MAX"])]
    lr_cur = a.lr
    PART_BASE = {}
    fixed = set(a.fix.split(",")) if a.fix else set()
    fit_names = [n for n in ("youngs_modulus", "poisson_ratio", "viscosity", "tool_friction_coefficient", "tool_adhesion", "floor_friction_coefficient", "von_mises_yield_stress_pa") if n not in fixed]
    start = dict(ds.shared_initial); start.update(json.loads(a.start))
    def to_u(p): return np.array([(math.log(p[n]) - math.log(bounds[n][0])) / (math.log(bounds[n][1]) - math.log(bounds[n][0])) if n in LOGP else (p[n] - bounds[n][0]) / (bounds[n][1] - bounds[n][0]) for n in fit_names])
    def to_p(u):
        p = dict(start)
        for n, v in zip(fit_names, np.clip(u, 0.0, 1.0)): p[n] = math.exp(math.log(bounds[n][0]) + v * (math.log(bounds[n][1]) - math.log(bounds[n][0]))) if n in LOGP else bounds[n][0] + v * (bounds[n][1] - bounds[n][0])
        return p
    def dL_du(p, g): return np.array([g.get(n, 0.0) * (p[n] * (math.log(bounds[n][1]) - math.log(bounds[n][0])) if n in LOGP else (bounds[n][1] - bounds[n][0])) for n in fit_names])
    # rollouts with the combined loss
    rollouts, recs, probes = {}, {}, {}
    for pid in a.parts + list(a.validate):
        ep = eps[pid]; cfg = ep.config
        cfg = dataclasses.replace(cfg, loss={**cfg.loss, "depth_reference": "frame"}) if isinstance(cfg.loss, dict) else cfg
        prepared = prepare_experiment(cfg, split=ep.membership, build_sdf=True, scored_window=ep.scored_window); _, ro = make_rollout(prepared, progress=lambda *x, **k: None)
        clouds = torch.load(D + f"recordings/{pid}/pointclouds_interpolated.pt", map_location="cpu", weights_only=True)[0]
        rec = {}; pts = {}
        for i in range(len(clouds)):
            q = np.asarray(clouds[i])[:, :3].astype(np.float64); q = q[np.isfinite(q).all(1)] @ T[:3, :3].T + T[:3, 3]; rec[i] = descriptors(q, 0.004)[0]; pts[i] = q
        frames = sorted(o.frame_index for o in ro.observations); n = len(frames); fw = {f: 2.0 * (k + 1) / n for k, f in enumerate(frames)}
        ro.loss = ShapeDescriptorLoss(ro.loss, rec, tau=0.004, scale=0.01, shape_weight=1.0, orig_weight=a.orig_weight, relative=True, frame_weights=fw, initial_positions=np.asarray(ro.initial_state.x, dtype=np.float64))
        rollouts[pid] = (ep, ro); recs[pid] = pts; print(f"prepared {pid}: {prepared.total_steps} steps, {n} scored frames", flush=True)
    state_path = os.path.join(a.out, "state.json"); hist_path = os.path.join(a.out, "history.jsonl")
    if a.resume and os.path.exists(state_path):
        st = json.load(open(state_path)); u = np.array(st["u"]); m = np.array(st["m"]); v = np.array(st["v"]); it0 = st["iteration"]; best = st["best"]
    else:
        u = to_u(start); m = np.zeros_like(u); v = np.zeros_like(u); it0 = 0; best = {"value": float("inf")}
    b1, b2, epsl = 0.9, 0.999, 1e-8
    def validate(p):
        out = {}
        for pid in a.validate:
            ep, ro = rollouts[pid]; sim_ext = {}
            def cb(step, state): sim_ext[step] = ext(np.asarray(state.x, dtype=np.float64))
            try:
                ev = ro.value_and_gradient(ep.parameters_for({k: p[k] for k in ds.shared_initial}) | {"von_mises_yield_stress_pa": p["von_mises_yield_stress_pa"]}, compute_grad=False, frame_callback=cb, frame_steps=sorted(ro.by_step))
                steps = sorted(sim_ext); zw = [sim_ext[s][1] * 1000 for s in steps]; hs = [sim_ext[s][2] * 1000 for s in steps]
                rec_pts = recs[pid]; rz = [ext(rec_pts[i])[1] * 1000 for i in sorted(rec_pts)]; rh = [ext(rec_pts[i])[2] * 1000 for i in sorted(rec_pts)]
                out[pid] = {"loss": float(ev.value), "sim_width_min_end": (min(zw), zw[-1]), "rec_width_min_end": (min(rz), rz[-1]), "sim_height_max": max(hs), "rec_height_max": max(rh)}
            except Exception as exc: out[pid] = {"error": str(exc)[:200]}
        return out
    for it in range(it0 + 1, a.updates + 1):
        p = to_p(u); t0 = time.perf_counter(); total = 0.0; grad = {n: 0.0 for n in fit_names}; per_part = {}; u_before = u.copy()
        for pid in a.parts:
            ep, ro = rollouts[pid]; params = ep.parameters_for({k: p[k] for k in ds.shared_initial}); params["von_mises_yield_stress_pa"] = p["von_mises_yield_stress_pa"]
            try: ev = ro.value_and_gradient(params, compute_grad=True)
            except Exception as exc:
                per_part[pid] = {"error": str(exc)[:200]}; print(f"update {it}: {pid} FAILED {exc}", flush=True); continue
            w = 1.0
            if os.environ.get("FIT_PART_NORMALIZE"):  # opt-in: divide each part by its first-update loss so every part starts at 1
                PART_BASE.setdefault(pid, max(float(ev.value), 1e-9)); w = 1.0 / PART_BASE[pid]
            total += w * float(ev.value) / len(a.parts); per_part[pid] = {"value": float(ev.value), "weight": w, "shape": float(np.mean([f["components"].get("shape", 0.0) for f in ev.frames]))}
            for n in fit_names: grad[n] += w * float(ev.gradient.get(n, 0.0)) / len(a.parts)
        if any("error" in x for x in per_part.values()):
            lr_cur *= 0.5; u = u_before
            row = {"iteration": it, "value": None, "failed": True, "lr_next": lr_cur, "parameters": {n: p[n] for n in fit_names}, "per_part": per_part, "seconds": round(time.perf_counter() - t0, 1)}
            with open(hist_path, "a") as f: f.write(json.dumps(row) + "\n")
            print(f"update {it:3d}: FAILED step rejected, lr -> {lr_cur:.4g}", flush=True); continue
        g = dL_du(p, grad); m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it); vh = v / (1 - b2 ** it); u = np.clip(u - lr_cur * mh / (np.sqrt(vh) + epsl), 0.0, 1.0)
        row = {"iteration": it, "value": total, "parameters": {n: p[n] for n in fit_names}, "per_part": per_part, "grad_u": g.tolist(), "seconds": round(time.perf_counter() - t0, 1)}
        if total < best["value"]: best = {"value": total, "iteration": it, "parameters": {n: p[n] for n in fit_names}}
        if a.validate and (it % a.validate_every == 0 or it == a.updates): row["validation"] = validate(p)
        with open(hist_path, "a") as f: f.write(json.dumps(row) + "\n")
        json.dump({"u": u.tolist(), "m": m.tolist(), "v": v.tolist(), "iteration": it, "best": best, "next_parameters": to_p(u)}, open(state_path, "w"), indent=1)
        print(f"update {it:3d}: loss {total:.4f} | " + " ".join(f"{n[:6]}={p[n]:.4g}" for n in fit_names) + f" | {row['seconds']:.0f} s" + (f" | val {row['validation']}" if "validation" in row else ""), flush=True)
    print("DONE best:", best, flush=True)

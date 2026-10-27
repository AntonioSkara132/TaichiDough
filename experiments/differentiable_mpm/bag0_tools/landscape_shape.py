"""Shape-descriptor loss over a parameter grid for one episode (forward only, resumable).
usage: landscape_shape.py ROOT DATASET OUT_JSONL GRID_JSON EPISODE
GRID_JSON: {"axes": {"youngs_modulus": [...], "von_mises_yield_stress_pa": [...], "viscosity": [...]}}"""
import itertools, json, os, sys, time, numpy as np, torch
root, dataset, out, grid_path, episode = sys.argv[1:6]
os.chdir(root); sys.path.insert(0, root); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.differentiable_mpm.calibrate import make_rollout
from experiments.differentiable_mpm.data import prepare_experiment
from experiments.differentiable_mpm.dataset_config import load_dataset
from experiments.differentiable_mpm.reference_adapter import reference_policy
from experiments.differentiable_mpm.runtime import init_runtime
from shape_loss import ShapeDescriptorLoss, descriptors
grid = json.load(open(grid_path)); names = list(grid["axes"]); base = grid.get("base", {})
sets = [{**base, **dict(zip(names, c))} for c in itertools.product(*(grid["axes"][n] for n in names))]
done = set()
if os.path.exists(out):
    for l in open(out): r = json.loads(l); done.add(json.dumps(r["set"], sort_keys=True))
D = os.path.dirname(dataset) + "/"; T = np.asarray(json.load(open(D + "scene_calibration_v2.json"))["scene_from_source"])
with reference_policy("frozen"):
    ds = load_dataset(dataset, backend="cuda", precision="f32", p2g_mode="atomic"); ep = next(e for e in ds.episodes if e.id == episode)
    init_runtime("cuda", "f32", cpu_threads=1, seed=ep.config.seed)
    prepared = prepare_experiment(ep.config, split=ep.membership, build_sdf=True, scored_window=ep.scored_window); _, ro = make_rollout(prepared, progress=lambda *x, **k: None)
    clouds = torch.load(str(ep.config.paths["episode"]) + "/pointclouds_interpolated.pt", map_location="cpu", weights_only=True)[0]; rec = {}
    for i in range(len(clouds)):
        p = np.asarray(clouds[i])[:, :3].astype(np.float64); p = p[np.isfinite(p).all(1)] @ T[:3, :3].T + T[:3, 3]; rec[i] = descriptors(p, 0.004)[0]
    frames = sorted(o.frame_index for o in ro.observations); n = len(frames); fw = {f: 2.0 * (k + 1) / n for k, f in enumerate(frames)}
    ro.loss = ShapeDescriptorLoss(ro.loss, rec, tau=0.004, scale=0.01, shape_weight=1.0, orig_weight=float(os.environ.get("LANDSCAPE_ORIG_WEIGHT", "0.0")), relative=True, frame_weights=fw,
                                  initial_positions=np.asarray(ro.initial_state.x, dtype=np.float64))
    for s in sets:
        if json.dumps(s, sort_keys=True) in done: continue
        t0 = time.perf_counter(); shared = dict(ds.shared_initial)
        shared.update({k: v for k, v in s.items() if k in ep.shared_parameter_names})
        params = ep.parameters_for(shared); params.update({k: v for k, v in s.items() if k not in ep.shared_parameter_names})
        try: row = {"episode": episode, "set": s, "value": float(ro.value_and_gradient(params, compute_grad=False).value)}
        except Exception as exc: row = {"episode": episode, "set": s, "error": f"{type(exc).__name__}: {exc}"[:200]}
        row["seconds"] = round(time.perf_counter() - t0, 1)
        with open(out, "a") as f: f.write(json.dumps(row) + "\n")
        print(row, flush=True)
print("LANDSCAPE DONE", episode, flush=True)

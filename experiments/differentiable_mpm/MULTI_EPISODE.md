# Shared-material dataset calibration

`calibrate_dataset` fits one shared material across explicitly selected episodes. The single-episode `calibrate` command remains available and uses the same fit logging and stability options. New source is confined to this experiment.

## Scope and current readiness

Schema v1 fits E, Poisson ratio, viscosity and the two principal-stretch limits. Tool retention is fixed at **1**, and tool absorption and stickiness are **0**. Its optional top-level `floor_retention` and `tool_friction_coefficient` remain fixed, preserving existing behavior and fingerprints. Schema v2 moves floor retention, tool friction, and stickiness into the shared physical vector so they can be fitted across episodes. Each episode keeps its own mass, reconstruction, camera and tool trajectories.

Implementation verification is in progress; see `VALIDATION.md` for completed checks. No full real dataset calibration has been launched.

The local inventory in `runs/dataset_inventory_20260912T132102Z_4a3449/` finds many recording variants, but only Episode18 has the locally identified calibration/reconstruction inputs used by this workflow. All four discovered reconstruction metadata files concern Episode18. The 370 episode-named directories across ten roots are **not 370 independent recordings**: the report flags 64 session-qualified duplicate groups. Discovery does not load tensors or qualify inputs, and marks zero episodes as fully validated by design.

There is also a measured Episode18 floor inconsistency: the observed table plane is about 47.6 mm above the configured floor beneath the initial dough, with about 5° tilt. Its cause is unresolved. Input validation checks consistency among saved metadata, not whether the saved camera/floor calibration agrees with the actual table. Do not interpret a successful input check as resolution of that geometry problem.

## Joint objective

For positive episode weights, the training objective is

`L = sum_e alpha_e * L_e`, where `alpha_e = weight_e / sum_training_weights`.

The existing rollout already averages observations within an episode. The dataset driver does not divide by frame count again. Equal episode weights therefore give equal influence to each episode's mean loss, rather than automatically favoring longer recordings.

All training episodes evaluate the same shared material candidate. Physical gradients are accumulated in deterministic manifest order using host float64; the existing optimizer performs its coordinate transformation. A failure in any training episode rejects the complete candidate, without dropping that episode or renormalizing the weights.

The current `ProjectedAdam` evaluates **loss and gradients for every proposal/backtracking candidate**. One optimizer attempt can therefore require several complete passes through all training episodes. No loss-only acceptance shortcut or independent per-episode fit is substituted.

Only one episode subprocess is active at a time. Its exit releases its Taichi allocations before the next starts. Both input preparation and kernel compilation repeat per episode evaluation; the runtime's offline cache remains disabled. This bounds memory by one episode but is deliberately not a high-throughput implementation. The parent retains summaries and optimizer state, not every episode's particle/renderer allocation. Compilation/preparation cost is included in worker timings.

## Discover inputs

Run from the repository root:

```bash
python3 -m experiments.differentiable_mpm.calibrate_dataset inventory \
  --root /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/DeformPath3 \
  --root /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/DeformPath2_interpolated \
  --reconstruction-root data/single_episode_calibration \
  --episode-config experiments/differentiable_mpm/configs/episode18_stretch_clamp.json
```

`--root`, `--reconstruction-root` and `--episode-config` are repeatable. Outputs are saved in a fresh run: `inventory.json`, `draft.json`, and `result.json`. Malformed metadata and missing/ambiguous prerequisites are reported. Symlinks are not followed. No raw tensors or large tensor hashes are read by discovery.

The draft is **not a runnable dataset manifest**. It contains unresolved configuration/membership/window entries; it is not accepted by `load_dataset`. Pick one processed variant per physical recording and supply each episode's verified configuration. Never copy Episode18's calibration, mass, reconstruction or expected hashes merely to fill missing fields.

## Manifest

A dataset manifest uses `taichidough/differentiable-dataset/v1` or `taichidough/differentiable-dataset/v2` and declares:

- `name`.
- `shared_parameters.initial`: exactly the five material values for v1; the same five plus `floor_retention`, `tool_friction_coefficient`, and `tool_stickiness` for v2.
- `shared_parameters.fit` and optional `bounds`. V2 may fit any distinct subset of its eight shared values.
- Optional `tool_contact`. V1 restricts it to retention1/absorption0/stickiness0. V2 fixes retention at 1 and absorption at 0 while taking stickiness from `shared_parameters`.
- V1 alone accepts top-level fixed `floor_retention` and `tool_friction_coefficient`; v2 requires these values under `shared_parameters`.
- Ordered `episodes`, each with `id`, `config`, `membership` (`training` or `validation`), positive `weight` (default1), `scored_window`, and optional `path_overrides`.

When v2 fits both friction and stickiness, every episode must use SDF collision and `coulomb-adhesive-v1`, with absorption 0. The loader rejects other combinations before numerical execution.

`scored_window` contains explicit `start_frame`, `end_frame` and `stride`. Episodes always replay from reconstruction frame0 to the scored endpoint; their state is never initialized from an observed intermediate deformation. Whole-episode membership is independent of the legacy config's within-episode training/validation windows. Scored frame indices reach the differentiable loss, export provenance and strict evaluator consistently.

Configuration references resolve relative to the dataset manifest. Inputs written inside an existing episode config keep the existing repository-relative path rules. Manifest path overrides resolve relative to the manifest. CLI overrides use explicit `--path EPISODE.INPUT=PATH` and resolve relative to the current directory. No automatic machine-path rewriting occurs.

The loader rejects incompatible shared physics/material/loss/runtime settings, duplicate IDs/resolved recordings/declared sequence fingerprints, invalid bounds and malformed windows. Preflight additionally rejects identical verified recording fingerprints under different episode IDs. Inventory's duplicate-variant warnings still require deliberate choice: different processed versions can have different content fingerprints despite describing the same physical trial.

The ten-episode manifest currently bounds E to 2,000–60,000 Pa, Poisson ratio to 0.45–0.49, and viscosity to 0–60 Pa.s. Its existing plastic-stretch bounds are unchanged. `--initial-youngs-modulus VALUE` and `--initial-viscosity VALUE` replace only the resolved initial candidate; they do not edit or create a dataset manifest. Each value must name a fitted, bounded parameter and lie inside the declared interval. The immutable source-document hash remains the hash of the original manifest, while the resolved dataset fingerprint changes with the initial values. Exact resume therefore rejects a different initial state.

A runnable **single-episode, two-frame smoke example**, not an all-dataset configuration, is provided at:

`configs/dataset_episode18_nonsticky_smoke.json`

It references the existing Episode18 stretch-clamp config, uses its candidate material values as initial guesses, and scores only frames1–2. It demonstrates configuration and path handling without pretending the other recordings are ready.

## Validate before fitting

Input-only smoke validation (no Taichi runtime initialization, no simulation):

```bash
python3 -m experiments.differentiable_mpm.calibrate_dataset validate \
  --dataset experiments/differentiable_mpm/configs/dataset_episode18_nonsticky_smoke.json \
  --reference-policy frozen --no-runtime
```

Omit `--no-runtime` to also initialize the explicitly selected backend in each validation worker. That still does not execute forward or backward kernels. Every declared episode is checked before fitting; missing prerequisites prevent fitting. Preflight writes its own durable `dataset_preflight_*` run with per-episode errors, even when the main fit cannot start.

For the `/mnt` checkout, override the paths explicitly:

```bash
python3 -m experiments.differentiable_mpm.calibrate_dataset validate \
  --dataset experiments/differentiable_mpm/configs/dataset_episode18_nonsticky_smoke.json \
  --reference-policy frozen --backend cuda \
  --path snimanje_23_10-episode18.episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --path snimanje_23_10-episode18.calibration=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla/scene_calibration_v2.json
```

Override `initial_particles` and `reconstruction_metadata` too if their files are elsewhere. Expected input hashes still have to match. The remote dataset beyond Episode18 has not been inventoried by this implementation work.

## Gradient, fit and frozen evaluation

For a completed dataset manifest, set `DATASET` to its path. These are explicit commands to run when inputs/geometry are ready, not commands already executed on the real dataset:

```bash
DATASET=experiments/differentiable_mpm/configs/your_verified_dataset.json

python3 -m experiments.differentiable_mpm.calibrate_dataset gradient \
  --dataset "$DATASET" --reference-policy frozen --backend cpu --precision f64

python3 -m experiments.differentiable_mpm.calibrate_dataset fit \
  --dataset "$DATASET" --reference-policy frozen --backend cuda \
  --physics-version corrected-v1 --iterations 20
```

`gradient --finite-difference` checks all fitted coordinates against multiple complete dataset perturbations and can be expensive. `--p2g-mode serial` preserves the existing fixed-order option; serial GPU transfer can be slow. Strict replay checks are the default. `--ignore-recompute-mismatch` explicitly permits finite replay mismatches, logs them and marks affected combined gradients approximate; invalid states and nonfinite derivatives still fail. A GPU's successful initialization is not evidence of full-dataset gradient qualification.

Dataset fitting defaults to `--fit-log concise`: one candidate line, one complete weighted-dataset loss/gradient line, and one accepted/rejected update line. Physical derivatives are labeled `dL/d<name>` and transformed Adam-coordinate derivatives are labeled `dL/du_<name>`. Routine worker and episode messages remain in `events.jsonl` but are omitted from concise output. `--fit-log detailed` also prints them; `--fit-log quiet` retains warnings, terminal status and result paths.

The default stability rule stops with `converged_parameters` after every fitted physical value remains stable for three consecutive accepted updates. It uses `rtol=1e-4` and absolute tolerances E=5 Pa, Poisson ratio=`1e-5`, viscosity=`1e-3` Pa.s, and each plastic stretch limit=`1e-5`. The comparison is `abs(new-old) <= atol + rtol*max(abs(old),abs(new))`. Repeat `--parameter-stability-atol NAME=VALUE` for overrides; use `--parameter-stability-updates` and `--parameter-stability-rtol` for the other settings. `--parameter-stability-updates 0` disables this rule. Rejected attempts do not change the streak. `budget_exhausted` still means only that the requested optimizer-attempt count ended.

A fit selects using **training episodes only**, then by default exports/replays all declared episodes for independent strict scoring. `--no-evaluate` explicitly skips that final step. All-training manifests are supported but their results state that no independent held-out episodes were declared. A lower combined training loss does not imply every individual episode improved or that material values were uniquely recovered.

Frozen replay/evaluation uses the dataset selection file, not a single-episode selection:

```bash
python3 -m experiments.differentiable_mpm.calibrate_dataset evaluate \
  --dataset "$DATASET" --reference-policy frozen --backend cuda \
  --parameters experiments/differentiable_mpm/runs/YOUR_FIT/selected_parameters.json \
  --split validation
```

Use `--split all` for all declared episodes, or `replay` instead of `evaluate` to export without strict scoring. Frozen selections must match the verified dataset/model/runtime identity. Different physics versions or changed input geometry require a new fit, not constant parameter rescaling.

Use a fresh output directory for new work. `fit --resume --output-dir EXISTING_RUN` requires exact ordered membership, weights, windows, inputs, source/dependencies, parameter bounds, backend/precision and replay policy. `--iterations` adds attempts. No partial worker result counts as an accepted update.

## Cartesian E/viscosity attempts

`calibrate_dataset_attempts` runs a deterministic Cartesian product of initial E and viscosity values. It starts one `calibrate_dataset fit` subprocess at a time, so total cost is approximately the sum of all child calibrations. Repeated E values form the outer loop and repeated viscosity values form the inner loop.

Example for the `/mnt` checkout:

```bash
cd /mnt/Data/studenti/antonio_skara/TaichiDough
PYTHON=/path/to/your/venv/bin/python
DATASET=experiments/differentiable_mpm/data/ten_episode_shared_alignment_v1/dataset.json
OUT=experiments/differentiable_mpm/runs/ten_episode_E_viscosity_attempts

"$PYTHON" -m experiments.differentiable_mpm.calibrate_dataset_attempts \
  --dataset "$DATASET" --python "$PYTHON" --output-dir "$OUT" \
  --initial-youngs-modulus 4000 --initial-youngs-modulus 12000 --initial-youngs-modulus 40000 \
  --initial-viscosity 0 --initial-viscosity 15 --initial-viscosity 45 \
  --reference-policy frozen --backend cuda --precision f32 \
  --physics-version corrected-v1 --iterations 20 --fit-log concise \
  --path EPISODE_ID.episode=/mnt/path/to/that/episode
```

Replace the example `EPISODE_ID` path and repeat `--path EPISODE.INPUT=PATH` for every input that differs on that machine. Run the same command first with `--validate-only`; it resolves the interpreter, bounds, paths, fingerprints, attempt IDs and child argument lists without creating the output directory or starting calibration.

A normal run stores each logical attempt under `attempts/ATTEMPT_ID/`, with immutable numbered invocation records and the child calibration directory. Child concise output is streamed with its attempt ID and retained in that invocation's `console.log`. The root `summary.json` and `summary.csv` are updated before and after each attempt. `best_attempt.json` names and hashes the completed child with minimum verified training loss; this is not held-out model selection when all episodes are training episodes.

Repeat the exact command with `--resume` to verify and skip completed attempts, resume a compatible interrupted child checkpoint, and continue pending attempts. Add `--retry-failed` only with `--resume`; a retry receives a new invocation directory and a fresh child calibration directory, preserving the failed child and its logs. On interruption, the launcher stops scheduling new attempts, sends SIGINT to the active coordinator, waits for its checkpoint cleanup, then uses SIGTERM and SIGKILL only if the configured grace periods expire. `completed_with_failures`, `failed`, and `interrupted` return nonzero. Only zero-exit child runs with matching identities, finite training loss, selected parameters and successful final strict evaluations are ranked.

## Logs and tests

In detailed mode the coordinator prints episode IDs, loss contributions, gradients, elapsed time and worker log directories. Concise and quiet modes retain those records without printing routine episode progress. Each request has separate `request.json`, `stdout.log`, `stderr.log`, and `worker/` records with detailed step events/results. Dataset runs save validated fingerprints, optimizer snapshots, combined events, per-episode fixed contacts and shared material selections. Infrastructure/protocol failures abort rather than masquerading as numerical optimizer rejections. Optional `--worker-timeout-s` terminates and reaps a worker; no timeout is imposed by default.

```bash
python3 -m experiments.differentiable_mpm.check --reference-policy frozen --quick
python3 -m experiments.differentiable_mpm.check --reference-policy frozen --only test_multi_trajectory
```

The numerical multi-episode test uses actual corrected CPUf64 simulation/AD and a subprocess comparison, but mocks the raw-data preparation boundary for the tiny synthetic fixtures. It does not qualify missing real-episode input preparation or full-horizon GPU fitting. See `VALIDATION.md` for actual pass/failure evidence.

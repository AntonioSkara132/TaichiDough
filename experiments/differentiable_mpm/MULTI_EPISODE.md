# Shared-material dataset calibration

`calibrate_dataset` fits one shared material across explicitly selected episodes. The single-episode `calibrate` command remains available unchanged. New source is confined to this experiment.

## Scope and current readiness

The first setup fits E, Poisson ratio, viscosity and the two principal-stretch limits. Tool retention is fixed at **1**, tool absorption and stickiness at **0**. This removes the existing retention damping; it is approximately frictionless tool contact, **not a new Coulomb-friction model**. Each episode keeps its own mass, reconstruction, camera, tool trajectories and floor parameters. The user's separate floor/contact investigations do not change these inputs automatically.

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

A dataset manifest uses `taichidough/differentiable-dataset/v1` and declares:

- `name`.
- `shared_parameters.initial`: exactly the five material values, including fixed material values if only a subset is fitted.
- `shared_parameters.fit` and optional `bounds`.
- Optional `tool_contact`, currently restricted to retention1/absorption0/stickiness0.
- Ordered `episodes`, each with `id`, `config`, `membership` (`training` or `validation`), positive `weight` (default1), `scored_window`, and optional `path_overrides`.

`scored_window` contains explicit `start_frame`, `end_frame` and `stride`. Episodes always replay from reconstruction frame0 to the scored endpoint; their state is never initialized from an observed intermediate deformation. Whole-episode membership is independent of the legacy config's within-episode training/validation windows. Scored frame indices reach the differentiable loss, export provenance and strict evaluator consistently.

Configuration references resolve relative to the dataset manifest. Inputs written inside an existing episode config keep the existing repository-relative path rules. Manifest path overrides resolve relative to the manifest. CLI overrides use explicit `--path EPISODE.INPUT=PATH` and resolve relative to the current directory. No automatic machine-path rewriting occurs.

The loader rejects incompatible shared physics/material/loss/runtime settings, duplicate IDs/resolved recordings/declared sequence fingerprints, invalid bounds and malformed windows. Preflight additionally rejects identical verified recording fingerprints under different episode IDs. Inventory's duplicate-variant warnings still require deliberate choice: different processed versions can have different content fingerprints despite describing the same physical trial.

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

## Logs and tests

The coordinator prints episode IDs, loss contributions, gradients, elapsed time and worker log directories. Each request has separate `request.json`, `stdout.log`, `stderr.log`, and `worker/` records with detailed step events/results. Dataset runs save validated fingerprints, optimizer snapshots, combined events, per-episode fixed contacts and shared material selections. Infrastructure/protocol failures abort rather than masquerading as numerical optimizer rejections. Optional `--worker-timeout-s` terminates and reaps a worker; no timeout is imposed by default.

```bash
python3 -m experiments.differentiable_mpm.check --reference-policy frozen --quick
python3 -m experiments.differentiable_mpm.check --reference-policy frozen --only test_multi_trajectory
```

The numerical multi-episode test uses actual corrected CPUf64 simulation/AD and a subprocess comparison, but mocks the raw-data preparation boundary for the tiny synthetic fixtures. It does not qualify missing real-episode input preparation or full-horizon GPU fitting. See `VALIDATION.md` for actual pass/failure evidence.

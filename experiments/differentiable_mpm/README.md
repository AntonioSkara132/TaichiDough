# Differentiable TaichiDough

An isolated experimental version of TaichiDough, with reverse-mode trajectory derivatives and gradient-based joint parameter fitting. **This task edits only this experiment directory.** It contains exact-byte reference snapshots and reads existing data and meshes without changing them.

**Frozen baseline:** the simulator snapshot has SHA-256 `6653543ac16c8fcbdc111c73ebaa2c5e2d8c1cdc899e3750dce539b2730a2f07`, G2P affine factor `4 * inv_dx`, and the original truncating stencil. The production implementation uses the corrected affine factor `4 * inv_dx * inv_dx` and nonnegative B-spline weights on a padded floor grid, changed independently in another session. Those corrections are outside this experimental solver and its reference comparisons. Experimental fitted parameters therefore apply to the preserved pre-correction implementation, not automatically to the corrected production model.

The default strict reference policy stops if live sources differ. To deliberately run this preserved version after such a change, add **`--reference-policy frozen`** to `calibrate`, `check`, or `real_parity` commands. Frozen mode requires a byte-verified baseline copy for each changed source, preserves the original expected hashes, and records live-source differences. Sources without snapshots must still match. Snapshot corruption is always an error. No working source is overwritten. The six additional helper copies in `reference_snapshots.json` were recovered from Git only after their bytes matched the original manifest hashes; that supplemental file cannot replace expected hashes. Strict evaluation executes those verified helper copies, including their lazy imports.

## Validation status

See [VALIDATION.md](VALIDATION.md) for measured results and limitations. Component and short-trajectory gradients pass finite-difference tests, and the full recorded forward replay passes position tolerances. A real CPU f32 backward pass through 334 steps now completes with finite gradients and exact checkpoint recomputation. A fresh short Vulkan test also passes its CPU comparisons. Full-horizon real backward/parameter fitting is still under qualification.

The synthetic example explicitly uses a 10 mm prediction visibility temperature with fixed 2 mm-generated targets. Current-source checks pass for all five coordinate gradients and two combined directions, with unchanged target hashes. All eight synthetic tests pass; four optimizer updates reduce training loss by 33.18% and also improve the unused excitation. The previous 2 mm predictor had poor optimization despite accurate local derivatives. That failure remains recorded, and the >1% short-fit assertion has not been weakened.

The commands below explicitly select the preserved baseline. Use `--reference-policy strict` when checking that production sources still equal that baseline.

## What is differentiated

The state is particle position `x`, velocity `v`, affine velocity `C`, deformation `F`, and plastic-volume history `Jp`. The solver separates material update, P2G, grid response, G2P, and particle contact into forward/reverse stages. Polar rotation and principal-stretch projection use composite spectral adjoints that remain well-defined at repeated positive stretches.

The forward physics follows `reference/taichi_viscoelastic_mpm_scene.py`:

- Fixed-corotated elasticity with E and Poisson ratio.
- Additive `viscosity * (C + C.T)` stress.
- Optional principal-stretch-clamp plasticity, optional Jp accumulation and hardening.
- Recorded rigid SDF tools, floor response, and existing damping/order of operations.
- Existing interpolation and G2P affine scaling, including the reference's `inv_dx * dpos` factor.

The experiment does not change the model to von Mises plasticity or Coulomb friction. `tool_retention` and `floor_retention` name the existing velocity multipliers more accurately: smaller values remove more remaining relative velocity. Absorption/stickiness remain fixed because their products with retention cannot be identified separately by this law.

Parameters are differentiated through every replay step. Host checkpoints and a configurable device history bound memory. At segment boundaries, **all five state adjoints are carried backward**, rather than detached. Grid/material/contact intermediates are recomputed. Checkpoint endpoint and contact/yield-count disagreement rejects a backward evaluation.

### Partial-observation training loss

The new, versioned objective combines calibrated local splats, front-biased depth, initial-referenced depth-change residuals, positive observed-foreground coverage, and directed distance to visible predicted coordinates. Its position gradients are explicitly injected into the simulation adjoint. Unknown pixels are not treated as known empty space; hidden volumetric particles are not symmetrically compared with the observed exterior. Missing/invalid support is an error, not a zero-loss frame.

Version 2 blends each fixed-support depth residual with its missing-depth penalty using predicted coverage. This removes the version-1 jump when an observed pixel loses its final splat. Version 1 remains explicitly selectable for reproducing early experiments; their loss values and optimization histories must not be compared as if the objectives were identical.

The existing strict evaluator remains separate and unchanged. Its numerical loss values are **not interchangeable** with the differentiable training loss.

## Installation and location

Use the repository's existing Python environment. No new global dependencies are required; this implementation uses the installed Taichi 1.7.4, NumPy and SciPy, with the existing Torch input loader.

Run commands below from the TaichiDough repository root. New outputs belong under `experiments/differentiable_mpm/runs/`. Existing run directories are never deleted or automatically overwritten. Choose a new directory for a new experiment; `--resume` is only for an exactly matching fit.

## Commands

### 1. Verify data, geometry, and the requested backend

```bash
python3 -m experiments.differentiable_mpm.calibrate validate \
  --config experiments/differentiable_mpm/configs/episode18_viscoelastic.json \
  --reference-policy frozen \
  --backend cpu \
  --output-dir experiments/differentiable_mpm/runs/episode18_preflight
```

This verifies hashes, mass/volume/density, camera, frame/control schedules and signed collision solids. Add `--no-runtime` for input-only validation; that does not verify any backend. Initialization itself is not evidence that forward/backward kernels execute.

### 2. Run component and trajectory checks

```bash
python3 -m experiments.differentiable_mpm.check --reference-policy frozen --quick --regressions
python3 -m experiments.differentiable_mpm.check --reference-policy frozen
```

The runner executes test files in separate processes and writes logs plus `result.json` in a new experiment run. `--only test_spectral test_trajectory` selects specific tests. Real full-replay and GPU qualification are separate from this unit suite:

```bash
python3 -m experiments.differentiable_mpm.real_parity \
  --reference-policy frozen --end-frame 60 \
  --output-dir experiments/differentiable_mpm/runs/real_forward_parity_new
python3 -m experiments.differentiable_mpm.backend_check \
  --backend vulkan --precision f32 --compare-cpu
```

The first command compares the complete recorded forward replay with the frozen original, using separate processes and fixed source snapshots. The second actually runs a small contact/plasticity simulation and loss backward pass; substitute `cuda` on a working NVIDIA system. A successful small GPU check does not establish full-horizon GPU recomputation accuracy.

### Synthetic joint-fit example

```bash
python3 -m experiments.differentiable_mpm.synthetic \
  --backend cpu --precision f64 --iterations 48 \
  --output-dir experiments/differentiable_mpm/runs/synthetic_joint
```

This fits all five material parameters to partial camera observations of a declared prestrained/shearing patch and scores a separate unused excitation. Prediction temperature defaults to `--visibility-temperature-m 0.01`; observations are generated with fixed `--target-visibility-temperature-m 0.002`. Set the prediction temperature to `0.002` explicitly to reproduce the poorly conditioned earlier objective. These settings are recorded in run identity and must match on resume.

The test checks loss reduction and gradient propagation, not unique parameter recovery. Parameters can compensate for the difference between prediction and target rendering; the selected loss can therefore be lower than the loss at the generating material parameters. Real data uses its own declared loss settings and is not changed by these synthetic defaults.

### 3. Compute a real replay gradient before fitting

```bash
python3 -m experiments.differentiable_mpm.calibrate gradient \
  --config experiments/differentiable_mpm/configs/episode18_viscoelastic.json \
  --reference-policy frozen \
  --backend cpu --end-frame 2 --segment-length 64 \
  --output-dir experiments/differentiable_mpm/runs/episode18_gradient_short
```

The short command still uses all 24,000 particles and the fixed grid/timestep. Omit `--end-frame 2` for the complete training horizon. `--finite-difference` adds coordinate checks at multiple perturbation sizes; it is a test, not the optimizer. Hard contact and visibility switches can make finite differences inconsistent across perturbation sizes.

### 4. Fit parameters using simulation gradients

Pure-viscoelastic E/ν/viscosity fitting:

```bash
python3 -m experiments.differentiable_mpm.calibrate fit \
  --config experiments/differentiable_mpm/configs/episode18_viscoelastic.json \
  --reference-policy frozen \
  --backend cpu --iterations 20 \
  --output-dir experiments/differentiable_mpm/runs/episode18_joint_viscoelastic
```

To fit all five material parameters, explicitly activate stretch-clamp plasticity using its separate configuration:

```bash
python3 -m experiments.differentiable_mpm.calibrate fit \
  --config experiments/differentiable_mpm/configs/episode18_stretch_clamp.json \
  --reference-policy frozen \
  --backend cpu --iterations 20 \
  --output-dir experiments/differentiable_mpm/runs/episode18_joint_stretch_clamp
```

That configuration changes the enabled plasticity model relative to the recent pure-viscoelastic runs; it is an explicit experiment, not an automatic reinterpretation of those results. Fitting inactive plastic limits is refused.

For a short integration run, add `--end-frame 2 --iterations 2 --no-evaluate`. For an existing matching fit, repeat its command with `--resume`; iterations are additional attempts. Source, fixed settings, observations, parameter transformations, backend, precision and optimizer settings must match. After code changes, create a fresh run.

Projected Adam receives physical simulation gradients, transforms them into bounded coordinates, and uses checked/backtracked proposals. Invalid candidates leave the accepted parameter state intact. The default `persistent-v1` retains reductions to the proposal rate. Optional `--learning-rate-policy recover-v1 --learning-rate-growth 1.25` allows the next proposal rate to recover toward its initial value; every proposal still requires finite gradients and the same sufficient-decrease test. This can improve efficiency after temporary backtracking, but is not a fix for every visibility transition. Rate policy and growth are part of exact-resume identity.

`budget_exhausted` means the requested work completed, not that calibration converged. A stalled/invalid optimizer is reported separately.

By default a fit selects on training frames 1–60 only, then exports the frozen best parameters for unchanged strict evaluation. Validation replays from frame 0 through 97 and scores only 61–97. It never initializes from an observed held-out deformation. `--no-evaluate` skips these independent checks explicitly.

### 5. Replay and independently evaluate a frozen selection

```bash
python3 -m experiments.differentiable_mpm.calibrate evaluate \
  --config experiments/differentiable_mpm/configs/episode18_viscoelastic.json \
  --reference-policy frozen \
  --parameters experiments/differentiable_mpm/runs/episode18_joint_viscoelastic/selected_parameters.json \
  --backend cpu --split validation \
  --output-dir experiments/differentiable_mpm/runs/episode18_held_out
```

The `replay` action exports simulation data without running the strict evaluator. `evaluate` requires a frozen parameter file. Every expected replay and scoring frame must be present.

### Remote `/mnt` checkout

The example configurations use repository-relative files and explicit local episode paths. Dataset layouts differ between the two machines. Override the episode and calibration paths without editing the original manifest:

```bash
python3 -m experiments.differentiable_mpm.calibrate validate \
  --config experiments/differentiable_mpm/configs/episode18_viscoelastic.json \
  --reference-policy frozen \
  --path episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --path calibration=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla/scene_calibration_v2.json \
  --backend cuda \
  --output-dir experiments/differentiable_mpm/runs/episode18_cuda_preflight
```

Run from `/mnt/Data/studenti/antonio_skara/TaichiDough`. If the reconstruction is stored elsewhere, use `--path initial_particles=...` and `--path reconstruction_metadata=...` too. Overrides still have to match the recorded hashes; matching filenames alone are insufficient. Do not replace expected hashes merely to suppress an unexplained mismatch.

CUDA is requested with fallback disabled. The previous `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE` is a driver/library initialization failure; this implementation does not repair it. CPU, Vulkan and CUDA execution evidence must be reported separately.

## Run records

- `run_manifest.json`: complete run identity, numerical configuration and input/code/dependency fingerprints.
- `resolved_inputs.json`: resolved paths, original/derived metadata hashes, camera, observations and controls.
- `memory_estimate.json`: particle/checkpoint/grid allocation estimates, with uncounted overhead named explicitly.
- `events.jsonl`: progress, objective attempts, accepted/rejected optimizer updates, failures and resumptions.
- `optimizer_state.json`: atomically saved consistent coordinates, moments, objective records and history.
- `last_objective.json`: most recent valid objective and physical gradients.
- `selected_parameters.json`: training-selected fixed candidate and scoring frames.
- `result.json`: action/optimization results; independent evaluations use separate directories.

The run lock prevents concurrent writers and is released by the OS if the process exits. A stale lock filename is not evidence that a process still holds the lock.

## Numerical limitations and interpretation

1. Hard-contact, integer-stencil and visibility selections are piecewise. These derivatives do not differentiate across creation/removal of a contact or a change of visible association.
2. The composite spectral VJP differentiates the intended polar/clamp map. Taichi's finite-iteration SVD has measurable forward approximation error, especially apparent in very small fp64 finite differences. The tests quantify it without changing reference forward values.
3. Near-singular/inverted states outside the spectral derivative's supported domain are rejected explicitly.
4. GPU atomic accumulation can change recomputed values or branches. Endpoint and branch-count checks detect some disagreement; equal aggregate counts alone do not prove identical per-particle decisions.
5. Simultaneously fitting five parameters does not guarantee they are separately identifiable. Motion must excite elasticity, rate response, and both plastic bounds. Report held-out behavior and parameter ambiguity, not just a small training loss.
6. Parameters are conditional on fixed mass, volume, geometry, contact padding, timestep, damping and the existing constitutive/contact laws. The previous ≈131 kPa candidate was conditional on ν=0.3, with a broad minimum, not a uniquely accepted material constant.

## Implementation references

- [Taichi differentiable programming](https://docs.taichi-lang.org/docs/differentiable_programming)
- [DiffTaichi examples](https://github.com/taichi-dev/difftaichi)
- [Differentiable physics-based system identification for robotic manipulation of elastoplastic materials](https://arxiv.org/abs/2411.00554)

The experiment adapts the project's own frozen implementation and independently implements the matrix derivatives/checkpointing. Third-party simulator source has not been copied into the differentiable implementation.

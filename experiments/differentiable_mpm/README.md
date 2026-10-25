# Differentiable TaichiDough

An isolated experimental version of TaichiDough, with reverse-mode trajectory derivatives and gradient-based joint parameter fitting. **This task edits only this experiment directory.** It contains exact-byte reference snapshots and reads existing data and meshes without changing them.

**Physics-version update:** configuration now defaults to `corrected-v1`, with `legacy-v1` available explicitly for reproducing old runs. Corrected physics uses the G2P affine factor `4 * inv_dx * inv_dx` with metric `dpos`, floor-based stencil indices, and a padded floor grid. Local qualification passes: 18 solver/transfer tests, both-version forward-reference comparisons, 14 synthetic/trajectory tests, 101 host-integration tests, and the small Vulkan/CPU forward/backward comparison. Full-episode corrected calibration and CUDA qualification remain unverified. See [VALIDATION.md](VALIDATION.md) for corrected-model evidence; the older numerical results below describe legacy physics. Do not reuse earlier fitted parameters as calibrated values for the corrected model.

The original simulator snapshot remains byte-identical at SHA-256 `6653543ac16c8fcbdc111c73ebaa2c5e2d8c1cdc899e3750dce539b2730a2f07`. A separate corrected reference is pinned to `d33f0aec3952fa72282cd181757f678b2e2a48e5b51016c23d40fd05d6a3ac3e`. Corrected and legacy results must record their selected physics version and reference.

The default strict reference policy stops if live sources differ. To deliberately run this preserved version after such a change, add **`--reference-policy frozen`** to `calibrate`, `check`, or `real_parity` commands. Frozen mode requires a byte-verified baseline copy for each changed source, preserves the original expected hashes, and records live-source differences. Sources without snapshots must still match. Snapshot corruption is always an error. No working source is overwritten. The six additional helper copies in `reference_snapshots.json` were recovered from Git only after their bytes matched the original manifest hashes; that supplemental file cannot replace expected hashes. Strict evaluation executes those verified helper copies, including their lazy imports.

## Episode 18 table-aligned inputs

The robust table estimate has been applied through a new rigid calibration, with regenerated particles and mass/volume metadata under `data/episode18_table_aligned_v1/`. The new configuration uses non-adhesive Coulomb tools. Input/domain checks passed; no simulation or fit has run on this geometry. The derived density is unusually high and needs a mass/volume sanity check. See [TABLE_ALIGNMENT.md](TABLE_ALIGNMENT.md) for files, limitations and the CUDA calibration command.

## Validation status

See [VALIDATION.md](VALIDATION.md) for measured results and limitations. Component and short-trajectory gradients pass finite-difference tests, and the full recorded forward replay passes position tolerances. A real CPU f32 backward pass through 334 steps completes with finite gradients and exact checkpoint recomputation on the stable-normalization revision. The optional serial-P2G revision passes all 197 cases in the complete local test/regression run, two additional segmented-trajectory tests, and short Vulkan forward/backward/replay checks. Full-horizon real backward/parameter fitting is still under qualification.

The synthetic example explicitly uses a 10 mm prediction visibility temperature with fixed 2 mm-generated targets. Current-source checks pass for all five coordinate gradients and two combined directions, with unchanged target hashes. All eight synthetic tests pass; four optimizer updates reduce training loss by 33.18% and also improve the unused excitation. The previous 2 mm predictor had poor optimization despite accurate local derivatives. That failure remains recorded, and the >1% short-fit assertion has not been weakened.

Reference policy and physics version are independent. `--reference-policy frozen` permits verified helper snapshots when live sources differ; it does not select legacy physics. Use `--physics-version legacy-v1` explicitly when reproducing the historical numerical results. The updated defaults select corrected physics, whose qualification status is stated above.

## Learned reduced-order dynamics

A separate fixed-particle POD model learns full-state updates conditioned on tool
motion and material parameters. Collection, training, differentiable inference and
held-out evaluation commands are in [REDUCED_ORDER.md](REDUCED_ORDER.md).
The first check uses small synthetic Taichi contact trajectories; calibrated
real-dough accuracy and long-horizon stability are not established.

## What is differentiated

The state is particle position `x`, velocity `v`, affine velocity `C`, deformation `F`, and plastic-volume history `Jp`. The solver separates material update, P2G, grid response, G2P, and particle contact into forward/reverse stages. Polar rotation and principal-stretch projection use composite spectral adjoints that remain well-defined at repeated positive stretches.

Material and contact laws are shared by both physics versions:

- Fixed-corotated elasticity with E and Poisson ratio.
- Additive `viscosity * (C + C.T)` stress.
- Optional principal-stretch-clamp plasticity, optional Jp accumulation and hardening.
- Recorded rigid SDF tools, floor response, and existing damping/order of operations.
- Quadratic B-spline interpolation. `corrected-v1` reconstructs the affine velocity gradient with `4 * inv_dx**2 * dpos`; `legacy-v1` retains `4 * inv_dx * dpos` and the truncating stencil.

For a complete stencil sampling an affine velocity field with gradient A, corrected G2P reconstructs `C = A`; legacy G2P reconstructs `C = dx * A`. Since C also drives deformation and APIC momentum transfer, the correction changes more than viscosity. Multiplying an old viscosity estimate by a grid-dependent constant does not convert the full model or its calibrated parameters. Physical grid spacing remains `dx = 1/grid`; corrected allocation is `(grid+1)^3` with logical indices -1 through grid-1.

Tool contact is selected explicitly with `simulation.tool_contact_model`:

- `retention-v1` is the default and preserves existing results. `tool_retention` multiplies the complete remaining relative velocity, so a small value can make dough follow a separating tool while the SDF contact test remains active.
- `coulomb-v1` is unilateral, non-adhesive velocity-level contact. Inward normal relative velocity is removed and supplies the Coulomb limit `mu * delta_v_n` for tangential slip. Separating relative velocity is returned unchanged, including its tangential component. Set `simulation.tool_contact_model` to `"coulomb-v1"` and `simulation.tool_friction_coefficient` to a fixed nonnegative coefficient. This mode requires zero `tool_contact_absorption` and `tool_stickiness`; `tool_retention` is not reinterpreted as friction and is inactive in this mode.

- `coulomb-adhesive-v1` applies Coulomb contact followed by a `tool_stickiness` fraction of velocity matching in all directions, including withdrawal inside the contact region.
- `coulomb-bounded-adhesive-v2` uses `tool_adhesion` in m/s². With `L = tool_adhesion * dt`, normal separating speed `vn >= 0` loses `L*vn/sqrt(vn² + L² + epsilon²)`; inward normal speed is removed completely. The magnitude of either normal correction supplies the tangential Coulomb budget `mu * normal_change`.
- **`coulomb-bounded-shear-adhesive-v3` is opt-in** and preserves v2's normal response. Its tangential speed budget is `mu * normal_change + tool_adhesion * dt`, using the same adhesion parameter with no additional fitted value. Tangential relative velocity is multiplied by `max(0, 1 - budget/sqrt(|vt|² + epsilon²))`, so the response cannot reverse slip. It can resist sideways motion even at zero normal relative velocity. `epsilon = 1e-8 m/s` keeps zero-slip adjoints finite. Setting adhesion to zero gives v2's response.

Both bounded models require zero `tool_contact_absorption` and `tool_stickiness`. Select v3 with `"tool_contact_model": "coulomb-bounded-shear-adhesive-v3"` in the simulation configuration; `tool_adhesion` can be fixed there or fitted through the existing physical-parameter field. Explicit physical values take precedence over simulation values. Existing model identifiers, saved configurations, and defaults are unchanged; the frozen reference simulator rejects v3 rather than substituting another law.

In v3, normal and tangential adhesion have **independent caps**, not a single cap on their combined vector. Adhesion is an effective acceleration, not interface stress in Pa: there is no contact-area calibration, persistent bond, damage, or separation-energy law. Resistance ends when the point leaves the contact region; smooth normal adhesion is not an exact sticking constraint or a sharp acceleration threshold. Grid and particle contact can both apply corrections during one substep, so these are per-response limits, not a guaranteed bound on a particle's total substep acceleration.

By default, the same response is used for the closest-tool grid contact and sequential particle projection. Angular collider velocity in the particle path is evaluated at the projected position. Floor contact is selected independently. Coulomb contact uses normal velocity corrections after stress, gravity and grid forces as its normal-impulse proxy; it is not a full complementarity solve and can underestimate static friction when resting contact has no inward velocity in that step. Active friction and adhesion parameters support adjoint differentiation within the selected contact branches.

### Grid-only Dirichlet contact

`simulation.tool_grid_contact_mode` defaults to `"inherit"`, which uses `tool_contact_model` on the grid. The opt-in `"dirichlet-v1"` mode requires SDF tools and sets each selected grid node's full velocity to `tool_velocity + tool_omega.cross(node_position - tool_center)`. This occurs before G2P, so particle velocity and affine velocity receive the constrained grid field. It is an instantaneous rigid-velocity constraint, not a port of DPSI's next-pose collision timing or finite-displacement tool velocity.

`simulation.tool_grid_contact_padding` is an optional finite nonnegative distance in metres. Its default `null` inherits `tool_contact_padding`; setting it changes only grid contact selection. Particle velocity response and positional push-out retain the existing `tool_contact_model` and `tool_contact_padding`. Neither new option is a fitted material parameter. To test a half-cell grid band at grid 128, merge these fields into an existing simulation configuration, keeping its other settings unchanged:

```json
{
  "grid": 128,
  "tool_collision": "sdf",
  "tool_grid_contact_mode": "dirichlet-v1",
  "tool_grid_contact_padding": 0.00390625,
  "tool_contact_padding": 0.0013
}
```

Grid selection retains the strict `distance < padding` test, SDF-volume validity checks, and smallest signed-distance tool selection (the first tool wins ties). The full constraint applies during approach, sliding and separation while the node remains selected. There is no persistent bond or additional physical release criterion. A wider band also extends the region in which tool motion is imposed; it does not guarantee particle nonpenetration. Existing ordering remains floor response, tool response, then domain constraints, so incompatible tool/table constraints are not solved jointly.

The response remains in the Taichi reverse computation. Within a fixed contact/selected-tool branch, gradients propagate through rigid tool velocity, tool center and G2P. The overwritten incoming grid velocity has zero derivative. Grid friction/adhesion parameters do not affect the prescribed velocity, but those parameters still act through particle contact and other steps. Enable tool-control adjoints with `Stepper(..., control_gradients=True)`. Contact activation, nearest-tool changes and release are discrete decisions, not globally smooth functions; finite-difference checks must stay within the same selection. There is no detached or straight-through gradient approximation.

The frozen reference simulator rejects Dirichlet mode or unequal grid/particle padding. Both settings are recorded in resolved configurations and fingerprints; changed configurations require new runs rather than resuming an existing fit. Full recorded-motion accuracy and GPU validation of this mode must be checked separately.

Parameters are differentiated through every replay step. Host checkpoints and a configurable device history bound memory. At segment boundaries, **all five state adjoints are carried backward**, rather than detached. Grid/material/contact intermediates are recomputed. By default, checkpoint endpoint and contact/yield-count disagreement rejects a backward evaluation.

### Triangle-based tool SDFs

The top-level `tool_sdf_builder` setting defaults to `"legacy-voxel-v1"`. Select `"triangle-distance-v1"` explicitly in a new configuration to use actual triangle distances with Open3D (tested implementation targets 0.17.0). Open3D is imported only for this mode. Original collision meshes, datasets and preserved builders are not rewritten.

The new builder requires a valid, consistently oriented closed solid; edge closure alone is insufficient. Intersection validation enumerates overlapping triangle bounding boxes and uses exact predicates on the stored coordinates, allowing only the expected shared vertex or edge between adjacent faces. This avoids treating Open3D's near-coplanar false positives as real intersections, without accepting genuine overlaps. Intersecting triangles, ambiguous overlapping components and invalid geometry are rejected rather than repaired automatically. Blade-only collision assets may omit the disconnected OptiTrack mount while retaining blade coordinates and all tracking transforms. Such extraction is not itself a validity certificate.

New field bounds cover the larger particle/grid contact padding plus an interpolation margin. Enlarging bounds does not dilate the mesh. Stored values are float32 signed distances and outward normals; the solver still interpolates distance and normals separately and normalizes the latter. Exact samples alone cannot represent an arbitrarily thin blade: interpolation may lose its negative interior or cancel opposite normals.

Preparation first tests the candidate resolution by querying only the grid corners needed by deterministic mesh-normal profiles. A failed candidate is rejected before allocating its dense arrays. A passing candidate is built and the stored interpolated field is checked again before use. Qualification policy `triangle-interpolation-v2` uses 0.1 mm distance and geometric boundary-offset limits, a 10% physical-thickness limit, and a 10-degree limit for every evaluated normal. It compares all genuine mesh crossings, including concave re-entry, and refines field roots before measuring their distance to the mesh. Ray displacement remains a diagnostic, not a normal boundary-distance measurement. The thickness limit applies only where all incident exit-face normals are within 5 degrees of the opposite entry normal; other chord-length errors are reported separately. Normal reports distinguish regular interpolation support from support spanning strongly varying exact normals, but both remain subject to the 10-degree limit. Reports describe sampled coverage, not a proof at every position, and do not establish physical measurement accuracy. Failed fields are rejected, including on cache reads. A higher resolution is not accepted without these checks.

The top-level `tool_sdf_qualification` setting selects the acceptance profile for `triangle-distance-v1`. `"strict-v2"` (default) is the policy above. `"grid-relative-v1"` sets the distance and boundary limits to 5% of the MPM grid spacing (1.04 mm at grid 48) and applies the 10-degree normal limit only where the exact normals at all eight interpolation corners agree within 1 degree; normals at stencils spanning an edge, corner or hole rim are still reported. It also accepts a missing interior or a crossing-count mismatch on a profile whose deepest material lies within the distance limit of the surface and, for a thickness profile, within 5% of its thickness. A line running just inside a wall passes; a plate that disappears does not, because its depth is half its thickness. The profile and its resolved limits are part of the cache and qualification identity, and the setting is serialized, so configuration fingerprints change. `prepare_bag0_dynamics.py` writes these settings from the optional `tool_sdf` block of a sources file (see `configs/bag0_dynamics_sources_v2.json`).

Cache identity includes the new algorithm, implementation/dependency identity, geometry, transforms, bounds inputs and qualification policy. Legacy cache keys remain unchanged. Resolved configurations now serialize the builder setting, so their fingerprints change even when the legacy default is selected; do not resume earlier runs under a different fingerprint. Dataset loading requires consistent builder, SDF resolution and mesh scale across members.

Extract the pinned shared blades without modifying their triangle records:

```bash
/usr/bin/python3 -m experiments.differentiable_mpm.prepare_blade_collision_meshes \
  --output-dir experiments/differentiable_mpm/runs/blade_only_geometry_v1
```

The output directory must be new. The command writes a manifest, retained triangle indices, exact intersection diagnostics and before/after views. Exit status 2 means the extracted geometry is not qualified; do not use that manifest for the new builder. The current shared assets retain exact intersections after mount omission, including adjacent-face overlaps. The new mode rejects their `not-qualified` manifest; extraction alone does not produce usable SDFs.

`prepare_valid_blade_meshes` derives separate corrected assets using pinned local edge flips: 16 face pairs for UR and 15 for Kinova. Every vertex and the raw bounds remain unchanged. Exact per-patch displacement bounds limit the resulting Hausdorff difference to 13.4055 nm and 14.3653 nm respectively. It checks the saved meshes and their production-transformed float32 geometry before marking the mesh qualified. Field qualification is still required:

```bash
/usr/bin/python3 -m experiments.differentiable_mpm.prepare_valid_blade_meshes \
  --output-dir experiments/differentiable_mpm/runs/valid_blade_geometry_v1
```

Focused tests are registered as `test_triangle_sdf`, `test_mesh_validation`, `test_prepare_blade_collision_meshes`, `test_sdf_qualification` and `test_sdf_builder_integration` in the experiment test runner. They include exact adjacent-face overlap checks, independent analytic distances, rejected subvoxel slabs, cache separation, and actual f32/f64 sampler finite differences. Sampler diagnostic output is collected in a separate kernel: writing a non-gradient normal diagnostic inside the differentiated probe triggered incorrect f32 adjoints with the installed Taichi version.

For a single valid mesh, `python3 -m experiments.differentiable_mpm.qualify_tool_sdf --help` lists standalone construction and qualification options. Its scale/rotation/translation must explicitly match the intended coordinate frame; raw-STL field output is not automatically a tool-frame field. Normal experiment preparation applies the existing tool-frame transforms through `data.py`.

On the shared retriangulated blades with a 3.90625 mm contact band, v2 sparse preflight rejects every tested cubic resolution (128, 256, 384, 512, 768 and 1024). At 512, both tools pass crossing-count, geometric boundary-offset and eligible physical-thickness checks; worst boundary offsets are 0.08745 mm (UR) and 0.04948 mm (Kinova), and eligible thickness errors are 0.982% and 0.628%. Regular-support normal checks pass, but varying-support normal checks and several distance probes fail. At 768, sampled distance errors pass the 0.1 mm limit for both tools, but normal checks still fail and UR has one crossing-count failure. These results do not qualify a field for use. No actual-tool dense field was built or accepted; the validated meshes and construction code are separate deliverables from a qualified runtime field.

This changes collision geometry construction, not the contact law, particle push-out or MPM transfer. No improvement in full-motion simulation is implied by passing geometry checks. Subsequent penetration measurements should use independent mesh-based checks, including swept crossings: an endpoint outside the tool does not rule out passage through it during the step.

### Partial-observation training loss

The new, versioned objective combines calibrated local splats, front-biased depth, initial-referenced depth-change residuals, positive observed-foreground coverage, and directed distance to visible predicted coordinates. Its position gradients are explicitly injected into the simulation adjoint. Unknown pixels are not treated as known empty space; hidden volumetric particles are not symmetrically compared with the observed exterior. Missing/invalid support is an error, not a zero-loss frame.

Version 2 blends each fixed-support depth residual with its missing-depth penalty using predicted coverage. This removes the version-1 jump when an observed pixel loses its final splat. Version 1 remains explicitly selectable for reproducing early experiments; their loss values and optimization histories must not be compared as if the objectives were identical.

The existing strict evaluator remains separate and unchanged. Its numerical loss values are **not interchangeable** with the differentiable training loss.

### Selectable DPSI and EMPM losses

`loss.version` also selects DPSI endpoint Chamfer/assignment objectives, an EMPM offline point-and-track objective, or an explicitly EMPM-inspired segmentation objective. See [LOSS_OPTIONS.md](LOSS_OPTIONS.md) for equations, required target files, partial-observation limitations, optimizer scaling, verification results, and new Episode18 examples. The partial v2 loss remains the default. Full tracking, volume and segmentation modes require verified additional observations; missing data does not trigger an automatic geometry-only fallback.

## Installation and location

Use the repository's existing Python environment. No new global dependencies are required; this implementation uses the installed Taichi 1.7.4, NumPy and SciPy, with the existing Torch input loader.

Run commands below from the TaichiDough repository root. New outputs belong under `experiments/differentiable_mpm/runs/`. Existing run directories are never deleted or automatically overwritten. Choose a new directory for a new experiment; `--resume` is only for an exactly matching fit.

## Multi-episode calibration

The `calibrate_dataset` command fits shared physical parameters over an explicit episode manifest, with equal-episode or declared weighted objectives and one numerical subprocess at a time. Dataset schema v1 remains material-only with its existing fixed-contact behavior and fingerprints. Schema v2 can also share and fit floor retention, tool friction, and tool stickiness. Joint friction-and-stickiness fitting requires SDF collision, `coulomb-adhesive-v1`, and zero absorption. See [MULTI_EPISODE.md](MULTI_EPISODE.md) for manifests, inventory, validation, fitting, exact resume and remote commands.

The driver is implemented and locally verified: 276 host/regression tests pass, and four actual corrected CPU f64 multi-motion tests pass. A weighted two-motion synthetic fit reduces combined training loss by 38.01%, all five material AD/FD checks pass, and held-out motion data is excluded from selection. See `VALIDATION.md` for exact evidence and limitations. The [ten-episode two-second dataset](data/ten_episode_shared_alignment_v1/README.md) is ready for input preparation and fitting under the user's shared-alignment assumption: common simulated floor y=0 and exact Episode18 tool transforms, with existing per-recording coordinate conversions and downward-filled initial reconstructions retained. Its current shared bounds are E=2,000–60,000 Pa, Poisson ratio=0.45–0.49 and viscosity=0–60 Pa.s; the existing plastic-stretch bounds remain unchanged. All ten passed routine input checks. The dataset uses a separate 1200 kg/m³ density assumption and calculated per-episode masses. Existing 250 g configurations are unchanged. Episode18's observed floor inconsistency remains unresolved. No real dataset fit has been launched.

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

With `plasticity="von-mises"`, `von_mises_yield_stress_pa` is also a physical parameter and can be listed in `fit_parameters` (or in a dataset's `shared_parameters`). The simulation setting supplies its initial value and records that lack it; the fitted value is written back to the simulation setting. It uses a log coordinate like E, default bounds 10 Pa–1 MPa, and receives gradient only from yielded particles: with radius `r = σ_y(1+ν)/E`, the projection adjoint gives `∂L/∂σ_y = (∂L/∂r)·r/σ_y`. Fitting it with any other plasticity model is refused. The value is the radius of the Hencky deviatoric stress at yield, so the uniaxial yield stress is `√(3/2)·σ_y`. Adding this twelfth parameter changes serialized parameter records and configuration fingerprints; older records still load.

For non-adhesive tool friction, create a new configuration rather than changing an existing run's identity. Its `simulation` object must explicitly contain:

```json
{
  "tool_contact_model": "coulomb-v1",
  "tool_friction_coefficient": 0.5,
  "tool_contact_absorption": 0.0,
  "tool_stickiness": 0.0
}
```

Keep `tool_retention` out of `fit_parameters` for this model. The value `0.5` is only an initial friction candidate, not a calibrated coefficient. Changing the contact model requires a fresh calibration; retention-model parameters and objective histories are not comparable as if only a parameter changed.

For a short integration run, add `--end-frame 2 --iterations 2 --no-evaluate`. For an existing matching fit, repeat its command with `--resume`; iterations are additional attempts. Source, fixed settings, observations, parameter transformations, backend, precision and optimizer settings must match. After code changes, create a fresh run.

Projected Adam receives physical simulation gradients, transforms them into bounded coordinates, and uses checked/backtracked proposals. Invalid candidates leave the accepted parameter state intact. The default `persistent-v1` retains reductions to the proposal rate. Optional `--learning-rate-policy recover-v1 --learning-rate-growth 1.25` allows the next proposal rate to recover toward its initial value; every proposal still requires finite gradients and the same sufficient-decrease test. This can improve efficiency after temporary backtracking, but is not a fix for every visibility transition. Rate policy and growth are part of exact-resume identity.

Fit logging defaults to `--fit-log concise`. It prints one candidate line with the fitted physical values, one result line with loss plus both gradient forms, and one optimizer-update line. `dL/d<name>` is the derivative with respect to the named physical parameter; `dL/du_<name>` is the derivative after the parameter transformation used by Adam. Use `--fit-log detailed` to retain replay and episode progress, or `--fit-log quiet` to print warnings, terminal status and result paths only. Complete evaluation and optimizer records remain in `events.jsonl` and `optimizer_state.json` in every mode.

Both fit commands apply physical-parameter stability stopping by default. An accepted update is stable for parameter `p` when

```text
abs(p_new - p_old) <= atol[p] + rtol * max(abs(p_old), abs(p_new))
```

All fitted parameters must satisfy the test for three consecutive accepted updates. The balanced defaults use `rtol=1e-4`, with absolute tolerances E=5 Pa, Poisson ratio=`1e-5`, viscosity=`1e-3` Pa.s, and each plastic stretch limit=`1e-5`. Rejected optimizer attempts neither increment nor reset the streak. The terminal status is `converged_parameters`; projected-gradient convergence remains `converged_gradient`. Override the rule with `--parameter-stability-updates`, `--parameter-stability-rtol`, and repeatable `--parameter-stability-atol NAME=VALUE`. Set `--parameter-stability-updates 0` to disable it.

`budget_exhausted` means the requested work completed, not that calibration converged. `evaluation_budget_exhausted` means the objective-call limit was reached. A stalled or invalid optimizer is reported separately.

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

### CUDA checkpoint-replay failures

A full remote CUDA f32 run completed the forward trajectory but rejected the first backward evaluation at step 9774 with `Contact/yield summaries differ on recomputation`. No parameter update was accepted. This is a replay-consistency failure, not evidence that the initial material parameters are physically invalid. The accompanying Taichi compiler warnings are separate from the explicit rejection.

The user's subsequent CUDA diagnostic on 2026-09-12 isolates the observed non-repeatability to P2G grid reduction: all three isolated repetitions at step 9754 restore identical inputs and produce identical material intermediates, then differ in grid mass/momentum. All three repeated segments have small state differences (maximum x component difference 1.1921e-7 m), but matching aggregate contact counts. This diagnostic reproduces the numerical differences, not the earlier contact-count flip. Parallel floating-point additions in P2G can change grid values enough to change a later hard-contact decision; readbacks can also affect GPU scheduling.

Strict consistency checks remain the default; the explicit approximate-gradient override is described below. A smaller segment length, fixed seed, `ti.sync()`, or higher precision alone does not guarantee deterministic CUDA accumulation. For a full-horizon reference gradient, use a fresh run with `gradient --backend cpu --cpu-threads 1 --precision f64 --end-frame 60`, retaining the same config and input overrides. This is a qualification step; a full real CPU gradient has not yet been verified.

Mismatch diagnostics now print expected/recomputed counts and segment bounds, persist `last_recompute_failure.json`, and write `result.json` for an invalid initial fit. Existing failed runs are preserved. Source/backend/precision changes require a new run, not `--resume`.

### Allow finite replay mismatches

`calibrate fit` and `calibrate gradient` accept `--ignore-recompute-mismatch`. It is off by default. Add it to the existing command, and use `--p2g-mode atomic` to keep parallel P2G instead of serial transfer.

When enabled, finite contact/yield-count differences, checkpoint-state tolerance violations and observation-loss recomputation differences produce warnings rather than rejecting the gradient evaluation. State, loss and gradient validity checks remain active: NaNs, infinities, invalid deformation and missing derivatives are not ignored. Numerical equations, replay tolerances, adjoint propagation and optimizer acceptance rules are unchanged.

Every mismatch is saved to `events.jsonl`; the latest ignored mismatch is saved to `last_recompute_warning.json`. The console prints the first three and every hundredth mismatch per objective, plus a summary when a fitting objective completes with mismatches. Diagnostics contain counts by kind and `replay_consistent`; gradient/fit runtime records distinguish completed backward execution from replay consistency. `backward_verified` means execution completed, not that an ignored mismatch was validated as harmless.

The flag is part of run identity and is recorded with selected parameters. Use a new run when changing it; `--resume` rejects a changed policy. This is an explicit approximate-gradient experiment, not a demonstrated fix for the full-episode CUDA gradients.

### Targeted forward replay probe

To investigate the failing CUDA segment without computing gradients or fitting parameters:

```bash
python3 -u -m experiments.differentiable_mpm.recompute_check \
  --config experiments/differentiable_mpm/configs/episode18_viscoelastic.json \
  --reference-policy frozen --backend cuda --precision f32 \
  --end-frame 60 --start-step 9728 --steps 64 --probe-step 9774 --repeats 3 \
  --output-dir experiments/differentiable_mpm/runs/cuda_recompute_9728
```

Use the same explicit `--path episode=...` and `--path calibration=...` overrides on the remote machine. The configured checkpoint length stays at 64; `--steps` specifies the diagnostic interval, not a different state-buffer layout.

The diagnostic advances the original prefix through step 9792, records the target segment, then repeats only its 64 steps three times from the saved checkpoint. It also repeats step 9774 independently from its original input state, comparing material intermediates and grid mass/momentum/velocity. Target states are streamed to disk rather than retaining the whole prefix in RAM. Output includes `stdout.log`, `comparison.json`, `result.json`, exact checkpoints, control hashes and source/runtime records.

Exit 2 means a recorded mismatch or a source change was detected. Exit 0 means only that these recorded comparisons agree: readbacks can affect GPU scheduling, matching aggregate counts do not prove equal contact identities, and this forward-only probe does not qualify gradients or full calibration.

### Optional fixed-order P2G

`calibrate`, `recompute_check` and `backend_check` accept `--p2g-mode serial`. The equivalent configuration field is `simulation.p2g_mode`; its default is `atomic`. CLI overrides take precedence. The selected mode is printed and recorded in run identity, and changing it rejects resume. No additional particle/grid buffers are allocated.

Serial mode fixes the particle iteration order for grid accumulation. Taichi also serializes the generated P2G reverse loop. Material/contact equations and exact replay checks are unchanged. Other material/contact parameter-gradient and observation-loss reductions remain as before; **serial P2G is not a claim of bitwise deterministic whole-objective gradients**.

The four targeted CPU solver tests pass, including exact repeated forward states/grids, atomic/serial forward agreement, and parameter/state finite differences with plasticity, Jp and moving SDF tools. A serial three-step Vulkan solver/loss backward check passes all 20 CPU comparisons, and a separate tiny Vulkan forward replay diagnostic matches every recorded state/material/grid byte across three repetitions. Independent CPU/Vulkan scatter tests confirm actual serialization in compiled forward/reverse code. See `VALIDATION.md` for evidence and the remaining full-episode/CUDA checks.

**Serial GPU transfer can be much slower.** A warmed isolated 24,000-particle/grid48 Vulkan probe measures 161.82 ms serial versus 1.383 ms atomic for P2G forward (117×), and 36.72 ms versus 5.352 ms for its reverse (6.86×). These are local Vulkan stage timings, not CUDA or complete-simulation estimates. This option first tests repeatability; fast deterministic parallel reduction would require a separate implementation.

Transfer the updated experimental code together, including `solver.py`. A user-provided remote run accepted the serial flag but recorded the pre-serialization solver hash `75110847…`; its passing CUDA test therefore does not qualify serial transfer. The locally qualified serial-option solver has SHA-256 `3960d45a81d7ab406f029af50669d99d7254557a36bdbe2bc50fcc9cd5cae8d8`. Check the run's recorded solver hash, not only its command line.

First run the small CUDA execution/CPU-comparison check. It requires no Episode 18 inputs:

```bash
python3 -u -m experiments.differentiable_mpm.backend_check \
  --backend cuda --precision f32 --p2g-mode serial --compare-cpu
```

Only if that passes, repeat the real forward-only diagnostic with the same mode:

```bash
python3 -u -m experiments.differentiable_mpm.recompute_check \
  --config experiments/differentiable_mpm/configs/episode18_viscoelastic.json \
  --reference-policy frozen --backend cuda --precision f32 --p2g-mode serial \
  --path episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --path calibration=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla/scene_calibration_v2.json \
  --end-frame 60 --start-step 9728 --steps 64 --probe-step 9754 --repeats 3
```

Both commands automatically choose new output directories. Keep the failed atomic runs. This diagnostic advances through 9,792 original steps before repeating the 64-step interval, so the serial cost applies to the entire prefix. Passing the probe is followed by a full `calibrate gradient` check with `--p2g-mode serial` and the same inputs/backend/precision; fitting still requires a valid complete gradient. No successful full real serial/CUDA gradient or calibration is established by the local tests.

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

# Fresh current-system audit for visual dough parameter identification

Audit date: 11 September 2026. Repository revision: `cb76169436d334c127f165ce5c3f779100bb82d4`.

Scope: independent reading of current simulator, calibration and sweep code, camera/reconstruction/replay/evaluation code, and stored JSON results. Earlier prose reports were not used as evidence. No simulation was run and no simulator code was edited. One small NumPy calculation reproduced the affine-transfer formula to check its scaling. Only this evidence file was written. Source references below are repository-relative and refer to the audited revision.

## Executive findings

1. **The idea is a valid system-identification research problem.** Depth observations and prescribed tool trajectories can support effective-parameter calibration by forward sweeps; automatic differentiation is not a requirement for science. Scientific claims depend on observability, model adequacy, independent testing and uncertainty, not on whether gradients generate candidates.
2. **The implemented model has elastic, viscous and plastic mechanisms, but the completed modulus fit does not estimate all three.** Its manifest disables plasticity, fixes viscosity to zero and searches only Young's modulus. A separate five-value viscosity runner now exists; no successful viscosity sweep was found. No dedicated plasticity/joint elasticity–viscosity–plasticity calibration runner was found in the inspected `scripts/`, `configs/` and test paths.
3. **The solver is MLS-MPM/APIC-derived, but its current G2P affine reconstruction has a concrete scaling inconsistency.** It uses physical node offsets with a coefficient appropriate to dimensionless offsets, reproducing an affine velocity gradient as `dx*A` rather than `A`. This directly affects deformation-gradient evolution, viscous stress, and plastic activation. This is an implementation issue, not a reason to reject the framework idea. It must be resolved before interpreting fitted SI material parameters or claiming standard MLS-MPM numerical behavior.
4. **The calibration result must not be misread.** The default-E score `0.799745` is a **training** score, not a held-out score. The selected-E held-out score is `0.828075`; the comparable frozen-state held-out score is `0.835860`. Thus the stored record shows a small (~0.93%) held-out aggregate improvement over frozen state, with no statistical significance established. It does not establish a held-out comparison against default E.
5. **The current local friction sweep completed all three candidates.** Training scores are 0.359191, 0.361067 and 0.361463 for retention values 0.1, 0.2 and 0.3. The older path-failed sweep remains in a different directory. The runner has no held-out stage or accepted-winner logic. This is a modest, completed training sensitivity result, not a validated physical friction estimate.
6. **A useful potential contribution is experimentally identifying when single-view, trajectory-conditioned calibration can distinguish rate dependence, recoverable deformation and residual deformation.** A benchmark with informative load–hold–release and multi-rate motions, nuisance-parameter sensitivity, and whole-trajectory tests would be stronger than arguing that combining established components is intrinsically novel.

## 1. The actual forward mechanics

### 1.1 Particle–grid update and stress

Primary source: `scripts/taichi_viscoelastic_mpm_scene.py:814–910`, `1256–1411`.

Particle fields are position `x`, velocity `v`, affine velocity `C`, deformation-gradient-like state `F`, plastic-volume scalar `Jp`, and a yielded flag. The background grid carries mass and momentum/velocity. The one-metre scene grid spacing is `dx=1/grid`; input time step is `dt`. Quadratic tensor-product B-spline weights use a 3×3×3 stencil (`1287–1293`, `1363–1369`).

The intended update is:

- Trial deformation: `F_trial = (I + dt*C) F` (`1268`).
- Optional singular-value projection, described below.
- Moduli: `mu0=E/[2(1+nu)]`, `lambda0=E*nu/[(1+nu)(1-2nu)]` (`829–833`).
- Hardening factor: `h=exp(hardening*(1-Jp))`; `mu=h*mu0`, `lambda=h*lambda0` (`1294–1297`).
- Elastic stress-like matrix: `tau_e = 2*mu*(F-R)*F^T + lambda*J*(J-1)*I`, where `R` is the polar rotation and `J=det(F)` (`1298–1300`).
- Viscous matrix: `tau_v = eta*(C+C^T)` (`1301`).
- P2G affine momentum matrix: `B = m_p*C - 4*dt*V_p/dx^2*(tau_e+tau_v)` (`1302–1303`).
- Grid momentum: `sum_p w_ip [m_p*v_p+B_p*(x_i-x_p)]`; mass: `sum_p w_ip*m_p` (`1305–1310`).
- Divide by grid mass, apply gravity and contact/boundaries, then G2P (`1312–1407`).

`tau_e` is the corotated Kirchhoff-style quantity `P*F^T`, not the first Piola stress itself. For nonsingular F, the associated first Piola expression is `P=2*mu*(F-R)+lambda*J*(J-1)*F^(-T)`. The metadata description mentions both corotated and Neo-Hookean (`2343`), but the actual formula above is the corotated form and should determine manuscript wording.

If `C` has the intended units of inverse seconds, `eta*(C+C^T)` has stress units for `eta` in Pa·s. However, finite-deformation use requires explicitly defining whether this is a Kirchhoff viscous addition or spatial Cauchy viscosity: a spatial Newtonian Cauchy term normally becomes `J*sigma_v` when added to Kirchhoff stress. The code directly adds `eta*(C+C^T)` to `P*F^T`. That is a definable phenomenological choice, not automatically a complete rheological model.

The model is a corotated elastic response plus an instantaneous rate term and optional plastic projection. It does not contain a Maxwell branch, a relaxation spectrum, or a general memory kernel. It should not be described as a complete dough viscoelastic law merely because it has a parameter named viscosity.

### 1.2 Verified affine-transfer scaling inconsistency

Sources: `scripts/taichi_viscoelastic_mpm_scene.py:1373–1378`, contrasted with its use of `C` in `1268`, `1301–1303`.

The code defines:

```python
dpos = (offset.cast(float) - fx) * dx
new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
```

With physical `dpos=x_i-x_p`, the usual quadratic MLS/APIC affine reconstruction is `C = 4/dx^2 * sum_i w_ip*v_i*(x_i-x_p)^T`. Equivalently, `4/dx` is correct when the outer product uses the **dimensionless** offset `(offset-fx)` without multiplying by `dx`.

This is directly checkable without relying on a named solver. On an interior stencil, the quadratic weights satisfy `sum_i w_i*d_i=0` and `sum_i w_i*d_i*d_i^T=(dx^2/4)*I`. For an exact affine grid velocity `v_i=b+A*d_i`, the implemented expression therefore returns `C_code=dx*A`, while an affine-preserving reconstruction returns `A`.

I reproduced the exact weight/offset expression in NumPy at grids 24, 48 and 96. Maximum numerical discrepancies were:

| Grid | max absolute `C_code-dx*A` | max absolute `C_correct-A` |
|---|---:|---:|
| 24 | 3.47e-17 | 8.88e-16 |
| 48 | 4.34e-17 | 2.44e-15 |
| 96 | 9.71e-17 | 9.77e-15 |

At grid 48, a gradient component `A_yy=3/s` is reconstructed as `0.0625` rather than `3`. This is a deterministic algebraic check of the current expression, not a newly run Taichi simulation. No matching affine-reproduction regression test was found by searching `test/`, `scripts/test*.py`, and `configs/` for affine/linear reproduction and the `new_C` expression.

Scientific consequence: `C` drives trial strain, viscosity and plastic yield. Fitted E or eta can compensate for this implementation-dependent behavior, so SI labels alone do not make them measurements of material properties. Fixing the factor would change the dynamics and require redoing calibration; it is not valid to rescale the existing fitted E by an assumed constant without experiments. The old calibration result's simulator SHA-256 exactly matches the current file (`6653543ac16c8fcbdc111c73ebaa2c5e2d8c1cdc899e3750dce539b2730a2f07`), so this finding is relevant to that stored result.

### 1.3 Plasticity controls

Sources: simulator `847–855`, `1271–1282`, `1397–1407`, `2013–2039`.

When `--pure-viscoelastic` is absent, the code computes `F_trial=U*diag(s_i)*V^T`, clips each singular value into `[plastic_min,plastic_max]`, and replaces F by the clipped product. Any clip exceeding 1e-6 marks the particle as yielded. If `--use-jp` is enabled, it updates `Jp <- clip(Jp*det(F_trial)/det(F_clipped), jp_min,jp_max)`.

These are **dimensionless principal-stretch limits**, not a yield stress parameter expressed in Pa. Their physical effect depends on E, nu, the deformation path and numerical transfer. A separate full plastic deformation tensor is not stored; optional Jp retains only a scalar volume-related history for hardening. Yield-dependent velocity and affine damping are additional numerical mechanisms. The same affine-damping setting also acts within a near-floor band.

`--pure-viscoelastic` disables singular-value clipping and yielding. Therefore scanning plastic bounds while leaving this flag enabled cannot identify plasticity. The completed Episode 18 E calibration includes this flag, viscosity zero, hardening zero, Jp disabled, velocity damping one and plastic damping values one. It is not a completed elastoviscoplastic identification experiment.

### 1.4 Contact geometry and response

Sources: simulator `956–1133`, `1319–1343`, `1387–1407`, `2271–2306`; `scripts/build_tool_collision_meshes.py:126–149`, `169–197`.

The code supports none, rotated boxes and mesh SDFs. Mesh SDF values and precomputed gradients are trilinearly sampled; gradients are normalized. A particle/node contacts when signed distance is less than the numerical padding. Grid contact selects the closer tool; particle correction handles both tools sequentially. Particle projection adds `(padding-distance+1e-4)*normal` to position.

Tool point velocity includes both translation and `omega cross (x-centre)`. For relative velocity u, the code removes the inward normal component and then applies

`u_after = r*(1-a)*(1-s) * [u - min(u·n,0)*n]`,

where `r=tool_contact_friction`, `a=tool_contact_absorption`, `s=tool_stickiness`; world velocity adds tool velocity afterward. It scales **all remaining relative components**, including outward normal velocity, rather than just a Coulomb-limited tangential component. Lower r means more attenuation/sticking; r=0 sets the postcontact velocity to the tool velocity, while r=1 retains it after inward-normal removal. This is a dimensionless velocity-retention parameter, not a conventional Coulomb coefficient.

Floor response similarly scales horizontal velocity by `floor_friction*(1-floor_stickiness)` and uses a normal restitution-like multiplier. Several factors can be applied at grid and particle stages. The damping accumulates per step/contact, so changing time step can change effective dissipation even if numerical values stay fixed. For repeated application of a factor r over time, the equivalent continuous decay rate would be `-log(r)/dt`; no such timestep normalization is implemented.

Collision solids are generated from bundled visual STL triangles using voxelization, morphological closing, exterior fill, and marching cubes. The program checks topology/interior samples and default-asset hashes. These checks establish usable closed solids; they do not alone establish physical dimensional accuracy or marker-to-tool registration accuracy. That requires a separate measurement record.

## 2. Measurement, initialization and replay

### 2.1 Coordinates and hidden volume

Sources: `scripts/deformpath_topview.py:109–206`; `scripts/reconstruct_voxel_dough_from_deformpath.py:223–372`; simulator `581–724`.

Metric v2 requires rigid transforms, exact camera intrinsics, a scene frame and a floor plane. Legacy calibration can include axis remapping, scale and translation and is marked non-metric. Floor-mode reconstruction transforms segmented visible points and fills columns toward an intersected floor, within configured thickness/bounds limits. This is an explicit single-view hidden-volume prior: it assumes filled material between the visible top and floor, not measured rear/underside geometry or internal cavities.

Mass initialization is resolution-independent when reconstruction metadata is supplied: `V_total=N_voxels*voxel_size^3`, `V_particle=V_total/N_particles`; with supplied object mass, `rho=M/V_total` and `m_particle=M/N_particles`. Without reconstruction metadata, particle volume is grid-derived and changing resolution changes total modeled mass/volume. The reconstructed path prevents additional fitting scales, axis maps and offsets and verifies metadata hashes/calibration consistency.

The current local reconstruction records 3,970 retained points, 32,710 three-millimetre voxels, volume 0.00088317 m³ and 24,000 particles. With the configured 0.25 kg input, density is 283.0712 kg/m³. This is derived from supplied mass and reconstructed volume; it is not an independently measured density. Entrained air, hidden volume and segmentation can all affect its interpretation.

### 2.2 Known tool trajectories

Source: `scripts/deformpath_dynamics.py:390–516`.

The data loader reads `pointclouds_interpolated.pt` and `paths_interpolated.pt`, validates both tool streams, common finite timestamps, and monotonicity. Replay applies marker-to-tool transforms, linearly interpolates translation, uses quaternion SLERP for rotation, and derives linear/angular velocities. Maximum interpolation gaps and extrapolation are checked. Known tool trajectories are therefore a useful conditioning input, but they are kinematics, not force measurements; repeated interpolated observation frames should not be counted as independent physical samples.

### 2.3 Observation operator and loss

Sources: `scripts/evaluate_dynamic_topview_match.py:131–297`; `scripts/material_calibration.py:74–332`; simulator `2419`.

Real point clouds and simulated particles are rasterized into the calibrated camera. Timing comparison uses a declared tolerance, rejects reuse of a simulated frame for multiple observations and preserves unavailable observations. No per-frame geometric registration/time warping is performed inside the evaluator. It compares visible geometry rather than material-point correspondence.

The calibration objective is a normalized weighted average of four terms:

- Huber loss on simulated-versus-observed **depth change from the initial frame**, on pixels valid in all four initial/current masks;
- `1-mask IoU`;
- Huber loss of one-sided observed-to-visible-simulation nearest-neighbour distance;
- `1-real coverage`.

The existing Episode 18 runs weight all four equally and use 0.01 m depth/distance scales, Huber delta 1 and minimum support counts of 50. These aggregate scores are dimensionless and are **not** depth errors in metres. Depth-only comparison is accompanied by mask/coverage terms, which is better than silently rewarding missing overlap.

Limits: rendered depth currently contains dough only; tool occlusion is explicitly not modeled. Occlusion cannot be distinguished from segmentation dropout. A one-sided distance alone does not penalize extra unmatched simulated geometry; mask and coverage terms address some but not all of that problem. The four-mask intersection can exclude newly exposed or lost regions, so it needs reporting and sensitivity checks rather than being assumed unbiased.

## 3. Calibration actually available

### Young's modulus

`calibrate_youngs_modulus.py:1184–1379` performs logarithmic E candidates, optional boundary expansion, local refinement, deterministic minimum selection, a configured near-minimum tolerance diagnosis, and whole-window bootstrap. It fits using training windows only and reruns the chosen E on validation windows without refitting. Fixed settings include viscosity, plasticity controls, density/mass, contact response and numerical resolution.

The `accepted` status is based on a valid interior, non-flat training minimum plus a valid held-out evaluation. It does **not** require held-out improvement over default/frozen baselines. A flat loss profile under a hand-chosen tolerance is evidence of weak **practical sensitivity for this experiment/objective**; it is not a theorem of structural non-identifiability and not a confidence interval.

### Friction-retention and viscosity sweeps

- `scripts/sweep_episode18_tool_friction.py:19–43`, `138–147`, `184–284`: r in {0.1,0.2,0.3}, E fixed to 130579.320726, padding fixed to 1/384 m, replay/scoring through 60. It now supports path rebasing, manifests, retries, and GPU execution by removing the CPU flag.
- `scripts/sweep_episode18_viscosity.py:30–55`, `58–95`, `124–157`: eta in {0,1,2.5,5,10} Pa·s by naming convention, fixed E/padding and caller-selected r. It scores frames 1–60 only. It imports baseline command construction and evaluation utilities from the friction runner.

Neither sweep implements held-out evaluation or a statistical accepted-winner decision. Neither does a joint search; their report alone does not prove parameter independence. No plasticity sweep was found. The small-run sweep caches reuse complete cases by path/status without the full main E driver's input-hash validation, so experiments need a manifest audit when commands/settings change.

## 4. Current stored evidence — with the correct splits

### 4.1 E search

File: `data/single_episode_calibration/episode18_kugla_fixed_collision_11_9/material_calibration_result.json`.

- Status `needs_review`; selected E=130579.320726 Pa, accepted false.
- Training frames 1–60, time 0.0333508–2.0010524 s.
- Validation frames 61–97, time 2.0344031–3.2350459 s; simulation restarts from source frame zero for the full history.
- Ten candidates 91422.7597–186506.7196 Pa are within the configured 2% training-loss tolerance.
- Whole-window bootstrap has only **one** training window. All 2,000 resamples therefore select the same candidate; its zero-width interval provides no between-trial uncertainty evidence.

| Setting | Split | Aggregate loss | Correct interpretation |
|---|---|---:|---|
| Selected E=130579.320726 | Training 1–60 | 0.361058936 | Minimum among explored E candidates |
| Default E=2000 | Training 1–60 | 0.799745342 | Same-window baseline; selection improves it by ~54.85% |
| Frozen initial simulation | Training 1–60 | 0.343190531 | Slightly better than every fitted-E candidate on aggregate |
| Selected E=130579.320726 | Validation 61–97 | 0.828075209 | Later-window forecast score |
| Frozen initial simulation | Validation 61–97 | 0.835859842 | Selected E improves aggregate by ~0.93%; no CI/repeats |
| Default E=2000 | Validation 61–97 | Not in the result | No valid held-out default-E comparison can be made from 0.799745 |

Key references: selection/flat/one-window bootstrap at result `227–289`; validation aggregate at `86830–86837`; default baseline at `93862–93899` explicitly has split training; frozen training at `106840`; frozen held-out values inside the validation window include `88172` and `89518`.

This is more nuanced than either 'calibration succeeded' or 'calibrated E is worse than default on held-out data'. The latter comparison would mix splits and is invalid. The actual stored evidence establishes a search minimum, weak local identifiability under this objective, a small chronological improvement over frozen state, and no demonstrated transfer to new trials.

### 4.2 Friction retention

File: `output/episode18_e130579_dx8_tool_friction_sweep_local/friction_sweep.json` (tracked at audit time).

All cases completed with valid evaluation and training frames 1–60:

| Retention r | Training loss | Simulator wall time | Evaluator wall time |
|---:|---:|---:|---:|
| 0.1 | 0.359190823 | 255.81 s | 20.98 s |
| 0.2 | 0.361067263 | 163.17 s | 17.56 s |
| 0.3 | 0.361462632 | 160.08 s | 16.82 s |

Scores appear at lines `2149`, `4305`, `6461`; completed status follows each. r=0.1 is lowest of these three on training only, improving about 0.52% relative to r=0.2. It is at the tested range boundary, the differences are small, and there is no repeated/held-out result. The larger first-run runtime could include initialization; these logs alone are not a controlled runtime comparison.

A different older directory `output/episode18_e130579_dx8_tool_friction_sweep/` has failed cases. Do not report those older failures as the state of all current friction experiments.

### 4.3 Viscosity

File found during audit: `output/episode18_e130579_dx8_viscosity_sweep2/viscosity_sweep.json`. All five cases failed at simulator startup; no loss/evaluation is available. First stderr: `.../cases/viscosity_0/simulator.stderr.log:1–11` reports `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE` during `ti.init` on the remote recorded machine. That result says nothing about viscosity response, material validity or numerical stability of the five values.

The current code's `--validate-only` behavior was not executed in this audit; no claim is made that these runs would complete in the current local environment.

### 4.4 Static geometry and older dynamics

Current metric-v2 static result `data/single_episode_calibration/episode18_kugla/static/evaluation/static_topview_metrics.json`:

- footprint IoU 0.830587 (`1376`);
- extent ratios 1.076925 and 1.106248;
- centroid offsets −0.9754 mm and +1.2524 mm;
- visible-depth median absolute error 0.1480 mm, p95 1.1553 mm (`1399`);
- calibration labels metric v2 with method `moving_mocap_tag_correspondence`.

This supports initial visible-geometry reproduction under the recorded transformation. It is not an independent depth-camera metrology test or dynamic material validation: the initial observation supplied the reconstruction.

The legacy static record has IoU 0.880807 and p95 2.0000 mm, but uses scale 2 and the bootstrap transform. It is a separate setup and should not be mixed with the metric result.

The older proxy rollout has 65 post-initial observations, mean pixel IoU 0.424007, visible-depth MAE 95.9501 mm and depth-change MAE 83.8271 mm (`data/dynamic_topview_match/episode18_kugla/proxy_rollout/evaluation/dynamic_topview_metrics.json`). This is a useful documented failure case for that proxy setup, not a measurement of the current SDF configuration's accuracy. The three-frame smoke rollout checks execution, not long-horizon prediction.

### 4.5 Reproducibility interpretation

The current friction-local aggregate is tracked. The main material/static results under `data/` are not tracked in the audited Git index, and use external absolute observation paths. Hashes and manifests help identify intended inputs, but an experiment archive or data download procedure is still needed to reproduce the results from a fresh clone. The audited simulator hash matches the imported E result's simulator hash. No full artifact-reproduction run or full test suite was performed here.

## 5. What is scientifically identifiable in principle?

These are deductions and proposed tests, not measured results.

### Elasticity

Known displacement-controlled tool motion does not automatically determine absolute stiffness from geometry. In quasistatic homogeneous elasticity with purely imposed displacement boundaries, no body force and fixed nu, scaling E can leave the displacement field unchanged while scaling reaction forces. Known gravity, inertia, measured density, free boundaries and additional force observations can break that ambiguity. Therefore a single slow prescribed motion can fit geometry well while leaving E weakly constrained. The loss profile should be treated as evidence about the observation/action combination, not merely an optimizer defect.

### Viscosity

Multi-rate trajectories and free recovery give potential information about rate dependence. Fixed-step velocity attenuation, floor/tool contact damping, numerical transfer, E and unknown tool timing can imitate viscous dissipation. Estimating viscosity requires either fixing or explicitly varying these alternatives.

As a dimensional warning only, E=130579 Pa and nu=0.3 imply mu≈50223 Pa. Nominal eta/mu for eta=1–10 Pa·s is roughly 20–200 microseconds, compared with dt=200 microseconds and the recorded depth-frame interval ≈33 milliseconds. The actual relevant timescale depends on the constitutive model, geometry and boundary conditions, and the affine-transfer issue prevents direct physical interpretation. Nonetheless, these values are a reason to justify the viscosity range through observed rate/relaxation scales rather than choose a short numerical list without analysis.

A displacement-hold test does not necessarily reveal stress relaxation in depth: if the visible geometry is constrained to stay fixed, the force may relax while depth does not. Include release/recovery or an independent force channel when that distinction matters. A Kelvin–Voigt-like rate addition should not be expected to reproduce a general stress-relaxation spectrum.

### Plasticity

Residual deformation after unloading is informative, but finite-time viscous recovery can look permanent over a short clip. Repeated load–unload cycles, variable wait durations and several deformation amplitudes can test the distinction. The current principal-stretch clipping law provides candidate dimensionless parameters for such a study; it does not directly provide independently interpretable yield stress or a full elasto-viscoplastic law. Plasticity must be enabled during those trials.

### Coupling and uncertainty

- E and nu change elastic stiffness combinations.
- E/rho affects inertial response; rho is coupled to reconstructed hidden volume when mass is fixed.
- E and stretch-clamp thresholds together determine the stress level at yield.
- eta, global damping, contact retention, stickiness and numerical transfer affect dissipation.
- Tool padding, marker registration, camera/floor transforms and hidden-volume assumptions change apparent strain/contact timing.
- Changing grid/time step can change both discretization error and effective per-step contact damping.

Forward sweeps can map these sensitivities without autodiff. Pairwise or joint loss slices and nuisance-parameter profiles are needed because one-at-a-time fits can give a misleadingly sharp conditional optimum. Finite-difference trajectory/feature sensitivities can assess whether different parameters produce distinguishable observed responses. Flat profiles should produce intervals or sets of plausible effective parameters, not extra printed decimal places.

## 6. Contribution opportunities supported by this codebase

A defensible experimental method could test these hypotheses:

1. **Action diversity:** multi-rate load–hold–release probes constrain elastic/rate/residual mechanisms better than one continuous shaping episode, as measured by narrower out-of-sample-stable parameter sets and better independent-trajectory prediction.
2. **Observation adequacy:** one calibrated depth view with a stated floor-fill prior can support useful prediction for some motion regimes; the limits can be quantified by extra-view or volume-reference experiments rather than assumed away.
3. **Contact/material separation:** independently measuring tool geometry and timing, and including controlled sliding/pressing/release motions, reduces compensation between bulk parameters and contact-retention/padding choices.
4. **Model selection:** compare elastic-only, elastic-plus-rate, elastic-plus-plastic and combined models under matched parameter-search budgets. Extra model complexity earns a claim only through untouched-trajectory improvement, not training fit alone.
5. **Numerical transferability:** demonstrate which inferred effective parameters remain stable under corrected affine transfer, grid/time-step changes and mass/volume-preserving particle resolution. Report cases where parameters must be resolution-specific.

The framework already provides much of the execution and measurement infrastructure for these tests. The code alone cannot establish that the combined method is new relative to all literature, but it supports a serious, falsifiable calibration study. A paper can make an experimental contribution without a new solver or optimizer; its main evidence would be carefully designed physical measurements, comparisons and transferable prediction. Conversely, using gradient-free sweeps should neither be treated as a scientific weakness by itself nor presented as algorithmic novelty.

# Differentiable MPM validation

## Multi-episode driver

The shared-material dataset driver is implemented, and its small corrected multi-motion numerical checks pass. Its objective averages existing episode-mean losses with fixed normalized weights and accumulates shared physical gradients. The existing optimizer evaluates every training episode for each proposal/backtrack. Subprocesses run sequentially; input preparation and compilation repeat. First-setup tools use retention1/absorption0/stickiness0, with each episode's floor parameters preserved.

- **Four actual corrected CPU f64 multi-motion tests pass** in `runs/multi_trajectory_validation_20260912T133705307972Z/`, with complete Python source identity unchanged. A weighted 1:2 two-motion fit reduces combined training loss from 3.0445477861 to 1.8871840942 (38.0143%) in four accepted updates without retuning. All five material AD/FD checks at two perturbation sizes pass (largest printed relative error 2.573e-7). The held-out motion is not called during fitting and is evaluated only after selection. Actual `EpisodeProcessEvaluator` → `episode_worker` execution matches direct value and all seven returned physical gradients exactly; raw-data loading/preparation is mocked only for this tiny synthetic worker comparison.
- **276 host/regression tests pass** across 15 suites in `runs/checks_20260912T133643_504575/`. This includes existing single-episode input/CLI/optimizer/storage checks, new manifest/inventory/worker/objective/dataset-CLI checks, and 25 existing production sweep regressions. Python/dependency identity matched at verification. Numerical kernels are not exercised by this host/regression count.
- The genuine Episode18 two-frame dataset smoke configuration passes **input-only, no-runtime validation**: `runs/dataset_validate_20260912T133654_e29e974e/result.json`, with detailed worker/preflight records in `runs/dataset_preflight_20260912T133654_1a0eb3a4/`. This executes saved-input preparation, not MPM forward/backward, and does not resolve the observed floor inconsistency.
- Metadata-only inventory is recorded in `runs/dataset_inventory_20260912T132102Z_4a3449/`. Across ten known roots it finds 370 episode-named directories and 64 duplicate recording groups; these are processed variants, not 370 independent trials. Only Episode18 has the identified attached calibration and reconstructed initialization. Other selected episodes require their own verified inputs before a real dataset fit.
- The separate Episode18 floor investigation (`runs/episode18_floor_height_20260912_hsv/`) reports an observed table plane about47.6mm above the configured floor beneath the dough and about5° tilt. Saved metadata consistency is not evidence that this physical discrepancy is corrected. This implementation changes neither floor nor camera calibration.

No real multi-episode fit has been launched. See [MULTI_EPISODE.md](MULTI_EPISODE.md) for operation, limitations and remote commands.

This record concerns the experimental copy. No production file was changed by this experiment's implementation work.

## Non-adhesive Coulomb tool contact

A separate `coulomb-v1` tool-contact model is implemented without changing the default `retention-v1` behavior. It removes inward normal relative velocity, limits tangential velocity change by `mu * delta_v_n`, and leaves separating velocity unchanged. Grid and particle contact use the same response; particle angular collider velocity is evaluated after position projection. Absorption and stickiness must be zero, and the fixed friction coefficient is not a fitted parameter.

Evidence in `runs/coulomb_contact_cpu_20260912T164000Z/`:

- Six CPU f64 analytic/integration tests pass: zero friction, kinetic sliding, static stop, separating and withdrawing contact, moving-tool drag, continuity away from switches, both grid/particle paths, projected-position angular velocity and invalid configuration.
- One actual one-step moving/rotating SDF test passes state and E/Poisson/viscosity AD-versus-finite-difference checks away from branch switches. `tool_retention` has exactly zero derivative in `coulomb-v1`.
- The existing `retention-v1` moving/rotating SDF state/parameter derivative regression passes.
- A 12-step corrected f32 SDF forward comparison against the independent reference passes with unchanged tolerances. Maximum differences are x0, v9.31e-10, C5.96e-8, F3.64e-12 and Jp0.
- Solver/state/test hashes are identical before and after the archived tests. Taichi's existing reverse-kernel uninitialized-local warnings remain in the logs.

This is a velocity-level approximation, not a full pressure/impulse complementarity solve. Its friction limit is based on inward relative normal velocity after stress/gravity/grid updates. Resting contact with no inward normal velocity can therefore receive no tangential friction in a step. No real Episode18 replay, coefficient calibration, CUDA test or full-horizon gradient was run for this model.

## Corrected physics local qualification

The experimental configuration now distinguishes `corrected-v1` (default) from explicit `legacy-v1`. The correction uses `4 * inv_dx**2` in G2P with metric `dpos`, floor-based stencil indices, and a grid padded at logical index -1. The original reference snapshot remains unchanged at SHA-256 `6653543ac16c8fcbdc111c73ebaa2c5e2d8c1cdc899e3750dce539b2730a2f07`. A separate corrected reference is pinned to SHA-256 `d33f0aec3952fa72282cd181757f678b2e2a48e5b51016c23d40fd05d6a3ac3e`.

**Corrected local numerical checks pass; full-episode calibration remains unqualified.** Five CPU f64 analytic transfer tests pass on solver `263e3568…`: affine reconstruction at grids 24/48/96, padded-grid G2P adjoints, floor conservation/nonnegative masses, known-viscosity stress/transfer, and stencil bounds. Evidence: `runs/corrected_transfer_cpu_20260912T114149_c6306f/`.

The corrected-v1 serial-P2G small Vulkan f32 forward/loss/backward fixture also passes its CPU comparison: `runs/backend_20260912T114258Z_d56b4b15/backend_check.json`. The report records actual `Arch.vulkan`, solver `263e3568…`, and amdgpu compute-counter increase of 49,953,951 ns. This is a four-particle, three-step test, not full-horizon GPU or CUDA qualification. Corrected and legacy short forward parity both pass: nine fixtures per version, twelve steps per fixture, comparing all x/v/C/F/Jp arrays against each version's independent reference snapshot. Original tolerances are unchanged; corrected floor masses are nonnegative and the legacy negative-mass fixture remains explicit. All eight recorded source hashes stayed unchanged. Evidence: `runs/physics_reference_parity_20260912T114148Z/{parity.log,verification.json}`. All 18 solver/transfer tests pass with zero failures, errors or skips, including material, plasticity/Jp, floor/SDF derivatives, explicit legacy behavior, and corrected atomic/serial forward and gradient checks. Numerical/test hashes are unchanged across all five groups. Combined evidence: `runs/corrected_solver_combined_20260912T120613Z/summary.json`.

CPU f64 synthetic (9 tests) and trajectory (5 tests) suites pass in `runs/corrected_synthetic_trajectory_20260912T114053144307Z/`. Corrected full-storage versus segment-4 observation gradients and five material finite differences pass (largest printed relative error 1.288e-7). The unchanged 12-step, four-update synthetic fit reduces training loss from 3.5580313124 to 2.1879261617 (38.5074%) and held-out loss from 2.8736226660 to 1.8772705569; held-out data does not select updates. This demonstrates loss reduction, not unique parameter recovery. Solver/test sources were unchanged, but `checkpoint.py` changed during these tests while the memory-estimate integration was in progress. Exact before/after hashes were subsequently recovered and verified in `runs/physics_cli_20260912T114826_cf8f76/checkpoint_source_transition/`: only `estimate_memory` changed, every other AST node is equal, and `CheckpointedRollout` onward is byte-identical. The tested replay/adjoint implementation therefore did not change; this run is still not described as whole-source-stable. Reverse-kernel warnings remain in the logs.

The physics-version integration passes 101 host-only tests, recorded in `runs/physics_cli_20260912T114826_cf8f76/`: defaults and overrides, invalid versions, reference labels, changed-version resume rejection, worker propagation, padded memory estimates, and precision-matched input stencil bounds. These tests do not execute MPM kernels.

The historical results below concern the original G2P/stencil physics, even where their original descriptions say “current” or “stable.” They are not corrected-model validation. No corrected full Episode18 calibration has been launched. Old fitted material parameters remain initial guesses; there is no constant conversion that preserves the full trajectory.

## Historical legacy-physics status

**The differentiable implementation is experimental. Full-horizon real-data calibration is not yet qualified.** Component and short-trajectory derivatives pass tests. A 334-step real CPU f32 backward pass and a fresh Vulkan/CPU comparison pass with the stable grid-normalization adjoint. Current-source synthetic coordinate/directional gradients and the unchanged short-fit acceptance test pass with a declared 10 mm prediction temperature and fixed 2 mm targets; both training and held-out losses improve. A completed full forward replay does not establish full-horizon backward stability or parameter identifiability.

| Check | Evidence | Result and limits |
|---|---|---|
| Composite polar/stretch-clamp derivatives | `tests/test_spectral.py` | CPU f32 and f64 tests passed, including identity, repeated positive stretches, active bounds and finite differences. The Taichi SVD forward approximation is measured separately. |
| One-step and short MPM derivatives | `tests/test_solver.py` | CPU f64 tests passed for all particle-state components, material parameters, Jp, floor and moving/rotating SDF tools. A dedicated floor-zero regression also passed. |
| Complete segmented adjoint | `tests/test_checkpoint.py`, `tests/test_trajectory.py` | Analytic recurrence tests and actual eight-step MPM tests passed; full storage and segment lengths 1, 3 and 4 agree. Every observation derivative is injected once; x/v/C/F/Jp adjoints all cross boundaries. |
| Partial-view objective | `tests/test_loss.py` | Eleven v2/compatibility tests passed, including calibrated clipping, coverage, occlusion, non-overlap, gradients and v1 reproduction. This does not establish that the objective is easy to optimize over every visibility transition. |
| Workflow/storage/input validation | `tests/test_inputs.py`, `tests/test_results.py`, `tests/test_calibrate_cli.py` | Input geometry, timing, mass, export, safe resume and CLI error cases were tested. CLI tests use analytic mocks and are not MPM execution evidence. |
| Full Episode 18 forward positions | `runs/real_forward_parity_stable_normalization/comparison_summary.json` | Stable solver: frames 0–60, 24,000 particles, 10,006 steps pass the declared position tolerances. Worst frame RMS 12.95 µm; maximum particle-position difference 0.342 mm. Contact histories and full states are not identical; details below. |
| Stable grid-normalization adjoint | `runs/episode18_normalization_diagnosis/fix_verification.json` | Two f32 tiny-mass tests, three contact/trajectory FD tests and three 12-step frozen-reference forward fixtures pass. Forward positions are bit-identical in those fixtures; no epsilon or mass cutoff. |
| Actual local GPU execution | `runs/backend_stable_normalization_vulkan/backend_check.json` | Fresh three-step Vulkan f32 forward/loss/backward passed all 20 CPU comparisons on solver `75110847…`. Per-process amdgpu compute time increased by 52,552,988 ns. This is not full-horizon checkpoint qualification. |
| Synthetic full-objective derivatives | `runs/synthetic_stable_normalization_gradient/result.json` | Current stable solver: all five coordinate and two directional FD checks pass for the 10 mm predictor. All fixed 2 mm target hashes match the preserved targets; source hashes were unchanged during the check. |
| Synthetic fitting | `runs/tests/synthetic_412d75dddf7f422198aa6a0821516be8/result.json` | Current stable solver: 12 steps / 4 accepted updates reduce training by 33.18%; held-out loss also decreases. All eight synthetic tests pass, retaining the >1% assertion. Longer earlier evidence and the original 2 mm failure are described below. No unique parameter recovery claim. |
| Real backward | `runs/episode18_gradient_stable_f32/result.json` | CPU f32 frames 0–2 / 334 steps complete with finite E/ν/viscosity derivatives and exact checkpoint/loss recomputation. A prior f64 short run also passed. This interval has floor response but no tool contacts; full-horizon backward and real fitting remain unqualified. |
| CUDA | Remote user-provided fit/diagnostic logs, 2026-09-12 | Full forward replay completes, but two initial backward evaluations fail at steps 9,774 and 9,754 in segment `[9728, 9792)`. No optimizer update is accepted. Three later isolated diagnostic probes first differ at grid reduction, after identical restored inputs and material intermediates. Repeated segment states differ, while counts match in that diagnostic. Full CUDA calibration remains unqualified. |

## Full Episode 18 forward comparison

Both implementations use separate single-thread CPU f32 processes, identical serialized inputs, and fixed candidate source bytes. The comparison targets the preserved baseline.

### Stable-normalization solver

`runs/real_forward_parity_stable_normalization/comparison_summary.json` records the fresh complete comparison on solver `75110847…`. All pinned source hashes still match:

- All 61 frames, 24,000 particles and 10,006 steps are present.
- Worst frame position RMS is 1.29519462e-5 m; maximum particle-position norm difference is 3.42110428e-4 m, both at frame 60. Both pass the predeclared RMS ≤1e-4 m and maximum ≤1e-3 m limits.
- Contact counts differ on 4,554 steps, first at step 4,392 (0.8784 s): tool-0 particle responses are 18 in the reference and 19 in the candidate.
- Maximum velocity-component difference is 0.396757 m/s; aggregate velocity-component RMS is 0.00192444 m/s. Maximum C difference is 0.00516912, F difference 0.000508939, Jp difference zero.
- The tighter one-step array thresholds are exceeded from frame 27 for x/v/C and frame 28 for F. Those diagnostic failures remain in the report; they were not silently loosened to claim full-state agreement.
- Reference forward time is 235.71 s and candidate forward time 316.99 s, including initial compilation.

These results establish close forward positions, not identical contact histories or full states, and do not establish correct full-horizon gradients.

### Earlier solver comparison

The original result below remains available at `runs/real_forward_parity_cpu_v1/`. It predates the stable normalization adjoint and is not substituted for the fresh result above.

| Quantity | Frames 0–2 | Frames 0–60 |
|---|---:|---:|
| Integration steps | 334 | 10,006 |
| Worst frame position RMS | 8.97998e-9 m | 1.29328e-5 m |
| Maximum particle-position difference | 1.19267e-7 m | 2.07264e-4 m |

The full-horizon position criteria were RMS ≤1e-4 m and maximum ≤1e-3 m. All 61 expected frame states were present.

The long replay does **not** pass as identical full state or contact history:

- Aggregate SDF response counts differ on 4,551 steps, first at step 4,392 (0.8784 s).
- Maximum velocity-component difference is 0.3962 m/s; maximum vector difference is 0.4677 m/s.
- These velocity differences are sparse: 0.0747% of saved particle/frame samples exceed 0.01 m/s. The 99th percentile is 0.0002226 m/s.
- The particle with the worst velocity difference has an 86.3 µm position difference at that frame.
- Maximum C difference is 0.0101086; F difference is 0.000514104; Jp difference is zero.

Reference forward/export time was 265.95 s; candidate time was 362.66 s, including initial compilation. See `runs/real_forward_parity_cpu_v1/velocity_difference_summary.json` for the velocity distribution.

The experimental floor-zero regression deliberately preserves the baseline's nonpositive-grid-mass behavior. Passing that regression means faithful reproduction, not that the original floor discretization is physically preferable.

## Float32 normalization and real short replay

The stable solver SHA-256 is `75110847bd75b3ade3c9b0473ea8fd062fbaf7204763f1368c7c4e504d04dfa2`. At real reverse step 287, the generated normalization derivative formed `1 / mass²` for mass `1.50686393589534e-20`, overflowing float32 even though the incoming gradients and correct derivative were finite. The custom VJP computes `u = momentum / mass`, `momentum_bar += u_bar / mass`, and `mass_bar -= dot(u, u_bar / mass)`. For nonpositive mass, raw momentum and its identity derivative are preserved. Forward physics, mass thresholds and gradients across checkpoints are unchanged.

The isolated recorded-node mass derivative is -0.0006515889545 in f32, versus -0.0006515889625 analytically in f64. Unit tests cover positive masses down to 1e-35, ordinary/zero/negative masses, additive adjoints and finite differences. The original failed real run remains in `runs/episode18_gradient_frozen_v2/`; it has not been overwritten.

The fixed real f32 run uses 24,000 particles, 334 steps, checkpoint length 64 and two observations:

- Mean loss: 0.03858269006.
- Physical derivatives: E = -1.77687358e-8, ν = -0.00570298173, viscosity = -8.86728458e-7.
- Seven checkpoints, zero endpoint and loss differences, once-only observation injections `[1, 1]`.
- Forward 19.39 s and backward 136.17 s, including compilation.
- 30,267 grid-floor responses; no tool responses and inactive plasticity. Zero tool/plastic-limit derivatives are expected on this interval.

Compared with the earlier short f64 run, the relative loss difference is 0.0114%; E/ν/viscosity derivatives differ by 1.59% / 0.868% / 0.268%. This precision comparison is not a real-data finite-difference check. The fresh full forward comparison above and the three local post-fix forward fixtures use the stable-normalization solver.

## Synthetic objective and optimizer

The synthetic setup uses 64 particles, a 32-step horizon, checkpoint length 8, partial camera observations, all five material parameters, and a separate unused excitation. Both plastic limits are exercised. It does not use particle-identity matching as the training objective.

The v2 run starts at training loss 3.016126 and reaches 2.986024 after 48 accepted updates. Held-out loss changes from 2.457926 to 2.597576. Target generation, initial parameters and truth are unchanged from the earlier v1 experiment, but **loss values across versions are not directly comparable**.

At the saved v2 plateau:

- The next Adam direction has AD slope -0.967154486 versus finite-difference slope -0.967154166.
- The persisted step size is 9.155e-7. A tested 1e-4 proposal gives about 100 times more objective reduction with the same reported decision counts.
- A 1e-3 proposal encounters a steep depth transition and increases loss.
- Increasing proposal rates without checked backtracking is therefore not justified.
- Optional `recover-v1` rate recovery is implemented with 28 optimizer tests passing, including exact resume and rollback. A read-only short-run probe showed that recovery alone does not fix the 2 mm-temperature short-fit acceptance failure.
- A dense boundary probe established that the problematic v2 transition is continuous but extremely steep: a 50.7 mm front/rear gap at 2 mm temperature creates an exponential depth-weight contrast. The measured large local derivative agrees with finite differences. See `runs/synthetic_v2_depth_transition/boundary_curve.json`.

### Explicit 10 mm predictor with fixed targets

The synthetic default predictor uses 10 mm temperature; target generation remains at 2 mm. The real-data configuration retains its separately declared 5 mm temperature. These are different objectives, so raw loss values must not be compared directly with the 2 mm synthetic predictor.

Stable-normalization revision evidence (solver `75110847…`):

- All five initial coordinate derivatives pass at multiple step sizes. Best relative errors are at most 1.39e-6. Both mixed and descent-direction checks pass, with best errors 4.79e-7 and 1.30e-6.
- Target summary hashes match the preserved 2 mm-generated training and held-out targets exactly. Numerical/source files were unchanged throughout the derivative check.
- All eight synthetic tests pass. The 12-step, four-update integration test reduces training 2.961285 → 1.978642 (33.18%) and held-out 1.523875 → 1.136965. Its >1% assertion remains unchanged.

The earlier 32-step, 48-update run at `runs/synthetic_joint_cpu_f64_v2_10mm_fixed_48/result.json` reduced training 4.539283 → 2.158032 (52.46%) and held-out 3.826273 → 2.348899. That longer fit predates the stable-normalization solver revision; it is not a rerun of 48 updates on the current source. `budget_exhausted` records completion of the requested updates, not convergence.

The selected training loss can be below the truth-reference loss because parameters can compensate for observation smoothing. These results demonstrate gradient propagation and loss reduction, not unique E/ν/viscosity/plastic-bound identification.

## Remote CUDA replay failure

The user-provided full CUDA f32 log from 2026-09-12 reaches frame 60 / step 10,006, then rejects the initial objective while recomputing step 9,774 within segment `[9728, 9792)`. Earlier segments may already have been differentiated, but the rejected evaluation supplies no usable complete gradient and no optimizer update is accepted.

A second user-provided full CUDA attempt rejects recomputation at step 9,754 in the same segment, with zero accepted updates. The subsequent forward-only diagnostic at remote `runs/recompute_20260912T091744_58db44/` repeats that segment and independently repeats step 9,754 three times. Every isolated probe restores identical state bytes and has identical trial/projected deformation, Jp history, rotation and stress affine arrays. Its first recorded difference is `grid_reduction` (grid mass/momentum), isolating the observed non-repeatability to P2G reduction.

All three repeated segments retain matching aggregate contact counts but differ in state bytes. Their maximum x component difference is 1.1920929e-7 m (0.119 µm); maximum C difference across the repetitions is 1.4542602e-6. Jp is unchanged. The diagnostic does not reproduce the prior contact-count flip. Readbacks can change GPU scheduling, and matching aggregate counts do not establish identical contact identities. Reverse-step scratch is recomputed again, so shorter segments alone do not guarantee consistent adjoint intermediates.

Exact count checks remain enabled. The new checkpoint diagnostics identify changed counts, both values and segment limits; the CLI stores those details and an invalid-initial result without claiming successful backward execution. Analytic checkpoint and mocked CLI tests pass in `runs/cuda_recompute_diagnostics_checks/`. Those tests verify reporting and rejection, not a fix for CUDA nondeterminism.

The targeted `recompute_check.py` tool repeats the failing interval and independently repeats the chosen substep from its original input. It preserves full state/control bytes and compares material scratch before interpreting differences in grid mass/momentum. All 49 tests across checkpoint, CLI and diagnostic suites pass in `runs/cuda_recompute_diagnostics_final/`. A real four-particle CPU f64 forward-only diagnostic, including nonzero-slot restoration and state-buffer rollover, also passes exact repeated-state/count/intermediate comparisons at `runs/recompute_diagnostic_real_cpu_smoke/result.json`; its source identity is unchanged during execution. These tests do not reproduce or fix the remote CUDA failure.

No full real CPU gradient or deterministic CUDA implementation has been qualified. Single-thread CPU removes competing scatter additions and is the next reference test. CUDA seeds, synchronization, one-thread blocks or float64 alone are not determinism guarantees.

### Remote small CUDA backward check

The user-provided `runs/backend_20260912T101209Z_a748cb1d/backend_check.json` passes CUDA f32 forward, observation loss, backward execution and all 20 CPU comparisons. Its recorded solver hash is `75110847…`, the pre-serialization solver, while all six other backend-check source hashes match the serial-option revision. Although the command and configuration request `serial`, that solver does not contain the serialization line. This is useful small-fixture CUDA backward evidence for the older solver, not CUDA-serial execution or full-episode qualification. Transfer the matching solver file and rerun before interpreting the flag as proof of execution mode.

The runtime records `Arch.cuda`; the CPU comparison separately records `Arch.x64`. Optional DRM counters are absent on that NVIDIA system, so their `verified=false` value is not evidence of CPU fallback. No traceback, fatal runtime error or initialization failure is present. Each backend stderr log contains 81 generated-gradient warnings (54 particle-operation and 27 grid-operation warnings); this passing fixture does not establish that every possible branch is valid. The report SHA-256 is `b52303bf4bf194d2e6f4deb36a2fcddc2a44f2714c604f50e152aa8c9eeaa857`.

## Optional serial P2G

Solver SHA-256 `3960d45a81d7ab406f029af50669d99d7254557a36bdbe2bc50fcc9cd5cae8d8` adds exactly one solver line: `ti.loop_config(serialize=ti.static(self.config.p2g_mode == "serial"))` before the existing P2G particle loop. Removing that line exactly recovers solver `75110847…`. The default is atomic. This changes accumulation order only when selected; no material/contact equation, mass threshold, checkpoint check or memory allocation is changed.

### Complete local regression run

`runs/serial_p2g_final_suite/result.json` passes all 17 suites, totaling 197 unittest cases with no reported skips: 172 experiment tests and 25 existing production sweep/padding tests. The experiment Python/dependency identity still matches the launch manifest after completion. This includes preservation, default atomic forward/derivative checks, serial solver tests, loss, checkpointing, optimizer, input/CLI/storage, forward-reference fixtures and the unchanged synthetic fitting tests. Current production files are exercised by their own regression tests; the experiment's frozen-reference tests still target the preserved baseline. No real Episode 18 calibration is launched by this suite.

### CPU solver derivatives and replay

All four targeted tests pass without failures, errors or skips in `runs/serial_p2g_cpu_tests_20260912T094240_b4ec71/result.json`. Pinned solver/spectral/state/test hashes remain unchanged during execution:

- Actual P2G on CPU f32 with four threads: cancellation-sensitive inputs produce the expected sequential result on repeated calls; generated velocity adjoints match analytical values.
- CPU f64 atomic/serial elastic, plastic/floor and SDF forward arrays agree at rtol/atol 2e-12. Repeated serial five-step trajectories match all stored state/grid bytes exactly.
- Five-step serial plastic/Jp material and all-state derivatives pass finite differences using the existing tolerances. Atomic/serial numerical gradients agree at rtol 1e-9 / atol 1e-10.
- Serial moving/rotating two-tool SDF parameter and state derivatives pass finite differences.

The full log retains Taichi generated-gradient warnings from the particle/grid kernels. Their presence alongside passing short tests does not establish validity for every input or full-horizon CUDA replay.

Mode configuration, CLI propagation and changed-mode resume refusal pass 56 nonkernel tests at `runs/serial_p2g_cli_checks/`. Backend CLI propagation and failure/report tests pass separately at `runs/serial_p2g_backend_cli_checks/`. These mocked tests do not execute simulator kernels.

The additional `runs/serial_p2g_segmented_cpu_checks/result.json` passes both existing eight-particle/eight-step trajectory tests with only the fixture's P2G mode set to serial. Source hashes are unchanged. Elastic/viscous and active plastic/Jp cases retain exact endpoint replay checks, compare full storage with segment lengths 1, 3 and 4, carry all five state adjoints, and inject every observation exactly once. Parameter finite differences pass the unchanged 1e-3 error limit; the largest best error across the checked parameters/cases is 9.29e-6. A reproducer and complete log are saved with the result.

### Actual Vulkan solver checks

`runs/serial_p2g_vulkan_backend/backend_check.json` passes all 20 unchanged same-precision CPU comparisons with serial P2G. The three-step, four-particle fixture includes active stretch plasticity/Jp, viscosity, moving/rotating SDF tools, floor response, observation loss and all-state/material reverse differentiation. Actual architecture is `Arch.vulkan`, with an increased per-process amdgpu compute counter of 44,938,421 ns. Solver/source hashes remain fixed. Maximum position-component difference is 8.94e-8 m; maximum scaled material/contact derivative difference is 3.48e-7. This checks numerical gradient agreement, not bitwise equality of repeated gradients.

The separate forward-only diagnostic at `runs/vulkan_serial_recompute_20260912T100139_f4f194/result.json` uses the same contact/plastic/Jp fixture, start step 2, interval length 3, probe step 3, three repeats and layout length 4. Every repeated state and grid array matches exactly. Every isolated probe matches restored input, all material intermediates and output bytes; all recorded maximum differences are zero. Aggregate counts also match. Hardware activity and unchanged experiment source identity are recorded; `reproduce.py` is archived alongside the results. This does not exercise loss or reverse differentiation.

Both checks use tiny fixtures, not the 24,000-particle Episode 18 trajectory. Neither establishes full-episode or CUDA qualification.

### Isolated CPU/Vulkan serialization and cost

`runs/serial_p2g_microprobe_20260912_v1/summary.json` and `evidence_index.json` preserve the script, results and exact compiler-IR excerpts. For a minimal scalar scatter, serial mode compiles to one serial offload with a nested particle loop and zero atomic additions in both forward and generated reverse, on CPU and Vulkan. Atomic mode retains a parallel range offload. A separate 27-neighbor scatter probe passes order-sensitive, analytical and finite-difference checks.

At 24,000 particles/grid48, five warmed repetitions produce one forward and one adjoint hash per backend in serial mode; atomic forward produces five distinct hashes per backend. This result concerns that scatter probe, not every solver or observation-loss reduction.

| Backend | Atomic forward | Serial forward | Atomic reverse | Serial reverse |
|---|---:|---:|---:|---:|
| CPU, 8 threads | 3.170 ms | 1.047 ms | 0.959 ms | 1.904 ms |
| Vulkan | 1.383 ms | 161.820 ms | 5.352 ms | 36.721 ms |

These are medians of five warmed isolated scatter calls, excluding compilation, clearing/uploads and all other stages. Vulkan serial is 117× slower forward and 6.86× slower reverse. They are not full-solver or CUDA timing estimates. Local CUDA support is unavailable; `cuda_unavailable.json` records that initialization failure rather than substituting a backend.

The option serializes P2G's forward and generated reverse particle loops. Other shared parameter-gradient and renderer/loss reductions remain parallel where they were parallel before. No bitwise whole-gradient, full-episode or CUDA qualification follows from these results.

## Opt-in finite mismatch override

`--ignore-recompute-mismatch` lets calibration continue through finite contact-count, checkpoint-state and observation-loss replay differences. It is disabled by default, independent of atomic/serial P2G, and included in run identity. Every ignored discrepancy remains recorded; objective diagnostics identify counts and replay inconsistency. State/loss/gradient validation and optimizer acceptance rules remain active. A completed backward pass with ignored discrepancies is not established as the derivative of the recorded forward loss.

All 97 focused nonkernel tests pass at `runs/ignore_recompute_final_checks/result.json`, with the experiment source identity unchanged: 16 checkpoint, 29 CLI, 28 optimizer, 4 result-storage and 20 recomputation-diagnostic tests. Coverage includes strict rejection, explicit continuation for all three mismatch kinds, invalid/nonfinite rejection, per-evaluation counter resets and retained diagnostic snapshots, warning persistence/throttling, policy propagation, resume refusal after policy changes, and approximate-gradient metadata. No full-episode CUDA run or calibration was executed to qualify this override; numerical MPM kernels and production files are unchanged by it.

## Interpreting execution records

Initialization, forward execution and backward execution are separate facts. A requested backend is not proof of device execution. Fit runtime records count successfully completed objective evaluations in the current process; restored optimizer state is not a newly executed simulation.

Default reference verification rejects changed production sources. Explicit `--reference-policy frozen` selects the preserved baseline only when exact expected bytes are available. Historical results keep their source hashes; production corrections must be evaluated as a separate physics version.

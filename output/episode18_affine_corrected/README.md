# Corrected affine transfer

Full Young's modulus calibration and viscosity fitting were **not run**, following the user's instruction. The work here corrects and tests the transfer, checks short recorded replays, and prepares commands for later use. Existing experiment results and visualizations were not modified.

## Solver changes

In `scripts/taichi_viscoelastic_mpm_scene.py`, G2P now reconstructs the affine velocity gradient with

```text
C = (4 / dx²) Σ weight × grid_velocity ⊗ physical_offset.
```

For an interior quadratic B-spline stencil, the weighted second moment of physical offsets is `dx²/4` times the identity. The previous coefficient `4/dx` returned `dx × A` for an affine input gradient `A`. An actual-kernel regression reproduced `A/24` at grid 24 before the correction. The corrected test reproduces `A` at grids 24, 48 and 96.

The corrected transfer exposed a separate floor-adjacent stencil error. Truncating a negative base coordinate toward zero gave invalid weights near scene height zero. For example, a fractional coordinate of 0.016 gave a middle weight of about −0.218. The existing floor-contact test then failed with a nonfinite state.

Stencil bases now use mathematical floor. One extra grid node at `−dx` on each axis provides complete support at scene zero. Logical grid indices are `−1 … grid−1`; spacing remains `1/grid`. The physical floor, tool poses, material law, time step and damping settings are unchanged. Safety checks and diagnostic stencil bounds use the same indexing. The floor-contact regression now passes, and a new force-free affine test checks vertical positions at 0, 0.016 × dx and 0.49 × dx.

These changes affect APIC momentum transfer and deformation evolution as well as viscosity. **Do not multiply the old fitted viscosity by 48 or treat the old fitted E as calibrated for this solver.** No replacement fitted parameters are claimed here.

## Regression checks

The following targeted suites passed after the solver corrections:

- 21 tests: `test/test_mpm_affine_transfer.py` and `scripts/test_mpm_mass.py`.
- 45 tests: material calibration, replay SDF diagnostics, contact-padding sweep, renderer, tool-friction sweep and collision-ablation tests.
- 16 tests: viscosity runner configuration, fingerprints, resume behavior and scoring validity.

**82 distinct targeted tests passed** across these three runs. This was not an execution of every test in the repository.

The new actual-kernel tests cover affine reproduction and deformation evolution across resolutions, translation, rigid spin with zero symmetric strain rate, the analytical one-particle viscous response, and floor-adjacent affine transfer. They are numerical regressions, not evidence that a physical dough material model has been identified.

Reproduce the numerical tests from the repository root:

```bash
python3 -m pytest -q test/test_mpm_affine_transfer.py scripts/test_mpm_mass.py
```

## Short recorded replays

Only source frames 0–5 were simulated, approximately 0.167 seconds, with 24,000 particles, grid 48, dt 0.0002 s, E 2000 Pa, viscosity 0, the original initial particles, and the fixed SDF/contact setup.

- `benchmark_cpu/`: CPU simulation completed, six finite particle frames, no particles below the physical floor. The calibration runner rejected metadata before evaluation because explicit collision-mesh overrides suppress the simulator's default collision-manifest record.
- `benchmark_gpu/`: Vulkan simulation completed with the same checks; its calibration metadata was rejected for the same configuration reason.
- `benchmark_gpu_verified/`: **simulation, metadata verification and evaluation passed** using the default collision solids. All six saved particle frames are finite and at or above the physical floor; the initial particle array matches the original exactly. All five scored frames are valid. The candidate report records the full result. Its short-window loss of 0.2278136528 is not comparable with earlier 60-frame training losses.

The first two directories are retained as rejected-metadata benchmark records, not successful calibration results. The preparation script now omits those redundant overrides and still fingerprints the collision solids and their manifest. No source-code change was needed for this configuration correction.

A successful five-frame smoke test does not establish stability over the full trajectory or all parameter candidates. CPU/Vulkan timings include initialization, SDF construction and compilation and are not a controlled speed comparison.

## Viscosity runner

`scripts/sweep_episode18_viscosity.py` now:

- accepts positive finite `--youngs-modulus`;
- preserves the source backend by default, with explicit `--cpu` or `--gpu` overrides;
- records source/input/settings fingerprints and rejects incompatible or unidentified completed cases on resume;
- preserves failed-attempt outputs when retrying;
- requires a valid finite loss containing all scored frames 1–60.

For command compatibility, omitting `--youngs-modulus` still selects the previous numerical default, 130579.320726 Pa. That default is **not** a corrected-solver calibration. Pass E explicitly for new work.

`viscosity_command_validation_final.json` contains a no-execution check using the local recorded inputs, E=2000 Pa and CPU. It verifies all five expanded command sets. E=2000 is a test setting, not a newly selected parameter. No viscosity sweep directory was created.

## Commands for later use — not executed

Prepare a new elasticity experiment without changing existing results:

```bash
python3 output/episode18_affine_corrected/prepare_calibration.py \
  --output-dir output/episode18_affine_corrected/elasticity \
  --backend cpu

python3 scripts/calibrate_youngs_modulus.py \
  output/episode18_affine_corrected/elasticity/material_calibration_manifest.json \
  --validate-only
```

If a full calibration is later wanted, omit `--validate-only` from the second command. The prepared profile preserves training frames 1–60 and later-window frames 61–97, with both replays initialized at frame 0. A later window of the same episode is not independent-motion validation. Inspect result acceptance and failures rather than relying only on a “Wrote result” message.

After choosing E from a corrected-solver experiment, inspect viscosity commands before running them:

```bash
python3 scripts/sweep_episode18_viscosity.py \
  --material-manifest output/episode18_affine_corrected/elasticity/material_calibration_manifest.json \
  --youngs-modulus "$SELECTED_E_PA" \
  --tool-contact-friction 0.2 --cpu \
  --output-dir output/episode18_affine_corrected/viscosity \
  --validate-only
```

The existing five viscosities are 0, 1, 2.5, 5 and 10 Pa·s. The viscosity runner scores only frames 1–60; these commands do not add a later-window viscosity evaluation.

All preparation and candidate execution records are under this directory. `checks/verification.json` records the results and final source hashes; `checks/changes.patch` contains the code/test changes, and the numerical/workflow test logs are alongside it. No commits or branches were created.

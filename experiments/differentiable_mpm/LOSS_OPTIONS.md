# Selectable calibration losses

The existing `partial-visible-splats-v2` objective remains the default. The new objectives are selected by `loss.version` in a separate experiment JSON. Version 1 of the partial loss remains available for reproducing earlier experiments. The strict evaluator is separate and unchanged.

**Verification status: all 77 tests in the five new suites passed; the final legacy trajectory regression is still running.** All six new modes passed CPU-f64 position/MPM gradient checks and short synthetic fitting. Three usable Episode18 examples passed input-only validation on the same source. The test-only optimizer tolerances and validation-before-runtime checks passed. No real-data simulation or calibration has been launched for this work.

## Options and data requirements

| `loss.version` | Objective | Required observations |
| --- | --- | --- |
| `partial-visible-splats-v2` | Mean of the existing depth-change, positive coverage and directed robust point terms | Current partial depth observations |
| `dpsi-pcd-cd-v1` | Endpoint symmetric unsquared Chamfer sums | Recorded calibrated point cloud, or supplied observed-cloud target |
| `dpsi-prt-cd-v1` | Same distance to an endpoint volume target | Separately reconstructed endpoint volume |
| `dpsi-pcd-emd-v1` | Endpoint injective target-to-simulation assignment distance | Recorded or supplied observed-cloud target |
| `dpsi-prt-emd-v1` | Same assignment to an endpoint volume target | Separately reconstructed endpoint volume |
| `empm-offline-v1` | Sequence squared Chamfer plus tracked-point error | Per-frame point clouds and genuine tracked dough-point observations |
| `empm-mask-inspired-v1` | Sequence squared Chamfer plus soft-IoU mask loss | Per-frame clouds and supplied foreground/known-pixel masks |

The prepared real recordings currently have partial visible clouds, not persistent measured dough-point tracks or complete later-frame volumes. Tool trajectories and frame indices are not dough-point correspondences. Saved simulated states must not be passed off as measured targets.

The full tracking mode refuses missing tracks when `tracking_weight > 0`. Setting `tracking_weight: 0.0` explicitly selects a **geometry-only ablation**. This is not the complete EMPM offline objective.

## DPSI definitions

Let `Q` be the observed/reconstructed target and `P` the simulated particle positions, both in meters in the same calibrated scene frame.

### Chamfer

```text
CD(Q,P) = sum_q min_p ||q-p||₂ + sum_p min_q ||p-q||₂
```

Distances are unsquared, not Huber-transformed, and not divided by point count. The scalar has units of meters but is a sum, not a mean geometric error. Changing sampling density changes its scale.

### Injective assignment

```text
EMD(Q,P) = min over injective assignments φ:Q→P of sum_q ||q-φ(q)||₂
```

Every target gets a distinct simulated partner; require `len(Q) <= len(P)`. Unmatched simulated points are not penalized. This differs from symmetric Chamfer. SciPy's linear assignment solver computes the exact assignment on the explicitly selected point sets; there is no implicit Sinkhorn or nearest-neighbor replacement.

Assignment checks cardinality, pair count and estimated memory before allocating its cost matrix. A 24,000-by-24,000 float64 cost matrix alone requires about 4.6 GB, before workspace. Oversized requests fail rather than removing points silently.

Optional `target_voxel_size_m`, `target_sample_count`, `predicted_sample_count` and `sampling_seed` describe explicit deterministic sampling. Particle selection remains fixed across evaluations. Counts and selected-index fingerprints are reported. If sampling is enabled, assignment is exact on those sampled sets, not on all original particles.

The paper uses 5 mm voxel downsampling of fused clouds and its own volumetric sampling density. Our examples preserve the existing initial particles; the sampled assignment example is explicitly an adaptation. Do not equate particle sampling density with mass density.

### Endpoint scoring

Only the selected window's explicit `end_frame` contributes to a DPSI objective. Replay still starts at reconstructed frame0 and includes all intervening steps. Export/strict-evaluation frames are kept separate from the endpoint objective frames.

Current PCD examples use calibrated/filtered recorded clouds before image rasterization. These are partial single-view observations, unlike DPSI's fused multi-view acquisition. Symmetric matching can penalize legitimate hidden simulated geometry. The objective does not silently switch to visible-only particles to conceal that difference.

A PRT target must be a separate reconstructed endpoint volume. The initial volume is not the endpoint observation, and filling the later partial cloud to the floor is not automatically justified.

## EMPM definitions and implementation choices

EMPM's paper specifies offline Chamfer plus squared tracked-coordinate errors, summed over time. Its official project currently lists code as coming soon. The paper does not specify the exact Chamfer norm/reduction or mask-loss operator. The following choices are therefore explicit adaptations, not a claim of exact implementation reproduction.

### Offline

```text
C_t = mean_q min_p ||q-p||₂² + mean_p min_q ||p-q||₂²
T_t = sum over valid tracks j of ||x_t[particle_id_j] - observed_track_tj||₂²
L   = sum over selected frames t of (geometric_weight*C_t + tracking_weight*T_t)
```

Both terms use squared metric distances. Geometric terms use means; tracking uses a sum. Weights are used as supplied and are not normalized to sum to one. Example weights are not values recovered from the paper.

Tracks keep fixed correspondences to the exact initial particle file. Reassigning the nearest simulated particle independently each frame is not tracking. Validity masks exclude missing/occluded tracks. The default `empty_track_policy: "error"` refuses a scored frame with no valid tracks; explicit `"skip"` removes only that frame's tracking term, records the missing support, and retains its geometric score.

### Mask-inspired

For foreground mask `M`, known-pixel set `V`, rendered coverage `c`, and positive epsilon:

```text
I = sum over V of c*M
U = sum over V of (c + M - c*M)
mask_loss = 1 - (I + epsilon)/(U + epsilon)
L = sum_t (geometric_weight*C_t + mask_weight*mask_loss_t)
```

Unknown pixels and occluders must be excluded by the supplied known-pixel mask. Sparse depth validity is not a dense foreground/background segmentation. Empty known support is invalid.

The renderer's coverage is a smooth normalized footprint estimate, not binary occupancy. This option reuses its actual coverage adjoint. The mask term is dimensionless, while Chamfer is in m²; weights therefore determine their relative scale explicitly.

This adds a selectable loss only. It does **not** implement EMPM's online controller, quasi-static detection, state resetting, or material updates during replay.

## Checkpoints, windows and datasets

Checkpoint boundaries are for memory management. Positions, velocities, affine velocity, deformation and internal variables continue across them; observations never reconstruct or reset the simulated dough at a boundary.

The current partial loss averages frames. EMPM sums selected frames; DPSI has one scored endpoint. Values and observation-position gradients use the same reduction. Dataset fitting keeps the normalized weighted mean of episode objectives without another frame-count division. Equal weights mean equal endpoint objectives for DPSI and equal sequence totals for EMPM. Per-episode target paths/hashes can differ, while objective definitions/reductions must agree.

Short informative windows can reduce cost and accumulated modeling error. Existing Episode18 frames1–60 already span roughly two seconds. Choose other endpoints from their actual timestamps and contact timing, not a universal frame number. Scoring after an idle period still simulates that preceding interval. Test frozen parameters on later continuation and distinct held-out episodes.

Changing a loss definition, target file, sampling or scoring policy changes run identity and prevents exact resume of an older optimizer trajectory. Strict checkpoint replay remains the default; allowing finite mismatches does not qualify the resulting gradients.

## Example configurations

The following new files are under `configs/`:

- `episode18_dpsi_pcd_cd_v1.json`: recorded-cloud Chamfer with explicit 5 mm target voxelization.
- `episode18_dpsi_pcd_emd_sampled_v1.json`: recorded-cloud assignment with at most 512 targets and 1,024 selected simulation particles; explicit sampling adaptation.
- `episode18_empm_geometry_only_v1.json`: explicit tracking-disabled EMPM geometry ablation.
- `episode18_dpsi_prt_cd_TEMPLATE_NOT_RUNNABLE.json` and `episode18_dpsi_prt_emd_TEMPLATE_NOT_RUNNABLE.json`: require verified endpoint volume files.
- `episode18_empm_tracks_TEMPLATE_NOT_RUNNABLE.json`: requires real tracks and their fixed particle mapping.
- `episode18_empm_masks_TEMPLATE_NOT_RUNNABLE.json`: requires foreground and known-pixel masks.

These files copy the original registered Episode18 material, contact, density and windows unchanged. In particular, they retain its **0.25 kg / approximately 2,196 kg/m³** assignment and fixed tool friction0.5; they do not silently adopt a later fit or the separate ten-episode 1,200 kg/m³ model. To compare losses on another verified scene, use its configuration as the base and change only the loss definition.

`create_loss_examples.py --base-config <verified-config> --output-dir <new-directory>` creates fresh examples and refuses existing filenames. Paths keep the repository-relative semantics of the supplied config. Templates have deliberate path/hash placeholders and must fail validation until completed; replacing a filename alone is insufficient.

### Optimizer tolerances depend on objective units

The existing optimizer defaults remain unchanged: `gradient_tolerance: 1e-6`, `epsilon: 1e-8` and `loss_tolerance: 1e-12`. The stopping test uses the projected gradient in optimizer coordinates, not the physical parameter derivative. Switching from a dimensionless robust loss to raw squared-meter errors can make these absolute values inappropriate.

In the eight-step CPU-f64 synthetic test, the EMPM offline log-stiffness gradient is about `1.1e-10`. The default optimizer therefore reports `converged_gradient` without taking a step, despite a nonzero, finite-difference-verified gradient. The test uses explicit `gradient_tolerance: 1e-14`, `epsilon: 1e-14` and `loss_tolerance: 1e-20` to exercise fitting on that tiny objective. These are **test-specific settings, not recommended defaults for real data**. No loss values or derivatives are secretly rescaled.

For a new real experiment, inspect the initial objective, coordinate gradients and numerical gradient checks before choosing tolerances. Explicit EMPM weights can also set the intended objective scale. Changing weights or tolerances changes run identity. `converged_gradient` alone is not evidence that a material has been identified.

## Additional input files

Optional path pairs use the existing `paths` and `expected_sha256` dictionaries:

- `loss_point_targets` (NPZ) and `loss_point_targets_metadata` (JSON).
- `loss_tracks` (NPZ) and `loss_tracks_metadata` (JSON).
- `loss_masks` (NPZ) and `loss_masks_metadata` (JSON).

Both files in each supplied pair require expected hashes. Metadata binds observations to sequence and calibration identities; tracking additionally binds to initial particles, and segmentation to the actual camera. Frame indices and timestamps must agree with the prepared recording. Loader schema details and exact required metadata fields are defined in `loss_targets.py` and its tests; unknown or inconsistent data is rejected before numerical runtime initialization.

Archive arrays are exact (extra arrays are rejected):

| Archive | Arrays |
| --- | --- |
| Point/volume targets | `frame_indices[F]`, `timestamps[F]`, `offsets[F+1]`, `points[N,3]` |
| Tracks | `frame_indices[F]`, `timestamps[F]`, `track_ids[K]`, `particle_ids[K]`, `positions[F,K,3]`, `valid[F,K]` |
| Masks | `frame_indices[F]`, `timestamps[F]`, `foreground[F,H,W]`, `known[F,H,W]` |

Indices and IDs are integers; mask arrays are boolean or integer0/1. Timestamps are the original processed-sequence timestamps in seconds, not integration-step numbers or wall-clock timestamps. Scored frames must be present. Extra archived frames are permitted and checked against the full sequence. Timestamp matching uses absolute tolerance1e-6s with no relative tolerance.

For example, track metadata uses this structure; the placeholder hashes must be replaced with actual values:

```json
{
  "schema": "taichidough/loss-tracks/v1",
  "units": "m",
  "coordinate_frame": "scene",
  "timestamp_reference": "sequence",
  "sequence_fingerprint": "REPLACE_WITH_VERIFIED_SEQUENCE_FINGERPRINT",
  "calibration_sha256": "REPLACE_WITH_CALIBRATION_FILE_SHA256",
  "initial_particles_sha256": "REPLACE_WITH_INITIAL_PARTICLE_FILE_SHA256",
  "provenance": {
    "kind": "observed",
    "description": "Describe acquisition, tracking, validity and initial particle mapping."
  }
}
```

Point metadata uses schema `taichidough/loss-point-targets/v1`, adds `target_representation` (`partial_observed`, `full_observed`, or `inferred_volume`) and does not include the initial-particle hash. Volume provenance must declare `kind: "inferred_volume"`. Point/track coordinates may instead specify `camera_optical`; the checked camera transform converts them to scene coordinates once.

Mask metadata uses schema `taichidough/loss-masks/v1`, `units: "pixel"`, `coordinate_frame: "image"`, and `camera_fingerprint` instead of the initial-particle hash. That camera fingerprint is the SHA256 of canonical JSON from the actual training camera's `as_dict()`; `loss_targets.camera_fingerprint(camera)` computes it. It binds resolution, intrinsics and transform. Metadata fields are exact; extra source details belong inside `provenance`.

`synthetic_fixture` provenance is accepted for labeled tests, not as evidence of measured real data. All NPZ reads disable pickle. Only finite valid entries contribute; excluded track coordinates are set to zero in the returned copy, with validity retained, so missing values cannot propagate NaNs through residual arithmetic. Input archives and original arrays remain unchanged.

## Commands

From the updated repository root, validate the current-data Chamfer example without initializing MPM:

```bash
python3 -m experiments.differentiable_mpm.calibrate validate \
  --config experiments/differentiable_mpm/configs/episode18_dpsi_pcd_cd_v1.json \
  --reference-policy frozen --no-runtime
```

On `/mnt`, relocate the recorded input explicitly:

```bash
python3 -m experiments.differentiable_mpm.calibrate validate \
  --config experiments/differentiable_mpm/configs/episode18_dpsi_pcd_cd_v1.json \
  --path episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --reference-policy frozen --no-runtime
```

After successful validation and checking the objective scale, an operator may run this command on `/mnt`:

```bash
python3 -m experiments.differentiable_mpm.calibrate fit \
  --config experiments/differentiable_mpm/configs/episode18_dpsi_pcd_cd_v1.json \
  --path episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --reference-policy frozen --backend cuda --iterations 20
```

For local fitting, omit the `/mnt` path override. This work does not launch either fitting command or establish CUDA gradient accuracy. Strict checkpoint replay remains enabled; the finite-mismatch override is not added automatically. Change the config filename to select another mode; tracking/volume/mask templates need verified extra inputs first.

### Updated source on another machine

The earlier `episode18_registered_tools_v1.tar.gz` remains unchanged and does not contain these loss options. Copying a new JSON into that old implementation is insufficient. Use a separate working copy with the current experimental implementation, including:

- New `loss_options.py`, `point_set_loss.py` and `loss_targets.py`.
- Updated `config.py`, `data.py`, `calibrate.py`, `checkpoint.py`, `dataset_config.py`, `calibrate_dataset.py`, `episode_worker.py`, `multi_episode.py` and `run_logging.py`.
- Their current experimental dependencies, including the contact-parameter updates if the remote copy predates them, and the matching unchanged reference files/manifests.
- The new example configurations and their verified Episode18 input assets.

Keep old runs and the original archive intact. No files were transferred to `/mnt` by this task. Run input-only validation in the updated copy before requesting any simulation.

## Verified synthetic checks

All 77 tests in the five new suites passed in the final-source run, including the actual renderer and MPM tests. The broader regression run is recorded separately below when complete.

The MPM fixture uses eight particles, eight steps, CPU-f64 and serial P2G. Full-history and three-step-checkpoint parameter/state adjoints agree under the tests' tolerances; checkpoint state replay uses zero absolute and relative mismatch tolerances. Each observation's gradient is injected exactly once. All six modes pass Young's modulus and viscosity finite differences at two step sizes. The largest reported relative difference across those 24 comparisons is approximately `1.35e-5`.

Four accepted stiffness-only updates reduce every tested objective:

| Synthetic objective | Initial value | Best value after four updates |
| --- | ---: | ---: |
| DPSI PCD-CD and PRT-CD, each tested | `2.18333106815e-5` | `2.86031762716e-7` |
| DPSI PCD-EMD and PRT-EMD, each tested | `1.09166553407e-5` | `1.45780347228e-7` |
| EMPM offline, geometry plus tracks | `2.70741182326e-11` | `6.11784295968e-14` |
| EMPM-inspired mask term in isolation | `0.872189016040` | `0.872155098716` |

These values have different units and reductions and must not be ranked against each other. The mask-only decrease is small; this test checks derivative propagation and accepted descent, not material recovery. The EMPM offline test uses the explicit test-only tolerances described above. All six fits end at the four-update budget, not a demonstrated physical optimum. No replay-mismatch override is used.

Exact output: `runs/paper_loss_verification_20260913_v2/test_paper_loss_trajectory.log`. Taichi emitted existing reverse-kernel uninitialized-local warnings; they remain in the logs. Passing these CPU tests does not qualify all branches, long real trajectories, or CUDA differentiation.

## Verification commands

The new suites are registered with the isolated-process test runner:

```bash
python3 -m experiments.differentiable_mpm.check \
  --only test_point_set_loss test_loss_targets test_loss_options_integration \
         test_runtime_target_validation test_paper_loss_trajectory \
  --reference-policy frozen
```

Point tests cover exact tiny examples, zero-distance subgradients, assignment cardinality and memory checks, repeated nearest neighbors, deterministic sampling, tracking validity and mask derivatives. The trajectory suite uses small synthetic CPU-f64 serial-P2G MPM problems to compare full/checkpointed adjoints, multiple finite-difference steps, temporal reductions, and short optimizer runs. This is not a real-data calibration.

`--quick` includes host target-loader and integration checks but excludes renderer/MPM compilation. The explicit frozen-reference policy uses the preserved simulator where needed; it does not change reference hashes or disable numerical validity checks. Exact test results and unresolved failures will replace the verification-in-progress notice after execution.

## Sources

- [DPSI primary paper](https://arxiv.org/pdf/2411.00554v3), printed pp.7–9: endpoint supervision, Chamfer/assignment equations, omitted point averaging and sampling.
- [EMPM primary paper](https://arxiv.org/html/2601.17251v1), §3.3 Eq.(8) and §3.4 Eq.(9).
- [EMPM official project](https://embodied-mpm.github.io/): implementation listed as coming soon when inspected.

Different objective values and units should not be ranked directly. A smaller paper-style loss does not establish accurate forces, hidden volume, density or material identifiability.

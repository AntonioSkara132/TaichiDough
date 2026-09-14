# Manuscript evidence record

This draft uses existing saved evidence only. It does not treat figure generation or recompilation as a new experiment. The immutable source archive `/home/antonio/Downloads/askara_icra2026(1).zip` has SHA-256 `122a54864bb82e9ce6e8716f09895bb94df3163974cc812d8427e6a1319a1a38`.

## Metric initialization

Source: `data/single_episode_calibration/episode18_kugla/static/evaluation/static_topview_metrics.json`

- Calibration schema: `taichidough/scene-calibration/v2`
- Moving-mocap calibration inliers: 384
- Observed points after filtering: 3,970
- Virtually visible points: 25,495
- Footprint IoU: 0.8305870237
- Extent ratios: 1.0769252, 1.1062478
- Area ratio: 1.1913462
- Centroid error: -0.9754 mm, 1.2524 mm
- Median / p95 absolute depth error: 0.1480 mm / 1.1553 mm

Interpretation: initial visible-geometry alignment only.

## Derivative-free Young's-modulus search

Source: `data/single_episode_calibration/episode18_kugla_fixed_collision_11_9/material_calibration_result.json`

- Result status: `needs_review`
- Selected candidate: 130579.320726 Pa
- Training loss: 0.361058936
- Selection accepted: false
- Reason: broad/flat minimum under the configured tolerance
- One training window; its bootstrap is not uncertainty across independent trials

Interpretation: an effective E-only candidate under the historical fixed setup, not an accepted material estimate.

## Interrupted differentiable dataset fit

Sources:

- `experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd/run_manifest.json`
- `experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd/optimizer_state.json`
- `experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd/result.json`
- `experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd_visualizations_prancer/summary.json`

- Status: `fit_failed`, `KeyboardInterrupt`
- Completed objective evaluations: 17
- Accepted updates: 15
- Initial/best objective: 0.4245516498 / 0.1622877831
- Relative reduction: 61.7743%
- Best-so-far E: 8803.0007 Pa
- Poisson ratio: 0.4870814
- Viscosity: 30.370366 Pa s
- Principal-stretch limits: 0.8626237, 1.0671377
- Episode density: fixed at 1200 kg/m^3
- Reconstructed volume: 0.000113832 m^3; episode mass is derived as 0.1365984 kg
- `use_jp`: false; principal-stretch projection remains active
- Training loss: `partial-visible-splats-v2`, with normalized effective weights 0.5 depth change, 0.25 positive observed coverage, and 0.25 directed observed-to-visible-particle distance
- Depth and distance scales: 0.01 m
- Renderer: 160 x 120, compact polynomial footprint radius 2 px, soft-visibility beta 0.005 m, opacity 8; not Gaussian splatting
- Full-prefix reverse evaluation: 9,839 steps, 59 observations, segment length 64, and 155 host checkpoints
- `ignore_recompute_mismatch`: true
- `replay_consistent`: false
- Recompute mismatches: 2,879 total = 192 checkpoint endpoint-state arrays + 2,687 per-step contact/yield summaries
- Maximum endpoint differences: x = 99.12 um, v = 0.18425 m/s, C = 0.02916 s^-1, F = 0.00015061, Jp = 0
- Maximum scalar observation-loss difference: 3.5763e-7; no observation-loss tolerance failures were recorded

Interpretation: the approximate differentiable optimizer reduced its declared objective. The run is interrupted, not converged or independently validated. The small scalar recomputation-loss difference does not verify the returned gradient as the exact derivative of the recorded forward objective because recomputed states and contact/yield decisions differ.

## Separate strict evaluated fit

Sources:

- `experiments/differentiable_mpm/runs/episode18-table-aligned-registered-tools_fit_20260912T222338_9ab2d0/selected_parameters.json`
- `experiments/differentiable_mpm/runs/episode18-table-aligned-registered-tools_fit_20260912T222338_9ab2d0/training_evaluation.json`
- `experiments/differentiable_mpm/runs/episode18-table-aligned-registered-tools_fit_20260912T222338_9ab2d0/validation_evaluation.json`

- Selected E: 300000 Pa, at its upper bound
- Training strict weighted score: 0.2820641; frozen-initial baseline: 0.3415937
- Frames 61--97 strict weighted score: 0.3939011; frozen-initial baseline: 0.8265162
- All 60 training and 37 later scored frames are valid
- `ignore_recompute_mismatch`: true

Interpretation: visible-geometry prediction improves over frozen initial geometry under this separate configuration. Frames 61--97 are temporal continuation of the same manipulation, not independent-motion generalization.

## Known limitations cited in the paper

Sources: `experiments/differentiable_mpm/VALIDATION.md`, run manifests above, and strict evaluator documentation.

- Full-horizon corrected real-data gradients are not strictly qualified on CUDA.
- The evaluator does not render the tools as depth occluders.
- The observed table estimate differs from the configured floor by about 47.6 mm and 5 degrees in a separate diagnostic.
- Hidden volume, contact, mass/density, reconstruction, calibration, and discretization can compensate for fitted material parameters.
- Current evidence uses one recorded dough sequence and does not establish universal rheological constants.

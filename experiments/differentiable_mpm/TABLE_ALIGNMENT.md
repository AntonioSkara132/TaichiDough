# Episode 18 table alignment

## Generated inputs

All new data is under:

```text
experiments/differentiable_mpm/data/episode18_table_aligned_v1/
```

- `scene_calibration_v2.json`: metric `mocap` → `table-aligned` calibration.
- `episode18_table_aligned_coulomb.json`: differentiable calibration configuration.
- `observed_points_scene.npz`: transformed, source-trimmed observations for all 388 processed frames, with offsets, timestamps and original frame indices.
- `tool_trajectories_scene.npz`: transformed mesh-tool poses for all 388 frames and linear/angular velocities for each interpolation interval.
- `reconstruction/episode18_kugla/frame_0000/`: new voxel centers, visible points and sampled particles, with reconstruction metadata.
- `mass_properties.json`: recomputed volume, density and per-particle mass/volume.
- `reconstruction_command.json` and `reconstruction.log`: exact regeneration command and output.
- `validation.json`: geometry checks, source/output hashes and actual input-loader results for both scoring windows.

The original recordings, original calibration/reconstruction, production implementation and frozen reference files remain unchanged. The old source material manifest is intentionally absent from the new configuration because its calibration and reconstruction hashes describe the old geometry.

The generated directory is about 13 MB and is ignored by Git. Copy it to the same repository-relative location on the CUDA machine; updating source code alone does not transfer these inputs. The experimental source changes must also be available there.

## Transform and conventions

Verified estimator result:

```text
runs/episode18_table_plane_20260912_robust_multiframe_47e2fad1/summary.json
plane = [0.08762950795145376, 0.9961169753375586,
         0.008487684050421995, -0.07360768236928783]
```

For the upward normalized plane `n·p + d = 0`, the rigid rotation maps `n` to `+Y`. The transformed height is exactly the previous signed distance, `y_new = n·p_old + d`. An additional X/Z translation centers the complete observed motion and padded tool-mesh bounds near X=Z=0.5 without scaling the scene. The floor remains `y=0`.

Both `scene_from_source` and `scene_from_camera` are left-multiplied by the identical transform. Recorded source tensors are not rewritten. The existing loader applies the derived calibration to observations and tool positions/orientations, and calculates velocities in the new scene. The exported scene arrays are diagnostic copies, not replacement source inputs; feeding them back through the calibration would transform them twice.

**Gravity assumption:** the measured table normal defines physical up, treating the previous tilt as coordinate-calibration error. Gravity is `[0,-9.81,0]` in the new frame. This is not a model of a physically inclined table under the previous gravity vector.

## Reconstruction and mass

The existing reconstruction script was reused with the original settings: 3 mm voxels, 3 mm observed-point floor clearance, negative-Y floor filling, 1 mm minimum thickness, one footprint dilation, 0.005 source-coordinate trimming and seed 0. No preview or simulation was run.

| Quantity | New value |
|---|---:|
| Occupied voxels | 4,216 |
| Sampled particles | 24,000 |
| Reconstructed volume | 0.000113832 m³ = 113.832 mL |
| Recorded total mass | 0.25 kg |
| Density | 2196.21898938787 kg/m³ |
| Particle volume | 4.743e-9 m³ |
| Particle mass | 1.0416666666666666e-5 kg |
| Median visible-top-to-floor thickness | 7.045 mm |
| Maximum initial particle height | 15.945 mm |

**The resulting density is unusually high for dough.** The measured mass was preserved rather than adjusted to obtain a preferred density. Check the recorded 250 g mass and the visible-top-to-floor volume assumption before interpreting fitted parameters as physical material measurements. The inherited initial material values are starting guesses, not a fit of this geometry.

## Validation performed

- Six focused host tests passed: rigid plane mapping, camera/tool composition, invalid transforms, accepted/rejected frame names and unchanged strict-evaluator loading of the explicit calibration.
- Actual regeneration/input-validation command exited successfully.
- All recorded input and generation-source hashes matched before and after generation.
- Camera-to-source optical transform changed by at most `1.11e-16`.
- Observation and tool-position composition errors were below `3e-8 m`.
- Tool rotation-matrix error was below `1.67e-15`; velocity comparison error was below `1.77e-6` in the respective linear/angular units.
- All initial particles and all retained observed points are in `[0,1)^3` and have valid corrected grid-48 interpolation stencils.
- Continuous tool motion was conservatively bounded using linear center interpolation and a SLERP rotational-displacement bound. X/Z and the upper Y limit stay inside the domain. Tools may extend below the floor; they are prescribed colliders, not simulated particles. No claim is made that every tool vertex is above Y=0.
- Actual preparation passed for training frames 1–60 (10,006 substeps) and validation frames 61–97 (16,176 substeps from initialization). Training preparation also built both collision SDFs.

No Taichi simulation, gradient evaluation, parameter calibration, CUDA qualification or broad regression suite was run for this geometry. Domain validation describes inputs, not the trajectory of a future simulation.

## Calibration settings

- Corrected MLS-MPM with stretch-clamp plasticity.
- Fit E, Poisson ratio, viscosity, plastic minimum and plastic maximum together.
- Non-adhesive `coulomb-v1` tool contact with fixed `mu=0.5`.
- Tool retention is inactive in this contact model; it is not fitted or interpreted as friction.
- Training remains frames 1–60 and held-out scoring remains frames 61–97. All 388 processed frames were transformed, but the configuration does not fit all of them.
- The command below uses atomic P2G and the explicitly requested finite replay-mismatch override. Finite mismatches are logged and can make gradients approximate; invalid states and nonfinite values still reject evaluations. Omit the override for strict rejection.

From the CUDA machine, after copying the new derived directory and updated experimental code:

```bash
cd /mnt/Data/studenti/antonio_skara/TaichiDough

python -u -m experiments.differentiable_mpm.calibrate fit \
  --config experiments/differentiable_mpm/data/episode18_table_aligned_v1/episode18_table_aligned_coulomb.json \
  --path episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --backend cuda \
  --precision f32 \
  --physics-version corrected-v1 \
  --p2g-mode atomic \
  --reference-policy frozen \
  --segment-length 64 \
  --ignore-recompute-mismatch \
  --iterations 20
```

A fresh uniquely named run is created automatically, with optimizer/forward/backward events and final selected-parameter evaluations. Do not resume an old geometry fit with this configuration.

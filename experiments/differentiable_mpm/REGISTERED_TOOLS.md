# Episode 18 registered-tool calibration

## Inputs

Configuration:

```text
experiments/differentiable_mpm/configs/episode18_table_aligned_registered_tools.json
```

Immutable derived inputs:

```text
experiments/differentiable_mpm/data/episode18_registered_tools_v1/
```

- UR5e: full `candidate_marker_from_mesh` from `ur5e_mirrored_collision_registration_20260912_v1/tool_0_result.json`, with the exact mirrored collision STL fitted in that run (SHA256 `65e2afa80894c45dde9ef1791786c8652d1de22a0a6b39de098fb2567552b251`). No additional reflection.
- Gen3: full candidate from `tool_registration_20260912_shared_v1/tool_1_result.json`, with the original collision solid used alongside that visual-mesh registration (SHA256 `c4d7a910e928d265f387003f33638a3ac123f7c8472af5d691882a099fb4867a`).
- The current user-edited Gen3 STL contains all original triangles plus a separate 2 mm cube at the raw STL origin. The original solid was recovered read-only from Git for this experiment. The registered UR STL also includes a separate 2 mm origin cube, preserved to match its actual registration target. Neither user mesh was changed.
- The table-aligned calibration, 24,000 particles and reconstruction metadata are exact copies of the verified table-aligned inputs. The raw episode tensors remain external, unchanged inputs.
- `tool_geometry.json`, `collision_manifest.json`, copied registration results, `preparation_provenance.json` and `validation.json` record the geometry, actual hashes and checks.

Transforms are applied once:

```text
scene_from_source @ source_from_marker(t) @ candidate_marker_from_mesh
                  @ visual_origin_RPY @ millimetres_to_metres
```

The inherited box proxies are inactive in SDF mode; they have not been registered as boxes. Do not switch this configuration to box collision without rebuilding those proxies.

The mesh winding is retained. The simulator determines SDF sign by flood fill from outside the voxelized triangle surface, not STL normals or signed volume. The collision normals are gradients of that signed distance. Negative mesh signed volume alone does not reverse contact in this implementation.

## Validation

Host-only validation passed for training frames 1–60 and held-out frames 61–97, including actual observation preparation, replay controls, mass/reconstruction consistency and both resolution-64 SDFs.

- UR: 98,736 triangles, 49,370 welded vertices; Gen3: 110,712 triangles, 55,356 vertices.
- Both: zero boundary edges, nonmanifold edges, inconsistent-winding edges or degenerate triangles.
- UR SDF: 652 negative voxels; Gen3: 1,815. Independent triangle solid-angle checks classified all 16 sampled interior points per tool as inside and all eight volume corners as outside. All six volume-face gradient checks pointed outward.
- Maximum full-episode transform error: `5.57e-8 m` in position/camera coordinates and `1.67e-15` in rotation.
- Conservative full-episode tool support remains within domain X/Z and upper Y. Prescribed tools can extend below the floor; this check does not assert that every tool vertex is above the table.
- The initial particles retain the previously verified corrected grid-48 stencils.

No MPM simulation, backward pass, CUDA qualification or material fit was run for these registered inputs. Input validation does not establish trajectory stability, gradient quality, or that the previous floating-dough problem is fixed.

## Physics and fitting

- Corrected MLS-MPM, grid 48, `dt=0.0002`, stretch-clamp plasticity, no Jp hardening.
- Floor `y=0`; gravity `[0,-9.81,0]`. The measured table normal defines physical up; the original tilt is treated as coordinate-calibration error.
- Non-adhesive Coulomb tool contact, fixed `mu=0.5`, padding `dx/8`, no stickiness/absorption.
- Jointly fit Young's modulus, Poisson ratio, viscosity, plastic minimum and plastic maximum.
- Initial E = 130,579.320726 Pa, nu = 0.3, viscosity = 0, plastic limits = 0.9/1.1.
- Tool retention is inactive and not fitted. Floor retention remains 0.4.
- Training is frames **1–60**, validation **61–97**. These are not full-episode or multi-episode material fits.
- Total mass 0.25 kg, volume 113.832 mL, density 2196.21898938787 kg/m³. The unusually high density is preserved; check the measured mass and reconstructed volume before interpreting fitted parameters physically.
- Registration uncertainty remains, particularly for UR5e's alternative fits. The accepted transforms are fixed inputs, not parameters fitted here.

## Transfer without overwriting existing work

The archive is created locally at:

```text
experiments/differentiable_mpm/bundles/episode18_registered_tools_v1.tar.gz
experiments/differentiable_mpm/bundles/episode18_registered_tools_v1.tar.gz.sha256
```

It includes the experimental Python sources, frozen reference snapshots, new configuration and derived inputs. It excludes raw recordings, old runs, production sources and user-owned mesh paths. Git alone does not transfer the ignored data directory.

Copy those two archive files to `/mnt/Data/studenti/antonio_skara/` using your usual transfer method. Use the same Python environment as your previous CUDA calibration. Then extract into a **new standalone directory**, not over the existing repository:

```bash
cd /mnt/Data/studenti/antonio_skara
sha256sum -c episode18_registered_tools_v1.tar.gz.sha256 && \
mkdir TaichiDough_registered_tools_v1 && \
tar -xzf episode18_registered_tools_v1.tar.gz -C TaichiDough_registered_tools_v1 && \
cd TaichiDough_registered_tools_v1 && \
sha256sum -c experiments/differentiable_mpm/data/episode18_registered_tools_v1/BUNDLE_SHA256SUMS
```

`mkdir` intentionally fails if this directory already exists. Choose a fresh directory instead of overwriting an earlier extraction. Frozen reference mode supports this standalone copy without production `scripts/` files. Outputs live under this copy's `experiments/differentiable_mpm/runs/`.

### Validate inputs first (no simulation)

```bash
cd /mnt/Data/studenti/antonio_skara/TaichiDough_registered_tools_v1
python -u -m experiments.differentiable_mpm.calibrate validate \
  --config experiments/differentiable_mpm/configs/episode18_table_aligned_registered_tools.json \
  --path episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --backend cuda --precision f32 --physics-version corrected-v1 \
  --reference-policy frozen --no-runtime
```

The original metadata may contain local provenance paths. Leave it unchanged: explicit content/sequence hashes allow the loader to create a path-only relocated metadata copy automatically. Missing required episode files or mismatched content are errors, not reasons to disable hash checks.

### Start calibration when ready

```bash
cd /mnt/Data/studenti/antonio_skara/TaichiDough_registered_tools_v1
python -u -m experiments.differentiable_mpm.calibrate fit \
  --config experiments/differentiable_mpm/configs/episode18_table_aligned_registered_tools.json \
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

A unique run directory is created automatically. Existing console/file logging reports forward/backward progress, optimizer attempts and replay-mismatch warnings; final selected parameters are evaluated on training and held-out frames. Twenty iterations means optimizer attempts, not twenty simulator calls.

The mismatch flag tolerates finite recomputation discrepancies and can make gradients approximate. Invalid/nonfinite states and gradients still reject evaluations. Omit it for strict mismatch rejection. Do not resume an older geometry run with these inputs.

# DeformPath episode preparation for TaichiDough

This recipe turns one ROS 2 DeformPath recording into data that can be used by the TaichiDough material and forward-simulation workflows.

It keeps four things separate:

1. the raw ROS bag;
2. the full mocap-coordinate tensor export;
3. temporal chunk recordings selected from the bag annotation;
4. reconstructed particles and simulation configuration files.

The commands below use `episode20_kugla` as an example. Replace `EPISODE`, the bag path, and the output names for another recording.

## 0. Set paths and the ROS environment

Run from the TaichiDough repository:

```bash
cd /home/antonio/diplomski_antonio/diplomski/TaichiDough

export DATA_ROOT=/home/antonio/diplomski_antonio/diplomski/data/deformpath_training
export EPISODE=episode20_kugla
export RECORDING=snimanje_23_10

# Use the actual bag directory. Release archives may not use the old DeformPath2 path.
export BAG_DIR="$DATA_ROOT/DeformPath2_release/DeformPath/DeformPath2Bags/$RECORDING/$EPISODE"
export FULL_EXPORT="$DATA_ROOT/DeformPath3/$RECORDING/${EPISODE}_taichi_ready"
export TEMPORAL_DIR="$PWD/experiments/differentiable_mpm/data/${EPISODE}_temporal_v2"
export TOOL_BUNDLE="$PWD/experiments/differentiable_mpm/data/episode18_registered_tools_v1"
export BASE_CONFIG="$PWD/experiments/differentiable_mpm/configs/episode18_registered_adhesive_fit.json"

source /opt/ros/humble/setup.bash
source /home/antonio/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Keep the ROS paths while adding the repository to Python's import path.
# Do not use: export PYTHONPATH=.
export PYTHONPATH=.:${PYTHONPATH:-}
export ROS_LOG_DIR=/tmp/roslogs
mkdir -p "$ROS_LOG_DIR"
```

Check these before starting:

```bash
test -f "$BAG_DIR/metadata.yaml"
test -f "$BAG_DIR/temporal_annotations.json"
test -f "$TOOL_BUNDLE/tool_geometry.json"
test -f "$TOOL_BUNDLE/collision_manifest.json"
test ! -e "$FULL_EXPORT"
test ! -e "$TEMPORAL_DIR"
df -h "$DATA_ROOT"
```

Use a new output directory. The preparation scripts refuse to replace existing directories.

## 1. Create or validate temporal annotations

If the bag already contains `temporal_annotations.json`, validate it first:

```bash
cd /home/antonio/diplomski_antonio/diplomski
source /opt/ros/humble/setup.bash
source /home/antonio/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

python3 scripts/annotate_rosbag_temporal_segments.py \
  "$BAG_DIR" \
  --validate-only
```

If it does not exist, create it interactively:

```bash
python3 scripts/annotate_rosbag_temporal_segments.py \
  "$BAG_DIR" \
  --playback-rate 0.2
```

The annotation uses point-cloud header order and half-open ranges `[start, end)`. Do not select ranges by SQLite row number or by a fixed offset after filtering. Later materialization maps the annotation's raw point-cloud ordinals through `sequence_metadata.json["pointcloud_indices"]`.

Record the annotation fingerprint and revision before continuing:

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BAG_DIR"]) / "temporal_annotations.json"
d = json.loads(p.read_text())
print("schema:", d.get("schema_name"), d.get("schema_version"))
print("revision:", d.get("annotation_revision"))
print("fingerprint:", d.get("source", {}).get("fingerprint", {}).get("value"))
print("segments:", len(d.get("segments", [])))
PY
```

## 2. Export the bag to mocap coordinates

Use the calibrated offline exporter. Keep the source tensors in mocap coordinates; the table alignment is applied later through `scene_calibration_v2.json`.

```bash
cd /home/antonio/diplomski_antonio/diplomski/TaichiDough

python3 "$DATA_ROOT/export_deformpath2_offline.py" \
  --bag-dir "$BAG_DIR" \
  --output-dir "$FULL_EXPORT" \
  --output-frame mocap \
  --pose-reference-frame pointcloud \
  --pose-frames UR5e_spathla gen3_spathla \
  --max-points 4096 \
  --seed 0 \
  --camera-tag-parent-frame tag16h5:3 \
  --camera-tag-frame tag3_real \
  --mocap-tag-frame apriltag3 \
  --mocap-frame mocap \
  --camera-frame camera_link
```

For another recording, confirm the tag frames and tool stream names from the bag before running this command. Do not substitute a color-optical transform for a depth-optical transform without a verified calibration chain.

The export should contain at least:

```text
pointclouds.pt
paths.pt
conversion_metadata.json
scene_calibration_v2.json
```

## 3. Interpolate tool poses and filter the point clouds

Use the same point-cloud timeline and keep both tool streams valid. These HSV and DBSCAN settings are the settings used by the Episode 20 table/reconstruction checks; change them only with a recorded reason.

```bash
python3 "$DATA_ROOT/interpolate_deformpath_sequence.py" \
  --input-dir "$FULL_EXPORT" \
  --output-dir "$FULL_EXPORT" \
  --timeline pointcloud \
  --paths-format episode-tensor \
  --require-all-streams \
  --max-points 4096 \
  --seed 0 \
  --h-min 56 --h-max 84 \
  --s-min 18 --s-max 73 \
  --v-min 156 --v-max 255 \
  --dbscan-eps 0.025 \
  --dbscan-min-samples 10
```

Expected additional files:

```text
pointclouds_interpolated.pt
paths_interpolated.pt
sequence_metadata.json
```

Check that the raw and processed sequence lengths are understood before chunking:

```bash
python3 - <<'PY'
import json, os, torch
from pathlib import Path
p = Path(os.environ["FULL_EXPORT"])
raw = torch.load(p / "pointclouds.pt", weights_only=False)[0]
processed = torch.load(p / "pointclouds_interpolated.pt", weights_only=False)[0]
paths = torch.load(p / "paths_interpolated.pt", weights_only=False)[0]
meta = json.loads((p / "sequence_metadata.json").read_text())
print("raw frames:", len(raw))
print("processed frames:", len(processed))
print("path shape:", tuple(paths["path"].shape))
print("tool streams:", paths["pose_frames"])
print("pointcloud indices:", len(meta["pointcloud_indices"]))
PY
```

## 4. Choose the table calibration explicitly

The scene calibration must declare:

```text
source_frame = mocap
scene_frame = table-aligned
floor_plane_scene = [0, 1, 0, 0]
scene_from_source = finite, nonidentity 4x4 rigid transform
```

There are two valid policies:

### New table calibration

Estimate the table plane from the current recording with the repository's table estimator. The estimator requires the HSV settings from step 3 and currently writes JSON using NumPy scalar values; if that repository bug is present, fix JSON serialization in the estimator before using the report.

### Reuse a validated calibration

If the recording was made with the same physical table and coordinate setup, reuse a known calibration, for example:

```text
TaichiDough/experiments/differentiable_mpm/data/episode18_table_aligned_v1/scene_calibration_v2.json
```

Do not silently copy it. Create a new Episode-specific `scene_calibration_v2.json` that keeps the transform unchanged and records:

- the source calibration path and SHA-256;
- `independent_episode_plane_estimation: false`;
- `source_tensors_transformed: false`;
- the fact that the table calibration was reused by decision.

Update the calibration records in both `conversion_metadata.json` and `sequence_metadata.json`, including their SHA-256 values and the sequence's `source_conversion_metadata_sha256`.

Create an Episode-specific table-validation report. It may report the reused calibration as accepted, but it must say that this is reuse and not an independent measurement.

## 5. Materialize annotated temporal ranges

For a reusable episode-specific package, create a `range_selection.json` that contains:

- the annotation fingerprint;
- the source recording ID;
- the selected half-open raw point-cloud ranges;
- the source hashes;
- the calibration policy.

Then run the shared materializer:

```bash
cd /home/antonio/diplomski_antonio/diplomski/TaichiDough

python3 -m experiments.differentiable_mpm.materialize_deformpath_ranges \
  --full-episode-dir "$FULL_EXPORT" \
  --conversion-metadata "$FULL_EXPORT/conversion_metadata.json" \
  --range-selection "$TEMPORAL_DIR/range_selection.json" \
  --annotation "$TEMPORAL_DIR/accepted_temporal_annotations.json" \
  --output-root "$TEMPORAL_DIR/recordings" \
  --max-gap-s 0.1
```

Each materialized recording must contain:

```text
pointclouds_interpolated.pt
paths_interpolated.pt
sequence_metadata.json
```

Check every chunk's frame count, timestamps, tool-stream validity, and source hash. Do not rebase timestamps unless the simulation configuration explicitly requires it.

## 6. Reconstruct the initial material state

For each chunk, reconstruct frame zero using the calibrated scene transform:

```bash
python3 scripts/reconstruct_voxel_dough_from_deformpath.py \
  --episode-dir "$CHUNK_DIR" \
  --pointclouds-name pointclouds_interpolated.pt \
  --frame 0 \
  --output-dir "$RECONSTRUCTION_OUTPUT" \
  --num-particles 24000 \
  --voxel-size 0.003 \
  --fill-mode floor \
  --fill-axis y \
  --fill-direction negative \
  --floor-clearance 0.003 \
  --floor-min-thickness 0.001 \
  --floor-max-thickness 0.25 \
  --bbox-padding 0.006 \
  --trim-quantile 0.005 \
  --footprint-dilate 1 \
  --seed 0 \
  --calibration "$CALIBRATION"
```

The reconstruction output must include:

```text
source_filtered_points_xyz.npy
scene_transformed_points_xyz.npy
real_points_xyz.npy
voxel_centers_xyz.npy
surface_particles_xyz.npy
sampled_particles_xyz.npy
reconstruction_metadata.json
```

Validate that every `sampled_particles_xyz.npy` has shape `(24000, 3)`, contains finite values, stays inside the corrected-v1 MPM grid stencil, and is labeled `table-aligned` in its metadata.

## 7. Build the Taichi dataset manifest

Create one config per materialized chunk. Reuse the checked physics settings unless the experiment requires another configuration:

```text
physics_version: corrected-v1
particles: 24000
grid: 48
plasticity: stretch-clamp
tool_collision: sdf
tool_contact_model: coulomb-adhesive-v1
tool_contact_absorption: 0
```

Each config must point to:

- its chunk recording;
- its calibration;
- its reconstructed `sampled_particles_xyz.npy`;
- its reconstruction metadata;
- tool geometry and collision assets;
- a verified expected SHA-256 for every input.

The dataset manifest should identify training and validation frames explicitly. For independent temporal chunks, frame zero is initialization and the final frame can be reserved for within-chunk validation. Do not call these independent held-out episodes unless they are independent recordings.

## 8. Final validation checklist

Before running Taichi, all of these should pass:

```bash
# Export
# - 250 raw point-cloud frames, if the annotation says 250
# - mocap source coordinates
# - calibration v2 present
# - nonidentity mocap-to-table transform
# - calibration and conversion hashes agree

# Temporal package
# - annotation copied byte-for-byte
# - expected fingerprint and revision match
# - all selected ranges are half-open and non-overlapping
# - every chunk has increasing timestamps and both tool streams
# - SHA256SUMS passes

# Reconstruction
# - every chunk has a valid reconstruction manifest
# - every particle array has shape (24000, 3)
# - source arrays are mocap; reconstructed arrays are table-aligned
# - floor and MPM stencil checks pass

# Final dataset
# - every config loads
# - every input hash verifies
# - tool/collision assets are present
# - mass and geometry assumptions are recorded
# - preparation says simulation_run=false
```

Run the repository's focused tests before changing the workflow:

```bash
pytest -q \
  experiments/differentiable_mpm/tests/test_materialize_deformpath_ranges.py \
  experiments/differentiable_mpm/tests/test_prepare_episode20_reconstructions.py \
  experiments/differentiable_mpm/tests/test_finalize_episode20_kugla_temporal.py
```

## Assumptions to record for every new episode

- Whether the table calibration was measured for this recording or reused.
- Whether tool geometry and collision meshes were measured or inherited.
- Whether mass and density were measured or inherited.
- The exact point cap, HSV thresholds, DBSCAN settings, voxel size, and particle count.
- The annotation fingerprint and selected raw frame ranges.
- Whether chunks are independent episodes or consecutive sections of one recording.
- Whether material fitting was run. Preparation alone must say that fitting and simulation were not run.

Never replace a mismatched calibration, tool manifest, or temporal annotation just to make a validator pass. Stop and record the mismatch first.

# TaichiDough

TaichiDough is a Taichi MLS-MPM prototype for reconstructing, simulating, and evaluating deformable dough. It includes:

- a two-tool viscoelastic and viscoplastic MPM scene;
- metric scene calibration from one selected AprilTag;
- floor-terminated voxel reconstruction from DeformPath point clouds;
- mass-preserving initialization from reconstructed volume;
- timestamped replay of two measured tool colliders;
- static and dynamic visible-geometry evaluation;
- effective Young's-modulus identification from recorded motion;
- GUI, video, virtual depth-camera, UDP, Gymnasium, and Stable-Baselines3 utilities.

The repository includes the ROS 2 AprilTag calibration collector. `rosbag2`, `image_proc`, `apriltag_ros`, `apriltag_msgs`, `tf2_ros`, and the camera message packages remain ROS 2 runtime dependencies. A robot-control bridge is not included.

## Install

Use Python 3.10 or newer:

```bash
python3 -m pip install -r requirements.txt
```

For NVIDIA acceleration, install a compatible NVIDIA driver first. Taichi prints the selected backend at startup. Pass `--cpu` when deterministic CPU execution or a machine without a supported GPU is required.

Source the ROS 2 installation and workspace that contain the AprilTag packages before collecting calibration data:

```bash
source /opt/ros/<ROS_DISTRO>/setup.bash
source <ROS_WORKSPACE>/install/setup.bash
```

## Run the procedural MPM scene

```bash
python3 scripts/taichi_viscoelastic_mpm_scene.py \
  --gui \
  --no-save \
  --free-camera \
  --steps 10000 \
  --tool-contact-padding 0.05 \
  --tool-contact-friction 0.5
```

CPU version:

```bash
python3 scripts/taichi_viscoelastic_mpm_scene.py \
  --cpu \
  --gui \
  --no-save \
  --free-camera \
  --steps 10000 \
  --tool-contact-padding 0.05 \
  --tool-contact-friction 0.5
```

Procedural initialization derives particle volume from the MPM grid. Calibrated experiments must instead use reconstructed particles and `--initial-particles-metadata`, as shown below.

## STL/SDF tool collision

The default `--tool-collision box` preserves calibrated proxy geometry used by replay, LeRobot, and UDP workflows. Use `--tool-collision sdf` to build a voxel SDF from the bundled `ur_spathla.stl` and `gen3_spathla.stl` meshes, or provide replacements with `--ur-tool-mesh` and `--kinova-tool-mesh`. Mesh coordinates are millimetres by default (`--tool-mesh-scale 0.001`); `--tool-sdf-resolution` controls the approximation resolution.

Captured replay supports both representations. Box replay composes each captured marker pose with `marker_from_collider`; SDF replay composes it with `marker_from_mesh`, which maps the marker into the mesh/tool-link frame. `marker_from_mesh` defaults to identity for existing geometry documents, but a non-identity value must be calibrated when the captured marker is not already the tool-link frame. The URDF visual origin and rotation are already applied while each SDF is built, so they must not be included in `marker_from_mesh`.

## Record video

```bash
python3 scripts/taichi_viscoelastic_mpm_scene.py \
  --gui \
  --free-camera \
  --record-video \
  --video-path data/taichi_mpm_camera_views/viscoelastic_mpm_gui.mp4 \
  --video-simulation-time \
  --steps 10000
```

`--video-simulation-time` sets the video frame rate so playback duration matches simulated time.

## Export virtual depth point clouds

Saved frames can include occlusion-aware point clouds from the virtual pinhole camera:

```bash
python3 scripts/taichi_viscoelastic_mpm_scene.py \
  --save-depth-pointclouds \
  --depth-pointcloud-save-pt \
  --depth-pointcloud-frame camera \
  --depth-pointcloud-max-points 1024 \
  --depth-width 640 \
  --depth-height 480 \
  --save-every 10 \
  --steps 10000
```

The default point-cloud format is `deformpath7`: XYZ plus `[0, 0, 1, 0]`, matching the seven-column DeformPath tensors. Use `--depth-pointcloud-format xyz` for plain XYZ arrays. A metric v2 calibration uses its exact `width`, `height`, `fx`, `fy`, `cx`, and `cy` instead of an approximate field-of-view camera.

## Stream Taichi point clouds over UDP

```bash
python3 scripts/taichi_depth_udp_observation_sender.py \
  --send-host 127.0.0.1 \
  --send-port 6005 \
  --view top_dough \
  --pointcloud-frame camera \
  --hz 30 \
  --max-points 256 \
  --steps 300
```

Port `6005` avoids the port commonly used by the Taichi/SOFA-style tool-control bridge. Keep UDP observations small enough for one datagram. Save dense point clouds to files with `--save-depth-pointclouds`.

## Metric AprilTag scene calibration

### Required measurements

A real calibrated run requires all of the following:

1. One tagged recording. Directories labelled `nema_apriltag` cannot produce this calibration.
2. The selected tag family and ID. This setup uses family `16h5`. Existing recording notes disagree between IDs 3 and 4, so every command must name the ID explicitly.
3. The tag's physical detection-edge length. This is the nominal 57 mm tag; use the configured value **0.0565 m** for both `apriltag_ros` and the collector.
4. A measured rigid `scene_from_tag` 4×4 matrix in row-major order. It defines the tag position and orientation in the simulation scene; the collector does not estimate tag placement relative to the floor.
5. A scene-frame floor plane `[nx, ny, nz, d]`, where `n·p + d = 0`. The intended scene uses metres and `+Y` as the upward floor normal. For a horizontal floor at `y = floor_y`, use `[0, 1, 0, -floor_y]`.

The configured 56.5 mm edge must span the four corners used by the detector. The physical tag is nominally 57 mm wide. Do not include a white quiet zone, paper margin, or mounting plate. This follows the [AprilTag detection-edge convention](https://docs.ros.org/en/rolling/p/apriltag/) used by [`apriltag_ros`](https://docs.ros.org/en/jazzy/p/apriltag_ros/index.html). An incorrect edge length scales the estimated translation by the same proportion.

Inspect the AprilTag parameter file instead of trusting its filename. For example, the installed `tags_36h11.yaml` may contain a different `family` or `size`. Use a recording-specific file with explicit values:

```yaml
/**:
  ros__parameters:
    image_transport: raw
    family: 16h5
    size: 0.0565
    max_hamming: 0
    pose_estimation_method: pnp
    tag:
      ids: [<TAG_ID>]
```

Replace `<TAG_ID>` with the selected tag. The YAML `size` and collector `--tag-edge-size-m` must both equal `0.0565`; the collector intentionally rejects mismatched values.

### Replay, rectify, detect, and collect

The inspected recordings use these topics:

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/depth/color/points
/tf
/tf_static
```

Message header frame IDs determine transform directions; topic names do not determine coordinate frames.

Run bag replay in one terminal:

```bash
ros2 bag play <TAGGED_ROSBAG_DIRECTORY> --clock
```

If the color image requires rectification, run `image_proc` in another terminal:

```bash
ros2 run image_proc rectify_node --ros-args \
  -r image:=/camera/camera/color/image_raw \
  -r camera_info:=/camera/camera/color/camera_info \
  -r image_rect:=/camera/camera/color/image_rect
```

If the camera driver already publishes a rectified color image with matching `CameraInfo`, use that topic directly. Verify this from the camera model rather than assuming that an `image_raw` topic is rectified.

Run `apriltag_ros` with the checked parameter file:

```bash
ros2 run apriltag_ros apriltag_node --ros-args \
  -r image_rect:=/camera/camera/color/image_rect \
  -r camera_info:=/camera/camera/color/camera_info \
  -r detections:=/apriltag/detections \
  --params-file configs/apriltag_recording.yaml
```

Then collect the metric calibration. Replace the 16 matrix entries and floor coefficient with measured values:

```bash
python3 scripts/collect_apriltag_scene_calibration.py \
  --output data/calibration/episode18_scene_v2.json \
  --raw-log data/calibration/episode18_scene_v2.observations.jsonl \
  --detections-topic /apriltag/detections \
  --camera-info-topic /camera/camera/color/camera_info \
  --pointcloud-topic /camera/camera/depth/color/points \
  --tag-family 16h5 \
  --tag-id <TAG_ID> \
  --tag-edge-size-m 0.0565 \
  --scene-frame taichi_scene \
  --scene-from-tag \
    <R00> <R01> <R02> <TX> \
    <R10> <R11> <R12> <TY> \
    <R20> <R21> <R22> <TZ> \
    0 0 0 1 \
  --floor-plane-scene 0 1 0 <NEGATIVE_FLOOR_Y> \
  --samples 30 \
  --min-inliers 10
```

The collector accepts only the configured family and ID. It also checks Hamming distance, decision margin, finite poses, camera and point-cloud timestamps, frame IDs, and the TF path between the point-cloud and color optical frames. It robustly rejects translation and rotation outliers and writes both the accepted calibration and every raw acceptance or rejection record.

The resulting `taichidough/scene-calibration/v2` file contains metric rigid transforms only:

```text
scene_from_camera = scene_from_tag @ inverse(camera_from_tag)
scene_from_source = scene_from_camera @ camera_from_source
```

It also records exact color-camera intrinsics, the floor plane, tag provenance, TF records, diagnostics, and SHA-256 fingerprints. There is no fitted axis map, scale, translation, registration, or approximate field of view in a v2 calibrated run.

### Offline moving-tag mocap-world export

`collect_apriltag_scene_calibration.py` is for a fixed tag observed during ROS replay. The DeformPath recordings instead provide a moving `apriltag3` pose in `/poses` while the camera is fixed in `mocap`. Export those recordings with the offline exporter:

```bash
cd /home/antonio/diplomski_antonio/diplomski/data/deformpath_training
python3 export_deformpath2_offline.py \
  --bag-dir DeformPath2/snimanje_23_10/episode18_kugla \
  --output-dir DeformPath3/snimanje_23_10/episode18_kugla \
  --camera-tag-parent-frame tag16h5:3 \
  --camera-tag-frame tag3_real \
  --mocap-tag-frame apriltag3 \
  --output-frame mocap
python3 interpolate_deformpath_sequence.py \
  --input-dir DeformPath3/snimanje_23_10/episode18_kugla \
  --output-dir DeformPath3/snimanje_23_10/episode18_kugla
```

For each synchronized frame, it estimates `mocap_from_camera = mocap_from_apriltag3 @ inverse(camera_from_tag3_real)`, rejects stale samples and outliers, then stores one robust camera pose in `scene_calibration_v2.json`. It transforms every point cloud into `mocap`, keeps both tool paths in `mocap`, and records `source_frame = scene_frame = mocap`, `scene_from_source = identity`, and the confirmed floor `[0, 1, 0, 0]`. Interpolation verifies and copies that artifact into the processed episode and writes its SHA-256 and canonical fingerprint to `sequence_metadata.json`.

A processed episode with no attached calibration remains a legacy/non-metric episode: supply an explicit calibration file, or re-export the tagged recording. Do not use a synthetic calibration for a real recording.

## Reconstruct dough down to the calibrated floor

Floor mode transforms the selected DeformPath frame into scene metres exactly once, removes points at or below the floor clearance, groups the visible dough by occupied columns, and intersects each column with the calibrated floor:

```bash
EPISODE=/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/Deformapth2_downsampled_interpolated/snimanje_23_10/episode18_kugla
CALIBRATION=data/calibration/episode18_scene_v2.json

python3 scripts/reconstruct_voxel_dough_from_deformpath.py \
  --episode-dir "$EPISODE" \
  --frame 0 \
  --output-dir data/voxel_reconstruction_v2 \
  --num-particles 24000 \
  --voxel-size 0.003 \
  --fill-mode floor \
  --fill-axis y \
  --fill-direction negative \
  --floor-clearance 0.003 \
  --floor-min-thickness 0.001 \
  --floor-max-thickness 0.25 \
  --footprint-dilate 1 \
  --calibration "$CALIBRATION" \
  --save-pt
```

Add `--scene-bounds X_MIN X_MAX Y_MIN Y_MAX Z_MIN Z_MAX` after measuring the usable workspace. Bounds are applied in scene metres after calibration.

For the command above, files are written under:

```text
data/voxel_reconstruction_v2/episode18_kugla/frame_0000/
```

Important outputs are:

- `source_filtered_points_xyz.npy`: filtered points in the point-cloud source frame;
- `scene_transformed_points_xyz.npy`: transformed points in the scene frame;
- `real_points_xyz.npy`: retained visible dough points in the reconstruction frame;
- `voxel_centers_xyz.npy`: occupied voxel centres;
- `surface_particles_xyz.npy`: visible points retained for particle sampling;
- `sampled_particles_xyz.npy`: deterministic MPM initial particles;
- `reconstruction_metadata.json`: frames, floor, rejection counts, thickness statistics, settings, fingerprints, voxel count, and volume.

The reconstructed object volume is:

```text
object_volume_m3 = voxel_count * voxel_size^3
```

`--fill-mode floor` requires a metric v2 calibration and rejects `--thickness` and `--fill-limit`. Omitting `--fill-mode` keeps the legacy fixed-thickness behavior for reproducing earlier runs; do not use that implicit mode for new calibrated results.

## Measured two-tool replay geometry

Recorded replay uses `taichidough/tool-geometry/v1`. Tool names must match the two names in `paths_interpolated.pt` under `pose_frames`. Each `half_extents_m` entry contains half the measured full collider dimensions in scene metres. Each `marker_from_collider` matrix gives the collider pose in that tool marker's local frame:

```json
{
  "schema": "taichidough/tool-geometry/v1",
  "tools": [
    {
      "name": "<FIRST_POSE_FRAME>",
      "half_extents_m": ["<HALF_X_M>", "<HALF_Y_M>", "<HALF_Z_M>"],
      "marker_from_collider": [
        ["<R00>", "<R01>", "<R02>", "<TX_M>"],
        ["<R10>", "<R11>", "<R12>", "<TY_M>"],
        ["<R20>", "<R21>", "<R22>", "<TZ_M>"],
        [0, 0, 0, 1]
      ],
      "marker_from_mesh": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ]
    },
    {
      "name": "<SECOND_POSE_FRAME>",
      "half_extents_m": ["<HALF_X_M>", "<HALF_Y_M>", "<HALF_Z_M>"],
      "marker_from_collider": [
        ["<R00>", "<R01>", "<R02>", "<TX_M>"],
        ["<R10>", "<R11>", "<R12>", "<TY_M>"],
        ["<R20>", "<R21>", "<R22>", "<TZ_M>"],
        [0, 0, 0, 1]
      ],
      "marker_from_mesh": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ]
    }
  ]
}
```

Store measured values as JSON numbers, not the strings shown as placeholders. The simulator reorders the two entries to match the recorded pose-stream order and applies the complete local rotation and translation. `marker_from_collider` is required and supplies `source_from_marker @ marker_from_collider` for box replay. `marker_from_mesh` is optional and supplies `source_from_marker @ marker_from_mesh` for `--tool-collision sdf`; omitting it selects an identity transform for legacy v1 documents. It maps marker to the SDF mesh/tool-link frame, not raw STL visual coordinates: do not include URDF visual origins or RPY rotations because SDF construction already applies them. Collider contact velocity includes both linear velocity and `angular_velocity × (particle_position - collider_center)`.

`--tool-half-extents` and `--tool-marker-offset` remain available for smoke tests. They create two identical proxy boxes and identity mesh transforms, and must not be described as measured tool geometry.

## Volume- and mass-preserving MPM initialization

Use the reconstructed particle file together with its metadata:

```bash
RECONSTRUCTION=data/voxel_reconstruction_v2/episode18_kugla/frame_0000
DOUGH_MASS_KG=<MEASURED_DOUGH_MASS_KG>

python3 scripts/taichi_viscoelastic_mpm_scene.py \
  --cpu \
  --steps 0 \
  --save-initial-frame \
  --no-publish-dough-center \
  --view deformpath_top \
  --initial-particles "$RECONSTRUCTION/sampled_particles_xyz.npy" \
  --initial-particles-metadata "$RECONSTRUCTION/reconstruction_metadata.json" \
  --initial-particles-calibration "$CALIBRATION" \
  --initial-particles-fit none \
  --initial-particles-axis-map xyz \
  --object-mass-kg "$DOUGH_MASS_KG" \
  --save-depth-pointclouds \
  --depth-pointcloud-frame camera_optical \
  --depth-pointcloud-format xyz \
  --output-dir "$RECONSTRUCTION/taichi_static"
```

For reconstructed initialization:

```text
particle_volume_m3 = object_volume_m3 / particle_count
density_kg_m3      = measured_object_mass_kg / object_volume_m3
particle_mass_kg   = density_kg_m3 * particle_volume_m3
```

If mass is unavailable, omit `--object-mass-kg` and pass a justified `--density` with its source recorded outside the command. Measured mass is preferred. Total object volume and mass stay constant when `--particles` or `--grid` changes.

Reconstructed particles are already in scene coordinates. The simulator verifies their path, SHA-256 value, calibration fingerprint, frame, volume, and floor plane. It rejects an additional axis permutation, fit, scale, or offset.

## Static initialization evaluation

The static evaluator compares the captured and simulated initial visible geometry without translating, scaling, or registering either point set:

```bash
python3 scripts/evaluate_static_topview_match.py \
  --episode-dir "$EPISODE" \
  --frame 0 \
  --calibration "$CALIBRATION" \
  --taichi-metadata "$RECONSTRUCTION/taichi_static/camera_parameters.json" \
  --output-dir "$RECONSTRUCTION/static_evaluation"

python3 scripts/visualize_static_topview_benchmark.py \
  --reconstruction-metadata "$RECONSTRUCTION/reconstruction_metadata.json" \
  --taichi-metadata "$RECONSTRUCTION/taichi_static/camera_parameters.json" \
  --metrics "$RECONSTRUCTION/static_evaluation/static_topview_metrics.json" \
  --calibration "$CALIBRATION" \
  --output-dir "$RECONSTRUCTION/static_diagnostics"
```

The evaluator records footprint IoU, metric extents, area, centroid offset, visible coverage, and depth residuals. The report adds retained and rejected points, floor-plane samples, per-column floor intersections, local thickness, reconstructed volume, and scene/source/camera/tag axes. Inspect these diagnostics before running material identification.

## Timestamped two-tool replay

This example uses reduced numerical settings to test the replay path. It is not a resolution study:

```bash
TOOL_GEOMETRY=configs/tool_geometry_measured.json
RUN=data/dynamic_topview_match/episode18_kugla/run_01

python3 scripts/taichi_viscoelastic_mpm_scene.py \
  --cpu \
  --no-publish-dough-center \
  --replay-episode "$EPISODE" \
  --replay-start-frame 0 \
  --replay-end-frame 60 \
  --replay-stride 6 \
  --replay-max-gap 0.1 \
  --initial-particles "$RECONSTRUCTION/sampled_particles_xyz.npy" \
  --initial-particles-metadata "$RECONSTRUCTION/reconstruction_metadata.json" \
  --initial-particles-calibration "$CALIBRATION" \
  --initial-particles-fit none \
  --initial-particles-axis-map xyz \
  --object-mass-kg "$DOUGH_MASS_KG" \
  --tool-geometry "$TOOL_GEOMETRY" \
  --particles 3000 \
  --grid 24 \
  --depth-splat-radius 1 \
  --output-dir "$RUN/taichi"

python3 scripts/evaluate_dynamic_topview_match.py \
  --episode-dir "$EPISODE" \
  --taichi-metadata "$RUN/taichi/camera_parameters.json" \
  --calibration "$CALIBRATION" \
  --frame-stride 6 \
  --output-dir "$RUN/evaluation"

python3 scripts/visualize_dynamic_topview_benchmark.py \
  --metrics "$RUN/evaluation/dynamic_topview_metrics.json" \
  --output-dir "$RUN/diagnostics"
```

Replay uses recorded timestamps, linear interpolation for collider position, and shortest-path quaternion interpolation for orientation. `--steps` is ignored. Both tool streams must remain finite and valid throughout the inclusive start/end interval. A gap larger than `--replay-max-gap` is rejected. `--replay-stride` controls saved observation cadence; every captured pose still contributes to interpolation.

The dynamic evaluator pairs frames by timestamp. Its default tolerance is one integration timestep. Observations outside the tolerance remain unpaired, and frames without enough common visible support remain unavailable rather than receiving zero error. The comparison uses the reconstruction's floor clearance and scene bounds to remove floor and tag points consistently.

Errors are simulation minus observation on common visible support. The report includes depth, silhouette, coverage, one-sided visible-point distance, change relative to the initial frame, timing, and tool-motion diagnostics. It also reports a frozen-initial-state baseline. It does not fit translation, scale, rigid registration, or time warping.

## Single-episode material launcher

For a mocap-world processed episode, the launcher resolves and verifies the calibration attached to `sequence_metadata.json`; do not set `CALIBRATION`:

```bash
EPISODE=/data/DeformPath3/snimanje_23_10/episode18_kugla \
TOOL_GEOMETRY=configs/tool_geometry_measured.json \
DOUGH_MASS_KG=<measured_kg> \
TRAIN_END_FRAME=60 VALIDATION_END_FRAME=120 \
bash scripts/run_single_episode_material_calibration.sh
```

It uses the bundled UR and Gen3 STL meshes with SDF collision at millimetre-to-metre scale `0.001` unless overridden. The recorded `UR5e_spathla` and `gen3_spathla` frames match the mesh tool-link frames, so both `marker_from_mesh` matrices are identity. An explicit `CALIBRATION` is accepted only if its fingerprint matches the episode attachment; `ALLOW_CALIBRATION_OVERRIDE=1` documents an intentional mismatch.

## Identify an effective Young's modulus

The material-calibration driver starts every candidate from the same reconstructed state, replays the same measured tool motion, varies only Young's modulus, and compares predicted visible geometry with later recorded geometry. Force measurements are not required for this trajectory-based fit.

Start from the complete manifest template:

```bash
cp configs/material_calibration.example.json configs/material_calibration.local.json
```

Fill every required placeholder. The manifest records:

- calibration, reconstruction, particle, sequence, simulator, evaluator, and tool-geometry paths and SHA-256 values;
- measured dough mass and measured tool-geometry fingerprint;
- fixed constitutive, plasticity, contact, gravity, timing, camera, particle, grid, timestep, backend, and seed values;
- one or more training deformation windows and at least one held-out validation window;
- component weights and minimum valid-support thresholds;
- logarithmic search, refinement, boundary expansion, bootstrap, cache, and timeout settings.

Every rollout begins at the frame recorded in `reconstruction_metadata.json` and continues through that window's `end_frame`. A window's inclusive `start_frame` and `end_frame` select only the evaluator frames included in its loss. This preserves the complete recorded motion before a later validation interval while keeping those earlier frames out of held-out scoring.

Generate file hashes with `sha256sum`. Sequence and tool-geometry fingerprints are reported by their loaders and simulation metadata; copy them exactly rather than hashing a different representation.

Validate the experiment before spending time on simulations:

```bash
python3 scripts/calibrate_youngs_modulus.py \
  configs/material_calibration.local.json \
  --validate-only
```

Run or resume the search:

```bash
python3 scripts/calibrate_youngs_modulus.py \
  configs/material_calibration.local.json
```

The initial range can use `1e4` to `3e5 Pa` as a literature-informed starting interval. The existing `2000 Pa` setting remains a separately scored baseline. These are search choices, not measured answers. The driver evaluates a logarithmic coarse grid, refines around an interior minimum, and expands the range when the best candidate lies at a boundary, subject to the manifest's hard limits.

Each simulation/evaluation pair is cached under a hash of the complete fixed inputs, code fingerprints, settings, window, and candidate modulus. Repeating the command reuses only complete, verified entries. Add `--retry-failures` to rerun candidates recorded as failed.

Create the report after a successful run:

```bash
python3 scripts/visualize_material_calibration.py \
  <MATERIAL_RESULT_JSON> \
  --output-dir data/material_calibration/report
```

The report contains HTML, JSON, candidate CSV, and per-window CSV files. It plots total and component losses on a logarithmic modulus axis, marks baselines and selection status, reports per-window optima, and gives a bootstrap interval obtained by resampling whole deformation windows.

A usable result requires an informative interior training minimum and acceptable held-out prediction without refitting. A boundary or flat minimum indicates that the data does not identify Young's modulus under the current setup.

The selected value is an **effective Young's modulus for this constitutive model, viscosity, plasticity, contact treatment, reconstructed volume, mass assumption, and numerical discretization**. It is not automatically a universal material constant for dough.

### Held-out validation and numerical convergence

Keep validation windows out of candidate selection. Prefer motion that differs from the training window while retaining the same calibrated object and tools. Compare the selected value against the default, deliberately softer and stiffer candidates, and the frozen-initial-state baseline on every loss component.

Repeat the full fit with multiple particle counts, grid resolutions, and timesteps while preserving reconstructed volume and total mass. Report selected modulus and held-out loss for each setting. Do not claim a calibrated value if either changes materially with numerical resolution.

Use CPU when the selected GPU backend does not reproduce candidate losses within the documented tolerance. Run each candidate twice before relying on a narrow minimum.

## Legacy reproduction modes

Version 1 calibration files with signed axis maps, fitted scales, and translations remain loadable for reproducing earlier results. Reconstruction also retains explicit `thickness` and `limit` modes, and old particle files can still use axis mapping and fit options.

Do not mix those options into a metric v2 result. New floor-calibrated particles must use identity axis order, no fit, unit scale, zero offset, matching reconstruction metadata, and matching calibration fingerprints.

## Regression and compilation checks

Run the complete regression set from the repository root:

```bash
python3 scripts/test_deformpath_topview.py
python3 scripts/test_deformpath_dynamics.py
python3 scripts/test_apriltag_scene_calibration.py
python3 scripts/test_floor_reconstruction.py
python3 scripts/test_mpm_mass.py
python3 scripts/test_material_calibration.py
python3 -m compileall scripts
```

The ROS-independent collector tests and `--help` command run without ROS imports. Real calibration collection still requires the ROS 2 packages listed above.

## Dough falling scene

```bash
python3 scripts/taichi_dough_fall_mpm.py --gui --no-save --free-camera
```

## Gymnasium wrapper

```python
from scripts.taichi_lerobot_env import TaichiViscoelasticMPMEnv

env = TaichiViscoelasticMPMEnv()
observation, info = env.reset()
action = [0, 0, -0.05, 0, 0, 0.05]
observation, reward, terminated, truncated, info = env.step(action)
```

The reward is:

```text
-mean_i ||x_i - mean_j(x_j)||^2
```

It rewards compact dough particle distributions.

## Stable-Baselines3 smoke training

```bash
python3 scripts/train_taichi_sb3.py \
  --cpu \
  --algo sac \
  --timesteps 80 \
  --particles 700 \
  --grid 20 \
  --observation-particles 96 \
  --episode-steps 32 \
  --action-substeps 4 \
  --output-dir data/taichi_sb3_runs/sac_smoke
```

For a longer GPU run, remove `--cpu` and increase `--timesteps`.

## Main files

```text
scripts/
  collect_apriltag_scene_calibration.py
  reconstruct_voxel_dough_from_deformpath.py
  taichi_viscoelastic_mpm_scene.py
  evaluate_static_topview_match.py
  evaluate_dynamic_topview_match.py
  calibrate_youngs_modulus.py
  visualize_static_topview_benchmark.py
  visualize_dynamic_topview_benchmark.py
  visualize_material_calibration.py
  taichi_depth_udp_observation_sender.py
  taichi_dough_fall_mpm.py
  taichi_lerobot_env.py
  train_taichi_sb3.py

configs/
  episode18_topview_calibration.json
  material_calibration.example.json

docs/
  taichi_mpm_model.tex
  taichi_mpm_model.pdf
```

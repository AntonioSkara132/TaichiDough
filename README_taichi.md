# TaichiDough DeformPath Pipeline

This is the current working pipeline for taking a DeformPath ROS bag through
mocap-frame export, interpolation, chunking, materialization, calibration, and
forward simulation video.

The commands below assume:

- TaichiDough repo: `/home/antonio/diplomski_antonio/diplomski/TaichiDough`
- DeformPath tools/data: `/home/antonio/diplomski_antonio/diplomski/data/deformpath_training`
- ROS 2 Humble
- The bag uses mocap poses and an AprilTag transform for camera-to-mocap calibration

## 1. Source ROS Correctly

Use zsh setup when your shell is zsh:

```zsh
source /opt/ros/humble/setup.zsh
source /home/antonio/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

The ROS workspace overlay is needed for custom mocap message types such as
`motion_capture_tracking_interfaces`.

## 2. Export Bag To Mocap Frame

Use the calibrated exporter, not the old proven-only mocap exporter, when the
data will be used for reconstruction or optimization. The calibrated exporter
writes `scene_calibration_v2.json`.

Example for `episode18_dynamics` with tag 4:

```zsh
cd /home/antonio/diplomski_antonio/diplomski/data/deformpath_training

python3 export_deformpath2_offline.py \
  --bag-dir DeformPathDataset/snimanje_16_6/episode18_dynamics \
  --output-dir episode18_dynamics_mocap_export_calibrated \
  --output-frame mocap \
  --max-points 4096 \
  --seed 0 \
  --camera-tag-parent-frame tag16h5:4 \
  --camera-tag-frame tag4_real \
  --mocap-tag-frame apriltag4 \
  --tag2real-quaternion 0.5 -0.5 -0.5 0.5
```

Check that metric calibration was written:

```zsh
ls episode18_dynamics_mocap_export_calibrated/scene_calibration_v2.json
```

If this file is missing, preprocessing can only run in staging mode and will not
be ready for optimization.

## 3. Create Chunk Ranges

For the episode18 dynamics split, the first and last checkpoints are omitted.
The ranges file should look like this:

```json
{
  "allow_omitted_edges": true,
  "ranges": [
    {"start": 83, "end": 102},
    {"start": 102, "end": 120},
    {"start": 120, "end": 138},
    {"start": 138, "end": 152},
    {"start": 152, "end": 171},
    {"start": 171, "end": 188},
    {"start": 188, "end": 206},
    {"start": 206, "end": 225},
    {"start": 225, "end": 244},
    {"start": 244, "end": 261},
    {"start": 261, "end": 278},
    {"start": 278, "end": 294},
    {"start": 294, "end": 311},
    {"start": 311, "end": 327},
    {"start": 327, "end": 343},
    {"start": 343, "end": 360},
    {"start": 360, "end": 376},
    {"start": 376, "end": 393},
    {"start": 393, "end": 408},
    {"start": 408, "end": 424},
    {"start": 424, "end": 439},
    {"start": 439, "end": 455},
    {"start": 455, "end": 472},
    {"start": 472, "end": 489},
    {"start": 489, "end": 504},
    {"start": 504, "end": 524}
  ]
}
```

Save it as:

```text
/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/episode18_dynamics_ranges.json
```

## 4. Preprocess Export

This copies the calibrated export, interpolates point clouds and paths, chunks
the sequence, reconstructs initial particles for the full episode, and writes a
base smoke config.

```zsh
cd /home/antonio/diplomski_antonio/diplomski/TaichiDough

python3 scripts/preprocess_deformpath_episode.py \
  --input-dir /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/episode18_dynamics_mocap_export_calibrated \
  --output-dir /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_dynamics \
  --ranges-file /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/episode18_dynamics_ranges.json \
  --max-points 4096 \
  --num-particles 24000 \
  --seed 0 \
  --overwrite
```

Expected final status:

```text
Preprocessing status: ready_for_optimization
Smoke config: .../differentiable_mpm_smoke.json
```

## 5. Visualize Interpolated Data

Full preprocessed sequence:

```zsh
cd /home/antonio/diplomski_antonio/diplomski/data/deformpath_training

python3 visualize_deformpath.py \
  preprocessed_dataset/episode18_dynamics \
  --pointclouds-name pointclouds_interpolated.pt \
  --paths-name paths_interpolated.pt \
  --max-points 4096
```

One static preview:

```zsh
python3 visualize_deformpath.py \
  preprocessed_dataset/episode18_dynamics \
  --pointclouds-name pointclouds_interpolated.pt \
  --paths-name paths_interpolated.pt \
  --frame 0 \
  --max-points 4096 \
  --output episode18_dynamics_interpolated_preview.png
```

Single chunk:

```zsh
python3 visualize_deformpath.py \
  preprocessed_dataset/episode18_dynamics/chunks/chunk01 \
  --pointclouds-name pointclouds_interpolated.pt \
  --paths-name paths_interpolated.pt \
  --max-points 4096
```

If Matplotlib fails with NumPy 2.x, use the local NumPy/Matplotlib shim:

```zsh
mkdir -p /tmp/np_mpl_shim
ln -sfn /usr/lib/python3/dist-packages/numpy /tmp/np_mpl_shim/numpy
ln -sfn /usr/lib/python3/dist-packages/numpy-1.21.5.egg-info /tmp/np_mpl_shim/numpy-1.21.5.egg-info
ln -sfn /usr/lib/python3/dist-packages/matplotlib /tmp/np_mpl_shim/matplotlib
ln -sfn /usr/lib/python3/dist-packages/matplotlib-3.5.1.egg-info /tmp/np_mpl_shim/matplotlib-3.5.1.egg-info

MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=/tmp/np_mpl_shim:/home/antonio/.local/lib/python3.10/site-packages \
python3 visualize_deformpath.py \
  preprocessed_dataset/episode18_dynamics \
  --pointclouds-name pointclouds_interpolated.pt \
  --paths-name paths_interpolated.pt \
  --max-points 4096
```

## 6. Materialize Chunks As Separate Episodes

This reconstructs `frame 0` separately inside every chunk. Each chunk is treated
as its own episode, so dynamics do not carry over from previous chunks.

```zsh
cd /home/antonio/diplomski_antonio/diplomski/TaichiDough

python3 scripts/materialize_chunked_deformpath_dataset.py \
  --chunks-dir /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_dynamics/chunks \
  --base-config /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_dynamics/differentiable_mpm_smoke.json \
  --validation-chunk chunk25,chunk26 \
  --overwrite
```

Expected output:

```text
Dataset manifest: .../chunks/dataset.json
```

## 7. Calibrate Material Parameters

Run chunk-minibatch calibration. `--episode-batch-size 4` evaluates four chunk
episodes in parallel, then updates parameters once per batch.

```zsh
cd /home/antonio/diplomski_antonio/diplomski/TaichiDough

python3 -m experiments.differentiable_mpm.calibrate_dataset fit \
  --dataset /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_dynamics/chunks/dataset.json \
  --episode-batch-size 4 \
  --iterations 20 \
  --reference-policy frozen \
  --backend cpu \
  --precision f64 \
  --cpu-threads 1 \
  --output-dir experiments/differentiable_mpm/runs/episode18_dynamics_chunk_minibatch_fit
```

The command logs batch starts and parameter updates. The selected fitted
parameters are written to:

```text
experiments/differentiable_mpm/runs/episode18_dynamics_chunk_minibatch_fit/selected_parameters.json
```

## 8. Forward Simulation Video

Render one chunk with manual parameters before calibration:

```zsh
cd /home/antonio/diplomski_antonio/diplomski/TaichiDough

python3 experiments/differentiable_mpm/forward_video_v2/run.py \
  --dataset /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_dynamics/chunks/dataset.json \
  --episode-id chunk01 \
  --youngs-modulus 130579.320726 \
  --poisson-ratio 0.3 \
  --viscosity 0.0 \
  --plastic-min 0.9 \
  --plastic-max 1.1 \
  --floor-retention 0.4 \
  --tool-friction-coefficient 0.5 \
  --tool-stickiness 0.0 \
  --backend cpu \
  --precision f64 \
  --cpu-threads 1 \
  --camera-zoom 1.0
```

For a faster smoke render, add:

```zsh
--end-frame 8
```

After calibration, render with the selected fitted parameters:

```zsh
python3 experiments/differentiable_mpm/forward_video_v2/run.py \
  --dataset /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_dynamics/chunks/dataset.json \
  --episode-id chunk01 \
  --parameters experiments/differentiable_mpm/runs/episode18_dynamics_chunk_minibatch_fit/selected_parameters.json \
  --backend cpu \
  --precision f64 \
  --cpu-threads 1 \
  --camera-zoom 1.0
```

The video path is printed as:

```text
VIDEO: .../perspective/requested_material_perspective.mp4
```

## 9. Recover Render If Simulation Finished But Rendering Failed

If simulation succeeds but rendering fails because of the NumPy 2.x / Matplotlib
binary mismatch, do not rerun the simulation. Render the saved snapshots with
the shim:

```zsh
cd /home/antonio/diplomski_antonio/diplomski/TaichiDough

mkdir -p /tmp/np_mpl_shim
ln -sfn /usr/lib/python3/dist-packages/numpy /tmp/np_mpl_shim/numpy
ln -sfn /usr/lib/python3/dist-packages/numpy-1.21.5.egg-info /tmp/np_mpl_shim/numpy-1.21.5.egg-info
ln -sfn /usr/lib/python3/dist-packages/matplotlib /tmp/np_mpl_shim/matplotlib
ln -sfn /usr/lib/python3/dist-packages/matplotlib-3.5.1.egg-info /tmp/np_mpl_shim/matplotlib-3.5.1.egg-info

MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=/tmp/np_mpl_shim:/home/antonio/.local/lib/python3.10/site-packages \
python3 experiments/differentiable_mpm/forward_video_v2/recover.py \
  --run-dir experiments/differentiable_mpm/runs/<forward_video_run_dir> \
  --camera-zoom 1.0
```

## Notes

- `export_deformpath2_offline_proven_mocap.py` can export mocap-frame tensors,
  but it does not produce `scene_calibration_v2.json`.
- `--allow-missing-calibration` in preprocessing is only for staging/debugging.
  It will not produce an optimization-ready dataset.
- For chunked dynamics, materialization reconstructs each chunk independently
  from its first interpolated frame.
- Train and validation chunks may come from the same original recording. In this
  pipeline they are treated as independent episodes.

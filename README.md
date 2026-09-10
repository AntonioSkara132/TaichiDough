# TaichiDough

TaichiDough is a small Taichi MLS-MPM prototype for deformable dough simulation.
It includes:

- a two-tool viscoelastic / viscoplastic MPM scene,
- a dough falling scene,
- GUI and video recording support,
- a Gymnasium / LeRobot-style wrapper,
- a Stable-Baselines3 training smoke script.

The main scene includes optional ROS 2 point-cloud publishing and tool-pose input,
synchronized with `taichi_dough_ros` on 2026-09-08. Standalone GUI, UDP control,
and training usage do not require ROS.

## Install

Use Python 3.10+.

```bash
python3 -m pip install -r requirements.txt
```

For NVIDIA CUDA acceleration, install a working NVIDIA driver first. Taichi should print `Starting on arch=cuda` or another GPU backend when it starts.

## Run The Main Scene

```bash
python3 scripts/taichi_viscoelastic_mpm_scene.py \
  --gui \
  --no-save \
  --free-camera \
  --steps 10000 \
  --tool-contact-padding 0.05 \
  --tool-contact-friction 0.5
```

To keep the two demo tools stationary, add `--no-scripted-tools`.

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

## Tool meshes and scene settings

The GUI uses `ur_spathla.stl` and `gen3_spathla.stl` with the RF lab URDF visual
origins and millimetre-to-metre scale. Meshes are loaded from the sourced
`ur_dual_bringup` package when available, otherwise from this repository's
`meshes/` directory. Override them with `--ur-tool-mesh PATH` and
`--kinova-tool-mesh PATH`.

By default, `--tool-collision sdf` rasterizes the same transformed STL geometry
into a local 64³ distance field at startup and uses it for dough contact.
`--tool-contact-padding` sets clearance around the mesh surface. Use
`--tool-collision box` for the former 10 cm box proxy or `--tool-collision none`
to disable tool contacts. The fixed robot-to-tool joints are already included in
`/tool_poses`; only the link's visual origin is applied to each mesh.

Scene defaults match the ROS simulation:

| Setting | Value (metres) |
| --- | --- |
| Initial dough centre | `[0.5, 0.30, 0.5]` |
| Dough ellipsoid radii | `[0.07, 0.026, 0.06]` |
| Floor Y | `0.33` |
| Top-view camera position | `[0.5, 0.75, 0.5]` |

The Gymnasium wrapper shares these geometry settings, including the updated
floor, while retaining its training-specific resolution and material defaults.

## ROS tool poses and dough cloud

From this repository, with the ROS workspace next to it:

```bash
export ROS_DOMAIN_ID=5
source ../ros2_ws/install/setup.bash
source ../ros2_ws/.venv-pytorch/bin/activate
python scripts/taichi_viscoelastic_mpm_scene.py \
  --cpu --gui --no-save --no-publish-dough-center --steps 10000 \
  --ros-tool-poses --ros-tool-poses-topic /tool_poses \
  --ros-pointcloud --ros-pointcloud-view top_dough \
  --ros-pointcloud-topic /taichi_dough/top_dough/points \
  --ros-pointcloud-frame camera_depth_optical_frame \
  --max-pointcloud-points 4096
```

ROS support requires the sourced workspace's `rclpy`, message packages, and
`dual_description/PoseStampedArray`. The two input poses must be ordered UR5e,
Kinova and expressed in the point cloud's frame.

Tool positions use the inverse of the camera projection used for the dough
cloud. For `top_dough`: `x = 0.5 - camera_x`, `y = 0.75 - camera_z`, and
`z = 0.5 - camera_y`. Mesh orientations use the same optical-to-scene rotation.
The interactive viewer camera does not affect this mapping. Matrix and offset
arguments default to `auto`; explicit overrides provide manual calibration.

## Verification

```bash
python -m unittest discover -s test -v
```

The tests check optical projection, both STL/URDF transform chains, standalone
imports, and mesh lookup. The subscriber test runs when ROS messages are available.

## Record Video

```bash
python3 scripts/taichi_viscoelastic_mpm_scene.py \
  --gui \
  --free-camera \
  --record-video \
  --video-path data/taichi_mpm_camera_views/viscoelastic_mpm_gui.mp4 \
  --video-simulation-time \
  --steps 10000 \
  --tool-contact-padding 0.05 \
  --tool-contact-friction 0.5
```

`--video-simulation-time` sets the video FPS so playback duration matches simulated time.

## Dough Falling Scene

```bash
python3 scripts/taichi_dough_fall_mpm.py --gui --no-save --free-camera
```

## RL Wrapper

The wrapper exposes a Gymnasium-compatible environment:

```python
from scripts.taichi_lerobot_env import TaichiViscoelasticMPMEnv

env = TaichiViscoelasticMPMEnv()
obs, info = env.reset()
action = [0, 0, -0.05, 0, 0, 0.05]
obs, reward, terminated, truncated, info = env.step(action)
```

Reward:

```text
-mean_i ||x_i - mean_j(x_j)||^2
```

This rewards compact dough particle distributions.

## Stable-Baselines3 Smoke Training

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

## Contents

```text
scripts/
  taichi_viscoelastic_mpm_scene.py
  taichi_dough_fall_mpm.py
  taichi_lerobot_env.py
  train_taichi_sb3.py

docs/
  taichi_mpm_model.tex
  taichi_mpm_model.pdf

examples/
  starting_scene.png
  starting_scene_sofa_boxes.png
  first_scene_screenshot.png
  episode36_deformpath_paths.png
  deformpath_run.mp4
```

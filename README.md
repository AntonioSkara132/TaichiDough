# TaichiDough

TaichiDough is a small Taichi MLS-MPM prototype for deformable dough simulation.
It includes:

- a two-tool viscoelastic / viscoplastic MPM scene,
- a dough falling scene,
- GUI and video recording support,
- a Gymnasium / LeRobot-style wrapper,
- a Stable-Baselines3 training smoke script.

The ROS2 bridge/control code is intentionally not included in this repository folder.

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

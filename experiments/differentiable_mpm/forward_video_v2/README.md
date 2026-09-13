# Dataset-aware forward video

`forward_video_v2/run.py` replays one episode from a current dataset using saved calibration parameters and renders an opaque perspective MP4. It preserves the selected episode's mass, density, frame-zero particles, calibration, collision meshes, prescribed tool poses, floor, and fixed contact settings.

It performs forward simulation only. An optimizer checkpoint supplies material parameters, not a mid-trajectory particle state: every replay starts from the configured frame-zero reconstruction.

## Active calibration run

The run directory may contain only `optimizer_state.json`; `selected_parameters.json` is not required. `best` uses the lowest-loss accepted candidate saved so far. `current` uses the latest accepted candidate.

```bash
cd /mnt/Data/studenti/antonio_skara/TaichiDough

SIM_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
RENDER_PYTHON="/absolute/path/to/python-with-pyvista"

python3 experiments/differentiable_mpm/forward_video_v2/run.py \
  --dataset experiments/differentiable_mpm/data/ten_episode_shared_alignment_v1/dataset_episode18_only.json \
  --episode-id snimanje_23_10-episode18 \
  --run-dir experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd \
  --checkpoint-choice best \
  --path snimanje_23_10-episode18.episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --simulation-python "$SIM_PYTHON" \
  --render-python "$RENDER_PYTHON" \
  --backend cuda
```

Omit `--end-frame` to replay every processed frame available in the recording. Add `--end-frame 59` for the prepared two-second Episode18 prefix.

The command prints the final `VIDEO:` path. Every invocation creates a new directory under `experiments/differentiable_mpm/runs/`; existing calibration and video runs are not changed.

## Completed calibration

Pass a completed dataset selection directly:

```bash
python3 experiments/differentiable_mpm/forward_video_v2/run.py \
  --dataset experiments/differentiable_mpm/data/ten_episode_shared_alignment_v1/dataset_episode18_only.json \
  --episode-id snimanje_23_10-episode18 \
  --parameters experiments/differentiable_mpm/runs/YOUR_FIT/selected_parameters.json \
  --path snimanje_23_10-episode18.episode=/absolute/path/to/episode18_kugla \
  --simulation-python /absolute/path/to/python-with-taichi \
  --render-python /absolute/path/to/python-with-pyvista \
  --backend cuda
```

When `--run-dir` contains `selected_parameters.json`, the launcher uses that completed selection. Otherwise it reads `optimizer_state.json` and applies `--checkpoint-choice`.

## Explicit parameters

All five material values are required together:

```bash
python3 experiments/differentiable_mpm/forward_video_v2/run.py \
  --dataset DATASET.json --episode-id EPISODE_ID \
  --youngs-modulus 8803.876082216937 \
  --poisson-ratio 0.48886576288903083 \
  --viscosity 28.133366421671834 \
  --plastic-min 0.8647389601015605 \
  --plastic-max 1.0676327831737862 \
  --simulation-python /path/to/simulation/python \
  --render-python /path/to/render/python
```

Fixed floor and tool contact values still come from the dataset. Explicit material arguments do not replace them.

## Preparation check

`--prepare-only` verifies the dataset, parameter source, episode, recording length, interpreter paths, and derived configuration. It does not initialize Taichi, simulate, or render:

```bash
python3 experiments/differentiable_mpm/forward_video_v2/run.py [inputs above] --prepare-only
```

Inspect `launcher_manifest.json` and `forward_config.json` in the printed output directory. They record parameter provenance, current dataset identity, mass, density, fixed settings, Python interpreters, path overrides, and requested endpoint.

## Rendering recovery

If simulation completed but rendering failed, rerender the saved states without rerunning physics:

```bash
python3 experiments/differentiable_mpm/forward_video_v2/recover.py \
  --run-dir experiments/differentiable_mpm/runs/forward_video_EPISODE_TIMESTAMP_ID \
  --render-python /absolute/path/to/python-with-pyvista \
  --camera-zoom 1.0
```

The rendering interpreter needs NumPy, SciPy, scikit-image, PyVista/VTK, and Pillow. `ffmpeg` and `ffprobe` must be available on `PATH`, or passed to `render.py` during manual recovery. The simulation interpreter needs the project's normal Taichi and input dependencies.

## Numerical interpretation

A calibration run made with `--ignore-recompute-mismatch` may have approximate gradients. The launcher records that provenance. Forward replay does not differentiate and does not ignore nonfinite states.

This launcher uses the live experimental implementation and records its inputs. `forward_video_v1` remains the historical fixed-bundle launcher for the 0.25 kg / approximately 2196 kg/m³ Episode18 setup.

# Visualize a saved calibration run

`visualize_run.py` is a reusable, standalone visualization CLI for the differentiable calibration outputs. It reads saved strict-evaluation images; it does not initialize Taichi, rerun simulation, optimize parameters, or change the original run.

## Run it locally

From the repository root:

```bash
python -m experiments.differentiable_mpm.visualize_run \
  --run-dir experiments/differentiable_mpm/runs/episode18-table-aligned-registered-tools_fit_20260912T222338_9ab2d0 \
  --video
```

Use a Python environment with NumPy and Pillow. The locally verified environment is:

```bash
/home/antonio/miniconda3/envs/prancer/bin/python \
  /home/antonio/diplomski_antonio/diplomski/TaichiDough/experiments/differentiable_mpm/visualize_run.py \
  --run-dir /home/antonio/diplomski_antonio/diplomski/TaichiDough/experiments/differentiable_mpm/runs/episode18-table-aligned-registered-tools_fit_20260912T222338_9ab2d0 \
  --video
```

The command prints a newly created output directory's `index.html`, contact sheet, and optional MP4 paths. Open `index.html` with your browser. Some browsers restrict canvas pixel inspection for `file://` images. For full pixel-inspector functionality, serve only the generated output directory on loopback:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory /path/to/generated_visualization
```

Then open `http://127.0.0.1:8000/` in a browser on that computer. The server is optional and is not started by the tool. All generated files remain local; nothing is published or uploaded.

## Options

- `--run-dir PATH`: run directory, or `selected_parameters.json` / `result.json` inside it. A parameter JSON is used to locate its containing run, not to reconstruct missing trajectories.
- `--split validation`: default. Finds `validation_*_strict/evaluation/dynamic_topview_metrics.json`.
- `--split training`: use the saved training report instead.
- `--metrics PATH`: explicitly select one local strict metrics JSON instead of supplying `--run-dir`. Use this if more than one matching report exists; the tool will not silently choose the newest one.
- `--list`: print available reports without writing anything.
- `--output-dir PATH`: choose a new directory outside the original run. Existing directories are refused. By default, a uniquely named sibling directory is created.
- `--video`: also create `depth_comparison.mp4`; requires `ffmpeg` on PATH.
- `--fps 30`: video sampling rate. The video samples the recorded elapsed-time interval uniformly, holding the most recent available observation. The HTML player uses the original timestamp gaps.
- `--full-frame`: disable the common camera-image crop used for comparison stills and MP4. The interactive report always retains the full saved camera image and original pixel coordinates.

For example, inspect inputs before rendering:

```bash
python -m experiments.differentiable_mpm.visualize_run --run-dir /path/to/fit_run --list
```

## Output

- `index.html`: synchronized interactive observed/simulated depth, outlines, residuals, masks, metric plots, numerical tables, and tool-trajectory context.
- `frames/`: per-frame depth/overlay/error/mask PNGs.
- `temporal_contact_sheet.png`: six observation-time samples.
- `comparison_frame_*.png`: first, middle, and final three-panel comparison stills.
- `depth_comparison.mp4`: optional H.264 1280×600 video of observed depth, fitted depth, and signed depth error.
- `visualization_manifest.json`: report data, metric scales, and display-crop description.
- `saved_run_visualization.json`: source hashes, selected-parameter context, input frame count, and video sampling records.
- `paired_metrics.csv`: copied when present in the source evaluation.

The video/still crop is calculated once from the union of all observed and predicted valid pixels, with a margin. The same pixel crop is applied to every frame and every panel. This is only a display crop, not a state translation, per-frame registration, geometry change, or loss recalculation. Depth and residual color ranges are fixed over the entire report. Dark/unsupported pixels are unavailable observations, not zero error.

The interactive page shows stored tool trajectories, **not 3D tool meshes**. The comparison video likewise displays saved optical depth. A separate 3D/tool renderer is required for spatula meshes, table geometry and particle side views.

## Copied runs and `/mnt`

The tool discovers strict reports using their local directory names and resolves each report's `arrays/` entries relative to that report. It deliberately ignores stale absolute `/mnt` or `/home` paths in the outer evaluation JSONs and replay camera metadata. It needs no raw recording, calibration file, original mesh, or production simulator merely to render these saved comparisons.

Transfer these two files together:

```text
experiments/differentiable_mpm/visualize_run.py
experiments/differentiable_mpm/visualization_report.py
```

`visualization_report.py` is an experimental copy of the previously existing depth-report implementation, with run/scoring annotations. The production script is not imported and does not need to exist in a standalone calibration bundle. An optional transfer archive is `bundles/saved_run_visualizer_v1.zip`; it contains these files and this guide at their repository-relative paths.

On `/mnt`, from your standalone calibration directory:

```bash
python -m experiments.differentiable_mpm.visualize_run \
  --run-dir experiments/differentiable_mpm/runs/YOUR_FIT_RUN_DIRECTORY \
  --video
```

A moved run must include the chosen `*_strict/evaluation/dynamic_topview_metrics.json` and its complete relative `arrays/` directory. Copy `selected_parameters.json` as well for material-selection labels and fitted-parameter context. A parameter JSON by itself cannot supply an animation.

## Interpretation

Selecting a validation **report** does not necessarily restrict the displayed frames to the held-out scoring window. The example Episode18 validation export contains all 98 frames 0–97, including initialization and training context. Labels use `selected_parameters.json` to distinguish initialization, material-fitting frames, and frames outside material selection. They do not claim independence from scene/tool calibration preprocessing.

The gray frozen baseline in metric plots is the **frozen initial geometry**, not an unfitted moving-simulation rollout. Saved parameters and metrics are displayed as recorded; the tool does not replace them with the original bundle settings. For example, this run's stored tool friction differs from the initial bundle, and rendering must not silently restore that initial setting.

## Verification

Seven focused unit tests cover report discovery, ambiguous matches, relocation through relative arrays, nonfinite valid depths, rejected path escapes, input preservation/output refusal, scoring annotations, and video timeline endpoints. No simulation tests are run by this tool. The supplied Episode18 run was rendered directly from all 98 saved observations and its MP4 inspected with ffprobe.

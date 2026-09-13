# Parameterized Episode 18 video command

From the TaichiDough repository:

```bash
python3 experiments/differentiable_mpm/forward_video_v1/run.py \
  --youngs-modulus 100000 \
  --viscosity 10 \
  --plastic-min 0.09 \
  --plastic-max 1.2 \
  --backend vulkan
```

Young's modulus is in Pa, viscosity in Pa·s, and plastic limits are stretch ratios. `0.09` and `0.9` are different inputs; the launcher uses exactly what you supply. Optional values include `--poisson-ratio`, `--tool-friction`, and `--floor-retention`. Run `--help` for all options.

The command performs the full 388-frame Episode 18 forward simulation, followed by the opaque perspective movie with both registered tools and table. It prints a final `VIDEO:` path. Each invocation creates a new `experiments/differentiable_mpm/runs/episode18_forward_video_TIMESTAMP_ID/` directory. Existing outputs are never reused or overwritten. No calibration is performed.

The defaults retain the actual fitted-run Poisson ratio 0.4047302679702308, tool friction 0.3, and floor retention 0.4. Numerical validity checks remain enabled. Forward configuration optimizer bounds are expanded only when needed to include an explicitly requested value; this does not change the physical-validity checks or clamp your input. Invalid physical parameters are rejected by the isolated simulator.

## Interpreter and input locations

On the current computer the defaults are system Python for Vulkan/Taichi simulation and `~/miniconda3/envs/prancer/bin/python` for rendering. They can be selected explicitly:

```bash
python3 experiments/differentiable_mpm/forward_video_v1/run.py \
  --episode /absolute/path/to/episode18_kugla \
  --simulation-python /path/to/python-with-taichi \
  --render-python /path/to/python-with-pyvista \
  --youngs-modulus 100000 --viscosity 10 \
  --plastic-min 0.09 --plastic-max 1.2 --backend vulkan
```

The renderer needs NumPy, SciPy, scikit-image, PyVista/VTK and Pillow, working offscreen graphics, plus `ffmpeg` and `ffprobe` on PATH. The launcher now checks rendering imports and executable availability before starting simulation; that check does not establish offscreen graphics or mounted-file access. The system Python plotting installation on the current computer has a NumPy/Matplotlib binary mismatch; use the existing prancer environment for rendering. No dependency installation is done by this launcher.

The episode override relocates the verified Episode 18 recording; it does not authorize a different episode with different hashes. The bundled geometry and initial particles are specific to Episode 18.

## Numerical isolation

Each invocation verifies and extracts the original archive `../bundles/episode18_registered_tools_v1.tar.gz` with SHA256 `740658218a57a15f62ac2184c45d4a286ad8e31524b6687e48e6bf124f2621d0`. All numerical imports are from this new extracted copy, not the actively edited main experiment. The original archive and source files are unchanged. `import_identity.json` records solver/state/spectral paths and hashes before runtime initialization. This command intentionally uses the original corrected-v1/coulomb-v1 implementation; it does not include later contact-model changes.

Raw particle states are retained even if rendering fails. A numerical validity stop produces a clearly labeled partial movie when a valid completion record exists; the command exits nonzero. It does not relax physics to finish the episode.

The displayed boundary is a rendering-only density approximation using 1.5 mm voxels, 1.2 mm Gaussian width, and level 0.20. It does not fill to the floor or move simulated particles. Sparse particles may be absent from the density boundary; raw snapshots remain intact. Material labels are read from the actual generated simulation configuration.

## Recover a failed visualization without rerunning simulation

Transfer this complete `forward_video_v1/` directory to the remote experimental directory, including the new `render_support.py` and `recover.py`. Existing saved runs, render scripts, images, and numerical snapshots do not need editing. New launcher runs also copy the updated rendering helper.

For the saved run whose density grid exceeded 30 million voxels, use a Python environment with the rendering dependencies:

```bash
RUN=/mnt/Data/studenti/antonio_skara/TaichiDough_registered_tools_v1/experiments/differentiable_mpm/runs/episode18_forward_video_20260913T134409_0fc3cf

python experiments/differentiable_mpm/forward_video_v1/recover.py \
  --run-dir "$RUN" \
  --ffmpeg /usr/bin/ffmpeg \
  --ffprobe /usr/bin/ffprobe
```

The command renders saved states into a fresh `perspective_recovery_TIMESTAMP_ID/` directory inside the run and prints `VIDEO:` on success. It never imports or executes the numerical simulator. Paths recorded in the saved provenance must still exist and match their hashes; this is not a general run-relocation tool.

The density calculation now uses disjoint 64-cell tiles with Gaussian halos rather than one grid spanning the entire particle bounding box. It retains 1.5 mm voxels, the same Gaussian width and density threshold, and contributions from every particle. Each default tile allocates at most 357,911 grid samples (about 1.37 MiB for its float32 density array), plus filtering and mesh memory. The final triangle array still grows with mesh complexity and has an explicit two-million-triangle limit. Sparse particles below the density threshold can remain absent from the displayed boundary, as before; they are not removed from the saved states. No adaptive coarsening, percentile clipping, or floor filling is performed.

### Bring the camera view closer

The default camera fits the full trajectory bounds, including distant particles and all tool positions. That can make the main scene look small. Use `--camera-zoom 1.8` to enlarge the view without rerunning physics:

```bash
python experiments/differentiable_mpm/forward_video_v1/recover.py \
  --run-dir "$RUN" \
  --camera-zoom 1.8 \
  --ffmpeg /usr/bin/ffmpeg \
  --ffprobe /usr/bin/ffprobe
```

`1` retains full-scene framing; larger values give a closer optical view, and smaller values zoom out. The allowed range is 0.25–4. The camera stays fixed throughout the movie. Zoom does not move or delete particles, but particles or tools can leave the image; the movie footer and manifest explicitly identify the zoomed view. `--camera-zoom` is also available on `run.py` for future simulations. It cannot be combined with `--encode-only`: changing the 3D view requires rendering the saved states again, not rerunning the simulation.

### Encode images that already exist

If rendering completed and only MP4 encoding failed, use the saved timing file and images instead:

```bash
python experiments/differentiable_mpm/forward_video_v1/recover.py \
  --run-dir "$RUN" \
  --encode-only \
  --frames-dir "$RUN/perspective" \
  --ffmpeg /usr/bin/ffmpeg \
  --ffprobe /usr/bin/ffprobe
```

`--frames-dir` means the directory containing `frame_timing.txt`, not its nested `frames/` directory. Omit it when that directory is `RUN/perspective`. Encoding-only recovery needs Python and FFmpeg/ffprobe, not NumPy, scikit-image, or an offscreen graphics environment. It creates a fresh output directory and does not overwrite the original video, timing file, images, or logs.

The `/usr/bin` examples require those executables to be installed already. Other installations can be selected with `--ffmpeg` and `--ffprobe`. New launcher rendering also accepts `FFMPEG` and `FFPROBE` environment variables. When PATH selects Snap and a non-Snap `/usr/bin` executable already exists, the default resolver prefers that existing executable. Nothing installs packages or changes Snap/filesystem permissions. A failure reports the actual FFmpeg log text, not only an exit code. A successful encode is checked with ffprobe.

Missing `scikit-image` or PyVista is an environment issue: select an interpreter that has the documented dependencies and rerun recovery, not the simulation. Working offscreen graphics is still required for full rendering.

## Verification performed

Recovery checks on 2026-09-13:

- Eleven tests in `tests/test_render_recovery.py` passed, including zoom input validation and rejection of zoom changes in encoding-only mode: tiled/dense geometry equivalence across positive and negative tile boundaries, distant clusters whose dense grid exceeds 30 million voxels, input immutability, explicit empty-boundary and triangle-limit errors, FFmpeg error details, overwrite refusal, missing executable handling, and actual encoding-only CLI output checked by ffprobe.
- The complete local saved Episode18 run was rendered successfully: 131 sampled states, H.264 1200×800, 392 video frames, 13.066667 seconds. No physics was rerun. Its manifest confirms unchanged simulation inputs and all original snapshot files. The final still was visually inspected.
- Verified output: `runs/episode18_E100k_eta10_plastic009_20260912T233343/perspective_recovery_20260913T154921_d32a43/requested_material_perspective.mp4`, relative to the experiment.
- A second complete saved-state render with `--camera-zoom 1.8` passed and its initial still was inspected; part of the UR5e tool leaves this closer view, as the movie label warns. Video: `runs/episode18_E100k_eta10_plastic009_20260912T233343/perspective_recovery_20260913T161037_c3f1d6/requested_material_perspective.mp4`. It also contains 392 frames and preserves original input/snapshot hashes.
- Updated launcher extraction, helper copying, and zoom forwarding passed `--prepare-only`; no new simulation was started. Remote `/mnt` recovery was not executed here, and remote Snap permissions remain unchanged.

The underlying forward and rendering drivers completed the full requested-material run with a verified movie. For this reusable launcher, command parsing, verified extraction, changed-parameter configuration preparation, and Python syntax were checked using `--prepare-only`; no second simulation was launched merely to test the command. Preparation-only output is not a completed simulation. Use the ordinary command above, without `--prepare-only`, to run simulation and rendering.

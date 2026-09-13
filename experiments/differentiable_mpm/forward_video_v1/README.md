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

The renderer needs NumPy, SciPy, scikit-image, PyVista/VTK and Pillow, working offscreen graphics, plus `ffmpeg` and `ffprobe` on PATH. The system Python plotting installation on the current computer has a NumPy/Matplotlib binary mismatch; use the existing prancer environment for rendering. No dependency installation is done by this launcher.

The episode override relocates the verified Episode 18 recording; it does not authorize a different episode with different hashes. The bundled geometry and initial particles are specific to Episode 18.

## Numerical isolation

Each invocation verifies and extracts the original archive `../bundles/episode18_registered_tools_v1.tar.gz` with SHA256 `740658218a57a15f62ac2184c45d4a286ad8e31524b6687e48e6bf124f2621d0`. All numerical imports are from this new extracted copy, not the actively edited main experiment. The original archive and source files are unchanged. `import_identity.json` records solver/state/spectral paths and hashes before runtime initialization. This command intentionally uses the original corrected-v1/coulomb-v1 implementation; it does not include later contact-model changes.

Raw particle states are retained even if rendering fails. A numerical validity stop produces a clearly labeled partial movie when a valid completion record exists; the command exits nonzero. It does not relax physics to finish the episode.

The displayed boundary is a rendering-only density approximation using 1.5 mm voxels, 1.2 mm Gaussian width, and level 0.20. It does not fill to the floor or move simulated particles. Sparse particles may be absent from the density boundary; raw snapshots remain intact. Material labels are read from the actual generated simulation configuration.

## Verification performed

The underlying forward and rendering drivers completed the full requested-material run with a verified movie. For this reusable launcher, command parsing, verified extraction, changed-parameter configuration preparation, and Python syntax were checked using `--prepare-only`; no second simulation was launched merely to test the command. Preparation-only output is not a completed simulation. Use the ordinary command above, without `--prepare-only`, to run simulation and rendering.

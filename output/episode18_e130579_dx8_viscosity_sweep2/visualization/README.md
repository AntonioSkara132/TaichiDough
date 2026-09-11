# Episode 18 viscosity visualizations

Rendered on 11 September 2026 from the five completed cases in the parent directory. No material simulations were rerun, and no original result, metadata or particle files were modified.

## Open

- `index.html` — local viewer with the comparison, result table and individual video links.
- `viscosity_comparison_isometric.mp4` — synchronized five-case comparison and score panel, 1920×960.
- `viscosity_comparison_preview.png` — final-frame comparison.
- `viscosity_0_isometric.mp4`, `viscosity_1_isometric.mp4`, `viscosity_2p5_isometric.mp4`, `viscosity_5_isometric.mp4`, `viscosity_10_isometric.mp4` — individual 960×720 videos.

All six videos contain 61 frames, use H.264/yuv420p, and last 8.056 seconds at 7.572 fps. The physical replay spans approximately 2.001 seconds; playback matches the earlier approximately four-times-slower isometric video. Source frame 0 is the initial state; the scores use source frames 1–60.

## Result

| Effective viscosity, nominal Pa·s | Training aggregate loss |
|---:|---:|
| 0 | 0.3611063537133981 |
| 1 | 0.36109166663665204 |
| 2.5 | 0.36103418347008154 |
| 5 | 0.36101766252581863 |
| 10 | 0.3610061713928395 |

Viscosity 10 is the lowest-loss tested value, at the upper boundary of this sweep. Relative improvement over viscosity 0 is only **0.027743%**. The results are visually very close and do not establish a unique physical viscosity. The evaluation is training-only; no held-out results were added.

E remains 130579.320726 Pa, Poisson's ratio 0.3, grid 48, time step 0.0002 s, 24,000 particles, SDF resolution 64, padding 0.0026041667 m, and tool-contact retention 0.2. Plastic projection is disabled.

This directory now contains completed results, unlike the failed-startup snapshot inspected during the earlier literature report. That report was not edited by this visualization task.

## Rendering and checks

- Reuses `scripts/render_replay_depth_tools.py` camera construction, STL loading, particle rasterization, triangle rasterization and depth shading.
- Same isometric preset as the existing renderer: position (0.72, 0.42, 1.34), look-at (0.23, 0.04, 0.73), vertical field of view 52 degrees.
- Orange dough, blue UR tool, red Kinova tool. The image is rendered geometry, not a camera photograph or a force/friction map.
- A common depth-shading range, approximately 0.812263–0.966414 m from the virtual camera, is computed across all five cases and all frames. It is not independently rescaled per case.
- Identical recorded tool poses and timestamps are verified before rendering. Tools are rasterized once per time step and composited with each case's dough.
- The initial viscosity-0 and final viscosity-10 images match the existing renderer exactly before annotations, using the shared shading range.
- FFprobe verified all six video frame counts, dimensions, codecs and pixel format. The comparison preview was visually inspected.
- Original copied-machine paths are only rebased in derived metadata under `resolved_replays/`; no particle copies are necessary.
- Input/renderer hashes, camera, shading range, equivalence checks and video metadata are in `visualization_manifest.json`.

To reproduce without overwriting these videos, use a new output directory:

```sh
python3 render_visualizations.py --output-dir ../visualization_new
```

The output script imports the existing renderer through the repository's `scripts` path. Project-language-server import/type-stub diagnostics for that dynamic import or the Pillow compatibility constant do not describe a rendering failure: the executed run completed successfully. No repository configuration or simulator dependencies were changed.

# Episode 18 article stills

These figures render saved MPM particle states and the corresponding recorded tool poses. Rendering does not run the simulator, optimization, or calibration, and does not change the saved geometry.

## Publication files

- **`figure1_t0p5004.png` and `figure1_t0p5004.pdf`**: the recommended, tightly framed, unlabelled figure at **0.5004 s**.
- **`figure1_t1p2008.png` and `figure1_t1p2008.pdf`**: a tightly framed alternative at **1.2008 s**, with more visible deformation.
- `figure1_preview.png`: a labelled comparison for viewing, not the publication file.

The publication figures are **2822 × 1701 pixels**, with no resizing of the original scene pixels. Their PDFs embed the RGB images losslessly at 7.1 inches wide, approximately 397 pixels per inch. PNG resolution metadata is 500 dpi; use the desired physical width when placing the figure in the article. The PDF scene is raster, not vector geometry.

## Full renders and provenance

- `episode18_t0p5004_clean.png` and `.pdf`: full, unlabelled view at **0.5004 s**, replay source frame 15 (original source frame 17), substep 2502. This is the requested approximately 0.5-second image.
- `episode18_t0p5004.png` and `.pdf`: the same view with a time label, color legend, and floor-grid spacing.
- `episode18_t1p2008_clean.png` and `.pdf`: a later alternative at **1.2008 s**, replay source frame 36 (original source frame 38), substep 6004, with more visible deformation.
- `episode18_t1p2008.png` and `.pdf`: the later alternative with labels.
- `preview.png`: reduced-resolution comparison for inspection, not the publication file.
- `figure_manifest.json`: source/output hashes, frames, particle bounds, tool poses, parameters, and camera settings.
- `render_article_stills.py`: the rendering script. It refuses to replace an existing figure.
- `compose_article_figure.py`: applies the same integer-pixel crop to both times, removing empty white margins without resampling or changing scene pixels.
- `composition_manifest.json`: crop rectangle, extraction settings, dimensions, and source/output hashes for the publication files.
- `verification.json`: 28 passing source/output hash comparisons; exact crop-pixel equality; exact equality between each publication PNG and the decoded RGB image embedded in its PDF; and successful independent `pdfinfo` parsing of both publication PDFs.

The preserved full renders are 3600 × 2300 pixels. Their PNG resolution metadata is 500 dpi; a 7.1-inch-wide placement gives approximately 507 pixels per inch. Their PDFs also embed the RGB images with lossless compression. Both times were visually inspected.

## Suggested captions

**Requested-time image:**

> Qualitative MPM replay of the dough-manipulation scene at t = 0.5004 s, showing the simulated dough (ochre) and two registered rigid tool meshes (blue and red). The dough boundary is extracted from 24,000 saved material-point positions; tool meshes use the corresponding recorded poses. The floor grid has 5 cm spacing.

**Later alternative:**

> Qualitative MPM replay of dough manipulation at t = 1.2008 s, showing the deformed dough (ochre) and registered rigid tool meshes (blue and red). The dough boundary is extracted from 24,000 saved material-point positions; tool meshes use the corresponding recorded poses. The floor grid has 5 cm spacing.

The captions describe a simulation visualization, not a camera observation or a validated material-identification result. The later state is provided as an alternative rather than labelled as the requested 0.5-second state.

## Rendering details

- Source run: `experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd_best_preview_v2_fixed/`.
- Source state: `simulation/simulation_result.json` with `simulation/snapshots/particles_000015.npy` and `particles_000036.npy`.
- Collision meshes and registration transforms: the source run's `simulation/prepared_inputs.json`, in recorded stream order.
- Tool transform: STL vertices × recorded mesh scale; apply recorded visual RPY rotation and visual-origin translation; then apply the saved xyzw quaternion and tool translation.
- Density boundary: existing `forward_video_v2/render_support.py::density_boundary`, voxel spacing 1.5 mm, Gaussian sigma 0.8 voxel, density isovalue 0.20, tiled marching cubes with 64-cell cores. Every saved particle contributes to the density field; the threshold may exclude sparse particles from the displayed boundary.
- Smooth shading changes lighting normals, not saved particle positions. Density extraction may extend the displayed boundary below the physical floor; the floor occludes that part without moving particles.
- Both times use the same orthographic camera, scale, lighting, and 5 cm display grid. Colors identify objects and do not encode stress, loss, or material parameters.

## Interpretation

The completed forward replay uses best-so-far parameters from an interrupted fit: E ≈ 8.803 kPa, Poisson ratio ≈ 0.4871, viscosity ≈ 30.37 Pa·s, and principal-stretch clamp limits ≈ [0.8626, 1.0671]. These figures are qualitative illustrations. They do not establish final parameter recovery, replay-consistent gradients, or held-out predictive accuracy.

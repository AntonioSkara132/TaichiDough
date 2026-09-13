# Drafting evidence and remaining work

## Status

This example contains an abstract, introduction, substantially expanded related work, and method. It describes the implemented experimental differentiable pipeline, while treating independent real-data prediction as the research hypothesis still to be tested. It contains no invented real calibration improvements, runtimes, physical-parameter recovery rates, or accepted-paper claims.

All manuscript files are new and confined to `experiments/differentiable_mpm/paper_draft_v1/`. Production simulation, frozen references, recordings, user meshes, previous papers, and the supplied PDF are unchanged.

## Layout source

Reference document: `/home/antonio/Downloads/askara_icra2026.pdf`.

The title and displayed author names/affiliation are retained. Layout uses `ieeeconf`, letter paper, 10-point conference text, two columns, and Latin Modern (the scalable Computer Modern family). The provided PDF also uses Computer Modern. The original empty introductory figure placeholder is omitted. No conference logo, acceptance notice, or proceedings record is added.

Class source: https://ras.papercept.net/conferences/support/files/ieeeconf.zip

The downloaded class is unmodified. The build script disables TeX shell escape. This generic official robotics template is a formatting basis, not a guarantee of compliance with a particular submission-year page limit or anonymity rule.

## Literature basis

Primary local survey:

`output/reports/taichidough_identification_survey/REPORT.md`

Bibliography metadata:

`output/reports/taichidough_identification_survey/reference_catalog.json`

The draft uses 25 references from that catalog, not the older project bibliography's incorrect Fabbri first name. Fabbri is Angelo Fabbri. Final journal/proceedings metadata are used where supplied by the checked catalog. Broad bibliographic descriptions for foundational and learned-model works are kept within the scope of the survey's evidence, without invented quantitative claims.

A focused primary-text verification independently checked the important comparisons against the local evidence copies:

| Reference | Inspected text and important qualification |
|---|---|
| DPSI | `evidence/dpsi_full.txt`, arXiv v3, February 2025. Final IJRR metadata from the catalog, not the preprint's placeholder journal header. One camera at six endpoint viewpoints; three unseen motion types with two starting configurations each. |
| EMPM | `evidence/empm_full.txt`, arXiv v1, January 2026. Final RA-L metadata from the catalog. Experimental homogeneous E and nu fitting; three D455 cameras; two arms and bread dough. The online example uses ongoing correction. |
| DiffCal | `evidence/diffcal_full.txt`, early-online 2022 text, final 2023 issue citation. Known geometry; single L515 camera; dynamic example estimates E, damping beta and density, with nu fixed. Beta is not the present eta. |
| DiffCloud | `evidence/diffcloud_full.txt`, later 2025 revision of the 2022 IROS paper. Two cameras; selected-frame examples. Its directed distance runs simulated-to-observed, opposite to this implementation's observed-to-visible-prediction term. |
| Hahn et al. | `evidence/hahn2019_full.txt`. Marker motion capture and moving clamps, not markerless depth. Elasticity and power-law viscosity. |
| PAC-NeRF | `evidence/pacnerf_full.txt`. Synthetic multi-law coverage; real falling-ball example uses RGB from four cameras, not their depth measurements. |
| GIC | `evidence/gic_full.txt`. Real evidence includes future states and grasp examples; do not reduce it to rendering alone or equate it with independent dough rheometry. |
| PhysTwin | `evidence/phystwin_full.txt`. Three-camera interaction data; unseen-interaction testing registers the fitted model to each new initial state. |

No fresh web search or external publication was needed for these comparisons. Survey commentary about the earlier code state is not copied as the current method.

## Method verification

The current implementation was checked read-only by a separate technical pass. Important source locations, relative to `experiments/differentiable_mpm/`:

- `solver.py:254–347,470–508`: stretch projection before stress; corrected metric-offset P2G/G2P factors; tool/floor rules.
- `renderer.py:131–203`: compact splats, soft depth and normalized mean-footprint coverage.
- `loss.py:134–175,194–269`: depth mask, missing-prediction penalty, directed visible-point term, normalized component weights.
- `checkpoint.py:88,290–320`: observation averaging and full-state checkpoint adjoints.
- `spectral.py:1–18,44–88`: composite spectral-map adjoints and threshold conventions.
- `parameters.py:103–160`: scaled physical coordinates and gradient pullback.
- `optimize.py:239–316`: projected Adam, validity checks, backtracking, rollback.
- `multi_episode.py:41–49,150–158`: weighted episode means and gradients.
- `configs/episode18_table_aligned_registered_tools.json`: actual initial settings, bounds, observation resolution and scored windows.
- `TABLE_ALIGNMENT.md`, `REGISTERED_TOOLS.md`: reconstruction, geometry validation, contact and density qualifications.

The method's tau is the transfer stress-like quantity. Its elastic part is P_e F^T, not first Piola P_e. The viscous term is a simplified additive rate term. Principal-stretch projection is not von Mises/J2 and does not include active Jp hardening in this configuration.

The renderer's coverage is a normalized mean footprint, not summed opacity. The depth mask excludes no pixel merely because its current prediction disappeared. Configured weights (1, 0.5, 0.5) normalize to (0.5, 0.25, 0.25). Each episode is averaged over frames once. Nearest-neighbor and visibility choices are fixed in the local derivative. Finite replay-mismatch tolerance gives approximate gradients, not exact CUDA adjoints.

## Evaluation qualifications

1. **No finished real-data claim.** Small synthetic derivative checks do not establish full corrected CUDA calibration, joint real-episode identification, or real held-out action prediction.
2. **Geometry affects evaluation independence.** Current frames 61–97 are excluded from the material objective, but table estimation and tool registration use data across Episode18. This is held-out material scoring conditional on that geometry, not an independent end-to-end test.
3. **Prospective test rule.** Fix global scene/tool calibration and material parameters from separate calibration/training data. Initialize each held-out episode from its first dough observation only. Recorded tool motion is an allowed input; future dough observations may only be used for evaluation.
4. **Mass and volume.** The retained 250 g mass and 113.832 mL reconstruction imply 2196 kg/m³. Check the measurement and the hidden-volume assumption rather than choosing a more appealing density without evidence.
5. **Contact validation.** An earlier full manipulation replay using original tool transforms ended with the entire simulated dough above the table. Tools-disabled settling did not reproduce the gap. The registered-tool input bundle passed preparation checks, but that does not prove it resolves the gap.
6. **Tool uncertainty.** Candidate registrations reduce measured residuals but partial views admit alternative transforms. Fixing these candidates makes material estimates conditional on them.
7. **Model limits.** The non-adhesive tool rule does not model tensile adhesion or a full static-friction pressure solve; the floor still uses velocity retention. The renderer does not explicitly account for tool occlusion. A short residual deformation is not evidence of uniquely identified yield plasticity.
8. **No novelty by component list.** MPM, AD, depth-based identification, bread dough, and bimanual interaction have prior art. Claims should follow measured comparisons, especially time-resolved versus endpoint data and fixed-parameter prediction on independent motions.

## Before a submission

- Complete geometry/contact and mass-volume checks before interpreting fitted values.
- Define independent episodes, specimen preparation and recovery periods.
- Run initial-versus-calibrated, endpoint-versus-time-series, restricted-versus-joint and single-versus-multi-episode comparisons.
- Include multiple initializations and numerical-resolution sensitivity.
- Measure prediction using interpretable geometric quantities as well as the training objective.
- Add real figures/results and revise the abstract only when supported by those results.
- Have every listed author review names, affiliation, claims, bibliography and intended submission status.

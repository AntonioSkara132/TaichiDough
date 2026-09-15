# Local article revision summary

## Files

- Preserved Overleaf snapshot: `source_original/`
- Revised LaTeX project: `source_revised/`
- Reviewer-comment report: `OVERLEAF_COMMENTS.md`
- Structured review data: `overleaf_review_items.json`
- Original downloaded PDF: `download/askara_icra2026.pdf`
- Revised compiled PDF: `askara_icra2026_revised.pdf`
- Revised source archive: `askara_icra2026_revised_source.zip`

The live Overleaf project was not edited.

## Revision constraints

- Preserve the existing title, main section order, experiment, and figures.
- Target an ICRA-style six-page paper.
- Present the work conservatively as an implemented single-view fitting pipeline.
- Keep the five-parameter `E`, `ν`, `η`, `σ_min`, `σ_max` run as a preliminary result, not final material calibration.
- Do not discuss whether a program or operator stopped the run.
- Retain numerical claims only when traceable to stored run artifacts.

## Main changes

- Rewrote unclear Abstract and Introduction sentences without changing the paper's subject.
- Completed the contribution list with a third experimental-analysis contribution.
- Broadened and synthesized Related Work while retaining its four existing themes.
- Added a short Method overview and renamed the setup subsection to `Experimental Setup and Data Collection`.
- Corrected the data-collection caption: OptiTrack markers are mounted on the spatulas, not the dough.
- Distinguished recorded tool poses from pose-derived translational and angular velocities.
- Defined the camera operator, grid spacing, SVD factors, singular values, principal-stretch limits, tool center, and objective masks.
- Corrected the trial-deformation/stretch-projection description and used the Kirchhoff stress symbol in particle-to-grid transfer.
- Added the explicit table-aligned scene transform used by both reported runs.
- Corrected density reporting: the preliminary history run uses `1200 kg/m^3`; the separate matched-sequence run uses `2196.219 kg/m^3`.
- Described missing-depth support, the normalized missing-depth residual of `4.0`, and masking of unobserved pixels.
- Qualified checkpoint-replay gradients as approximate optimization evidence.
- Kept the five-parameter result and described its late-stage objective plateau from the saved accepted-state history.
- Explicitly separated the preliminary parameter-history run from the run shown in the matched-sequence figure.
- Restored verified strict visible-geometry scores and compared them only with frozen initial geometry.
- Replaced unsupported state-of-the-art and unique-calibration implications with a conservative system claim.
- Completed the force/torque and single-view limitations discussion.
- Rewrote the Conclusion so every result is supported in the active Results section.
- Changed the plot title from `Interrupted` to `Preliminary` without altering plotted data.
- Changed the pipeline title from `Completed identification pipeline` to `TaichiDough fitting pipeline`.
- Removed the internal title from `matched_sequence.pdf` without changing its selected frames, source data, labels, crop, or depth color scale.
- Removed unused or conflicting LaTeX packages, restored IEEE Times-compatible body fonts, hid hyperlink boxes, fixed the author block and unmatched brace, and corrected float ordering.
- Removed the duplicate `arriola2020modeling` BibTeX entry, added the missing PhysTwin entry, corrected MLS-MPM/APIC citations, and completed verified bibliography metadata.

## Build verification

- Compiler: `pdflatex` with `bibtex`, followed by two final `pdflatex` passes.
- Output: 6 US-letter pages.
- All citations and cross-references resolve.
- BibTeX reports no warnings or duplicate entries.
- LaTeX reports no errors, undefined references, or overfull boxes.
- Body text uses embedded Nimbus Roman/Times-compatible Type 1 fonts.
- The result figures appear before the Conclusion and References.
- The revised source and compiled PDF contain no reference to interruption.

# Document review

Review date: 11 September 2026.

## Content checks

- Forty bibliography entries are cited, including three official publication-guidance pages. This is not a claim that forty primary experiments or forty complete papers were read.
- The bibliography build reports no unresolved citation identifiers.
- Primary-source access and version qualifications are included in the report, the reading catalogue, and the topic evidence notes.
- Proposed experiments and hypotheses are explicitly distinguished from completed TaichiDough results.
- The selected/default/frozen-state table preserves the recorded training/held-out distinction. The PDF text contains 0.361059, 0.828075, 0.835860 and 0.799745 as expected.
- Completed local retention sweeps are distinguished from older failed sweeps. Viscosity startup failures are not interpreted as material failures.
- The affine-transfer test is explicitly labelled as an algebraic NumPy calculation, not a full Taichi or real-dough experiment.

## PDF checks

- Thirty A4 pages; linked contents and references; selectable text.
- A ten-page preview montage was visually inspected, including the contents, executive assessment, methodology, comparative tables, evidence appendix and bibliography. Pages 16–18 were also inspected at full page resolution for the actual constitutive equations, transfer derivation, contact interpretation and corrected score table.
- The final build has no unresolved citations, equation-conversion failures or overfull-box warnings. The document-build and PDF-quality JSON files preserve the automated results.
- Four word bounding boxes extend slightly past the nominal right text margin. They end in quotation marks or an em dash and are consistent with optical punctuation protrusion; the greatest excess is about 3.3 points. No body/table clipping was observed.
- The TeX log retains a font-language feature warning and underfull-box warnings. The inspected text is legible and no missing-glyph warning was reported. These are documented rather than described as a warning-free build.

## HTML and publication

- The HTML is generated from the same Markdown and bibliography. Equations use native MathML, with no externally loaded script or math service.
- A standalone version includes charset, viewport and document structure; the Artifact version contains the page content without the host-provided outer structure.
- The stylesheet defines light/dark palettes, responsive gutters, independently scrollable tables/equations and navigation. No browser-render test was performed, so responsive behaviour is an authored design rather than a reported browser measurement.
- Artifact publication was attempted but rejected because the session uses ANTHROPIC_AUTH_TOKEN authentication rather than a Claude account login. No hosted page or URL was created. The standalone local HTML is the delivered companion.

## Boundaries of the verification

No new physical experiment, material fit, full simulator test suite or independent reproduction of stored TaichiDough dynamics was performed. No simulator/sweep source was modified. The PDF checks verify the document, not the physical validity of the current implementation.

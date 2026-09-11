# Dough Parameter Identification

A fresh critical literature survey and technical assessment of TaichiDough's proposed calibration of elastic, viscous, and plastic response from depth recordings and known tool trajectories.

Review date: **11 September 2026**. Audited repository revision: `cb76169436d334c127f165ce5c3f779100bb82d4`.

## Read these files

- **`Dough_Parameter_Identification.pdf`** — the typeset report, with linked contents and references.
- **`Dough_Parameter_Identification.html`** — the standalone read-only companion; open it directly in a browser.
- **`dough-identification.html`** — the same article prepared for Artifact publication. The upload was blocked because this session uses API-token authentication, not a Claude account login. **No hosted URL was created.**
- **`REPORT.md`** — the editable source shared by the PDF and web report.
- **`reading_catalog.csv`** — 40 references/guidance sources, primary links, read-version links, and access qualifications.
- **`reference_catalog.json`**, **`evidence/source_access.json`**, and **`references.json`** — reviewed bibliographic records, access notes, and generated CSL bibliography.

Use this report for the current assessment rather than the older `TaichiDough_novelty_publication_readiness_report.pdf` in the parent directory. In particular, the earlier report incorrectly treated the default-E training loss as a held-out score. The new report corrects that comparison and distinguishes the completed local contact-retention sweep from an older failed run. The older files were not overwritten.

## Main conclusions

The research idea is scientifically valid, including forward-only inference. The broad ingredients have strong precedents: DPSI, EMPM, DiffCal, visual soft-body calibration, and inverse food rheometry. The proposed contribution is a measured answer to which real-dough behaviours a restricted depth/tool-motion experiment can distinguish, and whether the calibrated model predicts new motions after fitting stops.

The current solver also contains an affine-transfer scaling inconsistency verified by an isolated algebraic test. It should be resolved and tested before interpreting SI-valued fits; affected calibrations need rerunning. This report **does not change simulator or sweep source**, run new material calibrations, or claim proposed experiments have been completed.

## Evidence and verification

- `evidence/current_system.md` — current-code and stored-experiment audit, source locations, split-aware metrics.
- `evidence/visual_inverse.md` — twelve primary visual/inverse studies, exact fitted parameters, evaluation distinctions, source versions, and search log.
- `evidence/dough_rheology.md` — food rheology, robotic measurement, preparation and experiment-design evidence.
- `evidence/methods_evaluation.md` — numerical foundations, identifiability, discrepancy and manipulation benchmarks.
- `evidence/review_protocol.md` — scope, evidence policy and document design.
- `evidence/check_affine_transfer.py` and `.json` — reproducible NumPy transfer calculation, explicitly not a Taichi simulation.
- `evidence/document_build.json`, `pdf_quality.json`, and `typesetting_warnings.txt` — document verification outputs.
- `evidence/document_review.md` — human-readable review of those document checks.

Downloaded third-party papers remain local reading material. They are not part of the web publication or a licence to redistribute the papers. The report is a targeted critical scoping review, not an exhaustive priority search or a registered systematic review.

## Rebuild

The bibliography build is offline:

```sh
python3 build_bibliography.py
```

The document build needs Pandoc, LuaLaTeX, TeX Gyre Pagella/Heros and DejaVu Sans Mono, BeautifulSoup, and the `markdown-latex-report` skill's Lua filter and bundled LaTeX packages. Optional PDF checks use PyMuPDF. Python package requirements are listed in `requirements-report.txt`; they belong in a separate document environment, not the simulator environment.

With those tools installed:

```sh
export REPORT_SKILL_DIR="$HOME/.claude/skills/markdown-latex-report"
export TEXINPUTS="$REPORT_SKILL_DIR//:${TEXINPUTS:-}"
python3 build_report.py
python3 evidence/verify_pdf.py
```

Set `PANDOC` to an explicit executable if Pandoc is not on `PATH`. Set `LATEX_ENGINE` to select a compatible LuaLaTeX executable. `--html-only` builds the web companion without TeX. Intermediate files use `$CLAUDE_JOB_DIR/tmp` when that environment variable is present, otherwise `.build/` in this directory.

The build used Pandoc 3.6.1 and LuaTeX 1.14 with a compatible isolated zlib/texlive-luatex extraction because the machine's default LuaTeX/zlib combination failed. No system package or simulator dependency was replaced. A normal compatible TeX installation does not need that workaround. PyMuPDF 1.26.3 was used from a separate virtual environment; a project-language-server missing-import diagnostic does not mean the executed document check lacked that dependency.

`reference_specs.json`, partial `evidence/metadata/` responses and `evidence/typesetting_check.*` are retained research/build intermediates. The authoritative bibliography inputs are `reference_catalog.json` and `evidence/source_access.json`, not the initial DOI-only list.

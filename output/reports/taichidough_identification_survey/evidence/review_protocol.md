# Review protocol and editorial decisions

Review date: 2026-09-11.
Repository revision at start: cb76169436d334c127f165ce5c3f779100bb82d4.

## Research question

Under what observation, excitation, model and evaluation conditions can depth-camera recordings and recorded tool trajectories support calibration of the elastic, viscous and plastic response of dough? Which scientifically testable contributions remain for TaichiDough when parameter estimation uses forward sweeps rather than automatic differentiation?

## Scope

This is a critical scoping review and research proposal, not a registered systematic review or proof of priority. Earlier conversation conclusions and the previous report are not treated as evidence. Paper names from that context are search leads only. Current code, stored experiment records, primary full texts and publisher metadata are assessed separately.

The review covers: (1) visual inverse dynamics and material identification; (2) dough rheology and robotic/visual material characterization; (3) MPM numerical foundations and contact; (4) predictive manipulation and datasets; (5) identifiability, model discrepancy, derivative-free inference and experimental design. Studies are distinguished by actual specimen, observation modality, known action/force, constitutive model, fitted parameters, optimization, validation split and availability of physical reference measurements.

Primary-paper evidence takes precedence over search snippets, repository documentation, demonstrations or previous summaries. Abstract-only and metadata-only evidence are labeled. Quantitative values from separate systems are not ranked when tasks, sensors, units, visibility or evaluation protocols differ. Search notes are saved in the topic evidence files. No total screened-paper count or acceptance probability will be invented.

## Main-session searches

- `dough material parameter identification depth camera robot trajectory viscoelastic plastic simulation inverse calibration`: initial orientation; assigned detailed primary-paper follow-up to the visual-inverse and dough-rheology researchers.
- `survey review deformable object physical parameter estimation vision inverse simulation robotics 2024 2025 2026`: checked for overlap with broad reviews. Search results alone do not establish a review's availability, exact scope or conclusions.

## Official publication criteria checked

- IEEE RA-L author information: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/ — explicitly includes innovative ideas, theoretical findings and application case studies. No official requirement for a new optimizer or differentiability found on this page.
- IEEE RA-P: https://www.ieee-ras.org/publications/ra-p/ — practical robotics investigations, case studies and reproducible/verifiable real-world advances; a potentially appropriate systems/experimental-paper direction.
- JOSS submitting guidance: https://joss.readthedocs.io/en/latest/submitting.html — research-software value can include engineering, not only new algorithms. Must show research use, mature open development, complete functionality, installability, tests and license; eligibility must be rechecked before submission.
- IJRR author instructions: https://journals.sagepub.com/author-instructions/ijr — access returned HTTP 403; no detailed rule attributed to this inaccessible page.

## Document design

Subject: identifying dough response from partial visual observations and prescribed tool motion.
Audience: the developer/researcher and prospective academic collaborators.
Job: explain the field, establish a defensible research question, and specify evidence for a publication.

PDF: single-column A4, 11 pt TeX Gyre Pagella body, TeX Gyre Heros headings, DejaVu Sans Mono for code; blue section headings; short comparative tables with wrapping; linked citations; numbered equations where useful; table of contents and page headers. Existing report remains unchanged.

Read-only HTML companion: palette tokens paper #F7FAFC, ink #192E3D, muted #4D6675, accent #216B86, border #CDDCE4; dark equivalents chosen together. Georgia serif headings paired with a system sans-serif body. Reading column around 70 characters, a compact contents sidebar at wide widths and stacked navigation on phones. Use units and explicit observation/parameter distinctions as content rather than decorative graphics. No saving, credentials, analytics or external data access.

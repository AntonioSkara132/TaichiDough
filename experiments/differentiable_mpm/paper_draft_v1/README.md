# Example paper draft

This is a separate working draft based on the title, author line, affiliation, and two-column US Letter layout of `/home/antonio/Downloads/askara_icra2026.pdf`. The original PDF and earlier project manuscripts are unchanged.

## Files

- `root.tex`: main document and conference formatting.
- `abstract.tex`: example abstract.
- `introduction.tex`: motivation, research hypothesis, and contribution scope.
- `related_work.tex`: expanded discussion of 25 cited sources.
- `method.tex`: current implemented geometry preparation, material/contact model, observation objective, checkpointed differentiation, optimization, and evaluation scope.
- `references.bib`: editable bibliography adapted from the project survey's reference catalog.
- `ieeeconf.cls`: unmodified conference class from the official PaperCept archive.
- `build.py`: reproducible pdfLaTeX/BibTeX build with shell escape disabled.
- `DRAFTING_NOTES.md`: source evidence, qualifications, and unfinished experimental claims.

A successful build creates `askara_draft.pdf` and `root.bbl` here. Those are generated outputs, not edits to the supplied PDF. Build artifacts and logs go to the selected build directory.

## Build

Requires an existing installation of Python 3, pdfLaTeX, BibTeX, Latin Modern, amsmath/amssymb, booktabs, cite, microtype, flushend, and hyperref. No package installation is performed by the script.

From this directory:

```bash
python build.py
```

Or keep intermediate files in a chosen directory:

```bash
python build.py --build-dir /path/to/build-directory
```

The script runs pdfLaTeX, BibTeX, and two further pdfLaTeX passes. The title and author details are preserved for the requested example; their inclusion is not a statement that every listed author has reviewed or approved this draft.

## Scope

The document contains the requested sections and references, not fabricated results, a claimed accepted submission, or a finished experimental paper. A visible working-draft note identifies its status. The original empty figure placeholder is omitted rather than presented as an actual figure. Expanded related work and a complete method naturally require more space than the supplied two-page outline.

There has been no external publication. The current conference's submission page limit, anonymity requirements, and official submission-year template should be checked before using this as a submission.

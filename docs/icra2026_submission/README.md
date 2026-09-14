# TaichiDough ICRA 2026 draft

This directory preserves the structure of `/home/antonio/Downloads/askara_icra2026(1).zip`. The source ZIP is unchanged.

## Build

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Run these commands from this directory. `EVIDENCE.md` records the source of every quantitative claim. Each generated figure has a JSON file with its data and display provenance.

## Interpretation

The manuscript reports existing single-sequence evidence. Parameter values are effective simulator parameters conditional on the camera calibration, reconstructed volume, mass/density, contact law, constitutive model, and numerical resolution. The draft does not claim universal dough constants, independent-motion generalization, completed robot experiments, or verified full-horizon CUDA gradients.

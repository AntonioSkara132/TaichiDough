#!/usr/bin/env python3
"""Remove empty white margins from the saved render without changing scene pixels."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from render_article_stills import font, save_pdf_lossless

OUTPUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = ("t0p5004", "t1p2008")
    targets = [OUTPUT / f"figure1_{name}.{extension}" for name in names for extension in ("png", "pdf")]
    targets += [OUTPUT / "figure1_preview.png", OUTPUT / "composition_manifest.json"]
    for path in targets:
        if path.exists():
            raise FileExistsError(path)
    sources = [OUTPUT / f"episode18_{name}_clean.png" for name in names]
    render_manifest = json.loads((OUTPUT / "figure_manifest.json").read_text())
    images, bounds = [], []
    for path in sources:
        if digest(path) != render_manifest["output_sha256"][path.name]:
            raise ValueError(f"Saved render changed: {path}")
        with Image.open(path) as image:
            image.load()
            image = image.convert("RGB")
        pixels = np.asarray(image)
        y, x = np.where(np.any(pixels < 250, axis=2))
        bounds.append([int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)])
        images.append(image)
    padding = 64
    box = [max(0, min(b[0] for b in bounds) - padding), max(0, min(b[1] for b in bounds) - padding),
           min(images[0].width, max(b[2] for b in bounds) + padding),
           min(images[0].height, max(b[3] for b in bounds) + padding)]
    previews = []
    for name, image in zip(names, images):
        cropped = image.crop(box)
        cropped.save(OUTPUT / f"figure1_{name}.png", dpi=(500, 500))
        save_pdf_lossless(cropped, OUTPUT / f"figure1_{name}.pdf")
        preview = cropped.copy()
        preview.thumbnail((1411, 900))
        previews.append(preview)
        print(f"figure1_{name}: {cropped.width} x {cropped.height} px", flush=True)
    sheet = Image.new("RGB", (1411, sum(p.height + 90 for p in previews)), "white")
    y = 0
    for label, image in zip(("t = 0.5004 s", "t = 1.2008 s"), previews):
        ImageDraw.Draw(sheet).text((32, y + 24), label, font=font(34), fill="#263341")
        sheet.paste(image, (0, y + 90))
        y += image.height + 90
    sheet.save(OUTPUT / "figure1_preview.png")
    manifest = {
        "operation": "Common integer-pixel crop, no resampling of publication images, no scene modification",
        "crop_xyxy_px": box,
        "image_size_px": [box[2] - box[0], box[3] - box[1]],
        "png_dpi_metadata": 500,
        "pdf_width_inches": 7.1,
        "pdf_effective_pixels_per_inch": (box[2] - box[0]) / 7.1,
        "pdf_encoding": "Losslessly Flate-compressed RGB raster",
        "input_sha256": {p.name: digest(p) for p in [*sources, OUTPUT / "figure_manifest.json", Path(__file__), OUTPUT / "render_article_stills.py"]},
        "output_sha256": {p.name: digest(p) for p in targets if p.suffix in (".png", ".pdf")},
        "rendering_parameters": {"voxel_m": .0015, "gaussian_sigma_voxels": .8, "density_isovalue": .20, "tile_cells": 64},
        "no_simulation_or_calibration_run": True,
    }
    with (OUTPUT / "composition_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

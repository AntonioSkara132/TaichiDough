#!/usr/bin/env python3
"""Render article stills from saved particles and tool poses; do not run physics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
RUN = ROOT / "experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd_best_preview_v2_fixed"
SUPPORT = ROOT / "experiments/differentiable_mpm/forward_video_v2"
sys.path.insert(0, str(SUPPORT))
from render_support import density_boundary

COLORS = {"dough": "#d9a45d", "tool_1": "#326bb1", "tool_2": "#c85049"}
SIZE = (3600, 2300)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stl(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(raw, dtype=dtype, offset=84, count=count)["vertices"].astype(float)


def mesh_from_triangles(triangles):
    faces = np.column_stack((np.full(len(triangles), 3), np.arange(len(triangles) * 3).reshape(-1, 3)))
    return pv.PolyData(triangles.reshape(-1, 3), faces.ravel()).clean(tolerance=0.0)


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def save_pdf_lossless(image, path, width_inches=7.1):
    """Embed the RGB raster using lossless Flate compression, without JPEG conversion."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    pw = width_inches * 72
    ph = pw * h / w
    stream = f"q {pw:.6f} 0 0 {ph:.6f} 0 0 cm /Im0 Do Q\n".encode("ascii")
    pixels = zlib.compress(rgb.tobytes(), 9)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.6f} {ph:.6f}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode("ascii"),
        f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(pixels)} >>\nstream\n".encode("ascii") + pixels + b"\nendstream",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream",
    ]
    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    with path.open("xb") as f:
        f.write(data)


def render(triangles, tools, lower, upper, floor_y, name):
    plot = pv.Plotter(off_screen=True, window_size=SIZE, lighting="none")
    plot.set_background("white")
    dough = mesh_from_triangles(triangles)
    plot.add_mesh(dough, color=COLORS["dough"], smooth_shading=True, ambient=0.32,
                  diffuse=0.68, specular=0.12, specular_power=24, show_edges=False)
    for index, vertices in enumerate(tools, 1):
        plot.add_mesh(mesh_from_triangles(vertices), color=COLORS[f"tool_{index}"],
                      smooth_shading=True, split_sharp_edges=True, feature_angle=40,
                      ambient=0.28, diffuse=0.72, specular=0.16, specular_power=28,
                      show_edges=False)
    floor = np.array([[lower[0], floor_y, lower[2]], [upper[0], floor_y, lower[2]],
                      [upper[0], floor_y, upper[2]], [lower[0], floor_y, upper[2]]])
    plot.add_mesh(pv.PolyData(floor, [4, 0, 1, 2, 3]), color="#f1f3f5", lighting=False)
    # The grid is a display-only metric reference with 5 cm spacing.
    for x in np.arange(np.ceil(lower[0] / .05) * .05, upper[0], .05):
        plot.add_mesh(pv.Line((x, floor_y + .00002, lower[2]), (x, floor_y + .00002, upper[2])),
                      color="#dbe0e5", line_width=1.2, lighting=False)
    for z in np.arange(np.ceil(lower[2] / .05) * .05, upper[2], .05):
        plot.add_mesh(pv.Line((lower[0], floor_y + .00002, z), (upper[0], floor_y + .00002, z)),
                      color="#dbe0e5", line_width=1.2, lighting=False)
    target = (lower + upper) * .5
    direction = np.array([1.75, 1.8, -2.1]); direction /= np.linalg.norm(direction)
    right = np.cross([0, 1, 0], direction); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    corners = np.array([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])
                        for z in (lower[2], upper[2])]) - target
    aspect = SIZE[0] / SIZE[1]
    half_height = 1.05 * max(np.abs(corners @ up).max(), np.abs(corners @ right).max() / aspect)
    plot.camera_position = [target + direction * 1.5, target, [0, 1, 0]]
    plot.enable_parallel_projection()
    plot.camera.parallel_scale = float(half_height)
    plot.camera.clipping_range = (.001, 10)
    for position, intensity in [([1.2, 1.4, -.2], .75), ([.0, .8, .8], .42), ([.9, .4, 1.2], .20)]:
        plot.add_light(pv.Light(position=position, focal_point=target, color="white", intensity=intensity))
    plot.enable_anti_aliasing("ssaa")
    rgb = plot.screenshot(return_img=True)
    plot.close()
    if rgb is None:
        raise RuntimeError("Offscreen renderer returned no image")
    image = Image.fromarray(rgb).convert("RGB")
    image.save(OUTPUT / f"{name}_clean.png", dpi=(500, 500))
    save_pdf_lossless(image, OUTPUT / f"{name}_clean.pdf")
    return image, {
        "projection": "orthographic", "position_m": (target + direction * 1.5).tolist(),
        "target_m": target.tolist(), "up": [0, 1, 0], "parallel_scale_m": float(half_height),
        "image_size_px": list(SIZE), "grid_spacing_m": .05,
    }


def main():
    names = ("episode18_t0p5004", "episode18_t1p2008")
    for name in names:
        for suffix in ("_clean.png", "_clean.pdf", ".png", ".pdf"):
            if (OUTPUT / (name + suffix)).exists():
                raise FileExistsError(OUTPUT / (name + suffix))
    result_path = RUN / "simulation/simulation_result.json"
    prepared_path = RUN / "simulation/prepared_inputs.json"
    config_path = RUN / "forward_config.json"
    result = json.loads(result_path.read_text())
    prepared = json.loads(prepared_path.read_text())
    config = json.loads(config_path.read_text())
    if result["status"] != "completed":
        raise ValueError("Selected replay did not complete")
    collision = prepared["provenance"]["collision"]
    sources = [result_path, prepared_path, config_path, Path(__file__), SUPPORT / "render_support.py"]
    tool_local = []
    for asset in collision["assets_in_stream_order"]:
        path = Path(asset["path"])
        if sha256(path) != asset["sha256"]:
            raise ValueError(f"Tool mesh changed: {path}")
        sources.append(path)
        vertices = stl(path) * collision["mesh_scale"]
        vertices = vertices @ Rotation.from_euler("xyz", asset["visual_rpy_rad"]).as_matrix().T
        vertices += np.asarray(asset["visual_origin_m"])
        tool_local.append(vertices)
    selected = []
    bounds_lower, bounds_upper = np.full(3, np.inf), np.full(3, -np.inf)
    for target_time, name in zip((.5, 1.2), names):
        record = min(result["frames"], key=lambda value: abs(value["sim_time_s"] - target_time))
        path = RUN / "simulation" / record["particles"]
        points = np.load(path, allow_pickle=False)
        if points.shape != (24000, 3) or not np.isfinite(points).all():
            raise ValueError("Saved particle array failed validation")
        sources.append(path)
        triangles, extraction = density_boundary(points, prepared["mass"]["particle_volume_m3"])
        tools = [vertices @ Rotation.from_quat(np.asarray(pose)[3:]).as_matrix().T + np.asarray(pose)[:3]
                 for vertices, pose in zip(tool_local, record["tool_poses"])]
        vertices = np.concatenate([triangles.reshape(-1, 3), *(mesh.reshape(-1, 3) for mesh in tools)])
        bounds_lower = np.minimum(bounds_lower, vertices.min(0))
        bounds_upper = np.maximum(bounds_upper, vertices.max(0))
        selected.append((name, record, points, triangles, tools, extraction))
    before = {str(path): sha256(path) for path in sources}
    lower = bounds_lower - np.array([.025, .003, .025])
    upper = bounds_upper + np.array([.025, .015, .025])
    details, previews = [], []
    for name, record, points, triangles, tools, extraction in selected:
        image, camera = render(triangles, tools, lower, upper, config["simulation"]["floor_y"], name)
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.text((120, 95), f"t = {record['sim_time_s']:.4f} s", font=font(57), fill="#263341")
        x = 120
        for label, color in [("Simulated dough", COLORS["dough"]), ("UR tool", COLORS["tool_1"]), ("Kinova tool", COLORS["tool_2"])]:
            draw.ellipse((x, SIZE[1] - 130, x + 29, SIZE[1] - 101), fill=color)
            draw.text((x + 48, SIZE[1] - 148), label, font=font(40), fill="#263341")
            x += 510
        draw.text((SIZE[0] - 690, SIZE[1] - 148), "Floor grid: 5 cm", font=font(40), fill="#5c6671")
        annotated.save(OUTPUT / f"{name}.png", dpi=(500, 500))
        save_pdf_lossless(annotated, OUTPUT / f"{name}.pdf")
        preview = annotated.copy(); preview.thumbnail((1440, 920)); previews.append(preview)
        details.append({"name": name, "source_frame": record["source_frame"],
                        "original_source_frame": record.get("original_source_frame"), "step": record["step"],
                        "simulation_time_s": record["sim_time_s"], "particle_count": len(points),
                        "particle_bounds_m": [points.min(0).tolist(), points.max(0).tolist()],
                        "tool_poses": record["tool_poses"], "extraction": extraction, "camera": camera})
        print(f"Rendered {name}, source frame {record['source_frame']}, t={record['sim_time_s']:.7f}s", flush=True)
    if {path: sha256(path) for path in before} != before:
        raise RuntimeError("Input files changed during rendering")
    sheet = Image.new("RGB", (1440, sum(p.height for p in previews)), "white")
    y = 0
    for preview in previews:
        sheet.paste(preview, (0, y)); y += preview.height
    sheet.save(OUTPUT / "preview.png")
    outputs = sorted(path for path in OUTPUT.iterdir() if path.suffix in (".png", ".pdf"))
    manifest = {"source_run": str(RUN), "source_simulation_status": result["status"],
                "simulation_rerun": False, "calibration_rerun": False, "generative_image_used": False,
                "parameters": config["parameters"], "mass": prepared["mass"], "colors": COLORS,
                "frames": details, "input_sha256": before, "inputs_unchanged": True,
                "output_sha256": {path.name: sha256(path) for path in outputs},
                "limitations": ["Qualitative render of a saved forward replay, not an observed camera image.",
                                "Uses best-so-far parameters from an interrupted fit, not validated final material identification.",
                                "Display geometry is a density isosurface; particles and recorded tool poses are unchanged.",
                                "Colors and 5 cm floor grid are display choices; they do not encode stress or error.",
                                "PDFs contain losslessly embedded high-resolution raster images, not vector geometry."]}
    with (OUTPUT / "figure_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__":
    main()

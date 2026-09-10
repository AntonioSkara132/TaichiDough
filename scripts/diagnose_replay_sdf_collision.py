#!/usr/bin/env python3
"""Inspect SDF collision meshes and optional replay collision-response counters.

The replay does not persist its SDF volume. This tool rebuilds a diagnostic volume
from the collision-solid STL, using the same construction function as the simulator.
It reports the reconstruction as an inspection artifact, not as a runtime SDF dump.
"""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
import hashlib
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from taichi_viscoelastic_mpm_scene import (
        KINOVA_TOOL_VISUAL_ORIGIN,
        KINOVA_TOOL_VISUAL_RPY,
        UR_TOOL_VISUAL_ORIGIN,
        UR_TOOL_VISUAL_RPY,
        build_mesh_sdf,
        load_binary_stl,
        mesh_in_tool_frame,
        rpy_to_matrix,
    )
except ImportError:
    from .taichi_viscoelastic_mpm_scene import (
        KINOVA_TOOL_VISUAL_ORIGIN,
        KINOVA_TOOL_VISUAL_RPY,
        UR_TOOL_VISUAL_ORIGIN,
        UR_TOOL_VISUAL_RPY,
        build_mesh_sdf,
        load_binary_stl,
        mesh_in_tool_frame,
        rpy_to_matrix,
    )


BACKGROUND = (24, 25, 25)
PANEL = (39, 40, 39)
INK = (242, 242, 236)
MUTED = (194, 194, 185)
VISUAL = (80, 159, 220)
COLLISION = (234, 142, 53)
NEGATIVE = np.array([50, 126, 202], dtype=np.float64)
ZERO = np.array([229, 229, 221], dtype=np.float64)
POSITIVE = np.array([215, 82, 68], dtype=np.float64)
CONTACT_DEBUG_SCHEMA = "taichidough/replay-sdf-contact-debug/v1"
MANIFEST_SCHEMA = "taichidough/tool-collision-meshes/v1"


@dataclass(frozen=True)
class ToolAsset:
    name: str
    visual_path: Path
    collision_path: Path
    visual_sha256: str | None
    collision_sha256: str | None
    manifest: dict[str, Any]


@dataclass
class SdfInspection:
    asset: ToolAsset
    visual_vertices: np.ndarray
    collision_vertices: np.ndarray
    signed_distance: np.ndarray
    minimum: np.ndarray
    spacing: np.ndarray
    statistics: dict[str, Any]


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_asset_sha256(path: Path, expected: str | None, label: str) -> str:
    actual = sha256(path)
    if expected is not None and actual.lower() != str(expected).lower():
        raise ValueError(f"{label} SHA-256 mismatch for {path}: expected {expected}, found {actual}")
    return actual


def _two_names(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two tool names")
    if any(not isinstance(name, str) or not name.strip() for name in value) or len(set(value)) != 2:
        raise ValueError(f"{label} must contain two unique nonempty tool names")
    return value


def _finite_scalar(value: Any, label: str, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not np.isfinite(parsed) or (positive and parsed <= 0):
        raise ValueError(f"{label} must be {'positive ' if positive else ''}and finite")
    return parsed


def load_replay_metadata(replay_dir: Path) -> dict[str, Any]:
    return read_json(replay_dir / "camera_parameters.json", "Replay metadata")


def validate_sdf_replay_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    replay = metadata.get("replay")
    if not isinstance(replay, dict):
        raise ValueError("Replay metadata is missing the replay object")
    if replay.get("tool_collision") != "sdf":
        raise ValueError("Replay metadata must record replay.tool_collision as 'sdf'")
    if replay.get("tool_pose_frame") != "mesh_tool_link":
        raise ValueError("Replay metadata must record replay.tool_pose_frame as 'mesh_tool_link'")
    names = _two_names(replay.get("tool_names"), "replay.tool_names")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay metadata must contain a nonempty frames list")
    seen: set[int] = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("frame"), int):
            raise ValueError(f"replay.frames[{index}] must identify an integer frame")
        if frame["frame"] in seen:
            raise ValueError(f"Replay metadata repeats frame {frame['frame']}")
        seen.add(frame["frame"])
    parameters = metadata.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Replay metadata parameters must be an object")
    return {"replay": replay, "frames": frames, "parameters": parameters, "tool_names": names}


def load_collision_manifest(path: Path) -> dict[str, Any]:
    return read_json(path, "Collision manifest")


def validate_collision_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Collision manifest must use schema {MANIFEST_SCHEMA}")
    tools = manifest.get("tools")
    if not isinstance(tools, list) or len(tools) != 2:
        raise ValueError("Collision manifest must contain exactly two tools")
    indexed: dict[str, dict[str, Any]] = {}
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"collision manifest tools[{index}] must be an object")
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in indexed:
            raise ValueError("Collision manifest tools must have unique nonempty names")
        for field in ("source_mesh", "collision_mesh"):
            if not isinstance(tool.get(field), str) or not tool[field]:
                raise ValueError(f"collision manifest tool {name} is missing {field}")
        if tool.get("coordinate_frame") != "raw_stl_visual":
            raise ValueError(f"collision manifest tool {name} must use coordinate_frame raw_stl_visual")
        indexed[name] = tool
    return indexed


def _tool_transform(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "UR5e_spathla":
        return UR_TOOL_VISUAL_ORIGIN, rpy_to_matrix(UR_TOOL_VISUAL_RPY)
    if name == "gen3_spathla":
        return KINOVA_TOOL_VISUAL_ORIGIN, rpy_to_matrix(KINOVA_TOOL_VISUAL_RPY)
    raise ValueError(f"No raw-STL-to-link transform is registered for tool {name!r}")


def resolve_tool_assets(
    replay: dict[str, Any], manifest_path: Path, manifest: dict[str, dict[str, Any]],
    visual_overrides: dict[str, Path | None], collision_overrides: dict[str, Path | None],
) -> list[ToolAsset]:
    result: list[ToolAsset] = []
    for name in replay["tool_names"]:
        if name not in manifest:
            raise ValueError(f"Collision manifest has no entry for replay tool {name!r}")
        entry = manifest[name]
        visual = visual_overrides.get(name) or manifest_path.parent / entry["source_mesh"]
        collision = collision_overrides.get(name) or manifest_path.parent / entry["collision_mesh"]
        visual, collision = Path(visual).expanduser(), Path(collision).expanduser()
        for path, label in ((visual, "Visual STL"), (collision, "Collision-solid STL")):
            if not path.is_file():
                raise FileNotFoundError(f"{label} for {name} does not exist: {path}")
        verify_asset_sha256(visual, entry.get("source_sha256"), f"Visual STL for {name}")
        verify_asset_sha256(collision, entry.get("collision_sha256"), f"Collision-solid STL for {name}")
        result.append(ToolAsset(name, visual, collision, entry.get("source_sha256"), entry.get("collision_sha256"), entry))
    return result


def summarize_sdf(sdf: np.ndarray, minimum: np.ndarray, spacing: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(sdf)
    total = int(sdf.size)
    negative, zero, positive = (int(np.count_nonzero(sdf < 0)), int(np.count_nonzero(sdf == 0)), int(np.count_nonzero(sdf > 0)))
    voxel_volume = float(np.prod(spacing))
    maximum = minimum + spacing * (np.asarray(sdf.shape) - 1)
    return {
        "resolution": list(sdf.shape), "finite_voxels": int(np.count_nonzero(finite)), "nonfinite_voxels": total - int(np.count_nonzero(finite)),
        "negative_voxels": negative, "zero_voxels": zero, "positive_voxels": positive,
        "negative_fraction": negative / total, "zero_fraction": zero / total, "positive_fraction": positive / total,
        "minimum_signed_distance_m": float(np.nanmin(sdf)), "maximum_signed_distance_m": float(np.nanmax(sdf)),
        "minimum_scene_m": minimum.tolist(), "maximum_scene_m": maximum.tolist(), "voxel_spacing_scene_m": spacing.tolist(),
        "voxel_volume_m3": voxel_volume, "estimated_negative_volume_m3": negative * voxel_volume,
    }


def rebuild_tool_sdf(asset: ToolAsset, resolution: int, padding_m: float, scale: float) -> SdfInspection:
    visual_raw, _ = load_binary_stl(asset.visual_path, scale)
    collision_raw, _ = load_binary_stl(asset.collision_path, scale)
    origin, rotation = _tool_transform(asset.name)
    visual = mesh_in_tool_frame(visual_raw, origin, rotation)
    collision = mesh_in_tool_frame(collision_raw, origin, rotation)
    sdf, _, minimum, spacing = build_mesh_sdf(collision, resolution, padding_m)
    return SdfInspection(asset, visual, collision, sdf, minimum, spacing, summarize_sdf(sdf, minimum, spacing))


def _font(size: int):
    try:
        from PIL import ImageFont
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        from PIL import ImageFont
        return ImageFont.load_default()


def _line(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, y: int, *, fill=INK, font_size=14) -> int:
    draw.text(xy, text, fill=fill, font=_font(font_size))
    return y + font_size + 5


def _project(vertices: np.ndarray, axis: int, bounds: tuple[np.ndarray, np.ndarray], width: int, height: int, pad: int = 14) -> np.ndarray:
    keep = [index for index in range(3) if index != axis]
    low, high = bounds[0][keep], bounds[1][keep]
    span = np.maximum(high - low, 1e-8)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    points = np.empty((len(vertices), 2), dtype=float)
    points[:, 0] = pad + (vertices[:, keep[0]] - low[0]) * scale
    points[:, 1] = height - pad - (vertices[:, keep[1]] - low[1]) * scale
    return points


def render_mesh_panel(title: str, vertices: np.ndarray, axis: int, bounds: tuple[np.ndarray, np.ndarray], color: tuple[int, int, int], subtitle: str) -> Image.Image:
    image = Image.new("RGB", (360, 360), PANEL)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 359, 50), fill=BACKGROUND)
    draw.text((10, 8), title, fill=INK, font=_font(16))
    draw.text((10, 30), subtitle, fill=MUTED, font=_font(10))
    projected = _project(vertices, axis, bounds, image.width, image.height - 52)
    projected[:, 1] += 52
    triangles = projected.reshape(-1, 3, 2)
    stride = max(1, len(triangles) // 12000)
    for triangle in triangles[::stride]:
        draw.line([tuple(point) for point in (*triangle, triangle[0])], fill=color, width=1)
    draw.rectangle((0, 51, 359, 359), outline=(90, 91, 86))
    return image


def _slice_rgb(values: np.ndarray, limit_m: float) -> np.ndarray:
    value = np.clip(values / max(limit_m, 1e-9), -1, 1)
    rgb = np.empty((*values.shape, 3), dtype=np.uint8)
    negative = value < 0
    rgb[negative] = (ZERO + (-value[negative, None]) * (NEGATIVE - ZERO)).astype(np.uint8)
    rgb[~negative] = (ZERO + value[~negative, None] * (POSITIVE - ZERO)).astype(np.uint8)
    return rgb


def render_sdf_slice(inspection: SdfInspection, axis: int, fraction: float, limit_m: float) -> Image.Image:
    mesh_minimum = float(inspection.collision_vertices[:, axis].min())
    mesh_maximum = float(inspection.collision_vertices[:, axis].max())
    requested_coordinate = mesh_minimum + fraction * (mesh_maximum - mesh_minimum)
    index = int(round((requested_coordinate - inspection.minimum[axis]) / inspection.spacing[axis]))
    index = int(np.clip(index, 0, inspection.signed_distance.shape[axis] - 1))
    values = np.take(inspection.signed_distance, index, axis=axis).T
    rgb = _slice_rgb(values, limit_m)
    panel = Image.fromarray(rgb, "RGB").resize((360, 300), Image.Resampling.NEAREST)
    image = Image.new("RGB", (360, 360), PANEL)
    image.paste(panel, (0, 50))
    draw = ImageDraw.Draw(image)
    planes = ("YZ", "XZ", "XY")
    coordinate = float(inspection.minimum[axis] + index * inspection.spacing[axis])
    draw.rectangle((0, 0, 359, 49), fill=BACKGROUND)
    draw.text((10, 7), f"{planes[axis]} SDF slice", fill=INK, font=_font(16))
    draw.text((10, 28), f"{coordinate * 1000:.2f} mm | blue: inside; red: exterior", fill=MUTED, font=_font(10))
    draw.line((8, 348, 112, 348), fill=tuple(NEGATIVE.astype(int)), width=4)
    draw.line((112, 348, 220, 348), fill=tuple(ZERO.astype(int)), width=4)
    draw.line((220, 348, 352, 348), fill=tuple(POSITIVE.astype(int)), width=4)
    draw.text((8, 332), f"−{limit_m * 1000:.1f} mm", fill=MUTED, font=_font(10))
    draw.text((260, 332), f"+{limit_m * 1000:.1f} mm", fill=MUTED, font=_font(10))
    return image


def _stat_lines(stats: dict[str, Any]) -> list[str]:
    return [
        f"Grid: {'×'.join(map(str, stats['resolution']))}; finite {stats['finite_voxels']:,}",
        f"Inside: {stats['negative_voxels']:,} ({stats['negative_fraction']:.1%})", f"Surface: {stats['zero_voxels']:,} ({stats['zero_fraction']:.1%})",
        f"Exterior: {stats['positive_voxels']:,} ({stats['positive_fraction']:.1%})", f"Range: {stats['minimum_signed_distance_m'] * 1000:.2f} to {stats['maximum_signed_distance_m'] * 1000:.2f} mm",
        "Negative voxels estimate collision-solid interior, not observed contact.",
    ]


def render_tool_section(inspection: SdfInspection, fraction: float, limit_m: float) -> Image.Image:
    all_vertices = np.concatenate((inspection.visual_vertices, inspection.collision_vertices))
    lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
    margin = max(float(np.max(upper - lower)) * .05, .001)
    bounds = (lower - margin, upper + margin)
    visual = render_mesh_panel("Visual STL surface", inspection.visual_vertices, 1, bounds, VISUAL, "Rendering/reference geometry")
    collision = render_mesh_panel("Collision-solid STL", inspection.collision_vertices, 1, bounds, COLLISION, "Geometry voxelized for SDF collision")
    slices = [render_sdf_slice(inspection, axis, fraction, limit_m) for axis in range(3)]
    image = Image.new("RGB", (1100, 840), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, 58), fill=(16, 17, 17))
    draw.text((16, 11), inspection.asset.name, fill=INK, font=_font(24))
    draw.text((16, 38), "All geometry is shown in mesh-tool-link coordinates. SDF rebuilt for inspection; the replay did not store a runtime volume.", fill=MUTED, font=_font(11))
    image.paste(visual, (10, 70)); image.paste(collision, (380, 70))
    y = 82
    for line in _stat_lines(inspection.statistics):
        y = _line(draw, (760, y), line, y, fill=INK if y == 82 else MUTED, font_size=12)
    spacing = inspection.statistics["voxel_spacing_scene_m"]
    _line(draw, (760, y + 8), "Spacing: " + ", ".join(f"{item * 1000:.3f} mm" for item in spacing), y, fill=MUTED, font_size=12)
    for index, panel in enumerate(slices):
        image.paste(panel, (10 + index * 365, 450))
    return image


def _counter(tool: dict[str, Any], group: str, field: str) -> int:
    try:
        value = tool[group][field]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Contact debug tool counter is missing {group}.{field}") from exc
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"Contact debug tool counter {group}.{field} must be a nonnegative integer")
    return value


def validate_contact_debug(debug: dict[str, Any], replay: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if debug.get("schema") != CONTACT_DEBUG_SCHEMA:
        raise ValueError(f"Contact debug must use schema {CONTACT_DEBUG_SCHEMA}")
    if debug.get("tool_collision") != "sdf":
        raise ValueError("Contact debug must record tool_collision as 'sdf'")
    if _two_names(debug.get("tool_names"), "contact debug tool_names") != replay["tool_names"]:
        raise ValueError("Contact debug tool_names do not match replay tool_names and ordering")
    frames = debug.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Contact debug frames must be a list")
    replay_frames = {row["frame"] for row in replay["frames"]}
    indexed: dict[int, dict[str, Any]] = {}
    for row in frames:
        if not isinstance(row, dict) or not isinstance(row.get("frame"), int) or row["frame"] not in replay_frames:
            raise ValueError("Contact debug frame is missing or does not align with the replay")
        if row["frame"] in indexed or not isinstance(row.get("tools"), list) or len(row["tools"]) != 2:
            raise ValueError(f"Contact debug frame {row['frame']} must have two unique tool counter records")
        for tool in row["tools"]:
            for group in ("grid_nodes", "particles"):
                for field in ("contact_candidates", "applied_responses", "inward_normal_velocity_removed"):
                    _counter(tool, group, field)
        indexed[row["frame"]] = row
    return indexed


def selected_contact_rows(replay: dict[str, Any], indexed: dict[int, dict[str, Any]] | None, limit: int) -> list[dict[str, Any]]:
    if indexed is None:
        return []
    present = [row for row in replay["frames"] if row["frame"] in indexed]
    if not present:
        return []
    positions = sorted(set(np.linspace(0, len(present) - 1, min(limit, len(present))).round().astype(int)))
    result = []
    for position in positions:
        frame = present[position]; debug = indexed[frame["frame"]]
        result.append({"frame": frame["frame"], "source_frame": frame.get("source_frame"), "sim_time_s": frame.get("sim_time_s"), "snapshot_without_substep_evaluations": debug.get("snapshot_without_substep_evaluations", False), "tools": debug["tools"]})
    return result


def _png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO(); image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_html(output: Path, report: dict[str, Any], sheet: Image.Image) -> Path:
    data = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    sheet_uri = _png_data_uri(sheet)
    rows = report["contact_rows"]
    if rows:
        table_rows = "".join(
            "<tr><td>{frame}</td><td>{source}</td><td>{time:.4f}</td><td>{snapshot}</td>{tools}</tr>".format(
                frame=row["frame"], source=row.get("source_frame", "—"), time=float(row.get("sim_time_s") or 0),
                snapshot="yes" if row["snapshot_without_substep_evaluations"] else "no",
                tools="".join("<td>{}</td><td>{}</td><td>{}</td><td>{}</td>".format(_counter(tool, "grid_nodes", "contact_candidates"), _counter(tool, "grid_nodes", "applied_responses"), _counter(tool, "particles", "contact_candidates"), _counter(tool, "particles", "applied_responses")) for tool in row["tools"]),
            ) for row in rows
        )
        contact_section = "<table><thead><tr><th>Replay</th><th>Source</th><th>t (s)</th><th>Initial snapshot</th><th>Tool 0 grid proximity</th><th>Tool 0 grid responses</th><th>Tool 0 particle proximity</th><th>Tool 0 particle responses</th><th>Tool 1 grid proximity</th><th>Tool 1 grid responses</th><th>Tool 1 particle proximity</th><th>Tool 1 particle responses</th></tr></thead><tbody>" + table_rows + "</tbody></table>"
    else:
        contact_section = "<p class='notice'>No replay SDF contact-debug file was supplied or found. Geometry inspection is available, but per-frame proximity and collision-response counters cannot be measured.</p>"
    output.write_text(f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>SDF collision inspection</title><style>
:root{{--bg:#181919;--panel:#272827;--ink:#f2f2ec;--muted:#c2c2b9;--line:#5a5c58}}@media(prefers-color-scheme:light){{:root{{--bg:#f6f6f3;--panel:#fff;--ink:#171817;--muted:#51534f;--line:#cccfc8}}}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui,sans-serif}}main{{max-width:1140px;margin:auto;padding:28px}}h1{{margin:0 0 8px}}h2{{margin-top:32px}}p,li{{line-height:1.55;color:var(--muted)}}.panel{{background:var(--panel);padding:18px;border:1px solid var(--line);border-radius:8px;margin:16px 0}}img{{display:block;max-width:100%;height:auto;margin:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{color:var(--muted)}}.scroll{{overflow-x:auto}}.notice{{border-left:4px solid #e89235;padding-left:12px}}code{{color:var(--ink)}}@media(max-width:600px){{main{{padding:16px}}}}</style></head><body><main>
<h1>SDF collision inspection</h1><p>Visual STL and collision-solid geometry are compared in mesh-tool-link coordinates. The signed-distance volumes below were rebuilt from the collision-solid STL for inspection; no runtime SDF array was saved in the replay.</p>
<section class='panel'><h2>Scope and provenance</h2><pre>{html.escape(json.dumps(report['provenance'], indent=2))}</pre></section>
<section class='panel'><h2>Collision geometry and reconstructed SDF</h2><img alt='SDF collision geometry inspection contact sheet' src='{sheet_uri}'></section>
<section class='panel'><h2>Replay collision-response diagnostics</h2><p>“Proximity” is the solver candidate predicate, not a confirmed collision. “Responses” means the corresponding solver branch applied a grid or particle response. A zero padding value makes candidates strictly inside-solid samples.</p><div class='scroll'>{contact_section}</div></section>
<section class='panel'><h2>Definitions and limitations</h2><ul><li>Negative signed distance: inside the collision solid. Positive: exterior. Positive distance is not automatically non-contact when padding is positive.</li><li>Visual geometry can differ from the watertight collision solid intentionally.</li><li>Depth exports contain dough only; they do not show tool occlusion and cannot establish collision.</li><li>Voxel resolution and reconstruction settings affect the displayed SDF.</li></ul><p><a href='visualization_manifest.json'>Machine-readable report JSON</a> · <a href='sdf_collision_contact_sheet.png'>Static contact sheet</a></p></section>
<script type='application/json' id='report-data'>{data}</script></main></body></html>""", encoding="utf-8")
    return output


def create_report(args: argparse.Namespace) -> Path:
    replay_dir = args.replay_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Diagnostic output directory must be empty: {output_dir}")
    replay = validate_sdf_replay_metadata(load_replay_metadata(replay_dir))
    manifest_path = args.collision_manifest.expanduser().resolve()
    manifest = validate_collision_manifest(load_collision_manifest(manifest_path))
    visual_overrides = {"UR5e_spathla": args.ur_visual_mesh, "gen3_spathla": args.kinova_visual_mesh}
    collision_overrides = {"UR5e_spathla": args.ur_collision_mesh, "gen3_spathla": args.kinova_collision_mesh}
    assets = resolve_tool_assets(replay, manifest_path, manifest, visual_overrides, collision_overrides)
    scale = _finite_scalar(args.tool_mesh_scale if args.tool_mesh_scale is not None else replay["parameters"].get("tool_mesh_scale", .001), "tool mesh scale", positive=True)
    resolution = int(args.sdf_resolution if args.sdf_resolution is not None else replay["parameters"].get("tool_sdf_resolution", 64))
    if resolution < 16:
        raise ValueError("SDF resolution must be at least 16")
    padding = _finite_scalar(args.contact_padding if args.contact_padding is not None else replay["replay"].get("tool_contact_padding_scene_m", 0), "contact padding")
    fraction = _finite_scalar(args.slice_fraction, "slice fraction")
    if not 0 <= fraction <= 1:
        raise ValueError("slice fraction must be between zero and one")
    inspections = [rebuild_tool_sdf(asset, resolution, padding, scale) for asset in assets]
    negative_magnitudes = np.concatenate([np.abs(item.signed_distance[item.signed_distance < 0]) for item in inspections])
    default_slice_limit_m = max(float(np.quantile(negative_magnitudes, .99)) * 1.5, 1e-4)
    limit_m = _finite_scalar(args.max_slice_distance_mm / 1000 if args.max_slice_distance_mm is not None else default_slice_limit_m, "slice distance limit", positive=True)
    sections = [render_tool_section(item, fraction, limit_m) for item in inspections]
    sheet = Image.new("RGB", (1100, 80 + sum(section.height + 12 for section in sections)), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), "SDF collision geometry inspection", fill=INK, font=_font(26))
    draw.text((16, 45), "Blue: visual STL | orange: collision-solid STL | SDF: blue inside, red exterior", fill=MUTED, font=_font(12))
    y = 80
    for section in sections:
        sheet.paste(section, (0, y)); y += section.height + 12
    debug_path = args.contact_debug
    if debug_path is None:
        ref = replay["replay"].get("sdf_contact_diagnostics", {})
        if isinstance(ref, dict) and ref.get("path"):
            candidate = Path(ref["path"])
            debug_path = candidate if candidate.is_file() else None
        if debug_path is None:
            candidate = replay_dir / "replay_sdf_contact_debug.json"
            debug_path = candidate if candidate.is_file() else None
    indexed = validate_contact_debug(read_json(debug_path, "Contact debug"), replay) if debug_path is not None else None
    rows = selected_contact_rows(replay, indexed, args.selected_frames)
    provenance = {"replay_dir": str(replay_dir), "collision_manifest": str(manifest_path), "tool_names": replay["tool_names"], "replay_frames": len(replay["frames"]), "contact_debug": str(debug_path) if debug_path else None, "sdf_reconstruction": {"resolution": resolution, "padding_m": padding, "mesh_scale": scale, "slice_fraction": fraction, "slice_limit_m": limit_m, "runtime_sdf_persisted": False}, "tools": [{"name": item.asset.name, "visual_stl": str(item.asset.visual_path), "collision_solid_stl": str(item.asset.collision_path), "visual_sha256": sha256(item.asset.visual_path), "collision_sha256": sha256(item.asset.collision_path), "manifest_build": item.asset.manifest.get("build"), "manifest_topology": item.asset.manifest.get("topology"), "sdf_statistics": item.statistics} for item in inspections]}
    report = {"schema": "taichidough/sdf-collision-inspection/v1", "provenance": provenance, "contact_rows": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(output_dir / "sdf_collision_contact_sheet.png")
    (output_dir / "visualization_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if rows:
        with (output_dir / "sdf_contact_counters.csv").open("w", newline="") as destination:
            writer = csv.writer(destination); writer.writerow(["frame", "source_frame", "sim_time_s", "tool", "grid_proximity_candidates", "grid_responses_applied", "particle_proximity_candidates", "particle_responses_applied"])
            for row in rows:
                for name, tool in zip(replay["tool_names"], row["tools"]): writer.writerow([row["frame"], row.get("source_frame"), row.get("sim_time_s"), name, _counter(tool, "grid_nodes", "contact_candidates"), _counter(tool, "grid_nodes", "applied_responses"), _counter(tool, "particles", "contact_candidates"), _counter(tool, "particles", "applied_responses")])
    write_html(output_dir / "index.html", report, sheet)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True, help="Directory containing camera_parameters.json")
    parser.add_argument("--collision-manifest", type=Path, required=True, help="taichidough/tool-collision-meshes/v1 JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty directory for inspection artifacts")
    parser.add_argument("--contact-debug", type=Path, help="Optional replay_sdf_contact_debug.json")
    parser.add_argument("--ur-visual-mesh", type=Path); parser.add_argument("--kinova-visual-mesh", type=Path)
    parser.add_argument("--ur-collision-mesh", type=Path); parser.add_argument("--kinova-collision-mesh", type=Path)
    parser.add_argument("--tool-mesh-scale", type=float); parser.add_argument("--sdf-resolution", type=int); parser.add_argument("--contact-padding", type=float)
    parser.add_argument("--slice-fraction", type=float, default=.5); parser.add_argument("--selected-frames", type=int, default=6)
    parser.add_argument("--max-slice-distance-mm", type=float, help="Symmetric signed-distance display limit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selected_frames < 1: raise ValueError("selected frames must be at least one")
    output = create_report(args)
    print(f"Wrote SDF collision inspection report to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}")

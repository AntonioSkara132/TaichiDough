#!/usr/bin/env python3
"""Build a local, synchronized visible-surface report from paired dynamics metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BACKGROUND = np.array([26, 26, 25], dtype=np.uint8)
REAL = np.array([57, 135, 229], dtype=np.uint8)
SIM = np.array([217, 89, 38], dtype=np.uint8)
NEUTRAL = np.array([180, 180, 176], dtype=np.uint8)


def depth_colors(depth, valid, low, high):
    rgb = np.broadcast_to(BACKGROUND, (*depth.shape, 3)).copy()
    t = np.clip((depth[valid] - low) / max(high - low, 1e-9), 0, 1)[:, None]
    rgb[valid] = ((1 - t) * np.array([205, 226, 251]) + t * np.array([13, 54, 107])).astype(np.uint8)
    return rgb


def residual_colors(real, rv, sim, sv, limit):
    rgb = np.broadcast_to(BACKGROUND, (*real.shape, 3)).copy()
    common = rv & sv
    t = np.clip((sim[common] - real[common]) / limit, -1, 1)[:, None]
    poles = np.where(t < 0, REAL, np.array([230, 103, 103]))
    rgb[common] = ((1 - np.abs(t)) * np.array([200, 200, 196]) + np.abs(t) * poles).astype(np.uint8)
    return rgb


def edge(mask):
    p = np.pad(mask, 1)
    return mask & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])


def normalize_report_context(report: dict) -> None:
    calibration = report.setdefault("calibration", {})
    schema = calibration.get("schema", "taichidough/topview-calibration/v1")
    if schema == "taichidough/scene-calibration/v2" or calibration.get("is_metric"):
        calibration["display_description"] = (
            f"{calibration.get('name', 'unnamed')}; metric rigid transform from "
            f"{calibration.get('source_frame', 'source')} to {calibration.get('scene_frame', 'scene')}; no fitted scale"
        )
    elif "uniform_scale" in calibration:
        calibration["display_description"] = (
            f"{calibration.get('name', 'unnamed')}; legacy signed-axis map with uniform scale "
            f"{calibration['uniform_scale']}"
        )
    else:
        calibration["display_description"] = calibration.get("name", "unspecified calibration")

    replay = report.setdefault("replay", {})
    geometry = replay.get("tool_geometry")
    if geometry is not None:
        if geometry.get("schema") != "taichidough/tool-geometry/v1":
            raise ValueError("Embedded tool geometry has an unsupported schema")
        names = geometry.get("names")
        half_extents = np.asarray(geometry.get("half_extents_m"), dtype=float)
        proxy = bool(geometry.get("proxy", False))
    else:
        names = replay.get("tool_names", ["tool_0", "tool_1"])
        half_extents = np.asarray(replay.get("tool_half_extents_scene_m", [.05, .05, .05]), dtype=float)
        proxy = True
    if not isinstance(names, list) or len(names) != 2 or len(set(names)) != 2 or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("Replay metadata must identify two uniquely named tools")
    if half_extents.shape == (3,):
        half_extents = np.broadcast_to(half_extents, (2, 3)).copy()
    if half_extents.shape != (2, 3) or not np.isfinite(half_extents).all() or np.any(half_extents <= 0):
        raise ValueError("Replay metadata tool half-extents must be two positive finite XYZ vectors")
    replay["tool_names"] = names
    replay["tool_half_extents_by_tool_m"] = half_extents.tolist()
    replay["tool_geometry_proxy"] = proxy
    replay["tool_geometry_description"] = (
        "proxy boxes derived from legacy replay options" if geometry is None else
        ("per-tool proxy collider geometry" if proxy else "measured per-tool collider geometry")
    )


def create_report(metrics_path: Path, output_dir: Path) -> Path:
    report = json.loads(metrics_path.read_text())
    if report.get("benchmark") != "dynamic-topview-proxy-replay/v1" or not report.get("frames"):
        raise ValueError("Expected a nonempty dynamic-topview-proxy-replay/v1 evaluation")
    normalize_report_context(report)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Visualization output directory must be empty")
    low, high, limit = float("inf"), -float("inf"), .001
    for row in report["frames"]:
        with np.load(metrics_path.parent / row["arrays"]) as arrays:
            for prefix in ("real", "sim"):
                values = arrays[f"{prefix}_depth"][arrays[f"{prefix}_valid"]]
                if len(values):
                    low, high = min(low, float(values.min())), max(high, float(values.max()))
            common = arrays["real_valid"] & arrays["sim_valid"]
            if common.any():
                limit = max(limit, float(np.max(np.abs(arrays["sim_depth"][common] - arrays["real_depth"][common]))))
    if not np.isfinite(low):
        low, high = 0.0, 1.0
    high = max(high, low + .001)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "frames"
    image_dir.mkdir()
    for row in report["frames"]:
        with np.load(metrics_path.parent / row["arrays"]) as arrays:
            real, rv, sim, sv = [arrays[key] for key in ("real_depth", "real_valid", "sim_depth", "sim_valid")]
            real_rgb, sim_rgb = depth_colors(real, rv, low, high), depth_colors(sim, sv, low, high)
            overlay = np.broadcast_to(BACKGROUND, (*real.shape, 3)).copy()
            overlay[rv | sv] = [48, 48, 46]
            overlay[edge(rv)] = REAL
            overlay[edge(sv)] = SIM
            both_edge = edge(rv) & edge(sv)
            overlay[both_edge] = NEUTRAL
            visibility = np.broadcast_to(BACKGROUND, (*real.shape, 3)).copy()
            visibility[rv & ~sv] = REAL
            visibility[sv & ~rv] = SIM
            visibility[rv & sv] = NEUTRAL
            encoded = []
            for depth, valid in ((real, rv), (sim, sv)):
                codes = np.zeros(depth.shape, dtype=np.uint16)
                codes[valid] = (1 + np.rint(np.clip((depth[valid] - low) / (high - low), 0, 1) * 65534)).astype(np.uint16)
                encoded.append(np.stack([(codes >> 8).astype(np.uint8), (codes & 255).astype(np.uint8), np.zeros_like(codes, dtype=np.uint8)], axis=-1))
            images = {"real": real_rgb, "sim": sim_rgb, "overlay": overlay,
                      "residual": residual_colors(real, rv, sim, sv, limit), "visibility": visibility,
                      "depth_values": np.concatenate(encoded, axis=1)}
            row["images"] = {}
            for name, rgb in images.items():
                path = image_dir / f"{row['source_frame']:06d}_{name}.png"
                Image.fromarray(rgb).save(path)
                row["images"][name] = str(path.relative_to(output_dir))
    report["display"] = {"depth_min_m": low, "depth_max_m": high, "residual_limit_m": limit,
                         "scale_policy": "one global metric range over all real and simulated frames; no per-frame normalization",
                         "inspector_quantization_step_m": (high - low) / 65534}
    data = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    content = HTML.replace("__REPORT_JSON__", data)
    output = output_dir / "index.html"
    output.write_text(content)
    (output_dir / "visualization_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    csv_path = metrics_path.parent / "paired_metrics.csv"
    if csv_path.exists():
        (output_dir / "paired_metrics.csv").write_bytes(csv_path.read_bytes())
    selected = sorted(set(np.linspace(0, len(report["frames"]) - 1, min(6, len(report["frames"]))).round().astype(int).tolist()))
    sheet = Image.new("RGB", (660, 210 * len(selected)), tuple(BACKGROUND))
    draw = ImageDraw.Draw(sheet)
    for i, index in enumerate(selected):
        row = report["frames"][index]
        draw.text((10, i * 210 + 5), f"Source {row['source_frame']} | t={row['time_s']:.4f}s | {row['status']} | real / simulation / residual", fill="white")
        for col, name in enumerate(("real", "sim", "residual")):
            image = Image.open(output_dir / row["images"][name])
            image.thumbnail((210, 170))
            sheet.paste(image, (10 + col * 220, i * 210 + 30))
    sheet.save(output_dir / "temporal_contact_sheet.png")
    return output


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Time-aligned dough dynamics diagnostics</title>
<style>
:root{color-scheme:dark;--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--secondary:#c3c2b7;--grid:#383835;--real:#3987e5;--sim:#d95926}
:root[data-theme="light"]{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--secondary:#52514e;--grid:#e1e0d9;--real:#2a78d6;--sim:#eb6834}
*{box-sizing:border-box}body{margin:0;background:var(--plane);color:var(--ink);font:15px system-ui,sans-serif}main{max-width:1480px;margin:auto;padding:28px}h1{font-size:30px;margin:0 0 10px}h2{font-size:18px;margin:0 0 12px}p{line-height:1.5;color:var(--secondary)}aside{border:1px solid #fab219;border-radius:8px;padding:18px;line-height:1.6}button,select{padding:8px 12px;border:1px solid var(--grid);border-radius:6px;background:var(--surface);color:var(--ink)}button{cursor:pointer}a{color:var(--ink)}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:16px 0;position:sticky;top:0;background:var(--plane);z-index:5}input[type=range]{flex:1;min-width:180px}.readout{font-variant-numeric:tabular-nums;line-height:1.65}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:18px 0}.panel{background:var(--surface);padding:18px;border:1px solid var(--grid);border-radius:8px;min-width:0}.image{width:100%;height:auto;display:block;background:#1a1a19;image-rendering:pixelated}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0;color:var(--secondary)}.key{display:inline-block;width:24px;height:3px;vertical-align:middle;margin-right:6px;background:var(--real)}.key.sim{background:var(--sim)}.key.base{background:#999}.bar{height:14px;background:linear-gradient(to right,#cde2fb,#0d366b)}.resbar{background:linear-gradient(to right,#3987e5,#c8c8c4,#e66767)}.scale{display:flex;justify-content:space-between;font-size:12px;color:var(--secondary)}canvas{display:block;width:100%;height:240px;touch-action:pan-y}.context{height:320px}.tooltip{min-height:44px;font-size:13px;font-variant-numeric:tabular-nums;color:var(--secondary)}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--grid);white-space:nowrap}.tablewrap{overflow-x:auto}details{margin:18px 0}summary{cursor:pointer}li{margin:8px 0;line-height:1.45}.cards{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}.cards article{padding:14px;border:1px solid var(--grid);border-radius:8px;min-width:170px}.cards strong{display:block;font-size:24px;margin-top:8px}.small{font-size:12px}.warning{font-weight:600}.swatch{display:inline-block;width:12px;height:12px;margin-right:5px}.unpaired{opacity:.6}@media(max-width:800px){main{padding:14px}.grid{grid-template-columns:1fr}}@media print{.controls{position:static}canvas{break-inside:avoid}}
</style></head><body><main>
<h1>Time-aligned dough dynamics diagnostics</h1>
<aside id="scope"></aside>
<p id="provenance"></p><div class="cards" id="cards"></div>
<div class="controls"><button id="prev" aria-label="Previous frame">← Frame</button><button id="play">Play</button><button id="next" aria-label="Next frame">Frame →</button><label for="timeline">Observation</label><input id="timeline" type="range" min="0" step="1" value="0"><label>Speed <select id="speed"><option value=".25">0.25×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><label>Theme <select id="theme"><option value="dark">Dark</option><option value="light">Light</option></select></label><label><input type="checkbox" id="texture"> Pattern masks</label></div>
<div class="readout" id="readout" aria-live="polite"></div>
<p><label>Pixel X <input id="pixelx" type="number" min="0" value="0" style="width:75px"></label> <label>Y <input id="pixely" type="number" min="0" value="0" style="width:75px"></label> <output id="pixelreadout">Move over a depth/error image or enter pixel coordinates to inspect metric values.</output></p>
<div class="grid">
<section class="panel"><h2>Captured visible surface</h2><img class="image" id="real" alt="Captured raw metric depth"><div class="bar"></div><div class="scale depthscale"></div></section>
<section class="panel"><h2>Simulated visible surface</h2><img class="image" id="sim" alt="Simulation raw metric depth"><div class="bar"></div><div class="scale depthscale"></div></section>
<section class="panel"><h2>Synchronized silhouette overlay</h2><div class="legend"><span><i class="key"></i>Real contour</span><span><i class="key sim"></i>Simulation contour</span><span>Gray: coincident contour</span></div><img class="image" id="overlay" alt="Real and simulated contour overlap"></section>
<section class="panel"><h2>Signed depth error: simulation minus real</h2><img class="image" id="residual" alt="Signed depth error on common visible pixels"><div class="bar resbar"></div><div class="scale" id="residualscale"></div><p class="small">Blue: nearer. Red: farther. Only common visible pixels are scored; dark pixels are unavailable, not zero error.</p></section>
<section class="panel"><h2>Visibility and missing support</h2><canvas id="visibilityCanvas" class="context" aria-label="Pixel support mask" role="img"></canvas><div class="legend"><span><i class="key"></i>Real only / ↗ pattern</span><span><i class="key sim"></i>Simulation only / ↘ pattern</span><span>Gray: both · Dark: neither/unknown</span></div><p class="small" id="support"></p></section>
<section class="panel"><h2>Tool trajectory context — scene X/Z</h2><canvas id="tools" class="context" role="img" aria-label="Captured and replayed tool trajectories"></canvas><div class="legend"><span><i class="key"></i>Captured tool path / circle</span><span><i class="key sim"></i>Replayed collider pose / square</span></div><p class="small" id="toolGeometry"></p></section>
</div>
<h2>Temporal diagnostics</h2><div class="legend" id="worstframes"></div><p>Every plot uses actual captured elapsed seconds. Hover or focus a plot and use arrow keys to inspect all values at a timestamp. Click a plot to synchronize the surface panels. Gray lines show the frozen initial simulation where relevant. Initialization is excluded from summary averages.</p>
<div class="grid" id="charts"></div>
<details open><summary>Metric definitions, assumptions and provenance</summary><ul id="limitations"></ul><p class="small">Footprint area counts occupied optical XY cells; it is not true curved surface area. Centroids describe visible samples, not a material center of mass. Depth-change error subtracts each sequence's initial depth on four-way common support. No time warping, per-frame registration, inferred contact labels or particle correspondence is used. Fixed display ranges include all exported real and simulation depths; long tails can reduce contrast rather than being hidden by rescaling.</p><p><a href="paired_metrics.csv">Download paired metrics CSV</a> · <a href="visualization_manifest.json">Full metrics and provenance JSON</a> · <a href="temporal_contact_sheet.png">Temporal contact sheet</a></p></details>
<details><summary>Per-frame numerical table (also available without hover)</summary><div class="tablewrap"><table id="table"></table></div></details>
</main><script>
'use strict';
const report=__REPORT_JSON__, rows=report.frames, $=id=>document.getElementById(id);
let selected=0,timer=null;
const fmt=(v,d=3)=>v==null||!Number.isFinite(v)?'N/A':v.toFixed(d);
const get=(r,path)=>path.split('.').reduce((o,k)=>o==null?null:o[k],r);
const css=k=>getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const geometryStatus=report.replay.tool_geometry_proxy?'tool collider geometry is a proxy':'tool collider geometry is recorded per tool';
$('scope').textContent=report.scope==='initialization_only'?'⚠ INITIALIZATION ONLY — no paired nonzero simulation time; dynamics have not been evaluated.':`⚠ TRAJECTORY-CONDITIONED ROLLOUT — simulated evolution is compared with captured observations, but physical fidelity is NOT validated; ${geometryStatus}.`;
$('provenance').textContent=`${report.counts.observations} observations; ${report.counts.paired} paired, ${report.counts.post_initial_paired} after initialization. Calibration: ${report.calibration.display_description}. Distances below are scene metres/mm. Pairing tolerance ${fmt(report.pair_tolerance_s*1000,3)} ms. Source time origin ${fmt(report.replay.source_time_origin_s,6)} s. Fixed ${report.camera.width}×${report.camera.height} pinhole image, splat radius ${report.camera.splat_radius} px. Playback follows actual timestamp gaps. Simulation: ${report.simulation_parameters?.particles??'unspecified'} particles, grid ${report.simulation_parameters?.grid??'unspecified'}, dt ${report.simulation_parameters?.dt??'unspecified'} s.`;
$('toolGeometry').textContent=`Tool names identify separate paths. The box boundary is the unit simulation domain, not a measured table boundary. Solid orange wireframes show ${report.replay.tool_geometry_description}; dashed orange wireframes add the configured contact padding. Neither denotes a confirmed contact event.`;
for(const [label,value] of [['Evaluated span',fmt(rows.at(-1).time_s,3)+' s'],['Nonzero paired frames',String(report.counts.post_initial_paired)],['Physical fidelity validated','No']]){const a=document.createElement('article');a.textContent=label;const s=document.createElement('strong');s.textContent=value;a.append(s);$('cards').append(a)}
for(const text of report.limitations){const li=document.createElement('li');li.textContent=text;$('limitations').append(li)}
for(const el of document.querySelectorAll('.depthscale')){el.textContent=`${fmt(report.display.depth_min_m,4)} m (near) — shared metric range — ${fmt(report.display.depth_max_m,4)} m (far)`}
for(const label of [`−${fmt(report.display.residual_limit_m*1000,2)} mm`,'0',`+${fmt(report.display.residual_limit_m*1000,2)} mm`]){const span=document.createElement('span');span.textContent=label;$('residualscale').append(span)}
$('timeline').max=rows.length-1;
function setupCanvas(canvas){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.round(rect.height*dpr);const c=canvas.getContext('2d');c.scale(dpr,dpr);return[c,rect.width,rect.height]}
const specs=[
 ['Footprint IoU','ratio',[['metrics.footprint_iou','Simulation vs real',1]]],
 ['Pixel IoU and frozen baseline','ratio',[['metrics.pixel_iou','Simulation vs real',1],['frozen_baseline.pixel_iou','Frozen initial shape',1]]],
 ['Pixel depth p95','mm',[['metrics.pixel_p95_m','Simulation error',1000],['frozen_baseline.pixel_p95_m','Frozen error',1000]]],
 ['Signed pixel depth bias','mm',[['metrics.pixel_bias_m','Simulation minus real',1000]]],
 ['Coverage by reference support','ratio',[['metrics.real_coverage','Fraction of real covered',1],['metrics.sim_coverage','Fraction of simulation covered',1]]],
 ['Common visible support','pixels',[['metrics.common_visible_pixels','Scored pixels',1]]],
 ['Occupied optical footprint area','cm²',[['real_summary.area_m2','Real',10000],['sim_summary.area_m2','Simulation',10000]]],
 ['Visible centroid X','mm',[['real_summary.centroid_x_m','Real',1000],['sim_summary.centroid_x_m','Simulation',1000]]],
 ['Visible centroid Y','mm',[['real_summary.centroid_y_m','Real',1000],['sim_summary.centroid_y_m','Simulation',1000]]],
 ['Visible footprint width X','mm',[['real_summary.width_x_m','Real',1000],['sim_summary.width_x_m','Simulation',1000]]],
 ['Visible footprint width Y','mm',[['real_summary.width_y_m','Real',1000],['sim_summary.width_y_m','Simulation',1000]]],
 ['Depth change error since initialization','mm',[['metrics.depth_change_mae_m','Change MAE',1000]]],
 ['Mean symmetric silhouette distance','pixels',[['metrics.boundary_mean_distance_px','Boundary distance',1]]],
 ['Tool separation','scene m',[['tool_separation_scene_m','Tool-center distance',1]]],
 ['Tool speed: '+report.replay.tool_names[0],'scene m/s',[['tool_speed_scene_m_s.0','Captured path derivative',1]]],
 ['Tool speed: '+report.replay.tool_names[1],'scene m/s',[['tool_speed_scene_m_s.1','Captured path derivative',1]]],
 ['Time pairing error','ms',[['pairing_error_s','Simulation minus real time',1000]]]
];
for(const [key,summary] of Object.entries(report.summary??{})){const button=document.createElement('button');button.textContent=({pixel_iou:'Worst pixel overlap',pixel_mae_m:'Worst depth MAE',pixel_p95_m:'Worst depth p95',depth_change_mae_m:'Worst depth change'}[key]??key)+': frame '+summary.worst_source_frame;button.onclick=()=>{stop();select(rows.findIndex(r=>r.source_frame===summary.worst_source_frame))};$('worstframes').append(button)}
const chartObjects=[];
for(const [title,unit,series] of specs){const panel=document.createElement('section');panel.className='panel';const heading=document.createElement('h2');heading.textContent=title+' ('+unit+')';panel.append(heading);const legend=document.createElement('div');legend.className='legend';for(let s=0;s<series.length;s++){const span=document.createElement('span'),key=document.createElement('i');key.className='key'+(s?' sim':'');if(series[s][0].startsWith('frozen'))key.className='key base';span.append(key,document.createTextNode(series[s][1]));legend.append(span)}panel.append(legend);const canvas=document.createElement('canvas');canvas.tabIndex=0;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',title+'; arrow keys inspect observations');panel.append(canvas);const tooltip=document.createElement('div');tooltip.className='tooltip';panel.append(tooltip);$('charts').append(panel);const obj={canvas,tooltip,unit,series,index:0};chartObjects.push(obj);canvas.addEventListener('pointermove',e=>{const rect=canvas.getBoundingClientRect(),t=Math.max(0,Math.min(1,(e.clientX-rect.left-65)/(rect.width-90)))*Math.max(rows.at(-1).time_s,.001);obj.index=rows.reduce((best,r,i)=>Math.abs(r.time_s-t)<Math.abs(rows[best].time_s-t)?i:best,0);drawChart(obj)});canvas.addEventListener('pointerleave',()=>{obj.index=selected;drawChart(obj)});canvas.addEventListener('click',()=>select(obj.index));canvas.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();obj.index=Math.max(0,Math.min(rows.length-1,obj.index+(e.key==='ArrowRight'?1:-1)));select(obj.index)}})}
function drawChart(obj){const[c,w,h]=setupCanvas(obj.canvas),left=65,right=w-25,top=18,bottom=h-38,maxT=Math.max(rows.at(-1).time_s,.001);let values=[];for(const [path,,factor]of obj.series)for(const r of rows){const v=get(r,path);if(v!=null&&Number.isFinite(v))values.push(v*factor)}let lo=values.length?Math.min(...values):0,hi=values.length?Math.max(...values):1;if(obj.unit==='ratio'){lo=0;hi=1}else{const pad=Math.max((hi-lo)*.08,Math.abs(hi)*.01,.001);lo-=pad;hi+=pad}const x=t=>left+t/maxT*(right-left),y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);c.font='11px system-ui';c.textBaseline='middle';for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,yy=y(v);c.strokeStyle=css('--grid');c.beginPath();c.moveTo(left,yy);c.lineTo(right,yy);c.stroke();c.fillStyle=css('--secondary');c.textAlign='right';c.fillText(fmt(v,obj.unit==='pixels'?0:2),left-8,yy)}for(let i=0;i<5;i++){const t=maxT*i/4;c.textAlign='center';c.fillText(fmt(t,2),x(t),bottom+15)}c.fillText('Elapsed captured time (s)',(left+right)/2,h-4);for(let s=0;s<obj.series.length;s++){const[path,,factor]=obj.series[s];c.strokeStyle=path.startsWith('frozen')?'#999':css(s?'--sim':'--real');c.lineWidth=2;c.setLineDash(s?[6,4]:[]);c.beginPath();let open=false;for(const r of rows){const v=get(r,path);if(v==null||!Number.isFinite(v)){open=false;continue}if(open)c.lineTo(x(r.time_s),y(v*factor));else{c.moveTo(x(r.time_s),y(v*factor));open=true}}c.stroke();c.setLineDash([])}const r=rows[obj.index];c.strokeStyle=css('--secondary');c.lineWidth=1;c.beginPath();c.moveTo(x(r.time_s),top);c.lineTo(x(r.time_s),bottom);c.stroke();let readout=[`t=${fmt(r.time_s,4)} s | frame ${r.source_frame}`];for(let s=0;s<obj.series.length;s++){const[path,label,factor]=obj.series[s],v=get(r,path);readout.push(label+': '+fmt(v==null?null:v*factor)+' '+obj.unit);if(v!=null){c.fillStyle=path.startsWith('frozen')?'#999':css(s?'--sim':'--real');c.beginPath();c.arc(x(r.time_s),y(v*factor),4,0,Math.PI*2);c.fill()}}obj.tooltip.textContent=readout.join(' · ')}
function drawProxy(c,pose,half,padding,x,y){const[qx,qy,qz,qw]=pose.slice(3),rotation=[[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],[2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],[2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]],vertices=[];for(let i=0;i<8;i++){const local=half.map((v,j)=>(i&(1<<j)?1:-1)*(v+padding));vertices.push(rotation.map((r,j)=>pose[j]+r.reduce((sum,v,k)=>sum+v*local[k],0)))}c.strokeStyle=css('--sim');c.lineWidth=1;c.setLineDash(padding?[3,3]:[]);c.beginPath();for(let i=0;i<8;i++)for(let j=0;j<3;j++){const k=i^(1<<j);if(k>i){c.moveTo(x(vertices[i][0]),y(vertices[i][2]));c.lineTo(x(vertices[k][0]),y(vertices[k][2]))}}c.stroke();c.setLineDash([])}
function drawTools(){const[c,w,h]=setupCanvas($('tools')),all=rows.flatMap(r=>(r.captured_tools_scene||[]).map(p=>[p[0],p[2]])),xs=all.map(p=>p[0]),zs=all.map(p=>p[1]);let loX=Math.min(0,...xs),hiX=Math.max(1,...xs),loZ=Math.min(0,...zs),hiZ=Math.max(1,...zs);const scale=Math.min((w-70)/(hiX-loX),(h-55)/(hiZ-loZ)),x=v=>35+(v-loX)*scale,y=v=>h-30-(v-loZ)*scale;c.strokeStyle=css('--grid');c.strokeRect(x(0),y(1),scale,scale);c.fillStyle=css('--secondary');c.font='11px system-ui';c.fillText('X → (m)',w-80,h-8);c.fillText('Z ↑ (m)',8,14);for(let tool=0;tool<2;tool++){c.strokeStyle=css('--real');c.lineWidth=2;c.setLineDash(tool?[5,4]:[]);c.beginPath();for(let i=0;i<rows.length;i++){const p=rows[i].captured_tools_scene[tool];i?c.lineTo(x(p[0]),y(p[2])):c.moveTo(x(p[0]),y(p[2]))}c.stroke();c.setLineDash([]);const p=rows[selected].captured_tools_scene[tool];c.fillStyle=css('--real');c.beginPath();c.arc(x(p[0]),y(p[2]),5,0,Math.PI*2);c.fill();c.fillStyle=css('--ink');c.fillText(report.replay.tool_names[tool],Math.min(w-125,Math.max(5,x(p[0])+8)),Math.max(20,y(p[2])-10));const q=rows[selected].replayed_tools_scene?.[tool];if(q){const half=report.replay.tool_half_extents_by_tool_m[tool];drawProxy(c,q,half,0,x,y);drawProxy(c,q,half,report.replay.tool_contact_padding_scene_m??0,x,y);c.strokeStyle=css('--sim');c.lineWidth=2;c.strokeRect(x(q[0])-5,y(q[2])-5,10,10)}}}
function drawVisibility(){const img=new Image(),index=selected;img.onload=()=>{if(index!==selected)return;const[c,w,h]=setupCanvas($('visibilityCanvas'));const scale=Math.min(w/img.width,h/img.height),iw=img.width*scale,ih=img.height*scale,ox=(w-iw)/2,oy=(h-ih)/2;c.imageSmoothingEnabled=false;c.drawImage(img,ox,oy,iw,ih);if($('texture').checked){const off=document.createElement('canvas');off.width=img.width;off.height=img.height;const oc=off.getContext('2d');oc.drawImage(img,0,0);const data=oc.getImageData(0,0,img.width,img.height);for(let y=0;y<img.height;y++)for(let x=0;x<img.width;x++){const p=(y*img.width+x)*4,isReal=data.data[p]===57&&data.data[p+1]===135,isSim=data.data[p]===217&&data.data[p+1]===89;if((isReal&&(x+y)%7===0)||(isSim&&(x-y+img.height*7)%7===0)){data.data[p]=data.data[p+1]=data.data[p+2]=235}}oc.putImageData(data,0,0);c.drawImage(off,ox,oy,iw,ih)}};img.src=rows[selected].images.visibility}
let pixelData=null;
$('pixelx').max=report.camera.width-1;$('pixely').max=report.camera.height-1;
function inspectPixel(){if(!pixelData)return;const w=report.camera.width,h=report.camera.height,x=Math.max(0,Math.min(w-1,Number($('pixelx').value)||0)),y=Math.max(0,Math.min(h-1,Number($('pixely').value)||0));const read=offset=>{const i=(y*w*2+x+offset)*4,code=pixelData[i]*256+pixelData[i+1];return code?report.display.depth_min_m+(code-1)*report.display.inspector_quantization_step_m:null},real=read(0),sim=read(w);$('pixelreadout').textContent=`Pixel (${x},${y}): real ${fmt(real,5)} m; simulation ${fmt(sim,5)} m; error ${fmt(real==null||sim==null?null:(sim-real)*1000,3)} mm. Inspector precision ±${fmt(report.display.inspector_quantization_step_m*500,4)} mm; metrics use original floats.`}
function loadPixelData(){pixelData=null;const image=new Image(),index=selected;image.onload=()=>{if(index!==selected)return;const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;const c=canvas.getContext('2d');c.drawImage(image,0,0);pixelData=c.getImageData(0,0,image.width,image.height).data;inspectPixel()};image.src=rows[selected].images.depth_values}
$('pixelx').oninput=inspectPixel;$('pixely').oninput=inspectPixel;for(const name of ['real','sim','overlay','residual'])$(name).onpointermove=e=>{const rect=e.currentTarget.getBoundingClientRect();$('pixelx').value=Math.min(report.camera.width-1,Math.max(0,Math.floor((e.clientX-rect.left)/rect.width*report.camera.width)));$('pixely').value=Math.min(report.camera.height-1,Math.max(0,Math.floor((e.clientY-rect.top)/rect.height*report.camera.height)));inspectPixel()};
function select(index){selected=Math.max(0,Math.min(rows.length-1,index));$('timeline').value=selected;const r=rows[selected];for(const name of['real','sim','overlay','residual'])$(name).src=r.images[name];$('readout').textContent=`Retained frame ${r.source_frame} (original ${r.original_source_frame}) | source ${fmt(r.source_timestamp_s,6)} s | elapsed ${fmt(r.time_s,6)} s | simulation ${fmt(r.sim_time_s,6)} s | lag ${fmt(r.pairing_error_s==null?null:r.pairing_error_s*1000,3)} ms | substeps ${r.simulation_step??'N/A'} | ${r.status} | ${r.phase}`;const m=r.metrics;$('support').textContent=m?`Common ${m.common_visible_pixels}; real only ${m.real_only_pixels}; simulation only ${m.sim_only_pixels}. Both absent or tool-occluded pixels are not distinguishable with the available dough-only capture.`:'No paired metric support. Missing data is not zero error.';loadPixelData();drawVisibility();drawTools();for(const obj of chartObjects){obj.index=selected;drawChart(obj)}}
function stop(){clearTimeout(timer);timer=null;$('play').textContent='Play'}function tick(){if(selected>=rows.length-1){stop();return}const delay=Math.max(1,(rows[selected+1].time_s-rows[selected].time_s)*1000/Number($('speed').value));timer=setTimeout(()=>{select(selected+1);tick()},delay)}$('play').onclick=()=>{if(timer){stop();return}if(selected===rows.length-1)select(0);$('play').textContent='Pause';tick()};$('timeline').oninput=()=>{stop();select(Number($('timeline').value))};$('prev').onclick=()=>{stop();select(selected-1)};$('next').onclick=()=>{stop();select(selected+1)};$('theme').onchange=()=>{document.documentElement.dataset.theme=$('theme').value;select(selected)};$('texture').onchange=drawVisibility;
const columns=[['Frame','source_frame'],['Time (s)','time_s'],['Status','status'],['Lag (s)','pairing_error_s'],['Pixel IoU','metrics.pixel_iou'],['Footprint IoU','metrics.footprint_iou'],['p95 (m)','metrics.pixel_p95_m'],['Bias (m)','metrics.pixel_bias_m'],['Common pixels','metrics.common_visible_pixels'],['Change MAE (m)','metrics.depth_change_mae_m'],['Frozen IoU','frozen_baseline.pixel_iou']];for(const [,unit,series] of specs)for(const [path,label] of series){if(!columns.some(c=>c[1]===path))columns.push([label+' (raw SI; '+unit+' on plot)',path])}const thead=document.createElement('thead'),tr=document.createElement('tr');for(const [label]of columns){const th=document.createElement('th');th.textContent=label;tr.append(th)}thead.append(tr);$('table').append(thead);const tbody=document.createElement('tbody');for(const r of rows){const tr=document.createElement('tr');for(const[,path]of columns){const td=document.createElement('td'),v=get(r,path);td.textContent=typeof v==='number'?fmt(v,6):v??'N/A';tr.append(td)}tbody.append(tr)}$('table').append(tbody);window.addEventListener('resize',()=>select(selected));select(0);
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Wrote {create_report(args.metrics, args.output_dir)}")


if __name__ == "__main__":
    main()

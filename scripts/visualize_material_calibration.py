#!/usr/bin/env python3
"""Write an HTML, JSON, and CSV report for effective Young's-modulus identification."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from material_calibration import LOSS_COMPONENTS
except ImportError:
    from .material_calibration import LOSS_COMPONENTS


RESULT_SCHEMA = "taichidough/material-calibration/v1"
COMPONENT_LABELS = {
    "depth_change": "Normalized robust depth-change error",
    "mask_iou": "1 − pixel IoU",
    "observed_to_simulation_distance": "Observed-to-visible-simulation distance",
    "real_coverage": "1 − real coverage",
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _candidate_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in result["search"].get("candidates", []):
        aggregate = candidate.get("aggregate", {})
        row = {
            "split": "training",
            "youngs_modulus_pa": _finite(candidate.get("youngs_modulus_pa")),
            "status": candidate.get("status", "unknown"),
            "weighted_total": _finite(aggregate.get("weighted_total")),
            "failure_reason": candidate.get("failure_reason") or aggregate.get("failure_reason"),
        }
        components = aggregate.get("components", {})
        row.update({name: _finite(components.get(name)) for name in LOSS_COMPONENTS})
        rows.append(row)
    validation = result.get("validation", {}).get("result")
    if validation:
        aggregate = validation.get("aggregate", {})
        row = {
            "split": "validation",
            "youngs_modulus_pa": _finite(validation.get("youngs_modulus_pa")),
            "status": validation.get("status", "unknown"),
            "weighted_total": _finite(aggregate.get("weighted_total")),
            "failure_reason": validation.get("failure_reason") or aggregate.get("failure_reason"),
        }
        components = aggregate.get("components", {})
        row.update({name: _finite(components.get(name)) for name in LOSS_COMPONENTS})
        rows.append(row)
    return rows


def _window_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in result["search"].get("candidates", []):
        for name, loss in candidate.get("windows", {}).items():
            row = {
                "split": "training",
                "window": name,
                "youngs_modulus_pa": candidate.get("youngs_modulus_pa"),
                "valid": loss.get("valid", False),
                "weighted_total": loss.get("weighted_total"),
                "failure_reason": loss.get("failure_reason"),
            }
            row.update({component: loss.get("components", {}).get(component) for component in LOSS_COMPONENTS})
            rows.append(row)
    validation = result.get("validation", {}).get("result")
    if validation:
        for name, loss in validation.get("windows", {}).items():
            row = {
                "split": "validation",
                "window": name,
                "youngs_modulus_pa": validation.get("youngs_modulus_pa"),
                "valid": loss.get("valid", False),
                "weighted_total": loss.get("weighted_total"),
                "failure_reason": loss.get("failure_reason"),
            }
            row.update({component: loss.get("components", {}).get(component) for component in LOSS_COMPONENTS})
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _svg_text(x: float, y: float, text: str, css_class: str = "axis-label", anchor: str = "middle") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{css_class}" text-anchor="{anchor}">{html.escape(text)}</text>'


def _format_pa(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g} MPa"
    if value >= 1_000:
        return f"{value / 1_000:.3g} kPa"
    return f"{value:.4g} Pa"


def _format_loss(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.6g}"


def _chart_svg(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    title: str,
    chart_id: str,
    *,
    frozen_value: float | None,
    baselines: Mapping[str, float | None],
    bootstrap: Mapping[str, Any] | None,
) -> str:
    width, height = 920.0, 390.0
    left, right, top, bottom = 82.0, 28.0, 48.0, 62.0
    plot_width, plot_height = width - left - right, height - top - bottom
    candidates = [float(row["youngs_modulus_pa"]) for row in rows if row.get("youngs_modulus_pa")]
    if not candidates:
        return f'<section class="chart-card"><h3>{html.escape(title)}</h3><p class="empty">No candidate values are available.</p></section>'
    x_min, x_max = min(candidates), max(candidates)
    if math.isclose(x_min, x_max):
        x_min, x_max = x_min / 2.0, x_max * 2.0
    values = [
        float(row[value_key])
        for row in rows
        if row.get("status") == "complete" and _finite(row.get(value_key)) is not None
    ]
    if frozen_value is not None:
        values.append(frozen_value)
    y_max = max(values, default=1.0)
    y_max = max(y_max * 1.12, 1e-9)

    def x_position(value: float) -> float:
        return left + (math.log(value) - math.log(x_min)) / (math.log(x_max) - math.log(x_min)) * plot_width

    def y_position(value: float) -> float:
        return top + plot_height - max(0.0, min(value, y_max)) / y_max * plot_height

    elements = [
        f'<section class="chart-card"><h3>{html.escape(title)}</h3>',
        f'<svg class="loss-chart" id="{html.escape(chart_id)}" viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{html.escape(title)}. Young’s modulus uses a logarithmic axis.">',
    ]
    if bootstrap:
        lower = _finite(bootstrap.get("lower_pa"))
        upper = _finite(bootstrap.get("upper_pa"))
        if lower and upper:
            lo = max(lower, x_min)
            hi = min(upper, x_max)
            if lo <= hi:
                elements.append(
                    f'<rect x="{x_position(lo):.2f}" y="{top:.2f}" width="{max(1.0, x_position(hi) - x_position(lo)):.2f}" height="{plot_height:.2f}" class="bootstrap-band"><title>Whole-window bootstrap interval: {_format_pa(lower)} to {_format_pa(upper)}</title></rect>'
                )
    for tick in range(5):
        value = y_max * tick / 4.0
        y = y_position(value)
        elements.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        elements.append(_svg_text(left - 12, y + 4, f"{value:.3g}", anchor="end"))
    log_low = math.log10(x_min)
    log_high = math.log10(x_max)
    tick_exponents = range(math.floor(log_low), math.ceil(log_high) + 1)
    tick_values = [10.0 ** exponent for exponent in tick_exponents if x_min <= 10.0 ** exponent <= x_max]
    tick_values.extend([x_min, x_max])
    for value in sorted(set(tick_values)):
        x = x_position(value)
        elements.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top + plot_height}" y2="{top + plot_height + 6}" class="axis"/>')
        elements.append(_svg_text(x, top + plot_height + 24, _format_pa(value)))
    elements.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_height}" class="axis"/>')
    elements.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{top + plot_height}" y2="{top + plot_height}" class="axis"/>')
    elements.append(_svg_text(left + plot_width / 2, height - 12, "Effective Young’s modulus (Pa, logarithmic scale)", "axis-title"))
    elements.append(f'<text x="18" y="{top + plot_height / 2}" class="axis-title" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2})">Loss</text>')

    if frozen_value is not None:
        y = y_position(frozen_value)
        elements.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.2f}" y2="{y:.2f}" class="frozen-line"/>')
        elements.append(_svg_text(left + plot_width - 4, max(top + 14, y - 7), f"Frozen {_format_loss(frozen_value)}", "direct-label", "end"))

    vertical_styles = {
        "soft": "baseline-soft",
        "stiff": "baseline-stiff",
        "default": "baseline-default",
        "best": "baseline-best",
    }
    for name in ("soft", "stiff", "default", "best"):
        value = baselines.get(name)
        if value is None or not x_min <= value <= x_max:
            continue
        x = x_position(value)
        elements.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_height}" class="{vertical_styles[name]}"><title>{name.title()} baseline: {_format_pa(value)}</title></line>')

    ordered_training = sorted((row for row in rows if row.get("split") == "training"), key=lambda row: float(row["youngs_modulus_pa"]))
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for row in ordered_training:
        value = _finite(row.get(value_key)) if row.get("status") == "complete" else None
        if value is None:
            if current:
                segments.append(current)
                current = []
        else:
            current.append((x_position(float(row["youngs_modulus_pa"])), y_position(value)))
    if current:
        segments.append(current)
    for segment in segments:
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in segment)
        elements.append(f'<polyline points="{points}" class="training-line"/>')
    for row in rows:
        candidate = float(row["youngs_modulus_pa"])
        x = x_position(candidate)
        value = _finite(row.get(value_key)) if row.get("status") == "complete" else None
        if value is None:
            if row.get("split") == "training":
                y = top + 16
                elements.append(f'<path d="M {x - 5:.2f} {y - 5:.2f} L {x + 5:.2f} {y + 5:.2f} M {x + 5:.2f} {y - 5:.2f} L {x - 5:.2f} {y + 5:.2f}" class="failure-mark"><title>Failed candidate {_format_pa(candidate)}: {html.escape(str(row.get("failure_reason") or "unknown failure"))}</title></path>')
            continue
        y = y_position(value)
        point_class = "training-point" if row.get("split") == "training" else "validation-point"
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" class="{point_class}"/>')

    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        candidate = float(row["youngs_modulus_pa"])
        value = _finite(row.get(value_key)) if row.get("status") == "complete" else None
        item = {
            "series": str(row.get("split", "unknown")).title(),
            "value": _format_loss(value),
            "status": str(row.get("status", "unknown")),
            "reason": str(row.get("failure_reason") or ""),
        }
        grouped.setdefault(candidate, []).append(item)
    sorted_x = sorted(grouped)
    x_positions = [x_position(value) for value in sorted_x]
    for index, candidate in enumerate(sorted_x):
        x = x_positions[index]
        zone_left = left if index == 0 else (x_positions[index - 1] + x) / 2.0
        zone_right = left + plot_width if index + 1 == len(sorted_x) else (x + x_positions[index + 1]) / 2.0
        encoded = html.escape(json.dumps(grouped[candidate], separators=(",", ":")), quote=True)
        label = html.escape(f"{_format_pa(candidate)}; " + "; ".join(f"{item['series']} {item['value']}" for item in grouped[candidate]), quote=True)
        elements.append(f'<rect x="{zone_left:.2f}" y="{top}" width="{max(1.0, zone_right - zone_left):.2f}" height="{plot_height}" class="hit-zone" tabindex="0" data-x="{x:.2f}" data-candidate="{html.escape(_format_pa(candidate), quote=True)}" data-items="{encoded}" aria-label="{label}"/>')
    elements.append(f'<line class="crosshair" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_height}" hidden/>')
    elements.append("</svg></section>")
    return "".join(elements)


def _baseline_value(record: Mapping[str, Any] | None) -> float | None:
    return _finite(record.get("youngs_modulus_pa")) if record else None


def _frozen_component(result: Mapping[str, Any], key: str) -> float | None:
    frozen = result.get("baselines", {}).get("frozen_training") or {}
    if key == "weighted_total":
        return _finite(frozen.get("weighted_total"))
    return _finite(frozen.get("components", {}).get(key))


def _status_block(result: Mapping[str, Any]) -> str:
    status = str(result.get("status", "unknown"))
    selection = result.get("search", {}).get("selection", {})
    best = _finite(selection.get("youngs_modulus_pa"))
    reasons = selection.get("rejection_reasons", [])
    icon = {"accepted": "✓", "needs_review": "!", "failed": "×"}.get(status, "?")
    detail = "Accepted by the configured diagnostics." if status == "accepted" else "; ".join(map(str, reasons)) or "No accepted selection."
    return (
        f'<div class="status status-{html.escape(status)}"><span class="status-icon" aria-hidden="true">{icon}</span>'
        f'<div><strong>{html.escape(status.replace("_", " ").title())}</strong><p>{html.escape(detail)}</p></div></div>'
        f'<div class="hero"><span>Selected effective Young’s modulus</span><strong>{html.escape(_format_pa(best))}</strong></div>'
    )


def build_html(result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    baselines_data = result.get("baselines", {})
    baselines = {
        "best": _baseline_value(baselines_data.get("best")),
        "default": _baseline_value(baselines_data.get("default_e_2000_pa")),
        "soft": _baseline_value(baselines_data.get("softest_successful")),
        "stiff": _baseline_value(baselines_data.get("stiffest_successful")),
    }
    bootstrap = result.get("search", {}).get("bootstrap_interval")
    cards = "".join(
        f'<div class="baseline-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in (
            ("Best training candidate", _format_pa(baselines["best"])),
            ("Default comparison", _format_pa(baselines["default"])),
            ("Soft successful candidate", _format_pa(baselines["soft"])),
            ("Stiff successful candidate", _format_pa(baselines["stiff"])),
            ("Frozen training loss", _format_loss(_frozen_component(result, "weighted_total"))),
        )
    )
    charts = [
        _chart_svg(
            rows,
            "weighted_total",
            "Weighted total loss",
            "total-loss",
            frozen_value=_frozen_component(result, "weighted_total"),
            baselines=baselines,
            bootstrap=bootstrap,
        )
    ]
    for component in LOSS_COMPONENTS:
        charts.append(
            _chart_svg(
                rows,
                component,
                COMPONENT_LABELS[component],
                "component-" + component.replace("_", "-"),
                frozen_value=_frozen_component(result, component),
                baselines=baselines,
                bootstrap=bootstrap,
            )
        )
    table_rows = []
    for row in sorted(rows, key=lambda item: (str(item["split"]), float(item["youngs_modulus_pa"]))):
        table_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    row["split"],
                    _format_pa(_finite(row["youngs_modulus_pa"])),
                    row["status"],
                    _format_loss(_finite(row["weighted_total"])),
                    *(_format_loss(_finite(row[name])) for name in LOSS_COMPONENTS),
                    row.get("failure_reason") or "",
                )
            )
            + "</tr>"
        )
    interval_text = "Unavailable"
    if bootstrap:
        interval_text = f"{_format_pa(_finite(bootstrap.get('lower_pa')))} to {_format_pa(_finite(bootstrap.get('upper_pa')))} ({float(bootstrap.get('confidence', 0)):.1%})"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Material Calibration</title>
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --text:#0b0b0b; --secondary:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10); --training:#2a78d6; --validation:#eb6834; --best:#4a3aa7; --failure:#d03b3b; --good:#0ca30c; --warning:#fab219; --band:#cde2fb; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme:dark; --page:#0d0d0d; --surface:#1a1a19; --text:#fff; --secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --training:#3987e5; --validation:#d95926; --best:#9085e9; --failure:#d03b3b; --good:#0ca30c; --warning:#fab219; --band:#184f95; }} }}
:root[data-theme="dark"] {{ color-scheme:dark; --page:#0d0d0d; --surface:#1a1a19; --text:#fff; --secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --training:#3987e5; --validation:#d95926; --best:#9085e9; --failure:#d03b3b; --good:#0ca30c; --warning:#fab219; --band:#184f95; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--page); color:var(--text); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ width:min(1180px, calc(100% - 32px)); margin:0 auto; padding:38px 0 64px; }} h1 {{ font-size:clamp(28px,4vw,44px); line-height:1.08; margin:0 0 10px; letter-spacing:-.025em; }}
.subtitle {{ color:var(--secondary); max-width:780px; font-size:16px; margin:0 0 24px; }} .status {{ display:flex; gap:12px; align-items:flex-start; padding:16px; border:1px solid var(--border); border-radius:12px; background:var(--surface); }}
.status p {{ margin:3px 0 0; color:var(--secondary); }} .status-icon {{ width:28px; height:28px; display:grid; place-items:center; border-radius:50%; font-weight:800; color:#fff; background:var(--failure); }} .status-accepted .status-icon {{ background:var(--good); }} .status-needs_review .status-icon {{ color:#0b0b0b; background:var(--warning); }}
.hero {{ margin:26px 0; }} .hero span {{ display:block; color:var(--secondary); }} .hero strong {{ display:block; font-size:clamp(38px,7vw,70px); line-height:1; margin-top:6px; letter-spacing:-.035em; }}
.baselines {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:22px 0; }} .baseline-card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px; }} .baseline-card span {{ display:block; color:var(--secondary); font-size:12px; }} .baseline-card strong {{ display:block; margin-top:5px; font-size:17px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:18px; align-items:center; color:var(--secondary); margin:20px 0 8px; }} .legend-key {{ display:inline-flex; align-items:center; gap:7px; }} .legend-line {{ width:24px; height:2px; background:var(--training); }} .legend-dot {{ width:10px; height:10px; border-radius:50%; background:var(--validation); box-shadow:0 0 0 2px var(--surface); }} .legend-failure {{ color:var(--failure); font-weight:800; font-size:18px; }}
.interval {{ color:var(--secondary); margin:0 0 22px; }} .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr)); gap:16px; }} .chart-card:first-child {{ grid-column:1/-1; }} .chart-card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; min-width:0; }} .chart-card h3 {{ margin:0 0 6px; font-size:16px; }}
.loss-chart {{ display:block; width:100%; height:auto; overflow:visible; }} .grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }} .axis-label {{ fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }} .axis-title,.direct-label {{ fill:var(--secondary); font-size:12px; }}
.training-line {{ fill:none; stroke:var(--training); stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }} .training-point {{ fill:var(--training); stroke:var(--surface); stroke-width:2; }} .validation-point {{ fill:var(--validation); stroke:var(--surface); stroke-width:2; }} .failure-mark {{ fill:none; stroke:var(--failure); stroke-width:2.5; stroke-linecap:round; }}
.bootstrap-band {{ fill:var(--band); opacity:.16; }} .frozen-line {{ stroke:var(--secondary); stroke-width:1.5; stroke-dasharray:7 5; }} .baseline-soft,.baseline-stiff {{ stroke:var(--axis); stroke-width:1; }} .baseline-default {{ stroke:var(--secondary); stroke-width:1.5; stroke-dasharray:3 4; }} .baseline-best {{ stroke:var(--best); stroke-width:2; }} .hit-zone {{ fill:transparent; cursor:crosshair; }} .hit-zone:focus {{ outline:none; stroke:var(--text); stroke-width:1; }} .crosshair {{ stroke:var(--secondary); stroke-width:1; pointer-events:none; }}
.tooltip {{ position:fixed; z-index:10; pointer-events:none; min-width:190px; max-width:300px; padding:10px 12px; border-radius:8px; background:var(--surface); border:1px solid var(--border); box-shadow:0 8px 28px rgba(0,0,0,.18); color:var(--text); }} .tooltip strong {{ display:block; margin-bottom:5px; }} .tooltip-row {{ display:flex; justify-content:space-between; gap:18px; }} .tooltip-row span {{ color:var(--secondary); }}
details {{ margin-top:24px; background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }} summary {{ cursor:pointer; font-weight:650; }} .table-wrap {{ overflow-x:auto; margin-top:12px; }} table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }} th,td {{ text-align:left; border-bottom:1px solid var(--grid); padding:8px 10px; white-space:nowrap; }} th {{ color:var(--secondary); }} td:last-child {{ white-space:normal; min-width:220px; }} .note {{ margin-top:22px; color:var(--secondary); }}
@media (forced-colors:active) {{ .training-line,.frozen-line,.baseline-default {{ stroke-dasharray:8 4; }} .validation-point {{ fill:CanvasText; }} }}
</style>
</head>
<body><main>
<h1>Effective Young’s-modulus identification</h1>
<p class="subtitle">This result applies only to the recorded geometry, reconstruction, proxy-tool replay, simulator model, and fixed parameters listed in the calibration result. It is not a universal dough constant.</p>
{_status_block(result)}
<div class="baselines">{cards}</div>
<div class="legend" aria-label="Series legend"><span class="legend-key"><i class="legend-line"></i>Training</span><span class="legend-key"><i class="legend-dot"></i>Held-out validation</span><span class="legend-key"><b class="legend-failure">×</b>Failed candidate</span></div>
<p class="interval"><strong>Whole-window bootstrap interval:</strong> {html.escape(interval_text)}</p>
<div class="chart-grid">{''.join(charts)}</div>
<details><summary>Candidate table</summary><div class="table-wrap"><table><thead><tr><th>Split</th><th>Effective E</th><th>Status</th><th>Total</th><th>Depth change</th><th>1 − IoU</th><th>Observed → visible simulation</th><th>1 − coverage</th><th>Failure</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></details>
<p class="note">Loss components use visible surfaces only. The nearest-distance term is one-sided from observed points to visible simulated points; hidden or interior simulation particles are not compared.</p>
</main><div class="tooltip" hidden></div>
<script>
const tip=document.querySelector('.tooltip');
function showTip(target,event){{
  const svg=target.closest('svg'); const cross=svg.querySelector('.crosshair');
  cross.hidden=false; cross.setAttribute('x1',target.dataset.x); cross.setAttribute('x2',target.dataset.x);
  tip.replaceChildren(); const title=document.createElement('strong'); title.textContent=target.dataset.candidate; tip.append(title);
  for(const item of JSON.parse(target.dataset.items)){{ const row=document.createElement('div'); row.className='tooltip-row'; const label=document.createElement('span'); label.textContent=item.series; const value=document.createElement('b'); value.textContent=item.status==='complete'?item.value:'Failed'; row.append(label,value); tip.append(row); if(item.reason){{ const reason=document.createElement('small'); reason.textContent=item.reason; tip.append(reason); }} }}
  tip.hidden=false; const x=(event&&event.clientX)||target.getBoundingClientRect().left; const y=(event&&event.clientY)||target.getBoundingClientRect().top; tip.style.left=Math.min(innerWidth-tip.offsetWidth-12,x+14)+'px'; tip.style.top=Math.min(innerHeight-tip.offsetHeight-12,y+14)+'px';
}}
function hideTip(target){{ target.closest('svg').querySelector('.crosshair').hidden=true; tip.hidden=true; }}
for(const zone of document.querySelectorAll('.hit-zone')){{ zone.addEventListener('pointermove',e=>showTip(zone,e)); zone.addEventListener('pointerleave',()=>hideTip(zone)); zone.addEventListener('focus',e=>showTip(zone,e)); zone.addEventListener('blur',()=>hideTip(zone)); }}
</script></body></html>"""


def write_report(result_path: Path, output_dir: Path) -> dict[str, Path]:
    result = json.loads(result_path.read_text())
    if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"Expected a {RESULT_SCHEMA!r} result")
    rows = _candidate_rows(result)
    window_rows = _window_rows(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = output_dir / "material_calibration_candidates.csv"
    window_csv = output_dir / "material_calibration_windows.csv"
    summary_json = output_dir / "material_calibration_summary.json"
    report_html = output_dir / "material_calibration_report.html"
    candidate_columns = ["split", "youngs_modulus_pa", "status", "weighted_total", *LOSS_COMPONENTS, "failure_reason"]
    window_columns = ["split", "window", "youngs_modulus_pa", "valid", "weighted_total", *LOSS_COMPONENTS, "failure_reason"]
    _write_csv(candidate_csv, rows, candidate_columns)
    _write_csv(window_csv, window_rows, window_columns)
    summary = {
        "schema": RESULT_SCHEMA,
        "status": result.get("status"),
        "interpretation": result.get("interpretation"),
        "selection": result.get("search", {}).get("selection"),
        "bootstrap_interval": result.get("search", {}).get("bootstrap_interval"),
        "baselines": {
            name: (
                value.get("youngs_modulus_pa") if isinstance(value, dict) and "youngs_modulus_pa" in value
                else value
            )
            for name, value in result.get("baselines", {}).items()
        },
        "candidate_rows": rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2, allow_nan=False))
    report_html.write_text(build_html(result, rows))
    return {"html": report_html, "summary_json": summary_json, "candidate_csv": candidate_csv, "window_csv": window_csv}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="taichidough/material-calibration/v1 JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        outputs = write_report(args.result.resolve(), args.output_dir.resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()

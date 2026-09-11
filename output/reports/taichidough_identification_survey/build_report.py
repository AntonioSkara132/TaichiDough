"""Build the PDF and read-only HTML from the reviewed Markdown source.

Requires pandoc, LuaLaTeX, the markdown-latex-report filter/packages, and
BeautifulSoup for wrapping HTML tables. All bibliography processing is offline.
Use PANDOC, REPORT_SKILL_DIR, and standard TeX environment variables to override
local tool locations. Intermediate output goes to CLAUDE_JOB_DIR/tmp when set.
"""

import argparse
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
STEM = "Dough_Parameter_Identification"


def run(command, *, log=None):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if log is not None:
        log.write_text(result.stdout + result.stderr)
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout


def build_html(pandoc, common, work):
    fragment = run([
        pandoc, str(ROOT / "REPORT.md"), *common,
        "--to=html5", "--mathml", "--section-divs",
    ], log=work / "pandoc-html.log")
    soup = BeautifulSoup(fragment, "html.parser")
    for table in soup.find_all("table"):
        wrapper = soup.new_tag("div", attrs={"class": "table-scroll", "tabindex": "0"})
        wrapper["role"] = "region"
        wrapper["aria-label"] = "Comparison table; scroll horizontally if needed"
        table.wrap(wrapper)
    contents = []
    for heading in soup.find_all("h1"):
        identifier = heading.get("id") or heading.parent.get("id")
        if identifier:
            label = heading.get_text(" ", strip=True)
            contents.append(f'<li><a href="#{html.escape(identifier, quote=True)}">{html.escape(label)}</a></li>')
    toc = '<ol class="toc-list">' + "\n".join(contents) + "</ol>"
    style = (ROOT / "web_style.css").read_text()
    page = f'''<title>Dough Parameter Identification</title>
<style>
{style}
</style>
<a class="skip-link" href="#report-text">Skip to the report</a>
<div class="page-shell">
<header class="report-cover">
<p class="eyebrow">Critical literature survey · 11 September 2026</p>
<h1>Dough Parameter Identification</h1>
<p class="subtitle">Depth recordings, tool trajectories, and the scientific case for TaichiDough</p>
<p class="cover-question">What can a camera and known tool motion identify—and what remains predictable when material parameters are uncertain?</p>
<div class="cover-meta"><span>40 references and guidance sources</span><span>Primary-source evidence with access qualifications</span><span>Current-code assessment</span></div>
</header>
<details class="mobile-contents">
<summary>Contents</summary>
<nav aria-label="Report contents on small screens">{toc}</nav>
</details>
<div class="reading-layout">
<aside class="desktop-contents">
<nav aria-label="Report contents"><p class="nav-label">Contents</p>{toc}</nav>
</aside>
<main id="report-text" class="report-text">
{str(soup)}
</main>
</div>
<footer class="report-footer">
<p>Critical scoping review, not an exhaustive priority claim. Proposed experiments are distinguished from recorded TaichiDough results. Source files, the PDF, and annotated evidence are in <code>output/reports/taichidough_identification_survey/</code>.</p>
<p>Read-only companion to the PDF. No simulator source was changed and no new material calibration was run for this assessment.</p>
</footer>
</div>
'''
    (ROOT / "dough-identification.html").write_text(page)
    head, body = page.split("</style>", 1)
    standalone = (
        '<!doctype html>\n<html lang="en-GB">\n<head>\n'
        '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + head + '</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n'
    )
    (ROOT / f"{STEM}.html").write_text(standalone)
    return len(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    pandoc = os.environ.get("PANDOC") or shutil.which("pandoc")
    if not pandoc:
        raise SystemExit("Pandoc not found; set PANDOC to its executable path.")
    skill = Path(os.environ.get("REPORT_SKILL_DIR", str(Path.home() / ".claude/skills/markdown-latex-report")))
    job = os.environ.get("CLAUDE_JOB_DIR")
    work = Path(job) / "tmp/dough-report-build" if job else ROOT / ".build"
    work.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(ROOT / "build_bibliography.py")])
    source = (ROOT / "REPORT.md").read_text()
    references = json.loads((ROOT / "references.json").read_text())
    ids = {record["id"] for record in references}
    cited = set(re.findall(r"@([A-Za-z0-9][A-Za-z0-9_-]+)", source))
    if cited - ids:
        raise SystemExit(f"Unresolved citations: {sorted(cited - ids)}")
    common = [
        "--from=markdown+raw_tex+raw_attribute-smart",
        "--citeproc", f"--bibliography={ROOT / 'references.json'}",
        f"--csl={ROOT / 'ieee.csl'}", "--number-sections",
    ]
    if not args.html_only:
        latex = run([
            pandoc, str(ROOT / "REPORT.md"), *common, "--to=latex", "--standalone",
            "--toc", "--toc-depth=1", f"--lua-filter={skill / 'filter.lua'}",
            f"--include-in-header={ROOT / 'report_header.tex'}",
            "-V", "mainfont=TeX Gyre Pagella", "-V", "sansfont=TeX Gyre Heros",
            "-V", "monofont=DejaVu Sans Mono", "-V", "linkcolor=accent",
            "-V", "urlcolor=accent", "-V", "citecolor=accent",
        ], log=work / "pandoc-latex.log")
        tex = ROOT / f"{STEM}.tex"
        tex.write_text(latex)
        engine = os.environ.get("LATEX_ENGINE", "lualatex")
        for iteration in range(3):
            run([
                engine, "-interaction=nonstopmode", "-halt-on-error",
                f"-output-directory={work}", str(tex),
            ], log=work / f"latex-pass-{iteration + 1}.log")
        shutil.copy2(work / f"{STEM}.pdf", ROOT / f"{STEM}.pdf")
        logtext = (work / f"{STEM}.log").read_text(errors="replace")
        warnings = [line for line in logtext.splitlines() if any(
            word in line for word in ("Overfull", "Underfull", "Warning", "Missing character", "undefined")
        )]
        (ROOT / "evidence/typesetting_warnings.txt").write_text("\n".join(warnings) + "\n")
    sections = build_html(pandoc, common, work)
    summary = {"reference_count": len(ids), "cited_reference_count": len(cited),
               "unresolved_citations": sorted(cited - ids), "top_level_sections": sections,
               "source_words_approx": len(source.split()), "intermediate_directory": str(work)}
    (ROOT / "evidence/document_build.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not args.html_only:
        print(ROOT / f"{STEM}.pdf")
    print(ROOT / "dough-identification.html")


if __name__ == "__main__":
    main()

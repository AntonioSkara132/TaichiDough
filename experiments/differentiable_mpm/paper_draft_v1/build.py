#!/usr/bin/env python3
"""Build the local example manuscript using pdfLaTeX and BibTeX."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('.build'))
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['BIBINPUTS'] = str(source) + os.pathsep + env.get('BIBINPUTS', '')
    latex = ['pdflatex', '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error',
             '-file-line-error', '-output-directory=' + str(build), 'root.tex']
    commands = [latex, ['bibtex', 'root'], latex, latex]
    for index, command in enumerate(commands, 1):
        working_directory = build if command[0] == 'bibtex' else source
        result = subprocess.run(command, cwd=working_directory, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = build / f'pass_{index}.stdout'
        log.write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            print(result.stdout)
            raise SystemExit(f'Build failed; complete output: {log}')
    pdf = source / 'askara_draft.pdf'
    shutil.copyfile(build / 'root.pdf', pdf)
    shutil.copyfile(build / 'root.bbl', source / 'root.bbl')
    print(f'PDF: {pdf}')
    print(f'Build log: {build / "root.log"}')
    for line in (build / 'root.log').read_text(errors='replace').splitlines():
        if any(marker in line for marker in ('Overfull', 'undefined', 'Warning:')):
            print(line)


if __name__ == '__main__':
    main()

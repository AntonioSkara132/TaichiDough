"""Inspect the generated PDF and render selected pages for document review.

Requires PyMuPDF. This checks document layout, not simulator behavior.
"""

import json
import os
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
pdf = fitz.open(ROOT / "Dough_Parameter_Identification.pdf")
work = Path(os.environ.get("CLAUDE_JOB_DIR", str(ROOT / ".build"))) / "tmp/pdf-review"
work.mkdir(parents=True, exist_ok=True)
margin = 24 * 72 / 25.4
pages = []
outside = []
for index, page in enumerate(pdf):
    text = page.get_text()
    words = page.get_text("words")
    too_wide = [word for word in words if word[0] < margin - 2 or word[2] > page.rect.width - margin + 2]
    outside.extend({"page": index + 1, "word": word[4], "bbox": list(word[:4])} for word in too_wide)
    sizes = [span["size"] for block in page.get_text("dict")["blocks"] if "lines" in block
             for line in block["lines"] for span in line["spans"] if span["text"].strip()]
    pages.append({"page": index + 1, "words": len(words), "smallest_font_pt": round(min(sizes), 2),
                  "first_lines": text.splitlines()[:6], "replacement_character": "�" in text})
text = "\n".join(page.get_text() for page in pdf)
summary = {"pages": len(pdf), "page_size_points": list(pdf[0].rect),
           "words_outside_text_margins": outside, "page_details": pages,
           "bibliography_present": "References" in text,
           "corrected_scores_present": all(value in text for value in ["0.361059", "0.828075", "0.835860", "0.799745"])}
(ROOT / "evidence/pdf_quality.json").write_text(json.dumps(summary, indent=2) + "\n")
(ROOT / "evidence/report_extracted.txt").write_text(text)
selected = [0, 1, 3, 6, 10, 14, 18, 22, 26, len(pdf) - 1]
mosaic = fitz.open()
canvas = mosaic.new_page(width=595.276 * 2, height=841.89 * 5)
for slot, page_number in enumerate(selected):
    column, row = slot % 2, slot // 2
    rect = fitz.Rect(column * 595.276, row * 841.89, (column + 1) * 595.276, (row + 1) * 841.89)
    canvas.show_pdf_page(rect, pdf, page_number)
canvas.get_pixmap(matrix=fitz.Matrix(.75, .75)).save(work / "selected-pages.png")
for page_number in [0, 6, 14, 26]:
    pdf[page_number].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(work / f"page-{page_number + 1}.png")
print(json.dumps({"pages": len(pdf), "outside_margin_word_count": len(outside),
                  "score_check": summary["corrected_scores_present"], "preview_directory": str(work)}, indent=2))
for page in pages:
    print(f"Page {page['page']:2d}: {' / '.join(page['first_lines'][2:5])}")

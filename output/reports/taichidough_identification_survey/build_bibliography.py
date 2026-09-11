"""Build the survey bibliography from reviewed primary-source records.

The topic evidence files document technical access and claim verification.
No network requests are made by this build.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
catalog = json.loads((ROOT / "reference_catalog.json").read_text())
access = json.loads((ROOT / "evidence/source_access.json").read_text())
catalog = [{**access.get(source["id"], {}), **source} for source in catalog]
records = []
for source in catalog:
    authors = []
    if source.get("organisation"):
        authors = [{"literal": source["organisation"]}]
    else:
        for name in source["authors"].split("; "):
            family, given = name.split(", ", 1)
            authors.append({"family": family, "given": given})
    record = {
        "id": source["id"],
        "type": source.get("type", "article-journal"),
        "title": source["title"],
        "author": authors,
        "issued": {"date-parts": [[source["year"]]]},
        "URL": source.get("url") or "https://doi.org/" + source["doi"],
    }
    for key in ("volume", "issue", "page", "number"):
        if source.get(key):
            record[key] = source[key]
    if source.get("venue"):
        record["container-title"] = source["venue"]
    if source.get("doi"):
        record["DOI"] = source["doi"]
    if record["type"] == "webpage":
        record["accessed"] = {"date-parts": [[2026, 9, 11]]}
        del record["issued"]
    records.append(record)
if len({r["id"] for r in records}) != len(records):
    raise ValueError("Bibliography identifiers must be unique")
(ROOT / "references.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
with (ROOT / "reading_catalog.csv").open("w", newline="") as stream:
    fields = ["id", "year", "title", "authors", "venue", "doi", "url", "read_url", "evidence"]
    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
    writer.writeheader()
    for source in catalog:
        writer.writerow({
            **source,
            "authors": source.get("authors", source.get("organisation", "")),
            "url": source.get("url") or "https://doi.org/" + source["doi"],
        })
print(f"Built {len(records)} bibliography records")

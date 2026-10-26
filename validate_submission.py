"""Validate compact Nature submission and rebuild SHA-256 file manifest."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
required = ["README.md", "requirements.txt", "SUBMISSION_CHECKLIST.md", "RUN_SUMMARY.md",
            "demo/run_demo.py", "demo/data/association_example.csv", "demo/output/demo_run.json"]
for rel in required:
    assert (ROOT / rel).is_file(), f"Missing {rel}"

with (ROOT / "demo/data/association_example.csv").open(encoding="utf-8-sig", newline="") as f:
    records = list(csv.DictReader(f))
assert records and all(r["B_county_id"].startswith("SYN") for r in records)
result = json.loads((ROOT / "demo/output/demo_run.json").read_text(encoding="utf-8"))
assert result["status"] == "PASS"

source_files = [p for p in (ROOT / "source").rglob("*") if p.is_file() and p.suffix.lower() in {".py", ".r"}]
assert len(source_files) >= 50

manifest = []
for file in sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "FILE_MANIFEST.json"):
    data = file.read_bytes()
    manifest.append({"path": file.relative_to(ROOT).as_posix(), "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()})
(ROOT / "FILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"status": "PASS", "source_code_files": len(source_files),
                  "manifest_files": len(manifest), "uncompressed_bytes": sum(x["bytes"] for x in manifest)}, indent=2))

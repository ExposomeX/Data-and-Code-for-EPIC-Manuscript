"""Dependency-free smoke test of all four public data interfaces.

This checks schemas and computes descriptive example outputs. It does not
retrain XGBoost or fit the restricted-data hospitalization models.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

def read(name):
    with (HERE / "data" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def write(name, fields, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def numeric(row, key):
    try:
        value = float(row[key])
        return value if math.isfinite(value) else 0.0
    except (ValueError, TypeError, KeyError):
        return 0.0

def main():
    t0 = time.perf_counter()
    emission = read("emission_example.csv")
    assert emission and {"Year", "lat", "lon", "As_CK"} <= emission[0].keys()
    e = defaultdict(lambda: [0, 0.0])
    for row in emission:
        key = row["Year"]
        e[key][0] += 1
        e[key][1] += numeric(row, "As_CK")
    write("emission_example_summary.csv", ["year", "grid_rows", "As_CK_sum_source_units"],
          [{"year": k, "grid_rows": v[0], "As_CK_sum_source_units": v[1]} for k, v in sorted(e.items())])

    assoc = read("association_example.csv")
    assert assoc and {"B_county_id", "B_year", "C_POP", "Y_Circulation", "X_As_CK"} <= assoc[0].keys()
    assert all(row["B_county_id"].startswith("SYN") for row in assoc), "demo must use synthetic counties"
    a = defaultdict(lambda: [0, 0.0, 0.0])
    for row in assoc:
        item = a[row["B_year"]]
        item[0] += 1
        item[1] += numeric(row, "C_POP")
        item[2] += numeric(row, "Y_Circulation")
    write("association_demo_qc.csv", ["year", "records", "population_sum", "circulation_count_sum"],
          [{"year": k, "records": v[0], "population_sum": v[1], "circulation_count_sum": v[2]} for k, v in sorted(a.items())])

    priority = read("priority_example.csv")
    assert priority and {"Disease_Name", "Exposure", "LogRR", "P_value", "PDS_EPPD", "PDS_EGoD"} <= priority[0].keys()
    p = defaultdict(list)
    for row in priority:
        # An illustrative epidemiological score; the complete biological
        # normalization and selection are in source/.../code_3_select_pols.py.
        score = max(0.0, numeric(row, "LogRR")) * -math.log10(max(numeric(row, "P_value"), 1e-300))
        p[row["Disease_Name"]].append((score, row["Exposure"]))
    write("priority_demo_epi_screen.csv", ["disease", "example_top_exposure", "epi_score"],
          [{"disease": k, "example_top_exposure": v[0][1], "epi_score": v[0][0]}
           for k, v in sorted((k, sorted(v, reverse=True)) for k, v in p.items())])

    future = read("future_example.csv")
    assert future and {"scenario", "lat", "lon", "time"} <= future[0].keys()
    f = defaultdict(int)
    for row in future:
        f[row["scenario"]] += 1
    write("future_example_summary.csv", ["scenario", "sample_grid_rows"],
          [{"scenario": k, "sample_grid_rows": v} for k, v in sorted(f.items())])
    report = {"status": "PASS", "elapsed_seconds": round(time.perf_counter() - t0, 3),
              "input_rows": {"emission": len(emission), "association_synthetic": len(assoc),
                             "priority": len(priority), "future": len(future)},
              "scope": "Interface and descriptive smoke test; no model refitting"}
    (OUT / "demo_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

"""Optional: regenerate compact examples from the author's full local outputs."""
from __future__ import annotations
import csv
import argparse
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"

def stratified(source, target, key, per_group=12):
    counts = {}
    with source.open(encoding="utf-8-sig", newline="") as inp, target.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            group = row[key]
            if counts.get(group, 0) < per_group:
                writer.writerow(row)
                counts[group] = counts.get(group, 0) + 1
    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate stratified demo rows from full local outputs")
    parser.add_argument("--source-root", type=Path, required=True)
    ROOT = parser.parse_args().source_root
    a = ROOT / "Supplementary Code_3_Create_Emission_Index/create_index/code_1_tot_results/3_combined_2_pds_data.csv"
    b = ROOT / "Supplementary Code_4_Pred_Future_Emission/code_5_results/pred_results.csv"
    print("priority", stratified(a, OUT / "priority_example.csv", "Disease_Name", 12))
    print("future", stratified(b, OUT / "future_example.csv", "scenario", 12))

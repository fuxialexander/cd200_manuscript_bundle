#!/usr/bin/env python3
"""Reconstruct current default phase membership; does not prove historical training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def partition(rows):
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    order = np.random.RandomState(42).permutation(len(rows))
    nt, nv = int(len(rows) * .75), int(len(rows) * .15)
    return {name: {rows[i][:3] for i in indices} for name, indices in
            [("train", order[:nt]), ("val", order[nt:nt + nv]), ("test", order[nt + nv:])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("training/data/scc_all_peaks.deduplicated.bed"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.source.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "Diff_DZ_vs_LZ" in reader.fieldnames:
            parser.error("This reconstruction is only for the observed missing Diff_DZ_vs_LZ fallback")
        rows = [(r["Chromosome"], int(r["Start"]), int(r["End"]), r["peak_type"]) for r in reader]
    if len(rows) > 500000:
        parser.error("This reconstruction assumes the default region cap does not truncate input")
    p1rows = [r for r in rows if r[3] == "background"]
    p2rows = [r for r in rows if r[3] in {"for_count", "for_profile"}]
    p1, p2, p3 = partition(p1rows), partition(p2rows), partition(rows)
    result = {
        "status": "RECONSTRUCTION_OF_CURRENT_SOURCE_DEFAULT_MEMBERSHIP_NOT_HISTORICAL_TRAINING_PROOF",
        "source": str(args.source),
        "assumptions": [
            "Seed 42; ratios .75/.15/.10; cap500000 does not truncate 184167 source rows",
            "Current phase3 differential filter falls back to all rows because Diff_DZ_vs_LZ is absent",
            "Measure exact source interval identity before curriculum; no model fit performed",
            "Phase1 optional: default cleaned launcher runs phase1_epochs=0; phase1 overlaps apply only if phase1 is trained on these rows"
        ],
        "phase_source_rows": {"phase1": len(p1rows), "phase2": len(p2rows), "phase3": len(rows)},
        "comparisons": [],
    }
    for name in ("val", "test"):
        for previous_name, previous in (("phase1_optional", p1), ("phase2", p2)):
            overlap, total = len(p3[name] & previous["train"]), len(p3[name])
            result["comparisons"].append({
                "prior_phase": previous_name, "later_phase": "phase3", "later_split": name,
                "later_split_unique_intervals": total, "identical_intervals_in_prior_training": overlap,
                "fraction": overlap / total,
            })
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()

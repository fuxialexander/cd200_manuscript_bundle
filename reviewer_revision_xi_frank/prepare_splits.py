#!/usr/bin/env python3
"""Prepare provisional chromosome holdout manifests; never train/evaluate a model.

Run from the repository root. Coordinates are 0-based half-open. A manifest
records unique model input windows and source line references, not target values.
The current training loaders DO NOT consume these manifests automatically.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


AUTOSOMES = {f"chr{i}" for i in range(1, 20)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_windows(path, half_width):
    windows = {}
    excluded = Counter()
    input_rows = 0
    with Path(path).open(newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not row or row[0].startswith("#"):
                continue
            if line_no == 1 and row[0] == "Chromosome":
                continue
            if len(row) < 3:
                raise ValueError(f"{path}:{line_no}: fewer than three fields")
            chrom = row[0]
            start, end = int(row[1]), int(row[2])
            if start < 0 or end <= start:
                raise ValueError(f"{path}:{line_no}: invalid source interval")
            input_rows += 1
            if chrom not in AUTOSOMES:
                excluded[chrom] += 1
                continue
            center = (start + end) // 2
            window = (chrom, center - half_width, center + half_width)
            if window[1] < 0:
                raise ValueError(f"{path}:{line_no}: input window crosses chromosome start")
            windows.setdefault(window, []).append(line_no)
    ordered = sorted(windows, key=lambda w: (int(w[0][3:]), w[1], w[2]))
    return windows, ordered, input_rows, dict(sorted(excluded.items()))


def assign_split(chrom, val_chroms, test_chroms):
    return "test" if chrom in test_chroms else "val" if chrom in val_chroms else "train"


def component_ids(ordered):
    """Put directly or transitively overlapping input windows in one block."""
    result = {}
    members = []
    current_chrom = None
    left = right = None

    def finish():
        if members:
            block = f"{current_chrom}:{left}-{right}"
            for member in members:
                result[member] = block

    for window in ordered:
        chrom, start, end = window
        if chrom != current_chrom or start >= right:
            finish()
            members = []
            current_chrom, left, right = chrom, start, end
        else:
            right = max(right, end)
        members.append(window)
    finish()
    return result


def prepare(source, output, val_chroms, test_chroms, half_width):
    windows, ordered, input_rows, excluded = load_windows(source, half_width)
    chroms = {window[0] for window in ordered}
    missing = (val_chroms | test_chroms) - chroms
    if missing:
        raise ValueError(f"{source}: holdout chromosomes absent: {sorted(missing)}")
    blocks = component_ids(ordered)
    counts = Counter()
    chrom_counts = Counter()
    block_sets = {split: set() for split in ("train", "val", "test")}
    # Use exclusive creation so existing review artifacts cannot be overwritten.
    with Path(output).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, delimiter="\t", lineterminator="\n")
                writer.writerow(["chrom", "input_start", "input_end", "split", "overlap_block", "source_lines"])
                for window in ordered:
                    split = assign_split(window[0], val_chroms, test_chroms)
                    counts[split] += 1
                    chrom_counts[window[0]] += 1
                    block_sets[split].add(blocks[window])
                    writer.writerow([*window, split, blocks[window], ",".join(map(str, windows[window]))])
    present = {split: {w[0] for w in ordered if assign_split(w[0], val_chroms, test_chroms) == split}
               for split in ("train", "val", "test")}
    assert not (present["train"] & present["val"] | present["train"] & present["test"] | present["val"] & present["test"])
    return {
        "source": str(source), "source_sha256": digest(source),
        "manifest": str(output), "manifest_sha256": digest(output),
        "source_rows": input_rows, "excluded_non_autosomal_rows": excluded,
        "unique_input_windows": len(ordered),
        "collapsed_repeated_input_windows": sum(len(v) - 1 for v in windows.values()),
        "split_unique_window_counts": dict(sorted(counts.items())),
        "chromosome_unique_window_counts": dict(chrom_counts),
        "split_overlap_block_counts": {s: len(v) for s, v in block_sets.items()},
        "cross_split_input_window_overlap_count": 0,
        "cross_split_overlap_basis": "Disjoint chromosome sets; all overlapping windows share chromosome and split.",
        "chromosome_end_bounds_checked": False,
        "metadata_conflicts_resolved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-bed", type=Path, default=Path("training/data/total.bed"))
    parser.add_argument("--phase3-bed", type=Path, default=Path("training/data/scc_all_peaks.deduplicated.bed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reviewer_revision_xi_frank/split_audit"))
    parser.add_argument("--val-chroms", nargs="+", default=["chr14"])
    parser.add_argument("--test-chroms", nargs="+", default=["chr16", "chr8", "chr9"])
    parser.add_argument("--input-half-width", type=int, default=2048)
    args = parser.parse_args()
    val, test = set(args.val_chroms), set(args.test_chroms)
    if not val or not test or val & test or not (val | test) <= AUTOSOMES:
        parser.error("Use nonempty, disjoint mm10 autosomal validation and test chromosome sets")
    if args.input_half_width <= 0:
        parser.error("Input half-width must be positive")
    outputs = [args.output_dir / f"{name}.windows.tsv.gz" for name in ("regression", "phase3")]
    audit_path = args.output_dir / "audit.json"
    if any(p.exists() for p in [*outputs, audit_path]):
        parser.error("Output already exists; choose a new output directory to preserve artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "PROVISIONAL_SPLIT_DESIGN_NOT_MODEL_EVALUATION",
        "genome": "mm10", "coordinate_convention": "0-based half-open",
        "input_length": 2 * args.input_half_width,
        "val_chromosomes": sorted(val), "test_chromosomes": sorted(test),
        "selection": "Proposed before new benchmark; chromosome choices require author review and are not historical checkpoint provenance.",
        "curriculum": "None applied. Fit curriculum groups and sample only after extracting training rows.",
        "integration": "Training loaders require explicit manifest integration; passing comma-separated chromosome names to current scalar CLI is invalid.",
        "datasets": [],
    }
    for source, output in zip((args.regression_bed, args.phase3_bed), outputs):
        audit["datasets"].append(prepare(source, output, val, test, args.input_half_width))
    with audit_path.open("x") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"audit": str(audit_path), "datasets": [{"source": d["source"], "counts": d["split_unique_window_counts"]} for d in audit["datasets"]]}, indent=2))


if __name__ == "__main__":
    main()

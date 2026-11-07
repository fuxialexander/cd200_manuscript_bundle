#!/usr/bin/env python3
"""Cache released GM12878 fold-0 regions with shared raw insertion targets.

Uses official post-filtered peak/nonpeak splits, never signal-based resampling or
test-dependent fitting. Sparse bigWig omitted positions represent zero insertion
counts as in ChromBPNet's native data loader; IO errors never become dummy rows.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyBigWig
from pyfaidx import Fasta


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_regions(folder, split, allowed):
    source_split = {"train": "trainingset", "valid": "validationset", "test": "testset"}[split]
    records, sources = [], []
    for category in ("peaks", "nonpeaks"):
        matches = list(folder.glob(f"{category}.{source_split}.fold_0.*.bed.gz"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {category} {split} source: {matches}")
        path = matches[0]
        sources.append({"path": str(path), "sha256": digest(path)})
        with gzip.open(path, "rt") as handle:
            for line_no, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
                chrom, start, summit = fields[0], int(fields[1]), int(fields[9])
                if chrom not in allowed or summit < 0:
                    raise ValueError(f"Unexpected chromosome/summit {path}:{line_no}")
                center = start + summit
                peak = int(category == "peaks")
                records.append({"region_id": f"{split}:{category}:{line_no}:{chrom}:{center}", "chrom": chrom, "center": center, "is_peak": peak,
                                "source_file": path.name, "source_line": line_no, "input_start": center - 1057, "input_end": center + 1057,
                                "target_start": center - 500, "target_end": center + 500})
    records.sort(key=lambda row: (row["chrom"], row["center"], -row["is_peak"], row["source_line"]))
    return records, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--bigwig", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-bigwig-md5", default="ccd7670dd53f297cc670ea0637a357d7")
    p.add_argument("--experiment", default="ENCSR637XSC")
    p.add_argument("--annotation", default="ENCSR389HIH")
    p.add_argument("--signal-accession", default="ENCFF180XQC")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists; preserve prior cache and choose another path")
    actual_md5 = digest(args.bigwig, "md5")
    if actual_md5 != args.expected_bigwig_md5:
        p.error(f"Observed bigWig checksum mismatch: {actual_md5}")
    fold_file = args.regions / "cv_params.fold_0.json"
    fold = json.loads(fold_file.read_text())
    if fold["test"] != ["chr1", "chr3", "chr6"] or fold["valid"] != ["chr8", "chr20"]:
        p.error("Unexpected released fold 0")
    sets = [set(fold[key]) for key in ("train", "valid", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        p.error("Chromosome partitions overlap")
    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True, rebuild=False)
    bw = pyBigWig.open(str(args.bigwig))
    lengths = bw.chroms()
    for chrom in set.union(*sets):
        if chrom not in fasta or chrom not in lengths or len(fasta[chrom]) != lengths[chrom]:
            p.error(f"Reference length mismatch for {chrom}")
    args.output.mkdir(parents=True)
    lut = np.full(256, 4, dtype=np.uint8)
    for i, base in enumerate(b"ACGT"):
        lut[base] = i
    report = {"status": "cache_generation_in_progress", "assembly": "GRCh38", "dataset": args.experiment, "model_annotation": args.annotation, "fold": fold,
              "input_length": 2114, "output_length": 1000, "cache_sequence_length": 3114, "cache_target_length": 2000,
              "cache_target_sequence_offset": 557, "evaluation_cache_offset": 500,
              "missing_signal_policy": "bigWig omitted positions converted from NaN to zero as native insertion loader; infinity/negative/IO failures rejected",
              "cache_is_filtered_by_target": False, "source_region_filtering": "Official released post-filtered training/validation/test BEDs; no additional target filtering",
              "reference": {"path": str(args.fasta), "fai_sha256": digest(str(args.fasta) + ".fai"), "sha256": digest(args.fasta)},
              "observed_signal": {"path": str(args.bigwig), "accession": args.signal_accession, "md5": actual_md5, "sha256": digest(args.bigwig)},
              "fold_file": {"path": str(fold_file), "sha256": digest(fold_file)}, "splits": {}}
    invalid = []
    all_identifiers = set()
    for split in ("train", "valid", "test"):
        started = time.monotonic()
        records, sources = read_regions(args.regions, split, set(fold[split]))
        good = []
        for row in records:
            center, chrom = row["center"], row["chrom"]
            if center - 1557 < 0 or center + 1557 > lengths[chrom]:
                invalid.append({"region_id": row["region_id"], "reason": "3114bp_jitter_context_crosses_chromosome_boundary"})
            else:
                good.append(row)
        records = good
        x = np.lib.format.open_memmap(args.output / f"{split}.x.npy", mode="w+", dtype=np.uint8, shape=(len(records), 3114))
        y = np.lib.format.open_memmap(args.output / f"{split}.y.npy", mode="w+", dtype=np.float32, shape=(len(records), 2000))
        omitted, fractional, maxima = 0, 0, 0.0
        for i, row in enumerate(records):
            center, chrom = row["center"], row["chrom"]
            seq = str(fasta[chrom][center - 1557:center + 1557])
            if len(seq) != 3114:
                raise ValueError(f"Truncated FASTA extraction {row}")
            x[i] = lut[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
            vals = np.asarray(bw.values(chrom, center - 1000, center + 1000), dtype=np.float32)
            omitted += int(np.isnan(vals).sum())
            if np.isinf(vals).any() or np.any(vals < 0):
                raise ValueError(f"Nonfinite/negative insertion signal at {row}")
            vals = np.nan_to_num(vals, nan=0.0)
            fractional += int(np.count_nonzero(vals != np.round(vals)))
            maxima = max(maxima, float(vals.max()))
            y[i] = vals
            if (i + 1) % 20000 == 0:
                print(f"{split} {i+1}/{len(records)} {time.monotonic()-started:.1f}s", flush=True)
        x.flush(); y.flush()
        columns = list(records[0])
        path = args.output / f"{split}.regions.tsv"
        with path.open("x") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(records)
        ids = "".join(row["region_id"] + "\n" for row in records)
        identities = {(row["chrom"], row["center"]) for row in records}
        if identities & all_identifiers:
            raise ValueError("Same centered interval in more than one split")
        all_identifiers |= identities
        report["splits"][split] = {"regions": len(records), "peaks": sum(row["is_peak"] for row in records), "chromosome_counts": dict(Counter(row["chrom"] for row in records)),
                                     "sources": sources, "region_ids_sha256": hashlib.sha256(ids.encode()).hexdigest(), "regions_sha256": digest(path),
                                     "omitted_bigwig_bases_set_to_zero": omitted, "fractional_signal_bases": fractional, "maximum_insertion_signal": maxima,
                                     "x_sha256": digest(args.output / f"{split}.x.npy"), "y_sha256": digest(args.output / f"{split}.y.npy"), "elapsed_seconds": time.monotonic() - started}
        print(f"{split} finished {len(records)} rows", flush=True)
        (args.output / "cache_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        del x, y
    report["status"] = "complete"
    report["excluded_boundary_regions"] = invalid
    report["cross_split_chromosomes_overlap"] = False
    (args.output / "cache_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    bw.close(); fasta.close()
    print("Cache complete", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare shared mouse chromosome-heldout caches from unsmoothed stored ATAC.

No normalization, convolution, signal threshold or test-outcome filter is applied.
Upstream insertion/Tn5-shift provenance is unverified. All model fits must exclude
chr16/chr8/chr9 test and chr14 validation, including learned bias components.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numcodecs
import numpy as np
from pyfaidx import Fasta


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TARGET_UNITS = "unsmoothed_per_base_ATAC_counts_from_bundled_Zarr"
TARGET_KIND = "stored_atac_signal_counts"
AUTOSOMES = {f"chr{i}" for i in range(1, 20)}
TEST = {"chr16", "chr8", "chr9"}
VALID = {"chr14"}
LUT = np.full(256, 4, np.uint8)
for i, base in enumerate(b"ACGT"):
    LUT[base] = i


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


class DenseZarrReader:
    """Read the inspected v2 Blosc arrays directly with a bounded chunk cache.

    This avoids whole-store discovery and records every accessed metadata/chunk
    checksum. Absent chunks use the array's declared fill value as Zarr specifies.
    """
    def __init__(self, root, expected_dtype, columns=1, max_cached_chunks=8):
        self.root, self.dtype, self.columns = Path(root), np.dtype(expected_dtype), columns
        self.max_cached = max_cached_chunks
        self.arrays, self.cache, self.sources = {}, OrderedDict(), {}

    def metadata(self, chrom):
        if chrom not in self.arrays:
            path = self.root / chrom / ".zarray"
            meta = json.loads(path.read_text())
            if meta["zarr_format"] != 2 or np.dtype(meta["dtype"]) != self.dtype or meta["order"] != "C" or meta.get("filters"):
                raise ValueError(f"Unsupported array representation: {path}")
            if len(meta["shape"]) != (1 if self.columns == 1 else 2) or (self.columns > 1 and meta["shape"][1] != self.columns):
                raise ValueError(f"Unexpected array shape: {path}")
            if meta["compressor"]["id"] != "blosc" or (self.columns > 1 and meta["chunks"][1] != self.columns):
                raise ValueError(f"Unsupported compression/chunk geometry: {path}")
            self.arrays[chrom] = meta
            self.sources[str(path)] = {"sha256": digest(path), "kind": "array_metadata"}
        return self.arrays[chrom]

    def chunk(self, chrom, index):
        key = (chrom, index)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        meta = self.metadata(chrom)
        name = str(index) if self.columns == 1 else f"{index}.0"
        path = self.root / chrom / name
        rows = meta["chunks"][0]
        if path.exists():
            compressed = path.read_bytes()
            raw = numcodecs.Blosc().decode(compressed)
            arr = np.frombuffer(raw, dtype=self.dtype)
            if arr.size != rows * self.columns:
                raise ValueError(f"Decoded chunk shape mismatch: {path}")
            arr = arr.reshape((rows,) if self.columns == 1 else (rows, self.columns))
            self.sources[str(path)] = {"sha256": hashlib.sha256(compressed).hexdigest(), "kind": "data_chunk"}
        else:
            fill = meta.get("fill_value")
            if fill is None:
                raise ValueError(f"Absent chunk without concrete fill value: {path}")
            arr = np.full((rows,) if self.columns == 1 else (rows, self.columns), fill, self.dtype)
            self.sources[str(path)] = {"sha256": None, "kind": "absent_chunk_declared_fill", "fill_value": fill}
        self.cache[key] = arr
        if len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
        return arr

    def read(self, chrom, start, end):
        meta = self.metadata(chrom)
        if start < 0 or end <= start or end > meta["shape"][0]:
            raise ValueError(f"Out-of-bounds read: {chrom}:{start}-{end}")
        width = meta["chunks"][0]
        chunks = []
        for index in range(start // width, (end - 1) // width + 1):
            a, b = max(start - index * width, 0), min(end - index * width, width)
            chunks.append(self.chunk(chrom, index)[a:b])
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else np.array(chunks[0], copy=True)


def split_for(chrom):
    if chrom not in AUTOSOMES:
        return None
    return "test" if chrom in TEST else "valid" if chrom in VALID else "train"


def source_regions(path):
    result, excluded, identities = [], Counter(), set()
    with Path(path).open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Chromosome", "Start", "End", "peak_type"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Source BED lacks required columns")
        for line, row in enumerate(reader, 2):
            chrom, a, b, kind = row["Chromosome"], int(row["Start"]), int(row["End"]), row["peak_type"]
            if a < 0 or b <= a or kind not in {"background", "for_count", "for_profile"}:
                raise ValueError(f"Invalid interval/category at source line {line}")
            split = split_for(chrom)
            if split is None:
                excluded[chrom] += 1
                continue
            center = (a + b) // 2
            key = (chrom, center)
            if key in identities:
                raise ValueError(f"Repeated input center at source line {line}: {key}")
            identities.add(key)
            result.append({"region_id": f"{chrom}:{center}", "chrom": chrom, "center": center,
                           "is_peak": int(kind != "background"), "peak_type": kind,
                           "source_line": line, "source_start": a, "source_end": b, "split": split})
    result.sort(key=lambda r: (int(r["chrom"][3:]), r["center"]))
    return result, dict(excluded)


def encode_sequence(seq):
    return LUT[np.frombuffer(str(seq).upper().encode("ascii"), dtype=np.uint8)]


def eligibility(chrom, center, chrom_length, sequence, max_n_fraction):
    if center - 1557 < 0 or center + 1557 > chrom_length:
        return "3114bp_context_outside_chromosome"
    if sequence is None or len(sequence) != 3114:
        raise ValueError("Truncated sequence extraction")
    return "N_fraction_exceeds_0.05_frozen_limit" if np.mean(sequence == 4) > max_n_fraction else None


def loci_records():
    # Reuse the evaluator's frozen seven windows and exact ordering.
    from evaluate_loci import planned_windows
    windows = planned_windows()
    records = []
    for w in windows:
        center = (w["output_start"] + w["output_end"]) // 2
        records.append({"region_id": w["window_id"], "chrom": w["chrom"], "center": center,
                        "is_peak": 1, "locus": w["locus"], "role": w["role"],
                        "output_start": w["output_start"], "output_end": w["output_end"], "split": "loci"})
    return records, windows


def cache_one_split(split, rows, fasta, signal, output, max_n, reject_path):
    good, rejected = [], []
    for row in rows:
        chrom, center = row["chrom"], row["center"]
        length = len(fasta[chrom])
        codes = None if center - 1557 < 0 or center + 1557 > length else encode_sequence(fasta[chrom][center - 1557:center + 1557])
        reason = eligibility(chrom, center, length, codes, max_n)
        if reason:
            rejected.append({"region_id": row["region_id"], "chrom": chrom, "center": center, "reason": reason})
        else:
            good.append(row)
    if split == "loci" and rejected:
        raise ValueError(f"A required Cd200 locus window failed frozen eligibility: {rejected}")
    if not good:
        raise ValueError(f"No eligible regions in {split}")
    x = np.lib.format.open_memmap(output / f"{split}.x.npy", mode="w+", dtype=np.uint8, shape=(len(good), 3114))
    y = np.lib.format.open_memmap(output / f"{split}.y.npy", mode="w+", dtype=np.float32, shape=(len(good), 2000))
    maximum, zeros = 0, 0
    for i, row in enumerate(good):
        chrom, center = row["chrom"], row["center"]
        x[i] = encode_sequence(fasta[chrom][center - 1557:center + 1557])
        values = signal.read(chrom, center - 1000, center + 1000)
        if values.shape != (2000,):
            raise ValueError("Signal read has wrong shape")
        y[i] = values
        maximum = max(maximum, int(values.max()))
        zeros += int(np.count_nonzero(values == 0))
        if (i + 1) % 20000 == 0:
            print(f"{split}: {i+1}/{len(good)}", flush=True)
    x.flush(); y.flush()
    table = output / f"{split}.regions.tsv"
    with table.open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(good[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(good)
    with reject_path.open("a") as handle:
        for row in rejected:
            handle.write(json.dumps({"split": split, **row}) + "\n")
    report = {"regions": len(good), "peaks": sum(r["is_peak"] for r in good),
              "nonpeaks": sum(not r["is_peak"] for r in good), "excluded": len(rejected),
              "excluded_reasons": dict(Counter(r["reason"] for r in rejected)),
              "chromosome_counts": dict(Counter(r["chrom"] for r in good)),
              "region_ids_sha256": hashlib.sha256("".join(r["region_id"] + "\n" for r in good).encode()).hexdigest(),
              "regions_sha256": digest(table), "x_sha256": digest(output / f"{split}.x.npy"),
              "y_sha256": digest(output / f"{split}.y.npy"), "maximum_stored_count": maximum,
              "zero_signal_bases": zeros, "signal_bases": len(good) * 2000}
    del x, y
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions", type=Path, default=REPO / "training/data/scc_all_peaks.deduplicated.bed")
    p.add_argument("--fasta", type=Path, default=Path("/mnt/storage/get_data/mm10.fa"))
    p.add_argument("--genome-zarr", type=Path, default=REPO / "data/zarr/mm10.zarr")
    p.add_argument("--signal-zarr", type=Path, default=REPO / "data/zarr/cd200_scc_dense_new.zarr")
    p.add_argument("--sample", default="SCCApos.cd200_scc")
    p.add_argument("--output", type=Path, default=Path("/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/cache"))
    p.add_argument("--report", type=Path, default=HERE / "cache_manifest.json")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; preserve previous cache and choose a fresh output path")
    if args.report.exists():
        p.error("Report exists; choose a fresh report path")
    for root in (args.genome_zarr, args.signal_zarr):
        if json.loads((root / ".zattrs").read_text()).get("assembly") != "mm10":
            p.error(f"Expected explicit mm10 assembly: {root}")
    rows, excluded_non_autosomal = source_regions(args.regions)
    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True, rebuild=False)
    signal = DenseZarrReader(args.signal_zarr / args.sample / "chrs", "<u2")
    genome = DenseZarrReader(args.genome_zarr / "chrs", "u1", columns=4, max_cached_chunks=2)
    for chrom in AUTOSOMES:
        if chrom not in fasta or len(fasta[chrom]) != signal.metadata(chrom)["shape"][0] or len(fasta[chrom]) != genome.metadata(chrom)["shape"][0]:
            p.error(f"FASTA/signal/genome Zarr chromosome length mismatch: {chrom}")
    # Verify actual sequence concordance at one source center per autosome and
    # each Cd200 window, not just chromosome lengths or assembly labels.
    locus_rows, windows = loci_records()
    checks, seen, check_rows = [], set(), []
    for row in rows:
        if row["chrom"] not in seen:
            seen.add(row["chrom"])
            check_rows.append(row)
    verification_rng = np.random.default_rng(1234)
    check_rows += [rows[int(i)] for i in verification_rng.choice(len(rows), size=min(50, len(rows)), replace=False)]
    check_rows += locus_rows
    for row in check_rows:
        chrom, center = row["chrom"], row["center"]
        if center - 1557 < 0 or center + 1557 > len(fasta[chrom]):
            continue
        codes = encode_sequence(fasta[chrom][center - 1557:center + 1557])
        onehot = np.zeros((3114, 4), np.uint8)
        valid = codes < 4
        onehot[np.flatnonzero(valid), codes[valid]] = 1
        equal = bool(np.array_equal(onehot, genome.read(chrom, center - 1557, center + 1557)))
        checks.append({"chrom": chrom, "center": center, "length": 3114, "identical": equal})
        if not equal:
            p.error(f"FASTA/genome Zarr sequence mismatch at {chrom}:{center}")
    args.output.mkdir(parents=True)
    reject_path = args.output / "excluded_regions.jsonl"
    reject_path.touch(exist_ok=False)
    report = {
        "status": "in_progress", "assembly": "mm10", "sample": args.sample,
        "target_kind": TARGET_KIND, "target_units": TARGET_UNITS,
        "upstream_provenance": "Stored ATAC signal counts; upstream insertion/Tn5-shift and replicate-merge provenance unverified. Not asserted to be confirmed raw shifted insertions.",
        "normalization": "none", "smoothing": "none", "test_outcome_filtering": False,
        "input_length": 2114, "output_length": 1000, "cache_sequence_length": 3114,
        "cache_target_length": 2000, "cache_target_sequence_offset": 557, "evaluation_cache_offset": 500,
        "count_transform": "ln1p_stored_atac_count", "frozen_masks": {
            "autosomes_only": True, "context_bounds": "entire3114bp", "exclude_N_fraction_greater_than": 0.05,
            "blacklist": "none supplied; no blacklist filter applied", "signal_threshold": None},
        "fold": {"train": sorted(AUTOSOMES - TEST - VALID), "valid": sorted(VALID), "test": sorted(TEST)},
        "excluded_non_autosomal_source_rows": excluded_non_autosomal,
        "reference": {"path": str(args.fasta), "sha256": digest(args.fasta), "fai_sha256": digest(str(args.fasta) + ".fai")},
        "source_regions": {"path": str(args.regions), "sha256": digest(args.regions)},
        "source_code_sha256": digest(__file__), "genome_sequence_concordance_checks": checks,
        "genome_verification_design": "first source center per autosome + 50 random source centers (seed1234, independent of signal) + seven Cd200 windows",
        "zarr_root_metadata": {str(path): {"sha256": digest(path), "content": json.loads(path.read_text())}
                               for path in [args.signal_zarr / ".zattrs", args.signal_zarr / args.sample / ".zattrs", args.genome_zarr / ".zattrs"]},
        "splits": {}, "cache_directory": str(args.output)}
    for split in ("train", "valid", "test", "loci"):
        started = time.monotonic()
        subset = locus_rows if split == "loci" else [r for r in rows if r["split"] == split]
        report["splits"][split] = cache_one_split(split, subset, fasta, signal, args.output, 0.05, reject_path)
        report["splits"][split]["elapsed_seconds"] = time.monotonic() - started
        write_json(args.report, report)
        print(f"{split}: complete {report['splits'][split]['regions']} rows", flush=True)
    window_path = args.output / "locus_windows.tsv"
    with window_path.open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(windows)
    observed = np.load(args.output / "loci.y.npy", mmap_mode="r")[:, 500:1500]
    np.save(args.output / "observed_profiles.npy", np.asarray(observed))
    report["stored_signal_locus_audit"] = [{
        "window_id": w["window_id"], "locus": w["locus"], "role": w["role"],
        "output_start": w["output_start"], "output_end": w["output_end"],
        "stored_signal_sum": float(values.sum(dtype=np.float64)), "nonzero_bases": int(np.count_nonzero(values)),
        "maximum_stored_count": int(values.max()), "all_values_integer": bool(np.all(values == np.round(values)))
    } for w, values in zip(windows, observed)]
    locus_ids = [w["window_id"] for w in windows]
    write_json(args.output / "target_metadata.json", {
        "genome_build": "mm10", "target_units": TARGET_UNITS, "target_kind": TARGET_KIND,
        "input_length": 2114, "output_length": 1000, "count_transform": "ln1p_stored_atac_count",
        "window_ids_sha256": hashlib.sha256("".join(i + "\n" for i in locus_ids).encode()).hexdigest(),
        "locus_windows_sha256": digest(window_path), "upstream_provenance": report["upstream_provenance"],
        "source_cache": str(args.output / "loci.y.npy")})
    chunk_log = args.output / "source_chunks.json"
    write_json(chunk_log, {"signal": signal.sources, "genome_sequence_checks": genome.sources})
    report["source_chunk_log"] = {"path": str(chunk_log), "sha256": digest(chunk_log),
                                  "signal_chunks_and_metadata": len(signal.sources), "genome_chunks_and_metadata": len(genome.sources)}
    report["excluded_regions_sha256"] = digest(reject_path)
    report["cross_split_chromosomes_overlap"] = False
    report["loci_are_descriptive_selected_examples"] = True
    report["status"] = "complete"
    write_json(args.report, report)
    write_json(args.output / "cache_manifest.json", report)
    fasta.close()
    print("Mouse cache complete", flush=True)


if __name__ == "__main__":
    main()

"""Convert local reference/signal files to the stores used by MoDL training.

Fragments follow the adjusted, deduplicated 10x format. Each retained row
contributes one event at start and one at end, matching Caesarion's fragment
converter. readSupport is not a molecule count; no further Tn5 shift is applied.
"""

from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_left
import csv
import gzip
import hashlib
from pathlib import Path
import re

import numpy as np
import zarr
from numcodecs import Blosc


def fingerprint(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def open_text(path: str | Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid chromosome/track name: {value!r}")
    return value


def blacklist_index(intervals):
    """Merge blacklist intervals once; window queries then take O(log n)."""
    merged = {}
    for chrom, start, end in sorted(intervals):
        rows = merged.setdefault(chrom, [])
        if rows and start <= rows[-1][1]:
            rows[-1] = rows[-1][0], max(end, rows[-1][1])
        else:
            rows.append((start, end))
    return {chrom: ([row[0] for row in rows], [row[1] for row in rows])
            for chrom, rows in merged.items()}


def overlaps_blacklist(index, chrom, start, end):
    starts, ends = index.get(chrom, ([], []))
    previous = bisect_left(starts, end) - 1
    return previous >= 0 and ends[previous] > start


def fasta_sizes(path: str | Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    chrom = None
    with open_text(path) as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                chrom = safe_name(line[1:].split()[0])
                if chrom in sizes:
                    raise ValueError(f"Duplicate FASTA record {chrom!r}")
                sizes[chrom] = 0
            elif chrom is None:
                raise ValueError(f"FASTA sequence before a header at line {number}")
            else:
                if not re.fullmatch(r"[ACGTRYSWKMBDHVNacgtryswkmbdhvn]+", line):
                    raise ValueError(f"Invalid FASTA bases at line {number}")
                sizes[chrom] += len(line)
    if not sizes or not all(sizes.values()):
        raise ValueError("FASTA must contain nonempty, uniquely named records")
    return sizes


def read_peaks(path: str | Path) -> list[tuple[str, int, int]]:
    peaks = []
    with open_text(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"BED needs at least three columns at line {number}")
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(f"Invalid BED coordinates at line {number}") from exc
            if start < 0 or end <= start:
                raise ValueError(f"Invalid BED interval at line {number}")
            peaks.append((fields[0], start, end))
    if not peaks:
        raise ValueError("BED contains no intervals")
    return peaks


def barcode_groups(path: str | Path) -> dict[str, str]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"barcode", "group"}.issubset(reader.fieldnames):
            raise ValueError("Barcode table requires tab-separated barcode and group headers")
        mapping = {}
        for number, row in enumerate(reader, 2):
            barcode, group = row["barcode"], safe_name(row["group"])
            if not barcode or barcode in mapping:
                raise ValueError(f"Empty or duplicate barcode at line {number}")
            mapping[barcode] = group
    if not mapping:
        raise ValueError("Barcode table contains no cells")
    return mapping


def create_store(path: Path, assembly: str):
    root = zarr.open_group(str(path), mode="w-", zarr_format=2)
    root.attrs["assembly"] = assembly
    return root


def create_array(group, name: str, shape: tuple, chunks: tuple, dtype: str):
    return group.create_array(
        name, shape=shape, chunks=chunks, dtype=dtype, fill_value=0,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )


def write_genome(path: Path, fasta: Path, sizes: dict, assembly: str, chunk_bp: int):
    root = create_store(path, assembly)
    group = root.create_group("chrs")
    arrays = {
        chrom: create_array(group, chrom, (length, 4), (chunk_bp, 4), "uint8")
        for chrom, length in sizes.items()
    }
    lookup = np.zeros((256, 4), dtype=np.uint8)
    for index, base in enumerate("ACGT"):
        lookup[ord(base), index] = lookup[ord(base.lower()), index] = 1
    chrom, offset, buffer = None, 0, ""

    def flush(text):
        nonlocal offset
        arrays[chrom][offset:offset + len(text)] = lookup[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
        offset += len(text)

    with open_text(fasta) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if buffer:
                    flush(buffer)
                chrom, offset, buffer = line[1:].split()[0], 0, ""
            else:
                buffer += line
                while len(buffer) >= chunk_bp:
                    flush(buffer[:chunk_bp])
                    buffer = buffer[chunk_bp:]
        if buffer:
            flush(buffer)


def fragment_signal(path: Path, source: Path, sizes: dict, assembly: str,
                    groups: dict[str, str] | None, track_id: str, chunk_bp: int) -> dict:
    root = create_store(path, assembly)
    tracks = sorted(set(groups.values())) if groups is not None else [safe_name(track_id)]
    root.attrs["celltype_ids"] = tracks
    arrays, totals = {}, dict.fromkeys(tracks, 0)
    for track in tracks:
        group = root.create_group(track).create_group("chrs")
        for chrom, length in sizes.items():
            arrays[track, chrom] = create_array(group, chrom, (length,), (chunk_bp,), "float32")

    # Bounded LRU avoids holding a whole chromosome (or all cell groups) in RAM.
    cache = OrderedDict()

    def flush_one():
        (track, chrom, chunk), values = cache.popitem(last=False)
        start = chunk * chunk_bp
        arrays[track, chrom][start:start + len(values)] = values

    def add(track, chrom, position):
        key = track, chrom, position // chunk_bp
        if key not in cache:
            if len(cache) >= 64:
                flush_one()
            start = key[2] * chunk_bp
            cache[key] = np.asarray(arrays[track, chrom][start:min(start + chunk_bp, sizes[chrom])])
        cache.move_to_end(key)
        cache[key][position % chunk_bp] += 1
        totals[track] += 1

    stats = {"rows": 0, "retained_fragments": 0, "unselected_barcodes": 0,
             "out_of_bounds_endpoints": 0, "right_endpoint": "end", "tn5_shift": "none",
             "read_support": "ignored; one molecule per deduplicated input row"}
    with open_text(source) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"10x fragments require five columns at line {number}")
            stats["rows"] += 1
            track = groups.get(fields[3]) if groups is not None else track_id
            if track is None:
                stats["unselected_barcodes"] += 1
                continue
            chrom = fields[0]
            try:
                start, end, support = int(fields[1]), int(fields[2]), int(fields[4])
            except ValueError as exc:
                raise ValueError(f"Invalid fragment values at line {number}") from exc
            if chrom not in sizes or start < 0 or end <= start or end > sizes[chrom] or support < 1:
                raise ValueError(f"Fragment outside the supplied FASTA or invalid at line {number}")
            stats["retained_fragments"] += 1
            for position in (start, end):
                if position < sizes[chrom]:
                    add(track, chrom, position)
                else:
                    stats["out_of_bounds_endpoints"] += 1
    while cache:
        flush_one()
    for track in tracks:
        if totals[track] == 0:
            raise ValueError(f"No retained insertion events for track {track!r}")
        root[track].attrs["library_size"] = totals[track]
    stats["library_sizes"] = totals
    return stats


def bigwig_signal(path: Path, source: Path, sizes: dict, assembly: str,
                  track_id: str, chunk_bp: int) -> dict:
    import pyBigWig

    root = create_store(path, assembly)
    root.attrs["celltype_ids"] = [safe_name(track_id)]
    track = root.create_group(track_id)
    group = track.create_group("chrs")
    total = 0.0
    with pyBigWig.open(str(source)) as bw:
        bw_sizes = bw.chroms()
        if not bw_sizes or any(chrom not in sizes or sizes[chrom] != length
                               for chrom, length in bw_sizes.items()):
            raise ValueError("bigWig chromosome names and lengths must agree with the supplied FASTA")
        for chrom, length in sizes.items():
            array = create_array(group, chrom, (length,), (chunk_bp,), "float32")
            if chrom not in bw_sizes:
                continue
            for start in range(0, length, chunk_bp):
                values = np.asarray(bw.values(chrom, start, min(start + chunk_bp, length), numpy=True))
                values = np.nan_to_num(values, nan=0, posinf=np.inf, neginf=-np.inf)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("bigWig must contain finite, nonnegative raw insertion counts")
                array[start:start + len(values)] = values
                total += float(values.sum(dtype=np.float64))
    if total <= 0:
        raise ValueError("bigWig has zero total signal")
    track.attrs["library_size"] = total
    return {"units": "raw insertion counts", "missing_values": "zero",
            "missing_chromosomes": sorted(set(sizes) - set(bw_sizes)),
            "library_sizes": {track_id: total}}


def prepare(args) -> dict:
    if args.context_bp < 32 or args.context_bp % 2 or args.chunk_bp < 1:
        raise ValueError("context-bp must be even and >=32; chunk-bp must be positive")
    sizes = fasta_sizes(args.fasta)
    peaks = read_peaks(args.peaks)
    for chrom, start, end in peaks:
        if chrom not in sizes or end > sizes[chrom]:
            raise ValueError(f"Peak {chrom}:{start}-{end} does not match the supplied FASTA")
    mapping = barcode_groups(args.barcode_groups) if args.barcode_groups else None
    if args.fragments and mapping is None and not args.pool_all:
        raise ValueError("Fragments need --barcode-groups, or explicit --pool-all")
    if args.bigwig and (mapping is not None or args.pool_all):
        raise ValueError("Barcode selection applies only to fragments")
    blacklist = read_peaks(args.blacklist) if args.blacklist else []
    for chrom, start, end in blacklist:
        if chrom not in sizes or end > sizes[chrom]:
            raise ValueError("Blacklist does not match the supplied FASTA")
    index = blacklist_index(blacklist)
    half = args.context_bp // 2
    retained = []
    excluded_boundary = excluded_blacklist = 0
    for chrom, start, end in sorted(set(peaks)):
        center = (start + end) // 2
        if center - half < 0 or center + half > sizes[chrom]:
            excluded_boundary += 1
        elif overlaps_blacklist(index, chrom, center - half, center + half):
            excluded_blacklist += 1
        else:
            retained.append((chrom, start, end))
    if not retained:
        raise ValueError("No peaks remain after context-boundary/blacklist filtering")

    args.out.mkdir(parents=True, exist_ok=False)
    args._created_output = True
    write_genome(args.out / "genome.zarr", args.fasta, sizes, args.assembly, args.chunk_bp)
    if args.fragments:
        signal = fragment_signal(args.out / "atac.zarr", args.fragments, sizes, args.assembly,
                                 mapping, args.track_id, args.chunk_bp)
    else:
        signal = bigwig_signal(args.out / "atac.zarr", args.bigwig, sizes, args.assembly,
                              args.track_id, args.chunk_bp)
    with (args.out / "peaks.tsv").open("x") as handle:
        handle.write("Chromosome\tStart\tEnd\n")
        for row in retained:
            handle.write("\t".join(map(str, row)) + "\n")
    inputs = {key: fingerprint(getattr(args, key))
              for key in ("fasta", "peaks", "fragments", "bigwig", "barcode_groups", "blacklist")
              if getattr(args, key)}
    return {"schema_version": 1, "assembly": args.assembly, "context_bp": args.context_bp,
            "chromosome_sizes": sizes, "tracks": list(signal["library_sizes"]),
            "inputs": inputs, "signal": signal, "peaks": {"input": len(peaks),
            "retained": len(retained), "duplicates_removed": len(peaks) - len(set(peaks)),
            "boundary_excluded": excluded_boundary, "blacklist_excluded": excluded_blacklist},
            "blacklist_scope": "Exclude full model windows; signal/library sizes retain all selected events"}

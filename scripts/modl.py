#!/usr/bin/env python3
"""Local MoDL workflow. stdout is JSON; training/inference logs stay in the run directory."""

from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def write_json(path: Path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for name in ("torch", "pytorch-lightning", "numpy", "zarr", "numcodecs", "pyBigWig", "caesar"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def code_state() -> dict:
    from atac_prediction.prepare_local_data import fingerprint

    paths = ["scripts/modl.py", "src/atac_prediction/prepare_local_data.py",
             "src/atac_prediction/train_atac.py", "src/atac_prediction/dna_atac_zarr_datamodule.py",
             "src/atac_prediction/dna_atac_zarr_dataset.py", "src/atac_prediction/dna_atac_lightning_module.py",
             "src/mutation_analysis/model.py", "src/checkpoint_manager.py",
             "model_studio/adapters/cd200.py", "model_studio/sequences.py", "model_studio/perturbations.py"]
    return {path: fingerprint(ROOT / path)["sha256"] for path in paths}


def doctor() -> dict:
    required = ["numpy", "zarr", "numcodecs", "torch", "pytorch_lightning", "caesar", "pyBigWig"]
    found = {name: importlib.util.find_spec(name) is not None for name in required}
    result = {"ok": all(found.values()), "dependencies": found, "versions": versions(),
              "bundle": str(ROOT), "python": sys.executable}
    if found["torch"]:
        import torch
        result["devices"] = {"cuda": torch.cuda.is_available(),
                             "mps": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                             "cpu": True}
    return result


def validate_data(data: Path, track_id: str | None, val_chr: str, test_chr: str) -> dict:
    import numpy as np
    import zarr

    data = data.expanduser().resolve()
    manifest = json.loads((data / "prepared.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported prepared-data schema")
    tracks = manifest["tracks"]
    if track_id is None and len(tracks) == 1:
        track_id = tracks[0]
    if track_id not in tracks:
        raise ValueError(f"Choose --track-id from {tracks}")
    if val_chr == test_chr:
        raise ValueError("Validation and test chromosomes must differ")
    genome = zarr.open_group(str(data / "genome.zarr"), mode="r")
    atac = zarr.open_group(str(data / "atac.zarr"), mode="r")
    for store in (genome, atac):
        if store.attrs.get("assembly") != manifest["assembly"]:
            raise ValueError("Store and manifest assemblies disagree")
    libsize = atac[track_id].attrs["library_size"]
    if not np.isfinite(libsize) or libsize <= 0:
        raise ValueError("Signal library size must be positive and finite")
    with (data / "peaks.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    counts, examples, seen = dict.fromkeys(("train", "val", "test"), 0), {}, set()
    for row in rows:
        chrom, start, end = row["Chromosome"], int(row["Start"]), int(row["End"])
        key = chrom, start, end
        if key in seen:
            raise ValueError("Prepared peaks contain duplicate intervals")
        seen.add(key)
        length = genome["chrs"][chrom].shape[0]
        center, half = (start + end) // 2, manifest["context_bp"] // 2
        if not 0 <= start < end <= length or center - half < 0 or center + half > length:
            raise ValueError(f"Invalid prepared model window for {key}")
        if atac[track_id]["chrs"][chrom].shape != (length,):
            raise ValueError(f"Signal and genome lengths differ for {chrom}")
        split = "val" if chrom == val_chr else "test" if chrom == test_chr else "train"
        counts[split] += 1
        examples.setdefault(split, (chrom, center - half, center + half))
    if not all(counts.values()):
        raise ValueError(f"Every chromosome split needs peaks: {counts}")
    for chrom, start, end in examples.values():
        dna = np.asarray(genome["chrs"][chrom][start:end])
        signal = np.asarray(atac[track_id]["chrs"][chrom][start:end])
        if dna.shape != (manifest["context_bp"], 4) or not np.isin(dna, [0, 1]).all() or (dna.sum(axis=1) > 1).any():
            raise ValueError("Invalid one-hot DNA in prepared data")
        if not np.isfinite(signal).all() or (signal < 0).any():
            raise ValueError("Signal must be finite and nonnegative")
    return {"ok": True, "data": str(data), "track_id": track_id,
            "assembly": manifest["assembly"], "context_bp": manifest["context_bp"],
            "splits": {"validation_chromosome": val_chr, "test_chromosome": test_chr, "counts": counts},
            "validation_scope": "All peak coordinates and shapes; one DNA/signal window per split"}


def train(args) -> dict:
    from atac_prediction.prepare_local_data import fingerprint

    checked = validate_data(args.data, args.track_id, args.val_chr, args.test_chr)
    if min(args.batch_size, args.max_epochs) < 1 or args.num_workers < 0:
        raise ValueError("Batch/epochs must be positive; workers must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not args.motifs.is_file():
        raise FileNotFoundError(args.motifs)
    command = [sys.executable, "-m", "atac_prediction.train_atac",
               "--atac-zarr-path", str(args.data / "atac.zarr"),
               "--genome-zarr-path", str(args.data / "genome.zarr"),
               "--summits-csv-path", str(args.data / "peaks.tsv"),
               "--celltype-ids", checked["track_id"], "--simple-motifs-path", str(args.motifs),
               "--extend-bp", str(checked["context_bp"] // 2), "--split-strategy", "chromosome",
               "--val-chr", args.val_chr, "--test-chr", args.test_chr,
               "--max-regions", str(sum(checked["splits"]["counts"].values())),
               "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
               "--max-epochs", str(args.max_epochs), "--learning-rate", str(args.learning_rate),
               "--precision", "32-true",
               "--run-name", args.out.name, "--seed", str(args.seed),
               "--wandb-mode", "disabled", "--no-compile", "--val-check-interval", "1.0",
               "--accelerator", {"cuda": "gpu"}.get(args.device, args.device),
               "--devices", "1", "--output-dir", str(args.out)]
    if args.smoke:
        command.append("--fast-dev-run")
    if args.resume_from:
        if not args.resume_from.is_file():
            raise FileNotFoundError(args.resume_from)
        command += ["--resume-from", str(args.resume_from)]
    plan = {**checked, "command": command, "shell_display": shlex.join(command),
            "cwd": str(ROOT), "out": str(args.out), "smoke_only": args.smoke,
            "network_logging": "disabled"}
    if args.dry_run:
        return {**plan, "dry_run": True}
    args.out.mkdir(parents=True, exist_ok=False)
    plan.update(status="running", versions=versions(),
                code_sha256=code_state(),
                inputs={"prepared": fingerprint(args.data / "prepared.json"), "motifs": fingerprint(args.motifs)})
    if args.resume_from:
        plan["inputs"]["resume_checkpoint"] = fingerprint(args.resume_from)
    write_json(args.out / "run.json", plan)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    env.update(WANDB_MODE="disabled", WANDB_SILENT="true", NUMBA_DISABLE_JIT="1", MPLBACKEND="Agg")
    try:
        with (args.out / "train.log").open("x") as handle:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        plan.update(returncode=process.returncode, status="complete" if process.returncode == 0 else "failed")
    except BaseException:
        plan["status"] = "interrupted"
        write_json(args.out / "run.json", plan)
        raise
    if process.returncode == 0:
        plan["summary"] = json.loads((args.out / "summary.json").read_text())
    write_json(args.out / "run.json", plan)
    return {"ok": process.returncode == 0, "out": str(args.out), "run": str(args.out / "run.json"),
            "log": str(args.out / "train.log"), "summary": plan.get("summary")}


def interpret(args) -> dict:
    from atac_prediction.prepare_local_data import fingerprint
    from model_studio.adapters.cd200 import CD200MotifCNNAdapter
    from model_studio.devices import DeviceManager
    from model_studio.domain import DeletionSettings
    from model_studio.perturbations import DeletionAnalyzer
    from model_studio.sequences import GenomeZarrSequenceSource

    manifest = json.loads((args.data / "prepared.json").read_text())
    chrom, start, end = GenomeZarrSequenceSource.parse_region(args.region)
    context = manifest["context_bp"]
    if end - start > context:
        raise ValueError(f"Requested region exceeds the {context}-bp model context")
    center = (start + end) // 2
    if center < context // 2:
        raise ValueError("Model context would extend before the start of the chromosome")
    record = GenomeZarrSequenceSource(
        region=f"{chrom}:{center - context // 2}-{center + context // 2}",
        genome_zarr=args.data / "genome.zarr").load()
    settings = DeletionSettings(deletion_size=args.deletion_size, stride=args.stride,
                               batch_size=args.batch_size, output_head=args.head, scan_scope="display")
    settings.validate(context)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    adapter = CD200MotifCNNAdapter(str(args.checkpoint), DeviceManager.resolve(args.device))
    inputs = {"checkpoint": fingerprint(args.checkpoint), "prepared": fingerprint(args.data / "prepared.json")}
    run = {"schema_version": 1, "status": "running", "region": args.region, "inputs": inputs,
           "versions": versions(), "code_sha256": code_state(),
           "coordinate_convention": "0-based, half-open"}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "run.json", run)
    try:
        with (args.out / "interpret.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            adapter.load()
            if getattr(adapter.model, "num_atac_samples", 1) != 1:
                raise ValueError("This workflow interprets a single-track checkpoint; use the legacy CLI for multi-output models")
            result = DeletionAnalyzer(adapter).run(record, settings, core_region=(start, end), display_region=(start, end))
        result.provenance["workflow_inputs"] = inputs
        write_json(args.out / "result.json", result.to_dict())
        with (args.out / "deletions.csv").open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["position", "deletion_effect", "plot_effect_pct"])
            writer.writerows(zip(result.nucleotide_positions, result.nucleotide_effects, result.plot_effects, strict=True))
        run.update(status="complete", artifacts=["result.json", "deletions.csv", "interpret.log"])
    except BaseException as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        adapter.close()
        write_json(args.out / "run.json", run)
    return {"ok": True, "out": str(args.out), "result": str(args.out / "result.json"),
            "csv": str(args.out / "deletions.csv")}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report local dependencies, versions and devices")
    prep = sub.add_parser("prepare", help="Convert FASTA + 10x fragments/raw bigWig + peaks to Zarr")
    prep.add_argument("--fasta", type=Path, required=True)
    prep.add_argument("--peaks", type=Path, required=True, help="Headerless BED/narrowPeak; first 3 columns are used")
    signal = prep.add_mutually_exclusive_group(required=True)
    signal.add_argument("--fragments", type=Path, help="Adjusted, deduplicated 10x fragments.tsv[.gz]")
    signal.add_argument("--bigwig", type=Path, help="Raw, nonnegative per-base insertion counts; not CPM/log/coverage")
    cells = prep.add_mutually_exclusive_group()
    cells.add_argument("--barcode-groups", type=Path, help="TSV with barcode and group headers; unlisted cells excluded")
    cells.add_argument("--pool-all", action="store_true", help="Explicitly pool every input barcode into one track")
    prep.add_argument("--track-id", default="sample", help="Track name for --pool-all or --bigwig")
    prep.add_argument("--assembly", required=True, help="Explicit reference assembly; never inferred")
    prep.add_argument("--blacklist", type=Path, help="BED; exclude overlapping full training windows")
    prep.add_argument("--context-bp", type=int, default=4096)
    prep.add_argument("--chunk-bp", type=int, default=65536)
    prep.add_argument("--out", type=Path, required=True, help="New directory; existing paths are refused")
    for name in ("validate", "train"):
        item = sub.add_parser(name, help="Check prepared data and disjoint chromosome splits" if name == "validate" else "Train one track with local logs")
        item.add_argument("--data", type=Path, required=True)
        item.add_argument("--track-id")
        item.add_argument("--val-chr", required=True)
        item.add_argument("--test-chr", required=True)
        if name == "train":
            item.add_argument("--out", type=Path, required=True)
            item.add_argument("--motifs", type=Path, default=ROOT / "artifacts/motifs_with_rc_aligned.pt")
            item.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
            item.add_argument("--max-epochs", type=int, default=100)
            item.add_argument("--learning-rate", type=float, default=1e-3)
            item.add_argument("--batch-size", type=int, default=8)
            item.add_argument("--num-workers", type=int, default=0)
            item.add_argument("--seed", type=int, default=42)
            item.add_argument("--resume-from", type=Path)
            item.add_argument("--dry-run", action="store_true", help="Validate and print exact argv; write nothing")
            item.add_argument("--smoke", action="store_true", help="One train/validation batch, followed by smoke.ckpt")
    item = sub.add_parser("interpret", help="Run batched sliding deletions using the existing analysis engine")
    item.add_argument("--data", type=Path, required=True)
    item.add_argument("--checkpoint", type=Path, required=True)
    item.add_argument("--region", required=True, help="0-based half-open chromosome:start-end")
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    item.add_argument("--head", choices=["regression", "count"], default="regression")
    item.add_argument("--deletion-size", type=int, default=15)
    item.add_argument("--stride", type=int, default=1)
    item.add_argument("--batch-size", type=int, default=4)
    return result


def main() -> int:
    args = parser().parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    try:
        if args.command == "doctor":
            output = doctor()
        elif args.command == "prepare":
            from atac_prediction.prepare_local_data import prepare
            output = prepare(args)
            output.update(created_at=datetime.now(timezone.utc).isoformat(), versions=versions(), code_sha256=code_state())
            write_json(args.out / "prepared.json", output)
            output = {"ok": True, "out": str(args.out), "tracks": output["tracks"], "peaks": output["peaks"]}
        elif args.command == "validate":
            output = validate_data(args.data, args.track_id, args.val_chr, args.test_chr)
        elif args.command == "train":
            output = train(args)
        else:
            output = interpret(args)
    except Exception as exc:
        output = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        if getattr(args, "_created_output", False):
            write_json(args.out / "prepare.failure.json", output)
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0 if output.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Smoke-test the CD200 manuscript bundle without running full inference."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "analysis/scripts/run_cd200_mutation.sh",
    "analysis/scripts/run_cd200_promoter_mutation.sh",
    "analysis/scripts/run_cd200_promoter_regression_mutation.sh",
    "analysis/scripts/run_cd200_promoter_regression_deg_mutation.sh",
    "analysis/checkpoints/shapescale_loss_scca_pos_with_count_head_fixed_motif_curriculum_from_scratch-epoch=105-val_loss=6.76.ckpt",
    "analysis/checkpoints/ATAC-Sequential-20250919-022148_phase3_final_model.ckpt",
    "analysis/fimo_mappings_cache.pkl",
    "artifacts/motifs_with_rc_aligned.pt",
    "artifacts/motifs_with_rc_aligned.txt",
    "artifacts/bias.meme",
    "artifacts/data/hocomoco/H13CORE_annotation.jsonl",
    "artifacts/data/hocomoco/H13CORE_pwms.txt",
    "configs/train_atac_phased.yaml",
    "configs/env.yaml",
    "training/data/total.bed",
    "training/data/scc_all_peaks.deduplicated.bed",
]

REQUIRED_DIRS = [
    "data/zarr/cd200_scc_dense_new.zarr",
    "data/zarr/mm10.zarr",
]


def run(command: list[str]) -> None:
    env = os.environ.copy()
    pythonpath = [str(ROOT / "src")]
    caesar_src = env.get("CAESAR_SRC")
    if not caesar_src:
        sibling = ROOT.parent / "Caesarion" / "src"
        if sibling.is_dir():
            caesar_src = str(sibling)
    if caesar_src:
        pythonpath.append(caesar_src)
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env["NUMBA_CACHE_DIR"] = env.get("NUMBA_CACHE_DIR", str(ROOT / ".numba-cache"))
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def check_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    missing_dirs = [path for path in REQUIRED_DIRS if not (ROOT / path).is_dir()]
    if missing or missing_dirs:
        msg = []
        if missing:
            msg.append("Missing required files:\n" + "\n".join(missing))
        if missing_dirs:
            msg.append("Missing required directories:\n" + "\n".join(missing_dirs))
        raise SystemExit("\n".join(msg))
    print(f"file check OK: {len(REQUIRED_FILES)} required files present")
    print(f"directory check OK: {len(REQUIRED_DIRS)} required directories present")


def check_checkpoints() -> None:
    checkpoints = {
        "regression": ROOT / "analysis/checkpoints/shapescale_loss_scca_pos_with_count_head_fixed_motif_curriculum_from_scratch-epoch=105-val_loss=6.76.ckpt",
        "phase3": ROOT / "analysis/checkpoints/ATAC-Sequential-20250919-022148_phase3_final_model.ckpt",
    }
    for label, path in checkpoints.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        has_differential = (
            "differential_head_state_dict" in checkpoint
            or str(checkpoint.get("phase", "")).lower() == "phase3"
        )
        print(f"checkpoint {label} OK: differential_head={has_differential}")


def check_imports_and_cli() -> None:
    import mutation_analysis  # noqa: F401
    import atac_prediction.train_atac  # noqa: F401
    import atac_diff_prediction.train_sequential_phased  # noqa: F401

    print("python import check OK")
    run([sys.executable, "analysis/mutation_analysis_cli.py", "--help"])
    run([sys.executable, "analysis/motif_scan_cli.py", "--help"])
    run([sys.executable, "analysis/plot_scan_results.py", "--help"])
    run([sys.executable, "-m", "atac_prediction.train_atac", "--help"])
    run([sys.executable, "-m", "atac_diff_prediction.train_sequential_phased", "--help"])
    print("CLI help check OK")


def main() -> None:
    check_files()
    check_checkpoints()
    check_imports_and_cli()
    print("bundle validation OK")


if __name__ == "__main__":
    main()

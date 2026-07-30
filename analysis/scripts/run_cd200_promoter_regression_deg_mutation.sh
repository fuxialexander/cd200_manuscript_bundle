#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/bundle_env.sh"

ATAC_ZARR_PATH="${ATAC_ZARR_PATH:-data/zarr/cd200_scc_dense_new.zarr}"
GENOME_ZARR_PATH="${GENOME_ZARR_PATH:-data/zarr/mm10.zarr}"
CENTER_WINDOW_BP="${CENTER_WINDOW_BP:-2000}"
PROMOTER_REGION="${PROMOTER_REGION:-chr16:45408000-45409426}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-analysis/checkpoints/shapescale_loss_scca_pos_with_count_head_fixed_motif_curriculum_from_scratch-epoch=105-val_loss=6.76.ckpt}"
CONFIG_FILE="${CONFIG_FILE:-configs/train_atac_phased.yaml}"
SIMPLE_MOTIFS_PATH="${SIMPLE_MOTIFS_PATH:-artifacts/motifs_with_rc_aligned.pt}"
MOTIF_NAMES_FILE="${MOTIF_NAMES_FILE:-artifacts/motifs_with_rc_aligned.txt}"
FIMO_CACHE="${FIMO_CACHE:-analysis/fimo_mappings_cache.pkl}"
DEG_FILE_1="${DEG_FILE_1:-analysis/rnaseq_deg_analysis/1F.deg.gene.csv}"
DEG_FILE_2="${DEG_FILE_2:-analysis/rnaseq_deg_analysis/AF.deg.gene.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/results/cd200_promoter_regression_deg_analysis}"
TSS_POSITION="${TSS_POSITION:-45409053}"
TSS_LABEL="${TSS_LABEL:-Cd200}"
MOTIF_PVALUE_CUTOFF="${MOTIF_PVALUE_CUTOFF:-1e-4}"
DEG_BASE_MEAN_THRESHOLD="${DEG_BASE_MEAN_THRESHOLD:-10}"
DEG_LOG2FC_THRESHOLD="${DEG_LOG2FC_THRESHOLD:-0}"
DEG_PADJ_THRESHOLD="${DEG_PADJ_THRESHOLD:-0.05}"

mkdir -p "${OUTPUT_DIR}"

for required_file in "${CHECKPOINT}" "${DEG_FILE_1}" "${DEG_FILE_2}"; do
    if [ ! -f "${required_file}" ]; then
        echo "Error: required file not found: ${required_file}"
        exit 1
    fi
done

HAS_DIFFERENTIAL_HEAD="$(CHECKPOINT_PATH="${CHECKPOINT}" "${PYTHON_CMD[@]}" - <<'PY'
import os
import torch

checkpoint = torch.load(os.environ["CHECKPOINT_PATH"], map_location="cpu")
has_differential = (
    "differential_head_state_dict" in checkpoint or
    str(checkpoint.get("phase", "")).lower() == "phase3"
)
print("true" if has_differential else "false")
PY
)"

if [ "${HAS_DIFFERENTIAL_HEAD}" = "true" ]; then
    echo "Error: ${CHECKPOINT} contains a differential head. Use run_cd200_promoter_mutation.sh for phase 3 differential analysis."
    exit 1
fi

IFS=':' read -r CHROM COORDS <<< "${PROMOTER_REGION}"
IFS='-' read -r START END <<< "${COORDS}"
REGION_SUFFIX="${CHROM}_${START}_${END}"

rm -f \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_deletion.png" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_count_nucleotide.csv" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_regression_nucleotide.csv" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_motifs_count.png" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_motifs_regression.png" \
  "${OUTPUT_DIR}/motif_scan_promoter_deg_${REGION_SUFFIX}.tsv" \
  "${OUTPUT_DIR}/combined_plot_highlighted_regression_promoter_deg_${REGION_SUFFIX}.png"

"${PYTHON_CMD[@]}" analysis/mutation_analysis_cli.py region "${PROMOTER_REGION}" \
  --checkpoint "${CHECKPOINT}" \
  --config-file "${CONFIG_FILE}" \
  --simple-motifs-path "${SIMPLE_MOTIFS_PATH}" \
  --genome-zarr "${GENOME_ZARR_PATH}" \
  --atac-zarr "${ATAC_ZARR_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --cell-types SCCApos.cd200_scc \
  --extend-bp 4096 \
  --deletion-size 15 \
  --stride 1 \
  --real-deletion \
  --distribute-effects \
  --smoothing-window 1 \
  --extract-motifs \
  --motif-names-file "${MOTIF_NAMES_FILE}" \
  --motif-threshold-sigma 4.0 \
  --save-baseline-pred \
  --save-observed-atac

REGRESSION_SCORES_FILE="${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_regression_nucleotide.csv"
if [ ! -f "${REGRESSION_SCORES_FILE}" ]; then
    echo "Error: regression scores file not found: ${REGRESSION_SCORES_FILE}"
    exit 1
fi

FULL_CORE_REGION="$(CHROM="${CHROM}" REGRESSION_SCORES_FILE="${REGRESSION_SCORES_FILE}" "${PYTHON_CMD[@]}" - <<'PY'
import csv
import os

scores_path = os.environ["REGRESSION_SCORES_FILE"]
chrom = os.environ["CHROM"]
with open(scores_path, newline="") as handle:
    positions = [int(row["position"]) for row in csv.DictReader(handle)]
if not positions:
    raise SystemExit("No positions found in regression scores CSV.")
print(f"{chrom}:{min(positions)}-{max(positions)}")
PY
)"

MOTIF_SCAN_OUTPUT="${OUTPUT_DIR}/motif_scan_promoter_deg_${REGION_SUFFIX}.tsv"
PLOT_OUTPUT_REGRESSION="${OUTPUT_DIR}/combined_plot_highlighted_regression_promoter_deg_${REGION_SUFFIX}.png"

"${PYTHON_CMD[@]}" analysis/motif_scan_cli.py \
  --scores-csv "${REGRESSION_SCORES_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --genome-zarr "${GENOME_ZARR_PATH}" \
  --motif-names "${MOTIF_NAMES_FILE}" \
  --fimo-cache "${FIMO_CACHE}" \
  --motif-filter "deg" \
  --deg-file-1 "${DEG_FILE_1}" \
  --deg-file-2 "${DEG_FILE_2}" \
  --deg-base-mean-threshold "${DEG_BASE_MEAN_THRESHOLD}" \
  --deg-log2fc-threshold "${DEG_LOG2FC_THRESHOLD}" \
  --deg-padj-threshold "${DEG_PADJ_THRESHOLD}" \
  --p-value-cutoff "${MOTIF_PVALUE_CUTOFF}" \
  --output-file "${MOTIF_SCAN_OUTPUT}"

CENTER_WINDOW_ARGS=()
if [ -n "${CENTER_WINDOW_BP}" ]; then
    CENTER_WINDOW_ARGS=(--center-window-bp "${CENTER_WINDOW_BP}")
fi

"${PYTHON_CMD[@]}" analysis/plot_scan_results.py \
  --motif-scan-file "${MOTIF_SCAN_OUTPUT}" \
  --deletion-scores-file "${REGRESSION_SCORES_FILE}" \
  --output-file "${PLOT_OUTPUT_REGRESSION}" \
  --dpi 300 \
  --highlight-regions \
  --merge-motifs \
  --plot-absolute-effect \
  --normalize-effect-to-max-predicted-atac \
  --motif-effect-threshold-pct 10 \
  --tss-highlight "${TSS_POSITION}" "${TSS_LABEL}" \
  --core-region "${FULL_CORE_REGION}" \
  "${CENTER_WINDOW_ARGS[@]}"

echo "=== Promoter regression DEG analysis complete ==="

#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/bundle_env.sh"

ATAC_ZARR_PATH="${ATAC_ZARR_PATH:-data/zarr/cd200_scc_dense_new.zarr}"
GENOME_ZARR_PATH="${GENOME_ZARR_PATH:-data/zarr/mm10.zarr}"
CENTER_WINDOW_BP="${CENTER_WINDOW_BP:-2000}"
PROMOTER_REGION="${PROMOTER_REGION:-chr16:45408000-45409426}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-analysis/checkpoints/ATAC-Sequential-20250919-022148_phase3_final_model.ckpt}"
REGRESSION_CHECKPOINT="${REGRESSION_CHECKPOINT:-analysis/checkpoints/shapescale_loss_scca_pos_with_count_head_fixed_motif_curriculum_from_scratch-epoch=105-val_loss=6.76.ckpt}"
CONFIG_FILE="${CONFIG_FILE:-configs/train_atac_phased.yaml}"
SIMPLE_MOTIFS_PATH="${SIMPLE_MOTIFS_PATH:-artifacts/motifs_with_rc_aligned.pt}"
MOTIF_NAMES_FILE="${MOTIF_NAMES_FILE:-artifacts/motifs_with_rc_aligned.txt}"
FIMO_CACHE="${FIMO_CACHE:-analysis/fimo_mappings_cache.pkl}"
RELAXED_TF_LIST="${RELAXED_TF_LIST:-analysis/rnaseq_deg_analysis/shared_DEGs_1F_AF_tf_relaxed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/results/cd200_promoter_analysis}"
PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-${OUTPUT_DIR}/regression_profile_overlay}"
TSS_POSITION="${TSS_POSITION:-45409053}"
TSS_LABEL="${TSS_LABEL:-Cd200}"

mkdir -p "${OUTPUT_DIR}" "${PROFILE_OUTPUT_DIR}"

for required_file in "${CHECKPOINT}" "${REGRESSION_CHECKPOINT}"; do
    if [ ! -f "${required_file}" ]; then
        echo "Error: required checkpoint not found: ${required_file}"
        exit 1
    fi
done

if [ ! -f "${RELAXED_TF_LIST}" ]; then
    echo "Relaxed TF list not found: ${RELAXED_TF_LIST}; rebuilding."
    "${PYTHON_CMD[@]}" analysis/rnaseq_deg_analysis/make_shared_deg_excel.py
    "${PYTHON_CMD[@]}" analysis/rnaseq_deg_analysis/rebuild_relaxed_tf.py
fi

if [ ! -f "${RELAXED_TF_LIST}" ]; then
    echo "Error: relaxed TF list is still missing: ${RELAXED_TF_LIST}"
    exit 1
fi

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

if [ "${HAS_DIFFERENTIAL_HEAD}" != "true" ]; then
    echo "Error: ${CHECKPOINT} does not contain a differential head."
    exit 1
fi

IFS=':' read -r CHROM COORDS <<< "${PROMOTER_REGION}"
IFS='-' read -r START END <<< "${COORDS}"
REGION_SUFFIX="${CHROM}_${START}_${END}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Regression overlay checkpoint: ${REGRESSION_CHECKPOINT}"
echo "Promoter region: ${PROMOTER_REGION}"
echo "Using relaxed TF list: ${RELAXED_TF_LIST}"

rm -f \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_deletion.png" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_count_nucleotide.csv" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_differential_heatmap.npy" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_differential_nucleotide.csv" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_regression_nucleotide.csv" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_motifs_count.png" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_motifs_differential.png" \
  "${OUTPUT_DIR}/${REGION_SUFFIX}_motifs_regression.png" \
  "${OUTPUT_DIR}/motif_scan_promoter_relaxed_${REGION_SUFFIX}.tsv" \
  "${OUTPUT_DIR}/combined_plot_highlighted_regression_promoter_relaxed_${REGION_SUFFIX}.png" \
  "${OUTPUT_DIR}/combined_plot_highlighted_differential_promoter_relaxed_${REGION_SUFFIX}.png"

rm -f \
  "${PROFILE_OUTPUT_DIR}/${REGION_SUFFIX}_deletion.png" \
  "${PROFILE_OUTPUT_DIR}/${REGION_SUFFIX}_deletion_count_nucleotide.csv" \
  "${PROFILE_OUTPUT_DIR}/${REGION_SUFFIX}_deletion_regression_nucleotide.csv"

echo "=== Running promoter deletion analysis ==="
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
  --save-observed-atac \
  --save-diff-matrix-path "${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_differential_heatmap.npy"

REGRESSION_SCORES_FILE="${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_regression_nucleotide.csv"
DIFFERENTIAL_SCORES_FILE="${OUTPUT_DIR}/${REGION_SUFFIX}_deletion_differential_nucleotide.csv"
for expected_file in "${REGRESSION_SCORES_FILE}" "${DIFFERENTIAL_SCORES_FILE}"; do
    if [ ! -f "${expected_file}" ]; then
        echo "Error: expected scores file not found: ${expected_file}"
        exit 1
    fi
done

echo "=== Running regression overlay prediction ==="
"${PYTHON_CMD[@]}" analysis/mutation_analysis_cli.py region "${PROMOTER_REGION}" \
  --checkpoint "${REGRESSION_CHECKPOINT}" \
  --config-file "${CONFIG_FILE}" \
  --simple-motifs-path "${SIMPLE_MOTIFS_PATH}" \
  --genome-zarr "${GENOME_ZARR_PATH}" \
  --atac-zarr "${ATAC_ZARR_PATH}" \
  --output-dir "${PROFILE_OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --cell-types SCCApos.cd200_scc \
  --extend-bp 4096 \
  --deletion-size 15 \
  --stride 1 \
  --real-deletion \
  --distribute-effects \
  --smoothing-window 1 \
  --save-baseline-pred \
  --save-observed-atac

PROFILE_SCORES_FILE="${PROFILE_OUTPUT_DIR}/${REGION_SUFFIX}_deletion_regression_nucleotide.csv"
if [ ! -f "${PROFILE_SCORES_FILE}" ]; then
    echo "Error: regression overlay scores file not found: ${PROFILE_SCORES_FILE}"
    exit 1
fi

MOTIF_SCAN_OUTPUT="${OUTPUT_DIR}/motif_scan_promoter_relaxed_${REGION_SUFFIX}.tsv"
PLOT_OUTPUT_REGRESSION="${OUTPUT_DIR}/combined_plot_highlighted_regression_promoter_relaxed_${REGION_SUFFIX}.png"
PLOT_OUTPUT_DIFFERENTIAL="${OUTPUT_DIR}/combined_plot_highlighted_differential_promoter_relaxed_${REGION_SUFFIX}.png"

echo "=== Scanning motifs with relaxed TF list ==="
"${PYTHON_CMD[@]}" analysis/motif_scan_cli.py \
  --scores-csv "${DIFFERENTIAL_SCORES_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --genome-zarr "${GENOME_ZARR_PATH}" \
  --motif-names "${MOTIF_NAMES_FILE}" \
  --fimo-cache "${FIMO_CACHE}" \
  --motif-filter "gene_list" \
  --gene-list-file "${RELAXED_TF_LIST}" \
  --p-value-cutoff 1e-4 \
  --output-file "${MOTIF_SCAN_OUTPUT}"

CENTER_WINDOW_ARGS=()
if [ -n "${CENTER_WINDOW_BP}" ]; then
    CENTER_WINDOW_ARGS=(--center-window-bp "${CENTER_WINDOW_BP}")
fi

echo "=== Plotting regression promoter scores ==="
"${PYTHON_CMD[@]}" analysis/plot_scan_results.py \
  --motif-scan-file "${MOTIF_SCAN_OUTPUT}" \
  --deletion-scores-file "${REGRESSION_SCORES_FILE}" \
  --profile-scores-file "${PROFILE_SCORES_FILE}" \
  --output-file "${PLOT_OUTPUT_REGRESSION}" \
  --dpi 300 \
  --fig-scale 0.5 \
  --highlight-regions \
  --merge-motifs \
  --plot-absolute-effect \
  --normalize-effect-to-max-predicted-atac \
  --motif-effect-threshold-pct 10 \
  --tss-highlight "${TSS_POSITION}" "${TSS_LABEL}" \
  --core-region "${PROMOTER_REGION}" \
  "${CENTER_WINDOW_ARGS[@]}"

echo "=== Plotting differential promoter scores ==="
"${PYTHON_CMD[@]}" analysis/plot_scan_results.py \
  --motif-scan-file "${MOTIF_SCAN_OUTPUT}" \
  --deletion-scores-file "${DIFFERENTIAL_SCORES_FILE}" \
  --profile-scores-file "${PROFILE_SCORES_FILE}" \
  --output-file "${PLOT_OUTPUT_DIFFERENTIAL}" \
  --dpi 300 \
  --fig-scale 0.5 \
  --highlight-regions \
  --merge-motifs \
  --plot-absolute-effect \
  --normalize-effect-to-max-predicted-atac \
  --motif-effect-threshold-pct 10 \
  --tss-highlight "${TSS_POSITION}" "${TSS_LABEL}" \
  --core-region "${PROMOTER_REGION}" \
  "${CENTER_WINDOW_ARGS[@]}"

echo "=== Promoter analysis complete ==="

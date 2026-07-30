#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/bundle_env.sh"

ATAC_ZARR_PATH="${ATAC_ZARR_PATH:-data/zarr/cd200_scc_dense_new.zarr}"
GENOME_ZARR_PATH="${GENOME_ZARR_PATH:-data/zarr/mm10.zarr}"
CENTER_WINDOW_BP="${CENTER_WINDOW_BP:-2000}"
DEVICE="${DEVICE:-cuda}"

CHECKPOINT="${CHECKPOINT:-analysis/checkpoints/shapescale_loss_scca_pos_with_count_head_fixed_motif_curriculum_from_scratch-epoch=105-val_loss=6.76.ckpt}"
CONFIG_FILE="${CONFIG_FILE:-configs/train_atac_phased.yaml}"
SIMPLE_MOTIFS_PATH="${SIMPLE_MOTIFS_PATH:-artifacts/motifs_with_rc_aligned.pt}"
MOTIF_NAMES_FILE="${MOTIF_NAMES_FILE:-artifacts/motifs_with_rc_aligned.txt}"
FIMO_CACHE="${FIMO_CACHE:-analysis/fimo_mappings_cache.pkl}"

REGIONS=(
    "chr16:45408000-45409426"
    "chr16:45383863-45384762"
    "chr16:45356164-45357191"
)

OUTPUT_DIR="${OUTPUT_DIR:-analysis/results/cd200_analysis_new}"
mkdir -p "${OUTPUT_DIR}"

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

echo "Checkpoint: ${CHECKPOINT}"
echo "Differential head available: ${HAS_DIFFERENTIAL_HEAD}"

for region in "${REGIONS[@]}"; do
    IFS=':' read -r CHROM COORDS <<< "${region}"
    IFS='-' read -r START END <<< "${COORDS}"
    region_suffix="${CHROM}_${START}_${END}"

    rm -f \
      "${OUTPUT_DIR}/${region_suffix}_deletion.png" \
      "${OUTPUT_DIR}/${region_suffix}_deletion_count_nucleotide.csv" \
      "${OUTPUT_DIR}/${region_suffix}_deletion_differential_heatmap.npy" \
      "${OUTPUT_DIR}/${region_suffix}_deletion_differential_nucleotide.csv" \
      "${OUTPUT_DIR}/${region_suffix}_deletion_regression_nucleotide.csv" \
      "${OUTPUT_DIR}/${region_suffix}_motifs_count.png" \
      "${OUTPUT_DIR}/${region_suffix}_motifs_differential.png" \
      "${OUTPUT_DIR}/${region_suffix}_motifs_regression.png" \
      "${OUTPUT_DIR}/motif_scan_deg_${region_suffix}.tsv" \
      "${OUTPUT_DIR}/combined_plot_highlighted_count_deg_${region_suffix}.png" \
      "${OUTPUT_DIR}/combined_plot_highlighted_differential_deg_${region_suffix}.png" \
      "${OUTPUT_DIR}/combined_plot_highlighted_regression_deg_${region_suffix}.png"
done

for region in "${REGIONS[@]}"; do
    echo "=== Processing region: ${region} ==="
    IFS=':' read -r CHROM COORDS <<< "${region}"
    IFS='-' read -r START END <<< "${COORDS}"

    "${PYTHON_CMD[@]}" analysis/mutation_analysis_cli.py region "${region}" \
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
      --save-diff-matrix-path "${OUTPUT_DIR}/${CHROM}_${START}_${END}_deletion_differential_heatmap.npy"

    echo "=== Completed region: ${region} ==="
done

DEG_FILE_1="${DEG_FILE_1:-analysis/rnaseq_deg_analysis/1F.deg.gene.csv}"
DEG_FILE_2="${DEG_FILE_2:-analysis/rnaseq_deg_analysis/AF.deg.gene.csv}"
RELAXED_TF_LIST="${RELAXED_TF_LIST:-analysis/rnaseq_deg_analysis/shared_DEGs_1F_AF_tf_relaxed.csv}"

if [ -f "${DEG_FILE_1}" ] && [ -f "${DEG_FILE_2}" ]; then
    echo "=== Starting motif scans using DEG files ==="

    for region in "${REGIONS[@]}"; do
        IFS=':' read -r CHROM COORDS <<< "${region}"
        IFS='-' read -r START END <<< "${COORDS}"
        region_suffix="${CHROM}_${START}_${END}"

        motif_scan_output="${OUTPUT_DIR}/motif_scan_deg_${region_suffix}.tsv"
        plot_output_regression="${OUTPUT_DIR}/combined_plot_highlighted_regression_deg_${region_suffix}.png"
        plot_output_count="${OUTPUT_DIR}/combined_plot_highlighted_count_deg_${region_suffix}.png"
        plot_output_differential="${OUTPUT_DIR}/combined_plot_highlighted_differential_deg_${region_suffix}.png"
        scores_file="${OUTPUT_DIR}/${CHROM}_${START}_${END}_deletion_regression_nucleotide.csv"

        if [ ! -f "${scores_file}" ]; then
            echo "Warning: scores file not found: ${scores_file}. Skipping."
            continue
        fi

        if [ "${region}" = "chr16:45408000-45409426" ]; then
            if [ ! -f "${RELAXED_TF_LIST}" ]; then
                echo "Warning: relaxed TF list not found: ${RELAXED_TF_LIST}. Skipping."
                continue
            fi
            motif_filter="gene_list"
            extra_filter_args=(--gene-list-file "${RELAXED_TF_LIST}")
            deg_base_mean_threshold="10"
            deg_padj_threshold="0.05"
        else
            motif_filter="deg"
            extra_filter_args=(--deg-file-1 "${DEG_FILE_1}" --deg-file-2 "${DEG_FILE_2}")
            deg_base_mean_threshold="2000"
            deg_padj_threshold="1"
        fi

        "${PYTHON_CMD[@]}" analysis/motif_scan_cli.py \
            --scores-csv "${scores_file}" \
            --checkpoint "${CHECKPOINT}" \
            --genome-zarr "${GENOME_ZARR_PATH}" \
            --motif-names "${MOTIF_NAMES_FILE}" \
            --fimo-cache "${FIMO_CACHE}" \
            --motif-filter "${motif_filter}" \
            "${extra_filter_args[@]}" \
            --deg-base-mean-threshold "${deg_base_mean_threshold}" \
            --deg-log2fc-threshold 0 \
            --deg-padj-threshold "${deg_padj_threshold}" \
            --p-value-cutoff 1e-4 \
            --output-file "${motif_scan_output}"

        center_window_args=()
        if [ -n "${CENTER_WINDOW_BP}" ]; then
            center_window_args=(--center-window-bp "${CENTER_WINDOW_BP}")
        fi

        tss_highlight_args=()
        if [ "${region}" = "chr16:45408000-45409426" ]; then
            tss_highlight_args=(--tss-highlight 45409053 Cd200)
        fi

        "${PYTHON_CMD[@]}" analysis/plot_scan_results.py \
            --motif-scan-file "${motif_scan_output}" \
            --deletion-scores-file "${OUTPUT_DIR}/${CHROM}_${START}_${END}_deletion_regression_nucleotide.csv" \
            --output-file "${plot_output_regression}" \
            --dpi 300 \
            --fig-scale 0.5 \
            --highlight-regions \
            --merge-motifs \
            --plot-absolute-effect \
            --normalize-effect-to-max-predicted-atac \
            --motif-effect-threshold-pct 10 \
            "${tss_highlight_args[@]}" \
            --core-region "${region}" \
            "${center_window_args[@]}"

        "${PYTHON_CMD[@]}" analysis/plot_scan_results.py \
            --motif-scan-file "${motif_scan_output}" \
            --deletion-scores-file "${OUTPUT_DIR}/${CHROM}_${START}_${END}_deletion_count_nucleotide.csv" \
            --output-file "${plot_output_count}" \
            --dpi 300 \
            --fig-scale 0.5 \
            --highlight-regions \
            --merge-motifs \
            --plot-absolute-effect \
            --normalize-effect-to-max-predicted-atac \
            --motif-effect-threshold-pct 10 \
            "${tss_highlight_args[@]}" \
            --core-region "${region}" \
            "${center_window_args[@]}"

        if [ "${HAS_DIFFERENTIAL_HEAD}" = "true" ] && [ -f "${OUTPUT_DIR}/${CHROM}_${START}_${END}_deletion_differential_nucleotide.csv" ]; then
            "${PYTHON_CMD[@]}" analysis/plot_scan_results.py \
                --motif-scan-file "${motif_scan_output}" \
                --deletion-scores-file "${OUTPUT_DIR}/${CHROM}_${START}_${END}_deletion_differential_nucleotide.csv" \
                --output-file "${plot_output_differential}" \
                --dpi 300 \
                --fig-scale 0.5 \
                --highlight-regions \
                --merge-motifs \
                --plot-absolute-effect \
                --normalize-effect-to-max-predicted-atac \
                --motif-effect-threshold-pct 10 \
                "${tss_highlight_args[@]}" \
                --core-region "${region}" \
                "${center_window_args[@]}"
        else
            echo "Skipping differential plot because checkpoint has no differential head."
        fi
    done

    echo "=== All motif scans and plots complete ==="
else
    echo "DEG files not found. Skipping motif scanning."
fi

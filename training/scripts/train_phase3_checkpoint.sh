#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/bundle_env.sh"

RUN_NAME="${RUN_NAME:-ATAC-Sequential-20250919-022148}"
CONFIG_FILE="${CONFIG_FILE:-configs/train_atac_phased.yaml}"
ATAC_ZARR_PATH="${ATAC_ZARR_PATH:-data/zarr/cd200_scc_dense_new.zarr}"
GENOME_ZARR_PATH="${GENOME_ZARR_PATH:-data/zarr/mm10.zarr}"
SUMMITS_CSV_PATH="${SUMMITS_CSV_PATH:-training/data/scc_all_peaks.deduplicated.bed}"
CELLTYPE_IDS="${CELLTYPE_IDS:-SCCAneg.cd200_scc}"

mkdir -p checkpoints

phase_checkpoint_args=()
if [ -n "${PHASE1_CHECKPOINT:-}" ]; then
    phase_checkpoint_args+=(--load-phase1-checkpoint "${PHASE1_CHECKPOINT}")
fi
if [ -n "${PHASE2_CHECKPOINT:-}" ]; then
    phase_checkpoint_args+=(--load-phase2-checkpoint "${PHASE2_CHECKPOINT}")
fi
if [ -n "${PHASE3_CHECKPOINT:-}" ]; then
    phase_checkpoint_args+=(--load-phase3-checkpoint "${PHASE3_CHECKPOINT}")
fi
if [ "${STRICT_LOADING:-false}" = "true" ]; then
    phase_checkpoint_args+=(--strict-loading)
fi

"${PYTHON_CMD[@]}" -m atac_diff_prediction.train_sequential_phased \
  --config "${CONFIG_FILE}" \
  --atac-zarr-path "${ATAC_ZARR_PATH}" \
  --genome-zarr-path "${GENOME_ZARR_PATH}" \
  --summits-csv-path "${SUMMITS_CSV_PATH}" \
  --celltype-ids "${CELLTYPE_IDS}" \
  --run-name "${RUN_NAME}" \
  --max-regions "${MAX_REGIONS:-500000}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-32}" \
  --phase1-epochs "${PHASE1_EPOCHS:-0}" \
  --phase2-epochs "${PHASE2_EPOCHS:-50}" \
  --phase3-epochs "${PHASE3_EPOCHS:-300}" \
  --differential-threshold "${DIFFERENTIAL_THRESHOLD:-0.05}" \
  --tn5-motifs-path artifacts/bias.meme \
  "${phase_checkpoint_args[@]}"

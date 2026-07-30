#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/bundle_env.sh"

ATAC_ZARR_PATH="${ATAC_ZARR_PATH:-data/zarr/cd200_scc_dense_new.zarr}"
GENOME_ZARR_PATH="${GENOME_ZARR_PATH:-data/zarr/mm10.zarr}"
SUMMITS_CSV_PATH="${SUMMITS_CSV_PATH:-training/data/total.bed}"
CELLTYPE_IDS="${CELLTYPE_IDS:-SCCApos.cd200_scc}"
RUN_NAME="${RUN_NAME:-shapescale_loss_scca_pos_with_count_head_fixed_motif_curriculum_from_scratch}"

"${PYTHON_CMD[@]}" -m atac_prediction.train_atac \
  --atac-zarr-path "${ATAC_ZARR_PATH}" \
  --genome-zarr-path "${GENOME_ZARR_PATH}" \
  --summits-csv-path "${SUMMITS_CSV_PATH}" \
  --celltype-ids "${CELLTYPE_IDS}" \
  --simple-motifs-path artifacts/motifs_with_rc_aligned.pt \
  --batch-size "${BATCH_SIZE:-32}" \
  --max-epochs "${MAX_EPOCHS:-200}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --use-curriculum-learning \
  --curriculum-groups "${CURRICULUM_GROUPS:-5}" \
  --curriculum-sampling-strategy "${CURRICULUM_SAMPLING_STRATEGY:-equal}" \
  --max-regions "${MAX_REGIONS:-100000}" \
  --precision "${PRECISION:-32-true}" \
  --split-strategy "${SPLIT_STRATEGY:-chromosome}" \
  --val-chr "${VAL_CHR:-chr14}" \
  --test-chr "${TEST_CHR:-chr10}" \
  --run-name "${RUN_NAME}"

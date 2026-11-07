# GM12878 chromosome held-out benchmark

This is the external-data prediction benchmark requested on 26 October 2026. It uses the published ChromBPNet GM12878 ATAC-seq data and fold 0, comparing newly trained MoDL, native scalar ExplaiNN, and freshly trained native ChromBPNet, with the released fold-0 ChromBPNet checkpoint as an additional reference. **Results are available only after the corresponding training/evaluation stages finish.** Read `benchmark_status.json` for the live stage and logs; response-draft text elsewhere is not a benchmark result.

**Recovery, 27 October 2026 at 02:53 UTC:** the original supervisor and MoDL process were no longer present; their status timestamp had stopped at 00:32 UTC. The termination cause is not established. ExplaiNN's completed fit/predictions and the published ChromBPNet reference are reused. The three-epoch partial MoDL run remains unchanged in `runs/modl_seed1234/`; a new MoDL fit starts from scratch in a timestamped directory because the checkpoint did not contain optimizer state. The remaining queue now runs as persistent user service `cd200-gm12878-benchmark-recovery.service`, with a live status file and separate recovery logs. Two recovery tests verify completed-run reuse, partial-run preservation and refusal to launch when the previous supervisor is alive.

## Fixed setting

| Field | Value |
| --- | --- |
| Source experiment | GM12878 ATAC-seq [ENCSR637XSC](https://www.encodeproject.org/experiments/ENCSR637XSC/) |
| Published model annotation | [ENCSR389HIH](https://www.encodeproject.org/annotations/ENCSR389HIH/) |
| Regions / observed insertion signal / model archive | ENCFF971WEQ / ENCFF180XQC / ENCFF142IOR; official checksums in `data/download_manifest.json` |
| Fold | Released fold 0, `cv_params.fold_0.json`; no multi-fold prediction ensemble |
| Test chromosomes | chr1, chr3, chr6 |
| Validation chromosomes | chr8, chr20 |
| Input / output | 2,114 bp DNA / central 1,000 bp raw insertion counts |
| Full cache | Train 387,457 = 193,894 peaks + 193,563 nonpeaks; validation 37,769 = 18,719 peaks + 19,050 nonpeaks; test 314,645 = 62,929 peaks + 251,716 nonpeaks |
| Sampling | Training: all peaks plus 0.1× as many nonpeaks, sampled per epoch. Validation: all peaks plus one fixed 0.1× nonpeak subset, shared by the new models. Test: all released test peaks and nonpeaks |
| Augmentation | Training peaks jitter ±500 bp; nonpeaks centered; random reverse complement in training only |
| First run | Seed 1234, Adam 0.001, batch 64, at most 50 epochs, validation early-stopping patience 5 |
| Shape/count loss weight | 93.3, as recorded by the released fold-0 model; scalar-only ExplaiNN uses count MSE alone |
| Hardware | Local RTX 5090, 32 GB; CUDA access requires execution outside the filesystem sandbox |

The cache contains integer-valued shifted insertions: the verified signal has no fractional values in extracted windows. Omitted sparse bigWig positions become zero as in the native ChromBPNet loader; errors are not replaced with samples. Signal is not maximum-normalized, library-scaled or smoothed. The local hg38 FASTA's canonical chromosome lengths match the signal; its full SHA-256 and source path are recorded in `data/cache_manifest.json`. The source of that existing FASTA is not inferred from its filename.

## Models and declared differences

- **MoDL:** original `DNAATACMotifCNN` and all 3,222 fixed motif kernels. It receives the common 2,114 bp input; its positive profile is cropped to the central 1,000 bp and normalized to shape probabilities. Count prediction uses the same `ln(1 + insertions)` target as the other models. These crop/target/loss changes are an explicitly documented benchmark adaptation of the manuscript model; no original source file was edited.
- **ExplaiNN:** the official implementation, 100 independent convolutional units, native scalar output, MSE on the common log-count target. This is scalar regression supported by its upstream trainer. Profile metrics are **not applicable**. A 300-unit sensitivity analysis and additional seeds remain future work, rather than being chosen after inspecting test performance.
- **Fresh ChromBPNet:** the official bias-factorized 512-filter/eight-dilation architecture and native multinomial/count objectives, using the released, frozen, same-fold GM12878 bias component. The cache-backed batch loop uses the same new-model validation subset as MoDL and ExplaiNN. The learned bias component is pretrained, so this is new TF-network training with a fixed published bias model, not de novo bias training.
- **Published ChromBPNet reference:** only the released fold-0 components, combined using the native profile-addition/log-count-logsumexp equations. Independent reconstruction of the original combined H5 graph gives exactly zero maximum discrepancy for both outputs on eight random sequences (`logs/chrombpnet_component_parity.json`). No cross-version marshaled Lambda is executed.

PyTorch training uses bfloat16 autocast and stores FP32 weights; held-out inference is FP32. Native TensorFlow runs FP32 with legacy Keras compatibility and the existing CUDA 12 libraries. Framework versions and source hashes are recorded per run. This reproduces the released data partitions and native ChromBPNet architecture, with declared modern-runtime and batch-loop changes; it is not a claim of bitwise reproduction of the original TensorFlow 2.8 training.

The published reference used its original validation nonpeak sample and preprocessing. The newly trained models share a fixed cache-order/seed-selected validation subset. Published outlier/region filtering has already occurred upstream; the archive is reused unchanged. The explicit 93.3 loss weight is part of the published configuration, not fitted to test data.

## Execution and outputs

The GPU checks completed successfully: original MoDL and native ExplaiNN each ran two training steps and two validation batches; native full ChromBPNet completed a forward/backward batch. These are explicitly labeled pilots and excluded from final evaluation.

The published reference has completed held-out prediction and a standalone evaluation. For **peaks**, log-count Pearson r is **0.6331** (95% paired-block framework interval **0.6167–0.6476**); mean native profile Jensen–Shannon distance is **0.3267** (0.3242–0.3296). For all test peaks and nonpeaks, count r is 0.7114 (0.6885–0.7348). See `reference_sanity_evaluation/metrics.tsv` and `evaluation.json`; these are reference-only results, not evidence of a comparative advantage for any new model. The joint evaluator will recompute the comparison after the fresh predictions exist.

The paper **does report GM12878-specific performance** in [Extended Figure 2b–c](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/#F9). The retrieved GM12878 ATAC count box is visually around **0.63**, with a rough across-fold range of **0.60–0.65**, consistent with our recomputed 0.6331. That range is an approximate reading of the raster figure, not a confidence interval. The prose's approximately 0.70 for K562 and pooled median 0.69 across other contexts are not GM12878 ATAC-specific targets. Exact fold-0 reproduction remains unverified because the plot does not label individual folds or provide their numerical values; there is no basis to call fold 0 unusually poor. See [PUBLICATION_RECONCILIATION.md](PUBLICATION_RECONCILIATION.md) and the preserved original figure/source workbook.

`run_benchmark.py` queues GPU work sequentially:

1. Wait for completion of the already-started published reference inference.
2. Train and evaluate ExplaiNN, then MoDL, using validation-only checkpoint selection.
3. Write `evaluation_published_reference/` and `EVALUATION_PUBLISHED_REFERENCE.md` for the initial comparison against the released baseline.
4. Train the native ChromBPNet TF network with the shared training setup and frozen fold-0 bias model; predict the same held-out rows.
5. Write `evaluation_matched_training/` and `EVALUATION_MATCHED_TRAINING.md`, including the published checkpoint as a reference alongside all three newly trained models.

Each prediction directory contains an ordered scalar log-count array, optional profile-probability array, and identity/checkpoint metadata. Evaluation checks exact region-order hashes before computing metrics. Count metrics are Pearson/Spearman and RMSE on the common natural-log count scale. Profile metrics include the native Jensen–Shannon **distance**, separately labeled divergence, and per-window correlations at 1 bp and 20 bp aggregation. Peaks, nonpeaks and each test chromosome are reported separately. All-zero target profiles and undefined correlations are counted explicitly.

The initial ExplaiNN training source is retained in `source_snapshots/`. A narrowly allowlisted NumPy-numeric checkpoint-loading fix was needed for `weights_only=True`; `logs/checkpoint_compatibility.json` verifies both pilot checkpoint reloads and identical ASTs for every training/model/loss function. Prediction metadata records both source hashes when this compatibility path is used. No training computation or saved tensor was changed.

Uncertainty uses 2,000 paired genomic-block bootstrap draws stratified by chromosome, merging block boundaries bridged by overlapping 2,114-bp inputs. Confidence intervals describe test-region sampling conditional on a fitted checkpoint, not training-seed uncertainty. Spearman is a point estimate only. Paired model differences are in `evaluation.json`.

From the bundle root, the supervisor command is:

```bash
../motif_chip_model/.venv/bin/python -u \
  reviewer_revision_xi_frank/gm12878_benchmark/run_benchmark.py
```

**Do not launch a second copy while `benchmark_status.json` records an active run.** The supervisor refuses to overwrite its status file or prior run outputs. The cache, downloaded archives and released models live under `/tmp/cd200_gm12878_benchmark/`; all run checkpoints, logs and final prediction/evaluation outputs live in this folder. Temporary inputs can be regenerated from the registered official URLs and `prepare_cache.py`; do not clear them while the job is running.

Check the persistent service with `systemctl --user status cd200-gm12878-benchmark-recovery.service`. Recovery uses `--restart-interrupted`, archives the previous stale status, skips completed outputs and preserves incomplete fits in their original directories. It refuses recovery if the recorded old processes are still present. There is no automatic retry of scientific/code failures; those are recorded in the status file for inspection.

## Acceptance and remaining scope

- [x] Dataset/model lineage, region-archive MD5 and observed-signal MD5 verified.
- [x] Shared cache completed with hashes and disjoint chromosomes; all test rows retained.
- [x] GPU forward/backward checks and model/target alignment checks passed.
- [x] Released component inference checked against the original full-model graph.
- [ ] Full fresh MoDL and ExplaiNN fits and all held-out predictions completed.
- [ ] Initial published-reference evaluation and source tables produced.
- [ ] Fresh native ChromBPNet comparison completed.
- [ ] Additional seeds/folds, capacity sensitivity, and publication figure completed if required for manuscript claims.

This is an external human-context benchmark. It strengthens the method evaluation but does **not** remedy historical leakage in the mouse manuscript checkpoints; R1.1 still needs the separate clean mouse retraining. SEACells/CollecTRI remains Frank-owned and deferred.

Primary-source setting details are in [PUBLISHED_SETTING.md](PUBLISHED_SETTING.md); model integration is in [LOCAL_MODEL_INTEGRATION.md](LOCAL_MODEL_INTEGRATION.md). No ranking or superiority claim is supported until the full metrics and paired uncertainty exist.

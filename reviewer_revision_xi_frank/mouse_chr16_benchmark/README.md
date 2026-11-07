# Mouse cSCC chromosome-held-out benchmark

User-authorized 2026-11-01. This extends the completed human benchmark to the manuscript's mouse signal. Code and small metadata are in this folder; large caches/runs are under `/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/`.

## Target and acceptance criteria

Train MoDL, native ChromBPNet (including a new mouse-only bias component) and native scalar ExplaiNN on identical mouse partitions and targets. Report genome-wide supplied-region count/profile metrics, paired genomic-block intervals, common profile/count losses, and separate P1/P2/promoter results. Retain genuine zero targets. Never select checkpoints or tune parameters using chr16 or other test outcomes.

- Test: chr16, chr8, chr9. Validation: chr14. Training: other mm10 autosomes.
- Sample: `SCCApos.cd200_scc`, corresponding to the PDSCC2 CD200-positive context. No historical model weights are reused.
- Input: 2,114 bp; supervised output: central 1,000 bp. Cache geometry matches the prior human benchmark, including training-only peak jitter and reverse complement.
- Observations: unsmoothed, unnormalized integer ATAC signal from the bundled Zarr. **Original Tn5 shift/endpoint and sample-merge provenance is not independently established.** The benchmark does not call this verified raw shifted insertion data or use fold-change tracks as counts.
- Region universe: `training/data/scc_all_peaks.deduplicated.bed`, without differential-effect filtering. `for_profile` and `for_count` are supplied peaks; `background` is supplied background. The universe is not exhaustive genome tiling.
- Frozen exclusions: non-autosomes, out-of-bounds full 3,114-bp cache context, and >5% ambiguous sequence in that context. No test-target exclusion or outcome-based resampling.
- Count objective: mean squared error on ln(1 + observed central-1,000-bp signal count). Common main-model weight is fitted from training peak counts only (median/10, minimum1).
- Profile objective: count-weighted categorical cross-entropy / multinomial NLL. The common held-out loss evaluator reports both inclusion/exclusion of the multinomial constant; raw TF and Torch training loss magnitudes should not be directly compared.
- One seed (1234), maximum50 epochs, validation patience5, batch64, Adam0.001. Torch training uses bfloat16 autocast; inference FP32. TF FP32. Block intervals do not measure training-seed uncertainty.

## Native mouse bias component

Randomly initialized native128-filter/4-dilation BPNet is fitted only on low-count training nonpeaks. The fixed selection rule uses 0.5 times the first percentile of training peak counts, followed by train-fitted strict count-outlier limits; validation applies those same gates. Bias count intercept scaling uses training nonpeaks only. The component is frozen for the native512-filter/8-dilation main model. Human bias weights and components trained on test chromosomes are rejected. Empty/degenerate background selection fails rather than silently substituting a component.

## Three selected loci

Coordinates are 0-based half-open and come from the original plotting/mutation scripts:

| Locus | Original plotted interval |
| --- | --- |
| P1 / Enhancer1 | chr16:45383863–45384762 |
| P2 / Enhancer2 | chr16:45356164–45357191 |
| Cd200 promoter | chr16:45408000–45409426 |

`evaluate_loci.py` reports native central1,000-bp count loss and profile NLL/cross-entropy, Jensen–Shannon distance, and profile Pearson at1bp and20bp. Additional tiles cover the longer original intervals. Full-interval profiles average overlapping count-scaled predictions per base; counts are then summed once. Such reconstructed counts are explicitly distinct from a native fixed-window count head. No count correlation across only three loci is promoted as a benchmark, and bases are not bootstrapped as independent examples.

These loci were already selected in the manuscript and are descriptive interpretation examples even though their chromosome is excluded from this new training. Their performance never determines checkpoint choice.

## Execution and artifacts

- `prepare_mouse_cache.py`: source metadata/chunk checksums, FASTA/Zarr concordance, immutable split/cache files; source audit in `cache_manifest.json`.
- `train_mouse_torch.py`: original shared Torch model/training functions with correct stored-signal metadata.
- `train_mouse_chrombpnet.py`: mouse-native bias fitting/lineage guards and shared native full-model fitting.
- `predict_loci.py`: FP32 predictions on frozen locus windows, with checkpoint and region identity checks.
- `evaluate_mouse.py`: shared count/profile/paired-block evaluator with mouse target semantics.
- `evaluate_losses.py`, `evaluate_loci.py`: common losses and locus metrics.
- `run_mouse_benchmark.py`: persistent sequential supervisor; live `benchmark_status.json`; completed `BENCHMARK_RESULTS.md` when every stage succeeds.
- `tests.log`: focused mathematical, geometry, lineage and source-reader checks. GPU pilot artifacts are separately marked and never used for final model selection.

Final runs are expected in `runs/modl_seed1234`, `runs/mouse_bias_seed1234`, `runs/chrombpnet_seed1234`, and `runs/explainn_seed1234` under the storage directory. Intermediate evaluations are labeled `modl_preliminary` and `modl_chrombpnet`; `final` includes all three methods. Each has common metrics, losses and locus summaries. The original mouse checkpoints and human benchmark outputs remain untouched.

No results are claimed until the corresponding completion marker exists. Consult the live status file rather than directory names alone.

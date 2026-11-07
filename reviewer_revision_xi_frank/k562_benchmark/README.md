# K562 chromosome held-out benchmark

Prepared and queued on 27 October 2026 at the user's request. This repeats the external prediction benchmark in K562 with freshly trained MoDL, native scalar ExplaiNN and native ChromBPNet, plus the released fold-0 ChromBPNet reference. It is an independently trained second cellular context, not a zero-shot transfer test.

The persistent user service is **`cd200-k562-benchmark.service`**. It waits for the existing GM12878 benchmark to finish or record a failure before starting any GPU stage. See [benchmark_status.json](benchmark_status.json) for the live waiting/training state. The current GM12878 jobs were not interrupted.

## Verified published setting

| Item | K562 setting |
| --- | --- |
| Experiment | [ENCSR868FGK](https://www.encodeproject.org/experiments/ENCSR868FGK/) |
| Published model annotation | [ENCSR467RSV](https://www.encodeproject.org/annotations/ENCSR467RSV/) |
| Observed insertion signal | ENCFF874FUM; MD5 `7265bdcfaf55605530eff48f98901ff2` |
| Released regions | ENCFF991RUK; MD5 `103ee3a49e9ff701da2acfdb59b67d70` |
| Released models | ENCFF984RAF; MD5 `0da121e548f1fc977e6ea206ef5b7a52` |
| Fold | Released fold 0; test chr1/chr3/chr6; validation chr8/chr20 |
| Geometry | 2,114 bp input; central 1,000 bp raw insertion target |
| Training cache | 366,656 regions, including 183,494 peaks |
| Validation cache | 35,360 regions, including 17,514 peaks |
| Test cache | 332,370 regions, including 66,474 peaks; no test subsampling |
| Training recipe | Seed 1234; Adam 0.001; batch 64; maximum 50 epochs; validation patience 5 |
| Training/validation nonpeaks | 0.1× the peak count; training resampled each epoch; fixed shared validation sample |
| Augmentation | Peak-only ±500 bp jitter and random reverse complement in training; centered validation/test |
| Count-loss weight | **76.0**, from the released K562 fold-0 parameter TSV |
| Frozen bias component | Released K562 same-fold bias model; archive run identifies 128 filters, four dilated layers, seed 1234 and threshold factor 0.5 |

All three downloaded files passed their official ENCODE MD5 checks. The primary metadata and verification record are in `data/`; the complete cache manifest records source/output hashes. Five native model/data tests passed, including independent equivalence of the original combined K562 H5 model and the separately loaded bias/residual components. Three supervisor tests include the requirement to wait for the existing GPU queue and use K562-specific assets/configuration.

## Execution and output

This reuses the inspected trainers, evaluator and cache builder in `../gm12878_benchmark/`. The source code is shared; K562's inputs, count-loss weight, output paths and status are separate. Existing GM12878 training code and active processes were not changed. The original GM12878 supervisor/cache-builder snapshots are retained in that benchmark's `source_snapshots/`.

After the GM12878 queue ends, the service automatically:

1. Predicts all K562 test regions with the released fold-0 ChromBPNet model.
2. Trains/predicts native scalar ExplaiNN, then MoDL, from scratch.
3. Writes the initial comparison against the published reference.
4. Trains/predicts fresh native ChromBPNet with the frozen K562 bias component.
5. Writes the matched-training comparison, source tables and paired genomic-block confidence intervals.

Count prediction is compared for all methods. Profiles are compared for MoDL and ChromBPNet; native scalar ExplaiNN has no profile output. Counts use `ln(1 + total raw insertions)`. Evaluation retains all released test windows and reports peaks, nonpeaks, all windows and each test chromosome separately. There are 2,000 paired block-bootstrap draws, with chromosome stratification and overlap-aware blocks. This first run is one fold and seed; those intervals do not represent training-seed variability.

Inputs and caches are under `/tmp/cd200_k562_benchmark/`; checkpoints, predictions, logs and final tables will be under this folder. Do not remove the temporary inputs while the service is waiting or running. Check the service with:

```bash
systemctl --user status cd200-k562-benchmark.service
```

No K562 model-performance result is claimed until inference/evaluation completes. Numerical replication of the paper's particular figure still requires matching its exact model and population; the original publication includes K562 and GM12878 separately in Extended Figure 2.

# Published GM12878 benchmark setting

This specification separates primary-source facts from the choices needed to compare MoDL and ExplaiNN on ChromBPNet's public setting. It is not a record of completed training. Run logs, checkpoints and result artifacts determine execution status.

## Dataset identity and fixed assets

Use the released [ENCODE model annotation ENCSR389HIH](https://www.encodeproject.org/annotations/ENCSR389HIH/), whose downloaded primary JSON is `data/ENCSR389HIH.json`. Its description identifies the models submitted for the ChromBPNet manuscript and its experimental input is **ENCSR637XSC**. This is GM12878 ATAC-seq; [the experiment record](https://www.encodeproject.org/experiments/ENCSR637XSC/) lists three isogenic replicates and NovaSeq 6000 sequencing. **ENCSR095QNB is a separate GM12878 experiment**, with its own annotation ENCSR067KBX; mixing those accessions would change the dataset.

Primary metadata pins the following files. The download process must verify the published MD5 and record a local SHA256. `published_sources.json` preserves their exact S3 URLs and metadata.

| Asset | Accession | Size in bytes | Published MD5 |
| --- | --- | ---: | --- |
| Training and test regions archive | ENCFF971WEQ | 78,117,459 | `6b7ccb352e0fc2f76853144ed1cd846d` |
| Observed signal profile bigWig | ENCFF180XQC | 1,207,521,844 | `ccd7670dd53f297cc670ea0637a357d7` |
| Models archive | ENCFF142IOR | 718,106,576 | `474bf4752d917573552e95e5e6a8612f` |
| Original bias models archive | ENCFF699ZFA | 14,983,210 | `2ede1749e65499b6c6019bc1329cfdef` |

The signal file's `derived_from` entries are ENCFF440GRZ, ENCFF981FXV and ENCFF962FMH. The model archive derives from the signal, region archive and ENCFF699ZFA. The separately fetched primary record `data/ENCFF699ZFA.json` verifies **output type “bias models”**, derived from ENCFF180XQC and ENCFF623LOS. Preserve those provenance links; verify each remaining linked file's role from its own metadata.

Use the released observation track as the common target instead of silently replacing it with a normalized ATAC signal or another GM12878 dataset. Verify its numeric scale and coordinate conventions. The [original ChromBPNet preprint](https://doi.org/10.1101/2024.12.25.630221) describes ENCSR637XSC as the 572-million-read GM12878 dataset and uses corrected unstranded ATAC insertion positions (+4/−4) in Methods §1.2. A new BAM-to-bigWig conversion must reproduce that convention.

## Chromosome partitions and geometry

The [official fold 0 JSON](https://zenodo.org/records/7445373/files/fold_0.json) specifies:

- Test: **chr1, chr3, chr6**.
- Validation: **chr8, chr20**.
- Training: chr2, chr4, chr5, chr7, chr9–chr19, chr21, chr22, chrX, chrY.

The [official preprocessing guide](https://github.com/kundajelab/chrombpnet/wiki/Preprocessing) links this fixed Zenodo file. Confirm that the downloaded region/model archive's own fold files agree; a tutorial fold alone is not proof of a particular checkpoint's split. Compare **fold 0**, rather than averaging five pretrained folds: other folds may have trained on fold 0 test chromosomes.

The geometry is **2,114 bp input → central 1,000 bp target**, centered at `narrowPeak.start + narrowPeak.summit` (column 10 is an offset, not an absolute coordinate). Half-open target intervals are `[summit−500, summit+500)`; input intervals are `[summit−1057, summit+1057)`. Keep the identical eligible windows and observations for all three models. Exclude insufficient-context regions deterministically and record every exclusion. The mm10 Cd200 holdout choice from the separate manuscript-repair protocol does not apply to this human fold.

## Native ChromBPNet configuration

The following facts can be checked directly in the downloaded official source under `vendor/chrombpnet/`; pin that snapshot's actual upstream commit/archive digest before interpreting results.

| Setting | Evidence in official source |
| --- | --- |
| 512 filters; 8 residual dilated convolutions; first filter width 21; profile output filter width 75 | `chrombpnet/training/models/chrombpnet_with_bias_model.py:27–98`; `chrombpnet/helpers/hyperparameters/find_chrombpnet_hyperparams.py:29–38` |
| Input 2114; output 1000; default seed 1234; maximum 50 epochs; patience 5; Adam learning rate 0.001; batch size 64 | `chrombpnet/parsers.py:65–70`, `:121–122` and `chrombpnet/training/models/chrombpnet_with_bias_model.py` |
| Peak jitter up to ±500 bp; negative sampling ratio 0.1 | `chrombpnet/parsers.py:121`; `chrombpnet/helpers/hyperparameters/find_chrombpnet_hyperparams.py:15–20` |
| Bias network loaded and frozen before joint training | `chrombpnet/training/models/chrombpnet_with_bias_model.py:14–25` |

The [official training guide](https://github.com/kundajelab/chrombpnet/wiki/ChromBPNet-training) distinguishes the full model and the residual bias-corrected model. Evaluate observed insertion profiles with the **full combined model**; retain the residual model for biological interpretation. Do not compare a bias-corrected profile directly to raw assay signal and call it predictive parity.

The original paper's Methods §1.9 independently reports a 1:10 nonpeak:peak training mixture, ±500 bp peak jitter, Adam 0.001, five-fold cross-validation and validation patience five. Its supplement/Methods discusses transferred HepG2 bias models in selected GM12878 analyses. The actual released fold 0 differs from that assumption: its archive parameter TSV records `GM12878_03.01.2022_bias_128_4_1234_0.4_fold_0`, identifying the same GM12878 dataset's fold 0 bias model, with 128 filters, four dilated layers, seed 1234 and background threshold factor 0.4. The archive's full-model count-loss weight is **93.3**. Source: `/tmp/cd200_gm12878_benchmark/models/fold_0/logs.models.fold_0.ENCSR637XSC/logfile.modelling.fold_0.ENCSR637XSC.chrombpnet_model_params.tsv`. The released bias H5 confirms four dilated convolutions. The tutorial's K562 ENCSR868FGK bias model remains an example, not GM12878 manuscript lineage.

**Published preprocessing caveat:** `find_chrombpnet_hyperparams.py:84–175` intentionally uses **training plus validation**, and excludes test chromosomes, to estimate count outlier thresholds, loss weight and the constant rescaling of the bias count head. The weight is based on median counts divided by 10, bounded below by 1. Reproducing that behavior is acceptable only if explicitly recorded. Restricting these estimates to training alone is a defensible stricter protocol, but it is a declared deviation from the released recipe. Never use test counts to fit these quantities.

## ExplaiNN as a native scalar comparator

Use the [official Wasserman lab repository](https://github.com/wassermanlab/ExplaiNN), downloaded under `vendor/ExplaiNN/`. Its parameterized `ExplaiNN` class has independent units and a final unconstrained linear output (`explainn/models/networks.py:44–68`, `:147–162`). The official CLI explicitly supports **MSE**, **Pearson**, and **PoissonNLL** objectives, including ATAC peak counts (`scripts/train.py:101–106`, `:209–224`). Thus a native scalar count comparison does not require adding a profile decoder.

**Recommended benchmark choice:** instantiate the upstream architecture with `input_length=2114`, `num_classes=1`, and use MSE against `log1p(sum of central 1000-bp observed insertions)`. Record this as a new single-track quantitative-accessibility application of ExplaiNN, not a reproduction of its original multi-cell-type AI-TAC task. Keep the upstream class intact; the target transformation, optimizer and candidate model size are benchmark settings. Native defaults are filter width 19, two fully connected layers per unit, pooling size/stride 7, and 100 units (`scripts/train.py:59–99`). A 300-unit candidate is motivated by the original study but must be declared and selected using validation only.

PoissonNLL is also supported upstream, but uses a different loss/target combination; if chosen, document how its log-rate output is converted for evaluation. Do not use a per-example correlation-across-cell-types loss for a single output: a one-element vector cannot supply that correlation.

The main comparison should report count metrics for **all three models**. Report profile metrics for **MoDL and ChromBPNet**; ExplaiNN's scalar configuration has no predicted base profile, so its profile cells should read “not applicable.” Adding 1,000 positional outputs or a decoder would be a separately labeled model adaptation with new training semantics, rather than native ExplaiNN.

## Evaluation contract and scope

For exact compatibility, the official source defines count Pearson/Spearman/MSE in `chrombpnet/training/metrics.py:20–41`; `training/predict.py` evaluates log-transformed observed counts. Its profile function at `metrics.py:43–94` calls SciPy `jensenshannon` on observed and predicted probability profiles. This is **Jensen–Shannon distance**, not divergence; its default logarithm base is natural. Do not silently compare it to base-2 divergence or smoothed Pearson profiles. Record zero-total-count exclusions for profile metrics, while retaining these windows for count evaluation.

Prespecify peak-only and peak-plus-background strata, with all test windows and their IDs saved. Count correlation on peaks alone and on a mixture with background are different endpoints. Add paired genomic-block bootstrap intervals and pairwise metric differences using the same resampled blocks for each model; these uncertainty estimates are additions for the reviewer request, not claims about the original paper's reporting.

For this experiment, “held-out” means the exact fold 0 chromosome partition is respected by every newly trained model and by every component reused for comparison. Pretrained ChromBPNet fold 0 can serve as the published reference; a new native ChromBPNet run with the same data and geometry establishes local reproducibility. Comparing only a full published checkpoint to shortened MoDL/ExplaiNN runs measures a constrained pilot and must be labeled accordingly. Match selection opportunities and report actual updates, epochs, wall time and parameter counts; equal epoch caps alone do not establish convergence.

This public experiment can establish predictive performance in a second, human biological context. It cannot retroactively validate the legacy mouse CD200 checkpoints, prove state-specific TEAD occupancy, or replace the clean mouse retraining requested in R1.1.

## Items to verify against downloaded archives before running

- [ ] Exact region filenames, summit offsets, duplicate handling and fold 0 train/validation/test chromosomes.
- [ ] ENCODE MD5 and locally computed SHA256 for all assets and upstream source snapshots.
- [ ] Full versus residual checkpoint identities, architecture shapes, and frozen-bias lineage.
- [ ] Reference assembly and chromosome lengths agree with the released model's reference provenance.
- [ ] Numerical observation scale, +4/−4 insertion convention, missing-value policy and profile zero-count handling.
- [ ] Same eligible windows, target totals and output transforms for all models.
- [ ] Every change from native MoDL, native ExplaiNN or the published ChromBPNet recipe declared in the run configuration.

## Verified native GPU runtime and model loading

`train_chrombpnet.py` imports the original vendored architecture and TensorFlow Probability multinomial loss; it does not relabel a bare BPNet as ChromBPNet. It requires a discovered CUDA GPU and a GPU-placed convolution. TensorFlow 2.21 in the existing `get_model_seq` environment was built against CUDA 12.5.1; configuring the existing `motif_chip_model` CUDA12 libraries resolves discovery. The isolated `runtime_deps/` contains legacy `tf_keras` and the small probability dependencies. `runtime_cuda/` points to the existing Triton CUDA compiler/libdevice artifacts. No parent environment was modified.

The full 512-filter/eight-layer bias-factorized model passed a native GPU forward/backward smoke test on RTX 5090; see `logs/chrombpnet_gpu_smoke.log`. The four cache-geometry/partition tests in `test_train_chrombpnet.py` passed. These verify runtime and input contracts, not final model accuracy.

The original combined H5 contains Python-version-specific Lambda bytecode. Released prediction instead loads its two non-Lambda components and reconstructs the exact native sum-of-logits/logsumexp-of-counts equations. An independent parity check read the original combined graph JSON, replaced only the inspected Lambda with a named `reduce_logsumexp`, loaded the original combined H5 weights, and compared eight random 2114-bp inputs. Profile logits and logcounts matched the component reconstruction **exactly (maximum absolute difference 0.0)**. `logs/chrombpnet_component_parity.json` records the combined H5 SHA256 and method; no marshaled bytecode was executed. That small parity check ran on CPU for verification; the benchmark inference runs on GPU.

The independent H5 parity check is reproducible with the opt-in `ReleasedGraphParity` test in `test_train_chrombpnet.py`: set `CHROMBPNET_PARITY_MODELS` to the extracted `fold_0` directory, enable `TF_USE_LEGACY_KERAS=1`, and include `runtime_deps` plus `vendor/chrombpnet` on `PYTHONPATH`. The ordinary four unit checks do not require TensorFlow.

**Completed published-reference evaluation:** all **314,645** common fold 0 test windows on chr1/chr3/chr6 were predicted on the local GPU, with exit status 0. Arrays and ordering metadata are in `runs/chrombpnet_published_fold0/test_predictions/`; `logs/chrombpnet_prediction_validation.json` records finite-value and shape checks. The largest profile probability-sum error was **2.3842×10⁻⁷**. This is inference with the released fold 0 weights, not newly trained ChromBPNet; predictive accuracy must be obtained by the shared evaluator against the preserved observations.

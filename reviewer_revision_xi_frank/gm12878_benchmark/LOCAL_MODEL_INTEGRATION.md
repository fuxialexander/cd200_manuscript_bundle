# Local model integration for the GM12878 benchmark

The original MoDL architecture accepts the native ChromBPNet 2114 bp input without source edits. Its full-length profile can be cropped to the same central 1000 bp, while its native scalar head is retrained against that window's insertion count. The supplied manuscript Lightning wrapper uses a central 2048 bp crop and different normalized targets; this benchmark therefore uses the original bare model with an explicitly adapted training objective. No manuscript checkpoint is reused.

## Implemented training entrypoint

[`train_torch.py`](train_torch.py) trains original MoDL or the vendored official ExplaiNN scalar model. Fitting reads only training and validation caches. Separate `--predict` restores the checkpoint selected by validation and then reads the test set. CUDA is required for both actual fitting and prediction; the script refuses CPU fallback.

```bash
/home/xf2217/Repos/motif_chip_model/.venv/bin/python \
  reviewer_revision_xi_frank/gm12878_benchmark/train_torch.py \
  --model modl --cache-dir PATH_TO_SHARED_CACHE --output-dir NEW_RUN_DIRECTORY
```

Use `--model explainn` for the official scalar baseline. Add `--predict` with the same run directory to emit its heldout predictions. Do not replace the placeholders with guessed cache locations: dataset preparation must finish and establish its manifest first.

The cache contract is `{train,valid,test}.x.npy`, uint8 `(N,3114)` DNA codes A=0/C=1/G=2/T=3/N=4; `.y.npy`, float32 `(N,2000)` raw insertion profiles; and `.regions.tsv` with `region_id`, `chrom`, `center`, `is_peak`. For a training-peak offset `j` drawn uniformly from 0 through 1000, the input is `x[j:j+2114]`, the aligned common target is `y[j:j+1000]`. Training nonpeaks use `j=500`; random reverse complement applies to both kinds of training example. The target cache starts 557 bp after the DNA cache. Validation/test use `j=500`. Unknown bases become all-zero one-hot vectors. Read/shape/value errors raise exceptions.

Defaults: seed 1234, batch size 64, learning rate 0.001, at most 50 epochs, patience 5. All peaks plus a freshly selected number of nonpeaks equal to 0.1 times peak count are sampled per training epoch. Validation uses a fixed, seed-selected 0.1 nonpeak-to-peak ratio, retaining all peaks; indices are sorted in cache order and the selected region-ID hash is recorded. The complete validation cache is preserved. Test prediction keeps every test row. The shared sampler uses NumPy `default_rng`; this is a declared implementation difference from the native global `RandomState` sampler. Both models learn `ln(1+sum(raw observed insertions in common1000))` with MSE. MoDL additionally learns a conditional profile with `-sum(y*ln(q))` averaged over examples; the multinomial constant is omitted. Count-loss weight defaults to `max(1, median centered training-peak counts /10)` and is recorded; `--count-weight` can instead fix the same published-archive value across models. This uses the original positive MoDL profile output normalized to unit mass, rather than replacing its architecture with a profile-logit head.

Original ExplaiNN includes per-unit BatchNorm: a final singleton training batch is skipped, and the scored example count is recorded. This affects at most one sampled training example per epoch. Evaluation keeps every row. Explicit `--max-train-steps` or `--max-valid-batches` labels a run as a **pilot**; its output must not be presented as the final benchmark.

Outputs: `metadata.json`, `history.json`, `best.pt`, `training_result.json`; prediction mode creates `test_predictions/prediction_counts.npy` in predicted natural-log count units, optionally MoDL `prediction_profiles.npy` as unit-sum shapes, and `prediction_metadata.json`. Prediction metadata includes the ordered region-ID SHA-256, target offsets 500:1500, `count_transform=ln1p_raw_insertion_count`, checkpoint hash and test-cache hashes. ExplaiNN has no native base-resolution profile and its profile result must be reported as N/A.

## Original architecture evidence and checks

The original factory is `atac_prediction.dna_atac_lightning_module.DNAATACMotifCNN(motif_kernels, num_atac_samples=1)`. Fixed weights are established at `src/atac_prediction/dna_atac_lightning_module.py:504–521`; all convolutions preserve length, scalar count features use global pooling (`:671–718`). Its native count head ends in Softplus (`:577–585`). The old wrapper crops central2048 (`:949–954`) and supervises a log10 count derived from normalized/smoothed ATAC (`:989–1005`); those training semantics are deliberately replaced in this external-context benchmark and declared in metadata.

The motif artifact is a dictionary containing `motif_kernels` tensor `(3222,31,4)`, motif names and a NumPy similarity matrix. The trainer loads this local artifact with a restricted NumPy allowlist under `weights_only=True`, extracts the kernels, and keeps them frozen. It does not deserialize an external untrusted model object.

CPU shape/backpropagation check of the **original imported class**, full 3222-kernel bank, batch1:

| Check | Observed |
| --- | --- |
| DNA input | `(1,4,2114)` |
| Original profile output | `(1,1,2114)` |
| Profile after `[...,557:1557]` | `(1,1000)` |
| Count | `(1,)` |
| Parameters | 3,113,174 total; 2,713,646 trainable |
| Frozen motif gradient | Absent, as required |
| Trainable gradients | All finite in the synthetic forward/backward check |

The official ExplaiNN import is `from explainn.models.networks import ExplaiNN`; `ExplaiNN(num_cnns=100,input_length=2114,num_classes=1,filter_size=19)` produces a scalar and has 3,038,301 parameters. A CPU batch2 forward/backward check against the vendored code passed with finite gradients. It uses random trainable filters; a PWM-initialized/frozen variant would be a separate baseline, not an automatic property of this constructor. Constructor and scalar-output behavior are verified against the [official implementation](https://github.com/wassermanlab/ExplaiNN/blob/master/explainn/models/networks.py).

Six focused tests passed with the existing PyTorch environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xf2217/Repos/motif_chip_model/.venv/bin/python \
  -m unittest discover -s reviewer_revision_xi_frank/gm12878_benchmark \
  -p test_train_torch.py -v
```

They cover exact jitter/target alignment, reverse complement, invalid values, deterministic training-only negative sampling, the common count/profile losses, chromosome overlap rejection, and constructing training data without any test cache. These tests and synthetic CPU model checks establish integration behavior; they do not establish GPU access or predictive performance.

## Local runtime observations

`/home/xf2217/Repos/motif_chip_model/.venv/bin/python` supplies torch 2.8.0, Lightning 2.5.5, NumPy 2.2.0, pyBigWig 0.3.24, pyfaidx 0.9.0.4 and pysam 0.24.0. Its original MoDL import succeeds, but `wandb` resolves as a namespace from the neighboring repository's `wandb/` directory; a complete installed W&B distribution was not found. This direct training loop does not use W&B logging. A future run that uses the manuscript Lightning logger must install and pin the real dependency rather than relying on this namespace import.

The neighboring `get_model_seq/.venv` has TensorFlow 2.21.0/Keras 3.14.0 and no installed ChromBPNet or ExplaiNN distribution at inspection. That environment is not evidence that the official ChromBPNet pipeline can run. The public ChromBPNet master requirements list TensorFlow 2.8.0 and NumPy 1.23.4, so native installation needs a compatible isolated environment or a validated container, with exact revision recorded. See the [official requirements](https://github.com/kundajelab/chrombpnet/blob/master/requirements.txt).

Official ChromBPNet combines a frozen bias model with the TF model and compiles multinomial profile loss plus count MSE. The bare `bpnet_model(...)` function in that source omits bias; using it alone would be a BPNet architecture baseline. A PyTorch reimplementation needs an explicit label and equivalence validation before it is represented as the official model. See [native bias integration](https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/training/models/chrombpnet_with_bias_model.py).

Local `/mnt/storage/get_data/hg38.fa` and `.fai`, plus `/mnt/storage/get_data/hg38-blacklist.v2.bed.gz`, exist. A targeted inspection of `encode_hg38atac_dense.zarr` found THP1/K562 groups, not a ready GM12878 group. This does not establish the absence of GM12878 elsewhere. The neighboring `get_model_seq` ChromBPNet scripts are GET fine-tuning integrations, not a verified native GM12878 training dataset or standalone trainer.

At float32, the MoDL motif activation alone uses about 27.2 MB per input example (`3222*2114*4` bytes). Training also retains residual activations and optimizer state; this arithmetic is not a measured GPU-memory peak. Begin with an explicitly labelled small CUDA pilot, then use batch64 only if memory permits. `--amp` requests bfloat16 and fails if unsupported. Large DNA caches stay memory mapped; no full cache is copied to GPU. A broken GPU driver/runtime must be repaired or bypassed through an authorized GPU host before any actual local GPU benchmark can be claimed.

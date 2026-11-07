# Local MoDL training and interpretation

This workflow accepts a local reference FASTA, called peaks BED, and either
adjusted 10x fragments or a raw insertion-count bigWig. It prepares the existing
Zarr formats, trains one pseudobulk track per checkpoint, and runs the same
batched deletion engine used by Model Studio. All computation and run outputs
stay on the machine executing the commands. The CLI disables W&B network logging.

## Environment

Use the repository environment described in [README.md](README.md#environment).
Caesarion is still required by the existing training/model-loading code. It must
be installed in that environment or available through `CAESAR_SRC` as described
there. This workflow does not install dependencies, download a reference, or
choose a genome assembly automatically.

```bash
source scripts/bundle_env.sh
"${PYTHON_CMD[@]}" scripts/modl.py doctor
```

`bash scripts/modl.sh ...` is the equivalent launcher using that same environment.
The examples below use it. Paths supplied to the CLI are resolved against the
invocation directory; the shell launcher first changes into the repository.
Use absolute paths for data outside the repository.

CLI results and errors are JSON on stdout. Success exits 0; workflow errors exit
2. `--help` describes each command. Training/inference logs go into the run
directory. Existing output directories are refused. A failed preparation leaves
its partial directory and `prepare.failure.json` for inspection; choose a new
output path after correcting the input.

## Input contract

| Input | Contract |
| --- | --- |
| Reference | FASTA or FASTA.gz; named records with IUPAC bases. A/C/G/T become one-hot channels in that order; ambiguous bases become zero. Supply the actual assembly explicitly. |
| Fragments | `fragments.tsv` or `.tsv.gz`, with chromosome, start, end, barcode, readSupport. Coordinates must already be adjusted for Tn5 as in 10x outputs. Additional columns are ignored. |
| Barcode groups | TSV with `barcode` and `group` headers. One row per selected barcode; unlisted barcodes are excluded. Group names become track IDs. Barcodes must match the fragment file exactly. |
| Peaks | Headerless BED or narrowPeak; the first three columns define 0-based, half-open intervals. Training windows are centered on the interval midpoint, not narrowPeak's summit-offset column. |
| Optional blacklist | BED in the same reference. Any training window overlapping it is excluded. Signal and library-size totals still include all selected insertion events. |
| Alternative signal | Nonnegative, raw per-base insertion counts in bigWig. Do not supply normalized CPM/RPKM, log signal, or fragment-span coverage. Missing positions/chromosomes become zero; present chromosome lengths must agree with FASTA. |

Each retained fragment row contributes one event at `start` and one at `end`.
This explicitly follows the existing Caesarion fragment converter. No additional
Tn5 shift is applied. An endpoint equal to chromosome length is outside the
base-indexed array and is reported and omitted. Column five includes PCR support
and is **not** used to weight molecules. The input is assumed deduplicated;
this CLI does not perform fragment/cell QC or deduplication.

The 10x conventions and column definitions are documented in the official
[fragments format](https://www.10xgenomics.com/support/software/cell-ranger-atac/2.1/tutorials/outputs/fragments-file).

Without cell metadata, `--pool-all --track-id sample` explicitly pools every input
barcode. Use this only for an already selected/QC-filtered population. Cell
grouping is a scientific input: the agent must obtain it from supplied metadata
or the user rather than inventing cell labels. This first workflow accepts one
fragment file at a time; prepare different samples separately to avoid barcode
collisions between samples.

## Prepare

Example barcode table:

```tsv
barcode	group
AAACGAAAGACGACTG-1	tumor
AAACGAAAGAGGTGCT-1	normal
```

Substitute actual files and an unused output path:

```bash
bash scripts/modl.sh prepare \
  --fasta /data/reference/genome.fa \
  --assembly hg38 \
  --fragments /data/sample/fragments.tsv.gz \
  --barcode-groups /data/sample/cells.tsv \
  --peaks /data/sample/peaks.bed \
  --blacklist /data/reference/blacklist.bed \
  --out /data/modl/prepared
```

Omit `--blacklist` when none is supplied and report that omission. For raw
bigWig, replace `--fragments` and `--barcode-groups` with
`--bigwig /data/sample/insertions.bw --track-id sample`.

Outputs: `genome.zarr`, `atac.zarr`, `peaks.tsv`, and `prepared.json`. The manifest
records input SHA-256 hashes, assembly, context size, library sizes, conversion
semantics, excluded/duplicate peaks, software versions, and source-code hashes.
The converter uses bounded buffers rather than whole-genome dense arrays in RAM.
Reference FASTA is streamed twice; hashing makes an additional pass over inputs.
Preparation can therefore be I/O intensive for large fragment/reference files.

## Validate and inspect the training plan

Choose validation and test chromosomes with peaks, leaving at least one other
chromosome for training. There is no random-split fallback. Do not choose
holdouts by looking at their eventual performance.

```bash
bash scripts/modl.sh validate \
  --data /data/modl/prepared --track-id tumor \
  --val-chr chr14 --test-chr chr10

bash scripts/modl.sh train \
  --data /data/modl/prepared --track-id tumor \
  --val-chr chr14 --test-chr chr10 \
  --out /data/modl/train-tumor --dry-run
```

Validation checks every peak's coordinates, store shapes, assembly metadata,
positive library size, and nonempty chromosome-disjoint splits. It inspects one
DNA/signal window per split; it is not an exhaustive rehash of prepared stores.
The dry run emits the exact underlying training argv and writes nothing.

## Smoke test, then train

```bash
bash scripts/modl.sh train \
  --data /data/modl/prepared --track-id tumor \
  --val-chr chr14 --test-chr chr10 \
  --out /data/modl/smoke-tumor --device cpu --batch-size 1 --smoke

bash scripts/modl.sh train \
  --data /data/modl/prepared --track-id tumor \
  --val-chr chr14 --test-chr chr10 \
  --out /data/modl/train-tumor --device cuda \
  --max-epochs 100 --batch-size 8 --seed 42
```

The smoke run executes one train and validation batch and exports `smoke.ckpt`.
It checks the execution path and provides no performance evidence. Full runs
write their selected checkpoints under `checkpoints/`; inspect `summary.json`'s
`best_checkpoint` rather than guessing a filename. `run.json` records exact argv,
input/code hashes, versions, status and the final summary. `train.log` contains
the training trace. Undefined scalar metrics are recorded as JSON null.

Full training requires adequate compute; `auto` selects available hardware and
`cpu` is supported. The CLI uses full precision and disables `torch.compile`
for a portable first workflow. It preserves the existing model, loss functions,
library normalization and signal smoothing. Reproducible seeds do not promise
bit-identical results across different hardware or software versions.

Train other barcode groups in separate run directories. These are independent
profile/count models. The existing phased differential model requires its
special target/peak definitions and remains available through the older
documented launchers; this CLI does not infer differential targets from labels.
The test chromosome is held aside, but this first CLI does not yet perform a
held-out benchmark. Training completion is not a generalization claim.

## Interpret a checkpoint

```bash
bash scripts/modl.sh interpret \
  --data /data/modl/prepared \
  --checkpoint /data/modl/train-tumor/checkpoints/CHOSEN.ckpt \
  --region chr1:100000-100500 \
  --out /data/modl/interpret-region1 --device cuda
```

Replace the example region and checkpoint with actual values. Region coordinates
are 0-based, half-open. The model receives the surrounding context recorded at
preparation (default 4,096 bp); deletion effects are scored/displayed within the
requested interval. The default is real 15-bp deletions at stride 1. Small CPU
execution checks can use a short region and larger `--stride`; record that
subsampling and do not present it as a dense scan.

Outputs are `result.json`, `deletions.csv`, `run.json`, and `interpret.log`.
`result.json` contains baseline predictions, deletion geometry, raw signed
effects, plot transformations, model metadata and provenance. CSV contains
`position`, `deletion_effect`, and `plot_effect_pct`. Load only trusted checkpoints.
The current command supports single-track regression/count checkpoints.

Deletion sensitivity is a model prediction. Motif overlays need actual motif
hits in the same reference and separately documented RNA filters; the CLI does
not assign TF identity from a sensitive interval alone. Existing motif-scanning
and annotation entrypoints remain available in `analysis/` and Model Studio.
Do not call a sensitive sequence a bound TF, causal regulator, or validated
enhancer without the corresponding evidence.

## Instructions for a coding agent

1. Read this file and run `doctor`. Establish the assembly, selected cell
   population, FASTA, fragments/signal, peaks, and available hardware from actual
   files. Ask only for missing scientific choices or unavailable inputs.
2. Prepare into an unused directory. Inspect `prepared.json` and report cell
   selection, molecule/endpoint conventions, dropped peaks and blacklist status.
3. Validate disjoint chromosome splits. Produce the dry-run argv and inspect
   sizes/hardware before launching the requested training. Do not upload data,
   enable remote logging, or start paid cloud compute.
4. Run a smoke test, then the authorized training. On failure inspect the log
   and recorded status; preserve partial outputs and use a new output directory.
5. Select the checkpoint by validation results. Interpret requested regions
   with explicit scoring heads and deletion settings. Keep test data out of
   checkpoint/hyperparameter selection.
6. Report commands, manifests, checkpoints, numerical artifacts, limitations and
   checks performed. Distinguish execution verification from predictive or
   biological validation.

## Workflow checks

```bash
source scripts/bundle_env.sh
"${PYTHON_CMD[@]}" -m unittest tests.test_modl_workflow
```

The tests use small synthetic files to check grouping, 10x molecule counts,
endpoint conventions, FASTA encoding, bigWig conversion, context/blacklist
filtering, split isolation, output protection, cache eviction, interpretation
coordinates and a read-only training plan. In the current agent sandbox Zarr's
asynchronous file operations stalled; the same tests passed outside that sandbox.
Treat that as an execution-permission issue rather than changing input data or
scientific parameters to work around it.

Verified on 2026-11-03 with Python 3.12.11, PyTorch 2.8.0 and CPU: all 30 tests
in `tests.test_modl_workflow`, `tests.test_model_studio_core` and
`tests.test_model_studio_http` passed; `scripts/validate_bundle.py` passed.
A synthetic three-chromosome, six-peak fixture with eight motif kernels completed
fragment conversion, one training epoch through the documented launcher,
checkpoint selection, checkpoint reload, and deletion JSON/CSV export. This
verifies execution only; large real-data runs, GPU execution and held-out
predictive evaluation remain unverified.

# CD200 Model Studio

The model studio is a local browser interface for checkpoint-driven DNA profile
inference and batched deletion analysis. Model weights and DNA remain on the
machine running the server.

For what the model predicts, how ATAC/DNA are processed, and how to train the
bundled checkpoints, see the main [`README.md`](README.md). This file focuses
on the website workflow.

## Start

From this bundle:

```bash
source scripts/bundle_env.sh
"${PYTHON_CMD[@]}" -m model_studio.server
```

Open <http://127.0.0.1:8787>.

The server binds only to `127.0.0.1` unless `--host` is explicitly changed.

## Workflow

1. Choose a workflow preset. `Manuscript promoter · differential` is the
   figure-parity workflow; `Promoter · regression model` is a separate
   regression-scoring workflow; `Custom checkpoint` exposes every field.
2. Pan or zoom the observed-ATAC browser. Shift-drag across its ruler to mark
   a region, then choose `Run selection / view` for one-click validation,
   checkpoint loading (when needed), and inference.
3. Alternatively, select `Auto`, `MPS`, `CUDA`, or `CPU` and load the
   checkpoint role(s) manually.
4. Validate the sequence. A genomic workflow records the model context,
   scoring core, and displayed interval separately.
5. Configure deletion size, stride, batch size, smoothing, and perturbation.
6. Run inference and inspect the profile and nucleotide-level deletion tracks.
7. Load motif/gene evidence and choose an explicit filter definition:
   expression (`baseMean`), single differential, consensus differential,
   precomputed gene list, or one TF.
8. Download the complete result as JSON or the deletion track as CSV.

`Auto` resolves devices in this order: CUDA, Apple MPS, CPU.

The manuscript preset uses:

- Phase-3 checkpoint, `differential` head: deletion effects.
- Regression checkpoint: current predicted accessibility profile.
- `chr16:45406665-45410761`: 4,096-bp model context.
- `chr16:45408000-45409426`: scoring core.
- `chr16:45407712-45409712`: central 2-kb displayed interval.
- `SCCApos.cd200_scc`: observed ATAC overlay.

The visible accessibility panel plots current regression-checkpoint inference
and observed ATAC only. Predicted accessibility, deletion effect, and motif
annotations use the same genomic domain and identical horizontal plot margins.

## Observed-ATAC genome browser

The embedded IGV.js browser is bundled locally. Its mm10 chromosome sizes,
observed `SCCApos.cd200_scc` signal, and inference sequence all come from the
bundle's Zarr stores. No package-CDN or hosted reference-genome request is
needed at runtime.

IGV displays conventional 1-based inclusive coordinates. Model Studio passes
IGV's raw 0-based half-open reference-frame or ROI coordinates to the existing
region input, so the selected bases are not shifted during inference.
`Run selection / view` uses the latest visible Shift-drag ROI; when none is
visible, it uses the current viewport. A run interval must be at most 2,000 bp;
the model still receives the fixed 4,096-bp surrounding context.

The track API reads only compressed chunks around the current viewport, applies
the same 20-bp observed-signal smoothing as inference, and bins wide views to
at most 50,000 bedGraph intervals. Panning outside the loaded window
transparently replaces the track with a new local view-sized window.

PyTorch checkpoints can execute Python during deserialization. Load only
checkpoints you trust; uploading a checkpoint copies it into
`model_studio/uploads/` with a unique name.

## Exact deletion-effect plot transformation

The browser plot uses the same plotting transformation as
`analysis/plot_scan_results.py::prepare_plot_effects` with
`use_absolute_effect=True` and
`normalize_to_max_predicted_atac=True`:

```text
plot effect (%) =
    100 * abs(raw nucleotide deletion effect)
        / max(abs(baseline prediction))
```

Raw signed effects remain in JSON and CSV. The transformed values, denominator,
formula, units, and source function are also stored in each result. The CSV
columns are `position`, `deletion_effect`, and `plot_effect_pct`.

## Motif and RNA annotations

Motif tables may be TSV, CSV, BED-like, or text-delimited files and need
chromosome, start, end, and TF/gene columns. Common alternatives such as
`chrom`, `gene_name`, and `motif_name` are detected automatically. Overlapping
hits for the same TF are merged using the manuscript plotting convention.
The built-in promoter workflows use
`model_studio/data/cd200_promoter_all_motifs_p1e-4.tsv`, an unfiltered
promoter-context scan at motif p-value <= 1e-4. Gene selection is applied
after loading this catalog, so changing filter modes can recover motifs that
are absent from the manuscript gene-list subset.

RNA tables may be CSV or TSV. Common gene, logFC, expression, and adjusted-p
columns are auto-detected, including the DESeq2 defaults `Unnamed: 0`,
`log2FoldChange`, `baseMean`, and `padj`.

- Expression mode applies only the `baseMean` threshold.
- Single differential mode applies `baseMean`, `|log2FoldChange|`, and `padj`.
- Consensus differential mode requires both RNA tables to pass expression and
  fold-change thresholds, the signs to agree, and `padj` to pass in at least
  one table.
- Gene-list mode uses the supplied precomputed list. The manuscript preset
  selects `shared_DEGs_1F_AF_tf_relaxed.csv`.
- Direct-TF mode bypasses RNA/gene-list filters.

All modes then apply the independent deletion-effect threshold. In the
manuscript preset, overlapping retained motifs are clustered into figure-style
rows and numbered members of the same family are deduplicated for display
(`SNAI1/SNAI3/SNAI2` becomes `SNAI`).

## Speed and reproducibility

Deletion variants are inferred in batches. The manuscript preset defaults to
batch size 32. It also defaults to display-only scanning: the model still sees
the full 4,096-bp context, but the app infers only deletion windows that can
contribute to the displayed 2-kb result. Each displayed value is therefore
identical to the corresponding value from a full-context scan, while roughly
half as many perturbed sequences are evaluated. Choose `Full model context`
when the complete 4,096-bp deletion export is needed.

Full-context batch-32 scans took about 8 seconds in warm local CUDA tests.
Display-only scanning is faster and matches the saved differential curve at
correlation 0.99997 with a mean absolute difference of about 0.042 percentage
points. Set batch size to 1 to use the legacy one-deletion-at-a-time numerical
path. The UI starts conservatively at batch 32 for CUDA, 8 for Apple MPS, and
4 for CPU; users with more available memory can raise it. MPS and CPU use the
same scientific definitions but have hardware-dependent runtimes.

Uncached observed-ATAC requests use the same direct local Zarr chunk reader as
the genome browser instead of importing the full Caesarion IO stack. This keeps
arbitrary-region setup small and avoids decoding unrelated chromosome chunks.

## Architecture

- `model_studio/adapters/`: model-family reconstruction and normalized
  prediction protocol.
- `model_studio/annotations.py`: motif-table normalization, same-TF merging,
  RNA joining, and exact deletion-effect overlap filtering.
- `model_studio/sequences.py`: sequence validation, zarr loading, and one-hot
  encoding.
- `model_studio/perturbations.py`: architecture-neutral batched deletion
  construction, scoring, distribution, and smoothing.
- `model_studio/jobs.py`: background progress, cancellation, and result
  artifacts.
- `model_studio/service.py`: application orchestration.
- `model_studio/server.py`: local HTTP API and static-file server.
- `model_studio/static/`: browser UI and canvas visualization.

The canonical model-adapter input is a float32 NumPy array shaped
`(batch, length, 4)`. Adapters normalize all outputs to named arrays shaped
`(batch, output_length)`.

## GET v2 pool16 transformer

Not included in this public release yet.

## Verification

Core tests:

```bash
source scripts/bundle_env.sh
"${PYTHON_CMD[@]}" -m unittest \
  tests.test_model_studio_core \
  tests.test_model_studio_http
```

The core test uses a deterministic fake profile adapter to prove that deletion
construction, batching, effect sign, nucleotide distribution, registry
switching, and the capabilities API do not depend on one checkpoint family.

Real-checkpoint parity test:

```bash
source scripts/bundle_env.sh
"${PYTHON_CMD[@]}" tests/validate_manuscript_runtime.py \
  --bundle "$PWD" --device auto --batch-size 32
```

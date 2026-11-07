# Get started with MoDL

MoDL predicts chromatin accessibility from DNA. You can train a model on your own
ATAC-seq data, score sequence deletions, and explore predictions in the local
Model Studio website.

## Install

Download `modl-0.2.2-source.tar.gz` and
`cd200_mutation_manuscript_bundle-0.2.2-py3-none-any.whl` into the same folder.
Use **Python 3.12**.

```bash
tar -xzf modl-0.2.2-source.tar.gz
cd modl-0.2.2
python3.12 -m venv .venv
.venv/bin/python -m pip install ../cd200_mutation_manuscript_bundle-0.2.2-py3-none-any.whl
.venv/bin/cd200-modl doctor
```

CPU, Apple Silicon MPS, and NVIDIA CUDA are supported for local interpretation.
On macOS, source-built dependencies may require Apple's Command Line Tools
(`xcode-select --install`).

## Use your own data

Start with a reference FASTA, called peaks in BED format, and either adjusted
10x fragments or a raw insertion-count bigWig. Open the extracted folder in
Codex or Claude Code and paste:

> Read AGENTS.md and AGENT_WORKFLOW.md. Check my inputs, help me prepare my data,
> run a small training test, and open Model Studio with my checkpoint. Ask me for
> the reference assembly, cell groups, and any missing inputs before analysis.

The CLI provides `prepare`, `validate`, `train`, and `interpret` commands.
Run `.venv/bin/cd200-modl --help` or follow [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md).
Use Model Studio for differential-head analysis and motif/RNA filtering.

## Explore the supplied Cd200 models

Download `modl_checkpoints_20261107.tar.gz` beside the source archive. From inside
the extracted source folder, install the two Cd200 checkpoints:

```bash
mkdir -p analysis/checkpoints
tar -xzf ../modl_checkpoints_20261107.tar.gz --strip-components=2 -C analysis/checkpoints weights/historical_MoDL
```

The Cd200 preset also needs the full reference and observed ATAC stores at
`data/zarr/mm10.zarr` and `data/zarr/cd200_scc_dense_new.zarr`. These genomic data
stores are separate inputs and are not included in the downloads. Store formats
and input settings are documented in [MODEL_STUDIO.md](MODEL_STUDIO.md).

Once the inputs are in place:

```bash
.venv/bin/python -m model_studio.server
```

Open **http://127.0.0.1:8787**, choose **Manuscript promoter · differential**, and
select **Run selection / view**. The app shows observed accessibility, the
predicted profile, deletion effects, and filtered motif annotations. Computation
runs on your machine; Auto selects CUDA, then Apple MPS, then CPU.

## Optional benchmark files

`modl_exact_region_splits_20261107.tar.gz` contains the exact evaluation regions.
The checkpoint archive also contains MoDL, ChromBPNet, and ExplaiNN benchmark
models. These are separate from the two Cd200 Model Studio checkpoints.

License: MIT. Software creators: Xi Fu and Anqi Shao.

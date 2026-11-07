# Genome Motif Analysis Tutorial

A comprehensive guide to using Caesar's genome motif analysis module for transcription factor binding site analysis, co-occurrence patterns, and spatial relationships.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Analysis Modes](#analysis-modes)
- [Common Usage Examples](#common-usage-examples)
- [Advanced Features](#advanced-features)
- [Output Files](#output-files)
- [Performance Tips](#performance-tips)

## Overview

The genome motif analysis module scans genomic sequences for transcription factor (TF) binding motifs and analyzes their spatial relationships. It supports multiple analysis modes:

- **Co-occurrence analysis**: Analyze spatial relationships between all motif pairs
- **Target-centered analysis**: Focus on patterns around one specific motif
- **Signal accumulation**: Aggregate genomic signals (e.g., ATAC-seq) around motif hits
- **Sequence content analysis**: Examine DNA composition around binding sites
- **Hit collection**: Extract and store sparse motif hits for downstream analysis
- **Full-curve subset reuse**: Cache query-motif hits once for overlapping peak-set families and replay one-to-all motif curves across many subsets
- **Flexible p-value thresholds**: FIMO-style p-value calculation supports any p-value threshold (not limited to predefined values)

## Quick Start

### Basic whole-genome analysis

Analyze all chromosomes for motif co-occurrence patterns:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr /path/to/genome.zarr \
    --output results/
```

### Test run on single chromosome

Quick test on chromosome 2:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr /path/to/genome.zarr \
    --output results/ \
    --test
```

### Analyze specific chromosomes

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr /path/to/genome.zarr \
    --chr chr1 chr2 chrX \
    --output results/
```

## Analysis Modes

### 1. Co-occurrence Analysis (Default)

Analyzes spatial relationships between all pairs of transcription factors. This reveals which TFs tend to bind near each other.

```bash
# Basic co-occurrence analysis
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --output results/cooccurrence/

# With visualization
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --output results/cooccurrence/ \
    --plot \
    --plot-pairs "CTCF,TP53;NFKB1,STAT1"
```

**Output**: A 3D tensor of shape `(n_motifs, n_motifs, 2*window_size)` showing how often each motif pair occurs at various distances.

### 2. Target-Centered Analysis (One-to-All)

Focuses analysis around a specific motif of interest. Useful for studying what other TFs bind near your target TF.

```bash
# Analyze patterns around CTCF binding sites
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --one-to-all CTCF \
    --output results/ctcf_centered/ \
    --plot

# Using motif index instead of name
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --one-to-all 0 \
    --output results/motif0_centered/
```

**Output**: A 2D tensor of shape `(n_motifs, 2*window_size)` showing the average pattern of all motifs around the query motif.

### Scalable Full-Curve One-to-All Reuse

Use the scalable profile workflow when the question is **“how does motif
organization differ between peak groups?”** It produces one complete profile
for every group: `(group, motif, position)`. Build on it in this order:

1. **Prepare groups.** Supply a TSV with `subset_id, chrom, start, end`, or use
   `prepare-chip-atlas-groups` to create one from a ChIP-Atlas BED.
2. **Profile groups.** Run `profile-peak-groups` once, rather than running one
   legacy command per group.
3. **Compare exclusive loci, if needed.** Run `compare-peak-sets` on individual
   BED/narrowPeak files. It constructs a shared union of overlap-connected
   loci, then emits ordered A-minus-B groups.
4. **Rank motif differences.** Run `rank-context-motifs` on the completed
   profile Zarr and the same group-membership TSV.

The descriptive commands are aliases; the older backend names remain valid.
Peak-set inputs are centered to `--peak-set-window-bp` (2,000 bp by default).
`--min-peaks 1000` retains groups with strictly more than 1,000 union loci.
Every direct peak-set run writes `peak_set_memberships.tsv`,
`peak_set_inventory.tsv`, and `peak_set_union_loci.tsv` for audit.

```bash
python -m caesar.analysis.genome_motif_analysis compare-peak-sets \
    --peak-set SJSA1=SJSA1_Nutlin.narrowPeak \
    --peak-set A2780=A2780_Nutlin.narrowPeak \
    --min-peaks 1000 \
    --output-dir results/nutlin_pairwise_full_curves \
    --query-motif P53.H13CORE.0.P.B \
    --window 500 \
    --output-accumulator auto
```

The profile workflow caches query-motif hits once over the shared peak universe, then builds complete profiles for every group. On the 100-subset Chip-Atlas TP53 benchmark, the measured strict replay-only run kept the complete output checksum unchanged while reducing subset-update wall time from about 33.3s to about 10.5s. Only complete profile tensors are considered valid biological outputs.

The current bottleneck is sequence extraction, not motif convolution. The block provider bypasses generic `get_track` query/aggregation for internally chunked dense chromosome arrays such as unmasked hg38, then reuses raw zarr chunks through an exact CPU LRU controlled by `--provider-max-sequence-cache-gb` with a 0.5GB default. In the promoted replay, the full-score provider spent about 13.7s fetching unmasked hg38 sequence chunks out of 17.2s provider wall, while forward plus reverse motif scans took about 2.6s combined. The previous direct chromosome-array replay had 33.8s sequence fetch and 37.5s provider wall, so the chunk cache cuts provider time by about 20.4s while preserving the complete output checksum. A 2GB chunk-cache A/B also preserved the checksum but did not materially improve wall time; a stricter 100kb block-gap A/B preserved the checksum but regressed workflow wall from 111.8s to 137.0s because it doubled loaded bp to 427.8M.

An optional `--anchor-contribution-mode stack_all` mode is retained for throughput experiments, but it is not the current recommendation: on the pre-preload benchmark it reached about 121.8s, now slower than the 92.4s strict path, and it is labeled `PASS_WARN` because the raw max absolute delta versus strict output is about 0.00195, above the strict 1e-4 gate.

An optional `--output-accumulator auto` mode uses Torch on CUDA when safe; otherwise it uses NumPy. It keeps the full output tensor, but carries a floating-point ordering warning. Use `numpy` when strict numerical equality matters.

```bash
python -m caesar.analysis.genome_motif_analysis full-curve-reuse \
    --regions-tsv subset_memberships.tsv \
    --output-dir results/p53_full_curve_reuse \
    --query-motif P53.H13CORE.0.P.B \
    --window 500 \
    --p-value-threshold p0.0001 \
    --output-accumulator auto
```

Each direct backend run writes `workflow_summary.json`,
`eureka_backend_payload.json`, and `eureka_backend_manifest.json` in the output
directory. The manifest is the run contract for Eureka callers: it declares
unmasked hg38, aligned HOCOMOCO PT motifs, complete full-curve tensor output,
the active query-motif hit-cache construction method, and
many-subset full-curve aggregation.

### 3. Signal Accumulation (All-to-One)

Aggregates genomic signals (ATAC-seq, ChIP-seq, etc.) around all motif binding sites.

```bash
# Accumulate ATAC-seq signal around each motif
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --all-to-one \
    --signal-zarr atac_signal.zarr \
    --output results/signal_accumulation/

# Accumulate signal for a single cell type only (reduced memory)
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --all-to-one \
    --signal-zarr celltype_atac.zarr \
    --signal-celltype-id CD4_T \
    --output results/signal_accumulation_single_ct/
```

**Output**:
- With multiple cell types: a 3D tensor `(n_motifs, 2*window_size, n_celltypes)`.
- With `--signal-celltype-id`: a 2D tensor `(n_motifs, 2*window_size)` for that single cell type.

### 4. Sequence Content Analysis

Analyzes DNA sequence composition around binding sites (useful for discovering sequence preferences).

```bash
# Analyze sequence content around all motifs
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --sequence-accumulation \
    --output results/sequence_content/

# Combined with narrowPeak regions
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --narrowpeak peaks.bed \
    --sequence-accumulation \
    --output results/peak_sequences/
```

**Output**: A 2D tensor of shape `(4, 2*window_size)` showing nucleotide frequencies (A, C, G, T) around binding sites.

### 5. Hit Collection (Sparse Motif Hits)

Extracts and stores motif scores for each motif in each peak region. Supports two aggregation modes:

**Max Mode (Default)**: Stores the maximum score per motif per peak
```bash
# Collect sparse hits with maximum scores (default)
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --narrowpeak peaks.bed \
    --collect-hits \
    --no-accumulation \
    --hit-aggregation max \
    --output results/hits/
```

**Sum Mode**: Stores the sum of all scores above threshold per motif per peak
```bash
# Collect sparse hits with sum of scores
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --narrowpeak peaks.bed \
    --collect-hits \
    --no-accumulation \
    --hit-aggregation sum \
    --output results/hits/
```

**Also works with tiling mode**:
```bash
# Collect hits from genome-wide tiles
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --tiling \
    --collect-hits \
    --no-accumulation \
    --hit-aggregation sum \
    --output results/hits/
```

**Output**: A sparse zarr store at `results/hits/motif_hits.zarr` containing:
- **Max mode**: Maximum score per motif per peak (includes position of max score)
- **Sum mode**: Sum of all scores above threshold per motif per peak (rel_pos = -1 indicates aggregation)

## Common Usage Examples

### Example 1: Analyze CTCF with ChIP-seq Data

Use ReMap database to automatically download CTCF ChIP-seq peaks:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --one-to-all CTCF \
    --remap \
    --plot \
    --output results/ctcf_remap/
```

### Example 2: Custom Peak Regions

Analyze motifs within your own peak regions:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --narrowpeak my_peaks.bed \
    --extend 500 \
    --output results/custom_peaks/
```

### Example 3: Motif Subset Analysis

Analyze only specific motifs using prefix matching, indices, or ranges:

```bash
# By TF family prefix
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --motifs "CTCF,TP53,STAT" \
    --output results/tf_families/

# Mix of prefixes, indices, and ranges
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --motifs "CTCF,TP53,0-5,10" \
    --output results/mixed_selection/
```

### Example 4: Multi-Celltype Signal Analysis

Analyze ATAC-seq signal from multiple cell types:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --all-to-one \
    --signal-zarr celltype_atac.zarr \
    --output results/multicelltype/
```

**Notes**:
- The output will have shape `(n_motifs, 2*window_size, n_celltypes)` when using multi-dimensional signal data.
- To reduce memory usage and load time, select a single cell type with `--signal-celltype-id <ID>`.

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --all-to-one \
    --signal-zarr celltype_atac.zarr \
    --signal-celltype-id CD4_T \
    --output results/one_celltype/
```

### Example 5: Strand-Specific Analysis

Analyze only the forward strand with reverse complement motifs:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --strand-specific \
    --output results/strand_specific/
```

### Example 6: Collect Hits with Sum Aggregation

Collect sparse motif hits using sum of scores instead of max:

```bash
# Collect hits with sum aggregation (sums all scores above threshold per peak)
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --narrowpeak peaks.bed \
    --collect-hits \
    --hit-aggregation sum \
    --output results/hits_sum/

# Works with tiling mode too
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --tiling \
    --collect-hits \
    --hit-aggregation sum \
    --output results/tiles_sum/
```

### Example 7: High-Confidence Hits Only

Use custom p-value threshold (any value supported):

```bash
# Use stricter threshold
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --p-value-threshold p0.001 \
    --output results/high_confidence/

# Use very strict threshold
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --p-value-threshold p1e-5 \
    --output results/very_strict/

# Use explicit score threshold instead of p-value
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --score-threshold 5.0 \
    --output results/score_based/
```

**P-value Threshold System**:
- **Default**: Uses FIMO p-value mapping to compute accurate score thresholds for any p-value
- **Flexible**: Supports any numeric p-value (e.g., `p0.001`, `p0.0001`, `p1e-5`, `p0.01`)
- **Fallback**: Use `--no-pvalue-mapping` to use annotation file thresholds (limited to `p0.001`, `p0.0005`, `p0.0001`)
- **Score-based**: Use `--score-threshold <value>` for direct score cutoffs (overrides p-value)

## Advanced Features

### P-Value and Score Thresholds

The system uses FIMO-style p-value calculation to provide accurate thresholds for any p-value:

```bash
# Use any p-value threshold (default: uses computed mappings)
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --p-value-threshold p0.0001 \
    --output results/

# Use a very strict threshold
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --p-value-threshold p1e-6 \
    --output results/

# Disable p-value mapping, use annotation file thresholds only
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --p-value-threshold p0.0001 \
    --no-pvalue-mapping \
    --output results/

# Use explicit score threshold (same for all motifs)
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --score-threshold 4.5 \
    --output results/
```

**How it works**:
- P-value mappings are computed once per motif using FIMO algorithm and cached
- Cached mappings stored in `~/.gcell_data/annotations/hocomoco/pvalue_mappings/`
- Each motif gets its own accurate score threshold for the specified p-value
- Much more accurate than using fixed thresholds from annotation files

### Custom Analysis Window

Adjust the spatial window size for analysis:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --analysis-window 1000 \
    --output results/large_window/
```

### Batch Size Optimization

Increase batch size for better GPU utilization (requires more memory):

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --batch-size 4 \
    --output results/
```

### Full Precision Mode

Use 32-bit precision for maximum accuracy (slower):

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --fp32 \
    --output results/
```

### Select a single cell type from CelltypeDenseZarrIO

When supplying a multi-celltype signal zarr, you can restrict loading and querying to a single `celltype_id` to save memory and speed up I/O:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --all-to-one \
    --signal-zarr celltype_atac.zarr \
    --signal-celltype-id CD4_T \
    --output results/signal_single_ct/
```

This returns a 2D result `(n_motifs, 2*window_size)` for that cell type.

### Hit Counting Only

Disable spatial accumulation to just count motif hits:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --no-accumulation \
    --output results/counts_only/
```

### Visualization Options

Generate comprehensive plots:

```bash
python -m caesar.analysis.genome_motif_analysis \
    --genome-zarr genome.zarr \
    --plot \
    --plot-types cooccurrence,power_spectrum,autocorrelation \
    --plot-pairs "CTCF,TP53;NFKB1,STAT1" \
    --plot-smooth 3 \
    --plot-max-delay 500 \
    --output results/with_plots/
```

## Output Files

The analysis generates several output files in the specified output directory:

### Standard Outputs

- **`cooccurrence_matrix.npy`**: Main co-occurrence tensor (3D for all-to-all, 2D for one-to-all)
- **`motif_names.txt`**: List of motif names in order
- **`analysis_stats.json`**: Summary statistics and metadata

### Optional Outputs (when `--plot` is enabled)

- **`cooccurrence_heatmap.png`**: Heatmap of motif co-occurrence patterns
- **`power_spectrum.png`**: Frequency domain analysis
- **`autocorrelation.png`**: Temporal correlation patterns
- **`quality_scatter.png`**: Motif quality metrics

### Sparse Hit Collection (when `--collect-hits` is enabled)

- **`motif_hits.zarr/`**: Zarr store containing sparse motif hits
  - Coordinates: peak regions
  - Variables: motif scores
  - Format depends on `--hit-aggregation` mode:
    - **`max` (default)**: Maximum score per motif per peak (includes position of max)
    - **`sum`**: Sum of all scores above threshold per motif per peak (rel_pos = -1)

## Performance Tips

### Memory Management

1. **Reduce batch size** if you encounter out-of-memory errors:
   ```bash
   --batch-size 1
   ```

2. **Use mixed precision** (default) for better memory efficiency:
   ```bash
   # Mixed precision is default, explicitly disable with:
   --fp32
   ```

3. **Analyze chromosomes sequentially** (automatic) rather than all at once

### Speed Optimization

1. **Use GPU** when available (automatically detected)

2. **Increase batch size** if you have sufficient GPU memory:
   ```bash
   --batch-size 4
   ```

3. **Reduce analysis window** for faster processing:
   ```bash
   --analysis-window 250
   ```

4. **Use narrowPeak regions** instead of whole genome for targeted analysis

### Typical Processing Times

- Single chromosome (chr1): ~5-10 minutes on GPU
- Whole genome (chr1-22): ~2-3 hours on GPU
- narrowPeak regions (~50k peaks): ~10-30 minutes on GPU

*Times vary based on hardware, motif count, and analysis parameters*

## Troubleshooting

### Common Issues

**Issue**: Out of memory errors
```bash
# Solution: Reduce batch size
--batch-size 1
```

**Issue**: No GPU detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Issue**: ReMap download fails
```bash
# Use explicit gene name
--remap --remap-gene FOXA1
```

**Issue**: P-value mapping computation fails
```bash
# Fall back to annotation file thresholds
--no-pvalue-mapping
```

**Issue**: Want to use custom p-value threshold
```bash
# Any numeric p-value is supported (default uses FIMO mapping)
--p-value-threshold p1e-5
```


## Python API Usage

You can also use the module programmatically:

```python
from caesar.analysis.genome_motif_analysis import analyze_genome_sequences

# Run analysis with custom p-value threshold
analyze_genome_sequences(
    genome_zarr_path="/path/to/genome.zarr",
    target_chromosomes=["chr1", "chr2"],
    output_directory="results/",
    batch_size=2,
    use_mixed_precision=True,
    enable_accumulation=True,
    analysis_window_size=500,
    query_motif_name="CTCF",
    p_value_threshold="p1e-5",  # Any p-value supported
    use_pvalue_mapping=True,  # Default: uses FIMO p-value mapping
    plot_config={
        "enable_plotting": True,
        "plot_types": ["cooccurrence", "power_spectrum"],
        "motif_pairs": [("CTCF", "TP53")],
    }
)

# Or use explicit score threshold
analyze_genome_sequences(
    genome_zarr_path="/path/to/genome.zarr",
    output_directory="results/",
    score_threshold=5.0,  # Overrides p-value threshold
    # ... other parameters
)
```

## Questions?

For more information:
- Check the [main Caesar documentation](../../../README.md)
- See example workflows in `workflows/motif_processing.yaml`
- Review the source code in [main.py](main.py)

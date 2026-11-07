# Caesar Preprocessing Module

## Overview

The Caesar preprocessing module provides a unified framework for processing genomic data through modular processors and configurable workflows. It supports multiple storage backends with different performance and compatibility characteristics.

## Storage Backends

Caesar supports two primary storage backends:

### 1. **Zarr Backend** (Default)
- **Full compatibility** with all processors
- **Flexible** - supports in-place operations
- **Moderate efficiency** - good compression, reasonable speed
- **Best for**: Standard analysis pipelines, smaller datasets

### 2. **BPCells Backend** (High Performance)
- **Limited compatibility** - see compatibility matrix below
- **High efficiency** - optimized C++ implementation
- **Excellent performance** - fast queries, low memory
- **Best for**: Large datasets, storage optimization

## Backend Compatibility

### ⚠️ IMPORTANT: Not all processors work with all backends!

| Processor | Zarr Support | BPCells Support | Notes |
|-----------|--------------|----------------|-------|
| FragmentProcessor | ✅ Full | ❌ No | Use BPCellsProcessor for BPCells |
| BPCellsProcessor | ❌ No | ✅ Full | BPCells-specific processor |
| ClusteringProcessor | ✅ Full | ✅ Full | **BPCells via AnnData (v2.1.0+)** |
| PseudoBulkProcessor | ✅ Full | ❌ No | Use BPCells celltype_dense instead |
| PeakCallingProcessor | ✅ Full | ✅ Yes | BPCells dense formats only |
| NormalizationProcessor | ✅ Full | ⚠️ Partial | BPCells stores factors only |
| FootprintProcessor | ✅ Full | ❌ No | Use BPCells IO methods instead |
| LibrarySizeProcessor | ✅ Full | ✅ Yes | BPCells dense formats |
| **Footprinting (IO)** | N/A | ✅ Full | Via `get_genome_wide_footprint_all_motifs()` |

**For detailed compatibility information, see:**
- [`docs/backends.md`](../../../docs/backends.md) - **Comprehensive backend comparison guide**
- [`docs/bpcells_backend_support.md`](../../../docs/bpcells_backend_support.md) - BPCells-specific details

## Quick Start

### Command Line Interface

Caesar provides a modern CLI with automatic processor discovery:

```bash
# View all available commands
caesar --help

# Get help for any specific command
caesar fragment --help
caesar workflow --help
```

### Available Commands

#### **Data Processing**
```bash
# Fragment file processing (auto-detects single-cell vs bulk)
caesar fragment -i sample.fragments.tsv.gz -o output/

# BAM/CRAM file processing
caesar bam -i sample.bam -o coverage.zarr --tn5-shift

# BPCells format processing (high performance)
caesar bpcells -i fragments.tsv.gz --output-type celltype_dense
```

#### **Analysis & Clustering**
```bash
# Single-cell clustering
caesar clustering -i single_cell.zarr --resolution 0.8

# Pseudobulk aggregation
caesar pseudobulk -i single_cell.zarr --groupby labels
```

#### **Peak Analysis**
```bash
# Peak calling
caesar peak-calling -i pseudobulk.zarr --qvalue 0.05

# Quantile normalization
caesar normalization -i pseudobulk.zarr
```

#### **Motif Analysis**
```bash
# Process motif scan results
caesar motif --motif-scan-results vierstra.bed.gz

# Create ML training datasets
caesar motif-curator --motif-zarr motifs.zarr --genome genome.zarr
```

#### **Advanced Analysis**
```bash
# TF footprint detection
caesar footprint -i normalized.zarr

# Library size calculation
caesar library-size -i data.zarr
```

#### **Workflows**
```bash
# Complete analysis pipelines
caesar workflow -c config.yaml
caesar workflow -c config.yaml --dry-run
caesar workflow -c config.yaml workflow.name=bulk
```

### Standard Zarr Workflow
```bash
# Full pipeline with Zarr backend
caesar workflow -c config.yaml
```

### BPCells Workflow
```bash
# BPCells for storage only
caesar workflow -c workflows/single-cell-bpcell-complete.yaml

# Note: Limited downstream processing available
```

### Hybrid Approach (Recommended for Large Data)
```yaml
# Step 1: Store in BPCells
workflow:
  name: single-cell-bpcells
  
# Step 2: Convert to Zarr for analysis
# (Use IO classes to read BPCells and save as Zarr)

# Step 3: Run full pipeline on Zarr data
```

## Choosing a Backend

```
Dataset > 1M cells?
├─ YES → BPCells for storage, convert to Zarr for analysis
└─ NO → Use Zarr throughout

Need clustering/footprinting?
├─ YES → Must use Zarr (or convert from BPCells)
└─ NO → BPCells may be sufficient

Memory constrained?
├─ YES → BPCells highly recommended
└─ NO → Either backend works
```

## Available Processors

### Data Processors
- **FragmentProcessor**: Process fragment files → Zarr
- **BPCellsProcessor**: Process fragment files → BPCells  
- **BAMProcessor**: Process BAM files → Zarr

### Transform Processors  
- **ClusteringProcessor**: Add cell clusters (Zarr only)
- **PseudoBulkProcessor**: Aggregate cells (Zarr only)

### Analysis Processors
- **PeakCallingProcessor**: Call peaks (both backends, restrictions apply)
- **NormalizationProcessor**: Normalize signals (Zarr only)
- **FootprintProcessor**: Detect TF footprints (Zarr only)
- **LibrarySizeProcessor**: Annotate library sizes (Zarr only)

## Workflows

Pre-configured workflows handle backend selection:

| Workflow | Backend | Use Case |
|----------|---------|----------|
| single-cell | Zarr | Standard scATAC-seq pipeline |
| single-cell-bpcells | BPCells | Efficient storage only |
| bulk | Zarr | Bulk ATAC/DNase-seq |
| reprocessing | Zarr | Re-analyze existing data |
| footprint | Zarr | Add TF footprints |

## Configuration Examples

### Zarr Backend (Full Pipeline)
```yaml
workflow:
  name: single-cell
  
data:
  fragments:
    - path: data.fragments.tsv.gz
      
output:
  format: zarr  # Default
```

### BPCells Backend (Storage)
```yaml
workflow:
  name: single-cell-bpcells
  
bpcells:
  output_type: single_sparse  # or celltype_dense, signal_dense
  
output:
  format: bpcells
```

## Best Practices

1. **Start with your end goal**: If you need clustering/footprinting, plan for Zarr
2. **Use BPCells for storage**: Especially for large datasets
3. **Convert as needed**: BPCells → Zarr for analysis
4. **Check compatibility**: Not all processors work with all backends
5. **Read the docs**: Backend-specific limitations and workarounds

## Troubleshooting

### "Processor X doesn't work with BPCells!"
- This is expected - see compatibility matrix
- Convert to Zarr first, or use alternative approach

### "BPCells installation failed"
- Requires C++ compiler
- See BPCells documentation for platform-specific instructions

### "Out of memory with Zarr"
- Consider using BPCells for storage
- Process in smaller chunks
- Increase available memory

## Additional Resources

- [Backend Comparison Guide](../../../docs/backends.md) - Comprehensive Zarr vs BPCells guide
- [BPCells Backend Documentation](../../../docs/bpcells_backend_support.md)
- [Workflow Examples](../../../workflows/)
- [Configuration Guide](../../../docs/configuration.md)
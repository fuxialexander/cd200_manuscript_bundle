#!/usr/bin/env python3
"""
Command-line interface for on-demand, FIMO-based motif scanning.

This tool performs fast motif scanning on a specified genomic region using a
pre-trained model's convolutional kernels and a pre-cached FIMO p-value mapping.

It offers two ways to define the scanning region:
1.  Directly specifying a genomic region with the `--region` argument.
    Example:
        --region chr2:136117414-136117614

2.  Automatically identifying a high-impact region from a deletion scan by
    providing a nucleotide scores CSV file with the `--scores-csv` argument.
    Example:
        --scores-csv results/dlbcl_analysis/chr2_136117500_136117800_deletion_regression_nucleotide.csv

Full Example:
    python motif_scan_cli.py \\
        --region chr16:45408000-45409426 \\
        --checkpoint analysis/checkpoints/ATAC-Sequential-20250919-022148_phase3_final_model.ckpt \\
        --genome-zarr data/zarr/mm10.zarr \\
        --motif-names artifacts/motifs_with_rc_aligned.txt \\
        --fimo-cache analysis/fimo_mappings_cache.pkl \\
        --verbose
"""

import os
import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mutation_analysis.analyzer import MutationAnalyzer, FimoMotifScanner
from mutation_analysis.dataset import MutationDataset
from mutation_analysis.motif_cache import FimoMappingsCache
from mutation_analysis.inference import ModelInference
from mutation_analysis.motif_filters import create_annotation_filter, PromoterAccessibilityFilter, get_hocomoco_motif_mapping

def _normalize_deg_columns(deg_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DEG column names used by the CLI."""
    if 'Unnamed: 0' in deg_df.columns:
        deg_df = deg_df.rename(columns={'Unnamed: 0': 'gene_name'})
    return deg_df


def _filter_deg_dataframe(
    deg_df: pd.DataFrame,
    base_mean_threshold: float,
    log2fc_threshold: float,
) -> pd.DataFrame:
    """Filter a DEG table to genes that pass expression and fold-change thresholds."""
    required_columns = {'gene_name', 'baseMean', 'log2FoldChange'}
    missing_columns = required_columns - set(deg_df.columns)
    if missing_columns:
        missing_str = ', '.join(sorted(missing_columns))
        raise ValueError(f"DEG file is missing required columns: {missing_str}")

    mask = (
        (deg_df['baseMean'] > base_mean_threshold) &
        (deg_df['log2FoldChange'].abs() > log2fc_threshold)
    )

    return deg_df.loc[mask].copy()


def _get_same_sign_deg_genes(deg1_filtered: pd.DataFrame, deg2_filtered: pd.DataFrame) -> set[str]:
    """Return genes that pass both DEG filters and change in the same direction."""
    deg1_fc = (
        deg1_filtered
        .dropna(subset=['gene_name', 'log2FoldChange'])
        .drop_duplicates(subset=['gene_name'])
        .set_index('gene_name')['log2FoldChange']
    )
    deg2_fc = (
        deg2_filtered
        .dropna(subset=['gene_name', 'log2FoldChange'])
        .drop_duplicates(subset=['gene_name'])
        .set_index('gene_name')['log2FoldChange']
    )

    shared_genes = deg1_fc.index.intersection(deg2_fc.index)
    same_sign_mask = np.sign(deg1_fc.loc[shared_genes]) == np.sign(deg2_fc.loc[shared_genes])
    return set(shared_genes[same_sign_mask])


def _get_genes_passing_padj(
    deg_df: pd.DataFrame,
    padj_threshold: float,
) -> set[str] | None:
    """Return genes passing the padj threshold, or None if padj is unavailable."""
    if 'padj' not in deg_df.columns:
        return None

    padj_filtered = deg_df.loc[deg_df['padj'] < padj_threshold, 'gene_name'].dropna()
    return set(padj_filtered.unique())


def create_motif_filter(args):
    """Create motif filter from command-line arguments."""
    if args.motif_filter == 'none':
        return None
    if args.motif_filter == 'promoter':
        print("Creating Promoter Accessibility Filter...")
        return PromoterAccessibilityFilter(cell_type=args.promoter_cell_type)

    if args.motif_filter == 'gene_list':
        if not args.gene_list_file:
            raise ValueError("--gene-list-file is required for 'gene_list' filter.")
        print("Creating filter from gene list...")
        
        # Read the gene list. Prefer an explicit gene column when present so
        # richer CSV exports can be used directly.
        gene_df = pd.read_csv(args.gene_list_file)
        if 'gene' in gene_df.columns:
            gene_series = gene_df['gene']
        elif 'gene_name' in gene_df.columns:
            gene_series = gene_df['gene_name']
        else:
            gene_series = gene_df.iloc[:, 0]
        input_gene_list = set(gene_series.dropna().unique())

        sample_genes = list(input_gene_list)[:10] if input_gene_list else []
        is_mouse_format = False
        if sample_genes:
            mouse_format_count = sum(
                1 for gene in sample_genes
                if isinstance(gene, str) and gene and gene[0].isupper() and (len(gene) == 1 or gene[1:].islower())
            )
            is_mouse_format = mouse_format_count > len(sample_genes) / 2

        if is_mouse_format:
            print("   ✓ Detected mouse gene format, converting HOCOMOCO mapping to mouse format...")
            motif_to_gene = get_hocomoco_motif_mapping(to_mouse=True)
        else:
            print("   ✓ Using human gene format for HOCOMOCO mapping...")
            motif_to_gene = get_hocomoco_motif_mapping(to_mouse=False)

        hocomoco_tfs = set(motif_to_gene.values())
        normalized_hocomoco_tfs = {tf.upper(): tf for tf in hocomoco_tfs}
        normalized_input_gene_list = {
            gene.upper() for gene in input_gene_list
            if isinstance(gene, str) and gene
        }

        # Find the intersection to get the list of TFs to whitelist
        tfs_to_keep = sorted(
            normalized_hocomoco_tfs[gene]
            for gene in normalized_input_gene_list.intersection(normalized_hocomoco_tfs)
        )
        
        print(f"   ✓ Found {len(tfs_to_keep)} TFs in both the gene list and HOCOMOCO.")
        if args.verbose and tfs_to_keep:
            print(f"   - TFs to keep: {', '.join(tfs_to_keep[:10])}{'...' if len(tfs_to_keep) > 10 else ''}")
            
        # Use the whitelist filter with the generated TF list
        # annotation_file is not needed when tf_list is provided to whitelist
        return create_annotation_filter(
            annotation_file=None,
            filter_type='whitelist',
            tf_list=tfs_to_keep
        )

    if args.motif_filter == 'deg':
        if not args.deg_file_1 or not args.deg_file_2:
            raise ValueError("--deg-file-1 and --deg-file-2 are required for 'deg' filter.")
        print("Creating filter from DEG files...")
        
        # Read both DEG files (assumes first column is gene_name)
        deg1 = _normalize_deg_columns(pd.read_csv(args.deg_file_1))
        deg2 = _normalize_deg_columns(pd.read_csv(args.deg_file_2))

        if 'padj' not in deg1.columns or 'padj' not in deg2.columns:
            print("   Warning: 'padj' column not found in one or both DEG files, using only log2FoldChange filtering")

        deg1_filtered = _filter_deg_dataframe(
            deg1,
            base_mean_threshold=args.deg_base_mean_threshold,
            log2fc_threshold=args.deg_log2fc_threshold,
        )
        deg2_filtered = _filter_deg_dataframe(
            deg2,
            base_mean_threshold=args.deg_base_mean_threshold,
            log2fc_threshold=args.deg_log2fc_threshold,
        )

        expressed1 = set(deg1_filtered['gene_name'].dropna().unique())
        expressed2 = set(deg2_filtered['gene_name'].dropna().unique())
        same_sign_genes = _get_same_sign_deg_genes(deg1_filtered, deg2_filtered)
        opposite_sign_genes = expressed1.intersection(expressed2) - same_sign_genes

        padj_genes_1 = _get_genes_passing_padj(deg1_filtered, args.deg_padj_threshold)
        padj_genes_2 = _get_genes_passing_padj(deg2_filtered, args.deg_padj_threshold)

        if padj_genes_1 is None and padj_genes_2 is None:
            expressed_genes = same_sign_genes
            padj_fail_genes = set()
        else:
            padj_genes_either = (padj_genes_1 or set()).union(padj_genes_2 or set())
            expressed_genes = same_sign_genes.intersection(padj_genes_either)
            padj_fail_genes = same_sign_genes - expressed_genes

        print(f"   ✓ Found {len(expressed1)} expressed+DE genes in file 1, {len(expressed2)} in file 2")
        print(f"   ✓ Genes passing both files with matching log2FC sign: {len(same_sign_genes)}")
        if opposite_sign_genes:
            print(f"   ✓ Excluded {len(opposite_sign_genes)} genes with opposite log2FC signs between files")
        if padj_fail_genes:
            print(f"   ✓ Excluded {len(padj_fail_genes)} same-sign genes failing padj in both files")
        print(
            f"   ✓ Filter criteria: baseMean > {args.deg_base_mean_threshold}, "
            f"|log2FC| > {args.deg_log2fc_threshold}, same log2FC sign in both files"
            + (f", and padj < {args.deg_padj_threshold} in at least one file" if 'padj' in deg1.columns or 'padj' in deg2.columns else "")
        )
        
        # Get all TFs from the HOCOMOCO mapping (convert to mouse format for mouse DEG files)
        # Check if genes appear to be in mouse format (first letter capitalized, rest lowercase)
        # Sample multiple genes to be more robust
        sample_genes = list(expressed_genes)[:10] if len(expressed_genes) > 0 else []
        is_mouse_format = False
        if sample_genes:
            # Check if majority of sample genes are in mouse format
            mouse_format_count = sum(1 for g in sample_genes 
                                   if g and g[0].isupper() and (len(g) == 1 or g[1:].islower()))
            is_mouse_format = mouse_format_count > len(sample_genes) / 2
        
        if is_mouse_format:
            print("   ✓ Detected mouse gene format, converting HOCOMOCO mapping to mouse format...")
            motif_to_gene = get_hocomoco_motif_mapping(to_mouse=True)
        else:
            print("   ✓ Using human gene format for HOCOMOCO mapping...")
            motif_to_gene = get_hocomoco_motif_mapping(to_mouse=False)
        
        hocomoco_tfs = set(motif_to_gene.values())
        
        # Find the intersection: expressed genes that are also TFs in HOCOMOCO
        tfs_to_keep = sorted(list(expressed_genes.intersection(hocomoco_tfs)))
        
        print(f"   ✓ Found {len(tfs_to_keep)} expressed TFs in HOCOMOCO.")
        if args.verbose and tfs_to_keep:
            print(f"   - TFs to keep: {', '.join(tfs_to_keep[:10])}{'...' if len(tfs_to_keep) > 10 else ''}")
            
        # Use the whitelist filter with the generated TF list
        return create_annotation_filter(
            annotation_file=None,
            filter_type='whitelist',
            tf_list=tfs_to_keep
        )

    # Annotation file is required for filters that read from it, but not
    # for whitelist/blacklist which can use Hocomoco as a fallback.
    if not args.annotation_file and args.motif_filter not in ['whitelist', 'blacklist']:
        print(f"Warning: Motif filter '{args.motif_filter}' requires --annotation-file. Skipping.")
        return None
    
    try:
        if args.motif_filter in ['whitelist', 'blacklist'] and not args.filter_tf_list:
            raise ValueError(f"{args.motif_filter} requires --filter-tf-list.")
            
        # For this CLI, we only have tf_list as an extra argument.
        # Other filter types that need more args would require adding them to the parser.
        return create_annotation_filter(
            args.annotation_file,
            args.motif_filter,
            tf_list=args.filter_tf_list
        )
    except Exception as e:
        print(f"Error creating motif filter: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- Region Selection ---
    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument("--region", type=str,
                              help="Genomic region to scan, e.g., 'chr2:136117414-136117614'.")
    region_group.add_argument("--scores-csv",
                              help="Path to CSV from deletion analysis. The entire region "
                                   "spanned by the file will be scanned.")

    # --- Core Inputs ---
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--genome-zarr", required=True, help="Path to the hg38 genome Zarr.")
    parser.add_argument("--motif-names", required=True, help="Path to file with motif names.")
    parser.add_argument("--fimo-cache", required=True, help="Path to precached FIMO mappings.")

    # --- Optional Parameters ---
    parser.add_argument("--device", default="cpu", help="Device for inference ('cuda' or 'cpu').")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed p-value logs.")
    parser.add_argument("--p-value-cutoff", type=float, default=1e-4,
                        help="P-value cutoff for reporting significant motif hits.")
    parser.add_argument("--output-file", type=str, help="Path to save scan results as a TSV file.")

    # --- Motif Filtering ---
    filter_group = parser.add_argument_group('Motif Filtering')
    filter_group.add_argument('--motif-filter', type=str, default='none',
                              choices=['none', 'expressed', 'de', 'whitelist', 'blacklist', 'promoter', 'gene_list', 'deg'],
                              help='Type of motif filter to apply.')
    filter_group.add_argument('--annotation-file', type=str, help='Annotation file for filters.')
    filter_group.add_argument('--gene-list-file', type=str, help="Path to a file with a list of genes for the 'gene_list' filter.")
    filter_group.add_argument('--deg-file-1', type=str, help="Path to first DEG file (for 'deg' filter).")
    filter_group.add_argument('--deg-file-2', type=str, help="Path to second DEG file (for 'deg' filter).")
    filter_group.add_argument('--deg-base-mean-threshold', type=float, default=10.0,
                              help='Minimum baseMean for expressed genes (for deg filter).')
    filter_group.add_argument('--deg-log2fc-threshold', type=float, default=2.0,
                              help='Minimum absolute log2FoldChange for differential expression (for deg filter).')
    filter_group.add_argument('--deg-padj-threshold', type=float, default=0.05,
                              help='Maximum padj for significant differential expression (for deg filter).')
    filter_group.add_argument('--filter-tf-list', type=str, nargs='+', help='List of TFs for whitelist/blacklist.')
    # Add other filter-specific arguments here as needed...
    filter_group.add_argument('--promoter-cell-type', type=str, default='HT365B1-S1H1', 
                              help='Cell type for promoter accessibility filter.')
    
    args = parser.parse_args()

    print("=== On-Demand Motif Scan ===")

    # 1. Determine scan region
    if args.region:
        print(f"\n1. Using specified region: {args.region}")
        try:
            chrom_part, range_part = args.region.split(':')
            start, end = map(int, range_part.split('-'))
            chromosome = chrom_part
        except ValueError:
            raise ValueError("Invalid --region format. Expected 'chr:start-end'.")
    else:
        print(f"\n1. Scanning full region from: {args.scores_csv}")
        scores_df = pd.read_csv(args.scores_csv)
        filename = os.path.basename(args.scores_csv)
        chromosome = filename.split('_')[0]
        
        start = int(scores_df['position'].min())
        end = int(scores_df['position'].max())
        
        print(f"   ✓ Defining scan region from score file bounds: {chromosome}:{start}-{end}")

    # 2. Set up analyzer
    print("\n2. Initializing Analyzer...")
    with open(args.motif_names) as f:
        motif_names = [line.strip() for line in f]
        
    fimo_cache = FimoMappingsCache.load(args.fimo_cache)
    model_inference = ModelInference(args.checkpoint, model_type='phase3', device=args.device)
    model = model_inference.model
    
    motif_scanner = FimoMotifScanner(
        model=model,
        motif_names=motif_names,
        fimo_cache=fimo_cache,
        p_value_threshold=args.p_value_cutoff
    )
    
    from caesar.io.zarr_io import SequenceDenseZarrIO
    genome_io = SequenceDenseZarrIO(args.genome_zarr)
    dataset = MutationDataset(genome_io)
    
    motif_filter = create_motif_filter(args)
    if motif_filter:
        print(f"   ✓ Applying motif filter: {motif_filter.get_description()}")

    analyzer = MutationAnalyzer(
        model=model, output_heads=[], dataset=dataset,
        motif_scanner=motif_scanner, motif_filter=motif_filter
    )
    print("   ✓ Analyzer ready.")

    # 3. Perform scan
    print(f"\n3. Scanning {chromosome}:{start}-{end} for motifs...")
    motif_hits = analyzer.scan_motifs(chromosome, start, end, verbose=args.verbose)

    # 4. Report results
    print("\n4. Scan Results:")
    if motif_hits.empty:
        print("   - No significant motifs found in the specified region.")
    else:
        print(f"   ✓ Found {len(motif_hits)} significant motif hits:")
        print(motif_hits[['motif_name', 'start', 'end', 'p_value', 'score']].head(20).to_string(index=False))

    # 5. Save results if an output file is specified
    if args.output_file:
        print(f"\n5. Saving results to {args.output_file}...")
        try:
            # Add chromosome column
            motif_hits['chromosome'] = chromosome
            
            # Get motif-to-gene mapping
            motif_to_gene = get_hocomoco_motif_mapping()
            motif_hits['gene_name'] = motif_hits['motif_name'].map(motif_to_gene)
            
            # Select and reorder columns
            output_df = motif_hits[[
                'chromosome', 'start', 'end', 'gene_name', 
                'motif_name', 'p_value', 'score'
            ]]
            
            # Save to TSV
            output_df.to_csv(args.output_file, sep='\t', index=False)
            print("   ✓ Results saved successfully.")
        except Exception as e:
            print(f"   ✗ Error saving results: {e}")

    print("\n=== Scan Complete ===")

if __name__ == "__main__":
    main()

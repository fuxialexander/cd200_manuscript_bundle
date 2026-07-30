#!/usr/bin/env python3
"""
Command-line interface for mutation analysis module.

Provides easy-to-use commands for analyzing variants and genomic regions
with deep learning models, supporting ISM, deletion scanning, and motif analysis.

Usage:
    # Analyze single variant
    python mutation_analysis_cli.py variant chr2:136117726:A:G \\
        --checkpoint model.ckpt \\
        --genome-zarr hg38.zarr \\
        --output-dir results/

    # Analyze region with deletion scanning
    python mutation_analysis_cli.py region chr2:136117626-136117842 \\
        --checkpoint model.ckpt \\
        --genome-zarr hg38.zarr \\
        --deletion-scan \\
        --real-deletion \\
        --extract-motifs \\
        --output-dir results/

    # Batch analyze VCF file
    python mutation_analysis_cli.py batch variants.vcf \\
        --checkpoint model.ckpt \\
        --genome-zarr hg38.zarr \\
        --output-dir results/ \\
        --parallel 4
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import pandas as pd
import numpy as np
from datetime import datetime

from caesar.io.bpcell_io import CelltypeDenseBPCellsIO
from caesar.io.zarr_io import CelltypeDenseZarrIO

# Add mutation_analysis to path
sys.path.insert(0, str(Path(__file__).parent))

from mutation_analysis import (
    load_model, MutationDataset, MutationAnalyzer,
    Variant
)
from mutation_analysis.motif_filters import create_annotation_filter
from caesar.io.zarr_io import SequenceDenseZarrIO


def setup_parser() -> argparse.ArgumentParser:
    """Create argument parser with all options."""
    parser = argparse.ArgumentParser(
        description='Mutation Analysis Command-Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Analysis command')

    # ========== Variant Command ==========
    variant_parser = subparsers.add_parser(
        'variant',
        help='Analyze single variant (ISM and optionally deletion scan)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    variant_parser.add_argument('variant', type=str,
                               help='Variant in format chr:pos:ref:alt (e.g., chr2:136117726:A:G)')
    add_common_args(variant_parser)
    add_analysis_args(variant_parser)
    add_deletion_args(variant_parser)
    add_output_args(variant_parser)

    # ========== Region Command ==========
    region_parser = subparsers.add_parser(
        'region',
        help='Analyze genomic region with deletion scanning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    region_parser.add_argument('region', type=str,
                              help='Region in format chr:start-end (e.g., chr2:136117626-136117842)')
    add_common_args(region_parser)
    add_deletion_args(region_parser)
    add_output_args(region_parser)

    # ========== Batch Command ==========
    batch_parser = subparsers.add_parser(
        'batch',
        help='Batch analyze variants from VCF or BED file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    batch_parser.add_argument('input_file', type=str,
                             help='Input file (VCF for variants, BED for regions)')
    batch_parser.add_argument('--format', choices=['vcf', 'bed'], default='vcf',
                             help='Input file format')
    batch_parser.add_argument('--parallel', type=int, default=1,
                             help='Number of parallel workers')
    batch_parser.add_argument('--max-variants', type=int, default=None,
                             help='Maximum number of variants to process')
    add_common_args(batch_parser)
    add_analysis_args(batch_parser)
    add_output_args(batch_parser)

    return parser


def add_common_args(parser: argparse.ArgumentParser):
    """Add arguments common to all commands."""
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--genome-zarr', type=str, required=True,
                       help='Path to genome zarr file')
    parser.add_argument('--atac-zarr', type=str, default=None,
                       help='Path to ATAC zarr file (optional, for observed data)')
    parser.add_argument('--cell-types', type=str, nargs='+', default=None,
                       help='Cell types to use from ATAC zarr file (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference (cuda/cpu)')
    parser.add_argument('--extend-bp', type=int, default=2048,
                       help='Context size for model input')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to config file for Phase 3 models (optional)')
    parser.add_argument('--simple-motifs-path', type=str, default=None,
                       help='Path to simple motifs .pt file (optional, for model loading)')


def add_analysis_args(parser: argparse.ArgumentParser):
    """Add analysis configuration arguments."""
    parser.add_argument('--ism', action='store_true',
                       help='Perform in-silico mutagenesis')
    parser.add_argument('--deletion-scan', action='store_true',
                       help='Perform deletion scanning')
    parser.add_argument('--ism-region-size', type=int, default=200,
                       help='Size of region for ISM (centered on variant)')


def add_deletion_args(parser: argparse.ArgumentParser):
    """Add deletion scanning arguments."""
    parser.add_argument('--deletion-size', type=int, default=15,
                       help='Size of deletion window (bp)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Step size between windows')
    parser.add_argument('--real-deletion', action='store_true',
                       help='Use real deletion (stitching) instead of zeroing')
    parser.add_argument('--distribute-effects', action='store_true',
                       help='Distribute window effects to nucleotide resolution')
    parser.add_argument('--smoothing-window', type=int, default=31,
                       help='Smoothing window size')
    parser.add_argument('--core-region', type=str, default=None,
                       help='Core region for scoring (chr:start-end)')
    parser.add_argument('--save-diff-matrix-path', type=str, default=None,
                        help='Save regression difference matrix to this .npy file path.')

    # Motif analysis
    parser.add_argument('--extract-motifs', action='store_true',
                       help='Extract motif activations from model')
    parser.add_argument('--motif-names-file', type=str, default=None,
                       help='File with motif names (one per line)')
    parser.add_argument('--motif-threshold-sigma', type=float, default=4.0,
                       help='Sigma threshold for high-scoring regions')

    # Motif filtering
    parser.add_argument('--motif-filter', type=str, default='none',
                       choices=['none', 'expressed', 'de', 'whitelist', 'blacklist'],
                       help='Type of motif filter to apply')
    parser.add_argument('--annotation-file', type=str, default=None,
                       help='Annotation file for motif filtering')
    parser.add_argument('--filter-expression-min', type=float, default=10.0,
                       help='Minimum expression for expressed filter')
    parser.add_argument('--filter-expression-cols', type=str, nargs='+',
                       default=['baseMean_1f', 'baseMean_af'],
                       help='Expression column names')
    parser.add_argument('--filter-padj-threshold', type=float, default=0.05,
                       help='P-value threshold for DE filter')
    parser.add_argument('--filter-padj-cols', type=str, nargs='+',
                       default=['padj_1f', 'padj_af'],
                       help='P-value column names for DE filter')
    parser.add_argument('--filter-tf-list', type=str, nargs='+', default=None,
                       help='List of TFs for whitelist/blacklist filter')


def add_output_args(parser: argparse.ArgumentParser):
    """Add output configuration arguments."""
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save result plots')
    parser.add_argument('--save-data', action='store_true', default=True,
                       help='Save result data (CSV/JSON)')
    parser.add_argument('--save-baseline-pred', action='store_true',
                       help='Save baseline prediction along with deletion effects')
    parser.add_argument('--save-observed-atac', action='store_true',
                          help='Save observed ATAC-seq data along with deletion effects')
    parser.add_argument('--plot-format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Format for saved plots')


def parse_variant(variant_str: str) -> Variant:
    """Parse variant string to Variant object."""
    parts = variant_str.split(':')
    if len(parts) != 4:
        raise ValueError(f"Invalid variant format: {variant_str}. "
                        f"Expected format: chr:pos:ref:alt")

    chromosome, pos, ref, alt = parts
    return Variant(
        chromosome=chromosome,
        position=int(pos) - 1,  # Convert to 0-based
        ref=ref,
        alt=alt
    )


def parse_region(region_str: str) -> tuple:
    """Parse region string to (chr, start, end)."""
    parts = region_str.replace(':', '-').split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid region format: {region_str}. "
                        f"Expected format: chr:start-end")

    chromosome, start, end = parts
    return chromosome, int(start), int(end)


def load_motif_names(motif_names_file: Optional[str]) -> Optional[List[str]]:
    """Load motif names from file."""
    if motif_names_file is None:
        return None

    with open(motif_names_file) as f:
        return [line.strip() for line in f if line.strip()]


def create_motif_filter(args):
    """Create motif filter from arguments."""
    if args.motif_filter == 'none':
        return None

    if args.annotation_file is None:
        print("Warning: Motif filter requested but no annotation file provided. "
              "Skipping filtering.")
        return None

    try:
        if args.motif_filter == 'expressed':
            return create_annotation_filter(
                args.annotation_file,
                'expressed',
                expression_columns=args.filter_expression_cols,
                min_expression=args.filter_expression_min
            )
        elif args.motif_filter == 'de':
            return create_annotation_filter(
                args.annotation_file,
                'de',
                padj_columns=args.filter_padj_cols,
                padj_threshold=args.filter_padj_threshold
            )
        elif args.motif_filter in ['whitelist', 'blacklist']:
            if args.filter_tf_list is None:
                raise ValueError(f"{args.motif_filter} filter requires --filter-tf-list")
            return create_annotation_filter(
                args.annotation_file,
                args.motif_filter,
                tf_list=args.filter_tf_list
            )
    except Exception as e:
        print(f"Error creating motif filter: {e}")
        return None


def setup_analyzer(args) -> tuple:
    """Setup analyzer from command-line arguments."""
    print(f"Loading model from: {args.checkpoint}")
    # Prepare kwargs for load_model
    load_kwargs = {}
    if args.config_file:
        load_kwargs['config_file'] = args.config_file
    if args.simple_motifs_path:
        load_kwargs['simple_motifs_path'] = args.simple_motifs_path
    model, output_heads = load_model(args.checkpoint, device=args.device, **load_kwargs)
    print(f"  Output heads: {output_heads}")

    print(f"Loading genome from: {args.genome_zarr}")
    genome = SequenceDenseZarrIO(args.genome_zarr, mode='r')

    # Load ATAC data if provided
    atac_io = None
    if args.atac_zarr:
        print(f"Loading ATAC data from: {args.atac_zarr}")
        try:
            if args.atac_zarr.endswith('.zarr'):
                atac_io = CelltypeDenseZarrIO(args.atac_zarr, mode='r')
            else:
                atac_io = CelltypeDenseBPCellsIO(path=args.atac_zarr, mode='r')
            
            if args.cell_types:
                print(f"  Filtering for cell types: {args.cell_types}")
                atac_io = atac_io.subset(args.cell_types)
        except Exception as e:
            print(f"  Warning: Could not load ATAC data. {e}")

    # Create dataset
    dataset = MutationDataset(genome, atac_io=atac_io, cell_types=args.cell_types)

    # Load motif names if provided
    motif_names = None
    if hasattr(args, 'extract_motifs') and args.extract_motifs:
        if args.motif_names_file:
            motif_names = load_motif_names(args.motif_names_file)
            print(f"  Loaded {len(motif_names)} motif names")

    # Create motif filter
    motif_filter = None
    if hasattr(args, 'motif_filter'):
        motif_filter = create_motif_filter(args)
        if motif_filter:
            print(f"  Motif filter: {motif_filter.get_description()}")

    # Create analyzer
    analyzer = MutationAnalyzer(
        model, output_heads, dataset,
        device=args.device,
        motif_filter=motif_filter,
        motif_names=motif_names
    )

    return analyzer, output_heads


def analyze_variant_command(args):
    """Handle variant analysis command."""
    # Parse variant
    variant = parse_variant(args.variant)
    print(f"\nAnalyzing variant: {variant}")

    # Setup
    analyzer, output_heads = setup_analyzer(args)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize variant name for filenames
    variant_name = str(variant).replace(':', '_').replace('>', '_')

    results = {}

    # ISM Analysis
    if args.ism:
        print(f"\nRunning ISM analysis (region size: {args.ism_region_size}bp)...")
        ism_result = analyzer.in_silico_mutagenesis(
            variant,
            region_size=args.ism_region_size,
            extend_bp=args.extend_bp
        )
        results['ism'] = ism_result

        if args.save_plots:
            plot_path = output_dir / f"{variant_name}_ism.{args.plot_format}"
            print(f"  Saving ISM plot: {plot_path}")
            ism_result.plot_summary(save_path=str(plot_path))

        if args.save_data:
            # Save mean-centered effects
            for head in output_heads:
                data_path = output_dir / f"{variant_name}_ism_{head}.csv"
                effects = ism_result.mean_centered_effects[head]
                df = pd.DataFrame(
                    effects,
                    columns=['A', 'C', 'G', 'T'],
                    index=range(ism_result.region_start, ism_result.region_end)
                )
                df.to_csv(data_path)
                print(f"  Saved ISM data: {data_path}")

    # Variant Effect Analysis
    print(f"\nRunning variant effect analysis...")
    variant_result = analyzer.analyze_variant(variant, extend_bp=args.extend_bp)
    results['variant'] = variant_result

    if args.save_plots:
        plot_path = output_dir / f"{variant_name}_effect.{args.plot_format}"
        print(f"  Saving variant effect plot: {plot_path}")
        variant_result.plot_summary(save_path=str(plot_path))

    if args.save_data:
        # Save variant effects as JSON
        data_path = output_dir / f"{variant_name}_effect.json"
        effect_data = {
            'variant': str(variant),
            'effects': {
                head: {
                    'primary_metric': float(variant_result.effects[head]['primary_metric']),
                    'all_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                                   for k, v in variant_result.effects[head].items()}
                }
                for head in output_heads
            }
        }
        with open(data_path, 'w') as f:
            json.dump(effect_data, f, indent=2)
        print(f"  Saved variant effect data: {data_path}")

    # Deletion Scan (optional)
    if args.deletion_scan:
        print(f"\nRunning deletion scan...")

        # Parse core region if provided
        core_region = None
        if args.core_region:
            _, core_start, core_end = parse_region(args.core_region)
            core_region = (core_start, core_end)

        deletion_result = analyzer.sliding_window_deletion(
            variant,
            deletion_size=args.deletion_size,
            stride=args.stride,
            extend_bp=args.extend_bp,
            distribute_effects=args.distribute_effects,
            smoothing_window=args.smoothing_window,
            core_region=core_region,
            real_deletion=args.real_deletion,
            extract_motif_activations=args.extract_motifs,
            motif_threshold_sigma=args.motif_threshold_sigma,
            save_diff_matrix_path=args.save_diff_matrix_path
        )
        results['deletion'] = deletion_result

        if args.save_plots:
            plot_path = output_dir / f"{variant_name}_deletion.{args.plot_format}"
            print(f"  Saving deletion plot: {plot_path}")
            deletion_result.plot_summary(save_path=str(plot_path))

            # Plot motif activations if available
            if deletion_result.motif_activations is not None:
                for head in output_heads:
                    motif_plot_path = output_dir / f"{variant_name}_motifs_{head}.{args.plot_format}"
                    print(f"  Saving motif activation plot: {motif_plot_path}")
                    deletion_result.plot_motif_activations(head, save_path=str(motif_plot_path))

        if args.save_data:
            # Save deletion effects
            for head in output_heads:
                if args.distribute_effects and deletion_result.nucleotide_scores is not None:
                    data_path = output_dir / f"{variant_name}_deletion_{head}_nucleotide.csv"
                    scores = deletion_result.nucleotide_scores[head]
                    positions = np.arange(deletion_result.region_start,
                                        deletion_result.region_start + len(scores))
                    data = {
                        'position': positions,
                        'effect': scores
                    }
                    if args.save_baseline_pred and deletion_result.baseline_predictions and head in deletion_result.baseline_predictions:
                        baseline_pred = deletion_result.baseline_predictions[head]
                    baseline_array = np.asarray(baseline_pred)
                    if baseline_array.ndim == 0 or baseline_array.size == 1:
                        data['baseline_prediction'] = np.full(len(scores), float(baseline_array.reshape(-1)[0]))
                    elif baseline_array.ndim > 0:
                        if len(baseline_array) == len(scores):
                            data['observed_prediction'] = baseline_array
                        else:
                            print(f"  Warning: baseline prediction length ({len(baseline_array)}) does not match scores length ({len(scores)}) for head {head}. Skipping.")
                    if args.save_observed_atac and deletion_result.observed_atac is not None:
                        observed_atac = deletion_result.observed_atac
                        if observed_atac.ndim > 1:
                            observed_atac = observed_atac.squeeze()

                        if len(observed_atac) == len(scores):
                            data['observed_atac'] = observed_atac
                        else:
                            print(f"  Warning: observed ATAC length ({len(observed_atac)}) does not match scores length ({len(scores)}) for head {head}. Skipping.")
                    df = pd.DataFrame(data)
                else:
                    data_path = output_dir / f"{variant_name}_deletion_{head}_window.csv"
                    df = pd.DataFrame({
                        'position': deletion_result.window_positions,
                        'effect': deletion_result.deletion_effects[head]
                    })
                df.to_csv(data_path, index=False)
                print(f"  Saved deletion data: {data_path}")

    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
    return results


def analyze_region_command(args):
    """Handle region analysis command."""
    # Parse region
    chromosome, start, end = parse_region(args.region)
    center = (start + end) // 2

    # Create pseudo-variant at center
    variant = Variant(
        chromosome=chromosome,
        position=center,
        ref='N',
        alt='N'
    )

    print(f"\nAnalyzing region: {args.region}")
    print(f"  Center position: {chromosome}:{center}")

    # Setup
    analyzer, output_heads = setup_analyzer(args)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region_name = f"{chromosome}_{start}_{end}"

    # Parse core region if provided (use full region if not specified)
    core_region = None
    if args.core_region:
        _, core_start, core_end = parse_region(args.core_region)
        core_region = (core_start, core_end)
    else:
        core_region = (start, end)

    print(f"\nRunning deletion scan on region...")
    deletion_result = analyzer.sliding_window_deletion(
        variant,
        deletion_size=args.deletion_size,
        stride=args.stride,
        extend_bp=args.extend_bp,
        distribute_effects=args.distribute_effects,
        smoothing_window=args.smoothing_window,
        core_region=core_region,
        real_deletion=args.real_deletion,
        extract_motif_activations=args.extract_motifs,
        motif_threshold_sigma=args.motif_threshold_sigma,
        save_diff_matrix_path=args.save_diff_matrix_path
    )

    if args.save_plots:
        plot_path = output_dir / f"{region_name}_deletion.{args.plot_format}"
        print(f"  Saving deletion plot: {plot_path}")
        deletion_result.plot_summary(save_path=str(plot_path))

        # Plot motif activations if available
        if deletion_result.motif_activations is not None:
            for head in output_heads:
                motif_plot_path = output_dir / f"{region_name}_motifs_{head}.{args.plot_format}"
                print(f"  Saving motif activation plot: {motif_plot_path}")
                deletion_result.plot_motif_activations(head, save_path=str(motif_plot_path))

    if args.save_data:
        # Save deletion effects
        for head in output_heads:
            if args.distribute_effects and deletion_result.nucleotide_scores is not None:
                data_path = output_dir / f"{region_name}_deletion_{head}_nucleotide.csv"
                scores = deletion_result.nucleotide_scores[head]
                positions = np.arange(deletion_result.region_start,
                                    deletion_result.region_start + len(scores))
                data = {
                    'position': positions,
                    'effect': scores
                }
                if args.save_baseline_pred and deletion_result.baseline_predictions and head in deletion_result.baseline_predictions:
                    baseline_pred = deletion_result.baseline_predictions[head]
                    print(f"  Baseline prediction: {baseline_pred}")
                    print(f"  Baseline prediction shape: {baseline_pred.shape}")
                    print(f"  Scores shape: {scores.shape}")
                    baseline_array = np.asarray(baseline_pred)
                    if baseline_array.ndim == 0 or baseline_array.size == 1:
                        data['baseline_prediction'] = np.full(len(scores), float(baseline_array.reshape(-1)[0]))
                    elif baseline_array.ndim > 0:
                        if len(baseline_array) == len(scores):
                            data['observed_prediction'] = baseline_array
                        else:
                            print(f"  Warning: baseline prediction length ({len(baseline_array)}) does not match scores length ({len(scores)}) for head {head}. Skipping.")
                if args.save_observed_atac and deletion_result.observed_atac is not None:
                    observed_atac = deletion_result.observed_atac
                    if observed_atac.ndim > 1:
                        observed_atac = observed_atac.squeeze()
                        
                    if len(observed_atac) == len(scores):
                        data['observed_atac'] = observed_atac
                    else:
                        print(f"  Warning: observed ATAC length ({len(observed_atac)}) does not match scores length ({len(scores)}) for head {head}. Skipping.")

                df = pd.DataFrame(data)
            else:
                data_path = output_dir / f"{region_name}_deletion_{head}_window.csv"
                df = pd.DataFrame({
                    'position': deletion_result.window_positions,
                    'effect': deletion_result.deletion_effects[head]
                })
            df.to_csv(data_path, index=False)
            print(f"  Saved deletion data: {data_path}")

    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
    return deletion_result


def analyze_batch_command(args):
    """Handle batch analysis command."""
    print(f"\nBatch analysis: {args.input_file}")
    print(f"  Format: {args.format}")

    # Load input file
    if args.format == 'vcf':
        # Parse VCF
        variants = parse_vcf_file(args.input_file, max_variants=args.max_variants)
        print(f"  Loaded {len(variants)} variants from VCF")
    else:
        # Parse BED
        regions = parse_bed_file(args.input_file, max_regions=args.max_variants)
        print(f"  Loaded {len(regions)} regions from BED")
        # Convert regions to pseudo-variants
        variants = [
            Variant(chromosome=chr, position=(start + end) // 2, ref='N', alt='N')
            for chr, start, end in regions
        ]

    # Setup
    analyzer, output_heads = setup_analyzer(args)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each variant
    results_summary = []

    for i, variant in enumerate(variants):
        print(f"\n[{i+1}/{len(variants)}] Analyzing: {variant}")

        try:
            # Analyze variant
            result = analyzer.analyze_variant(variant, extend_bp=args.extend_bp)

            # Save results
            variant_name = str(variant).replace(':', '_').replace('>', '_')

            if args.save_plots:
                plot_path = output_dir / f"{variant_name}_effect.{args.plot_format}"
                result.plot_summary(save_path=str(plot_path))

            # Collect summary data
            summary_row = {
                'variant': str(variant),
                'chromosome': variant.chromosome,
                'position': variant.position,
            }
            for head in output_heads:
                summary_row[f'{head}_effect'] = result.effects[head]['primary_metric']
            results_summary.append(summary_row)

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary table
    if results_summary:
        summary_path = output_dir / 'batch_summary.csv'
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Batch analysis complete! Summary saved to: {summary_path}")

    return results_summary


def parse_vcf_file(vcf_path: str, max_variants: Optional[int] = None) -> List[Variant]:
    """Parse VCF file and return list of variants."""
    variants = []

    with open(vcf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 5:
                continue

            chr, pos, _, ref, alt = fields[:5]

            # Handle multiple alts
            alts = alt.split(',')
            for a in alts:
                variants.append(Variant(chromosome=chr, position=int(pos) - 1, ref=ref, alt=a))  # Convert to 0-based

            if max_variants and len(variants) >= max_variants:
                break

    return variants


def parse_bed_file(bed_path: str, max_regions: Optional[int] = None) -> List[tuple]:
    """Parse BED file and return list of (chr, start, end)."""
    regions = []

    with open(bed_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('track'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 3:
                continue

            chr, start, end = fields[:3]
            regions.append((chr, int(start), int(end)))

            if max_regions and len(regions) >= max_regions:
                break

    return regions


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == 'variant':
            analyze_variant_command(args)
        elif args.command == 'region':
            analyze_region_command(args)
        elif args.command == 'batch':
            analyze_batch_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

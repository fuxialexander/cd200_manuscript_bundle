"""
Genomic Query Module - High-level functions for answering genomic data questions.

This module provides convenient functions for common genomic data queries,
designed to be used interactively or by AI assistants to answer questions
about genomic data.

Output files are organized in: analysis_results/{YYYYMMDD_HHMMSS}_{analysis_title}/
Each analysis includes a methods.md file documenting the approach and data used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from caesar.analysis.track import Track
    from caesar.io.genomic_io import GenomicIO
    from gcell.rna.gencode import Gencode

logger = logging.getLogger(__name__)

# Default base directory for analysis results
DEFAULT_OUTPUT_BASE = "analysis_results"


def create_output_dir(
    analysis_title: str,
    base_dir: str | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Create organized output directory for analysis results.

    Args:
        analysis_title: Short descriptive title for the analysis (e.g., "TERT_promoter_peaks")
        base_dir: Base directory for results (default: analysis_results/)
        timestamp: Timestamp to use (default: current time)

    Returns:
        Path to the created output directory
    """
    if base_dir is None:
        base_dir = DEFAULT_OUTPUT_BASE

    if timestamp is None:
        timestamp = datetime.now()

    # Format: YYYYMMDD_HHMMSS
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")

    # Clean analysis title for filesystem
    clean_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in analysis_title)

    output_path = Path(base_dir) / f"{time_str}_{clean_title}"
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def write_methods_file(
    output_dir: Path,
    analysis_type: str,
    parameters: dict[str, Any],
    data_source: str,
    description: str,
    additional_info: dict[str, Any] | None = None,
) -> Path:
    """Write methods.md file documenting the analysis approach and data.

    Args:
        output_dir: Directory to write methods.md
        analysis_type: Type of analysis performed
        parameters: Analysis parameters used
        data_source: Path or description of data source
        description: Brief description of the analysis
        additional_info: Additional information to include

    Returns:
        Path to the methods.md file
    """
    methods_path = output_dir / "methods.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# {analysis_type}",
        "",
        f"**Generated**: {timestamp}",
        "",
        "## Description",
        "",
        description,
        "",
        "## Data Source",
        "",
        f"- **Path**: `{data_source}`",
        "",
        "## Parameters",
        "",
    ]

    for key, value in parameters.items():
        lines.append(f"- **{key}**: {value}")

    if additional_info:
        lines.append("")
        lines.append("## Additional Information")
        lines.append("")
        for key, value in additional_info.items():
            if isinstance(value, dict):
                lines.append(f"### {key}")
                for k, v in value.items():
                    lines.append(f"- **{k}**: {v}")
            elif isinstance(value, list):
                lines.append(f"### {key}")
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append(f"- **{key}**: {value}")

    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("See this directory for generated figures and data files.")
    lines.append("")

    methods_path.write_text("\n".join(lines))
    return methods_path


@dataclass
class QueryResult:
    """Container for genomic query results."""

    success: bool
    summary: str
    findings: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    output_dir: str | None = None
    methods_file: str | None = None

    def to_markdown(self) -> str:
        """Generate a markdown report of the results."""
        lines = []

        # Summary
        lines.append("## Analysis Summary")
        lines.append(self.summary)
        lines.append("")

        # Findings
        if self.findings:
            lines.append("## Findings")
            for key, value in self.findings.items():
                if isinstance(value, pd.DataFrame):
                    lines.append(f"### {key}")
                    lines.append(value.to_markdown(index=False))
                elif isinstance(value, dict):
                    lines.append(f"### {key}")
                    for k, v in value.items():
                        lines.append(f"- **{k}**: {v}")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")

        # Figures
        if self.figures:
            lines.append("## Figures")
            for fig_path in self.figures:
                fig_name = Path(fig_path).name
                lines.append(f"- [{fig_name}]({fig_path})")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## Warnings")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        return "\n".join(lines)


def load_gencode(assembly: str = "hg38", version: int = 40) -> Gencode:
    """Load GENCODE gene annotations.

    Args:
        assembly: Genome assembly (hg38, hg19, mm10)
        version: GENCODE version number

    Returns:
        Gencode object with gene annotations
    """
    from gcell.rna.gencode import Gencode

    return Gencode(assembly=assembly, version=version)


def find_gene(gene_name: str, gencode: Gencode) -> dict | None:
    """Look up gene information by name.

    Args:
        gene_name: Gene symbol (case-insensitive)
        gencode: Gencode annotation object

    Returns:
        Dict with gene info or None if not found
    """
    # Try exact match first
    if gene_name in gencode.gene_to_tss:
        return {
            "name": gene_name,
            "chrom": gencode.gene_to_chrom[gene_name],
            "tss": gencode.gene_to_tss[gene_name],
            "tes": gencode.gene_to_tes.get(gene_name),
            "strand": gencode.gene_to_strand[gene_name],
            "type": gencode.gene_to_type.get(gene_name, "unknown"),
        }

    # Try case-insensitive match
    gene_upper = gene_name.upper()
    matches = [g for g in gencode.gene_to_tss.keys() if g.upper() == gene_upper]
    if matches:
        actual_name = matches[0]
        return {
            "name": actual_name,
            "chrom": gencode.gene_to_chrom[actual_name],
            "tss": gencode.gene_to_tss[actual_name],
            "tes": gencode.gene_to_tes.get(actual_name),
            "strand": gencode.gene_to_strand[actual_name],
            "type": gencode.gene_to_type.get(actual_name, "unknown"),
        }

    return None


def check_peaks_near_promoter(
    io: GenomicIO,
    gene_name: str,
    gencode: Gencode | None = None,
    upstream: int = 2000,
    downstream: int = 2000,
    peak_name: str = "peaks",
    output_base: str | None = None,
) -> QueryResult:
    """Check if there are peaks near a gene's promoter.

    Args:
        io: GenomicIO object with loaded data
        gene_name: Gene symbol to query
        gencode: Gencode object (will load if None)
        upstream: bp upstream of TSS to check
        downstream: bp downstream of TSS to check
        peak_name: Name of peak set to query
        output_base: Base directory for results (default: analysis_results/)

    Returns:
        QueryResult with analysis results
    """
    result = QueryResult(success=False, summary="")

    # Create organized output directory
    output_dir = create_output_dir(f"{gene_name}_promoter_peaks", base_dir=output_base)
    result.output_dir = str(output_dir)

    # Load gencode if not provided
    if gencode is None:
        assembly = getattr(io, "assembly", "hg38") or "hg38"
        gencode = load_gencode(assembly=assembly)

    # Find gene
    gene_info = find_gene(gene_name, gencode)
    if gene_info is None:
        result.summary = f"Gene '{gene_name}' not found in GENCODE annotations."
        return result

    chrom = gene_info["chrom"]
    tss = gene_info["tss"]
    strand = gene_info["strand"]

    # Define promoter region based on strand
    if strand == "+":
        promoter_start = tss - upstream
        promoter_end = tss + downstream
    else:
        promoter_start = tss - downstream
        promoter_end = tss + upstream

    # Find overlapping peaks
    peaks_found = []
    celltypes_checked = []

    for celltype_id in io.ids:
        try:
            peaks = io.get_peaks(celltype_id, name=peak_name)
            if peaks is not None and len(peaks) > 0:
                overlapping = peaks[
                    (peaks["Chromosome"] == chrom)
                    & (peaks["Start"] < promoter_end)
                    & (peaks["End"] > promoter_start)
                ]
                celltypes_checked.append(celltype_id)
                if len(overlapping) > 0:
                    overlapping = overlapping.copy()
                    overlapping["celltype"] = celltype_id
                    peaks_found.append(overlapping)
        except Exception as e:
            logger.debug(f"Could not get peaks for {celltype_id}: {e}")

    # Generate visualization
    try:
        # Get extended region for visualization
        vis_start = promoter_start - 3000
        vis_end = promoter_end + 3000

        track = io.get_default_track_obj(chrom, vis_start, vis_end, conv_size=100)

        # Prepare beds for visualization
        beds_to_plot = []
        if peaks_found:
            all_peaks = pd.concat(peaks_found, ignore_index=True)
            beds_to_plot.append(all_peaks[["Start", "End"]].values)

        fig, axes = track.plot_tracks_with_genebody(
            gene_annot=gencode,
            genes_to_highlight=[gene_info["name"]],
        )

        # Add vertical lines at TSS
        for ax in axes[:-1]:  # Skip gene track axis
            ax.axvline(x=tss - vis_start, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # Save figure
        fig_path = output_dir / f"{gene_info['name']}_promoter_peaks.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        result.figures.append(str(fig_path))

    except Exception as e:
        result.warnings.append(f"Could not generate visualization: {e}")

    # Compile results
    if peaks_found:
        all_peaks = pd.concat(peaks_found, ignore_index=True)
        n_peaks = len(all_peaks)
        celltypes_with_peaks = all_peaks["celltype"].nunique()

        result.success = True
        result.summary = (
            f"**Yes**, found {n_peaks} peak(s) near the {gene_info['name']} promoter "
            f"(TSS +/- {upstream}/{downstream}bp) across {celltypes_with_peaks} cell type(s)."
        )
        result.findings["peaks"] = all_peaks[
            ["Chromosome", "Start", "End", "celltype"]
        ].reset_index(drop=True)
        result.findings["gene_info"] = gene_info
        result.data["peaks_df"] = all_peaks
    else:
        result.success = True
        result.summary = (
            f"**No peaks found** near the {gene_info['name']} promoter "
            f"(TSS +/- {upstream}/{downstream}bp) in {len(celltypes_checked)} cell type(s) checked."
        )
        result.findings["gene_info"] = gene_info
        result.findings["region_checked"] = f"{chrom}:{promoter_start}-{promoter_end}"

    # Write methods file
    data_source = getattr(io, "path", str(type(io).__name__))
    methods_path = write_methods_file(
        output_dir=output_dir,
        analysis_type="Peaks Near Promoter Analysis",
        parameters={
            "gene": gene_info["name"],
            "upstream": f"{upstream} bp",
            "downstream": f"{downstream} bp",
            "peak_name": peak_name,
            "region": f"{chrom}:{promoter_start}-{promoter_end}",
        },
        data_source=data_source,
        description=(
            f"Checked for ATAC-seq peaks overlapping the promoter region of {gene_info['name']}. "
            f"The promoter was defined as TSS +/- {upstream}/{downstream}bp based on strand orientation."
        ),
        additional_info={
            "gene_info": gene_info,
            "celltypes_checked": celltypes_checked,
        },
    )
    result.methods_file = str(methods_path)

    return result


def visualize_gene_accessibility(
    io: GenomicIO,
    gene_name: str,
    gencode: Gencode | None = None,
    upstream: int = 5000,
    downstream: int = 5000,
    conv_size: int = 100,
    output_base: str | None = None,
) -> QueryResult:
    """Visualize accessibility profile around a gene.

    Args:
        io: GenomicIO object with loaded data
        gene_name: Gene symbol to visualize
        gencode: Gencode object (will load if None)
        upstream: bp upstream of gene start
        downstream: bp downstream of gene end
        conv_size: Smoothing window size
        output_base: Base directory for results (default: analysis_results/)

    Returns:
        QueryResult with analysis results
    """
    result = QueryResult(success=False, summary="")

    # Create organized output directory
    output_dir = create_output_dir(f"{gene_name}_accessibility", base_dir=output_base)
    result.output_dir = str(output_dir)

    # Load gencode if not provided
    if gencode is None:
        assembly = getattr(io, "assembly", "hg38") or "hg38"
        gencode = load_gencode(assembly=assembly)

    # Find gene
    gene_info = find_gene(gene_name, gencode)
    if gene_info is None:
        result.summary = f"Gene '{gene_name}' not found in GENCODE annotations."
        return result

    try:
        # Get gene track using built-in method
        track = io.get_default_gene_track_obj(
            gene_info["name"], upstream=upstream, downstream=downstream
        )

        # Plot with gene body
        fig, axes = track.plot_tracks_with_genebody(
            gene_annot=gencode,
            genes_to_highlight=[gene_info["name"]],
        )

        # Calculate some statistics
        stats = {}
        for label, signal in track.convoluted_tracks.items():
            stats[label] = {
                "max_signal": float(np.max(signal)),
                "mean_signal": float(np.mean(signal)),
                "total_signal": float(np.sum(signal)),
            }

        # Save figure
        fig_path = output_dir / f"{gene_info['name']}_accessibility.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        result.figures.append(str(fig_path))

        result.success = True
        result.summary = (
            f"Generated accessibility profile for **{gene_info['name']}** "
            f"({gene_info['chrom']}:{gene_info['tss']}, {gene_info['strand']} strand)."
        )
        result.findings["gene_info"] = gene_info
        result.findings["signal_stats"] = stats
        result.data["track"] = track

        # Write methods file
        data_source = getattr(io, "path", str(type(io).__name__))
        methods_path = write_methods_file(
            output_dir=output_dir,
            analysis_type="Gene Accessibility Visualization",
            parameters={
                "gene": gene_info["name"],
                "upstream": f"{upstream} bp",
                "downstream": f"{downstream} bp",
                "conv_size": f"{conv_size} bp smoothing",
            },
            data_source=data_source,
            description=(
                f"Visualized ATAC-seq accessibility profile around the {gene_info['name']} gene. "
                f"Signal was smoothed with a {conv_size}bp convolution window."
            ),
            additional_info={"gene_info": gene_info, "signal_stats": stats},
        )
        result.methods_file = str(methods_path)

    except Exception as e:
        result.summary = f"Error generating visualization: {e}"
        result.warnings.append(str(e))

    return result


def query_region_signal(
    io: GenomicIO,
    chrom: str,
    start: int,
    end: int,
    conv_size: int = 100,
    output_base: str | None = None,
    gencode: Gencode | None = None,
) -> QueryResult:
    """Query signal statistics for a genomic region.

    Args:
        io: GenomicIO object with loaded data
        chrom: Chromosome name
        start: Start position
        end: End position
        conv_size: Smoothing window size
        output_base: Base directory for results (default: analysis_results/)
        gencode: Optional Gencode object for gene annotations

    Returns:
        QueryResult with analysis results
    """
    result = QueryResult(success=False, summary="")

    # Create organized output directory
    output_dir = create_output_dir(f"region_{chrom}_{start}_{end}", base_dir=output_base)
    result.output_dir = str(output_dir)

    try:
        # Get track data
        track = io.get_default_track_obj(chrom, start, end, conv_size=conv_size)

        # Calculate statistics for each cell type
        stats = {}
        for label, signal in track.convoluted_tracks.items():
            stats[label] = {
                "max_signal": float(np.max(signal)),
                "mean_signal": float(np.mean(signal)),
                "total_signal": float(np.sum(signal)),
                "nonzero_fraction": float(np.mean(signal > 0)),
            }

        # Create summary DataFrame
        stats_df = pd.DataFrame(stats).T
        stats_df.index.name = "celltype"
        stats_df = stats_df.reset_index()

        # Load gencode for annotation if not provided
        if gencode is None:
            assembly = getattr(io, "assembly", "hg38") or "hg38"
            try:
                gencode = load_gencode(assembly=assembly)
            except Exception:
                gencode = None

        # Plot
        if gencode is not None:
            fig, axes = track.plot_tracks_with_genebody(gene_annot=gencode)
        else:
            fig, axes = track.plot_tracks()

        # Save figure
        fig_path = output_dir / f"region_{chrom}_{start}_{end}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        result.figures.append(str(fig_path))

        result.success = True
        result.summary = (
            f"Signal statistics for **{chrom}:{start:,}-{end:,}** "
            f"({end - start:,} bp region)."
        )
        result.findings["statistics"] = stats_df
        result.findings["region"] = {"chrom": chrom, "start": start, "end": end}
        result.data["track"] = track
        result.data["stats"] = stats

        # Write methods file
        data_source = getattr(io, "path", str(type(io).__name__))
        methods_path = write_methods_file(
            output_dir=output_dir,
            analysis_type="Region Signal Query",
            parameters={
                "region": f"{chrom}:{start:,}-{end:,}",
                "region_size": f"{end - start:,} bp",
                "conv_size": f"{conv_size} bp smoothing",
            },
            data_source=data_source,
            description=(
                f"Queried ATAC-seq signal statistics for the genomic region {chrom}:{start:,}-{end:,}. "
                f"Signal was smoothed with a {conv_size}bp convolution window."
            ),
        )
        result.methods_file = str(methods_path)

    except Exception as e:
        result.summary = f"Error querying region: {e}"
        result.warnings.append(str(e))

    return result


def compare_celltypes_at_gene(
    io: GenomicIO,
    gene_name: str,
    celltypes: list[str] | None = None,
    gencode: Gencode | None = None,
    upstream: int = 2000,
    downstream: int = 2000,
    output_base: str | None = None,
) -> QueryResult:
    """Compare accessibility across cell types at a gene.

    Args:
        io: GenomicIO object with loaded data
        gene_name: Gene symbol to compare
        celltypes: List of cell types to compare (None = all)
        gencode: Gencode object (will load if None)
        upstream: bp upstream of TSS
        downstream: bp downstream of TSS
        output_base: Base directory for results (default: analysis_results/)

    Returns:
        QueryResult with analysis results
    """
    from caesar.io.aggregator import ByKeyAggregator
    from caesar.io.normalizer import DictNormalizer, NoNormalizer
    from caesar.io.query import KeyQuery

    result = QueryResult(success=False, summary="")

    # Create organized output directory
    output_dir = create_output_dir(f"{gene_name}_celltype_comparison", base_dir=output_base)
    result.output_dir = str(output_dir)

    # Load gencode if not provided
    if gencode is None:
        assembly = getattr(io, "assembly", "hg38") or "hg38"
        gencode = load_gencode(assembly=assembly)

    # Find gene
    gene_info = find_gene(gene_name, gencode)
    if gene_info is None:
        result.summary = f"Gene '{gene_name}' not found in GENCODE annotations."
        return result

    # Use all celltypes if not specified
    if celltypes is None:
        celltypes = list(io.ids)

    # Validate celltypes
    valid_celltypes = [ct for ct in celltypes if ct in io.ids]
    if not valid_celltypes:
        result.summary = "None of the specified cell types were found in the data."
        return result

    chrom = gene_info["chrom"]
    tss = gene_info["tss"]
    strand = gene_info["strand"]

    # Define region
    if strand == "+":
        region_start = tss - upstream
        region_end = tss + downstream
    else:
        region_start = tss - downstream
        region_end = tss + upstream

    try:
        # Get normalized track for each celltype
        normalizer = (
            DictNormalizer(io.libsize) if isinstance(io.libsize, dict) else NoNormalizer()
        )

        track = io.get_track(
            chr_name=chrom,
            start=region_start,
            end=region_end,
            query=KeyQuery(valid_celltypes),
            aggregator=ByKeyAggregator(),
            normalizer=normalizer,
            conv_size=100,
            output_format="track_object",
        )

        # Calculate statistics and rank
        stats = {}
        for label, signal in track.convoluted_tracks.items():
            stats[label] = {
                "max_signal": float(np.max(signal)),
                "mean_signal": float(np.mean(signal)),
                "total_signal": float(np.sum(signal)),
            }

        stats_df = pd.DataFrame(stats).T
        stats_df.index.name = "celltype"
        stats_df = stats_df.sort_values("max_signal", ascending=False)
        stats_df = stats_df.reset_index()

        # Plot
        fig, axes = track.plot_tracks_with_genebody(
            gene_annot=gencode,
            genes_to_highlight=[gene_info["name"]],
        )

        # Save figure
        fig_path = output_dir / f"{gene_info['name']}_celltype_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        result.figures.append(str(fig_path))

        # Identify top cell types
        top_ct = stats_df.iloc[0]["celltype"]
        top_signal = stats_df.iloc[0]["max_signal"]

        result.success = True
        result.summary = (
            f"Compared {len(valid_celltypes)} cell types at **{gene_info['name']}** TSS region. "
            f"Highest accessibility in **{top_ct}** (max signal: {top_signal:.2f})."
        )
        result.findings["ranking"] = stats_df
        result.findings["gene_info"] = gene_info
        result.data["track"] = track

        # Write methods file
        data_source = getattr(io, "path", str(type(io).__name__))
        methods_path = write_methods_file(
            output_dir=output_dir,
            analysis_type="Cell Type Comparison Analysis",
            parameters={
                "gene": gene_info["name"],
                "upstream": f"{upstream} bp",
                "downstream": f"{downstream} bp",
                "celltypes_compared": len(valid_celltypes),
                "region": f"{chrom}:{region_start}-{region_end}",
            },
            data_source=data_source,
            description=(
                f"Compared ATAC-seq accessibility across {len(valid_celltypes)} cell types at the "
                f"{gene_info['name']} TSS region. Cell types were ranked by maximum signal intensity."
            ),
            additional_info={
                "gene_info": gene_info,
                "celltypes_analyzed": valid_celltypes,
                "top_celltype": top_ct,
            },
        )
        result.methods_file = str(methods_path)

    except Exception as e:
        result.summary = f"Error comparing cell types: {e}"
        result.warnings.append(str(e))

    return result

"""Shared utilities for extracting fragment information from config.

This module provides unified logic for handling fragment file configurations
across different processors (FragmentProcessor, BPCellsProcessor, etc.).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Common fragment file suffixes to recognize and remove
# Patterns are ordered from most specific to least specific
FRAGMENT_SUFFIXES = [
    # Patterns with additional suffixes (e.g., _with_prefix)
    r"_atac_fragments_with_prefix\.tsv\.gz$",
    r"_ATAC_fragments_with_prefix\.tsv\.gz$",
    r"_atac_fragments_with_prefix\.tsv$",
    r"_ATAC_fragments_with_prefix\.tsv$",
    # Standard patterns
    r"_fragments\.tsv\.gz$",
    r"\.fragments\.tsv\.gz$",
    r"-fragments\.tsv\.gz$",
    r"_atac_fragments\.tsv\.gz$",
    r"-atac_fragments\.tsv\.gz$",
    r"_ATAC_fragments\.tsv\.gz$",
    r"_scATAC_fragments\.tsv\.gz$",
    r"_aggregated_fragments\.tsv\.gz$",
    r"_Aggregated_atac_fragments\.tsv\.gz$",
    r"_Aggregated_ATAC_fragments\.tsv\.gz$",
    r"_aggr_fragments\.tsv\.gz$",
    # Also handle without .gz extension
    r"_fragments\.tsv$",
    r"\.fragments\.tsv$",
    r"-fragments\.tsv$",
    r"_atac_fragments\.tsv$",
    r"-atac_fragments\.tsv$",
    r"_ATAC_fragments\.tsv$",
    r"_scATAC_fragments\.tsv$",
    r"_aggregated_fragments\.tsv$",
    r"_Aggregated_atac_fragments\.tsv$",
    r"_Aggregated_ATAC_fragments\.tsv$",
    r"_aggr_fragments\.tsv$",
]


def extract_id_from_path(frag_path: str | Path, verbose: bool = False) -> str:
    """Extract sample ID from fragment file path by removing common suffixes.
    
    This function recognizes common fragment file naming patterns and removes
    them to extract a clean sample ID. For example:
    - "sample1_atac_fragments.tsv.gz" -> "sample1"
    - "sample1.fragments.tsv.gz" -> "sample1"
    - "sample1-Aggregated_ATAC_fragments.tsv.gz" -> "sample1"
    
    If no recognized suffix is found, extracts the filename stem (without extension).
    
    Args:
        frag_path: Path to fragment file or directory
        verbose: If True, log the extraction process
    
    Returns:
        Extracted sample ID (filename prefix without common fragment suffixes)
    
    Examples:
        >>> extract_id_from_path("sample1_atac_fragments.tsv.gz")
        'sample1'
        >>> extract_id_from_path("sample1.fragments.tsv.gz")
        'sample1'
        >>> extract_id_from_path("/path/to/sample1-Aggregated_ATAC_fragments.tsv.gz")
        'sample1'
        >>> extract_id_from_path("sample1.tsv.gz")  # No recognized suffix
        'sample1'
    """
    frag_path_obj = Path(frag_path)
    
    if frag_path_obj.is_dir():
        # For directories, use the directory name
        base_name = frag_path_obj.name
    else:
        # For files, get the full filename (with extension)
        base_name = frag_path_obj.name
    
    if verbose:
        logger.debug(f"  Extracting ID from: {base_name}")
    
    # Try to match and remove common fragment suffixes
    original_name = base_name
    for suffix_pattern in FRAGMENT_SUFFIXES:
        if re.search(suffix_pattern, base_name, re.IGNORECASE):
            # Remove the matched suffix
            sample_id = re.sub(suffix_pattern, "", base_name, flags=re.IGNORECASE)
            if verbose:
                logger.debug(f"    Matched suffix pattern: {suffix_pattern}")
                logger.debug(f"    Extracted ID: {sample_id} (removed suffix from {original_name})")
            return sample_id
    
    # No recognized suffix found - extract stem (filename without extension)
    sample_id = frag_path_obj.stem.split(".")[0]
    if verbose:
        logger.debug(f"    No recognized suffix found, using stem: {sample_id}")
    
    return sample_id


def extract_fragment_info(
    frag_path: str | Path,
    fragments_config: list[Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Extract fragment information from config or path.
    
    This function provides unified logic for extracting fragment metadata:
    - Matches fragments by path (not index) for robustness
    - Extracts id, celltype, metadata path, and metadata_column
    - Falls back to filename-based extraction if not in config
    
    Args:
        frag_path: Path to fragment file or directory
        fragments_config: List of fragment config objects (with .path, .id, .celltype, etc.)
        verbose: If True, log detailed information about extraction process
    
    Returns:
        Dictionary with keys:
        - id: Fragment ID (from config or filename prefix)
        - celltype: Cell type label (from config or "unknown")
        - metadata: Path to metadata file (if specified)
        - metadata_column: Column name for cell type labels (if specified)
    
    Examples:
        >>> frag_info = extract_fragment_info(
        ...     frag_path="sample1.fragments.tsv.gz",
        ...     fragments_config=[FragmentFile(path="sample1.fragments.tsv.gz", id="sample1", celltype="pbmc")]
        ... )
        >>> frag_info["id"]
        'sample1'
        >>> frag_info["celltype"]
        'pbmc'
        
        >>> # Without config - uses filename
        >>> frag_info = extract_fragment_info("sample1.fragments.tsv.gz")
        >>> frag_info["id"]
        'sample1'
    """
    frag_path_obj = Path(frag_path)
    
    if verbose:
        logger.info(f"Extracting fragment info for: {frag_path}")
    
    # Extract ID using intelligent suffix removal
    frag_prefix = extract_id_from_path(frag_path, verbose=verbose)
    
    sample_id = frag_prefix
    celltype = "unknown"
    metadata_path = None
    metadata_column = None
    matched_config = False
    
    # Try to find matching fragment config by path
    if fragments_config:
        if verbose:
            logger.info(f"  Checking {len(fragments_config)} fragment config(s) for match...")
        
        for frag_config in fragments_config:
            if not hasattr(frag_config, "path"):
                if verbose:
                    logger.debug(f"  Skipping config without 'path' attribute")
                continue
                
            config_path = Path(frag_config.path)
            
            # Match by path comparison
            if config_path.is_dir() and frag_path_obj.is_dir():
                # Both are directories - match by full path or name
                if str(frag_path_obj) == str(config_path) or frag_path_obj.name == config_path.name:
                    matched_config = True
                    sample_id = getattr(frag_config, "id", None) or frag_prefix
                    celltype = getattr(frag_config, "celltype", None) or "unknown"
                    metadata_path = getattr(frag_config, "metadata", None)
                    metadata_column = getattr(frag_config, "metadata_column", None)
                    if verbose:
                        logger.info(f"  ✓ Matched directory: {config_path}")
                    break
            elif not config_path.is_dir() and not frag_path_obj.is_dir():
                # Both are files - match by extracted ID or full path
                config_prefix = extract_id_from_path(config_path, verbose=False)
                if config_prefix == frag_prefix or str(frag_path_obj) == str(config_path):
                    matched_config = True
                    sample_id = getattr(frag_config, "id", None) or frag_prefix
                    celltype = getattr(frag_config, "celltype", None) or "unknown"
                    metadata_path = getattr(frag_config, "metadata", None)
                    metadata_column = getattr(frag_config, "metadata_column", None)
                    if verbose:
                        logger.info(f"  ✓ Matched file: {config_path} (ID: {config_prefix} == {frag_prefix})")
                    break
            elif verbose:
                logger.debug(f"  No match: {config_path} (type mismatch)")
    else:
        if verbose:
            logger.info("  No fragment config provided - using filename-based extraction")
    
    if verbose:
        logger.info(f"  Extracted fragment info:")
        logger.info(f"    ID: {sample_id} {'(from config)' if matched_config and sample_id != frag_prefix else f'(from filename: {frag_prefix})'}")
        logger.info(f"    Celltype: {celltype} {'(from config)' if matched_config and celltype != 'unknown' else '(default: unknown)'}")
        if metadata_path:
            logger.info(f"    Metadata: {metadata_path}")
        if metadata_column:
            logger.info(f"    Metadata column: {metadata_column}")
        if not metadata_path and not metadata_column:
            logger.info(f"    Metadata: None (will use uniform celltype '{celltype}')")
    
    return {
        "id": sample_id,
        "celltype": celltype,
        "metadata": metadata_path,
        "metadata_column": metadata_column,
    }


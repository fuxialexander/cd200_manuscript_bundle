#!/usr/bin/env python
"""Main entry point for the caesar CLI with automatic processor registration."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

from caesar.preprocess.core.config import (
    export_config,
    load_caesar_config,
)
from caesar.preprocess.core.factory import ProcessorFactory, register_all_processors

# Import workflow functions only when needed to avoid circular imports
try:
    from caesar.preprocess.core.workflow import (
        WorkflowEngine,
        load_and_dry_run_workflow,
        load_and_semi_wet_run_workflow,
    )
except ImportError as e:
    logging.error(f"ImportError: {e}")
    # Workflow functionality will be limited if imports fail
    WorkflowEngine = None
    load_and_dry_run_workflow = None
    load_and_semi_wet_run_workflow = None


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Force reconfiguration if already configured
    )
    # Explicitly set root logger level to ensure all loggers inherit it
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Also ensure the handler level is set correctly
    # (basicConfig should do this, but be explicit)
    for handler in root_logger.handlers:
        handler.setLevel(level)

    # Ensure all existing loggers respect the verbose setting
    # Reset any loggers that have a more restrictive level to NOTSET
    # so they inherit from the root logger
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        # Ensure propagation is enabled (default, but be explicit)
        logger.propagate = True
        # If logger has an explicit level that's more restrictive than desired,
        # reset it to NOTSET so it inherits from root
        if logger.level != logging.NOTSET and logger.level > level:
            logger.setLevel(logging.NOTSET)


def _get_overwrite_mapping(command: str) -> str | None:
    """Get the config path for overwrite based on command.

    Returns None for commands that don't have overwrite in their config.

    Args:
        command: Command name

    Returns:
        Config path string or None if not supported
    """
    # Map command names to config section names that have overwrite field
    overwrite_configs = {
        "clustering": "clustering.overwrite",
        "library-size": "library_size.overwrite",
        "bpcells": "bpcells.overwrite",
        "fragment": "fragments.overwrite",
        "normalization": "normalization.overwrite",
        "footprint": "footprints.overwrite",
        "peak-calling": "peaks.overwrite",
        "pseudobulk": "pseudobulk.overwrite",
        "tss-accessibility": "tss_accessibility.overwrite",
        "tes-accessibility": "tes_accessibility.overwrite",
        "cnv-detection": "cnv_detection.overwrite",
        "isolibsize-metacell": "isolibsize_metacell.overwrite",
        "cellstate-tile-count": "cellstate_tile_count.overwrite",
    }
    return overwrite_configs.get(command)


def args_to_hydra_overrides(args: argparse.Namespace, command: str) -> list[str]:
    """Convert argparse namespace to Hydra-style overrides.

    Args:
        args: Parsed command-line arguments
        command: Command name

    Returns:
        List of Hydra-style override strings
    """
    overrides = []

    # Convert args to dict and filter out None values and special attrs
    args_dict = {
        k: v
        for k, v in vars(args).items()
        if v is not None and not k.startswith("_") and k not in ["command", "func"]
    }

    # Handle multiome-specific arguments
    if command in ["multiome", "multiome-pseudobulk"] and "atac" in args_dict:
        atac_files = args_dict.pop("atac")
        import json

        fragments_list = []
        for i, path in enumerate(atac_files):
            path_obj = Path(path)
            frag_prefix = path_obj.stem.split(".")[0]
            frag_dict = {"path": path, "id": frag_prefix}
            fragments_list.append(frag_dict)
        fragments_json = json.dumps(fragments_list)
        overrides.append(f"data.fragments={fragments_json}")

    if (
        command in ["multiome", "multiome-rna", "multiome-pseudobulk"]
        and "rna" in args_dict
    ):
        rna_files = args_dict.pop("rna")
        import json

        # data.rna is a list of string paths (not dicts)
        rna_json = json.dumps(rna_files)
        overrides.append(f"data.rna={rna_json}")

    # Handle multiome config options
    if command == "multiome":
        if "min_shared_cells" in args_dict:
            value = args_dict.pop("min_shared_cells")
            overrides.append(f"multiome.min_shared_cells={value}")
        if "barcode_suffix" in args_dict:
            value = args_dict.pop("barcode_suffix")
            overrides.append(f"multiome.barcode_suffix={value}")

    # Enable processors that default to disabled when run directly via CLI
    # These processors have enabled=False by default since they're optional in workflows
    # but should be enabled when run as standalone commands
    processors_to_enable = {
        "cellstate-tile-count": "cellstate_tile_count.enabled=true",
        "cnv-detection": "cnv_detection.enabled=true",
        "normalization": "normalization.enabled=true",
    }
    if command in processors_to_enable:
        overrides.append(processors_to_enable[command])

    # Handle input files
    if "input" in args_dict:
        input_files = args_dict.pop("input")
        # CNV detection uses cnv_detection.input
        if command == "cnv-detection":
            overrides.append(f"cnv_detection.input={input_files}")
        # Cellstate tile count uses cellstate_tile_count.input
        elif command == "cellstate-tile-count":
            overrides.append(f"cellstate_tile_count.input={input_files}")
        elif isinstance(input_files, list):
            if command == "bam":
                # Build BAM list as JSON to ensure proper list initialization
                import json

                bam_list = []
                for i, path in enumerate(input_files):
                    bam_list.append(
                        {"path": path, "id": f"sample_{i}", "celltype": "bulk"}
                    )
                bam_json = json.dumps(bam_list)
                overrides.append(f"data.bam={bam_json}")
            elif command == "bigwig":
                # Build BigWig list as JSON to ensure proper list initialization
                import json

                bigwig_list = []
                # Get sample_ids and celltypes from args if provided
                sample_ids = args_dict.pop("sample_ids", None)
                celltypes = args_dict.pop("celltypes", None)
                for i, path in enumerate(input_files):
                    path_obj = Path(path)
                    # Use provided sample_id or derive from filename
                    sample_id = (
                        sample_ids[i]
                        if sample_ids and i < len(sample_ids)
                        else path_obj.stem
                    )
                    # Use provided celltype or default to "signal"
                    celltype = (
                        celltypes[i] if celltypes and i < len(celltypes) else "signal"
                    )
                    bigwig_list.append(
                        {"path": path, "id": sample_id, "celltype": celltype}
                    )
                bigwig_json = json.dumps(bigwig_list)
                overrides.append(f"data.bigwig={bigwig_json}")
            elif command == "bpcells":
                # Handle BPCells inputs (can be fragment files or BPCells directories)
                # Build fragments list as a single override to ensure proper list structure
                fragments_list = []
                metadata_files = (
                    args_dict.pop("metadata", None) if "metadata" in args_dict else None
                )

                for i, path in enumerate(input_files):
                    # Extract fragment file prefix for sample_id matching
                    path_obj = Path(path)
                    frag_prefix = path_obj.stem.split(".")[
                        0
                    ]  # Get prefix before .tsv.gz or directory name
                    frag_dict = {
                        "path": path,
                        "id": frag_prefix,
                    }  # Use file prefix as id

                    # Add metadata if provided
                    if metadata_files:
                        if isinstance(metadata_files, list):
                            if len(metadata_files) == 1:
                                # Single metadata file for all inputs
                                # The metadata should have a sample_id column matching fragment prefixes
                                frag_dict["metadata"] = metadata_files[0]
                            elif len(metadata_files) == len(input_files):
                                # One metadata file per input
                                frag_dict["metadata"] = metadata_files[i]
                            else:
                                raise ValueError(
                                    f"Number of metadata files ({len(metadata_files)}) must match "
                                    f"number of inputs ({len(input_files)}) or be 1"
                                )
                        else:
                            frag_dict["metadata"] = metadata_files
                    fragments_list.append(frag_dict)

                # Create single override for fragments list
                import json

                fragments_json = json.dumps(fragments_list)
                overrides.append(f"data.fragments={fragments_json}")
            elif command == "fragment":
                # Handle fragment inputs similar to bpcells
                fragments_list = []
                metadata_files = (
                    args_dict.pop("metadata", None) if "metadata" in args_dict else None
                )

                for i, path in enumerate(input_files):
                    # Extract fragment file prefix for sample_id matching
                    path_obj = Path(path)
                    frag_prefix = path_obj.stem.split(".")[
                        0
                    ]  # Get prefix before .tsv.gz
                    frag_dict = {
                        "path": path,
                        "id": frag_prefix,
                    }  # Use file prefix as id

                    # Add metadata if provided
                    if metadata_files:
                        if isinstance(metadata_files, list):
                            if len(metadata_files) == 1:
                                # Single metadata file for all inputs
                                frag_dict["metadata"] = metadata_files[0]
                            elif len(metadata_files) == len(input_files):
                                # One metadata file per input
                                frag_dict["metadata"] = metadata_files[i]
                            else:
                                raise ValueError(
                                    f"Number of metadata files ({len(metadata_files)}) must match "
                                    f"number of inputs ({len(input_files)}) or be 1"
                                )
                        else:
                            frag_dict["metadata"] = metadata_files
                    fragments_list.append(frag_dict)

                # Create single override for fragments list
                import json

                fragments_json = json.dumps(fragments_list)
                overrides.append(f"data.fragments={fragments_json}")
            else:  # other commands
                for i, path in enumerate(input_files):
                    overrides.append(f"data.fragments[{i}].path={path}")
                    overrides.append(f"data.fragments[{i}].id=sample_{i}")
        else:
            overrides.append(f"data.input={input_files}")

    # Handle output
    if "output" in args_dict:
        output = args_dict.pop("output")
        # Map CLI output argument to config field based on processor type
        if command in [
            "pseudobulk",
            "clustering",
            "peak-calling",
            "normalization",
            "footprint",
            "library-size",
            "multiome-pseudobulk",
            "isolibsize-metacell",
            "cellstate-tile-count",
        ]:
            # Transformation processors expect a specific output file path
            section = (
                "footprints" if command == "footprint" else command.replace("-", "_")
            )
            overrides.append(f"{section}.output={output}")
        else:
            # Preprocessing processors expect output directory
            overrides.append(f"output.dir={output}")
    elif "output_dir" in args_dict:
        output_dir = args_dict.pop("output_dir")
        overrides.append(f"output.dir={output_dir}")

    # Handle mode -> workflow.type (for preprocessing commands)
    if "mode" in args_dict:
        mode = args_dict.pop("mode")
        if mode != "auto":
            overrides.append(f"workflow.type={mode}")

    # Handle special argument mappings
    arg_mappings = {
        "n_jobs": "resources.n_jobs",
        "min_fragments": "fragments.min_fragments"
        if command != "bpcells"
        else "bpcells.min_fragments",
        "max_fragments": "fragments.max_fragments"
        if command != "bpcells"
        else "bpcells.max_fragments",
        "single_end": "bam.single_end",
        "tn5_shift": "bam.tn5_shift",  # Deprecated, kept for backwards compatibility
        "backend": "multiome.backend" if command == "multiome" else "bam.backend",
        "assembly": "assembly",  # Top-level config field
        "n_components": "clustering.n_components",
        "resolution": "clustering.resolution",
        "device": "clustering.device",
        "no_blacklist": (lambda v: f"cellstate_tile_count.apply_blacklist={not v}")
        if command == "cellstate-tile-count"
        else (lambda v: f"clustering.blacklist_filtering={not v}"),
        "overwrite": _get_overwrite_mapping(command),
        "groupby": "cellstate_tile_count.groupby"
        if command == "cellstate-tile-count"
        else "pseudobulk.groupby",
        "grouping_file": "pseudobulk.grouping_file",
        "metadata_column": "pseudobulk.metadata_column",
        "external_metadata": "cellstate_tile_count.external_metadata"
        if command == "cellstate-tile-count"
        else "pseudobulk.external_metadata",
        "barcode_column": "cellstate_tile_count.barcode_column"
        if command == "cellstate-tile-count"
        else "pseudobulk.barcode_column",
        "min_libsize": "cellstate_tile_count.min_libsize"
        if command == "cellstate-tile-count"
        else "pseudobulk.min_libsize",
        "qvalue": "peaks.qvalue",
        "method": "normalization.method"
        if command == "normalization"
        else ("peaks.method" if command == "peak-calling" else None),
        "no_extract_summits": (lambda v: f"peaks.extract_summits={not v}")
        if command == "peak-calling"
        else None,
        "summit_threshold": "peaks.summit_threshold"
        if command == "peak-calling"
        else None,
        "generate_background": "peaks.generate_background"
        if command == "peak-calling"
        else None,
        "background_reference_celltype": "peaks.background_reference_celltype"
        if command == "peak-calling"
        else None,
        "classify_peaks": "peaks.classify_peaks" if command == "peak-calling" else None,
        "extend": "footprints.extend"
        if command == "footprint"
        else None,  # extend only valid for footprint
        "peak_file": "normalization.peak_file",
        "reference": "normalization.reference",
        "write_output": "normalization.write_output",
        "motif_zarr": "footprints.motif_zarr",
        "footprint_mode": "footprints.mode",
        "cutoff_range": "footprints.cutoff_range",
        "core_extend": "footprints.core_extend",
        "context_extend": "footprints.context_extend",
        "n_bins": "footprints.n_bins",
        "shift_offsets": "footprints.shift_offsets",
        "n_processes": "footprints.n_processes",
        "blacklist": "blacklist_file",
        # Fragment-specific mappings
        "barcode_col": "fragments.barcode_col"
        if command == "fragment"
        else ("bpcells.barcode_col" if command == "bpcells" else None),
        "sample_id_col": "fragments.sample_id_col"
        if command == "fragment"
        else ("bpcells.sample_id_col" if command == "bpcells" else None),
        # CUT&RUN-specific mappings (fragment processor)
        "assay_type": "fragments.assay_type" if command == "fragment" else None,
        "antibody_target": None,  # Handled separately for per-fragment config
        "spike_in_chroms": (
            lambda v: f"fragments.spike_in_chroms={json.dumps(v)}"
        ) if command == "fragment" else None,
        "spike_in_factor_file": "fragments.spike_in_factor_file"
        if command == "fragment" else None,
        "no_validate_fragment_sizes": (
            lambda v: f"fragments.validate_fragment_sizes={not v}"
        ) if command == "fragment" else None,
        # BPCells-specific mappings
        "input_format": "bpcells.input_format",
        "annotation_col": "bpcells.annotation_col",
        "cluster_mapping": "bpcells.cluster_mapping_file",
        "target_groups": lambda v: f"bpcells.target_groups=[{','.join(str(x) for x in v)}]"
        if isinstance(v, list)
        else f"bpcells.target_groups=[{v}]",
        "group_by": "bpcells.group_by",
        "bin_size": "bpcells.bin_size" if command == "bpcells" else None,
        "threads": "bpcells.threads" if command == "bpcells" else "resources.n_jobs",
        "cleanup_temp_dirs": "bpcells.cleanup_temp_dirs"
        if command == "bpcells"
        else None,
        # BigWig-specific mappings
        "chunk_size": "bigwig.chunk_size" if command == "bigwig" else None,
        # Iso-libsize metacell mappings
        "target_libsize": "isolibsize_metacell.target_libsize"
        if command == "isolibsize-metacell"
        else None,
        "n_metacells": "isolibsize_metacell.n_metacells"
        if command == "isolibsize-metacell"
        else None,
        "embedding_key": "isolibsize_metacell.embedding_key"
        if command == "isolibsize-metacell"
        else None,
        "n_neighbors": "isolibsize_metacell.n_neighbors"
        if command == "isolibsize-metacell"
        else None,
        "min_libsize_fraction": "isolibsize_metacell.min_libsize_fraction"
        if command == "isolibsize-metacell"
        else None,
        "max_cells_per_metacell": "isolibsize_metacell.max_cells_per_metacell"
        if command == "isolibsize-metacell"
        else None,
        "seed": "isolibsize_metacell.seed"
        if command == "isolibsize-metacell"
        else None,
        "metacell_column": "isolibsize_metacell.metacell_column"
        if command == "isolibsize-metacell"
        else None,
        "adata_path": "isolibsize_metacell.adata_path"
        if command == "isolibsize-metacell"
        else None,
        "use_shared_cells_only": "isolibsize_metacell.use_shared_cells_only"
        if command == "isolibsize-metacell"
        else None,
        "tolerance": "isolibsize_metacell.tolerance"
        if command == "isolibsize-metacell"
        else None,
        # TSS/TES accessibility mappings
        "upstream": "tss_accessibility.upstream"
        if command == "tss-accessibility"
        else ("tes_accessibility.upstream" if command == "tes-accessibility" else None),
        "downstream": "tss_accessibility.downstream"
        if command == "tss-accessibility"
        else (
            "tes_accessibility.downstream" if command == "tes-accessibility" else None
        ),
        "gene_list": "tss_accessibility.gene_list"
        if command == "tss-accessibility"
        else (
            "tes_accessibility.gene_list" if command == "tes-accessibility" else None
        ),
        "gene_set": "tss_accessibility.gene_set"
        if command == "tss-accessibility"
        else ("tes_accessibility.gene_set" if command == "tes-accessibility" else None),
        "gencode_version": "tss_accessibility.gencode_version"
        if command == "tss-accessibility"
        else (
            "tes_accessibility.gencode_version"
            if command == "tes-accessibility"
            else None
        ),
        "dataset_name": "tss_accessibility.dataset_name"
        if command == "tss-accessibility"
        else (
            "tes_accessibility.dataset_name" if command == "tes-accessibility" else None
        ),
        "tile_size": "tss_accessibility.tile_size"
        if command == "tss-accessibility"
        else (
            "tes_accessibility.tile_size"
            if command == "tes-accessibility"
            else (
                "cnv_detection.tile_size"
                if command == "cnv-detection"
                else (
                    "cellstate_tile_count.tile_size"
                    if command == "cellstate-tile-count"
                    else None
                )
            )
        ),
        "no_aggregate": (lambda v: f"tss_accessibility.aggregate={not v}")
        if command == "tss-accessibility"
        else (
            (lambda v: f"tes_accessibility.aggregate={not v}")
            if command == "tes-accessibility"
            else None
        ),
        "all_transcripts": "tss_accessibility.all_transcripts"
        if command == "tss-accessibility"
        else (
            "tes_accessibility.all_transcripts"
            if command == "tes-accessibility"
            else None
        ),
        # CNV detection mappings
        "regenerate_tiles": "cnv_detection.regenerate_tiles"
        if command == "cnv-detection"
        else None,
        "no_gc_correction": (lambda v: f"cnv_detection.gc_correction={not v}")
        if command == "cnv-detection"
        else None,
        "gc_method": "cnv_detection.gc_method" if command == "cnv-detection" else None,
        "smooth_method": "cnv_detection.smooth_method"
        if command == "cnv-detection"
        else None,
        "window_size": "cnv_detection.window_size"
        if command == "cnv-detection"
        else None,
        "per_cell_segment": (lambda v: f"cnv_detection.joint_segment={not v}")
        if command == "cnv-detection"
        else None,
        "min_segment_size": "cnv_detection.min_segment_size"
        if command == "cnv-detection"
        else None,
        "loss_threshold": "cnv_detection.loss_threshold"
        if command == "cnv-detection"
        else None,
        "gain_threshold": "cnv_detection.gain_threshold"
        if command == "cnv-detection"
        else None,
        "no_iterative": (lambda v: f"cnv_detection.iterative={not v}")
        if command == "cnv-detection"
        else None,
        "diploid_threshold": "cnv_detection.diploid_threshold"
        if command == "cnv-detection"
        else None,
        # Cell-state tile count mappings
        "dense": (lambda v: f"cellstate_tile_count.sparse_output={not v}")
        if command == "cellstate-tile-count"
        else None,
        "normalize": "cellstate_tile_count.normalize"
        if command == "cellstate-tile-count"
        else None,
    }

    # Apply mappings
    for arg_name, config_path in arg_mappings.items():
        if arg_name in args_dict:
            value = args_dict.pop(arg_name)
            # Skip if config_path is None (mapping not valid for this command)
            if config_path is None:
                continue
            if callable(config_path):
                override = config_path(value)
            else:
                override = f"{config_path}={value}"
            overrides.append(override)

    # Handle command-specific mappings for min_cells
    if "min_cells" in args_dict:
        value = args_dict.pop("min_cells")
        if command == "pseudobulk":
            overrides.append(f"pseudobulk.min_cells_per_group={value}")
        elif command == "cellstate-tile-count":
            overrides.append(f"cellstate_tile_count.min_cells_per_group={value}")
        else:
            overrides.append(f"fragments.min_cells={value}")

    # Handle groupby alias for isolibsize-metacell (maps to metacell_column)
    if "groupby" in args_dict and command == "isolibsize-metacell":
        value = args_dict.pop("groupby")
        if value is not None:
            overrides.append(f"isolibsize_metacell.metacell_column={value}")

    # Handle --tn5 / --chip signal mode for BAM processor
    if command == "bam":
        # The argparse mutually exclusive group gives us 'tn5' and 'chip' args
        # Default is tn5=True when neither is specified
        chip_mode = args_dict.pop("chip", False)
        args_dict.pop("tn5", None)  # Remove tn5 from args_dict (it's the default)
        args_dict.pop("tn5_shift_deprecated", None)  # Remove deprecated arg
        if chip_mode:
            overrides.append("bam.signal_mode=chip")
        else:
            overrides.append("bam.signal_mode=tn5")

    # Handle remaining arguments generically
    for key, value in args_dict.items():
        if key not in [
            "config",
            "export_config",
            "verbose",
            "metadata",
        ]:  # metadata handled separately
            # Convert key to config path
            config_key = key.replace("-", "_")
            # Map to appropriate config section based on command
            if command == "bpcells" and config_key in [
                "output_type",
                "input_format",
                "barcode_col",
                "annotation_col",
                "sample_id_col",
                "cluster_mapping_file",
                "target_groups",
                "group_by",
                "bin_size",
            ]:
                overrides.append(f"bpcells.{config_key}={value}")
            else:
                overrides.append(f"{config_key}={value}")

    return overrides


def unified_command_handler(args: argparse.Namespace) -> int:
    """Unified command handler for all processors.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    command = args.command

    try:
        # Build Hydra overrides from arguments
        overrides = args_to_hydra_overrides(args, command)

        # Load config with overrides
        config = load_caesar_config(
            config_path=args.config
            if hasattr(args, "config") and args.config
            else None,
            overrides=overrides,
        )

        # Export config if requested
        if hasattr(args, "export_config") and args.export_config:
            export_config(config, args.export_config)
            print(f"Exported configuration to: {args.export_config}")
            # Check if we should also process (i.e., if required inputs are provided)
            if not hasattr(args, "input") or not args.input:
                return 0  # Only export, don't process

        # Create and run processor
        processor = ProcessorFactory.create(command, config)
        result = processor.process()

        # Standard success output
        print(f"Successfully completed {command}")
        if isinstance(result, dict) and "output_path" in result:
            print(f"Output: {result['output_path']}")
        elif isinstance(result, dict):
            # Print any interesting results
            for key, value in result.items():
                if key.startswith("n_") or key in [
                    "output_path",
                    "n_groups",
                    "n_clusters",
                    "n_peaks",
                ]:
                    print(f"{key.replace('_', ' ').title()}: {value}")

        return 0

    except Exception as e:
        logging.error(f"{command} failed: {e}")
        import traceback

        if hasattr(args, "verbose") and args.verbose:
            traceback.print_exc()
        return 1


def workflow_command_handler(args: argparse.Namespace) -> int:
    """Special handler for workflow commands.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Handle verbose flag - check both parent parser (before workflow) and
    # workflow parser (after workflow) arguments
    verbose = False
    if hasattr(args, "verbose"):
        verbose = args.verbose
    # Also check if parent parser set it (when using caesar -v workflow)
    if not verbose and hasattr(args, "__dict__"):
        # Check if verbose was set in parent namespace
        verbose = getattr(args, "_parent_verbose", False)

    # Ensure verbose logging is set up
    if verbose:
        setup_logging(verbose=True)

    try:
        # Build overrides from args
        overrides = []

        # Process positional arguments to extract workflow name and additional overrides
        positional_args = getattr(args, "args", [])
        # Filter out "run" as it's redundant (workflow already runs by default)
        positional_args = [arg for arg in positional_args if arg != "run"]

        workflow_choices = [
            "single-cell-complete",
            "single-cell-simple",
            "bpcells-import-fragment",
            "bpcells-complete",
            "bulk",
            "bulk-simple",
            "multiome",
            "reprocessing",
            "footprint",
            # Legacy names for backward compatibility
            "single-cell",
            "custom",
        ]

        workflow_name_from_cli = None
        if positional_args and positional_args[0] in workflow_choices:
            workflow_name_from_cli = positional_args[0]
            overrides.extend(positional_args[1:])
        else:
            overrides.extend(positional_args)

        # If workflow name specified via CLI, add it to overrides
        if workflow_name_from_cli:
            overrides.append(f"workflow.name={workflow_name_from_cli}")

        # Export config if requested
        if hasattr(args, "export_config") and args.export_config:
            config = load_caesar_config(config_path=args.config, overrides=overrides)
            export_config(config, args.export_config)
            print(f"Exported configuration to: {args.export_config}")
            if args.dry_run:
                return 0

        # Handle dry run
        if args.dry_run:
            if load_and_dry_run_workflow:
                results = load_and_dry_run_workflow(args.config, overrides=overrides)
                return 0 if results["ready"] else 1
            else:
                print("Dry run functionality not available due to import issues")
                return 1

        # Handle semi-wet run
        if hasattr(args, "semi_wet_run") and args.semi_wet_run:
            if load_and_semi_wet_run_workflow:
                results = load_and_semi_wet_run_workflow(
                    args.config, overrides=overrides
                )
                return 0
            else:
                print("Semi-wet run functionality not available due to import issues")
                return 1

        # Execute workflow using existing workflow engine
        if not WorkflowEngine:
            print("Workflow engine not available due to import issues")
            return 1

        config = load_caesar_config(config_path=args.config, overrides=overrides)

        # Auto-detect workflow if not specified
        if not workflow_name_from_cli and not hasattr(config.workflow, "name"):
            # Auto-detection logic (simplified from original)
            if hasattr(config.data, "fragments") and config.data.fragments:
                config.workflow.name = "single-cell-complete"  # Default for fragments
            elif hasattr(config.data, "bam") and config.data.bam:
                config.workflow.name = "bulk"
            elif hasattr(config.data, "input") and config.data.input:
                # Check input type for sparse vs dense zarr
                if isinstance(config.data.input, str) and "dense" in config.data.input:
                    config.workflow.name = "footprint-jaspar"  # Default for dense zarr
                else:
                    config.workflow.name = "reprocessing"  # Default for sparse zarr
            else:
                raise ValueError("Could not auto-detect workflow type")

        # Execute workflow
        engine = WorkflowEngine(config)
        results = engine.execute()

        print(f"Successfully completed workflow: {config.workflow.name}")
        print("Results:")
        for step_name, result in results["results"].items():
            output = result.get("output_path", "completed")
            print(f"  - {step_name}: {output}")

        return 0

    except Exception as e:
        logging.error(f"Workflow execution failed: {e}")
        import traceback

        if hasattr(args, "verbose") and args.verbose:
            traceback.print_exc()
        return 1


def create_unified_parser() -> argparse.ArgumentParser:
    """Create the unified argument parser with automatic processor registration.

    Returns:
        Configured ArgumentParser
    """
    # Register all processors
    register_all_processors()

    parser = argparse.ArgumentParser(
        prog="caesar",
        description=(
            "Caesar - A comprehensive framework for single-cell multiomics data analysis\n\n"
            "Caesar provides a unified interface for processing, analyzing, and annotating\n"
            "chromatin accessibility data from both single-cell and bulk experiments.\n\n"
            "Key features:\n"
            "  • Efficient Zarr-based storage for massive datasets\n"
            "  • Unified processing for single-cell and bulk data\n"
            "  • Modular architecture for flexible analysis pipelines\n"
            "  • Integration with standard genomics tools and formats\n"
            "  • Scalable from laptop to HPC cluster\n\n"
            "Choose a command below to get started. For detailed help on any command:\n"
            "  caesar <command> --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global arguments
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available preprocessing commands",
        description="Choose a command to run. Use 'caesar <command> --help' for command-specific help.",
    )

    # Auto-register all processor commands
    for command_name in ProcessorFactory.get_registered_commands():
        processor_class = ProcessorFactory.get_processor_class(command_name)

        # Create subparser for this command
        cmd_parser = subparsers.add_parser(
            command_name,
            help=processor_class.get_description(),
            description=processor_class.get_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Add processor-specific arguments
        processor_class.add_cli_args(cmd_parser)

        # Set the unified handler
        cmd_parser.set_defaults(func=unified_command_handler)

    # Add special workflow command
    add_workflow_parser(subparsers)

    # Add batch command (can be handled by existing logic)
    add_batch_parser(subparsers)

    return parser


def add_workflow_parser(subparsers) -> None:
    """Add workflow command parser."""
    parser = subparsers.add_parser(
        "workflow",
        help="Run predefined analysis workflows",
        description=(
            "Run complete analysis pipelines with predefined or custom workflows.\n\n"
            "Available workflows:\n"
            "  • single-cell-complete    - Full single-cell ATAC-seq pipeline (Zarr)\n"
            "  • single-cell-simple      - Alias for single-cell-complete\n"
            "  • bpcells-import-fragment - Import fragments to BPCells format\n"
            "  • bpcells-complete        - Full pipeline with BPCells backend\n"
            "  • bulk                    - Bulk ATAC-seq pipeline (fragments or BAM)\n"
            "  • bulk-simple             - Alias for bulk\n"
            "  • multiome                - Multimodal single-cell analysis\n"
            "  • reprocessing            - Reprocess existing zarr data\n"
            "  • footprint               - TF footprinting analysis\n\n"
            "Workflow is auto-detected from input data if not specified:\n"
            "  • Fragment files → single-cell or bulk (based on celltype presence)\n"
            "  • BAM files → bulk workflow\n"
            "  • Sparse zarr → reprocessing workflow\n"
            "  • Dense zarr → footprint workflow\n\n"
            "Examples:\n"
            "  caesar workflow -c config.yaml                                # Auto-detect\n"
            "  caesar workflow -c config.yaml --dry-run                      # Preview steps\n"
            "  caesar workflow -c config.yaml workflow.name=single-cell-complete\n"
            "  caesar workflow -c config.yaml clustering.resolution=1.2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Configuration file (YAML)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show workflow steps without executing"
    )
    parser.add_argument(
        "--semi-wet-run",
        action="store_true",
        help="Execute workflow but skip existing outputs",
    )
    parser.add_argument(
        "--nextflow-mode",
        action="store_true",
        help="Running from Nextflow (internal use)",
    )
    parser.add_argument(
        "--export-config", metavar="PATH", help="Export resolved configuration to file"
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Hydra-style overrides (e.g., workflow.name=bulk clustering.resolution=1.2)",
    )

    parser.set_defaults(func=workflow_command_handler)


def simplified_batch_handler(args: argparse.Namespace) -> int:
    """Simplified batch handler that doesn't depend on the old CLI."""
    print("Batch processing functionality not yet implemented in new CLI architecture")
    print("Please use individual processor commands for now")
    return 1


def add_batch_parser(subparsers) -> None:
    """Add batch command parser (simplified version)."""
    parser = subparsers.add_parser(
        "batch",
        help="Process multiple samples in parallel (not yet implemented)",
        description="Process multiple samples in parallel using batch configuration. (Not yet implemented in new CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Batch configuration file (YAML)"
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=1,
        help="Number of samples to process in parallel",
    )
    parser.add_argument(
        "--executor",
        choices=["local", "slurm", "sge"],
        default="local",
        help="Execution backend",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without executing",
    )
    parser.add_argument(
        "--export-config", metavar="PATH", help="Export resolved configuration to file"
    )

    parser.set_defaults(func=simplified_batch_handler)


def main() -> int:
    """Main entry point for the caesar CLI.

    Returns:
        Exit code
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

    # Parse arguments
    parser = create_unified_parser()
    args = parser.parse_args()

    # Handle verbose flag - works both before and after subcommand
    # - 'caesar -v workflow ...' sets verbose on parent parser
    # - 'caesar workflow -v ...' sets verbose on subcommand parser
    verbose = getattr(args, "verbose", False)

    # Setup logging (workflow handler will re-setup if needed)
    setup_logging(verbose)

    # Handle command
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

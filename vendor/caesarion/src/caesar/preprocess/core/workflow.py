"""Workflow engine with simplified processor mapping.

This replaces the complex PROCESSOR_REGISTRY and parameter mapping
with direct processor instantiation using the full CaesarConfig.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from caesar.preprocess.core.config import (
    WORKFLOW_TEMPLATES,
    CaesarConfig,
    WorkflowStep,
    validate_caesar_config,
)
from caesar.preprocess.processor.add_expression import AddExpressionProcessor
from caesar.preprocess.processor.add_peak_processor import AddPeakProcessor
from caesar.preprocess.processor.bam_processor import BAMProcessor
from caesar.preprocess.processor.bpcells_processor import BPCellsProcessor
from caesar.preprocess.processor.clustering_processor import ClusteringProcessor
from caesar.preprocess.processor.footprint_processor import FootprintProcessor

# Import processors directly - no indirection
from caesar.preprocess.processor.fragment_processor import FragmentProcessor
from caesar.preprocess.processor.library_size_processor import LibrarySizeProcessor
from caesar.preprocess.processor.metadata_filter_processor import (
    MetadataFilterProcessor,
)
from caesar.preprocess.processor.motif_curator_processor import (
    NonRedundantMotifCuratorProcessor,
)
from caesar.preprocess.processor.motif_processor import NonRedundantMotifProcessor
from caesar.preprocess.processor.multiome_processor import MultiomeProcessor
from caesar.preprocess.processor.multiome_pseudobulk_processor import (
    MultiomePseudobulkProcessor,
)
from caesar.preprocess.processor.multiome_rna_processor import MultiomeRNAProcessor
from caesar.preprocess.processor.normalization_processor import NormalizationProcessor
from caesar.preprocess.processor.peak_calling_processor import PeakCallingProcessor
from caesar.preprocess.processor.pseudobulk_processor import PseudobulkProcessor
from caesar.preprocess.processor.split_annotator_processor import (
    SplitAnnotatorProcessor,
)
from caesar.preprocess.processor.peak_deduplication_processor import (
    PeakDeduplicationProcessor,
)
from caesar.preprocess.processor.differential_signal_processor import (
    DifferentialSignalProcessor,
)
from caesar.preprocess.processor.isolibsize_metacell_processor import (
    IsoLibsizeMetacellProcessor,
)

if TYPE_CHECKING:
    from caesar.preprocess.core.base import BaseProcessor

# Direct processor mapping - simple and clear
PROCESSORS: dict[str, type[BaseProcessor]] = {
    # Data processors
    "fragments": FragmentProcessor,
    "bam": BAMProcessor,
    "bpcells": BPCellsProcessor,
    "clustering": ClusteringProcessor,
    "pseudobulk": PseudobulkProcessor,
    "peaks": PeakCallingProcessor,
    "normalization": NormalizationProcessor,
    "library_size_annotation": LibrarySizeProcessor,
    "footprints": FootprintProcessor,
    "add_peak": AddPeakProcessor,
    "add_expression": AddExpressionProcessor,
    "metadata_filter": MetadataFilterProcessor,
    "non_redundant_motif": NonRedundantMotifProcessor,
    "non_redundant_motif_curation": NonRedundantMotifCuratorProcessor,
    "multiome": MultiomeProcessor,
    "multiome_rna": MultiomeRNAProcessor,
    "multiome_pseudobulk": MultiomePseudobulkProcessor,
    "split_annotator": SplitAnnotatorProcessor,
    "peak_deduplication": PeakDeduplicationProcessor,
    "differential_signal": DifferentialSignalProcessor,
    "isolibsize_metacell": IsoLibsizeMetacellProcessor,
}


class WorkflowEngine:
    """Execute multi-step preprocessing workflows with configuration.

    This engine simplifies workflow execution by:
    1. Using a single CaesarConfig for all processors
    2. Direct processor mapping without complex registries
    3. Simple parameter passing - processors get the full config
    """

    def __init__(self, config: CaesarConfig):
        """Initialize workflow engine with unified configuration.

        Args:
            config: Full CaesarConfig object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate configuration
        validate_caesar_config(config)

        # Track execution state
        self.results: dict[str, Any] = {}
        self.current_output: Path | None = None

        # Store processor config overrides (bypasses OmegaConf validation)
        # Key: processor_name, Value: config object
        self._processor_config_overrides: dict[str, Any] = {}

    def execute(self) -> dict[str, Any]:
        """Execute the workflow defined in configuration.

        Returns:
            Workflow execution results
        """
        start_time = time.time()

        # Get workflow steps
        steps = self._get_workflow_steps()

        if not steps:
            raise ValueError("No workflow steps defined")

        self.logger.info(f"Starting workflow with {len(steps)} steps")

        # Check for BPCells processor and warn about compatibility
        bpcell_step_indices = [i for i, step in enumerate(steps) if step.processor == "bpcells"]
        if bpcell_step_indices:
            # Check for incompatible processors that come AFTER any BPCells step
            # Note: LibrarySizeProcessor and PeakCallingProcessor now support BPCells
            incompatible_processors = []
            incompatible_processor_names = {
                "pseudobulk": "PseudoBulkProcessor",  # Use BPCells celltype_dense output instead
                # Note: NormalizationProcessor now supports BPCells (stores params, preserves raw counts)
            }

            for bpcell_idx in bpcell_step_indices:
                for i in range(bpcell_idx + 1, len(steps)):
                    proc_name = steps[i].processor
                    if proc_name in incompatible_processor_names:
                        incompatible_processors.append(
                            f"{incompatible_processor_names[proc_name]} (step {i+1})"
                        )

            if incompatible_processors:
                self.logger.warning(
                    f"⚠️  BPCells processor detected with incompatible downstream processors: "
                    f"{', '.join(set(incompatible_processors))}. "
                    f"These processors require Zarr format and are not compatible with BPCells output. "
                    f"\nReason: They use Zarr-specific IO classes (SingleSparseZarrIO, CelltypeDenseZarrIO, etc.) "
                    f"and methods that don't exist in BPCells format. "
                    f"\nSee BPCellsProcessor class docstring for detailed explanations of each incompatibility. "
                    f"\nAlternatives: Use BPCells celltype_dense output directly, or convert to Zarr first."
                )
            else:
                self.logger.info(
                    "✓ BPCells processor detected - Fully compatible with core Caesar operations. "
                    "Supported: clustering (via AnnData), peak calling (dense formats), library size, "
                    "normalization (stores params), footprinting (IO methods). "
                    "See docs/bpcells_backend_support.md for details."
                )

        # Execute each step
        for i, step in enumerate(steps):
            self.logger.info(f"Step {i + 1}/{len(steps)}: {step.name}")

            try:
                result = self._execute_step(step)
                self.results[step.name] = result

                # Update current output if available
                # For AnnotationProcessor steps (in-place modifications), preserve current_output
                # even if they don't return output_path, since they modify the same store
                if "output_path" in result and result["output_path"] is not None:
                    self.current_output = Path(result["output_path"])
                # If AnnotationProcessor doesn't return output_path, keep current_output unchanged
                # (it modifies the same store in-place)

            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                self.logger.error(f"Step {step.name} failed: {e}")
                self.logger.error(f"Full traceback:\n{error_traceback}")
                if not self.config.workflow.continue_on_error:
                    raise
                self.results[step.name] = {"error": str(e), "traceback": error_traceback}

        runtime = time.time() - start_time

        return {
            "workflow": self.config.workflow.name or "custom",
            "steps_executed": len(steps),
            "results": self.results,
            "final_output": str(self.current_output) if self.current_output else None,
            "runtime": runtime,
        }

    def _get_workflow_steps(self) -> list[WorkflowStep]:
        """Get workflow steps from configuration.

        Returns:
            List of workflow steps
        """
        # Check if using a predefined workflow
        if (
            self.config.workflow.name
            and self.config.workflow.name in WORKFLOW_TEMPLATES
        ):
            self.logger.info(f"Using predefined workflow: {self.config.workflow.name}")
            return WORKFLOW_TEMPLATES[self.config.workflow.name]()

        # Use custom steps
        if self.config.workflow.steps:
            return self.config.workflow.steps

        # Auto-detect workflow based on input data
        return self._auto_detect_workflow()

    def _auto_detect_workflow(self) -> list[WorkflowStep]:
        """Auto-detect workflow based on input data and backend.

        Returns:
            Appropriate workflow steps
        """
        # Check what data is available
        has_fragments = bool(self.config.data.fragments)
        has_bam = bool(self.config.data.bam)
        has_input = bool(self.config.data.input)

        # Check backend setting
        backend = getattr(self.config.output, "backend", "zarr")

        if has_input:
            # Working with existing zarr - footprint or reprocessing workflow
            self.logger.info("Detected existing zarr input - using footprint workflow")
            return WORKFLOW_TEMPLATES["footprint"]()
        elif has_fragments:
            # Fragment-based workflow - check backend
            if backend == "bpcells":
                self.logger.info("Detected fragment files with BPCells backend - using BPCells complete workflow")
                return WORKFLOW_TEMPLATES["bpcells-complete"]()
            else:
                self.logger.info("Detected fragment files - using single-cell workflow")
                return WORKFLOW_TEMPLATES["single-cell"]()
        elif has_bam:
            # BAM-based workflow
            self.logger.info("Detected BAM files - using bulk workflow")
            return WORKFLOW_TEMPLATES["bulk"]()
        else:
            raise ValueError("No input data detected")

    def _execute_step(self, step: WorkflowStep) -> dict[str, Any]:
        """Execute a single workflow step.

        Args:
            step: Workflow step to execute

        Returns:
            Step execution results
        """
        # Handle special 'auto' processor
        if step.processor == "auto":
            processor_name = self._auto_select_processor()
        else:
            processor_name = step.processor

        # Get processor class
        if processor_name not in PROCESSORS:
            raise ValueError(f"Unknown processor: {processor_name}")

        processor_class = PROCESSORS[processor_name]

        # Select processor-specific config based on step name (if needed)
        # This makes the workflow explicit and removes complex config selection logic
        processor_config = self._select_processor_config(processor_name, step.name)
        if processor_name == "bpcells" and processor_config:
            # Store override in our dict (bypasses OmegaConf validation)
            self._processor_config_overrides["bpcells"] = processor_config
            # Also set on config for processor to access (using object.__setattr__ to bypass OmegaConf)
            object.__setattr__(self.config, "_bpcells_config_override", processor_config)

        # Create processor instance with config
        processor = processor_class(self.config)

        # Set input path for processors that need it
        if self.current_output:
            processor.input_path = self.current_output

        # Execute processor
        return processor.process()

    def _select_bpcells_config(self, step_name: str):
        """Select the appropriate BPCells config based on workflow step name.

        Args:
            step_name: Name of the workflow step

        Returns:
            Selected BPCellsConfig or None
        """
        # Map step names to config sections
        step_to_config = {
            "import_fragments": "bpcells_import",
            "generate_anndata": "bpcells_anndata",
            "pseudobulk_celltype_dense": "bpcells_pseudobulk",
        }

        config_section = step_to_config.get(step_name)
        if config_section:
            config = getattr(self.config, config_section, None)
            if config is not None:
                return config

        # Fallback: use legacy bpcells config if step-specific config not found
        # This provides backward compatibility
        if hasattr(self.config, "bpcells") and self.config.bpcells:
            return self.config.bpcells

        return None

    def _select_processor_config(self, processor_name: str, step_name: str):
        """Select the appropriate config for a processor based on workflow step name.

        This method provides a consistent pattern for config selection across all processors.
        Currently only BPCells processor needs step-specific config selection, but this
        pattern can be extended for other processors if needed.

        Args:
            processor_name: Name of the processor
            step_name: Name of the workflow step

        Returns:
            Selected config object or None (processor will use default)
        """
        if processor_name == "bpcells":
            return self._select_bpcells_config(step_name)
        # Add other processors here if they need step-specific config selection
        # elif processor_name == "fragments":
        #     return self._select_fragments_config(step_name)

        return None

    def _auto_select_processor(self) -> str:
        """Auto-select processor based on input type.

        Returns:
            Processor name
        """
        if self.config.data.fragments:
            return "fragments"
        elif self.config.data.bam:
            return "bam"
        else:
            raise ValueError("Cannot auto-select processor - no input data")

    def dry_run(self) -> dict[str, Any]:
        """Perform a dry run of the workflow without executing processors.

        This method validates the configuration, shows what steps would be
        executed, and identifies any potential issues without making changes.

        Returns:
            Dictionary containing dry run analysis results
        """
        print("\n" + "=" * 60)
        print("=== CAESAR WORKFLOW DRY RUN ===")
        print("=" * 60 + "\n")

        # Configuration summary
        print("Configuration Summary:")
        print(f"- Assembly: {self.config.assembly}")

        # Input summary
        if self.config.data.fragments:
            print(f"- Input: {len(self.config.data.fragments)} fragment file(s)")
        elif self.config.data.bam:
            print(f"- Input: {len(self.config.data.bam)} BAM file(s)")
        elif self.config.data.input:
            print(f"- Input: Existing zarr at {self.config.data.input}")
        else:
            print("- Input: ⚠️  No input data specified")

        print(
            f"- Output: {self.config.output.dir} ({self.config.output.backend} backend)"
        )
        print(
            f"- Resources: {self.config.resources.n_jobs} jobs, {self.config.resources.memory_limit} memory"
        )

        # Get workflow steps
        validation_issues = []  # Initialize before try block
        step_analysis = []

        try:
            steps = self._get_workflow_steps()
            workflow_name = self.config.workflow.name or "custom"
            print(f"\nWorkflow: {workflow_name}")
            print(f"Steps to execute ({len(steps)} total):")

            # Initialize current_output for tracking between steps
            self.current_output = None
            if self.config.data.input:
                self.current_output = Path(self.config.data.input)

            # Analyze each step

            for i, step in enumerate(steps):
                print(f"\n{i + 1}. {step.name} ", end="")

                # Determine processor
                if step.processor == "auto":
                    try:
                        processor_name = self._auto_select_processor()
                        print(f"(auto-detected: {processor_name})")
                    except ValueError as e:
                        processor_name = None
                        print(f"(⚠️  auto-detection failed: {e})")
                        validation_issues.append(f"Step {i + 1}: {e}")
                else:
                    processor_name = step.processor
                    print(f"({processor_name})")

                if processor_name and processor_name in PROCESSORS:
                    processor_class = PROCESSORS[processor_name]
                    print(f"   Processor: {processor_class.__name__}")

                    # Create a temporary processor instance to get I/O paths
                    temp_processor = None
                    try:
                        # Select processor-specific config based on step name (if needed)
                        processor_config = self._select_processor_config(processor_name, step.name)
                        if processor_name == "bpcells" and processor_config:
                            # Store override in our dict (bypasses OmegaConf validation)
                            self._processor_config_overrides["bpcells"] = processor_config
                            # Also set on config for processor to access (using object.__setattr__ to bypass OmegaConf)
                            object.__setattr__(self.config, "_bpcells_config_override", processor_config)
                        temp_processor = processor_class(self.config)

                        # Set input path from workflow if needed
                        if self.current_output and i > 0:
                            temp_processor.input_path = self.current_output

                        # Determine processor type and show I/O paths
                        if hasattr(
                            temp_processor, "get_input_files"
                        ):  # PreprocessingProcessor
                            input_files = temp_processor.get_input_files()
                            if input_files:
                                print(f"   Input: {len(input_files)} file(s)")
                                for f in input_files[:3]:  # Show first 3 files
                                    print(f"          - {f}")
                                if len(input_files) > 3:
                                    print(
                                        f"          ... and {len(input_files) - 3} more"
                                    )

                            # Get output path for preprocessing processors
                            if processor_name == "fragments":
                                if temp_processor.config.fragments.dual_output:
                                    print(
                                        f"   Output: {self.config.output.dir}/single_sparse.zarr"
                                    )
                                    print(
                                        f"           {self.config.output.dir}/celltype_dense.zarr"
                                    )
                                else:
                                    output_name = temp_processor.config.fragments.output
                                    print(
                                        f"   Output: {self.config.output.dir}/{output_name}"
                                    )
                            else:
                                # Other preprocessing processors
                                processor_config = getattr(
                                    self.config, processor_name.replace("-", "_"), None
                                )
                                if processor_config and hasattr(
                                    processor_config, "output"
                                ):
                                    print(
                                        f"   Output: {self.config.output.dir}/{processor_config.output}"
                                    )

                        elif hasattr(
                            temp_processor, "get_target_path"
                        ):  # AnnotationProcessor
                            try:
                                # Remove override before accessing config (to avoid OmegaConf validation errors)
                                override = None
                                if hasattr(self.config, "_bpcells_config_override"):
                                    override = getattr(self.config, "_bpcells_config_override")
                                    object.__delattr__(self.config, "_bpcells_config_override")
                                target_path = temp_processor.get_input_path()
                                # Restore override if it existed
                                if override is not None:
                                    object.__setattr__(self.config, "_bpcells_config_override", override)
                                if target_path:
                                    print(
                                        f"   Target: {target_path} (in-place modification)"
                                    )
                                elif self.current_output:
                                    print(
                                        f"   Target: {self.current_output} (in-place modification)"
                                    )
                                else:
                                    print(
                                        "   Target: [to be determined from previous step]"
                                    )
                            except Exception as e:
                                # Restore override if it was removed
                                if override is not None and not hasattr(self.config, "_bpcells_config_override"):
                                    object.__setattr__(self.config, "_bpcells_config_override", override)
                                print(
                                    f"   Target: [to be determined from previous step]"
                                )

                        else:  # TransformationProcessor
                            try:
                                input_path = temp_processor.get_input_path()
                                if input_path:
                                    print(f"   Input: {input_path}")
                                elif self.current_output:
                                    print(f"   Input: {self.current_output}")
                                else:
                                    print(
                                        "   Input: [to be determined from previous step]"
                                    )
                            except:
                                print("   Input: [to be determined from previous step]")

                            # Get output path
                            processor_config = getattr(
                                self.config, processor_name.replace("-", "_"), None
                            )
                            if processor_config and hasattr(processor_config, "output"):
                                print(
                                    f"   Output: {self.config.output.dir}/{processor_config.output}"
                                )
                            else:
                                print(
                                    f"   Output: {self.config.output.dir}/[processor-specific]"
                                )

                    except Exception as e:
                        # If we can't instantiate the processor, show what we can
                        print(
                            f"   [Unable to determine I/O paths: {type(e).__name__}: {str(e)}]"
                        )

                    # Show relevant config for this processor
                    if processor_name == "fragments":
                        print(f"   - chunk_size: {self.config.fragments.chunk_size:,}")
                        print(
                            f"   - min_fragments: {self.config.fragments.min_fragments:,}"
                        )
                        print(
                            f"   - max_fragments: {self.config.fragments.max_fragments:,}"
                        )
                        print(f"   - tile_size: {self.config.fragments.tile_size}")
                        print(f"   - min_cells: {self.config.fragments.min_cells}")
                        print(
                            f"   - max_fragments_per_cell: {self.config.fragments.max_fragments_per_cell}"
                        )
                        print(f"   - barcode_tag: {self.config.fragments.barcode_tag}")
                        if self.config.fragments.peak_file:
                            print(f"   - peak_file: {self.config.fragments.peak_file}")
                        print(f"   - overwrite: {self.config.fragments.overwrite}")
                    elif processor_name == "bam":
                        print(f"   - single_end: {self.config.bam.single_end}")
                        print(f"   - tn5_shift: {self.config.bam.tn5_shift}")
                        print(f"   - n_jobs: {self.config.bam.n_jobs}")
                        print(f"   - chunk_size: {self.config.bam.chunk_size:,}")
                    elif processor_name == "clustering":
                        print(f"   - enabled: {self.config.clustering.enabled}")
                        print(f"   - method: {self.config.clustering.method}")
                        print(
                            f"   - n_components: {self.config.clustering.n_components}"
                        )
                        print(f"   - resolution: {self.config.clustering.resolution}")
                        print(f"   - device: {self.config.clustering.device}")
                        print(
                            f"   - blacklist_filtering: {self.config.clustering.blacklist_filtering}"
                        )
                        if self.config.blacklist_file:
                            print(f"   - blacklist_file: {self.config.blacklist_file}")
                        print(f"   - tile_size: {self.config.clustering.tile_size}")
                        print(
                            f"   - total_features: {self.config.clustering.total_features:,}"
                        )
                        print(f"   - overwrite: {self.config.clustering.overwrite}")
                    elif processor_name == "pseudobulk":
                        print(f"   - groupby: {self.config.pseudobulk.groupby}")
                        print(
                            f"   - min_cells_per_group: {self.config.pseudobulk.min_cells_per_group}"
                        )
                        print(
                            f"   - min_libsize: {self.config.pseudobulk.min_libsize:,}"
                        )
                        if self.config.pseudobulk.source:
                            print(f"   - source: {self.config.pseudobulk.source}")
                        if self.config.pseudobulk.grouping_file:
                            print(
                                f"   - grouping_file: {self.config.pseudobulk.grouping_file}"
                            )
                    elif processor_name == "peaks":
                        print(f"   - method: {self.config.peaks.method}")
                        print(f"   - qvalue: {self.config.peaks.qvalue}")
                    elif processor_name == "library_size_annotation":
                        print(f"   - n_jobs: {self.config.library_size.n_jobs}")
                        print(f"   - overwrite: {self.config.library_size.overwrite}")
                        if self.config.blacklist_file:
                            print(f"   - blacklist_file: {self.config.blacklist_file}")
                    elif processor_name == "normalization":
                        print(f"   - method: {self.config.normalization.method}")
                        print(f"   - enabled: {self.config.normalization.enabled}")
                        if self.config.normalization.peak_file:
                            print(
                                f"   - peak_file: {self.config.normalization.peak_file}"
                            )
                        print(f"   - reference: {self.config.normalization.reference}")
                    elif processor_name == "footprints":
                        print(f"   - mode: {self.config.footprints.mode}")
                        print(
                            f"   - cutoff_range: {self.config.footprints.cutoff_range}"
                        )
                        if self.config.footprints.motif_zarr:
                            print(
                                f"   - motif_zarr: {self.config.footprints.motif_zarr}"
                            )
                        print(f"   - extend: {self.config.footprints.extend}")
                        print(f"   - n_processes: {self.config.footprints.n_processes}")
                    elif processor_name == "bpcells":
                        # Get actual output_type from processor (which selects the right config)
                        # This handles bpcells_import, bpcells, and bpcells_pseudobulk configs
                        # Get config from processor (which uses workflow-selected config)
                        if temp_processor and hasattr(temp_processor, 'bpcells_config'):
                            actual_output_type = temp_processor.output_type
                            actual_config = temp_processor.bpcells_config
                        else:
                            # Fallback: try to get from step-specific config
                            bpcells_config = self._select_bpcells_config(step.name)
                            if bpcells_config:
                                actual_output_type = bpcells_config.output_type
                                actual_config = bpcells_config
                            else:
                                actual_output_type = self.config.bpcells_anndata.output_type if self.config.bpcells_anndata else self.config.bpcells.output_type
                                actual_config = self.config.bpcells_anndata if self.config.bpcells_anndata else self.config.bpcells
                        print(f"   - output_type: {actual_output_type}")
                        print(f"   - input_format: {actual_config.input_format}")
                        print(f"   - shift_end: {actual_config.shift_end}")
                        print(
                            f"   - threads: {actual_config.threads or self.config.resources.n_jobs}"
                        )
                        print(
                            f"   - min_fragments: {actual_config.min_fragments:,}"
                        )
                        print(f"   - group_by: {actual_config.group_by}")
                        print(f"   - barcode_col: {actual_config.barcode_col}")
                        print(f"   - annotation_col: {actual_config.annotation_col}")
                        print(f"   - sample_id_col: {actual_config.sample_id_col}")
                        if self.config.bpcells.cluster_mapping_file:
                            print(
                                f"   - cluster_mapping_file: {self.config.bpcells.cluster_mapping_file}"
                            )
                        if self.config.bpcells.target_groups:
                            print(
                                f"   - target_groups: {self.config.bpcells.target_groups}"
                            )
                        print(
                            f"   - save_metadata: {self.config.bpcells.save_metadata}"
                        )
                        print(f"   - overwrite: {self.config.bpcells.overwrite}")
                        if self.config.bpcells.normalization_factor_file:
                            print(
                                f"   - normalization_factor_file: {self.config.bpcells.normalization_factor_file}"
                            )
                        # Show metadata requirements
                        if actual_output_type == "celltype_dense":
                            has_metadata = any(
                                getattr(frag, "metadata", None) for frag in self.config.data.fragments
                            )
                            if has_metadata:
                                print(f"   - metadata: Provided ({sum(1 for f in self.config.data.fragments if getattr(f, 'metadata', None))} file(s))")
                            else:
                                print(f"   - metadata: ⚠️  REQUIRED for celltype_dense output")
                        elif actual_output_type in ["single_sparse", "signal_dense"]:
                            has_metadata = any(
                                getattr(frag, "metadata", None) for frag in self.config.data.fragments
                            )
                            print(f"   - metadata: {'Provided' if has_metadata else 'Optional'}")
                    if step.description:
                        print(f"   Description: {step.description}")

                    # Update current_output for next step based on processor type
                    try:
                        if temp_processor and hasattr(
                            temp_processor, "get_input_files"
                        ):  # PreprocessingProcessor
                            if processor_name == "fragments":
                                if temp_processor.config.fragments.dual_output:
                                    # For dual output, use configured output name (typically single_sparse.zarr)
                                    self.current_output = (
                                        Path(self.config.output.dir)
                                        / temp_processor.config.fragments.output
                                    )
                                else:
                                    self.current_output = (
                                        Path(self.config.output.dir)
                                        / temp_processor.config.fragments.output
                                    )
                            else:
                                processor_config = getattr(
                                    self.config, processor_name.replace("-", "_"), None
                                )
                                if processor_config and hasattr(
                                    processor_config, "output"
                                ):
                                    self.current_output = (
                                        Path(self.config.output.dir)
                                        / processor_config.output
                                    )
                        elif hasattr(
                            temp_processor, "get_target_path"
                        ):  # AnnotationProcessor
                            # Output remains the same as input for in-place modifications
                            pass
                        else:  # TransformationProcessor
                            processor_config = getattr(
                                self.config, processor_name.replace("-", "_"), None
                            )
                            if processor_config and hasattr(processor_config, "output"):
                                self.current_output = (
                                    Path(self.config.output.dir)
                                    / processor_config.output
                                )
                    except:
                        pass

                    step_analysis.append(
                        {
                            "step": step.name,
                            "processor": processor_name,
                            "processor_class": processor_class.__name__,
                            "valid": True,
                        }
                    )
                elif processor_name:
                    print(f"   ⚠️  Unknown processor: {processor_name}")
                    validation_issues.append(
                        f"Step {i + 1}: Unknown processor '{processor_name}'"
                    )
                    step_analysis.append(
                        {
                            "step": step.name,
                            "processor": processor_name,
                            "valid": False,
                            "error": "Unknown processor",
                        }
                    )

        except Exception as e:
            print(f"\n⚠️  Error analyzing workflow: {e}")
            validation_issues.append(f"Workflow analysis error: {e}")
            steps = []
            step_analysis = []

        # Configuration validation
        print("\n" + "-" * 40)
        print("Validation Results:")

        # Check for configuration issues
        config_issues = self._check_config_issues()

        if not validation_issues and not config_issues:
            print("✅ All configuration checks passed")
        else:
            for issue in validation_issues:
                print(f"❌ {issue}")
            for issue in config_issues:
                print(f"⚠️  {issue}")

        # File accessibility checks
        print("\nFile Accessibility:")
        file_checks = self._check_file_accessibility()
        for check, result in file_checks.items():
            if result["status"] == "ok":
                print(f"✅ {check}: {result['message']}")
            else:
                print(f"❌ {check}: {result['message']}")

        print("\n" + "=" * 60)

        if validation_issues or any(r["status"] != "ok" for r in file_checks.values()):
            print("❌ Workflow has issues that need to be resolved")
        else:
            print("✅ Ready to execute! Use engine.execute() to run.")

        print("=" * 60 + "\n")

        return {
            "workflow_name": self.config.workflow.name or "custom",
            "steps": step_analysis,
            "validation_issues": validation_issues,
            "config_issues": config_issues,
            "file_checks": file_checks,
            "ready": not validation_issues and not config_issues,
        }

    def semi_wet_run(self) -> dict[str, Any]:
        """Perform a semi-wet run of the workflow with overwrite disabled.

        This mode executes the workflow but skips any steps where output already exists.
        This is useful for resuming interrupted workflows or re-running with updates.

        Returns:
            Workflow execution results
        """
        print("\n" + "=" * 60)
        print("=== CAESAR WORKFLOW SEMI-WET RUN ===")
        print("=" * 60 + "\n")
        print("Mode: Skip existing outputs (overwrite=False)")
        print("This will execute the workflow but skip steps with existing output.\n")

        # Enable skip mode in the config
        self._enable_skip_mode()

        # Execute the workflow with skip mode enabled
        return self.execute()

    def _enable_skip_mode(self) -> None:
        """Disable overwrite mode for all processors that support it."""
        # For processors that have overwrite parameters, set them to False
        # This is done by modifying the config before processor initialization

        # Fragment processor - set overwrite=False to skip existing outputs
        if hasattr(self.config, "fragments"):
            self.config.fragments.overwrite = False

        # Clustering processor has overwrite parameter
        if hasattr(self.config, "clustering"):
            self.config.clustering.overwrite = False

        # Library size processor has overwrite parameter
        if hasattr(self.config, "library_size"):
            self.config.library_size.overwrite = False

        # Normalization processor has overwrite parameter
        if hasattr(self.config, "normalization"):
            self.config.normalization.overwrite = False

        # Footprint processor has overwrite parameter
        if hasattr(self.config, "footprints"):
            self.config.footprints.overwrite = False

        # Peak calling processor has overwrite parameter
        if hasattr(self.config, "peaks"):
            self.config.peaks.overwrite = False

        # Pseudobulk processor has overwrite parameter
        if hasattr(self.config, "pseudobulk"):
            self.config.pseudobulk.overwrite = False

        # BPCells processor has overwrite parameter
        if hasattr(self.config, "bpcells"):
            self.config.bpcells.overwrite = False

        self.logger.info("Skip mode enabled for processors that support it")

    def _check_config_issues(self) -> list[str]:
        """Check for configuration redundancies and issues.

        Returns:
            List of configuration issues found
        """
        issues = []

        # Note: n_jobs can be defined in multiple places intentionally
        # Different processors may need different levels of parallelism
        # resources.n_jobs serves as a default, but processors can override

        return issues

    def _check_file_accessibility(self) -> dict[str, dict[str, str]]:
        """Check if input/output files and directories are accessible.

        Returns:
            Dictionary of file checks and their results
        """
        checks = {}

        # Check input files
        if self.config.data.fragments:
            for i, frag in enumerate(self.config.data.fragments):
                path = Path(frag.path)
                if path.exists():
                    checks[f"Fragment file {i + 1}"] = {
                        "status": "ok",
                        "message": f"{path.name} exists",
                    }
                else:
                    checks[f"Fragment file {i + 1}"] = {
                        "status": "error",
                        "message": f"{path} not found",
                    }

        if self.config.data.bam:
            for i, bam in enumerate(self.config.data.bam):
                path = Path(bam.path)
                if path.exists():
                    checks[f"BAM file {i + 1}"] = {
                        "status": "ok",
                        "message": f"{path.name} exists",
                    }
                else:
                    checks[f"BAM file {i + 1}"] = {
                        "status": "error",
                        "message": f"{path} not found",
                    }

        if self.config.data.input:
            path = Path(self.config.data.input)
            if path.exists():
                checks["Input zarr"] = {
                    "status": "ok",
                    "message": f"{path.name} exists",
                }
            else:
                checks["Input zarr"] = {
                    "status": "error",
                    "message": f"{path} not found",
                }

        # Check output directory
        output_dir = Path(self.config.output.dir)
        if output_dir.exists():
            checks["Output directory"] = {
                "status": "ok",
                "message": f"{output_dir} exists and is writable",
            }
        else:
            # Check if parent exists and is writable
            parent = output_dir.parent
            if parent.exists() and os.access(parent, os.W_OK):
                checks["Output directory"] = {
                    "status": "ok",
                    "message": f"Will be created in {parent}",
                }
            else:
                checks["Output directory"] = {
                    "status": "error",
                    "message": f"Cannot create {output_dir} - parent not writable",
                }

        return checks


def load_and_execute_workflow(config_path: str) -> dict[str, Any]:
    """Load configuration and execute workflow.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Workflow execution results
    """
    # Load configuration using OmegaConf
    yaml_config = OmegaConf.load(config_path)

    # Merge with structured config for validation
    base_config = OmegaConf.structured(CaesarConfig)
    merged_config = OmegaConf.merge(base_config, yaml_config)

    # Convert to CaesarConfig object
    config = OmegaConf.to_object(merged_config)

    # Create and execute workflow
    engine = WorkflowEngine(config)
    return engine.execute()


def load_and_dry_run_workflow(
    config_path: str, overrides: list[str] | None = None
) -> dict[str, Any]:
    """Load configuration and perform a dry run of the workflow.

    Args:
        config_path: Path to YAML configuration file
        overrides: Optional list of Hydra-style overrides

    Returns:
        Dry run analysis results
    """
    from .config import load_caesar_config

    print(f"Loading configuration from: {config_path}")

    # Load config with overrides
    config = load_caesar_config(config_path, overrides=overrides or [])

    # Create engine and run dry run
    engine = WorkflowEngine(config)
    return engine.dry_run()


def load_and_semi_wet_run_workflow(
    config_path: str, overrides: list[str] | None = None
) -> dict[str, Any]:
    """Load configuration and perform a semi-wet run of the workflow.

    Args:
        config_path: Path to YAML configuration file
        overrides: Optional list of Hydra-style overrides

    Returns:
        Workflow execution results
    """
    from .config import load_caesar_config

    print(f"Loading configuration from: {config_path}")

    # Load config with overrides
    config = load_caesar_config(config_path, overrides=overrides or [])

    # Create engine and run semi-wet run
    engine = WorkflowEngine(config)
    return engine.semi_wet_run()

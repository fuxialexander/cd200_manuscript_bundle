from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GenomicMaskingMixin:
    """
    Mixin class for applying genomic masking to track data.

    This mixin provides functionality to mask out blacklist regions and
    centromere/heterochromatin/telomere regions with a configurable buffer
    (default ±2Mbp) by setting values to zero.

    The mixin can be used with SignalDenseIO and CelltypeDenseIO classes
    to automatically apply masking during data retrieval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._genome = None
        self._masking_regions_cache = {}

    def _get_genome(self, assembly: str = None):
        """Get Genome object for specified assembly."""
        if assembly is None:
            assembly = getattr(self, "assembly", "hg38")

        # Initialize genome cache if it doesn't exist (handles mixin initialization issues)
        if not hasattr(self, "_genome"):
            self._genome = None

        # Use cached genome only if assembly matches
        if (
            self._genome is not None
            and getattr(self._genome, "assembly", None) == assembly
        ):
            return self._genome

        try:
            from gcell.dna.genome import Genome

            self._genome = Genome(assembly, load_genome_seq=False)
            return self._genome
        except ImportError:
            logger.error(
                "gcell.dna.genome.Genome not available. Cannot apply genomic masking."
            )
            return None

    def _get_masking_regions(
        self, chromosome: str, buffer_size: int, assembly: str = None
    ) -> list[tuple[int, int]]:
        """
        Get regions to mask (blacklist + centromere/heterochromatin/telomeres ±buffer) for a chromosome.

        Parameters
        ----------
        chromosome : str
            Chromosome name (e.g., 'chr1', 'chr2', etc.)
        buffer_size : int
            Buffer size in base pairs to add around centromere/heterochromatin/telomere regions.
        assembly : str, optional
            Genome assembly to use. If None, uses self.assembly from the IO object.

        Returns
        -------
        list[tuple[int, int]]
            List of (start, end) tuples representing regions to mask.
        """
        if assembly is None:
            assembly = getattr(self, "assembly", "hg38")

        # Initialize cache if it doesn't exist (handles mixin initialization issues)
        if not hasattr(self, "_masking_regions_cache"):
            self._masking_regions_cache = {}

        # Create cache key that includes buffer size and assembly
        cache_key = f"{chromosome}_{buffer_size}_{assembly}"
        if cache_key in self._masking_regions_cache:
            return self._masking_regions_cache[cache_key]

        genome = self._get_genome(assembly)
        if genome is None:
            return []

        masking_regions = []

        # Get blacklist regions
        try:
            blacklist = genome.blacklist
            if not blacklist.empty:
                chrom_blacklist = blacklist[blacklist["Chromosome"] == chromosome]
                for _, region in chrom_blacklist.iterrows():
                    masking_regions.append((region["Start"], region["End"]))
        except Exception as e:
            logger.warning(f"Could not load blacklist regions for {chromosome}: {e}")

        # Get centromere and heterochromatin regions with buffer
        try:
            chrom_gap = genome.chrom_gap
            heterochromatin_df = chrom_gap.get_heterochromatin()
            telomeres_df = chrom_gap.get_telomeres()

            # Process heterochromatin and centromere regions
            if not heterochromatin_df.empty:
                chrom_heterochromatin = heterochromatin_df[
                    heterochromatin_df["Chromosome"] == chromosome
                ]
                for _, region in chrom_heterochromatin.iterrows():
                    # Add ±buffer
                    start = max(0, region["Start"] - buffer_size)
                    end = region["End"] + buffer_size
                    masking_regions.append((start, end))

            # Process telomere regions
            if not telomeres_df.empty:
                chrom_telomeres = telomeres_df[telomeres_df["Chromosome"] == chromosome]
                for _, region in chrom_telomeres.iterrows():
                    # Add ±buffer
                    start = max(0, region["Start"] - buffer_size)
                    end = region["End"] + buffer_size
                    masking_regions.append((start, end))

        except Exception as e:
            logger.warning(f"Could not load gap regions for {chromosome}: {e}")

        # Merge overlapping regions
        if masking_regions:
            masking_regions.sort()
            merged_regions = []
            current_start, current_end = masking_regions[0]

            for start, end in masking_regions[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged_regions.append((current_start, current_end))
                    current_start, current_end = start, end

            merged_regions.append((current_start, current_end))
            self._masking_regions_cache[cache_key] = merged_regions
        else:
            self._masking_regions_cache[cache_key] = []

        if self._masking_regions_cache[cache_key]:
            logger.debug(
                f"Loaded {len(self._masking_regions_cache[cache_key])} masking regions for {chromosome}"
            )

        return self._masking_regions_cache[cache_key]

    def _process_chromosome_masking(
        self,
        chromosome: str,
        buffer_size: int,
        assembly: str,
        target_ids: list,
        chrom_size: int,
    ) -> tuple[str, bool, str]:
        """
        Process masking for a single chromosome.

        Returns
        -------
        tuple[str, bool, str]
            (chromosome, success, error_message)
        """
        try:
            masking_regions = self._get_masking_regions(
                chromosome, buffer_size, assembly
            )

            if not masking_regions:
                return chromosome, True, f"No masking regions found for {chromosome}"

            logger.info(f"Masking {len(masking_regions)} regions in {chromosome}")

            for target_id in target_ids:
                try:
                    # Get zarr array reference
                    if target_id is None:
                        # SignalDenseIO case
                        zarr_array = self.dataset["chrs"][chromosome]
                    else:
                        # CelltypeDenseIO case
                        zarr_array = self.dataset[target_id]["chrs"][chromosome]

                    # Apply masking to each region
                    for mask_start, mask_end in masking_regions:
                        # Ensure coordinates are within chromosome bounds
                        mask_start = max(0, min(mask_start, chrom_size - 1))
                        mask_end = max(0, min(mask_end, chrom_size))

                        if mask_start < mask_end:
                            # Directly overwrite zarr data with zeros
                            if zarr_array.ndim == 1:
                                zarr_array[mask_start:mask_end] = 0.0
                            else:
                                # For multi-dimensional data, mask along the first axis
                                zarr_array[mask_start:mask_end] = 0.0

                    if target_id is not None:
                        logger.debug(f"Applied masking to {target_id}:{chromosome}")

                except Exception as e:
                    error_msg = (
                        f"Error applying masking to {target_id}:{chromosome}: {e}"
                        if target_id
                        else f"Error applying masking to {chromosome}: {e}"
                    )
                    logger.error(error_msg)
                    return chromosome, False, error_msg

            return chromosome, True, ""

        except Exception as e:
            error_msg = f"Error processing {chromosome}: {e}"
            logger.error(error_msg)
            return chromosome, False, error_msg

    def apply_genomic_masking_to_zarr(
        self,
        buffer_size: int = 2_000_000,
        assembly: str = None,
        celltype_ids=None,
        overwrite_existing=False,
        max_workers: int = None,
    ):
        """
        Apply genomic masking directly to zarr data by overwriting masked regions with zeros.

        This method permanently modifies the zarr data by setting blacklist regions and
        centromere/heterochromatin/telomere regions (with buffer) to zero.
        Use with caution as this operation cannot be undone without reloading from source data.

        Parameters
        ----------
        buffer_size : int, optional
            Buffer size in base pairs to add around centromere/heterochromatin/telomere regions.
            Default is 2,000,000 (2Mbp).
        assembly : str, optional
            Genome assembly to use. If None, uses self.assembly from the IO object.
        celltype_ids : list, optional
            List of celltype IDs to apply masking to. If None, applies to all available IDs.
            For SignalDenseIO, this parameter is ignored.
        overwrite_existing : bool, optional
            Whether to proceed even if masking attributes suggest masking was already applied.
            Default is False.
        max_workers : int, optional
            Maximum number of parallel workers for chromosome processing.
            If None, uses the number of CPU cores available.
        """
        if self.mode == "r":
            logger.error(
                "Cannot apply masking: Zarr dataset is read-only. Open in 'a' mode."
            )
            return

        if assembly is None:
            assembly = getattr(self, "assembly", "hg38")

        # Check if masking was already applied
        if hasattr(self.dataset, "attrs"):
            if (
                self.dataset.attrs.get("genomic_masking_applied", False)
                and not overwrite_existing
            ):
                logger.warning(
                    "Genomic masking appears to have been already applied. "
                    "Use overwrite_existing=True to proceed anyway."
                )
                return

        logger.info(
            f"Applying genomic masking directly to zarr data (±{buffer_size / 1_000_000:.1f}Mbp buffer, {assembly})"
        )

        # Determine which IDs to process
        if hasattr(self, "ids"):
            # CelltypeDenseIO case
            target_ids = celltype_ids if celltype_ids is not None else self.ids
            target_ids = [id for id in target_ids if id in self.ids]
        else:
            # SignalDenseIO case
            target_ids = [None]  # Will be handled differently

        if hasattr(self, "ids") and not target_ids:
            logger.error("No valid celltype IDs found to apply masking to.")
            return

        # Prepare chromosome processing tasks
        chromosome_tasks = []
        for chromosome in self.chroms:
            chrom_size = self.chrom_sizes.get(chromosome, 0)
            if chrom_size == 0:
                logger.warning(f"Could not determine size for {chromosome}, skipping")
                continue
            chromosome_tasks.append(
                (chromosome, buffer_size, assembly, target_ids, chrom_size)
            )

        if not chromosome_tasks:
            logger.error("No valid chromosomes found to process")
            return

        # Process chromosomes in parallel
        logger.info(
            f"Processing {len(chromosome_tasks)} chromosomes with max_workers={max_workers}"
        )

        # Use ThreadPoolExecutor instead of ProcessPoolExecutor since we're accessing shared zarr data
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_chromosome = {
                executor.submit(self._process_chromosome_masking, *task): task[0]
                for task in chromosome_tasks
            }

            # Process results
            failed_chromosomes = []
            for future in future_to_chromosome:
                chromosome = future_to_chromosome[future]
                try:
                    chrom, success, message = future.result()
                    if not success:
                        failed_chromosomes.append(chrom)
                        logger.error(f"Failed to process {chrom}: {message}")
                    elif message:
                        logger.debug(message)
                except Exception as e:
                    failed_chromosomes.append(chromosome)
                    logger.error(f"Exception processing {chromosome}: {e}")

        if failed_chromosomes:
            logger.warning(
                f"Failed to process {len(failed_chromosomes)} chromosomes: {failed_chromosomes}"
            )

        # Mark that masking has been applied
        self.dataset.attrs["genomic_masking_applied"] = True
        self.dataset.attrs["genomic_masking_buffer"] = buffer_size

        logger.info("Genomic masking applied successfully to zarr data")

    def clear_masking_cache(self):
        """Clear the cache of masking regions."""
        if hasattr(self, "_masking_regions_cache"):
            self._masking_regions_cache = {}
            logger.info("Cleared genomic masking regions cache")
        else:
            logger.info("No masking cache to clear")

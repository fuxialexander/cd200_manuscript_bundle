"""
Modular motif filtering system.

Provides pluggable filters for selecting motifs based on various criteria
(expression, differential expression, custom lists, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Set, Optional, Dict, Callable
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_hocomoco_motif_mapping(
    hocomoco_aligned_path: str = 'artifacts/motifs_with_rc_aligned.pt',
    hocomoco_annot_path: str = 'artifacts/data/hocomoco/H13CORE_annotation.jsonl',
    to_mouse: bool = False
) -> Dict[str, str]:
    """
    Load motif-to-TF mapping from Hocomoco files.
    
    Args:
        hocomoco_aligned_path: Path to aligned motifs file
        hocomoco_annot_path: Path to HOCOMOCO annotation JSONL file
        to_mouse: If True, convert human gene symbols to mouse format (capitalize first letter only)
    
    Returns:
        Dictionary mapping motif names to TF gene symbols
    """
    try:
        from gcell.dna.hocomoco import HocomocoIO
    except ImportError:
        logger.error("Caesar library not found. Please install it to use Hocomoco annotations.")
        raise

    if to_mouse:
        # Directly read from JSONL to convert human to mouse format
        import json
        mapping = {}
        with open(hocomoco_annot_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                motif_name = data.get('name')
                if not motif_name:
                    continue
                
                # Extract human gene symbol and convert to mouse format
                if 'masterlist_info' in data and 'species' in data['masterlist_info']:
                    human_info = data['masterlist_info']['species'].get('HUMAN', {})
                    human_gene = human_info.get('gene_symbol')
                    if human_gene:
                        # Convert from all caps to first letter capitalized (human to mouse conversion)
                        mouse_gene = human_gene.capitalize()
                        mapping[motif_name] = mouse_gene
        return mapping
    else:
        # Use HocomocoIO for standard mapping
        hoco_io = HocomocoIO(hocomoco_aligned_path, annotation_path=hocomoco_annot_path)
        
        mapping = {}
        for motif_name in hoco_io.motif_names:
            try:
                # The get_tf_name_from_motif method handles RC motifs correctly
                tf_name = hoco_io.get_tf_name_from_motif(motif_name)
                mapping[motif_name] = tf_name
            except ValueError:
                # This can happen if a motif in the .pt file is not in the annotation
                logger.warning(f"Could not find TF name for motif: {motif_name}")
                continue
                
        return mapping


class MotifFilter(ABC):
    """
    Abstract base class for motif filters.

    Filters determine which motifs should be included in visualization
    or analysis based on biological criteria (expression, DE, etc.).
    """

    @abstractmethod
    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        """
        Filter motif names based on criteria.

        Args:
            motif_names: List of candidate motif names

        Returns:
            Filtered list of motif names
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return human-readable description of filter."""
        pass


class PassThroughFilter(MotifFilter):
    """No filtering - pass through all motifs."""

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        return motif_names

    def get_description(self) -> str:
        return "No filtering (all motifs)"


class WhitelistFilter(MotifFilter):
    """
    Keep only motifs for specified transcription factors.

    Uses motif-to-TF mapping to convert TF names to motif names.
    """

    def __init__(self, tf_list: List[str], motif_to_tf_mapping: Dict[str, str],
                 case_sensitive: bool = False):
        """
        Args:
            tf_list: List of TF gene names to keep
            motif_to_tf_mapping: Dict mapping motif_name -> gene_name
            case_sensitive: Whether TF name matching is case-sensitive
        """
        if not case_sensitive:
            self.tf_set = {tf.upper() for tf in tf_list}
        else:
            self.tf_set = set(tf_list)

        self.motif_to_tf = motif_to_tf_mapping
        self.case_sensitive = case_sensitive

        logger.info(f"WhitelistFilter: {len(self.tf_set)} TFs allowed")

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        filtered = []
        for motif in motif_names:
            tf = self.motif_to_tf.get(motif)
            if tf:
                tf_check = tf if self.case_sensitive else tf.upper()
                if tf_check in self.tf_set:
                    filtered.append(motif)

        logger.debug(f"WhitelistFilter: {len(filtered)}/{len(motif_names)} motifs passed")
        return filtered

    def get_description(self) -> str:
        return f"Whitelist: {len(self.tf_set)} TFs"


class BlacklistFilter(MotifFilter):
    """
    Exclude motifs for specified transcription factors.

    Uses motif-to-TF mapping to convert TF names to motif names.
    """

    def __init__(self, tf_list: List[str], motif_to_tf_mapping: Dict[str, str],
                 case_sensitive: bool = False):
        """
        Args:
            tf_list: List of TF gene names to exclude
            motif_to_tf_mapping: Dict mapping motif_name -> gene_name
            case_sensitive: Whether TF name matching is case-sensitive
        """
        if not case_sensitive:
            self.tf_set = {tf.upper() for tf in tf_list}
        else:
            self.tf_set = set(tf_list)

        self.motif_to_tf = motif_to_tf_mapping
        self.case_sensitive = case_sensitive

        logger.info(f"BlacklistFilter: {len(self.tf_set)} TFs excluded")

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        filtered = []
        for motif in motif_names:
            tf = self.motif_to_tf.get(motif)
            if tf:
                tf_check = tf if self.case_sensitive else tf.upper()
                if tf_check not in self.tf_set:
                    filtered.append(motif)
            else:
                # Keep motifs without TF mapping
                filtered.append(motif)

        logger.debug(f"BlacklistFilter: {len(filtered)}/{len(motif_names)} motifs passed")
        return filtered

    def get_description(self) -> str:
        return f"Blacklist: {len(self.tf_set)} TFs excluded"


class ExpressionFilter(MotifFilter):
    """
    Filter motifs based on TF expression levels.

    Requires annotation data with gene expression information.
    """

    def __init__(self, annotation_df: pd.DataFrame,
                 min_expression: float = 10.0,
                 expression_column: str = 'baseMean',
                 motif_column: str = 'Motif',
                 gene_column: str = 'gene_name'):
        """
        Args:
            annotation_df: DataFrame with motif annotations and expression data
            min_expression: Minimum expression level threshold
            expression_column: Column name for expression values
            motif_column: Column name for motif names
            gene_column: Column name for gene names
        """
        self.annotation = annotation_df
        self.min_expr = min_expression
        self.expr_col = expression_column
        self.motif_col = motif_column
        self.gene_col = gene_column

        # Create motif -> expression mapping
        self.motif_to_expr = dict(zip(
            annotation_df[motif_column],
            annotation_df[expression_column]
        ))

        logger.info(f"ExpressionFilter: min_expression={min_expression}")

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        filtered = []
        for motif in motif_names:
            expr = self.motif_to_expr.get(motif)
            if expr is not None and expr >= self.min_expr:
                filtered.append(motif)

        logger.debug(f"ExpressionFilter: {len(filtered)}/{len(motif_names)} motifs passed")
        return filtered

    def get_description(self) -> str:
        return f"Expression: {self.expr_col} >= {self.min_expr}"


class MultiConditionExpressionFilter(MotifFilter):
    """
    Filter based on expression in multiple conditions.

    Motif passes if TF is expressed above threshold in ANY condition.
    """

    def __init__(self, annotation_df: pd.DataFrame,
                 expression_columns: List[str],
                 min_expression: float = 10.0,
                 motif_column: str = 'Motif',
                 require_all: bool = False):
        """
        Args:
            annotation_df: DataFrame with motif annotations
            expression_columns: List of column names for different conditions
            min_expression: Minimum expression threshold
            motif_column: Column name for motif names
            require_all: If True, require expression in ALL conditions (AND logic).
                        If False, require expression in ANY condition (OR logic).
        """
        self.annotation = annotation_df
        self.expr_cols = expression_columns
        self.min_expr = min_expression
        self.motif_col = motif_column
        self.require_all = require_all

        # Create motif -> expression mapping for each condition
        self.motif_to_expr = {
            col: dict(zip(annotation_df[motif_column], annotation_df[col]))
            for col in expression_columns
        }

        logger.info(f"MultiConditionExpressionFilter: {len(expression_columns)} conditions, "
                   f"require_all={require_all}")

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        filtered = []
        for motif in motif_names:
            # Check expression in each condition
            expressed = []
            for col in self.expr_cols:
                expr = self.motif_to_expr[col].get(motif)
                if expr is not None:
                    expressed.append(expr >= self.min_expr)
                else:
                    expressed.append(False)

            # Apply logic
            if self.require_all:
                if all(expressed):
                    filtered.append(motif)
            else:
                if any(expressed):
                    filtered.append(motif)

        logger.debug(f"MultiConditionExpressionFilter: {len(filtered)}/{len(motif_names)} motifs passed")
        return filtered

    def get_description(self) -> str:
        logic = "ALL" if self.require_all else "ANY"
        return f"Expression ({logic}): {', '.join(self.expr_cols)} >= {self.min_expr}"


class DifferentialExpressionFilter(MotifFilter):
    """
    Filter motifs based on differential expression of their TFs.

    Requires annotation data with DE statistics (padj, log2FC).
    """

    def __init__(self, annotation_df: pd.DataFrame,
                 padj_threshold: float = 0.05,
                 min_log2fc: float = 0.0,
                 padj_column: str = 'padj',
                 log2fc_column: Optional[str] = None,
                 motif_column: str = 'Motif'):
        """
        Args:
            annotation_df: DataFrame with DE results
            padj_threshold: Maximum adjusted p-value
            min_log2fc: Minimum absolute log2 fold-change (if log2fc_column provided)
            padj_column: Column name for adjusted p-value
            log2fc_column: Optional column name for log2 fold-change
            motif_column: Column name for motif names
        """
        self.annotation = annotation_df
        self.padj_thresh = padj_threshold
        self.min_log2fc = min_log2fc
        self.padj_col = padj_column
        self.log2fc_col = log2fc_column
        self.motif_col = motif_column

        # Create mappings
        self.motif_to_padj = dict(zip(
            annotation_df[motif_column],
            annotation_df[padj_column]
        ))

        if log2fc_column:
            self.motif_to_log2fc = dict(zip(
                annotation_df[motif_column],
                annotation_df[log2fc_column]
            ))
        else:
            self.motif_to_log2fc = None

        logger.info(f"DifferentialExpressionFilter: padj < {padj_threshold}, "
                   f"min_log2fc={min_log2fc if log2fc_column else 'N/A'}")

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        filtered = []
        for motif in motif_names:
            padj = self.motif_to_padj.get(motif)
            if padj is None or pd.isna(padj):
                continue

            # Check p-value threshold
            if padj >= self.padj_thresh:
                continue

            # Check log2FC threshold if applicable
            if self.motif_to_log2fc:
                log2fc = self.motif_to_log2fc.get(motif)
                if log2fc is None or pd.isna(log2fc):
                    continue
                if abs(log2fc) < self.min_log2fc:
                    continue

            filtered.append(motif)

        logger.debug(f"DifferentialExpressionFilter: {len(filtered)}/{len(motif_names)} motifs passed")
        return filtered

    def get_description(self) -> str:
        desc = f"DE: {self.padj_col} < {self.padj_thresh}"
        if self.log2fc_col:
            desc += f", |{self.log2fc_col}| >= {self.min_log2fc}"
        return desc


class MultiConditionDEFilter(MotifFilter):
    """
    Filter based on differential expression in multiple comparisons.

    Motif passes if TF is DE in ANY comparison.
    """

    def __init__(self, annotation_df: pd.DataFrame,
                 padj_columns: List[str],
                 padj_threshold: float = 0.05,
                 motif_column: str = 'Motif',
                 require_all: bool = False):
        """
        Args:
            annotation_df: DataFrame with DE results
            padj_columns: List of padj column names for different comparisons
            padj_threshold: Maximum adjusted p-value
            motif_column: Column name for motif names
            require_all: If True, require DE in ALL comparisons. If False, ANY.
        """
        self.annotation = annotation_df
        self.padj_cols = padj_columns
        self.padj_thresh = padj_threshold
        self.motif_col = motif_column
        self.require_all = require_all

        # Create mappings for each comparison
        self.motif_to_padj = {
            col: dict(zip(annotation_df[motif_column], annotation_df[col]))
            for col in padj_columns
        }

        logger.info(f"MultiConditionDEFilter: {len(padj_columns)} comparisons, "
                   f"require_all={require_all}")

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        filtered = []
        for motif in motif_names:
            # Check DE in each comparison
            is_de = []
            for col in self.padj_cols:
                padj = self.motif_to_padj[col].get(motif)
                if padj is not None and not pd.isna(padj):
                    is_de.append(padj < self.padj_thresh)
                else:
                    is_de.append(False)

            # Apply logic
            if self.require_all:
                if all(is_de):
                    filtered.append(motif)
            else:
                if any(is_de):
                    filtered.append(motif)

        logger.debug(f"MultiConditionDEFilter: {len(filtered)}/{len(motif_names)} motifs passed")
        return filtered

    def get_description(self) -> str:
        logic = "ALL" if self.require_all else "ANY"
        return f"DE ({logic}): padj < {self.padj_thresh}"


class CustomFilter(MotifFilter):
    """
    Custom filter using user-provided function.

    Allows arbitrary filtering logic.
    """

    def __init__(self, filter_func: Callable[[str], bool],
                 description: str = "Custom filter"):
        """
        Args:
            filter_func: Function that takes motif_name and returns True to keep
            description: Human-readable description
        """
        self.filter_func = filter_func
        self.description = description

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        filtered = [m for m in motif_names if self.filter_func(m)]
        logger.debug(f"CustomFilter: {len(filtered)}/{len(motif_names)} motifs passed")
        return filtered

    def get_description(self) -> str:
        return self.description


class CompositeFilter(MotifFilter):
    """
    Combine multiple filters with AND or OR logic.
    """

    def __init__(self, filters: List[MotifFilter], mode: str = 'AND'):
        """
        Args:
            filters: List of MotifFilter instances
            mode: 'AND' (all filters must pass) or 'OR' (any filter must pass)
        """
        if mode not in ['AND', 'OR']:
            raise ValueError("mode must be 'AND' or 'OR'")

        self.filters = filters
        self.mode = mode

        logger.info(f"CompositeFilter: {len(filters)} filters with {mode} logic")

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        if not self.filters:
            return motif_names

        if self.mode == 'AND':
            # All filters must pass - iteratively filter
            result = motif_names
            for filt in self.filters:
                result = filt.filter_motifs(result)
            return result
        else:
            # Any filter passes - union of results
            result_sets = [set(filt.filter_motifs(motif_names)) for filt in self.filters]
            union = set.union(*result_sets) if result_sets else set()
            # Preserve original order
            return [m for m in motif_names if m in union]

    def get_description(self) -> str:
        filter_descs = [f.get_description() for f in self.filters]
        return f"({f' {self.mode} '.join(filter_descs)})"


class PromoterAccessibilityFilter(MotifFilter):
    """
    Filter motifs based on transcription factor accessibility at promoters.

    This filter identifies TFs that are active in a given cell type by measuring
    ATAC-seq signal accessibility around transcription start sites (TSS). Only
    motifs corresponding to TFs with accessibility above a certain quantile
    are kept.

    NOTE: This is an advanced filter with external dependencies on project-specific
    data loaders (`caesar` and `gcell`).
    """
    def __init__(self, cell_type: str,
                 quantile_threshold: float = 0.7,
                 hocomoco_aligned_path: str = 'artifacts/motifs_with_rc_aligned.pt',
                 hocomoco_annot_path: str = 'artifacts/data/hocomoco/H13CORE_annotation.jsonl',
                 atac_zarr_path: str = 'data/zarr/cd200_scc_dense_new.zarr',
                 motif_to_tf_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            cell_type: The name of the cell type to use for accessibility scoring.
            quantile_threshold: Keep motifs with accessibility > this quantile.
            hocomoco_aligned_path: Path to Hocomoco aligned motifs (.pt).
            hocomoco_annot_path: Path to Hocomoco annotation file (.jsonl).
            atac_zarr_path: Path to the ATAC-seq data in Zarr format.
            motif_to_tf_mapping: Optional dict mapping motif name to TF gene name.
                                If not provided, it will be inferred.
        """
        self.cell_type = cell_type
        self.quantile_threshold = quantile_threshold
        self.hocomoco_aligned_path = hocomoco_aligned_path
        self.hocomoco_annot_path = hocomoco_annot_path
        self.atac_zarr_path = atac_zarr_path
        self._motif_to_tf = motif_to_tf_mapping
        
        # Lazy-loaded set of allowed TF names
        self._allowed_tfs: Optional[Set[str]] = None
        self.description = f"Promoter Accessibility in {cell_type} > {quantile_threshold} quantile"

    @property
    def motif_to_tf_mapping(self) -> Dict[str, str]:
        """Loads or returns the motif-to-TF mapping."""
        if self._motif_to_tf is None:
            # This is a simplified version; a real implementation might need
            # to parse the motif names more robustly.
            from gcell.dna.hocomoco.HocomocoIO import HocomocoIO
            hoco = HocomocoIO(self.hocomoco_aligned_path, self.hocomoco_annot_path)
            self._motif_to_tf = {
                motif: data['gene_name']
                for motif, data in hoco.motifs_data.items()
            }
        return self._motif_to_tf
        
    def _initialize_filter(self):
        """
        Calculates the set of TFs that pass the accessibility threshold.
        This is done once on the first call to filter_motifs.
        """
        if self._allowed_tfs is not None:
            return

        logger.info(f"Initializing PromoterAccessibilityFilter for cell type '{self.cell_type}'...")
        try:
            from gcell.dna.hocomoco.HocomocoIO import HocomocoIO
            from caesar.io.zarr_io import CelltypeDenseZarrIO
        except ImportError as e:
            logger.error(
                f"Failed to import `caesar` library. This is required for "
                f"PromoterAccessibilityFilter. Please install it. Error: {e}"
            )
            raise

        hoco = HocomocoIO(self.hocomoco_aligned_path, self.hocomoco_annot_path)
        cdz = CelltypeDenseZarrIO(self.atac_zarr_path, mode='r')
        cdz_subset = cdz.subset([self.cell_type])

        # Get TF-TSS accessibility scores
        data = hoco.get_tf_tss_scores(
            zarr_io=cdz_subset,
            window_size=1000,
            conv_size=100,
            verbose=False,
            return_dataframe=True
        )

        accessibility_scores = data[self.cell_type]
        threshold = accessibility_scores.quantile(self.quantile_threshold)
        
        self._allowed_tfs = set(
            accessibility_scores[accessibility_scores > threshold].index
        )
        logger.info(f"  ✓ Filter initialized: {len(self._allowed_tfs)} TFs passed threshold {threshold:.3f}")

    def filter_motifs(self, motif_names: List[str]) -> List[str]:
        self._initialize_filter()
        if self._allowed_tfs is None:
            return [] # Initialization failed

        filtered = []
        for motif in motif_names:
            tf_name = self.motif_to_tf_mapping.get(motif)
            # Handle cases like MA0004.1_Arnt by splitting
            if tf_name is None:
                tf_name = motif.split('.')[0]

            if tf_name in self._allowed_tfs:
                filtered.append(motif)

        logger.debug(f"PromoterAccessibilityFilter: {len(filtered)}/{len(motif_names)} motifs passed")
        return filtered

    def get_description(self) -> str:
        return self.description


def create_annotation_filter(annotation_file: Optional[str],
                            filter_type: str = 'expressed',
                            **kwargs) -> MotifFilter:
    """
    Convenience function to create filter from annotation file.

    Args:
        annotation_file: Path to annotation CSV/TSV
        filter_type: Type of filter ('expressed', 'de', 'whitelist', 'blacklist')
        **kwargs: Additional arguments for specific filter type

    Returns:
        MotifFilter instance

    Examples:
        >>> filt = create_annotation_filter('annot.csv', 'expressed', min_expression=10)
        >>> filt = create_annotation_filter('annot.csv', 'de', padj_threshold=0.01)
        >>> filt = create_annotation_filter('annot.csv', 'whitelist', tf_list=['TP53', 'MYC'])
    """
    # For whitelist/blacklist, motif mapping can come from Hocomoco if no file
    if filter_type in ['whitelist', 'blacklist'] and not annotation_file:
        motif_to_tf = get_hocomoco_motif_mapping()
        df = None # No annotation dataframe
    elif annotation_file:
        # Read annotation file
        if annotation_file.endswith('.csv'):
            df = pd.read_csv(annotation_file)
        else:
            df = pd.read_csv(annotation_file, sep='\t')
        logger.info(f"Loaded annotation file: {len(df)} rows")
        motif_to_tf = dict(zip(df['Motif'], df['gene_name']))
    else:
        raise ValueError(f"Annotation file is required for filter type '{filter_type}'")

    if filter_type == 'expressed':
        if df is None:
            raise ValueError("Annotation DataFrame is missing for 'expressed' filter.")
        return MultiConditionExpressionFilter(
            df,
            expression_columns=kwargs.get('expression_columns', ['baseMean']),
            min_expression=kwargs.get('min_expression', 10.0)
        )

    elif filter_type == 'de':
        if df is None:
            raise ValueError("Annotation DataFrame is missing for 'de' filter.")
        return MultiConditionDEFilter(
            df,
            padj_columns=kwargs.get('padj_columns', ['padj']),
            padj_threshold=kwargs.get('padj_threshold', 0.05)
        )

    elif filter_type == 'whitelist':
        return WhitelistFilter(
            kwargs['tf_list'],
            motif_to_tf,
            case_sensitive=kwargs.get('case_sensitive', False)
        )

    elif filter_type == 'blacklist':
        return BlacklistFilter(
            kwargs['tf_list'],
            motif_to_tf,
            case_sensitive=kwargs.get('case_sensitive', False)
        )

    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

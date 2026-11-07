from __future__ import annotations

from caesar.io.mixins import FootprintMixin

from .genomic_query import (
    QueryResult,
    check_peaks_near_promoter,
    compare_celltypes_at_gene,
    create_output_dir,
    find_gene,
    load_gencode,
    query_region_signal,
    visualize_gene_accessibility,
    write_methods_file,
)
from .iso_libsize_sampler import (
    IsoLibsizeSampler,
    Metacell,
    sample_iso_libsize_metacells,
)

__all__ = [
    "FootprintMixin",
    "IsoLibsizeSampler",
    "Metacell",
    "sample_iso_libsize_metacells",
    # Genomic query functions
    "QueryResult",
    "check_peaks_near_promoter",
    "compare_celltypes_at_gene",
    "create_output_dir",
    "find_gene",
    "load_gencode",
    "query_region_signal",
    "visualize_gene_accessibility",
    "write_methods_file",
]

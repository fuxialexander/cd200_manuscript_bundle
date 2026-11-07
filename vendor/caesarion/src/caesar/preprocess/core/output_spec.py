"""Unified output specification for pseudobulk pipelines.

This module defines the canonical output structure that both pseudobulk approaches
(cell-type aggregation and iso-libsize metacell sampling) must produce. It provides:

1. Schema definitions for manifest and output components
2. Validation utilities for ensuring output conformance
3. Manifest generation utilities

The specification ensures interoperability with downstream consumers including
GET model training pipelines and analysis notebooks.

Example usage:
    >>> from caesar.preprocess.core.output_spec import (
    ...     OutputManifest,
    ...     validate_pseudobulk_output,
    ...     generate_manifest
    ... )
    >>> # Validate an existing output directory
    >>> errors = validate_pseudobulk_output("/path/to/output")
    >>> if errors:
    ...     print(f"Validation failed: {errors}")
    >>> # Generate manifest for new output
    >>> manifest = generate_manifest(
    ...     output_dir="/path/to/output",
    ...     study_id="GSE12345",
    ...     aggregation_method="isolibsize_metacell",
    ...     pipeline_version="0.5.0"
    ... )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Specification version
SPEC_VERSION = "1.0.0"
SCHEMA_URL = "https://caesar.io/schemas/manifest/v1.json"

# Aggregation method types
AggregationMethod = Literal["clustering", "isolibsize_metacell"]
UnitType = Literal["celltype", "metacell"]
Modality = Literal["atac", "rna"]


@dataclass
class InputFileInfo:
    """Information about an input file."""

    type: str  # "fragments", "rna", "metadata"
    path: str
    md5: str | None = None


@dataclass
class UnitInfo:
    """Information about an aggregated unit (celltype or metacell)."""

    id: str
    library_size: int
    n_cells: int
    seed_barcode: str | None = None  # Only for metacells


@dataclass
class PipelineInfo:
    """Pipeline execution information."""

    name: str = "caesar"
    version: str = ""
    workflow: str = ""
    execution_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    runtime_seconds: float | None = None


@dataclass
class AggregationInfo:
    """Aggregation method information."""

    method: AggregationMethod
    groupby_column: str
    n_units: int
    unit_type: UnitType
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceDataInfo:
    """Source data information."""

    n_cells: int
    total_fragments: int
    assembly: str
    modalities: list[Modality]
    input_files: list[InputFileInfo] = field(default_factory=list)


@dataclass
class OutputPaths:
    """Output file paths relative to output directory.

    Note: Naming follows the unified convention where Zarr backend uses
    the same base names as BPCells (e.g., `celltype_dense.zarr` vs
    `celltype_dense_bpcells/`).
    """

    dense_zarr: str = "celltype_dense.zarr"
    tile_counts: str = "tile_counts.h5ad"
    footprint_features: str | None = None
    tss_accessibility: str | None = None
    tes_accessibility: str | None = None
    normalized: str | None = None
    cellstate_tile_counts: str | None = None


@dataclass
class OutputManifest:
    """Complete manifest for a pseudobulk output.

    This is the canonical representation of pipeline metadata that must
    accompany all pseudobulk outputs.
    """

    version: str
    study_id: str
    pipeline: PipelineInfo
    aggregation: AggregationInfo
    source_data: SourceDataInfo
    outputs: OutputPaths
    units: list[UnitInfo]

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "$schema": SCHEMA_URL,
            "version": self.version,
            "study_id": self.study_id,
            "pipeline": {
                "name": self.pipeline.name,
                "version": self.pipeline.version,
                "workflow": self.pipeline.workflow,
                "execution_date": self.pipeline.execution_date,
                "runtime_seconds": self.pipeline.runtime_seconds,
            },
            "aggregation": {
                "method": self.aggregation.method,
                "groupby_column": self.aggregation.groupby_column,
                "n_units": self.aggregation.n_units,
                "unit_type": self.aggregation.unit_type,
                "parameters": self.aggregation.parameters,
            },
            "source_data": {
                "n_cells": self.source_data.n_cells,
                "total_fragments": self.source_data.total_fragments,
                "assembly": self.source_data.assembly,
                "modalities": self.source_data.modalities,
                "input_files": [
                    {"type": f.type, "path": f.path, "md5": f.md5}
                    for f in self.source_data.input_files
                ],
            },
            "outputs": {
                "dense_zarr": self.outputs.dense_zarr,
                "tile_counts": self.outputs.tile_counts,
                "cellstate_tile_counts": self.outputs.cellstate_tile_counts,
                "footprint_features": self.outputs.footprint_features,
                "tss_accessibility": self.outputs.tss_accessibility,
                "tes_accessibility": self.outputs.tes_accessibility,
                "normalized": self.outputs.normalized,
            },
            "units": [
                {
                    "id": u.id,
                    "library_size": u.library_size,
                    "n_cells": u.n_cells,
                    "seed_barcode": u.seed_barcode,
                }
                for u in self.units
            ],
        }

    def save(self, output_dir: str | Path) -> Path:
        """Save manifest to output directory.

        Args:
            output_dir: Directory to save manifest.json

        Returns:
            Path to saved manifest file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"

        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved manifest to {manifest_path}")
        return manifest_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputManifest":
        """Create manifest from dictionary.

        Args:
            data: Dictionary representation of manifest

        Returns:
            OutputManifest instance
        """
        return cls(
            version=data["version"],
            study_id=data["study_id"],
            pipeline=PipelineInfo(
                name=data["pipeline"]["name"],
                version=data["pipeline"]["version"],
                workflow=data["pipeline"]["workflow"],
                execution_date=data["pipeline"]["execution_date"],
                runtime_seconds=data["pipeline"].get("runtime_seconds"),
            ),
            aggregation=AggregationInfo(
                method=data["aggregation"]["method"],
                groupby_column=data["aggregation"]["groupby_column"],
                n_units=data["aggregation"]["n_units"],
                unit_type=data["aggregation"]["unit_type"],
                parameters=data["aggregation"].get("parameters", {}),
            ),
            source_data=SourceDataInfo(
                n_cells=data["source_data"]["n_cells"],
                total_fragments=data["source_data"]["total_fragments"],
                assembly=data["source_data"]["assembly"],
                modalities=data["source_data"]["modalities"],
                input_files=[
                    InputFileInfo(
                        type=f["type"],
                        path=f["path"],
                        md5=f.get("md5"),
                    )
                    for f in data["source_data"].get("input_files", [])
                ],
            ),
            outputs=OutputPaths(
                dense_zarr=data["outputs"]["dense_zarr"],
                tile_counts=data["outputs"]["tile_counts"],
                cellstate_tile_counts=data["outputs"].get("cellstate_tile_counts"),
                footprint_features=data["outputs"].get("footprint_features"),
                tss_accessibility=data["outputs"].get("tss_accessibility"),
                tes_accessibility=data["outputs"].get("tes_accessibility"),
                normalized=data["outputs"].get("normalized"),
            ),
            units=[
                UnitInfo(
                    id=u["id"],
                    library_size=u["library_size"],
                    n_cells=u["n_cells"],
                    seed_barcode=u.get("seed_barcode"),
                )
                for u in data.get("units", [])
            ],
        )

    @classmethod
    def load(cls, output_dir: str | Path) -> "OutputManifest":
        """Load manifest from output directory.

        Args:
            output_dir: Directory containing manifest.json

        Returns:
            OutputManifest instance

        Raises:
            FileNotFoundError: If manifest.json doesn't exist
            json.JSONDecodeError: If manifest is invalid JSON
        """
        manifest_path = Path(output_dir) / "manifest.json"

        with open(manifest_path) as f:
            data = json.load(f)

        return cls.from_dict(data)


@dataclass
class ValidationError:
    """Represents a validation error."""

    component: str
    field: str | None
    message: str
    severity: Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        if self.field:
            return f"[{self.severity.upper()}] {self.component}.{self.field}: {self.message}"
        return f"[{self.severity.upper()}] {self.component}: {self.message}"


def validate_manifest(manifest: OutputManifest) -> list[ValidationError]:
    """Validate manifest internal consistency.

    Args:
        manifest: Manifest to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[ValidationError] = []

    # Version check
    if manifest.version != SPEC_VERSION:
        errors.append(
            ValidationError(
                component="manifest",
                field="version",
                message=f"Version mismatch: expected {SPEC_VERSION}, got {manifest.version}",
                severity="warning",
            )
        )

    # Aggregation consistency
    if manifest.aggregation.method == "clustering":
        expected_unit_type = "celltype"
    else:
        expected_unit_type = "metacell"

    if manifest.aggregation.unit_type != expected_unit_type:
        errors.append(
            ValidationError(
                component="aggregation",
                field="unit_type",
                message=f"Expected unit_type '{expected_unit_type}' for method '{manifest.aggregation.method}'",
            )
        )

    # Unit count consistency
    if len(manifest.units) != manifest.aggregation.n_units:
        errors.append(
            ValidationError(
                component="manifest",
                field="units",
                message=f"Unit count mismatch: n_units={manifest.aggregation.n_units}, "
                f"actual units={len(manifest.units)}",
            )
        )

    # Library size sanity check
    total_unit_libsize = sum(u.library_size for u in manifest.units)
    if manifest.source_data.total_fragments > 0:
        ratio = total_unit_libsize / manifest.source_data.total_fragments
        # Allow some tolerance (80-120%) for fragment sampling
        if ratio < 0.5 or ratio > 1.5:
            errors.append(
                ValidationError(
                    component="manifest",
                    field="library_size",
                    message=f"Library size ratio unusual: {ratio:.2%} of total fragments",
                    severity="warning",
                )
            )

    # Required iso-libsize parameters
    if manifest.aggregation.method == "isolibsize_metacell":
        required_params = ["target_libsize", "n_metacells", "embedding_key"]
        missing = [p for p in required_params if p not in manifest.aggregation.parameters]
        if missing:
            errors.append(
                ValidationError(
                    component="aggregation",
                    field="parameters",
                    message=f"Missing required iso-libsize parameters: {missing}",
                )
            )

    return errors


def validate_dense_zarr(output_dir: Path, manifest: OutputManifest) -> list[ValidationError]:
    """Validate dense zarr structure.

    Args:
        output_dir: Output directory path
        manifest: Manifest for expected structure

    Returns:
        List of validation errors
    """
    import zarr

    errors: list[ValidationError] = []
    dense_path = output_dir / manifest.outputs.dense_zarr

    if not dense_path.exists():
        errors.append(
            ValidationError(
                component="dense_zarr",
                field=None,
                message=f"Dense zarr not found at {dense_path}",
            )
        )
        return errors

    try:
        store = zarr.open(dense_path, mode="r")

        # Check root attributes
        attrs = dict(store.attrs)

        if attrs.get("zarr_type") != "celltype_dense":
            errors.append(
                ValidationError(
                    component="dense_zarr",
                    field="zarr_type",
                    message=f"Expected zarr_type 'celltype_dense', got '{attrs.get('zarr_type')}'",
                )
            )

        if attrs.get("assembly") != manifest.source_data.assembly:
            errors.append(
                ValidationError(
                    component="dense_zarr",
                    field="assembly",
                    message=f"Assembly mismatch: manifest={manifest.source_data.assembly}, "
                    f"zarr={attrs.get('assembly')}",
                )
            )

        # Check unit groups exist
        expected_ids = {u.id for u in manifest.units}
        actual_ids = {k for k in store.keys() if not k.startswith(".")}

        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids

        if missing:
            errors.append(
                ValidationError(
                    component="dense_zarr",
                    field="units",
                    message=f"Missing unit groups: {missing}",
                )
            )
        if extra:
            errors.append(
                ValidationError(
                    component="dense_zarr",
                    field="units",
                    message=f"Unexpected unit groups: {extra}",
                    severity="warning",
                )
            )

    except Exception as e:
        errors.append(
            ValidationError(
                component="dense_zarr",
                field=None,
                message=f"Failed to read zarr: {e}",
            )
        )

    return errors


def validate_tile_counts(output_dir: Path, manifest: OutputManifest) -> list[ValidationError]:
    """Validate tile counts h5ad structure.

    Args:
        output_dir: Output directory path
        manifest: Manifest for expected structure

    Returns:
        List of validation errors
    """
    import anndata as ad

    errors: list[ValidationError] = []
    tile_path = output_dir / manifest.outputs.tile_counts

    if not tile_path.exists():
        errors.append(
            ValidationError(
                component="tile_counts",
                field=None,
                message=f"Tile counts file not found at {tile_path}",
            )
        )
        return errors

    try:
        adata = ad.read_h5ad(tile_path)

        # Check obs (units)
        expected_n_units = manifest.aggregation.n_units
        actual_n_units = adata.n_obs

        if actual_n_units != expected_n_units:
            errors.append(
                ValidationError(
                    component="tile_counts",
                    field="n_obs",
                    message=f"Unit count mismatch: expected {expected_n_units}, got {actual_n_units}",
                )
            )

        # Check required obs columns
        required_obs = ["library_size", "n_cells", "unit_type"]
        missing_obs = [c for c in required_obs if c not in adata.obs.columns]
        if missing_obs:
            errors.append(
                ValidationError(
                    component="tile_counts",
                    field="obs",
                    message=f"Missing required obs columns: {missing_obs}",
                )
            )

        # Check required var columns
        required_var = ["Chromosome", "Start", "End"]
        missing_var = [c for c in required_var if c not in adata.var.columns]
        if missing_var:
            errors.append(
                ValidationError(
                    component="tile_counts",
                    field="var",
                    message=f"Missing required var columns: {missing_var}",
                )
            )

        # Check uns metadata
        if "assembly" not in adata.uns:
            errors.append(
                ValidationError(
                    component="tile_counts",
                    field="uns",
                    message="Missing 'assembly' in uns",
                )
            )
        elif adata.uns["assembly"] != manifest.source_data.assembly:
            errors.append(
                ValidationError(
                    component="tile_counts",
                    field="uns.assembly",
                    message=f"Assembly mismatch: manifest={manifest.source_data.assembly}, "
                    f"h5ad={adata.uns['assembly']}",
                )
            )

    except Exception as e:
        errors.append(
            ValidationError(
                component="tile_counts",
                field=None,
                message=f"Failed to read h5ad: {e}",
            )
        )

    return errors


def validate_footprint_features(
    output_dir: Path, manifest: OutputManifest
) -> list[ValidationError]:
    """Validate footprint features zarr structure.

    Args:
        output_dir: Output directory path
        manifest: Manifest for expected structure

    Returns:
        List of validation errors
    """
    import zarr

    errors: list[ValidationError] = []

    if manifest.outputs.footprint_features is None:
        return errors  # Optional component

    footprint_path = output_dir / manifest.outputs.footprint_features

    if not footprint_path.exists():
        errors.append(
            ValidationError(
                component="footprint_features",
                field=None,
                message=f"Footprint features not found at {footprint_path}",
                severity="warning",
            )
        )
        return errors

    try:
        store = zarr.open(footprint_path, mode="r")
        attrs = dict(store.attrs)

        # Check zarr type
        if attrs.get("zarr_type") != "footprint_features":
            errors.append(
                ValidationError(
                    component="footprint_features",
                    field="zarr_type",
                    message=f"Expected zarr_type 'footprint_features', got '{attrs.get('zarr_type')}'",
                )
            )

        # Check required attributes
        required_attrs = ["n_units", "n_motifs", "n_features", "feature_names"]
        missing_attrs = [a for a in required_attrs if a not in attrs]
        if missing_attrs:
            errors.append(
                ValidationError(
                    component="footprint_features",
                    field="attrs",
                    message=f"Missing required attributes: {missing_attrs}",
                )
            )

        # Check unit count consistency
        if attrs.get("n_units") != manifest.aggregation.n_units:
            errors.append(
                ValidationError(
                    component="footprint_features",
                    field="n_units",
                    message=f"Unit count mismatch: manifest={manifest.aggregation.n_units}, "
                    f"zarr={attrs.get('n_units')}",
                )
            )

    except Exception as e:
        errors.append(
            ValidationError(
                component="footprint_features",
                field=None,
                message=f"Failed to read zarr: {e}",
            )
        )

    return errors


def validate_accessibility_profiles(
    output_dir: Path,
    manifest: OutputManifest,
    component: Literal["tss", "tes"],
) -> list[ValidationError]:
    """Validate TSS or TES accessibility profiles.

    Args:
        output_dir: Output directory path
        manifest: Manifest for expected structure
        component: "tss" or "tes"

    Returns:
        List of validation errors
    """
    import zarr

    errors: list[ValidationError] = []

    if component == "tss":
        output_path = manifest.outputs.tss_accessibility
        expected_type = "tss_accessibility"
    else:
        output_path = manifest.outputs.tes_accessibility
        expected_type = "tes_accessibility"

    if output_path is None:
        return errors  # Optional component

    profile_path = output_dir / output_path

    if not profile_path.exists():
        errors.append(
            ValidationError(
                component=f"{component}_accessibility",
                field=None,
                message=f"{component.upper()} accessibility not found at {profile_path}",
                severity="warning",
            )
        )
        return errors

    try:
        store = zarr.open(profile_path, mode="r")
        attrs = dict(store.attrs)

        # Check zarr type
        if attrs.get("zarr_type") != expected_type:
            errors.append(
                ValidationError(
                    component=f"{component}_accessibility",
                    field="zarr_type",
                    message=f"Expected zarr_type '{expected_type}', got '{attrs.get('zarr_type')}'",
                )
            )

        # Check required attributes
        required_attrs = ["n_units", "n_genes", "upstream", "downstream", "profile_width"]
        missing_attrs = [a for a in required_attrs if a not in attrs]
        if missing_attrs:
            errors.append(
                ValidationError(
                    component=f"{component}_accessibility",
                    field="attrs",
                    message=f"Missing required attributes: {missing_attrs}",
                )
            )

        # Check unit count consistency
        if attrs.get("n_units") != manifest.aggregation.n_units:
            errors.append(
                ValidationError(
                    component=f"{component}_accessibility",
                    field="n_units",
                    message=f"Unit count mismatch: manifest={manifest.aggregation.n_units}, "
                    f"zarr={attrs.get('n_units')}",
                )
            )

        # Check profile width consistency
        expected_width = attrs.get("upstream", 0) + attrs.get("downstream", 0)
        if attrs.get("profile_width") != expected_width:
            errors.append(
                ValidationError(
                    component=f"{component}_accessibility",
                    field="profile_width",
                    message=f"Profile width inconsistency: {attrs.get('profile_width')} != "
                    f"upstream ({attrs.get('upstream')}) + downstream ({attrs.get('downstream')})",
                    severity="warning",
                )
            )

        # Check required arrays
        required_arrays = ["profiles", "gene_ids", "gene_names", "unit_ids"]
        missing_arrays = [a for a in required_arrays if a not in store]
        if missing_arrays:
            errors.append(
                ValidationError(
                    component=f"{component}_accessibility",
                    field="arrays",
                    message=f"Missing required arrays: {missing_arrays}",
                )
            )

    except Exception as e:
        errors.append(
            ValidationError(
                component=f"{component}_accessibility",
                field=None,
                message=f"Failed to read zarr: {e}",
            )
        )

    return errors


def validate_pseudobulk_output(output_dir: str | Path) -> list[ValidationError]:
    """Validate complete pseudobulk output against specification.

    This is the main validation entry point that checks all components.

    Args:
        output_dir: Path to output directory containing manifest.json

    Returns:
        List of validation errors (empty if valid)

    Example:
        >>> errors = validate_pseudobulk_output("/path/to/output")
        >>> if errors:
        ...     for e in errors:
        ...         print(e)
        ... else:
        ...     print("Output is valid")
    """
    output_dir = Path(output_dir)
    errors: list[ValidationError] = []

    # Check manifest exists
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        errors.append(
            ValidationError(
                component="manifest",
                field=None,
                message="manifest.json not found in output directory",
            )
        )
        return errors

    # Load and validate manifest
    try:
        manifest = OutputManifest.load(output_dir)
    except Exception as e:
        errors.append(
            ValidationError(
                component="manifest",
                field=None,
                message=f"Failed to load manifest: {e}",
            )
        )
        return errors

    # Manifest internal validation
    errors.extend(validate_manifest(manifest))

    # Component validation
    errors.extend(validate_dense_zarr(output_dir, manifest))
    errors.extend(validate_tile_counts(output_dir, manifest))
    errors.extend(validate_footprint_features(output_dir, manifest))
    errors.extend(validate_accessibility_profiles(output_dir, manifest, "tss"))
    errors.extend(validate_accessibility_profiles(output_dir, manifest, "tes"))

    return errors


def generate_manifest(
    output_dir: str | Path,
    study_id: str,
    aggregation_method: AggregationMethod,
    pipeline_version: str,
    workflow_name: str = "",
    assembly: str = "hg38",
    modalities: list[Modality] | None = None,
    aggregation_params: dict[str, Any] | None = None,
    input_files: list[dict[str, str]] | None = None,
    runtime_seconds: float | None = None,
    save: bool = True,
) -> OutputManifest:
    """Generate manifest by introspecting output directory.

    This function examines the output directory structure and creates
    a manifest that describes the outputs.

    Args:
        output_dir: Path to output directory
        study_id: Study identifier (e.g., "GSE12345")
        aggregation_method: "clustering" or "isolibsize_metacell"
        pipeline_version: Caesar version string
        workflow_name: Workflow name (e.g., "multiome-isolibsize")
        assembly: Genome assembly (default "hg38")
        modalities: List of modalities (default ["atac"])
        aggregation_params: Method-specific parameters
        input_files: List of input file info dicts
        runtime_seconds: Pipeline runtime in seconds
        save: Whether to save manifest.json to output_dir

    Returns:
        Generated OutputManifest

    Example:
        >>> manifest = generate_manifest(
        ...     output_dir="/path/to/output",
        ...     study_id="GSE12345",
        ...     aggregation_method="isolibsize_metacell",
        ...     pipeline_version="0.5.0",
        ...     workflow_name="multiome-isolibsize",
        ...     aggregation_params={
        ...         "target_libsize": 5000000,
        ...         "n_metacells": 100,
        ...         "embedding_key": "X_pca"
        ...     }
        ... )
    """
    import zarr

    output_dir = Path(output_dir)
    modalities = modalities or ["atac"]
    aggregation_params = aggregation_params or {}
    input_files = input_files or []

    # Determine unit type
    unit_type: UnitType = "metacell" if aggregation_method == "isolibsize_metacell" else "celltype"

    # Detect outputs
    outputs = OutputPaths()

    # Check dense zarr - use canonical name first, then check for iso-libsize variants
    dense_path = output_dir / "celltype_dense.zarr"
    if dense_path.exists():
        outputs.dense_zarr = "celltype_dense.zarr"
    else:
        # Look for iso-libsize variants (e.g., celltype_dense_isolibsize.2M.zarr)
        isolibsize_patterns = list(output_dir.glob("celltype_dense_isolibsize.*.zarr"))
        if isolibsize_patterns:
            # Use the first one found (typically only one per output dir)
            outputs.dense_zarr = isolibsize_patterns[0].name
            dense_path = isolibsize_patterns[0]
        else:
            # Check legacy names for backward compatibility (will be removed in future)
            for name in ["dense.zarr", "pseudobulk.zarr"]:
                if (output_dir / name).exists():
                    outputs.dense_zarr = name
                    dense_path = output_dir / name
                    logger.warning(
                        f"Found legacy dense zarr name '{name}'. "
                        "Please migrate to 'celltype_dense.zarr' or "
                        "'celltype_dense_isolibsize.{{N}}M.zarr'."
                    )
                    break

    # Check tile counts
    tile_path = output_dir / "tile_counts.h5ad"
    if tile_path.exists():
        outputs.tile_counts = "tile_counts.h5ad"
    else:
        for name in ["tiles.h5ad", "variable_tiles.h5ad"]:
            if (output_dir / name).exists():
                outputs.tile_counts = name
                tile_path = output_dir / name
                break

    # Check cellstate tile counts
    if (output_dir / "cellstate_tile_counts.h5ad").exists():
        outputs.cellstate_tile_counts = "cellstate_tile_counts.h5ad"

    # Check optional components
    if (output_dir / "footprint_features.zarr").exists():
        outputs.footprint_features = "footprint_features.zarr"
    elif (output_dir / "ml_features.zarr").exists():
        outputs.footprint_features = "ml_features.zarr"

    if (output_dir / "tss_accessibility.h5ad").exists():
        outputs.tss_accessibility = "tss_accessibility.h5ad"
    elif (output_dir / "tss_accessibility.zarr").exists():
        outputs.tss_accessibility = "tss_accessibility.zarr"

    if (output_dir / "tes_accessibility.h5ad").exists():
        outputs.tes_accessibility = "tes_accessibility.h5ad"
    elif (output_dir / "tes_accessibility.zarr").exists():
        outputs.tes_accessibility = "tes_accessibility.zarr"

    if (output_dir / "normalized.zarr").exists():
        outputs.normalized = "normalized.zarr"

    # Extract unit information from dense zarr
    units: list[UnitInfo] = []
    n_cells = 0
    total_fragments = 0
    groupby_column = "labels"

    if dense_path.exists():
        try:
            store = zarr.open(dense_path, mode="r")
            attrs = dict(store.attrs)

            groupby_column = attrs.get("groupby", "labels")
            if aggregation_method == "isolibsize_metacell":
                groupby_column = attrs.get("metacell_column", "metacell_id")

            # Get unit IDs and info
            unit_ids = [k for k in store.keys() if not k.startswith(".")]
            for unit_id in unit_ids:
                unit_group = store[unit_id]
                unit_attrs = dict(unit_group.attrs)
                units.append(
                    UnitInfo(
                        id=unit_id,
                        library_size=unit_attrs.get("library_size", 0),
                        n_cells=unit_attrs.get("n_cells", 0),
                        seed_barcode=unit_attrs.get("seed_barcode"),
                    )
                )
                n_cells += unit_attrs.get("n_cells", 0)
                total_fragments += unit_attrs.get("library_size", 0)

        except Exception as e:
            logger.warning(f"Failed to read dense zarr for manifest generation: {e}")

    # Create manifest
    manifest = OutputManifest(
        version=SPEC_VERSION,
        study_id=study_id,
        pipeline=PipelineInfo(
            name="caesar",
            version=pipeline_version,
            workflow=workflow_name,
            runtime_seconds=runtime_seconds,
        ),
        aggregation=AggregationInfo(
            method=aggregation_method,
            groupby_column=groupby_column,
            n_units=len(units),
            unit_type=unit_type,
            parameters=aggregation_params,
        ),
        source_data=SourceDataInfo(
            n_cells=n_cells,
            total_fragments=total_fragments,
            assembly=assembly,
            modalities=modalities,
            input_files=[
                InputFileInfo(
                    type=f.get("type", "unknown"),
                    path=f.get("path", ""),
                    md5=f.get("md5"),
                )
                for f in input_files
            ],
        ),
        outputs=outputs,
        units=units,
    )

    if save:
        manifest.save(output_dir)

    return manifest


def check_output_completeness(output_dir: str | Path) -> dict[str, bool]:
    """Check which output components are present.

    Args:
        output_dir: Path to output directory

    Returns:
        Dictionary mapping component names to presence boolean

    Example:
        >>> status = check_output_completeness("/path/to/output")
        >>> print(status)
        {'manifest': True, 'dense_zarr': True, 'tile_counts': True,
         'footprint_features': False, 'tss_accessibility': True, ...}
    """
    output_dir = Path(output_dir)

    return {
        "manifest": (output_dir / "manifest.json").exists(),
        "dense_zarr": (output_dir / "dense.zarr").exists()
        or (output_dir / "celltype_dense.zarr").exists(),
        "tile_counts": (output_dir / "tile_counts.h5ad").exists()
        or (output_dir / "tiles.h5ad").exists(),
        "footprint_features": (output_dir / "footprint_features.zarr").exists()
        or (output_dir / "ml_features.zarr").exists(),
        "tss_accessibility": (output_dir / "tss_accessibility.zarr").exists(),
        "tes_accessibility": (output_dir / "tes_accessibility.zarr").exists(),
        "normalized": (output_dir / "normalized.zarr").exists(),
    }


def get_required_components() -> list[str]:
    """Get list of required output components.

    Returns:
        List of required component names
    """
    return ["manifest", "dense_zarr", "tile_counts"]


def get_optional_components() -> list[str]:
    """Get list of optional output components.

    Returns:
        List of optional component names
    """
    return ["footprint_features", "tss_accessibility", "tes_accessibility", "normalized"]


def create_tile_counts_h5ad(
    dense_zarr_path: str | Path,
    output_path: str | Path,
    tile_size: int = 500,
    selection_method: str = "variance",
    variance_threshold: float = 0.1,
    max_tiles: int | None = 500000,
    peaks_path: str | Path | None = None,
    assembly: str = "hg38",
) -> Path:
    """Create tile counts h5ad conforming to output specification.

    This function generates the tile_counts.h5ad file from a pseudobulk
    dense zarr store. It extracts variable tiles (non-background) and
    stores them in the specification-compliant AnnData format.

    Args:
        dense_zarr_path: Path to celltype_dense or multiome_dense zarr
        output_path: Path for output h5ad file
        tile_size: Size of genomic tiles in bp (default 500)
        selection_method: Method for selecting variable tiles
            - "variance": Select tiles with highest variance
            - "peaks": Use provided peaks file
            - "all": Keep all tiles (may be very large)
        variance_threshold: Minimum variance for tile selection (if variance method)
        max_tiles: Maximum number of tiles to keep (None for no limit)
        peaks_path: Path to peaks file (BED or narrowPeak) if using peaks selection
        assembly: Genome assembly (default "hg38")

    Returns:
        Path to created h5ad file

    Example:
        >>> path = create_tile_counts_h5ad(
        ...     dense_zarr_path="/path/to/dense.zarr",
        ...     output_path="/path/to/tile_counts.h5ad",
        ...     selection_method="variance",
        ...     variance_threshold=0.1
        ... )
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import zarr
    from scipy.sparse import csr_matrix

    from gcell.dna.genome import ChromSize

    dense_zarr_path = Path(dense_zarr_path)
    output_path = Path(output_path)

    # Open dense zarr
    store = zarr.open(dense_zarr_path, mode="r")
    attrs = dict(store.attrs)

    # Get unit IDs
    unit_ids = [k for k in store.keys() if not k.startswith(".")]
    n_units = len(unit_ids)

    if n_units == 0:
        raise ValueError(f"No units found in {dense_zarr_path}")

    # Get chromosome sizes
    chrom_sizes = ChromSize(assembly)

    # Determine chromosomes to process
    sample_unit = store[unit_ids[0]]
    if "chrs" in sample_unit:
        chroms = list(sample_unit["chrs"].keys())
    else:
        chroms = [k for k in sample_unit.keys() if k.startswith("chr") and "_" not in k]

    logger.info(f"Processing {n_units} units across {len(chroms)} chromosomes")

    # Create tile coordinates
    tiles_list = []
    for chrom in chroms:
        chrom_len = chrom_sizes.get(chrom, 0)
        if chrom_len == 0:
            continue

        n_tiles = (chrom_len + tile_size - 1) // tile_size
        for i in range(n_tiles):
            start = i * tile_size
            end = min((i + 1) * tile_size, chrom_len)
            tiles_list.append({
                "tile_id": f"{chrom}_{start}_{end}",
                "Chromosome": chrom,
                "Start": start,
                "End": end,
            })

    tiles_df = pd.DataFrame(tiles_list)
    tiles_df = tiles_df.set_index("tile_id")
    n_total_tiles = len(tiles_df)

    logger.info(f"Created {n_total_tiles} tiles from genome")

    # Build count matrix
    # For efficiency, we process chromosome by chromosome
    counts_by_unit = []

    for unit_id in unit_ids:
        unit_group = store[unit_id]
        unit_counts = []

        for chrom in chroms:
            chrom_len = chrom_sizes.get(chrom, 0)
            if chrom_len == 0:
                continue

            # Get coverage array
            if "chrs" in unit_group:
                if chrom not in unit_group["chrs"]:
                    # Fill with zeros
                    n_tiles = (chrom_len + tile_size - 1) // tile_size
                    unit_counts.extend([0] * n_tiles)
                    continue
                coverage = unit_group["chrs"][chrom][:]
            else:
                if chrom not in unit_group:
                    n_tiles = (chrom_len + tile_size - 1) // tile_size
                    unit_counts.extend([0] * n_tiles)
                    continue
                coverage = unit_group[chrom][:]

            # Sum counts per tile
            n_tiles = (chrom_len + tile_size - 1) // tile_size
            for i in range(n_tiles):
                start = i * tile_size
                end = min((i + 1) * tile_size, len(coverage))
                if start < len(coverage):
                    tile_count = int(np.sum(coverage[start:end]))
                else:
                    tile_count = 0
                unit_counts.append(tile_count)

        counts_by_unit.append(unit_counts)

    # Create count matrix
    count_matrix = np.array(counts_by_unit, dtype=np.float32)

    logger.info(f"Built count matrix: {count_matrix.shape}")

    # Select variable tiles
    if selection_method == "variance":
        # Compute variance across units
        tile_variance = np.var(count_matrix, axis=0)

        # Filter by variance threshold
        variable_mask = tile_variance >= variance_threshold

        # Apply max_tiles limit
        if max_tiles is not None and np.sum(variable_mask) > max_tiles:
            # Keep top max_tiles by variance
            threshold = np.sort(tile_variance[variable_mask])[-max_tiles]
            variable_mask = tile_variance >= threshold

        selected_indices = np.where(variable_mask)[0]

    elif selection_method == "peaks":
        if peaks_path is None:
            raise ValueError("peaks_path required when selection_method='peaks'")

        # Load peaks
        peaks_df = pd.read_csv(peaks_path, sep="\t", header=None)
        peaks_df.columns = ["Chromosome", "Start", "End"] + [f"col_{i}" for i in range(3, len(peaks_df.columns))]

        # Find tiles overlapping peaks
        import pyranges as pr

        tiles_pr = pr.PyRanges(tiles_df.reset_index()[["Chromosome", "Start", "End"]])
        peaks_pr = pr.PyRanges(peaks_df[["Chromosome", "Start", "End"]])

        overlaps = tiles_pr.overlap(peaks_pr)
        overlap_keys = set(
            f"{row.Chromosome}_{row.Start}_{row.End}"
            for _, row in overlaps.df.iterrows()
        )

        variable_mask = tiles_df.index.isin(overlap_keys)
        selected_indices = np.where(variable_mask)[0]

    elif selection_method == "all":
        selected_indices = np.arange(n_total_tiles)
        if max_tiles is not None and len(selected_indices) > max_tiles:
            # Subsample
            selected_indices = np.random.choice(selected_indices, max_tiles, replace=False)
            selected_indices = np.sort(selected_indices)

    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")

    logger.info(f"Selected {len(selected_indices)} variable tiles")

    # Subset count matrix and tiles
    count_matrix = count_matrix[:, selected_indices]
    tiles_df = tiles_df.iloc[selected_indices]

    # Create unit metadata
    unit_metadata = []
    for unit_id in unit_ids:
        unit_group = store[unit_id]
        unit_attrs = dict(unit_group.attrs)
        unit_metadata.append({
            "unit_id": unit_id,
            "library_size": unit_attrs.get("library_size", 0),
            "n_cells": unit_attrs.get("n_cells", 0),
            "unit_type": attrs.get("aggregation_method", "celltype"),
        })

    obs_df = pd.DataFrame(unit_metadata)
    obs_df = obs_df.set_index("unit_id")

    # Add tile type classification
    tiles_df = tiles_df.copy()
    tiles_df["tile_type"] = "variable"
    tiles_df["peak_score"] = np.nan
    tiles_df["is_promoter"] = False  # Could be annotated later

    # Create sparse matrix for storage efficiency
    sparse_matrix = csr_matrix(count_matrix)

    # Create AnnData
    adata = ad.AnnData(
        X=sparse_matrix,
        obs=obs_df,
        var=tiles_df,
    )

    # Add uns metadata
    adata.uns["assembly"] = assembly
    adata.uns["tile_size"] = tile_size
    adata.uns["n_total_tiles"] = n_total_tiles
    adata.uns["n_variable_tiles"] = len(selected_indices)
    adata.uns["selection_method"] = selection_method
    adata.uns["variance_threshold"] = variance_threshold if selection_method == "variance" else None
    adata.uns["aggregation_method"] = attrs.get("aggregation_method", "clustering")
    adata.uns["groupby_column"] = attrs.get("groupby", "labels")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    logger.info(f"Saved tile counts to {output_path}")

    return output_path


def create_accessibility_profiles_zarr(
    dense_zarr_path: str | Path,
    output_path: str | Path,
    site_type: Literal["tss", "tes"],
    upstream: int = 2000,
    downstream: int = 2000,
    gene_annotation: str = "GENCODE_v43",
    gene_set: str = "protein_coding",
    assembly: str = "hg38",
) -> Path:
    """Create TSS or TES accessibility profiles zarr conforming to specification.

    This function generates accessibility profiles centered on transcription
    start sites (TSS) or transcription end sites (TES) for each unit.

    Args:
        dense_zarr_path: Path to celltype_dense or multiome_dense zarr
        output_path: Path for output zarr
        site_type: "tss" or "tes"
        upstream: Bases upstream of site (default 2000)
        downstream: Bases downstream of site (default 2000)
        gene_annotation: Gene annotation version
        gene_set: Gene set to use (e.g., "protein_coding")
        assembly: Genome assembly (default "hg38")

    Returns:
        Path to created zarr

    Example:
        >>> path = create_accessibility_profiles_zarr(
        ...     dense_zarr_path="/path/to/dense.zarr",
        ...     output_path="/path/to/tss_accessibility.zarr",
        ...     site_type="tss",
        ...     upstream=2000,
        ...     downstream=2000
        ... )
    """
    import numpy as np
    import zarr
    from numcodecs import Blosc

    from gcell.dna.gene import get_gene_positions

    dense_zarr_path = Path(dense_zarr_path)
    output_path = Path(output_path)

    # Open dense zarr
    store = zarr.open(dense_zarr_path, mode="r")
    attrs = dict(store.attrs)

    # Get unit IDs
    unit_ids = [k for k in store.keys() if not k.startswith(".")]
    n_units = len(unit_ids)

    if n_units == 0:
        raise ValueError(f"No units found in {dense_zarr_path}")

    # Get gene positions
    gene_df = get_gene_positions(
        assembly=assembly,
        gene_set=gene_set,
        site_type=site_type,
    )

    gene_ids = gene_df["gene_id"].tolist()
    gene_names = gene_df["gene_name"].tolist()
    n_genes = len(gene_ids)

    logger.info(f"Processing {n_units} units for {n_genes} genes")

    # Profile dimensions
    profile_width = upstream + downstream

    # Initialize profiles array
    profiles = np.zeros((n_units, n_genes, profile_width), dtype=np.float32)

    # Extract profiles for each unit
    for unit_idx, unit_id in enumerate(unit_ids):
        unit_group = store[unit_id]

        for gene_idx, (_, row) in enumerate(gene_df.iterrows()):
            chrom = row["chromosome"]
            position = row["position"]
            strand = row["strand"]

            # Determine extraction window
            if strand == "+":
                start = position - upstream
                end = position + downstream
            else:
                start = position - downstream
                end = position + upstream

            # Get coverage
            if "chrs" in unit_group:
                if chrom not in unit_group["chrs"]:
                    continue
                coverage = unit_group["chrs"][chrom][:]
            else:
                if chrom not in unit_group:
                    continue
                coverage = unit_group[chrom][:]

            # Extract profile
            profile_start = max(0, start)
            profile_end = min(len(coverage), end)

            if profile_start >= len(coverage) or profile_end <= 0:
                continue

            # Map to profile coordinates
            out_start = max(0, -start)
            out_end = out_start + (profile_end - profile_start)

            profile = np.zeros(profile_width, dtype=np.float32)
            profile[out_start:out_end] = coverage[profile_start:profile_end]

            # Reverse if minus strand
            if strand == "-":
                profile = profile[::-1]

            profiles[unit_idx, gene_idx, :] = profile

    # Create output zarr
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compressor = Blosc(cname="zstd", clevel=4, shuffle=Blosc.BITSHUFFLE)
    out_store = zarr.open(output_path, mode="w")

    # Set attributes
    out_store.attrs["zarr_type"] = f"{site_type}_accessibility"
    out_store.attrs["n_units"] = n_units
    out_store.attrs["n_genes"] = n_genes
    out_store.attrs["upstream"] = upstream
    out_store.attrs["downstream"] = downstream
    out_store.attrs["profile_width"] = profile_width
    out_store.attrs["resolution"] = 1
    out_store.attrs["gene_annotation"] = gene_annotation
    out_store.attrs["gene_set"] = gene_set
    out_store.attrs["strand_aware"] = True
    out_store.attrs["aggregation_method"] = attrs.get("aggregation_method", "clustering")
    out_store.attrs["completed"] = True

    # Store arrays
    out_store.create_dataset(
        "profiles",
        data=profiles,
        chunks=(min(10, n_units), min(1000, n_genes), profile_width),
        compressor=compressor,
    )

    out_store.create_dataset("gene_ids", data=np.array(gene_ids, dtype=object))
    out_store.create_dataset("gene_names", data=np.array(gene_names, dtype=object))
    out_store.create_dataset("unit_ids", data=np.array(unit_ids, dtype=object))

    # Store TSS/TES positions
    tss_positions = np.array([
        [i, row["position"], 1 if row["strand"] == "+" else -1]
        for i, (_, row) in enumerate(gene_df.iterrows())
    ], dtype=np.int64)
    out_store.create_dataset("tss_positions", data=tss_positions)

    logger.info(f"Saved {site_type.upper()} accessibility profiles to {output_path}")

    return output_path

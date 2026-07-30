"""Motif annotation catalogs with RNA and deletion-effect filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .perturbations import manuscript_plot_effects


def _normalized_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _find_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    normalized = {_normalized_name(column): str(column) for column in frame.columns}
    for candidate in candidates:
        found = normalized.get(_normalized_name(candidate))
        if found:
            return found
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".bed"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep=None, engine="python")


@dataclass(frozen=True)
class RNAColumnMapping:
    gene: str
    logfc: str | None
    expression: str | None
    padj: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotifRecord:
    chromosome: str
    start: int
    end: int
    tf: str
    motif_name: str
    p_value: float | None
    score: float | None
    hit_count: int
    rna_logfc: float | None
    rna_expression: float | None
    rna_padj: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RNAExpressionTable:
    """Normalize a user-provided RNA table to TF, logFC, and expression."""

    GENE_COLUMNS = (
        "gene_name",
        "gene",
        "gene_symbol",
        "symbol",
        "tf",
        "transcription_factor",
        "unnamed: 0",
    )
    LOGFC_COLUMNS = (
        "log2FoldChange",
        "log2FC",
        "logFC",
        "lfc",
        "lfc_1f",
        "fold_change",
    )
    EXPRESSION_COLUMNS = (
        "baseMean",
        "expression",
        "mean_expression",
        "avg_expression",
        "mean_count",
        "normalized_count",
        "TPM",
        "FPKM",
        "CPM",
    )
    PADJ_COLUMNS = (
        "padj",
        "adjusted_p_value",
        "adjusted_pvalue",
        "fdr",
        "q_value",
        "qvalue",
    )

    def __init__(
        self,
        path: str | Path,
        column_overrides: dict[str, str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"RNA table not found: {self.path}")
        self.frame = _read_table(self.path)
        if self.frame.empty:
            raise ValueError(f"RNA table is empty: {self.path}")
        overrides = column_overrides or {}
        gene_column = overrides.get("gene") or _find_column(
            self.frame,
            self.GENE_COLUMNS,
        )
        logfc_column = overrides.get("logfc") or _find_column(
            self.frame,
            self.LOGFC_COLUMNS,
        )
        expression_column = overrides.get("expression") or _find_column(
            self.frame,
            self.EXPRESSION_COLUMNS,
        )
        padj_column = overrides.get("padj") or _find_column(
            self.frame,
            self.PADJ_COLUMNS,
        )
        if not gene_column or gene_column not in self.frame:
            raise ValueError(
                "RNA table needs a gene column such as gene, gene_name, "
                "symbol, or an unnamed first column"
            )
        for label, column in (
            ("logFC", logfc_column),
            ("expression", expression_column),
        ):
            if column and column not in self.frame:
                raise ValueError(f"Requested RNA {label} column is absent: {column}")
        if not logfc_column and not expression_column:
            raise ValueError(
                "RNA table needs a logFC and/or expression column "
                "(for example log2FoldChange and baseMean)"
            )
        self.mapping = RNAColumnMapping(
            gene=gene_column,
            logfc=logfc_column,
            expression=expression_column,
            padj=padj_column,
        )
        normalized = pd.DataFrame(
            {
                "tf_key": self.frame[gene_column]
                .astype(str)
                .str.strip()
                .str.upper(),
                "rna_logfc": (
                    pd.to_numeric(self.frame[logfc_column], errors="coerce")
                    if logfc_column
                    else np.nan
                ),
                "rna_expression": (
                    pd.to_numeric(self.frame[expression_column], errors="coerce")
                    if expression_column
                    else np.nan
                ),
                "rna_padj": (
                    pd.to_numeric(self.frame[padj_column], errors="coerce")
                    if padj_column
                    else np.nan
                ),
            }
        )
        normalized = normalized[normalized["tf_key"].ne("")]
        normalized["abs_logfc"] = normalized["rna_logfc"].abs()
        normalized = normalized.sort_values(
            ["tf_key", "abs_logfc", "rna_expression"],
            ascending=[True, False, False],
            na_position="last",
        )
        self.normalized = normalized.drop_duplicates("tf_key").set_index("tf_key")

    def lookup(self, tf: str) -> tuple[float | None, float | None]:
        details = self.lookup_details(tf)
        return details["logfc"], details["expression"]

    def lookup_details(self, tf: str) -> dict[str, float | None]:
        key = str(tf).strip().upper()
        if key not in self.normalized.index:
            return {"logfc": None, "expression": None, "padj": None}
        row = self.normalized.loc[key]
        logfc = row["rna_logfc"]
        expression = row["rna_expression"]
        padj = row["rna_padj"]
        return {
            "logfc": float(logfc) if pd.notna(logfc) else None,
            "expression": (
                float(expression) if pd.notna(expression) else None
            ),
            "padj": float(padj) if pd.notna(padj) else None,
        }

    def ranges(self) -> dict[str, float | None]:
        logfc = self.normalized["rna_logfc"].dropna().abs()
        expression = self.normalized["rna_expression"].dropna()
        return {
            "abs_logfc_max": float(logfc.max()) if not logfc.empty else None,
            "expression_min": (
                float(expression.min()) if not expression.empty else None
            ),
            "expression_max": (
                float(expression.max()) if not expression.empty else None
            ),
        }


class MotifAnnotationCatalog:
    """Region-scoped motif hits joined to optional RNA measurements."""

    CHROMOSOME_COLUMNS = ("chromosome", "chrom", "chr", "seqname")
    START_COLUMNS = ("start", "chromstart", "position")
    END_COLUMNS = ("end", "chromend", "stop")
    TF_COLUMNS = (
        "gene_name",
        "tf",
        "gene",
        "gene_symbol",
        "transcription_factor",
    )
    MOTIF_COLUMNS = ("motif_name", "motif", "name", "pattern")
    PVALUE_COLUMNS = ("p_value", "pvalue", "p-value", "q_value", "qvalue")
    SCORE_COLUMNS = ("score", "motif_score", "log_odds")

    def __init__(
        self,
        motif_path: str | Path,
        *,
        chromosome: str,
        start: int,
        end: int,
        rna_path: str | Path | None = None,
        rna_path_2: str | Path | None = None,
        gene_list_path: str | Path | None = None,
        rna_columns: dict[str, str] | None = None,
        merge_overlaps: bool = True,
    ) -> None:
        self.motif_path = Path(motif_path).expanduser().resolve()
        if not self.motif_path.is_file():
            raise FileNotFoundError(f"Motif table not found: {self.motif_path}")
        self.chromosome = chromosome
        self.start = int(start)
        self.end = int(end)
        self.rna = (
            RNAExpressionTable(rna_path, rna_columns)
            if rna_path
            else None
        )
        self.rna_2 = (
            RNAExpressionTable(rna_path_2, rna_columns)
            if rna_path_2
            else None
        )
        self.gene_list_path = (
            Path(gene_list_path).expanduser().resolve()
            if gene_list_path
            else None
        )
        self.gene_list: set[str] = set()
        if self.gene_list_path is not None:
            if not self.gene_list_path.is_file():
                raise FileNotFoundError(
                    f"Gene list not found: {self.gene_list_path}"
                )
            genes = _read_table(self.gene_list_path)
            gene_column = _find_column(genes, RNAExpressionTable.GENE_COLUMNS)
            if not gene_column:
                raise ValueError(
                    "Gene list needs a gene, gene_name, symbol, or TF column"
                )
            self.gene_list = {
                str(value).strip().upper()
                for value in genes[gene_column]
                if str(value).strip()
            }
        frame = _read_table(self.motif_path)
        if frame.empty:
            raise ValueError(f"Motif table is empty: {self.motif_path}")
        columns = {
            "chromosome": _find_column(frame, self.CHROMOSOME_COLUMNS),
            "start": _find_column(frame, self.START_COLUMNS),
            "end": _find_column(frame, self.END_COLUMNS),
            "tf": _find_column(frame, self.TF_COLUMNS),
            "motif_name": _find_column(frame, self.MOTIF_COLUMNS),
            "p_value": _find_column(frame, self.PVALUE_COLUMNS),
            "score": _find_column(frame, self.SCORE_COLUMNS),
        }
        missing = [
            name
            for name in ("chromosome", "start", "end", "tf")
            if not columns[name]
        ]
        if missing:
            raise ValueError(
                "Motif table is missing required columns: " + ", ".join(missing)
            )
        normalized = pd.DataFrame(
            {
                "chromosome": frame[columns["chromosome"]].astype(str),
                "start": pd.to_numeric(frame[columns["start"]], errors="coerce"),
                "end": pd.to_numeric(frame[columns["end"]], errors="coerce"),
                "tf": frame[columns["tf"]].astype(str).str.strip(),
                "motif_name": (
                    frame[columns["motif_name"]].astype(str)
                    if columns["motif_name"]
                    else frame[columns["tf"]].astype(str)
                ),
                "p_value": (
                    pd.to_numeric(frame[columns["p_value"]], errors="coerce")
                    if columns["p_value"]
                    else np.nan
                ),
                "score": (
                    pd.to_numeric(frame[columns["score"]], errors="coerce")
                    if columns["score"]
                    else np.nan
                ),
            }
        ).dropna(subset=["start", "end"])
        normalized["start"] = normalized["start"].astype(int)
        normalized["end"] = normalized["end"].astype(int)
        normalized = normalized[
            normalized["chromosome"].eq(chromosome)
            & normalized["end"].gt(self.start)
            & normalized["start"].lt(self.end)
            & normalized["tf"].ne("")
            & normalized["end"].gt(normalized["start"])
        ].copy()
        if normalized.empty:
            raise ValueError(
                f"No motif hits overlap {chromosome}:{self.start}-{self.end}"
            )
        self.raw_count = len(normalized)
        if merge_overlaps:
            normalized = self._merge_overlaps(normalized)
        self.records = [self._record(row) for row in normalized.itertuples()]

    @staticmethod
    def _merge_overlaps(frame: pd.DataFrame) -> pd.DataFrame:
        merged: list[dict[str, Any]] = []
        ordered = frame.sort_values(["chromosome", "tf", "start", "end"])
        for (_chromosome, _tf), group in ordered.groupby(
            ["chromosome", "tf"],
            sort=False,
        ):
            active: list[Any] = []
            active_end: int | None = None
            for row in group.itertuples(index=False):
                if active and active_end is not None and row.start > active_end:
                    merged.append(MotifAnnotationCatalog._merge_cluster(active))
                    active = []
                    active_end = None
                active.append(row)
                active_end = max(active_end or row.end, row.end)
            if active:
                merged.append(MotifAnnotationCatalog._merge_cluster(active))
        return pd.DataFrame(merged).sort_values(["start", "tf"]).reset_index(drop=True)

    @staticmethod
    def _merge_cluster(rows: list[Any]) -> dict[str, Any]:
        best = min(
            rows,
            key=lambda row: (
                float(row.p_value) if pd.notna(row.p_value) else float("inf")
            ),
        )
        return {
            "chromosome": best.chromosome,
            "start": min(row.start for row in rows),
            "end": max(row.end for row in rows),
            "tf": best.tf,
            "motif_name": best.motif_name,
            "p_value": best.p_value,
            "score": max(
                (
                    float(row.score)
                    for row in rows
                    if pd.notna(row.score)
                ),
                default=np.nan,
            ),
            "hit_count": len(rows),
        }

    def _record(self, row: Any) -> MotifRecord:
        rna = (
            self.rna.lookup_details(row.tf)
            if self.rna
            else {"logfc": None, "expression": None, "padj": None}
        )
        return MotifRecord(
            chromosome=str(row.chromosome),
            start=int(row.start),
            end=int(row.end),
            tf=str(row.tf),
            motif_name=str(row.motif_name),
            p_value=(
                float(row.p_value)
                if hasattr(row, "p_value") and pd.notna(row.p_value)
                else None
            ),
            score=(
                float(row.score)
                if hasattr(row, "score") and pd.notna(row.score)
                else None
            ),
            hit_count=int(getattr(row, "hit_count", 1)),
            rna_logfc=rna["logfc"],
            rna_expression=rna["expression"],
            rna_padj=rna["padj"],
        )

    def summary(self) -> dict[str, Any]:
        record_logfc = [
            abs(record.rna_logfc)
            for record in self.records
            if record.rna_logfc is not None
        ]
        record_expression = [
            record.rna_expression
            for record in self.records
            if record.rna_expression is not None
        ]
        rna_ranges = (
            {
                "abs_logfc_max": max(record_logfc) if record_logfc else None,
                "expression_min": (
                    min(record_expression) if record_expression else None
                ),
                "expression_max": (
                    max(record_expression) if record_expression else None
                ),
            }
            if self.rna
            else None
        )
        return {
            "motif_path": str(self.motif_path),
            "rna_path": str(self.rna.path) if self.rna else None,
            "rna_path_2": str(self.rna_2.path) if self.rna_2 else None,
            "gene_list_path": (
                str(self.gene_list_path) if self.gene_list_path else None
            ),
            "gene_list_count": len(self.gene_list),
            "region": (
                f"{self.chromosome}:{self.start}-{self.end}"
            ),
            "raw_motif_count": self.raw_count,
            "merged_motif_count": len(self.records),
            "available_tfs": sorted({record.tf for record in self.records}),
            "rna_columns": (
                self.rna.mapping.to_dict() if self.rna else None
            ),
            "rna_columns_2": (
                self.rna_2.mapping.to_dict() if self.rna_2 else None
            ),
            "rna_ranges": rna_ranges,
        }

    @staticmethod
    def _plot_track(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(result["nucleotide_positions"], dtype=np.int64)
        transformed = result.get("plot_effects")
        if transformed is None:
            head = result["settings"]["output_head"]
            transformed, _metadata = manuscript_plot_effects(
                np.asarray(result["nucleotide_effects"], dtype=np.float64),
                np.asarray(result["baseline"][head], dtype=np.float64),
            )
        effects = np.asarray(transformed, dtype=np.float64)
        if len(positions) != len(effects):
            raise ValueError("Deletion positions and transformed effects differ in length")
        return positions, effects

    def filter(
        self,
        result: dict[str, Any],
        *,
        mode: str = "rna",
        min_abs_logfc: float = 2.0,
        min_expression: float = 10.0,
        max_padj: float = 0.05,
        selected_tf: str | None = None,
        min_effect_pct: float = 10.0,
    ) -> dict[str, Any]:
        legacy_rna_mode = mode == "rna"
        if mode == "rna":
            mode = "differential"
        modes = {
            "expression",
            "differential",
            "consensus_de",
            "gene_list",
            "tf",
        }
        if mode not in modes:
            raise ValueError(
                "Annotation mode must be expression, differential, "
                "consensus_de, gene_list, or tf"
            )
        if mode in {"expression", "differential", "consensus_de"} and self.rna is None:
            raise ValueError(f"{mode} mode requires a primary RNA table")
        if mode == "consensus_de" and self.rna_2 is None:
            raise ValueError("Consensus differential mode requires two RNA tables")
        if mode == "gene_list" and not self.gene_list:
            raise ValueError("Gene-list mode requires a non-empty gene list")
        if (
            min_abs_logfc < 0
            or min_expression < 0
            or min_effect_pct < 0
            or max_padj < 0
        ):
            raise ValueError("Annotation thresholds cannot be negative")
        if mode == "tf" and not selected_tf:
            raise ValueError("Direct TF mode requires a selected TF")
        positions, effects = self._plot_track(result)
        selected_key = str(selected_tf or "").strip().upper()
        output: list[dict[str, Any]] = []
        for record in self.records:
            if mode == "tf":
                if record.tf.upper() != selected_key:
                    continue
            elif mode == "gene_list":
                if record.tf.upper() not in self.gene_list:
                    continue
            elif mode == "expression":
                if record.rna_expression is None:
                    continue
                if record.rna_expression <= min_expression:
                    continue
            elif mode == "differential":
                if record.rna_logfc is None or record.rna_expression is None:
                    continue
                if abs(record.rna_logfc) <= min_abs_logfc:
                    continue
                if record.rna_expression <= min_expression:
                    continue
                if (
                    not legacy_rna_mode
                    and (
                    record.rna_padj is None
                    or record.rna_padj >= max_padj
                    )
                ):
                    continue
            else:
                first = self.rna.lookup_details(record.tf)
                second = self.rna_2.lookup_details(record.tf)
                values = (first, second)
                if any(
                    item["logfc"] is None or item["expression"] is None
                    for item in values
                ):
                    continue
                if any(
                    abs(float(item["logfc"])) <= min_abs_logfc
                    or float(item["expression"]) <= min_expression
                    for item in values
                ):
                    continue
                if float(first["logfc"]) * float(second["logfc"]) <= 0:
                    continue
                adjusted = [
                    float(item["padj"])
                    for item in values
                    if item["padj"] is not None
                ]
                if not adjusted or min(adjusted) >= max_padj:
                    continue
            overlap = (positions >= record.start) & (positions < record.end)
            overlap_effect = (
                float(np.nanmax(effects[overlap]))
                if np.any(overlap)
                else 0.0
            )
            if overlap_effect < min_effect_pct:
                continue
            payload = record.to_dict()
            payload["overlap_effect_pct"] = overlap_effect
            output.append(payload)
        output.sort(
            key=lambda item: (
                item["tf"],
                item["start"],
                item["end"],
            )
        )
        groups = self._figure_groups(output)
        return {
            "mode": mode,
            "filters": {
                "min_abs_logfc": min_abs_logfc,
                "min_expression": min_expression,
                "max_padj": max_padj,
                "selected_tf": selected_tf,
                "min_effect_pct": min_effect_pct,
            },
            "motifs": output,
            "groups": groups,
            "group_count": len(groups),
            "motif_count": len(output),
            "tf_count": len({item["tf"] for item in output}),
        }

    @staticmethod
    def _figure_groups(motifs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cluster retained motif intervals and create manuscript-style labels."""
        if not motifs:
            return []
        ordered = sorted(motifs, key=lambda item: (item["start"], item["end"]))
        clusters: list[list[dict[str, Any]]] = []
        active: list[dict[str, Any]] = []
        active_end: int | None = None
        for motif in ordered:
            if active and active_end is not None and motif["start"] > active_end:
                clusters.append(active)
                active = []
                active_end = None
            active.append(motif)
            active_end = max(active_end or motif["end"], motif["end"])
        if active:
            clusters.append(active)

        output: list[dict[str, Any]] = []
        for cluster in clusters:
            ranked = sorted(
                cluster,
                key=lambda item: (
                    -float(item.get("overlap_effect_pct") or 0),
                    item["tf"],
                ),
            )
            full_labels: list[str] = []
            display_labels: list[str] = []
            for motif in ranked:
                tf = str(motif["tf"])
                if tf not in full_labels:
                    full_labels.append(tf)
                family = tf.rstrip("0123456789") or tf
                if family not in display_labels:
                    display_labels.append(family)
            output.append(
                {
                    "start": min(item["start"] for item in cluster),
                    "end": max(item["end"] for item in cluster),
                    "label": "/".join(display_labels),
                    "full_label": "/".join(full_labels),
                    "max_effect_pct": max(
                        float(item.get("overlap_effect_pct") or 0)
                        for item in cluster
                    ),
                    "motifs": cluster,
                }
            )
        return sorted(
            output,
            key=lambda item: (-item["max_effect_pct"], item["start"]),
        )


class MotifAnnotationService:
    """Own the currently loaded annotation catalog for the local app."""

    def __init__(self) -> None:
        self.catalog: MotifAnnotationCatalog | None = None

    def load(
        self,
        motif_path: str,
        *,
        chromosome: str,
        start: int,
        end: int,
        rna_path: str | None = None,
        rna_path_2: str | None = None,
        gene_list_path: str | None = None,
        rna_columns: dict[str, str] | None = None,
        merge_overlaps: bool = True,
    ) -> dict[str, Any]:
        candidate = MotifAnnotationCatalog(
            motif_path,
            chromosome=chromosome,
            start=start,
            end=end,
            rna_path=rna_path or None,
            rna_path_2=rna_path_2 or None,
            gene_list_path=gene_list_path or None,
            rna_columns=rna_columns,
            merge_overlaps=merge_overlaps,
        )
        self.catalog = candidate
        return candidate.summary()

    def filter(self, result: dict[str, Any], **filters: Any) -> dict[str, Any]:
        if self.catalog is None:
            raise RuntimeError("Load motif annotations before filtering")
        return self.catalog.filter(result, **filters)

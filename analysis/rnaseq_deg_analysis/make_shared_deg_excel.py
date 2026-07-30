#!/usr/bin/env python3
"""Generate Excel of shared DEGs between 1F and AF, with TF annotation."""

import csv
import os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

WORKDIR = os.path.dirname(os.path.abspath(__file__))
LFC_CUTOFF = 0.5
PVAL_CUTOFF = 0.05


def parse_float(val):
    val = val.strip()
    if val == "" or val.upper() == "NA":
        return None
    return float(val)


def load_degs(filepath):
    """Load DEGs as dict keyed by gene name."""
    degs = {}
    with open(filepath, "r") as f:
        for row in csv.DictReader(f):
            gene = row[""].strip('"')
            lfc = parse_float(row["log2FoldChange"])
            pval = parse_float(row["pvalue"])
            padj = parse_float(row["padj"])
            if any(v is None for v in [lfc, pval, padj]):
                continue
            if abs(lfc) > LFC_CUTOFF and pval < PVAL_CUTOFF:
                degs[gene] = {"log2FoldChange": lfc, "pvalue": pval, "padj": padj}
    return degs


def load_tfs(filepath):
    with open(filepath) as f:
        return set(line.strip() for line in f if line.strip())


def write_sheet(ws, genes, title_suffix=""):
    """Write gene data to a worksheet."""
    headers = [
        "Gene", "log2FC_1F", "pvalue_1F", "padj_1F",
        "log2FC_AF", "pvalue_AF", "padj_AF",
    ]
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for i, g in enumerate(genes, 2):
        ws.cell(row=i, column=1, value=g["gene"])
        ws.cell(row=i, column=2, value=round(g["lfc_1f"], 4))
        ws.cell(row=i, column=3, value=g["pval_1f"]).number_format = "0.00E+00"
        ws.cell(row=i, column=4, value=g["padj_1f"]).number_format = "0.00E+00"
        ws.cell(row=i, column=5, value=round(g["lfc_af"], 4))
        ws.cell(row=i, column=6, value=g["pval_af"]).number_format = "0.00E+00"
        ws.cell(row=i, column=7, value=g["padj_af"]).number_format = "0.00E+00"

    # Auto-width
    for col in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 14
    ws.column_dimensions["A"].width = 18


def main():
    degs_1f = load_degs(os.path.join(WORKDIR, "1F.deg.gene.csv"))
    degs_af = load_degs(os.path.join(WORKDIR, "AF.deg.gene.csv"))
    tfs = load_tfs(os.path.join(WORKDIR, "mouse_tfs.txt"))

    # Find shared DEGs
    shared_genes = set(degs_1f.keys()) & set(degs_af.keys())
    print(f"1F DEGs: {len(degs_1f)}")
    print(f"AF DEGs: {len(degs_af)}")
    print(f"Shared DEGs: {len(shared_genes)}")

    # Build shared DEG records — classify by direction in BOTH cohorts
    # "up" = upregulated in both, "down" = downregulated in both
    shared_up = []
    shared_down = []
    mixed = []

    for gene in sorted(shared_genes):
        rec = {
            "gene": gene,
            "lfc_1f": degs_1f[gene]["log2FoldChange"],
            "pval_1f": degs_1f[gene]["pvalue"],
            "padj_1f": degs_1f[gene]["padj"],
            "lfc_af": degs_af[gene]["log2FoldChange"],
            "pval_af": degs_af[gene]["pvalue"],
            "padj_af": degs_af[gene]["padj"],
            "is_tf": gene in tfs,
        }
        up_1f = rec["lfc_1f"] > 0
        up_af = rec["lfc_af"] > 0
        if up_1f and up_af:
            shared_up.append(rec)
        elif not up_1f and not up_af:
            shared_down.append(rec)
        else:
            mixed.append(rec)

    # Sort by average |LFC| descending
    shared_up.sort(key=lambda g: (abs(g["lfc_1f"]) + abs(g["lfc_af"])) / 2, reverse=True)
    shared_down.sort(key=lambda g: (abs(g["lfc_1f"]) + abs(g["lfc_af"])) / 2, reverse=True)

    tf_up = [g for g in shared_up if g["is_tf"]]
    tf_down = [g for g in shared_down if g["is_tf"]]

    print(f"  Up in both: {len(shared_up)} ({len(tf_up)} TFs)")
    print(f"  Down in both: {len(shared_down)} ({len(tf_down)} TFs)")
    print(f"  Mixed direction: {len(mixed)}")

    # Create Excel
    wb = Workbook()

    ws_up = wb.active
    ws_up.title = "Shared_Up"
    write_sheet(ws_up, shared_up)

    ws_down = wb.create_sheet("Shared_Down")
    write_sheet(ws_down, shared_down)

    ws_tf_up = wb.create_sheet("TF_Up")
    write_sheet(ws_tf_up, tf_up)

    ws_tf_down = wb.create_sheet("TF_Down")
    write_sheet(ws_tf_down, tf_down)

    outpath = os.path.join(WORKDIR, "shared_DEGs_1F_AF.xlsx")
    wb.save(outpath)
    print(f"\nExcel saved to: {outpath}")


if __name__ == "__main__":
    main()

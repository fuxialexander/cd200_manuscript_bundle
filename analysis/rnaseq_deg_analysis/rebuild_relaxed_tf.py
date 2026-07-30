#!/usr/bin/env python3
"""Rebuild TF_Up_relaxed and TF_Down_relaxed sheets with 1C stats included."""

import csv
import os
from openpyxl import load_workbook
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


def load_all_genes(filepath):
    genes = {}
    with open(filepath, "r") as f:
        for row in csv.DictReader(f):
            gene = row[""].strip('"')
            genes[gene] = {
                "log2FoldChange": parse_float(row["log2FoldChange"]),
                "pvalue": parse_float(row["pvalue"]),
                "padj": parse_float(row["padj"]),
            }
    return genes


def is_deg(g):
    lfc, pval, padj = g["log2FoldChange"], g["pvalue"], g["padj"]
    if any(v is None for v in [lfc, pval, padj]):
        return False
    return abs(lfc) > LFC_CUTOFF and pval < PVAL_CUTOFF


def load_tfs(filepath):
    with open(filepath) as f:
        return set(line.strip() for line in f if line.strip())


def write_csv(filepath, tf_up, tf_down):
    fieldnames = [
        "direction",
        "gene",
        "lfc_1f", "pval_1f", "padj_1f", "deg_1f",
        "lfc_af", "pval_af", "padj_af", "deg_af",
        "lfc_1c", "pval_1c", "padj_1c", "deg_1c",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for direction, records in [("up", tf_up), ("down", tf_down)]:
            for record in records:
                row = {"direction": direction, **record}
                writer.writerow(row)


def write_sheet(ws, records):
    headers = [
        "Gene",
        "log2FC_1F", "pvalue_1F", "padj_1F", "DEG_1F",
        "log2FC_AF", "pvalue_AF", "padj_AF", "DEG_AF",
        "log2FC_1C", "pvalue_1C", "padj_1C", "DEG_1C",
    ]
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for i, r in enumerate(records, 2):
        ws.cell(row=i, column=1, value=r["gene"])
        for j, cohort in enumerate(["1f", "af", "1c"]):
            base = 2 + j * 4
            lfc = r[f"lfc_{cohort}"]
            pval = r[f"pval_{cohort}"]
            padj = r[f"padj_{cohort}"]
            ws.cell(row=i, column=base, value=round(lfc, 4) if lfc is not None else "NA")
            c_pval = ws.cell(row=i, column=base + 1, value=pval if pval is not None else "NA")
            c_padj = ws.cell(row=i, column=base + 2, value=padj if padj is not None else "NA")
            if pval is not None:
                c_pval.number_format = "0.00E+00"
            if padj is not None:
                c_padj.number_format = "0.00E+00"
            ws.cell(row=i, column=base + 3, value=r[f"deg_{cohort}"])

    for col in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 14
    ws.column_dimensions["A"].width = 18


def main():
    all_1f = load_all_genes(os.path.join(WORKDIR, "1F.deg.gene.csv"))
    all_af = load_all_genes(os.path.join(WORKDIR, "AF.deg.gene.csv"))
    all_1c = load_all_genes(os.path.join(WORKDIR, "1C.deg.gene.csv"))
    tfs = load_tfs(os.path.join(WORKDIR, "mouse_tfs.txt"))

    common_genes = set(all_1f.keys()) & set(all_af.keys())

    tf_up = []
    tf_down = []

    for gene in sorted(common_genes):
        if gene not in tfs:
            continue

        g1, ga, gc = all_1f[gene], all_af[gene], all_1c.get(gene)
        lfc_1f = g1["log2FoldChange"]
        lfc_af = ga["log2FoldChange"]
        lfc_1c = gc["log2FoldChange"] if gc else None

        if lfc_1f is None or lfc_af is None:
            continue
        # Same trend in 1F and AF
        if (lfc_1f > 0) != (lfc_af > 0):
            continue
        # DEG in at least one of 1F/AF
        if not (is_deg(g1) or is_deg(ga)):
            continue
        # 1C must exist and have same trend
        if lfc_1c is None:
            continue
        if (lfc_1f > 0) != (lfc_1c > 0):
            continue

        rec = {
            "gene": gene,
            "lfc_1f": lfc_1f, "pval_1f": g1["pvalue"], "padj_1f": g1["padj"],
            "deg_1f": "Yes" if is_deg(g1) else "No",
            "lfc_af": lfc_af, "pval_af": ga["pvalue"], "padj_af": ga["padj"],
            "deg_af": "Yes" if is_deg(ga) else "No",
            "lfc_1c": lfc_1c,
            "pval_1c": gc["pvalue"] if gc else None,
            "padj_1c": gc["padj"] if gc else None,
            "deg_1c": "Yes" if gc and is_deg(gc) else "No",
        }

        if lfc_1f > 0:
            tf_up.append(rec)
        else:
            tf_down.append(rec)

    tf_up.sort(key=lambda g: (abs(g["lfc_1f"]) + abs(g["lfc_af"])) / 2, reverse=True)
    tf_down.sort(key=lambda g: (abs(g["lfc_1f"]) + abs(g["lfc_af"])) / 2, reverse=True)

    print(f"TF_Up_relaxed: {len(tf_up)}")
    print(f"TF_Down_relaxed: {len(tf_down)}")

    xlpath = os.path.join(WORKDIR, "shared_DEGs_1F_AF.xlsx")
    wb = load_workbook(xlpath)

    for name in ["TF_Up_relaxed", "TF_Down_relaxed"]:
        if name in wb.sheetnames:
            del wb[name]

    ws_up = wb.create_sheet("TF_Up_relaxed")
    write_sheet(ws_up, tf_up)

    ws_down = wb.create_sheet("TF_Down_relaxed")
    write_sheet(ws_down, tf_down)

    wb.save(xlpath)
    print(f"Excel updated: {xlpath}")

    csv_path = os.path.join(WORKDIR, "shared_DEGs_1F_AF_tf_relaxed.csv")
    write_csv(csv_path, tf_up, tf_down)
    print(f"CSV updated: {csv_path}")


if __name__ == "__main__":
    main()

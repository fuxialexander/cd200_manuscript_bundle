# GM12878 performance in the original ChromBPNet paper

**Yes: the paper reports GM12878-specific count and profile performance in Extended Figure 2b–c.** The GM12878 ATAC count box is visually around **0.63**, roughly **0.60–0.65** across the displayed folds. Our independently recomputed count Pearson **0.6330755** on 62,929 test peaks is visually consistent with that panel. The figure neither labels each fold nor prints exact values; it does not establish an exact published fold-0 number. The visual range is not a confidence interval.

## Primary figure and source tables

- [Original publication, Extended Figure 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/#F9), corresponding to figure element `F9` in the downloaded `data/chrombpnet_publication.xml`.
- Original figure: [nihpp-2024.12.25.630221v2-f0009.jpg](data/publication_sources/nihpp-2024.12.25.630221v2-f0009.jpg), 763 × 195 pixels. Panel b separates ATAC-seq and DNase-seq, showing separate boxes for each cell line; panel c reports profile distance. Individual fold labels are absent.
- Original supplementary workbook: [media-1.xlsx](data/publication_sources/media-1.xlsx). Tables 1–5 are configurations, folds and additional study tables; no fold-specific numerical performance table was found.
- Source and integrity manifest: [media_manifest.json](data/publication_sources/media_manifest.json). Both selected members were extracted from a complete, valid 40-member Europe PMC ZIP and checked against their ZIP CRCs. Download source: [Europe PMC supplementary files](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11741299/supplementaryFiles). The complete archive remains in `/tmp/cd200_gm12878_benchmark/chrombpnet_publication_media_complete.zip`; no partial archive was deleted.

The small raster figure supports an approximate reading, not precise digitization. The original Results text gives K562 ATAC count correlation approximately 0.70 and a pooled median 0.69 for other contexts. That pooled median is not a GM12878 ATAC-specific expected value and cannot be used to call the local 0.633 a failed reproduction or an anomalously weak fold.

## Verified dataset and fold provenance

The workbook's **Table 1** records the following original ATAC-seq settings:

| Context | Experiment | Depth | Bias method and threshold | Bias annotation | Full-model annotation |
| --- | --- | --- | --- | --- | --- |
| GM12878 | ENCSR637XSC | 572M | In-vivo, 0.4 | ENCSR637GXZ | ENCSR389HIH |
| K562 | ENCSR868FGK | 546M | In-vivo, 0.5 | ENCSR748YRV | ENCSR467RSV |

Both rows describe paired-end data. This is additional source-table verification of the lineage already established from ENCODE metadata, not a new selection of data. The GM12878 DNase-seq row is a separate experiment, ENCSR000EMT at 68M reads, and must not be substituted for the ATAC result.

**Table 2** specifies fold 0 test chr1/chr3/chr6 and validation chr8/chr20. The local extraction is [Table_2.tsv](data/publication_sources/Table_2.tsv); its full original training chromosome list matches the benchmark's canonical fold 0. [Table_1.tsv](data/publication_sources/Table_1.tsv) preserves the configuration rows.

## What was and was not numerically verified

| Claim | Status and evidence |
| --- | --- |
| The original paper includes GM12878-specific results | **Verified** in Extended Figure 2b–c. |
| Its GM12878 ATAC count box is near 0.63 | **Approximate visual reading**, around 0.60–0.65; no exact extracted points. |
| Our peak-only count Pearson is 0.6330755 | **Independently verified local result**; see `logs/count_metric_verification.json` and `PUBLICATION_RECONCILIATION.md`. |
| Our value is visually consistent with the published GM12878 ATAC box | **Supported comparison**, without identifying which plotted fold equals which local checkpoint. |
| The paper reports exactly 0.6330755 for GM12878 fold 0 | **Not established**. No exact fold-labeled source value found. |
| GM12878 fold 0 is unusually poor | **Not established**, and unnecessary to explain the plot. |
| The local model reconstruction faithfully loads released weights | **Verified** independently against the original full H5 graph on eight random sequences, with zero numerical discrepancy; this tests loading, not publication-point identity. |

The authors' public [figure-source repository](https://github.com/kundajelab/chrombpnet-figures) was inspected for an exact Extended Figure 2 source table. Retrieved notebooks in `data/publication_sources/` include examples of K562 count evaluation and GM12878 **bias-model** diagnostics. Their diagnostic values must not be relabeled as full ChromBPNet GM12878 count performance. A fold-labeled numerical source table for the cross-cell-line boxplot was not located there.

Profile metrics need separate reconciliation: the paper's Methods 1.10 describes median distance and base-2 logs, while the current local summary uses mean native SciPy distance with natural logs. The GM12878-specific profile box should not be equated numerically to the local mean without matching log base, aggregation and region population.

## Consequence for reporting

Use: “The published GM12878 ATAC fold-0 checkpoint achieved peak-only count Pearson r = 0.6331 in our evaluation, consistent with the GM12878 ATAC distribution displayed in the original paper's Extended Figure 2b.”

Do not write that 0.6331 is an exact value quoted from the paper, that the paper omitted GM12878, or that the difference from a pooled 0.69 proves a poor fold. K562 remains a separate requested biological context, rather than a replacement chosen to avoid the GM12878 result. No training setting should be tuned against either context's held-out test metrics.

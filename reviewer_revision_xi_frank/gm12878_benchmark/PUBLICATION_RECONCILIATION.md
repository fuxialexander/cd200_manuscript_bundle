# Published results versus this benchmark

Status checked 27 October 2026. The original paper **does report GM12878-specific performance** in Extended Figure 2b–c. Its GM12878 ATAC count box is visually centered around **0.63**, with a rough across-fold range of **0.60–0.65**. The independently verified local value **0.6330755** is visually consistent with that plot. The figure does not label folds or print their numerical values, so an exact match to the paper's fold-0 point has **not** been established. There is no evidence here that fold 0 is unusually weak.

## Primary paper

Source: [ChromBPNet paper, PMC11741299](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/). Full XML retrieved from the [Europe PMC primary full-text endpoint](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11741299/fullTextXML), saved as `data/chrombpnet_publication.xml`.

- The first Results section reports held-out total-count **Pearson r = 0.70 ± 0.02 for K562 ATAC-seq**, and 0.71 ± 0.02 for K562 DNase-seq, initially discussing expanded BPNet models.
- The bias-factorized model Results report a **median r = 0.69** for count prediction across the other four cell lines. Extended Figure 2b shows variation across assays, cell lines and folds.
- Methods section 1.10 defines total-signal evaluation as Pearson correlation between predicted and observed log counts. This agrees with the type of count metric computed locally.

The original [Extended Figure 2b–c](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/#F9) has now been retrieved and inspected. The count panel separates GM12878 ATAC-seq from GM12878 DNase-seq, the other cell lines and their folds. Its ATAC box sits around 0.63; the prose's 0.69 is a pooled summary across other contexts, not a GM12878 ATAC-specific target. The rough 0.60–0.65 range is a visual reading of a small raster plot, not source-data extraction or a confidence interval.

The figure is preserved as `data/publication_sources/nihpp-2024.12.25.630221v2-f0009.jpg`, with provenance and SHA256 in `data/publication_sources/media_manifest.json`. The original `media-1.xlsx` workbook's Table 1 confirms GM12878 ATAC experiment ENCSR637XSC, 572M reads, in-vivo bias threshold 0.4, bias annotation ENCSR637GXZ and full-model annotation ENCSR389HIH. Table 2 confirms the chromosome folds. The workbook has five tables covering configurations, splits and biological/variant/motif resources; no exact GM12878 fold-by-fold performance values were found. The public model archive contains training logs but no released evaluation metric table that establishes an exact plotted point. See [GM12878_PUBLICATION_DETAIL.md](GM12878_PUBLICATION_DETAIL.md).

## Independently verified local evaluation

| Population | Windows | Published fold-0 model count Pearson |
| --- | ---: | ---: |
| Peaks only, test chr1/chr3/chr6 | 62,929 | 0.6330755 |
| Peaks and background, same test chromosomes | 314,645 | 0.7113881 |

Targets are `ln(1 + sum of observed ENCFF180XQC ATAC insertions in the central 1000 bp))`. Predictions come from the full, bias-factorized released fold-0 model, with component equivalence verified against the original full H5 graph. Region ordering and checksums are recorded in `logs/count_metric_verification.json` and the prediction metadata. No QTL labels or allele-effect predictions enter these numbers.

Including background changes the estimand and cannot be used to manufacture agreement with the paper's peak-only benchmark. A cross-cell-line or cross-fold summary is also not an exact fold-0 reference.

## Interpretation and remaining checks

The local count result is reproducible on its stated regions and **visually agrees with the GM12878 ATAC panel**. Exact fold-specific numerical reproduction remains unverified because the source point is unlabeled and its underlying value was not located. Recover those source values and match the exact checkpoint/bias choice, peak population, preprocessing, reference and aggregation before claiming exact numerical reproduction. Do not characterize the local value as a discrepancy from 0.69 or attribute it to an unfavorable fold. The separately retrained ChromBPNet comparator remains important for the matched-training comparison. No training settings were changed in response to these test metrics.

Profile reporting also needs care: this benchmark currently reports mean Jensen–Shannon distance using natural logs to match the downloaded native metric implementation. The paper discusses median JSD and describes base-2 logs in Methods 1.10. Those summary values should not be compared as identical numerical metrics without conversion and matching aggregation.

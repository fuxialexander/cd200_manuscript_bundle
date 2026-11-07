# GM12878 held-out prediction results

GM12878 ATAC-seq ENCSR637XSC, published ChromBPNet fold 0. Test chromosomes: chr1, chr3, chr6; validation: chr8, chr20. All eligible released test regions are retained. Estimates below are conditional on one fitted model per method; they do not measure training-seed variability.

| Population | Model | Metric | Estimate | 95% block CI |
| --- | --- | --- | ---: | --- |
| all | ChromBPNet_published | count_pearson_ln1p | 0.7114 | [0.6885, 0.7348] |
| all | ChromBPNet_published | count_rmse_ln1p | 1.6418 | [1.5812, 1.7079] |
| all | ChromBPNet_published | count_spearman_ln1p | 0.6372 | not computed |
| all | ChromBPNet_published | profile_js_distance | 0.5582 | [0.5539, 0.5625] |
| all | ChromBPNet_published | profile_pearson_1bp | 0.5610 | [0.5546, 0.5673] |
| all | ChromBPNet_published | profile_pearson_20bp | 0.6234 | [0.6152, 0.6309] |
| all | ExplaiNN | count_pearson_ln1p | 0.6155 | [0.5940, 0.6363] |
| all | ExplaiNN | count_rmse_ln1p | 1.8087 | [1.7496, 1.8731] |
| all | ExplaiNN | count_spearman_ln1p | 0.5708 | not computed |
| all | MoDL | count_pearson_ln1p | 0.7007 | [0.6790, 0.7221] |
| all | MoDL | count_rmse_ln1p | 1.4018 | [1.3389, 1.4711] |
| all | MoDL | count_spearman_ln1p | 0.6356 | not computed |
| all | MoDL | profile_js_distance | 0.5566 | [0.5522, 0.5610] |
| all | MoDL | profile_pearson_1bp | 0.5585 | [0.5521, 0.5651] |
| all | MoDL | profile_pearson_20bp | 0.6134 | [0.6054, 0.6208] |
| peaks | ChromBPNet_published | count_pearson_ln1p | 0.6331 | [0.6167, 0.6476] |
| peaks | ChromBPNet_published | count_rmse_ln1p | 0.8456 | [0.8295, 0.8617] |
| peaks | ChromBPNet_published | count_spearman_ln1p | 0.5714 | not computed |
| peaks | ChromBPNet_published | profile_js_distance | 0.3267 | [0.3242, 0.3296] |
| peaks | ChromBPNet_published | profile_pearson_1bp | 0.8504 | [0.8490, 0.8517] |
| peaks | ChromBPNet_published | profile_pearson_20bp | 0.8645 | [0.8616, 0.8675] |
| peaks | ExplaiNN | count_pearson_ln1p | 0.6006 | [0.5853, 0.6150] |
| peaks | ExplaiNN | count_rmse_ln1p | 0.8715 | [0.8570, 0.8859] |
| peaks | ExplaiNN | count_spearman_ln1p | 0.5346 | not computed |
| peaks | MoDL | count_pearson_ln1p | 0.6796 | [0.6638, 0.6942] |
| peaks | MoDL | count_rmse_ln1p | 0.8027 | [0.7851, 0.8214] |
| peaks | MoDL | count_spearman_ln1p | 0.6293 | not computed |
| peaks | MoDL | profile_js_distance | 0.3185 | [0.3159, 0.3214] |
| peaks | MoDL | profile_pearson_1bp | 0.8633 | [0.8618, 0.8647] |
| peaks | MoDL | profile_pearson_20bp | 0.8786 | [0.8760, 0.8811] |
| nonpeaks | ChromBPNet_published | count_pearson_ln1p | 0.3828 | [0.3422, 0.4279] |
| nonpeaks | ChromBPNet_published | count_rmse_ln1p | 1.7862 | [1.7194, 1.8582] |
| nonpeaks | ChromBPNet_published | count_spearman_ln1p | 0.4283 | not computed |
| nonpeaks | ChromBPNet_published | profile_js_distance | 0.6170 | [0.6147, 0.6194] |
| nonpeaks | ChromBPNet_published | profile_pearson_1bp | 0.4875 | [0.4825, 0.4921] |
| nonpeaks | ChromBPNet_published | profile_pearson_20bp | 0.5622 | [0.5541, 0.5689] |
| nonpeaks | ExplaiNN | count_pearson_ln1p | 0.3278 | [0.2965, 0.3612] |
| nonpeaks | ExplaiNN | count_rmse_ln1p | 1.9747 | [1.9098, 2.0435] |
| nonpeaks | ExplaiNN | count_spearman_ln1p | 0.4117 | not computed |
| nonpeaks | MoDL | count_pearson_ln1p | 0.2804 | [0.2497, 0.3120] |
| nonpeaks | MoDL | count_rmse_ln1p | 1.5150 | [1.4448, 1.5920] |
| nonpeaks | MoDL | count_spearman_ln1p | 0.3950 | not computed |
| nonpeaks | MoDL | profile_js_distance | 0.6171 | [0.6147, 0.6195] |
| nonpeaks | MoDL | profile_pearson_1bp | 0.4811 | [0.4762, 0.4856] |
| nonpeaks | MoDL | profile_pearson_20bp | 0.5461 | [0.5384, 0.5528] |

ExplaiNN uses its native scalar output and has no profile scores. MoDL uses its unchanged fixed-motif architecture with a declared 1000-bp crop and common raw-insertion training objectives. The published ChromBPNet checkpoint is a reference; the separately retrained native ChromBPNet is the matched-training comparator when present. See evaluation.json for exact metric definitions, undefined-profile coverage and paired model-difference confidence intervals.

This external human-data benchmark does not replace leakage-free retraining on the manuscript's mouse data. No method was selected or tuned using these test metrics.

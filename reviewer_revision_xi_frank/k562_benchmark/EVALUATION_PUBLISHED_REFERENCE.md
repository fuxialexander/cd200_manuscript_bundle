# K562 held-out prediction results

K562 ATAC-seq ENCSR868FGK, published ChromBPNet fold 0. Test chromosomes: chr1, chr3, chr6; validation: chr8, chr20. All eligible released test regions are retained. Estimates below are conditional on one fitted model per method; they do not measure training-seed variability.

| Population | Model | Metric | Estimate | 95% block CI |
| --- | --- | --- | ---: | --- |
| all | ChromBPNet_published | count_pearson_ln1p | 0.6881 | [0.6734, 0.7033] |
| all | ChromBPNet_published | count_rmse_ln1p | 1.5166 | [1.4631, 1.5719] |
| all | ChromBPNet_published | count_spearman_ln1p | 0.6037 | not computed |
| all | ChromBPNet_published | profile_js_distance | 0.5465 | [0.5398, 0.5532] |
| all | ChromBPNet_published | profile_pearson_1bp | 0.5767 | [0.5676, 0.5856] |
| all | ChromBPNet_published | profile_pearson_20bp | 0.6417 | [0.6336, 0.6495] |
| all | ExplaiNN | count_pearson_ln1p | 0.6180 | [0.6016, 0.6347] |
| all | ExplaiNN | count_rmse_ln1p | 1.6819 | [1.6264, 1.7383] |
| all | ExplaiNN | count_spearman_ln1p | 0.5634 | not computed |
| all | MoDL | count_pearson_ln1p | 0.6407 | [0.6265, 0.6556] |
| all | MoDL | count_rmse_ln1p | 1.5349 | [1.4794, 1.5920] |
| all | MoDL | count_spearman_ln1p | 0.5652 | not computed |
| all | MoDL | profile_js_distance | 0.5444 | [0.5376, 0.5512] |
| all | MoDL | profile_pearson_1bp | 0.5841 | [0.5746, 0.5932] |
| all | MoDL | profile_pearson_20bp | 0.6491 | [0.6409, 0.6572] |
| peaks | ChromBPNet_published | count_pearson_ln1p | 0.6995 | [0.6878, 0.7105] |
| peaks | ChromBPNet_published | count_rmse_ln1p | 0.8019 | [0.7846, 0.8198] |
| peaks | ChromBPNet_published | count_spearman_ln1p | 0.6048 | not computed |
| peaks | ChromBPNet_published | profile_js_distance | 0.3378 | [0.3351, 0.3401] |
| peaks | ChromBPNet_published | profile_pearson_1bp | 0.8339 | [0.8326, 0.8352] |
| peaks | ChromBPNet_published | profile_pearson_20bp | 0.8473 | [0.8450, 0.8495] |
| peaks | ExplaiNN | count_pearson_ln1p | 0.6815 | [0.6700, 0.6926] |
| peaks | ExplaiNN | count_rmse_ln1p | 0.7979 | [0.7829, 0.8128] |
| peaks | ExplaiNN | count_spearman_ln1p | 0.5684 | not computed |
| peaks | MoDL | count_pearson_ln1p | 0.7286 | [0.7170, 0.7400] |
| peaks | MoDL | count_rmse_ln1p | 0.7482 | [0.7293, 0.7689] |
| peaks | MoDL | count_spearman_ln1p | 0.6371 | not computed |
| peaks | MoDL | profile_js_distance | 0.3326 | [0.3299, 0.3351] |
| peaks | MoDL | profile_pearson_1bp | 0.8460 | [0.8446, 0.8474] |
| peaks | MoDL | profile_pearson_20bp | 0.8599 | [0.8576, 0.8621] |
| nonpeaks | ChromBPNet_published | count_pearson_ln1p | 0.4568 | [0.4282, 0.4853] |
| nonpeaks | ChromBPNet_published | count_rmse_ln1p | 1.6475 | [1.5898, 1.7088] |
| nonpeaks | ChromBPNet_published | count_spearman_ln1p | 0.4268 | not computed |
| nonpeaks | ChromBPNet_published | profile_js_distance | 0.5996 | [0.5954, 0.6042] |
| nonpeaks | ChromBPNet_published | profile_pearson_1bp | 0.5112 | [0.5040, 0.5180] |
| nonpeaks | ChromBPNet_published | profile_pearson_20bp | 0.5893 | [0.5823, 0.5961] |
| nonpeaks | ExplaiNN | count_pearson_ln1p | 0.3445 | [0.3197, 0.3685] |
| nonpeaks | ExplaiNN | count_rmse_ln1p | 1.8376 | [1.7806, 1.8979] |
| nonpeaks | ExplaiNN | count_spearman_ln1p | 0.3992 | not computed |
| nonpeaks | MoDL | count_pearson_ln1p | 0.2739 | [0.2557, 0.2916] |
| nonpeaks | MoDL | count_rmse_ln1p | 1.6748 | [1.6173, 1.7363] |
| nonpeaks | MoDL | count_spearman_ln1p | 0.3333 | not computed |
| nonpeaks | MoDL | profile_js_distance | 0.5983 | [0.5940, 0.6029] |
| nonpeaks | MoDL | profile_pearson_1bp | 0.5174 | [0.5100, 0.5243] |
| nonpeaks | MoDL | profile_pearson_20bp | 0.5954 | [0.5880, 0.6023] |

ExplaiNN uses its native scalar output and has no profile scores. MoDL uses its unchanged fixed-motif architecture with a declared 1000-bp crop and common raw-insertion training objectives. The published ChromBPNet checkpoint is a reference; the separately retrained native ChromBPNet is the matched-training comparator when present. See evaluation.json for exact metric definitions, undefined-profile coverage and paired model-difference confidence intervals.

This external human-data benchmark does not replace leakage-free retraining on the manuscript's mouse data. No method was selected or tuned using these test metrics.

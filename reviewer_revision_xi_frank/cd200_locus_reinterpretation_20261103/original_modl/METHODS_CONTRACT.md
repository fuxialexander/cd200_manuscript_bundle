# Original MoDL locus inference and S5 reconstruction

Use **`results_core_reproduction/`** as the final export. `results/` is retained as an initial diagnostic that used the wrong full-window maximum for deletion checks; its reference profiles are valid, but its P1 deletion checks do not follow the original core-region definition. No original source, checkpoint or manuscript result was modified. All inference ran on CPU with four threads; no GPU was used.

## Files for the revised figure

- `results_core_reproduction/matched1000_windows.tsv` preserves the existing seven Cd200 window IDs and their order.
- `matched1000_profiles.npz` contains `native4096_profile`, `adapted2114_profile`, `native4096_shape_probability`, `adapted2114_shape_probability`, `observed_stored_counts`, and `observed_overlay_conv20`, each shaped `(7,1000)`. `native4096_count_head_scalar` and `adapted2114_count_head_scalar` are seven native scalar outputs and must not be interpreted as shared1000bp raw-count predictions.
- `native4096_{P1,P2,promoter}.npz` preserves full4096bp positions, original regression predictions, observed raw stored signal, the reconstructed original smoothed overlay, and the native scalar output.
- `phase3_8192_{P1,P2,promoter}.npz` preserves the original S5B8192bp-context differential inference, including its regression/count outputs.
- `native_diff_normalization_audit.json` and `native_diff_recovered_contrasts.tsv` recover the **actual original differential transform**. These resolve the provisional final-BED-denominator mismatch described in the earlier export manifest. Historical row-sum versus current signal differences remain separately identified.
- `archived_deletion_tracks/` holds copies of the canonical epoch105 regression/count tracks and original phase3 promoter differential track. Their sources and SHA-256 hashes are in `archived_deletion_tracks.tsv`. `deletion_validation.tsv` records selected fresh checks rather than a complete new deletion scan.

For an S5A profile comparison with new ChromBPNet, retain original MoDL's4096bp context and use the corresponding1000bp output crop. The2114bp-context output is a separate adaptation and can change the prediction because GroupNorm/global pooling use the input window. The original observed overlay is a20bp mean convolution. If smoothing ChromBPNet predictions within its1000bp output, scoring the central980bp avoids the10bp boundary where the surrounding output is unavailable. No native scalar count loss is interchangeable across these outputs.

## Models and native preprocessing

`export_original_modl.py` imports the original sibling source in `/home/xf2217/Repos/motif_chip_model/`. It reconstructs the exact architecture from saved tensor dimensions and loads weights with `strict=True`. The phase3 main, bias and differential components are checked against the saved assembled full state. Regression uses the shipped epoch105 checkpoint; differential prediction uses `ATAC-Sequential-20250919-022148_phase3_final_model.ckpt`.

The epoch105 model has3222 frozen31bp motif kernels, a same-length dilated residual trunk with GroupNorm, a positive profile head and a scalar count head. Original deletion scripts use4096bp centered input (`analysis/scripts/run_cd200_mutation.sh:76`). Original S5B differential inference uses8192bp centered input (`analysis/plot_cd200_profile_corr_and_diff_logfc.py:143`). The phase3 model predicts one negative-state profile/count plus a separately learned positive-minus-negative differential scalar; it does not expose independent positive and negative profile heads.

The original overlay query uses `normalize_factor=1e8, conv_size=20` (`src/mutation_analysis/dataset.py:302`). In the inspected Caesarion `FactorNormalizer`, division by the supplied factor and multiplication by1e8 makes that specific factor an identity transform (`Caesarion/src/caesar/io/normalizer.py:92`). Reconstructed20bp mean-convolved stored counts reproduce every archived overlay value to maximum absolute error below`1e-7`. This establishes the **displayed original target**, not all historical training-time preprocessing. The modern training loader's library-normalized target should not be substituted for this verified overlay.

Fresh native4096 profiles have correlation greater than`0.99999986` with archived profiles, mean absolute differences`7.0e-5–1.05e-4`, and maximum absolute differences`0.000895–0.001409` across the three loci. These are close numerical reproductions, not bitwise equality; the difference has not been assigned to a particular backend.

## Original deletion effect scope

The `region` CLI defaults `core_region` to the annotated interval (`analysis/mutation_analysis_cli.py:520`). A true15bp deletion shortens the4096bp input to4081bp. The primary regression effect is the difference between maxima **within the annotated core**, using the same sequence-index slice for reference and deletion predictions (`src/mutation_analysis/metrics.py:124`). It is not the difference between maxima over the full4096bp context. Each deletion contributes one fifteenth of its score to every deleted nucleotide; the supplied tracks use1bp stride and smoothing window1.

Scalar count deletion effects are `log2((alt_native_output+1)/(ref_native_output+1))`; differential effects are `alt_native_diff-ref_native_diff`. These are different targets/sign conventions. The plotting conversion takes absolute effects and divides by the maximum absolute baseline prediction before multiplying by100; it must not be called a signed DeepSHAP or DeepLIFT contribution.

Nine selected regression checks, including the P1 TEAD motif at`chr16:45384064–45384077`, agree with archived scores to maximum absolute difference`0.000723`. Nine scalar count checks agree to`0.0000333`; three phase3 promoter differential checks agree to`0.0000177`. Selected positions include locus midpoints, fixed interior checks and maximum archived effects; they are reproducibility checks, not an unbiased evaluation set. The initial full-window probe exposed the core-scope distinction at P1 and is retained for that reason.

## Recovered S5B units

The original curation script computes `logSumValue_scca_pos`, `logSumValue_scca_neg` and `Diff_scca` before later signal filtering and sequence deduplication (`/home/xf2217/Repos/motif_chip_model/atac_prediction/cd200_plot.py:60`,`:62`,`:69`,`:89`). The retained log columns encode the original denominators exactly:

```text
Lpos = 97,975,505
Lneg = 110,158,866
Diff_scca = log10(1 + pos_sum/Lpos*1e5)
          - log10(1 + neg_sum/Lneg*1e5)
```

Each state has95,366 positive rows independently encoding its denominator through `SumValue*1e5/expm1(logSumValue*ln(10))`. Maximum inverse-denominator residuals are about`1.34e-6` and`2.52e-6`. Reconstructed differential values agree with all184,167 retained labels to maximum absolute residual below`1e-16`. This algebraic recovery fits no calibration and uses no new model predictions.

Summing the **final** BED instead gives57,847,436 and65,910,183; those are not the original native denominators. The historical S5B script reads stored `Diff_scca` labels for P1/promoter but falls back to a raw-signal formula using final-BED sums for P2, which has no overlapping training row. Therefore the historical observed bars themselves mix normalizers. Its phase3 output is additionally multiplied by the legacy factor`3.307237044473427`, estimated from2123 selected positive-differential peaks. The export retains unscaled predictions as primary and keeps that legacy display only as a labelled historical variant.

Using recovered constants permits new CBP positive/negative predicted sums and current observed sums over the **same central2048bp interval** to share the original mathematical transform. Retain the historical BED labels separately: at P1, for example, historical row sums343/186 reproduce its label`0.0626042`, whereas the current Zarr over that representative interval contains515/272. Recovering the transform does not make those different underlying observations identical. Current-data fit and historical-label fit answer different questions, and neither warrants fitting a new scale on these selected loci.

## The apparently missing Peak1 bar

`analysis/results/cd200_deletion_vs_chrombpnet_original/diff_scca_logfc_summary.csv` records a finite Peak1/Enhancer1 ChromBPNet contrast of`0.0004166613`, with positive/negative sums142.1759/161.2049. It is very small at the displayed scale, but it is not exactly zero. The corresponding profile correlation is`0.8375925`; the no-bias variant records a negative differential contrast`-0.0234315` and profile correlation`0.8559502`. The existing track audit finds890 finite positive values and9 omitted bases over the899bp Peak1 interval in both state tracks. These facts do not support an all-zero-prediction explanation or explain an absent correlation bar. Match the precise submitted panel/version before asserting the cause.

## Reproduction and provenance

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/cd200-original-modl-mpl \
  /home/xf2217/Repos/motif_chip_model/.venv/bin/python \
  reviewer_revision_xi_frank/cd200_locus_reinterpretation_20261103/original_modl/export_original_modl.py \
  --output NEW_EXPORT_DIRECTORY

/home/xf2217/Repos/motif_chip_model/.venv/bin/python \
  reviewer_revision_xi_frank/cd200_locus_reinterpretation_20261103/original_modl/recover_native_diff_normalization.py \
  --original-exports NEW_EXPORT_DIRECTORY
```

Each command refuses to overwrite its results. The export manifest records checkpoint/source/output hashes and every Zarr chunk read. The normalization audit separately records the curation-source hash, retained-BED hash and recovered-contrast table hash. The historical split concerns remain applicable to the original checkpoints; these selected-locus reproductions do not constitute new heldout validation.

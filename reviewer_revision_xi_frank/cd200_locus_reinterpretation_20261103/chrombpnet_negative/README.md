# Negative-state ChromBPNet fit

This run trains a separate native ChromBPNet on `SCCAneg.cd200_scc`, using the same frozen regions, DNA, chromosome partition and model geometry as the completed positive-state mouse benchmark. Its purpose is to support a matched positive-versus-negative prediction contrast at the Cd200 loci.

Live state and exact commands: `job_status.json`. Persistent artifacts: `/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16_negative_20261103/`. The supervisor is the user systemd service `cd200-negative-chrombpnet-20261103.service`; model training continues independently of this tool session.

The new cache retains 143,439 training, 7,965 validation, 26,350 test and seven locus windows. DNA arrays are symlinks to checksum-verified positive-cache arrays; region tables are exact copies. All negative target arrays are newly extracted from the negative sample's unsmoothed uint16 Zarr tracks, stored as float32 without normalization or smoothing. No signal-dependent filter changes the common region universe. Thirty-seven checks compare cached rows with the negative source at their declared coordinates. Source chunks, sample metadata, array/region checksums and inherited DNA eligibility are recorded in `cache_manifest.json`.

Test chromosomes are chr16/chr8/chr9; validation is chr14. Native bias fitting starts from random initialization and uses only negative-state training nonpeaks for learned parameters and intercept calibration. Validation selects checkpoints. The full model uses 512 filters, eight dilation layers, 2,114-bp inputs, central 1,000-bp outputs, seed 1234, batch size 64, negative ratio 0.1, maximum 50 epochs and patience five, matching the positive benchmark. The full-model count-loss weight is estimated from negative training peaks as `max(1, median central1000 counts / 10) = 37.5`; the corresponding native bias-stage weight is 3.8. These values are not copied from the positive model.

The queue trains the bias model, trains the full model, predicts held-out and seven locus windows, and runs the existing common evaluations. Completion writes `RESULTS.json` with checkpoint hashes and result paths. Source snapshots and unchanged-source checks are retained. Training runs are fresh; incomplete outputs are preserved rather than overwritten.

Counts remain in stored ATAC-count units. Original fragment-endpoint/Tn5-shift provenance and motif purity of the fitted bias model are not independently established. Per-state predictions are not depth-normalized here; positive-versus-negative calibration and interpretation are separate analyses. One seed and one chromosome partition do not quantify seed-to-seed reproducibility.


## Completed run

Both the fitting/evaluation supervisor and the postcontrast supervisor finished successfully. The negative-state bias network completed 15 epochs with best epoch 9; the full model completed 20 epochs with best epoch 14, using validation-only selection. The final full-checkpoint SHA256 is `883a642b90a10e365bf2cc1ea28176adac569e99e72805677eb3321a15a74916`.

Held-out negative-state count Pearson correlation is 0.684654 on 13,593 peaks and 0.826971 on all 26,350 regions; peak profile JS distance is 0.493599. Full per-scope metrics and 2,000 genomic-block intervals are in the persistent `evaluation_final/evaluation.json`. The seven-window and full-interval locus metrics are in `locus_evaluation_final/locus_evaluation.json`.

The authorized nine-window, 2,048-bp negative-state prediction and regenerated state-contrast figure also completed. Outputs are `../negative_contrast/` and `../figures/regenerated_s5b_contrast.{pdf,svg,png}`. Exact numerical contrasts are registered in `../figures/state_contrast_source_data.tsv` with raw observed/predicted sums. The separate root workflow applies the recovered historical pre-filter state normalizers 97,975,505 and 110,158,866. These are distinct from sums of the final filtered BED or the Zarr library-size metadata. Historical curated labels and original MoDL outputs remain separate in the source table; no empirical original-MoDL display multiplier is applied to the new panel.

All source snapshots were unchanged across the run, three final checkpoint hashes were independently rechecked, all queued commands exited zero, and the regenerated PNG was visually inspected. Original working figures and previous benchmark caches/checkpoints were preserved. New models, negative targets, logs and source snapshots are retained at the paths above; no cleanup deletion or external publication was performed. `delivery_manifest.json` registers the completed outputs.

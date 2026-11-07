# Mouse chromosome-held-out benchmark

SCCApos; test chr16/chr8/chr9, validation chr14. One fold and seed. Targets: unsmoothed_per_base_ATAC_counts_from_bundled_Zarr. Upstream insertion/Tn5-shift provenance remains unverified.

| Model | Peak count Pearson | All-region count Pearson | Peak profile JS distance |
| --- | ---: | ---: | ---: |
| MoDL | 0.7082 | 0.8289 | 0.49745760206229506 |
| ChromBPNet | 0.6986 | 0.8316 | 0.5011767091820607 |
| ExplaiNN | 0.5338 | 0.7039 | N/A |

Full metrics and paired block intervals: /mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/evaluation_final/
Common profile/count losses: /mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/losses_final/
P1/P2/promoter native-window and full-interval metrics: /mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/locus_evaluation_final/
Original human benchmark files and original manuscript checkpoints were not changed.

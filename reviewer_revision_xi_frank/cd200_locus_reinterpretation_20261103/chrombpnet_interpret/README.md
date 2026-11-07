# Mouse Cd200: ChromBPNet contribution scores

The three chromosome-held-out loci now have official ChromBPNet DeepSHAP scores and separately calculated Captum DeepLIFT Rescale scores. These are explanations of the new **SCCApos mouse** model, not the old human browser tracks. No negative-state subtraction or TF-MoDISco was performed here.

## Inputs and model

`export_fp32/export_metadata.json` records checkpoint/input hashes, exact mm10 windows, and the bias-free component extracted from the full newly trained model. The order is **P1, P2, promoter**. Every attribution covers 2114 input bases in A,C,G,T channel order. Full-model profile fits must use the separate native combined-model predictions; attribution uses the bias-free component as [recommended by ChromBPNet](https://github.com/kundajelab/chrombpnet/wiki/Generate-contribution-score-bigwigs).

References are 100 verified dinucleotide shuffles per sequence, generated with `numpy.RandomState(1234 + locus_index)`. The 20-reference analysis uses the exact first 20. This is a deterministic preserved reference sample, not a claim that changing reference samples has no effect. Target training units are unsmoothed per-base ATAC counts from the bundled Zarr; upstream Tn5 shift/endpoint provenance remains unknown. Do not describe these targets as confirmed shifted insertion counts.

## File contract

| Files | Meaning |
|---|---|
| `deepshap_100ref/counts_scores.npz` | Primary official logcount contribution scores |
| `deepshap_100ref/profile_scores.npz` | Primary official dynamically weighted profile scores |
| `deepshap_100ref/profile_frozen_input_scores.npz` | Fixed-profile target for the matched implementation check |
| `deepshap_20ref/*_scores.npz` | Official default-size reference sensitivity |
| `deeplift_100ref/counts_scores.npz` | Mean of 100 genuine single-reference Rescale explanations |
| `deeplift_100ref/profile_frozen_input_scores.npz` | Matched fixed-profile Rescale explanations |
| `deeplift_100ref/*_locus{i}_per_reference.npz` | Each reference explanation; index 0 is the single-reference sensitivity |
| `export_fp32/native_predictions.npz` | Bias-free native forward logits/counts, not full-model accessibility profiles |
| `implementation_comparison.json`, `reference_sensitivity.json` | Quantitative implementation and reference checks |
| `artifact_registry.json` | Artifact sizes and SHA256 hashes |

Every aggregate score NPZ contains `hypothetical` and `projected`, each `(3,2114,4)`, `inputs` `(3,2114,4)`, and `references` `(3,N,2114,4)`. `projected = hypothetical * inputs`. The hypothetical projection is exactly the official ChromBPNet multiplier/reference projection, not a gradient placeholder. DeepLIFT aggregate NPZs additionally retain `difference_attributions` `(3,100,2114,4)`; each such per-reference array sums to the associated scalar output difference. Per-reference NPZs contain `hypothetical` and `difference_attributions` `(100,2114,4)`.

`native_predictions.npz` has `logits` `(303,1000)` and `counts` `(303,)`. Row order is the first three original inputs, then 100 P1 references, then 100 P2 references, then 100 promoter references. It contains the bias-free component only. `inputs_references.npz` preserves the exact sequences.

## Algorithms and checks

The official Kundaje TensorFlow DeepExplainer backend and ChromBPNet `shap_utils.py` are used without changing attribution rules. The only operator compatibility alias is modern linear `AddV2` to the existing linear-add passthrough. The unchanged backend uses original derivatives for linear dilated-convolution reshapes. Reference pairs are evaluated in chunks of two and their scores are averaged; this CNN has no batch-dependent normalization or dropout. Source SHA256 values and operator types are in run metadata. Runtime dependencies are isolated here: `kundajelab-shap==1`, `deeplift==0.6.13.0`, `captum==0.8.0`.

Standalone [Captum DeepLift](https://captum.ai/api/deep_lift.html) uses the Rescale rule on a PyTorch copy of the same native convolutions, unique ReLUs, residual crops, and output heads. All 303 original/reference forward predictions were checked against native TensorFlow before explanations: maximum absolute errors were 4.77e-6 for counts and 2.44e-5 for profile logits. TF32 was disabled. Each single-reference Rescale explanation passed scalar conservation within 2e-6 and differs substantially from an ordinary gradient. Averaging these same 100 reference explanations agrees with official DeepSHAP: maximum hypothetical-score difference 6.71e-8 and relative L2 error below 9.3e-7 for counts and fixed profile. These are **independent implementations of closely related DeepLIFT-family calculations**, not two unrelated methods; agreement is a numerical implementation check, not independent biological evidence. The [DeepLIFT project](https://github.com/kundajelab/deeplift) also describes this relationship to DeepExplainer.

The count scalar is the native logcount head. For the official profile scalar, let `z` be mean-centered logits and `w=softmax(z)`. The official target is `sum(stop_gradient(w)*z)`. Its blocked weight path means ordinary `f(input)-mean(f(reference))` completeness does **not** apply. The fork's product rule assigns the retained logit branch the pair-averaged weight, so the correct partial conservation check is the reference mean of `sum(0.5*(w_input+w_reference)*(z_input-z_reference))`. All three 100-reference official profile residuals are below 1.5e-7 against this quantity. Metadata preserves the ordinary forward residual separately and marks it inapplicable.

The supplementary matched profile target fixes weights to the unperturbed input's softmax for both input and references, then explains `sum(w_input*z)`. Ordinary scalar conservation applies to this fixed target, and both implementations agree. Do not relabel these fixed-target scores as the official dynamic profile output.

The 20-versus-100 reference hypothetical correlations span 0.83–0.92 for counts and 0.77–0.89 for the official profile. This sensitivity is recorded, and the 100-reference result is primary. A single-reference track can differ further; its preserved scores should be treated as sensitivity, not substituted for the reference ensemble.

## Reproduction

From the repository root, use the native runtime and the existing bootstrap (CUDA paths/legacy Keras are isolated). Output folders must be new to avoid overwriting artifacts.

```bash
../get_model_seq/.venv/bin/python reviewer_revision_xi_frank/cd200_locus_reinterpretation_20261103/chrombpnet_interpret/interpret_native.py --stage export --output NEW_EXPORT
../get_model_seq/.venv/bin/python reviewer_revision_xi_frank/cd200_locus_reinterpretation_20261103/chrombpnet_interpret/interpret_native.py --stage deepshap --export-dir NEW_EXPORT --output NEW_SHAP --references 100 --reference-batch-size 2
../motif_chip_model/.venv/bin/python reviewer_revision_xi_frank/cd200_locus_reinterpretation_20261103/chrombpnet_interpret/deeplift_rescale.py --export-dir NEW_EXPORT --output NEW_DEEPLIFT --references 100 --reference-batch-size 2
```

Acceptance checks ran on the local RTX5090 GPU. Full metadata and logs are preserved. `export/` is an earlier diagnostic artifact whose TF32-sensitive component-comparison assertion failed; it is retained for audit and excluded from all final analyses. `smoke_shap_2ref/` is only the preliminary compatibility check. The final export is `export_fp32/`. No existing runtime installation was replaced, and no diagnostic artifact was deleted.

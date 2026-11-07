#!/usr/bin/env python3
"""Mouse-only bias training and native ChromBPNet on the common cache.

No human checkpoint is accepted. Bias filters, background thresholds, count-loss
weight and bias rescaling use training chromosomes only. Validation chromosomes
select the checkpoint. No held-out cache arrays are read while fitting.

The immutable shared human trainer supplies the existing runtime bootstrap,
cache contract, native full-model architecture and full-model train/predict loops.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

BASE = Path(__file__).resolve().parent
COUNT_TRANSFORM = "ln1p_stored_atac_count"
TARGET_DESCRIPTION = "Bundled unsmoothed, unnormalized per-base ATAC counts as stored; ln(1 + central 1000-bp sum)"
TARGET_LIMITATION = "Upstream Tn5 shift and fragment-endpoint provenance have not been established; these are not certified +4/-4 insertion tracks"
SHARED_PATH = BASE.parent / "gm12878_benchmark/train_chrombpnet.py"
SPEC = importlib.util.spec_from_file_location("mouse_shared_native_chrombpnet", SHARED_PATH)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)


def centered_counts(cache, block_size=8192):
    result = np.empty(len(cache), dtype=np.float64)
    for begin in range(0, len(cache), block_size):
        y = cache.y[begin:begin + block_size, 500:1500]
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Invalid observed signal: nonfinite/negative stored ATAC counts")
        result[begin:begin + len(y)] = y.sum(-1, dtype=np.float64)
    return result


def background_gates(train_counts, is_peak, threshold_factor=0.5, outlier_quantile=0.9999):
    """Native bias filter rule, fitted to TRAIN only rather than train+valid."""
    counts = np.asarray(train_counts, dtype=np.float64)
    peak = np.asarray(is_peak, dtype=bool)
    if counts.shape != peak.shape or counts.ndim != 1 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Invalid training counts/peak labels")
    if not 0 < threshold_factor <= 1 or not 0.5 < outlier_quantile < 1:
        raise ValueError("Invalid prespecified bias threshold or outlier quantile")
    if not peak.any() or peak.all():
        raise ValueError("Bias fitting requires both training peaks and nonpeaks")
    peak_quantile = float(np.quantile(counts[peak], 0.01))
    upper_background = peak_quantile * threshold_factor
    if upper_background <= 0:
        raise ValueError("Training peak first percentile is zero: native background cutoff cannot be estimated; inspect the peak universe")
    low_background = counts[(~peak) & (counts < upper_background)]
    if len(low_background) < 2:
        raise ValueError("Fewer than two training backgrounds below prespecified native bias cutoff")
    lower, upper = np.quantile(low_background, [1 - outlier_quantile, outlier_quantile])
    selected = np.flatnonzero((~peak) & (counts > lower) & (counts < upper))
    if len(selected) < 2:
        raise ValueError("Native strict count-outlier gates leave fewer than two training backgrounds")
    weight = max(1.0, float(np.median(counts[selected])) / 10.0)
    return selected, {"peak_count_quantile": 0.01, "training_peak_count_quantile_value": peak_quantile,
                      "bias_threshold_factor": threshold_factor, "background_count_upper_cutoff": upper_background,
                      "outlier_quantile": outlier_quantile, "strict_count_lower": float(lower),
                      "strict_count_upper": float(upper), "count_loss_weight": weight,
                      "training_nonpeaks_before_filtering": int((~peak).sum()),
                      "training_nonpeaks_below_background_cutoff": int(len(low_background)),
                      "training_nonpeaks_selected": int(len(selected)), "fitted_on": "training chromosomes only"}


def apply_background_gates(counts, is_peak, gates):
    return np.flatnonzero((~np.asarray(is_peak, dtype=bool)) &
                         (counts > gates["strict_count_lower"]) &
                         (counts < gates["strict_count_upper"]))


def mouse_inputs(args):
    manifest_path = args.cache_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("assembly") != "mm10":
        raise ValueError("Mouse training requires a cache manifest explicitly declaring assembly mm10")
    train, valid = shared.Cache(args.cache_dir, "train"), shared.Cache(args.cache_dir, "valid")
    shared.assert_partitions(train, valid)
    forbidden = set(args.heldout_chroms)
    if "chr16" not in forbidden:
        raise ValueError("Cd200 chromosome chr16 must be explicitly held out")
    for name, cache in [("train", train), ("valid", valid)]:
        overlap = forbidden & {row["chrom"] for row in cache.regions}
        if overlap:
            raise ValueError(f"Held-out chromosomes occur in {name}: {sorted(overlap)}")
    return train, valid, manifest


def native_bias_model(tf, args, count_weight):
    from chrombpnet.training.models import bpnet_model
    return bpnet_model.getModelGivenModelOptionsAndWeightInits(
        SimpleNamespace(seed=args.seed, learning_rate=args.learning_rate),
        {"filters": 128, "n_dil_layers": 4, "counts_loss_weight": count_weight,
         "inputlen": 2114, "outputlen": 1000})


def bias_batch(cache, indices, rng=None):
    # Centered always: bias training has no coordinate jitter. Native reverse
    # complement augmentation is retained only for the training pass.
    x, targets = cache.batch(indices)
    if rng is not None:
        rc = rng.random(len(indices)) < 0.5
        x[rc] = x[rc, ::-1, ::-1]
        targets[0][rc] = targets[0][rc, ::-1]
    return x, targets


def batch_pass(model, cache, indices, args, rng=None, training=False):
    totals, seen = None, 0
    limit = args.max_train_steps if training else args.max_valid_batches
    for step, begin in enumerate(range(0, len(indices), args.batch_size)):
        if limit and step >= limit:
            break
        ix = indices[begin:begin + args.batch_size]
        x, targets = bias_batch(cache, ix, rng if training else None)
        metrics = model.train_on_batch(x, targets, return_dict=True) if training else model.test_on_batch(x, targets, return_dict=True)
        values = {key: float(value) for key, value in metrics.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("Native mouse bias loss became nonfinite")
        if totals is None:
            totals = {key: 0.0 for key in values}
        for key, value in values.items():
            totals[key] += len(ix) * value
        seen += len(ix)
    if not seen:
        raise ValueError("Empty native bias pass")
    return {**{key: value / seen for key, value in totals.items()}, "examples": seen}


def region_hash(cache, indices):
    return hashlib.sha256("".join(cache.regions[int(i)]["region_id"] + "\n" for i in indices).encode()).hexdigest()


def fit_bias(args, tf):
    train, valid, manifest = mouse_inputs(args)
    train_counts, valid_counts = centered_counts(train), centered_counts(valid)
    train_ix, gates = background_gates(train_counts, train.is_peak, args.bias_threshold_factor)
    valid_ix = apply_background_gates(valid_counts, valid.is_peak, gates)
    if len(valid_ix) < 2:
        raise ValueError("Training-derived native bias gates leave fewer than two validation backgrounds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Bias output directory must be empty; existing runs are preserved")
    np.save(args.output_dir / "training_indices.npy", train_ix)
    np.save(args.output_dir / "validation_indices.npy", valid_ix)
    model = native_bias_model(tf, args, gates["count_loss_weight"])
    metadata = {"model": "mouse_native_bias_bpnet", "assembly": "mm10", "initialization": "random; no pretrained checkpoint",
                "seed": args.seed, "filters": 128, "n_dilation_layers": 4, "input_length": 2114, "output_length": 1000,
                "gates": gates, "training_chromosomes": sorted({r["chrom"] for r in train.regions}),
                "validation_chromosomes": sorted({r["chrom"] for r in valid.regions}),
                "heldout_chromosomes": list(args.heldout_chroms), "test_opened_during_fitting": False,
                "train_region_ids_sha256": region_hash(train, train_ix), "valid_region_ids_sha256": region_hash(valid, valid_ix),
                "cache_manifest_sha256": shared.sha256(args.cache_dir / "cache_manifest.json"),
                "tensorflow": tf.__version__, "parameters": int(model.count_params()),
                "pilot": bool(args.max_train_steps or args.max_valid_batches),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "source_sha256": shared.sha256(__file__), "shared_trainer_sha256": shared.sha256(SHARED_PATH),
                "native_bias_source_sha256": shared.sha256(shared.VENDOR / "chrombpnet/training/models/bpnet_model.py"),
                "bias_motif_purity_verified": False,
                "declared_changes": ["Background and outlier thresholds fitted on train only instead of native train+valid", "Training uses centered sequences and random reverse complements; validation has neither augmentation", "Train-only intercept scaling after best-validation checkpoint selection"]}
    shared.save_json(args.output_dir / "metadata.json", metadata)
    best, best_epoch, history = float("inf"), -1, []
    for epoch in range(args.max_epochs):
        start = time.monotonic()
        rng = np.random.default_rng(args.seed + epoch)
        order = train_ix.copy()
        rng.shuffle(order)
        tr = batch_pass(model, train, order, args, rng, True)
        va = batch_pass(model, valid, valid_ix, args)
        row = {"epoch": epoch, "train": tr, "valid": va, "seconds": time.monotonic() - start}
        history.append(row)
        if va["loss"] < best:
            best, best_epoch = va["loss"], epoch
            model.save_weights(args.output_dir / "best.weights.h5")
        shared.save_json(args.output_dir / "history.json", history)
        print(json.dumps(row), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model.load_weights(args.output_dir / "best.weights.h5")
    model.save(args.output_dir / "bias_unscaled.h5", include_optimizer=False)
    # Native adjustment minimizes scalar log-count MSE on nonpeak regions. Only
    # TRAIN data are used here; no model parameter is refit on test/validation.
    scale_ix = np.flatnonzero(~train.is_peak)
    residual_sum, scale_n = 0.0, 0
    for begin in range(0, len(scale_ix), args.batch_size):
        ix = scale_ix[begin:begin + args.batch_size]
        x, targets = train.batch(ix)
        _, prediction = model(x, training=False)
        residual = targets[1][:, 0] - prediction.numpy()[:, 0]
        if not np.isfinite(residual).all():
            raise FloatingPointError("Nonfinite native bias-rescaling residual")
        residual_sum += residual.sum(dtype=np.float64)
        scale_n += len(ix)
    delta = float(residual_sum / scale_n)
    last = model.layers[-1]
    if not isinstance(last, tf.keras.layers.Dense) or last.units != 1:
        raise ValueError("Native bias last layer is not scalar Dense")
    weights, intercept = last.get_weights()
    last.set_weights([weights, intercept + delta])
    model.trainable = False
    model.save(args.output_dir / "bias_model_scaled.h5", include_optimizer=False)
    provenance = {**metadata, "stage": "completed_mouse_bias_fit_and_train_only_rescaling",
                  "best_epoch": best_epoch, "best_validation_loss": best, "epochs_completed": len(history),
                  "scaling_training_nonpeaks": scale_n, "scaling_region_ids_sha256": region_hash(train, scale_ix),
                  "logcount_intercept_shift": delta,
                  "unscaled_model_sha256": shared.sha256(args.output_dir / "bias_unscaled.h5"),
                  "scaled_model_sha256": shared.sha256(args.output_dir / "bias_model_scaled.h5")}
    shared.save_json(args.output_dir / "bias_provenance.json", provenance)
    shared.save_json(args.output_dir / "training_result.json", {"best_epoch": best_epoch, "best_validation_loss": best,
                     "epochs_completed": len(history), "pilot": metadata["pilot"], "test_evaluated": False})


def require_mouse_bias(args):
    if args.bias_model is None:
        raise ValueError("Fresh mouse main-model training requires --bias-model from the mouse bias stage")
    sidecar = args.bias_model.parent / "bias_provenance.json"
    provenance = json.loads(sidecar.read_text())
    if provenance.get("stage") != "completed_mouse_bias_fit_and_train_only_rescaling" or provenance.get("assembly") != "mm10":
        raise ValueError("Rejecting a pretrained human/unknown-lineage bias component")
    if provenance.get("initialization") != "random; no pretrained checkpoint":
        raise ValueError("Mouse bias must have been trained from random initialization")
    if provenance["scaled_model_sha256"] != shared.sha256(args.bias_model):
        raise ValueError("Mouse bias weights differ from the verified provenance")
    if provenance["cache_manifest_sha256"] != shared.sha256(args.cache_dir / "cache_manifest.json"):
        raise ValueError("Mouse bias and main model must use the exact same cache/split manifest")
    if set(args.heldout_chroms) & set(provenance["training_chromosomes"] + provenance["validation_chromosomes"]):
        raise ValueError("Bias component has seen a requested held-out chromosome")
    if provenance.get("pilot") and not (args.max_train_steps or args.max_valid_batches):
        raise ValueError("A bias pilot cannot be used as the final benchmark component")
    return sidecar, provenance


def install_mouse_metadata_contract():
    """Relabel this imported runner instance without editing shared human code."""
    original = shared.save_json
    def save(path, obj):
        if isinstance(obj, dict):
            obj = dict(obj)
            obj["count_transform"] = COUNT_TRANSFORM
            obj["target_units"] = "unsmoothed_per_base_ATAC_counts_from_bundled_Zarr"
            obj["genome_build"] = "mm10"
            obj["target_kind"] = "stored_atac_signal_counts"
            obj["input_length"] = 2114
            obj["output_length"] = 1000
            obj["target_description"] = TARGET_DESCRIPTION
            obj["target_provenance_limitation"] = TARGET_LIMITATION
            if "prediction_counts_units" in obj:
                obj["prediction_counts_units"] = "predicted natural_log(1 + common central-1000 stored ATAC counts)"
            if "native_changes" in obj:
                obj["native_changes"] = [s.replace("released bias", "newly trained mouse bias") for s in obj["native_changes"]]
        original(path, obj)
    shared.save_json = save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["bias", "train", "predict"])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bias-model", type=Path)
    parser.add_argument("--heldout-chroms", nargs="+", default=["chr16"])
    parser.add_argument("--bias-threshold-factor", type=float, default=0.5)
    parser.add_argument("--count-weight", type=float)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--negative-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-valid-batches", type=int, default=0)
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size) <= 0:
        parser.error("Epoch, patience and batch size must be positive")
    # The shared bootstrap re-execs this entry point with the existing CUDA12
    # runtime; no shared human source or environment is edited.
    shared.bootstrap()
    install_mouse_metadata_contract()
    mouse_inputs(args)
    if args.mode != "bias":
        sidecar, provenance = require_mouse_bias(args)
    tf = shared.setup_tf(args.seed)
    if args.mode == "bias":
        fit_bias(args, tf)
    elif args.mode == "train":
        shared.fit(args, tf)
        p = args.output_dir / "metadata.json"
        metadata = json.loads(p.read_text())
        metadata["mouse_bias_provenance_sha256"] = shared.sha256(sidecar)
        metadata["mouse_entrypoint_sha256"] = shared.sha256(__file__)
        metadata["heldout_chromosomes"] = list(args.heldout_chroms)
        shared.save_json(p, metadata)
    else:
        shared.predict(args, tf)
        p = args.output_dir / "test_predictions/prediction_metadata.json"
        metadata = json.loads(p.read_text())
        metadata["mouse_bias_provenance_sha256"] = shared.sha256(sidecar)
        metadata["mouse_entrypoint_sha256"] = shared.sha256(__file__)
        metadata["bias_trained_from_scratch_on_train_chromosomes"] = True
        shared.save_json(p, metadata)


if __name__ == "__main__":
    main()

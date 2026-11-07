#!/usr/bin/env python3
"""Official bias-factorized ChromBPNet on the shared cache; GPU required.

Use legacy tf.keras with existing CUDA12 libraries. The upstream model and loss
are imported unchanged. Released inference loads non-Lambda component models and
recombines their outputs with the inspected upstream Add/logsumexp equations.
Cache geometry: DNA3114, insertions2000, aligned offset500 at evaluation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np

BASE = Path(__file__).resolve().parent
VENDOR = BASE / "vendor/chrombpnet"


def bootstrap():
    """Dynamic loader search paths must exist before Python process startup."""
    if os.environ.get("CD200_TF_RUNTIME") != "1":
        env = dict(os.environ)
        cuda = BASE.parents[2] / "motif_chip_model/.venv/lib/python3.12/site-packages/nvidia"
        # BASE.parents[2] is Repos for this repository's explicit layout.
        paths = [str(p) for p in cuda.glob("*/lib")]
        if not paths:
            raise RuntimeError(f"Existing CUDA12 library root not found: {cuda}")
        env["LD_LIBRARY_PATH"] = ":".join(["/usr/lib/x86_64-linux-gnu", *paths, env.get("LD_LIBRARY_PATH", "")])
        env["TF_USE_LEGACY_KERAS"] = "1"
        env["CD200_TF_RUNTIME"] = "1"
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
        triton = cuda.parent / "triton/backends/nvidia"
        compiler = BASE / "runtime_cuda"
        for source, relative in [(triton / "lib/libdevice.10.bc", "nvvm/libdevice/libdevice.10.bc"),
                                 (triton / "bin/ptxas", "bin/ptxas")]:
            if not source.is_file():
                raise RuntimeError(f"Existing CUDA compiler artifact missing: {source}")
            target = compiler / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(source)
        if "--xla_gpu_cuda_data_dir=" not in env.get("XLA_FLAGS", ""):
            env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + f" --xla_gpu_cuda_data_dir={compiler}").strip()
        env["PATH"] = str(compiler / "bin") + ":" + env.get("PATH", "")
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    sys.path.insert(0, str(BASE / "runtime_deps"))
    sys.path.insert(0, str(VENDOR))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def setup_tf(seed):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("CUDA GPU required; CPU training/inference fallback prohibited")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Verify that a convolution actually executes on GPU, not merely discovery.
    with tf.device("/GPU:0"):
        probe = tf.nn.conv1d(tf.ones((1, 32, 4)), tf.ones((3, 4, 2)), 1, "VALID")
    if "GPU" not in probe.device:
        raise RuntimeError("TensorFlow convolution was not placed on CUDA GPU")
    return tf


class Cache:
    def __init__(self, directory, split):
        self.directory, self.split = Path(directory), split
        self.x = np.load(self.directory / f"{split}.x.npy", mmap_mode="r")
        self.y = np.load(self.directory / f"{split}.y.npy", mmap_mode="r")
        with (self.directory / f"{split}.regions.tsv").open() as handle:
            self.regions = list(csv.DictReader(handle, delimiter="\t"))
        if self.x.dtype != np.uint8 or self.x.shape != (len(self.regions), 3114):
            raise ValueError("Expected uint8 DNA(N,3114) and matching regions")
        if self.y.dtype != np.float32 or self.y.shape != (len(self.regions), 2000):
            raise ValueError("Expected float32 insertions(N,2000)")
        ids = [r["region_id"] for r in self.regions]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty cache or duplicate region IDs")
        self.is_peak = np.array([r["is_peak"].lower() in ("1", "true", "yes") for r in self.regions])

    def __len__(self):
        return len(self.regions)

    def batch(self, indices, rng=None):
        indices = np.asarray(indices, dtype=np.int64)
        offsets = np.full(len(indices), 500, dtype=np.int64)
        if rng is not None:
            mask = self.is_peak[indices]
            offsets[mask] = rng.integers(0, 1001, int(mask.sum()))
        dna = np.asarray(self.x[indices[:, None], offsets[:, None] + np.arange(2114)], dtype=np.int64)
        target = np.asarray(self.y[indices[:, None], offsets[:, None] + np.arange(1000)], dtype=np.float32).copy()
        if (dna > 4).any() or not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("Nonfinite/negative target or invalid nucleotide encoding")
        onehot = np.eye(5, dtype=np.float32)[dna, :4]
        if rng is not None:
            rc = rng.random(len(indices)) < 0.5
            onehot[rc] = onehot[rc, ::-1, ::-1]
            target[rc] = target[rc, ::-1]
        return onehot, [target, np.log1p(target.sum(-1, keepdims=True))]


def assert_partitions(train, valid):
    tc = {r["chrom"] for r in train.regions}
    vc = {r["chrom"] for r in valid.regions}
    if tc & vc:
        raise ValueError(f"Training/validation chromosomes overlap: {tc & vc}")


def no_lambda_component(tf, path):
    """Reject marshaled code rather than deserialize old Python Lambda bytecode."""
    import h5py
    with h5py.File(path) as handle:
        config = json.loads(handle.attrs["model_config"])
    def inspect(node):
        if isinstance(node, dict):
            if node.get("class_name") == "Lambda":
                raise ValueError("Component contains serialized Lambda; do not deserialize")
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)
    inspect(config)
    model = tf.keras.models.load_model(path, compile=False)
    if model.input_shape != (None, 2114, 4) or model.output_shape != [(None, 1000), (None, 1)]:
        raise ValueError(f"Unexpected released component geometry: {model.input_shape}, {model.output_shape}")
    return model


def fresh_model(tf, bias_path, count_weight, learning_rate, seed=1234):
    from chrombpnet.training.models import chrombpnet_with_bias_model as native
    # Upstream loader compiles an old training config unnecessarily; override only
    # deserialization, preserving its explicit freezing and native construction.
    def load_bias(path):
        model = no_lambda_component(tf, path)
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
        return model
    native.load_pretrained_bias = load_bias
    params = {"filters": 512, "n_dil_layers": 8, "counts_loss_weight": count_weight,
              "bias_model_path": str(bias_path), "inputlen": 2114, "outputlen": 1000}
    model = native.getModelGivenModelOptionsAndWeightInits(
        SimpleNamespace(seed=seed, learning_rate=learning_rate), params)
    bias = next(m for m in model.layers if isinstance(m, tf.keras.Model) and m.name != "model_wo_bias")
    if bias.trainable_weights:
        raise ValueError("Bias network unexpectedly has trainable weights")
    return model


def released_model(tf, bias_path, nobias_path):
    bias = no_lambda_component(tf, bias_path)
    nobias = no_lambda_component(tf, nobias_path)
    x = tf.keras.Input((2114, 4), name="sequence")
    bp, bc = bias(x)
    tp, tc = nobias(x)
    # Exact native equations inspected in upstream source; no marshaled Lambda.
    profile = tf.keras.layers.Add(name="logits_profile_predictions")([tp, bp])
    concat = tf.keras.layers.Concatenate(axis=-1)([tc, bc])
    count = tf.keras.layers.Lambda(
        lambda t: tf.reduce_logsumexp(t, axis=-1, keepdims=True), name="logcount_predictions")(concat)
    return tf.keras.Model(x, [profile, count])


def fit(args, tf):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Use a fresh output directory; overwriting a run is prohibited")
    train, valid = Cache(args.cache_dir, "train"), Cache(args.cache_dir, "valid")
    assert_partitions(train, valid)
    peaks, negatives = np.flatnonzero(train.is_peak), np.flatnonzero(~train.is_peak)
    if not len(peaks):
        raise ValueError("No training peaks")
    centered_counts = np.asarray(train.y[peaks, 500:1500]).sum(-1)
    weight = args.count_weight if args.count_weight is not None else max(1.0, float(np.median(centered_counts)) / 10)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("Invalid count loss weight")
    model = fresh_model(tf, args.bias_model, weight, args.learning_rate, args.seed)
    valid_rng = np.random.default_rng(args.seed)
    valid_peaks, valid_negatives = np.flatnonzero(valid.is_peak), np.flatnonzero(~valid.is_peak)
    valid_nneg = min(len(valid_negatives), int(len(valid_peaks) * args.negative_ratio))
    # Same fixed validation sample as the shared Torch runner; never reshuffle.
    valid_indices = np.sort(np.concatenate([valid_peaks, valid_rng.choice(valid_negatives, valid_nneg, replace=False)]))
    np.save(args.output_dir / "validation_indices.npy", valid_indices)
    sources = [Path(__file__), args.bias_model,
               VENDOR / "chrombpnet/training/models/chrombpnet_with_bias_model.py",
               VENDOR / "chrombpnet/training/utils/losses.py"]
    meta = {"model": "chrombpnet_native_fresh", "seed": args.seed,
            "tensorflow": tf.__version__, "keras": "legacy tf_keras",
            "gpu": tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]),
            "count_weight": weight, "count_weight_rule": "explicit" if args.count_weight is not None else "max(1,median centered training peak counts/10)",
            "input_length": 2114, "output_length": 1000, "filters": 512, "dilations": 8,
            "pilot": bool(args.max_train_steps or args.max_valid_batches),
            "source_hashes": {str(p): sha256(p) for p in sources},
            "training_chromosomes": sorted({r["chrom"] for r in train.regions}),
            "validation_chromosomes": sorted({r["chrom"] for r in valid.regions}),
            "cache_metadata_sha256": sha256(args.cache_dir / "cache_manifest.json"),
            "native_changes": ["Cache-backed batch iteration instead of genomic data generator; native model and TFP multinomial loss unchanged", "Deserialize released bias with compile=False; freeze exactly as upstream", "Train-only count-weight estimate unless explicit; released original used train+valid", "No test data read during fitting"],
            "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            "parameters": int(model.count_params()),
            "bias_model_sha256": sha256(args.bias_model)}
    meta["validation_indices_sha256"] = sha256(args.output_dir / "validation_indices.npy")
    save_json(args.output_dir / "metadata.json", meta)
    best, best_epoch, history = float("inf"), -1, []
    for epoch in range(args.max_epochs):
        start = time.monotonic()
        rng = np.random.default_rng(args.seed + epoch)
        nneg = min(len(negatives), int(len(peaks) * args.negative_ratio))
        ix = np.concatenate([peaks, rng.choice(negatives, nneg, replace=False)])
        rng.shuffle(ix)
        rows = {}
        for name, data, indices, training in [("train", train, ix, True), ("valid", valid, valid_indices, False)]:
            totals, seen = None, 0
            limit = args.max_train_steps if training else args.max_valid_batches
            for step, begin in enumerate(range(0, len(indices), args.batch_size)):
                if limit and step >= limit:
                    break
                sub = indices[begin:begin+args.batch_size]
                x, targets = data.batch(sub, rng if training else None)
                metric = model.train_on_batch(x, targets, return_dict=True) if training else model.test_on_batch(x, targets, return_dict=True)
                values = {k: float(v) for k,v in metric.items()}
                if not all(np.isfinite(v) for v in values.values()):
                    raise FloatingPointError(f"Nonfinite {name} loss")
                if totals is None:
                    totals = {k: 0.0 for k in values}
                for key, value in values.items():
                    totals[key] += len(sub) * value
                seen += len(sub)
            if not seen:
                raise ValueError("No examples evaluated")
            rows[name] = {**{k: v / seen for k,v in totals.items()}, "examples": seen}
        row = {"epoch": epoch, **rows, "seconds": time.monotonic()-start}
        history.append(row)
        if rows["valid"]["loss"] < best:
            best, best_epoch = rows["valid"]["loss"], epoch
            model.save_weights(args.output_dir / "best.weights.h5")
        save_json(args.output_dir / "history.json", history)
        print(json.dumps(row), flush=True)
        if epoch-best_epoch >= args.patience:
            break
    save_json(args.output_dir / "training_result.json", {"best_epoch": best_epoch, "best_validation_loss": best,
              "epochs_completed": len(history), "pilot": meta["pilot"], "test_evaluated": False})


def predict(args, tf):
    is_released = args.mode == "released"
    if is_released:
        if args.released_nobias is None:
            raise ValueError("Released inference requires --released-nobias component")
        model = released_model(tf, args.bias_model, args.released_nobias)
        meta = {"pilot": False, "training_chromosomes": ["chr2","chr4","chr5","chr7","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr21","chr22","chrX","chrY"], "validation_chromosomes": ["chr8","chr20"]}
    else:
        meta = json.loads((args.output_dir / "metadata.json").read_text())
        if meta["bias_model_sha256"] != sha256(args.bias_model):
            raise ValueError("Bias model differs from training")
        model = fresh_model(tf, args.bias_model, meta["count_weight"], args.learning_rate, meta["seed"])
        model.load_weights(args.output_dir / "best.weights.h5")
    test = Cache(args.cache_dir, "test")
    if set(meta["training_chromosomes"] + meta["validation_chromosomes"]) & {r["chrom"] for r in test.regions}:
        raise ValueError("Test chromosomes overlap model training or validation")
    directory = args.output_dir / "test_predictions"
    directory.mkdir(parents=True, exist_ok=False)
    counts = np.lib.format.open_memmap(directory / "prediction_counts.npy", mode="w+", dtype=np.float32, shape=(len(test),))
    profiles = np.lib.format.open_memmap(directory / "prediction_profiles.npy", mode="w+", dtype=np.float32, shape=(len(test),1000))
    for begin in range(0, len(test), args.batch_size):
        ix = np.arange(begin, min(begin+args.batch_size,len(test)))
        x, _ = test.batch(ix)
        logits, count = model(x, training=False)
        prob = tf.nn.softmax(logits, axis=-1).numpy()
        count = count.numpy()[:,0]
        if not np.isfinite(prob).all() or not np.isfinite(count).all():
            raise FloatingPointError("Nonfinite test predictions")
        profiles[ix], counts[ix] = prob, count
    profiles.flush(); counts.flush()
    checkpoint = args.released_nobias if is_released else args.output_dir / "best.weights.h5"
    save_json(directory / "prediction_metadata.json", {
        "model": "chrombpnet_published_fold0" if is_released else "chrombpnet_native_fresh",
        "pilot": meta["pilot"], "checkpoint_sha256": sha256(checkpoint),
        "bias_model_sha256": sha256(args.bias_model), "recombined_components": is_released,
        "region_ids_sha256": hashlib.sha256("".join(r["region_id"]+"\n" for r in test.regions).encode()).hexdigest(),
        "target_start": 500, "target_end": 1500, "count_transform": "ln1p_raw_insertion_count",
        "test_rows": len(test), "test_chromosomes": sorted({r["chrom"] for r in test.regions}),
        "prediction_profiles_units": "softmax native full-model profile logits",
        "prediction_counts_units": "native full-model ln1p total counts",
        "tensorflow": tf.__version__, "source_sha256": sha256(__file__)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train","predict","released","smoke"], required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bias-model", type=Path, required=True)
    parser.add_argument("--released-nobias", type=Path)
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
    bootstrap()
    tf = setup_tf(args.seed)
    if args.mode == "smoke":
        m = fresh_model(tf,args.bias_model,93.3,args.learning_rate,args.seed)
        x = np.eye(4,dtype=np.float32)[np.random.randint(4,size=(2,2114))]
        y = np.ones((2,1000),dtype=np.float32)
        result = m.train_on_batch(x,[y,np.log1p(y.sum(-1,keepdims=True))],return_dict=True)
        print(json.dumps({"gpu_smoke":{k:float(v) for k,v in result.items()},"parameters":m.count_params(),"GPU":str(tf.config.list_physical_devices("GPU"))}),flush=True)
    elif args.cache_dir is None or args.output_dir is None:
        parser.error("--cache-dir and --output-dir are required")
    elif args.mode == "train":
        fit(args,tf)
    else:
        predict(args,tf)


if __name__ == "__main__":
    main()

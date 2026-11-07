#!/usr/bin/env python3
"""Train original MoDL or official ExplaiNN on the shared GM12878 cache.

Actual fitting/prediction requires CUDA. CPU import and unit checks are supported.
Cache x is uint8 A/C/G/T/N=0/1/2/3/4, shape(N,3114); y is float32(N,2000).
For jitter j the aligned input is x[j:j+2114], target y[j:j+1000].
The cache's y starts 557bp downstream of its x. Test is read only by --predict.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
RC = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def aligned_window(x, y, offset=500, reverse=False):
    if not 0 <= offset <= 1000:
        raise ValueError("Jitter offset outside [0,1000]")
    codes = np.array(x[offset:offset + 2114], copy=True)
    target = np.array(y[offset:offset + 1000], dtype=np.float32, copy=True)
    if codes.shape != (2114,) or target.shape != (1000,):
        raise ValueError("Cache window has invalid length")
    if (codes > 4).any() or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Invalid DNA codes or nonfinite/negative observed insertions")
    if reverse:
        codes, target = RC[codes[::-1]], target[::-1].copy()
    sequence = F.one_hot(torch.from_numpy(codes.astype(np.int64)), num_classes=5)[..., :4].T.float()
    return sequence, torch.from_numpy(target)


class CacheDataset(Dataset):
    def __init__(self, cache, split, training=False):
        self.cache, self.split, self.training = Path(cache), split, training
        self.x = np.load(self.cache / f"{split}.x.npy", mmap_mode="r")
        self.y = np.load(self.cache / f"{split}.y.npy", mmap_mode="r")
        if self.x.dtype != np.uint8 or self.x.ndim != 2 or self.x.shape[1] != 3114:
            raise ValueError(f"{split}: expected uint8 x(N,3114), got {self.x.dtype} {self.x.shape}")
        if self.y.dtype != np.float32 or self.y.shape != (len(self.x), 2000):
            raise ValueError(f"{split}: expected float32 y(N,2000)")
        with (self.cache / f"{split}.regions.tsv").open() as handle:
            self.regions = list(csv.DictReader(handle, delimiter="\t"))
        if len(self.regions) != len(self.x) or not self.regions:
            raise ValueError(f"{split}: empty or mismatched region rows")
        if not all({"region_id", "chrom", "center", "is_peak"} <= row.keys() for row in self.regions):
            raise ValueError("Missing region identity/partition columns")
        ids = [row["region_id"] for row in self.regions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{split}: duplicated region_id")
        self.is_peak = np.array([row["is_peak"].lower() in ("1", "true", "yes") for row in self.regions])
        if any(row["is_peak"].lower() not in ("0", "1", "true", "false", "yes", "no") for row in self.regions):
            raise ValueError("is_peak must be a recognized boolean")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        offset = np.random.randint(0, 1001) if self.training and self.is_peak[index] else 500
        reverse = bool(np.random.randint(2)) if self.training else False
        x, y = aligned_window(self.x[index], self.y[index], offset, reverse)
        return x, y, int(index)


def sample_training_indices(is_peak, ratio, seed):
    rng = np.random.default_rng(seed)
    peaks, negatives = np.flatnonzero(is_peak), np.flatnonzero(~is_peak)
    if ratio < 0 or not len(peaks):
        raise ValueError("Need peaks and nonnegative negative ratio")
    n_negative = min(len(negatives), int(len(peaks) * ratio))
    chosen = np.concatenate([peaks, rng.choice(negatives, n_negative, replace=False)])
    rng.shuffle(chosen)
    return chosen.tolist()


def check_partition_pair(train, valid):
    tchrom = {row["chrom"] for row in train.regions}
    vchrom = {row["chrom"] for row in valid.regions}
    if tchrom & vchrom:
        raise ValueError(f"Training/validation chromosomes overlap: {sorted(tchrom & vchrom)}")
    if {row["region_id"] for row in train.regions} & {row["region_id"] for row in valid.regions}:
        raise ValueError("Training/validation region IDs overlap")


def make_model(kind, motif_path, num_cnns=100):
    if kind == "modl":
        sys.path.insert(0, str(REPO / "src"))
        from atac_prediction.dna_atac_lightning_module import DNAATACMotifCNN
        allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
                   type(np.dtype("float32")), type(np.dtype("float64"))]
        with torch.serialization.safe_globals(allowed):
            obj = torch.load(motif_path, map_location="cpu", weights_only=True)
        kernels = obj["motif_kernels"].float().numpy()
        model = DNAATACMotifCNN(kernels, num_atac_samples=1)
        if model.motif_conv.weight.requires_grad:
            raise ValueError("MoDL motif filters must remain frozen")
        return model
    sys.path.insert(0, str(BASE / "vendor" / "ExplaiNN"))
    from explainn.models.networks import ExplaiNN
    return ExplaiNN(num_cnns=num_cnns, input_length=2114, num_classes=1, filter_size=19)


def predictions(model, kind, x):
    out = model(x)
    if kind == "explainn":
        return out[:, 0].float(), None
    raw_profile = out["regression"][:, 0, 557:1557].float()
    if raw_profile.shape[1] != 1000 or not bool(torch.isfinite(raw_profile).all()) or bool((raw_profile <= 0).any()):
        raise FloatingPointError("MoDL profile must be finite and strictly positive")
    profile = raw_profile / raw_profile.sum(-1, keepdim=True)
    return out["count"][:, 0].float(), profile


def losses(logcount, profile, target, count_weight):
    expected = torch.log1p(target.sum(-1))
    count = F.mse_loss(logcount, expected)
    shape = torch.zeros((), device=target.device)
    if profile is not None:
        shape = -(target * profile.clamp_min(1e-8).log()).sum(-1).mean()
    total = count if profile is None else shape + count_weight * count
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite loss")
    return total, shape, count


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def loader(dataset, args, indices=None, epoch=0):
    generator = torch.Generator().manual_seed(args.seed + epoch)
    return DataLoader(dataset, batch_size=args.batch_size, sampler=indices,
                      shuffle=False, num_workers=args.workers, pin_memory=True,
                      drop_last=False, worker_init_fn=seed_worker, generator=generator)


def autocast(args):
    return torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else contextlib.nullcontext()


def epoch_pass(model, kind, batches, args, count_weight, optimizer=None):
    model.train(optimizer is not None)
    totals = np.zeros(3, dtype=np.float64)
    seen = 0
    maximum = args.max_train_steps if optimizer else args.max_valid_batches
    with torch.set_grad_enabled(optimizer is not None):
        for step, (x, y, indices) in enumerate(batches):
            if maximum and step >= maximum:
                break
            # ExplaiNN's original per-unit BN needs >1 training example.
            if optimizer and kind == "explainn" and len(x) == 1:
                continue
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                count, profile = predictions(model, kind, x)
                if not bool(torch.isfinite(count).all()):
                    raise FloatingPointError("Nonfinite scalar prediction")
                loss, shape, count_loss = losses(count, profile, y, count_weight)
            if optimizer:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
            n = len(x)
            totals += n * np.array([loss.item(), shape.item(), count_loss.item()])
            seen += n
    if not seen:
        raise RuntimeError("No examples scored in epoch")
    return {"loss": totals[0] / seen, "profile_nll_without_constant": totals[1] / seen,
            "logcount_mse": totals[2] / seen, "examples": seen}


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def provenance(args, train, valid, count_weight, validation_indices):
    files = [args.cache_dir / f"{s}.{suffix}" for s in ("train", "valid") for suffix in ("x.npy", "y.npy", "regions.tsv")]
    files += [Path(__file__), REPO / "src/atac_prediction/dna_atac_lightning_module.py"]
    if args.model == "modl":
        files.append(args.motif_path)
    else:
        files.append(BASE / "vendor/ExplaiNN/explainn/models/networks.py")
    return {
        "model": args.model, "seed": args.seed, "pilot": bool(args.max_train_steps or args.max_valid_batches),
        "training_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "files_sha256": {str(p): sha256(p) for p in files},
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "input_length": 2114, "output_length": 1000,
        "count_target": "natural_log(1 + sum(raw observed insertions in the common central1000))",
        "modl_adaptation": "Original fixed-motif architecture; center1000 profile crop and common raw-count multinomial/logcount objectives replace manuscript2048 smoothed-target wrapper.",
        "explainn_adaptation": "Official native scalar architecture, random trainable filters, scalar regression on the same logcount target; profile output is N/A.",
        "count_loss_weight": count_weight,
        "profile_loss": "-sum(raw_target * log(normalized_positive_profile)) averaged over examples; multinomial constant omitted",
        "negative_ratio": args.negative_ratio,
        "train_only_jitter": {"peaks": [0, 1000], "nonpeaks_offset": 500},
        "train_only_reverse_complement": True,
        "validation_offset": 500, "training_chromosomes": sorted({r["chrom"] for r in train.regions}),
        "validation_chromosomes": sorted({r["chrom"] for r in valid.regions}),
        "validation_selection": {
            "rule": "All peaks plus fixed seed-selected nonpeaks at declared negative_ratio; indices sorted in original cache order",
            "seed": args.seed, "available_rows": len(valid), "selected_rows": len(validation_indices),
            "region_ids_sha256": hashlib.sha256("".join(valid.regions[i]["region_id"] + "\n" for i in validation_indices).encode()).hexdigest(),
            "sampler_rng": "numpy.default_rng; shared deterministic sampler differs from native global RandomState implementation",
        },
        "selection": "best validation total loss (ExplaiNN scalar count MSE); test never read during fitting",
        "explainn_singleton_training_batch": "skip singleton final batch because original BatchNorm needs batch>=2; examples count recorded per epoch",
    }


def fit(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Training output directory must be empty; use a distinct run directory")
    train = CacheDataset(args.cache_dir, "train", True)
    valid = CacheDataset(args.cache_dir, "valid", False)
    check_partition_pair(train, valid)
    validation_indices = sorted(sample_training_indices(valid.is_peak, args.negative_ratio, args.seed))
    # Estimate from training peaks only, centered geometry; no validation/test fit.
    peak_indices = np.flatnonzero(train.is_peak)
    if not len(peak_indices):
        raise ValueError("No training peaks")
    peak_counts = np.array([train.y[int(i), 500:1500].sum(dtype=np.float64) for i in peak_indices])
    median_count = float(np.median(peak_counts))
    count_weight = args.count_weight if args.count_weight is not None else max(1.0, median_count / 10.0)
    if count_weight <= 0 or not np.isfinite(count_weight):
        raise ValueError("Invalid training-derived count loss weight")
    model = make_model(args.model, args.motif_path, args.num_cnns).cuda()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    meta = provenance(args, train, valid, count_weight, validation_indices)
    meta["training_peak_median_count"] = median_count
    meta["count_weight_rule"] = "max(1,median centered training peak count/10), unless explicitly overridden"
    meta["parameters"] = sum(p.numel() for p in model.parameters())
    meta["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    save_json(args.output_dir / "metadata.json", meta)
    best, best_epoch, history = float("inf"), -1, []
    for epoch in range(args.max_epochs):
        start = time.monotonic()
        indices = sample_training_indices(train.is_peak, args.negative_ratio, args.seed + epoch)
        tr = epoch_pass(model, args.model, loader(train, args, indices, epoch), args, count_weight, optimizer)
        va = epoch_pass(model, args.model, loader(valid, args, validation_indices), args, count_weight)
        row = {"epoch": epoch, "train": tr, "valid": va, "seconds": time.monotonic() - start}
        history.append(row)
        if va["loss"] < best:
            best, best_epoch = va["loss"], epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "valid_loss": best,
                        "model": args.model, "count_weight": count_weight, "seed": args.seed,
                        "num_cnns": args.num_cnns, "provenance": meta}, args.output_dir / "best.pt")
        save_json(args.output_dir / "history.json", history)
        print(json.dumps(row), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    save_json(args.output_dir / "training_result.json", {"best_epoch": best_epoch, "best_validation_loss": best,
              "epochs_completed": len(history), "pilot": meta["pilot"], "test_evaluated": False})


def predict(args):
    checkpoint_path = args.output_dir / "best.pt"
    # Our own checkpoint stores np.float64 loss summaries. Allow these exact
    # numeric types without permitting arbitrary pickle execution.
    allowed = [np._core.multiarray.scalar, np.dtype, type(np.dtype("float64")), type(np.dtype("float32"))]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model"] != args.model:
        raise ValueError("Requested model differs from checkpoint")
    meta = checkpoint["provenance"]
    training_source_hash = meta["files_sha256"].get(str(Path(__file__)))
    source_compatibility = None
    if training_source_hash != sha256(__file__):
        snapshot = BASE / "source_snapshots/train_torch_initial.py"
        expected = "e7f1d16803164609accddfb1dff5584c90373b8f29e0e662a99236d709043095"
        if training_source_hash != expected or not snapshot.exists() or sha256(snapshot) != expected:
            raise ValueError("Training/prediction code hash differs without a verified compatible source snapshot")
        source_compatibility = "Verified initial training source retained; prediction-only numeric checkpoint deserialization fix. Architecture, training and prediction computations unchanged."
    for name in ("train", "valid"):
        path = args.cache_dir / f"{name}.regions.tsv"
        if meta["files_sha256"].get(str(path)) != sha256(path):
            raise ValueError("Training/prediction region provenance differs")
    test = CacheDataset(args.cache_dir, "test", False)
    excluded = set(meta["training_chromosomes"]) | set(meta["validation_chromosomes"])
    if excluded & {r["chrom"] for r in test.regions}:
        raise ValueError("Test chromosomes overlap training/validation")
    prediction_dir = args.output_dir / "test_predictions"
    prediction_dir.mkdir(exist_ok=False)
    model = make_model(args.model, args.motif_path, checkpoint["num_cnns"]).cuda()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    counts = np.lib.format.open_memmap(prediction_dir / "prediction_counts.npy", mode="w+", dtype=np.float32, shape=(len(test),))
    profiles = None
    if args.model == "modl":
        profiles = np.lib.format.open_memmap(prediction_dir / "prediction_profiles.npy", mode="w+", dtype=np.float32, shape=(len(test), 1000))
    with torch.no_grad():
        for x, y, indices in loader(test, args):
            with autocast(args):
                count, profile = predictions(model, args.model, x.cuda(non_blocking=True))
            if not bool(torch.isfinite(count).all()):
                raise FloatingPointError("Nonfinite heldout prediction")
            counts[indices.numpy()] = count.cpu().numpy()
            if profiles is not None:
                profiles[indices.numpy()] = profile.cpu().numpy()
    counts.flush()
    if profiles is not None:
        profiles.flush()
    save_json(prediction_dir / "prediction_metadata.json", {
        "model": args.model, "checkpoint_sha256": sha256(checkpoint_path), "pilot": meta["pilot"],
        "training_source_sha256": training_source_hash, "prediction_source_sha256": sha256(__file__),
        "source_compatibility": source_compatibility,
        "prediction_counts_units": "predicted natural_log(1+raw insertions)",
        "prediction_profiles_units": "unit-sum probability shape" if profiles is not None else "N/A-native scalar ExplaiNN",
        "test_rows": len(test), "test_chromosomes": sorted({r["chrom"] for r in test.regions}),
        "test_files_sha256": {str(args.cache_dir / f"test.{s}"): sha256(args.cache_dir / f"test.{s}") for s in ("x.npy", "y.npy", "regions.tsv")},
        "region_order": str(args.cache_dir / "test.regions.tsv"), "test_offset": 500,
        "region_ids_sha256": hashlib.sha256("".join(r["region_id"] + "\n" for r in test.regions).encode()).hexdigest(),
        "target_start": 500, "target_end": 1500, "count_transform": "ln1p_raw_insertion_count",
    })


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("modl", "explainn"), required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--motif-path", type=Path, default=REPO / "artifacts/motifs_with_rc_aligned.pt")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-cnns", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--count-weight", type=float)
    p.add_argument("--negative-ratio", type=float, default=0.1)
    p.add_argument("--gradient-clip", type=float, default=5.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--max-train-steps", type=int, default=0, help="Explicit pilot only")
    p.add_argument("--max-valid-batches", type=int, default=0, help="Explicit pilot only")
    args = p.parse_args()
    if args.batch_size < 2 or args.max_epochs < 1 or args.workers < 0 or args.max_train_steps < 0 or args.max_valid_batches < 0:
        p.error("Invalid batch size, epoch count, workers or pilot caps")
    if args.learning_rate <= 0 or args.negative_ratio < 0 or args.gradient_clip <= 0:
        p.error("Invalid optimizer or sampling arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; refusing CPU fallback for training or predictions")
    if args.amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested bfloat16 unavailable on this GPU")
    args.cache_dir, args.output_dir, args.motif_path = args.cache_dir.resolve(), args.output_dir.resolve(), args.motif_path.resolve()
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    predict(args) if args.predict else fit(args)

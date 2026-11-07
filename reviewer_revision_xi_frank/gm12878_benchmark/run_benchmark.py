#!/usr/bin/env python3
"""Run the fixed fold-0 GPU benchmark sequentially and evaluate automatically."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
CODE = BASE
ROOT = BASE.parents[1]
REPOS = ROOT.parent
TORCH = REPOS / "motif_chip_model/.venv/bin/python"
TF = REPOS / "get_model_seq/.venv/bin/python"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    # Atomic status replacement is confined to this run's generated artifacts.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def summarize(evaluation, path, heading, dataset="GM12878", experiment="ENCSR637XSC"):
    rows = list(csv.DictReader((evaluation / "metrics.tsv").open(), delimiter="\t"))
    text = [f"# {heading}", "", f"{dataset} ATAC-seq {experiment}, published ChromBPNet fold 0. Test chromosomes: chr1, chr3, chr6; validation: chr8, chr20. All eligible released test regions are retained. Estimates below are conditional on one fitted model per method; they do not measure training-seed variability.", "",
            "| Population | Model | Metric | Estimate | 95% block CI |", "| --- | --- | --- | ---: | --- |"]
    wanted = {"count_pearson_ln1p", "count_spearman_ln1p", "count_rmse_ln1p", "profile_js_distance", "profile_pearson_1bp", "profile_pearson_20bp"}
    for row in rows:
        if row["scope"] not in {"all", "peaks", "nonpeaks"} or row["metric"] not in wanted:
            continue
        value = row["value"]
        value = f"{float(value):.4f}" if value not in {"", "None"} else "undefined"
        lo, hi = row["ci_lower"], row["ci_upper"]
        ci = f"[{float(lo):.4f}, {float(hi):.4f}]" if lo not in {"", "None"} else "not computed"
        text.append(f"| {row['scope']} | {row['model']} | {row['metric']} | {value} | {ci} |")
    text += ["", "ExplaiNN uses its native scalar output and has no profile scores. MoDL uses its unchanged fixed-motif architecture with a declared 1000-bp crop and common raw-insertion training objectives. The published ChromBPNet checkpoint is a reference; the separately retrained native ChromBPNet is the matched-training comparator when present. See evaluation.json for exact metric definitions, undefined-profile coverage and paired model-difference confidence intervals.", "",
             "This external human-data benchmark does not replace leakage-free retraining on the manuscript's mouse data. No method was selected or tuned using these test metrics.", ""]
    path.write_text("\n".join(text))


def main():
    global BASE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-dir", type=Path, default=BASE)
    p.add_argument("--dataset", default="GM12878")
    p.add_argument("--experiment", default="ENCSR637XSC")
    p.add_argument("--annotation", default="ENCSR389HIH")
    p.add_argument("--count-weight", type=float, default=93.3)
    p.add_argument("--wait-for-benchmark", type=Path, help="Do not use GPU until this other benchmark is complete or failed")
    p.add_argument("--run-published-inference", action="store_true", help="Run released fold-0 inference as the first GPU stage")
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/cd200_gm12878_benchmark/cache"))
    p.add_argument("--released-model-dir", type=Path, default=Path("/tmp/cd200_gm12878_benchmark/models/fold_0"))
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-fresh-chrombpnet", action="store_true", help="Only the published ChromBPNet reference, explicitly not matched fresh training")
    p.add_argument("--restart-interrupted", action="store_true", help="Reuse completed runs and preserve/restart incomplete fits in new directories")
    args = p.parse_args()
    BASE = args.benchmark_dir.resolve()
    tag = ""
    prior_status = None
    status_path = BASE / "benchmark_status.json"
    if status_path.exists():
        if not args.restart_interrupted:
            p.error("Status file exists; inspect the existing run rather than launch a duplicate GPU queue")
        prior_status = json.loads(status_path.read_text())
        if prior_status.get("state") == "complete":
            p.error("Benchmark already complete")
        pids = [prior_status.get("supervisor_pid")] + [s.get("pid") for s in prior_status.get("stages", []) if s.get("state") == "running"]
        for pid in pids:
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                p.error(f"Recorded process {pid} is still present; refusing a duplicate queue")
        tag = "restart_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prior_status.update(state="interrupted", interruption_detected_utc=stamp(),
                            interruption_reason="Recorded supervisor and running child no longer exist; termination cause is not established")
        archive = BASE / f"benchmark_status.{tag}.json"
        with archive.open("x") as handle:
            json.dump(prior_status, handle, indent=2)
            handle.write("\n")
    cache = json.loads((args.cache_dir / "cache_manifest.json").read_text())
    if cache["status"] != "complete":
        raise ValueError("Shared cache is not complete")
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="8", CUDA_VISIBLE_DEVICES="0")
    status = {"started_utc": stamp(), "supervisor_pid": os.getpid(), "state": "starting", "stages": [],
              "configuration": {"seed": 1234, "epochs_max": args.max_epochs, "early_stopping_patience": 5, "batch_size": args.batch_size,
                                "learning_rate": 0.001, "negative_ratio": 0.1, "count_weight": args.count_weight,
                                "dataset": args.dataset, "experiment": args.experiment, "annotation": args.annotation,
                                "torch_training_precision": "bfloat16 autocast; stored FP32 weights; FP32 inference", "tensorflow_precision": "FP32",
                                "published_reference": f"{args.annotation} fold0 only, no test-fold ensemble", "fresh_native_comparator": not args.skip_fresh_chrombpnet},
              "cache_manifest_sha256": hashlib.sha256((args.cache_dir / "cache_manifest.json").read_bytes()).hexdigest()}
    if prior_status:
        status["prior_status_archive"] = str(archive)
        status["recovery"] = "Completed outputs reused; incomplete training restarts from scratch in a fresh directory because optimizer state was not saved"
    path = BASE / "benchmark_status.json"
    save(path, status)

    def run(name, command):
        log = BASE / "logs" / f"{name}{'_' + tag if tag else ''}.log"
        entry = {"name": name, "started_utc": stamp(), "command": [str(v) for v in command], "log": str(log), "state": "running"}
        status["stages"].append(entry)
        status.update(state="running", current_stage=name, updated_utc=stamp())
        print(f"{stamp()} START {name}", flush=True)
        with log.open("x") as handle:
            process = subprocess.Popen([str(v) for v in command], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            entry["pid"] = process.pid
            save(path, status)
            while process.poll() is None:
                time.sleep(20)
                status["updated_utc"] = stamp()
                argv = [str(v) for v in command]
                history_path = Path(argv[argv.index("--output-dir") + 1]) / "history.json" if "--output-dir" in argv else BASE / "unused_history.json"
                if history_path.exists():
                    try:
                        history = json.loads(history_path.read_text())
                        entry["last_completed_epoch"] = history[-1]
                    except (ValueError, IndexError):
                        pass
                save(path, status)
        entry.update(state="passed" if process.returncode == 0 else "failed", exit_code=process.returncode, finished_utc=stamp())
        save(path, status)
        print(f"{stamp()} END {name} exit={process.returncode}", flush=True)
        if process.returncode:
            raise RuntimeError(f"Stage {name} failed; inspect {log}")

    try:
        if args.wait_for_benchmark:
            while True:
                other = json.loads(args.wait_for_benchmark.read_text())
                if other.get("state") in {"complete", "failed"}:
                    break
                status.update(state="waiting_for_other_benchmark", waiting_for=str(args.wait_for_benchmark), updated_utc=stamp())
                save(path, status)
                time.sleep(20)
        published = BASE / "runs/chrombpnet_published_fold0/test_predictions"
        if args.run_published_inference and not (published / "prediction_metadata.json").exists():
            run("chrombpnet_published_predict", [TF, CODE / "train_chrombpnet.py", "--mode", "released", "--cache-dir", args.cache_dir,
                "--output-dir", published.parent, "--bias-model", args.released_model_dir / f"model.bias_scaled.fold_0.{args.experiment}.h5",
                "--released-nobias", args.released_model_dir / f"model.chrombpnet_nobias.fold_0.{args.experiment}.h5", "--batch-size", args.batch_size])
        # The already-started reference inference owns its output; wait for its
        # completion marker rather than race it or overwrite its arrays.
        started = time.monotonic()
        while not (published / "prediction_metadata.json").exists():
            status.update(state="waiting_for_published_reference", updated_utc=stamp())
            save(path, status)
            if time.monotonic() - started > 3600:
                raise TimeoutError("Published-reference inference has no completion marker after one hour")
            time.sleep(20)
        pred = {"ChromBPNet_published": published}
        for kind, label in (("explainn", "ExplaiNN"), ("modl", "MoDL")):
            output = BASE / "runs" / f"{kind}_seed1234"
            result_path = output / "training_result.json"
            completed = result_path.exists() and not json.loads(result_path.read_text()).get("pilot", True)
            if output.exists() and not completed:
                if not tag:
                    raise FileExistsError(f"Incomplete output exists: {output}")
                output = BASE / "runs" / f"{kind}_seed1234_{tag}"
                status.setdefault("preserved_partial_runs", []).append(str(BASE / "runs" / f"{kind}_seed1234"))
            common = [TORCH, CODE / "train_torch.py", "--model", kind, "--cache-dir", args.cache_dir, "--output-dir", output,
                      "--seed", 1234, "--batch-size", args.batch_size, "--count-weight", args.count_weight, "--workers", 4]
            if not completed:
                run(f"{kind}_train", common + ["--max-epochs", args.max_epochs, "--patience", 5, "--amp"])
            if not (output / "test_predictions/prediction_metadata.json").exists():
                run(f"{kind}_predict", common + ["--predict"])
            else:
                status.setdefault("reused_completed_predictions", []).append(str(output / "test_predictions"))
            pred[label] = output / "test_predictions"
        def evaluate(name, predictions):
            output = BASE / (name + ("_" + tag if tag else ""))
            cmd = [TORCH, CODE / "evaluate.py", "--data-dir", args.cache_dir, "--output", output, "--bootstrap", 2000]
            for label, directory in predictions.items():
                cmd += ["--prediction", f"{label}={directory}"]
            run(name, cmd)
            summarize(output, BASE / f"{name.upper()}.md", f"{args.dataset} held-out prediction results", args.dataset, args.experiment)
        evaluate("evaluation_published_reference", pred)
        status["initial_comparison_available"] = str(BASE / "EVALUATION_PUBLISHED_REFERENCE.md")
        if not args.skip_fresh_chrombpnet:
            output = BASE / "runs/chrombpnet_seed1234"
            if output.exists():
                if not tag:
                    raise FileExistsError(f"Output exists: {output}")
                output = BASE / "runs" / f"chrombpnet_seed1234_{tag}"
            common = [TF, CODE / "train_chrombpnet.py", "--cache-dir", args.cache_dir, "--output-dir", output,
                      "--bias-model", args.released_model_dir / f"model.bias_scaled.fold_0.{args.experiment}.h5",
                      "--count-weight", args.count_weight, "--seed", 1234, "--batch-size", args.batch_size]
            run("chrombpnet_train", common + ["--mode", "train", "--max-epochs", args.max_epochs, "--patience", 5])
            run("chrombpnet_predict", common + ["--mode", "predict"])
            pred["ChromBPNet_fresh"] = output / "test_predictions"
            evaluate("evaluation_matched_training", pred)
            status["matched_comparison_available"] = str(BASE / "EVALUATION_MATCHED_TRAINING.md")
        status.update(state="complete", finished_utc=stamp(), updated_utc=stamp())
        save(path, status)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}", updated_utc=stamp())
        save(path, status)
        raise


if __name__ == "__main__":
    main()

"""Evaluate the four saved 750M checkpoints, with RULER 8K run last."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
ROOT = EXPERIMENTS.parent
RULER = EXPERIMENTS / "ruler-8k-ablations-20261113"
sys.path.insert(0, str(RULER))
from run_ruler_8k import read_json, submit as submit_ruler, summarize as summarize_ruler, write_json


TRAINING = {
    "GPTA2-750M": "gpta2-750M-20261113",
    "RWKV7-750M": "mixers-750M-20261113",
    "KVM256-750M": "mixers-750M-20261113",
    "KVMsqrt-750M": "mixers-750M-20261113",
}
DEFERRED_350M = {
    "350M-E1": "25046-kvm-ruler8k-350M-E1",
    "350M-E2": "25047-kvm-ruler8k-350M-E2",
    "350M-B1": "25048-kvm-ruler8k-350M-B1",
}
SUITES = {"smoke", "short", "niah-4k-8k-16k", "niah-32k", "ruler-4k", "longbench-fewshot", "textbook-32k"}


def prepare():
    control = read_json(EXPERIMENTS / "kvm-sqrt-triton-350M-20261111/eval/manifest.json")
    ruler = read_json(RULER / "manifest.json")
    records = {}
    for name, training_dir in TRAINING.items():
        train_root = EXPERIMENTS / training_dir
        training = read_json(train_root / "manifest.json")
        source = Path(training["source"])
        for file, digest in training["source_files"].items():
            if hashlib.sha256((source / file).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen source changed for {name}: {file}")
        checkpoint = train_root / "runs" / name
        for file in ("config.json", "model.safetensors", "tokenizer.json"):
            if not (checkpoint / file).is_file():
                raise FileNotFoundError(checkpoint / file)
        if hashlib.sha256((checkpoint / "tokenizer.json").read_bytes()).hexdigest() != ruler["tokenizer_sha256"]:
            raise ValueError(f"RULER tokenizer differs for {name}")
        destination = OUTPUT / name
        destination.mkdir(exist_ok=True)
        path = destination / "manifest.json"
        manifest = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": str(source), "source_files": training["source_files"],
            "source_sha256": training["source_sha256"],
            "environment": control["environment"], "nltk_data": control["nltk_data"],
            "protocol": control["protocol"], "models": {name: str(checkpoint)},
            "training_manifest": str(train_root / "manifest.json"),
        }
        if name == "RWKV7-750M":
            # Keep FLA's chunk-state indexing within signed 32-bit range.
            manifest["batch_size_overrides"] = {
                "niah-4k-8k-16k": 8, "niah-32k": 4, "longbench-fewshot": 4,
            }
        if path.is_file():
            previous = read_json(path)
            manifest["created_at"] = previous["created_at"]
            if manifest != previous:
                raise ValueError(f"Existing evaluation manifest differs for {name}")
        else:
            write_json(path, manifest)
        record = {"checkpoint": str(checkpoint), "eval_manifest": str(path)}
        if name == "RWKV7-750M":
            record["batch_size"] = 16
        if ruler["models"].get(name, record) != record:
            raise ValueError(f"Existing RULER record differs for {name}")
        records[name] = record
    ruler["models"].update(records)
    write_json(RULER / "manifest.json", ruler)


def submit():
    for name in TRAINING:
        destination = OUTPUT / name
        marker = destination / "submission.json"
        if marker.is_file():
            continue
        manifest = read_json(destination / "manifest.json")
        source = Path(manifest["source"])
        command = [
            "cluster-run", "--detach", "--num-gpus", "8", "--name", f"kvm-eval-{name}", "--",
            "env", f"UV_PROJECT_ENVIRONMENT={manifest['environment']}", "WANDB_MODE=disabled",
            "uv", "run", "--project", str(source), "--no-sync", "python",
            str(source / "scripts/eval_wave1.py"), "worker", "--output", str(destination),
        ]
        result = subprocess.check_output(command, cwd=ROOT, text=True)
        match = re.search(r"^Submitted (\S+)", result, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not parse cluster submission: {result}")
        write_json(marker, {"job_id": match.group(1), "command": command})
        print(f"{name}: {match.group(1)}", flush=True)


def summary():
    sys.path.insert(0, str(EXPERIMENTS / "hybrid-results-350M-20261111"))
    from plot_hybrids import BENCHMARK_KEYS, evaluation_metrics
    from plot_report import ruler_8k_scores

    scores = ruler_8k_scores()
    columns = ["run", "eval_state", "dclm_val_nll", *BENCHMARK_KEYS, "ruler_8k_pct"]
    pending = OUTPUT / f"metrics-{os.getpid()}.tmp"
    with pending.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for name in TRAINING:
            status = OUTPUT / name / "status.json"
            manifest = read_json(status.with_name("manifest.json"))
            train_log = Path(manifest["models"][name]) / "train.log"
            loss, = re.findall(r"^step:15000/15000 val_loss:([0-9.]+)", train_log.read_text(), re.M)
            writer.writerow({
                "run": name, "eval_state": read_json(status).get("state", "running") if status.is_file() else "queued",
                "dclm_val_nll": loss, **evaluation_metrics(status, name),
                "ruler_8k_pct": scores.get(name, ""),
            })
    pending.replace(OUTPUT / "metrics.csv")


def wait_then_submit_ruler():
    previous_states = None
    while True:
        states = {}
        for name in TRAINING:
            path = OUTPUT / name / "status.json"
            if path.is_file():
                try:
                    states[name] = read_json(path)
                except json.JSONDecodeError:
                    break
        completed = {
            name for name, state in states.items() if state.get("state") == "complete"
            and {stage["suite"] for stage in state["stages"] if stage["state"] == "complete"} == SUITES
        }
        if completed == TRAINING.keys():
            break
        progress = {
            name: (state.get("state", "running"), tuple(
                stage["suite"] for stage in state["stages"] if stage["state"] == "complete"
            )) for name, state in states.items()
        }
        if progress != previous_states:
            print(f"750M standard evaluations: {progress}", flush=True)
            try:
                summary()
            except json.JSONDecodeError:
                pass
            else:
                previous_states = progress
        time.sleep(60)
    summary()
    for name in (*TRAINING, *DEFERRED_350M):
        submit_ruler(name)
    summarize_ruler()
    pending = set(TRAINING) | DEFERRED_350M.keys()
    finished = {}
    while pending:
        for name in sorted(pending):
            path = RULER / name / "status.json"
            if path.is_file():
                state = read_json(path)["state"]
                if state in {"complete", "failed"}:
                    finished[name] = state
                    pending.remove(name)
                    print(f"RULER 8K {name}: {state}", flush=True)
                    summary()
        if all(finished.get(name) == "complete" for name in DEFERRED_350M):
            output = EXPERIMENTS / "ablation-results-350M/kvm-append-policies-350M-vs-triton-K0-ruler-8k.png"
            if not output.is_file():
                subprocess.run([
                    sys.executable, str(output.parent / "plot_append_policies.py"), "--k0", "--ruler-8k",
                ], check=True)
        if pending:
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "summary", "wait-then-submit-ruler"))
    action = parser.parse_args().action
    {
        "prepare": prepare, "submit": submit, "summary": summary,
        "wait-then-submit-ruler": wait_then_submit_ruler,
    }[action]()

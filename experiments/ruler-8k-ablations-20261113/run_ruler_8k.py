"""Submit and record full RULER 8K evaluations with the saved model sources."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
ROOT = EXPERIMENTS.parent
MANIFEST = OUTPUT / "manifest.json"
GROUP = "ruler_8k"
METRIC = "8192,none"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(path)


def submission_path(model):
    return OUTPUT / model / "submission.json"


def submit(model):
    record = read_json(MANIFEST)["models"][model]
    checkpoint = Path(record["checkpoint"])
    if not all((checkpoint / name).is_file() for name in (
        "config.json", "model.safetensors", "tokenizer.json",
    )):
        raise FileNotFoundError(f"Incomplete checkpoint for {model}: {checkpoint}")
    marker = submission_path(model)
    if marker.exists():
        return read_json(marker)["job_id"]
    marker.parent.mkdir(exist_ok=True)
    command = [
        "cluster-run", "--detach", "--num-gpus", "8", "--name", f"kvm-ruler8k-{model}", "--",
        "env", f"UV_PROJECT_ENVIRONMENT={ROOT / '.venv-eval'}", "WANDB_MODE=disabled",
        "uv", "run", "--project", str(ROOT), "--no-sync", "python", str(Path(__file__).resolve()),
        "worker", model,
    ]
    result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    match = re.search(r"^Submitted (\S+)", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse cluster submission: {result.stdout}")
    job = match.group(1)
    write_json(marker, {"job_id": job, "command": command})
    print(f"{model}: {job}", flush=True)
    return job


def score_result(path, tasks):
    result = read_json(path)
    group = result["groups"][GROUP]
    scores = {task: result["results"][task][METRIC] for task in tasks}
    counts = {task: result["n-samples"][task]["effective"] for task in tasks}
    if set(result["group_subtasks"][GROUP]) != set(tasks):
        raise ValueError("RULER result has a different task set")
    if group["sample_count"][METRIC] != 6500 or set(counts.values()) != {500}:
        raise ValueError(f"Unexpected RULER sample counts: {counts}")
    score = group[METRIC]
    if not all(0 <= value <= 1 for value in scores.values()) or not math.isclose(
        score, sum(scores.values()) / len(tasks), abs_tol=1e-10,
    ):
        raise ValueError("RULER group score does not match the task macro average")
    return {"score_pct": 100 * score, "task_pct": {
        task: 100 * value for task, value in scores.items()
    }, "samples": 6500, "result_file": str(path)}


def worker(model):
    manifest = read_json(MANIFEST)
    record = manifest["models"][model]
    reference = read_json(Path(record["eval_manifest"]))
    source = Path(reference["source"])
    target = OUTPUT / model
    destination = target / "results"
    status = {"model": model, "state": "running", "checkpoint": record["checkpoint"]}
    status_path = target / "status.json"
    write_json(status_path, status)
    try:
        tokenizer = Path(record["checkpoint"]) / "tokenizer.json"
        if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != manifest["tokenizer_sha256"]:
            raise ValueError(f"Tokenizer differs from the shared RULER tokenizer for {model}")
        for name, expected in reference["source_files"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen source changed: {name}")
        with tempfile.TemporaryDirectory(prefix="kvm-ruler8k-") as cache:
            env = os.environ | {
                "HF_WRAPPER_BASE_PATH": str(source), "PYTHONPATH": str(source),
                "KVM_EVAL_MODULE_CACHE": f"{cache}/hf-modules",
                "HF_MODULES_CACHE": f"{cache}/hf-modules",
                "TRITON_CACHE_DIR": f"{cache}/triton", "TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor",
                "TORCHINDUCTOR_DIR": f"{cache}/inductor",
                "NLTK_DATA": reference["nltk_data"], "WANDB_MODE": "disabled",
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
            }
            command = [
                sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
                "--num_processes", "8", "--num_machines", "1", "--mixed_precision", "no",
                "--main_process_port", env["MASTER_PORT"], "-m", "wrap_lmeval",
                "--model", "hf", "--model_args",
                f"pretrained={record['checkpoint']},trust_remote_code=True,dtype=bfloat16",
                "--batch_size", str(record.get("batch_size", manifest["batch_size"])),
                "--include_path", str(OUTPUT / "tasks"), "--tasks", GROUP,
                "--seed", manifest["seeds"], "--metadata", '{"max_seq_lengths": [8192]}',
                "--output_path", str(destination), "--log_samples",
            ]
            status["command"] = command
            write_json(status_path, status)
            if "KVM_EVAL_MODULE_CACHE" not in (source / "wrap_lmeval.py").read_text():
                subprocess.run([
                    sys.executable, "-c", "import sys; from transformers import AutoConfig; "
                    "AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)",
                    record["checkpoint"],
                ], cwd=source, env=env, check=True)
            subprocess.run(command, cwd=source, env=env, check=True)
        result_file, = destination.rglob("results_*.json")
        status.update(score_result(result_file, manifest["tasks"]), state="complete")
        print(f"{model}: {status['score_pct']:.4f}% RULER 8K", flush=True)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        write_json(status_path, status)
        summarize()


def submit_after_training(model):
    record = read_json(MANIFEST)["models"][model]
    pipeline = Path(record["training_status"])
    status = OUTPUT / model / "dependency.json"
    status.parent.mkdir(exist_ok=True)
    write_json(status, {"state": "waiting", "training_status": str(pipeline)})
    while True:
        if pipeline.is_file():
            try:
                training = read_json(pipeline)
            except json.JSONDecodeError:
                time.sleep(10)
                continue
            if training.get("final_validation_loss") is not None:
                try:
                    job = submit(model)
                except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                    print(f"Waiting to submit {model}: {exc}", flush=True)
                else:
                    write_json(status, {"state": "submitted", "job_id": job})
                    summarize()
                    return
            elif training["state"] == "failed":
                write_json(status, {"state": "training_failed", "training_status": str(pipeline)})
                raise RuntimeError(f"Training failed for {model}")
        time.sleep(60)


def summarize():
    manifest = read_json(MANIFEST)
    tasks = manifest["tasks"]
    with (OUTPUT / ".summary.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = OUTPUT / "ruler_8k_scores.tmp"
        with pending.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["model", "status", "ruler_8k_pct", *tasks, "result_file"])
            for model in manifest["models"]:
                target = OUTPUT / model
                status_file = target / "status.json"
                status = read_json(status_file) if status_file.is_file() else {
                    "state": "submitted" if submission_path(model).is_file() else "waiting",
                }
                writer.writerow([
                    model, status["state"], status.get("score_pct", ""),
                    *(status.get("task_pct", {}).get(task, "") for task in tasks),
                    status.get("result_file", ""),
                ])
        pending.replace(OUTPUT / "ruler_8k_scores.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "worker", "wait", "summarize"))
    parser.add_argument("models", nargs="*")
    options = parser.parse_args()
    if options.action == "summarize":
        summarize()
    elif options.action == "submit":
        for model in options.models:
            submit(model)
    elif options.action == "worker":
        worker(*options.models)
    else:
        submit_after_training(*options.models)

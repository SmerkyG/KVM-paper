"""Run the paper's evaluation suites on eight GPUs, reusing published baselines.

Launch through cluster-run and the isolated uv evaluation environment. The
driver records commands, per-suite logs, and results locally. W&B is reserved
for training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SHORT = ["lambada_openai", "piqa", "winogrande", "arc_easy", "arc_challenge", "hellaswag"]
NIAH = ["niah_single_1", "niah_single_2", "niah_single_3"]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def prepare_nltk_data():
    """Provision readable sentence-tokenizer resources before reserving GPUs."""
    import nltk

    cache = Path.home() / ".cache" / "kvm-eval-nltk"
    previous_umask = os.umask(0o077)
    try:
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        nltk.data.path.insert(0, str(cache))
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab", download_dir=str(cache), raise_on_error=True)
        assert nltk.sent_tokenize("First sentence. Second sentence.") == [
            "First sentence.", "Second sentence."
        ]
    finally:
        os.umask(previous_umask)
    return str(cache)


def prepare(output, qwen, gpta, resume_from=None, models=None):
    nltk_data = prepare_nltk_data()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    hashes = {}
    for name in sorted(set(filter(None, files))):
        src = ROOT / name
        if src.is_file():
            dst = source / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    manifest = {
        "created_at": now(), "source": str(source), "source_files": hashes,
        "environment": str(ROOT / ".venv-eval"),
        "nltk_data": nltk_data,
        "models": {"Q0": str(qwen.resolve())} if qwen is not None else {},
        "protocol": {
            "harness": "0.4.13", "dtype": "bfloat16", "gpus": 8,
            "seeds": "0,1234,1234,1234", "short_tasks": SHORT,
            "niah_tasks": NIAH, "niah_lengths": [4096, 8192, 16384, 32768],
            "niah_samples_per_task_per_length": 500,
            "ruler": {"length": 4096, "tasks": 13},
            "longbench": "longbench_fewshot (all four tasks)",
            "textbook": {"length": 32768, "max_doc_count": 1000, "seed": 0},
            "max_length_override": None,
        },
    }
    if gpta is not None:
        manifest["models"]["GPTA2"] = str(gpta.resolve())
    for item in models or []:
        name, separator, path = item.partition("=")
        if not separator or not name or not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError("--model requires NAME=/absolute/path/to/checkpoint")
        if name in manifest["models"]:
            raise ValueError(f"Duplicate model name: {name}")
        manifest["models"][name] = str(Path(path).expanduser().resolve())
    if not manifest["models"]:
        raise ValueError("Specify --qwen or --model NAME=CHECKPOINT")
    for model_path in manifest["models"].values():
        for name in ["config.json", "model.safetensors", "tokenizer.json"]:
            if not (Path(model_path) / name).is_file():
                raise FileNotFoundError(Path(model_path) / name)
    if resume_from is not None:
        previous = json.loads((resume_from / "manifest.json").read_text())
        if previous["models"] != manifest["models"] or previous["protocol"] != manifest["protocol"]:
            raise ValueError("Resumption requires identical models and evaluation protocol")
        manifest["reused_stages"] = reuse_completed_stages(resume_from, output)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


def reuse_completed_stages(previous, output):
    status = json.loads((previous / "status.json").read_text())
    stages = []
    for stage in status["stages"]:
        if stage["state"] != "complete" or stage.get("exit_code") != 0:
            continue
        relative = Path(stage["model"], stage["suite"])
        shutil.copytree(previous / relative, output / relative)
        reused = dict(stage, reused_from=str(previous / relative))
        for key in ("log", "result_file"):
            if key in reused:
                reused[key] = str(output / Path(reused[key]).relative_to(previous))
        stages.append(reused)
    return stages


def result_metrics(path):
    files = list(path.rglob("results_*.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one result JSON under {path}, found {len(files)}")
    data = json.loads(files[0].read_text())
    metrics = {}
    for task, values in data["results"].items():
        for metric, value in values.items():
            if isinstance(value, (int, float)) and value >= 0:
                metrics[f"{task}/{metric}"] = value
    return str(files[0]), metrics


def worker(output):
    manifest = json.loads((output / "manifest.json").read_text())
    source = Path(manifest["source"])
    for name, expected in manifest["source_files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source snapshot changed: {name}")
    os.chdir(source)
    env = os.environ.copy()
    env.update({
        "HF_WRAPPER_BASE_PATH": str(source), "PYTHONPATH": str(source),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
        "WANDB_MODE": "disabled",
        "NLTK_DATA": manifest["nltk_data"],
    })
    cache = Path(tempfile.mkdtemp(prefix="kvm-wave1-eval-"))
    env.update({"TRITON_CACHE_DIR": str(cache / "triton"),
                "TORCHINDUCTOR_DIR": str(cache / "inductor"),
                "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                "KVM_EVAL_MODULE_CACHE": str(cache / "hf-modules")})
    status = {"started_at": now(), "logging": "local",
              "stages": manifest.get("reused_stages", [])}
    status_path = output / "status.json"

    def save():
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    def launch(model, suite, tasks=None, lengths=None, batch=32, smoke=False):
        for completed in status["stages"]:
            if (completed["model"] == model and completed["suite"] == suite
                    and completed["state"] == "complete"):
                print(f"REUSE {model} {suite}: {completed['result_file']}", flush=True)
                return 0
        target = output / model / suite
        target.mkdir(parents=True)
        if suite == "textbook-32k":
            command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
                       "--nproc-per-node=8", "--master-addr=127.0.0.1",
                       f"--master-port={os.environ['MASTER_PORT']}", "-m", "loss_plot.eval",
                       "--logs_path", manifest["models"][model],
                       "--eval.dataset", "princeton-nlp/TextbookChapters",
                       "--eval.text_column", "chapter", "--eval.split", "train",
                       "--eval.context_length", "32768", "--eval.max_doc_count", "1000",
                       "--eval.seed", "0", "--eval.num_workers", "0",
                       "--eval.output_path", str(target / "loss.json")]
        else:
            batch = manifest.get("batch_size_overrides", {}).get(suite, batch)
            command = [sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
                       "--num_processes", "8", "--num_machines", "1",
                       "--mixed_precision", "no", "--main_process_port", os.environ["MASTER_PORT"],
                       "-m", "wrap_lmeval", "--model", "hf", "--model_args",
                       f"pretrained={manifest['models'][model]},trust_remote_code=True,dtype=bfloat16",
                       "--batch_size", str(batch), "--tasks", *tasks,
                       "--seed", manifest["protocol"]["seeds"],
                       "--output_path", str(target), "--log_samples"]
            if lengths:
                command += ["--metadata", json.dumps({"max_seq_lengths": lengths})]
            if smoke:
                command += ["--limit", "8"]
        stage = {"model": model, "suite": suite, "command": command,
                 "started_at": now(), "state": "running", "log": str(target / "run.log")}
        status["stages"].append(stage)
        save()
        print(f"START {model} {suite}: {stage['log']}", flush=True)
        started = time.monotonic()
        with (target / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    stage["elapsed_seconds"] = round(time.monotonic() - started, 1)
                    save()
        stage.update(finished_at=now(), exit_code=code,
                     elapsed_seconds=round(time.monotonic() - started, 1),
                     state="complete" if code == 0 else "failed")
        if code == 0 and suite != "textbook-32k":
            stage["result_file"], stage["metrics"] = result_metrics(target)
        elif code == 0:
            stage["result_file"] = str(target / "loss.json")
        save()
        print(f"{stage['state'].upper()} {model} {suite}: {stage['elapsed_seconds']} s", flush=True)
        if code:
            print((target / "run.log").read_text()[-5000:], flush=True)
        return code

    try:
        for model in manifest["models"]:
            if launch(model, "smoke", ["piqa", "niah_single_1"], [4096], batch=2, smoke=True):
                raise RuntimeError(f"{model} evaluation preflight failed")
        suites = [
            ("short", SHORT, None, 32),
            ("niah-4k-8k-16k", NIAH, [4096, 8192, 16384], 32),
            ("niah-32k", NIAH, [32768], 8),
            ("ruler-4k", ["ruler"], [4096], 32),
            ("longbench-fewshot", ["longbench_fewshot"], None, 32),
            ("textbook-32k", None, None, 1),
        ]
        for suite, tasks, lengths, batch in suites:
            for model in manifest["models"]:
                launch(model, suite, tasks, lengths, batch)
        status["state"] = "complete" if all(s["exit_code"] == 0 for s in status["stages"]) else "failed"
        if status["state"] != "complete":
            failed = [f"{s['model']}/{s['suite']}" for s in status["stages"] if s["exit_code"] != 0]
            raise RuntimeError(f"Evaluation suites failed: {', '.join(failed)}")
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        status["finished_at"] = now()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--gpta", type=Path)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output.resolve(), args.qwen, args.gpta,
                args.resume_from.resolve() if args.resume_from else None, args.model)
    else:
        worker(args.output.resolve())

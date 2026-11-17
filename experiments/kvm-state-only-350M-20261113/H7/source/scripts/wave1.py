"""Prepare immutable sources, validate, and submit eight-GPU training jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CONFIGS = [
    "configs/prolong/base.yaml", "configs/prolong/tokens_3B.yaml",
    "configs/prolong/120M/kvm_sqrt16_wd.yaml", "configs/prolong/wave1/common.yaml",
]
DESCRIPTIONS = {
    "K0": "GPTAlpha-2 KVM-sqrt reference",
    "E1": "GPTAlpha-2 KVM-sqrt with random append",
    "B1": "GPTAlpha-2 KVM-sqrt with chunked bipartite grouping",
    "Q0": "Qwen3-style full attention with matched half RoPE",
}
DEFAULT_EXPERIMENTS = tuple(DESCRIPTIONS)
QWEN_LR_EXPERIMENTS = {
    "Q-lr-5e-4": "5e-4",
    "Q-lr-1e-3": "1e-3",
    "Q-lr-4e-3": "4e-3",
    "Q-lr-8e-3": "8e-3",
    "Q-lr-1e-2": "1e-2",
}
DESCRIPTIONS.update({name: f"Qwen3-style full attention, peak LR {lr}"
                     for name, lr in QWEN_LR_EXPERIMENTS.items()})
WAVE2_EXPERIMENTS = {
    "Q1": "Qwen3-style KVM-sqrt, peak LR 4e-3",
    "F0": "KVM-256 with birth-radius readout",
    "F1": "KVM-256 with corrected count readout",
    "A3": "KVM-sqrt with corrected count readout",
}
DESCRIPTIONS.update(WAVE2_EXPERIMENTS)
WAVE3_EXPERIMENTS = {
    "E2": "KVM-sqrt with evenly spaced append selection",
    "A1": "KVM-sqrt without protected sink slots",
    "A2": "KVM-sqrt with unit regional temperatures",
    "A4": "KVM-sqrt with unit merge weights",
}
DESCRIPTIONS.update(WAVE3_EXPERIMENTS)
WAVE4_EXPERIMENTS = {
    "A5": "KVM-sqrt with gate-weight-normalized values",
    "A6": "KVM-sqrt with merges restricted to pre-existing slots",
    "S1": "KVM capped-linear growth to 1448 slots at 4096 tokens",
    "S2": "KVM capped-linear growth to 1448 slots at 8192 tokens",
}
WAVE5_EXPERIMENTS = {
    "H1": "3:1 RoPE sliding window / half-RoPE full attention",
    "H2": "3:1 RWKV-7 / half-RoPE full attention",
    "H3": "3:1 RoPE sliding window / KVM-sqrt",
    "A7": "KVM-sqrt with a learned value radius per head",
}
DESCRIPTIONS.update(WAVE4_EXPERIMENTS)
DESCRIPTIONS.update(WAVE5_EXPERIMENTS)


def config_files(experiment):
    for wave, experiments in ((4, WAVE4_EXPERIMENTS), (5, WAVE5_EXPERIMENTS)):
        if experiment in experiments:
            return CONFIGS + [f"configs/prolong/wave{wave}/{experiment}.yaml"]
    if experiment in WAVE3_EXPERIMENTS:
        return CONFIGS + [f"configs/prolong/wave3/{experiment}.yaml"]
    if experiment in WAVE2_EXPERIMENTS:
        overlay = f"configs/prolong/wave2/{experiment}.yaml"
        if experiment == "Q1":
            return config_files("Q-lr-4e-3") + [overlay]
        if experiment in {"F0", "F1"}:
            return [*CONFIGS[:2], "configs/prolong/120M/kvm_wd.yaml", CONFIGS[-1], overlay]
        return CONFIGS + [overlay]
    if experiment in QWEN_LR_EXPERIMENTS:
        return CONFIGS + ["configs/prolong/wave1/Q0.yaml",
                          f"configs/prolong/qwen_lr/{QWEN_LR_EXPERIMENTS[experiment]}.yaml"]
    return CONFIGS + [f"configs/prolong/wave1/{experiment}.yaml"]


def read_config(experiment):
    from config import load_configs
    from train import CLI_Config
    return CLI_Config(**load_configs(config_files(experiment)))


def model_class(config):
    module, name = config.model.model_class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def inspect_configs(experiments=DEFAULT_EXPERIMENTS):
    import torch
    from model.rwkv7_backbone import MixerConfig
    summary = {}
    for experiment in experiments:
        description = DESCRIPTIONS[experiment]
        config = read_config(experiment)
        with torch.device("meta"):
            model = model_class(config)(MixerConfig(**config.model.model_dump()))
        parameters = sum(p.numel() for p in model.parameters())
        if not all(getattr(p, "label", None) for p in model.parameters()):
            raise ValueError(f"Unlabeled optimizer parameter in {experiment}")
        assert config.train.train_data.device_batch_size == 8
        assert config.train.train_data.sequence_length == 8192
        assert config.train.num_iterations == 6000
        summary[experiment] = {
            "description": description, "parameters": parameters,
            "gpus": 8, "global_batch_tokens": 524288,
            "training_tokens": 3145728000,
            "config_files": config_files(experiment),
            "resolved_config": config.model_dump(),
        }
        del model
    if "K0" in summary:
        reference = summary["K0"]["parameters"]
        for name in ("E1", "B1"):
            if name in summary:
                assert summary[name]["parameters"] == reference
        if "Q0" in summary:
            assert abs(summary["Q0"]["parameters"] / reference - 1) < 0.01
    return summary


def prepare(output, experiments=DEFAULT_EXPERIMENTS, source_overrides=None):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=REPO
    ).decode().split("\0")
    hashes = {}
    for relative in sorted(set(filter(None, files)) | set(source_overrides or ())):
        src = REPO / relative
        overridden = source_overrides is not None and relative in source_overrides
        if not overridden and not src.is_file():
            continue
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if overridden:
            target.write_bytes(source_overrides[relative])
        else:
            shutil.copy2(src, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": str(REPO), "source": str(source),
        "uv_environment": str(REPO / ".venv"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "source_sha256": source_hash, "source_files": hashes,
        "experiments": inspect_configs(experiments),
        "submissions": {},
    }
    from datasets import load_dataset
    from datasets import config as dataset_settings
    from huggingface_hub import constants as hub_settings
    dataset_settings.HF_DATASETS_OFFLINE = True
    hub_settings.HF_HUB_OFFLINE = True
    manifest["datasets"] = {}
    for name in ("Seerkfang/prolong-64k-512-new", "SmerkyG/dclm-10B"):
        dataset = load_dataset(name, split="train")
        manifest["datasets"][name] = {
            "rows": len(dataset), "fingerprint": dataset._fingerprint,
            "cache_files": dataset.cache_files,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPO)
    (output / "source.diff").write_bytes(diff)
    print(json.dumps({"output": str(output), "source_sha256": source_hash,
                      "parameters": {k: v["parameters"] for k, v in manifest["experiments"].items()}}, indent=2))


def load_manifest(output):
    return json.loads((output / "manifest.json").read_text())


def verify_source(manifest):
    source = Path(manifest["source"])
    for relative, expected in manifest["source_files"].items():
        actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source changed: {relative}")


def submit(output, experiments):
    manifest = load_manifest(output)
    verify_source(manifest)
    for experiment in experiments or list(manifest["experiments"]):
        if experiment not in manifest["experiments"]:
            raise ValueError(f"{experiment} is not present in the prepared manifest")
        if experiment in manifest["submissions"]:
            raise ValueError(f"{experiment} has already been submitted")
        command = [
            "cluster-run", "--detach", "--num-gpus", "8", "--name", f"kvm-wave1-{experiment}",
            "--", "env", f"UV_PROJECT_ENVIRONMENT={manifest['uv_environment']}",
            "PYTHONUNBUFFERED=1", f"WANDB_MODE={os.environ.get('WANDB_MODE', 'online')}",
            "uv", "run", "--project", manifest["source"], "--no-sync", "python",
            str(Path(manifest["source"]) / "scripts/wave1.py"),
            "worker", "--output", str(output), "--experiment", experiment,
        ]
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=True)
        print(result.stdout, end="", flush=True)
        match = re.search(r"^Submitted (\S+)", result.stdout, re.MULTILINE)
        if not match:
            raise RuntimeError("Could not identify cluster-run submission: " + result.stdout)
        manifest["submissions"][experiment] = {
            "run_id": match.group(1), "command": command,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "output_dir": str(output / "runs" / experiment),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def worker(output, experiment):
    manifest = load_manifest(output)
    verify_source(manifest)
    os.chdir(manifest["source"])
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    wandb_dir = output / "telemetry" / experiment
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    cache = Path(tempfile.gettempdir()) / f"kvm-wave1-{os.getuid()}" / manifest["source_sha256"][:12]
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ["TORCHINDUCTOR_DIR"] = str(cache / "inductor")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    import torch
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"Expected exactly 8 visible GPUs, found {torch.cuda.device_count()}")
    print(json.dumps({"experiment": experiment, "source_sha256": manifest["source_sha256"],
                      "node": os.environ.get("CLUSTER_RUN_NODE"), "torch": torch.__version__,
                      "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0)}, indent=2), flush=True)
    run_output = output / "runs" / experiment
    arguments = ["uv", "run", "--no-sync", "torchrun", "--nnodes=1", "--nproc-per-node=8",
                 "--master-addr=127.0.0.1", f"--master-port={os.environ['MASTER_PORT']}", "train.py"]
    for path in manifest["experiments"][experiment]["config_files"]:
        arguments.extend(["-c", path])
    arguments.extend(["--train.output_dir", str(run_output)])
    print("Training arguments:", json.dumps(arguments), flush=True)
    raise SystemExit(subprocess.call(arguments))


def smoke(device, context):
    import torch
    from model.rwkv7_backbone import MixerConfig
    torch.set_num_threads(2)
    if device == "cuda" and torch.cuda.device_count() != 8:
        raise RuntimeError("Submit the GPU smoke check as an eight-GPU cluster-run job")
    from model.kvm_experiment_mixer import SequenceMixer
    from model.kvm_mixer import SequenceMixer as ReferenceMixer
    if device == "cuda":
        cfg = read_config("K0")
        with torch.device(device):
            candidate = SequenceMixer(MixerConfig(**cfg.model.model_dump()), 0).bfloat16()
            reference = ReferenceMixer(MixerConfig(**cfg.model.model_dump()), 0).bfloat16()
            q = torch.randn(1, 6, 256, 128, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(1, 6, 1536, 128, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(1, 6, 1536, 128, dtype=torch.bfloat16, requires_grad=True)
        actual, expected = candidate._state_attention(q, k, v), reference._state_attention(q, k, v)
        torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)
        actual_grads = torch.autograd.grad(actual.float().square().mean(), (q, k, v), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.float().square().mean(), (q, k, v))
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=0.1)
        del candidate, reference, q, k, v, actual, expected, actual_grads, expected_grads
        torch.cuda.empty_cache()
        print("Original dense-mask attention output/gradient check passed", flush=True)
    for experiment in DEFAULT_EXPERIMENTS:
        config = read_config(experiment)
        torch.manual_seed(config.train.init_seed)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = model_class(config)(MixerConfig(**config.model.model_dump()))
        torch.set_default_dtype(old_dtype)
        model.train()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.monotonic()
        tokens = torch.randint(0, 50257, (1, context + 1), device=device)
        result = model(tokens[:, :-1], labels=tokens[:, 1:].contiguous(), return_logits=False)
        if not torch.isfinite(result.loss):
            raise RuntimeError(f"Nonfinite smoke-test loss for {experiment}")
        result.loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Nonfinite gradient: {experiment}/{name}")
        if device == "cuda":
            torch.cuda.synchronize()
        print(json.dumps({"experiment": experiment, "context": context,
                          "loss": result.loss.item(), "seconds": time.monotonic() - start,
                          "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}), flush=True)
        del result, model, tokens
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--experiments", nargs="+", choices=DESCRIPTIONS, default=DEFAULT_EXPERIMENTS)
    for name in ("prepare", "submit", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
        if name == "prepare":
            p.add_argument("--experiments", nargs="+", choices=DESCRIPTIONS, default=DEFAULT_EXPERIMENTS)
        if name == "submit":
            p.add_argument("--experiments", nargs="+", choices=DESCRIPTIONS)
        if name == "worker":
            p.add_argument("--experiment", required=True, help="experiment key in the frozen manifest")
    p = sub.add_parser("smoke")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--context", type=int, default=1024)
    args = parser.parse_args()
    if args.action == "inspect":
        print(json.dumps(inspect_configs(args.experiments), indent=2))
    elif args.action == "prepare":
        prepare(args.output, args.experiments)
    elif args.action == "submit":
        submit(args.output, args.experiments)
    elif args.action == "worker":
        worker(args.output, args.experiment)
    elif args.action == "smoke":
        smoke(args.device, args.context)


if __name__ == "__main__":
    main()

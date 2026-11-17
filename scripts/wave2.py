"""Prepare and submit the four 120M Wave 2 jobs with original KVM attention."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess

from wave1 import REPO, WAVE2_EXPERIMENTS, load_manifest, prepare as freeze, verify_source


MIXER = "model/kvm_mixer.py"


def original_with_count_fix(original):
    """Preserve the original attention call; patch only the count-readout branch."""
    edits = (
        ("s_v_norm = s_v / s_vlen", "s_v_norm = (s_v.float() / s_vlen).to(s_v.dtype)"),
        ("torch.ones_like(v_new[..., :1])", "torch.ones_like(v_new[..., :1], dtype=torch.float32)"),
        ("s_vlen = s_vlen + scores.sum(-1, keepdim=True)",
         "s_vlen = s_vlen.float() + scores.float().sum(dim=-2).unsqueeze(-1)"),
        ("torch.ones_like(v[..., :init_state_len, :1])",
         "torch.ones_like(v[..., :init_state_len, :1], dtype=torch.float32)"),
    )
    for before, after in edits:
        if original.count(before) != 1:
            raise ValueError(f"Original mixer does not contain exactly one {before!r}")
        original = original.replace(before, after)
    return original.encode()


def check_pair(left, right, expected):
    differences = {
        f"{section}.{name}"
        for section in ("train", "model")
        for name in left[section].keys() | right[section].keys()
        if left[section].get(name) != right[section].get(name)
    }
    if differences != expected | {"train.wandb_name_suffix"}:
        raise ValueError(f"Unexpected configuration differences: {sorted(differences)}")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(output, kvm_reference, qwen_reference, eval_reference):
    original = load_manifest(kvm_reference)
    qwen = load_manifest(qwen_reference)
    evaluation = load_manifest(eval_reference)
    raw = (Path(original["source"]) / MIXER).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != original["comparison"]["original_mixer_sha256"]:
        raise ValueError("Original K0 mixer does not match the recorded Git source")
    overrides = {MIXER: original_with_count_fix(raw.decode())}
    # Use the same evaluation driver and data/scoring code as the tuned Qwen run.
    for name in ("scripts/eval_wave1.py", "wrap_lmeval.py", "loss_plot/eval.py",
                 "loss_plot/common.py"):
        content = (Path(evaluation["source"]) / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != evaluation["source_files"][name]:
            raise ValueError(f"Evaluation reference changed: {name}")
        overrides[name] = content
    freeze(output, tuple(WAVE2_EXPERIMENTS), source_overrides=overrides)
    manifest = load_manifest(output)
    runs = manifest["experiments"]
    resolved = {name: run["resolved_config"] for name, run in runs.items()}
    check_pair(resolved["F0"], resolved["F1"], {"model.kvm_use_vlens"})
    check_pair(original["experiments"]["K0"]["resolved_config"], resolved["A3"], {"model.kvm_use_vlens"})
    check_pair(qwen["experiments"]["Q-lr-4e-3"]["resolved_config"], resolved["Q1"],
               {"model.model_class_path"})
    if abs(runs["Q1"]["parameters"] / qwen["experiments"]["Q-lr-4e-3"]["parameters"] - 1) >= 0.01:
        raise ValueError("Q1 parameter budget differs from the tuned Qwen control")
    for name, data in manifest["datasets"].items():
        if any(data["fingerprint"] != ref["datasets"][name]["fingerprint"] for ref in (original, qwen)):
            raise ValueError(f"Dataset changed: {name}")
    for name in ("train.py", "model/value_residual_mixin.py", "utils/defer.py"):
        if manifest["source_files"][name] != original["source_files"][name]:
            raise ValueError(f"Unexpected training source change: {name}")
    manifest["comparison"] = {
        "original_kvm_reference": str(kvm_reference),
        "original_mixer_sha256": digest,
        "mixer_changes": "FP32 assignment counts, correct count axis, and count division only",
        "qwen_reference": str(qwen_reference),
        "evaluation_reference": str(eval_reference),
        "config_pairs_verified": ["F0/F1", "K0/A3", "Qwen-4e-3/Q1"],
    }
    for name, run in runs.items():
        target = output / name
        eval_output = target / "eval"
        eval_output.mkdir(parents=True)
        run["evaluation_output"] = str(eval_output)
        experiment = dict(manifest, experiments={name: run}, submissions={})
        write_json(target / "manifest.json", experiment)
        assess = copy.deepcopy(evaluation)
        for key in ("selection", "reused_evaluation_output", "reused_stages"):
            assess.pop(key, None)
        assess.update(
            created_at=manifest["created_at"], source=manifest["source"],
            source_files=manifest["source_files"], models={name: str(target / "runs" / name)},
            checkpoint_pending={"training_output": str(target), "experiment": name},
        )
        write_json(eval_output / "manifest.json", assess)
    write_json(output / "manifest.json", manifest)


def submit(output, wave="wave2"):
    manifest = load_manifest(output)
    for name in manifest["experiments"]:
        if name in manifest["submissions"]:
            continue
        target = output / name
        job = load_manifest(target)
        verify_source(job)
        source = Path(job["source"])
        command = [
            "cluster-run", "--detach", "--num-gpus", "8", "--name", f"kvm-{wave}-{name}",
            "--", "env", f"UV_PROJECT_ENVIRONMENT={job['uv_environment']}",
            "WANDB_MODE=online", "WANDB_ENTITY=smerky", "PYTHONUNBUFFERED=1",
            "uv", "run", "--project", str(source), "--no-sync", "python",
            str(source / "scripts/train_eval_pipeline.py"),
            "--train-output", str(target), "--experiment", name,
            "--eval-output", str(target / "eval"),
        ]
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        print(result.stdout, end="", flush=True)
        match = re.search(r"^Submitted (\S+)", result.stdout, re.MULTILINE)
        if not match:
            raise RuntimeError("Could not identify submitted cluster job")
        row = {
            "run_id": match.group(1), "command": command,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "output_dir": str(target / "runs" / name),
        }
        manifest["submissions"][name] = row
        write_json(output / "manifest.json", manifest)
        job["submissions"][name] = row
        write_json(target / "manifest.json", job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "submit"):
        command = commands.add_parser(action)
        command.add_argument("--output", required=True, type=lambda s: Path(s).resolve())
        if action == "prepare":
            for reference in ("kvm", "qwen", "eval"):
                command.add_argument(f"--{reference}-reference", required=True,
                                     type=lambda s: Path(s).resolve())
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output, args.kvm_reference, args.qwen_reference, args.eval_reference)
    else:
        submit(args.output)

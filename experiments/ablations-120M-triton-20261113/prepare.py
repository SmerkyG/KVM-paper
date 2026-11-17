"""Prepare and submit 120M Triton KVM-sqrt ablations from the saved K0 run."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
ROOT = EXPERIMENTS.parent
RULER = EXPERIMENTS / "ruler-8k-ablations-20261113"
sys.path.insert(0, str(ROOT / "scripts"))
from wave1 import load_manifest, verify_source
from wave2 import check_pair, submit, write_json
from wave3 import read_verified


ABLATIONS = ("A1", "A2", "A3", "A4")
PIPELINE = "scripts/train_eval_pipeline.py"
BASE_PIPELINE = "scripts/standard_train_eval_pipeline.py"
EVAL_FILES = ("wrap_lmeval.py", "loss_plot/common.py", "scripts/eval_wave1.py")
TRAIN_LAUNCHER = "scripts/wave1.py"
PATCH_DEFAULTS = {
    "alt_layer_offset": 0, "kvm_merge_old_slots_only": 0,
    "kvm_use_gate_weight_vlen": 0, "kvm_use_shared_value_radius": 0,
}

CHECK = """
import importlib
import json
import sys

import torch
from config import load_configs
from model.kvm_triton_mixer import SequenceMixer
from model.rwkv7_backbone import MixerConfig
from train import CLI_Config

config = CLI_Config(**load_configs(sys.argv[1:]))
module, name = config.model.model_class_path.rsplit('.', 1)
model_class = getattr(importlib.import_module(module), name)
with torch.device('meta'):
    model = model_class(MixerConfig(**config.model.model_dump()))
layers = model.transformer.h
assert len(layers) == 12 and all(type(layer.attn) is SequenceMixer for layer in layers)
assert all(getattr(p, 'label', None) in config.train.optimization_labels for p in model.parameters())
print(json.dumps({'resolved_config': config.model_dump(),
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'triton_layers': len(layers)}))
"""


def prepare(ablations=ABLATIONS):
    batch_path = OUTPUT / "manifest.json"
    batch = load_manifest(OUTPUT) if batch_path.is_file() else {"experiments": {}, "submissions": {}}
    existing = batch["experiments"].keys() & set(ablations)
    if existing:
        raise FileExistsError(f"Already prepared: {', '.join(sorted(existing))}")
    control_dir = EXPERIMENTS / "kvm-sqrt-triton-20261111"
    control = load_manifest(control_dir)
    evaluation_dir = control_dir / "eval-r2"
    evaluation = load_manifest(evaluation_dir)
    launcher_dir = EXPERIMENTS / "ablations-350M-triton-20261111-r2/A1"
    launcher = load_manifest(launcher_dir)
    recipe = control["experiments"]["K0"]
    base_sources = {name: read_verified(control, name) for name in control["source_files"]}
    base_sources.update({name: read_verified(evaluation, name) for name in EVAL_FILES})
    base_sources[TRAIN_LAUNCHER] = read_verified(launcher, TRAIN_LAUNCHER)
    base_sources[BASE_PIPELINE] = base_sources[PIPELINE]
    base_sources[PIPELINE] = (OUTPUT / "pipeline.py").read_bytes()
    records = {}

    for name in ablations:
        previous_dir = EXPERIMENTS / "ablations-350M-triton-20261111-r2" / name
        previous = load_manifest(previous_dir)
        comparison = previous["comparison"]
        delta = comparison["configuration_delta"]
        patches = comparison["patched_sources"]
        sources = base_sources | {path: read_verified(previous, path) for path in patches}
        if name == "A6":
            test = "tests/test_kvm_triton_ablations.py"
            sources[test] = (ROOT / test).read_bytes()
        overlay = f"configs/prolong/ablation120M/{name}.json"
        sources[overlay] = (json.dumps({
            "model": delta, "train": {"wandb_name_suffix": f"kvmsqrt-120M-triton-{name}"},
        }, indent=2) + "\n").encode()
        target = OUTPUT / name
        source = target / "source"
        source.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for path, content in sources.items():
            destination = source / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.read_bytes() != content:
                    raise ValueError(f"Existing snapshot differs: {destination}")
            else:
                destination.write_bytes(content)
            hashes[path] = hashlib.sha256(content).hexdigest()
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        files = [*recipe["config_files"], overlay]
        verified = json.loads(subprocess.check_output(
            ["uv", "run", "--no-sync", "python", "-c", CHECK, *files], cwd=source,
            env=os.environ | {"UV_PROJECT_ENVIRONMENT": control["uv_environment"]},
            text=True,
        ).splitlines()[-1])
        expected = recipe["resolved_config"]
        if patches:
            expected = expected | {"model": PATCH_DEFAULTS | expected["model"]}
        check_pair(expected, verified["resolved_config"],
                   {f"model.{key}" for key in delta})
        ruler_id = f"120M-KVMsqrt-triton-{name}"
        checkpoint = target / "runs" / name
        eval_dir = target / "eval"
        eval_dir.mkdir(exist_ok=True)
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        experiment = recipe | verified | {
            "description": previous["experiments"][name]["description"].replace("350M", "120M"),
            "config_files": files, "evaluation_output": str(eval_dir),
        }
        if name == "A6":
            experiment["gpu_preflight"] = [
                "tests.test_kvm_triton_ablations.KvmTritonAblationTests."
                "test_a6_keeps_append_selection_and_excludes_new_slots",
            ]
        write_json(target / "manifest.json", control | {
            "created_at": now, "source": str(source), "source_files": hashes,
            "source_sha256": digest, "experiments": {name: experiment}, "submissions": {},
            "comparison": {
                "control": str(control_dir / "manifest.json"), "configuration_delta": delta,
                "patched_sources": patches, "ablation_reference": str(previous_dir / "manifest.json"),
                "launcher_reference": str(launcher_dir / "manifest.json"),
            },
        })
        write_json(eval_dir / "manifest.json", {
            "created_at": now, "source": str(source), "source_files": hashes,
            "source_sha256": digest, "environment": evaluation["environment"],
            "nltk_data": evaluation["nltk_data"], "protocol": evaluation["protocol"],
            "models": {name: str(checkpoint)},
            "checkpoint_pending": {"training_output": str(target), "experiment": name},
            "ruler_8k": {"runner": str(RULER / "run_ruler_8k.py"), "model": ruler_id},
        })
        batch["experiments"][name] = experiment
        records[ruler_id] = {"checkpoint": str(checkpoint), "eval_manifest": str(eval_dir / "manifest.json")}
        print(f"Prepared {name}: {verified['parameters']:,} parameters, {recipe['training_tokens']:,} tokens", flush=True)

    ruler_manifest = load_manifest(RULER)
    for model, record in records.items():
        assert ruler_manifest["models"].get(model, record) == record
        (RULER / model).mkdir(exist_ok=True)
        ruler_manifest["models"][model] = record
    write_json(RULER / "manifest.json", ruler_manifest)
    write_json(batch_path, batch)
    for name in ablations:
        verify_source(load_manifest(OUTPUT / name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit"))
    parser.add_argument("--experiments", nargs="+", choices=(*ABLATIONS, "A6"), default=ABLATIONS)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.experiments)
    else:
        submit(OUTPUT, wave="120M-triton-ablations")

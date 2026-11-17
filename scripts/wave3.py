"""Freeze and queue the 120M KVM-sqrt Wave 3 ablations."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

from wave1 import WAVE3_EXPERIMENTS, load_manifest, prepare as freeze
from wave2 import check_pair, submit, write_json


DELTAS = {
    "E2": {"model.kvm_append_policy"},
    "A1": {"model.sink_len"},
    "A2": {"model.kvm_use_head_temps"},
    "A4": {"model.kvm_use_merge_gate_keys", "model.kvm_use_merge_gate_values"},
}


def read_verified(manifest, name):
    content = (Path(manifest["source"]) / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["source_files"][name]:
        raise ValueError(f"Reference source changed: {name}")
    return content


def prepare(output, kvm_reference, append_reference, eval_reference):
    kvm, append, evaluation = map(load_manifest, (kvm_reference, append_reference, eval_reference))
    eval_paths = ("scripts/eval_wave1.py", "wrap_lmeval.py", "loss_plot/eval.py", "loss_plot/common.py")
    eval_sources = {name: read_verified(evaluation, name) for name in eval_paths}
    output.mkdir(parents=True, exist_ok=False)
    batch = {"experiments": {}, "submissions": {}}
    for name in WAVE3_EXPERIMENTS:
        control_id, control = ("E1", append) if name == "E2" else ("K0", kvm)
        overrides = {path: read_verified(control, path) for path in control["source_files"]
                     if path.startswith(("model/", "data/", "utils/"))
                     or path in {"config.py", "train.py", "trainer.py"}}
        overrides.update(eval_sources)
        target = output / name
        freeze(target, (name,), source_overrides=overrides)
        job = load_manifest(target)
        run = job["experiments"][name]
        resolved = run["resolved_config"]
        reference = control["experiments"][control_id]
        check_pair(reference["resolved_config"], resolved, DELTAS[name])
        if name == "E2":
            check_pair(kvm["experiments"]["K0"]["resolved_config"], resolved,
                       DELTAS[name] | {"model.token_mixer_class_path"})
        for path, content in overrides.items():
            if job["source_files"].get(path) != hashlib.sha256(content).hexdigest():
                raise ValueError(f"Did not freeze the verified source: {path}")
        for dataset, info in job["datasets"].items():
            if info["fingerprint"] != control["datasets"][dataset]["fingerprint"]:
                raise ValueError(f"Dataset changed: {dataset}")
        eval_output = target / "eval"
        eval_output.mkdir()
        assess = copy.deepcopy(evaluation)
        for key in ("selection", "reused_evaluation_output", "reused_stages"):
            assess.pop(key, None)
        assess.update(created_at=job["created_at"], source=job["source"],
                      source_files=job["source_files"],
                      models={name: str(target / "runs" / name)},
                      checkpoint_pending={"training_output": str(target), "experiment": name})
        write_json(eval_output / "manifest.json", assess)
        run["evaluation_output"] = str(eval_output)
        job["comparison"] = {
            "control": control_id,
            "control_source": control["source"],
            "configuration_delta": sorted(DELTAS[name]),
            "mixer_source_sha256": job["source_files"]["model/kvm_mixer.py"],
            "evaluation_reference": str(eval_reference),
        }
        write_json(target / "manifest.json", job)
        batch["experiments"][name] = run
    write_json(output / "manifest.json", batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "submit"):
        command = sub.add_parser(action)
        command.add_argument("--output", required=True, type=lambda p: Path(p).resolve())
        if action == "prepare":
            for reference in ("kvm", "append", "eval"):
                command.add_argument(f"--{reference}-reference", required=True,
                                     type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output, args.kvm_reference, args.append_reference, args.eval_reference)
    else:
        submit(args.output, wave="wave3")

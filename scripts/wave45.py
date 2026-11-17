"""Freeze, validate, and submit the remaining 120M KVM experiments."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys

from wave1 import (
    REPO, WAVE4_EXPERIMENTS, WAVE5_EXPERIMENTS, load_manifest, prepare as freeze,
    verify_source,
)
from wave2 import check_pair, submit, write_json
from wave3 import read_verified


WAVES = {"wave4": WAVE4_EXPERIMENTS, "wave5": WAVE5_EXPERIMENTS}
BACKBONE = "model/rwkv7_backbone.py"
MIXER = "model/kvm_mixer.py"
MODEL_DEFAULTS = {
    "alt_layer_offset": 0,
    "kvm_use_gate_weight_vlen": 0,
    "kvm_use_shared_value_radius": 0,
    "kvm_merge_old_slots_only": 0,
    "state_linear_cap": None,
    "state_linear_end": None,
}
LINEAR = {"model.state_budget_mode", "model.state_linear_cap", "model.state_linear_end"}
HYBRID = {
    "model.token_mixer", "model.alt_token_mixer", "model.alt_layer_every",
    "model.alt_layer_offset",
}
SWA = {"model.rope_partial_dim", "model.swa_window_size"}
DELTAS = {
    "A5": {"model.kvm_use_vlens", "model.kvm_use_gate_weight_vlen"},
    "A6": {"model.kvm_merge_old_slots_only"},
    "S1": LINEAR,
    "S2": LINEAR,
    "H1": HYBRID | SWA,
    "H2": HYBRID,
    "H3": HYBRID | SWA,
    "A7": {"model.kvm_use_shared_value_radius"},
    "H4": HYBRID,
}
LAYOUTS = {
    "H1": ("swa", "gptalpha"),
    "H2": ("rwkv7", "gptalpha"),
    "H3": ("swa", "kvm"),
    "H4": ("rwkv7", "kvm"),
}


def native_mixer(original):
    """Use the exact historical dense-mask SDPA call in the new snapshots."""
    original = original.decode()
    candidate = (REPO / MIXER).read_text()
    attention_begin = "\n        causal_mask = _lower_right_causal_mask("
    attention_end = "\n    def _append_to_state("
    historical_call = original[original.index(attention_begin):original.index(attention_end)]
    start = candidate.index("\n        return self._state_attention(q, k_star, v_star)")
    end = candidate.index(attention_end, start)
    if not candidate[start:end].endswith(historical_call):
        raise ValueError("KVM dense-mask SDPA implementation changed")
    return (candidate[:start] + historical_call + candidate[end:]).encode()


def check_frozen(experiment):
    """Check the model constructed from the snapshot, including hybrid placement."""
    import json
    import torch
    from model.rwkv7_backbone import MixerConfig
    from wave1 import model_class, read_config

    run = read_config(experiment)
    config = MixerConfig(**run.model.model_dump())
    with torch.device("meta"):
        model = model_class(run)(config)
    metadata = {"parameters": sum(p.numel() for p in model.parameters())}
    if experiment in LAYOUTS:
        primary, alternate = LAYOUTS[experiment]
        expected = [primary, primary, primary, alternate] * 3
        actual = [type(b.attn).__module__.split(".")[-1].removesuffix("_mixer")
                  for b in model.transformer.h]
        if actual != expected:
            raise ValueError(f"Invalid hybrid layout: {actual}")
        for layer, mixer in zip(model.transformer.h, expected):
            rotary = model.transformer.alt_rotary if layer.is_alt_layer else model.transformer.rotary
            if mixer == "swa":
                assert rotary.partial_dim == 128 and layer.attn.swa_window_size == 256
            elif mixer == "kvm":
                assert rotary.partial_dim == layer.attn.rope_partial_dim == 64
                assert layer.attn.state_budget_mode == "power_law"
            elif mixer == "gptalpha":
                assert rotary.partial_dim == 64
        metadata["layer_mixers"] = actual
    if experiment in {"S1", "S2"}:
        mixer = model.transformer.h[0].attn
        chunk = mixer.chunk_len
        cap_end = 4096 if experiment == "S1" else 8192
        current = chunk
        metadata["schedule"] = []
        for tokens in range(chunk * mixer.n_bswa_chunks, 32768 + chunk, chunk):
            available = mixer._bswa_begin_for_total_len(tokens + chunk)
            target = max(chunk, min(1448, 1448 * tokens // cap_end))
            slots = mixer._desired_state_len(tokens, available, current)
            assert slots == max(current, min(target, available))
            current = slots
            metadata["schedule"].append({
                "update_after_tokens": tokens,
                "nominal_slots": target,
                "slots_for_next_chunk": slots,
            })
        assert current == 1448
    print(json.dumps(metadata))


def prepare(output, wave, kvm_reference, eval_reference, rwkv_deps=None, experiments=None):
    import json

    selected = experiments or WAVES[wave]
    if set(selected) - WAVES[wave].keys():
        raise ValueError(f"Invalid experiments for {wave}: {selected}")
    kvm, evaluation = map(load_manifest, (kvm_reference, eval_reference))
    reference = copy.deepcopy(kvm["experiments"]["K0"]["resolved_config"])
    reference["model"].update(MODEL_DEFAULTS)
    eval_paths = ("scripts/eval_wave1.py", "wrap_lmeval.py", "loss_plot/eval.py", "loss_plot/common.py")
    baseline = {
        name: read_verified(kvm, name) for name in kvm["source_files"]
        if name.startswith(("model/", "data/", "utils/"))
        or name in {"config.py", "train.py", "trainer.py"}
    }
    baseline.update({name: read_verified(evaluation, name) for name in eval_paths})
    baseline[BACKBONE] = (REPO / BACKBONE).read_bytes()
    ablation_mixer = native_mixer(baseline[MIXER])
    dependencies = {}
    if {"H2", "H4"}.intersection(selected):
        if rwkv_deps is None or not (rwkv_deps / "fla/ops/rwkv7/chunk.py").is_file():
            raise ValueError("Wave 5 needs the pinned RWKV wheels installed to --rwkv-deps")
        sys.path.append(str(rwkv_deps))
        dependencies = {
            str(p.relative_to(rwkv_deps)): p.read_bytes()
            for p in rwkv_deps.rglob("*")
            if p.is_file() and p.suffix != ".pyc" and p.name != ".lock"
        }

    output.mkdir(parents=True, exist_ok=False)
    batch = {"experiments": {}, "submissions": {}}
    for experiment in selected:
        overrides = baseline.copy()
        if experiment not in {"H1", "H2"}:
            overrides[MIXER] = ablation_mixer
        if experiment in {"H2", "H4"}:
            overrides.update(dependencies)
        target = output / experiment
        freeze(target, (experiment,), source_overrides=overrides)
        job = load_manifest(target)
        verify_source(job)
        run = job["experiments"][experiment]
        check_pair(reference, run["resolved_config"], DELTAS[experiment])
        for dataset, info in job["datasets"].items():
            if info["fingerprint"] != kvm["datasets"][dataset]["fingerprint"]:
                raise ValueError(f"Dataset changed: {dataset}")
        for path, content in overrides.items():
            if (Path(job["source"]) / path).read_bytes() != content:
                raise ValueError(f"Did not freeze the intended source: {path}")

        # Validate the actual frozen imports, not the active checkout.
        source = Path(job["source"])
        verified = json.loads(subprocess.check_output(
            ["uv", "run", "--no-sync", "python", "scripts/wave45.py", "check", experiment],
            cwd=source, env=os.environ | {"UV_PROJECT_ENVIRONMENT": job["uv_environment"]},
            text=True,
        ).splitlines()[-1])
        if verified["parameters"] != run["parameters"]:
            raise ValueError(f"Frozen parameter count differs for {experiment}")
        run["validation"] = verified

        eval_output = target / "eval"
        eval_output.mkdir()
        assess = copy.deepcopy(evaluation)
        for key in ("selection", "reused_evaluation_output", "reused_stages"):
            assess.pop(key, None)
        assess.update(created_at=job["created_at"], source=job["source"],
                      source_files=job["source_files"],
                      models={experiment: str(target / "runs" / experiment)},
                      checkpoint_pending={"training_output": str(target), "experiment": experiment})
        write_json(eval_output / "manifest.json", assess)
        run["evaluation_output"] = str(eval_output)
        job["comparison"] = {
            "control": "K0", "control_source": kvm["source"],
            "configuration_delta": sorted(DELTAS[experiment]),
            "evaluation_reference": str(eval_reference),
        }
        write_json(target / "manifest.json", job)
        batch["experiments"][experiment] = run
    write_json(output / "manifest.json", batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check")
    check.add_argument("experiment", choices=DELTAS)
    for action in ("prepare", "submit"):
        command = commands.add_parser(action)
        command.add_argument("--wave", required=True, choices=WAVES)
        command.add_argument("--output", required=True, type=lambda p: Path(p).resolve())
        if action == "prepare":
            command.add_argument("--experiment", action="append", choices=DELTAS)
            for reference in ("kvm", "eval"):
                command.add_argument(f"--{reference}-reference", required=True,
                                     type=lambda p: Path(p).resolve())
            command.add_argument("--rwkv-deps", type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if args.action == "check":
        check_frozen(args.experiment)
    elif args.action == "prepare":
        prepare(args.output, args.wave, args.kvm_reference, args.eval_reference,
                args.rwkv_deps, args.experiment)
    else:
        submit(args.output, wave=args.wave)

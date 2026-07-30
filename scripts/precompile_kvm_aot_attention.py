"""Precompile the two 120M/8K KVM attention-forward specializations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.env import setup_env

setup_env()

import torch
import triton

from model.kernels.kvm_triton_training_kernels import (
    _run_aotriton_source_attention_forward,
    build_mixer_prefill_schedule,
)


def compile_specialization(
    args: SimpleNamespace,
    *,
    initial: bool,
    state_len: int = 0,
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    front_k: torch.Tensor,
    front_v: torch.Tensor,
    temperature: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
):
    if initial:
        params = dict(
            q_start=0,
            query_len=512,
            state_len=0,
            front_start=0,
            front_len=512,
            query_block=128,
            key_block=64,
            num_warps=4,
            waves_per_eu=1,
            is_initial_causal=True,
        )
    else:
        params = dict(
            q_start=512,
            query_len=256,
            state_len=state_len,
            front_start=256,
            front_len=512,
            query_block=64,
            key_block=64,
            num_warps=2,
            waves_per_eu=3,
            is_initial_causal=False,
        )
    return _run_aotriton_source_attention_forward(
        args,
        q_flat=q,
        state_k_attn=state_k,
        state_v_attn=state_v,
        bswa_k_flat=front_k,
        bswa_v_flat=front_v,
        state_temperature=temperature,
        front_temperature=temperature,
        out=out,
        lse=lse,
        **params,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parsed = parser.parse_args()

    schedule = build_mixer_prefill_schedule(
        q_len=8192,
        chunk_len=256,
        n_bswa_chunks=2,
        initial_state_len=256,
        schedule_factor=16.0,
        schedule_exponent=0.5,
        state_min_len=256,
        state_round_down=1,
        max_state_len=2_560_000,
        schedule_mode="power_law",
    )
    args = SimpleNamespace(
        batch=8,
        q_heads=6,
        kv_heads=6,
        q_len=8192,
        dim=128,
        value_dim=128,
    )
    rows = args.batch * args.q_heads
    device = torch.device("cuda")
    tensor_shape = (rows, args.q_len, args.dim)
    q = torch.empty(tensor_shape, device=device, dtype=torch.bfloat16)
    front_k = torch.empty_like(q)
    front_v = torch.empty_like(q)
    state_shape = (rows, schedule.final_state_len, args.dim)
    state_k = torch.empty(state_shape, device=device, dtype=torch.bfloat16)
    state_v = torch.empty_like(state_k)
    temperature = torch.ones(args.q_heads, device=device, dtype=torch.float32)
    out = torch.empty_like(q)
    lse = torch.empty((rows, args.q_len), device=device, dtype=torch.float32)

    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    binaries = {}
    specializations = (
        ("initial", True, 0),
        ("recurrent_aligned", False, 256),
        ("recurrent_unaligned", False, 443),
    )
    for name, initial, state_len in specializations:
        compiled = compile_specialization(
            args,
            initial=initial,
            state_len=state_len,
            q=q,
            state_k=state_k,
            state_v=state_v,
            front_k=front_k,
            front_v=front_v,
            temperature=temperature,
            out=out,
            lse=lse,
        )
        binary_path = parsed.output_dir / f"{name}.hsaco"
        binary_path.write_bytes(compiled.kernel)
        binaries[name] = {
            "path": str(binary_path.resolve()),
            "bytes": len(compiled.kernel),
            "shared": compiled.metadata.shared,
        }

    manifest = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "batch": args.batch,
        "q_len": args.q_len,
        "heads": args.q_heads,
        "head_dim": args.dim,
        "max_state_len": schedule.final_state_len,
        "binaries": binaries,
    }
    (parsed.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Validate precompiled forward binaries at the exact 120M/8K specialization."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.env import setup_env

setup_env()

import torch

from model.kernels.kvm_triton_training_kernels import (
    _run_aotriton_source_attention_forward,
    build_mixer_prefill_schedule,
)
from scripts.diagnose_kvm_aot_schedule_parity import dispatcher_forward


def stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float().flatten()
    candidate_f = candidate.float().flatten()
    difference = candidate_f - reference_f
    return {
        "exact": float((candidate == reference).float().mean()),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rel_l2": float(difference.norm() / reference_f.norm().clamp_min(1e-20)),
    }


def main() -> None:
    batch, heads, total_len, dim = 8, 6, 8192, 128
    rows = batch * heads
    schedule = build_mixer_prefill_schedule(
        q_len=total_len,
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
        batch=batch,
        q_heads=heads,
        kv_heads=heads,
        q_len=total_len,
        dim=dim,
        value_dim=dim,
    )
    torch.manual_seed(20260730)
    shape = (rows, total_len, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    front_k = torch.randn_like(q)
    front_v = torch.randn_like(q)
    state_shape = (rows, schedule.final_state_len, dim)
    state_k = torch.randn(state_shape, device="cuda", dtype=torch.bfloat16)
    state_v = torch.randn_like(state_k)
    state_temperature = torch.linspace(
        0.65, 1.35, heads, device="cuda", dtype=torch.float32
    )
    front_temperature = torch.linspace(
        1.25, 0.75, heads, device="cuda", dtype=torch.float32
    )
    out = torch.full_like(q, float("nan"))
    lse = torch.full((rows, total_len), float("nan"), device="cuda")

    initial_len = 512
    initial_q = q.reshape(batch, heads, total_len, dim)[:, :, :initial_len]
    initial_k = front_k.reshape(batch, heads, total_len, dim)[:, :, :initial_len]
    initial_v = front_v.reshape(batch, heads, total_len, dim)[:, :, :initial_len]
    initial_k = (
        initial_k
        * front_temperature.to(initial_k.dtype).view(1, heads, 1, 1)
    ).to(initial_k.dtype)
    initial_result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        initial_q,
        initial_k,
        initial_v,
        0.0,
        True,
        False,
        scale=1.0 / math.sqrt(float(dim)),
    )
    _run_aotriton_source_attention_forward(
        args,
        q_flat=q,
        state_k_attn=state_k,
        state_v_attn=state_v,
        bswa_k_flat=front_k,
        bswa_v_flat=front_v,
        state_temperature=front_temperature,
        front_temperature=front_temperature,
        out=out,
        lse=lse,
        q_start=0,
        query_len=initial_len,
        state_len=0,
        front_start=0,
        front_len=initial_len,
        query_block=128,
        key_block=64,
        num_warps=4,
        waves_per_eu=1,
        is_initial_causal=True,
    )
    initial_out = out.reshape(batch, heads, total_len, dim)[:, :, :initial_len]
    initial_lse = lse.reshape(batch, heads, total_len)[:, :, :initial_len]

    recurrent_stats = {}
    recurrent_cases = (("aligned", 512, 256, 256), ("unaligned", 768, 443, 512))
    for name, q_start, state_len, front_start in recurrent_cases:
        query_len, front_len = 256, 512
        recurrent_reference, recurrent_reference_lse = dispatcher_forward(
            q_flat=q,
            state_k=state_k,
            state_v=state_v,
            front_k=front_k,
            front_v=front_v,
            state_temperature=state_temperature,
            front_temperature=front_temperature,
            batch=batch,
            heads=heads,
            total_len=total_len,
            q_start=q_start,
            query_len=query_len,
            state_len=state_len,
            front_start=front_start,
            front_len=front_len,
            dim=dim,
        )
        _run_aotriton_source_attention_forward(
            args,
            q_flat=q,
            state_k_attn=state_k,
            state_v_attn=state_v,
            bswa_k_flat=front_k,
            bswa_v_flat=front_v,
            state_temperature=state_temperature,
            front_temperature=front_temperature,
            out=out,
            lse=lse,
            q_start=q_start,
            query_len=query_len,
            state_len=state_len,
            front_start=front_start,
            front_len=front_len,
            query_block=64,
            key_block=64,
            num_warps=2,
            waves_per_eu=3,
            is_initial_causal=False,
        )
        recurrent_out = out.reshape(batch, heads, total_len, dim)[
            :, :, q_start : q_start + query_len
        ]
        recurrent_lse = lse.reshape(batch, heads, total_len)[
            :, :, q_start : q_start + query_len
        ]
        recurrent_stats[f"recurrent_{name}_out"] = stats(
            recurrent_reference, recurrent_out
        )
        recurrent_stats[f"recurrent_{name}_lse"] = stats(
            recurrent_reference_lse, recurrent_lse
        )

    print(
        json.dumps(
            {
                "initial_out": stats(initial_result[0], initial_out),
                "initial_lse": stats(
                    initial_result[1] * math.log2(math.e), initial_lse
                ),
                **recurrent_stats,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

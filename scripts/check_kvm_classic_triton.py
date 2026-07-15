"""Check classic KVM Triton routing and mixer training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.kernels.kvm_classic_triton_training_kernels import (
    MixerPrefillSchedule,
    allocate_work_buffers,
    run_forward_state_update,
)
from model.kvm_classic_mixer import SequenceMixer
from model.rwkv7_backbone import MixerConfigDataclass


def _kernel_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        q_len=args.macro_block,
        initial_state_len=args.state_before,
        max_state_len=args.state_before + args.n_append,
        state_chunk=16,
        group_chunks=12,
        update_token_block=8,
        macro_block=args.macro_block,
        bswa_chunks=2,
        sub_block=128,
        attn_block=64,
        dim=args.dim,
        value_dim=args.value_dim,
        sink_len=1,
        ln_eps=1.0e-5,
        scan_num_warps=8,
        update_num_warps=8,
        attn_num_warps=4,
        waves_per_eu=1,
        scan_waves_per_eu=0,
        update_waves_per_eu=0,
        attn_waves_per_eu=1,
        undo_mode="stash",
        cache_from_rounded_state=True,
        append_policy="global",
        merge_order="append_before_merge",
    )


def check_routes(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    kernel_args = _kernel_args(args)
    rows = args.batch * args.kv_heads
    state_after = args.state_before + args.n_append
    schedule = MixerPrefillSchedule(
        before_by_macro=torch.tensor([args.state_before], dtype=torch.int32),
        after_by_macro=torch.tensor([state_after], dtype=torch.int32),
        n_append_by_macro=torch.tensor([args.n_append], dtype=torch.int32),
        valid_update_by_macro=torch.tensor([1], dtype=torch.int32),
        attention_state_len_by_macro=torch.tensor(
            [args.state_before], dtype=torch.int32
        ),
        front_len=args.macro_block,
        initial_state_len=args.state_before,
        final_state_len=state_after,
        final_state_coverage_len=args.macro_block,
    )
    buffers = allocate_work_buffers(kernel_args, schedule, device)

    state_k = torch.zeros(rows, state_after, args.dim, device=device, dtype=dtype)
    state_v = torch.zeros(
        rows, state_after, args.value_dim, device=device, dtype=dtype
    )
    state_k[:, : args.state_before].normal_()
    state_v[:, : args.state_before].normal_()
    ln_weight = torch.randn(args.dim, device=device, dtype=torch.float32) * 0.1 + 1.0
    ln_bias = torch.randn(args.dim, device=device, dtype=torch.float32) * 0.1
    state_k_attn = torch.zeros_like(state_k)
    state_k_attn[:, : args.state_before] = F.layer_norm(
        state_k[:, : args.state_before].float(),
        (args.dim,),
        ln_weight,
        ln_bias,
        kernel_args.ln_eps,
    ).to(dtype)
    state_v_attn = state_v.clone()
    state_vlen = torch.zeros(rows, state_after, device=device, dtype=torch.float32)
    state_vlen[:, : args.state_before] = torch.linalg.vector_norm(
        state_v[:, : args.state_before].float(), dim=-1
    )

    select_k = torch.randn(
        rows, args.macro_block, args.dim, device=device, dtype=dtype
    )
    append_v = torch.randn(
        rows, args.macro_block, args.value_dim, device=device, dtype=dtype
    )
    gate = (
        1.0
        + 0.25
        * torch.randn(rows, args.macro_block, 1, device=device, dtype=torch.float32)
    ).clamp_min(0.05)
    merge_k = (select_k.float() * gate).to(dtype)
    merge_v = (append_v.float() * gate).to(dtype)
    append_k = merge_k
    append_v = merge_v

    old_state_k_attn = state_k_attn[:, : args.state_before].clone()
    append_pos = torch.full(
        (rows, 1, args.macro_block), -1, device=device, dtype=torch.int32
    )
    best_idx = torch.full_like(append_pos, -1)
    undo_k = torch.empty(
        rows, 1, args.macro_block, args.dim, device=device, dtype=dtype
    )
    undo_v = torch.empty(
        rows, 1, args.macro_block, args.value_dim, device=device, dtype=dtype
    )

    run_forward_state_update(
        kernel_args,
        schedule,
        merge_k,
        merge_v,
        ln_weight,
        ln_bias,
        buffers,
        state_k,
        state_v,
        state_k_attn,
        state_v_attn,
        state_vlen,
        append_pos,
        best_idx,
        undo_k,
        undo_v,
        0,
        True,
        overflow_select_k_flat=select_k,
        overflow_append_k_flat=append_k,
        overflow_append_v_flat=append_v,
        overflow_merge_k_flat=merge_k,
        overflow_merge_v_flat=merge_v,
    )
    torch.cuda.synchronize()

    select_scores = torch.matmul(
        select_k.float(), old_state_k_attn.float().transpose(-1, -2)
    ).amax(dim=-1)
    selected = torch.argsort(select_scores, dim=-1, stable=True)[
        :, : args.n_append
    ].sort(dim=-1).values
    expected_append_pos = torch.full(
        (rows, args.macro_block), -1, device=device, dtype=torch.int32
    )
    expected_append_pos.scatter_(
        1,
        selected,
        torch.arange(
            args.state_before,
            state_after,
            device=device,
            dtype=torch.int32,
        ).expand(rows, -1),
    )
    torch.testing.assert_close(append_pos[:, 0], expected_append_pos, rtol=0, atol=0)

    expected_state_attn = torch.empty(
        rows, state_after, args.dim, device=device, dtype=dtype
    )
    expected_state_attn[:, : args.state_before] = old_state_k_attn
    for row in range(rows):
        selected_append_k = append_k[row, selected[row]]
        expected_state_attn[row, args.state_before :] = F.layer_norm(
            selected_append_k.float(),
            (args.dim,),
            ln_weight,
            ln_bias,
            kernel_args.ln_eps,
        ).to(dtype)
    merge_scores = torch.matmul(
        merge_k.float(), expected_state_attn.float().transpose(-1, -2)
    )
    merge_scores[..., : kernel_args.sink_len] = -torch.inf
    expected_best_idx = merge_scores.argmax(dim=-1).to(torch.int32)
    expected_best_idx.scatter_(1, selected, -1)
    torch.testing.assert_close(best_idx[:, 0], expected_best_idx, rtol=0, atol=0)

    appended_merge_targets = (
        best_idx[:, 0] >= args.state_before
    ).logical_and(append_pos[:, 0] < 0).sum()
    if int(appended_merge_targets) == 0:
        raise AssertionError("test did not exercise append-before-merge visibility")
    print(
        "classic_routes passed",
        {
            "rows": rows,
            "global_appends_per_row": args.n_append,
            "merges_into_new_appends": int(appended_merge_targets),
        },
    )


def check_mixer_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    hidden_size = args.q_heads * args.value_dim
    cfg = MixerConfigDataclass(
        hidden_size=hidden_size,
        num_attention_heads=args.q_heads,
        num_key_value_heads=args.kv_heads,
        d_qk_head=args.dim,
        d_v_head=args.value_dim,
        num_hidden_layers=1,
        chunk_len=args.macro_block,
        n_bswa_chunks=2,
        n_max_d_chunks=64,
        state_budget_mode="power_law",
        state_growth_factor=16.0,
        state_growth_exponent=0.5,
        state_min_len=args.macro_block,
        state_round_down=1,
        kvm_use_merge_gate_keys=1,
        kvm_use_merge_gate_values=1,
        kvm_apply_merge_gate_to_appends=1,
        kvm_use_head_temps=1,
        kvm_use_vlens=1,
    )
    mixer = SequenceMixer(cfg, 0).to(device=device, dtype=torch.bfloat16).train()
    torch.nn.init.normal_(mixer.c_proj.weight, std=0.01)
    q_len = args.macro_block * 4
    q = torch.randn(
        args.batch,
        args.q_heads,
        q_len,
        args.dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        args.batch,
        args.kv_heads,
        q_len,
        args.dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn(
        args.batch,
        args.kv_heads,
        q_len,
        args.value_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    merge_gate = (
        1.0
        + 0.25
        * torch.randn(
            args.batch,
            args.kv_heads,
            q_len,
            1,
            device=device,
            dtype=torch.float32,
        )
    ).clamp_min(0.05)
    out = mixer.forward_prefill(q, k, v, merge_gate, None, None, None)
    out.float().square().mean().backward()
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise AssertionError(f"missing or non-finite {name} gradient")
    print("mixer_training passed", {"out": tuple(out.shape)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--macro-block", type=int, default=256)
    parser.add_argument("--state-before", type=int, default=256)
    parser.add_argument("--n-append", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.macro_block % 128:
        raise ValueError("macro-block must be divisible by 128")
    torch.manual_seed(args.seed)
    check_routes(args)
    check_mixer_training(args)


if __name__ == "__main__":
    main()

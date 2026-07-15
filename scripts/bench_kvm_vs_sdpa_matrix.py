#!/usr/bin/env python3
"""Benchmark SDPA against the current integrated KVM Triton kernels.

This intentionally works at the Q/K/V kernel level. It does not include mixer
projections, token shift, RoPE, MLP, optimizer, or distributed training costs.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
from statistics import median
from typing import Callable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.kernels import kvm_classic_triton_training_kernels  # noqa: E402
from model.kernels import kvm_triton_training_kernels  # noqa: E402
from model.kvm_classic_mixer import _KvmClassicTritonTrainingFunction  # noqa: E402
from model.kvm_triton_mixer import _KvmTritonTrainingFunction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--chunk-len", type=int, default=256)
    parser.add_argument("--bswa-chunks", type=int, default=2)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--state-growth-exponent", type=float, default=0.5)
    parser.add_argument("--state-min-len", type=int, default=256)
    parser.add_argument("--state-round-down", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kvm-semantics",
        choices=("kvm2", "classic"),
        default="kvm2",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("prefill", "generation", "training"),
        default=["prefill", "generation", "training"],
    )
    parser.add_argument(
        "--impls",
        nargs="+",
        choices=("sdpa", "kvm"),
        default=["sdpa", "kvm"],
    )
    return parser.parse_args()


def get_kvm_kernels(args: argparse.Namespace):
    if args.kvm_semantics == "classic":
        return kvm_classic_triton_training_kernels
    return kvm_triton_training_kernels


def get_kvm_training_function(args: argparse.Namespace):
    if args.kvm_semantics == "classic":
        return _KvmClassicTritonTrainingFunction
    return _KvmTritonTrainingFunction


def kvm_impl_name(args: argparse.Namespace, suffix: str) -> str:
    base = "kvm_classic_triton" if args.kvm_semantics == "classic" else "kvm_triton"
    return f"{base}_{suffix}"


def bench_cuda(
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float, float, float, float]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    out = None
    for _ in range(warmup):
        out = fn()
        torch.cuda.synchronize()

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    if isinstance(out, tuple):
        checksum = sum(float(x.float().sum().item()) for x in out if torch.is_tensor(x))
    elif torch.is_tensor(out):
        checksum = float(out.float().sum().item())
    else:
        checksum = 0.0
    return (
        median(times),
        sum(times) / len(times),
        min(times),
        max(times),
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
    )


def print_result(
    *,
    phase: str,
    impl: str,
    head_mode: str,
    seq_len: int,
    batch: int,
    q_heads: int,
    kv_heads: int,
    dim: int,
    value_dim: int,
    state_len: int | None,
    kv_len: int | None,
    median_ms: float,
    mean_ms: float,
    min_ms: float,
    max_ms: float,
    peak_mib: float,
    units: str,
    unit_count: float,
    note: str = "",
) -> None:
    units_per_s = unit_count / (median_ms / 1000.0)
    fields = {
        "phase": phase,
        "impl": impl,
        "head_mode": head_mode,
        "seq_len": seq_len,
        "batch": batch,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "dim": dim,
        "value_dim": value_dim,
        "state_len": -1 if state_len is None else state_len,
        "kv_len": -1 if kv_len is None else kv_len,
        "median_ms": f"{median_ms:.3f}",
        "mean_ms": f"{mean_ms:.3f}",
        "min_ms": f"{min_ms:.3f}",
        "max_ms": f"{max_ms:.3f}",
        "peak_mib": f"{peak_mib:.1f}",
        "units": units,
        "units_per_s": f"{units_per_s:.1f}",
        "note": note,
    }
    print("RESULT " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


def print_error(
    *,
    phase: str,
    impl: str,
    head_mode: str,
    seq_len: int,
    batch: int,
    q_heads: int,
    kv_heads: int,
    exc: BaseException,
) -> None:
    message = str(exc).replace("\n", " ")[:240]
    print(
        "ERROR "
        f"phase={phase} impl={impl} head_mode={head_mode} seq_len={seq_len} "
        f"batch={batch} q_heads={q_heads} kv_heads={kv_heads} "
        f"type={type(exc).__name__} message={message}",
        flush=True,
    )


def make_kvm_args(args: argparse.Namespace, seq_len: int, kv_heads: int) -> argparse.Namespace:
    if seq_len % args.chunk_len:
        raise ValueError("seq_len must be divisible by chunk_len for this benchmark")
    if args.chunk_len % 128 == 0:
        sub_block = 128
    elif args.chunk_len % 64 == 0:
        sub_block = 64
    else:
        raise ValueError("chunk_len must be divisible by 64 or 128")
    return argparse.Namespace(
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=kv_heads,
        q_len=seq_len,
        logical_q_len=seq_len,
        initial_state_len=min(seq_len, args.chunk_len),
        max_state_len=0,
        state_chunk=16,
        group_chunks=12,
        update_token_block=8,
        macro_block=args.chunk_len,
        bswa_chunks=args.bswa_chunks,
        sub_block=sub_block,
        attn_block=64,
        schedule_factor=args.state_growth_factor,
        schedule_exponent=args.state_growth_exponent,
        state_min_len=args.state_min_len,
        state_round_down=args.state_round_down,
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
        q_head_loop_unroll_factor=4,
        skip_temperature_grad=False,
        skip_temperature_atomic=False,
        temperature_grad_backend="atomic",
        update_grad_backend="triton",
        attn_grad_backend="kv-owned",
        undo_mode="stash",
        cache_from_rounded_state=True,
        reconstruct_live_state_backward=True,
        fuse_restore_refresh=True,
        fuse_state_dkdv_raw=False,
        append_policy="global" if args.kvm_semantics == "classic" else "subblock_quota",
        merge_order=(
            "append_before_merge"
            if args.kvm_semantics == "classic"
            else "merge_before_append"
        ),
    )


def make_head_mode(q_heads: int, kv_heads: int) -> str:
    return "mha" if q_heads == kv_heads else f"gqa{q_heads // kv_heads}"


def make_sdpa_inputs(
    args: argparse.Namespace,
    seq_len: int,
    kv_heads: int,
    *,
    q_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_len = seq_len if q_len is None else q_len
    dtype = torch.bfloat16
    device = torch.device("cuda")
    q = torch.randn(
        args.batch, args.q_heads, q_len, args.dim, device=device, dtype=dtype
    )
    k = torch.randn(args.batch, kv_heads, seq_len, args.dim, device=device, dtype=dtype)
    v = torch.randn(
        args.batch, kv_heads, seq_len, args.value_dim, device=device, dtype=dtype
    )
    return q, k, v


def bench_sdpa_prefill(args: argparse.Namespace, seq_len: int, kv_heads: int) -> None:
    q, k, v = make_sdpa_inputs(args, seq_len, kv_heads)
    enable_gqa = args.q_heads != kv_heads
    run = lambda: F.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=enable_gqa
    )
    med, mean, min_t, max_t, peak = bench_cuda(
        run, warmup=args.warmup, iters=args.iters
    )
    print_result(
        phase="prefill",
        impl="sdpa_torch_fwd",
        head_mode=make_head_mode(args.q_heads, kv_heads),
        seq_len=seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=kv_heads,
        dim=args.dim,
        value_dim=args.value_dim,
        state_len=None,
        kv_len=seq_len,
        median_ms=med,
        mean_ms=mean,
        min_ms=min_t,
        max_ms=max_t,
        peak_mib=peak,
        units="tokens",
        unit_count=float(args.batch * seq_len),
        note="torch_sdpa_prefill",
    )


def bench_sdpa_generation(args: argparse.Namespace, seq_len: int, kv_heads: int) -> None:
    q, k, v = make_sdpa_inputs(args, seq_len, kv_heads, q_len=1)
    enable_gqa = args.q_heads != kv_heads
    med, mean, min_t, max_t, peak = bench_cuda(
        lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=enable_gqa
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    print_result(
        phase="generation",
        impl="sdpa_torch_decode_full",
        head_mode=make_head_mode(args.q_heads, kv_heads),
        seq_len=seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=kv_heads,
        dim=args.dim,
        value_dim=args.value_dim,
        state_len=None,
        kv_len=seq_len,
        median_ms=med,
        mean_ms=mean,
        min_ms=min_t,
        max_ms=max_t,
        peak_mib=peak,
        units="tokens",
        unit_count=float(args.batch),
        note="one_decode_token",
    )


def bench_sdpa_training(args: argparse.Namespace, seq_len: int, kv_heads: int) -> None:
    dtype = torch.bfloat16
    device = torch.device("cuda")
    q0 = torch.randn(
        args.batch,
        args.q_heads,
        seq_len,
        args.dim,
        device=device,
        dtype=dtype,
    )
    k0 = torch.randn(args.batch, kv_heads, seq_len, args.dim, device=device, dtype=dtype)
    v0 = torch.randn(
        args.batch, kv_heads, seq_len, args.value_dim, device=device, dtype=dtype
    )
    dout = torch.randn(
        args.batch,
        args.q_heads,
        seq_len,
        args.value_dim,
        device=device,
        dtype=dtype,
    )
    enable_gqa = args.q_heads != kv_heads

    def run() -> torch.Tensor:
        q = q0.detach().requires_grad_(True)
        k = k0.detach().requires_grad_(True)
        v = v0.detach().requires_grad_(True)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=enable_gqa
        )
        loss = (out.float() * dout.float()).sum()
        loss.backward()
        return out

    med, mean, min_t, max_t, peak = bench_cuda(
        run, warmup=args.warmup, iters=args.iters
    )
    print_result(
        phase="training",
        impl="sdpa_torch_fwd_bwd",
        head_mode=make_head_mode(args.q_heads, kv_heads),
        seq_len=seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=kv_heads,
        dim=args.dim,
        value_dim=args.value_dim,
        state_len=None,
        kv_len=seq_len,
        median_ms=med,
        mean_ms=mean,
        min_ms=min_t,
        max_ms=max_t,
        peak_mib=peak,
        units="tokens",
        unit_count=float(args.batch * seq_len),
        note="autograd_backend",
    )


def make_kvm_streams(
    args: argparse.Namespace, triton_args: argparse.Namespace
) -> dict[str, torch.Tensor]:
    dtype = torch.bfloat16
    device = torch.device("cuda")
    q_rows = args.batch * args.q_heads
    kv_rows = args.batch * triton_args.kv_heads
    q = torch.randn(q_rows, triton_args.q_len, args.dim, device=device, dtype=dtype)
    bswa_k = torch.randn(kv_rows, triton_args.q_len, args.dim, device=device, dtype=dtype)
    bswa_v = torch.randn(
        kv_rows, triton_args.q_len, args.value_dim, device=device, dtype=dtype
    )
    select_k = torch.randn_like(bswa_k)
    append_k = torch.randn_like(bswa_k)
    append_v = torch.randn_like(bswa_v)
    merge_k = torch.randn_like(bswa_k)
    merge_v = torch.randn_like(bswa_v)
    initial_k = torch.randn_like(bswa_k)
    initial_v = torch.randn_like(bswa_v)
    return {
        "q": q,
        "bswa_k": bswa_k,
        "bswa_v": bswa_v,
        "select_k": select_k,
        "append_k": append_k,
        "append_v": append_v,
        "merge_k": merge_k,
        "merge_v": merge_v,
        "initial_k": initial_k,
        "initial_v": initial_v,
        "ln_weight": torch.ones(args.dim, device=device, dtype=torch.float32),
        "ln_bias": torch.zeros(args.dim, device=device, dtype=torch.float32),
        "state_temperature": torch.ones(args.q_heads, device=device, dtype=torch.float32),
        "front_temperature": torch.ones(args.q_heads, device=device, dtype=torch.float32),
    }


def bench_kvm_prefill(args: argparse.Namespace, seq_len: int, kv_heads: int) -> None:
    kernels = get_kvm_kernels(args)
    triton_args = make_kvm_args(args, seq_len, kv_heads)
    schedule = kernels.make_schedule(triton_args)
    streams = make_kvm_streams(args, triton_args)

    def run() -> torch.Tensor:
        with torch.no_grad():
            forward = kernels.build_prefill_forward(
                triton_args,
                schedule,
                streams["q"],
                streams["merge_k"],
                streams["merge_v"],
                streams["bswa_k"],
                streams["bswa_v"],
                streams["ln_weight"],
                streams["ln_bias"],
                streams["state_temperature"],
                streams["front_temperature"],
                initial_k_flat=streams["initial_k"],
                initial_v_flat=streams["initial_v"],
                overflow_select_k_flat=streams["select_k"],
                overflow_append_k_flat=streams["append_k"],
                overflow_append_v_flat=streams["append_v"],
                overflow_merge_k_flat=streams["merge_k"],
                overflow_merge_v_flat=streams["merge_v"],
            )
            return forward["out"]

    med, mean, min_t, max_t, peak = bench_cuda(
        run, warmup=args.warmup, iters=args.iters
    )
    print_result(
        phase="prefill",
        impl=kvm_impl_name(args, "fwd"),
        head_mode=make_head_mode(args.q_heads, kv_heads),
        seq_len=seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=kv_heads,
        dim=args.dim,
        value_dim=args.value_dim,
        state_len=int(schedule.final_state_len),
        kv_len=int(schedule.final_state_len) + args.bswa_chunks * args.chunk_len,
        median_ms=med,
        mean_ms=mean,
        min_ms=min_t,
        max_ms=max_t,
        peak_mib=peak,
        units="tokens",
        unit_count=float(args.batch * seq_len),
        note="sqrt16_schedule",
    )


def bench_kvm_training(args: argparse.Namespace, seq_len: int, kv_heads: int) -> None:
    kernels = get_kvm_kernels(args)
    training_function = get_kvm_training_function(args)
    triton_args = make_kvm_args(args, seq_len, kv_heads)
    schedule = kernels.make_schedule(triton_args)
    streams = make_kvm_streams(args, triton_args)
    dout = torch.randn(
        args.batch * args.q_heads,
        triton_args.q_len,
        args.value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    def run() -> torch.Tensor:
        q = streams["q"].detach().requires_grad_(True)
        bswa_k = streams["bswa_k"].detach().requires_grad_(True)
        bswa_v = streams["bswa_v"].detach().requires_grad_(True)
        select_k = streams["select_k"].detach().requires_grad_(True)
        append_k = streams["append_k"].detach().requires_grad_(True)
        append_v = streams["append_v"].detach().requires_grad_(True)
        merge_k = streams["merge_k"].detach().requires_grad_(True)
        merge_v = streams["merge_v"].detach().requires_grad_(True)
        initial_k = streams["initial_k"].detach().requires_grad_(True)
        initial_v = streams["initial_v"].detach().requires_grad_(True)
        ln_weight = streams["ln_weight"].detach().requires_grad_(True)
        ln_bias = streams["ln_bias"].detach().requires_grad_(True)
        state_temperature = streams["state_temperature"].detach().requires_grad_(True)
        front_temperature = streams["front_temperature"].detach().requires_grad_(True)
        out = training_function.apply(
            q,
            bswa_k,
            bswa_v,
            select_k,
            merge_v,
            select_k,
            append_k,
            append_v,
            merge_k,
            merge_v,
            initial_k,
            initial_v,
            ln_weight,
            ln_bias,
            state_temperature,
            front_temperature,
            triton_args,
            schedule,
        )
        loss = (out.float() * dout.float()).sum()
        loss.backward()
        return out

    med, mean, min_t, max_t, peak = bench_cuda(
        run, warmup=args.warmup, iters=args.iters
    )
    print_result(
        phase="training",
        impl=kvm_impl_name(args, "fwd_bwd"),
        head_mode=make_head_mode(args.q_heads, kv_heads),
        seq_len=seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=kv_heads,
        dim=args.dim,
        value_dim=args.value_dim,
        state_len=int(schedule.final_state_len),
        kv_len=int(schedule.final_state_len) + args.bswa_chunks * args.chunk_len,
        median_ms=med,
        mean_ms=mean,
        min_ms=min_t,
        max_ms=max_t,
        peak_mib=peak,
        units="tokens",
        unit_count=float(args.batch * seq_len),
        note="sqrt16_reconstruct_live",
    )


def bench_kvm_generation(args: argparse.Namespace, seq_len: int, kv_heads: int) -> None:
    kernels = get_kvm_kernels(args)
    triton_args = make_kvm_args(args, seq_len, kv_heads)
    prefill_schedule = kernels.make_schedule(triton_args)
    current_state_len = int(prefill_schedule.final_state_len)
    state_coverage_len = int(prefill_schedule.final_state_coverage_len)
    state_after = kernels.desired_state_len(
        args.bswa_chunks * args.chunk_len + state_coverage_len,
        state_coverage_len + args.chunk_len,
        current_state_len,
        args.state_growth_factor,
        args.state_growth_exponent,
        args.state_min_len,
        args.state_round_down,
        1 << 30,
    )
    n_append = min(max(state_after - current_state_len, 0), args.chunk_len)
    state_after = current_state_len + n_append
    update_args = make_kvm_args(args, args.chunk_len, kv_heads)
    update_args.max_state_len = state_after
    update_args.initial_state_len = current_state_len
    schedule = kernels.MixerPrefillSchedule(
        before_by_macro=torch.tensor([current_state_len], dtype=torch.int32),
        after_by_macro=torch.tensor([state_after], dtype=torch.int32),
        n_append_by_macro=torch.tensor([n_append], dtype=torch.int32),
        valid_update_by_macro=torch.tensor([1], dtype=torch.int32),
        attention_state_len_by_macro=torch.tensor([current_state_len], dtype=torch.int32),
        front_len=args.chunk_len,
        initial_state_len=current_state_len,
        final_state_len=state_after,
        final_state_coverage_len=0,
    )

    dtype = torch.bfloat16
    device = torch.device("cuda")
    kv_rows = args.batch * kv_heads
    state_k = torch.randn(kv_rows, state_after, args.dim, device=device, dtype=dtype)
    state_v = torch.randn(kv_rows, state_after, args.value_dim, device=device, dtype=dtype)
    state_k_attn = torch.randn_like(state_k)
    state_v_attn = torch.randn_like(state_v)
    state_vlen = torch.ones(kv_rows, state_after, device=device, dtype=torch.float32)
    overflow_k = torch.randn(kv_rows, args.chunk_len, args.dim, device=device, dtype=dtype)
    overflow_v = torch.randn(
        kv_rows, args.chunk_len, args.value_dim, device=device, dtype=dtype
    )
    select_k = torch.randn_like(overflow_k)
    append_k = torch.randn_like(overflow_k)
    append_v = torch.randn_like(overflow_v)
    merge_k = torch.randn_like(overflow_k)
    merge_v = torch.randn_like(overflow_v)
    ln_weight = torch.ones(args.dim, device=device, dtype=torch.float32)
    ln_bias = torch.zeros(args.dim, device=device, dtype=torch.float32)
    buffers = kernels.allocate_work_buffers(update_args, schedule, device)
    append_pos_by_token = torch.full(
        (kv_rows, 1, args.chunk_len), -1, device=device, dtype=torch.int32
    )
    best_idx_by_token = torch.full_like(append_pos_by_token, -1)
    undo_k_by_token = torch.empty(
        kv_rows, 1, args.chunk_len, args.dim, device=device, dtype=dtype
    )
    undo_v_by_token = torch.empty(
        kv_rows, 1, args.chunk_len, args.value_dim, device=device, dtype=dtype
    )

    def run_update() -> torch.Tensor:
        kernels.run_forward_state_update(
            update_args,
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
            append_pos_by_token,
            best_idx_by_token,
            undo_k_by_token,
            undo_v_by_token,
            0,
            False,
            overflow_select_k_flat=select_k,
            overflow_append_k_flat=append_k,
            overflow_append_v_flat=append_v,
            overflow_merge_k_flat=merge_k,
            overflow_merge_v_flat=merge_v,
        )
        return state_k

    med, mean, min_t, max_t, peak = bench_cuda(
        run_update, warmup=args.warmup, iters=args.iters
    )
    print_result(
        phase="generation",
        impl=kvm_impl_name(args, "update_256"),
        head_mode=make_head_mode(args.q_heads, kv_heads),
        seq_len=seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=kv_heads,
        dim=args.dim,
        value_dim=args.value_dim,
        state_len=current_state_len,
        kv_len=args.chunk_len,
        median_ms=med,
        mean_ms=mean,
        min_ms=min_t,
        max_ms=max_t,
        peak_mib=peak,
        units="update_tokens",
        unit_count=float(args.batch * args.chunk_len),
        note=f"n_append={n_append}",
    )
    print_result(
        phase="generation",
        impl=kvm_impl_name(args, "update_amortized"),
        head_mode=make_head_mode(args.q_heads, kv_heads),
        seq_len=seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=kv_heads,
        dim=args.dim,
        value_dim=args.value_dim,
        state_len=current_state_len,
        kv_len=args.chunk_len,
        median_ms=med / args.chunk_len,
        mean_ms=mean / args.chunk_len,
        min_ms=min_t / args.chunk_len,
        max_ms=max_t / args.chunk_len,
        peak_mib=peak,
        units="tokens",
        unit_count=float(args.batch),
        note=f"per_decode_token_from_256_update,n_append={n_append}",
    )

    # Current decode attention is SDPA-shaped over compressed state + BSWA. Time
    # the corresponding PyTorch SDPA call separately from the periodic update.
    decode_kv_len = current_state_len + args.bswa_chunks * args.chunk_len
    q, k, v = make_sdpa_inputs(args, decode_kv_len, kv_heads, q_len=1)
    enable_gqa = args.q_heads != kv_heads
    med, mean, min_t, max_t, peak = bench_cuda(
        lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=enable_gqa
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    print_result(
        phase="generation",
        impl="kvm_shape_sdpa_torch_decode",
        head_mode=make_head_mode(args.q_heads, kv_heads),
        seq_len=seq_len,
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=kv_heads,
        dim=args.dim,
        value_dim=args.value_dim,
        state_len=current_state_len,
        kv_len=decode_kv_len,
        median_ms=med,
        mean_ms=mean,
        min_ms=min_t,
        max_ms=max_t,
        peak_mib=peak,
        units="tokens",
        unit_count=float(args.batch),
        note="state_plus_bswa_attention_shape",
    )


def run_case(
    args: argparse.Namespace,
    *,
    seq_len: int,
    kv_heads: int,
    phase: str,
    impl: str,
) -> None:
    head_mode = make_head_mode(args.q_heads, kv_heads)
    try:
        if impl == "sdpa" and phase == "prefill":
            bench_sdpa_prefill(args, seq_len, kv_heads)
        elif impl == "sdpa" and phase == "generation":
            bench_sdpa_generation(args, seq_len, kv_heads)
        elif impl == "sdpa" and phase == "training":
            bench_sdpa_training(args, seq_len, kv_heads)
        elif impl == "kvm" and phase == "prefill":
            bench_kvm_prefill(args, seq_len, kv_heads)
        elif impl == "kvm" and phase == "generation":
            bench_kvm_generation(args, seq_len, kv_heads)
        elif impl == "kvm" and phase == "training":
            bench_kvm_training(args, seq_len, kv_heads)
        else:
            raise AssertionError(f"unhandled case phase={phase} impl={impl}")
    except BaseException as exc:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print_error(
            phase=phase,
            impl=impl,
            head_mode=head_mode,
            seq_len=seq_len,
            batch=args.batch,
            q_heads=args.q_heads,
            kv_heads=kv_heads,
            exc=exc,
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required")
    if args.dim != args.value_dim:
        raise ValueError("This benchmark expects dim == value_dim for SDPA kernels")
    if args.q_heads % min(args.kv_heads) != 0:
        raise ValueError("q_heads must be divisible by kv_heads")

    torch.manual_seed(args.seed)
    print("torch", torch.__version__, flush=True)
    print("device", torch.cuda.get_device_name(), flush=True)
    print(
        "settings",
        {
            "batch": args.batch,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "seq_lens": args.seq_lens,
            "dim": args.dim,
            "value_dim": args.value_dim,
            "chunk_len": args.chunk_len,
            "bswa_chunks": args.bswa_chunks,
            "state_schedule": "sqrt16",
            "kvm_semantics": args.kvm_semantics,
            "sdpa_backend": "torch",
            "warmup": args.warmup,
            "iters": args.iters,
        },
        flush=True,
    )

    for seq_len in args.seq_lens:
        for kv_heads in args.kv_heads:
            if args.q_heads % kv_heads:
                print(
                    f"ERROR phase=all impl=all head_mode=invalid seq_len={seq_len} "
                    f"batch={args.batch} q_heads={args.q_heads} kv_heads={kv_heads} "
                    "type=ValueError message=q_heads_must_be_divisible_by_kv_heads",
                    flush=True,
                )
                continue
            for phase in args.phases:
                for impl in args.impls:
                    run_case(args, seq_len=seq_len, kv_heads=kv_heads, phase=phase, impl=impl)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare decode routing from centroid sums versus cached BF16 means."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

from model.kernels.paged_leaf_attention import (
    _decode_route_coarse_gqa_groups_fixed_prepare_kernel,
    _pack_route_score_index,
    _unpack_route_score_index,
)


@triton.jit
def _score_cached_mean_bias_kernel(
    q,
    mean_k,
    counts,
    log_counts,
    output_scores,
    output_indices,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    GROUP_N: tl.constexpr,
    USE_CACHED_BIAS: tl.constexpr,
):
    """Isolate the cost of computing log(count) in cached-mean routing."""
    sequence = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    kv_head = sequence % KV_HEADS
    batch = sequence // KV_HEADS
    lane = tl.arange(0, GROUP_N)
    slot = group * GROUP_N + lane
    valid = slot < state_len
    count = tl.load(
        counts + sequence * state_len + slot,
        mask=valid,
        other=1.0,
    ).to(tl.float32)
    valid &= (count > 0.0) & (count < 1024.0)
    dimension = tl.arange(0, HEAD_DIM)
    keys = tl.load(
        mean_k
        + (sequence * state_len + slot[:, None]) * HEAD_DIM
        + dimension[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    query_lane = tl.arange(0, 16)
    query_valid = query_lane < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_lane
    query_row = batch * QUERY_HEADS + query_head
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + dimension[None, :],
        mask=query_valid[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * SCALE
    if USE_CACHED_BIAS:
        bias = tl.load(
            log_counts + sequence * state_len + slot,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
    else:
        bias = tl.log(count)
    scores += bias[None, :]
    scores = tl.where(
        query_valid[:, None] & valid[None, :], scores, -float("inf")
    )
    packed = _pack_route_score_index(scores, slot[None, :])
    top_scores, top_indices = _unpack_route_score_index(
        tl.topk(packed, 8, dim=1)
    )
    rank = tl.arange(0, 8)
    base = (query_row * tl.cdiv(state_len, GROUP_N) + group) * 8
    tl.store(
        output_scores + base[:, None] + rank[None, :],
        top_scores,
        mask=query_valid[:, None],
    )
    tl.store(
        output_indices + base[:, None] + rank[None, :],
        top_indices,
        mask=query_valid[:, None],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(20260829)
    device = torch.device("cuda")
    batch, kv_heads, gqa, head_dim = 1, 2, 4, 256
    query_heads = kv_heads * gqa
    group_n = 32
    groups = triton.cdiv(args.slots, group_n)
    rows = batch * query_heads
    sequences = batch * kv_heads
    union_capacity = gqa * 8
    mask_capacity = args.slots

    q = torch.randn(
        batch, query_heads, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    state_sum_k = torch.randn(
        batch,
        kv_heads,
        args.slots,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    state_v = torch.empty_like(state_sum_k)
    counts = torch.randint(
        1,
        257,
        (batch, kv_heads, args.slots, 1),
        dtype=torch.int32,
        device=device,
    ).to(torch.float32)
    state_mean_k = (state_sum_k.float() / counts).to(torch.bfloat16)
    state_fp8_mean_k = state_mean_k.to(torch.float8_e4m3fnuz)
    log_counts_fp32 = counts.log().squeeze(-1)
    log_counts_fp16 = log_counts_fp32.to(torch.float16)
    cache_indices = torch.arange(batch, dtype=torch.int64, device=device)

    def scratch() -> dict[str, torch.Tensor]:
        return {
            "scores": torch.empty(
                rows, groups, 8, dtype=torch.float32, device=device
            ),
            "indices": torch.empty(
                rows, groups, 8, dtype=torch.int64, device=device
            ),
            "group_out": torch.empty(
                rows, groups, head_dim, dtype=torch.float32, device=device
            ),
            "group_lse": torch.empty(
                rows, groups, dtype=torch.float32, device=device
            ),
            "context": torch.empty(sequences, dtype=torch.int32, device=device),
            "launch": torch.empty(sequences, dtype=torch.int32, device=device),
            "marker": torch.zeros(1, dtype=torch.int32, device=device),
            "previous_rows": torch.full(
                (sequences,), -1, dtype=torch.int32, device=device
            ),
            "previous_counts": torch.zeros(
                sequences, dtype=torch.int32, device=device
            ),
            "previous_slots": torch.zeros(
                sequences, union_capacity, dtype=torch.int32, device=device
            ),
            "active_mask": torch.zeros(
                sequences, mask_capacity, dtype=torch.uint8, device=device
            ),
            "active_blocks": torch.zeros(
                sequences,
                triton.cdiv(mask_capacity, 64),
                dtype=torch.uint8,
                device=device,
            ),
        }

    sum_scratch = scratch()
    mean_scratch = scratch()
    local_lens = torch.zeros(batch, dtype=torch.int32, device=device)
    fixed_lengths = torch.full(
        (batch, kv_heads), args.slots, dtype=torch.int32, device=device
    )
    new_k = torch.empty(
        batch, kv_heads, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    new_v = torch.empty_like(new_k)
    arena_k = torch.empty(
        sequences * max(1, args.slots),
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    arena_v = torch.empty_like(arena_k)
    slot_offsets = torch.zeros(
        batch,
        kv_heads,
        args.slots + 1,
        dtype=torch.int32,
        device=device,
    )

    def launch(keys: torch.Tensor, buffers: dict[str, torch.Tensor], means: bool) -> None:
        _decode_route_coarse_gqa_groups_fixed_prepare_kernel[
            (sequences, groups)
        ](
            q,
            keys,
            state_v,
            counts,
            cache_indices,
            buffers["scores"],
            buffers["indices"],
            buffers["group_out"],
            buffers["group_lse"],
            local_lens,
            fixed_lengths,
            buffers["context"],
            buffers["launch"],
            new_k,
            new_v,
            arena_k,
            arena_v,
            buffers["marker"],
            buffers["previous_rows"],
            buffers["previous_counts"],
            buffers["previous_slots"],
            slot_offsets,
            buffers["active_mask"],
            buffers["active_blocks"],
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            state_v.stride(0),
            state_v.stride(1),
            state_v.stride(2),
            counts.stride(0),
            counts.stride(1),
            counts.stride(2),
            new_k.stride(0),
            new_k.stride(1),
            new_v.stride(0),
            new_v.stride(1),
            slot_offsets.stride(1),
            buffers["active_mask"].stride(0),
            buffers["active_blocks"].stride(0),
            args.slots,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=gqa,
            HEAD_DIM=head_dim,
            SCALE=head_dim**-0.5,
            GROUP_N=group_n,
            MAX_GROUPS=groups,
            PROTECTED_LEN=1,
            MAX_LEAF_TOKENS=1024,
            SCORE_ONLY=True,
            STATE_CAPACITY=args.slots,
            UNION_CAPACITY=union_capacity,
            LOCAL_OFFSET=0,
            LOCAL_CAPACITY=1,
            LOCAL_LIMIT=0,
            SINK_LEN=0,
            LEAF_BEGIN=args.slots,
            MASK_CAPACITY=mask_capacity,
            TILE_SIZE=64,
            RESET_BLOCK_N=64,
            RESET_BLOCKS_N=4,
            INCLUDE_NEW=False,
            SEPARATE_LOCAL_SINK=False,
            KEYS_ARE_MEANS=means,
            num_warps=4,
            num_stages=3,
            waves_per_eu=1,
        )

    launch(state_sum_k, sum_scratch, False)
    launch(state_mean_k, mean_scratch, True)
    fp8_scratch = scratch()
    launch(state_fp8_mean_k, fp8_scratch, True)
    live_log_scratch = scratch()
    fp32_log_scratch = scratch()
    fp16_log_scratch = scratch()

    def launch_bias(
        log_counts: torch.Tensor,
        buffers: dict[str, torch.Tensor],
        cached: bool,
    ) -> None:
        _score_cached_mean_bias_kernel[(sequences, groups)](
            q,
            state_mean_k,
            counts,
            log_counts,
            buffers["scores"],
            buffers["indices"],
            args.slots,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=gqa,
            HEAD_DIM=head_dim,
            SCALE=head_dim**-0.5,
            GROUP_N=group_n,
            USE_CACHED_BIAS=cached,
            num_warps=4,
            num_stages=3,
            waves_per_eu=1,
        )

    launch_bias(log_counts_fp32, live_log_scratch, False)
    launch_bias(log_counts_fp32, fp32_log_scratch, True)
    launch_bias(log_counts_fp16, fp16_log_scratch, True)
    torch.cuda.synchronize()
    exact_scores = torch.equal(sum_scratch["scores"], mean_scratch["scores"])
    exact_indices = torch.equal(sum_scratch["indices"], mean_scratch["indices"])
    maximum_score_error = float(
        (sum_scratch["scores"] - mean_scratch["scores"])
        .abs()
        .nan_to_num()
        .max()
        .item()
    )
    fp8_indices_equal = torch.equal(
        sum_scratch["indices"], fp8_scratch["indices"]
    )
    fp8_slot_agreement = float(
        (sum_scratch["indices"] == fp8_scratch["indices"])
        .to(torch.float32)
        .mean()
        .item()
    )
    fp32_bias_indices_equal = torch.equal(
        live_log_scratch["indices"], fp32_log_scratch["indices"]
    )
    fp32_bias_scores_equal = torch.equal(
        live_log_scratch["scores"], fp32_log_scratch["scores"]
    )
    fp16_bias_slot_agreement = float(
        (live_log_scratch["indices"] == fp16_log_scratch["indices"])
        .to(torch.float32)
        .mean()
        .item()
    )

    def time(call) -> float:
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            call()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000.0)
        return statistics.median(samples)

    result = {
        "slots": args.slots,
        "sum_route_us": time(lambda: launch(state_sum_k, sum_scratch, False)),
        "precomputed_mean_route_us": time(
            lambda: launch(state_mean_k, mean_scratch, True)
        ),
        "fp8_mean_route_us": time(
            lambda: launch(state_fp8_mean_k, fp8_scratch, True)
        ),
        "candidate_scores_bitwise_equal": exact_scores,
        "candidate_indices_equal": exact_indices,
        "maximum_score_error": maximum_score_error,
        "fp8_candidate_indices_equal": fp8_indices_equal,
        "fp8_candidate_slot_agreement": fp8_slot_agreement,
        "live_log_score_us": time(
            lambda: launch_bias(log_counts_fp32, live_log_scratch, False)
        ),
        "fp32_cached_log_score_us": time(
            lambda: launch_bias(log_counts_fp32, fp32_log_scratch, True)
        ),
        "fp16_cached_log_score_us": time(
            lambda: launch_bias(log_counts_fp16, fp16_log_scratch, True)
        ),
        "fp32_cached_log_scores_equal": fp32_bias_scores_equal,
        "fp32_cached_log_indices_equal": fp32_bias_indices_equal,
        "fp16_cached_log_slot_agreement": fp16_bias_slot_agreement,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

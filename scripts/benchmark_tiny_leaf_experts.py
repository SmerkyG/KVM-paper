#!/usr/bin/env python3
"""Benchmark exact N=1..4 leaf-expert attention on a real LOD prefill trace.

Routing and expert/query grouping are prepared outside the timed region.  The
baseline is the existing expert kernel, restricted to the same N<=4 routes.
The probe kernel uses one exact-N launch for each populated N bucket and never
materializes the baseline's padded 16x32 QK/PV tiles.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoTokenizer

from model.kernels.paged_leaf_attention import _paged_leaf_attention_kernel
from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.probe_qwen35_lod_niah import load_text_model


@triton.jit
def _tiny_leaf_expert_attention_kernel(
    q,
    packed_route_row,
    block_expert,
    block_starts,
    q_lengths,
    cu_q,
    expert_kv_row,
    expert_slot,
    leaf_k,
    leaf_v,
    page_indices,
    slot_pages,
    out,
    lse,
    PAGE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    KEY_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Exact attention for experts known to contain exactly 1..4 leaves."""
    program = tl.program_id(0)
    expert = tl.load(block_expert + program)
    query_block = program - tl.load(block_starts + expert)
    query_count = tl.load(q_lengths + expert)
    query_offset = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_query = query_offset < query_count
    packed_begin = tl.load(cu_q + expert).to(tl.int64)
    packed_row = packed_begin + query_offset.to(tl.int64)
    route_row = tl.load(
        packed_route_row + packed_row, mask=valid_query, other=0
    ).to(tl.int64)
    query_row = route_row // ROUTE_COUNT

    kv_row = tl.load(expert_kv_row + expert).to(tl.int64)
    slot = tl.load(expert_slot + expert).to(tl.int64)
    page_id = tl.load(
        slot_pages
        + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
    ).to(tl.int64)
    physical_page = (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE

    head_offset = tl.arange(0, HEAD_DIM)
    value_offset = tl.arange(0, VALUE_DIM)
    q_block = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=valid_query[:, None],
        other=0.0,
    ).to(tl.float32)

    score_columns = tl.full((BLOCK_M, 4), -float("inf"), tl.float32)
    key_column = tl.arange(0, 4)
    for key_index in tl.static_range(0, KEY_COUNT):
        leaf_index = tl.load(page_indices + physical_page + key_index).to(tl.int64)
        storage_token = kv_row * LEAF_CAPACITY + leaf_index
        key = tl.load(
            leaf_k + storage_token * HEAD_DIM + head_offset
        ).to(tl.float32)
        score = tl.sum(q_block * key[None, :], axis=1) * SCALE_LOG2
        score_columns = tl.where(
            key_column[None, :] == key_index,
            score[:, None],
            score_columns,
        )

    maximum = tl.max(score_columns, axis=1)
    probability = tl.math.exp2(score_columns - maximum[:, None])
    denominator = tl.sum(probability, axis=1)
    probability /= denominator[:, None]
    accumulator = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)
    for key_index in tl.static_range(0, KEY_COUNT):
        leaf_index = tl.load(page_indices + physical_page + key_index).to(tl.int64)
        storage_token = kv_row * LEAF_CAPACITY + leaf_index
        value = tl.load(
            leaf_v + storage_token * VALUE_DIM + value_offset
        ).to(tl.float32)
        key_probability = tl.sum(
            tl.where(
                key_column[None, :] == key_index,
                probability,
                0.0,
            ),
            axis=1,
        )
        accumulator += key_probability[:, None] * value[None, :]

    natural_lse = (
        maximum + tl.math.log2(denominator)
    ) * 0.6931471805599453
    tl.store(
        out + route_row[:, None] * VALUE_DIM + value_offset[None, :],
        accumulator,
        mask=valid_query[:, None],
    )
    tl.store(lse + route_row, natural_lse, mask=valid_query)


@dataclass
class ExpertBucket:
    order: torch.Tensor
    unique_expert: torch.Tensor
    q_lengths: torch.Tensor
    cu_q: torch.Tensor
    expert_kv_row: torch.Tensor
    expert_slot: torch.Tensor
    block_expert: dict[int, torch.Tensor]
    block_starts: dict[int, torch.Tensor]
    total_blocks: dict[int, int]


def _make_bucket(
    sorted_expert: torch.Tensor,
    order: torch.Tensor,
    flat_slot_lengths: torch.Tensor,
    state_capacity: int,
    *,
    minimum: int,
    maximum: int,
    block_sizes: tuple[int, ...],
) -> ExpertBucket | None:
    route_key_count = flat_slot_lengths.index_select(0, sorted_expert.to(torch.long))
    keep = (route_key_count >= minimum) & (route_key_count <= maximum)
    bucket_expert_rows = sorted_expert[keep]
    if int(bucket_expert_rows.numel()) == 0:
        return None
    bucket_order = order[keep]
    unique_expert, q_lengths_i64 = torch.unique_consecutive(
        bucket_expert_rows, return_counts=True
    )
    q_lengths = q_lengths_i64.to(torch.int32)
    cu_q = F.pad(q_lengths_i64.cumsum(0), (1, 0)).to(torch.int32)
    expert_index = torch.arange(
        q_lengths.numel(), device=q_lengths.device, dtype=torch.int32
    )
    block_expert: dict[int, torch.Tensor] = {}
    block_starts: dict[int, torch.Tensor] = {}
    total_blocks: dict[int, int] = {}
    for block_m in block_sizes:
        expert_blocks = torch.div(
            q_lengths + block_m - 1, block_m, rounding_mode="floor"
        )
        blocks = int(expert_blocks.sum().item())
        block_expert[block_m] = torch.repeat_interleave(
            expert_index, expert_blocks, output_size=blocks
        )
        block_starts[block_m] = F.pad(
            expert_blocks.cumsum(0), (1, 0)
        )[:-1].to(torch.int32)
        total_blocks[block_m] = blocks
    return ExpertBucket(
        order=bucket_order,
        unique_expert=unique_expert,
        q_lengths=q_lengths,
        cu_q=cu_q,
        expert_kv_row=torch.div(
            unique_expert, state_capacity, rounding_mode="floor"
        ),
        expert_slot=unique_expert % state_capacity,
        block_expert=block_expert,
        block_starts=block_starts,
        total_blocks=total_blocks,
    )


def _time_launches(launch, repeats: int) -> float:
    launch()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        launch()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def _probe_call(
    module: Qwen3_5TwoLevelAttention,
    q: torch.Tensor,
    top_slots: torch.Tensor,
    cache: dict[str, torch.Tensor | int],
    *,
    configs: tuple[tuple[int, int], ...],
    repeats: int,
) -> dict[str, object] | None:
    page_indices = cache.get("page_indices")
    if not isinstance(page_indices, torch.Tensor):
        raise RuntimeError("tiny-expert probe requires virtual indexed leaf storage")
    leaf_k = cache["leaf_k"]
    leaf_v = cache["leaf_v"]
    slot_pages = cache["slot_pages"]
    slot_lengths = cache["slot_lengths"]
    overflow_page_keys = cache["overflow_page_keys"]
    overflow_page_values = cache["overflow_page_values"]
    overflow_used = cache["overflow_used"]
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (
            leaf_k,
            leaf_v,
            slot_pages,
            slot_lengths,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
        )
    ):
        raise TypeError("incomplete virtual page cache")
    if leaf_k.dtype != torch.bfloat16 or leaf_v.dtype != torch.bfloat16:
        raise TypeError("tiny-expert probe currently measures BF16 K/V only")

    batch, query_heads, query_len, head_dim = q.shape
    route_count = int(top_slots.size(-1))
    kv_heads = int(leaf_k.size(1))
    kv_group_size = query_heads // kv_heads
    rows = batch * query_heads * query_len
    state_capacity = int(slot_lengths.size(-1))
    page_capacity = int(page_indices.size(2))
    page_size = int(page_indices.size(3))
    inline_pages = int(slot_pages.size(3))
    leaf_capacity = int(leaf_k.size(2))
    value_dim = int(leaf_v.size(-1))

    query_head = torch.arange(query_heads, device=q.device, dtype=torch.int32)
    kv_head_for_query = torch.div(
        query_head, kv_group_size, rounding_mode="floor"
    )
    kv_row_for_head = (
        torch.arange(batch, device=q.device, dtype=torch.int32).unsqueeze(1)
        * kv_heads
        + kv_head_for_query.unsqueeze(0)
    )
    expert_id = (
        kv_row_for_head[:, :, None, None] * state_capacity
        + top_slots.to(torch.int32)
    ).reshape(-1)
    sorted_expert, order = expert_id.sort(stable=False)
    flat_slot_lengths = slot_lengths.reshape(-1)
    block_sizes = tuple(sorted({16, *(block_m for block_m, _ in configs)}))
    small = _make_bucket(
        sorted_expert,
        order,
        flat_slot_lengths,
        state_capacity,
        minimum=1,
        maximum=4,
        block_sizes=block_sizes,
    )
    if small is None:
        return None
    exact = {
        key_count: _make_bucket(
            sorted_expert,
            order,
            flat_slot_lengths,
            state_capacity,
            minimum=key_count,
            maximum=key_count,
            block_sizes=block_sizes,
        )
        for key_count in range(1, 5)
    }
    torch.cuda.synchronize()

    route_out = torch.empty(
        rows * route_count, value_dim, dtype=q.dtype, device=q.device
    )
    route_lse = torch.empty(
        rows * route_count, dtype=torch.float32, device=q.device
    )
    scale_log2 = float(module.scaling) * math.log2(math.e)
    hash_probes = int(module._page_lookup_probes(cache))

    def launch_baseline() -> None:
        _paged_leaf_attention_kernel[(small.total_blocks[16],)](
            q,
            q,
            small.order,
            small.block_expert[16],
            small.block_starts[16],
            leaf_k,
            leaf_v,
            page_indices,
            leaf_k,
            leaf_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            small.q_lengths,
            small.cu_q,
            small.expert_kv_row,
            small.expert_slot,
            route_out,
            route_lse,
            0,
            PAGE_CAPACITY=page_capacity,
            LEAF_CAPACITY=leaf_capacity,
            STATE_CAPACITY=state_capacity,
            INLINE_PAGES_PER_SLOT=inline_pages,
            HASH_CAPACITY=int(overflow_page_values.size(2)),
            HASH_PROBES=hash_probes,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            PAGE_SIZE=page_size,
            ROUTE_COUNT=route_count,
            SCALE_LOG2=scale_log2,
            BLOCK_M=16,
            BLOCK_N=32,
            INT8_MMA=False,
            INT8_PV_MMA=False,
            INDEXED=True,
            num_warps=4,
            waves_per_eu=1,
        )

    baseline_ms = _time_launches(launch_baseline, repeats)
    launch_baseline()
    torch.cuda.synchronize()
    sample_route = torch.cat(
        [
            bucket.order[: min(1024, int(bucket.order.numel()))]
            for bucket in exact.values()
            if bucket is not None
        ]
    ).to(torch.long)
    baseline_out = route_out.index_select(0, sample_route).clone()
    baseline_lse = route_lse.index_select(0, sample_route).clone()

    config_records: list[dict[str, object]] = []
    for block_m, num_warps in configs:
        def launch_specialized() -> None:
            for key_count, bucket in exact.items():
                if bucket is None:
                    continue
                _tiny_leaf_expert_attention_kernel[
                    (bucket.total_blocks[block_m],)
                ](
                    q,
                    bucket.order,
                    bucket.block_expert[block_m],
                    bucket.block_starts[block_m],
                    bucket.q_lengths,
                    bucket.cu_q,
                    bucket.expert_kv_row,
                    bucket.expert_slot,
                    leaf_k,
                    leaf_v,
                    page_indices,
                    slot_pages,
                    route_out,
                    route_lse,
                    PAGE_CAPACITY=page_capacity,
                    LEAF_CAPACITY=leaf_capacity,
                    STATE_CAPACITY=state_capacity,
                    INLINE_PAGES_PER_SLOT=inline_pages,
                    HEAD_DIM=head_dim,
                    VALUE_DIM=value_dim,
                    PAGE_SIZE=page_size,
                    ROUTE_COUNT=route_count,
                    SCALE_LOG2=scale_log2,
                    KEY_COUNT=key_count,
                    BLOCK_M=block_m,
                    num_warps=num_warps,
                    waves_per_eu=1,
                )

        specialized_ms = _time_launches(launch_specialized, repeats)
        launch_specialized()
        torch.cuda.synchronize()
        special_out = route_out.index_select(0, sample_route)
        special_lse = route_lse.index_select(0, sample_route)
        config_records.append(
            {
                "block_m": block_m,
                "num_warps": num_warps,
                "calculation_ms": specialized_ms,
                "speedup_vs_current_subset": baseline_ms / specialized_ms,
                "sample_output_max_abs_error": float(
                    (special_out - baseline_out).abs().max().item()
                ),
                "sample_output_mean_abs_error": float(
                    (special_out - baseline_out).abs().float().mean().item()
                ),
                "sample_lse_max_abs_error": float(
                    (special_lse - baseline_lse).abs().max().item()
                ),
                "sample_output_finite": bool(torch.isfinite(special_out).all().item()),
                "sample_lse_finite": bool(torch.isfinite(special_lse).all().item()),
            }
        )

    key_histogram = {
        str(key_count): {
            "experts": 0 if bucket is None else int(bucket.q_lengths.numel()),
            "routes": 0 if bucket is None else int(bucket.order.numel()),
            "programs_m16": (
                0 if bucket is None else bucket.total_blocks[16]
            ),
        }
        for key_count, bucket in exact.items()
    }
    return {
        "routes": int(small.order.numel()),
        "experts": int(small.q_lengths.numel()),
        "current_subset_calculation_ms": baseline_ms,
        "key_histogram": key_histogram,
        "configs": config_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--prefill-two-level-topk", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = load_text_model(
        args.checkpoint,
        "two_level",
        8,
        args.state_growth_factor,
        device,
        "paged",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    sequence = select_sequences(
        tokenizer, args.dataset, args.sequence_length, 1, 0, 1
    )[0][1].unsqueeze(0).expand(args.batch_size, -1).contiguous().to(device)
    modules = [
        module
        for module in model.modules()
        if isinstance(module, Qwen3_5TwoLevelAttention)
    ]
    for module in modules:
        module.prefill_two_level_topk = args.prefill_two_level_topk
        module.leaf_layout = "expert"
        module.virtual_page_storage = True

    with torch.inference_mode():
        warm = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        del warm
        for module in modules:
            if hasattr(module, "_lod_state"):
                delattr(module, "_lod_state")

        call_records: list[dict[str, object]] = []
        configs = (
            (1, 1),
            (2, 1),
            (4, 1),
            (8, 1),
            (8, 2),
            (16, 2),
            (16, 4),
        )
        for layer_index, module in enumerate(modules):
            original = module._paged_leaf_attention

            def measured(self, q, top_slots, cache, *, __original=original, __layer=layer_index):
                record = _probe_call(
                    self,
                    q,
                    top_slots,
                    cache,
                    configs=configs,
                    repeats=args.repeats,
                )
                if record is not None:
                    record["layer"] = __layer
                    call_records.append(record)
                return __original(q, top_slots, cache)

            module._paged_leaf_attention = types.MethodType(measured, module)

        result = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)

    aggregate_configs: list[dict[str, object]] = []
    baseline_total = sum(
        float(record["current_subset_calculation_ms"]) for record in call_records
    )
    for block_m, num_warps in configs:
        matches = [
            config
            for record in call_records
            for config in record["configs"]
            if config["block_m"] == block_m and config["num_warps"] == num_warps
        ]
        calculation_total = sum(float(config["calculation_ms"]) for config in matches)
        aggregate_configs.append(
            {
                "block_m": block_m,
                "num_warps": num_warps,
                "calculation_ms": calculation_total,
                "speedup_vs_current_subset": baseline_total / calculation_total,
                "sample_output_max_abs_error": max(
                    float(config["sample_output_max_abs_error"]) for config in matches
                ),
                "sample_output_mean_abs_error": sum(
                    float(config["sample_output_mean_abs_error"])
                    for config in matches
                )
                / len(matches),
                "sample_lse_max_abs_error": max(
                    float(config["sample_lse_max_abs_error"]) for config in matches
                ),
                "all_samples_finite": all(
                    bool(config["sample_output_finite"])
                    and bool(config["sample_lse_finite"])
                    for config in matches
                ),
            }
        )
    record = {
        "checkpoint": args.checkpoint,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "attention_layers": len(modules),
        "calls": len(call_records),
        "state_growth_factor": args.state_growth_factor,
        "prefill_two_level_topk": args.prefill_two_level_topk,
        "repeats": args.repeats,
        "current_subset_calculation_ms": baseline_total,
        "tiny_expert_routes": sum(int(call["routes"]) for call in call_records),
        "tiny_experts": sum(int(call["experts"]) for call in call_records),
        "aggregate_configs": aggregate_configs,
        "call_records": call_records,
        "logit_finite": bool(torch.isfinite(result.logits).all().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check and microbenchmark uniform all-centroid top-1 LOD attention."""

from __future__ import annotations

import argparse
import json
import math

import torch
import triton

from model.kernels.all_centroid_lod_attention import (
    all_centroid_top1_attention,
)


def make_cache(
    *,
    batch: int,
    kv_heads: int,
    state_len: int,
    leaves_per_state: int,
    head_dim: int,
    query_len: int,
    query_heads: int,
    device: torch.device,
    uneven: bool,
) -> dict[str, torch.Tensor]:
    page_size = 16
    if uneven:
        lengths = torch.tensor(
            [1 + (slot * 7) % (2 * leaves_per_state) for slot in range(state_len)],
            dtype=torch.int32,
            device=device,
        )
        lengths[0] = 4 * leaves_per_state
    else:
        lengths = torch.full(
            (state_len,), leaves_per_state, dtype=torch.int32, device=device
        )
    leaf_count = int(lengths.sum().item())
    page_count_per_slot = math.ceil(int(lengths.max().item()) / page_size)
    page_capacity = state_len * page_count_per_slot
    leaf_k = torch.randn(
        batch,
        kv_heads,
        leaf_count,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    leaf_v = torch.randn_like(leaf_k)
    per_slot_indices = torch.full(
        (state_len, page_count_per_slot * page_size),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )
    begin = 0
    for slot, length_tensor in enumerate(lengths):
        length = int(length_tensor.item())
        per_slot_indices[slot, :length] = torch.arange(
            begin, begin + length, dtype=torch.int32, device=device
        )
        begin += length
    page_indices = per_slot_indices.view(page_capacity, page_size)[
        None, None
    ].expand(batch, kv_heads, -1, -1).contiguous()
    slot_pages = torch.arange(
        page_capacity, dtype=torch.int32, device=device
    ).view(state_len, page_count_per_slot)[None, None].expand(
        batch, kv_heads, -1, -1
    ).contiguous()
    state_k = leaf_k.new_zeros(batch, kv_heads, state_len, head_dim)
    state_v = leaf_v.new_zeros(batch, kv_heads, state_len, head_dim)
    begin = 0
    for slot, length_tensor in enumerate(lengths):
        length = int(length_tensor.item())
        state_k[:, :, slot] = leaf_k[:, :, begin : begin + length].sum(dim=2)
        state_v[:, :, slot] = leaf_v[:, :, begin : begin + length].sum(dim=2)
        begin += length
    slot_lengths = lengths.view(1, 1, state_len).expand(
        batch, kv_heads, -1
    ).contiguous()
    return {
        "q": torch.randn(
            batch,
            query_heads,
            query_len,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        "state_k": state_k,
        "state_v": state_v,
        "counts": slot_lengths[..., None].float(),
        "leaf_k": leaf_k,
        "leaf_v": leaf_v,
        "page_indices": page_indices,
        "slot_pages": slot_pages,
        "overflow_page_keys": torch.full(
            (batch, kv_heads, 1),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        "overflow_page_values": torch.full(
            (batch, kv_heads, 1),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        "overflow_used": torch.zeros((), dtype=torch.int32, device=device),
        "slot_lengths": slot_lengths,
    }


def reference(
    tensors: dict[str, torch.Tensor],
    *,
    kv_group_size: int,
    scale: float,
    local_branch: tuple[torch.Tensor, torch.Tensor] | None,
    disjoint_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = tensors["q"].float()
    state_k = tensors["state_k"].float()
    state_v = tensors["state_v"].float()
    leaf_k = tensors["leaf_k"].float()
    leaf_v = tensors["leaf_v"].float()
    counts = tensors["counts"].squeeze(-1)
    slot_lengths = tensors["slot_lengths"]
    batch, query_heads, query_len, _ = q.shape
    state_len = int(state_k.size(2))
    outputs = torch.empty_like(q)
    lses = torch.empty(batch, query_heads, query_len, device=q.device)
    for batch_index in range(batch):
        for query_head in range(query_heads):
            kv_head = query_head // kv_group_size
            members = []
            begin = 0
            for slot in range(state_len):
                count = int(slot_lengths[batch_index, kv_head, slot].item())
                members.append(torch.arange(begin, begin + count, device=q.device))
                begin += count
            for position in range(query_len):
                query = q[batch_index, query_head, position]
                scores = []
                values = []
                for slot, indices in enumerate(members):
                    keys = leaf_k[batch_index, kv_head, indices]
                    winner = int(torch.argmax(keys @ query).item())
                    leaf_index = int(indices[winner].item())
                    scores.append((query @ leaf_k[batch_index, kv_head, leaf_index]) * scale)
                    values.append(leaf_v[batch_index, kv_head, leaf_index])
                    residual_count = counts[batch_index, kv_head, slot] - 1.0
                    if float(residual_count.item()) > 0:
                        if disjoint_residual:
                            residual_key = (
                                state_k[batch_index, kv_head, slot]
                                - leaf_k[batch_index, kv_head, leaf_index]
                            ) / residual_count
                            residual_value = (
                                state_v[batch_index, kv_head, slot]
                                - leaf_v[batch_index, kv_head, leaf_index]
                            ) / residual_count
                        else:
                            residual_key = (
                                state_k[batch_index, kv_head, slot]
                                / counts[batch_index, kv_head, slot]
                            )
                            residual_value = (
                                state_v[batch_index, kv_head, slot]
                                / counts[batch_index, kv_head, slot]
                            )
                        scores.append(query @ residual_key * scale + residual_count.log())
                        values.append(residual_value)
                score = torch.stack(scores)
                value = torch.stack(values)
                if local_branch is not None:
                    local_output, local_lse = local_branch
                    score = torch.cat(
                        (score, local_lse[batch_index, query_head, position].view(1))
                    )
                    value = torch.cat(
                        (
                            value,
                            local_output[
                                batch_index, query_head, position
                            ].float().view(1, -1),
                        )
                    )
                probability = score.softmax(dim=0)
                outputs[batch_index, query_head, position] = probability @ value
                lses[batch_index, query_head, position] = score.logsumexp(dim=0)
    return outputs, lses


def run(args: argparse.Namespace) -> dict[str, float | int]:
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    kv_group_size = args.query_heads // args.kv_heads
    tensors = make_cache(
        batch=args.batch_size,
        kv_heads=args.kv_heads,
        state_len=args.state_len,
        leaves_per_state=args.leaves_per_state,
        head_dim=args.head_dim,
        query_len=args.query_len,
        query_heads=args.query_heads,
        device=device,
        uneven=args.uneven,
    )
    local_branch = None
    if args.local_branch:
        local_branch = (
            torch.randn_like(tensors["q"]),
            torch.randn(
                *tensors["q"].shape[:-1], dtype=torch.float32, device=device
            ),
        )
    kwargs = dict(
        state_len=args.state_len,
        kv_group_size=kv_group_size,
        scale=args.head_dim**-0.5,
        local_branch=local_branch,
        hash_probes=0,
        winner_block_m=args.winner_block_m,
        winner_block_n=args.winner_block_n,
        winner_block_d=args.winner_block_d,
        centroids_per_program=args.centroids_per_program,
        combine_centroids_per_program=args.combine_centroids_per_program,
        attention_block_m=args.attention_block_m,
        attention_block_n=args.attention_block_n,
        attention_block_d=args.attention_block_d,
        winner_num_warps=args.winner_num_warps,
        attention_num_warps=args.attention_num_warps,
        waves_per_eu=args.waves_per_eu,
        fused_prefill=args.fused_prefill,
        fused_block_m=args.fused_block_m,
        fused_leaf_block_n=args.fused_leaf_block_n,
        fused_block_d=args.fused_block_d,
        fused_centroids_per_program=args.fused_centroids_per_program,
        fused_num_warps=args.fused_num_warps,
        disjoint_residual=not args.mean_residual,
    )

    def invoke(timing_events=None):
        return all_centroid_top1_attention(
            tensors["q"],
            tensors["state_k"],
            tensors["state_v"],
            tensors["counts"],
            tensors["leaf_k"],
            tensors["leaf_v"],
            tensors["page_indices"],
            tensors["slot_pages"],
            tensors["overflow_page_keys"],
            tensors["overflow_page_values"],
            tensors["overflow_used"],
            tensors["slot_lengths"],
            timing_events=timing_events,
            **kwargs,
        )

    actual, actual_lse, winners, winner_scores, state_scores = invoke()
    torch.cuda.synchronize(device)
    result: dict[str, float | int] = {
        "winner_bytes": (
            winners.numel() * winners.element_size()
            + winner_scores.numel() * winner_scores.element_size()
            + state_scores.numel() * state_scores.element_size()
        ),
    }
    if args.check:
        if winners.numel():
            winner_mismatches = 0
            for batch_index in range(args.batch_size):
                for kv_head in range(args.kv_heads):
                    for slot in range(args.state_len):
                        begin = slot * args.leaves_per_state
                        keys = tensors["leaf_k"][
                            batch_index,
                            kv_head,
                            begin : begin + args.leaves_per_state,
                        ].float()
                        for query_group in range(kv_group_size):
                            query_head = kv_head * kv_group_size + query_group
                            expected_winner = (
                                torch.argmax(
                                    tensors["q"][
                                        batch_index, query_head
                                    ].float()
                                    @ keys.T,
                                    dim=1,
                                )
                                + begin
                            )
                            query_in_kv = (
                                query_group * args.query_len
                                + torch.arange(args.query_len, device=device)
                            )
                            actual_winner = winners[
                                batch_index,
                                kv_head,
                                slot,
                                query_in_kv // args.winner_block_m,
                                query_in_kv % args.winner_block_m,
                            ]
                            winner_mismatches += int(
                                (actual_winner != expected_winner).sum().item()
                            )
            result["winner_mismatches"] = winner_mismatches
            if winner_mismatches:
                raise AssertionError(
                    f"top-1 index table has {winner_mismatches} mismatches"
                )
        expected, expected_lse = reference(
            tensors,
            kv_group_size=kv_group_size,
            scale=args.head_dim**-0.5,
            local_branch=local_branch,
            disjoint_residual=not args.mean_residual,
        )
        result["max_output_error"] = float(
            (actual.float() - expected).abs().max().item()
        )
        result["max_lse_error"] = float(
            (actual_lse - expected_lse).abs().max().item()
        )
        torch.testing.assert_close(
            actual.float(), expected, rtol=3e-2, atol=2e-2
        )
        torch.testing.assert_close(
            actual_lse, expected_lse, rtol=2e-3, atol=2e-3
        )
    result["total_ms"] = float(
        triton.testing.do_bench(invoke, warmup=args.warmup, rep=args.repeats)
    )
    phase_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] = {}
    invoke(phase_events)
    torch.cuda.synchronize(device)
    for name, pairs in phase_events.items():
        result[f"{name}_ms"] = float(
            sum(begin.elapsed_time(end) for begin, end in pairs)
        )
    if args.compare_full:
        if local_branch is not None:
            raise ValueError("full-attention comparison excludes a synthetic local branch")

        def full_attention():
            return torch.ops.aten._scaled_dot_product_flash_attention.default(
                tensors["q"],
                tensors["leaf_k"],
                tensors["leaf_v"],
                0.0,
                False,
                False,
                scale=args.head_dim**-0.5,
            )[0]

        full_attention()
        torch.cuda.synchronize(device)
        result["full_attention_ms"] = float(
            triton.testing.do_bench(
                full_attention, warmup=args.warmup, rep=args.repeats
            )
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--query-len", type=int, default=17)
    parser.add_argument("--state-len", type=int, default=8)
    parser.add_argument("--leaves-per-state", type=int, default=7)
    parser.add_argument("--uneven", action="store_true")
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--winner-block-m", type=int, default=64)
    parser.add_argument("--winner-block-n", type=int, default=16)
    parser.add_argument("--winner-block-d", type=int, default=128)
    parser.add_argument("--centroids-per-program", type=int, default=8)
    parser.add_argument(
        "--combine-centroids-per-program", type=int, default=16
    )
    parser.add_argument("--attention-block-m", type=int, default=64)
    parser.add_argument("--attention-block-n", type=int, default=8)
    parser.add_argument("--attention-block-d", type=int, default=128)
    parser.add_argument("--winner-num-warps", type=int, default=4)
    parser.add_argument("--attention-num-warps", type=int, default=8)
    parser.add_argument("--waves-per-eu", type=int, default=1)
    parser.add_argument("--fused-prefill", action="store_true")
    parser.add_argument("--fused-block-m", type=int)
    parser.add_argument("--fused-leaf-block-n", type=int)
    parser.add_argument("--fused-block-d", type=int)
    parser.add_argument("--fused-centroids-per-program", type=int)
    parser.add_argument("--fused-num-warps", type=int)
    parser.add_argument("--mean-residual", action="store_true")
    parser.add_argument("--local-branch", action="store_true")
    parser.add_argument("--compare-full", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))

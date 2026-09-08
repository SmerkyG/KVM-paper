#!/usr/bin/env python3
"""Verify fused AITER centroid-remainder plus exact-leaf attention."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F


def _metadata(
    state_lengths: torch.Tensor,
    centroid_bases: torch.Tensor,
    kv_indptr: torch.Tensor,
    union_width: int,
) -> torch.Tensor:
    sequences = int(state_lengths.numel())
    metadata = torch.zeros(
        (1 + sequences, 4),
        dtype=torch.int32,
        device=state_lengths.device,
    )
    metadata[0, 0] = 0x4C4F44
    metadata[0, 1] = union_width
    metadata[0, 3] = 4
    sequence_metadata = metadata[1:]
    sequence_metadata[:, 0].copy_(state_lengths)
    sequence_metadata[:, 1].copy_(centroid_bases)
    sequence_metadata[:, 2].copy_(
        torch.arange(sequences, device=state_lengths.device) * 2 * union_width
    )
    sequence_metadata[:, 3].copy_(
        sequences * 2 * union_width + kv_indptr[:-1]
    )
    return metadata


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_bias: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    leaf_indices: torch.Tensor,
    leaf_owners: torch.Tensor,
    routes: torch.Tensor,
    state_lengths: torch.Tensor,
    centroid_bases: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(q)
    lse = torch.empty(q.size(0), dtype=torch.float32, device=q.device)
    for sequence in range(int(state_lengths.numel())):
        query_begin = int(qo_indptr[sequence].item())
        query_end = int(qo_indptr[sequence + 1].item())
        leaf_begin = int(kv_indptr[sequence].item())
        leaf_end = int(kv_indptr[sequence + 1].item())
        state_len = int(state_lengths[sequence].item())
        centroid_base = int(centroid_bases[sequence].item())
        centroids = torch.arange(
            centroid_base,
            centroid_base + state_len,
            dtype=torch.long,
            device=q.device,
        )
        sequence_leaf_indices = leaf_indices[leaf_begin:leaf_end].long()
        sequence_leaf_owners = leaf_owners[leaf_begin:leaf_end]
        for query_index in range(query_begin, query_end):
            selected_routes = routes[query_index]
            centroid_slots = torch.arange(state_len, device=q.device)
            selected_centroids = (
                centroid_slots[:, None] == selected_routes[None, :]
            ).any(dim=-1)
            selected_leaves = (
                sequence_leaf_owners[:, None] == selected_routes[None, :]
            ).any(dim=-1)
            physical = torch.cat(
                (centroids[~selected_centroids], sequence_leaf_indices[selected_leaves])
            )
            scores = (
                q[query_index, 0].float() @ k[physical, 0, 0].float().T * scale
                + key_bias[physical].float()
            )
            output[query_index, 0] = (
                torch.softmax(scores, dim=-1) @ v[physical, 0, 0].float()
            ).to(output.dtype)
            lse[query_index] = torch.logsumexp(scores, dim=-1)
    return output, lse


def _elapsed_ms(call, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / repeats


def _reference_unmasked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_bias: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    leaf_indices: torch.Tensor,
    state_lengths: torch.Tensor,
    centroid_bases: torch.Tensor,
    scale: float,
    *,
    include_centroids: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(q)
    lse = torch.empty(q.size(0), dtype=torch.float32, device=q.device)
    for sequence in range(int(state_lengths.numel())):
        qb = int(qo_indptr[sequence].item())
        qe = int(qo_indptr[sequence + 1].item())
        lb = int(kv_indptr[sequence].item())
        le = int(kv_indptr[sequence + 1].item())
        physical = leaf_indices[lb:le].long()
        if include_centroids:
            base = int(centroid_bases[sequence].item())
            centroids = torch.arange(
                base,
                base + int(state_lengths[sequence].item()),
                device=q.device,
            )
            physical = torch.cat((centroids, physical))
        local_k = k[physical, 0, 0].float()
        local_v = v[physical, 0, 0].float()
        for row in range(qb, qe):
            scores = q[row, 0].float() @ local_k.T * scale
            if include_centroids:
                scores += key_bias[physical].float()
            output[row, 0] = (torch.softmax(scores, dim=-1) @ local_v).to(
                output.dtype
            )
            lse[row] = torch.logsumexp(scores, dim=-1)
    return output, lse


def _verify_paged_wrapper(*, scale: float) -> None:
    from model.kernels.paged_leaf_attention import (
        aiter_fused_hilo_paged_attention,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, kv_heads, kv_group, query_len, head_dim = 2, 2, 2, 13, 128
    query_heads = kv_heads * kv_group
    state_len = state_capacity = 6
    page_size = 16
    leaf_capacity = state_capacity * page_size
    page_capacity = state_capacity
    q = torch.randn(
        batch, query_heads, query_len, head_dim, dtype=dtype, device=device
    )
    leaf_k = torch.randn(
        batch, kv_heads, leaf_capacity, head_dim, dtype=dtype, device=device
    )
    leaf_v = torch.randn_like(leaf_k)
    coarse_k = torch.randn(
        batch, kv_heads, state_capacity, head_dim, dtype=dtype, device=device
    )
    coarse_v = torch.randn_like(coarse_k)
    counts = torch.randint(
        1, 65, (batch, kv_heads, state_capacity), device=device
    )
    leaf_offset = 0
    coarse_offset = batch * kv_heads * leaf_capacity
    arena_k = torch.cat((leaf_k.reshape(-1, head_dim), coarse_k.reshape(-1, head_dim)))
    arena_v = torch.cat((leaf_v.reshape(-1, head_dim), coarse_v.reshape(-1, head_dim)))
    arena_bias = torch.zeros(arena_k.size(0), dtype=torch.float16, device=device)
    arena_bias[coarse_offset:] = counts.reshape(-1).float().log().half()
    page_indices = (
        torch.arange(leaf_capacity, dtype=torch.int32, device=device)
        .view(1, 1, page_capacity, page_size)
        .expand(batch, kv_heads, -1, -1)
        .contiguous()
    )
    slot_pages = (
        torch.arange(state_capacity, dtype=torch.int32, device=device)
        .view(1, 1, state_capacity, 1)
        .expand(batch, kv_heads, -1, -1)
        .contiguous()
    )
    slot_lengths = torch.full(
        (batch, kv_heads, state_capacity),
        page_size,
        dtype=torch.int32,
        device=device,
    )
    overflow_page_keys = torch.full(
        (batch, kv_heads, 1), -1, dtype=torch.int32, device=device
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), dtype=torch.int32, device=device)
    top_slots = torch.randint(
        0,
        state_capacity,
        (batch, query_heads, query_len, 3),
        dtype=torch.int64,
        device=device,
    )

    actual_out, actual_lse = aiter_fused_hilo_paged_attention(
        q,
        leaf_k,
        leaf_v,
        page_indices,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        arena_k,
        arena_v,
        arena_bias,
        arena_leaf_offset=leaf_offset,
        arena_coarse_offset=coarse_offset,
        arena_row_offset=0,
        state_len=state_len,
        state_capacity=state_capacity,
        kv_group_size=kv_group,
        scale=scale,
        hash_probes=0,
        query_tile=16,
    )
    expected_out = torch.empty_like(actual_out)
    expected_lse = torch.empty_like(actual_lse)
    for b in range(batch):
        for qh in range(query_heads):
            kh = qh // kv_group
            for row in range(query_len):
                selected = top_slots[b, qh, row].unique()
                keep = torch.ones(state_capacity, dtype=torch.bool, device=device)
                keep[selected] = False
                keys = torch.cat(
                    (
                        coarse_k[b, kh, keep],
                        leaf_k[b, kh].view(state_capacity, page_size, head_dim)[
                            selected
                        ].reshape(-1, head_dim),
                    )
                )
                values = torch.cat(
                    (
                        coarse_v[b, kh, keep],
                        leaf_v[b, kh].view(state_capacity, page_size, head_dim)[
                            selected
                        ].reshape(-1, head_dim),
                    )
                )
                biases = torch.cat(
                    (
                        counts[b, kh, keep].float().log(),
                        torch.zeros(selected.numel() * page_size, device=device),
                    )
                )
                scores = q[b, qh, row].float() @ keys.float().T * scale + biases
                expected_out[b, qh, row] = (
                    torch.softmax(scores, dim=-1) @ values.float()
                ).to(dtype)
                expected_lse[b, qh, row] = torch.logsumexp(scores, dim=-1)
    output_error = float((actual_out.float() - expected_out.float()).abs().max())
    lse_error = float((actual_lse - expected_lse).abs().max())
    print(f"paged_wrapper_output_max_abs_error={output_error:.8f}")
    print(f"paged_wrapper_lse_max_abs_error={lse_error:.8f}")
    if output_error > 0.025 or lse_error > 0.025:
        raise AssertionError("fused AITER HiLo wrapper disagrees with its reference")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-sequences", type=int, default=0)
    parser.add_argument("--state-len", type=int, default=1024)
    parser.add_argument("--leaf-count", type=int, default=256)
    parser.add_argument("--query-tile", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    from aiter.ops.mha import _mha_batch_prefill

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 128
    scale = 1.0 / math.sqrt(head_dim)

    query_lengths = torch.tensor([16, 11, 5], dtype=torch.int32, device=device)
    state_lengths = torch.tensor([13, 9, 17], dtype=torch.int32, device=device)
    leaf_lengths = torch.tensor([31, 23, 37], dtype=torch.int32, device=device)
    qo_indptr = F.pad(query_lengths.cumsum(0), (1, 0)).to(torch.int32)
    kv_indptr = F.pad(leaf_lengths.cumsum(0), (1, 0)).to(torch.int32)
    total_q = int(query_lengths.sum().item())
    total_leaves = int(leaf_lengths.sum().item())
    state_total = int(state_lengths.sum().item())
    centroid_bases = F.pad(state_lengths.cumsum(0), (1, 0))[:-1].to(torch.int32)
    leaf_base = state_total
    leaf_indices = torch.arange(
        leaf_base,
        leaf_base + total_leaves,
        dtype=torch.int32,
        device=device,
    )
    physical_count = state_total + total_leaves
    q = torch.randn(total_q, 1, head_dim, dtype=dtype, device=device)
    k = torch.randn(physical_count, 1, 1, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    key_bias = torch.zeros(physical_count, dtype=torch.float16, device=device)
    for sequence in range(3):
        base = int(centroid_bases[sequence].item())
        count = int(state_lengths[sequence].item())
        multiplicity = torch.randint(1, 65, (count,), device=device)
        key_bias[base : base + count] = multiplicity.float().log().half()

    route_stride = 4
    union_width = 16 * route_stride
    routes = torch.full(
        (total_q, route_stride), -1, dtype=torch.int32, device=device
    )
    leaf_owners = torch.empty(total_leaves, dtype=torch.int32, device=device)
    state_capacity = int(state_lengths.max().item())
    sparse_values = 3 * 2 * union_width
    query_masks = torch.zeros(
        sparse_values + total_leaves,
        dtype=torch.int32,
        device=device,
    )
    query_masks[:sparse_values].view(3, 2, union_width)[:, 0].fill_(
        state_capacity
    )
    for sequence in range(3):
        qb = int(qo_indptr[sequence].item())
        qe = int(qo_indptr[sequence + 1].item())
        lb = int(kv_indptr[sequence].item())
        le = int(kv_indptr[sequence + 1].item())
        state_len = int(state_lengths[sequence].item())
        routes[qb:qe, :2] = torch.randint(
            0, state_len, (qe - qb, 2), dtype=torch.int32, device=device
        )
        leaf_owners[lb:le] = torch.arange(le - lb, device=device) % state_len
        for local_row, query_index in enumerate(range(qb, qe)):
            bit = 1 << local_row
            selected = (
                leaf_owners[lb:le, None] == routes[query_index, None, :]
            ).any(dim=-1)
            query_masks[
                sparse_values + lb + torch.where(selected)[0]
            ] |= bit
        selected_union = routes[qb:qe].reshape(-1)
        selected_union = selected_union[selected_union >= 0].unique(sorted=True)
        sparse = query_masks[
            sequence * 2 * union_width : (sequence + 1) * 2 * union_width
        ].view(2, union_width)
        sparse[0, : selected_union.numel()] = selected_union
        for rank, slot in enumerate(selected_union.tolist()):
            membership = 0
            for local_row, query_index in enumerate(range(qb, qe)):
                if bool((routes[query_index] == slot).any()):
                    membership |= 1 << local_row
            sparse[1, rank] = membership

    metadata = _metadata(
        state_lengths,
        centroid_bases,
        kv_indptr,
        union_width,
    )

    def fused():
        out, lse, _, _ = _mha_batch_prefill(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            leaf_indices,
            int(query_lengths.max().item()),
            int((state_lengths + leaf_lengths).max().item()),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            return_lse=True,
            bias=key_bias.view(1, -1),
            block_table=metadata,
            seqlen_k=query_masks,
        )
        return out, lse

    with torch.inference_mode():
        actual_out, actual_lse = fused()
        expected_out, expected_lse = _reference(
            q,
            k,
            v,
            key_bias,
            qo_indptr,
            kv_indptr,
            leaf_indices,
            leaf_owners,
            routes,
            state_lengths,
            centroid_bases,
            scale,
        )
        leaf_only_out, leaf_only_lse = _reference_unmasked(
            q,
            k,
            v,
            key_bias,
            qo_indptr,
            kv_indptr,
            leaf_indices,
            state_lengths,
            centroid_bases,
            scale,
            include_centroids=False,
        )
        all_out, all_lse = _reference_unmasked(
            q,
            k,
            v,
            key_bias,
            qo_indptr,
            kv_indptr,
            leaf_indices,
            state_lengths,
            centroid_bases,
            scale,
            include_centroids=True,
        )
        torch.cuda.synchronize()
    actual_lse = actual_lse.reshape(-1)
    output_error = float((actual_out.float() - expected_out.float()).abs().max())
    lse_error = float((actual_lse - expected_lse).abs().max())
    print(f"output_max_abs_error={output_error:.8f}")
    print(f"lse_max_abs_error={lse_error:.8f}")
    print(
        "leaf_only_output_error="
        f"{float((actual_out.float() - leaf_only_out.float()).abs().max()):.8f}"
    )
    print(
        "leaf_only_lse_error="
        f"{float((actual_lse - leaf_only_lse).abs().max()):.8f}"
    )
    print(
        "all_unmasked_output_error="
        f"{float((actual_out.float() - all_out.float()).abs().max()):.8f}"
    )
    print(
        "all_unmasked_lse_error="
        f"{float((actual_lse - all_lse).abs().max()):.8f}"
    )
    if output_error > 0.025 or lse_error > 0.025:
        raise AssertionError("fused AITER LOD disagrees with the explicit reference")

    _verify_paged_wrapper(scale=scale)

    if args.benchmark_sequences <= 0:
        return
    sequences = args.benchmark_sequences
    query_tile = args.query_tile
    state_len = args.state_len
    leaf_count = args.leaf_count
    bench_q = torch.randn(
        sequences * query_tile, 1, head_dim, dtype=dtype, device=device
    )
    bench_k = torch.randn(
        state_len + leaf_count, 1, 1, head_dim, dtype=dtype, device=device
    )
    bench_v = torch.randn_like(bench_k)
    bench_bias = torch.zeros(
        state_len + leaf_count, dtype=torch.float16, device=device
    )
    bench_bias[:state_len] = math.log(16)
    bench_qo = torch.arange(
        0,
        (sequences + 1) * query_tile,
        query_tile,
        dtype=torch.int32,
        device=device,
    )
    bench_kv = torch.arange(
        0,
        (sequences + 1) * leaf_count,
        leaf_count,
        dtype=torch.int32,
        device=device,
    )
    bench_pages = torch.arange(
        state_len,
        state_len + leaf_count,
        dtype=torch.int32,
        device=device,
    ).repeat(sequences)
    bench_routes = torch.randint(
        0,
        state_len,
        (sequences * query_tile, route_stride),
        dtype=torch.int32,
        device=device,
    )
    bench_state_lengths = torch.full(
        (sequences,), state_len, dtype=torch.int32, device=device
    )
    bench_centroid_bases = torch.zeros(
        sequences, dtype=torch.int32, device=device
    )
    bench_metadata = _metadata(
        bench_state_lengths,
        bench_centroid_bases,
        bench_kv,
        union_width,
    )
    bench_leaf_masks = torch.zeros(
        sequences * 2 * union_width + sequences * leaf_count,
        dtype=torch.int32,
        device=device,
    )
    bench_sparse = bench_leaf_masks[: sequences * 2 * union_width].view(
        sequences, 2, union_width
    )
    bench_sparse[:, 0].fill_(state_len)
    bench_leaf_masks[sequences * 2 * union_width :].fill_(
        (1 << query_tile) - 1
    )

    def benchmark_call():
        out, lse, _, _ = _mha_batch_prefill(
            bench_q,
            bench_k,
            bench_v,
            bench_qo,
            bench_kv,
            bench_pages,
            query_tile,
            state_len + leaf_count,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            return_lse=True,
            bias=bench_bias.view(1, -1),
            block_table=bench_metadata,
            seqlen_k=bench_leaf_masks,
        )
        return out, lse

    with torch.inference_mode():
        fused_ms = _elapsed_ms(
            benchmark_call, warmup=args.warmup, repeats=args.repeats
        )
    print(f"fused_hilo_aiter_ms={fused_ms:.6f}")


if __name__ == "__main__":
    main()

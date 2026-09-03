#!/usr/bin/env python3
"""Compare query-major exact pages with literal gather-then-SDPA."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

from model.kernels.paged_leaf_attention import _dense_page_exact_attention_kernel


VENDOR_AITER = Path("/home/dan/subusers/agent/vendor/aiter")
if str(VENDOR_AITER) not in sys.path:
    sys.path.append(str(VENDOR_AITER))
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")


def timed_ms(function, warmups: int, repeats: int) -> tuple[float, object]:
    result = None
    for _ in range(warmups):
        result = function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        result = function()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / repeats, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--query-length", type=int, default=4096)
    parser.add_argument("--pages", type=int, default=2400)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--top-pages", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--query-tiles", type=int, nargs="+", default=(64, 256))
    parser.add_argument(
        "--regular-union-shapes",
        nargs="*",
        default=(),
        metavar="QUERY_TILE_UNION_PAGES",
        help=(
            "Synthetic normal-paged-attention shapes such as 16x64. These "
            "measure a query tile attending a shared union without masks."
        ),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    device = torch.device("cuda", 0)
    group = args.query_heads // args.kv_heads
    scale = args.head_dim**-0.5
    q = torch.randn(
        args.batch_size,
        args.query_heads,
        args.query_length,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    leaf_k = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.pages * args.page_size,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    leaf_v = torch.randn_like(leaf_k)
    page_indices = torch.arange(
        args.pages * args.page_size, device=device, dtype=torch.int32
    ).view(1, 1, args.pages, args.page_size)
    page_indices = page_indices.expand(
        args.batch_size, args.kv_heads, -1, -1
    ).contiguous()
    page_counts = torch.full(
        (args.batch_size, args.kv_heads, args.pages),
        args.page_size,
        device=device,
        dtype=torch.int32,
    )
    selected = torch.randint(
        args.pages,
        (
            args.batch_size,
            args.query_heads,
            args.query_length,
            args.top_pages,
        ),
        device=device,
        dtype=torch.int32,
    )
    cache_indices = torch.arange(args.batch_size, device=device, dtype=torch.long)
    exact_out = torch.empty_like(q)
    exact_lse = torch.empty(
        args.batch_size,
        args.query_heads,
        args.query_length,
        device=device,
        dtype=torch.float32,
    )
    rows = args.batch_size * args.query_heads * args.query_length

    def current_exact():
        return _dense_page_exact_attention_kernel[(rows,)](
            q,
            cache_indices,
            leaf_k,
            leaf_v,
            page_indices,
            page_counts,
            selected,
            exact_out,
            exact_lse,
            args.query_length,
            QUERY_HEADS=args.query_heads,
            KV_HEADS=args.kv_heads,
            KV_GROUP_SIZE=group,
            PAGE_CAPACITY=args.pages,
            LEAF_CAPACITY=args.pages * args.page_size,
            LEAF_K_BATCH_STRIDE=leaf_k.stride(0),
            LEAF_K_HEAD_STRIDE=leaf_k.stride(1),
            LEAF_K_TOKEN_STRIDE=leaf_k.stride(2),
            LEAF_V_BATCH_STRIDE=leaf_v.stride(0),
            LEAF_V_HEAD_STRIDE=leaf_v.stride(1),
            LEAF_V_TOKEN_STRIDE=leaf_v.stride(2),
            HEAD_DIM=args.head_dim,
            VALUE_DIM=args.head_dim,
            HEAD_BLOCK_DIM=args.head_dim,
            VALUE_BLOCK_DIM=args.head_dim,
            PAGE_SIZE=args.page_size,
            TOP_PAGES=args.top_pages,
            SCALE_LOG2=scale * math.log2(math.e),
            num_warps=2,
            waves_per_eu=1,
        )

    current_ms, compiled = timed_ms(current_exact, args.warmups, args.repeats)
    page_k = leaf_k.view(
        args.batch_size,
        args.kv_heads,
        args.pages,
        args.page_size,
        args.head_dim,
    )
    page_v = leaf_v.view_as(page_k)
    from aiter.ops.mha import mha_batch_prefill_func

    page_k_flat = page_k.reshape(-1, args.page_size, args.head_dim).unsqueeze(2)
    page_v_flat = page_v.reshape_as(page_k_flat)
    query_rows = rows
    qo_indptr = torch.arange(query_rows + 1, device=device, dtype=torch.int32)
    kv_indptr = qo_indptr * args.top_pages
    kv_last_page_lens = torch.full(
        (query_rows,), args.page_size, device=device, dtype=torch.int32
    )
    seqlen_k = torch.full(
        (query_rows,),
        args.top_pages * args.page_size,
        device=device,
        dtype=torch.int32,
    )
    physical_page_base = (
        torch.arange(args.batch_size, device=device, dtype=torch.int32)[:, None]
        * args.kv_heads
        + (
            torch.arange(args.query_heads, device=device, dtype=torch.int32)[None, :]
            // group
        )
    ) * args.pages

    def prepare_aiter_page_table():
        return (
            (selected + physical_page_base[:, :, None, None]).reshape(-1).contiguous()
        )

    aiter_page_indices = prepare_aiter_page_table()

    def aiter_kernel():
        return mha_batch_prefill_func(
            q.reshape(query_rows, 1, args.head_dim),
            page_k_flat,
            page_v_flat,
            qo_indptr,
            kv_indptr,
            aiter_page_indices,
            1,
            args.top_pages * args.page_size,
            softmax_scale=scale,
            causal=False,
            return_lse=True,
            kv_last_page_lens=kv_last_page_lens,
            seqlen_k=seqlen_k,
        )

    aiter_kernel_ms, aiter_result = timed_ms(aiter_kernel, args.warmups, args.repeats)
    aiter_table_ms, _ = timed_ms(prepare_aiter_page_table, args.warmups, args.repeats)

    def aiter_end_to_end():
        nonlocal aiter_page_indices
        aiter_page_indices = prepare_aiter_page_table()
        return aiter_kernel()

    aiter_total_ms, _ = timed_ms(aiter_end_to_end, args.warmups, args.repeats)
    aiter_output, aiter_lse = aiter_result
    aiter_output = aiter_output[:, 0].reshape_as(q)
    aiter_lse = aiter_lse.reshape_as(exact_lse)

    # A normal paged-attention request may use one block table for all query
    # heads in a GQA group.  Make that table the union of the pages selected by
    # those heads.  Every head attends every page in the union; the matching
    # production summary path would remove the same union from every head's
    # low-resolution attention.  No per-head mask is required here.
    union_sequences = args.batch_size * args.kv_heads * args.query_length
    union_queries = (
        q.view(
            args.batch_size,
            args.kv_heads,
            group,
            args.query_length,
            args.head_dim,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(union_sequences, group, args.head_dim)
        .contiguous()
    )
    union_candidates = (
        selected.view(
            args.batch_size,
            args.kv_heads,
            group,
            args.query_length,
            args.top_pages,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(union_sequences, group * args.top_pages)
    )
    union_qo_indptr = torch.arange(
        union_sequences + 1, device=device, dtype=torch.int32
    )
    union_physical_base = (
        torch.arange(
            args.batch_size * args.kv_heads,
            device=device,
            dtype=torch.int32,
        )
        .repeat_interleave(args.query_length)
        .mul(args.pages)
    )

    def prepare_gqa_union_page_table():
        sorted_pages = union_candidates.sort(dim=-1).values
        unique = torch.ones_like(sorted_pages, dtype=torch.bool)
        unique[:, 1:] = sorted_pages[:, 1:] != sorted_pages[:, :-1]
        union_counts = unique.sum(dim=-1, dtype=torch.int32)
        union_kv_indptr = torch.empty(
            union_sequences + 1, device=device, dtype=torch.int32
        )
        union_kv_indptr[0] = 0
        torch.cumsum(union_counts, dim=0, out=union_kv_indptr[1:])
        union_page_indices = (sorted_pages + union_physical_base[:, None])[
            unique
        ].contiguous()
        return union_page_indices, union_kv_indptr, union_counts

    (
        union_page_indices,
        union_kv_indptr,
        union_page_counts,
    ) = prepare_gqa_union_page_table()
    union_last_page_lens = torch.full(
        (union_sequences,), args.page_size, device=device, dtype=torch.int32
    )
    union_seqlen_k = union_page_counts * args.page_size

    def aiter_gqa_union_kernel():
        return mha_batch_prefill_func(
            union_queries,
            page_k_flat,
            page_v_flat,
            union_qo_indptr,
            union_kv_indptr,
            union_page_indices,
            1,
            group * args.top_pages * args.page_size,
            softmax_scale=scale,
            causal=False,
            return_lse=True,
            kv_last_page_lens=union_last_page_lens,
            seqlen_k=union_seqlen_k,
        )

    union_kernel_ms, union_result = timed_ms(
        aiter_gqa_union_kernel, args.warmups, args.repeats
    )
    union_table_ms, _ = timed_ms(
        prepare_gqa_union_page_table, args.warmups, args.repeats
    )
    union_output, union_lse = union_result
    union_output = (
        union_output.view(
            args.batch_size,
            args.kv_heads,
            args.query_length,
            group,
            args.head_dim,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape_as(q)
    )
    union_lse = (
        union_lse.view(
            args.batch_size,
            args.kv_heads,
            args.query_length,
            group,
        )
        .permute(0, 1, 3, 2)
        .reshape_as(exact_lse)
    )
    regular_union_records = []
    ps1_regular_union_records = []
    token_k_flat = leaf_k.reshape(-1, 1, args.head_dim).unsqueeze(2)
    token_v_flat = leaf_v.reshape_as(token_k_flat)
    grouped_queries = (
        q.view(
            args.batch_size,
            args.kv_heads,
            group,
            args.query_length,
            args.head_dim,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    for shape in args.regular_union_shapes:
        try:
            query_tile_text, union_pages_text = shape.lower().split("x", 1)
            query_tile = int(query_tile_text)
            union_pages = int(union_pages_text)
        except (ValueError, AttributeError) as error:
            raise ValueError(
                f"invalid regular union shape {shape!r}; expected e.g. 16x64"
            ) from error
        if (
            query_tile <= 0
            or union_pages <= 0
            or args.query_length % query_tile
            or union_pages > args.pages
        ):
            raise ValueError(f"regular union shape {shape!r} does not fit")
        tiles_per_kv_row = args.query_length // query_tile
        sequence_count = args.batch_size * args.kv_heads * tiles_per_kv_row
        packed_queries = grouped_queries.reshape(-1, group, args.head_dim)
        regular_qo_indptr = (
            torch.arange(sequence_count + 1, device=device, dtype=torch.int32)
            * query_tile
        )
        regular_kv_indptr = (
            torch.arange(sequence_count + 1, device=device, dtype=torch.int32)
            * union_pages
        )
        sequence = torch.arange(sequence_count, device=device, dtype=torch.int32)
        logical_page = (
            sequence[:, None] * 37
            + torch.arange(union_pages, device=device, dtype=torch.int32)[None, :]
        ) % args.pages
        regular_physical_base = (sequence // tiles_per_kv_row) * args.pages
        regular_page_indices = (
            (logical_page + regular_physical_base[:, None]).reshape(-1).contiguous()
        )
        regular_last_page_lens = torch.full(
            (sequence_count,), args.page_size, device=device, dtype=torch.int32
        )
        regular_seqlen_k = torch.full(
            (sequence_count,),
            union_pages * args.page_size,
            device=device,
            dtype=torch.int32,
        )

        def regular_union_kernel():
            return mha_batch_prefill_func(
                packed_queries,
                page_k_flat,
                page_v_flat,
                regular_qo_indptr,
                regular_kv_indptr,
                regular_page_indices,
                query_tile,
                union_pages * args.page_size,
                softmax_scale=scale,
                causal=False,
                return_lse=True,
                kv_last_page_lens=regular_last_page_lens,
                seqlen_k=regular_seqlen_k,
            )

        regular_ms, regular_result = timed_ms(
            regular_union_kernel, args.warmups, args.repeats
        )
        regular_union_records.append(
            {
                "query_tile": query_tile,
                "union_pages": union_pages,
                "exact_attention_work_inflation": union_pages / args.top_pages,
                "kernel_ms": regular_ms,
                "output_finite": bool(torch.isfinite(regular_result[0]).all().item()),
            }
        )

        # The production recursive cache keeps chronological leaves plus a
        # virtual page->leaf index table.  AITER's page-size-1 specialization
        # can consume those leaves directly: expand every logical 16-token
        # page into sixteen physical one-token page IDs.  A validity bit can
        # later be packed into these IDs for partially populated pages.
        token_page_indices = (
            (
                (logical_page + regular_physical_base[:, None])[:, :, None]
                * args.page_size
                + torch.arange(args.page_size, device=device, dtype=torch.int32)[
                    None, None, :
                ]
            )
            .reshape(-1)
            .contiguous()
        )
        token_count = union_pages * args.page_size
        ps1_kv_indptr = (
            torch.arange(sequence_count + 1, device=device, dtype=torch.int32)
            * token_count
        )
        ps1_last_page_lens = torch.ones(
            sequence_count, device=device, dtype=torch.int32
        )

        def ps1_regular_union_kernel():
            return mha_batch_prefill_func(
                packed_queries,
                token_k_flat,
                token_v_flat,
                regular_qo_indptr,
                ps1_kv_indptr,
                token_page_indices,
                query_tile,
                token_count,
                softmax_scale=scale,
                causal=False,
                return_lse=True,
                kv_last_page_lens=ps1_last_page_lens,
                seqlen_k=regular_seqlen_k,
            )

        ps1_ms, ps1_result = timed_ms(
            ps1_regular_union_kernel, args.warmups, args.repeats
        )
        ps1_regular_union_records.append(
            {
                "query_tile": query_tile,
                "union_pages": union_pages,
                "union_tokens": token_count,
                "exact_attention_work_inflation": union_pages / args.top_pages,
                "kernel_ms": ps1_ms,
                "output_finite": bool(torch.isfinite(ps1_result[0]).all().item()),
            }
        )
    batch_index = torch.arange(device=device, end=args.batch_size).view(
        args.batch_size, 1, 1, 1
    )
    kv_index = (torch.arange(device=device, end=args.query_heads) // group).view(
        1, args.query_heads, 1, 1
    )
    records = []
    for query_tile in args.query_tiles:
        if args.query_length % query_tile:
            continue

        def copy_sdpa():
            output_tiles = []
            lse_tiles = []
            for query_begin in range(0, args.query_length, query_tile):
                query_end = query_begin + query_tile
                selected_tile = selected[:, :, query_begin:query_end].long()
                gathered_k = page_k[batch_index, kv_index, selected_tile]
                gathered_v = page_v[batch_index, kv_index, selected_tile]
                sequence_count = args.batch_size * args.query_heads * query_tile
                query = (
                    q[:, :, query_begin:query_end]
                    .contiguous()
                    .view(sequence_count, 1, 1, args.head_dim)
                )
                keys = gathered_k.reshape(
                    sequence_count,
                    1,
                    args.top_pages * args.page_size,
                    args.head_dim,
                )
                values = gathered_v.reshape_as(keys)
                output, lse, *_ = (
                    torch.ops.aten._scaled_dot_product_flash_attention.default(
                        query,
                        keys,
                        values,
                        0.0,
                        False,
                        False,
                        scale=scale,
                    )
                )
                output_tiles.append(
                    output.view(
                        args.batch_size,
                        args.query_heads,
                        query_tile,
                        args.head_dim,
                    )
                )
                lse_tiles.append(
                    lse.view(args.batch_size, args.query_heads, query_tile)
                )
            return torch.cat(output_tiles, dim=2), torch.cat(lse_tiles, dim=2)

        torch.cuda.reset_peak_memory_stats(device)
        pipeline_ms, pipeline_result = timed_ms(copy_sdpa, args.warmups, args.repeats)
        peak_bytes = torch.cuda.max_memory_allocated(device)

        phases: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )
        output_tiles = []
        for query_begin in range(0, args.query_length, query_tile):
            query_end = query_begin + query_tile
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            selected_tile = selected[:, :, query_begin:query_end].long()
            gathered_k = page_k[batch_index, kv_index, selected_tile]
            gathered_v = page_v[batch_index, kv_index, selected_tile]
            query = q[:, :, query_begin:query_end].contiguous()
            end.record()
            phases["gather_and_query_copy"].append((begin, end))
            sequence_count = args.batch_size * args.query_heads * query_tile
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            output, *_ = torch.ops.aten._scaled_dot_product_flash_attention.default(
                query.view(sequence_count, 1, 1, args.head_dim),
                gathered_k.reshape(
                    sequence_count,
                    1,
                    args.top_pages * args.page_size,
                    args.head_dim,
                ),
                gathered_v.reshape(
                    sequence_count,
                    1,
                    args.top_pages * args.page_size,
                    args.head_dim,
                ),
                0.0,
                False,
                False,
                scale=scale,
            )
            end.record()
            phases["sdpa"].append((begin, end))
            output_tiles.append(output)
        torch.cuda.synchronize()
        phase_ms = {
            name: sum(start.elapsed_time(stop) for start, stop in pairs)
            for name, pairs in phases.items()
        }
        copied_output, copied_lse = pipeline_result
        records.append(
            {
                "query_tile": query_tile,
                "pipeline_ms": pipeline_ms,
                "phase_ms": phase_ms,
                "phase_sum_ms": sum(phase_ms.values()),
                "peak_allocated_bytes": peak_bytes,
                "output_max_abs": float(
                    (copied_output.float() - exact_out.float()).abs().max().item()
                ),
                "lse_max_abs": float((copied_lse - exact_lse).abs().max().item()),
            }
        )

    metadata = getattr(compiled, "metadata", None)
    result = {
        "shape": vars(args) | {"output": str(args.output)},
        "current_exact": {
            "milliseconds": current_ms,
            "registers_per_thread": getattr(compiled, "n_regs", None),
            "spills_per_thread": getattr(compiled, "n_spills", None),
            "shared_memory_bytes": getattr(metadata, "shared", None),
        },
        "aiter_direct_paged": {
            "page_table_ms": aiter_table_ms,
            "kernel_ms": aiter_kernel_ms,
            "end_to_end_ms": aiter_total_ms,
            "output_max_abs": float(
                (aiter_output.float() - exact_out.float()).abs().max().item()
            ),
            "lse_max_abs": float((aiter_lse - exact_lse).abs().max().item()),
        },
        "aiter_gqa_union_paged": {
            "kernel_ms": union_kernel_ms,
            "page_table_ms": union_table_ms,
            "sequences": union_sequences,
            "query_heads_per_sequence": group,
            "mean_union_pages": float(union_page_counts.float().mean().item()),
            "max_union_pages": int(union_page_counts.max().item()),
            "exact_attention_work_inflation": float(
                union_page_counts.float().mean().item() / args.top_pages
            ),
            "output_finite": bool(torch.isfinite(union_output).all().item()),
            "lse_finite": bool(torch.isfinite(union_lse).all().item()),
        },
        "aiter_regular_union_sweep": regular_union_records,
        "aiter_ps1_regular_union_sweep": ps1_regular_union_records,
        "copy_sdpa": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

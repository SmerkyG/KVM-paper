"""Exercise rank-owned INT4 page reconstruction at long Kimi-K3 geometry."""

from __future__ import annotations

import argparse

import torch

from lod_attention._config import LODMode, ModelFamily, PagedLODConfig
from lod_attention._engines import KernelRecursivePagedLODAttention
from lod_attention._profile import configure_engine
from lod_attention.kernels.paged_leaf_attention import (
    dequantize_owned_virtual_paged_keys,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=524_288)
    parser.add_argument("--head-dim", type=int, default=576)
    parser.add_argument("--dcp-size", type=int, default=8)
    parser.add_argument("--dcp-rank", type=int, default=2)
    parser.add_argument("--interleave", type=int, default=4)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--cached-prefill", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    pages = (args.tokens + 15) // 16
    leaves = pages * 16
    page_indices = torch.arange(
        leaves, dtype=torch.int32, device=device
    ).view(1, 1, pages, 16)
    page_counts = torch.full(
        (1, 1, pages), 16, dtype=torch.int32, device=device
    )
    if args.tokens % 16:
        page_counts[..., -1] = args.tokens % 16
    next_page = torch.full((1, 1), pages, dtype=torch.int32, device=device)
    packed_keys = torch.full(
        (1, 1, leaves, args.head_dim // 2),
        0x88,
        dtype=torch.uint8,
        device=device,
    )
    page_scales = torch.ones(
        1,
        1,
        pages,
        args.head_dim // 4,
        dtype=torch.bfloat16,
        device=device,
    )
    quantized_sums = torch.full(
        (1, 1, pages, args.head_dim),
        16,
        dtype=torch.int8,
        device=device,
    )
    summary_scales = torch.ones_like(page_scales)
    local_length = sum(
        (position // args.interleave) % args.dcp_size == args.dcp_rank
        for position in range(args.tokens)
    )
    output = dequantize_owned_virtual_paged_keys(
        page_indices,
        page_counts,
        next_page,
        packed_keys,
        page_scales,
        quantized_sums,
        summary_scales,
        source_slot=0,
        sink_len=0,
        local_length=local_length,
        dcp_rank=args.dcp_rank,
        dcp_world_size=args.dcp_size,
        dcp_interleave_size=args.interleave,
    )
    torch.cuda.synchronize()
    expected = torch.ones_like(output)
    torch.testing.assert_close(output, expected)
    print(
        "KIMI_DCP_DEQUANT_OK",
        f"tokens={args.tokens}",
        f"pages={pages}",
        f"local_length={local_length}",
        f"shape={tuple(output.shape)}",
    )
    if args.build_cache:
        config = PagedLODConfig(
            kv_bits=4,
            quant_group_size=4,
            state_clustering_normalization="cosine",
            state_clustering_centroid_rescale="none",
            routing_normalization="none",
        )
        engine = KernelRecursivePagedLODAttention(
            config,
            query_heads=96,
            key_value_heads=1,
            scale=args.head_dim**-0.5,
            default_open_count=8,
        )
        engine.head_dim = args.head_dim
        configure_engine(
            engine,
            family=ModelFamily.KIMI_K3,
            mode=LODMode.THREE_TIER_INT4,
            request_capacity=local_length,
            has_query_norm=True,
            has_key_norm=False,
        )
        engine.state_growth_factor /= args.dcp_size**0.5
        engine.state_min_len = max(1, engine.state_min_len // args.dcp_size)
        cache = engine.build_cache_from_bf16(
            output,
            output[..., :512],
            finalize_cache_for_decode=True,
        )
        torch.cuda.synchronize()
        print(
            "KIMI_DCP_REBUILD_OK",
            f"state_len={cache.state['state_len']}",
            f"coverage={cache.state['coverage']}",
            f"total_len={cache.state['total_len']}",
        )
    if args.cached_prefill:
        query_heads = 12
        config = PagedLODConfig(
            kv_bits=4,
            quant_group_size=4,
            state_clustering_normalization="cosine",
            state_clustering_centroid_rescale="none",
            routing_normalization="none",
        )
        engine = KernelRecursivePagedLODAttention(
            config,
            query_heads=query_heads,
            key_value_heads=1,
            scale=args.head_dim**-0.5,
            default_open_count=8,
        )
        engine.head_dim = args.head_dim
        configure_engine(
            engine,
            family=ModelFamily.KIMI_K3,
            mode=LODMode.THREE_TIER_INT4,
            request_capacity=args.tokens,
            has_query_norm=True,
            has_key_norm=False,
        )
        turn = 16_384
        prefix = args.tokens - turn
        generator = torch.Generator(device=device).manual_seed(31)
        prefix_k = torch.randn(
            1,
            1,
            prefix,
            args.head_dim,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        cache = engine.build_cache_from_bf16(
            prefix_k, prefix_k[..., :512], finalize_cache_for_decode=True
        )
        del prefix_k
        new_k = torch.randn(
            1,
            1,
            turn,
            args.head_dim,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        expanded_q = torch.randn(
            1,
            query_heads,
            turn,
            192,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        w_uk_t = torch.randn(
            query_heads,
            128,
            512,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        w_uv = torch.randn(
            query_heads,
            512,
            128,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        engine._lod_kimi_expanded_prefill_query = expanded_q
        engine._lod_kimi_w_uk_t = w_uk_t
        engine._lod_kimi_w_uv = w_uv
        engine._lod_stage_cached_prefill_update = True
        carrier = new_k.expand(-1, query_heads, -1, -1)
        result, _ = engine(
            carrier,
            new_k,
            new_k[..., :512],
            cache=cache,
            use_cache=True,
            finalize_cache_for_decode=False,
        )
        torch.cuda.synchronize()
        print(
            "KIMI_CACHED_PREFILL_OK",
            f"prefix={prefix}",
            f"turn={turn}",
            f"state_len={cache.state['state_len']}",
            f"shape={tuple(result.shape)}",
        )


if __name__ == "__main__":
    main()

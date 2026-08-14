#!/usr/bin/env python3
"""Microbenchmark legacy and block-aggregated page ordinal allocation."""

from __future__ import annotations

import statistics

import torch

from model.kernels.paged_leaf_attention import (
    _assign_block_page_ordinals_kernel,
    _assign_page_ordinals_kernel,
)


def make_metadata(
    batch: int,
    kv_heads: int,
    slots: int,
    tokens: int,
) -> dict[str, torch.Tensor]:
    pages = batch * kv_heads * tokens
    return {
        "slot_lengths": torch.zeros(
            batch, kv_heads, slots, device="cuda", dtype=torch.int32
        ),
        "next_page": torch.zeros(
            batch, kv_heads, device="cuda", dtype=torch.int32
        ),
        "slot_pages": torch.full(
            (batch, kv_heads, slots, 128),
            -1,
            device="cuda",
            dtype=torch.int32,
        ),
        "overflow_keys": torch.full(
            (batch, kv_heads, 1), -1, device="cuda", dtype=torch.int32
        ),
        "overflow_values": torch.full(
            (batch, kv_heads, 1), -1, device="cuda", dtype=torch.int32
        ),
        "overflow_used": torch.zeros((), device="cuda", dtype=torch.int32),
        "overflow_flag": torch.zeros((), device="cuda", dtype=torch.int32),
        "ordinals": torch.empty(
            batch, kv_heads, tokens, device="cuda", dtype=torch.int32
        ),
        "page_capacity": torch.empty(pages, device="cuda", dtype=torch.int8),
    }


def reset(metadata: dict[str, torch.Tensor]) -> None:
    metadata["slot_lengths"].zero_()
    metadata["next_page"].zero_()
    metadata["slot_pages"].fill_(-1)
    metadata["overflow_keys"].fill_(-1)
    metadata["overflow_values"].fill_(-1)
    metadata["overflow_used"].zero_()
    metadata["overflow_flag"].zero_()


def launch_legacy(
    owners: torch.Tensor, metadata: dict[str, torch.Tensor]
) -> None:
    batch, kv_heads, tokens = owners.shape
    _assign_page_ordinals_kernel[(batch * kv_heads * tokens,)](
        owners,
        metadata["slot_lengths"],
        metadata["next_page"],
        metadata["slot_pages"],
        metadata["overflow_keys"],
        metadata["overflow_values"],
        metadata["overflow_used"],
        metadata["overflow_flag"],
        metadata["ordinals"],
        TOKENS=tokens,
        KV_HEADS=kv_heads,
        STATE_CAPACITY=int(metadata["slot_lengths"].size(2)),
        INLINE_PAGES_PER_SLOT=int(metadata["slot_pages"].size(3)),
        PAGE_CAPACITY=int(metadata["page_capacity"].numel()),
        HASH_CAPACITY=1,
        HASH_PROBES=0,
        PAGE_SIZE=16,
        num_warps=1,
    )


def launch_block(
    owners: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    block_tokens: int,
) -> None:
    batch, kv_heads, tokens = owners.shape
    import triton

    _assign_block_page_ordinals_kernel[
        (batch * kv_heads, triton.cdiv(tokens, block_tokens))
    ](
        owners,
        metadata["slot_lengths"],
        metadata["next_page"],
        metadata["slot_pages"],
        metadata["overflow_keys"],
        metadata["overflow_values"],
        metadata["overflow_used"],
        metadata["overflow_flag"],
        metadata["ordinals"],
        TOKENS=tokens,
        STATE_CAPACITY=int(metadata["slot_lengths"].size(2)),
        INLINE_PAGES_PER_SLOT=int(metadata["slot_pages"].size(3)),
        HASH_CAPACITY=1,
        HASH_PROBES=0,
        PAGE_SIZE=16,
        BLOCK_TOKENS=block_tokens,
        num_warps=1,
    )


def measure(
    launch,
    owners: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    repeats: int = 100,
) -> float:
    timings = []
    for iteration in range(repeats + 5):
        reset(metadata)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch(owners, metadata)
        end.record()
        end.synchronize()
        if iteration >= 5:
            timings.append(start.elapsed_time(end) * 1_000.0)
    return statistics.median(timings)


def main() -> None:
    torch.manual_seed(0)
    batch, kv_heads, tokens = 8, 2, 1024
    for slots in (64, 256, 1024, 4096):
        owners = torch.randint(
            slots,
            (batch, kv_heads, tokens),
            device="cuda",
            dtype=torch.long,
        )
        metadata = make_metadata(batch, kv_heads, slots, tokens)
        legacy_us = measure(launch_legacy, owners, metadata)
        result = {"slots": slots, "legacy_us": legacy_us}
        for block_tokens in (8, 16, 32):
            block_us = measure(
                lambda current_owners, current_metadata: launch_block(
                    current_owners, current_metadata, block_tokens
                ),
                owners,
                metadata,
            )
            result[f"block{block_tokens}_us"] = block_us
            result[f"block{block_tokens}_speedup"] = legacy_us / block_us
        print(result)


if __name__ == "__main__":
    main()

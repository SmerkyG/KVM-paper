#!/usr/bin/env python3
"""Verify fixed-pool conversion, mapped decode, and eager catch-up."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "integrations",
        "vllm_lod",
    ),
)

from vllm_lod_plugin.config import VLLMLODSettings
from vllm_lod_plugin.pool import VLLMLayerLODPool

from model.kernels.paged_leaf_attention import rehash_overflow_pages
from model.triton_lod_engines import KernelLODCache


class _Layer:
    def __init__(self, device: torch.device) -> None:
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_size = 128
        self.head_size_v = 128
        self.impl = SimpleNamespace(scale=128**-0.5)
        self.kv_cache = torch.empty(1, dtype=torch.bfloat16, device=device)


def _clone_cache(cache: KernelLODCache) -> KernelLODCache:
    def clone(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {name: clone(item) for name, item in value.items()}
        return value

    return KernelLODCache(clone(cache.state))


def _hash_index(key: int, capacity: int) -> int:
    value = key & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & (capacity - 1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), required=True)
    parser.add_argument(
        "--routing-geometry",
        choices=("auto", "raw", "spherical", "coherence"),
        default="raw",
    )
    parser.add_argument("--normalized-qk", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this check requires a CUDA or ROCm GPU")
    torch.manual_seed(17)
    device = torch.device("cuda")
    settings = VLLMLODSettings(
        chunk_size=16,
        local_window=32,
        state_growth_factor=4.0,
        state_min_size=16,
        protected_prefix=1,
        open_count=8,
        kv_bits=args.kv_bits,
        quant_group_size=32,
        pool_size=4,
        request_capacity=128,
        routing_geometry=args.routing_geometry,
    )
    active = torch.zeros(4, dtype=torch.long, device=device)
    pool = VLLMLayerLODPool(
        _Layer(device),
        settings=settings,
        max_requests=4,
        request_capacity=128,
        active_indices=active,
        dtype=torch.bfloat16,
        device=device,
        has_query_norm=args.normalized_qk,
        has_key_norm=args.normalized_qk,
    )
    # Force frequent state maintenance so the short test exercises catch-up.
    pool.engine.decode_state_update_len = 4

    source_keys = torch.full((1, 2, 8), -1, dtype=torch.int32, device=device)
    source_values = torch.full_like(source_keys, -1)
    hash_keys = (3 * 65_536 + 2, 7 * 65_536 + 4)
    source_keys[0, 0, :2] = torch.tensor(hash_keys, dtype=torch.int32, device=device)
    source_values[0, 0, :2] = torch.tensor((11, 29), dtype=torch.int32, device=device)
    destination_keys = torch.full(
        (4, 2, 32), -1, dtype=torch.int32, device=device
    )
    destination_values = torch.full_like(destination_keys, -1)
    overflow_used = torch.zeros((), dtype=torch.int32, device=device)
    overflow_flag = torch.zeros_like(overflow_used)
    rehash_overflow_pages(
        source_keys,
        source_values,
        destination_keys,
        destination_values,
        overflow_used,
        overflow_flag,
        source_slot=0,
        destination_slot=2,
    )
    torch.cuda.synchronize(device)
    for key, expected_value in zip(hash_keys, (11, 29)):
        index = _hash_index(key, 32)
        for _ in range(32):
            if int(destination_keys[2, 0, index].item()) == key:
                assert int(destination_values[2, 0, index].item()) == expected_value
                break
            index = (index + 1) & 31
        else:
            raise AssertionError("rehash dropped an overflow page entry")
    assert int(overflow_used.item()) == 1
    assert int(overflow_flag.item()) == 0

    direct_length = 24
    direct_query = torch.randn(
        2 * direct_length, 8, 128, dtype=torch.bfloat16, device=device
    )
    direct_key = torch.randn(
        2 * direct_length, 2, 128, dtype=torch.bfloat16, device=device
    )
    direct_value = torch.randn_like(direct_key)
    direct_output = torch.empty_like(direct_query)
    reference_initial, _ = pool.engine(
        torch.stack(
            (
                direct_query[:direct_length].permute(1, 0, 2),
                direct_query[direct_length:].permute(1, 0, 2),
            )
        ),
        torch.stack(
            (
                direct_key[:direct_length].permute(1, 0, 2),
                direct_key[direct_length:].permute(1, 0, 2),
            )
        ),
        torch.stack(
            (
                direct_value[:direct_length].permute(1, 0, 2),
                direct_value[direct_length:].permute(1, 0, 2),
            )
        ),
    )
    pool.direct_prefill_plan = (
        (2, 0, direct_length, 0),
        (3, direct_length, 2 * direct_length, 0),
    )
    pool.direct_prefill(direct_query, direct_key, direct_value, direct_output)
    assert pool.ready[2]
    assert pool.ready[3]
    assert int(pool.metadata[2]["total_len"]) == direct_length
    assert bool(torch.isfinite(direct_output).all())
    torch.testing.assert_close(
        direct_output.reshape(2, direct_length, 8, 128).float(),
        reference_initial.permute(0, 2, 1, 3).float(),
        rtol=4e-2,
        atol=2e-2,
    )

    cached_length = 5
    cached_query = torch.randn(
        2 * cached_length, 8, 128, dtype=torch.bfloat16, device=device
    )
    cached_key = torch.randn(
        2 * cached_length, 2, 128, dtype=torch.bfloat16, device=device
    )
    cached_value = torch.randn_like(cached_key)
    cached_output = torch.empty_like(cached_query)
    # vLLM may reorder active requests between scheduler iterations. The pool
    # batches a contiguous slot set regardless of the incoming plan order.
    pool.direct_prefill_plan = (
        (3, cached_length, 2 * cached_length, direct_length),
        (2, 0, cached_length, direct_length),
    )
    pool.direct_prefill(cached_query, cached_key, cached_value, cached_output)
    assert bool(torch.isfinite(cached_output).all())
    assert int(pool.metadata[2]["total_len"]) == direct_length + cached_length
    assert int(pool.metadata[3]["total_len"]) == direct_length + cached_length
    assert pool.batched_cached_prefill_calls == 1
    assert pool.batched_cached_prefill_rows == 2

    native_key = torch.randn(1, 2, 128, dtype=torch.bfloat16, device=device)
    native_value = torch.randn_like(native_key)
    prior_recent = int(pool.local_lens[2].item())
    pool.native_append_plan = (
        (2, 0, 1, direct_length + cached_length),
    )
    pool.record_native_appends(native_key, native_value)
    assert int(pool.metadata[2]["total_len"]) == direct_length + cached_length + 1
    assert int(pool.local_lens[2].item()) == prior_recent + 1
    torch.testing.assert_close(
        pool.state["recent_k"][2, :, prior_recent, :], native_key[0]
    )

    length = 40
    keys = [
        torch.randn(1, 2, length, 128, dtype=torch.bfloat16, device=device)
        for _ in range(2)
    ]
    values = [torch.randn_like(key) for key in keys]
    # The conversion engine owns reusable update scratch, so finish installing
    # one result before constructing the next request's result.
    for row, (key, value) in enumerate(zip(keys, values)):
        pool.install(row, pool.engine.build_cache_from_bf16(key, value))

    # Rows 2 and 3 emulate full-CUDA-graph padding. They are distinct cache
    # rows and are reset before each replay, so their fake appends cannot race
    # with or accumulate into either live request.
    metadata = SimpleNamespace(num_actual_tokens=4)
    active.copy_(torch.tensor([1, 0, 2, 3], dtype=torch.long, device=device))
    for step in range(12):
        pool.catch_up_many([(0, length + step), (1, length + step)])
        pool.local_lens[2:].zero_()
        query = torch.randn(4, 8, 128, dtype=torch.bfloat16, device=device)
        new_key = torch.randn(4, 2, 128, dtype=torch.bfloat16, device=device)
        new_value = torch.randn_like(new_key)
        expected = []
        for batch_row, pool_row in enumerate((1, 0)):
            reference = _clone_cache(pool._row_cache(pool_row))
            output, _ = pool.engine(
                query[batch_row : batch_row + 1].unsqueeze(2),
                new_key[batch_row : batch_row + 1].unsqueeze(2),
                new_value[batch_row : batch_row + 1].unsqueeze(2),
                cache=reference,
                use_cache=True,
            )
            # The engine intentionally reuses its fixed decode output buffer.
            # Preserve this row before the next reference call overwrites it.
            expected.append(output.squeeze(2).clone())
        output = torch.empty_like(query)
        pool.decode(query, new_key, new_value, metadata, output)
        reference = torch.cat(expected).float()
        torch.testing.assert_close(
            output[:2].float(),
            reference,
            rtol=4e-2,
            atol=2e-2,
            msg=(
                f"mapped fixed-pool output diverged at decode step {step}; "
                f"max_abs={(output[:2].float() - reference).abs().max().item():.6f}"
            ),
        )
    torch.cuda.synchronize(device)
    print(
        f"vLLM LOD fixed pool KV{args.kv_bits} "
        f"routing={args.routing_geometry} parity: PASS"
    )


if __name__ == "__main__":
    main()

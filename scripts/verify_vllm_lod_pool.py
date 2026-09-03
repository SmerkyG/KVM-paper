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


def _physical_to_indexed_reference(cache: KernelLODCache) -> None:
    """Express physical pages as equivalent indexed storage for parity."""
    page = cache.state["page_cache"]
    if "page_indices" in page:
        return
    page_k = page["page_k"]
    page_v = page["page_v"]
    batches, heads, page_capacity, page_size, head_dim = page_k.shape
    leaf_capacity = page_capacity * page_size
    page["leaf_k"] = page_k.reshape(batches, heads, leaf_capacity, head_dim)
    page["leaf_v"] = page_v.reshape(batches, heads, leaf_capacity, head_dim)
    page["page_indices"] = (
        torch.arange(leaf_capacity, dtype=torch.int32, device=page_k.device)
        .reshape(1, 1, page_capacity, page_size)
        .expand(batches, heads, -1, -1)
        .contiguous()
    )
    if "page_k_token_scales" in page:
        page["page_k_token_scales"] = page["page_k_token_scales"].reshape(
            batches, heads, leaf_capacity
        )
        page["page_v_token_scales"] = page["page_v_token_scales"].reshape(
            batches, heads, leaf_capacity
        )
    page["leaf_capacity"] = leaf_capacity


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
    parser.add_argument("--kv-bits", type=int, choices=(0, 4, 8), required=True)
    parser.add_argument("--levels", type=int, choices=(2, 3))
    parser.add_argument("--key-bits", type=int, choices=(0, 4, 8))
    parser.add_argument("--value-bits", type=int, choices=(0, 4, 8))
    parser.add_argument(
        "--routing-geometry",
        choices=("auto", "raw", "spherical", "coherence"),
        default="raw",
    )
    parser.add_argument("--normalized-qk", action="store_true")
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument(
        "--physical-pages",
        action="store_true",
        help="exercise physical page K/V storage instead of indexed leaves",
    )
    parser.add_argument(
        "--serving-reuse-length",
        type=int,
        default=0,
        help="also replay a scheduler-chunked prefill at serving scale",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this check requires a CUDA or ROCm GPU")
    torch.manual_seed(17)
    device = torch.device("cuda")
    settings = VLLMLODSettings(
        levels=args.levels or (2 if args.kv_bits == 0 else 3),
        chunk_size=16,
        local_window=32,
        state_growth_factor=4.0,
        state_min_size=16,
        protected_prefix=1,
        open_count=8,
        kv_bits=args.kv_bits,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        quant_group_size=32,
        pool_size=4,
        request_capacity=128,
        routing_geometry=args.routing_geometry,
        prefill_local_backend="torch",
        prefill_chunk_size=16,
        prefill_local_window=32,
        prefill_state_update_size=16,
        leaf_layout="expert" if args.kv_bits == 8 and args.levels == 2 else "query",
        leaf_block_m=16,
        leaf_block_n=32,
        leaf_num_warps=1,
        dense_leaf_storage=not args.physical_pages,
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
    if settings.levels == 2 and args.kv_bits == 0:
        # Exercise the cache-indexed cooperative Triton path at a small test
        # capacity; production enables it automatically at long context.
        pool._use_cooperative_decode = lambda: True

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
        (0, 0, direct_length, 0),
        (2, direct_length, 2 * direct_length, 0),
    )
    pool.direct_prefill_prompt_lengths = {0: direct_length, 2: direct_length}
    pool.direct_prefill(direct_query, direct_key, direct_value, direct_output)
    assert pool.ready[0]
    assert pool.ready[2]
    assert int(pool.metadata[0]["total_len"]) == direct_length
    assert bool(torch.isfinite(direct_output).all())
    torch.testing.assert_close(
        direct_output.reshape(2, direct_length, 8, 128).float(),
        reference_initial.permute(0, 2, 1, 3).float(),
        rtol=4e-2,
        atol=2e-2,
    )
    if args.prefill_only:
        page = pool.state["page_cache"]
        if args.kv_bits == 8 and args.levels == 2:
            key_storage = page.get("leaf_k", page.get("page_k"))
            value_storage = page.get("leaf_v", page.get("page_v"))
            assert isinstance(key_storage, torch.Tensor)
            assert isinstance(value_storage, torch.Tensor)
            assert key_storage.dtype == torch.int8
            assert value_storage.dtype == torch.int8
            assert bool((page["page_k_token_scales"][:2] > 0).any())
            assert bool((page["page_v_token_scales"][:2] > 0).any())
        print("vLLM LOD prefill verification passed")
        return

    cached_length = 5
    cached_query = torch.randn(
        4 * cached_length, 8, 128, dtype=torch.bfloat16, device=device
    )
    cached_key = torch.randn(
        4 * cached_length, 2, 128, dtype=torch.bfloat16, device=device
    )
    cached_value = torch.randn_like(cached_key)
    cached_output = torch.empty_like(cached_query)
    # vLLM may mix fresh and continuing requests, and reorder active requests,
    # within one scheduler iteration. Batch each lifecycle group separately.
    pool.direct_prefill_plan = (
        (2, 0, cached_length, direct_length),
        (1, cached_length, 2 * cached_length, 0),
        (0, 2 * cached_length, 3 * cached_length, direct_length),
        (3, 3 * cached_length, 4 * cached_length, 0),
    )
    pool.direct_prefill_prompt_lengths = {
        0: direct_length + cached_length,
        1: cached_length,
        2: direct_length + cached_length,
        3: cached_length,
    }
    pool.direct_prefill(cached_query, cached_key, cached_value, cached_output)
    assert bool(torch.isfinite(cached_output).all())
    assert int(pool.metadata[1]["total_len"]) == cached_length
    assert int(pool.metadata[3]["total_len"]) == cached_length
    assert int(pool.metadata[0]["total_len"]) == direct_length + cached_length
    assert int(pool.metadata[2]["total_len"]) == direct_length + cached_length
    assert pool.batched_cached_prefill_calls == 1
    assert pool.batched_cached_prefill_rows == 2

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
            if args.physical_pages:
                # Compare physical addressing against the same pages exposed
                # through the already-verified indexed addressing path.
                _physical_to_indexed_reference(reference)
            # Captured vLLM decode specializes one fixed state extent for all
            # rows. Zero-count padding must therefore be semantically inert.
            reference.state["state_len"] = pool.state_capacity
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
        if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(output[:2]).all()):
            raise AssertionError(
                f"nonfinite decode: reference={torch.isfinite(reference).float().mean().item():.4f} "
                f"pool={torch.isfinite(output[:2]).float().mean().item():.4f}"
            )
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

    # Reusing a released serving row must reproduce the same chunked prefill.
    # This catches stale pool fields that ordinary fresh-row checks miss.
    reuse_split, reuse_length = 80, 112
    reuse_query = torch.randn(
        reuse_length, 8, 128, dtype=torch.bfloat16, device=device
    )
    reuse_key = torch.randn(
        reuse_length, 2, 128, dtype=torch.bfloat16, device=device
    )
    reuse_value = torch.randn_like(reuse_key)
    reuse_pool = VLLMLayerLODPool(
        _Layer(device),
        settings=settings,
        max_requests=4,
        request_capacity=128,
        active_indices=torch.zeros(4, dtype=torch.long, device=device),
        dtype=torch.bfloat16,
        device=device,
        has_query_norm=args.normalized_qk,
        has_key_norm=args.normalized_qk,
    )

    # Keep the source ranges and result buffer explicit because vLLM passes
    # flattened layer-wide Q/K/V tensors rather than turn-local slices.
    def run_reused_prefill() -> torch.Tensor:
        output = torch.empty_like(reuse_query)
        reuse_pool.direct_prefill_plan = ((0, 0, reuse_split, 0),)
        reuse_pool.direct_prefill_prompt_lengths = {0: reuse_length}
        reuse_pool.direct_prefill(reuse_query, reuse_key, reuse_value, output)
        reuse_pool.direct_prefill_plan = (
            (0, reuse_split, reuse_length, reuse_split),
        )
        reuse_pool.direct_prefill_prompt_lengths = {0: reuse_length}
        reuse_pool.direct_prefill(reuse_query, reuse_key, reuse_value, output)
        return output.clone()

    first_reuse = run_reused_prefill()
    reuse_pool.reset(0)
    second_reuse = run_reused_prefill()
    torch.testing.assert_close(
        second_reuse.float(),
        first_reuse.float(),
        rtol=0,
        atol=0,
        msg="reset LOD pool row changed an identical chunked prefill",
    )

    if args.serving_reuse_length:
        serving_length = args.serving_reuse_length
        serving_capacity = max(32_768, serving_length + 32)
        serving_settings = VLLMLODSettings(
            kv_bits=args.kv_bits,
            pool_size=1,
            request_capacity=serving_capacity,
            routing_geometry=args.routing_geometry,
            prefill_local_backend="torch",
        )
        serving_pool = VLLMLayerLODPool(
            _Layer(device),
            settings=serving_settings,
            max_requests=1,
            request_capacity=serving_capacity,
            active_indices=torch.zeros(1, dtype=torch.long, device=device),
            dtype=torch.bfloat16,
            device=device,
            has_query_norm=args.normalized_qk,
            has_key_norm=args.normalized_qk,
        )
        serving_query = torch.randn(
            serving_length, 8, 128, dtype=torch.bfloat16, device=device
        )
        serving_key = torch.randn(
            serving_length, 2, 128, dtype=torch.bfloat16, device=device
        )
        serving_value = torch.randn_like(serving_key)
        serving_decode_query = torch.randn(
            32, 8, 128, dtype=torch.bfloat16, device=device
        )
        serving_decode_key = torch.randn(
            32, 2, 128, dtype=torch.bfloat16, device=device
        )
        serving_decode_value = torch.randn_like(serving_decode_key)

        def run_serving_request() -> tuple[torch.Tensor, torch.Tensor]:
            output = torch.empty_like(serving_query)
            previous_length = 0
            while previous_length < serving_length:
                end = min(previous_length + 4096, serving_length)
                serving_pool.direct_prefill_plan = (
                    (0, previous_length, end, previous_length),
                )
                serving_pool.direct_prefill_prompt_lengths = {0: serving_length}
                serving_pool.direct_prefill(
                    serving_query, serving_key, serving_value, output
                )
                previous_length = end
            decode_outputs = []
            for decode_step in range(32):
                previous_length = serving_length + decode_step
                serving_pool.catch_up_many([(0, previous_length)])
                decode_output = torch.empty_like(
                    serving_decode_query[decode_step : decode_step + 1]
                )
                serving_pool.decode(
                    serving_decode_query[decode_step : decode_step + 1],
                    serving_decode_key[decode_step : decode_step + 1],
                    serving_decode_value[decode_step : decode_step + 1],
                    SimpleNamespace(num_actual_tokens=1),
                    decode_output,
                )
                decode_outputs.append(decode_output.clone())
            return output.clone(), torch.cat(decode_outputs)

        first_serving, first_serving_decode = run_serving_request()
        serving_pool.reset(0)
        second_serving, second_serving_decode = run_serving_request()
        torch.testing.assert_close(
            second_serving.float(),
            first_serving.float(),
            rtol=0,
            atol=0,
            msg="reset LOD pool row changed a serving-scale chunked prefill",
        )
        serving_decode_delta = (
            second_serving_decode.float() - first_serving_decode.float()
        ).abs()
        torch.testing.assert_close(
            second_serving_decode.float(),
            first_serving_decode.float(),
            rtol=0,
            atol=0,
            msg=(
                "reset LOD pool row changed serving-scale decode: "
                f"max={serving_decode_delta.max().item():.6f} "
                f"mean={serving_decode_delta.mean().item():.6f} "
                f"per_step_max="
                f"{serving_decode_delta.flatten(1).max(dim=1).values.tolist()}"
            ),
        )
    torch.cuda.synchronize(device)
    print(
        f"vLLM LOD fixed pool KV{args.kv_bits} "
        f"K{settings.resolved_key_bits}/V{settings.resolved_value_bits} "
        f"routing={args.routing_geometry} parity: PASS"
    )


if __name__ == "__main__":
    main()

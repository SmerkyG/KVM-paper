#!/usr/bin/env python3
"""Compare grouped and legacy page-wide INT4 cached-prefill appends."""

from __future__ import annotations

import json
import os

import torch

from model.kernels.paged_leaf_attention import (
    append_quantized_virtual_paged_kv,
    append_virtual_paged_kv,
    quantize_page_summaries_int8,
    quantize_virtual_paged_kv,
)


def main() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    batch, heads, dimension = 1, 2, 256
    state_capacity, page_capacity, page_size = 32, 128, 16
    initial_tokens, append_tokens = 512, 256
    leaf_capacity = initial_tokens + append_tokens
    initial_k = torch.randn(
        batch,
        heads,
        initial_tokens,
        dimension,
        dtype=torch.bfloat16,
        device=device,
    )
    initial_v = torch.randn_like(initial_k)
    append_k = torch.randn(
        batch,
        heads,
        append_tokens,
        dimension,
        dtype=torch.bfloat16,
        device=device,
    )
    append_v = torch.randn_like(append_k)
    initial_owners = (
        torch.arange(initial_tokens, dtype=torch.int32, device=device)
        .remainder(state_capacity)
        .reshape(1, 1, -1)
        .repeat(batch, heads, 1)
    )
    append_owners = (
        torch.arange(append_tokens, dtype=torch.int32, device=device)
        .remainder(state_capacity)
        .reshape(1, 1, -1)
        .repeat(batch, heads, 1)
    )
    leaf_k = torch.zeros(
        batch,
        heads,
        leaf_capacity,
        dimension,
        dtype=torch.bfloat16,
        device=device,
    )
    leaf_v = torch.zeros_like(leaf_k)
    leaf_k[..., :initial_tokens, :].copy_(initial_k)
    leaf_v[..., :initial_tokens, :].copy_(initial_v)
    page_indices = torch.full(
        (batch, heads, page_capacity, page_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    slot_pages = torch.full(
        (batch, heads, state_capacity, 4),
        -1,
        dtype=torch.int32,
        device=device,
    )
    overflow_page_keys = torch.full(
        (batch, heads, 256), -1, dtype=torch.int32, device=device
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), dtype=torch.int32, device=device)
    overflow_flag = torch.zeros((), dtype=torch.int32, device=device)
    slot_lengths = torch.zeros(
        batch, heads, state_capacity, dtype=torch.int32, device=device
    )
    next_page = torch.zeros(batch, heads, dtype=torch.int32, device=device)
    page_sum_k = torch.zeros(
        batch,
        heads,
        page_capacity,
        dimension,
        dtype=torch.bfloat16,
        device=device,
    )
    page_sum_v = torch.zeros_like(page_sum_k)
    page_counts = torch.zeros(
        batch, heads, page_capacity, dtype=torch.int32, device=device
    )
    append_virtual_paged_kv(
        leaf_k,
        leaf_v,
        0,
        initial_owners,
        page_indices,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        slot_lengths,
        next_page,
        page_sum_k,
        page_sum_v,
        page_counts,
        hash_probes=0,
        quantize_touched=False,
    )
    quantized_leaf_k = torch.zeros(
        batch,
        heads,
        leaf_capacity,
        dimension // 2,
        dtype=torch.uint8,
        device=device,
    )
    quantized_leaf_v = torch.zeros_like(quantized_leaf_k)
    page_k_scales = torch.zeros(
        batch,
        heads,
        page_capacity,
        dimension // 4,
        dtype=torch.bfloat16,
        device=device,
    )
    page_v_scales = torch.zeros_like(page_k_scales)
    page_quantized_counts = torch.zeros_like(page_counts)
    os.environ["VLLM_LOD_INT4_QUANT_GROUPS_PER_PROGRAM"] = "1"
    quantize_virtual_paged_kv(
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        quantized_leaf_k,
        quantized_leaf_v,
        page_k_scales,
        page_v_scales,
        page_quantized_counts,
        quant_group_size=4,
        quant_token_group_size=16,
        quant_bits=4,
        optimize_scale=True,
    )
    (
        quantized_page_sum_k,
        quantized_page_sum_v,
        page_sum_k_scales,
        page_sum_v_scales,
    ) = quantize_page_summaries_int8(
        page_sum_k,
        page_sum_v,
        quant_group_size=4,
        optimize_scale=True,
    )
    base = {
        "page_indices": page_indices,
        "slot_pages": slot_pages,
        "overflow_page_keys": overflow_page_keys,
        "overflow_page_values": overflow_page_values,
        "overflow_used": overflow_used,
        "overflow_flag": overflow_flag,
        "slot_lengths": slot_lengths,
        "next_page": next_page,
        "page_sum_k": page_sum_k,
        "page_sum_v": page_sum_v,
        "page_counts": page_counts,
        "quantized_leaf_k": quantized_leaf_k,
        "quantized_leaf_v": quantized_leaf_v,
        "page_k_scales": page_k_scales,
        "page_v_scales": page_v_scales,
        "page_quantized_counts": page_quantized_counts,
        "quantized_page_sum_k": quantized_page_sum_k,
        "quantized_page_sum_v": quantized_page_sum_v,
        "page_sum_k_scales": page_sum_k_scales,
        "page_sum_v_scales": page_sum_v_scales,
    }

    def execute(groups_per_program: int) -> dict[str, torch.Tensor]:
        state = {name: tensor.clone() for name, tensor in base.items()}
        os.environ["VLLM_LOD_INT4_QUANT_GROUPS_PER_PROGRAM"] = str(
            groups_per_program
        )
        append_quantized_virtual_paged_kv(
            append_k,
            append_v,
            initial_tokens,
            append_owners,
            state["page_indices"],
            state["slot_pages"],
            state["overflow_page_keys"],
            state["overflow_page_values"],
            state["overflow_used"],
            state["overflow_flag"],
            state["slot_lengths"],
            state["next_page"],
            state["page_sum_k"],
            state["page_sum_v"],
            state["page_counts"],
            state["quantized_leaf_k"],
            state["quantized_leaf_v"],
            state["page_k_scales"],
            state["page_v_scales"],
            state["page_quantized_counts"],
            hash_probes=0,
            quant_group_size=4,
            quant_token_group_size=16,
            quant_bits=4,
            quantized_page_sum_k=state["quantized_page_sum_k"],
            quantized_page_sum_v=state["quantized_page_sum_v"],
            page_sum_k_scales=state["page_sum_k_scales"],
            page_sum_v_scales=state["page_sum_v_scales"],
            optimize_summary_scale=True,
            optimize_leaf_scale=True,
        )
        torch.cuda.synchronize()
        return state

    reference = execute(1)
    result: dict[str, object] = {}
    for groups_per_program in (2, 4, 8):
        candidate = execute(groups_per_program)
        fields = {}
        for name, expected in reference.items():
            actual = candidate[name]
            if expected.is_floating_point():
                difference = float((expected.float() - actual.float()).abs().max())
                fields[name] = {"max_abs": difference}
            else:
                fields[name] = {"equal": bool(torch.equal(expected, actual))}
        result[str(groups_per_program)] = fields
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("LOD_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set LOD_RUN_GPU_TESTS=1 on a CUDA/ROCm worker",
)


def _page_cache(*, leaf_capacity: int, dimension: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    page_capacity = 4
    return {
        "page_indices": torch.full(
            (1, 1, page_capacity, 16), -1, dtype=torch.int32, device=device
        ),
        "slot_pages": torch.full((1, 1, 1, 1), -1, dtype=torch.int32, device=device),
        # One overflow bucket and one probe deliberately force the third page
        # to miss after page zero is inline and page one occupies this bucket.
        "overflow_page_keys": torch.full(
            (1, 1, 1), -1, dtype=torch.int32, device=device
        ),
        "overflow_page_values": torch.full(
            (1, 1, 1), -1, dtype=torch.int32, device=device
        ),
        "overflow_used": torch.zeros((), dtype=torch.int32, device=device),
        "overflow_flag": torch.zeros((), dtype=torch.int32, device=device),
        "slot_lengths": torch.zeros((1, 1, 1), dtype=torch.int32, device=device),
        "next_page": torch.zeros((1, 1), dtype=torch.int32, device=device),
        "page_sum_k": torch.zeros(
            (1, 1, page_capacity, dimension), dtype=torch.bfloat16, device=device
        ),
        "page_sum_v": torch.zeros(
            (1, 1, page_capacity, dimension), dtype=torch.bfloat16, device=device
        ),
        "page_counts": torch.zeros(
            (1, 1, page_capacity), dtype=torch.int32, device=device
        ),
        "quantized_leaf_k": torch.empty(
            (1, 1, leaf_capacity, dimension // 2),
            dtype=torch.uint8,
            device=device,
        ),
        "quantized_leaf_v": torch.empty(
            (1, 1, leaf_capacity, dimension // 2),
            dtype=torch.uint8,
            device=device,
        ),
        "page_k_scales": torch.empty(
            (1, 1, page_capacity, dimension // 4),
            dtype=torch.bfloat16,
            device=device,
        ),
        "page_v_scales": torch.empty(
            (1, 1, page_capacity, dimension // 4),
            dtype=torch.bfloat16,
            device=device,
        ),
        "page_quantized_counts": torch.zeros(
            (1, 1, page_capacity), dtype=torch.int32, device=device
        ),
    }


def test_bounded_page_hash_miss_never_forms_an_out_of_bounds_address() -> None:
    """A full bounded hash must safely leave the missing page at coarse LOD."""
    from lod_attention.kernels.paged_cache import (
        append_quantized_virtual_paged_kv,
        append_virtual_paged_kv,
        quantize_virtual_paged_kv,
    )
    from lod_attention.kernels.paged_prefill import paged_leaf_attention

    dimension = 16
    leaf_capacity = 48
    cache = _page_cache(leaf_capacity=leaf_capacity, dimension=dimension)
    leaf_k = torch.randn(
        (1, 1, leaf_capacity, dimension), dtype=torch.bfloat16, device="cuda"
    )
    leaf_v = torch.randn_like(leaf_k)

    first_owners = torch.zeros((1, 1, 32), dtype=torch.int32, device="cuda")
    append_virtual_paged_kv(
        leaf_k,
        leaf_v,
        0,
        first_owners,
        cache["page_indices"],
        cache["slot_pages"],
        cache["overflow_page_keys"],
        cache["overflow_page_values"],
        cache["overflow_used"],
        cache["overflow_flag"],
        cache["slot_lengths"],
        cache["next_page"],
        cache["page_sum_k"],
        cache["page_sum_v"],
        cache["page_counts"],
        hash_probes=1,
    )
    quantize_virtual_paged_kv(
        leaf_k,
        leaf_v,
        cache["page_indices"],
        cache["page_sum_k"],
        cache["page_sum_v"],
        cache["page_counts"],
        cache["quantized_leaf_k"],
        cache["quantized_leaf_v"],
        cache["page_k_scales"],
        cache["page_v_scales"],
        cache["page_quantized_counts"],
    )

    final_owners = torch.zeros((1, 1, 16), dtype=torch.int32, device="cuda")
    append_quantized_virtual_paged_kv(
        leaf_k[:, :, 32:],
        leaf_v[:, :, 32:],
        32,
        final_owners,
        cache["page_indices"],
        cache["slot_pages"],
        cache["overflow_page_keys"],
        cache["overflow_page_values"],
        cache["overflow_used"],
        cache["overflow_flag"],
        cache["slot_lengths"],
        cache["next_page"],
        cache["page_sum_k"],
        cache["page_sum_v"],
        cache["page_counts"],
        cache["quantized_leaf_k"],
        cache["quantized_leaf_v"],
        cache["page_k_scales"],
        cache["page_v_scales"],
        cache["page_quantized_counts"],
        hash_probes=1,
    )
    torch.cuda.synchronize()

    assert cache["overflow_flag"].item() == 1
    assert cache["slot_lengths"].item() == leaf_capacity
    assert cache["next_page"].item() == 3
    # The failed page remains represented in the parent centroid. The two
    # addressable pages are still complete and consistently quantized.
    counts = cache["page_counts"].cpu()
    quantized_counts = cache["page_quantized_counts"].cpu()
    assert torch.count_nonzero(counts).item() == 2
    assert torch.equal(counts, quantized_counts)

    query = torch.randn((1, 1, 1, dimension), dtype=torch.bfloat16, device="cuda")
    top_slots = torch.zeros((1, 1, 1, 1), dtype=torch.int64, device="cuda")
    output, lse = paged_leaf_attention(
        query,
        leaf_k,
        leaf_v,
        cache["slot_pages"],
        cache["overflow_page_keys"],
        cache["overflow_page_values"],
        cache["overflow_used"],
        cache["slot_lengths"],
        top_slots,
        page_indices=cache["page_indices"],
        page_k_scales=cache["page_k_scales"],
        page_v_scales=cache["page_v_scales"],
        quantized_leaf_k=cache["quantized_leaf_k"],
        quantized_leaf_v=cache["quantized_leaf_v"],
        page_sum_k=cache["page_sum_k"],
        page_sum_v=cache["page_sum_v"],
        page_counts=cache["page_counts"],
        quant_group_size=4,
        kv_group_size=1,
        active_slots=1,
        scale=dimension**-0.5,
        hash_probes=1,
        block_m=16,
        block_n=16,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()


@pytest.mark.parametrize("block_n", [16, 32, 64, 128])
@pytest.mark.parametrize("hash_probes", [0, -1, 1])
def test_grouped_page_lookup_matches_per_token_lookup(block_n, hash_probes) -> None:
    """Grouped BF16 lookup retains partial pages, GQA and safe hash misses."""
    from lod_attention.kernels.paged_cache import append_virtual_paged_kv
    from lod_attention.kernels.paged_prefill import paged_leaf_attention

    torch.manual_seed(194)
    dimension, leaves = 32, 61
    cache = _page_cache(leaf_capacity=leaves, dimension=dimension)
    if hash_probes == 0:
        cache['slot_pages'] = torch.full((1, 1, 1, 4), -1, dtype=torch.int32, device='cuda')
    if hash_probes == -1:
        cache['overflow_page_keys'] = torch.full((1, 1, 4), -1, dtype=torch.int32, device='cuda')
        cache['overflow_page_values'] = torch.full_like(cache['overflow_page_keys'], -1)
    key = torch.randn((1, 1, leaves, dimension), device='cuda').bfloat16()
    value = torch.randn_like(key)
    owners = torch.zeros((1, 1, leaves), dtype=torch.int32, device='cuda')
    append_virtual_paged_kv(
        key, value, 0, owners, cache['page_indices'], cache['slot_pages'],
        cache['overflow_page_keys'], cache['overflow_page_values'], cache['overflow_used'],
        cache['overflow_flag'], cache['slot_lengths'], cache['next_page'],
        cache['page_sum_k'], cache['page_sum_v'], cache['page_counts'],
        hash_probes=hash_probes)
    query = torch.randn((1, 2, 37, dimension), device='cuda').bfloat16()
    routes = torch.full((1, 2, 37, 4), -1, dtype=torch.int64, device='cuda')
    routes[..., 0] = 0

    def run(grouped):
        return paged_leaf_attention(
            query, key, value, cache['slot_pages'], cache['overflow_page_keys'],
            cache['overflow_page_values'], cache['overflow_used'], cache['slot_lengths'],
            routes, page_indices=cache['page_indices'], kv_group_size=2, active_slots=1,
            scale=dimension**-0.5, hash_probes=hash_probes, block_m=16, block_n=block_n,
            scalar_page_lookup=grouped)

    reference = run(False)
    grouped = run(True)
    for actual, expected in zip(grouped, reference, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)

"""Multi-page frontiers must replace each opened leaf exactly once."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("LOD_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set LOD_RUN_GPU_TESTS=1 on a CUDA/ROCm worker",
)


@pytest.mark.parametrize("dimension,heads", [(128, 8), (256, 6)])
@pytest.mark.parametrize("pages_per_slot", [None, 1, 2, 4, 8])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("summary_precision", ["bf16", "int8"])
def test_top_pages_match_disjoint_frontier(dimension, heads, pages_per_slot, quantized, summary_precision):
    from lod_attention.kernels.paged_prefill import query_major_indexed_residual_page_attention

    torch.manual_seed(13)
    device = "cuda"
    lengths = [7, 37, 69]
    leaf_k = torch.randn(1, 1, sum(lengths), dimension, dtype=torch.bfloat16, device=device)
    mla = dimension in (512, 576)
    value_dim = 512 if mla else dimension
    leaf_v = leaf_k[..., :value_dim] if mla else torch.randn_like(leaf_k)
    query = torch.randn(1, heads, 1, dimension, dtype=torch.bfloat16, device=device)
    indices = torch.full((1, 1, 9, 16), -1, dtype=torch.int32, device=device)
    table = torch.full((1, 1, 3, 5), -1, dtype=torch.int32, device=device)
    sums_k = torch.zeros(1, 1, 9, dimension, dtype=torch.bfloat16, device=device)
    sums_v = sums_k[..., :value_dim] if mla else torch.zeros_like(sums_k)
    counts = torch.zeros(1, 1, 9, dtype=torch.int32, device=device)
    state_k = torch.zeros(1, 1, 3, dimension, dtype=torch.bfloat16, device=device)
    state_v = state_k[..., :value_dim] if mla else torch.zeros_like(state_k)
    slot_lengths = torch.tensor([[lengths]], dtype=torch.int32, device=device)
    state_counts = slot_lengths.float()
    routes = torch.tensor([0, 1, 2, -1], dtype=torch.int64, device=device).expand(1, heads, 1, 4).contiguous()
    pages, token = [], 0
    for slot, length in enumerate(lengths):
        state_k[0, 0, slot] = leaf_k[0, 0, token:token+length].float().sum(0)
        state_v[0, 0, slot] = leaf_v[0, 0, token:token+length].float().sum(0)
        owned = []
        for ordinal, begin in enumerate(range(token, token+length, 16)):
            end = min(begin+16, token+length)
            page = len(pages)
            indices[0, 0, page, :end-begin] = torch.arange(begin, end, device=device)
            table[0, 0, slot, ordinal] = page
            sums_k[0, 0, page] = leaf_k[0, 0, begin:end].float().sum(0)
            sums_v[0, 0, page] = leaf_v[0, 0, begin:end].float().sum(0)
            counts[0, 0, page] = end-begin
            pages.append((begin, end))
            owned.append(page)
        token += length
    overflow = torch.zeros(1, 1, 1, dtype=torch.int32, device=device)
    options = {}
    if pages_per_slot is not None:
        options["pages_per_slot"] = pages_per_slot
    effective_sums_k, effective_sums_v = sums_k.float(), sums_v.float()
    if summary_precision == "int8":
        from lod_attention.kernels.paged_cache import quantize_page_summaries_int8

        group = 4 if quantized else 32
        code_k, code_v, scale_k, scale_v = quantize_page_summaries_int8(sums_k, sums_v, quant_group_size=group)
        if mla:
            code_v, scale_v = code_k[..., :value_dim], scale_k[..., :value_dim // group]
        options.update(quantized_page_sum_k=code_k, quantized_page_sum_v=code_v,
                       page_sum_k_scales=scale_k, page_sum_v_scales=scale_v,
                       quant_group_size=group)
        effective_sums_k = code_k.float() * scale_k.float().repeat_interleave(group, -1)
        effective_sums_v = code_v.float() * scale_v.float().repeat_interleave(group, -1)
    reference_k, reference_v = leaf_k.float(), leaf_v.float()
    if quantized:
        from lod_attention.kernels.paged_cache import quantize_virtual_paged_kv

        packed_k = torch.empty(1, 1, sum(lengths), dimension//2, dtype=torch.uint8, device=device)
        packed_v = packed_k[..., :value_dim // 2] if mla else torch.empty_like(packed_k)
        scales_k = torch.empty(1, 1, 9, dimension//4, dtype=torch.bfloat16, device=device)
        scales_v = scales_k[..., :value_dim // 4] if mla else torch.empty_like(scales_k)
        quantized_counts = torch.zeros_like(counts)
        quantize_virtual_paged_kv(leaf_k, leaf_v, indices, sums_k, sums_v, counts,
                                 packed_k, packed_v, scales_k, scales_v, quantized_counts)
        options.update(quantized_leaf_k=packed_k, quantized_leaf_v=packed_v,
                       page_k_scales=scales_k, page_v_scales=scales_v,
                       page_quantized_counts=quantized_counts, quant_group_size=4)
        # Independent decoding of actual stored nibbles and page anchors.
        for page, (begin, end) in enumerate(pages):
            for packed, scales, sums, destination in [(packed_k, scales_k, effective_sums_k, reference_k),
                                                       (packed_v, scales_v, effective_sums_v, reference_v)]:
                codes = packed[0, 0, begin:end].int()
                codes = torch.stack((codes & 15, (codes >> 4) & 15), -1).flatten(-2).float() - 8
                destination[0, 0, begin:end] = codes * scales[0, 0, page].float().repeat_interleave(4) + sums[0, 0, page].float()/int(counts[0, 0, page])
    arguments = (
        query, state_k, state_v, state_counts, leaf_k, leaf_v,
        indices, sums_k, sums_v, counts, table, overflow, overflow,
        torch.zeros((), dtype=torch.int32, device=device), slot_lengths, routes,
    )
    launch = dict(
        kv_group_size=heads, scale=dimension**-0.5, hash_probes=0,
        route_parallel=True, num_warps=1 if dimension == 128 else 2, **options,
    )
    output, lse = query_major_indexed_residual_page_attention(*arguments, **launch)
    if pages_per_slot is None and dimension == 128:
        # The batch-8 occupancy specialization must retain exactly the same
        # partial pages, fewer-than-two-page slots, residuals and invalid routes
        # as the one-row calculation checked against the independent oracle.
        def eight_rows(tensor):
            if isinstance(tensor, torch.Tensor) and tensor.ndim:
                return tensor.repeat(8, *([1] * (tensor.ndim - 1)))
            return tensor

        batched_output, batched_lse = query_major_indexed_residual_page_attention(
            *(eight_rows(tensor) for tensor in arguments),
            **{name: eight_rows(value) for name, value in launch.items()},
        )
        assert torch.equal(batched_output, output.expand(8, -1, -1, -1))
        assert torch.equal(batched_lse, lse.expand(8, -1, -1))
        # Serving uses the two-level 64-entry directory, rather than the
        # inline directory above. Compare each tuned batched row to its
        # untuned one-row calculation using that actual lookup path too.
        directory = torch.full((1, 1, 3, 64), -1, dtype=torch.int32, device=device)
        directory[:, :, :, :5] = table
        directory_arguments = list(arguments)
        directory_arguments[10] = torch.arange(3, dtype=torch.int32, device=device).reshape(1, 1, 3, 1)
        directory_arguments[11] = torch.full((1, 1, 3), -1, dtype=torch.int32, device=device)
        directory_arguments[12] = directory
        directory_launch = dict(launch, hash_probes=-1)
        one_output, one_lse = query_major_indexed_residual_page_attention(
            *directory_arguments, **directory_launch,
        )
        directory_output, directory_lse = query_major_indexed_residual_page_attention(
            *(eight_rows(tensor) for tensor in directory_arguments),
            **{name: eight_rows(value) for name, value in directory_launch.items()},
        )
        assert torch.equal(directory_output, one_output.expand(8, -1, -1, -1))
        assert torch.equal(directory_lse, one_lse.expand(8, -1, -1))
    for head in range(heads):
        q = query[0, head, 0].float()
        for slot in range(3):
            owned = table[0, 0, slot]
            owned = owned[owned >= 0].long()
            page_counts = counts[0, 0, owned].float()
            scores = (effective_sums_k[0, 0, owned] / page_counts[:, None]) @ q * dimension**-0.5 + page_counts.log()
            budget = 2 if pages_per_slot is None else pages_per_slot
            selected = owned[scores.topk(min(budget, len(owned))).indices]
            opened = torch.cat([torch.arange(*pages[int(p)], device=device) for p in selected])
            key = reference_k[0, 0, opened]
            value = reference_v[0, 0, opened]
            logits = key @ q * dimension**-0.5
            remaining = lengths[slot] - len(opened)
            if remaining:
                residual_k = (state_k[0, 0, slot].float() - effective_sums_k[0, 0, selected].sum(0)) / remaining
                residual_v = (state_v[0, 0, slot].float() - effective_sums_v[0, 0, selected].sum(0)) / remaining
                logits = torch.cat((logits, ((residual_k @ q) * dimension**-0.5 + torch.tensor(remaining, device=device).log())[None]))
                value = torch.cat((value, residual_v[None]))
            expected = logits.softmax(0) @ value
            torch.testing.assert_close(output[0, head, slot].float(), expected, atol=0.015, rtol=0.015)
            torch.testing.assert_close(lse[0, head, slot], logits.logsumexp(0), atol=0.002, rtol=0.002)
        assert output[0, head, 3].eq(0).all()
        assert lse[0, head, 3].isneginf()


@pytest.mark.parametrize("dimension", [512, 576])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("summary_precision", ["bf16", "int8"])
def test_mla_top2_shared_latent_and_strided_value_prefix(dimension, quantized, summary_precision):
    # Exercise the actual shared-latent layout, not independent contiguous V:
    # K3's 512-value channels retain the 576-key record's physical strides.
    test_top_pages_match_disjoint_frontier(dimension, 16, None, quantized, summary_precision)


@pytest.mark.parametrize("block", [4, 16])
@pytest.mark.parametrize("case", ["all_ties", "later_winner", "ties_across_blocks"])
def test_top2_preserves_ties_and_runner_up_across_scan_blocks(block, case):
    from lod_attention.kernels.paged_prefill import query_major_indexed_residual_page_attention

    torch.manual_seed(71)
    device, dimension, pages = "cuda", 128, 40
    key = torch.randn(1, 1, pages * 16, dimension, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn(1, 1, 1, dimension, device=device, dtype=torch.bfloat16)
    sums_k = key.reshape(1, 1, pages, 16, dimension).float().sum(3).to(torch.bfloat16)
    sums_v = value.reshape(1, 1, pages, 16, dimension).float().sum(3).to(torch.bfloat16)
    state_k = key.float().sum(2, keepdim=True).to(torch.bfloat16)
    state_v = value.float().sum(2, keepdim=True).to(torch.bfloat16)
    counts = torch.full((1, 1, pages), 16, device=device, dtype=torch.int32)
    lengths = torch.full((1, 1, 1), pages * 16, device=device, dtype=torch.int32)
    table = torch.tensor(list(range(29, 13, -1)) + list(range(14)) + list(range(30, 40)),
                         device=device, dtype=torch.int32).reshape(1, 1, 1, pages)
    scores = torch.zeros(1, pages, device=device)
    if case == "all_ties":
        selected = [29, 28]
    elif case == "later_winner":
        scores[0, 39], scores[0, 25] = 10, 5
        selected = [39, 25]
    else:
        scores[0, [3, 11, 38]] = 10
        selected = [11, 3] if block == 16 else [3, 11]
    indices = torch.arange(pages * 16, device=device, dtype=torch.int32).reshape(1, 1, pages, 16)
    overflow = torch.zeros(1, 1, 1, device=device, dtype=torch.int32)
    output, lse = query_major_indexed_residual_page_attention(
        query, state_k, state_v, lengths.float(), key, value,
        indices, sums_k, sums_v, counts, table, overflow, overflow,
        torch.zeros((), device=device, dtype=torch.int32), lengths,
        torch.zeros(1, 1, 1, 1, device=device, dtype=torch.int64),
        kv_group_size=1, scale=dimension**-0.5, hash_probes=0,
        page_block_n=block, materialized_page_scores=scores.reshape(1, 1, 1, pages),
        pages_per_slot=2, route_parallel=True, num_warps=1,
    )
    opened = torch.cat([torch.arange(p * 16, (p + 1) * 16, device=device) for p in selected])
    remaining = (pages - 2) * 16
    residual_k = (state_k[0, 0, 0].float() - sums_k[0, 0, selected].float().sum(0)) / remaining
    residual_v = (state_v[0, 0, 0].float() - sums_v[0, 0, selected].float().sum(0)) / remaining
    keys = torch.cat((key[0, 0, opened].float(), residual_k[None]))
    values = torch.cat((value[0, 0, opened].float(), residual_v[None]))
    logits = keys @ query[0, 0, 0].float() * dimension**-0.5
    logits[-1] += torch.tensor(float(remaining), device=device).log()
    torch.testing.assert_close(output[0, 0, 0].float(), logits.softmax(0) @ values, atol=0.015, rtol=0.015)
    torch.testing.assert_close(lse[0, 0, 0], logits.logsumexp(0), atol=0.002, rtol=0.002)

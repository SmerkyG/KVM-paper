"""Independent checks of CUDA routing, branch mass, and causal alignment."""

import os
from types import SimpleNamespace

import pytest
import torch

from lod_attention._config import ModelFamily, LODMode, model_family
from lod_attention._profile import configure_engine


def test_llama31_native_geometry_and_cuda_profile():
    config = SimpleNamespace(
        model_type="llama",
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    assert model_family(config) is ModelFamily.LLAMA31
    for mode in LODMode:
        engine = SimpleNamespace(config=config, head_dim=128)
        configure_engine(
            engine,
            family=ModelFamily.LLAMA31,
            mode=mode,
            request_capacity=131072,
            has_query_norm=False,
            has_key_norm=False,
        )
        assert engine.prefill_local_attention_backend == "flash_attn"
        assert (
            engine.prefill_cuda_route_coarse and not engine.prefill_aiter_route_coarse
        )
        assert engine.two_level_topk == engine.prefill_two_level_topk == 8
        assert engine.state_clustering_normalization == "cosine"
        assert engine.routing_normalization == "query"
        assert engine.decode_route_parallel_reduce
        assert engine.decode_route_group_size == 64
        assert engine.decode_route_num_warps == 2
        assert engine.decode_route_reduce_num_warps == 4
        assert (engine.leaf_block_m, engine.leaf_block_n) == (
            64,
            16 if mode.kv_bits else 32,
        )


gpu = pytest.mark.skipif(
    os.getenv("LOD_RUN_GPU_TESTS") != "1"
    or not torch.cuda.is_available()
    or torch.version.hip,
    reason="NVIDIA GPU tests require LOD_RUN_GPU_TESTS=1",
)


@gpu
@pytest.mark.parametrize(
    "batch,queries,states", [(1, 17, 73), (1, 129, 257), (2, 35, 1031)]
)
@pytest.mark.parametrize("normalize", [False, True])
def test_fused_route_coarse_matches_dense_mass_and_top8(
    batch, queries, states, normalize
):
    from lod_attention.kernels.cuda_prefill_attention import (
        cuda_prefill_route_coarse_attention,
    )

    torch.manual_seed(567)
    heads, kv_heads, dimension = 8, 2, 128
    q = torch.randn(
        batch, heads, queries, dimension, device="cuda", dtype=torch.bfloat16
    )
    state_k = torch.randn(
        batch, kv_heads, states + 19, dimension, device="cuda", dtype=torch.bfloat16
    )
    state_v = torch.randn_like(state_k)
    counts = torch.randint(
        1, 15, (batch, kv_heads, states + 19, 1), device="cuda"
    ).float()
    buffers = {}
    slots, coarse, head_counts, offsets = cuda_prefill_route_coarse_attention(
        q,
        state_k,
        state_v,
        counts,
        state_len=states,
        kv_group_size=4,
        scale=dimension**-0.5,
        normalize_route_query=normalize,
        buffers=buffers,
    )
    mean_k = (
        (state_k[..., :states, :].float() / counts[..., :states, :])
        .to(torch.bfloat16)
        .float()
        .repeat_interleave(4, dim=1)
    )
    mean_v = (
        (state_v[..., :states, :].float() / counts[..., :states, :])
        .to(torch.bfloat16)
        .float()
        .repeat_interleave(4, dim=1)
    )
    raw = torch.matmul(q.float(), mean_k.transpose(-1, -2)) * dimension**-0.5
    bias = counts[..., :states, 0].repeat_interleave(4, dim=1).log().unsqueeze(2)
    score = raw + bias
    expected = torch.matmul(score.softmax(-1), mean_v)
    torch.testing.assert_close(
        coarse.output_0.permute(0, 2, 1, 3).float(), expected, atol=0.005, rtol=0.02
    )
    torch.testing.assert_close(
        coarse.lse_0, score.logsumexp(-1), atol=2.0e-4, rtol=2.0e-4
    )
    rms = q.float().square().mean(-1, keepdim=True).clamp_min(1.0e-12).sqrt()
    ranking = (raw / rms if normalize else raw) + bias
    # Ignore the route output order: expert packing places the boundary last
    # and the other seven winners in ascending slot order.
    assert torch.equal(
        slots.sort(-1).values, ranking.topk(8, -1).indices.sort(-1).values
    )
    actual_counts = head_counts.reshape(batch, heads, states)
    for b in range(batch):
        for h in range(heads):
            expected_counts = torch.bincount(slots[b, h].reshape(-1), minlength=states)
            assert torch.equal(actual_counts[b, h].long(), expected_counts)
            for slot in slots[b, h].unique():
                assigned = offsets[b, h][slots[b, h] == slot].sort().values
                assert torch.equal(
                    assigned,
                    torch.arange(len(assigned), device="cuda", dtype=torch.int32),
                )
    assert not any(
        t.ndim == 4 and tuple(t.shape) == (batch, heads, queries, states)
        for t in buffers.values()
    )


@gpu
@pytest.mark.parametrize("batch,queries,keys", [(1, 7, 37), (2, 19, 19), (2, 35, 73)])
@pytest.mark.parametrize("provided_output", [False, True])
def test_flash3_local_attention_causal_suffix_and_output_buffer(
    batch, queries, keys, provided_output
):
    from lod_attention.kernels.cuda_prefill_attention import cuda_local_attention

    torch.manual_seed(12)
    q = torch.randn(batch, 8, queries, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 2, keys, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    storage = torch.empty(batch, queries, 8, 128, device="cuda", dtype=torch.bfloat16)
    target = storage.permute(0, 2, 1, 3) if provided_output else None
    output, lse = cuda_local_attention(q, k, v, scale=128**-0.5, output_buffer=target)
    logits = (
        torch.matmul(q.float(), k.float().repeat_interleave(4, 1).transpose(-1, -2))
        * 128**-0.5
    )
    mask = (
        torch.arange(keys, device="cuda")[None, :]
        <= (torch.arange(queries, device="cuda") + keys - queries)[:, None]
    )
    logits = logits.masked_fill(~mask, -float("inf"))
    expected = torch.matmul(logits.softmax(-1), v.float().repeat_interleave(4, 1))
    torch.testing.assert_close(output.float(), expected, atol=0.01, rtol=0.02)
    torch.testing.assert_close(lse, logits.logsumexp(-1), atol=2.0e-4, rtol=2.0e-4)
    if target is not None:
        assert output.data_ptr() == target.data_ptr()
    no_lse, _ = cuda_local_attention(q, k, v, scale=128**-0.5, return_lse=False)
    torch.testing.assert_close(no_lse, output)


@gpu
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_decode_routes_use_normalized_query_but_attention_uses_native_logits(
    batch, normalized, parallel
):
    import triton
    from lod_attention.kernels.paged_routing import (
        _decode_route_coarse_gqa_groups_kernel,
        _reduce_decode_route_coarse_kernel,
        _reduce_decode_route_coarse_vector_topk_kernel,
    )
    from lod_attention.kernels.cuda_prefill_attention import (
        restore_decode_attention_logits,
    )

    torch.manual_seed(814)
    heads, kv_heads, d, capacity, group = 8, 2, 128, 257, 64
    groups = triton.cdiv(capacity, group)
    q = torch.randn(batch, heads, 1, d, device="cuda", dtype=torch.bfloat16)
    q = (
        (q * torch.linspace(0.2, 4, heads, device="cuda").view(1, heads, 1, 1))
        .to(torch.bfloat16)
        .contiguous()
    )
    k = torch.randn(batch, kv_heads, capacity, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    counts = torch.randint(1, 25, (batch, kv_heads, capacity, 1), device="cuda").float()
    lens = torch.linspace(73, capacity, batch, device="cuda").int()
    indices = torch.arange(batch - 1, -1, -1, device="cuda").long()
    scores = torch.empty(batch, heads, groups, 8, device="cuda")
    candidates = torch.empty_like(scores, dtype=torch.int64)
    partial = torch.empty(batch, heads, groups, d, device="cuda")
    partial_lse = torch.empty(batch, heads, groups, device="cuda")
    slots = torch.empty(batch, heads, 1, 8, device="cuda", dtype=torch.int64)
    selected_scores = torch.empty_like(slots, dtype=torch.float32)
    coarse = torch.empty(batch, heads, d, device="cuda")
    coarse_lse = torch.empty(batch, heads, device="cuda")
    slot_lengths = counts.int().squeeze(-1)
    scratch = torch.zeros(batch * kv_heads, device="cuda", dtype=torch.int32)
    _decode_route_coarse_gqa_groups_kernel[(batch * kv_heads, groups)](
        q,
        k,
        v,
        counts,
        indices,
        scores,
        candidates,
        partial,
        partial_lse,
        *k.stride()[:3],
        *v.stride()[:3],
        *counts.stride()[:3],
        capacity,
        lens,
        scratch,
        scratch,
        scratch,
        counts,
        lens,
        k,
        v,
        q,
        q,
        lens,
        *counts.stride()[:3],
        QUERY_HEADS=heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=4,
        HEAD_DIM=d,
        SCALE=d**-0.5,
        GROUP_N=group,
        MAX_GROUPS=groups,
        PROTECTED_LEN=0,
        MAX_LEAF_TOKENS=0,
        USE_DOT=True,
        USE_STATE_LENS=True,
        NORMALIZE_ROUTE_QUERY=normalized,
        num_warps=4,
    )
    reducer = (
        _reduce_decode_route_coarse_vector_topk_kernel
        if parallel
        else _reduce_decode_route_coarse_kernel
    )
    reduction_geometry = (
        dict(
            MAX_SEGMENTS=groups,
            CANDIDATE_BLOCK=triton.next_power_of_2(groups * 8),
            SEGMENT_BLOCK=triton.next_power_of_2(groups),
        )
        if parallel
        else dict(MAX_GROUPS=groups, CANDIDATE_TILE=triton.next_power_of_2(groups * 8))
    )
    reducer[(batch * heads,)](
        scores,
        candidates,
        partial,
        partial_lse,
        slots,
        selected_scores,
        coarse,
        coarse_lse,
        slot_lengths,
        indices,
        groups,
        *([groups] if parallel else []),
        QUERY_HEADS=heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=4,
        HEAD_DIM=d,
        STATE_CAPACITY=capacity,
        ROUTE_COUNT=8,
        OPEN_COUNT=8,
        **reduction_geometry,
        APPLY_MASS_CUTOFF=False,
        LOG_MASS_FRACTION=0.0,
        num_warps=4,
    )
    if normalized:
        restore_decode_attention_logits(
            q,
            k,
            counts,
            indices,
            slots,
            selected_scores,
            kv_group_size=4,
            scale=d**-0.5,
        )
    mk = (
        (k.float() / counts).to(torch.bfloat16)[indices].float().repeat_interleave(4, 1)
    )
    mv = (
        (v.float() / counts).to(torch.bfloat16)[indices].float().repeat_interleave(4, 1)
    )
    raw = (q.float() @ mk.transpose(-1, -2)) * d**-0.5
    bias = counts[indices].squeeze(-1).repeat_interleave(4, 1).log().unsqueeze(2)
    active = (
        torch.arange(capacity, device="cuda")[None, None, None, :]
        < lens[indices, None, None, None]
    )
    native = (raw + bias).masked_fill(~active, -float("inf"))
    rms = q.float().square().mean(-1, keepdim=True).sqrt()
    ranking = ((raw / rms if normalized else raw) + bias).masked_fill(
        ~active, -float("inf")
    )
    assert torch.equal(
        slots.sort(-1).values, ranking.topk(8, -1).indices.sort(-1).values
    )
    if normalized:
        assert not torch.equal(
            slots.sort(-1).values, native.topk(8, -1).indices.sort(-1).values
        )
    torch.testing.assert_close(
        selected_scores, native.gather(-1, slots), atol=2.0e-4, rtol=2.0e-4
    )
    torch.testing.assert_close(
        coarse_lse, native.logsumexp(-1).squeeze(2), atol=2.0e-4, rtol=2.0e-4
    )
    expected = (native.softmax(-1) @ mv).squeeze(2)
    torch.testing.assert_close(coarse, expected, atol=0.005, rtol=0.02)

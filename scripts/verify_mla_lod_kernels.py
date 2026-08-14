#!/usr/bin/env python3
"""Exercise native DeepSeek-style MLA geometry through the LOD kernels."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from model.kernels.lod_kernels import (
    prepare_state_clustering_keys,
    route_logits_coarse_attention,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_engines import (
    KernelRecursivePagedLODAttention,
    KernelTwoLevelLODAttention,
)


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MLA kernel verification requires a GPU")
    torch.manual_seed(93)
    device = torch.device("cuda")
    batch = 8
    query_heads = 16
    key_value_heads = 1
    qk_dim = 576
    value_dim = 512
    scale = 192**-0.5

    config = LODConfig(
        chunk_size=64,
        local_window=128,
        state_growth_factor=8.0,
        state_min_size=64,
        protected_prefix=1,
        max_routes=8,
    )
    engine = KernelTwoLevelLODAttention(
        config,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        scale=scale,
        default_open_count=8,
    )

    # Raw MLA state keys must reproduce the model's per-token latent RMSNorm
    # for exact leaves, while coarse entries normalize only after averaging.
    norm_weight = torch.linspace(
        0.5, 1.5, value_dim, device=device, dtype=torch.float32
    )
    raw_key = torch.randn(
        batch,
        key_value_heads,
        7,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    raw_counts = torch.randint(
        1,
        9,
        (batch, key_value_heads, 7, 1),
        device=device,
    ).float()
    epsilon = 1e-6
    for mode in ("latent", "whole", "raw"):
        raw_engine = KernelTwoLevelLODAttention(
            replace(config, mla_state_key_normalization=mode),
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            scale=scale,
            default_open_count=8,
        )
        raw_engine.mla_key_norm_weight = norm_weight
        raw_engine.mla_key_norm_epsilon = epsilon
        exact_key = raw_engine._mla_normalize_key(
            raw_key, state_centroid=False
        )
        latent = raw_key[..., :value_dim].float()
        normalized_latent = (
            latent
            * torch.rsqrt(
                latent.square().mean(dim=-1, keepdim=True) + epsilon
            )
        )
        normalized_latent = normalized_latent.to(raw_key.dtype)
        normalized_latent = normalized_latent * norm_weight.to(raw_key.dtype)
        expected_exact = torch.cat(
            (normalized_latent, raw_key[..., value_dim:]), dim=-1
        )
        torch.testing.assert_close(
            exact_key, expected_exact
        )
        normalized_sum = raw_engine._mla_state_key_sum_for_attention(
            raw_key * raw_counts.to(raw_key.dtype),
            raw_counts,
            state_len=7,
        )
        normalized_mean = normalized_sum.float() / raw_counts
        if mode == "latent":
            expected_mean = expected_exact
        elif mode == "whole":
            expected_mean = raw_key.float() * torch.rsqrt(
                raw_key.float().square().mean(dim=-1, keepdim=True)
                + epsilon
            )
            expected_mean = expected_mean.to(raw_key.dtype)
            expected_mean[..., :value_dim] *= norm_weight.to(raw_key.dtype)
        else:
            expected_mean = raw_key.float()
        torch.testing.assert_close(
            normalized_mean,
            expected_mean.float(),
            atol=1.6e-2,
            rtol=1.6e-2,
        )
    print("raw MLA latent/full-key JIT normalization formulas passed")

    # MLA's concatenated latent/rotary key width is not a power of two.  Keep
    # the quality-routing preparation kernel covered at that native width.
    state_len = 32
    state_k = torch.randn(
        batch,
        key_value_heads,
        state_len,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    state_counts = torch.randint(
        1,
        17,
        (batch, key_value_heads, state_len, 1),
        device=device,
    ).float()
    prepared_route, prepared_append, _ = prepare_state_clustering_keys(
        state_k,
        state_counts,
        {},
        state_len=state_len,
        geometry="spherical",
    )
    mean_state_k = state_k / state_counts.to(state_k.dtype)
    expected_prepared = (
        mean_state_k.float()
        / mean_state_k.float().square().mean(dim=-1, keepdim=True).sqrt()
    ).to(state_k.dtype)
    torch.testing.assert_close(
        prepared_route,
        expected_prepared,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(prepared_append, prepared_route)

    local_len = 192
    local_offset = 64
    local_q = torch.randn(
        batch,
        query_heads,
        local_len,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    local_k = torch.randn(
        batch,
        key_value_heads,
        local_len,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    local_v = torch.randn(
        batch,
        key_value_heads,
        local_len,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    actual_local, actual_lse = engine._prefill_local_attention(
        local_q,
        local_k,
        local_v,
        query_offset=local_offset,
    )
    target_q = local_q[..., local_offset:, :]
    query_positions = local_offset + torch.arange(
        local_len - local_offset, device=device
    )
    key_positions = torch.arange(local_len, device=device)
    local_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    expected_local = F.scaled_dot_product_attention(
        target_q,
        local_k.repeat_interleave(query_heads, dim=1),
        local_v.repeat_interleave(query_heads, dim=1),
        attn_mask=local_mask,
        scale=scale,
    )
    torch.testing.assert_close(
        actual_local, expected_local, atol=3e-2, rtol=2e-2
    )
    if tuple(actual_lse.shape) != (*target_q.shape[:-1],):
        raise AssertionError("unexpected MLA local-attention LSE shape")

    coarse_query_len = 8
    coarse_state_len = 16
    coarse_local_len = 16
    coarse_q = local_q[..., :coarse_query_len, :].contiguous()
    coarse_state_v = torch.randn(
        batch,
        key_value_heads,
        coarse_state_len,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    coarse_counts = torch.randint(
        1,
        17,
        (batch, key_value_heads, coarse_state_len, 1),
        device=device,
        dtype=torch.int32,
    ).float()
    coarse_logits = torch.randn(
        batch,
        query_heads,
        coarse_query_len,
        coarse_state_len,
        device=device,
        dtype=torch.bfloat16,
    )
    coarse_routes = coarse_logits.float().topk(8, dim=-1).indices
    coarse_local_k = local_k[..., :coarse_local_len, :].contiguous()
    coarse_local_v = local_v[..., :coarse_local_len, :].contiguous()
    actual_coarse, actual_coarse_lse = engine._gemm_coarse_attention(
        coarse_q,
        coarse_logits,
        coarse_state_v,
        coarse_counts,
        coarse_local_k,
        coarse_local_v,
        coarse_routes,
        state_len=coarse_state_len,
    )
    expected_coarse, expected_coarse_lse = route_logits_coarse_attention(
        coarse_q,
        coarse_logits,
        coarse_state_v,
        coarse_counts,
        coarse_local_k,
        coarse_local_v,
        coarse_routes,
        state_len=coarse_state_len,
        kv_group_size=query_heads,
        scale=scale,
    )
    torch.testing.assert_close(
        actual_coarse, expected_coarse, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        actual_coarse_lse, expected_coarse_lse, atol=2e-3, rtol=2e-3
    )

    exact_len = 64
    exact_q = torch.randn(
        batch,
        query_heads,
        exact_len,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    exact_k = torch.randn(
        batch,
        key_value_heads,
        exact_len,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    exact_v = torch.randn(
        batch,
        key_value_heads,
        exact_len,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    actual_exact, _ = engine(exact_q, exact_k, exact_v)
    expected_exact = F.scaled_dot_product_attention(
        exact_q,
        exact_k.repeat_interleave(query_heads, dim=1),
        exact_v.repeat_interleave(query_heads, dim=1),
        is_causal=True,
        scale=scale,
    )
    torch.testing.assert_close(actual_exact, expected_exact, atol=2e-2, rtol=2e-2)

    prefill_len = 384
    q = torch.randn(
        batch,
        query_heads,
        prefill_len,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        batch,
        key_value_heads,
        prefill_len,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        batch,
        key_value_heads,
        prefill_len,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    output, cache = engine(q, k, v, use_cache=True)
    if tuple(output.shape) != (batch, query_heads, prefill_len, value_dim):
        raise AssertionError(f"unexpected MLA prefill shape {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all()):
        raise AssertionError("MLA prefill produced non-finite output")
    if cache is None or cache.total_length != prefill_len:
        raise AssertionError("MLA prefill cache length is wrong")

    next_q = torch.randn_like(q[..., :1, :])
    next_k = torch.randn_like(k[..., :1, :])
    next_v = torch.randn_like(v[..., :1, :])
    decoded, cache = engine(
        next_q, next_k, next_v, cache=cache, use_cache=True
    )
    if tuple(decoded.shape) != (batch, query_heads, 1, value_dim):
        raise AssertionError(f"unexpected MLA decode shape {tuple(decoded.shape)}")
    if not bool(torch.isfinite(decoded).all()):
        raise AssertionError("MLA decode produced non-finite output")
    if cache is None or cache.total_length != prefill_len + 1:
        raise AssertionError("MLA decode cache length is wrong")

    recursive = KernelRecursivePagedLODAttention(
        PagedLODConfig(
            chunk_size=64,
            local_window=128,
            state_growth_factor=8.0,
            state_min_size=64,
            protected_prefix=1,
            max_routes=8,
            page_size=16,
            mla_state_key_normalization="latent",
            mla_recursive_page_key_normalization=True,
        ),
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        scale=scale,
        default_open_count=8,
    )
    recursive.mla_key_norm_weight = norm_weight
    recursive.mla_key_norm_epsilon = epsilon
    recursive_len = 1536
    recursive_q = torch.randn(
        batch,
        query_heads,
        recursive_len,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    recursive_k = torch.randn(
        batch,
        key_value_heads,
        recursive_len,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    recursive_v = torch.randn(
        batch,
        key_value_heads,
        recursive_len,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    recursive_output, recursive_cache = recursive(
        recursive_q, recursive_k, recursive_v, use_cache=True
    )
    if tuple(recursive_output.shape) != (
        batch,
        query_heads,
        recursive_len,
        value_dim,
    ):
        raise AssertionError("recursive MLA prefill output shape is wrong")
    if not bool(torch.isfinite(recursive_output).all()):
        raise AssertionError("recursive MLA prefill produced non-finite output")
    if recursive_cache is None or recursive_cache.total_length != recursive_len:
        raise AssertionError("recursive MLA cache length is wrong")
    recursive_decode, recursive_cache = recursive(
        recursive_q[..., :1, :],
        recursive_k[..., :1, :],
        recursive_v[..., :1, :],
        cache=recursive_cache,
        use_cache=True,
    )
    if tuple(recursive_decode.shape) != (batch, query_heads, 1, value_dim):
        raise AssertionError("recursive MLA decode output shape is wrong")
    if not bool(torch.isfinite(recursive_decode).all()):
        raise AssertionError("recursive MLA decode produced non-finite output")
    print("native MLA QK=576/V=512 batch-8 kernel smokes passed")


if __name__ == "__main__":
    main()

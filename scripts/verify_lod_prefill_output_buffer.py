#!/usr/bin/env python3
"""Compare recursive LOD prefill with allocated and token-major output."""

from __future__ import annotations

import torch

from model.kernels.lod_kernels import merge_attention_branches_with_sink
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_engines import KernelRecursivePagedLODAttention


def make_engine() -> KernelRecursivePagedLODAttention:
    engine = KernelRecursivePagedLODAttention(
        PagedLODConfig(
            chunk_size=256,
            local_window=512,
            state_growth_factor=16.0,
            state_min_size=256,
            protected_prefix=1,
            max_routes=8,
            page_size=16,
            kv_bits=0,
            state_clustering_normalization="none",
            state_clustering_centroid_rescale="coherence",
            state_clustering_centroid_rescale_scope="assignment",
            routing_normalization="none",
        ),
        query_heads=8,
        key_value_heads=2,
        scale=256**-0.5,
        default_open_count=8,
    ).cuda()
    engine.separate_sink_cache = True
    return engine


def verify_sink_merge_strides() -> dict[str, float | int]:
    batch, query_heads, kv_heads, query_len, head_dim = 2, 8, 2, 37, 64
    group_size = query_heads // kv_heads
    scale = head_dim**-0.5
    q = torch.randn(
        batch,
        query_heads,
        query_len,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sink_k = torch.randn(
        batch, kv_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16
    )
    sink_v = torch.randn_like(sink_k)
    branch_out = torch.randn_like(q)
    branch_lse = torch.randn(
        batch, query_heads, query_len, device="cuda", dtype=torch.float32
    )
    storage = torch.full(
        (batch, query_len + 10, query_heads, head_dim),
        123.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = storage[:, 3 : 3 + query_len].permute(0, 2, 1, 3)
    actual = merge_attention_branches_with_sink(
        q,
        sink_k,
        sink_v,
        branch_out,
        branch_lse,
        kv_group_size=group_size,
        scale=scale,
        output_buffer=output,
    )
    q_grouped = q.float().reshape(
        batch, kv_heads, group_size, query_len, head_dim
    )
    sink_scores = (
        q_grouped * sink_k.float().unsqueeze(2)
    ).sum(dim=-1) * scale
    sink_lse = sink_scores.reshape(batch, query_heads, query_len)
    maximum = torch.maximum(branch_lse, sink_lse)
    branch_weight = torch.exp(branch_lse - maximum)
    sink_weight = torch.exp(sink_lse - maximum)
    sink_output = (
        sink_v.float()
        .unsqueeze(2)
        .expand(-1, -1, group_size, query_len, -1)
        .reshape(batch, query_heads, query_len, head_dim)
    )
    expected = (
        branch_weight.unsqueeze(-1) * branch_out.float()
        + sink_weight.unsqueeze(-1) * sink_output
    ) / (branch_weight + sink_weight).unsqueeze(-1)
    return {
        "sink_merge_oracle_max_abs": float((actual.float() - expected).abs().max()),
        "sink_merge_left_sentinel_changed": int(storage[:, :3].ne(123.0).sum()),
        "sink_merge_right_sentinel_changed": int(
            storage[:, 3 + query_len :].ne(123.0).sum()
        ),
    }


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(0)
    sink_metrics = verify_sink_merge_strides()
    batch, length, query_heads, kv_heads, dim = 2, 8192, 8, 2, 256
    token_q = torch.randn(
        batch,
        length,
        query_heads,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    token_k = torch.randn(
        batch,
        length,
        kv_heads,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    token_v = torch.randn_like(token_k)
    q = token_q.permute(0, 2, 1, 3)
    k_view = token_k.permute(0, 2, 1, 3)
    v_view = token_v.permute(0, 2, 1, 3)
    reference, _ = make_engine()(
        q, k_view.contiguous(), v_view.contiguous(), use_cache=True
    )
    strided_input, _ = make_engine()(q, k_view, v_view, use_cache=True)
    token_major = torch.empty(
        batch, length, query_heads, dim,
        device="cuda", dtype=torch.bfloat16,
    )
    output_view = token_major.permute(0, 2, 1, 3)
    buffered_contiguous_input, _ = make_engine()(
        q,
        k_view.contiguous(),
        v_view.contiguous(),
        use_cache=True,
        output_buffer=output_view,
    )
    contiguous_output = torch.empty(
        batch, query_heads, length, dim,
        device="cuda", dtype=torch.bfloat16,
    )
    buffered_contiguous_output, _ = make_engine()(
        q, k_view, v_view, use_cache=True, output_buffer=contiguous_output
    )
    guard_elements = 1 << 20
    payload_elements = batch * query_heads * length * dim
    guarded_storage = torch.full(
        (payload_elements + 2 * guard_elements,),
        123.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    guarded_output = guarded_storage[
        guard_elements : guard_elements + payload_elements
    ].view(batch, query_heads, length, dim)
    buffered_guarded_output, _ = make_engine()(
        q, k_view, v_view, use_cache=True, output_buffer=guarded_output
    )
    token_major_strided = torch.empty_like(token_major)
    strided_output_view = token_major_strided.permute(0, 2, 1, 3)
    actual, _ = make_engine()(
        q, k_view, v_view, use_cache=True, output_buffer=strided_output_view
    )
    torch.cuda.synchronize()
    metrics = {
        **sink_metrics,
        "returned_alias": actual.data_ptr() == strided_output_view.data_ptr(),
        "output_buffer_max_abs": float(
            (buffered_contiguous_input - reference).abs().max()
        ),
        "output_buffer_mean_abs": float(
            (buffered_contiguous_input - reference).abs().float().mean()
        ),
        "contiguous_output_max_abs": float(
            (buffered_contiguous_output - reference).abs().max()
        ),
        "contiguous_output_mean_abs": float(
            (buffered_contiguous_output - reference).abs().float().mean()
        ),
        "guarded_output_max_abs": float(
            (buffered_guarded_output - reference).abs().max()
        ),
        "prefix_guard_changed": int(
            guarded_storage[:guard_elements].ne(123.0).sum()
        ),
        "suffix_guard_changed": int(
            guarded_storage[-guard_elements:].ne(123.0).sum()
        ),
        "strided_input_max_abs": float((strided_input - reference).abs().max()),
        "strided_input_mean_abs": float(
            (strided_input - reference).abs().float().mean()
        ),
        "output_max_abs": float((actual - reference).abs().max()),
        "output_mean_abs": float((actual - reference).abs().float().mean()),
        "token_view_max_abs": float(
            (strided_output_view - reference).abs().max()
        ),
    }
    for name in (
        "output_buffer_max_abs",
        "contiguous_output_max_abs",
        "guarded_output_max_abs",
        "strided_input_max_abs",
        "output_max_abs",
        "token_view_max_abs",
    ):
        if metrics[name] > 1.0e-3:
            raise AssertionError(
                f"{name} exceeded BF16 parity tolerance: {metrics[name]}"
            )
    if metrics["prefix_guard_changed"] or metrics["suffix_guard_changed"]:
        raise AssertionError("prefill output wrote outside its guarded allocation")
    if metrics["sink_merge_oracle_max_abs"] > 1.0e-2:
        raise AssertionError("separate-sink merge differs from its FP32 oracle")
    if (
        metrics["sink_merge_left_sentinel_changed"]
        or metrics["sink_merge_right_sentinel_changed"]
    ):
        raise AssertionError("separate-sink merge wrote outside its destination view")
    if not metrics["returned_alias"]:
        raise AssertionError("prefill did not return the supplied output view")
    print(metrics)


if __name__ == "__main__":
    main()

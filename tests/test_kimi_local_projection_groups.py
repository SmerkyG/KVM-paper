"""Bound exact-field projection memory without removing heads or KV tokens."""

import pytest
import torch


@pytest.mark.parametrize("group", ["-1", "5"])
def test_reject_incompatible_local_head_group(monkeypatch, group):
    from lod_attention.kernels.aiter_mla_prefill_attention import aiter_kimi_local_prefill_attention

    monkeypatch.setenv("LOD_KIMI_LOCAL_PREFILL_HEAD_GROUP", group)
    q = torch.zeros(1, 96, 2, 576)
    with pytest.raises(ValueError, match="local prefill head group"):
        aiter_kimi_local_prefill_attention(q, torch.zeros(1, 1, 4, 576),
            query_offset=2, scale=1, expanded_q=torch.zeros(1, 96, 2, 192),
            w_uk_t=torch.zeros(96, 128, 512), w_uv=torch.zeros(96, 512, 128))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU MLA projection/AITER")
@torch.inference_mode()
def test_grouped_local_projection_matches_full_head_attention(monkeypatch):
    from lod_attention.kernels.aiter_mla_prefill_attention import aiter_kimi_local_prefill_attention

    torch.manual_seed(107)
    heads, keys, queries = 96, 32768, 33
    latent = torch.randn(1, 1, keys, 576, device="cuda", dtype=torch.bfloat16) * .1
    q = torch.zeros(1, heads, queries, 576, device="cuda", dtype=torch.bfloat16)
    expanded = torch.randn(1, heads, queries, 192, device="cuda", dtype=torch.bfloat16) * .1
    uk = torch.randn(heads, 128, 512, device="cuda", dtype=torch.bfloat16) * .02
    uv = torch.randn(heads, 512, 128, device="cuda", dtype=torch.bfloat16) * .02
    common = dict(query_offset=keys - queries, scale=192**-.5,
                  expanded_q=expanded, w_uk_t=uk, w_uv=uv)
    buffers = {}
    monkeypatch.delenv("LOD_KIMI_LOCAL_PREFILL_HEAD_GROUP", raising=False)
    reference = tuple(t.clone() for t in aiter_kimi_local_prefill_attention(q, latent, buffers=buffers, **common))
    large_bytes = sum(buffers[name].untyped_storage().nbytes() for name in (
        "kimi_local_expanded_k", "kimi_local_expanded_k_nope", "kimi_local_expanded_v"))
    grouped = {}
    monkeypatch.setenv("LOD_KIMI_LOCAL_PREFILL_HEAD_GROUP", "16")
    actual = aiter_kimi_local_prefill_attention(q, latent, buffers=grouped, **common)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], reference[0], atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(actual[1], reference[1], atol=2e-4, rtol=2e-4)
    small_bytes = sum(grouped[name].untyped_storage().nbytes() for name in (
        "kimi_local_expanded_k", "kimi_local_expanded_k_nope", "kimi_local_expanded_v"))
    assert large_bytes == small_bytes * 6
    # A second call with different Q proves all groups refresh and that the
    # assembled result does not alias the reused group's temporary output.
    common["expanded_q"] = expanded.flip(1).contiguous()
    second = aiter_kimi_local_prefill_attention(q, latent, buffers=grouped, **common)
    torch.cuda.synchronize()
    assert second[0].data_ptr() == actual[0].data_ptr()
    # Q changes but per-head projection weights do not: compare to a new
    # all-head control, not a permutation of the old head outputs.
    monkeypatch.delenv("LOD_KIMI_LOCAL_PREFILL_HEAD_GROUP")
    second_reference = aiter_kimi_local_prefill_attention(q, latent, **common)
    torch.testing.assert_close(second[0], second_reference[0], atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(second[1], second_reference[1], atol=2e-4, rtol=2e-4)
    print({"local_projection_bytes_all_heads": large_bytes,
           "local_projection_bytes_16_heads": small_bytes,
           "saved_bytes": large_bytes - small_bytes,
           "scope": "exact local K/V workspace only; same 96 heads and every key"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU fused MLA route/coarse")
@torch.inference_mode()
def test_grouped_coarse_projection_preserves_routes_and_replacement_inputs(monkeypatch):
    from lod_attention.kernels.aiter_mla_prefill_attention import aiter_kimi_expanded_prefill_route_coarse_attention

    torch.manual_seed(108)
    heads, states, queries = 96, 2048, 33
    q = torch.randn(1, heads, queries, 192, device="cuda", dtype=torch.bfloat16) * .1
    means = torch.randn(1, 1, states, 576, device="cuda", dtype=torch.bfloat16) * .1
    counts = torch.randint(1, 2000, (1, 1, states, 1), device="cuda").float()
    sums = means.float() * counts
    uk = torch.randn(heads, 128, 512, device="cuda", dtype=torch.bfloat16) * .02
    uv = torch.randn(heads, 512, 128, device="cuda", dtype=torch.bfloat16) * .02
    common = dict(state_len=states, scale=192**-.5, normalize_route_query=False,
                  slot_lengths=counts.squeeze(-1).int(), max_open_leaf_tokens=1024)
    monkeypatch.setenv("LOD_KIMI_SUBTILE64", "score")
    monkeypatch.setenv("LOD_KIMI_TILE_REFINE", "1")

    def run(buffers):
        slots, coarse, head_counts, offsets = aiter_kimi_expanded_prefill_route_coarse_attention(
            q, sums, sums[..., :512], counts, uk, uv, buffers=buffers, **common)
        assert head_counts is None and offsets is None
        if coarse.ready_stream is not None:
            torch.cuda.current_stream().wait_stream(coarse.ready_stream)
        return tuple(t.clone() for t in (slots, coarse.output_0, coarse.lse_0,
            coarse.mean_k, coarse.mean_v, coarse.counts, coarse.selected_route_scores))

    monkeypatch.delenv("LOD_KIMI_COARSE_PREFILL_HEAD_GROUP", raising=False)
    original = {}
    reference = run(original)
    monkeypatch.setenv("LOD_KIMI_COARSE_PREFILL_HEAD_GROUP", "16")
    grouped = {}
    actual = run(grouped)
    torch.cuda.synchronize()
    assert actual[0].dtype == reference[0].dtype
    torch.testing.assert_close(actual[0].sort(-1).values, reference[0].sort(-1).values, atol=0, rtol=0)
    for index, (got, expected) in enumerate(zip(actual[1:], reference[1:]), start=1):
        # Changing GEMM's output width can change the last BF16 bit of a
        # projected centroid value. Keep the attention/LSE checks tighter;
        # allow two BF16 relative ulps only for the raw projected means.
        relative = 2 * torch.finfo(torch.bfloat16).eps if index == 4 else 2e-3
        torch.testing.assert_close(got, expected, atol=2e-4, rtol=relative)
    fields = ("kimi_expanded_coarse_k", "kimi_expanded_coarse_k_nope", "kimi_expanded_coarse_v")
    original_bytes = sum(original[name].untyped_storage().nbytes() for name in fields)
    grouped_bytes = sum(grouped[name].untyped_storage().nbytes() for name in fields)
    assembled_values = grouped["kimi_grouped_coarse_values"].untyped_storage().nbytes()
    assert grouped_bytes * 6 == original_bytes
    assert grouped_bytes + assembled_values < original_bytes
    # Shared projection buffers must refresh after query and weights change.
    q.mul_(-.75)
    uk.mul_(.5)
    uv.mul_(-.5)
    monkeypatch.setenv("LOD_KIMI_COARSE_PREFILL_HEAD_GROUP", "0")
    expected = run({})
    monkeypatch.setenv("LOD_KIMI_COARSE_PREFILL_HEAD_GROUP", "16")
    actual = run(grouped)
    torch.testing.assert_close(actual[0].sort(-1).values, expected[0].sort(-1).values, atol=0, rtol=0)
    for index, (got, ref) in enumerate(zip(actual[1:], expected[1:]), start=1):
        relative = 2 * torch.finfo(torch.bfloat16).eps if index == 4 else 2e-3
        torch.testing.assert_close(got, ref, atol=2e-4, rtol=relative)
    print({"coarse_projection_bytes_all_heads": original_bytes,
           "grouped_projection_plus_complete_values_bytes": grouped_bytes + assembled_values,
           "saved_bytes": original_bytes - grouped_bytes - assembled_values,
           "scope": "96 heads, unchanged top-eight selection and complete centroid value means"})

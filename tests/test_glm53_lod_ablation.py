"""CPU checks for request-local exact prefill and an unchanged decode path."""
from types import SimpleNamespace

import pytest
import torch

from benchmarks._glm53_lod_ablation import exact_prefill_output, install_lod_ablation


def test_single_final_query_reuses_causal_kernel_without_changing_its_field():
    from benchmarks._glm53_lod_ablation import projected_exact_attention
    torch.manual_seed(60)
    q, latent = torch.randn(1, 2, 1, 3), torch.randn(1, 1, 7, 3)
    destination = torch.empty_like(q)

    def attention(q, kv, uk, uv, *, query_offset, scale, return_lse, buffers, output_buffer):
        assert query_offset == 5 and q.size(2) == 2 and kv.size(2) == 7
        assert output_buffer is None and not return_lse
        scores = q @ kv.transpose(-1, -2) * scale
        mask = torch.arange(7)[None] <= query_offset + torch.arange(2)[:, None]
        return scores.masked_fill(~mask, -torch.inf).softmax(-1) @ kv, None

    actual, _ = projected_exact_attention(q, latent, None, None, attention=attention,
        query_offset=6, scale=.5, return_lse=False, buffers={}, output_buffer=destination)
    expected = (q @ latent.transpose(-1, -2) * .5).softmax(-1) @ latent
    assert actual is destination
    torch.testing.assert_close(actual, expected)


def test_exact_prefill_includes_all_history_causally_and_releases_it():
    torch.manual_seed(41)
    heads, tokens, latent_width, dim = 2, 9, 5, 3
    q = torch.randn(tokens, heads, dim, dtype=torch.float64)
    latent = torch.randn(tokens, latent_width, dtype=torch.float64)
    layer = SimpleNamespace(num_heads=heads, v_head_dim=dim, scale=.5,
        W_UK_T=torch.randn(heads, dim, latent_width, dtype=torch.float64),
        W_UV=torch.randn(heads, latent_width, dim, dtype=torch.float64))
    offsets = []

    def attention(q, kv, uk, uv, *, query_offset, scale, return_lse, buffers, output_buffer):
        offsets.append(query_offset)
        k = torch.einsum("btl,hdl->bhtd", kv[:, 0], uk)
        v = torch.einsum("btl,hld->bhtd", kv[:, 0], uv)
        scores = q @ k.transpose(-1, -2) * scale
        mask = torch.arange(kv.size(2))[None] <= query_offset + torch.arange(q.size(2))[:, None]
        output_buffer.copy_(scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v)
        assert not return_lse
        return output_buffer, None

    outputs = []
    for begin, end in ((0, 4), (4, 7), (7, 9)):
        output = torch.full((end - begin, heads * dim), torch.nan, dtype=torch.float64)
        actual = exact_prefill_output(layer, [(2, 0, end - begin, begin)], {2: tokens},
            q[begin:end], latent[begin:end], output, attention=attention)
        assert actual is output and actual.isfinite().all()
        outputs.append(actual)
    k = torch.einsum("tl,hdl->htd", latent, layer.W_UK_T)
    v = torch.einsum("tl,hld->htd", latent, layer.W_UV)
    scores = q.transpose(0, 1) @ k.transpose(-1, -2) * layer.scale
    mask = torch.arange(tokens)[None] <= torch.arange(tokens)[:, None]
    expected = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v).transpose(0, 1).reshape(tokens, -1)
    torch.testing.assert_close(torch.cat(outputs), expected)
    assert offsets == [0, 4, 7]
    assert layer._glm53_exact_prefill_tokens == tokens
    assert layer._glm53_exact_prefill_history == {}
    with pytest.raises(RuntimeError, match="request offset"):
        exact_prefill_output(layer, [(2, 0, 2, 9)], {2: 11},
            q[:2], latent[:2], output[:2], attention=attention)


@pytest.mark.parametrize("control", ["exact_prefill", "exact_final_row"])
def test_hybrid_hook_does_not_change_decode_or_dummy_calls(monkeypatch, control):
    from vllm_lod_plugin.models import glm53_flash as glm
    seen = []
    sentinel = object()
    def original(*args):
        seen.append(args)
        return sentinel
    monkeypatch.setattr(glm, "latent_attention", original)
    install_lod_ablation(**{control: True})
    layer = SimpleNamespace(_vllm_lod_pool=SimpleNamespace(direct_prefill_plan=None))
    assert glm.latent_attention(layer, "q", "kv", "direct", "shape", "dcp") is sentinel
    assert seen == [(layer, "q", "kv", "direct", "shape", "dcp")]


def test_exact_final_row_preserves_ordinary_prefill_rows_and_uses_full_history():
    torch.manual_seed(59)
    tokens, heads, dim = 9, 2, 3
    q = torch.randn(tokens, heads, dim, dtype=torch.float64)
    latent = torch.randn(tokens, 5, dtype=torch.float64)
    layer = SimpleNamespace(num_heads=heads, v_head_dim=dim, scale=.5,
        W_UK_T=torch.randn(heads, dim, 5, dtype=torch.float64),
        W_UV=torch.randn(heads, 5, dim, dtype=torch.float64))
    calls = []

    def attention(q, kv, uk, uv, *, query_offset, scale, return_lse, buffers, output_buffer):
        calls.append((query_offset, q.size(2), kv.size(2)))
        assert (query_offset, q.size(2), kv.size(2)) == (tokens - 1, 1, tokens)
        k = torch.einsum("btl,hdl->bhtd", kv[:, 0], uk)
        v = torch.einsum("btl,hld->bhtd", kv[:, 0], uv)
        output_buffer.copy_((q @ k.transpose(-1, -2) * scale).softmax(-1) @ v)
        return output_buffer, None

    for begin, end in ((0, 4), (4, 7), (7, 9)):
        original = torch.randn(end - begin, heads * dim, dtype=torch.float64)
        output = original.clone()
        actual = exact_prefill_output(layer, [(3, 0, end - begin, begin)], {3: tokens},
            q[begin:end], latent[begin:end], output, attention=attention, final_row_only=True)
        assert actual is output
        unchanged = end - begin - (end == tokens)
        assert torch.equal(output[:unchanged], original[:unchanged])
        if end < tokens:
            assert not calls
            torch.testing.assert_close(layer._glm53_exact_prefill_history[3], latent[:end])
        else:
            k = torch.einsum("tl,hdl->htd", latent, layer.W_UK_T)
            v = torch.einsum("tl,hld->htd", latent, layer.W_UV)
            weights = torch.einsum("hd,htd->ht", q[-1], k).mul(layer.scale).softmax(-1)
            expected = torch.einsum("ht,htd->hd", weights, v).reshape(-1)
            torch.testing.assert_close(output[-1], expected)
    assert calls == [(8, 1, 9)]
    assert layer._glm53_exact_prefill_calls == layer._glm53_exact_prefill_tokens == 1
    assert layer._glm53_exact_prefill_history == {}


def test_final_row_is_request_local_for_ragged_packed_prefill():
    layer = SimpleNamespace(num_heads=1, v_head_dim=2, scale=1,
                            W_UK_T=None, W_UV=None)
    q = torch.arange(10).view(5, 1, 2).float()
    latent = q[:, 0].clone()
    original = torch.full((5, 2), -7.)
    calls = []

    def attention(q, kv, *args, **kwargs):
        calls.append(kv[0, 0].clone())
        kwargs["output_buffer"].copy_(kv.sum(2, keepdim=True))
        return kwargs["output_buffer"], None

    actual = exact_prefill_output(layer, [(1, 0, 2, 0), (3, 2, 5, 0)], {1: 2, 3: 3},
        q, latent, original.clone(), attention=attention, final_row_only=True)
    assert torch.equal(actual[[0, 2, 3]], original[[0, 2, 3]])
    torch.testing.assert_close(actual[1], latent[:2].sum(0))
    torch.testing.assert_close(actual[4], latent[2:].sum(0))
    assert len(calls) == 2 and calls[0].size(0) == 2 and calls[1].size(0) == 3
    assert layer._glm53_exact_prefill_tokens == layer._glm53_exact_prefill_calls == 2
    assert layer._glm53_exact_prefill_history == {}


def test_uncapping_changes_only_glm_and_not_the_top_eight_policy(monkeypatch):
    from vllm_lod_plugin import pool
    from lod_attention._config import ModelFamily
    def configure(engine, **kwargs):
        engine.max_open_centroid_leaves = 1024
        engine.two_level_topk = engine.prefill_two_level_topk = 8
    monkeypatch.setattr(pool, "configure_engine", configure)
    install_lod_ablation(uncapped=True)
    for family in (ModelFamily.GLM53_FLASH, ModelFamily.KIMI_K3, ModelFamily.QWEN38):
        engine = SimpleNamespace()
        pool.configure_engine(engine, family=family)
        assert engine.max_open_centroid_leaves == (None if family is ModelFamily.GLM53_FLASH else 1024)
        assert engine.two_level_topk == engine.prefill_two_level_topk == 8


def test_combined_ablation_retains_independent_original_functions(monkeypatch):
    from vllm_lod_plugin import pool
    from vllm_lod_plugin.models import glm53_flash as glm
    from lod_attention._config import ModelFamily
    calls = []
    sentinel = object()
    def attention(*args):
        calls.append("attention")
        return sentinel
    def configure(engine, **kwargs):
        calls.append("configuration")
        engine.max_open_centroid_leaves = 1024
    monkeypatch.setattr(glm, "latent_attention", attention)
    monkeypatch.setattr(pool, "configure_engine", configure)
    install_lod_ablation(exact_prefill=True, uncapped=True)
    layer = SimpleNamespace(_vllm_lod_pool=SimpleNamespace(direct_prefill_plan=None))
    assert glm.latent_attention(layer, None, None, None) is sentinel
    engine = SimpleNamespace()
    pool.configure_engine(engine, family=ModelFamily.GLM53_FLASH)
    assert engine.max_open_centroid_leaves is None
    assert calls == ["attention", "configuration"]

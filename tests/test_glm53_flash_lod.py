"""NoPE MLA algebra and adapter boundaries; no trained-model claims."""

from types import SimpleNamespace

import pytest
import torch

from lod_attention._config import ModelFamily, model_family
from vllm_lod_plugin.models.glm53_flash import latent_attention
from vllm_lod_plugin.models.kimi_k3 import absorb_query


def test_indexer_metadata_uses_storage_pages_not_kda_slabs():
    # A 4,352-token KDA slab contains 34 virtual 128-token kernel pages.
    # The existing builder combines pairs into physical 256-token pages,
    # each holding 64 pooled keys. All 17 pages must remain addressable.
    block_table = torch.arange(34).unsqueeze(0)
    factor = 256 // (4 * 32)
    converted = block_table[:, ::factor] // factor
    assert torch.equal(converted, torch.arange(17).unsqueeze(0))


@pytest.mark.parametrize("width", [1026, 1280])
def test_indexer_workspace_rounding_preserves_logical_scores(width):
    from vllm_lod_plugin.models.glm53_flash import _tile_safe_indexer_call
    arguments = [object() for _ in range(6)]
    seen = []

    def original(*args, **kwargs):
        assert list(args[:6]) == arguments
        padded_width = args[6]
        assert padded_width % 256 == 0 and padded_width >= width
        seen.append(padded_width)
        return torch.arange(padded_width).expand(2, -1)

    result = _tile_safe_indexer_call(original, *arguments, width)
    assert result.shape == (2, width)
    assert result.stride(1) == 1
    assert torch.equal(result, torch.arange(width).expand(2, -1))


def test_family_and_fixture_geometry():
    import json
    from pathlib import Path
    from vllm_lod_plugin.config import VLLMLODSettings

    config = json.loads((Path(__file__).parent / "fixtures/glm53-flash-mixed4-fp8/config.json").read_text())
    assert model_family(SimpleNamespace(**config)) is ModelFamily.GLM53_FLASH
    assert VLLMLODSettings.production().for_family(ModelFamily.GLM53_FLASH).family is ModelFamily.GLM53_FLASH
    assert (config["kv_lora_rank"], config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"]) == (512, 256, 0, 256)
    assert config["layer_types"] == ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
    assert config["index_topk"] == 2048 and config["index_kpool"] == 4
    assert config["quantization_config"]["quant_method"] == "fp8"


@pytest.mark.parametrize("mode", ["full", "two-tier", "three-tier-bf16", "three-tier-int4"])
def test_benchmark_uses_native_sparse_control_or_latent_lod(mode):
    from benchmarks._vllm import llm_kwargs

    kwargs = llm_kwargs(checkpoint="zai-org/GLM-5.3-Flash", mode=mode,
        max_model_len=32768, batch_size=1, tensor_parallel_size=4,
        gpu_memory_utilization=0.8, full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    assert kwargs["attention_config"] == (
        {"backend": "ROCM_AITER_MLA_SPARSE"} if mode == "full" else
        {"backend": "CUSTOM", "backend_per_kind": {"mla_attention": "TRITON_MLA"}})
    assert kwargs["language_model_only"]


def test_hf_not_silently_adapted():
    from lod_attention._hf_backend import install_hf_lod_attention
    model = SimpleNamespace(config=SimpleNamespace(model_type="glm5_next_text"))
    with pytest.raises(NotImplementedError, match="vLLM"):
        install_hf_lod_attention(model)


@pytest.mark.parametrize("prefill", [True, False])
def test_latent_adapter_matches_expanded_attention_without_direct_key(prefill):
    torch.manual_seed(18)
    tokens, heads = 3, 2
    query = torch.randn(tokens, heads, 256)
    latent = torch.randn(tokens, 512)
    direct = torch.empty(tokens, 1, 0)
    uk = torch.randn(heads, 256, 512) * 0.02
    uv = torch.randn(heads, 512, 256) * 0.02
    absorbed = absorb_query(query, uk, nope_dim=256)
    observed = []

    def attention(q, k, v, output, **kwargs):
        assert k.data_ptr() == v.data_ptr() == latent.data_ptr()
        assert k.size(-1) == q.size(-1) == 512  # No fake 64 channels.
        # No-RoPE absorption returns a head-major bmm view. Scratch outputs
        # must not silently inherit that layout from empty_like(absorbed).
        assert output.is_contiguous()
        observed.append(q)
        scores = torch.einsum("thl,sl->hts", q, k[:, 0]) * 0.0625
        causal = torch.arange(tokens)[None, :] <= torch.arange(tokens)[:, None]
        weights = scores.masked_fill(~causal, -torch.inf).softmax(-1)
        values = torch.einsum("hts,sl->thl", weights, v[:, 0])
        if kwargs:
            assert kwargs["mla_query"] is query
            assert kwargs["mla_w_uk_t"] is uk and kwargs["mla_w_uv"] is uv
            values = torch.einsum("thl,hlv->thv", values, uv)
        output.copy_(values)

    pool = SimpleNamespace(dcp_world_size=1, direct_prefill_plan=[(0, 0, tokens, 0)] if prefill else None,
                           decode_enabled=not prefill, max_requests=tokens)
    pool.direct_prefill = attention
    pool.decode = lambda q, k, v, _metadata, out: attention(q, k, v, out)
    def project(x, out):
        out.copy_(torch.einsum("thl,hlv->thv", x, uv).reshape(tokens, -1))
    layer = SimpleNamespace(_vllm_lod_pool=pool, W_UK_T=uk, W_UV=uv, num_heads=heads,
                            v_head_dim=256, _v_up_proj=project)
    actual = latent_attention(layer, query, latent, direct)
    key = torch.einsum("sl,hpl->shp", latent, uk)
    value = torch.einsum("sl,hlv->shv", latent, uv)
    scores = torch.einsum("thp,shp->hts", query, key) * 0.0625
    causal = torch.arange(tokens)[None, :] <= torch.arange(tokens)[:, None]
    weights = scores.masked_fill(~causal, -torch.inf).softmax(-1)
    expected = torch.einsum("hts,shv->thv", weights, value).reshape(tokens, -1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(observed[0], absorbed)


def test_dcp_not_silently_accepted():
    layer = SimpleNamespace(_vllm_lod_pool=SimpleNamespace(dcp_world_size=8))
    with pytest.raises(NotImplementedError, match="DCP1"):
        latent_attention(layer, torch.zeros(1, 1, 256), torch.zeros(1, 512), torch.empty(1, 1, 0))


def test_nope_absorption_does_not_copy_the_bmm_output(monkeypatch):
    original = torch.bmm
    products = []
    def bmm(*args, **kwargs):
        products.append(original(*args, **kwargs))
        return products[-1]
    monkeypatch.setattr(torch, "bmm", bmm)
    query, weight = torch.randn(7, 2, 256), torch.randn(2, 256, 512)
    actual = absorb_query(query, weight, nope_dim=256)
    assert actual.data_ptr() == products[0].data_ptr()
    torch.testing.assert_close(actual, torch.einsum("thd,hdl->thl", query, weight))


@pytest.mark.parametrize("project_leaves,previous", [(False, 0), (True, 16384)])
def test_projected_prefill_defers_unused_query_absorption(monkeypatch, project_leaves, previous):
    monkeypatch.setenv("LOD_GLM_PROJECTED_LEAVES", str(int(project_leaves)))
    import vllm_lod_plugin.models.kimi_k3 as kimi
    def unused(*args, **kwargs):
        raise AssertionError("whole-chunk query projection must not execute")
    monkeypatch.setattr(kimi, "absorb_query", unused)
    query, latent = torch.randn(5, 2, 256), torch.randn(5, 512)
    def prefill(q, k, v, out, **kwargs):
        assert q.data_ptr() == latent.data_ptr() and q.stride(1) == 0
        assert kwargs["mla_query"] is query
        assert kwargs["defer_mla_query_absorption"]
        out.copy_(query)
    pool = SimpleNamespace(dcp_world_size=1, engine=SimpleNamespace(
        prefill_exact_first_chunk=True, prefill_chunk_len=16384, _lod_glm_project_local=True),
        direct_prefill_plan=[(0, 0, 5, previous)], direct_prefill=prefill)
    layer = SimpleNamespace(_vllm_lod_pool=pool, W_UK_T=torch.randn(2, 256, 512),
        W_UV=torch.randn(2, 512, 256), num_heads=2, v_head_dim=256)
    actual = latent_attention(layer, query, latent, torch.empty(5, 1, 0))
    torch.testing.assert_close(actual, query.reshape(5, -1))
    assert layer._vllm_lod_deferred_query_tokens == 5


@pytest.mark.parametrize("plan,eligible", [
    (None, False), ([], False),
    ([(3, 0, 16384, 0)], True),
    ([(3, 0, 8192, 0), (4, 8192, 16384, 0)], True),
    ([(3, 0, 16385, 0)], False),
    ([(3, 0, 8192, 16384)], False),
    ([(3, 0, 8192, 0), (4, 8192, 16384, 4096)], False),
])
def test_native_prefix_only_accepts_first_scheduler_chunk(plan, eligible):
    from vllm_lod_plugin.models.glm53_flash import _native_prefix_plan
    layer = SimpleNamespace(_vllm_lod_pool=SimpleNamespace(direct_prefill_plan=plan))
    assert (_native_prefix_plan(layer) is not None) is eligible


def test_native_prefix_hook_is_scoped_and_cleanup_survives_failure():
    engine = SimpleNamespace()
    def prefill(*args, **kwargs):
        assert callable(engine._lod_initial_attention)
        raise RuntimeError("cache construction failed")
    pool = SimpleNamespace(dcp_world_size=1, engine=engine,
        direct_prefill_plan=[(0, 0, 3, 0)], direct_prefill=prefill)
    layer = SimpleNamespace(_vllm_lod_pool=pool, _vllm_lod_native_prefix=True,
                            W_UK_T=torch.randn(2, 256, 512), W_UV=torch.randn(2, 512, 256))
    with pytest.raises(RuntimeError, match="cache construction failed"):
        latent_attention(layer, torch.randn(3, 2, 256), torch.randn(3, 512),
                         torch.empty(3, 1, 0))
    assert not hasattr(engine, "_lod_initial_attention")


def test_combined_mla_projection_layout_and_layer_cache():
    from lod_attention._mla_projection import combined_kv_weight
    torch.manual_seed(81)
    uk, uv = torch.randn(3, 256, 512, dtype=torch.float64), torch.randn(3, 512, 256, dtype=torch.float64)
    latent = torch.randn(7, 512, dtype=torch.float64)
    buffers = {}
    weight = combined_kv_weight(uk, uv, buffers)
    assert combined_kv_weight(uk, uv, buffers) is weight
    packed = (latent @ weight).view(7, 3, 512)
    torch.testing.assert_close(packed[..., :256], torch.einsum("tl,hdl->thd", latent, uk))
    torch.testing.assert_close(packed[..., 256:], torch.einsum("tl,hld->thd", latent, uv))
    assert not torch.equal(packed[..., :256], packed[..., 256:])
    other = combined_kv_weight(uk.clone(), uv.clone(), buffers)
    assert other.data_ptr() != weight.data_ptr()
    uv.add_(1)
    assert combined_kv_weight(uk, uv, buffers).data_ptr() != weight.data_ptr()


def test_model_native_initial_attention_bypasses_only_attention_not_caller():
    from lod_attention._core import TritonLODAttentionCore
    observed = []
    def initial(q, k, v, **kwargs):
        observed.append(kwargs)
        return "native-prefix"
    engine = SimpleNamespace(_lod_initial_attention=initial)
    assert TritonLODAttentionCore._exact_attention(engine, None, None, None,
        causal=True, output_buffer="destination") == "native-prefix"
    assert observed == [dict(causal=True, valid_starts=None, output_buffer="destination")]


def test_glm_prefix_cache_has_one_backing_buffer_and_no_native_latent(monkeypatch):
    pytest.importorskip("vllm")
    from dataclasses import fields
    import vllm.v1.core.kv_cache_utils as utils
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, MambaSpec, KpoolTailSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm_lod_plugin.metadata_cache import LODMetadataOnlyFullAttentionSpec
    from vllm_lod_plugin.models.glm53_cache import install_native_prefix_cache_groups
    from vllm_lod_plugin.cache_ownership import _physical_groups

    for name in ("_get_kv_cache_groups_glm5_next", "get_kv_cache_config_from_groups",
                 "_pool_bytes_per_block"):
        monkeypatch.setattr(utils, name, getattr(utils, name))
    monkeypatch.setenv("VLLM_LOD_ENABLED", "1")
    install_native_prefix_cache_groups()
    native = MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=512,
                             dtype=torch.bfloat16)
    logical = LODMetadataOnlyFullAttentionSpec(**{
        field.name: getattr(native, field.name)
        for field in fields(LODMetadataOnlyFullAttentionSpec)})
    specs = {"model.layers.3.attn": logical,
        "model.layers.3.indexer.k_cache": MLAAttentionSpec(block_size=128,
            num_kv_heads=1, head_size=132, dtype=torch.uint8, tokens_per_state=4),
        "model.layers.3.indexer.tail_cache": KpoolTailSpec(block_size=4,
            num_kv_heads=2, head_size=128, dtype=torch.bfloat16, sliding_window=4),
        **{f"model.layers.{i}.kda": MambaSpec(block_size=128,
            shapes=((16, 16),), dtypes=(torch.float32,)) for i in range(3)}}
    config = SimpleNamespace(model_config=SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm5_next_text", lod_native_prefix=True)),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None,
            prefix_cache_retention_interval=1,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC))
    groups = utils._get_kv_cache_groups_glm5_next(config, specs)
    physical = _physical_groups(groups)
    allocation = utils.get_kv_cache_config_from_groups(config, physical, 8 << 20)
    tensors = allocation.kv_cache_tensors
    assert len({tensor.size for tensor in tensors}) == 1
    assert all("model.layers.3.attn" not in tensor.layers for tensor in tensors)
    assert {name for tensor in tensors for name in tensor.layers} == set(specs) - {"model.layers.3.attn"}
    # Group slabs must not overlap; their native blocks remain contiguous.
    bounds = sorted((tensor.offset, tensor.offset + allocation.num_blocks * tensor.block_stride)
                    for tensor in tensors)
    assert all(left[1] <= right[0] for left, right in zip(bounds, bounds[1:]))
    assert bounds[-1][1] <= tensors[0].size <= 8 << 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU native sparse MLA test")
def test_native_prefix_sparse_output_respects_batched_request_indices():
    pytest.importorskip("vllm")
    from vllm_lod_plugin.models.glm53_flash import _native_prefix_attention

    torch.manual_seed(37)
    tokens, heads, width = 173, 16, 8
    query = (torch.randn(2, tokens, heads, 512, device="cuda") * 0.3).bfloat16().permute(0, 2, 1, 3)
    latent = (torch.randn(2, 1, tokens, 512, device="cuda") * 0.4).bfloat16()
    # Each row has its own different selected keys; early rows have invalid
    # columns. Native indexer output is a packed prefix followed by -1s.
    indices = torch.full((2 * tokens, width), -1, device="cuda", dtype=torch.int32)
    for row in range(2):
        for position in range(tokens):
            count = min(position + 1, width)
            indices[row * tokens + position, :count] = torch.randperm(position + 1, device="cuda")[:count].int()
    layer = SimpleNamespace(indexer=SimpleNamespace(topk_indices_buffer=indices,
        topk_tokens=width), scale=0.0625, impl=SimpleNamespace(sinks=None))
    output = torch.empty(2, tokens, heads, 512, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    plan = [(4, 0, tokens, 0), (9, tokens, 2 * tokens, 0)]
    result = _native_prefix_attention(layer, plan, query, latent, latent,
                                      causal=True, output_buffer=output)
    assert result.data_ptr() == output.data_ptr()
    for row in range(2):
        index = indices[row * tokens:(row + 1) * tokens].long()
        selected = latent[row, 0].float()[index.clamp_min(0)]
        logits = torch.einsum("htd,tkd->htk", query[row].float(), selected) * layer.scale
        logits.masked_fill_(index[None] < 0, -torch.inf)
        expected = torch.einsum("htk,tkd->htd", logits.softmax(-1), selected)
        torch.testing.assert_close(result[row].float(), expected, atol=0.006, rtol=0.03)
    assert layer._vllm_lod_native_prefix_tokens == 2 * tokens


@pytest.mark.parametrize("family,heads,kv_heads,width", [
    (ModelFamily.GLM53_FLASH, 64, 1, 512),
    (ModelFamily.KIMI_K3, 96, 1, 576),
    (ModelFamily.QWEN38, 24, 4, 256),
    (ModelFamily.K2, 64, 8, 128),
])
def test_only_nope_mla_profile_declares_identical_latent_kv(family, heads, kv_heads, width):
    from lod_attention._config import LODMode
    from lod_attention._profile import configure_engine

    engine = SimpleNamespace(config=SimpleNamespace(num_attention_heads=heads,
                                                    num_key_value_heads=kv_heads), head_dim=width)
    configure_engine(engine, family=family, mode=LODMode.TWO_TIER,
                     request_capacity=32768, has_query_norm=True, has_key_norm=False)
    assert engine._lod_shared_latent_kv == (family is ModelFamily.GLM53_FLASH)


def test_nope_local_dispatch_survives_distinct_cached_tensor_allocations(monkeypatch):
    import sys
    from types import ModuleType
    from lod_attention._core import TritonLODAttentionCore

    module = ModuleType("lod_attention.kernels.latent_local_prefill")
    observed = []
    def fast(q, k, **kwargs):
        observed.append((q, k, kwargs))
        return "output", "lse"
    module.latent_local_prefill_attention = fast
    monkeypatch.setitem(sys.modules, module.__name__, module)
    q = torch.randn(1, 4, 7, 512)
    k = torch.randn(1, 1, 7, 512)
    v = k.clone()  # Concatenating K and V separately loses pointer identity.
    engine = SimpleNamespace(prefill_local_attention_backend="aiter",
        _lod_shared_latent_kv=True, scaling=0.0625)
    result = TritonLODAttentionCore._prefill_local_attention(
        engine, q, k, v, query_offset=0, return_lse=True)
    assert result == ("output", "lse") and len(observed) == 1
    assert observed[0][1] is k and observed[0][2]["scale"] == 0.0625


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
def test_scalar_leaf_lookup_matches_fp32_ragged_gqa_and_directory():
    from lod_attention.kernels.paged_prefill import paged_leaf_attention

    torch.manual_seed(56)
    batch, kv_heads, query_heads, queries, slots, capacity = 2, 2, 8, 37, 8, 768
    latent = (torch.randn(batch, kv_heads, capacity, 512, device="cuda") * 0.3).bfloat16()
    query = (torch.randn(batch, query_heads, queries, 512, device="cuda") * 0.2).bfloat16()
    lengths = torch.tensor([700, 19, 1, 0, 33, 0, 0, 0], device="cuda", dtype=torch.int32)
    lengths = lengths.expand(batch, kv_heads, slots).contiguous()
    indices = torch.full((batch, kv_heads, 50, 16), -1, device="cuda", dtype=torch.int32)
    flat = torch.full((batch, kv_heads, slots, 45), -1, device="cuda", dtype=torch.int32)
    offset = 0
    for slot, size in ((0, 700), (1, 19), (2, 1), (4, 33)):
        pages = (size + 15) // 16
        flat[:, :, slot, :pages] = torch.arange(offset, offset + pages, device="cuda")
        leaves = torch.randperm(capacity, device="cuda", dtype=torch.int32)[:size]
        indices[:, :, offset:offset + pages].flatten(2)[..., :size] = leaves
        offset += pages
    # 44+2+1+3 pages are required; use the actual filled directory length.
    # The large expert crosses the old 32-page inline-directory boundary.
    routes = torch.tensor([0, 1, 2, 3, 4, 5, -1, -1], device="cuda", dtype=torch.int64)
    routes = routes.expand(batch, query_heads, queries, 8).contiguous()
    routes[:, :, 0].fill_(-1)
    unused_keys = torch.zeros(batch, kv_heads, 1, device="cuda", dtype=torch.int32)
    used = torch.zeros(1, device="cuda", dtype=torch.int32)
    # Only the first 700+19+1+33=753 leaves are live; the directory indexes
    # arbitrary storage rows, not chronological or physically packed keys.
    for directory_kind in ("flat", "two-level", "hash"):
        root, overflow, probes = flat, unused_keys, 0
        hash_keys, directory_used = unused_keys, used
        if directory_kind == "two-level":
            root = torch.arange(slots, device="cuda", dtype=torch.int32)
            root = root.view(1, 1, slots, 1).expand(batch, kv_heads, slots, 1).contiguous()
            overflow = torch.full((batch, kv_heads, slots, 64), -1,
                                  device="cuda", dtype=torch.int32)
            overflow[..., :45] = flat
            probes = -1
        elif directory_kind == "hash":
            root = flat[..., :1].contiguous()
            key_table, value_table = [-1] * 128, [-1] * 128
            for slot, size in ((0, 700), (1, 19), (2, 1), (4, 33)):
                for ordinal in range(1, (size + 15) // 16):
                    lookup = slot * 65536 + ordinal
                    hashed = lookup
                    hashed ^= hashed >> 16
                    hashed = (hashed * 0x7FEB352D) & 0xFFFFFFFF
                    hashed ^= hashed >> 15
                    hashed = (hashed * 0x846CA68B) & 0xFFFFFFFF
                    hashed ^= hashed >> 16
                    index = hashed & 127
                    while key_table[index] != -1:
                        index = (index + 1) & 127
                    key_table[index] = lookup
                    value_table[index] = int(flat[0, 0, slot, ordinal])
            hash_keys = torch.tensor(key_table, device="cuda", dtype=torch.int32)
            hash_keys = hash_keys.expand(batch, kv_heads, 128).contiguous()
            overflow = torch.tensor(value_table, device="cuda", dtype=torch.int32)
            overflow = overflow.expand(batch, kv_heads, 128).contiguous()
            directory_used = torch.ones(1, device="cuda", dtype=torch.int32)
            probes = 32
        arguments = dict(page_indices=indices, kv_group_size=4, active_slots=slots,
            scale=0.0625, block_m=32, block_n=16, num_warps=2, hash_probes=probes,
            reduce_routes=False)
        output, lse = paged_leaf_attention(query, latent, latent, root, hash_keys,
            overflow, directory_used, lengths, routes, scalar_page_lookup=True, **arguments)
        for slot, route in ((0, 0), (1, 1), (2, 2), (4, 4)):
            page_ids = flat[0, 0, slot]
            leaf_ids = indices[0, 0, page_ids[page_ids >= 0].long()].flatten()
            leaf_ids = leaf_ids[leaf_ids >= 0].long()
            keys = latent[:, :, leaf_ids].float().repeat_interleave(4, dim=1)
            scores = query[:, :, 1:].float() @ keys.transpose(-1, -2) * 0.0625
            torch.testing.assert_close(lse[:, :, 1:, route], scores.logsumexp(-1),
                                       atol=0.003, rtol=0.003)
            torch.testing.assert_close(output[:, :, 1:, route].float(), scores.softmax(-1) @ keys,
                                       atol=0.004, rtol=0.025)
        assert torch.isneginf(lse[:, :, 1:, 3]).all()
        assert output[:, :, 1:, 3].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
def test_latent_only_fused_route_coarse_matches_pytorch():
    from lod_attention.kernels.aiter_mla_prefill_attention import aiter_mla_prefill_route_coarse_attention

    torch.manual_seed(19)
    query = torch.randn(1, 4, 3, 512, dtype=torch.bfloat16, device="cuda") * 0.2
    counts = torch.randint(1, 5, (1, 1, 12, 1), device="cuda").float()
    sums = (torch.randn(1, 1, 12, 512, device="cuda") * counts).bfloat16()
    routes, coarse, _, _ = aiter_mla_prefill_route_coarse_attention(
        query, sums, sums, counts, state_len=12, kv_group_size=4,
        scale=0.0625, normalize_route_query=False)
    # The fused kernel deliberately materializes centroid means in BF16.
    mean = (sums.float() / counts).bfloat16().float()
    logits = query.float() @ mean.transpose(-1, -2) * 0.0625 + counts.squeeze(-1).log().unsqueeze(2)
    expected = logits.softmax(-1) @ mean
    expected_lse = logits.logsumexp(-1)
    torch.testing.assert_close(coarse.output_0.permute(0, 2, 1, 3).float(), expected, atol=0.008, rtol=0.025)
    torch.testing.assert_close(coarse.lse_0.float(), expected_lse, atol=0.004, rtol=0.004)
    assert torch.equal(routes, logits.topk(8, dim=-1).indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
@pytest.mark.parametrize("normalize", [False, True])
def test_query_tiled_route_coarse_ragged_queries_and_multiple_kv_heads(normalize):
    from lod_attention.kernels.aiter_mla_prefill_attention import aiter_mla_prefill_route_coarse_attention

    torch.manual_seed(28)
    q = torch.randn(2, 8, 69, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    counts = torch.randint(1, 9, (2, 2, 137, 1), device="cuda").float()
    key = (torch.randn(2, 2, 137, 512, device="cuda") * counts * 0.3).bfloat16()
    value = (torch.randn_like(key.float()) * counts).bfloat16()
    routes, coarse, _, _ = aiter_mla_prefill_route_coarse_attention(
        q, key, value, counts, state_len=137, kv_group_size=4,
        scale=0.0625, normalize_route_query=normalize)
    mean_k = (key.float() / counts).bfloat16().float().repeat_interleave(4, dim=1)
    mean_v = (value.float() / counts).bfloat16().float().repeat_interleave(4, dim=1)
    log_counts = counts.squeeze(-1).log().repeat_interleave(4, dim=1).unsqueeze(2)
    similarity = q.float() @ mean_k.transpose(-1, -2) * 0.0625
    logits = similarity + log_counts
    torch.testing.assert_close(coarse.output_0.permute(0, 2, 1, 3).float(),
                               logits.softmax(-1) @ mean_v, atol=0.004, rtol=0.025)
    torch.testing.assert_close(coarse.lse_0, logits.logsumexp(-1), atol=0.003, rtol=0.003)
    rms = q.float().square().mean(-1, keepdim=True).sqrt() if normalize else 1.0
    route_scores = similarity + rms * log_counts
    assert torch.equal(routes, route_scores.topk(8, dim=-1).indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
@pytest.mark.parametrize("normalize,tied", [(False, False), (True, False), (True, True)])
def test_gluon_shared_latent_routes_ragged_gqa_and_ties(normalize, tied):
    from lod_attention.kernels.aiter_mla_prefill_attention import aiter_mla_prefill_route_coarse_attention

    torch.manual_seed(41)
    q = torch.randn(2, 8, 69, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    counts = torch.randint(1, 9, (2, 2, 137, 1), device="cuda").float()
    sums = (torch.randn(2, 2, 137, 512, device="cuda") * counts * 0.3).bfloat16()
    if tied:
        q.zero_()
        counts.fill_(2)
    routes, coarse, _, _ = aiter_mla_prefill_route_coarse_attention(
        q, sums, sums, counts, state_len=137, kv_group_size=4,
        scale=0.0625, normalize_route_query=normalize)
    mean = (sums.float() / counts).bfloat16().float().repeat_interleave(4, dim=1)
    logs = counts.squeeze(-1).log().repeat_interleave(4, dim=1).unsqueeze(2)
    similarity = q.float() @ mean.transpose(-1, -2) * 0.0625
    logits = similarity + logs
    torch.testing.assert_close(coarse.output_0.permute(0, 2, 1, 3).float(),
                               logits.softmax(-1) @ mean, atol=0.004, rtol=0.025)
    torch.testing.assert_close(coarse.lse_0, logits.logsumexp(-1), atol=0.003, rtol=0.003)
    rms = q.float().square().mean(-1, keepdim=True).clamp_min(1.e-12).sqrt() if normalize else 1.0
    ranking = similarity + rms * logs
    expected_routes = ranking.argsort(dim=-1, descending=True, stable=True)[..., :8]
    assert torch.equal(routes, expected_routes)
    torch.testing.assert_close(coarse.selected_route_scores,
                               ranking.gather(-1, expected_routes), atol=0.003, rtol=0.003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
@pytest.mark.parametrize("offset,suffix_only", [(0, False), (120, False), (120, True)])
def test_streaming_latent_local_prefill_matches_causal_reference(offset, suffix_only):
    from lod_attention.kernels.latent_local_prefill import latent_local_prefill_attention

    torch.manual_seed(24)
    # Token-major backing exercises the strides used by the HF/vLLM adapter.
    query = torch.randn(2, 173, 4, 512, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3) * 0.3
    latent = torch.randn(2, 1, 173, 512, device="cuda", dtype=torch.bfloat16) * 0.4
    actual_q = query[..., offset:, :]
    output = torch.empty(2, 173 - offset, 4, 512, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    result, lse = latent_local_prefill_attention(
        actual_q if suffix_only else query, latent, query_offset=offset,
        scale=0.0625, output_buffer=output)
    logits = actual_q.float() @ latent.float().transpose(-1, -2) * 0.0625
    visible = torch.arange(173, device="cuda")[None, :] <= offset + torch.arange(173 - offset, device="cuda")[:, None]
    logits.masked_fill_(~visible, -torch.inf)
    expected = logits.softmax(-1) @ latent.float()
    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result.float(), expected, atol=0.004, rtol=0.025)
    torch.testing.assert_close(lse, logits.logsumexp(-1), atol=0.003, rtol=0.003)
    # Output-only exact-front calls neither materialize nor recover an LSE.
    again, no_lse = latent_local_prefill_attention(
        actual_q, latent, query_offset=offset, scale=0.0625, return_lse=False)
    assert no_lse.numel() == 0
    torch.testing.assert_close(again, result, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
def test_exact_front_workspace_is_not_overwritten_by_refined_local_field():
    from lod_attention.kernels.latent_local_prefill import latent_local_prefill_attention

    torch.manual_seed(26)
    q = torch.randn(1, 2, 79, 512, device="cuda", dtype=torch.bfloat16) * 0.3
    kv = torch.randn(1, 1, 79, 512, device="cuda", dtype=torch.bfloat16) * 0.4
    workspace = {}
    front, _ = latent_local_prefill_attention(q, kv, query_offset=0,
        scale=0.0625, return_lse=False, buffers=workspace)
    saved = front.clone()
    local, _ = latent_local_prefill_attention(q * 0.5, kv, query_offset=0,
        scale=0.0625, buffers=workspace)
    assert front.data_ptr() != local.data_ptr()
    torch.testing.assert_close(front, saved, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
def test_native_pooled_indexer_non_tile_width_matches_reference():
    pytest.importorskip("vllm")
    from vllm.platforms import current_platform
    if not current_platform.is_rocm():
        pytest.skip("AMD native FP8 indexer")
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm_lod_plugin.models.glm53_flash import _tile_safe_indexer_call

    init_workspace_manager(torch.device("cuda:0"))
    torch.manual_seed(21)
    pages, page_size, dim, heads, length = 17, 64, 128, 32, 1025
    keys = (torch.randn(pages, page_size, dim, device="cuda") * 0.3).to(ops.FP8_DTYPE)
    query = (torch.randn(1, 1, heads, dim, device="cuda") * 0.2).to(ops.FP8_DTYPE)
    weights = torch.rand(1, heads, device="cuda") * 0.1
    scales = torch.rand(pages, page_size, device="cuda") + 0.5
    # Native indexer pages pack 16x16 shuffled key tiles, then FP32 scales.
    shuffled = keys.reshape(pages, 4, 16, 8, 16).permute(0, 1, 3, 2, 4).contiguous()
    packed = torch.empty(pages, page_size * (dim + 4), device="cuda", dtype=torch.uint8)
    packed[:, :page_size * dim] = shuffled.view(torch.uint8).reshape(pages, -1)
    packed[:, page_size * dim:] = scales.view(torch.uint8).reshape(pages, -1)
    cache = packed.view(pages, page_size, 1, dim + 4)
    blocks = torch.randperm(pages, device="cuda", dtype=torch.int32).unsqueeze(0)
    lengths = torch.tensor([[length]], device="cuda", dtype=torch.int32)
    actual = _tile_safe_indexer_call(ops.rocm_fp8_paged_mqa_logits,
        query, cache, weights, lengths, blocks, None, length + 1)
    ordered_keys = keys.float()[blocks[0].long()].reshape(-1, dim)[:length]
    ordered_scales = scales[blocks[0].long()].flatten()[:length]
    expected = ((query[0, 0].float() @ ordered_keys.T).relu()
                * weights[0, :, None]).sum(0) * ordered_scales
    torch.testing.assert_close(actual[0, :length], expected, atol=0.002, rtol=0.002)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
@pytest.mark.parametrize("width", [512, 576])
def test_absorbed_centroid_materialization_preserves_latent_and_optional_tail(width):
    from lod_attention.kernels.paged_decode_kernels import materialize_absorbed_mla_coarse_means

    torch.manual_seed(23)
    sums = torch.randn(1, 1, 9, width, device="cuda")
    counts = torch.arange(9, device="cuda").view(1, 1, 9, 1).float()
    means = torch.full_like(sums, torch.nan, dtype=torch.bfloat16)
    bias = torch.full((1, 1, 9), torch.nan, device="cuda", dtype=torch.float16)
    materialize_absorbed_mla_coarse_means(sums, counts, means, bias, active_state_len=7)
    expected = sums[:, :, :7] / counts[:, :, :7].clamp_min(1)
    expected[:, :, 0].zero_()
    torch.testing.assert_close(means[:, :, :7].float(), expected, atol=0.008, rtol=0.01)
    assert torch.isneginf(bias[0, 0, 0])
    torch.testing.assert_close(bias[0, 0, 1:7].float(), counts[0, 0, 1:7, 0].log(), atol=0.002, rtol=0.002)
    assert torch.isnan(means[:, :, 7:]).all() and torch.isnan(bias[:, :, 7:]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
@pytest.mark.parametrize("compact,heads", [(False, 16), (True, 8), (True, 16), (True, 32)])
def test_latent_gluon_decode_and_head_tiled_union_match_reference(compact, heads):
    from lod_attention.kernels.kimi_gluon_decode import (
        absorbed_mla_decode_gfx942, absorbed_mla_lod_decode_gfx942)

    torch.manual_seed(22)
    query = torch.randn(1, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    keys = torch.randn(80, 512, device="cuda", dtype=torch.bfloat16) * 0.3
    output = torch.empty(1, heads, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(1, heads, device="cuda", dtype=torch.float32)
    tiles = (heads + 15) // 16
    if compact:
        fixed = torch.randperm(80, device="cuda", dtype=torch.int32).unsqueeze(0)
        starts = (10, 30)[:tiles]
        descriptors = torch.tensor([[start | (16 << 24)] for start in starts], device="cuda", dtype=torch.int32)
        seq_lens = torch.full((tiles,), 19, device="cuda", dtype=torch.int32)
        bias = torch.zeros(80, device="cuda")
        bias[fixed[0, :3].long()] = torch.tensor([0.0, 0.7, 1.2], device="cuda")
        absorbed_mla_lod_decode_gfx942(query, keys, bias, output,
            descriptors, fixed, torch.zeros(1, device="cuda", dtype=torch.int32),
            seq_lens, torch.full_like(seq_lens, 16), torch.zeros(1, device="cuda", dtype=torch.int32),
            0.0625, local_limit=0, head_tiled_metadata=True, include_new=False,
            num_splits=4, final_lse=lse)
        ids = [torch.cat((fixed[0, :3], fixed[0, start:start + 16])).long() for start in starts]
    else:
        bias = torch.zeros(80, device="cuda")
        table = torch.randperm(5, device="cuda", dtype=torch.int32).unsqueeze(0)
        seq_lens = torch.tensor([73], device="cuda", dtype=torch.int32)
        absorbed_mla_decode_gfx942(query, keys.view(5, 16, 512), output,
            table, seq_lens, 0.0625, num_splits=4, final_lse=lse)
        indices = (table[0, :, None] * 16 + torch.arange(16, device="cuda")).flatten()[:73]
        ids = [indices.long()] * tiles
    for tile, indices in enumerate(ids):
        head_slice = slice(tile * 16, min(heads, (tile + 1) * 16))
        scores = query[0, head_slice].float() @ keys[indices].float().T * 0.0625 + bias[indices]
        expected = scores.softmax(-1) @ keys[indices].float()
        torch.testing.assert_close(output[0, head_slice].float(), expected, atol=0.003, rtol=0.02)
        torch.testing.assert_close(lse[0, head_slice], scores.logsumexp(-1), atol=0.003, rtol=0.003)

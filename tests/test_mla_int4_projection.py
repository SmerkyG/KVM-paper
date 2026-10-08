"""Selected INT4 leaves must project identically to their decoded records."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("LOD_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set LOD_RUN_GPU_TESTS=1 on a CUDA/ROCm worker",
)


@pytest.mark.parametrize("width", [512, 576])
@torch.inference_mode()
def test_bf16_summary_keeps_shared_latent_prefix(width):
    from lod_attention._engines import KernelRecursivePagedLODAttention
    torch.manual_seed(169)
    source = torch.randn(2, 1, 129, width, device="cuda").bfloat16()
    owners = (torch.arange(129, device="cuda") % 8).view(1, 1, -1).expand(2, 1, -1).int().contiguous()
    engine = KernelRecursivePagedLODAttention(query_heads=2, key_value_heads=1, scale=.0625).cuda().eval()
    cache = engine._new_page_cache(source, source[..., :512], owners,
        state_capacity=8, sequence_capacity=129, virtual_k=source, virtual_v=source[..., :512])
    # Populate the fixed-pool layout: one K sum with V aliasing its prefix.
    cache["page_sum_k"].zero_()
    cache["page_sum_v"] = cache["page_sum_k"][..., :512]
    for name in ("slot_lengths", "next_page", "page_counts"):
        cache[name].zero_()
    cache["slot_pages"].fill_(-1)
    cache["leaf_count"] = 0
    engine._append_page_cache(cache, source, source[..., :512], owners.long())
    for b in range(2):
        for page in range(int(cache["next_page"][b, 0])):
            n = int(cache["page_counts"][b, 0, page])
            ids = cache["page_indices"][b, 0, page, :n].long()
            expected = source[b, 0, ids].float().sum(0).bfloat16()
            torch.testing.assert_close(cache["page_sum_k"][b, 0, page], expected, rtol=0, atol=0)
            torch.testing.assert_close(cache["page_sum_v"][b, 0, page], expected[:512], rtol=0, atol=0)


@pytest.mark.parametrize("width", [512, 576])
@torch.inference_mode()
def test_shared_int4_append_matches_separate_storage(width):
    from types import SimpleNamespace
    from lod_attention._config import PagedLODConfig
    from lod_attention._engines import KernelRecursivePagedLODAttention
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool
    torch.manual_seed(177)
    batch, total, split, slots = 2, 193, 97, 9
    source = (torch.randn(batch, 1, total, width, device="cuda") * .4).bfloat16()
    owners = (torch.arange(total, device="cuda") % slots).view(1, 1, -1).expand(batch, 1, -1).int().contiguous()
    layer = SimpleNamespace(num_heads=2, num_kv_heads=1, head_size=width,
        head_size_v=512, kv_lora_rank=512, scale=.0625,
        _vllm_lod_absorbed_mla=True, _vllm_lod_glm53=width == 512)
    pool = VLLMLayerLODPool(layer,
        settings=VLLMLODSettings.production(mode="three-tier-int4", pool_size=batch),
        max_requests=batch, request_capacity=8192,
        active_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
        dtype=torch.bfloat16, device=torch.device("cuda"), has_query_norm=True)
    separate_engine = KernelRecursivePagedLODAttention(
        PagedLODConfig(kv_bits=4, quant_group_size=4), query_heads=2, key_value_heads=1, scale=.0625).cuda().eval()
    separate_engine.leaf_quant_scale_mode = pool.engine.leaf_quant_scale_mode
    separate_engine.leaf_append_quant_scale_mode = pool.engine.leaf_append_quant_scale_mode
    caches = []
    for engine, destination, capacity in ((pool.engine, pool.state["page_cache"], pool.state_capacity),
                                          (separate_engine, None, slots)):
        cache = engine._new_page_cache(source[..., :split, :], source[..., :split, :512],
            owners[..., :split], state_capacity=capacity, sequence_capacity=total,
            virtual_k=source, virtual_v=source[..., :512], destination=destination)
        engine._finalize_virtual_page_quantization(cache, destination_page=destination)
        engine._append_page_cache(cache, source[..., split:, :], source[..., split:, :512],
                                  owners[..., split:].long())
        caches.append(cache)
    shared, separate = caches
    for b in range(batch):
        n = int(shared["next_page"][b, 0])
        assert n == int(separate["next_page"][b, 0])
        # Page IDs are allocated atomically and may differ between launches.
        # Compare summaries/scales by page membership, not allocation order.
        def membership(cache, page):
            count = int(cache["page_counts"][b, 0, page])
            return tuple(cache["page_indices"][b, 0, page, :count].tolist())
        separate_pages = {membership(separate, p): p for p in range(n)}
        corresponding = torch.tensor(
            [separate_pages[membership(shared, p)] for p in range(n)],
            device=source.device,
        )
        for name in ("quantized_leaf_k", "quantized_leaf_v"):
            torch.testing.assert_close(shared[name][b, :, :total], separate[name][b, :, :total], rtol=0, atol=0, msg=name)
        for name in ("page_k_scales", "page_v_scales", "quantized_page_sum_k",
                     "quantized_page_sum_v", "page_sum_k_scales", "page_sum_v_scales"):
            torch.testing.assert_close(shared[name][b, :, :n], separate[name][b, :, corresponding], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("width", [512, 576])
@pytest.mark.parametrize("fixed_pool", [False, True])
@torch.inference_mode()
def test_compact_projection_reads_int4_not_bf16_placeholder(width, fixed_pool):
    from lod_attention._config import PagedLODConfig
    from lod_attention._engines import KernelRecursivePagedLODAttention
    from lod_attention.kernels.glm_compact_leaf_projection import project_compact_glm_leaves
    from lod_attention.kernels.kimi_compact_leaf_projection import project_compact_kimi_leaves

    torch.manual_seed(157)
    batch, heads, tokens, slots = 2, 2, 193, 9
    source = (torch.randn(batch, 1, tokens, width, device="cuda") * .4).bfloat16()
    owners = (torch.arange(tokens, device="cuda") // 19 % slots).int().view(1, 1, -1).expand(batch, -1, -1).contiguous()
    engine = KernelRecursivePagedLODAttention(PagedLODConfig(kv_bits=4, quant_group_size=4),
        query_heads=heads, key_value_heads=1, scale=.0625).cuda().eval()
    destination = None
    state_capacity = slots
    if fixed_pool:
        from types import SimpleNamespace
        from vllm_lod_plugin.config import VLLMLODSettings
        from vllm_lod_plugin.pool import VLLMLayerLODPool
        layer = SimpleNamespace(num_heads=heads, num_kv_heads=1, head_size=width,
            head_size_v=512, kv_lora_rank=512, scale=.0625,
            _vllm_lod_absorbed_mla=True, _vllm_lod_glm53=width == 512)
        pool = VLLMLayerLODPool(layer,
            settings=VLLMLODSettings.production(mode="three-tier-int4", pool_size=batch),
            max_requests=batch, request_capacity=8192,
            active_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
            dtype=torch.bfloat16, device=torch.device("cuda"), has_query_norm=True)
        engine = pool.engine
        destination, state_capacity = pool.state["page_cache"], pool.state_capacity
    cache = engine._new_page_cache(source, source[..., :512], owners,
        state_capacity=state_capacity, sequence_capacity=tokens, virtual_k=source,
        virtual_v=source[..., :512], destination=destination)
    engine._finalize_virtual_page_quantization(cache, destination_page=destination)
    decoded = torch.zeros_like(source)
    group = engine.leaf_quant_group_size
    # Independent nibble/anchor decoding of the stored representation.
    for b in range(batch):
        for page in range(int(cache["next_page"][b, 0])):
            n = int(cache["page_counts"][b, 0, page])
            index = cache["page_indices"][b, 0, page, :n].long()
            packed = cache["quantized_leaf_k"][b, 0, index].int()
            codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).float() - 8
            if cache.get("summary_quantization_finalized", False):
                sums = cache["quantized_page_sum_k"][b, 0, page].float() * cache["page_sum_k_scales"][b, 0, page].float().repeat_interleave(group)
            else:
                sums = cache["page_sum_k"][b, 0, page].float()
            decoded[b, 0, index] = (codes * cache["page_k_scales"][b, 0, page].float().repeat_interleave(group) + sums / n).bfloat16()
    assert ((decoded.float() - source.float()).norm() / source.float().norm()).item() < .15
    key_dim = 256 if width == 512 else 128
    uk = (torch.randn(heads, key_dim, 512, device="cuda") / 512**.5).bfloat16()
    uv = (torch.randn(heads, 512, key_dim, device="cuda") / 512**.5).bfloat16()
    counts = torch.zeros(batch, heads, slots, dtype=torch.int32, device="cuda")
    counts[0, 0, [0, 3]], counts[0, 1, [4, 8]] = 1, 1
    counts[1, 0, [2, 5]], counts[1, 1, [1, 7]] = 1, 1
    placeholder = torch.full((batch, 1, 1, width), float("nan"), device="cuda", dtype=torch.bfloat16)
    cache["leaf_k"] = placeholder
    projection = project_compact_glm_leaves if width == 512 else project_compact_kimi_leaves
    actual_k, actual_v, starts = projection(placeholder, uk, uv, cache, counts.flatten(),
        active_slots=slots, hash_probes=engine._page_lookup_probes(cache), buffers={})
    expected_k = torch.einsum("btl,hdl->bhtd", decoded[:, 0, :, :512].float(), uk.float()).bfloat16()
    if width == 576:
        expected_k = torch.cat((expected_k, decoded[..., 512:].expand(-1, heads, -1, -1)), -1)
    expected_v = torch.einsum("btl,hld->bhtd", decoded[:, 0, :, :512].float(), uv.float()).bfloat16()
    starts = starts.cpu()
    for b, h, slot in counts.nonzero().tolist():
        expert = (b * heads + h) * slots + slot
        a, z = int(starts[expert]), int(starts[expert + 1])
        mask = owners[b, 0] == slot
        torch.testing.assert_close(actual_k.reshape(-1, actual_k.size(-1))[a:z], expected_k[b, h, mask], atol=.004, rtol=.025)
        torch.testing.assert_close(actual_v.reshape(-1, actual_v.size(-1))[a:z], expected_v[b, h, mask], atol=.004, rtol=.025)

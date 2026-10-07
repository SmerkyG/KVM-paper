"""Exact-storage alternative for million-token K3 capacity experiments."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("compact", [False, True])
def test_compact_directory_is_explicit_and_preserves_latent_alias(monkeypatch, compact):
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool

    monkeypatch.setenv("LOD_KIMI_COMPACT_PAGE_DIRECTORY", str(int(compact)))
    layer = SimpleNamespace(num_heads=12, num_kv_heads=1, head_size=576,
                            kv_lora_rank=512, scale=192**-.5,
                            _vllm_lod_absorbed_mla=True)
    pool = VLLMLayerLODPool(layer, settings=VLLMLODSettings(), max_requests=1,
        request_capacity=4096, active_indices=torch.zeros(1, dtype=torch.long),
        dtype=torch.bfloat16, device=torch.device("cpu"), request_owner_prefill=False)
    page = pool.state["page_cache"]
    assert pool.engine.leaf_paged_directory is not compact
    assert page["paged_page_directory"] is not compact
    assert pool.engine._page_lookup_probes(page) == (32 if compact else -1)
    pool._assert_shared_latent_storage()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP page-directory kernels")
@torch.inference_mode()
def test_fixed_compact_directory_keeps_overflow_leaves_and_latent_alias(monkeypatch):
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool

    torch.manual_seed(91)
    device = torch.device("cuda")
    layer = SimpleNamespace(num_heads=96, num_kv_heads=1, head_size=576,
                            kv_lora_rank=512, scale=192**-.5,
                            _vllm_lod_absorbed_mla=True)
    # A large closed bucket exercises real overflow hashing, not just the
    # inline entries used by <=1024-leaf refined centroids. Capacity matches
    # the real million-token owner; this is not a trained serving benchmark.
    tokens, first = 70017, 69259
    source = torch.randn(1, 1, tokens, 576, device=device, dtype=torch.bfloat16)
    owners = torch.zeros(1, 1, tokens, device=device, dtype=torch.int32)
    owners[..., 68000:] = torch.arange(tokens - 68000, device=device) % 7 + 1
    semantic_lists, directory_bytes = [], []
    for compact in (False, True):
        monkeypatch.setenv("LOD_KIMI_COMPACT_PAGE_DIRECTORY", str(int(compact)))
        pool = VLLMLayerLODPool(layer, settings=VLLMLODSettings(), max_requests=1,
            request_capacity=1045514, active_indices=torch.zeros(1, device=device, dtype=torch.long),
            dtype=torch.bfloat16, device=device, request_owner_prefill=False)
        page = pool.engine._new_page_cache(
            source[..., :first, :], source[..., :first, :512], owners[..., :first],
            state_capacity=pool.state_capacity, sequence_capacity=pool.leaf_capacity,
            virtual_k=source[..., :first, :], virtual_v=source[..., :first, :512],
            destination=pool.state["page_cache"],
        )
        pointers = {name: page[name].data_ptr() for name in (
            "slot_pages", "overflow_page_keys", "overflow_page_values", "leaf_k")}
        pool.engine._append_page_cache(page, source[..., first:, :], source[..., first:, :512],
                                       owners[..., first:])
        torch.cuda.synchronize()
        assert int(page["overflow_flag"]) == 0
        assert page["paged_page_directory"] is not compact
        assert pool.engine._page_lookup_probes(page) == (32 if compact else -1)
        for name, pointer in pointers.items():
            assert page[name].data_ptr() == pointer
        assert page["leaf_k"].untyped_storage().data_ptr() == page["leaf_v"].untyped_storage().data_ptr()
        torch.testing.assert_close(page["leaf_k"][..., :tokens, :], source, atol=0, rtol=0)
        roots, keys, values, indices, lengths = [page[name][0, 0].cpu() for name in (
            "slot_pages", "overflow_page_keys", "overflow_page_values", "page_indices", "slot_lengths")]
        hashed = dict(zip(keys[keys.ge(0)].tolist(), values[keys.ge(0)].tolist())) if compact else {}
        listed = []
        for slot in range(8):
            count = int(lengths[slot])
            leaves = []
            for ordinal in range((count + 15) // 16):
                if compact:
                    physical = (int(roots[slot, ordinal]) if ordinal < roots.size(1)
                                else hashed[slot * 65536 + ordinal])
                else:
                    directory = int(roots[slot, ordinal // 64])
                    assert directory >= 0
                    physical = int(values[directory, ordinal % 64])
                assert physical >= 0
                leaves.extend(indices[physical, :min(16, count - ordinal * 16)].tolist())
            expected = torch.nonzero(owners[0, 0].cpu().eq(slot)).flatten().tolist()
            assert sorted(leaves) == expected
            listed.append(sorted(leaves))
        semantic_lists.append(listed)
        directory_bytes.append(sum(page[name].untyped_storage().nbytes() for name in (
            "slot_pages", "overflow_page_keys", "overflow_page_values")))
    assert semantic_lists[0] == semantic_lists[1]
    assert (directory_bytes[0] - directory_bytes[1]) * 24 > 1024**3
    print({"directory_bytes_per_layer": directory_bytes,
           "saved_bytes_24_layers": (directory_bytes[0] - directory_bytes[1]) * 24,
           "scope": "page indexing only; identical KV records and memberships"})

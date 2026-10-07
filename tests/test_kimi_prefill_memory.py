from types import SimpleNamespace as NS

import torch

from benchmarks._kimi_prefill_memory import (
    shared_decode_scratch_summary, snapshot_prefill_memory, unique_storage_groups,
)


def test_groups_count_backing_storage_once_across_aliases():
    latent = torch.empty(2, 576, dtype=torch.bfloat16)
    other = torch.empty(17, dtype=torch.int32)
    sizes = unique_storage_groups({
        "cache": {"k": latent, "v": latent[..., :512]},
        "scratch": [latent.view(-1)[:3], (other, other)],
        "metadata": [1, None, "not a tensor"],
    })
    assert sizes == {"cache": latent.numel() * 2, "scratch": 17 * 4, "metadata": 0}


def test_snapshot_deduplicates_shared_scratch_and_reports_only_real_shadows(monkeypatch):
    latent, workspace, weight = [torch.empty(n) for n in (576, 99, 999)]
    pool = NS(state={"state_k": latent, "state_v": latent[:512]},
              dcp_prefill_shadows={3: NS(state={"total_len": 16384, "leaf_k": workspace})},
              decode_buffer_storage=None, dcp_decode_buffer_storage={"tmp": workspace})
    runtime = NS(pools={"layer": pool},
                 _prefill_attention_buffers={"tmp": workspace, "weight_source": weight})
    worker = NS(model_runner=NS(model_state=NS(_vllm_lod_runtime=runtime)))
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda: 123)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (456, 789))
    snapshot = snapshot_prefill_memory(worker)
    assert snapshot["shadow_rows_by_layer"] == {"layer": [3]}
    assert snapshot["shadow_global_lengths_by_layer"] == {"layer": {"3": 16384}}
    assert sum(snapshot["allocation_groups_bytes"].values()) == (576 + 99) * 4
    assert snapshot["allocation_groups_bytes"]["shared_prefill_scratch"] == 0


def test_shared_decode_summary_counts_one_registry_not_24_copies():
    data = torch.empty(99)
    registry = {"temp": data, "alias": data[:7]}
    pools = {str(index): NS(shared_decode_scratch=registry) for index in range(24)}
    pools["private"] = NS(shared_decode_scratch=None)
    summary = shared_decode_scratch_summary(pools)
    assert summary["participating_layers"] == 24 and summary["registries"] == 1
    assert summary["storage_bytes"] == 99 * 4


def test_sharded_b1_memory_reports_backing_without_counting_the_archive_twice(monkeypatch):
    leaf = torch.empty(1, 1, 320, 576, dtype=torch.bfloat16)
    page = dict(leaf_k=leaf, leaf_v=leaf[..., :512], leaf_count=64,
                dcp_leaf_sharded=True, dcp_prefill_pool_backed=True)
    pool = NS(state={"leaf_k": leaf},
              dcp_prefill_shadows={0: NS(state=dict(total_len=513, page_cache=page))},
              decode_buffer_storage=None, dcp_decode_buffer_storage=None)
    worker = NS(model_runner=NS(model_state=NS(_vllm_lod_runtime=NS(pools={"layer": pool}))))
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda: 123)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (456, 789))
    snapshot = snapshot_prefill_memory(worker)
    assert snapshot["allocation_groups_bytes"]["persistent_semantic_cache"] == leaf.numel() * 2
    assert snapshot["allocation_groups_bytes"]["replicated_prefill_shadows"] == 0
    assert snapshot["sharded_prefill_archive_by_layer"] == {"layer": {"0": dict(
        pool_backed=True, local_leaf_capacity=320, covered_local_leaves=64, global_length=513)}}


def test_owner_memory_counts_child_cache_and_deduplicates_backed_prefill(monkeypatch):
    leaf, tail, pending, weights, construction = [torch.empty(n) for n in (576, 17, 19, 23, 37)]
    engine = NS(_lod_state_update_buffers={"delta": construction, "alias": leaf})
    child = NS(state={"page_cache": {"leaf_k": leaf, "leaf_v": leaf[:512]}},
               decode_buffer_storage=None, dcp_decode_buffer_storage=None, engine=engine)
    cache = NS(state={"leaf_k": leaf, "recent_k": tail, "owner_remote_pool_backed": True})
    pool = NS(state={}, owner_decode_pool=child, owner_decode_buffers={},
        dcp_prefill_shadows={}, decode_buffer_storage=None, dcp_decode_buffer_storage=None,
        _kimi_request_owner_rows={2: {"cache": cache, "parts": [pending]}},
        layer=NS(_lod_owner_uk=weights, _lod_owner_uv=weights), engine=engine)
    runtime = NS(pools={"layer": pool})
    worker = NS(model_runner=NS(model_state=NS(_vllm_lod_runtime=runtime)))
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda: 123)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (456, 789))
    snapshot = snapshot_prefill_memory(worker)
    groups = snapshot["allocation_groups_bytes"]
    assert groups["owner_persistent_semantic_cache"] == 576 * 4
    assert groups["owner_prefill_rows"] == (17 + 19) * 4
    assert groups["owner_projection_weights"] == 23 * 4
    assert groups["construction_workspaces"] == 37 * 4
    assert groups["owner_construction_workspaces"] == 0
    assert sum(groups.values()) == (576 + 17 + 19 + 23 + 37) * 4
    assert snapshot["owner_remote_pool_backed_by_layer"] == {"layer": {"2": True}}

"""Host-only allocation accounting for untimed prefill warmup.

Count backing storage, not logical tensor views: Kimi K/V alias the same
latent record, and multiple layers share a prefill workspace. This observes
live allocations at a chunk boundary, not an exclusive GPU-time attribution
or an estimate of the peak inside an attention/MoE kernel.
"""

from __future__ import annotations

import torch


def unique_storage_groups(groups: dict) -> dict[str, int]:
    """Assign shared storage once, to the first group containing it."""
    seen = set()

    def count(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            identity = (value.device.type, value.device.index, int(storage.data_ptr()))
            if identity in seen:
                return 0
            seen.add(identity)
            return int(storage.nbytes())
        if isinstance(value, dict):
            return sum(count(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count(child) for child in value)
        return 0

    return {name: count(value) for name, value in groups.items()}


def shared_decode_scratch_summary(pools) -> dict:
    """Host-only audit of reserved sharing; no events or GPU tensor reads."""
    layers = [pool for pool in pools.values()
              if getattr(pool, "shared_decode_scratch", None) is not None]
    registries = {id(pool.shared_decode_scratch): pool.shared_decode_scratch for pool in layers}
    return {
        "participating_layers": len(layers),
        "registries": len(registries),
        "unique_tensor_count": sum(map(len, registries.values())),
        "storage_bytes": sum(unique_storage_groups(registries).values()),
        "scope": "shared transient tensors only; per-layer outputs/state excluded",
    }


def snapshot_prefill_memory(worker) -> dict:
    """Called only by the warmup audit, never in measured generation."""
    runner = worker.model_runner
    runtime = getattr(getattr(runner, "model_state", None), "_vllm_lod_runtime", None)
    if runtime is None:
        runtime = getattr(runner, "_vllm_lod_runtime", None)
    pools = {} if runtime is None else runtime.pools
    owners = {name: pool.owner_decode_pool for name, pool in pools.items()
              if getattr(pool, "owner_decode_pool", None) is not None}
    owner_rows = {name: {row: {
        "cache": None if value.get("cache") is None else value["cache"].state,
        "parts": value.get("parts", []),
    } for row, value in getattr(pool, "_kimi_request_owner_rows", {}).items()}
                  for name, pool in pools.items()}
    shadows = {name: {row: cache.state for row, cache in pool.dcp_prefill_shadows.items()}
               for name, pool in pools.items()}
    # Immutable weight-source references may point into a daemon's much larger
    # IPC storage. They are not client scratch allocations. Actual client-owned
    # transformed weight layouts remain in the workspace accounting.
    scratch = {}
    for name, value in getattr(runtime, "_prefill_attention_buffers", {}).items():
        if isinstance(name, str) and name.endswith("_source"):
            continue
        if isinstance(name, tuple) and name[:1] == ("mla_combined_kv",):
            # (versions, packed layout, key source, value source). Only the
            # packed layout is client-owned; source views retain IPC weights.
            value = value[1]
        scratch[name] = value
    groups = unique_storage_groups({
        "persistent_semantic_cache": {name: pool.state for name, pool in pools.items()},
        "owner_persistent_semantic_cache": {name: pool.state for name, pool in owners.items()},
        "owner_prefill_rows": owner_rows,
        "replicated_prefill_shadows": shadows,
        "decode_scratch": {name: (pool.decode_buffer_storage, pool.dcp_decode_buffer_storage)
                           for name, pool in pools.items()},
        "owner_decode_scratch": {name: (
            pool.decode_buffer_storage, pool.dcp_decode_buffer_storage,
            pools[name].owner_decode_buffers,
        ) for name, pool in owners.items()},
        "owner_projection_weights": {name: (
            getattr(pool.layer, "_lod_owner_uk", None),
            getattr(pool.layer, "_lod_owner_uv", None),
        ) for name, pool in pools.items() if name in owners},
        "owner_transport": getattr(getattr(runtime, "dcp_group", None), "_lod_owner_transport", {}),
        "shared_prefill_scratch": scratch,
        "construction_workspaces": {name: (
            getattr(getattr(pool, "engine", None), "_lod_state_update_buffers", None),
            getattr(getattr(pool, "engine", None), "_lod_state_maxsim_buffers", None),
        ) for name, pool in pools.items()},
        "owner_construction_workspaces": {name: (
            getattr(getattr(pool, "engine", None), "_lod_state_update_buffers", None),
            getattr(getattr(pool, "engine", None), "_lod_state_maxsim_buffers", None),
        ) for name, pool in owners.items()},
        "queued_construction_sources": (
            getattr(runtime, "_initial_prefill_sources", {}),
            getattr(runtime, "_cached_prefill_sources", {}),
        ),
    })
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocation_groups_bytes": groups,
        "shared_decode_scratch": shared_decode_scratch_summary(pools),
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "torch_peak_allocated_bytes_so_far": torch.cuda.max_memory_allocated(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
        "shadow_rows_by_layer": {name: sorted(map(int, rows)) for name, rows in shadows.items()},
        "shadow_global_lengths_by_layer": {
            name: {str(row): int(state["total_len"]) for row, state in rows.items()}
            for name, rows in shadows.items()
        },
        "sharded_prefill_archive_by_layer": {
            name: {str(row): {
                "pool_backed": bool(state["page_cache"].get("dcp_prefill_pool_backed")),
                "local_leaf_capacity": int(state["page_cache"]["leaf_k"].size(2)),
                "covered_local_leaves": int(state["page_cache"]["leaf_count"]),
                "global_length": int(state["total_len"]),
            } for row, state in rows.items()
                   if state.get("page_cache", {}).get("dcp_leaf_sharded")}
            for name, rows in shadows.items()
        },
        "owner_remote_pool_backed_by_layer": {
            name: {str(row): bool(value.get("cache") and value["cache"].state.get(
                "owner_remote_pool_backed", False))
                for row, value in getattr(pool, "_kimi_request_owner_rows", {}).items()}
            for name, pool in pools.items() if name in owners
        },
        "scope": "live storage at untimed chunk boundary; excludes daemon weights and driver scratch",
    }

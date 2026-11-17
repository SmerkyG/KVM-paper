"""Measure persistent cache storage and allocator usage at the inference phases."""

from bisect import bisect_right

from large_batch_inference import CapturedGeneration, torch
from model.kvm_triton_decode import TritonKVMDecodeCache


def tensor_storages(values):
    storages = {}
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            storage = value.untyped_storage()
            if storage.nbytes():
                storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, dict):
            storages.update(tensor_storages(value.values()))
        elif isinstance(value, TritonKVMDecodeCache):
            storages.update(tensor_storages(vars(value).values()))
    return storages


def cache_storages(cache):
    all_storage = {}
    attention_storage = {}
    for layer in cache.layers:
        states = layer.states_dict
        all_storage.update(tensor_storages(states.values()))
        for value in states.values():
            if isinstance(value, TritonKVMDecodeCache):
                names = (
                    "state_k", "state_v", "state_k_attn", "state_v_attn", "state_vlen",
                    "recent_k", "recent_v", "recent_gate",
                )
                attention_storage.update(tensor_storages(getattr(value, name) for name in names))
        attention_storage.update(tensor_storages(states[key] for key in ("k", "v") if key in states))
    return all_storage, attention_storage


def bytes_in_segments(storages, segments):
    by_address = sorted((s["address"], s["total_size"]) for s in segments)
    starts = [address for address, _ in by_address]
    used = 0
    for address, size in storages.items():
        index = bisect_right(starts, address) - 1
        if index >= 0:
            start, length = by_address[index]
            if address + size <= start + length:
                used += size
    return used


def measure_memory(generation, prompt, *, decode_graph_steps):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before_prefill = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    first = generation.prefill(prompt)
    torch.cuda.synchronize()
    prefill_peak = torch.cuda.max_memory_allocated()
    prefill_live = torch.cuda.memory_allocated()
    prefill_cache, _ = cache_storages(generation.cache)
    del first
    torch.cuda.empty_cache()
    memory_after_prefill_capture = {}

    def record_prefill_pool(prefill_graph):
        segments = torch.cuda.memory_snapshot(prefill_graph.pool())
        storage, _ = cache_storages(generation.cache)
        memory_after_prefill_capture.update({
            "prefill_graph_pool_reserved_bytes": sum(s["total_size"] for s in segments),
            "prefill_graph_cache_bytes_in_pool": bytes_in_segments(storage, segments),
        })

    graph = CapturedGeneration(
        generation, prompt, 1, decode_graph_steps=decode_graph_steps,
        after_prefill_capture=record_prefill_pool,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    first = graph.prefill(prompt)
    generated, logits = graph.decode(first)
    torch.cuda.synchronize()
    replay_peak_reserved = torch.cuda.max_memory_reserved()
    if generated.shape != (prompt.size(0), len(generation.decode_positions) + 1):
        raise AssertionError(f"Unexpected generated shape: {generated.shape}")
    del first, generated, logits
    torch.cuda.empty_cache()
    segments = torch.cuda.memory_snapshot(graph.prefill_graph.pool())
    graph_pool = sum(s["total_size"] for s in segments)
    total_reserved = torch.cuda.memory_reserved()
    storage, attention_storage = cache_storages(generation.cache)
    model_storages = tensor_storages(generation.model.parameters())
    model_storages.update(tensor_storages(generation.model.buffers()))
    cache_bytes = sum(size for address, size in storage.items() if address not in model_storages)
    attention_bytes = sum(size for address, size in attention_storage.items() if address not in model_storages)
    cache_in_pool = bytes_in_segments(storage, segments)
    if not (0 < graph_pool <= total_reserved and 0 < attention_bytes <= cache_bytes):
        raise AssertionError("Unexpected graph-pool or cache size")
    return {
        "eager_before_prefill_allocated_bytes": before_prefill,
        "eager_prefill_peak_allocated_bytes": prefill_peak,
        "eager_after_prefill_allocated_bytes": prefill_live,
        "eager_after_prefill_cache_bytes": sum(prefill_cache.values()),
        "decode_cache_bytes": cache_bytes,
        "decode_attention_state_bytes": attention_bytes,
        "decode_cache_bytes_in_graph_pool": cache_in_pool,
        "graph_pool_reserved_bytes": graph_pool,
        "graph_pool_reserved_minus_cache_bytes": graph_pool - cache_in_pool,
        "replay_total_reserved_bytes": total_reserved,
        "replay_peak_reserved_bytes": replay_peak_reserved,
        "decode_graphs": len(graph.decode_graphs),
        **memory_after_prefill_capture,
    }

"""Opt-in worker GPU traces for diagnosis, never canonical wall timings."""


def start_prefill_profile(worker, collect_projection_usage=True):
    import os
    import torch

    if collect_projection_usage:
        os.environ["LOD_KIMI_PROFILE_PROJECTED_LEAVES"] = "1"
    capture_path = os.environ.get("LOD_KIMI_CAPTURE_PREFILL")
    if capture_path:
        _install_leaf_capture(worker, capture_path)
    _install_attention_shape_trace(worker)
    torch.cuda.synchronize()
    worker._lod_prefill_profile = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
    )
    worker._lod_prefill_profile.start()


def stop_prefill_profile(worker):
    import os
    import torch

    torch.cuda.synchronize()
    profiler = worker._lod_prefill_profile
    profiler.stop()
    totals = {}
    cpu_totals = {}
    gpu_start = float("inf")
    gpu_end = 0.0
    chunk_kernels = []
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CPU:
            item = cpu_totals.setdefault(event.name, {"calls": 0, "self_cpu_us": 0.0})
            item["calls"] += 1
            item["self_cpu_us"] += event.self_cpu_time_total
            continue
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        item = totals.setdefault(event.name, {"calls": 0, "total_gpu_us": 0.0})
        item["calls"] += 1
        item["total_gpu_us"] += event.device_time_total
        gpu_start = min(gpu_start, event.time_range.start)
        gpu_end = max(gpu_end, event.time_range.end)
        if any(name in event.name for name in (
            "_latent_local_prefill_gluon", "latent_route_coarse_gluon",
            "_mla_route_coarse_prefill_kernel", "_paged_leaf_attention_kernel",
        )):
            chunk_kernels.append(dict(name=event.name,
                start_us=event.time_range.start, gpu_ms=event.device_time_total / 1000))
    del worker._lod_prefill_profile
    for engine, original in getattr(worker, "_lod_prefill_capture_hooks", []):
        engine._paged_leaf_attention = original
    worker._lod_prefill_capture_hooks = []
    for engine, original in getattr(worker, "_lod_prefill_shape_hooks", []):
        engine._two_level_attention = original
    worker._lod_prefill_shape_hooks = []
    os.environ.pop("LOD_KIMI_PROFILE_PROJECTED_LEAVES", None)
    runtime = getattr(worker.model_runner, "_vllm_lod_runtime", None)
    # The V2 runner stores this extension on its model state instead.
    if runtime is None:
        runtime = getattr(getattr(worker.model_runner, "model_state", None),
                          "_vllm_lod_runtime", None)
    usage = []
    if runtime is not None:
        for pool in runtime.pools.values():
            usage.extend(getattr(pool.engine, "_lod_kimi_projection_usage", []))
    return {
        "scope": "diagnostic profiler; instrumentation overhead included",
        "gpu_timeline_span_ms": (gpu_end - gpu_start) / 1000,
        "summed_kernel_ms_not_wall_ms": sum(
            item["total_gpu_us"] for item in totals.values()
        ) / 1000,
        "kernels": sorted(
            [{"name": name, **item} for name, item in totals.items()],
            key=lambda item: item["total_gpu_us"], reverse=True,
        ),
        "cpu_events": sorted(
            [{"name": name, **item} for name, item in cpu_totals.items()],
            key=lambda item: item["self_cpu_us"], reverse=True,
        )[:40],
        "leaf_projection_usage": usage,
        "chunk_attention_kernels": sorted(chunk_kernels, key=lambda item: item["start_us"]),
        "attention_shapes": getattr(worker, "_lod_prefill_attention_shapes", []),
    }


def _install_attention_shape_trace(worker):
    """Record host-known geometry; no GPU reads, events or synchronization."""
    from types import MethodType

    runner = worker.model_runner
    runtime = getattr(runner, "_vllm_lod_runtime", None)
    if runtime is None:
        runtime = getattr(getattr(runner, "model_state", None), "_vllm_lod_runtime", None)
    shapes, hooks = [], []
    if runtime is not None:
        for name, pool in runtime.pools.items():
            engine, original = pool.engine, pool.engine._two_level_attention

            def attention(self, q, k, v, *args, _original=original, _name=str(name), **kwargs):
                shapes.append(dict(layer=_name, query_shape=list(q.shape),
                    local_keys=int(k.size(2)), state_len=int(kwargs["state_len"]),
                    context_len=kwargs.get("context_len")))
                return _original(q, k, v, *args, **kwargs)

            engine._two_level_attention = MethodType(attention, engine)
            hooks.append((engine, original))
    worker._lod_prefill_shape_hooks = hooks
    worker._lod_prefill_attention_shapes = shapes


def _install_leaf_capture(worker, capture_path):
    """Save one real late-prefill leaf input, outside canonical measurements.

    This is a diagnostic-only instance hook: no capture branch is added to
    production attention. A small replay can then tune kernels on trained
    queries and real centroid populations rather than dummy fixture routing.
    """
    import os
    from pathlib import Path
    from types import MethodType
    import torch

    runner = worker.model_runner
    runtime = getattr(runner, "_vllm_lod_runtime", None)
    if runtime is None:
        runtime = getattr(getattr(runner, "model_state", None),
                          "_vllm_lod_runtime", None)
    if runtime is None or runtime.dcp_rank != 0:
        return
    minimum_leaves = int(os.environ.get("LOD_KIMI_CAPTURE_MIN_LEAVES", "32768"))
    pool = next(iter(runtime.pools.values()))
    engine = pool.engine
    original = engine._paged_leaf_attention
    captured = False

    def capture(self, q, top_slots, cache, *, active_slots, reduce_routes=True):
        nonlocal captured
        leaf_count = int(cache.get("leaf_count", cache["leaf_k"].size(2)))
        expanded = getattr(self, "_lod_kimi_expanded_prefill_chunk", None)
        if (not captured and leaf_count >= minimum_leaves
                and isinstance(expanded, torch.Tensor) and int(q.size(2)) > 1):
            captured = True
            payload = {
                "scope": "real trained Kimi K3 late-prefill leaf inputs",
                "q": expanded.detach().cpu().contiguous(),
                "w_uk_t": self._lod_kimi_w_uk_t.detach().cpu().contiguous(),
                "w_uv": self._lod_kimi_w_uv.detach().cpu().contiguous(),
                "top_slots": top_slots.detach().cpu().contiguous(),
                "active_slots": active_slots,
                "scale": self.scaling,
                "hash_probes": self._page_lookup_probes(cache),
                "reduce_routes": reduce_routes,
                "leaf_count": leaf_count,
                "cache": {},
            }
            for key in ("slot_pages", "slot_lengths", "page_indices",
                        "overflow_page_keys", "overflow_page_values",
                        "overflow_used", "leaf_k", "leaf_v"):
                value = cache[key]
                if key in ("leaf_k", "leaf_v"):
                    value = value[..., :leaf_count, :]
                payload["cache"][key] = value.detach().cpu().contiguous()
            path = Path(capture_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            print(f"KIMI_PREFILL_CAPTURE path={path} leaves={leaf_count} "
                  f"slots={active_slots} q={tuple(expanded.shape)}", flush=True)
        return original(q, top_slots, cache, active_slots=active_slots,
                        reduce_routes=reduce_routes)

    worker._lod_prefill_capture_hooks = [(engine, original)]
    engine._paged_leaf_attention = MethodType(capture, engine)

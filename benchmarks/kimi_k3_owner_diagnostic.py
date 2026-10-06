"""Profile one warmed request-owner model chunk, not a serving benchmark.

Scopes are installed only by worker RPC for a separate diagnostic generation.
GPU activities are assigned to the innermost scope through Kineto correlation
IDs. Durations are not subtracted from canonical end-to-end measurements.
"""

from __future__ import annotations

from functools import wraps
from types import MethodType


def linear_replication_factor(module):
    """Infer actual weight partitioning, not a possibly vestigial tp_size."""
    factors = [1]
    for axis in ("input", "output"):
        full = getattr(module, axis + "_size", None)
        part = getattr(module, axis + "_size_per_partition", None)
        if full is not None and part is not None and part > 0:
            if full % part:
                raise ValueError(f"non-integral {axis} partition: {full}/{part}")
            factors.append(full // part)
    return max(factors)


def interval_union_us(intervals):
    end = float("-inf")
    total = 0.0
    for begin, stop in sorted(intervals):
        total += max(0.0, stop - max(begin, end))
        end = max(end, stop)
    return total


def summarize_profile(events, cpu_device, gpu_device):
    cpu = {event.id: event for event in events if event.device_type == cpu_device}
    groups, intervals = {}, []
    for event in events:
        if event.device_type != gpu_device:
            continue
        # ROCTracer also emits GPU-side user-annotation ranges. These are
        # enclosing scopes, not kernels; counting them would double-count
        # work and artificially erase idle gaps in the activity union.
        if getattr(event, "is_user_annotation", False) or event.name.startswith("K3/"):
            continue
        parent = cpu.get(event.linked_correlation_id)
        while parent is not None and not parent.name.startswith("K3/"):
            parent = parent.cpu_parent
        name = parent.name if parent is not None else "unattributed"
        group = groups.setdefault(name, {"calls": 0, "sum_us": 0.0, "intervals": []})
        interval = (event.time_range.start, event.time_range.end)
        group["calls"] += 1
        group["sum_us"] += event.device_time_total
        group["intervals"].append(interval)
        intervals.append(interval)
    span = (max(stop for _, stop in intervals) - min(begin for begin, _ in intervals)
            if intervals else 0.0)
    return {
        "gpu_timeline_span_ms": span / 1000,
        "gpu_activity_union_ms": interval_union_us(intervals) / 1000,
        "stages": {name: {
            "gpu_activity_count": group["calls"],
            "summed_gpu_ms_not_wall_ms": group["sum_us"] / 1000,
            "gpu_activity_union_ms": interval_union_us(group["intervals"]) / 1000,
        } for name, group in groups.items()},
    }


def arm_owner_diagnostic(worker, previous, trace_path=None):
    """Install temporary scopes; profile a single model forward on rank zero."""
    import torch
    import vllm.models.kimi_k3.amd.linear as native
    from vllm_lod_plugin.models import kimi_k3_request_prefill as owner

    if getattr(worker, "_kimi_owner_diagnostic", None) is not None:
        raise RuntimeError("owner diagnostic already armed")
    runner = worker.model_runner
    runtime = runner.model_state._vllm_lod_runtime
    root = runner.model
    core = next(module for module in root.modules()
                if type(module).__name__ == "KimiLinearModel")
    state = {"restores": [], "profile": None, "previous": previous,
             "captured": False, "chunk": None, "trace_path": trace_path}
    # Metadata-only lower-bound estimate for replicating non-expert TP
    # parameters. It excludes new communication buffers and allocator overhead.
    weight_groups = {}
    for path, module in root.named_modules():
        types = {kind.__name__ for kind in type(module).__mro__}
        multiplier = linear_replication_factor(module)
        for name, param in module.named_parameters(recurse=False):
            if ".experts" in path and "shared_experts" not in path:
                category, extra = "routed_expert_or_transform", 0
            elif types & {"VocabParallelEmbedding", "ParallelLMHead"}:
                category, extra = "embedding_or_head_kept_tp", 0
            else:
                category = "shared_experts" if "shared_experts" in path else "other_nonexpert"
                # Biases and nonlinear parameter shards are deliberately
                # excluded from this linear-weight lower bound.
                extra = (param.numel() * param.element_size() * (multiplier - 1)
                         if name == "weight" else 0)
            group = weight_groups.setdefault(category, {"local_parameter_bytes": 0,
                                                        "tp1_additional_parameter_bytes_estimate": 0})
            group["local_parameter_bytes"] += param.numel() * param.element_size()
            group["tp1_additional_parameter_bytes_estimate"] += extra
    state["weight_layout"] = weight_groups
    worker._kimi_owner_diagnostic = state

    def scope(target, attr, label):
        original = getattr(target, attr)
        @wraps(original)
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function("K3/" + label):
                return original(*args, **kwargs)
        state["restores"].append((target, attr, original))
        setattr(target, attr, wrapped)

    # Exclusive innermost stage labels distinguish owner transport/refinement
    # from native TP projection, recurrent attention, MoE and residual mixing.
    for attr, label in (("exchange_queries", "query_transport"),
                        ("exchange_outputs", "output_transport"),
                        ("advance_block", "state_update"),
                        ("attend_slice", "owner_attention_other")):
        scope(owner, attr, label)
    scope(native, "_apply_attn_res", "residual_mix_and_gather")
    for module in root.modules():
        name = type(module).__name__
        if name == "KimiMoE":
            scope(module, "forward", "moe")
        elif name == "KimiDecoderLayer":
            kind = "mla_projection_and_dispatch" if type(module.self_attn).__name__ == "KimiMLAAttention" else "recurrent_attention"
            scope(module, "_run_self_attn", kind)
    for pool in runtime.pools.values():
        for attr, label in (("_prefill_local_attention", "local_attention"),
                            ("_two_level_attention", "remote_attention"),
                            ("build_cache_from_bf16", "initial_state_construction")):
            scope(pool.engine, attr, label)

    original = core.forward
    state["restores"].append((core, "forward", original))
    def forward(self, input_ids, positions, intermediate_tensors, *args, **kwargs):
        # Synchronization and profiler overhead belong only to this diagnostic.
        capture = (worker.rank == 0 and not state["captured"]
                   and positions.numel() > 1 and int(positions.min()) >= previous)
        if not capture:
            return original(input_ids, positions, intermediate_tensors, *args, **kwargs)
        state["captured"] = True
        state["chunk"] = {"global_start": int(positions.min()),
                          "global_end_exclusive": int(positions.max()) + 1,
                          "total_tokens": positions.numel()}
        torch.cuda.synchronize()
        profiler = torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        state["profile"] = profiler
        with profiler:
            with torch.profiler.record_function("K3/model_other"):
                output = original(input_ids, positions, intermediate_tensors, *args, **kwargs)
            torch.cuda.synchronize()
        return output
    core.forward = MethodType(forward, core)
    return {"rank": worker.rank, "target_previous": previous}


def finish_owner_diagnostic(worker):
    import torch

    state = worker._kimi_owner_diagnostic
    for target, attr, original in reversed(state["restores"]):
        setattr(target, attr, original)
    del worker._kimi_owner_diagnostic
    profile = state["profile"]
    if worker.rank == 0 and profile is None:
        raise RuntimeError("diagnostic did not reach the requested global chunk")
    result = {"rank": worker.rank, "chunk": state["chunk"],
              "scope": "one rank, one instrumented model chunk; not serving latency",
              "weight_layout_estimate": state["weight_layout"]}
    if profile is not None:
        if state["trace_path"]:
            profile.export_chrome_trace(state["trace_path"])
            result["trace_path"] = state["trace_path"]
        result.update(summarize_profile(profile.events(),
                      torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA))
        events = profile.key_averages()
        result["kernels"] = sorted([
            {"name": event.key, "calls": event.count,
             "summed_gpu_ms_not_wall_ms": event.device_time_total / 1000}
            for event in events if event.device_type == torch.autograd.DeviceType.CUDA
            and not event.is_user_annotation and not event.key.startswith("K3/")
        ], key=lambda item: item["summed_gpu_ms_not_wall_ms"], reverse=True)[:40]
    return result

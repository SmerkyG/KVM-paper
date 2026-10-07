"""Measure matched K3 prefill and fixed-trace decode on the vLLM path."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from benchmarks._vllm import close_llm, llm_kwargs
from benchmarks._decode_update_audit import decode_update_deltas, read_decode_update_counters


# Spawned EngineCore workers import this module as well. A development-only
# watchdog locates a stuck startup without ptrace access to the child.
if os.environ.get("LOD_BENCHMARK_TRACE_TIMEOUT"):
    import faulthandler

    faulthandler.dump_traceback_later(
        float(os.environ["LOD_BENCHMARK_TRACE_TIMEOUT"]), repeat=True,
    )


def reference_trace_prefix(tokens, record, *, length: int, output_tokens: int):
    """Verify the archive; permit its 1025-output panel's canonical extra step.

    Old dense panels contain 1025 outputs/1024 decode steps. The current
    four-update LoD protocol needs 1026 outputs. Its extra token must come
    from the same frozen ProLong stream, after checking the whole old trace.
    Arbitrary trace extension remains an error.
    """
    from benchmarks.prolong import token_digest

    count = int(record["trace_tokens"])
    if not (1 <= output_tokens <= count or (count == 1025 and output_tokens == 1026)):
        raise ValueError("requested continuation exceeds the archived decode trace")
    trace = tokens[length:length + count]
    if len(trace) != count or token_digest(trace) != record["trace_token_sha256"]:
        raise RuntimeError("archived decode trace mismatch")
    extended = tokens[length:length + output_tokens]
    if len(extended) != output_tokens:
        raise ValueError("requested continuation exceeds the frozen source stream")
    return extended


def timed_sweep_generate(llm, prompts, params, *, synchronized_decode: bool):
    """Use the canonical ProLong timer, including its request-validity checks."""
    from benchmarks.prolong import timed_generate

    measurement = timed_generate(llm, prompts, params)
    timing = measurement[-1]
    if synchronized_decode and len(prompts) > 1 and (
        timing["last_token_spread_seconds"] != 0
        or timing["all_requests_live_overlap_seconds"] != timing["decode_window_seconds"]
    ):
        # First samples come from serial prefills, before the decode barrier.
        # Only subsequent decode must run as one live, aligned cohort.
        raise RuntimeError("requests did not remain a synchronized decode cohort")
    return measurement


def owner_prefill_layout(batch_size: int) -> tuple[int, int]:
    """Size scheduler slices independently of the global 16K LoD cadence."""
    row_chunk = int(os.getenv("LOD_KIMI_OWNER_QUERY_CHUNK", "16384"))
    if row_chunk < 1 or row_chunk > 16384 or 16384 % row_chunk:
        raise ValueError("owner query chunk must be a positive divisor of 16384")
    # Keep at least one normal 16K model batch, including the decode reserve.
    # Smaller row slices do not shrink logical attention/update blocks.
    return row_chunk, max(16384, row_chunk * batch_size) + batch_size


def configure_owner_prefill_environment(batch_size: int) -> tuple[int, int]:
    """Configure attention slices without overwriting a bounded MoE choice."""
    row_chunk, total_budget = owner_prefill_layout(batch_size)
    os.environ.update(LOD_KIMI_REQUEST_OWNER_PREFILL="1",
        LOD_KIMI_OWNER_QUERY_CHUNK=str(row_chunk),
        LOD_BENCHMARK_PREFILL_COHORT=str(batch_size), LOD_KIMI_OWNER_SHARD_RESIDUAL="0")
    os.environ.setdefault("LOD_KIMI_OWNER_MOE_CHUNK", str(total_budget))
    return row_chunk, total_budget


def reset_peak_memory(worker):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_memory(worker):
    import torch

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    result = {
        "rank": worker.rank,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "current_torch_allocated_bytes": torch.cuda.memory_allocated(),
        "device_free_bytes_after_generation": free,
        "device_total_bytes": total,
    }
    from benchmarks._kimi_prefill_memory import snapshot_prefill_memory
    result["lod_storage_inventory"] = snapshot_prefill_memory(worker)
    result["centroid_leaf_stats"] = getattr(worker, "_kimi_warmup_leaf_stats", {})
    result["centroid_leaf_stats_scope"] = "untimed warmup, before request cleanup; absent if not armed"
    return result


def memory_runtime(worker):
    runner = worker.model_runner
    runtime = getattr(getattr(runner, "model_state", None), "_vllm_lod_runtime", None)
    return runtime if runtime is not None else getattr(runner, "_vllm_lod_runtime", None)


def centroid_leaf_stats(worker):
    """Untimed capacity diagnostic; do not discard any leaves here.

    K3 counts are physical member counts (unit merge weights). Global counts
    decide closure; local directory lengths decide reclaimable shard bytes.
    A sealed centroid may still contribute its complete coarse sum and mass.
    """
    runtime = memory_runtime(worker)
    records = {}
    for name, parent in getattr(runtime, "pools", {}).items():
        pool = getattr(parent, "owner_decode_pool", None) or parent
        record = pool_leaf_stats(pool)
        if record is not None:
            records[name] = record
    return records


def pool_leaf_stats(pool):
    import torch

    if not getattr(pool, "is_absorbed_mla", False):
        return None
    page = pool.state.get("page_cache", {})
    lengths, counts = page.get("slot_lengths"), pool.state.get("counts")
    cap = pool.engine.max_open_centroid_leaves
    if not isinstance(lengths, torch.Tensor) or not isinstance(counts, torch.Tensor) or cap is None:
        return None
    local = lengths.detach().cpu().to(torch.int64)
    global_counts = counts.detach().cpu().squeeze(-1)
    closed = global_counts > cap
    return dict(cap=int(cap), nonempty_centroids=int((global_counts > 0).sum()),
        largest_global_centroid=float(global_counts.max()),
        global_member_count=float(global_counts.sum()),
        closed_centroids=int(closed.sum()), local_leaf_count=int(local.sum()),
        local_leaves_in_closed_centroids=int(local[closed].sum()),
        note="capacity opportunity only; current archive remains intact")


def arm_warmup_leaf_stats(worker):
    """Observe cleanup only in untimed warmup; remove all hooks before timing."""
    runtime = memory_runtime(worker)
    worker._kimi_warmup_leaf_stats = {}
    worker._kimi_warmup_leaf_stats_hooks = []
    for name, parent in getattr(runtime, "pools", {}).items():
        pool = getattr(parent, "owner_decode_pool", None) or parent
        if not getattr(pool, "is_absorbed_mla", False):
            continue
        for attribute in ("reset", "_reset_range"):
            # The scheduler resets the parent range, not necessarily the
            # child owner's reset method. Sample its semantic child before
            # that parent lifecycle operation clears or overwrites the row.
            original = getattr(parent, attribute)
            had_override = attribute in vars(parent)

            def observe(*args, _pool=pool, _name=name, _original=original, **kwargs):
                record = pool_leaf_stats(_pool)
                previous = worker._kimi_warmup_leaf_stats.get(_name, {})
                if record and record["global_member_count"] > previous.get("global_member_count", 0):
                    worker._kimi_warmup_leaf_stats[_name] = record
                return _original(*args, **kwargs)

            setattr(parent, attribute, observe)
            worker._kimi_warmup_leaf_stats_hooks.append((parent, attribute, original, had_override))


def finish_warmup_leaf_stats(worker):
    for pool, attribute, original, had_override in worker._kimi_warmup_leaf_stats_hooks:
        if had_override:
            setattr(pool, attribute, original)
        else:
            delattr(pool, attribute)
    worker._kimi_warmup_leaf_stats_hooks = []
    return worker._kimi_warmup_leaf_stats


def owner_prefill_audit(worker):
    """Report actual query sizes received by this rank, without GPU timers."""
    runtime = worker.model_runner.model_state._vllm_lod_runtime
    sizes = {}
    for pool in runtime.pools.values():
        for length, count in getattr(pool, "_kimi_owner_query_sizes", {}).items():
            sizes[length] = sizes.get(length, 0) + count
    pool = next(iter(runtime.pools.values()), None)
    group = getattr(pool, "dcp_group", None)
    buffers = getattr(group, "_lod_owner_transport", {})
    from vllm_lod_plugin.prefill_allocator import _PREFILL_ALLOCATOR_AUDIT
    return {
        "rank": worker.rank, "owned_query_sizes": sizes,
        "head_owner_ranges": sorted({getattr(pool, "_kimi_head_owner_range")
            for pool in runtime.pools.values() if hasattr(pool, "_kimi_head_owner_range")}),
        "head_owner_last_states": {name: getattr(pool, "_kimi_head_owner_last_state")
            for name, pool in runtime.pools.items() if hasattr(pool, "_kimi_head_owner_last_state")},
        "transport_arena_allocations": getattr(group, "_lod_owner_transport_allocations", 0),
        "transport_arena_bytes": sum(t.numel() * t.element_size() for t in buffers.values()),
        "allocator_pressure_audit": dict(_PREFILL_ALLOCATOR_AUDIT),
        "max_requests_per_prefill_call": max(
            (getattr(pool, "_kimi_owner_max_plan_rows", 0) for pool in runtime.pools.values()),
            default=0),
        "latest_full_cohort_previous": max(
            (getattr(pool, "_kimi_owner_full_cohort_previous", 0) for pool in runtime.pools.values()),
            default=0),
    }


def owner_decode_counters(worker):
    runtime = worker.model_runner.model_state._vllm_lod_runtime
    return {name: {
        "updates": int(getattr(pool, "_kimi_owner_decode_updates", 0)),
        "tokens": int(getattr(pool, "_kimi_owner_decode_tokens", 0)),
    } for name, pool in runtime.pools.items()}


def owner_decode_graph_audit(worker):
    """Read actual v2 graph handles and fixed-owner state outside timing."""
    runner = worker.model_runner
    runtime = runner.model_state._vllm_lod_runtime
    manager = runner.cudagraph_manager
    descriptors = [dict(num_tokens=int(d.num_tokens), graph_instantiated=graph is not None)
                   for d, graph in manager.graphs.items()]
    pools = list(runtime.pools.values())
    return dict(rank=worker.rank, captured_graphs=descriptors,
        owner_layer_count=sum(getattr(p,"kimi_captured_owner_decode",False) for p in pools),
        local_decode_world_sizes=[p.owner_decode_pool.dcp_world_size for p in pools],
        local_decode_heads=[p.owner_decode_pool.query_heads for p in pools],
        active_global_cadences=[p.owner_decode_pool.engine.decode_state_update_len for p in pools],
        projected_output_exchange="one reduce-scatter; no distributed route/LSE merge")


def owner_decode_graph_replays(worker):
    """Count actual model-graph replays, not merely allocated graph handles.

    This host-only audit adds no events, synchronization, or GPU work. Its
    wrapper is installed before warmup; counter snapshots are outside timing.
    """
    manager = worker.model_runner.cudagraph_manager
    if not hasattr(manager, "_lod_owner_graph_replays"):
        manager._lod_owner_graph_replays = 0
        original = manager.run_fullgraph

        def replay(descriptor):
            result = original(descriptor)
            if descriptor.num_tokens == 8:
                manager._lod_owner_graph_replays += 1
            return result

        manager.run_fullgraph = replay
    return int(manager._lod_owner_graph_replays)


def validate_owner_decode_counts(before, after, *, steps: int, world_size: int):
    """Every rank owns one row; four updates are per row, not across B8."""
    if len(before) != world_size or len(after) != world_size:
        raise RuntimeError("owner decode did not audit all workers")
    deltas = []
    for initial, final in zip(before, after, strict=True):
        if not initial or initial.keys() != final.keys():
            raise RuntimeError("owner decode layer set changed")
        delta = {name: {key: values[key] - initial[name][key] for key in values}
                 for name, values in final.items()}
        if any(values != {"updates": (steps - 1) // 256, "tokens": steps}
               for values in delta.values()):
            raise RuntimeError(f"wrong owner decode work/cadence: {delta}")
        deltas.append(delta)
    return deltas


def validate_owner_prefill_audits(audits, *, row_chunk: int, world_size: int,
                                batch_size: int | None = None,
                                head_owners: bool = False,
                                length: int | None = None) -> set[int]:
    """Audit every TP rank and require the actual number of attention owners."""
    owners = 6 if head_owners else world_size if batch_size is None else batch_size
    active = [audit for audit in audits if (
        audit["owned_query_sizes"].get(row_chunk, 0)
        or audit["owned_query_sizes"].get(str(row_chunk), 0)
    )]
    if (not 1 <= owners <= world_size
            or len(audits) != world_size
            or {audit["rank"] for audit in audits} != set(range(world_size))
            or len(active) != owners):
        raise RuntimeError("not every attention owner received its configured full query block")
    if head_owners:
        if world_size != 8 or batch_size != 1 or {a["rank"] for a in active} != set(range(6)):
            raise RuntimeError("six head owners did not cover all 96 heads")
        for audit in active:
            if audit.get("head_owner_ranges") not in (
                [(audit["rank"] * 16, (audit["rank"] + 1) * 16)],
                [[audit["rank"] * 16, (audit["rank"] + 1) * 16]],
            ):
                raise RuntimeError("head-owner audit has an incorrect head range")
            if length is not None:
                states = audit.get("head_owner_last_states", {})
                if not states or any(
                    state["total_len"] != length or state["coverage"] != length - 256
                    or state["prefill_update_len"] != 16384 or state["query_heads"] != 16
                    for state in states.values()
                ):
                    raise RuntimeError("head-owner history/cadence audit is incomplete")
        if length is not None and any(
            audit["head_owner_last_states"] != active[0]["head_owner_last_states"]
            for audit in active[1:]
        ):
            raise RuntimeError("head owners built different global centroid schedules")
    return {audit["rank"] for audit in active}


def report_phase(result: dict, save, *, length: int, phase: str, **details) -> None:
    """Publish progress only outside generation/timing windows."""
    entry = {"length": length, "phase": phase, "timestamp_unix": time.time(), **details}
    result["current_phase"] = entry
    result.setdefault("phase_history", []).append(entry)
    save()
    print("KIMI_PREFILL_PHASE " + json.dumps(entry), flush=True)


def ranked_attention_audit(worker):
    import torch
    from benchmarks.prolong import audit_worker_attention_mode

    runner = worker.model_runner
    runtime = getattr(runner, "_vllm_lod_runtime", None)
    if runtime is None:
        runtime = getattr(getattr(runner, "model_state", None), "_vllm_lod_runtime", None)
    from lod_attention.kernels import paged_decode
    from lod_attention.kernels.lod_kernels import _materialized_score_output
    from vllm_lod_plugin.models.kimi_k3_request_prefill import attend_slice
    from lod_attention.kernels._coarse_route_views import coarse_route_mean_view
    from lod_attention.kernels.kimi_gluon_decode import kimi_lod_decode_splits
    live_split_geometry = "consumer_splits" in paged_decode.fused_decode_paged_lod_attention.__code__.co_varnames
    decode_geometry = {}
    if runtime:
        for name, original_pool in runtime.pools.items():
            pool = getattr(original_pool, "owner_decode_pool", original_pool)
            state = getattr(pool, "state", None) or {}
            page = state.get("page_cache", {})
            means = None
            if isinstance(state.get("state_k"), torch.Tensor) and isinstance(
                page.get("unified_page1_k"), torch.Tensor
            ):
                means = coarse_route_mean_view(
                    state["state_k"], page["unified_page1_k"],
                    int(page["unified_page1_coarse_offset"]), pool.kv_heads,
                )
            decode_geometry[name] = dict(
                rank_major_owned_routes="distributed_global_topk_into" in
                    paged_decode.fused_decode_paged_lod_attention.__code__.co_names,
                splits_by_live_batch={str(rows): (
                    kimi_lod_decode_splits(rows, head_tiled_metadata=True)
                    if live_split_geometry else 64)
                    for rows in getattr(pool, "dcp_decode_buffers", {})},
                cached_centroid_mean_routing=(means is not None and
                    "coarse_route_mean_view" in
                    paged_decode.fused_decode_paged_lod_attention.__code__.co_names),
            )
    owner_slice = getattr(attend_slice, "__wrapped__", attend_slice)
    return {**audit_worker_attention_mode(worker), "rank": worker.rank,
            "decode_geometry": decode_geometry,
            "owner_shared_construction_scope": "owner_state_workspace" in owner_slice.__code__.co_names,
            "update_score_workspace_follows_overflow": "score_tokens" in _materialized_score_output.__code__.co_varnames,
            "compact_selected_projection_enabled": os.environ.get("LOD_KIMI_COMPACT_SELECTED_PROJECTION") == "1",
            "compact_projection_calls": {
                name: getattr(pool.engine, "_lod_kimi_compact_projection_calls", 0)
                for name, pool in runtime.pools.items()
            } if runtime else {}}


def audit_loaded_attention(llm, *, mode: str, length: int,
                           required_ranks: set[int] | None = None):
    """Audit a completed point outside timing, before trying a larger shape."""
    audits = llm.collective_rpc(ranked_attention_audit)
    if required_ranks is not None and not required_ranks.issubset(
        {audit["rank"] for audit in audits}
    ):
        raise RuntimeError("missing active attention owner's kernel audit")
    if mode == "two-tier" and length > 16_384:
        expected = {
            "CK_TILE_FMHA_ROUTE_QUERY_NORMALIZE": "0",
            "CK_TILE_FMHA_ROUTE_TOPK": "8",
            "CK_TILE_FMHA_ROUTE_GLOBAL_TOPK": "0",
            "CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY": "0",
        }
        for audit in audits:
            # B1 has one attention owner and seven projection/MoE workers.
            # Keep every worker's report, but require routing binaries only
            # on ranks independently verified to execute owner attention.
            if required_ranks is not None and audit["rank"] not in required_ranks:
                continue
            if os.environ.get("LOD_KIMI_COMPACT_SELECTED_PROJECTION") == "1" and (
                not audit.get("compact_selected_projection_enabled")
                or not audit.get("compact_projection_calls")
                or any(count < 1 for count in audit["compact_projection_calls"].values())
            ):
                raise RuntimeError("measured worker did not execute compact selected-leaf projection")
            fused = [module for module in audit["loaded_kimi_lod_modules"]
                     if "_asyncbias_" in module["module"]]
            if not fused or any(module["route_build_flags"] != (
                expected | {"CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY":
                            "1" if module["module"].endswith("_v13") else "0"}
            ) for module in fused):
                raise RuntimeError(
                    f"measured Kimi worker {audit['rank']} did not load correct top-eight route/coarse"
                )
    return audits


def validate_expert_scatter(worker):
    """Diagnostic-only bounds check before route scatter; never a timing path."""
    import torch
    import lod_attention.kernels.paged_prefill as module

    original = module._scatter_expert_routes_kernel

    class CheckedScatter:
        def __getitem__(self, grid):
            launch = original[grid]

            def checked(*args, **options):
                slots, starts, head_offsets, offsets, packed, blocks, experts = args[:7]
                items_per_head, active_slots = args[7:9]
                flat = slots.reshape(-1)
                valid = flat.ge(0)
                if bool((flat[valid] >= active_slots).any()):
                    raise RuntimeError(
                        f"route out of range: active={active_slots}, "
                        f"max={int(flat.max())}, shape={tuple(slots.shape)}"
                    )
                rows = torch.arange(flat.numel(), device=slots.device)[valid]
                selected = flat[valid].long()
                head = rows // items_per_head
                query_heads = options["QUERY_HEADS"]
                expert = ((head // query_heads) * options["KV_HEADS"]
                          + (head % query_heads) // options["KV_GROUP_SIZE"])
                expert = expert * active_slots + selected
                local = (head_offsets[head * active_slots + selected].long()
                         + offsets.reshape(-1)[valid].long())
                expected_counts = torch.bincount(expert, minlength=starts.numel())
                expected_starts = expected_counts.cumsum(0) - expected_counts
                if not torch.equal(starts.long(), expected_starts):
                    path = Path("results/kimi-k3-mla-stack") / f"route-scatter-rank-{worker.rank}.pt"
                    torch.save({
                        "slots": slots.cpu(), "starts": starts.cpu(),
                        "head_offsets": head_offsets.cpu(), "offsets": offsets.cpu(),
                        "active_slots": active_slots, "options": options,
                    }, path)
                    bad = starts.long().ne(expected_starts).nonzero()[0].item()
                    raise RuntimeError(
                        f"route count prefix mismatch at expert {bad}: "
                        f"actual={int(starts[bad])}, expected={int(expected_starts[bad])}; "
                        f"captured {path}"
                    )
                destination = starts[expert].long() + local
                if bool(((destination < 0) | (destination >= packed.numel())).any()):
                    raise RuntimeError(
                        f"route scatter destination overflow: min={int(destination.min())}, "
                        f"max={int(destination.max())}, capacity={packed.numel()}, "
                        f"active={active_slots}, shape={tuple(slots.shape)}, "
                        f"local_max={int(local.max())}, start_max={int(starts.max())}"
                    )
                first = local.remainder(options["BLOCK_M"]).eq(0)
                block_dest = (blocks[expert[first]].long()
                              + local[first] // options["BLOCK_M"])
                if bool(((block_dest < 0) | (block_dest >= experts.numel())).any()):
                    raise RuntimeError("route block scatter destination overflow")
                return launch(*args, **options)

            return checked

    module._scatter_expert_routes_kernel = CheckedScatter()
    return {"rank": worker.rank, "scope": "synchronizing scatter bounds diagnostic"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("full", "two-tier"), required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="match a control's context capacity without rerunning its entire sweep")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--reference-decode-trace", action="store_true",
                        help="replay and verify the archived dense ProLong continuation")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--decode-context-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile-length", type=int, default=None,
                        help="extra diagnostic pass after uninstrumented timing")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="warm once, then profile one model chunk; no serving timing")
    parser.add_argument("--report-memory", action="store_true",
                        help="read worker peak memory outside timed generation")
    parser.add_argument("--allocation-audit-only", action="store_true",
                        help="inspect reserved backing storage; do not run or report serving timing")
    parser.add_argument("--kda-prefill", choices=("current", "gluon_paged"),
                        default="gluon_paged",
                        help="use the same approved G8 KDA prefill baseline in dense and LoD")
    owner_group = parser.add_mutually_exclusive_group()
    owner_group.add_argument("--ordinary-dcp", action="store_true",
                             help="explicitly disable the default B8 request-owner layout")
    owner_group.add_argument("--owner-local-mla", action="store_true",
                             help="private B8 prefill experiment with local MLA Q/K/V/O")
    owner_group.add_argument("--owner-tp-mla", action="store_true",
                             help="same row-owned attention but retain native TP MLA projections")
    owner_group.add_argument("--head-owner-mla", action="store_true",
                             help="B1 prefill: six 16-head owners, with native TP8 projections")
    parser.add_argument("--audit-prefill-batches", action="store_true",
                        help="inspect actual row chunks during warmup, remove hooks before timing")
    parser.add_argument("--capacity-only", action="store_true",
                        help="one audited prefill pass; report fit, never a warm serving timing")
    parser.add_argument("--validate-route-scatter", action="store_true",
                        help="synchronizing bounds diagnostic; results are not serving timings")
    parser.add_argument("--weight-cache-id", default=None,
                        help="resident trained INT4-MoE K3 cache; otherwise use fixture dummy weights")
    parser.add_argument("--real-token-cache", type=Path, default=None,
                        help="frozen ProLong token cache instead of synthetic fixture tokens")
    parser.add_argument("--reference-baselines", type=Path, nargs="+", default=(),
                        help="reuse archived ProLong prompt cohorts and verify their hashes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmarks._vllm import configure_kimi_layout

    if args.ordinary_dcp:
        os.environ["LOD_KIMI_REQUEST_OWNER_PREFILL"] = "0"
        os.environ["LOD_KIMI_REQUEST_OWNER_DECODE"] = "0"
    elif not (args.owner_local_mla or args.owner_tp_mla or args.head_owner_mla):
        args.owner_tp_mla = configure_kimi_layout(
            checkpoint=args.checkpoint, mode=args.mode, batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            decode_context_parallel_size=args.decode_context_parallel_size,
            max_model_len=args.max_model_len or max(args.lengths) + args.decode_tokens + 8,
        )
    if min(*args.lengths, args.batch_size, args.decode_tokens, args.repeats) < 1:
        raise ValueError("lengths, token counts, batch size, and repeats must be positive")
    if args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes <= 0:
        raise ValueError("kv cache memory bytes must be positive")
    if args.max_model_len is not None and args.max_model_len < max(args.lengths) + args.decode_tokens + 8:
        parser.error("max-model-len must include the measured prompts and output/headroom")
    if args.profile_length is not None and args.profile_length not in args.lengths:
        raise ValueError("profile length must be one of the warmed measured lengths")
    if args.capacity_only and (
        args.decode_tokens not in (1, 2) or args.profile_length is not None
        or (args.decode_tokens == 2 and not args.reference_decode_trace)
    ):
        parser.error("capacity-only requires one output, or two archived trace outputs, and no extra profiling")
    if args.diagnostic_only and (args.capacity_only or args.profile_length is not None
                                or len(args.lengths) != 1 or args.decode_tokens != 1
                                or os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") != "1"):
        parser.error("diagnostic-only requires one owner-prefill length, max_tokens=1, and no other profile mode")
    if args.owner_local_mla or args.owner_tp_mla:
        if (args.mode != "two-tier" or args.batch_size not in (1, 8)
                or (args.owner_local_mla and args.batch_size != 8)
                or (args.owner_local_mla and args.decode_tokens != 1)
                or (args.batch_size == 1 and args.decode_tokens != 1)
                or args.tensor_parallel_size != 8 or args.decode_context_parallel_size != 8
                or args.real_token_cache is None):
            parser.error("owner MLA requires trained TP8/DCP8: native TP B1/B8 prefill, local projections B8")
        row_chunk, total_budget = configure_owner_prefill_environment(args.batch_size)
        if args.owner_tp_mla:
            # Like the local-projection experiment's shared output arena,
            # retain fixed-shape head transport buffers through steady prefill.
            os.environ["LOD_KIMI_OWNER_REUSE_TRANSPORT"] = "1"
            if args.decode_tokens > 1:
                os.environ["LOD_KIMI_REQUEST_OWNER_DECODE"] = "1"
    if args.head_owner_mla:
        if (args.mode != "two-tier" or args.batch_size != 1 or args.decode_tokens != 1
                or args.tensor_parallel_size != 8 or args.decode_context_parallel_size != 8):
            parser.error("six head owners require B1 TP8/DCP8 two-tier prefill only")
        os.environ.update(LOD_KIMI_REQUEST_OWNER_PREFILL="1", LOD_KIMI_HEAD_OWNER_PREFILL="1",
            LOD_KIMI_OWNER_QUERY_CHUNK="16384", LOD_BENCHMARK_PREFILL_COHORT="1",
            LOD_KIMI_OWNER_MOE_CHUNK="16385", LOD_KIMI_OWNER_SHARD_RESIDUAL="0")
        if os.getenv("LOD_KIMI_REQUEST_OWNER_DECODE") == "1":
            parser.error("six head owners do not implement owner decode")
    os.environ["LOD_BENCHMARK_SYNC_PREFILL_CACHE"] = "1"
    # The local worker audit uses the same callable RPC as the ProLong runner.
    # Newer vLLM disables callable serialization unless explicitly enabled.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM, SamplingParams

    if args.reference_decode_trace:
        if args.decode_tokens <= 1 or not args.reference_baselines:
            parser.error("reference decode requires an archived baseline and multiple output tokens")
        from benchmarks.prolong import configure_synchronized_decode_environment
        configure_synchronized_decode_environment(enabled=True, batch_size=args.batch_size)

    maximum = max(args.lengths)
    kwargs = llm_kwargs(
        checkpoint=args.checkpoint,
        mode=args.mode,
        max_model_len=args.max_model_len or maximum + args.decode_tokens + 8,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        decode_context_parallel_size=args.decode_context_parallel_size,
        dcp_comm_backend="ag_rs",
        gpu_memory_utilization=args.gpu_memory_utilization,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    kwargs.update(load_format="dummy", skip_tokenizer_init=True)
    if args.owner_local_mla or args.owner_tp_mla or args.head_owner_mla:
        # Keep the ordinary control's AITER/custom-op selection. The legacy
        # owner flag forces enforce_eager (which also changes norm/IR kernels),
        # but this private experiment runs only uncaptured prefill and never
        # uses the startup decode graphs. Its hooks are installed after startup.
        kwargs["enforce_eager"] = False
    if args.reference_decode_trace:
        kwargs["enable_trace_replay"] = True
    if args.weight_cache_id:
        if args.real_token_cache is None:
            parser.error("trained speed tests require frozen real ProLong tokens")
        kwargs.update(
            load_format="ipc_cache", enable_expert_parallel=True,
            disable_custom_all_reduce=False,
            quantization_config={"moe": {"weight": "int4_per_group_32"}},
            model_loader_extra_config={
                "auto_start": True, "cache_id": args.weight_cache_id,
                "backing_load_format": "auto", "broker_timeout": 1800.0,
            },
        )
    if args.kv_cache_memory_bytes is not None:
        # Explicit native-cache sizing leaves room for several unfinished
        # request-owned LoD prefill shadows without changing the algorithm.
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_tokens,
        seed=1234,
        ignore_eos=True,
        detokenize=False,
    )
    documents = None
    if args.real_token_cache:
        import torch
        from benchmarks.prolong import DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED

        cached = torch.load(args.real_token_cache, map_location="cpu", weights_only=False)
        if (cached.get("dataset"), cached.get("revision"), cached.get("seed")) != (
            DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED,
        ):
            raise ValueError("ProLong token cache does not match the frozen speed corpus")
        documents = cached["documents"]
    reference_prompts = {}
    reference_traces = {}
    reference_trace_extensions = {}
    if args.reference_baselines:
        from transformers import AutoTokenizer
        from benchmarks.prolong import SEPARATOR, token_digest

        if documents is None:
            parser.error("reference cohorts require a real token cache")
        separator = AutoTokenizer.from_pretrained(
            args.checkpoint, trust_remote_code=True,
        )(SEPARATOR, add_special_tokens=False)["input_ids"]
        for path in args.reference_baselines:
            archived = json.loads(path.read_text())
            for context, measured in archived["measurements"].items():
                length = int(context)
                if length not in args.lengths:
                    continue
                rows = []
                traces = []
                for record in measured["prompts"]:
                    tokens = []
                    for index in record["panel_source_stream_indices"]:
                        if tokens:
                            tokens.extend(separator)
                        tokens.extend(documents[index])
                    if token_digest(tokens[:length]) != record["token_sha256"]:
                        raise RuntimeError(f"archived ProLong prompt hash mismatch: {path}, {length}")
                    rows.append({"prompt_token_ids": tokens[:length]})
                    if args.reference_decode_trace:
                        traces.append(reference_trace_prefix(
                            tokens, record, length=length, output_tokens=args.decode_tokens,
                        ))
                        if args.decode_tokens > int(record["trace_tokens"]):
                            reference_trace_extensions.setdefault(str(length), []).append({
                                "archived_outputs": int(record["trace_tokens"]),
                                "measured_outputs": args.decode_tokens,
                                "archived_trace_sha256": record["trace_token_sha256"],
                                "extended_trace_sha256": token_digest(traces[-1]),
                                "source": "same frozen ProLong document stream",
                            })
                if len(rows) != args.batch_size:
                    raise ValueError("reference prompt cohort differs from requested batch size")
                reference_prompts[length] = rows
                reference_traces[length] = traces
        if set(reference_prompts) != set(args.lengths):
            raise ValueError("archived baselines do not cover all requested lengths")

    def prompts(length: int) -> list[dict[str, list[int]]]:
        if reference_prompts:
            return reference_prompts[length]
        if documents is not None:
            rows = []
            for request in range(args.batch_size):
                tokens = []
                cursor = request
                while len(tokens) < length:
                    tokens.extend(documents[cursor % len(documents)])
                    cursor += args.batch_size
                rows.append({"prompt_token_ids": tokens[:length]})
            return rows
        return [
            {
                "prompt_token_ids": [
                    3 + ((position + request) % 997) for position in range(length)
                ]
            }
            for request in range(args.batch_size)
        ]

    llm = LLM(**kwargs)
    result = None
    try:
        from benchmarks.kimi_k3_kda_dense_prefill import select_kda
        kda_audits = llm.collective_rpc(select_kda, args=(args.kda_prefill, 8))
        if {audit["rank"] for audit in kda_audits} != set(range(args.tensor_parallel_size)):
            raise RuntimeError("missing KDA baseline worker audit")
        owner_local_audits = None
        if args.owner_local_mla:
            from benchmarks._kimi_owner_local_mla import install_owner_local_mla
            owner_local_audits = llm.collective_rpc(install_owner_local_mla,
                args=(prompts(min(args.lengths))[0]["prompt_token_ids"][:128],), timeout=300)
            if {a["rank"] for a in owner_local_audits} != set(range(8)):
                raise RuntimeError("missing owner-local MLA projection audit rank")
        elif args.owner_tp_mla:
            from benchmarks._kimi_owner_local_mla import prepare_owner_tp_mla
            owner_local_audits = llm.collective_rpc(prepare_owner_tp_mla, timeout=300)
            if {a["rank"] for a in owner_local_audits} != set(range(8)):
                raise RuntimeError("missing owner-TP MLA projection audit rank")
        elif args.head_owner_mla:
            from benchmarks._kimi_owner_local_mla import prepare_head_owner_tp_mla
            owner_local_audits = llm.collective_rpc(prepare_head_owner_tp_mla,
                args=(prompts(min(args.lengths))[0]["prompt_token_ids"][:32],), timeout=300)
            if {a["rank"] for a in owner_local_audits} != set(range(8)):
                raise RuntimeError("missing six-head-owner preparation audit rank")
        if args.validate_route_scatter:
            llm.collective_rpc(validate_expert_scatter)
        measurements = {}
        result = {
            "checkpoint": args.checkpoint,
            "kda_prefill": args.kda_prefill,
            "kda_prefill_worker_audit": kda_audits,
            "max_model_len": kwargs["max_model_len"],
            "mode": args.mode,
            "allocation_audit_only": args.allocation_audit_only,
            "capacity_only": args.capacity_only,
            "diagnostic_only": args.diagnostic_only,
            "route_scatter_validation": args.validate_route_scatter,
            "weight_cache_id": args.weight_cache_id,
            "real_token_cache": str(args.real_token_cache) if args.real_token_cache else None,
            "reference_baselines": list(map(str, args.reference_baselines)),
            "request_owner_prefill": os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") == "1",
            "owner_local_mla": args.owner_local_mla,
            "owner_tp_mla": args.owner_tp_mla,
            "head_owner_mla": args.head_owner_mla,
            "attention_owners": 6 if args.head_owner_mla else None,
            "attention_heads_per_owner": 16 if args.head_owner_mla else None,
            "owner_local_projection_audits": owner_local_audits,
            "scheduler_row_chunk": int(kwargs["long_prefill_token_threshold"]),
            "scheduler_total_budget": int(kwargs["max_num_batched_tokens"]),
            "batch_size": args.batch_size,
            "decode_tokens": args.decode_tokens,
            "reference_decode_trace": args.reference_decode_trace,
            "reference_trace_extensions": reference_trace_extensions,
            "decode_execution": ("captured_request_owner"
                if os.getenv("LOD_KIMI_REQUEST_OWNER_DECODE") == "1" else
                "eager_request_owner" if os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") == "1" else "native"),
            "tensor_parallel_size": args.tensor_parallel_size,
            "decode_context_parallel_size": args.decode_context_parallel_size,
            "repeats": args.repeats,
            "rotating_prefills": os.environ.get("LOD_BENCHMARK_ROTATE_PREFILLS") == "1",
            "rotating_cohort": int(os.environ.get("LOD_BENCHMARK_ROTATING_COHORT", args.batch_size)),
            "distributed_prefill_construction": os.environ.get("LOD_KIMI_DISTRIBUTED_PREFILL_BUILD") == "1",
            "cross_layer_prefill_group": os.environ.get("LOD_KIMI_CROSS_LAYER_PREFILL_GROUP"),
            "independent_local_dcp_prefill": (
                os.environ.get("LOD_KIMI_DCP_LOCAL_PREFILL") == "1"
            ),
            "replicated_summary_dcp_prefill": (
                os.environ.get("LOD_KIMI_DCP_SHARED_PREFILL") == "1"
            ),
            "global_centroid_sharded_leaf_prefill": (
                os.environ.get("LOD_KIMI_DCP_SHARDED_LEAVES") == "1"
            ),
            "transient_reconstructed_prefill_workspace": (
                os.environ.get("LOD_KIMI_DCP_PREFILL_WORKSPACE") == "1"
            ),
            "sharded_workspace_layout": (
                "shared_stream_geometric_buffers_v3"
                if os.environ.get("LOD_KIMI_DCP_PREFILL_WORKSPACE") == "1" else (
                    ("distributed_fine_b1_pool_backed_tp_head_scratch_v5"
                     if args.batch_size == 1 else "distributed_fine_tp_head_scratch_v5")
                    if os.environ.get("LOD_KIMI_DCP_SHARDED_LEAVES") == "1" else None)
            ),
            "environment": {name: value for name, value in os.environ.items()
                            if name.startswith(("LOD_KIMI_", "LOD_BENCHMARK_")) or name in (
                                "HSA_NO_SCRATCH_RECLAIM", "CLUSTER_RUN_NODE",
                                "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
            "measurements": measurements,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "worker_attention_audit_status": "pending",
            "measurement_status": "in_progress",
            "graph_prefill_requested": os.environ.get("LOD_KIMI_GRAPH_PREFILL") == "1",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def save_result() -> None:
            # Readers may inspect a completed point during the next long
            # warmup. Never expose a half-written JSON artifact.
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(args.output)
            if (args.output.name.startswith("oct7-current-")
                    and result.get("current_phase", {}).get("phase") == "point_complete"
                    and not args.capacity_only and not args.diagnostic_only):
                # After generation/auditing, never inside the serving timer.
                # Keep the canonical panel current during a multi-hour sweep.
                from benchmarks.kimi_k3_current_timings import render
                render(args.output.parent)

        def audit_point(point: dict, length: int) -> None:
            owners = point.get("active_attention_owner_ranks")
            audits = audit_loaded_attention(llm, mode=args.mode, length=length,
                required_ranks=None if owners is None else set(owners))
            point.update(measurement_status="complete", worker_attention_audit=audits,
                         worker_attention_audit_status="passed")
            result.update(worker_attention_audit=audits, worker_attention_audit_status="passed")

        save_result()
        if args.allocation_audit_only:
            result["allocation_worker_memory"] = llm.collective_rpc(peak_memory)
            result["measurement_status"] = "complete"
            result["scope"] = "reserved storage only; no long-context generation or speed claim"
            save_result()
            return
        for length in args.lengths:
            if args.reference_decode_trace:
                params = [SamplingParams(temperature=0, max_tokens=args.decode_tokens,
                    seed=0, ignore_eos=True, detokenize=False, trace_decode_token_ids=trace)
                    for trace in reference_traces[length]]
            # Different prefix lengths can select different DCP merge and
            # state-update specializations. Warm the exact measured length so
            # JIT compilation is never charged to serving latency.
            from benchmarks.prolong import release_worker_allocator_cache

            retain_allocator = os.getenv("LOD_KIMI_REUSE_PREFILL_ALLOCATOR") == "1"
            before_warmup_allocator = llm.collective_rpc(
                release_worker_allocator_cache, args=(retain_allocator,))
            if args.report_memory:
                llm.collective_rpc(reset_peak_memory)
            report_phase(result, save_result, length=length, phase="warmup")
            captured_owner = os.getenv("LOD_KIMI_REQUEST_OWNER_DECODE") == "1"
            if captured_owner:
                llm.collective_rpc(owner_decode_graph_replays)
            if args.audit_prefill_batches:
                from benchmarks._prefill_batch_audit import (
                    arm_prefill_batch_audit, finish_prefill_batch_audit,
                    validate_prefill_batch_audits)
                llm.collective_rpc(arm_prefill_batch_audit)
            if args.report_memory:
                llm.collective_rpc(arm_warmup_leaf_stats)
            warmup_started = time.perf_counter()
            warmup_outputs = llm.generate(prompts(length), params, use_tqdm=False)
            warmup_elapsed = time.perf_counter() - warmup_started
            if args.report_memory:
                llm.collective_rpc(finish_warmup_leaf_stats)
            if args.reference_decode_trace and any(
                list(output.outputs[0].token_ids) != trace
                for output, trace in zip(warmup_outputs, reference_traces[length], strict=True)
            ):
                raise RuntimeError("warmup did not replay the verified dense decode trace")
            batch_audits = None
            if args.audit_prefill_batches:
                batch_audits = llm.collective_rpc(finish_prefill_batch_audit)
                validate_prefill_batch_audits(batch_audits, length=length,
                    batch_size=args.batch_size,
                    row_chunk=int(kwargs["long_prefill_token_threshold"]),
                    cohort=min(args.batch_size, int(os.getenv("LOD_BENCHMARK_PREFILL_COHORT", "1"))),
                    world_size=args.tensor_parallel_size)
                result.setdefault("warmup_prefill_batch_audits", {})[str(length)] = batch_audits
            warmup_memory = llm.collective_rpc(peak_memory) if args.report_memory else None
            if warmup_memory is not None:
                result.setdefault("warmup_worker_memory", {})[str(length)] = warmup_memory
            report_phase(result, save_result, length=length, phase="warmup_complete",
                         warmup_elapsed_seconds=warmup_elapsed)
            if args.diagnostic_only:
                from benchmarks.kimi_k3_owner_diagnostic import (
                    arm_owner_diagnostic, finish_owner_diagnostic)
                llm.collective_rpc(release_worker_allocator_cache, args=(retain_allocator,))
                row_chunk = int(kwargs["long_prefill_token_threshold"])
                llm.collective_rpc(arm_owner_diagnostic, args=(length - row_chunk,
                    str(args.output.with_suffix(".trace.json"))))
                report_phase(result, save_result, length=length, phase="diagnostic")
                llm.generate(prompts(length), params, use_tqdm=False)
                result["owner_chunk_diagnostic"] = llm.collective_rpc(finish_owner_diagnostic)
                audit_point({}, length)
                result["measurement_status"] = "complete"
                save_result()
                break
            if args.capacity_only:
                point = {
                    "prefill_completed": True,
                    "decode_steps_exercised": args.decode_tokens - 1,
                    "amortized_decode_benchmark": False,
                    "generated_token_ids": [output.outputs[0].token_ids for output in warmup_outputs],
                    "worker_memory": warmup_memory,
                    "before_warmup_allocator_policy": before_warmup_allocator,
                }
                if documents is not None:
                    from benchmarks.prolong import token_digest
                    point["prompt_token_sha256"] = [
                        token_digest(row["prompt_token_ids"]) for row in prompts(length)
                    ]
                if os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") == "1":
                    point["owner_prefill_audit"] = llm.collective_rpc(owner_prefill_audit)
                    if length >= kwargs["long_prefill_token_threshold"]:
                        validate_owner_prefill_audits(point["owner_prefill_audit"],
                            row_chunk=int(kwargs["long_prefill_token_threshold"]),
                            world_size=args.tensor_parallel_size, batch_size=args.batch_size,
                            head_owners=args.head_owner_mla, length=length)
                measurements[str(length)] = point
                audit_point(point, length)
                save_result()
                print("KIMI_CAPACITY_POINT " + json.dumps({"length": length, **point}), flush=True)
                continue
            warmup_allocator = llm.collective_rpc(
                release_worker_allocator_cache, args=(retain_allocator,))
            result.setdefault("warmup_allocator_policy", {})[str(length)] = warmup_allocator
            if args.report_memory:
                llm.collective_rpc(reset_peak_memory)
            elapsed_samples = []
            prefill_samples = []
            decode_samples = []
            owner_decode_deltas = []
            graph_replay_deltas = []
            decode_update_audits = []
            batch_timings = []
            for repeat in range(args.repeats):
                updates_before = (llm.collective_rpc(read_decode_update_counters)
                                  if args.decode_tokens > 1 else None)
                counts_before = llm.collective_rpc(owner_decode_counters) if (
                    os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") == "1" and args.decode_tokens > 1) else None
                graph_before = llm.collective_rpc(owner_decode_graph_replays) if captured_owner else None
                report_phase(result, save_result, length=length, phase="measurement",
                             repeat_index=repeat, repeats=args.repeats)
                elapsed, prefill, decode, token_ids, _, timing = timed_sweep_generate(
                    llm, prompts(length), params,
                    synchronized_decode=args.reference_decode_trace and args.decode_tokens > 1,
                )
                elapsed_samples.append(elapsed)
                prefill_samples.append(prefill)
                batch_timings.append(timing)
                if updates_before is not None:
                    update_delta = decode_update_deltas(
                        updates_before, llm.collective_rpc(read_decode_update_counters))
                    decode_update_audits.append(update_delta)
                    if (args.mode == "two-tier" and args.decode_tokens == 1026
                            and os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") != "1"):
                        expected = {"catch_up_batches": 4, "catch_up_rows": 4 * args.batch_size}
                        if len(update_delta) != args.tensor_parallel_size or any(
                            not worker or any(counts != expected for counts in worker.values())
                            for worker in update_delta
                        ):
                            raise RuntimeError("decode measurement did not include four updates per row")
                if graph_before is not None:
                    graph_after = llm.collective_rpc(owner_decode_graph_replays)
                    delta = [last - first for first, last in zip(graph_before, graph_after, strict=True)]
                    if delta != [args.decode_tokens - 1] * args.tensor_parallel_size:
                        raise RuntimeError(f"owner decode did not replay every B8 model graph: {delta}")
                    graph_replay_deltas.append(delta)
                if counts_before is not None:
                    owner_decode_deltas.append(validate_owner_decode_counts(
                        counts_before, llm.collective_rpc(owner_decode_counters),
                        steps=args.decode_tokens - 1, world_size=args.tensor_parallel_size))
                if args.reference_decode_trace and any(list(tokens) != trace
                    for tokens, trace in zip(token_ids, reference_traces[length], strict=True)):
                    raise RuntimeError("generation did not replay the verified dense decode trace")
                decode_steps = args.decode_tokens - 1
                if decode_steps:
                    decode_samples.append(decode / decode_steps)
            elapsed = statistics.median(elapsed_samples)
            prefill = statistics.median(prefill_samples)
            report_phase(result, save_result, length=length, phase="auditing",
                         prefill_seconds=prefill, elapsed_seconds=elapsed)
            measurements[str(length)] = {
                "warmup_elapsed_seconds": warmup_elapsed,
                "before_warmup_allocator_policy": before_warmup_allocator,
                "warmup_allocator_policy": warmup_allocator,
                "elapsed_seconds": elapsed,
                "prefill_seconds": prefill,
                "elapsed_samples_seconds": elapsed_samples,
                "prefill_samples_seconds": prefill_samples,
                "decode_ms_per_batch_step": (
                    1000.0 * statistics.median(decode_samples)
                    if decode_samples
                    else None
                ),
                "decode_samples_seconds_per_batch_step": decode_samples,
                "aggregate_prompt_tokens_per_second": (
                    args.batch_size * length / prefill
                ),
                "generated_token_ids": [list(tokens) for tokens in token_ids],
                "measured_batch_timings": batch_timings,
                "owner_decode_update_deltas": owner_decode_deltas,
                "measured_decode_update_counters": decode_update_audits,
                "owner_decode_graph_replay_deltas": graph_replay_deltas,
            }
            if os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") == "1":
                owner_audits = llm.collective_rpc(owner_prefill_audit)
                measurements[str(length)]["owner_prefill_audit"] = owner_audits
                row_chunk = int(kwargs["long_prefill_token_threshold"])
                if length >= row_chunk:
                    active_owners = validate_owner_prefill_audits(
                        owner_audits, row_chunk=row_chunk,
                        world_size=args.tensor_parallel_size, batch_size=args.batch_size,
                        head_owners=args.head_owner_mla, length=length)
                    measurements[str(length)]["active_attention_owner_ranks"] = sorted(active_owners)
            if documents is not None:
                from benchmarks.prolong import token_digest
                measurements[str(length)]["prompt_token_sha256"] = [
                    token_digest(row["prompt_token_ids"]) for row in prompts(length)
                ]
            if args.report_memory:
                measurements[str(length)]["warmup_worker_memory"] = warmup_memory
                measurements[str(length)]["worker_memory"] = llm.collective_rpc(peak_memory)
            audit_point(measurements[str(length)], length)
            if os.getenv("LOD_KIMI_REQUEST_OWNER_DECODE") == "1":
                graph_audits = llm.collective_rpc(owner_decode_graph_audit)
                if len(graph_audits) != 8 or any(
                    a["owner_layer_count"] != len(a["local_decode_heads"])
                    or not any(g["num_tokens"] == 8 and g["graph_instantiated"]
                               for g in a["captured_graphs"])
                    or set(a["local_decode_world_sizes"]) != {1}
                    or set(a["local_decode_heads"]) != {96}
                    or set(a["active_global_cadences"]) != {256}
                    for a in graph_audits):
                    raise RuntimeError("owner decode did not instantiate the B8 single-owner graph")
                measurements[str(length)]["owner_decode_graph_audits"] = graph_audits
            report_phase(result, save_result, length=length, phase="point_complete")
            print("KIMI_PREFILL_POINT " + json.dumps({
                "length": length, **measurements[str(length)],
            }), flush=True)
            # Completed points retain their own binary audit if a later,
            # larger shape fails. All RPCs are outside measured generation.
            save_result()
        result["measurement_status"] = "complete"
        if args.profile_length is not None:
            from benchmarks._prefill_profile import start_prefill_profile, stop_prefill_profile

            llm.collective_rpc(start_prefill_profile)
            llm.generate(prompts(args.profile_length), params, use_tqdm=False)
            result["diagnostic_gpu_profile"] = llm.collective_rpc(stop_prefill_profile)
        save_result()
        print(json.dumps(result, indent=2))
    except Exception as error:
        if result is not None:
            result.update(measurement_status="failed", failure={
                "exception_type": type(error).__name__, "message": str(error),
            })
            save_result()
        raise
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

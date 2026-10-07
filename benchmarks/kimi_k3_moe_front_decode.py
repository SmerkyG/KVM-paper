"""Matched decode-sized MoE-front variants, with one resident vLLM model.

Native and merged FULL_DECODE_ONLY graphs are captured separately outside
timing and retained for A/B/A comparison; weights are never reloaded. Both
arms use the approved G8 direct-state KDA prefill. Native expert routing,
FP32 correction bias, expert kernels, streams, norms, and tail are unchanged.
The frozen teacher-forced 1,025-step serving run includes four LoD updates.
The adapters gate on token count: large prefill stays native, but a small
prefill fragment can use the candidate too. Timings here isolate decode,
not an end-to-end front speedup or a demonstrated prediction-quality gain.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def prepare_fronts(worker, max_tokens, token_ids, front_kind="all"):
    import torch
    from vllm.forward_context import set_forward_context
    from benchmarks.experimental.kimi_moe_front import DecodeFront, EarlySharedFront

    model = worker.model_runner.model
    core = next(m for m in model.modules() if type(m).__name__ == "KimiLinearModel")
    layers = [m for m in model.modules() if type(m).__name__ == "KimiMoE"]
    fronts = [(EarlySharedFront(m, max_tokens) if front_kind == "early"
               else DecodeFront(m, max_tokens, front_kind)) for m in layers]
    worker._lod_moe_fronts = fronts
    ids = torch.tensor(token_ids[:max_tokens], device="cuda", dtype=torch.long)
    hidden = core.embed_input_ids(ids)
    checks = []
    # Check every front against the real router, including FP32 correction
    # bias and renormalization. Full expert output checks use two layers.
    with torch.inference_mode(), set_forward_context(None, worker.vllm_config, num_tokens=len(ids)):
        for index, front in enumerate(fronts):
            layer = front.layer
            gate_up = layer.shared_experts.gate_up_proj(hidden)[0]
            shared = layer.shared_experts.act_fn(gate_up)
            logits = layer.gate(hidden)[0]
            latent = layer.routed_expert_down_proj(hidden)[0]
            ref_weights, ref_ids = layer.experts.router.select_experts(latent, logits)
            a, b, c = front.project(hidden)
            got_weights, got_ids = layer.experts.router.select_experts(c, b)
            def error(ref, actual):
                return float((actual.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-20))
            check = dict(layer=index, expert_sets_equal=bool(torch.equal(
                ref_ids.sort().values, got_ids.sort().values)),
                shared_relative_l2=error(shared, a) if a is not None else 0.0, router_relative_l2=error(logits, b),
                latent_relative_l2=error(latent, c), routing_weights_relative_l2=error(ref_weights, got_weights),
                correction_bias_dtype=str(layer.gate.e_score_correction_bias.dtype))
            if index in (0, len(fronts)-1):
                native = layer(hidden)
                front.enabled = True
                merged = layer(hidden)
                front.enabled = False
                check["moe_output_relative_l2"] = error(native, merged)
                # Small roundoff is expected; no route mismatch or large
                # downstream deviation may be waved through as speed gain.
                if check["moe_output_relative_l2"] > 0.02:
                    raise RuntimeError(f"MoE output check failed: {check}")
            if not check["expert_sets_equal"] or max(check[k] for k in (
                    "shared_relative_l2", "router_relative_l2", "latent_relative_l2")) > 0.02:
                raise RuntimeError(f"merged projection/routing check failed: {check}")
            checks.append(check)
    manager = worker.model_runner.cudagraph_manager
    worker._lod_moe_graphs = {"native": dict(manager.graphs)}
    if not manager.graphs:
        raise RuntimeError("native decode graph must be instantiated")
    return dict(rank=worker.rank, layers=len(fronts), extra_bytes=sum(f.extra_bytes for f in fronts),
                checks=checks, shared_stream_retained=True)


def select_front(worker, variant):
    """Never time a new Python forward using the old captured native graph."""
    import torch
    torch.cuda.synchronize()
    fronts, snapshots = worker._lod_moe_fronts, worker._lod_moe_graphs
    for front in fronts:
        front.enabled = variant == "merged"
    manager = worker.model_runner.cudagraph_manager
    if variant not in snapshots:
        manager.graphs = {}
        manager._graphs_captured = False
        for front in fronts:
            if front.kind == "early":
                front.record_capture = True
        worker.model_runner.capture_model()
        for front in fronts:
            if front.kind == "early":
                front.record_capture = False
                if not front.capture_enqueues:
                    raise RuntimeError("native backend did not capture early shared-expert work")
        snapshots[variant] = dict(manager.graphs)
    manager.graphs = dict(snapshots[variant])
    manager._graphs_captured = True
    worker._lod_moe_decode_replays = 0
    if not hasattr(worker, "_lod_moe_original_replay"):
        worker._lod_moe_original_replay = manager.run_fullgraph
        def replay(desc):
            worker._lod_moe_decode_replays += 1
            return worker._lod_moe_original_replay(desc)
        manager.run_fullgraph = replay
    return dict(rank=worker.rank, variant=variant, captured_tokens=manager.captured_token_counts(),
                distinct_graphs=len({id(g) for graphs in snapshots.values() for g in graphs.values()}),
                early_capture_enqueues=[f.capture_enqueues for f in fronts if f.kind == "early"])


def replay_counts(worker):
    return dict(rank=worker.rank, decode_graph_replays=worker._lod_moe_decode_replays)


def expected_decode_updates(length, output_tokens):
    # The first output comes from prefill; do not count that boundary twice.
    return (length + output_tokens - 2)//256 - length//256


def arm_graph_kernel_check(worker):
    """Poison private scratch immediately before an untimed graph replay.

    Native graphs never touch these buffers. Merged graphs populate them on
    each decode. Arming after prefill is essential: a small prefill fragment
    could otherwise populate candidate scratch even with a wrong decode
    graph. The one-shot host hook removes itself before replay. It adds no
    graph node or measured instrumentation. Kineto crashes with this image's
    HSA/RCCL combination.
    """
    manager = worker.model_runner.cudagraph_manager
    original = manager.run_fullgraph
    worker._lod_moe_canary_replayed = False
    def first_replay(desc):
        manager.run_fullgraph = original
        for front in worker._lod_moe_fronts:
            if front.kind != "early":
                front.router.fill_(float("nan"))
        worker._lod_moe_canary_replayed = True
        return original(desc)
    manager.run_fullgraph = first_replay


def graph_kernel_check(worker):
    import torch
    if not worker._lod_moe_canary_replayed:
        raise RuntimeError("canary did not observe a decode graph replay")
    if worker._lod_moe_fronts[0].kind == "early":
        # There is deliberately no additional device operation to identify
        # this path. Native events are enqueued during separate graph capture
        # on every layer; actual FULL graph replays are audited separately.
        return dict(rank=worker.rank, written_front_layers=len(worker._lod_moe_fronts)
                    if worker._lod_moe_fronts[0].enabled else 0, untimed=True,
                    method="separate graph capture / native event enqueue / replay audit",
                    early_capture_enqueues=[f.capture_enqueues for f in worker._lod_moe_fronts])
    written = [bool(torch.isfinite(front.router).all()) for front in worker._lod_moe_fronts]
    untouched = [bool(torch.isnan(front.router).all()) for front in worker._lod_moe_fronts]
    if any(not (a or b) for a, b in zip(written, untouched)):
        raise RuntimeError("only part of a decode front's scratch was populated")
    return dict(rank=worker.rank, written_front_layers=sum(written), untimed=True,
                method="poison/read private scratch; no hot-path instrumentation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="tests/fixtures/kimi-k3-attention-moe")
    p.add_argument("--weight-cache-id")
    p.add_argument("--real-token-cache", type=Path)
    p.add_argument("--mode", choices=("full", "two-tier"), default="full")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--front", choices=("all", "routed", "early"), default="all")
    p.add_argument("--length", type=int, default=16384)
    p.add_argument("--batches", type=int, nargs="+", default=[8])
    p.add_argument("--decode-tokens", type=int, default=1026)
    p.add_argument("--validation-only", action="store_true",
                   help="check captured graph execution and greedy equality without repeating timings")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.length < 256 or args.length % 256 or args.decode_tokens < 2:
        p.error("use positive global 256-token multiples and at least one decode step")
    if not args.fixture and (not args.weight_cache_id or not args.real_token_cache):
        p.error("trained runs require the resident weight cache and frozen real tokens")
    if args.batches not in ([1], [8]):
        p.error("one B1 or B8 cohort per engine (admission barrier is engine-local)")
    os.environ.update(VLLM_ALLOW_INSECURE_SERIALIZATION="1")
    import torch
    from vllm import LLM, SamplingParams
    from benchmarks._vllm import close_llm, llm_kwargs, write_json
    from benchmarks.kimi_k3_kda_dense_prefill import select_kda
    from benchmarks.kimi_k3_prefill_sweep import timed_sweep_generate
    from benchmarks.prolong import configure_synchronized_decode_environment, token_digest
    from benchmarks.kimi_k3_decode_power2 import LOD_ENV
    os.environ.update(LOD_ENV)
    configure_synchronized_decode_environment(enabled=True, batch_size=max(args.batches))
    kwargs = llm_kwargs(checkpoint=args.checkpoint, mode=args.mode,
        max_model_len=args.length + args.decode_tokens + 16, batch_size=max(args.batches),
        tensor_parallel_size=8, decode_context_parallel_size=8, dcp_comm_backend="ag_rs",
        gpu_memory_utilization=0.8 if not args.fixture else 0.1,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(skip_tokenizer_init=True, enable_trace_replay=True,
        disable_custom_all_reduce=False, enable_expert_parallel=True,
        kv_cache_memory_bytes=(1 if args.fixture or max(args.batches) == 1 else 3) << 30,
        compilation_config=dict(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=args.batches,
                                max_cudagraph_capture_size=max(args.batches)))
    if args.fixture:
        # Two real-geometry native MoE layers (official shared width is
        # 2*3072 / TP8 = 768, not 768 / TP8). Dummy weights are graph/stream
        # evidence only, not trained-model quality evidence.
        kwargs.update(load_format="dummy", kernel_config={"moe_backend": "triton"},
            hf_overrides=dict(num_experts=896, num_experts_per_token=16, moe_intermediate_size=3072,
                              num_shared_experts=2, routed_expert_hidden_size=3584))
        kwargs["compilation_config"]["mode"] = 0
        docs = [[(i*17+row*31) % 256 for i in range(args.length+args.decode_tokens)] for row in range(8)]
    else:
        kwargs.update(load_format="ipc_cache",
            quantization_config={"moe": {"weight": "int4_per_group_32"}},
            model_loader_extra_config=dict(auto_start=True, cache_id=args.weight_cache_id,
                backing_load_format="auto", broker_timeout=1800.0))
        cached = torch.load(args.real_token_cache, map_location="cpu", weights_only=False)
        from benchmarks.prolong import DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED
        if (cached["dataset"], cached["revision"], cached["seed"]) != (DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED):
            raise ValueError("not the frozen ProLong speed corpus")
        docs = cached["documents"]
    result = dict(status="starting", fixture_only=args.fixture, mode=args.mode, engine_config=kwargs,
        length=args.length, batches=args.batches, timed_decode_steps=0 if args.validation_only else args.decode_tokens-1,
        front_kind=args.front, kda_prefill_baseline="gluon_paged_G8", production_changed=False, measurements={})
    def save():
        write_json(args.output, result)
    save()
    llm = LLM(**kwargs)
    try:
        if not args.fixture:
            result["kda_hooks"] = llm.collective_rpc(select_kda, args=("gluon_paged", 8))
        owner = args.mode == "two-tier" and args.batches == [8]
        if owner:
            from benchmarks._kimi_owner_local_mla import prepare_owner_tp_mla
            result["owner_preparation"] = llm.collective_rpc(prepare_owner_tp_mla)
        result["preparation"] = llm.collective_rpc(prepare_fronts, args=(max(args.batches), docs[0], args.front), timeout=300)
        save()
        for batch in args.batches:
            prompts, traces = [], []
            for row in range(batch):
                tokens, cursor = [], row
                while len(tokens) < args.length + args.decode_tokens:
                    tokens.extend(docs[cursor % len(docs)])
                    cursor += batch
                prompts.append(dict(prompt_token_ids=tokens[:args.length]))
                traces.append(tokens[args.length:args.length+args.decode_tokens])
            params = [SamplingParams(temperature=0, seed=0, max_tokens=args.decode_tokens,
                ignore_eos=True, detokenize=False, trace_decode_token_ids=t) for t in traces]
            points = result["measurements"][str(batch)] = {}
            for variant in (() if args.validation_only else ("native", "merged", "native_repeat")):
                actual = variant.replace("_repeat", "")
                result.update(status="warming", active_batch=batch, active_variant=variant)
                audits = llm.collective_rpc(select_front, args=(actual,), timeout=300)
                save()
                llm.generate(prompts, params, use_tqdm=False)
                before = llm.collective_rpc(replay_counts)
                if args.mode == "two-tier":
                    from benchmarks.kimi_k3_prefill_sweep import owner_decode_counters
                    from benchmarks._decode_update_audit import read_decode_update_counters, decode_update_deltas
                    counter_fn = owner_decode_counters if owner else read_decode_update_counters
                    updates_before = llm.collective_rpc(counter_fn)
                result["status"] = "measuring"
                save()
                elapsed, prefill, decode, ids, _, timing = timed_sweep_generate(
                    llm, prompts, params, synchronized_decode=True)
                after = llm.collective_rpc(replay_counts)
                if args.mode == "two-tier":
                    updates_after = llm.collective_rpc(counter_fn)
                    updates = decode_update_deltas(updates_before, updates_after)
                    expected_updates = expected_decode_updates(args.length, args.decode_tokens)
                    for worker_counters in updates:
                        if not worker_counters:
                            raise RuntimeError("no LoD update counters")
                        for counters in worker_counters.values():
                            expected = (dict(updates=expected_updates, tokens=args.decode_tokens-1)
                                if owner else dict(catch_up_batches=expected_updates, catch_up_rows=expected_updates*batch))
                            if counters != expected:
                                raise RuntimeError(f"global decode cadence mismatch: {counters} != {expected}")
                counts = [b["decode_graph_replays"] - a["decode_graph_replays"] for a, b in zip(before, after)]
                trace_match = [list(t) for t in ids] == traces
                if counts != [args.decode_tokens-1]*8 or not trace_match:
                    raise RuntimeError(f"invalid serving measurement: graph replays={counts}, trace_match={trace_match}")
                points[variant] = dict(decode_ms_per_batch_step=decode/(args.decode_tokens-1)*1000,
                    prefill_seconds=prefill, elapsed_seconds=elapsed, request_timings=timing,
                    decode_graph_replays=counts, worker_graphs=audits,
                    prompt_sha256=[token_digest(x["prompt_token_ids"]) for x in prompts],
                    trace_sha256=[token_digest(t) for t in traces])
                if args.mode == "two-tier":
                    points[variant]["decode_update_counters"] = updates
                print(f"MOE_MODEL_DECODE B={batch} variant={variant} ms={points[variant]['decode_ms_per_batch_step']:.6f}", flush=True)
                save()
            if not args.validation_only:
                baseline = (points["native"]["decode_ms_per_batch_step"]+points["native_repeat"]["decode_ms_per_batch_step"])/2
                points["speedup"] = baseline/points["merged"]["decode_ms_per_batch_step"]
            result.update(status="checking_greedy")
            save()
            greedy = SamplingParams(temperature=0, seed=0, max_tokens=32, ignore_eos=True, detokenize=False)
            outputs = {}
            for variant in ("native", "merged"):
                llm.collective_rpc(select_front, args=(variant,))
                llm.collective_rpc(arm_graph_kernel_check)
                outputs[variant] = [o.outputs[0].token_ids for o in llm.generate(prompts, greedy, use_tqdm=False)]
                kernel_check = llm.collective_rpc(graph_kernel_check)
                expected = len(result["preparation"][0]["checks"]) if variant == "merged" else 0
                if any(check["written_front_layers"] != expected for check in kernel_check):
                    raise RuntimeError(f"captured graph did not use {variant} front: {kernel_check}")
                points.setdefault("untimed_graph_kernel_checks", {})[variant] = kernel_check
            points["greedy_32_tokens"] = outputs
            points["greedy_outputs_identical"] = outputs["native"] == outputs["merged"]
            save()
        result["status"] = "complete"
        save()
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

"""Matched trained-K3 dense prefill: replace only KDA, in one resident engine.

The unmodified runtime, new KDA math with unchanged state/output copies, and
direct state/output I/O are measured independently. Native dense MLA, MoE,
weights, scheduler, convolution, normalization and HIP decode stay unchanged.
One exact-shape warmup and one ordinary ProLong serving measurement per point;
no profiler, internal timers, prefix-cache hits, or weight reloads in between.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def select_kda(worker, variant, groups):
    """Worker-local benchmark hook; never modify installed vLLM/AITER sources."""
    from vllm.models.kimi_k3.amd import kda as module

    if not hasattr(module, "_lod_probe_original_prefill"):
        module._lod_probe_original_prefill = module.chunk_kda_prefill
    original = module._lod_probe_original_prefill
    if variant == "current":
        module.chunk_kda_prefill = original
    else:
        from benchmarks.kimi_k3_kda_upstream_probe import candidate
        from vllm.model_executor.layers.mamba.ops.gather_initial_states import gather_initial_states

        def patched(*, q, k, v, raw_g, raw_beta, A_log, g_bias, **kw):
            if kw.get("checkpoint_offsets") is not None:
                raise NotImplementedError("KDA experiment does not export prefix checkpoints")
            if not kw.get("use_qk_l2norm_in_kernel", True) or kw.get("lower_bound", -5.0) != -5.0:
                raise ValueError("unexpected K3 gate/normalization contract")
            if kw.get("scale") not in (None, 128**-0.5):
                raise ValueError("unexpected K3 query scale")
            cache, indices = kw["state_cache"], kw["state_indices"]
            inp = dict(q=q, k=k, v=v, g=raw_g, beta=raw_beta, A_log=A_log,
                       dt_bias=g_bias, cu_seqlens=kw["cu_seqlens"],
                       chunk_indices=kw["chunk_indices"], chunk_offsets=kw["chunk_offsets"])
            if variant == "gluon_paged":
                return candidate(inp, config={"G": groups}, state_cache=cache,
                    state_indices=indices, has_initial_state=kw["has_initial_state"], out=kw["out"])
            initial = gather_initial_states(cache, indices, kw["has_initial_state"])
            out, state = candidate(inp, initial_state=initial, config={"G": groups})
            kw["out"].copy_(out)
            cache[indices.long()] = state
            return kw["out"], None

        module.chunk_kda_prefill = patched
    return dict(rank=worker.rank, variant=variant, groups=groups,
                implementation=module.chunk_kda_prefill.__module__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weight-cache-id", required=True)
    p.add_argument("--real-token-cache", type=Path, required=True)
    p.add_argument("--length", type=int, default=16384)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.length, args.groups, *args.batches) < 1:
        p.error("length, groups and batch sizes must be positive")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    import torch
    from benchmarks._vllm import close_llm, llm_kwargs, write_json
    from benchmarks.prolong import DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED, timed_generate, token_digest
    from vllm import LLM, SamplingParams

    cached = torch.load(args.real_token_cache, map_location="cpu", weights_only=False)
    if (cached["dataset"], cached["revision"], cached["seed"]) != (
            DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED):
        raise ValueError("not the frozen ProLong speed corpus")
    docs = cached["documents"]
    if len(docs) < max(args.batches) or not any(docs):
        raise ValueError("insufficient frozen documents")
    kwargs = llm_kwargs(checkpoint=args.checkpoint, mode="full", max_model_len=args.length + 9,
        batch_size=max(args.batches), tensor_parallel_size=8, decode_context_parallel_size=8,
        dcp_comm_backend="ag_rs", gpu_memory_utilization=0.8,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="ipc_cache", skip_tokenizer_init=True, enable_expert_parallel=True,
        disable_custom_all_reduce=False, kv_cache_memory_bytes=2 << 30,
        quantization_config={"moe": {"weight": "int4_per_group_32"}},
        model_loader_extra_config=dict(auto_start=True, cache_id=args.weight_cache_id,
            backing_load_format="auto", broker_timeout=1800.0),
        compilation_config=dict(cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=sorted(set(args.batches)), max_cudagraph_capture_size=max(args.batches)))
    result = dict(scope=__doc__, status="starting", length=args.length, tp=8, dcp=8,
        groups=args.groups, trained_weights=True, production_changed=False,
        weight_cache_id=args.weight_cache_id, engine_config=kwargs, measurements={})
    def save():
        write_json(args.output, result)
    save()
    llm = LLM(**kwargs)
    try:
        params = SamplingParams(temperature=0, seed=1234, max_tokens=1, ignore_eos=True, detokenize=False)
        for batch in args.batches:
            # Match the existing prefill-sweep corpus construction. Some
            # frozen documents are shorter than 16K, so continue the same
            # per-request document stream rather than inventing tokens.
            prompts = []
            for row in range(batch):
                tokens, cursor = [], row
                while len(tokens) < args.length:
                    tokens.extend(docs[cursor % len(docs)])
                    cursor += batch
                prompts.append(dict(prompt_token_ids=tokens[:args.length]))
            points = result["measurements"][str(batch)] = {}
            for variant in ("current", "gluon", "gluon_paged"):
                result.update(status="warming", active_batch=batch, active_variant=variant)
                audits = llm.collective_rpc(select_kda, args=(variant, args.groups))
                assert {a["rank"] for a in audits} == set(range(8))
                save()
                llm.generate(prompts, params, use_tqdm=False)
                result["status"] = "measuring"
                save()
                elapsed, prefill, _, ids, _, timing = timed_generate(llm, prompts, params)
                points[variant] = dict(prefill_seconds=prefill, elapsed_seconds=elapsed,
                    aggregate_prompt_tokens_per_second=batch * args.length / prefill,
                    generated_token_ids=ids, request_timings=timing, worker_hooks=audits)
                if variant != "current":
                    points[variant]["speedup"] = points["current"]["prefill_seconds"] / prefill
                    points[variant]["same_first_tokens_as_current"] = ids == points["current"]["generated_token_ids"]
                if variant == "gluon_paged":
                    points[variant]["same_first_tokens_as_copies"] = ids == points["gluon"]["generated_token_ids"]
                print(f"KDA_TRAINED_PREFILL B={batch} variant={variant} seconds={prefill:.6f}", flush=True)
                save()
            result.setdefault("prompt_token_sha256", {})[str(batch)] = [token_digest(x["prompt_token_ids"]) for x in prompts]
        result["status"] = "complete"
        save()
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        try:
            llm.collective_rpc(select_kda, args=("current", args.groups))
        finally:
            close_llm(llm)


if __name__ == "__main__":
    main()

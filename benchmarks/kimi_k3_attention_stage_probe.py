"""Two real K3 MLA+MoE layers; compare TP Q/K/V/O with one attention owner.

Prefill only. No embedding, LM head, sampling, scheduler, or pipeline overlap
is timed. Native AttnRes and EP/TP MoE are retained. Both variants use the
same single-owner LoD math/cache and identical weights and hidden inputs.
This small random-weight geometry fixture is not a trained-model quality or
full-K3 latency benchmark. Weight assembly is outside the prefill interval.
Expert choices are frozen from the control prefill: ordinary BF16 W_O
roundoff otherwise flips near-tied random MoE routes. Router GEMMs and native
expert kernels still execute; this is not natural-routing quality evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from types import MethodType
import zlib


def run_probe(worker, lengths, chunk_size, diagnostic=False):
    import torch
    import torch.nn.functional as F
    from vllm.distributed import get_tp_group
    from vllm.forward_context import set_forward_context
    from vllm_lod_plugin.models.kimi_k3_request_prefill import (
        attend_slice, exchange_queries, exchange_outputs,
    )
    from vllm_lod_plugin.models.kimi_k3_sharded_prefill import gather_prefill
    from benchmarks.kimi_k3_owner_diagnostic import linear_replication_factor

    runner = worker.model_runner
    core = next(m for m in runner.model.modules() if type(m).__name__ == "KimiLinearModel")
    runtime = runner.model_state._vllm_lod_runtime
    layers = list(core.layers)
    group = get_tp_group()
    rank, world = group.rank_in_group, group.world_size
    comm = group.device_communicator.pynccl_comm
    if world != 8 or len(layers) != 2 or comm is None or comm.disabled:
        raise RuntimeError("stage probe requires exactly two real MLA+MoE layers and TP8")
    if any(type(layer.mlp).__name__ != "KimiMoE" for layer in layers):
        raise RuntimeError("the fixture must execute native MoE, not Identity or a dense MLP")
    # Deterministic, non-degenerate random weights. Replicated parameters have
    # identical seeds across ranks; TP/expert shards intentionally differ.
    with torch.inference_mode():
        for name, module in core.named_modules():
            types = {t.__name__ for t in type(module).__mro__}
            factor = linear_replication_factor(module)
            for param_name, parameter in module.named_parameters(recurse=False):
                seed = zlib.adler32((name + "/" + param_name).encode())
                if factor > 1 or (".experts" in name and "shared_experts" not in name):
                    seed += rank
                generator = torch.Generator(device=parameter.device).manual_seed(seed)
                if "RMSNorm" in types:
                    parameter.fill_(1)
                elif parameter.ndim >= 2:
                    fan_in = getattr(module, "input_size", parameter.shape[-1])
                    parameter.normal_(0, float(fan_in) ** -0.5, generator=generator)
                else:
                    parameter.zero_()

    prepared = []
    for layer in layers:
        wrapper = layer.self_attn.mla_attn
        pool = wrapper.mla_attn._vllm_lod_pool
        if wrapper.q_b_proj.weight.shape[0] != 12 * 192:
            raise RuntimeError("unexpected q-projection partition; do not benchmark replicated Q as TP")
        # Gather only once before timing. This prototype retains original TP
        # shards, so its peak memory is not the final partitioned-stage layout.
        q_weight = gather_prefill(group, wrapper.q_b_proj.weight, dim=0)
        o_weight = gather_prefill(group, wrapper.o_proj.weight, dim=1)
        g_weight = gather_prefill(group, wrapper.g_proj.weight, dim=0)
        kv_weight = wrapper.kv_b_proj.weight.view(12, 256, 512)
        uk = gather_prefill(group, kv_weight[:, :128].contiguous(), dim=0)
        uv = gather_prefill(group, kv_weight[:, 128:].transpose(1, 2).contiguous(), dim=0)
        full = (q_weight, o_weight, g_weight, uk, uv) if rank == 0 else None
        local = (kv_weight[:, :128], kv_weight[:, 128:].transpose(1, 2))
        prepared.append((layer, wrapper, pool, full, local))
        del q_weight, o_weight, g_weight, uk, uv

    maximum = max(lengths)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    inputs = torch.randn(maximum, 7168, dtype=torch.bfloat16, device="cuda", generator=generator)
    positions = torch.arange(maximum, device="cuda")
    state = {"variant": None, "previous": 0, "prompt": 0, "trace": {}}
    audits = {}
    route_state = {"capture": True, "logits": {}}
    original_gates = []
    for layer in layers:
        gate = layer.mlp.gate
        original = gate.forward
        original_gates.append((gate, original))

        def routed_gate(self, hidden, *, original=original, index=layer.layer_idx):
            logits, bias = original(hidden)
            key = (state["previous"], index)
            if route_state["capture"]:
                route_state["logits"][key] = logits.clone()
            else:
                logits = route_state["logits"][key]
            return logits, bias

        gate.forward = MethodType(routed_gate, gate)

    def trace(name, tensor, layer_index):
        if diagnostic and rank == 0:
            state["trace"][(state["previous"], layer_index, name)] = tensor[:8].clone()

    def attention(layer, position, hidden):
        index = layer.layer_idx
        _, wrapper, pool, full, local = prepared[index]
        previous, prompt = state["previous"], state["prompt"]
        tokens = hidden.size(0)
        plan = ((0, 0, tokens, previous),)
        owner_local = state["variant"] == "owner_qkvo"
        if not owner_local or rank == 0:
            fused = wrapper.fused_qkv_a_proj(hidden)[0]
            q_c, kv = fused.split((1536, 576), dim=-1)
            latent, direct = kv.split((512, 64), dim=-1)
            q_c, latent = wrapper._normalize_q_kv(q_c, latent)
            q = (F.linear(q_c, full[0]) if owner_local else wrapper.q_b_proj(q_c)[0])
            q = q.view(tokens, 96 if owner_local else 12, 192).contiguous()
            record = torch.cat((latent, direct), dim=-1).unsqueeze(1)
        else:
            q = None
        if not owner_local:
            queries = exchange_queries(group, q, plan)
        else:
            queries = {0: q} if rank == 0 else {}
        outputs = {}
        if rank == 0:
            trace("query", queries[0], index)
            trace("record", record, index)
            outputs[0] = attend_slice(pool, 0, queries[0], record,
                                     full[3], full[4], previous=previous, prompt=prompt)
            trace("attention", outputs[0], index)
        if owner_local:
            if rank == 0:
                gate = F.linear(hidden, full[2]).sigmoid()
                result = F.linear(outputs[0].reshape(tokens, 12288) * gate, full[1])
            else:
                result = hidden.new_empty(tokens, 7168)
            # Normal native MoE expects the same hidden rows on each TP rank.
            # Include this interface cost, rather than timing attention alone.
            comm.broadcast(result, src=0)
            trace("output_projection", result, index)
            return result
        local_output = exchange_outputs(group, outputs, plan, q).reshape(tokens, 1536)
        local_output = local_output * wrapper.g_proj(hidden)[0].sigmoid()
        result = wrapper.o_proj(local_output)[0]
        trace("output_projection", result, index)
        return result

    originals = [layer._run_self_attn for layer in layers]
    for layer in layers:
        layer._run_self_attn = MethodType(attention, layer)

    def execute(variant, length):
        state.update(variant=variant, prompt=length, trace={})
        for _, _, pool, _, _ in prepared:
            pool._kimi_request_owner_rows = {}
            pool.engine.reset_runtime_cache()
        output = inputs.new_empty(length, 7168)
        for start in range(0, length, chunk_size):
            end = min(length, start + chunk_size)
            state["previous"] = start
            hidden = inputs[start:end]
            bank = hidden.new_empty(end - start, 1, 7168)
            prefix_delta = None
            with set_forward_context(None, worker.vllm_config, num_tokens=end-start):
                for layer in layers:
                    hidden, bank, prefix_delta = layer(
                        positions[start:end], hidden, bank, prefix_delta=prefix_delta)
                    trace("moe_output", prefix_delta, layer.layer_idx)
                # Native pre-output-norm hidden result, excluding LM head.
                output[start:end].copy_(hidden + prefix_delta)
        if rank == 0:
            audits[variant] = [{
                "total_len": pool._kimi_request_owner_rows[0]["total"],
                "coverage": int(pool._kimi_request_owner_rows[0]["cache"].state["coverage"]),
                "state_len": int(pool._kimi_request_owner_rows[0]["cache"].state["state_len"]),
                "leaf_count": int(pool._kimi_request_owner_rows[0]["cache"].state["page_cache"]["leaf_count"]),
            } for _, _, pool, _, _ in prepared]
        return output

    def measure(variant, length):
        # Exact-shape warmup includes attention and native expert kernels.
        execute(variant, length)
        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        output = execute(variant, length)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        return seconds, output

    points = {}
    try:
        with torch.inference_mode():
            for length in lengths:
                route_state.update(capture=True, logits={})
                execute("tp_qkvo", length)
                route_state["capture"] = False
                before, reference = measure("tp_qkvo", length)
                reference = reference.clone()
                reference_trace = state["trace"]
                after, observed = measure("owner_qkvo", length)
                delta = observed.float() - reference.float()
                relative = (delta.square().sum() / reference.float().square().sum().clamp_min(1e-20)).sqrt().item()
                finite = bool(torch.isfinite(observed).all())
                if diagnostic and rank == 0:
                    for key, expected in reference_trace.items():
                        actual = state["trace"][key]
                        rel = ((actual.float()-expected.float()).square().sum()
                               / expected.float().square().sum().clamp_min(1e-20)).sqrt().item()
                        print("KIMI_STAGE_EQUIVALENCE " + json.dumps({
                            "length": length, "chunk_start": key[0], "layer": key[1],
                            "stage": key[2], "relative_l2": rel,
                        }), flush=True)
                if not finite or relative > 0.025:
                    raise AssertionError(f"Q/K/V/O owner output fails equivalence: relative L2={relative}")
                points[str(length)] = {
                    "tp_qkvo_seconds": before, "owner_qkvo_seconds": after,
                    "relative_l2": relative, "max_absolute_error": delta.abs().max().item(),
                    "output_rms": observed.float().square().mean().sqrt().item(),
                    "cache_audit": dict(audits) if rank == 0 else None,
                }
                print("KIMI_ATTENTION_STAGE_POINT " + json.dumps({"rank": rank, "length": length,
                    "chunk_size": chunk_size, **{k:v for k,v in points[str(length)].items() if k != "cache_audit"}}), flush=True)
    finally:
        for layer, original in zip(layers, originals, strict=True):
            layer._run_self_attn = original
        for gate, original in original_gates:
            gate.forward = original
    return {"rank": rank, "world_size": world, "layers": len(layers),
            "moe_types": [type(layer.mlp).__name__ for layer in layers],
            "expert_parallel_size": layers[0].mlp.experts.moe_config.ep_size,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "points": points}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768, 65536])
    parser.add_argument("--chunk-size", type=int, choices=(2048, 16384), default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true",
                        help="trace small intermediate samples; not usable as benchmark timings")
    args = parser.parse_args()
    if any(length < 32768 or length % 16384 for length in args.lengths):
        parser.error("lengths must be multiples of 16K and at least 32K to exercise LoD, not only the exact prefix")
    os.environ.update(LOD_KIMI_REQUEST_OWNER_PREFILL="1", LOD_KIMI_OWNER_SHARD_RESIDUAL="0",
                      LOD_KIMI_OWNER_REUSE_TRANSPORT="1", VLLM_ALLOW_INSECURE_SERIALIZATION="1")
    from benchmarks._vllm import close_llm, llm_kwargs
    from vllm import LLM
    kwargs = llm_kwargs(checkpoint="tests/fixtures/kimi-k3-attention-moe", mode="two-tier",
                        max_model_len=max(args.lengths)+1, batch_size=8, tensor_parallel_size=8,
                        decode_context_parallel_size=8, gpu_memory_utilization=0.1,
                        full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="dummy", skip_tokenizer_init=True,
                  kv_cache_memory_bytes=268435456, enable_expert_parallel=True,
                  # The image's BF16 AITER instance generator cannot generate
                  # K3's SITU activation (its production INT4 path can). Use
                  # native Triton MoE for both layouts, without changing SITU.
                  kernel_config={"moe_backend": "triton"})
    llm = LLM(**kwargs)
    try:
        workers = llm.collective_rpc(run_probe, args=(args.lengths, args.chunk_size, args.diagnostic))
        if {w["rank"] for w in workers} != set(range(8)) or any(w["expert_parallel_size"] != 8 for w in workers):
            raise RuntimeError("missing worker or native EP8")
        points = {}
        for length in args.lengths:
            key = str(length)
            tp = max(w["points"][key]["tp_qkvo_seconds"] for w in workers)
            owner = max(w["points"][key]["owner_qkvo_seconds"] for w in workers)
            points[key] = {"tp_seconds": tp, "owner_seconds": owner, "speedup": tp/owner,
                           "max_relative_l2": max(w["points"][key]["relative_l2"] for w in workers)}
        result = {"scope": __doc__, "chunk_size": args.chunk_size, "layers": 2,
                  "owner_rank": 0, "pipeline_overlap": False, "cuda_graph_capture": False,
                  "moe_backend": "native vLLM Triton BF16 SITU, identical in both variants",
                  "moe_routes": "recorded control-prefill logits reused in both variants; router GEMMs still timed",
                  "quality_evidence": False,
                  "timing": "max of eight synchronized wall intervals; one measured pass after exact-shape warmup",
                  "geometry": {"hidden":7168,"heads":96,"latent":512,"direct":64,
                               "experts":32,"active_experts":4,"expert_latent":4096,"expert_intermediate":2048},
                  "points": points, "workers": workers}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print("KIMI_ATTENTION_STAGE_RESULT " + json.dumps(points), flush=True)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

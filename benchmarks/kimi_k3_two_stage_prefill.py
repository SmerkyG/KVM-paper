"""K3's first 24 layers: TP8 full attention versus owner-local LoD stages.

Native KDA convolution/recurrence/gated norm, MLA LoD, and attention-side
AttnRes are retained. Feed-forward branches, embeddings, and LM head are
excluded from both layouts. Random normalized weights are identical across
variants, not trained-model quality evidence. The baseline uses native AITER
full causal MLA on each query-head rank; there is no DCP. Owners keep full heads for their own
12 layers; the other TP ranks do no owner-stage work. This tests two stages
of a proposed eight-stage layout, not PP2 of
the entire model or a throughput claim for K3 with distributed MoE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace
import zlib


def microbatches(length: int, batch_size: int, chunk: int = 16384):
    """Interleave rows, preserving each row's chronological cache order."""
    if length <= 0 or length % chunk or batch_size <= 0:
        raise ValueError("positive batch and context divisible by chunk required")
    return tuple((row, start) for start in range(0, length, chunk)
                 for row in range(batch_size))


def assemble_kda_input(shards, projection: int, head_dim: int, heads: int):
    """Concatenate each head-partitioned field, retaining only one replicated f_a.

    A naive cat(rank0, rank1) interleaves Q/K/V/gate fields incorrectly and
    duplicates f_a. Padding rows are deliberately not carried into TP1.
    """
    import torch

    world = len(shards)
    if projection % world or heads % world:
        raise ValueError("KDA heads must divide across TP ranks")
    widths = (projection // world,) * 4 + (head_dim, heads // world)
    fields = [shard[:sum(widths)].split(widths, dim=0) for shard in shards]
    for other in fields[1:]:
        if not torch.equal(fields[0][4], other[4]):
            raise AssertionError("KDA's f_a projection must be replicated identically")
    return torch.cat([torch.cat([f[i] for f in fields], dim=0) if i != 4 else fields[0][i]
                      for i in range(6)], dim=0)


def handoff_bytes(tokens: int, hidden: int = 7168, *, include_input_bank: bool = False):
    """BF16 payload, not protocol overhead or a collective wire-byte estimate."""
    return tokens * hidden * 2 * (2 if include_input_bank else 1)


def kda_state_row(row: int):
    """vLLM reserves cache row 0 as NULL_BLOCK_ID; real rows start at 1."""
    if row < 0:
        raise ValueError("request row must be nonnegative")
    return row + 1


def completion_statistics(seconds):
    """Post-fill intervals, not total/cohort and not constant-context latency."""
    if (not seconds or any(not math.isfinite(t) or t < 0 for t in seconds)
            or any(b < a for a, b in zip(seconds, seconds[1:]))):
        raise ValueError("nonempty monotonic completion timestamps required")
    intervals = [b-a for a, b in zip(seconds, seconds[1:])]
    return dict(completion_seconds=list(seconds),
                post_first_completion_mean_interval_seconds=(
                    sum(intervals)/len(intervals) if intervals else None),
                completion_interval_seconds=intervals)


def validate_cache_audit(audit, *, length, batch, rank, full):
    """Reject partial contexts, missing requests, or hidden dense compression."""
    indices = [str(i) for i in range(3, 24, 4)
               if full or (rank < 2 and i//12 == rank)]
    if set(audit) != set(indices):
        raise AssertionError("missing or extra MLA layers in cache audit")
    for layer in audit.values():
        if set(layer) != {str(row) for row in range(batch)}:
            raise AssertionError("missing or extra requests in cache audit")
        for row in layer.values():
            if row["total_len"] != length:
                raise AssertionError("cache does not include the complete context")
            if full:
                if not row["full_attention"] or row["centroid_updates"] != 0:
                    raise AssertionError("dense control must not run LoD updates")
            elif row["coverage"] != length-256 or row["state_len"] <= 0:
                raise AssertionError("LoD cache is stale at the global update boundary")


def run_probe(worker, lengths, batches, include_input_bank=False, diagnostic=False):
    import torch
    import torch.nn.functional as F
    from aiter import flash_attn_varlen_func
    from vllm.distributed import get_tp_group
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices, prepare_chunk_offsets,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from vllm_lod_plugin.models.kimi_k3_request_prefill import attend_slice
    from vllm_lod_plugin.models.kimi_k3_sharded_prefill import gather_prefill

    core = next(m for m in worker.model_runner.model.modules()
                if type(m).__name__ == "KimiLinearModel")
    layers = list(core.layers)
    group = get_tp_group()
    rank, world = group.rank_in_group, group.world_size
    comm = group.device_communicator.pynccl_comm
    if world not in (2, 8) or len(layers) != 24 or comm is None or comm.disabled:
        raise RuntimeError("requires TP8 (or explicit historical TP2) and K3's first 24 layers")
    local_heads = 96 // world
    local_projection = 12288 // world
    baseline = f"full_tp{world}"
    numerical_control = f"lod_tp{world}"
    kinds = ["kda" if isinstance(layer.self_attn, KimiK3DeltaAttention) else "mla"
             for layer in layers]
    if kinds != ["mla" if (i + 1) % 4 == 0 else "kda" for i in range(24)]:
        raise AssertionError("wrong first-24 KDA/MLA placement")
    if any(type(layer.mlp).__name__ != "Identity" for layer in layers):
        raise AssertionError("MoE must be excluded equally, not unexpectedly timed")

    # Verify bottom-right causal alignment for a continued prefill before
    # accepting the dense control. This small FP32 oracle is untimed.
    with torch.inference_mode():
        generator = torch.Generator(device="cuda").manual_seed(4321)
        dq = torch.randn(3, local_heads, 192, dtype=torch.bfloat16, device="cuda", generator=generator)
        dk = torch.randn(7, local_heads, 192, dtype=torch.bfloat16, device="cuda", generator=generator)
        dv = torch.randn(7, local_heads, 128, dtype=torch.bfloat16, device="cuda", generator=generator)
        actual = flash_attn_varlen_func(
            dq, dk, dv, torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 7], dtype=torch.int32, device="cuda"), 3, 7,
            softmax_scale=192**-0.5, causal=True, return_lse=False)
        scores = torch.einsum("qhd,khd->hqk", dq.float(), dk.float()) * 192**-0.5
        causal = torch.arange(7, device="cuda")[None, :] <= torch.arange(4, 7, device="cuda")[:, None]
        expected = torch.einsum("hqk,khd->qhd", scores.masked_fill(~causal, -float("inf")).softmax(-1), dv.float())
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
        del dq, dk, dv, actual, expected, scores, causal

    # Normalize random weights, including KDA's otherwise uninitialized state
    # parameters. f_a is a replicated *field* inside a partitioned packed weight.
    with torch.inference_mode():
        for name, module in core.named_modules():
            for pname, parameter in module.named_parameters(recurse=False):
                seed = zlib.adler32((name + "/" + pname).encode())
                partitioned = any(getattr(module, a + "_size_per_partition", None)
                                  not in (None, getattr(module, a + "_size", None))
                                  for a in ("input", "output"))
                generator = torch.Generator(device="cuda").manual_seed(seed + (rank if partitioned else 0))
                if "norm" in name.lower() and parameter.ndim == 1:
                    parameter.fill_(1)
                elif pname in ("A_log", "dt_bias"):
                    parameter.zero_()
                elif parameter.ndim >= 2:
                    parameter.normal_(0, float(getattr(module, "input_size", parameter.shape[-1])) ** -0.5,
                                      generator=generator)
                else:
                    parameter.zero_()
        for layer, kind in zip(layers, kinds, strict=True):
            if kind == "kda":
                kda = layer.self_attn
                offset = 4 * kda.local_projection_size
                seed = zlib.adler32((str(layer.layer_idx) + "/replicated_f_a").encode())
                generator = torch.Generator(device="cuda").manual_seed(seed)
                kda.in_proj_qkvgfab.weight[offset:offset+128].normal_(
                    0, 7168 ** -0.5, generator=generator)
                if kda.in_proj_padding:
                    kda.in_proj_qkvgfab.weight[-kda.in_proj_padding:].zero_()

    max_batch, chunk = max(batches), 16384
    prepared, owner_kda = {}, {}
    for layer, kind in zip(layers, kinds, strict=True):
        index, owner = layer.layer_idx, layer.layer_idx // 12
        if kind == "mla":
            wrapper = layer.self_attn.mla_attn
            pool = wrapper.mla_attn._vllm_lod_pool
            pool.engine._lod_kimi_reduce_prefill_routes = True
            qw = gather_prefill(group, wrapper.q_b_proj.weight, dim=0)
            ow = gather_prefill(group, wrapper.o_proj.weight, dim=1)
            gw = gather_prefill(group, wrapper.g_proj.weight, dim=0)
            kv = wrapper.kv_b_proj.weight.view(local_heads, 256, 512)
            uk = gather_prefill(group, kv[:, :128].contiguous(), dim=0)
            uv = gather_prefill(group, kv[:, 128:].transpose(1, 2).contiguous(), dim=0)
            prepared[index] = dict(pool=pool, wrapper=wrapper,
                full=(qw, ow, gw, uk, uv) if rank == owner else None,
                local=(kv[:, :128], kv[:, 128:].transpose(1, 2)),
                full_history=torch.empty(max_batch, max(lengths), 1, 576,
                                         dtype=torch.bfloat16, device="cuda"),
                dense_lengths={})
            del qw, ow, gw, uk, uv
        else:
            original = layer.self_attn
            iw = gather_prefill(group, original.in_proj_qkvgfab.weight, dim=0)
            iw = assemble_kda_input(iw.chunk(world, dim=0), 12288, 128, 96)
            fw = gather_prefill(group, original.f_b_proj.weight, dim=0)
            ow = gather_prefill(group, original.o_proj.weight, dim=1)
            # Packed conv is [Q-local,K-local,V-local], not rank-major QKV.
            cw = gather_prefill(group, original.conv1d.weight.view(3, local_projection, 1, 4), dim=1).reshape(36864, 1, 4)
            al = gather_prefill(group, original.A_log, dim=0)
            dt = gather_prefill(group, original.dt_bias, dim=0)
            if rank == owner:
                full = SimpleNamespace(prefix=original.prefix, local_num_heads=96,
                    head_dim=128, local_projection_size=12288, in_proj_padding=0,
                    gate_lower_bound=original.gate_lower_bound, use_fused_chunk=original.use_fused_chunk,
                    A_log=al, dt_bias=dt, conv1d=SimpleNamespace(weight=cw, bias=None),
                    decode_conv1d_weight=None, decode_norm_weight=None, o_norm=original.o_norm)
                full.in_proj_qkvgfab = lambda x, w=iw: (F.linear(x, w), None)
                full.f_b_proj = lambda x, w=fw: (F.linear(x, w), None)
                full.o_proj = lambda x, w=ow: (F.linear(x, w), None)
                full._forward = MethodType(KimiK3DeltaAttention._forward, full)
                owner_kda[index] = full
            del iw, fw, ow, cw, al, dt
        if kind == "kda":
            for obj, heads in ((layer.self_attn, local_heads), (owner_kda.get(index), 96)):
                if obj is not None:
                    conv = torch.zeros(max_batch+1, 3*heads*128, 3, dtype=torch.bfloat16, device="cuda")
                    if not is_conv_state_dim_first():
                        conv = conv.transpose(-1, -2).contiguous()
                    obj.kv_cache = (conv, torch.zeros(max_batch+1, heads, 128, 128, dtype=torch.float32, device="cuda"))

    generator = torch.Generator(device="cuda").manual_seed(1234)
    inputs = torch.randn(max_batch, max(lengths), 7168, dtype=torch.bfloat16,
                         device="cuda", generator=generator)
    positions = torch.arange(max(lengths), device="cuda")
    cu_cpu = torch.tensor([0, chunk], dtype=torch.int32)
    cu = cu_cpu.to("cuda")
    cu_keys = {start: torch.tensor([0, start+chunk], dtype=torch.int32, device="cuda")
               for start in range(0, max(lengths), chunk)}
    nums, bp, tp = compute_causal_conv1d_metadata(cu_cpu, device=torch.device("cuda", torch.cuda.current_device()))
    ci = prepare_chunk_indices(cu_cpu, FLA_CHUNK_SIZE).to("cuda")
    co = prepare_chunk_offsets(cu_cpu, FLA_CHUNK_SIZE).to("cuda")
    metadata = {}
    for row in range(max_batch):
        for has_initial in (False, True):
            metadata[row, has_initial] = GDNAttentionMetadata(
                num_prefills=1, num_prefill_tokens=chunk, num_decodes=0,
                num_decode_tokens=0, num_spec_decodes=0, num_spec_decode_tokens=0,
                num_actual_tokens=chunk, has_initial_state=torch.tensor([has_initial], device="cuda"),
                non_spec_query_start_loc=cu, non_spec_state_indices_tensor=torch.tensor([kda_state_row(row)], dtype=torch.int32, device="cuda"),
                chunk_indices=ci, chunk_offsets=co, nums_dict=nums,
                batch_ptr=bp, token_chunk_offset_ptr=tp)
    context = dict(variant=baseline, row=0, start=0, length=0)
    original_attention = [layer._run_self_attn for layer in layers]

    def attention(layer, position, hidden):
        index = layer.layer_idx
        owned = context["variant"] in ("sequential", "pipeline")
        if kinds[index] == "kda":
            obj = owner_kda[index] if owned else layer.self_attn
            return KimiK3DeltaAttention.forward(obj, hidden, position)
        p = prepared[index]
        wrapper, pool = p["wrapper"], p["pool"]
        fused = wrapper.fused_qkv_a_proj(hidden)[0]
        qc, kv = fused.split((1536, 576), dim=-1)
        latent, direct = kv.split((512, 64), dim=-1)
        qc, latent = wrapper._normalize_q_kv(qc, latent)
        qw, ow, gw, uk, uv = p["full"] if owned else (None, None, None, *p["local"])
        q = F.linear(qc, qw) if owned else wrapper.q_b_proj(qc)[0]
        q = q.view(chunk, 96 if owned else local_heads, 192).contiguous()
        record = torch.cat((latent, direct), dim=-1).unsqueeze(1)
        if context["variant"] == baseline:
            row, start = context["row"], context["start"]
            if p["dense_lengths"].get(row, 0) != start:
                raise AssertionError("full-attention history is not chronological")
            total = start + chunk
            p["full_history"][row, start:total].copy_(record)
            history = p["full_history"][row, :total, 0]
            # Exactly the native expanded MLA geometry: one packed W_UKV
            # projection, D192 keys/queries, D128 values, no centroids/routing.
            projected = wrapper.kv_b_proj(history[:, :512])[0].view(total, local_heads, 256)
            keys = torch.cat((projected[..., :128],
                              history[:, None, 512:].expand(-1, local_heads, -1)), dim=-1)
            out = flash_attn_varlen_func(
                q=q, k=keys, v=projected[..., 128:],
                cu_seqlens_q=cu, cu_seqlens_k=cu_keys[start],
                max_seqlen_q=chunk, max_seqlen_k=total,
                softmax_scale=float(pool.engine.scaling), causal=True,
                return_lse=False,
            ).reshape(chunk, -1)
            p["dense_lengths"][row] = total
        else:
            out = attend_slice(pool, context["row"], q, record, uk, uv,
                previous=context["start"], prompt=context["length"]).reshape(chunk, -1)
        if owned:
            return F.linear(out * F.linear(hidden, gw).sigmoid(), ow)
        return wrapper.o_proj(out * wrapper.g_proj(hidden)[0].sigmoid())[0]

    for layer in layers:
        layer._run_self_attn = MethodType(attention, layer)

    # Fixed ring slots bound in-flight bank/payload memory. Receivers retain
    # the original inputs, so bank entry 0 need not travel at this boundary.
    slots = 2
    fields = 2 if include_input_bank else 1
    wire = [inputs.new_empty(fields, chunk, 7168) for _ in range(slots)]
    banks = [inputs.new_empty(chunk, 2, 7168) for _ in range(slots)]
    ack = torch.zeros(1, dtype=torch.int32, device="cuda")
    transfer = torch.cuda.Stream()
    compute = torch.cuda.current_stream()
    ready = [torch.cuda.Event() for _ in range(slots)]
    released = [torch.cuda.Event() for _ in range(slots)]
    timing_start = torch.cuda.Event(enable_timing=True)
    completions = [torch.cuda.Event(enable_timing=True)
                   for _ in microbatches(max(lengths), max_batch, chunk)]
    statistics = {"reductions": 0, "reduction_input_bytes": 0}
    native_reduce = group.all_reduce

    def count_reduce(tensor, *args, **kwargs):
        statistics["reductions"] += 1
        statistics["reduction_input_bytes"] += tensor.numel() * tensor.element_size()
        return native_reduce(tensor, *args, **kwargs)

    group.all_reduce = count_reduce

    def reset():
        for obj in [layer.self_attn for layer, kind in zip(layers, kinds, strict=True) if kind == "kda"] + list(owner_kda.values()):
            for cache in obj.kv_cache:
                cache.zero_()
        for p in prepared.values():
            p["pool"]._kimi_request_owner_rows = {}
            p["pool"].engine.reset_runtime_cache()
            p["dense_lengths"].clear()

    def block(begin, end, hidden, bank, row, start):
        context.update(row=row, start=start)
        m = metadata[row, start > 0]
        attn_metadata = {layers[i].self_attn.prefix: m for i in range(begin, end) if kinds[i] == "kda"}
        delta = None
        with set_forward_context(attn_metadata, worker.vllm_config, num_tokens=chunk):
            for layer in layers[begin:end]:
                hidden, bank, delta = layer(positions[start:start+chunk], hidden, bank, prefix_delta=delta)
                if diagnostic:
                    current = hidden if delta is None else delta
                    finite = bool(torch.isfinite(current).all())
                    print("KIMI_FIRST24_NUMERICS " + json.dumps(dict(rank=rank,
                        variant=context["variant"], row=row, start=start,
                        layer=layer.layer_idx, kind=kinds[layer.layer_idx],
                        finite=finite, rms=current.float().square().mean().sqrt().item())), flush=True)
                    if not finite:
                        raise AssertionError(f"first non-finite {context['variant']} layer {layer.layer_idx} ({kinds[layer.layer_idx]})")
        return hidden if delta is None else hidden + delta

    def execute(variant, length, batch):
        context.update(variant=variant, length=length)
        # One latent KV head, but TP has 96/world query heads and owners 96.
        for p in prepared.values():
            heads = 96 if variant in ("sequential", "pipeline") else local_heads
            p["pool"].engine.num_key_value_groups = heads
            p["pool"].engine.config.num_attention_heads = heads
        reset()
        # Keep outputs for complete validation, not only token-ID comparisons.
        parallel = variant in (baseline, numerical_control)
        output = inputs.new_empty(batch, length, 7168) if (rank == 1 or parallel) else None
        timing_start.record(compute)
        for step, (row, start) in enumerate(microbatches(length, batch, chunk)):
            slot = step % slots
            if parallel:
                out = block(0, 24, inputs[row, start:start+chunk], banks[slot], row, start)
                output[row, start:start+chunk].copy_(out)
            elif rank == 0:
                if variant == "pipeline" and step >= slots:
                    compute.wait_event(released[slot])
                out = block(0, 12, inputs[row, start:start+chunk], banks[slot], row, start)
                wire[slot][0].copy_(out)
                if include_input_bank:
                    wire[slot][1].copy_(banks[slot][:, 0])
                if variant == "pipeline":
                    ready[slot].record(compute)
                    transfer.wait_event(ready[slot])
                    comm.send(wire[slot], 1, stream=transfer)
                    released[slot].record(transfer)
                else:
                    comm.send(wire[slot], 1)
                    comm.recv(ack, 1)
            elif rank == 1:
                if variant == "pipeline":
                    if step >= slots:
                        transfer.wait_event(released[slot])
                    comm.recv(wire[slot], 0, stream=transfer)
                    ready[slot].record(transfer)
                    compute.wait_event(ready[slot])
                else:
                    comm.recv(wire[slot], 0)
                banks[slot][:, 0].copy_(wire[slot][1] if include_input_bank else inputs[row, start:start+chunk])
                out = block(12, 24, wire[slot][0], banks[slot], row, start)
                output[row, start:start+chunk].copy_(out)
                if variant == "pipeline":
                    released[slot].record(compute)
                else:
                    comm.send(ack, 0)
            if parallel or rank == 1:
                completions[step].record(compute)
        return output

    def audit(variant):
        if variant == baseline:
            return {str(i): {str(row): {"total_len": length,
                                       "full_attention": True, "centroid_updates": 0}
                             for row, length in p["dense_lengths"].items()}
                    for i, p in prepared.items()}
        selected = (range(24) if variant == baseline else
                    range(rank*12, rank*12+12) if rank < 2 else ())
        return {str(i): {str(row): {
            "total_len": data["total"], "coverage": int(data["cache"].state["coverage"]),
            "state_len": int(data["cache"].state["state_len"]),
        } for row, data in prepared[i]["pool"]._kimi_request_owner_rows.items()}
            for i in selected if kinds[i] == "mla"}

    def measure(variant, length, batch):
        execute(variant, length, batch)
        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        statistics.update(reductions=0, reduction_input_bytes=0)
        begin = time.perf_counter()
        output = execute(variant, length, batch)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        point = dict(seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     **statistics, cache_audit=audit(variant))
        if variant == baseline or rank == 1:
            point.update(completion_statistics([
                timing_start.elapsed_time(event)/1000
                for event in completions[:len(microbatches(length, batch, chunk))]]))
        return output, point

    points = {}
    try:
        with torch.inference_mode():
            for length in lengths:
                for batch in batches:
                    key = f"{length}/B{batch}"
                    reference, tp_point = measure(baseline, length, batch)
                    if not bool(torch.isfinite(reference).all()):
                        raise AssertionError(f"{baseline} control is non-finite; layout comparison is invalid")
                    variants = {baseline: tp_point}
                    # Untimed same-LoD control isolates repartition rounding
                    # from the intentional full-to-LoD approximation. The old
                    # 5% bound applies only to this arithmetic/layout check.
                    lod_reference = execute(numerical_control, length, batch)
                    torch.cuda.synchronize()
                    group.barrier()
                    if not bool(torch.isfinite(lod_reference).all()):
                        raise AssertionError("untimed TP LoD numerical control is non-finite")
                    sequential = None
                    for variant in ("sequential", "pipeline"):
                        output, point = measure(variant, length, batch)
                        if rank == 1:
                            relative = ((output.float()-reference.float()).square().sum()
                                / reference.float().square().sum().clamp_min(1e-20)).sqrt().item()
                            layout_relative = ((output.float()-lod_reference.float()).square().sum()
                                / lod_reference.float().square().sum().clamp_min(1e-20)).sqrt().item()
                            if not bool(torch.isfinite(output).all()) or not (layout_relative <= 0.05):
                                raise AssertionError(f"{variant} fails same-LoD layout check: L2={layout_relative}")
                            point[f"relative_l2_to_{baseline}"] = relative
                            point[f"relative_l2_to_{numerical_control}"] = layout_relative
                            if variant == "sequential":
                                sequential = output
                            else:
                                if not torch.equal(output, sequential):
                                    raise AssertionError("pipeline differs from sequential owner arithmetic")
                                point["bitwise_equal_to_sequential"] = True
                        variants[variant] = point
                        print("KIMI_FIRST24_POINT " + json.dumps({"rank":rank,"length":length,
                            "batch":batch,"variant":variant,"seconds":point["seconds"]}), flush=True)
                    points[key] = variants
                    del reference, lod_reference, sequential, output
    finally:
        group.all_reduce = native_reduce
        for layer, original in zip(layers, original_attention, strict=True):
            layer._run_self_attn = original
    return dict(rank=rank, world_size=world, layer_kinds=kinds, points=points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768, 65536])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--tensor-parallel-size", type=int, choices=(2, 8), default=8,
                        help="TP8 is the intended baseline; TP2 only reproduces the historical pilot")
    parser.add_argument("--include-input-bank", action="store_true",
                        help="transfer original input bank entry too, instead of receiver retention")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true",
                        help="synchronize/check every layer; not valid for benchmark timings")
    args = parser.parse_args()
    for length in args.lengths:
        microbatches(length, min(args.batches))
    # The production request-owner switch selects a different B8/TP8/DCP8
    # serving experiment. This direct first-24 worker probe needs no such switch.
    os.environ.pop("LOD_KIMI_REQUEST_OWNER_PREFILL", None)
    os.environ.update(LOD_KIMI_OWNER_SHARD_RESIDUAL="0",
                      VLLM_ALLOW_INSECURE_SERIALIZATION="1")
    from benchmarks._vllm import llm_kwargs, close_llm
    from vllm import LLM
    kwargs = llm_kwargs(checkpoint="tests/fixtures/kimi-k3-first24", mode="two-tier",
        max_model_len=max(args.lengths)+1, batch_size=max(args.batches), tensor_parallel_size=args.tensor_parallel_size,
        decode_context_parallel_size=1, gpu_memory_utilization=0.2,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="dummy", skip_tokenizer_init=True,
                  enforce_eager=True, kv_cache_memory_bytes=2147483648)
    llm = LLM(**kwargs)
    try:
        workers = llm.collective_rpc(run_probe, args=(args.lengths, args.batches, args.include_input_bank, args.diagnostic))
        if {w["rank"] for w in workers} != set(range(args.tensor_parallel_size)):
            raise AssertionError("missing worker")
        owner = next(w for w in workers if w["rank"] == 1)
        baseline = f"full_tp{args.tensor_parallel_size}"
        numerical_control = f"lod_tp{args.tensor_parallel_size}"
        points = {}
        for length in args.lengths:
            for batch in args.batches:
                key = f"{length}/B{batch}"
                for worker in workers:
                    for variant in (baseline, "sequential", "pipeline"):
                        validate_cache_audit(worker["points"][key][variant]["cache_audit"],
                            length=length, batch=batch, rank=worker["rank"], full=variant == baseline)
                points[key] = {variant: {
                    "seconds": max(w["points"][key][variant]["seconds"] for w in workers),
                    "peak_allocated_bytes": max(w["points"][key][variant]["peak_allocated_bytes"] for w in workers),
                    "sum_collective_input_bytes": sum(w["points"][key][variant]["reduction_input_bytes"] for w in workers),
                    **{k:v for k,v in owner["points"][key][variant].items()
                       if k in (f"relative_l2_to_{baseline}", f"relative_l2_to_{numerical_control}", "bitwise_equal_to_sequential",
                                "completion_seconds", "post_first_completion_mean_interval_seconds",
                                "completion_interval_seconds")},
                } for variant in (baseline, "sequential", "pipeline")}
                points[key]["stage_handoff_payload_bytes"] = handoff_bytes(length*batch,
                    include_input_bank=args.include_input_bank)
        result = dict(scope=__doc__, points=points, workers=workers,
            stage_layers=12, stages=2, chunk_size=16384, fixed_transport_slots=2,
            input_bank_retained=not args.include_input_bank,
            random_weights=True, quality_evidence=False, moe_included=False,
            cuda_graph_capture=False,
            baseline_tensor_parallel_size=args.tensor_parallel_size,
            baseline_decode_context_parallel_size=1,
            baseline_attention="native AITER full causal MLA, no LoD centroid construction or routing",
            dense_causal_alignment_check="passed against untimed small FP32 oracle",
            untimed_layout_control=numerical_control,
            diagnostic=args.diagnostic, valid_timing=not args.diagnostic,
            timing="max of all TP worker synchronized wall intervals; one warmed pass per variant; completion events report post-fill spacing separately",
            traffic_note="collective input bytes are logical payload, not measured wire bytes; p2p payload excludes protocol overhead")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print("KIMI_FIRST24_RESULT " + json.dumps(points), flush=True)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

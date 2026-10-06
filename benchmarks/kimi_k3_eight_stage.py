"""Experimental eight-stage K3 attention pipeline with native distributed FFNs.

Compare owner-local KDA+MLA with owner MLA/native TP8 KDA, keeping the
trained EP8 INT4 MoE at its original place between every attention layer.
This is a prefill layout benchmark, not an installed vLLM serving scheduler.
Embedding, both AttnRes mixes, FFNs, transfers and LoD updates are timed;
weight assembly, final output normalization/LM head and sampling are not.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

from benchmarks.kimi_k3_mla_owners import real_token_rows
from benchmarks.kimi_k3_two_stage_prefill import assemble_kda_input, microbatches


def stage_ranges(layer_count: int, stages: int = 8):
    """Eight real layer stages; full K3 boundaries align with 12-layer banks."""
    if stages != 8 or layer_count < stages:
        raise ValueError("requires eight nonempty stages")
    width = (layer_count + stages - 1)//stages
    ranges = tuple((s*width, min((s+1)*width, layer_count)) for s in range(stages))
    if any(a >= b for a, b in ranges):
        raise ValueError("the final stage would be empty")
    return ranges


@dataclass(frozen=True)
class Task:
    micro: int
    stage: int
    layer: int


def service_rounds(layer_count: int, count: int, *, pipelined: bool):
    """One deterministic collective order; distinct stages run attention ahead.

    Downstream-first service lets stage buffers be released before upstream
    handoff. Each stage sees microbatches in the same chronological order.
    FFNs remain one shared EP8 service, never eight unordered collective groups.
    """
    ranges = stage_ranges(layer_count)
    if count < 1:
        raise ValueError("positive microbatch count required")
    if not pipelined:
        return tuple((Task(m, s, layer),) for m in range(count)
                     for s, (a, b) in enumerate(ranges) for layer in range(a, b))
    active, next_micro, rounds = [None]*8, 0, []
    while next_micro < count or any(t is not None for t in active):
        if active[0] is None and next_micro < count:
            active[0] = Task(next_micro, 0, 0)
            next_micro += 1
        tasks = tuple(t for t in reversed(active) if t is not None)
        rounds.append(tasks)
        incoming = []
        for task in tasks:
            if task.layer + 1 < ranges[task.stage][1]:
                active[task.stage] = Task(task.micro, task.stage, task.layer+1)
            else:
                active[task.stage] = None
                if task.stage < 7:
                    incoming.append(Task(task.micro, task.stage+1, ranges[task.stage+1][0]))
        for task in incoming:
            if active[task.stage] is not None:
                raise AssertionError("handoff would overwrite a live stage")
            active[task.stage] = task
    return tuple(rounds)


def completion_summary(times, *, chunk: int, stages: int, pipelined: bool):
    """Discard drain intervals rather than labeling cohort/M steady throughput."""
    if not times or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("ordered completion times required")
    stop = len(times)-(stages-1) if pipelined else len(times)
    window = times[:stop] if stop > 1 else []
    spacing = (window[-1]-window[0])/(len(window)-1) if window else None
    return dict(completion_seconds=times, steady_completion_count=len(window),
                steady_microbatch_interval_seconds=spacing,
                steady_tokens_per_second=chunk/spacing if spacing and spacing > 0 else None,
                steady_note="first completion through last completion before pipeline drain; context-dependent finite stream")


def run_probe(worker, lengths, batch, chunk, layer_count, token_rows, validation_only, cohorts, progress_path=None):
    import torch
    import torch.nn.functional as F
    from aiter import flash_attn_varlen_func
    from vllm.distributed import get_tp_group
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
    from vllm.models.kimi_k3.amd.linear import _apply_attn_res
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from vllm_lod_plugin.models.kimi_k3_request_prefill import attend_slice
    from vllm_lod_plugin.models.kimi_k3_sharded_prefill import gather_prefill

    core = next(m for m in worker.model_runner.model.modules() if type(m).__name__ == "KimiLinearModel")
    layers = list(core.layers)[:layer_count or None]
    count = len(layers)
    ranges = stage_ranges(count)
    group = get_tp_group()
    rank, world = group.rank_in_group, group.world_size
    comm = group.device_communicator.pynccl_comm
    if world != 8 or comm is None or comm.disabled:
        raise RuntimeError("requires native TP8/EP8 and enabled RCCL")
    if type(layers[0].mlp).__name__ != "KimiMLP" or any(type(l.mlp).__name__ != "KimiMoE" for l in layers[1:]):
        raise AssertionError("every trained FFN must remain installed")
    if layers[1].mlp.experts.moe_config.ep_size != 8:
        raise AssertionError("MoE must remain native EP8")
    owner_of = {i:s for s, (a, b) in enumerate(ranges) for i in range(a, b)}
    kinds = ["kda" if isinstance(l.self_attn, KimiK3DeltaAttention) else "mla" for l in layers]
    maximum = max(max(lengths), 512)
    ids = torch.tensor(token_rows, dtype=torch.long, device="cuda")
    positions = torch.arange(maximum, device="cuda")
    original_kda, owner_kda, mla = [], {}, {}

    def kda_cache(heads):
        conv = torch.zeros(batch+1, 3*heads*128, 3, device="cuda", dtype=torch.bfloat16)
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2).contiguous()
        return conv, torch.zeros(batch+1, heads, 128, 128, device="cuda", dtype=torch.float32)

    # Only each stage's owner retains its complete QKVO weights. These are
    # new tensors assembled outside timing; no daemon parameter is modified.
    for index, (layer, kind) in enumerate(zip(layers, kinds, strict=True)):
        if kind == "kda":
            obj = layer.self_attn
            original_kda.append((obj, obj.kv_cache))
            obj.kv_cache = None
            iw = gather_prefill(group, obj.in_proj_qkvgfab.weight, dim=0)
            iw = assemble_kda_input(iw.chunk(world, 0), 12288, 128, 96)
            fw = gather_prefill(group, obj.f_b_proj.weight, dim=0)
            ow = gather_prefill(group, obj.o_proj.weight, dim=1)
            cw = gather_prefill(group, obj.conv1d.weight.view(3, 1536, 1, 4), dim=1).reshape(36864, 1, 4)
            al = gather_prefill(group, obj.A_log, dim=0)
            dt = gather_prefill(group, obj.dt_bias, dim=0)
            if rank == owner_of[index]:
                full = SimpleNamespace(prefix=obj.prefix, local_num_heads=96,
                    head_dim=128, local_projection_size=12288, in_proj_padding=0,
                    gate_lower_bound=obj.gate_lower_bound, use_fused_chunk=obj.use_fused_chunk,
                    A_log=al, dt_bias=dt, conv1d=SimpleNamespace(weight=cw, bias=None),
                    decode_conv1d_weight=None, decode_norm_weight=None, o_norm=obj.o_norm)
                full.in_proj_qkvgfab = lambda x, w=iw: (F.linear(x, w), None)
                full.f_b_proj = lambda x, w=fw: (F.linear(x, w), None)
                full.o_proj = lambda x, w=ow: (F.linear(x, w), None)
                full._forward = MethodType(KimiK3DeltaAttention._forward, full)
                full.kv_cache = None
                owner_kda[index] = full
            del iw, fw, ow, cw, al, dt
        else:
            w = layer.self_attn.mla_attn
            pool = w.mla_attn._vllm_lod_pool
            pool.engine._lod_kimi_reduce_prefill_routes = True
            qw = gather_prefill(group, w.q_b_proj.weight, dim=0)
            ow = gather_prefill(group, w.o_proj.weight, dim=1)
            gw = gather_prefill(group, w.g_proj.weight, dim=0)
            kv = w.kv_b_proj.weight.view(12, 256, 512)
            uk = gather_prefill(group, kv[:, :128].contiguous(), dim=0)
            uv = gather_prefill(group, kv[:, 128:].transpose(1, 2).contiguous(), dim=0)
            mla[index] = dict(wrapper=w, pool=pool, history=None, lengths={},
                              full=(qw, ow, gw, uk, uv) if rank == owner_of[index] else None)
            del qw, ow, gw, uk, uv
        if (index+1) % 12 == 0:
            # Avoid retaining inactive multi-layer gather allocations while
            # attaching a near-capacity trained model. This is untimed.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    print("KIMI_PIPELINE_PREPARED " + json.dumps(dict(rank=rank, layers=count, stages=ranges)), flush=True)

    bank_count = (count+11)//12
    # Two explicit stage slots; transfer includes prefix, pending FFN delta,
    # and the valid AttnRes bank, not merely the current normalized hidden.
    payload = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    payloads = {}
    local_bank_count = (ranges[rank][1]+11)//12
    attn_input = payload.new_empty(chunk, 7168)
    moe_input = payload.new_empty(chunk, 7168)
    service_input = payload.new_empty(chunk, 7168)
    compute, service = torch.cuda.Stream(), torch.cuda.current_stream()
    prepared_event, ff_done, arrival = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
    native_metadata, owner_metadata, cu = {}, {}, {}
    for n in {chunk, 256}:
        cpu = torch.tensor([0, n], dtype=torch.int32)
        gpu = cpu.to("cuda")
        cu[n] = gpu
        nums, bp, tp = compute_causal_conv1d_metadata(cpu, device=payload.device)
        ci = prepare_chunk_indices(cpu, FLA_CHUNK_SIZE).to("cuda")
        co = prepare_chunk_offsets(cpu, FLA_CHUNK_SIZE).to("cuda")
        for row in range(batch):
            for initial in (False, True):
                m = GDNAttentionMetadata(num_prefills=1, num_prefill_tokens=n,
                    num_decodes=0, num_decode_tokens=0, num_spec_decodes=0,
                    num_spec_decode_tokens=0, num_actual_tokens=n,
                    has_initial_state=torch.tensor([initial], device="cuda"),
                    non_spec_query_start_loc=gpu,
                    non_spec_state_indices_tensor=torch.tensor([row+1], device="cuda", dtype=torch.int32),
                    chunk_indices=ci, chunk_offsets=co, nums_dict=nums, batch_ptr=bp, token_chunk_offset_ptr=tp)
                native_metadata[n, row, initial] = owner_metadata[n, row, initial] = m
    torch.cuda.synchronize()
    group.barrier()
    compute.wait_stream(service)

    def reset(variant, length, n):
        torch.cuda.synchronize()
        if variant == "full_tp8":
            # The serial control needs one bank, not both owner ring slots.
            payloads.clear()
        elif n not in payloads:
            payloads[n] = payload.new_empty(2, 2+local_bank_count, n, 7168)
        # The two layouts never use both recurrent-state representations.
        # Drop the inactive one before allocating any FFN/cache workspace.
        for obj, _ in original_kda:
            if variant == "owner_kda":
                obj.kv_cache = None
        for obj in owner_kda.values():
            if variant != "owner_kda":
                obj.kv_cache = None
        for obj, _ in original_kda:
            if variant != "owner_kda":
                if obj.kv_cache is None:
                    obj.kv_cache = kda_cache(12)
                for cache in obj.kv_cache:
                    cache.zero_()
        for obj in owner_kda.values():
            if variant == "owner_kda":
                if obj.kv_cache is None:
                    obj.kv_cache = kda_cache(96)
                for cache in obj.kv_cache:
                    cache.zero_()
        for p in mla.values():
            p["pool"]._kimi_request_owner_rows = {}
            p["pool"].engine.reset_runtime_cache()
            p["pool"].engine.num_key_value_groups = 12 if variant == "full_tp8" else 96
            p["pool"].engine.config.num_attention_heads = 12 if variant == "full_tp8" else 96
            p["lengths"].clear()
            p["history"] = (payload.new_empty(batch, length, 576) if variant == "full_tp8" else None)
        for storage in payloads.values():
            storage.zero_()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        group.barrier()
        compute.wait_stream(service)

    def owned_attention(index, hidden, row, start, length):
        if kinds[index] == "kda":
            return KimiK3DeltaAttention.forward(owner_kda[index], hidden, positions[start:start+hidden.size(0)])
        p = mla[index]
        w, pool = p["wrapper"], p["pool"]
        if start == 0:
            # A later cohort reuses this slot only after the previous request
            # has passed this layer. Other stages keep their own old caches.
            pool._kimi_request_owner_rows.pop(row, None)
        qc, kv = w.fused_qkv_a_proj(hidden)[0].split((1536, 576), -1)
        latent, direct = kv.split((512, 64), -1)
        qc, latent = w._normalize_q_kv(qc, latent)
        qw, ow, gw, uk, uv = p["full"]
        query = F.linear(qc, qw).view(hidden.size(0), 96, 192).contiguous()
        record = torch.cat((latent, direct), -1).unsqueeze(1)
        out = attend_slice(pool, row, query, record, uk, uv, previous=start, prompt=length)
        return F.linear(out.reshape(hidden.size(0), 12288)*F.linear(hidden, gw).sigmoid(), ow)

    def dense_attention(index, hidden, row, start, length):
        if kinds[index] == "kda":
            return layers[index].self_attn(hidden_states=hidden, positions=positions[start:start+hidden.size(0)])
        p = mla[index]
        w = p["wrapper"]
        n, total = hidden.size(0), start+hidden.size(0)
        qc, kv = w.fused_qkv_a_proj(hidden)[0].split((1536, 576), -1)
        latent, direct = kv.split((512, 64), -1)
        qc, latent = w._normalize_q_kv(qc, latent)
        query = w.q_b_proj(qc)[0].view(n, 12, 192).contiguous()
        if start == 0:
            p["lengths"].pop(row, None)
        if p["lengths"].get(row, 0) != start:
            raise AssertionError("dense history is not chronological")
        p["history"][row, start:total].copy_(torch.cat((latent, direct), -1))
        history = p["history"][row, :total]
        projected = w.kv_b_proj(history[:, :512])[0].view(total, 12, 256)
        keys = torch.cat((projected[..., :128], history[:, None, 512:].expand(-1, 12, -1)), -1)
        out = flash_attn_varlen_func(query, keys, projected[..., 128:], cu[n],
            torch.tensor([0, total], device="cuda", dtype=torch.int32), n, total,
            softmax_scale=float(p["pool"].engine.scaling), causal=True, return_lse=False)
        p["lengths"][row] = total
        return w.o_proj(out.reshape(n, 1536)*w.g_proj(hidden)[0].sigmoid())[0]

    def audit(variant, length):
        records = {}
        for i, p in mla.items():
            if variant == "full_tp8":
                rows = {str(r): dict(total_len=n, centroid_updates=0) for r, n in p["lengths"].items()}
            else:
                rows = {str(r): dict(total_len=d["total"], coverage=int(d["cache"].state["coverage"]),
                                    state_len=int(d["cache"].state["state_len"]))
                        for r, d in p["pool"]._kimi_request_owner_rows.items()}
            expected = {str(r) for r in range(batch)} if variant == "full_tp8" or owner_of[i] == rank else set()
            if set(rows) != expected or any(d["total_len"] != length for d in rows.values()):
                raise AssertionError("MLA request/cache ownership or global history is incorrect")
            if variant != "full_tp8" and any(d["coverage"] != length-256 for d in rows.values()):
                raise AssertionError("global prefill cadence/coverage is incorrect")
            records[str(i)] = rows
        return records

    def execute(variant, length, n, *, pipeline=True, sample_all=False):
        reset(variant, length, n)
        cohort_micros = microbatches(length, batch, n)
        micros = cohort_micros*cohorts
        def request_id(micro):
            return micro//len(cohort_micros)*batch + micros[micro][0]
        rounds = service_rounds(count, len(micros), pipelined=pipeline)
        buffers = payloads.get(n)
        dense_bank = payload.new_empty(n, bank_count, 7168) if variant == "full_tp8" else None
        timings = [torch.cuda.Event(enable_timing=True) for _ in micros]
        admissions = [torch.cuda.Event(enable_timing=True) for _ in range(batch*cohorts)]
        row_done = [torch.cuda.Event(enable_timing=True) for _ in range(batch*cohorts)]
        outputs = payload.new_empty(len(micros), n if sample_all else 8, 7168) if rank == 7 or variant == "full_tp8" else None
        started = torch.cuda.Event(enable_timing=True)
        launched = set()
        start_wall = time.perf_counter()
        started.record(service)

        def prepare(task):
            if rank != task.stage or task in launched:
                return
            launched.add(task)
            row, start = micros[task.micro]
            layer = layers[task.layer]
            storage = buffers[task.micro % 2]
            prefix = storage[0]
            bank = storage[2:].permute(1, 0, 2)
            m = owner_metadata[n, row, start > 0]
            with torch.cuda.stream(compute), set_forward_context({layer.self_attn.prefix:m} if kinds[task.layer] == "kda" else None,
                    worker.vllm_config, num_tokens=n):
                hidden = _apply_attn_res(prefix, bank, layer.self_attention_res_proj,
                    layer.self_attention_res_norm, layer.prev_valid_blocks,
                    delta=None if task.layer == 0 else storage[1], output_norm=layer.input_layernorm,
                    block_write_idx=layer.block_write_idx if layer.is_block_write_layer else -1)
                if variant == "tp8_kda" and kinds[task.layer] == "kda":
                    attn_input[:n].copy_(hidden)
                else:
                    output = owned_attention(task.layer, hidden, row, start, length)
                    if layer.is_block_write_layer:
                        prefix.copy_(output)
                        delta = None
                    else:
                        delta = output
                    mixed = _apply_attn_res(prefix, bank, layer.mlp_res_proj, layer.mlp_res_norm,
                        layer.prev_valid_blocks + int(layer.is_block_write_layer), delta=delta,
                        output_norm=layer.post_attention_layernorm)
                    moe_input[:n].copy_(mixed)
                prepared_event.record(compute)

        if variant == "full_tp8":
            for micro, (row, start) in enumerate(micros):
                if start == 0:
                    admissions[request_id(micro)].record(service)
                hidden = core.embed_input_ids(ids[request_id(micro), start:start+n])
                bank = dense_bank
                delta = None
                m = native_metadata[n, row, start > 0]
                attn_metadata = {l.self_attn.prefix:m for i, l in enumerate(layers) if kinds[i] == "kda"}
                with set_forward_context(attn_metadata, worker.vllm_config, num_tokens=n):
                    for i, layer in enumerate(layers):
                        prefix = hidden
                        hidden = _apply_attn_res(prefix, bank, layer.self_attention_res_proj,
                            layer.self_attention_res_norm, layer.prev_valid_blocks, delta=delta,
                            output_norm=layer.input_layernorm,
                            block_write_idx=layer.block_write_idx if layer.is_block_write_layer else -1)
                        attention = dense_attention(i, hidden, row, start, length)
                        if layer.is_block_write_layer:
                            prefix, delta = attention, None
                        else:
                            delta = attention
                        mixed = _apply_attn_res(prefix, bank, layer.mlp_res_proj, layer.mlp_res_norm,
                            layer.prev_valid_blocks + int(layer.is_block_write_layer), delta=delta,
                            output_norm=layer.post_attention_layernorm)
                        delta, hidden = layer.mlp(mixed), prefix
                output = hidden+delta
                outputs[micro].copy_(output if sample_all else output[-8:])
                timings[micro].record(service)
                if start+n == length:
                    row_done[request_id(micro)].record(service)
        else:
            for tasks in rounds:
                # New inputs enter only stage 0. Embedding still uses native
                # TP8 on the shared service stream and is included in timing.
                for task in tasks:
                    if task.stage == 0 and task.layer == 0:
                        row, start = micros[task.micro]
                        if start == 0:
                            admissions[request_id(task.micro)].record(service)
                        hidden = core.embed_input_ids(ids[request_id(task.micro), start:start+n])
                        if rank == 0:
                            buffers[task.micro % 2, 0].copy_(hidden)
                            buffers[task.micro % 2, 2:].zero_()
                            arrival.record(service)
                            compute.wait_event(arrival)
                for task in tasks:
                    prepare(task)
                outgoing = []
                for task in tasks:
                    row, start = micros[task.micro]
                    layer = layers[task.layer]
                    if rank == task.stage:
                        service.wait_event(prepared_event)
                        service_input[:n].copy_(attn_input[:n] if variant == "tp8_kda" and kinds[task.layer] == "kda" else moe_input[:n])
                    comm.broadcast(service_input[:n], src=task.stage, stream=service)
                    m = native_metadata[n, row, start > 0]
                    with set_forward_context({layer.self_attn.prefix:m} if kinds[task.layer] == "kda" else None,
                            worker.vllm_config, num_tokens=n):
                        if variant == "tp8_kda" and kinds[task.layer] == "kda":
                            attention = layer.self_attn(hidden_states=service_input[:n], positions=positions[start:start+n])
                            if rank == task.stage:
                                storage = buffers[task.micro % 2]
                                prefix, bank = storage[0], storage[2:].permute(1, 0, 2)
                                if layer.is_block_write_layer:
                                    prefix.copy_(attention)
                                    delta = None
                                else:
                                    delta = attention
                                service_input[:n].copy_(_apply_attn_res(prefix, bank, layer.mlp_res_proj,
                                    layer.mlp_res_norm, layer.prev_valid_blocks + int(layer.is_block_write_layer),
                                    delta=delta, output_norm=layer.post_attention_layernorm))
                            comm.broadcast(service_input[:n], src=task.stage, stream=service)
                        result = layer.mlp(service_input[:n])
                    if rank == task.stage:
                        buffers[task.micro % 2, 1].copy_(result)
                        ff_done.record(service)
                        compute.wait_event(ff_done)
                    if task.layer+1 < ranges[task.stage][1]:
                        # Queue the next attention NOW, while peers service
                        # other stages' FFNs; not after the entire round ends.
                        prepare(Task(task.micro, task.stage, task.layer+1))
                    elif task.stage < 7:
                        outgoing.append(task)
                    else:
                        if rank == 7:
                            storage = buffers[task.micro % 2]
                            output = storage[0]+storage[1]
                            outputs[task.micro].copy_(output if sample_all else output[-8:])
                        timings[task.micro].record(service)
                        if start+n == length:
                            row_done[request_id(task.micro)].record(service)
                # Same communicator, same global order, one stream: no
                # cyclic P2P/EP collective ordering across separate groups.
                comm.group_start()
                for task in outgoing:
                    fields = 2 + (task.layer+12)//12
                    data = buffers[task.micro % 2, :fields]
                    if not data.is_contiguous():
                        raise AssertionError("P2P payload must be physically contiguous")
                    if rank == task.stage:
                        comm.send(data, task.stage+1, stream=service)
                    elif rank == task.stage+1:
                        comm.recv(data, task.stage, stream=service)
                comm.group_end()
                if any(task.stage+1 == rank for task in outgoing):
                    arrival.record(service)
                    compute.wait_event(arrival)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start_wall
        times = [started.elapsed_time(t)/1000 for t in timings]
        latencies = [a.elapsed_time(b)/1000 for a, b in zip(admissions, row_done)]
        point = dict(seconds=elapsed, cohort_tokens_per_second=length*batch*cohorts/elapsed,
            request_cohorts=cohorts, request_count=batch*cohorts, microbatch_count=len(micros),
            row_latency_seconds=latencies, cache_audit=audit(variant, length),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            service_order_sha256=hashlib.sha256(json.dumps([
                [(t.micro, t.stage, t.layer) for t in tasks] for tasks in rounds
            ]).encode()).hexdigest(),
            **completion_summary(times, chunk=n, stages=8, pipelined=variant != "full_tp8" and pipeline))
        return outputs, point

    points, checks = {}, {}
    try:
        with torch.inference_mode():
            # All tokens, all rows, all eight stages, native routed MoE, and
            # continued KDA states. Isolate schedule equivalence per layout;
            # cross-layout roundoff can legitimately alter natural MoE routes.
            for variant in ("owner_kda", "tp8_kda"):
                reference, _ = execute(variant, 512, 256, pipeline=False, sample_all=True)
                if rank == 7:
                    reference = reference.clone()
                actual, _ = execute(variant, 512, 256, pipeline=True, sample_all=True)
                if rank == 7:
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                    checks[variant] = "all first-512-token outputs bitwise equal to sequential same-layout execution"
                    print("KIMI_PIPELINE_VALIDATION " + json.dumps(dict(variant=variant, passed=True)), flush=True)
                group.barrier()
                del reference, actual
            if not validation_only:
                for length in lengths:
                    key = f"{length}/B{batch}"
                    points[key] = {}
                    for variant in ("full_tp8", "owner_kda", "tp8_kda"):
                        print("KIMI_PIPELINE_PASS " + json.dumps(dict(rank=rank,
                            length=length, variant=variant, phase="warmup")), flush=True)
                        execute(variant, length, chunk)
                        group.barrier()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        print("KIMI_PIPELINE_PASS " + json.dumps(dict(rank=rank,
                            length=length, variant=variant, phase="measured")), flush=True)
                        output, point = execute(variant, length, chunk)
                        if output is not None and not bool(torch.isfinite(output).all()):
                            raise AssertionError("nonfinite pipeline output")
                        points[key][variant] = point
                        if progress_path:
                            # Outside the measured interval: preserve accepted
                            # controls even if a later candidate fails.
                            target = Path(f"{progress_path}.rank{rank}.partial.json")
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(json.dumps(dict(rank=rank, layers=count,
                                ranges=ranges, checks=checks, points=points), indent=2)+"\n")
                        print("KIMI_EIGHT_STAGE_POINT " + json.dumps(dict(rank=rank, length=length,
                            batch=batch, variant=variant, seconds=point["seconds"],
                            steady_tokens_per_second=point["steady_tokens_per_second"])), flush=True)
                        del output
    finally:
        torch.cuda.synchronize()
        for obj, cache in original_kda:
            obj.kv_cache = cache
    return dict(rank=rank, layers=count, ranges=ranges, kinds=kinds,
                checks=checks, points=points)


def main():
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weight-cache-id", required=True)
    parser.add_argument("--real-token-cache", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--request-cohorts", type=int, default=2,
                        help="stream fresh prompts through eight reusable request slots")
    parser.add_argument("--chunk-size", type=int, choices=(2048, 4096, 16384), default=16384)
    parser.add_argument("--layer-count", type=int, choices=(0, 24), default=0,
                        help="0 executes every trained layer; 24 is an eight-stage correctness smoke only")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.request_cohorts < 1:
        parser.error("positive request cohort count required")
    for length in args.lengths:
        microbatches(length, args.batch, args.chunk_size)
    os.environ.pop("LOD_KIMI_REQUEST_OWNER_PREFILL", None)
    os.environ.update(VLLM_ALLOW_INSECURE_SERIALIZATION="1", LOD_KIMI_OWNER_SHARD_RESIDUAL="0")
    import torch
    from benchmarks.prolong import DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED, token_digest
    cached = torch.load(args.real_token_cache, map_location="cpu", weights_only=False)
    if (cached.get("dataset"), cached.get("revision"), cached.get("seed")) != (DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED):
        raise ValueError("wrong frozen ProLong corpus")
    maximum = max(args.lengths)
    rows = real_token_rows(cached["documents"], max(maximum, 512), args.batch*args.request_cohorts)
    from benchmarks._vllm import llm_kwargs, close_llm
    from vllm import LLM
    kwargs = llm_kwargs(checkpoint=args.checkpoint, mode="two-tier", max_model_len=maximum+1,
        batch_size=args.batch, tensor_parallel_size=8, decode_context_parallel_size=8,
        dcp_comm_backend="ag_rs", gpu_memory_utilization=0.8, full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="ipc_cache", skip_tokenizer_init=True, enforce_eager=True,
        kv_cache_memory_bytes=1073741824, enable_expert_parallel=True, disable_custom_all_reduce=False,
        quantization_config={"moe":{"weight":"int4_per_group_32"}},
        model_loader_extra_config=dict(auto_start=True, cache_id=args.weight_cache_id,
            backing_load_format="auto", broker_timeout=1800.0))
    llm = LLM(**kwargs)
    try:
        workers = llm.collective_rpc(run_probe, timeout=1200,
            args=(args.lengths, args.batch, args.chunk_size, args.layer_count, rows, args.validation_only,
                  args.request_cohorts, str(args.output)))
        if {w["rank"] for w in workers} != set(range(8)):
            raise AssertionError("missing worker")
        last = next(w for w in workers if w["rank"] == 7)
        result = dict(scope=__doc__, workers=workers, checks=last["checks"], points={},
            trained_weights=True, native_moe_ep=8, attention_stages=8,
            layer_count=last["layers"], stage_ranges=last["ranges"], chunk_size=args.chunk_size,
            pipeline_overlap=True, cuda_graph_capture=False, weight_assembly_timed=False,
            lm_head_timed=False, input_token_sha256=[token_digest(row) for row in rows],
            additional_weight_storage="prototype retains native TP shards and only stage-owned full attention weights",
            timing="one shape-exact warmup and one synchronized wall pass; slowest worker; final-stage completion events separately")
        for key in last["points"]:
            for variant in last["points"][key]:
                if len({w["points"][key][variant]["service_order_sha256"] for w in workers}) != 1:
                    raise AssertionError("workers disagree about collective service order")
            result["points"][key] = {}
            for v, p in last["points"][key].items():
                elapsed = max(w["points"][key][v]["seconds"] for w in workers)
                result["points"][key][v] = {**p, "seconds":elapsed,
                    "cohort_tokens_per_second":p["cohort_tokens_per_second"]*p["seconds"]/elapsed}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print("KIMI_EIGHT_STAGE_RESULT " + json.dumps(dict(checks=result["checks"], points=result["points"])), flush=True)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

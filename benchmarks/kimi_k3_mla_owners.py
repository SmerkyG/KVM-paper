"""Hybrid K3 prefill: native TP8 KDA/EP8 MoE, request-owner MLA Q/K/V/O.

The unchanged full checkpoint is attached to the existing weight daemon;
only the first N layers execute in the direct worker probe. This retains
trained, production-sized native INT4 experts and both AttnRes operations.
No shared checkpoint parameters are modified. LM head, sampling,
and scheduling are outside the timed interval; embeddings, all attention interfaces,
KDA, MoE, residuals, and LoD updates are inside. This is a layout/speed probe,
not an end-to-end serving latency or language-quality measurement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from types import MethodType


def owner_rank(row: int, layer_index: int, world: int = 8) -> int:
    """Spread B1 MLA layers; B8 assigns one complete row to each GPU."""
    if row < 0 or layer_index < 0 or world < 1:
        raise ValueError("nonnegative row/layer and positive world required")
    return (row + layer_index // 4) % world


def real_token_rows(documents, length, batch):
    """Match the existing speed sweep's real-token concatenation policy."""
    if not documents or not any(documents) or length <= 0 or batch <= 0:
        raise ValueError("nonempty real-token corpus and positive shape required")
    rows = []
    for request in range(batch):
        tokens, cursor, visited = [], request, 0
        while len(tokens) < length:
            tokens.extend(documents[cursor % len(documents)])
            cursor += batch
            visited += 1
            if visited >= len(documents) and not tokens:
                raise ValueError("this row's source stream is empty")
        rows.append(tokens[:length])
    return rows


def consume_attention_rows(output, chunk, attend):
    """Copy each result before another call can overwrite shared scratch."""
    if output.size(0) % chunk:
        raise ValueError("output must contain complete row slices")
    for row in range(output.size(0)//chunk):
        output[row*chunk:(row+1)*chunk].copy_(attend(row))


def validate_audit(audit, *, variant, length, batch, layer_count, rank):
    expected_layers = {str(i) for i in range(3, layer_count, 4)}
    if set(audit) != expected_layers:
        raise AssertionError("missing MLA layer audit")
    for index, rows in audit.items():
        expected_rows = {str(r) for r in range(batch)
                         if variant != "owner_mla" or owner_rank(r, int(index)) == rank}
        if set(rows) != expected_rows:
            raise AssertionError("missing request or cache exists on a non-owner")
        for data in rows.values():
            if data["total_len"] != length:
                raise AssertionError("cache history incomplete")
            if variant == "full_tp8":
                if data["centroid_updates"] != 0:
                    raise AssertionError("full control used LoD")
            elif data["coverage"] != length - 256 or data["state_len"] <= 0:
                raise AssertionError("global 16K update did not complete")


def run_probe(worker, lengths, batches, chunk, layer_count, token_rows):
    import torch
    import torch.nn.functional as F
    from aiter import flash_attn_varlen_func
    from vllm.distributed import get_tp_group
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from vllm_lod_plugin.models.kimi_k3_request_prefill import attend_slice
    from vllm_lod_plugin.models.kimi_k3_sharded_prefill import gather_prefill

    core = next(m for m in worker.model_runner.model.modules()
                if type(m).__name__ == "KimiLinearModel")
    layers = list(core.layers)[:layer_count]
    group = get_tp_group()
    rank, world = group.rank_in_group, group.world_size
    comm = group.device_communicator.pynccl_comm
    if world != 8 or comm is None or comm.disabled or len(layers) != layer_count:
        raise RuntimeError("requires TP8, enabled RCCL, and the requested native layers")
    kinds = ["kda" if isinstance(layer.self_attn, KimiK3DeltaAttention) else "mla" for layer in layers]
    if kinds != ["mla" if (i+1) % 4 == 0 else "kda" for i in range(layer_count)]:
        raise AssertionError("unexpected KDA/MLA placement")
    if (type(layers[0].mlp).__name__ != "KimiMLP"
            or any(type(layer.mlp).__name__ != "KimiMoE" for layer in layers[1:])):
        raise AssertionError("must retain the trained dense first FFN and every native MoE")
    moe = layers[1].mlp
    if moe.experts.moe_config.ep_size != 8:
        raise AssertionError("native MoE is not EP8")

    maximum, max_batch = max(lengths), max(batches)
    ids = torch.tensor(token_rows, dtype=torch.long, device="cuda")
    # Keep real token IDs, not a multi-GiB archive of all embedded hidden
    # states. Like normal vLLM, embed only the active scheduler slice. This
    # identical embedding/all-reduce is now explicitly included in timing.
    inputs = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(maximum, device="cuda")
    prepared, native_kda_caches = {}, []
    for layer, kind in zip(layers, kinds, strict=True):
        if kind == "kda":
            kda = layer.self_attn
            if kda.local_num_heads != 12:
                raise AssertionError("KDA was moved off its native TP8 head partition")
            native_kda_caches.append((kda, kda.kv_cache))
            conv = torch.zeros(max_batch+1, 3*12*128, 3, dtype=torch.bfloat16, device="cuda")
            if not is_conv_state_dim_first():
                conv = conv.transpose(-1, -2).contiguous()
            kda.kv_cache = (conv, torch.zeros(max_batch+1, 12, 128, 128,
                                            dtype=torch.float32, device="cuda"))
            continue
        wrapper = layer.self_attn.mla_attn
        pool = wrapper.mla_attn._vllm_lod_pool
        if wrapper.q_b_proj.weight.shape != (12*192, 1536):
            raise AssertionError("MLA Q projection is not a native 12-head shard")
        pool.engine._lod_kimi_reduce_prefill_routes = True
        q_weight = gather_prefill(group, wrapper.q_b_proj.weight, dim=0)
        o_weight = gather_prefill(group, wrapper.o_proj.weight, dim=1)
        g_weight = gather_prefill(group, wrapper.g_proj.weight, dim=0)
        kv = wrapper.kv_b_proj.weight.view(12, 256, 512)
        uk = gather_prefill(group, kv[:, :128].contiguous(), dim=0)
        uv = gather_prefill(group, kv[:, 128:].transpose(1, 2).contiguous(), dim=0)
        # Gathered weights are untimed prototype copies, not modifications to
        # daemon storage. Each rank keeps them only for rows it will own.
        owned = any(owner_rank(r, layer.layer_idx) == rank for r in range(max_batch))
        prepared[layer.layer_idx] = dict(wrapper=wrapper, pool=pool,
            full=(q_weight, o_weight, g_weight, uk, uv) if owned else None,
            local=(kv[:, :128], kv[:, 128:].transpose(1, 2)),
            history=None,
            dense_lengths={})
        del q_weight, o_weight, g_weight, uk, uv

    # One grow-only, layer-shared output transport. No headwise return
    # exchange or TP W_O all-reduce is required for owner-local MLA.
    local_output = inputs.new_empty(chunk, 7168)
    wire = inputs.new_empty(world, chunk, 7168)
    assembled = inputs.new_empty(max_batch*chunk, 7168)
    banks = inputs.new_empty(max_batch*chunk, (layer_count+11)//12, 7168)
    metadata, cu_gpu, cu_keys = {}, {}, {}
    for batch in batches:
        cu_cpu = torch.arange(batch+1, dtype=torch.int32)*chunk
        cu = cu_cpu.to("cuda")
        cu_gpu[batch] = cu
        nums, bp, tp = compute_causal_conv1d_metadata(cu_cpu, device=inputs.device)
        ci = prepare_chunk_indices(cu_cpu, FLA_CHUNK_SIZE).to("cuda")
        co = prepare_chunk_offsets(cu_cpu, FLA_CHUNK_SIZE).to("cuda")
        for initial in (False, True):
            metadata[batch, initial] = GDNAttentionMetadata(
                num_prefills=batch, num_prefill_tokens=batch*chunk,
                num_decodes=0, num_decode_tokens=0, num_spec_decodes=0,
                num_spec_decode_tokens=0, num_actual_tokens=batch*chunk,
                has_initial_state=torch.full((batch,), initial, device="cuda"),
                non_spec_query_start_loc=cu,
                non_spec_state_indices_tensor=torch.arange(1, batch+1, dtype=torch.int32, device="cuda"),
                chunk_indices=ci, chunk_offsets=co, nums_dict=nums,
                batch_ptr=bp, token_chunk_offset_ptr=tp)
        for start in range(0, maximum, chunk):
            cu_keys[batch, start] = torch.arange(batch+1, dtype=torch.int32, device="cuda")*(start+chunk)

    state = dict(variant="full_tp8", batch=1, previous=0, length=0)
    originals = [(layer, layer._run_self_attn) for layer in layers if kinds[layer.layer_idx] == "mla"]

    def attention(layer, position, hidden):
        p = prepared[layer.layer_idx]
        wrapper, pool = p["wrapper"], p["pool"]
        variant, batch, previous = state["variant"], state["batch"], state["previous"]
        if variant == "owner_mla":
            owned_rows = [r for r in range(batch) if owner_rank(r, layer.layer_idx) == rank]
            if len(owned_rows) > 1:
                raise AssertionError("probe expects at most eight rows")
            if owned_rows:
                row = owned_rows[0]
                row_hidden = hidden[row*chunk:(row+1)*chunk]
                fused = wrapper.fused_qkv_a_proj(row_hidden)[0]
                qc, kv = fused.split((1536, 576), dim=-1)
                latent, direct = kv.split((512, 64), dim=-1)
                qc, latent = wrapper._normalize_q_kv(qc, latent)
                qw, ow, gw, uk, uv = p["full"]
                query = F.linear(qc, qw).view(chunk, 96, 192).contiguous()
                record = torch.cat((latent, direct), -1).unsqueeze(1)
                output = attend_slice(pool, row, query, record, uk, uv,
                                      previous=previous, prompt=state["length"])
                local_output.copy_(F.linear(output.reshape(chunk, 12288)
                                  * F.linear(row_hidden, gw).sigmoid(), ow))
            if batch == 1:
                # All native TP/EP ranks need this hidden-width result.
                comm.broadcast(local_output, src=owner_rank(0, layer.layer_idx))
                return local_output
            if not owned_rows:
                local_output.zero_()
            comm.all_gather(wire.view(world*chunk, 7168), local_output)
            result = assembled[:batch*chunk]
            for row in range(batch):
                result[row*chunk:(row+1)*chunk].copy_(wire[owner_rank(row, layer.layer_idx)])
            return result

        fused = wrapper.fused_qkv_a_proj(hidden)[0]
        qc, kv = fused.split((1536, 576), dim=-1)
        latent, direct = kv.split((512, 64), dim=-1)
        qc, latent = wrapper._normalize_q_kv(qc, latent)
        query = wrapper.q_b_proj(qc)[0].view(batch*chunk, 12, 192).contiguous()
        record = torch.cat((latent, direct), -1).view(batch, chunk, 1, 576)
        if variant == "full_tp8":
            total = previous+chunk
            for row in range(batch):
                if p["dense_lengths"].get(row, 0) != previous:
                    raise AssertionError("dense history order drift")
                p["dense_lengths"][row] = total
            p["history"][:batch, previous:total].copy_(record)
            history = p["history"][:batch, :total, 0].reshape(batch*total, 576)
            projected = wrapper.kv_b_proj(history[:, :512])[0].view(batch*total, 12, 256)
            keys = torch.cat((projected[..., :128], history[:, None, 512:].expand(-1, 12, -1)), -1)
            output = flash_attn_varlen_func(query, keys, projected[..., 128:],
                cu_gpu[batch], cu_keys[batch, previous], chunk, total,
                softmax_scale=float(pool.engine.scaling), causal=True, return_lse=False)
        else:
            uk, uv = p["local"]
            output = hidden.new_empty(batch*chunk, 12, 128)
            consume_attention_rows(output, chunk, lambda row: attend_slice(
                pool, row, query[row*chunk:(row+1)*chunk], record[row],
                uk, uv, previous=previous, prompt=state["length"]))
        return wrapper.o_proj(output.reshape(batch*chunk, 1536)
                              * wrapper.g_proj(hidden)[0].sigmoid())[0]

    for layer, _ in originals:
        layer._run_self_attn = MethodType(attention, layer)

    def reset(variant):
        state["variant"] = variant
        for kda, _ in native_kda_caches:
            for cache in kda.kv_cache:
                cache.zero_()
        for p in prepared.values():
            pool = p["pool"]
            pool._kimi_request_owner_rows = {}
            pool.engine.reset_runtime_cache()
            heads = 96 if variant == "owner_mla" else 12
            pool.engine.num_key_value_groups = heads
            pool.engine.config.num_attention_heads = heads
            p["dense_lengths"].clear()
            if variant == "full_tp8":
                if p["history"] is None:
                    p["history"] = torch.empty(max_batch, maximum, 1, 576,
                                               dtype=torch.bfloat16, device="cuda")
            else:
                # This is solely the dense control's chronological archive.
                # Keeping another full BF16 copy alongside owner LoD is not
                # part of the proposed layout, and exhausted the probe VRAM.
                p["history"] = None

    def execute(variant, length, batch):
        reset(variant)
        state.update(length=length, batch=batch)
        last = None
        for start in range(0, length, chunk):
            state["previous"] = start
            # Identical packed rows in all layouts. B8 is a *true* KDA/MoE
            # batch, while MLA owners run their rows concurrently.
            hidden = core.embed_input_ids(ids[:batch, start:start+chunk].reshape(-1))
            bank, delta = banks[:batch*chunk], None
            pos = positions[start:start+chunk].repeat(batch)
            m = metadata[batch, start > 0]
            attn_metadata = {layer.self_attn.prefix: m for layer, kind in zip(layers, kinds) if kind == "kda"}
            with set_forward_context(attn_metadata, worker.vllm_config, num_tokens=batch*chunk):
                for layer in layers:
                    hidden, bank, delta = layer(pos, hidden, bank, prefix_delta=delta)
                last = (hidden+delta).view(batch, chunk, 7168)[:, -8:].clone()
        return last

    def audit(variant):
        return {str(index): ({str(row): dict(total_len=length, centroid_updates=0)
                  for row, length in p["dense_lengths"].items()} if variant == "full_tp8" else
                {str(row): dict(total_len=data["total"], coverage=int(data["cache"].state["coverage"]),
                                state_len=int(data["cache"].state["state_len"]))
                 for row, data in p["pool"]._kimi_request_owner_rows.items()})
                for index, p in prepared.items()}

    def layout_check():
        # Isolate pure MLA on identical trained inputs, without propagating
        # differences through token-dependent MoE routing or KDA recurrence.
        layer = next(layer for layer, kind in zip(layers, kinds) if kind == "mla")
        errors = {}
        for batch in batches:
            sampled = {}
            state.update(batch=batch, length=32768)
            for variant in ("lod_tp8", "owner_mla"):
                reset(variant)
                for start in (0, 16384):
                    # Use the selected scheduler shape, preserving 16K boundaries.
                    for offset in range(0, 16384, chunk):
                        state["previous"] = start+offset
                        hidden = core.embed_input_ids(ids[:batch, start+offset:start+offset+chunk].reshape(-1))
                        output = attention(layer, positions[:chunk], hidden)
                    # Check every row, including the all-gather's rotated
                    # row order, not just B1 broadcast or rank agreement.
                    sampled[variant, start] = output.view(batch, chunk, 7168)[:, -32:].clone()
            errors[str(batch)] = []
            for start in (0, 16384):
                expected, actual = sampled["lod_tp8", start].float(), sampled["owner_mla", start].float()
                relative = ((actual-expected).square().sum()/expected.square().sum().clamp_min(1e-20)).sqrt().item()
                if not bool(torch.isfinite(actual).all()) or relative > 0.03:
                    raise AssertionError(f"identical-input B{batch} MLA owner layout failed, block={start}, relative L2={relative}")
                errors[str(batch)].append(relative)
        return errors

    points = {}
    try:
        with torch.inference_mode():
            layout_errors = layout_check()
            print("KIMI_MLA_OWNER_LAYOUT_CHECK " + json.dumps(dict(rank=rank, relative_l2=layout_errors)), flush=True)
            for length in lengths:
                for batch in batches:
                    key, variants = f"{length}/B{batch}", {}
                    for variant in ("full_tp8", "owner_mla"):
                        # Finish old collectives before returning inactive
                        # allocator blocks. This is outside the warm/timed
                        # pass; never trigger reclaim while peers enter RCCL.
                        torch.cuda.synchronize()
                        group.barrier()
                        reset(variant)
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        execute(variant, length, batch)
                        torch.cuda.synchronize()
                        group.barrier()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        begin = time.perf_counter()
                        output = execute(variant, length, batch)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-begin
                        finite = bool(torch.isfinite(output).all())
                        if not finite:
                            raise AssertionError("nonfinite native KDA/MLA/MoE output")
                        variants[variant] = dict(seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            output_rms=output.float().square().mean().sqrt().item(), cache_audit=audit(variant))
                        validate_audit(variants[variant]["cache_audit"], variant=variant,
                            length=length, batch=batch, layer_count=layer_count, rank=rank)
                        print("KIMI_MLA_OWNER_POINT " + json.dumps(dict(rank=rank, length=length,
                            batch=batch, variant=variant, seconds=elapsed)), flush=True)
                    points[key] = variants
    finally:
        for layer, original in originals:
            layer._run_self_attn = original
        for kda, cache in native_kda_caches:
            kda.kv_cache = cache
    return dict(rank=rank, layer_kinds=kinds, ffn_types=[type(layer.mlp).__name__ for layer in layers],
                kda_local_heads=12, ep_size=moe.experts.moe_config.ep_size,
                layout_relative_l2=layout_errors, points=points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weight-cache-id", required=True)
    parser.add_argument("--real-token-cache", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768, 65536])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--chunk-size", type=int, choices=(2048, 16384), default=2048)
    parser.add_argument("--layer-count", type=int, choices=(4, 24), default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(n < 32768 or n % 16384 for n in args.lengths) or any(b not in (1, 8) for b in args.batches):
        parser.error("requires 32K+ multiples of 16K, and B1/B8")
    # Attach the same unmodified HF config/IPC entry. Cropping is solely at
    # runtime, never a new model identity or another daemon materialization.
    os.environ.pop("LOD_KIMI_REQUEST_OWNER_PREFILL", None)
    os.environ.update(VLLM_ALLOW_INSECURE_SERIALIZATION="1", LOD_KIMI_OWNER_SHARD_RESIDUAL="0")
    import torch
    from benchmarks.prolong import DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED, token_digest
    cached = torch.load(args.real_token_cache, map_location="cpu", weights_only=False)
    if (cached.get("dataset"), cached.get("revision"), cached.get("seed")) != (DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED):
        raise ValueError("wrong frozen ProLong corpus")
    maximum, batch = max(args.lengths), max(args.batches)
    rows = real_token_rows(cached["documents"], maximum, batch)
    from benchmarks._vllm import llm_kwargs, close_llm
    from vllm import LLM
    kwargs = llm_kwargs(checkpoint=args.checkpoint, mode="two-tier", max_model_len=maximum+1,
        batch_size=batch, tensor_parallel_size=8, decode_context_parallel_size=8,
        dcp_comm_backend="ag_rs", gpu_memory_utilization=0.8,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="ipc_cache", skip_tokenizer_init=True, enforce_eager=True,
        kv_cache_memory_bytes=1073741824, enable_expert_parallel=True, disable_custom_all_reduce=False,
        quantization_config={"moe": {"weight": "int4_per_group_32"}},
        model_loader_extra_config=dict(auto_start=True, cache_id=args.weight_cache_id,
                                      backing_load_format="auto", broker_timeout=1800.0))
    llm = LLM(**kwargs)
    try:
        workers = llm.collective_rpc(run_probe, args=(args.lengths, args.batches, args.chunk_size, args.layer_count, rows))
        if {w["rank"] for w in workers} != set(range(8)):
            raise AssertionError("missing worker")
        points = {}
        for length in args.lengths:
            for batch in args.batches:
                key = f"{length}/B{batch}"
                full = max(w["points"][key]["full_tp8"]["seconds"] for w in workers)
                owner = max(w["points"][key]["owner_mla"]["seconds"] for w in workers)
                points[key] = dict(full_tp8_seconds=full, owner_mla_seconds=owner, speedup=full/owner)
        result = dict(scope=__doc__, points=points, workers=workers, trained_weights=True,
            layer_count=args.layer_count, chunk_size=args.chunk_size, native_kda_tp=8, native_moe_ep=8,
            moe_included=True, cuda_graph_capture=False, pipeline_overlap=False,
            embeddings_included=True,
            input_token_sha256=[token_digest(row) for row in rows],
            timing="one exact-shape warmup and one synchronized wall pass; max across all eight workers",
            weight_assembly_timed=False, additional_weight_storage="prototype retains TP shards and owner copies")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print("KIMI_MLA_OWNER_RESULT " + json.dumps(points), flush=True)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

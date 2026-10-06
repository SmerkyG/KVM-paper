"""Opt-in, eager prefill experiment: one K3 row's MLA Q/K/V/O per GPU.

KDA, MoE, residual mixing and the rest of vLLM stay unchanged. Only small
MLA projection copies are assembled, outside timing, from resident TP shards.
This is an aligned eight-row prefill experiment, not a serving/decode backend.
"""

from types import MethodType

import torch
import torch.nn.functional as F


def validate_owner_plan(plan, *, world, tokens, chunk):
    if len(plan) != world or {slot % world for slot, *_ in plan} != set(range(world)):
        raise ValueError("local MLA requires exactly one row per GPU")
    if tokens != world * chunk or any(
        begin != index * chunk or end != begin + chunk or previous % chunk
        for index, (_, begin, end, previous) in enumerate(plan)
    ):
        raise ValueError("local MLA requires packed full 16K row blocks")


def assemble_owner_hidden(wire, plan):
    """Return packed scheduler order; the usual rank order is a zero-copy view."""
    order = [slot % wire.size(0) for slot, *_ in plan]
    if order == list(range(wire.size(0))):
        return wire.flatten(0, 1)
    return torch.cat([wire[rank] for rank in order], dim=0)


def local_mla_forward(self, positions, hidden_states, llama_4_scaling=None):
    from vllm_lod_plugin.models.kimi_k3_request_prefill import attend_slice

    pool = self.mla_attn._vllm_lod_pool
    plan = pool.direct_prefill_plan
    if not plan:
        return self._lod_owner_original_forward(positions, hidden_states, llama_4_scaling)
    if llama_4_scaling is not None or self.rotary_emb is not None:
        raise NotImplementedError("K3 owner probe requires unrotated MLA")
    group, chunk = pool.dcp_group, int(pool.engine.prefill_chunk_len)
    validate_owner_plan(plan, world=group.world_size, tokens=hidden_states.size(0), chunk=chunk)
    slot, begin, end, previous = next(item for item in plan if item[0] % group.world_size == group.rank_in_group)
    prompts = pool.direct_prefill_prompt_lengths
    if any(prior >= prompts[row] for row, _, _, prior in plan):
        raise NotImplementedError("local MLA experiment is prefill-only")
    pool.direct_prefill_plan = None
    pool.direct_prefill_prompt_lengths = {}
    row_hidden = hidden_states[begin:end]
    fused = self.fused_qkv_a_proj(row_hidden)[0]
    qc, kv = fused.split((1536, 576), dim=-1)
    latent, direct = kv.split((512, 64), dim=-1)
    qc, latent = self._normalize_q_kv(qc, latent)
    qw, ow, gw, uk, uv = self._lod_owner_mla_weights
    query = F.linear(qc, qw).view(chunk, 96, 192).contiguous()
    record = torch.cat((latent, direct), dim=-1).unsqueeze(1)
    attended = attend_slice(pool, slot, query, record, uk, uv,
                           previous=previous, prompt=prompts[slot])
    local = F.linear(attended.reshape(chunk, 12288)
                     * F.linear(row_hidden, gw).sigmoid(), ow)
    # One arena shared by all MLA layers. Its view is consumed on the current
    # stream before another layer overwrites it; concurrent forwards are not
    # supported by this private eager experiment.
    arena = getattr(group, "_lod_owner_hidden_arena", None)
    shape = (group.world_size, chunk, hidden_states.size(-1))
    if arena is None or tuple(arena.shape) != shape:
        arena = group._lod_owner_hidden_arena = hidden_states.new_empty(shape)
    group.device_communicator.pynccl_comm.all_gather(arena.flatten(0, 1), local)
    output = assemble_owner_hidden(arena, plan)
    pool._kimi_owner_max_plan_rows = max(getattr(pool, "_kimi_owner_max_plan_rows", 0), len(plan))
    pool._kimi_owner_full_cohort_previous = max(
        getattr(pool, "_kimi_owner_full_cohort_previous", 0), min(item[3] for item in plan))
    sizes = getattr(pool, "_kimi_owner_query_sizes", None)
    if sizes is None:
        sizes = pool._kimi_owner_query_sizes = {}
    sizes[chunk] = sizes.get(chunk, 0) + 1
    for row, first, last, prior in plan:
        pool.ready[row] = True
        pool.dcp_sharded[row] = False
        pool.metadata[row].update(total_len=prior + last - first, coverage=0)
    return output


def install_owner_local_mla(worker, validation_token_ids):
    """Assemble once; verify projection layout using real ProLong token inputs."""
    from vllm_lod_plugin.models.kimi_k3_sharded_prefill import gather_prefill

    model = worker.model_runner.model
    core = next(m for m in model.modules() if type(m).__name__ == "KimiLinearModel")
    layers = list(core.layers)
    kinds = [type(layer.self_attn).__name__ for layer in layers]
    if ([type(layer.mlp).__name__ for layer in layers]
            != ["KimiMLP"] + ["KimiMoE"] * (len(layers) - 1)
            or any(layer.mlp.experts.moe_config.ep_size != 8 for layer in layers[1:])
            or any(layer.self_attn.local_num_heads != 12 for layer in layers
                   if type(layer.self_attn).__name__ == "KimiK3DeltaAttention")):
        raise AssertionError("native TP8 KDA and EP8 MoE must remain unchanged")
    hidden = core.embed_input_ids(torch.tensor(validation_token_ids, device="cuda", dtype=torch.long))
    installed, errors, extra_bytes = [], {}, 0
    for layer in layers:
        if type(layer.self_attn).__name__ != "KimiMLAAttention":
            continue
        wrapper = layer.self_attn.mla_attn
        pool = wrapper.mla_attn._vllm_lod_pool
        group = pool.dcp_group
        if (not pool.kimi_request_owner_prefill or group.world_size != 8
                or pool.query_heads != 12 or wrapper.rotary_emb is not None
                or tuple(wrapper.q_b_proj.weight.shape) != (2304, 1536)):
            raise ValueError("requires ordinary K3 TP8 owner pools and BF16 MLA projections")
        if hasattr(wrapper, "_lod_owner_mla_weights"):
            raise RuntimeError("local MLA projections already installed")
        q = gather_prefill(group, wrapper.q_b_proj.weight, dim=0)
        o = gather_prefill(group, wrapper.o_proj.weight, dim=1)
        gate = gather_prefill(group, wrapper.g_proj.weight, dim=0)
        kv = wrapper.kv_b_proj.weight.view(12, 256, 512)
        uk = gather_prefill(group, kv[:, :128].contiguous(), dim=0)
        uv = gather_prefill(group, kv[:, 128:].transpose(1, 2).contiguous(), dim=0)
        weights = (q, o, gate, uk, uv)
        extra_bytes += sum(w.numel() * w.element_size() for w in weights)
        # Validate head ordering and local-vs-TP output projection, including
        # the gate, before replacing forward. This never changes shared weights.
        fused = wrapper.fused_qkv_a_proj(hidden)[0]
        qc, latent = fused[:, :1536], fused[:, 1536:2048]
        qc, _ = wrapper._normalize_q_kv(qc, latent)
        native_q = wrapper.q_b_proj(qc)[0].view(-1, 12, 192)
        expected_q = gather_prefill(group, native_q, dim=1)
        local_q = F.linear(qc, q).view(-1, 96, 192)
        # Use the same real-input-derived head values in both W_O evaluations.
        native_values = native_q[..., :128].reshape(hidden.size(0), 1536)
        expected = wrapper.o_proj(native_values * wrapper.g_proj(hidden)[0].sigmoid())[0]
        values = expected_q[..., :128].reshape(hidden.size(0), 12288)
        actual = F.linear(values * F.linear(hidden, gate).sigmoid(), o)
        def relative_l2(a, b):
            return ((a.float()-b.float()).square().sum() / b.float().square().sum().clamp_min(1e-20)).sqrt().item()
        qe, oe = relative_l2(local_q, expected_q), relative_l2(actual, expected)
        if not bool(torch.isfinite(actual).all()) or max(qe, oe) > 0.03:
            raise AssertionError(f"owner MLA projection check failed: Q={qe}, gated O={oe}")
        errors[str(layer.layer_idx)] = dict(query_relative_l2=qe, gated_output_relative_l2=oe)
        wrapper._lod_owner_mla_weights = weights
        wrapper._lod_owner_original_forward = wrapper.forward
        wrapper.forward = MethodType(local_mla_forward, wrapper)
        installed.append(int(layer.layer_idx))
    if not installed:
        raise ValueError("no K3 MLA layers found")
    torch.cuda.synchronize()
    return dict(rank=worker.rank, mla_layer_indices=installed,
                language_layers=len(layers), native_kda_layers=kinds.count("KimiK3DeltaAttention"),
                native_moe_layers=len(layers)-1, native_moe_ep=8, kda_local_heads=12,
                additional_projection_bytes=extra_bytes, projection_errors=errors,
                kda_and_moe_unchanged=True, shared_daemon_weights_modified=False,
                scope="eager prefill only; projections assembled and validated outside timing")


def prepare_owner_tp_mla(worker):
    """Keep native TP Q/K/V/O; prepare only the small latent head maps once."""
    from vllm_lod_plugin.models.kimi_k3_sharded_prefill import gather_prefill

    model = worker.model_runner.model
    core = next(m for m in model.modules() if type(m).__name__ == "KimiLinearModel")
    indices, map_bytes = [], 0
    for layer in core.layers:
        if type(layer.self_attn).__name__ != "KimiMLAAttention":
            continue
        wrapper = layer.self_attn.mla_attn
        attention = wrapper.mla_attn
        pool = attention._vllm_lod_pool
        if not pool.kimi_request_owner_prefill or pool.dcp_world_size != 8:
            raise ValueError("requires native TP8 K3 request-owner pools")
        if hasattr(wrapper, "_lod_owner_mla_weights"):
            raise AssertionError("TP owner must not retain local MLA projection copies")
        if getattr(attention, "_lod_owner_uk", None) is None:
            attention._lod_owner_uk = gather_prefill(pool.dcp_group, attention.W_UK_T, dim=0)
            attention._lod_owner_uv = gather_prefill(pool.dcp_group, attention.W_UV, dim=0)
        if (tuple(attention._lod_owner_uk.shape) != (96,128,512)
                or tuple(attention._lod_owner_uv.shape) != (96,512,128)):
            raise AssertionError("wrong full-head latent projection maps")
        map_bytes += sum(w.numel()*w.element_size()
                         for w in (attention._lod_owner_uk, attention._lod_owner_uv))
        indices.append(int(layer.layer_idx))
    if not indices:
        raise ValueError("no K3 MLA layers found")
    torch.cuda.synchronize()
    return dict(rank=worker.rank, mla_layer_indices=indices,
                language_layers=len(core.layers),
                native_kda_layers=sum(type(layer.self_attn).__name__ == "KimiK3DeltaAttention"
                                      for layer in core.layers),
                native_moe_layers=sum(type(layer.mlp).__name__ == "KimiMoE"
                                      for layer in core.layers),
                request_capacity=worker.model_runner.model_state._vllm_lod_runtime.pool_size,
                additional_projection_bytes=map_bytes, additional_latent_head_map_bytes=map_bytes,
                additional_q_gate_o_copy_bytes=0,
                q_k_v_o_tensor_parallel=True, shared_daemon_weights_modified=False,
                scope="native TP projections; only small latent head maps prepared outside timing")


def prepare_head_owner_tp_mla(worker, validation_token_ids):
    """Prepare only this owner's 16 latent maps; verify KV replication once."""
    from vllm_lod_plugin.models.kimi_k3_sharded_prefill import gather_prefill
    from vllm_lod_plugin.models.kimi_k3_head_prefill import exchange_heads

    core = next(m for m in worker.model_runner.model.modules()
                if type(m).__name__ == "KimiLinearModel")
    hidden = core.embed_input_ids(torch.tensor(validation_token_ids, device="cuda", dtype=torch.long))
    indices, map_bytes, replication_errors, transport_errors = [], 0, {}, {}
    rank = int(worker.rank)
    for layer in core.layers:
        if type(layer.self_attn).__name__ != "KimiMLAAttention":
            continue
        wrapper = layer.self_attn.mla_attn
        attention = wrapper.mla_attn
        pool = attention._vllm_lod_pool
        if (not pool.kimi_head_owner_prefill or pool.dcp_world_size != 8
                or pool.query_heads != 12 or pool.max_requests != 1):
            raise ValueError("requires B1 TP8 six-head-owner pools")
        if hasattr(wrapper, "_lod_owner_mla_weights"):
            raise AssertionError("head-owner probe must retain native TP Q/gate/O")
        uk = gather_prefill(pool.dcp_group, attention.W_UK_T, dim=0)
        uv = gather_prefill(pool.dcp_group, attention.W_UV, dim=0)
        if tuple(uk.shape) != (96, 128, 512) or tuple(uv.shape) != (96, 512, 128):
            raise AssertionError("wrong K3 latent head maps")
        if rank < 6:
            attention._lod_head_owner_uk = uk[rank * 16:(rank + 1) * 16].clone()
            attention._lod_head_owner_uv = uv[rank * 16:(rank + 1) * 16].clone()
            map_bytes += sum(w.numel() * w.element_size() for w in (
                attention._lod_head_owner_uk, attention._lod_head_owner_uv))
        fused = wrapper.fused_qkv_a_proj(hidden)[0]
        qc, kv = fused.split((1536, 576), dim=-1)
        latent, direct = kv.split((512, 64), dim=-1)
        qc, latent = wrapper._normalize_q_kv(qc, latent)
        record = torch.cat((latent, direct), dim=-1)
        gathered = gather_prefill(pool.dcp_group, record, dim=0).view(8, *record.shape)
        error = (gathered.float() - gathered[:1].float()).abs().max().item()
        if error != 0 or not bool(torch.isfinite(gathered).all()):
            raise AssertionError("head owners require replicated native latent records")
        replication_errors[str(layer.layer_idx)] = error
        # Check the actual grouped RCCL transport, using real-input native
        # Q projections. This is untimed; serving never all-gathers these Qs.
        query = wrapper.q_b_proj(qc)[0].view(hidden.size(0), 12, 192).contiguous()
        owned = exchange_heads(pool.dcp_group, query)
        full_query = gather_prefill(pool.dcp_group, query, dim=1)
        if rank < 6:
            torch.testing.assert_close(owned, full_query[:, rank * 16:(rank + 1) * 16], atol=0, rtol=0)
        returned = exchange_heads(pool.dcp_group, owned[..., :128].contiguous().clone(), returning=True)
        torch.testing.assert_close(returned, query[..., :128], atol=0, rtol=0)
        transport_errors[str(layer.layer_idx)] = 0
        indices.append(int(layer.layer_idx))
        del uk, uv, gathered
    if not indices:
        raise ValueError("no K3 MLA layers found")
    torch.cuda.synchronize()
    return dict(rank=rank, mla_layer_indices=indices, language_layers=len(core.layers),
                head_range=[rank * 16, (rank + 1) * 16] if rank < 6 else None,
                attention_owners=6, native_tp_heads=12,
                additional_latent_head_map_bytes=map_bytes, additional_q_gate_o_copy_bytes=0,
                native_record_replication_max_errors=replication_errors,
                head_exchange_round_trip_max_errors=transport_errors,
                shared_daemon_weights_modified=False,
                scope="six 16-head owners; native TP8/EP8; prefill only")

"""Experimental TP8 K3 prefill with one attention owner per request.

Only Q and projected attention outputs move between TP ranks. Model weights,
MoE, and W_O keep their ordinary TP/EP layout. Owners retain one full latent
history and the ordinary global centroids, without DCP sequence partitioning.
Decode retains request ownership and uses the existing single-GPU head-tiled
LoD decoder, optionally with captured B8 graphs. Prefix reuse is unsupported.
Scheduler slices may be 2K, but centroid updates remain at global 16K
boundaries and the unfinished logical block stays exact.
"""

from __future__ import annotations

from math import prod
import os
from typing import Any
from types import SimpleNamespace

import torch

from .kimi_k3_sharded_prefill import gather_prefill, projection_scope


def take_owner_row(free_rows: list[int], cursor: int, capacity: int) -> tuple[int, int]:
    """Rotate owner allocation even when a smaller cohort releases its rows."""
    row = min(free_rows, key=lambda item: (item - cursor) % capacity)
    free_rows.remove(row)
    return row, (row + 1) % capacity


def retain_owner_record(key: torch.Tensor) -> torch.Tensor:
    """Retain only this row, not a contiguous view of the whole TP batch."""
    return key.permute(1, 0, 2).unsqueeze(0).clone(memory_format=torch.contiguous_format)


def owner_prefill_storage(pool: Any) -> dict | None:
    """Reuse the graph-stable owner cache instead of reserving history twice.

    Only the completed remote state/directory/leaf archive use these buffers.
    The wider prefill exact tail remains temporary; it must not overwrite the
    much smaller decode tail. Installation later copies that tail and skips
    the already-aliased history. This does not change centroid assignments.
    """
    if os.getenv("LOD_KIMI_OWNER_POOL_BACKED_PREFILL") != "1":
        return None
    child = getattr(pool, "owner_decode_pool", None)
    if child is None:
        raise RuntimeError("pool-backed owner prefill requires the captured owner cache")
    storage = child._initial_prefill_storage((0,))
    if storage is None:
        raise RuntimeError("owner decode cache cannot provide direct prefill storage")
    return storage


def transport_workspace(group: Any, query: torch.Tensor, name: str, elements: int,
                        *, reuse: bool = True) -> torch.Tensor:
    """Share two grow-only arenas across sequential MLA layers on this rank.

    The wire arena holds rank-major receives/sends. The payload arena holds
    assembled queries, then return outputs after attention has consumed them.
    RCCL and attention enqueue on the same current stream; neither arena may
    be reused by a concurrent forward or by a caller retaining returned views.
    This eager owner experiment already excludes concurrent/captured forwards.
    """
    if not reuse:
        return query.new_empty(elements)
    buffers = getattr(group, "_lod_owner_transport", None)
    if buffers is None:
        buffers = group._lod_owner_transport = {}
    storage = buffers.get(name)
    if (storage is None or storage.numel() < elements
            or storage.dtype != query.dtype or storage.device != query.device):
        # Grow only on a larger scheduler shape, never at each layer/chunk.
        storage = query.new_empty(elements)
        buffers[name] = storage
        group._lod_owner_transport_allocations = getattr(group, "_lod_owner_transport_allocations", 0) + 1
    return storage[:elements]


def exchange_queries(group: Any, query: torch.Tensor, plan: tuple) -> dict:
    """Send each TP head slice to its row owner, in one grouped RCCL launch."""
    comm = group.device_communicator.pynccl_comm
    if comm is None or comm.disabled:
        raise RuntimeError("request-owner prefill needs an enabled pynccl communicator")
    rank, world = group.rank_in_group, group.world_size
    # Reuse is a capacity-sensitive experiment, not a safe default for the
    # trained model: retaining arenas leaves less room for ROCm launch scratch.
    reuse = os.getenv("LOD_KIMI_OWNER_REUSE_TRANSPORT") == "1"
    owned = [(slot, end - begin) for slot, begin, end, _ in plan if slot % world == rank]
    elements = sum(length for _, length in owned) * world * prod(query.shape[1:])
    wire = transport_workspace(group, query, "wire", elements, reuse=reuse)
    payload = transport_workspace(group, query, "payload", elements, reuse=reuse)
    received, assembled, offset = {}, {}, 0
    for slot, length in owned:
        size = length * world * prod(query.shape[1:])
        received[slot] = wire[offset:offset + size].view(world, length, *query.shape[1:])
        assembled[slot] = payload[offset:offset + size].view(
            length, world * query.size(1), query.size(2))
        offset += size
    comm.group_start()
    try:
        for slot, begin, end, _ in plan:
            owner = slot % world
            source = query[begin:end]
            if not source.is_contiguous():
                raise ValueError("request-owner queries must be contiguous")
            if owner == rank:
                for peer in range(world):
                    if peer == rank:
                        received[slot][peer].copy_(source)
                    else:
                        comm.recv(received[slot][peer], peer)
            else:
                comm.send(source, owner)
    finally:
        comm.group_end()
    for slot, tensor in received.items():
        assembled[slot].view(tensor.size(1), world, *query.shape[1:]).copy_(
            tensor.permute(1, 0, 2, 3))
    return assembled


def exchange_outputs(group: Any, outputs: dict, plan: tuple, query: torch.Tensor) -> torch.Tensor:
    """Return each head's projected output to its original W_O owner."""
    comm = group.device_communicator.pynccl_comm
    rank, world = group.rank_in_group, group.world_size
    reuse = os.getenv("LOD_KIMI_OWNER_REUSE_TRANSPORT") == "1"
    # Attention has consumed the assembled queries. Reuse that arena for
    # returned outputs; also reuse the now-idle receive arena for packed sends.
    result = transport_workspace(group, query, "payload", query.size(0) * query.size(1) * 128,
                                 reuse=reuse).view(
        query.size(0), query.size(1), 128)
    wire = transport_workspace(group, query, "wire", sum(output.numel() for output in outputs.values()),
                               reuse=reuse)
    sources, offset = {}, 0
    for slot, output in outputs.items():
        source = wire[offset:offset + output.numel()].view(world, output.size(0), query.size(1), 128)
        source.copy_(output.view(output.size(0), world, query.size(1), 128).permute(1, 0, 2, 3))
        sources[slot] = source
        offset += output.numel()
    comm.group_start()
    try:
        for slot, begin, end, _ in plan:
            owner = slot % world
            if owner == rank:
                for peer in range(world):
                    if peer == rank:
                        result[begin:end].copy_(sources[slot][peer])
                    else:
                        comm.send(sources[slot][peer], peer)
            else:
                comm.recv(result[begin:end], owner)
    finally:
        comm.group_end()
    return result


@torch.inference_mode()
def advance_block(engine: Any, cache: Any, block: torch.Tensor, *, total: int,
                  working: torch.Tensor | None = None) -> None:
    """The same final update as a non-finalized 16K cached-prefill call."""
    state = cache.state
    recent = state["recent_k"][..., :int(state["recent_len"]), :]
    if working is None:
        working = torch.cat((recent, block), dim=2)
    elif working.size(2) != recent.size(2) + block.size(2):
        raise ValueError("request-owner update workspace does not contain the recent tail and block")
    old_coverage = int(state["coverage"])
    target = total - (engine.prefill_local_len - engine.prefill_chunk_len)
    coverage = old_coverage
    while coverage < target:
        end = min(target, coverage + engine.prefill_state_update_len)
        overflow = working[..., coverage - old_coverage:end - old_coverage, :]
        sk, sv, counts, length, owners, remap = engine._update_state(
            state["state_k"], state["state_v"], state["counts"],
            state.get("key_norm_sums"), overflow, overflow[..., :512],
            state_len=int(state["state_len"]),
            ctx_len=min(total, end + engine.local_len), available_context=end,
            state_capacity=int(state["state_capacity"]),
            scheduled_state_len=int(state["scheduled_state_len"]),
        )
        if remap is not None:
            raise AssertionError("request-owner prefill cannot remap centroid IDs")
        engine._append_page_cache(state["page_cache"], overflow, overflow[..., :512], owners)
        state.update(state_k=sk, state_v=sv, counts=counts,
                     state_len=length, scheduled_state_len=length)
        coverage = end
    tail = working[..., coverage - old_coverage:, :].clone(memory_format=torch.contiguous_format)
    state.update(coverage=coverage, total_len=total, recent_len=tail.size(2),
                 recent_k=tail, recent_v=tail[..., :512])


@torch.inference_mode()
def attend_slice(pool: Any, slot: int, query: torch.Tensor, key: torch.Tensor,
                 uk: torch.Tensor, uv: torch.Tensor, *, previous: int, prompt: int,
                 head_group_limit: int = 12) -> torch.Tensor:
    """Attend a scheduler slice without changing the logical 16K blocks."""
    engine = pool.engine
    chunk = int(engine.prefill_chunk_len)
    length = query.size(0)
    if previous % chunk + length > chunk:
        raise ValueError("request-owner scheduler slices must not cross a 16K block")
    rows = pool._kimi_request_owner_rows
    row = rows.setdefault(slot, {"cache": None, "parts": [], "total": 0})
    if row["total"] != previous:
        raise RuntimeError("request-owner logical sequence length drifted")
    row["parts"].append(retain_owner_record(key))
    block = row["parts"][0] if len(row["parts"]) == 1 else torch.cat(row["parts"], dim=2)
    cache = row["cache"]
    local = block if cache is None else torch.cat((
        cache.state["recent_k"][..., :int(cache.state["recent_len"]), :], block,
    ), dim=2)
    expanded = query.permute(1, 0, 2).unsqueeze(0)
    carrier = key.permute(1, 0, 2).unsqueeze(0).expand(1, query.size(1), -1, -1)
    with projection_scope(engine, expanded, uk, uv, head_group_limit=head_group_limit):
        engine._lod_kimi_expanded_prefill_chunk = expanded
        try:
            local_branch = engine._prefill_local_attention(
                carrier, local, local[..., :512], query_offset=local.size(2) - length,
            )
            if cache is None:
                result = local_branch[0]
            else:
                state = cache.state
                result = engine._two_level_attention(
                    carrier, local, local[..., :512], state["state_k"], state["state_v"],
                    state["counts"], None, local, local[..., :512],
                    key_norm_sums=state.get("key_norm_sums"),
                    state_len=int(state["state_len"]), state_capacity=int(state["state_capacity"]),
                    page_cache=state["page_cache"], local_branch=local_branch,
                    sink_k=state["sink_k"], sink_v=state["sink_v"],
                    context_len=previous + length,
                )
        finally:
            del engine._lod_kimi_expanded_prefill_chunk
    total = previous + length
    if total % chunk == 0 or total == prompt:
        if cache is None:
            storage = owner_prefill_storage(pool)
            engine._lod_prefill_cache_capacity = prompt
            if storage is not None:
                engine._lod_prefill_storage = storage
            try:
                cache = engine.build_cache_from_bf16(
                    block, block[..., :512], finalize_cache_for_decode=False,
                )
            finally:
                del engine._lod_prefill_cache_capacity
                if storage is not None:
                    del engine._lod_prefill_storage
            if storage is not None:
                # The remote buffers alias the fixed pool, but the exact tail
                # is a prefill view. Use the normal partial-copy installation,
                # not the stricter all-tensors-alias pool-backed contract.
                cache.state["pool_backed"] = False
                cache.state["owner_remote_pool_backed"] = True
            row["cache"] = cache
        else:
            # Reuse the same chronological local field that attention just
            # consumed, rather than concatenating recent+block a second time.
            advance_block(engine, cache, block, total=total, working=local)
        row["parts"].clear()
        engine.reset_runtime_cache()
    row["total"] = total
    return result.squeeze(0).permute(1, 0, 2).contiguous()


def request_owner_prefill(layer: Any, pool: Any, query: torch.Tensor,
                          record: torch.Tensor) -> torch.Tensor:
    plan = pool.direct_prefill_plan
    pool.direct_prefill_plan = None
    if not plan:
        # vLLM's synthetic model profiling must not execute collectives or
        # create semantic rows; ordinary decode is intentionally unsupported.
        return query.new_zeros(query.size(0), query.size(1), 128)
    prompts = pool.direct_prefill_prompt_lengths
    pool.direct_prefill_prompt_lengths = {}
    # Distinguish eight live prefills from sequential smaller owner waves in
    # capacity reports. These are CPU metadata, with no timing events or GPU
    # synchronization added to the attention path.
    pool._kimi_owner_max_plan_rows = max(
        getattr(pool, "_kimi_owner_max_plan_rows", 0), len(plan))
    if len(plan) == pool.dcp_world_size:
        pool._kimi_owner_full_cohort_previous = max(
            getattr(pool, "_kimi_owner_full_cohort_previous", 0),
            min(previous for _, _, _, previous in plan))
    group = pool.dcp_group
    if getattr(layer, "_lod_owner_uk", None) is None:
        layer._lod_owner_uk = gather_prefill(group, layer.W_UK_T.contiguous(), dim=0)
        layer._lod_owner_uv = gather_prefill(group, layer.W_UV.contiguous(), dim=0)
    queries = exchange_queries(group, query, plan)
    sizes = getattr(pool, "_kimi_owner_query_sizes", None)
    if sizes is None:
        sizes = pool._kimi_owner_query_sizes = {}
    for owned_query in queries.values():
        length = int(owned_query.size(0))
        sizes[length] = sizes.get(length, 0) + 1
    outputs = {}
    for slot, begin, end, previous in plan:
        if slot in queries:
            if previous >= prompts[slot]:
                outputs[slot] = attend_decode(
                    pool, slot, queries[slot], record[begin:end],
                    layer._lod_owner_uk, layer._lod_owner_uv, previous=previous,
                )
            else:
                outputs[slot] = attend_slice(
                    pool, slot, queries[slot], record[begin:end],
                    layer._lod_owner_uk, layer._lod_owner_uv,
                    previous=previous, prompt=prompts[slot],
                    head_group_limit=int(os.getenv("LOD_KIMI_OWNER_PREFILL_HEAD_GROUP", "12")),
                )
        # Runtime row bookkeeping is replicated, not the semantic cache.
        pool.ready[slot] = True
        pool.dcp_sharded[slot] = False
        pool.metadata[slot].update(total_len=previous + end - begin, coverage=0)
    pool.direct_prefill_calls += 1
    return exchange_outputs(group, outputs, plan, query)


@torch.inference_mode()
def attend_decode(pool: Any, slot: int, query: torch.Tensor, record: torch.Tensor,
                  uk: torch.Tensor, uv: torch.Tensor, *, previous: int) -> torch.Tensor:
    """One request, all 96 heads, one cache; retain the native 256 cadence.

    Install the completed prefill into the normal fixed-address single-GPU
    pool once. The old cache is then released, not retained as a second copy.
    TP head exchange is outside this pool; no DCP routing/LSE collectives run.
    """
    from ..pool import VLLMLayerLODPool
    from .kimi_k3 import absorb_query

    if query.size(0) != 1 or record.size(0) != 1:
        raise NotImplementedError("request-owner decode requires one token per request")
    row = pool._kimi_request_owner_rows[slot]
    if row["total"] != previous or row["parts"]:
        raise RuntimeError("request-owner decode has no completed matching prefill")
    decode = row.get("decode_pool")
    if decode is None:
        layer = SimpleNamespace(
            num_heads=query.size(1), num_kv_heads=1, head_size=576,
            kv_lora_rank=512, scale=float(pool.engine.scaling),
            _vllm_lod_absorbed_mla=True,
        )
        decode = VLLMLayerLODPool(
            layer, settings=pool.settings, max_requests=1,
            request_capacity=pool.request_capacity,
            active_indices=torch.zeros(1, dtype=torch.long, device=query.device),
            dtype=query.dtype, device=query.device, request_owner_prefill=False,
        )
        # Prefill-only recent storage may have a wide unused capacity. Install
        # only the live tail, keeping K/V latent aliasing in the source cache.
        cache = row["cache"]
        recent_len = int(cache.state["recent_len"])
        cache.state["recent_k"] = cache.state["recent_k"][..., :recent_len, :]
        cache.state["recent_v"] = cache.state["recent_k"][..., :512]
        decode.install(0, cache)
        decode.active_decode_rows = (0,)
        row["decode_pool"] = decode
        row["cache"] = None
        pool.engine.reset_runtime_cache()
    before = decode.catch_up_batches
    decode.catch_up_many([(0, previous)])
    pool._kimi_owner_decode_updates = getattr(pool, "_kimi_owner_decode_updates", 0) + (
        decode.catch_up_batches - before)
    q = absorb_query(query, uk, nope_dim=128).contiguous()
    value = record[..., :512]
    output = q.new_empty(1, query.size(1), 512)
    decode.decode_dcp(q, record, value, output)
    # Native absorbed-MLA returns a latent value per head; W_UV is linear.
    result = torch.bmm(output.transpose(0, 1), uv).transpose(0, 1).contiguous()
    row["total"] = previous + 1
    pool._kimi_owner_decode_tokens = getattr(pool, "_kimi_owner_decode_tokens", 0) + 1
    return result

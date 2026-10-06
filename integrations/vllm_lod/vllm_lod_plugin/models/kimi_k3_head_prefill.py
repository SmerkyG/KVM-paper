"""Experimental six 16-head prefill owners, with native TP8 projections.

Heads, not history, are partitioned. Each owner retains the same full latent
history and global centroids; outputs return to their original 12-head TP
shards. KDA/MoE and model weights remain TP8/EP8. This B1, eager-prefill probe
does not implement decode, prefix reuse, or a production serving policy.
"""

from __future__ import annotations

from typing import Any

import torch

from .kimi_k3_request_prefill import attend_slice, transport_workspace


def head_transfers() -> tuple[tuple[int, int, int, int, int], ...]:
    """(owner, TP source, source offset, owner offset, width), in head order."""
    transfers = []
    for owner in range(6):
        for source in range(8):
            first = max(owner * 16, source * 12)
            last = min((owner + 1) * 16, (source + 1) * 12)
            if first < last:
                transfers.append((owner, source, first - source * 12,
                                  first - owner * 16, last - first))
    return tuple(transfers)


def exchange_heads(group: Any, tensor: torch.Tensor, *, returning: bool = False) -> torch.Tensor:
    """Regroup 8x12 into 6x16 heads, or return 6x16 to native 8x12 shards.

RCCL receives require contiguous packets: use one packed wire arena, then
copy the small head ranges into the token-major payload. Both arenas are
shared across sequential MLA layers; no all-head all-gather or LSE merge.
"""
    rank = group.rank_in_group
    if group.world_size != 8 or not 0 <= rank < 8 or tensor.ndim != 3:
        raise ValueError("head-owner exchange requires TP8 token/head/channel tensors")
    expected = (16 if rank < 6 else 0) if returning else 12
    if tensor.size(1) != expected:
        raise ValueError("head-owner exchange has the wrong input head range")
    comm = group.device_communicator.pynccl_comm
    if comm is None or comm.disabled:
        raise RuntimeError("head-owner prefill requires an enabled RCCL communicator")
    transfers = head_transfers()
    # Reversal preserves the same packet ordering on both endpoints.
    packets = [(source, owner, owner_offset, source_offset, width)
               if returning else (owner, source, source_offset, owner_offset, width)
               for owner, source, source_offset, owner_offset, width in transfers]
    tokens, _, channels = tensor.shape
    output_heads = 12 if returning else (16 if rank < 6 else 0)
    elements = sum(width * tokens * channels for dest, source, *_, width in packets
                   if dest != source and rank in (dest, source))
    wire = transport_workspace(group, tensor, "head_wire", elements)
    result = transport_workspace(group, tensor, "head_payload", tokens * output_heads * channels).view(
        tokens, output_heads, channels)
    sends, receives, offset = {}, {}, 0
    for index, (dest, source, source_offset, output_offset, width) in enumerate(packets):
        if rank == source == dest:
            result[:, output_offset:output_offset + width].copy_(
                tensor[:, source_offset:source_offset + width])
        elif rank in (dest, source):
            size = tokens * width * channels
            packet = wire[offset:offset + size].view(tokens, width, channels)
            offset += size
            if rank == source:
                packet.copy_(tensor[:, source_offset:source_offset + width])
                sends[index] = packet
            else:
                receives[index] = packet
    comm.group_start()
    try:
        for index, (dest, source, _, _, _) in enumerate(packets):
            if index in sends:
                comm.send(sends[index], dest)
            elif index in receives:
                comm.recv(receives[index], source)
    finally:
        comm.group_end()
    for index, packet in receives.items():
        _, _, _, output_offset, width = packets[index]
        result[:, output_offset:output_offset + width].copy_(packet)
    return result


def head_owner_prefill(layer: Any, pool: Any, query: torch.Tensor,
                       record: torch.Tensor) -> torch.Tensor:
    """Six independent head ranges; identical full-context state per owner."""
    plan = pool.direct_prefill_plan
    pool.direct_prefill_plan = None
    if not plan:
        return query.new_zeros(query.size(0), query.size(1), 128)
    prompts = pool.direct_prefill_prompt_lengths
    pool.direct_prefill_prompt_lengths = {}
    if len(plan) != 1:
        raise NotImplementedError("six-head-owner prototype requires one prefill request")
    slot, begin, end, previous = plan[0]
    if begin != 0 or end != query.size(0) or previous >= prompts[slot]:
        raise NotImplementedError("six-head-owner prototype is prefill only")
    rank = pool.dcp_rank
    owned_query = exchange_heads(pool.dcp_group, query)
    if rank < 6:
        if getattr(layer, "_lod_head_owner_uk", None) is None:
            raise RuntimeError("prepare six-owner latent head maps before prefill")
        sizes = getattr(pool, "_kimi_owner_query_sizes", None)
        if sizes is None:
            sizes = pool._kimi_owner_query_sizes = {}
        sizes[end - begin] = sizes.get(end - begin, 0) + 1
        pool._kimi_head_owner_range = (rank * 16, (rank + 1) * 16)
        # Native K3's normalized latent/direct record is already replicated
        # after TP projections. No additional KV broadcast is required.
        attended = attend_slice(pool, slot, owned_query, record,
            layer._lod_head_owner_uk, layer._lod_head_owner_uv,
            previous=previous, prompt=prompts[slot], head_group_limit=16)
        row = pool._kimi_request_owner_rows[slot]
        state = row["cache"].state if row["cache"] is not None else None
        pool._kimi_head_owner_last_state = dict(total_len=row["total"],
            coverage=None if state is None else int(state["coverage"]),
            state_len=0 if state is None else int(state["state_len"]),
            prefill_update_len=int(pool.engine.prefill_state_update_len), query_heads=16)
    else:
        attended = query.new_empty(query.size(0), 0, 128)
    result = exchange_heads(pool.dcp_group, attended, returning=True)
    pool.ready[slot] = True
    pool.dcp_sharded[slot] = False
    pool.metadata[slot].update(total_len=previous + end - begin, coverage=0)
    pool._kimi_owner_max_plan_rows = 1
    pool.direct_prefill_calls += 1
    return result

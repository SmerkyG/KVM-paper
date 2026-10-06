"""Global-centroid K3 prefill with DCP-owned BF16 leaf storage.

The existing cross-layer builder still updates identical whole-sequence
centroids on every rank. Only the exact chronological archive is sharded.
Routing runs on the query-head owner, selects eight GLOBAL centroids, and
uses GLOBAL leaf counts for the cap. Distributed fine attention replaces
each selected coarse contribution once, after combining disjoint leaves.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import prod
import os
from typing import Any

import torch


def owned_bf16_shadow_records(pool: Any, source: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract chronological DCP rows from sink, archived prefix, and exact tail.

    The archive is only authoritative below coverage. Its reserved tail can
    contain uninitialized bytes, even though its capacity covers the prompt.
    Unit-interleaved ownership gives static strided views; concatenate only
    when the row spans multiple authoritative fields.
    """
    if not pool.is_absorbed_mla or pool.dcp_interleave_size != 1:
        raise ValueError("strided BF16 shadow extraction requires unit-interleaved MLA")
    page = source["page_cache"]
    if page.get("quantization_finalized") or page.get("dcp_leaf_sharded"):
        raise ValueError("expected an unsharded BF16 chronological archive")
    total, coverage = int(source["total_len"]), int(source["coverage"])
    leaves, recent = page["leaf_k"], source["recent_k"]
    sink = source.get("sink_k")
    sink_len = int(sink.size(2)) if isinstance(sink, torch.Tensor) else 0
    rank, world = int(pool.dcp_rank), int(pool.dcp_world_size)
    if (not sink_len <= coverage <= total
            or int(leaves.size(2)) < coverage - sink_len
            or int(recent.size(2)) < total - coverage
            or int(source.get("recent_len", total - coverage)) != total - coverage):
        raise ValueError("DCP shadow has inconsistent authoritative prefix/tail extents")
    parts = []
    if rank < sink_len:
        parts.append(sink[..., rank:sink_len:world, :])
    first_archive = rank + max(0, (sink_len - rank + world - 1) // world) * world
    if first_archive < coverage:
        parts.append(leaves[..., first_archive - sink_len:coverage - sink_len:world, :])
    first_recent = (rank - coverage) % world
    if first_recent < total - coverage:
        parts.append(recent[..., first_recent:total - coverage:world, :])
    key = (leaves[..., :0, :] if not parts else parts[0] if len(parts) == 1
           else torch.cat(parts, dim=2))
    if int(key.size(2)) != pool._dcp_local_length(total):
        raise AssertionError("DCP shadow extraction produced the wrong local length")
    return key, key[..., :pool.value_dim], total


def gather_prefill(group: Any, tensor: torch.Tensor, *, dim: int,
                   buffer: torch.Tensor | None = None) -> torch.Tensor:
    """Enqueue RCCL on the current stream without a host-side wait per layer."""
    comm = getattr(getattr(group, "device_communicator", None), "pynccl_comm", None)
    if comm is None or comm.disabled:
        return group.all_gather(tensor, dim=dim)
    tensor = tensor.contiguous()
    shape = (group.world_size, *tensor.shape)
    if buffer is not None and (
        tuple(buffer.shape) != shape or not buffer.is_contiguous()
        or buffer.dtype != tensor.dtype or buffer.device != tensor.device
    ):
        raise ValueError("prefill gather workspace has incompatible geometry")
    gathered = buffer if buffer is not None else torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    comm.all_gather(gathered.view(-1, *tensor.shape[1:]), tensor)
    shape = (*tensor.shape[:dim], group.world_size * tensor.size(dim), *tensor.shape[dim + 1:])
    return gathered.movedim(0, dim).reshape(shape)


def reduce_scatter_prefill(group: Any, tensor: torch.Tensor, *, dim: int,
                           buffer: torch.Tensor | None = None) -> torch.Tensor:
    """Sum disjoint leaf outputs into head owners using reusable receive scratch.

    Like the native pynccl path, this enqueues on the current stream. It does
    not add a host wait or assume that avoiding an allocation is a speed win.
    """
    comm = getattr(getattr(group, "device_communicator", None), "pynccl_comm", None)
    if comm is None or comm.disabled:
        return group.reduce_scatter(tensor, dim=dim)
    if tensor.size(dim) % group.world_size:
        raise ValueError("prefill reduce-scatter heads must divide across ranks")
    source = tensor.movedim(dim, 0).contiguous()
    shape = (source.size(0) // group.world_size, *source.shape[1:])
    if buffer is not None and (
        tuple(buffer.shape) != shape or not buffer.is_contiguous()
        or buffer.dtype != tensor.dtype or buffer.device != tensor.device
    ):
        raise ValueError("prefill reduce-scatter workspace has incompatible geometry")
    output = buffer if buffer is not None else torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    comm.reduce_scatter(output, source)
    return output.movedim(0, dim).contiguous()


def archive_workspace(buffers: dict | None, name: str, shape: tuple, *,
                      token_axis: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Grow shared scratch geometrically, not at every 16K boundary."""
    from lod_attention.kernels.aiter_prefill_attention import _workspace_tensor

    capacity = list(shape)
    capacity[token_axis] = 1 << (int(shape[token_axis]) - 1).bit_length()
    storage = _workspace_tensor(buffers, name, tuple(capacity), dtype=dtype, device=device)
    return storage.reshape(-1)[:prod(shape)].view(shape)


def combine_prefill_partials(
    pool: Any, fine: torch.Tensor, fine_lse: torch.Tensor,
    *, heads: int, buffers: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LSE-weight disjoint leaf fields and scatter them to query-head owners."""
    from lod_attention.kernels.kimi_sharded_prefill import weight_prefill_partials_

    batch, all_heads, queries = fine_lse.shape
    world = pool.dcp_world_size
    if all_heads != heads * world or fine.shape[:3] != fine_lse.shape:
        raise ValueError("DCP partial outputs do not match query-head ownership")
    lse_buffer = archive_workspace(buffers, "sharded_prefill_lses", (world, *fine_lse.shape),
                                   token_axis=3, dtype=fine_lse.dtype, device=fine_lse.device)
    gathered_lses = gather_prefill(pool.dcp_group, fine_lse, dim=0, buffer=lse_buffer).view(
        world, batch, all_heads, queries,
    )
    total_buffer = archive_workspace(buffers, "sharded_prefill_total_lse", tuple(fine_lse.shape),
                                     token_axis=2, dtype=torch.float32, device=fine_lse.device)
    total_lse = weight_prefill_partials_(fine, gathered_lses, rank=pool.dcp_rank, total_lse=total_buffer)
    receive = archive_workspace(buffers, "sharded_prefill_output", (heads, batch, queries, fine.size(-1)),
                                token_axis=2, dtype=fine.dtype, device=fine.device)
    fine = reduce_scatter_prefill(pool.dcp_group, fine, dim=1, buffer=receive)
    return fine, total_lse[:, pool.dcp_rank * heads:(pool.dcp_rank + 1) * heads]


@contextmanager
def projection_scope(engine: Any, query: torch.Tensor, uk: torch.Tensor, uv: torch.Tensor,
                     *, head_group_limit: int | None = None):
    """Replace temporary maps safely even when vLLM supplied nn.Parameters."""
    values = dict(_lod_kimi_expanded_prefill_chunk=query, _lod_kimi_w_uk_t=uk, _lod_kimi_w_uv=uv)
    if head_group_limit is not None:
        if head_group_limit < 1 or query.size(1) % head_group_limit:
            raise ValueError("prefill head group must divide the query heads")
        values["_lod_kimi_prefill_head_group_limit"] = head_group_limit
    original = {name: getattr(engine, name) for name in values if hasattr(engine, name)}
    try:
        for name, value in values.items():
            if hasattr(engine, name):
                delattr(engine, name)  # clear Module's parameter registration
            setattr(engine, name, value)
        yield
    finally:
        for name in values:
            if hasattr(engine, name):
                delattr(engine, name)
            if name in original:
                setattr(engine, name, original[name])


def owned_slice(tensor: torch.Tensor, *, begin: int, rank: int, world: int) -> torch.Tensor:
    """Select globally owned positions without a GPU mask or synchronization."""
    if begin < 0 or world <= 0 or not 0 <= rank < world:
        raise ValueError("invalid global DCP ownership")
    return tensor[..., (rank - begin) % world :: world, :]


def owned_owners(owners: torch.Tensor, *, begin: int, rank: int, world: int) -> torch.Tensor:
    return owned_slice(owners.unsqueeze(-1), begin=begin, rank=rank, world=world).squeeze(-1)


def add_global_counts(lengths: torch.Tensor, owners: torch.Tensor) -> None:
    """Keep physical global leaf counts separate from rank-local page lengths."""
    valid = owners.ge(0) & owners.lt(lengths.size(-1))
    lengths.scatter_add_(-1, owners.clamp(0, lengths.size(-1) - 1).long(), valid.to(lengths.dtype))


def b1_leaf_storage(pool: Any, *, batch: int) -> torch.Tensor | None:
    """Use the idle B1 decode archive, but not its rank-local centroid directory.

    Prefill uses global centroid IDs; decode rebuilds rank-local centroids at
    handoff. Their directories cannot alias. The chronological rank-owned
    records can, because the final builder packs its inputs before installing
    decode state. Keep the exact prefill tail outside the smaller decode tail.
    Multi-request prefills retain the existing separate storage path.
    """
    if batch != 1 or getattr(pool, "max_requests", None) != 1:
        return None
    page = pool.state.get("page_cache")
    if not isinstance(page, dict) or page.get("quantization_finalized"):
        raise ValueError("B1 sharded prefill requires a BF16 decode archive")
    key = page.get("leaf_k")
    value = page.get("leaf_v")
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("B1 sharded prefill is missing decode leaf storage")
    if (key.size(0) != 1 or key.size(-1) != 576 or value.size(-1) != 512
            or key.data_ptr() != value.data_ptr() or key.stride() != value.stride()):
        raise ValueError("B1 sharded prefill must retain K/V latent-prefix aliasing")
    return key


def initial_page(
    pool: Any, archive: torch.Tensor, owners: torch.Tensor, *,
    sink_len: int, coverage: int, prompt_capacity: int, state_capacity: int,
) -> dict:
    engine = pool.engine
    rank, world = pool.dcp_rank, pool.dcp_world_size
    local = owned_slice(archive, begin=sink_len, rank=rank, world=world).contiguous()
    local_owners = owned_owners(owners, begin=sink_len, rank=rank, world=world).contiguous()
    covered = local[..., :local_owners.size(-1), :]
    capacity = pool._dcp_local_length(prompt_capacity) + max(32, engine.decode_cache_headroom)
    backing = b1_leaf_storage(pool, batch=int(local.size(0)))
    if backing is not None:
        # The fixed pool already sizes headroom in DCP-local tokens. Adding
        # the engine's global headroom again rejects a valid rank-local arena.
        if int(backing.size(2)) < pool._dcp_local_length(prompt_capacity):
            raise ValueError("B1 decode archive cannot hold the sharded prompt and local headroom")
        capacity = int(backing.size(2))
    storage_kwargs = {"metadata_only": True} if backing is not None else {}
    page = engine._new_page_cache(
        covered, covered[..., :pool.value_dim], local_owners,
        state_capacity=state_capacity, sequence_capacity=capacity,
        virtual_k=local, virtual_v=local[..., :pool.value_dim],
        **storage_kwargs,
    )
    if backing is not None:
        # Initial metadata construction deliberately does not allocate/write
        # chronological records. Copy the authoritative prefix/tail once,
        # then ordinary append kernels write directly into the fixed archive.
        backing[..., :local.size(2), :].copy_(local)
        page.update(leaf_k=backing, leaf_v=backing[..., :pool.value_dim],
                    leaf_capacity=int(backing.size(2)), metadata_only=False,
                    dcp_prefill_pool_backed=True)
    page["dcp_leaf_sharded"] = True
    if os.environ.get("LOD_KIMI_DCP_PREFILL_WORKSPACE") == "1":
        directory = engine._new_page_cache(
            archive[..., :owners.size(2), :], archive[..., :owners.size(2), :pool.value_dim], owners.long(),
            state_capacity=state_capacity, sequence_capacity=prompt_capacity + engine.decode_cache_headroom,
            virtual_k=archive, virtual_v=archive[..., :pool.value_dim], metadata_only=True,
        )
        page["global_directory"] = directory
        page["global_slot_lengths"] = directory["slot_lengths"]
    else:
        lengths = torch.zeros_like(page["slot_lengths"])
        add_global_counts(lengths, owners)
        page["global_slot_lengths"] = lengths
    expected = pool._dcp_local_length(coverage) - int(rank < sink_len)
    if int(page["leaf_count"]) != expected:
        raise AssertionError("initial DCP archive does not match global coverage")
    return page


def append_page(
    pool: Any, page: dict, working: torch.Tensor, owners: torch.Tensor, *,
    previous_coverage: int, coverage: int, total_len: int,
    owner_ranks: torch.Tensor | None = None,
) -> None:
    """Insert owned overflow, but retain global centroid IDs and cardinalities."""
    from lod_attention.kernels.paged_leaf_attention import stable_owner_ranks

    rank, world = pool.dcp_rank, pool.dcp_world_size
    overflow_len = coverage - previous_coverage
    local = owned_slice(working[..., :overflow_len, :], begin=previous_coverage,
                        rank=rank, world=world).contiguous()
    local_owners = owned_owners(owners, begin=previous_coverage, rank=rank, world=world).contiguous()
    pool.engine._append_page_cache(
        page, local, local[..., :pool.value_dim], local_owners.long(),
        owner_ranks=stable_owner_ranks(local_owners).long(),
    )
    if "global_directory" in page:
        pool.engine._append_page_cache(
            page["global_directory"], working[..., :overflow_len, :],
            working[..., :overflow_len, :pool.value_dim], owners.long(), owner_ranks=owner_ranks,
        )
    else:
        add_global_counts(page["global_slot_lengths"], owners)
    # Archive the owned exact tail too, so final decode construction reads a
    # complete chronological local prefix without re-gathering global history.
    tail = owned_slice(working[..., overflow_len:overflow_len + total_len - coverage, :],
                       begin=coverage, rank=rank, world=world)
    count = int(page["leaf_count"])
    page["leaf_k"][..., count:count + tail.size(2), :].copy_(tail)


def owned_history(pool: Any, source: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the already-sharded raw history, including this rank's sink."""
    total = int(source["total_len"])
    sink = source["sink_k"]
    sink_len = int(sink.size(2))
    owned_sink = owned_slice(sink, begin=0, rank=pool.dcp_rank, world=pool.dcp_world_size)
    expected = pool._dcp_local_length(total)
    archive_len = expected - owned_sink.size(2)
    archive = source["page_cache"]["leaf_k"][..., :archive_len, :]
    key = torch.cat((owned_sink, archive), dim=2) if owned_sink.size(2) else archive
    if key.size(2) != expected:
        raise AssertionError("sharded history has the wrong chronological length")
    return key, key[..., :pool.value_dim], total


def attention(
    pool: Any, cache: Any, query: torch.Tensor, record: torch.Tensor,
    w_uk_t: torch.Tensor, w_uv: torch.Tensor, *, output_buffer: torch.Tensor | None = None,
    replicated_query: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same global coarse-replacement calculation, distributed exact leaves."""
    from lod_attention.kernels.aiter_mla_prefill_attention import (
        aiter_kimi_expanded_prefill_route_coarse_attention,
        aiter_kimi_local_prefill_attention,
        merge_aiter_mla_prefill_refinement,
    )
    state, engine = cache.state, pool.engine
    page = state["page_cache"]
    batch, heads, queries, _ = query.shape
    # Workspace reconstruction keeps native head ownership, with no query
    # gather or partial-output reduction. Only one layer is reconstructed.
    workspace = "global_directory" in page
    layer = pool.layer
    if not workspace and getattr(layer, "_lod_sharded_w_uk_t", None) is None:
        layer._lod_sharded_w_uk_t = gather_prefill(pool.dcp_group, w_uk_t, dim=0)
        layer._lod_sharded_w_uv = gather_prefill(pool.dcp_group, w_uv, dim=0)
    exact = torch.cat((state["sink_k"], state["recent_k"][..., :int(state["recent_len"]), :], record), dim=2)
    carrier = record.expand(batch, heads, queries, 576)
    buffers = getattr(engine, "_lod_prefill_attention_buffers", None)
    foreground = torch.cuda.current_stream(query.device)
    local_stream = getattr(engine, "_lod_prefill_local_stream", None)
    if local_stream is None:
        local_stream = torch.cuda.Stream(device=query.device)
        engine._lod_prefill_local_stream = local_stream
    local_stream.wait_stream(foreground)
    with torch.cuda.stream(local_stream):
        local, local_lse = aiter_kimi_local_prefill_attention(
            carrier, exact, query_offset=exact.size(2) - queries,
            scale=float(engine.scaling), expanded_q=query, w_uk_t=w_uk_t, w_uv=w_uv,
            buffers=buffers,
        )
    if workspace:
        # Prefetch raw KV while coarse scoring and exact local attention run.
        comm_stream = getattr(pool.dcp_group, "_lod_kimi_prefill_gather_stream", None)
        if comm_stream is None:
            comm_stream = torch.cuda.Stream(device=query.device)
            pool.dcp_group._lod_kimi_prefill_gather_stream = comm_stream
        comm_stream.wait_stream(foreground)
        with torch.cuda.stream(comm_stream):
            full_keys = reconstruct_leaves(pool, state)
    slots, coarse, _, _ = aiter_kimi_expanded_prefill_route_coarse_attention(
        query, state["state_k"].contiguous(), state["state_v"].contiguous(),
        state["counts"].contiguous(), w_uk_t, w_uv,
        state_len=int(state["state_len"]), scale=float(engine.scaling),
        normalize_route_query=False, slot_lengths=page["global_slot_lengths"],
        max_open_leaf_tokens=engine.max_open_centroid_leaves, buffers=buffers,
    )
    if coarse.ready_stream is not None:
        torch.cuda.current_stream(query.device).wait_stream(coarse.ready_stream)
    if workspace:
        foreground.wait_stream(comm_stream)
        directory = dict(page["global_directory"], leaf_k=full_keys, leaf_v=full_keys[..., :pool.value_dim])
        with projection_scope(engine, query, w_uk_t, w_uv):
            fine, fine_lse = engine._paged_leaf_attention(
                carrier, slots, directory, active_slots=int(state["state_len"]), reduce_routes=True,
            )
        foreground.wait_stream(local_stream)
        return merge_aiter_mla_prefill_refinement(
            carrier, state["sink_k"][..., :0, :], coarse.mean_v[..., :0, :],
            coarse, slots, fine, fine_lse, local, local_lse,
            kv_group_size=heads, scale=float(engine.scaling), output_buffer=output_buffer,
        )
    world = pool.dcp_world_size
    # Fixed-size query/partial scratch is shared across layers just like the
    # projected leaf workspace. All consumers below are on the foreground
    # stream, so the next layer cannot overwrite it before reduce-scatter.
    slot_buffer = archive_workspace(buffers, "sharded_prefill_slots", (world, *slots.shape),
                                    token_axis=3, dtype=slots.dtype, device=slots.device)
    all_slots = gather_prefill(pool.dcp_group, slots, dim=1, buffer=slot_buffer)
    all_q = (replicated_query if replicated_query is not None
             else gather_prefill(pool.dcp_group, query, dim=1, buffer=archive_workspace(
                 buffers, "sharded_prefill_query", (world, *query.shape), token_axis=3,
                 dtype=query.dtype, device=query.device)))
    if tuple(all_q.shape) != (batch, heads * pool.dcp_world_size, queries, 192):
        raise ValueError("replicated prefill query does not match the DCP group heads")
    all_heads = all_q.size(1)
    lengths = page["slot_lengths"].unsqueeze(2).expand(batch, all_heads, queries, -1)
    local_lengths = torch.gather(lengths, -1, all_slots.clamp_min(0).long())
    local_slots = torch.where(all_slots.ge(0) & local_lengths.gt(0), all_slots, -1)
    fine_carrier = record.expand(batch, all_heads, queries, 576)
    with projection_scope(engine, all_q, layer._lod_sharded_w_uk_t, layer._lod_sharded_w_uv,
                          head_group_limit=heads):
        fine, fine_lse = engine._paged_leaf_attention(
            fine_carrier, local_slots, page,
            active_slots=int(state["state_len"]), reduce_routes=True,
        )
    fine, fine_lse = combine_prefill_partials(pool, fine, fine_lse, heads=heads, buffers=buffers)
    foreground.wait_stream(local_stream)
    # Sink is already in the exact local field. Remove globally selected
    # coarse terms once, not once per rank and not using local leaf counts.
    return merge_aiter_mla_prefill_refinement(
        carrier, state["sink_k"][..., :0, :], coarse.mean_v[..., :0, :],
        coarse, slots, fine, fine_lse, local, local_lse,
        kv_group_size=heads, scale=float(engine.scaling), output_buffer=output_buffer,
    )


def reconstruct_leaves(pool: Any, state: dict) -> torch.Tensor:
    """Gather only the current layer's owned archive into shared workspace.

    This is an experimental latency/memory tradeoff: unlike distributed fine
    attention, it communicates the covered history once per prefill chunk.
    Persistent storage remains sharded; the additional communication is
    quadratic in prompt length for a fixed 16K chunk size.
    """
    from lod_attention.kernels.kimi_sharded_prefill import interleave_shards

    page = state["page_cache"]
    tokens = int(state["coverage"]) - int(state["sink_k"].size(2))
    per_rank = (tokens + pool.dcp_world_size - 1) // pool.dcp_world_size
    keys = page["leaf_k"][..., :int(page["leaf_count"]), :]
    buffers = getattr(pool.engine, "_lod_prefill_attention_buffers", None)
    if keys.size(2) < per_rank or not keys.is_contiguous():
        source = keys
        keys = archive_workspace(buffers, "sharded_prefill_input",
                                 (*source.shape[:2], per_rank, source.size(-1)), token_axis=2,
                                 dtype=source.dtype, device=source.device)
        keys[..., :source.size(2), :].copy_(source)
        keys[..., source.size(2):, :].zero_()
    shape = (pool.dcp_world_size, *keys.shape)
    gather_buffer = archive_workspace(buffers, "sharded_prefill_gather", shape, token_axis=3,
                                      dtype=keys.dtype, device=keys.device)
    # All engines share this workspace and one communication stream. Foreground
    # attention is serialized; that stream waits for the prior fine consumer
    # before overwriting scratch. Separate uncached gathers on 24 streams would
    # retain 24 allocator freelists and erase the peak-VRAM benefit.
    gathered = gather_prefill(pool.dcp_group, keys, dim=0, buffer=gather_buffer).view(shape)
    output = archive_workspace(
        buffers, "sharded_prefill_keys",
        (keys.size(0), keys.size(1), tokens, keys.size(-1)), token_axis=2,
        dtype=keys.dtype, device=keys.device,
    )
    interleave_shards(gathered, output, begin=int(state["sink_k"].size(2)))
    return output

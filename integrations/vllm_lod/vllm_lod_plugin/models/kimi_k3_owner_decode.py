"""Graph-safe fixed B8/TP8 request-owned MLA decode.

Native K3 can supply replicated 96-head queries in decode. Each rank
selects its one request, runs the ordinary single-GPU LoD decoder, and returns
the twelve-head slices to native TP gating/W_O with one reduce-scatter. There
is no distributed routing or LSE merge. Installation and 256-token catch-ups
run in scheduler preprocessing, never inside the captured attention graph.
"""

from types import SimpleNamespace
from typing import Any

import torch


def initialize_owner_decode(pool: Any) -> None:
    from ..pool import VLLMLayerLODPool
    from .kimi_k3_sharded_prefill import gather_prefill

    layer = SimpleNamespace(num_heads=96, num_kv_heads=1, head_size=576,
        kv_lora_rank=512, scale=float(pool.engine.scaling),
        _vllm_lod_absorbed_mla=True)
    pool.owner_decode_pool = VLLMLayerLODPool(layer, settings=pool.settings,
        max_requests=1, request_capacity=pool.request_capacity,
        active_indices=torch.zeros(1, dtype=torch.long, device=pool.device),
        dtype=pool.dtype, device=pool.device, request_owner_prefill=False,
        shared_decode_scratch=getattr(pool, "shared_decode_scratch", None))
    pool.owner_decode_pool.active_decode_rows = (0,)
    pool.owner_decode_pool._dcp_buffers(
        torch.empty(1, 96, 1, 576, device=pool.device, dtype=pool.dtype), 1)
    attention = pool.layer
    attention._lod_owner_uk = gather_prefill(pool.dcp_group, attention.W_UK_T, dim=0)
    attention._lod_owner_uv = gather_prefill(pool.dcp_group, attention.W_UV, dim=0)
    empty = lambda *shape: torch.empty(*shape, dtype=pool.dtype, device=pool.device)
    pool.owner_decode_buffers = dict(
        input_index=torch.tensor([pool.dcp_rank], dtype=torch.long, device=pool.device),
        query=empty(1, 96, 192), key=empty(1, 1, 576),
        q_latent=empty(96, 1, 512), absorbed=empty(1, 96, 576),
        output=empty(1, 96, 512), projected=empty(96, 1, 128),
        send=empty(8, 8, 12, 128), receive=empty(8, 12, 128),
        query_wire=empty(8, 8, 12, 192), replicated_query=empty(8, 96, 192),
    )
    pool.owner_decode_input_row = pool.dcp_rank


def owner_row_index(rows: tuple[int, ...], rank: int) -> int:
    if len(rows) != 8 or set(rows) != set(range(8)):
        raise ValueError("captured owner decode requires eight distinct live owners")
    return rows.index(rank)


def prepare_owner_decode(pool: Any, requests: list[tuple[int, int]], *,
                         catch_up: bool = True) -> None:
    """Host-only lifecycle work, before vLLM launches/replays its graph."""
    rows = tuple(row for row, _ in requests)
    index = owner_row_index(rows, pool.dcp_rank)
    slot, previous = requests[index]
    row = pool._kimi_request_owner_rows[slot]
    if row["total"] != previous or row["parts"]:
        raise RuntimeError("owner decode does not have the matching completed prefix")
    decode = pool.owner_decode_pool
    if row.get("decode_pool") is None:
        cache = row["cache"]
        recent = int(cache.state["recent_len"])
        cache.state["recent_k"] = cache.state["recent_k"][..., :recent, :]
        cache.state["recent_v"] = cache.state["recent_k"][..., :512]
        decode.install(0, cache)
        row.update(decode_pool=decode, cache=None)
        pool.engine.reset_runtime_cache()
    if row["decode_pool"] is not decode:
        raise AssertionError("owner cache pointer changed after graph capture")
    if catch_up:
        before = decode.catch_up_batches
        decode.catch_up_many([(0, previous)])
        decode.ensure_unified_page1_fixed((0,))
        pool._kimi_owner_decode_updates = getattr(pool, "_kimi_owner_decode_updates", 0) + (
            decode.catch_up_batches - before)
    pool._kimi_owner_decode_tokens = getattr(pool, "_kimi_owner_decode_tokens", 0) + 1
    row["total"] = previous + 1
    if pool.owner_decode_input_row != index:
        pool.owner_decode_buffers["input_index"].fill_(index)
        pool.owner_decode_input_row = index
    pool.decode_enabled = True
    pool.direct_prefill_plan = None
    for physical, length in requests:
        pool.metadata[physical].update(total_len=length + 1, coverage=0)


def prepare_owner_decode_batch(runtime: Any, requests: list[tuple[int, int]]) -> None:
    """Install once, then batch compatible layer updates before graph replay."""
    parents = tuple(runtime.pools.values())
    slot = requests[owner_row_index(tuple(row for row, _ in requests), runtime.dcp_rank)][0]
    installing = any(pool._kimi_request_owner_rows[slot].get("decode_pool") is None
                     for pool in parents)
    for pool in parents:
        prepare_owner_decode(pool, requests, catch_up=False)
    if installing:
        # Owner prefill bypasses the ordinary final-construction cleanup.
        # These are the parents' construction-only temporaries, not the
        # children's fixed/captured decode buffers or installed semantic KV.
        # Keeping all 24 layers' large final-overflow workspaces alive can
        # make the next generation OOM even when the first generation fits.
        # Do not clear runtime-wide shared dictionaries: they can already
        # belong to the children's decode catch-up from a prior generation.
        for pool in parents:
            for name in ("_lod_state_update_buffers", "_lod_state_maxsim_buffers"):
                if hasattr(pool.engine, name):
                    delattr(pool.engine, name)
        # The serial owner-prefill scratch is separate from the runtime's
        # layer-batched decode scratch. Its work is complete at this handoff;
        # no captured decode graph or persistent leaf/centroid aliases it.
        getattr(runtime, "_prefill_attention_buffers", {}).pop("kimi_owner_state_update", None)
    children = tuple(pool.owner_decode_pool for pool in parents)
    previous = requests[owner_row_index(tuple(row for row, _ in requests), runtime.dcp_rank)][1]
    before = [pool.catch_up_batches for pool in children]
    # All MLA layers on this rank own the same physical request. Reuse the
    # established layer-batched update with identical global boundaries.
    if not runtime._catch_up_one_across_layers(0, previous, pools=children):
        for child in children:
            child.catch_up_many([(0, previous)])
    for parent, child, initial in zip(parents, children, before, strict=True):
        child.ensure_unified_page1_fixed((0,))
        parent._kimi_owner_decode_updates = getattr(parent, "_kimi_owner_decode_updates", 0) + (
            child.catch_up_batches - initial)
    for row, length in requests:
        runtime.logical_lengths[row] = length + 1


def owner_decode_attention(pool: Any, replicated_query: torch.Tensor,
                           record: torch.Tensor) -> torch.Tensor:
    """Fixed buffers/collective shapes; no host lengths or cache allocations."""
    if replicated_query.shape != (8, 96, 192) or record.shape != (8, 1, 576):
        raise ValueError("owner decode needs the native B8 replicated K3 queries")
    b = pool.owner_decode_buffers
    if replicated_query.is_cuda:
        from lod_attention.kernels.kimi_owner_decode import owner_query_projection
        owner_query_projection(replicated_query,record,pool.layer._lod_owner_uk,
            b["input_index"],b["absorbed"],b["key"])
    else:  # CPU layout tests use the explicit reference formula.
        torch.index_select(replicated_query, 0, b["input_index"], out=b["query"])
        torch.index_select(record, 0, b["input_index"], out=b["key"])
        torch.bmm(b["query"][..., :128].transpose(0, 1), pool.layer._lod_owner_uk,
                  out=b["q_latent"])
        b["absorbed"][..., :512].copy_(b["q_latent"].transpose(0, 1))
        b["absorbed"][..., 512:].copy_(b["query"][..., 128:])
    pool.owner_decode_pool.decode_dcp(
        b["absorbed"], b["key"], b["key"][..., :512], b["output"])
    # Each row has exactly one nonzero contributor. Ordinary sum is exact;
    # this is an output/head transpose, not an attention-mass combination.
    if replicated_query.is_cuda:
        from lod_attention.kernels.kimi_owner_decode import owner_value_projection
        owner_value_projection(b["output"],pool.layer._lod_owner_uv,b["input_index"],b["send"])
    else:
        torch.bmm(b["output"].transpose(0, 1), pool.layer._lod_owner_uv,
                  out=b["projected"])
        b["send"].zero_()
        torch.index_copy(b["send"], 1, b["input_index"],
            b["projected"].reshape(8, 1, 12, 128), out=b["send"])
    comm = pool.dcp_group.device_communicator.pynccl_comm
    if comm is None or comm.disabled:
        raise RuntimeError("captured owner decode requires graph-safe pynccl")
    comm.reduce_scatter(b["receive"], b["send"])
    return b["receive"]


def owner_decode_query(pool: Any, query: torch.Tensor,
                       replicated_query: torch.Tensor | None) -> torch.Tensor:
    """Reuse native head replication, or gather the local head shards once.

    vLLM's metadata-free synthetic startup can omit the replicated query even
    when real decode supplies it. Both branches have fixed graph-safe buffers.
    """
    if replicated_query is not None:
        return replicated_query
    if query.shape != (8,12,192):
        raise ValueError("owner query gather requires native B8 local head shards")
    b = pool.owner_decode_buffers
    comm = pool.dcp_group.device_communicator.pynccl_comm
    if comm is None or comm.disabled:
        raise RuntimeError("owner query gather requires graph-safe pynccl")
    comm.all_gather(b["query_wire"].flatten(0,1), query.contiguous())
    b["replicated_query"].view(8,8,12,192).copy_(
        b["query_wire"].permute(1,0,2,3))
    return b["replicated_query"]


__all__ = ["initialize_owner_decode", "owner_row_index", "prepare_owner_decode",
           "prepare_owner_decode_batch", "owner_decode_attention", "owner_decode_query"]

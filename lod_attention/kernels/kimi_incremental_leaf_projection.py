"""Experimental prefix projection cache for append-only initial prefill."""

from __future__ import annotations

import torch

from .aiter_mla_prefill_attention import expand_kimi_leaf_kv


def expand_incremental_kimi_leaves(
    key: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    page_cache: dict,
    *,
    head_begin: int,
    buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain projected immutable leaves in this request's page cache.

    This prototype uses additional persistent projection storage. It is not
    an INT4 or chat-cache conversion path; dropping the request cache drops
    its projections. The source archive may grow/reallocate, but must retain
    identical chronological prefix leaves. A new archive requires a new
    ``page_cache`` dictionary.
    """
    if key.size(0) != 1:
        raise ValueError("incremental projection prototype requires one prefill row")
    tokens, heads = int(key.size(2)), int(w_uk_t.size(0))
    prefixes = page_cache.setdefault("kimi_projected_leaf_prefixes", {})
    cache_key = (head_begin, heads)
    entry = prefixes.get(cache_key)
    if (entry is not None and
            (entry["uk"].data_ptr() != w_uk_t.data_ptr()
             or entry["uv"].data_ptr() != w_uv.data_ptr()
             or entry["tokens"] > tokens)):
        entry = None
    prefix = 0 if entry is None else int(entry["tokens"])
    if entry is None or entry["k"].size(2) < tokens:
        capacity = 1 << (max(tokens, 1) - 1).bit_length()
        expanded_k = key.new_empty(1, heads, capacity, 192)
        expanded_v = key.new_empty(1, heads, capacity, 128)
        if prefix:
            expanded_k[..., :prefix, :].copy_(entry["k"][..., :prefix, :])
            expanded_v[..., :prefix, :].copy_(entry["v"][..., :prefix, :])
        entry = {"k": expanded_k, "v": expanded_v, "tokens": prefix,
                 "uk": w_uk_t, "uv": w_uv}
        prefixes[cache_key] = entry
    if tokens > prefix:
        tail_k, tail_v = expand_kimi_leaf_kv(
            key[..., prefix:tokens, :], w_uk_t, w_uv, buffers=buffers,
        )
        entry["k"][..., prefix:tokens, :].copy_(tail_k)
        entry["v"][..., prefix:tokens, :].copy_(tail_v)
        entry["tokens"] = tokens
    return entry["k"][..., :tokens, :], entry["v"][..., :tokens, :]

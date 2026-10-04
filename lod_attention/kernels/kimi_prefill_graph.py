"""Opt-in fixed-shape K3 coarse/route graphs shared across model layers.

The graph captures GPU attention work, not host cache plans or update cadence.
Every query, centroid, count, directory length, and projection weight is
copied into stable inputs before replay. A bounded first-four-shapes cache
never evicts/re-captures in a warm run; other shapes use ordinary attention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .aiter_mla_prefill_attention import aiter_kimi_expanded_prefill_route_coarse_attention


@dataclass
class _Entry:
    inputs: tuple[torch.Tensor, ...]
    graph: torch.cuda.CUDAGraph
    result: tuple


class KimiPrefillCoarseGraphs:
    """One runtime-wide graph cache; no per-layer copies of graph workspaces."""

    def __init__(self, max_shapes: int = 4):
        if max_shapes < 1:
            raise ValueError("coarse graph cache needs a positive shape limit")
        self.max_shapes = max_shapes
        self.entries: dict[tuple, _Entry] = {}
        self.replay_count = 0
        self.fallback_count = 0

    def run(self, q, state_k, state_v, counts, uk, uv, *, state_len, scale,
            normalize_route_query, slot_lengths=None, max_open_leaf_tokens=None,
            buffers=None):
        sources = (q, state_k[..., :state_len, :], state_v[..., :state_len, :],
                   counts[..., :state_len, :], uk, uv,
                   slot_lengths[..., :state_len] if slot_lengths is not None else None)
        # The graph experiment deliberately targets just the aligned K3
        # production prefill shape, never decode, a ragged tail or normalized
        # routing. Fallback keeps the identical math for every other shape.
        supported = (q.size(2) == 16_384 and state_k.size(-1) == 576
                     and state_v.size(-1) == 512 and not normalize_route_query
                     and slot_lengths is not None and max_open_leaf_tokens is not None
                     and all(t.is_cuda for t in sources))
        key = (tuple((tuple(t.shape), t.dtype, t.device) for t in sources)
               if supported else None)
        key = (key, float(scale), int(max_open_leaf_tokens)) if supported else None
        if not supported or (key not in self.entries and len(self.entries) >= self.max_shapes):
            self.fallback_count += 1
            return aiter_kimi_expanded_prefill_route_coarse_attention(
                q, state_k, state_v, counts, uk, uv, state_len=state_len, scale=scale,
                normalize_route_query=normalize_route_query, slot_lengths=slot_lengths,
                max_open_leaf_tokens=max_open_leaf_tokens, buffers=buffers,
            )
        entry = self.entries.get(key)
        if entry is None:
            entry = self._capture(sources, state_len=state_len, scale=scale,
                                  max_open_leaf_tokens=max_open_leaf_tokens)
            self.entries[key] = entry
        for destination, source in zip(entry.inputs, sources, strict=True):
            destination.copy_(source)
        entry.graph.replay()
        self.replay_count += 1
        return entry.result

    def _capture(self, sources, *, state_len, scale, max_open_leaf_tokens):
        inputs = tuple(t.contiguous().clone() for t in sources)
        q, key, value, counts, uk, uv, lengths = inputs
        workspace = {}

        def run():
            routed, coarse, head_counts, offsets = aiter_kimi_expanded_prefill_route_coarse_attention(
                q, key, value, counts, uk, uv, state_len=state_len, scale=scale,
                normalize_route_query=False, slot_lengths=lengths,
                max_open_leaf_tokens=max_open_leaf_tokens, buffers=workspace,
                # These weights are *mutable graph inputs*, unlike immutable
                # layer parameters. Capture their layout transforms on every
                # replay; a pointer-based weight cache would freeze layer zero.
                cache_immutable_weights=False,
            )
            torch.cuda.current_stream(q.device).wait_stream(coarse.ready_stream)
            return routed, replace(coarse, ready_stream=None), head_counts, offsets

        foreground = torch.cuda.current_stream(q.device)
        capture_stream = torch.cuda.Stream(device=q.device)
        capture_stream.wait_stream(foreground)
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            run()
            run()
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream), torch.inference_mode():
            result = run()
        foreground.wait_stream(capture_stream)
        return _Entry(inputs=inputs, graph=graph, result=result)


@dataclass
class _BuildEntry:
    key: torch.Tensor
    graph: torch.cuda.CUDAGraph
    cache: object
    workspaces: dict


class KimiFinalCacheGraphs:
    """Graph the complete BF16 final-DCP cache build, not its host policy.

    One bounded cache is shared by the serial construction stream. Its result
    is scratch: the caller must install/copy each result before replaying for
    another layer group. No source archive or caller cache becomes graph-owned.
    """

    def __init__(self, max_shapes: int = 4):
        if max_shapes < 1:
            raise ValueError("cache construction graph limit must be positive")
        self.max_shapes = max_shapes
        self.entries = {}
        self.replay_count = 0
        self.fallback_count = 0

    def run(self, engine, key, value, *, final_cache_coverage):
        signature = (
            tuple(key.shape), key.dtype, key.device, int(final_cache_coverage),
            float(engine.state_growth_factor), int(engine.state_min_len),
            engine._streaming_state_geometry(), int(engine.sink_len), int(engine.state_size_offset),
            tuple(int(getattr(engine, name)) for name in (
                "chunk_len", "local_len", "prefill_chunk_len", "prefill_local_len",
                "prefill_state_update_len", "decode_state_update_len")),
        )
        supported = (
            key.is_cuda and key.ndim == 4 and key.size(1) == 1 and key.size(-1) == 576
            and value.data_ptr() == key.data_ptr() and value.size(-1) == 512
            and value.shape[:-1] == key.shape[:-1] and value.dtype == key.dtype
            and value.stride()[:-1] == key.stride()[:-1]
            and not engine.recursive_page_lod and not engine.leaf_key_quant_bits
            and not engine.leaf_value_quant_bits
            and engine._streaming_state_geometry() in {"raw", "spherical"}
            and engine.state_clustering_centroid_rescale == "none"
            and engine.state_premerge_factor == 1 and not engine.state_merge_before_append
            and not engine.overflow_bipartite_merge and not engine.state_union_bipartite
            and not engine.state_precompact_direct_append and not engine.state_append_subblock_size
            and engine.state_split_max_leaves is None and engine.separate_sink_cache
            and engine.leaf_paged_directory and engine.leaf_seal_capacity is None
        )
        if not supported or (signature not in self.entries and len(self.entries) >= self.max_shapes):
            self.fallback_count += 1
            return engine.build_cache_from_bf16(
                key, value, finalize_cache_for_decode=True,
                final_cache_coverage=final_cache_coverage,
            )
        entry = self.entries.get(signature)
        if entry is None:
            entry = self._capture(engine, key, final_cache_coverage)
            self.entries[signature] = entry
        entry.key.copy_(key)
        entry.graph.replay()
        self.replay_count += 1
        return entry.cache

    @staticmethod
    def _capture(engine, key, coverage):
        static = key.contiguous().clone()
        names = ("_lod_state", "_lod_state_update_buffers", "_lod_state_maxsim_buffers")
        saved = {name: getattr(engine, name) for name in names if hasattr(engine, name)}
        for name in saved:
            delattr(engine, name)

        def build():
            return engine.build_cache_from_bf16(
                static, static[..., :512], finalize_cache_for_decode=True,
                final_cache_coverage=coverage,
            )

        foreground = torch.cuda.current_stream(key.device)
        stream = torch.cuda.Stream(device=key.device)
        stream.wait_stream(foreground)
        try:
            with torch.cuda.stream(stream), torch.inference_mode():
                build()
                build()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
                cache = build()
            foreground.wait_stream(stream)
            workspaces = {name: getattr(engine, name) for name in names[1:]
                          if hasattr(engine, name)}
        finally:
            for name in names:
                if hasattr(engine, name):
                    delattr(engine, name)
                if name in saved:
                    setattr(engine, name, saved[name])
        return _BuildEntry(static, graph, cache, workspaces)

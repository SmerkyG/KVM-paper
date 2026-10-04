"""Benchmark-only fixed-geometry state-update graph experiment.

Only the GPU append/merge calculation is captured. The caller remains
responsible for the global sequence cadence, page metadata and cache lifetime.
Inputs are copied into stable storage; changing data must never be frozen by
capture. This deliberately pays those copies before claiming any speed gain.
"""

from __future__ import annotations

import torch


def _prefix_alias(key: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        key.data_ptr() == value.data_ptr()
        and key.dtype == value.dtype
        and key.shape[:-1] == value.shape[:-1]
        and key.stride()[:-1] == value.stride()[:-1]
        and value.size(-1) <= key.size(-1)
    )


class FixedStateUpdateGraph:
    """Capture one update shape without changing its append/merge math."""

    def __init__(self, update, inputs, options):
        state_k, state_v, counts, norms, overflow_k, overflow_v = inputs
        if norms is not None or not all(t.is_cuda for t in (
            state_k, state_v, counts, overflow_k, overflow_v,
        )):
            raise ValueError("this experiment requires CUDA state without norm sums")
        self.state_alias = _prefix_alias(state_k, state_v)
        self.overflow_alias = _prefix_alias(overflow_k, overflow_v)
        self.options = dict(options)
        static_k, static_overflow = state_k.clone(), overflow_k.clone()
        self.inputs = (
            static_k,
            static_k[..., :state_v.size(-1)] if self.state_alias else state_v.clone(),
            counts.clone(), None, static_overflow,
            static_overflow[..., :overflow_v.size(-1)]
            if self.overflow_alias else overflow_v.clone(),
        )
        foreground = torch.cuda.current_stream(state_k.device)
        stream = torch.cuda.Stream(device=state_k.device)
        stream.wait_stream(foreground)
        with torch.cuda.stream(stream), torch.inference_mode():
            # Warm the actual update kernels before capture, restoring state
            # after each warmup. No synthetic graph rows enter the live cache.
            for _ in range(2):
                self._copy_inputs(inputs)
                update(*self.inputs, **self.options)
            self._copy_inputs(inputs)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
            self.outputs = update(*self.inputs, **self.options)
        foreground.wait_stream(stream)
        if self.outputs[5] is not None:
            raise ValueError("state remapping is not supported by this graph experiment")

    def _copy_inputs(self, inputs):
        for index, (source, target) in enumerate(zip(inputs, self.inputs)):
            if source is None:
                if target is not None:
                    raise ValueError("state norm geometry changed")
                continue
            if (source.shape != target.shape or source.dtype != target.dtype
                    or source.device != target.device):
                raise ValueError("state-update graph input geometry changed")
            if (index == 1 and self.state_alias) or (
                index == 5 and self.overflow_alias
            ):
                if not _prefix_alias(inputs[index - 1], source):
                    raise ValueError("state-update graph input aliasing changed")
                continue
            target.copy_(source)

    @torch.inference_mode()
    def __call__(self, *inputs):
        self._copy_inputs(inputs)
        self.graph.replay()
        state_k, state_v, counts, *_ = inputs
        state_k.copy_(self.outputs[0])
        if not self.state_alias:
            state_v.copy_(self.outputs[1])
        counts.copy_(self.outputs[2])
        # Callers may retain ownership while another layer group replays this
        # graph. Return independent ownership storage, not a shared scratch.
        return state_k, state_v, counts, self.outputs[3], self.outputs[4].clone(), None


class DirectStateUpdateGraph:
    """Benchmark a replay on already-stable caller-owned update workspaces.

    The caller must refill the same storage before replay. Unlike the existing
    graph this adds no state/overflow input copies or state-output copies.
    Capture warmups restore live state before returning, and every call checks
    all pointers/strides so a new layer's storage cannot be silently ignored.
    This experiment is not attached to the serving runtime.
    """

    @staticmethod
    def _signature(inputs):
        return tuple(None if tensor is None else (
            tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device) for tensor in inputs)

    def __init__(self, update, inputs, options):
        state_k, state_v, counts, norms, overflow_k, overflow_v = inputs
        if norms is not None or not _prefix_alias(state_k, state_v):
            raise ValueError("direct experiment requires prefix-aliased state without norms")
        if not all(t.is_cuda for t in (state_k, state_v, counts, overflow_k, overflow_v)):
            raise ValueError("direct experiment requires CUDA inputs")
        self.signature = self._signature(inputs)
        self.inputs = inputs  # Own references until the graph is destroyed.
        original_key, original_count = state_k.clone(), counts.clone()

        def restore():
            state_k.copy_(original_key)
            counts.copy_(original_count)

        foreground = torch.cuda.current_stream(state_k.device)
        stream = torch.cuda.Stream(device=state_k.device)
        stream.wait_stream(foreground)
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                restore()
                update(*inputs, **options)
            restore()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
            self.outputs = update(*inputs, **options)
        foreground.wait_stream(stream)
        restore()
        if self.outputs[5] is not None:
            raise ValueError("direct experiment does not support state remapping")

    @torch.inference_mode()
    def __call__(self, *inputs):
        if self._signature(inputs) != self.signature:
            raise ValueError("direct state-update replay requires the same input storage")
        self.graph.replay()
        state_k, state_v, counts, *_ = inputs
        return state_k, state_v, counts, self.outputs[3], self.outputs[4].clone(), None

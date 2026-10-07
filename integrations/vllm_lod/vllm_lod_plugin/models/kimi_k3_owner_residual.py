"""Experimental token-sharded AttnRes bank for full-block request owners.

The native K3 model forward and every layer are retained. Only allocation of
its residual bank is intercepted: each TP rank stores its contiguous token
slice. Native AttnRes runs on that slice and gathers the resulting normalized
hidden states before the unchanged TP attention/MoE. Prefix sums are mutated
only on their owning rank and consumed only there by subsequent AttnRes calls.
This saves bank memory, at the cost of an additional all-gather per mix.

This eager, prefill-only experiment deliberately excludes auxiliary hidden
state outputs and pipeline parallelism, which require replicated prefix sums.
"""

from __future__ import annotations

from contextvars import ContextVar
from importlib import import_module
import os

import torch
from torch.overrides import TorchFunctionMode

from .kimi_k3_sharded_prefill import gather_prefill


_owner_group = ContextVar("kimi_owner_residual_group", default=None)


class ShardBankAllocation(TorchFunctionMode):
    """Intercept exactly one known native bank allocation, not model tensors."""

    def __init__(self, shape: tuple[int, int, int], world: int):
        self.shape = shape
        self.world = world
        self.used = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.new_empty and not self.used:
            size = (tuple(args[1]) if len(args) == 2 and isinstance(args[1], (tuple, list))
                    else tuple(args[1:]))
            if size == self.shape:
                self.used = True
                return func(args[0], (size[0] // self.world, *size[1:]), **kwargs)
        return func(*args, **kwargs)


def shard_attn_res(native, prefix, bank, proj, norm, valid_blocks, **kwargs):
    """Run unchanged per-token arithmetic; gather only its final output."""
    group = _owner_group.get()
    if group is None or bank.size(0) == prefix.size(0):
        return native(prefix, bank, proj, norm, valid_blocks, **kwargs)
    if prefix.size(0) != bank.size(0) * group.world_size:
        raise RuntimeError("sharded K3 residual bank has incompatible token geometry")
    begin = group.rank_in_group * bank.size(0)
    end = begin + bank.size(0)
    delta = kwargs.pop("delta", None)
    local = native(prefix[begin:end], bank, proj, norm, valid_blocks,
                   delta=None if delta is None else delta[begin:end], **kwargs)
    return gather_prefill(group, local, dim=0)


def install_owner_residual_sharding() -> None:
    if os.getenv("LOD_KIMI_OWNER_SHARD_RESIDUAL") != "1":
        return
    if os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") != "1":
        raise ValueError("sharded AttnRes requires the request-owner prefill experiment")
    module = import_module("vllm.models.kimi_k3.amd.linear")
    from vllm.distributed import get_pp_group, get_tp_group

    model = module.KimiLinearModel
    if getattr(model, "_lod_owner_sharded_residual_installed", False):
        return
    native_forward = model.forward
    native_mix = module._apply_attn_res

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs):
        group = get_tp_group()
        tokens = int(positions.numel())
        blocks = self.config.attn_res_block_size
        # Tiny/dummy or unaligned calls retain the complete native path.
        if blocks is None or tokens <= group.world_size or tokens % group.world_size:
            return native_forward(self, input_ids, positions, intermediate_tensors,
                                  inputs_embeds=inputs_embeds, **kwargs)
        pp = get_pp_group()
        if not (pp.is_first_rank and pp.is_last_rank) or intermediate_tensors is not None:
            raise NotImplementedError("owner residual sharding supports PP1 only")
        if self.aux_hidden_state_layers:
            raise NotImplementedError("owner residual sharding does not support auxiliary outputs")
        shape = (tokens, (self.end_layer + blocks - 1) // blocks, self.config.hidden_size)
        allocation = ShardBankAllocation(shape, group.world_size)
        token = _owner_group.set(group)
        try:
            with allocation:
                result = native_forward(self, input_ids, positions, intermediate_tensors,
                                        inputs_embeds=inputs_embeds, **kwargs)
            if not allocation.used:
                raise RuntimeError("native K3 forward no longer allocates the expected residual bank")
            return result
        finally:
            _owner_group.reset(token)

    def mix(prefix, bank, proj, norm, valid_blocks, **kwargs):
        return shard_attn_res(native_mix, prefix, bank, proj, norm, valid_blocks, **kwargs)

    model.forward = forward
    module._apply_attn_res = mix
    model._lod_owner_sharded_residual_installed = True

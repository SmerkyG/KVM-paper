"""Bound full K3 MoE temporaries independently of request-owner attention.

This experiment retains the native MoE forward, expert routing, collectives,
and weights. Only its tokenwise input is sliced. All TP ranks execute the same
slice order, while attention can consume a complete 16K block per request.
"""

from __future__ import annotations

import os

import torch


def install_owner_profile_cleanup() -> None:
    """Release dead dummy-forward workspaces before RCCL's sampler allocation.

    Only explicit-KV-budget startup profiling needs this: that path still runs
    the full model to compile it, but does not use its allocator high-water
    mark to size KV. RCCL cannot reclaim PyTorch's cached MoE temporaries.
    Normal sampling, graph replay and attention scheduling are unchanged.
    """
    if os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") != "1":
        return
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:
        return  # Older vLLM runners do not have this startup profile path.
    if getattr(GPUModelRunner, "_lod_owner_profile_cleanup_installed", False):
        return
    profile = GPUModelRunner.profile_run
    sampler = GPUModelRunner._dummy_sampler_run

    def profile_run(self, *args, **kwargs):
        previous = getattr(self, "_lod_owner_in_explicit_profile", False)
        budget = getattr(self.cache_config, "kv_cache_memory_bytes", None)
        self._lod_owner_in_explicit_profile = bool(budget)
        try:
            return profile(self, *args, **kwargs)
        finally:
            self._lod_owner_in_explicit_profile = previous

    def dummy_sampler_run(self, *args, **kwargs):
        if getattr(self, "_lod_owner_in_explicit_profile", False):
            torch.accelerator.empty_cache()
        return sampler(self, *args, **kwargs)

    GPUModelRunner.profile_run = profile_run
    GPUModelRunner._dummy_sampler_run = dummy_sampler_run
    GPUModelRunner._lod_owner_profile_cleanup_installed = True


def install_owner_moe_chunking() -> None:
    if os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") != "1":
        return
    tokens = int(os.getenv("LOD_KIMI_OWNER_MOE_CHUNK", "16384"))
    if tokens <= 0:
        raise ValueError("owner MoE chunk must be positive")
    from vllm.models.kimi_k3.amd.linear import KimiMoE

    if getattr(KimiMoE, "_lod_owner_chunking_installed", False):
        return
    original = KimiMoE.forward

    def forward(self, hidden_states):
        if hidden_states.size(0) <= tokens:
            return original(self, hidden_states)
        output = torch.empty_like(hidden_states)
        for begin in range(0, hidden_states.size(0), tokens):
            end = min(begin + tokens, hidden_states.size(0))
            output[begin:end].copy_(original(self, hidden_states[begin:end]))
        return output

    KimiMoE.forward = forward
    KimiMoE._lod_owner_chunking_installed = True

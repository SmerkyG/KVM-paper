"""Bound full K3 MoE temporaries independently of request-owner attention.

This experiment retains the native MoE forward, expert routing, collectives,
and weights. Only its tokenwise input is sliced. All TP ranks execute the same
slice order, while attention can consume a complete 16K block per request.
"""

from __future__ import annotations

import os

import torch


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

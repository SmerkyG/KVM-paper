"""Lazy model registrations missing from supported vLLM releases."""

from __future__ import annotations


def register_k2_horizon() -> None:
    """Backport upstream K2 Horizon without overriding future native support."""
    from vllm import ModelRegistry

    if "K2HorizonForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "K2HorizonForCausalLM",
            "vllm_lod_plugin.k2_horizon:K2HorizonForCausalLM",
        )


__all__ = ["register_k2_horizon"]

"""vLLM plugin registration for the LoD Attention paper release."""

from __future__ import annotations


_REGISTERED = False


def register() -> None:
    """Register production LoD, K2 Horizon, and Qwen3.8 DFlash2 support."""

    global _REGISTERED
    if _REGISTERED:
        return

    # Isolated experiment: reuse the native vLLM scheduler and cache without
    # installing LoD's ownership hooks or changing the production backend.
    import os
    if os.environ.get("VLLM_COMPARISON_BACKEND") == "quest":
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

        register_backend(
            AttentionBackendEnum.CUSTOM,
            "vllm_lod_plugin.quest_backend.QuestAttentionBackend",
        )
        _REGISTERED = True
        return

    # Parse first so removed research flags fail before vLLM is modified.
    from .config import VLLMLODSettings

    VLLMLODSettings.from_environment()

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "vllm_lod_plugin.backend.LODAttentionBackend",
    )

    from .models.dflash2 import register_dflash2_compat
    from .model_compat import register_k2_horizon

    register_dflash2_compat()
    register_k2_horizon()

    from .cache_ownership import install_cache_ownership_hooks
    from .runtime import install_model_state_hooks

    install_cache_ownership_hooks()
    install_model_state_hooks()
    _REGISTERED = True


__all__ = ["register"]

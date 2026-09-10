"""Minimal Hugging Face entry points for production LoD Attention."""

from __future__ import annotations

from typing import Any

from torch import nn

from ._config import LODMode, model_family
from ._hf_backend import (
    HFLODCache,
    convert_hf_full_cache_to_lod,
    install_hf_lod_attention,
    new_hf_lod_cache,
)


def install(model: nn.Module, mode: str | LODMode = LODMode.TWO_TIER) -> list[str]:
    """Replace supported global-attention layers with production top-four LoD.

    The model must be Qwen3.8 or K2 Horizon. The returned names are the layers
    changed in place. ``model.generate`` creates the appropriate owned cache;
    direct cached calls can use :func:`new_cache`.
    """

    resolved = LODMode.parse(mode)
    model_family(model)  # fail before mutating an unsupported model
    return install_hf_lod_attention(model, mode=resolved)


def new_cache(model: nn.Module) -> Any:
    """Create an empty model-bound LoD cache for direct Hugging Face calls."""

    return new_hf_lod_cache(model)


def convert_cache(model: nn.Module, full_cache: Any) -> HFLODCache:
    """Build an LoD cache from an existing post-RoPE BF16 Hugging Face cache."""

    return convert_hf_full_cache_to_lod(model, full_cache)


__all__ = ["convert_cache", "install", "new_cache"]

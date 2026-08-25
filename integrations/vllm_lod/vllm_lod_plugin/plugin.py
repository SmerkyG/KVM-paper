"""General-plugin entry point loaded in every vLLM process."""

from __future__ import annotations

import sys
from pathlib import Path

_REGISTERED = False


def _add_lod_source_tree() -> None:
    # Respect an explicitly selected source checkout (for example through
    # PYTHONPATH) instead of silently replacing it with this plugin's editable
    # install root. This matters when benchmarking changes from a worktree.
    for entry in sys.path:
        candidate = Path(entry or ".").resolve()
        if (candidate / "model" / "triton_lod_engines.py").is_file():
            return
    # Editable installs retain this package beneath integrations/vllm_lod.
    root = Path(__file__).resolve().parents[3]
    if not (root / "model" / "triton_lod_engines.py").is_file():
        raise RuntimeError(
            "Cannot locate the LOD Attention source tree. Install this plugin "
            "editable from integrations/vllm_lod or add the repository root "
            "to PYTHONPATH."
        )
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)


def register() -> None:
    """Register the CUSTOM backend and lifecycle hooks, idempotently."""
    global _REGISTERED
    if _REGISTERED:
        return
    _add_lod_source_tree()
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "vllm_lod_plugin.backend.LODAttentionBackend",
    )
    from .cache_ownership import install_cache_ownership_hooks
    from .runtime import install_model_state_hooks

    install_cache_ownership_hooks()
    install_model_state_hooks()
    from .weight_cache_loader import register_weight_cache_loader

    register_weight_cache_loader()
    _REGISTERED = True


__all__ = ["register"]

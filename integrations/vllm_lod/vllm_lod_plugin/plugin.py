"""General-plugin entry point loaded in every vLLM process."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REGISTERED = False


def _install_native_attention_skip_diagnostic() -> None:
    """Hold native attention arithmetic at zero for dispatch-overhead tests."""
    # ``eligible`` zeros only externally-owned LOD layers, leaving native
    # sliding-window layers intact. ``skip`` also patches native attention so
    # it remains the whole-model no-attention control.
    if os.getenv("VLLM_LOD_DIAGNOSTIC_EXTERNAL_EMPTY_ATTENTION") != "skip":
        return
    if not __import__("torch").version.hip:
        raise NotImplementedError("the external-empty diagnostic currently targets ROCm")
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionImpl,
    )

    if getattr(RocmAiterUnifiedAttentionImpl, "_vllm_lod_skip_installed", False):
        return

    def skip_forward(self, *args, **kwargs):
        output = kwargs.get("output")
        if output is None:
            if len(args) <= 6:
                raise TypeError("native attention diagnostic could not locate output")
            output = args[6]
        return output.zero_()

    RocmAiterUnifiedAttentionImpl.forward = skip_forward
    RocmAiterUnifiedAttentionImpl._vllm_lod_skip_installed = True


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
    from .dflash2_compat import register_dflash2_compat

    register_dflash2_compat()
    _install_native_attention_skip_diagnostic()
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "vllm_lod_plugin.backend.LODAttentionBackend",
    )
    # The benchmark vLLM release only supports Muse-Glimmer through the
    # generic Transformers runner, whose attention is not replaceable by an
    # out-of-tree vLLM backend. Register its text tower lazily so both native
    # full attention and LOD use the same serving stack.
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "MuseGlimmerForCausalLM",
        "vllm_lod_plugin.muse_glimmer:MuseGlimmerForCausalLM",
    )
    # K2 Horizon support landed after vLLM 0.28.0. Backport the upstream model
    # on older releases, while automatically deferring to native support after
    # an eventual vLLM upgrade. The weight-cache-only plugin calls this too so
    # native-attention K2 runs do not install the LOD lifecycle hooks.
    from .model_compat import register_k2_horizon

    register_k2_horizon()
    from .cache_ownership import install_cache_ownership_hooks
    from .runtime import install_model_state_hooks, install_tp_safe_vocab_padding

    install_tp_safe_vocab_padding()
    install_cache_ownership_hooks()
    install_model_state_hooks()
    from .weight_cache_loader import register_weight_cache_loader

    register_weight_cache_loader()
    _REGISTERED = True


__all__ = ["register"]

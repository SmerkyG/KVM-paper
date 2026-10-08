"""vLLM plugin registration for the LoD Attention paper release."""

from __future__ import annotations

import logging


_REGISTERED = False
logger = logging.getLogger(__name__)


def register() -> None:
    """Register production LoD, K2 Horizon, and Qwen3.8 DFlash2 support."""

    global _REGISTERED
    if _REGISTERED:
        return

    # Parse first so removed research flags fail before vLLM is modified.
    from .config import VLLMLODSettings, lod_enabled

    VLLMLODSettings.from_environment()
    # This optional AMD image fix retains the native GLM sparse-indexer math
    # in dense controls as well as LoD; older vLLM installations lack FlyDSL.
    from .models.glm53_flash import install_flydsl_shift_compat, install_native_indexer_compat

    install_flydsl_shift_compat()
    install_native_indexer_compat()
    import os
    if os.environ.get("LOD_GLM_KDA_PREFILL") == "1":
        from benchmarks.glm53_kda_prefill import install_glm_kda_prefill

        install_glm_kda_prefill()

    # Dense controls may still use the shared weight daemon, the K2 model
    # registration, and K3's optimized dense Gluon decoder.  Keep that path
    # entirely separate from LoD's attention-backend and model monkey patches:
    # merely making the LoD runtime return ``None`` is not sufficient because
    # its K3 MLA registration changes vLLM's static attention metadata.
    if not lod_enabled():
        from .model_compat import register_k2_horizon
        from .models.kimi_k3 import register_kimi_k3_dense
        from .weight_cache_loader import register_weight_cache_loader

        register_k2_horizon()
        register_kimi_k3_dense()
        register_weight_cache_loader()
        _REGISTERED = True
        return

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "vllm_lod_plugin.backend.LODAttentionBackend",
    )

    from .models.kimi_k3 import register_kimi_k3_lod
    from .model_compat import register_k2_horizon

    # DFlash2 is an optional Qwen-only adapter and vLLM has changed its private
    # sampler helpers across releases.  Do not let that optional compatibility
    # layer prevent the model-independent backend (or K2/K3 support) from
    # registering on a newer vLLM build.
    try:
        from .models.dflash2 import register_dflash2_compat
    except ImportError as exc:
        logger.warning("Qwen DFlash2 compatibility is unavailable: %s", exc)
    else:
        register_dflash2_compat()
    register_k2_horizon()
    register_kimi_k3_lod()
    from .models.glm53_flash import register_glm53_flash_lod

    register_glm53_flash_lod()

    from .cache_ownership import install_cache_ownership_hooks
    from .runtime import install_model_state_hooks

    install_cache_ownership_hooks()
    install_model_state_hooks()

    # Register the persistent post-load weight cache for both dense and LoD
    # engines.  The cache owns the final TP/EP-sharded tensors, so loader-side
    # conversions (including Kimi-K3 MXFP4 -> groupwise INT4) run only on a
    # cold miss and are shared with later workers through HIP IPC.
    from .weight_cache_loader import register_weight_cache_loader

    register_weight_cache_loader()
    _REGISTERED = True


__all__ = ["register"]

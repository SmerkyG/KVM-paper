"""Shared, production-only vLLM setup for the public benchmark runners."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MODES = ("full", "two-tier", "three-tier-bf16", "three-tier-int4")
SCHEDULER_CHUNK = 16_384


def is_qwen38(checkpoint: str) -> bool:
    """Return whether a model identifier names the supported Qwen3.8 family."""

    normalized = checkpoint.lower().replace("_", "").replace("-", "")
    return "qwen3.8" in checkpoint.lower() or "qwen38" in normalized


def configure_environment(mode: str, pool_size: int) -> None:
    """Select the release plugin and one of its three immutable LoD modes."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    # K2's vLLM model registration also lives in this plugin, so load it for
    # the full-attention control. CUSTOM attention is selected only below.
    os.environ["VLLM_PLUGINS"] = "lod_attention"
    os.environ["VLLM_LOD_MODE"] = "two-tier" if mode == "full" else mode
    os.environ["VLLM_LOD_POOL_SIZE"] = str(pool_size)
    os.environ.pop("VLLM_LOD_MAX_CONTEXT", None)


def llm_kwargs(
    *,
    checkpoint: str,
    mode: str,
    max_model_len: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    full_attention_backend: str,
) -> dict[str, Any]:
    """Build the matched vLLM configuration used by every offline runner."""

    if max_model_len < 2:
        raise ValueError("max_model_len must be at least two")
    if batch_size < 1 or tensor_parallel_size < 1:
        raise ValueError("batch_size and tensor_parallel_size must be positive")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    configure_environment(mode, batch_size)
    backend = "CUSTOM" if mode != "full" else full_attention_backend
    kwargs: dict[str, Any] = {
        "model": checkpoint,
        "model_impl": "vllm",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "max_model_len": max_model_len,
        "max_num_seqs": batch_size,
        "max_num_batched_tokens": SCHEDULER_CHUNK,
        "long_prefill_token_threshold": min(SCHEDULER_CHUNK, max_model_len),
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "disable_custom_all_reduce": True,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "attention_config": {"backend": backend},
    }
    if is_qwen38(checkpoint):
        kwargs["language_model_only"] = True
    return kwargs


def close_llm(llm: Any) -> None:
    """Deterministically stop the EngineCore child created by offline vLLM."""

    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if callable(shutdown):
        shutdown()


def write_json(path: Path, value: Any) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

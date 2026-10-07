"""Shared, production-only vLLM setup for the public benchmark runners."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MODES = ("full", "two-tier", "three-tier-bf16", "three-tier-int4")
# The release measurements use 16K chunks. Extremely long dense controls can
# lower this benchmark-only value when the native KV cache and one 16K model
# activation no longer fit simultaneously (for example K3 B=8 at 1.02M).
SCHEDULER_CHUNK = int(os.environ.get("LOD_BENCHMARK_SCHEDULER_CHUNK", "16384"))
if SCHEDULER_CHUNK <= 0:
    raise ValueError("LOD_BENCHMARK_SCHEDULER_CHUNK must be positive")
LOD_SCHEDULER = "vllm_lod_plugin.scheduler.LODChunkAlignedScheduler"
_automatic_kimi_owner_environment: dict[str, str] = {}


def configure_kimi_layout(
    *, checkpoint: str, mode: str, batch_size: int,
    tensor_parallel_size: int, decode_context_parallel_size: int,
    max_model_len: int,
) -> bool:
    """Default the tested K3 B8/TP8/DCP8 cohort to one request per GPU.

    This is a fixed eight-request execution layout, not variable-batch
    serving. Other geometries retain ordinary DCP. Explicit owner flags win;
    ``LOD_KIMI_REQUEST_OWNER_PREFILL=0`` selects the ordinary-DCP control.
    Clear only defaults this helper itself installed when constructing a
    different engine in the same process (in particular a dense control).
    """
    eligible = (mode == "two-tier" and batch_size == 8
                and tensor_parallel_size == decode_context_parallel_size == 8
                and is_kimi_k3(checkpoint))
    if not eligible or os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") == "0":
        for name, value in _automatic_kimi_owner_environment.items():
            if os.environ.get(name) == value:
                os.environ.pop(name)
        _automatic_kimi_owner_environment.clear()
        return False
    defaults = {
        "LOD_KIMI_REQUEST_OWNER_PREFILL": "1",
        "LOD_KIMI_REQUEST_OWNER_DECODE": "1",
        "LOD_KIMI_OWNER_QUERY_CHUNK": "2048",
        "LOD_BENCHMARK_PREFILL_COHORT": "8",
        "LOD_KIMI_OWNER_MOE_CHUNK": "16392",
        "LOD_KIMI_OWNER_REUSE_TRANSPORT": "1",
        "LOD_KIMI_OWNER_SHARD_RESIDUAL": "0",
        "LOD_KIMI_OWNER_POOL_BACKED_PREFILL": "1",
        # Bound scratch by live archived leaves inside the attention engine,
        # not a request's future capacity. Shared construction scratch lets
        # short live prefixes retain the tested twelve-head geometry.
        "LOD_KIMI_OWNER_PREFILL_HEAD_GROUP": "12",
        "LOD_KIMI_OWNER_PRESSURE_CHECK": "1",
    }
    # Explicit experimental prefill-only owner layouts remain opt-in. Do not
    # add captured decode behind the local-Q/K/V/O or six-head-owner probes.
    if (os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") == "1"
            and not _automatic_kimi_owner_environment):
        return False
    for name, value in defaults.items():
        previous = _automatic_kimi_owner_environment.get(name)
        if name not in os.environ or (previous is not None and os.environ[name] == previous):
            os.environ[name] = value
            _automatic_kimi_owner_environment[name] = value
        else:
            # A caller changed this setting after the first engine: it is now
            # an explicit override, not a default we own or may clean up.
            _automatic_kimi_owner_environment.pop(name, None)
    return True


def scheduler_budget(batch_size: int, speculative_tokens: int = 0) -> int:
    """Leave room for decode rows without shrinking the 16K prefill chunk."""

    query_tokens = speculative_tokens + 1 if speculative_tokens else 1
    prefill_cohort = int(os.environ.get("LOD_BENCHMARK_PREFILL_COHORT", "1"))
    if prefill_cohort < 1:
        raise ValueError("LOD_BENCHMARK_PREFILL_COHORT must be positive")
    return (
        SCHEDULER_CHUNK * min(batch_size, prefill_cohort)
        + batch_size * query_tokens
    )


def is_qwen38(checkpoint: str) -> bool:
    """Return whether a model identifier names the supported Qwen3.8 family."""

    normalized = checkpoint.lower().replace("_", "").replace("-", "")
    return "qwen3.8" in checkpoint.lower() or "qwen38" in normalized


def is_kimi_k3(checkpoint: str) -> bool:
    normalized = checkpoint.lower().replace("_", "").replace("-", "")
    if "kimik3" in normalized:
        return True

    # Release benchmarks normally use a node-local copy of the checkpoint so
    # loading does not repeatedly traverse shared storage.  That temporary
    # directory has a content hash rather than the model name, so identify K3
    # from its config as well.  Without this check the nominal dense control
    # silently falls back from the release Gluon decoder to TRITON_MLA.
    config_path = Path(checkpoint) / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    model_type = str(config.get("model_type", "")).lower()
    architectures = {
        str(name).lower() for name in config.get("architectures", ())
    }
    return model_type == "kimi_linear" or any(
        "kimilinear" in name or "kimik3" in name for name in architectures
    )


def _kimi_linear_architecture_override(checkpoint: str) -> dict[str, list[str]] | None:
    """Identify the text-only Kimi packaging whose config omits architectures."""

    config_path = Path(checkpoint) / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        str(config.get("model_type", "")).lower() == "kimi_linear"
        and not config.get("architectures")
    ):
        return {"architectures": ["KimiLinearForCausalLM"]}
    return None


def configure_environment(mode: str, pool_size: int) -> None:
    """Select the release plugin and one of its three immutable LoD modes."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    # K2's vLLM model registration also lives in this plugin, so load it for
    # the full-attention control. CUSTOM attention is selected only below.
    os.environ["VLLM_PLUGINS"] = "lod_attention"
    os.environ["VLLM_LOD_ENABLED"] = "0" if mode == "full" else "1"
    os.environ["VLLM_LOD_MODE"] = "two-tier" if mode == "full" else mode
    os.environ["VLLM_LOD_POOL_SIZE"] = str(pool_size)
    os.environ.pop("VLLM_LOD_MAX_CONTEXT", None)


def default_gpu_memory_utilization(
    checkpoint: str,
    mode: str,
    *,
    quality: bool = False,
) -> float:
    """Return the measured safe memory target for a release benchmark.

    K2's larger model-side 131K LoD pool already consumes more than 70% of an
    MI325X before vLLM allocates its small native-cache remainder. Qwen fits at
    70% but needs the remaining headroom for concurrent cache maintenance.
    """

    if quality:
        return 0.65
    if mode == "full":
        return 0.9
    return 0.7 if is_qwen38(checkpoint) else 0.8


def llm_kwargs(
    *,
    checkpoint: str,
    mode: str,
    max_model_len: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    full_attention_backend: str,
    decode_context_parallel_size: int = 1,
    dcp_comm_backend: str = "ag_rs",
    speculative_model: str | None = None,
    num_speculative_tokens: int = 7,
    speculative_attention_backend: str = "TRITON_ATTN",
) -> dict[str, Any]:
    """Build the matched vLLM configuration used by every offline runner."""

    if max_model_len < 2:
        raise ValueError("max_model_len must be at least two")
    if batch_size < 1 or tensor_parallel_size < 1:
        raise ValueError("batch_size and tensor_parallel_size must be positive")
    if (
        decode_context_parallel_size < 1
        or tensor_parallel_size % decode_context_parallel_size
    ):
        raise ValueError(
            "decode_context_parallel_size must be positive and divide TP"
        )
    if dcp_comm_backend not in ("ag_rs", "a2a"):
        raise ValueError("dcp_comm_backend must be 'ag_rs' or 'a2a'")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if speculative_model and not is_qwen38(checkpoint):
        raise ValueError("DFlash2 is supported only with Qwen3.8")
    if speculative_model and num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be positive")
    configure_kimi_layout(
        checkpoint=checkpoint, mode=mode, max_model_len=max_model_len,
        batch_size=batch_size, tensor_parallel_size=tensor_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
    )
    configure_environment(mode, batch_size)
    # Dense K3 comparisons use the faster absorbed-MLA Gluon decoder rather
    # than AMD's precompiled/full-attention decode path.  The plugin leaves
    # every non-K3 control unchanged.
    if mode == "full" and is_kimi_k3(checkpoint):
        os.environ["VLLM_KIMI_DENSE_GLUON"] = "1"
    else:
        os.environ.pop("VLLM_KIMI_DENSE_GLUON", None)
    backend = "CUSTOM" if mode != "full" else full_attention_backend
    active_speculative_tokens = num_speculative_tokens if speculative_model else 0
    kwargs: dict[str, Any] = {
        "model": checkpoint,
        "model_impl": "vllm",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "max_model_len": max_model_len,
        "max_num_seqs": batch_size,
        "max_num_batched_tokens": scheduler_budget(
            batch_size, active_speculative_tokens
        ),
        "long_prefill_token_threshold": min(SCHEDULER_CHUNK, max_model_len),
        "scheduler_cls": LOD_SCHEDULER,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "decode_context_parallel_size": decode_context_parallel_size,
        "dcp_comm_backend": dcp_comm_backend,
        "disable_custom_all_reduce": True,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "attention_config": {"backend": backend},
    }
    if is_kimi_k3(checkpoint):
        # LoD intercepts the outer MLA call, while construction still needs a
        # native MLA implementation to materialize W_UK/W_UV.  Full attention
        # should likewise use vLLM's native MLA auto-selection rather than a
        # conventional MHA backend forced by the other paper models.
        kwargs["attention_config"] = (
            {
                "backend": "CUSTOM",
                "backend_per_kind": {"mla_attention": "TRITON_MLA"},
            }
            if mode != "full"
            else {
                "backend": None,
                # TritonMLAImpl is the stable vLLM host ABI patched above to
                # dispatch K3 dense decode to our faster Gluon kernel.
                "backend_per_kind": {"mla_attention": "TRITON_MLA"},
            }
        )
        architecture_override = _kimi_linear_architecture_override(checkpoint)
        if architecture_override is not None:
            kwargs["hf_overrides"] = architecture_override
        if os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") == "1":
            if not 1 <= batch_size <= 8 or tensor_parallel_size != 8 or decode_context_parallel_size != 8:
                raise ValueError("request-owner probe requires B1–B8 TP8/DCP8")
            owner_chunk = int(os.getenv("LOD_KIMI_OWNER_QUERY_CHUNK", "2048"))
            if owner_chunk not in (2048, 4096, 8192, 16384):
                raise ValueError("owner query chunk must be 2K, 4K, 8K, or 16K")
            # The default retains the previous 16K aggregate budget. Larger
            # owner blocks explicitly opt into larger scheduler cohorts.
            kwargs["long_prefill_token_threshold"] = owner_chunk
            if "LOD_KIMI_OWNER_QUERY_CHUNK" in os.environ:
                owner_cohort = int(os.getenv("LOD_BENCHMARK_PREFILL_COHORT", "1"))
                kwargs["max_num_batched_tokens"] = max(
                    SCHEDULER_CHUNK, owner_chunk * min(batch_size, owner_cohort),
                ) + batch_size
            kwargs["enforce_eager"] = True
            if os.getenv("LOD_KIMI_REQUEST_OWNER_DECODE") == "1":
                if batch_size != 8:
                    raise ValueError("captured request-owner decode still requires B8")
                kwargs["enforce_eager"] = False
                kwargs["compilation_config"] = {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [8],
                    "max_cudagraph_capture_size": 8,
                }
        if os.environ.get("LOD_KIMI_GRAPH_PREFILL") == "1":
            os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
            # Capture only the exact fixed prefill shape. Mixed steps exceeding
            # 16K remain eager; their scheduler budget and cadence are unchanged.
            # A second almost-identical 16K+decode-reserve descriptor retains
            # additional graph/communication memory beside resident K3 weights.
            sizes = sorted(set(range(1, batch_size + 1)) | {
                min(SCHEDULER_CHUNK, max_model_len),
            })
            kwargs["compilation_config"] = {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": sizes,
                "max_cudagraph_capture_size": max(sizes),
            }
    if is_qwen38(checkpoint) or is_kimi_k3(checkpoint):
        kwargs["language_model_only"] = True
    if speculative_model:
        kwargs["speculative_config"] = {
            "method": "dflash",
            "model": speculative_model,
            "num_speculative_tokens": num_speculative_tokens,
            "attention_backend": speculative_attention_backend,
        }
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        if tensor_parallel_size > 1:
            # ROCm graph capture cannot coexist with ProcessGroupNCCL's
            # background HIP-event watchdog on the validated vLLM revision.
            os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
            os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
            os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
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

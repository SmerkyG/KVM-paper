"""OpenAI-compatible Hugging Face serving with LOD attention installed."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cli.serving.model_manager import ModelManager

from .hf_pytorch_lod_attention import install_hf_lod_attention
from .hf_lod_session_cache import (
    LOD_SESSION_HEADER,
    LODSessionCacheConfig,
    LODSessionGenerationState,
    bind_lod_session,
    reset_lod_session,
)
from .pytorch_lod_attention_paged import PagedLODConfig


@dataclass(frozen=True)
class LODServerConfig:
    """LOD settings shared by every attention layer in the served model."""

    chunk_size: int = 256
    local_window: int = 512
    state_growth_factor: float = 16.0
    state_min_size: int = 256
    protected_prefix: int = 1
    open_count: int = 8
    page_size: int = 16
    kv_bits: int = 4
    engine_backend: str = "kernel"
    left_padding_mode: str = "chunk_aligned"

    def __post_init__(self) -> None:
        if self.kv_bits not in (0, 4):
            raise ValueError("kv_bits must be 0 (BF16) or 4")
        if not 0 <= self.open_count <= 8:
            raise ValueError("open_count must be in [0, 8]")
        if self.engine_backend not in ("torch", "kernel"):
            raise ValueError("engine_backend must be 'torch' or 'kernel'")

    def attention_config(self) -> PagedLODConfig:
        return PagedLODConfig(
            chunk_size=self.chunk_size,
            local_window=self.local_window,
            state_growth_factor=self.state_growth_factor,
            state_min_size=self.state_min_size,
            protected_prefix=self.protected_prefix,
            max_routes=8,
            page_size=self.page_size,
            kv_bits=self.kv_bits,
            state_clustering_policy="qk_norm_aware",
            routing_normalization="qk_norm_aware",
        )


class LODModelManager(ModelManager):
    """Transformers Serve model manager that optionally installs LOD."""

    def __init__(
        self,
        checkpoint: str,
        *,
        lod_config: LODServerConfig,
        attention_mode: str = "lod",
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        require_qwen35_fast_path: bool = True,
    ) -> None:
        if attention_mode not in ("lod", "full"):
            raise ValueError("attention_mode must be 'lod' or 'full'")
        self.lod_config = lod_config
        self.attention_mode = attention_mode
        self._lod_device = device
        self.require_qwen35_fast_path = require_qwen35_fast_path
        self.installed_attention_layers: list[str] = []
        self.acceleration: dict[str, bool | int] | None = None
        super().__init__(
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
            model_timeout=-1,
            force_model=checkpoint,
        )

    def _load_processor(self, model_id_and_revision: str):
        model_id, revision = model_id_and_revision.split("@", 1)
        processor = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
        )
        # Composite checkpoints such as Qwen3.5 load a text-only causal model
        # whose model_type differs from the parent config.  Preserve the
        # parent's HF Serve response grammar so tool calls remain parseable.
        from transformers.cli.serving.utils import get_response_template

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
        )
        response_template = get_response_template(
            processor, SimpleNamespace(config=config)
        )
        if response_template is not None:
            processor.response_template = response_template
        return processor

    def _load_model(
        self,
        model_id_and_revision: str,
        tqdm_class: type | None = None,
        progress_callback: Callable | None = None,
    ):
        model_id, revision = model_id_and_revision.split("@", 1)
        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
        )
        get_text_config = getattr(config, "get_text_config", None)
        text_config = (
            get_text_config(decoder=True) if callable(get_text_config) else config
        )
        is_qwen35 = type(text_config).__module__.startswith(
            ("transformers.models.qwen3_5.", "transformers.models.qwen3_5_moe.")
        )
        if is_qwen35:
            from scripts.probe_qwen35_lod_niah import enable_fla_fast_path

            enable_fla_fast_path(required=self.require_qwen35_fast_path)
        config._attn_implementation = "sdpa"
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "loading",
                    "model": model_id_and_revision,
                    "stage": "model",
                }
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            config=config,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            tqdm_class=tqdm_class,
        )
        device = self._lod_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, int) or str(device).isdigit():
            device = f"cuda:{device}"
        model = model.to(torch.device(device)).eval()
        if self.attention_mode == "lod":
            self.installed_attention_layers = install_hf_lod_attention(
                model,
                config=self.lod_config.attention_config(),
                open_count=self.lod_config.open_count,
                engine_backend=self.lod_config.engine_backend,
                left_padding_mode=self.lod_config.left_padding_mode,
            )
        if is_qwen35:
            from scripts.probe_qwen35_lod_niah import require_qwen35_acceleration

            if self.require_qwen35_fast_path:
                self.acceleration = require_qwen35_acceleration(model)
        return model

    def get_gen_models(self, cache_dir: str | None = None) -> list[dict[str, Any]]:
        del cache_dir
        checkpoint = str(self.force_model)
        return [
            {
                "id": checkpoint,
                "object": "model",
                "created": int(time.time()),
                "owned_by": checkpoint.split("/", 1)[0]
                if "/" in checkpoint
                else "local",
            }
        ]


def build_lod_openai_app(
    checkpoint: str,
    *,
    lod_config: LODServerConfig | None = None,
    attention_mode: str = "lod",
    device: str = "cuda",
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    require_qwen35_fast_path: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
    session_cache_config: LODSessionCacheConfig | None = None,
    enable_cors: bool = False,
):
    """Build the standard Transformers Serve app around an LOD model."""
    from transformers.cli.serving.chat_completion import ChatCompletionHandler
    from transformers.cli.serving.completion import CompletionHandler
    from transformers.cli.serving.response import ResponseHandler
    from transformers.cli.serving.server import build_server
    from transformers.cli.serving.transcription import TranscriptionHandler
    from fastapi import Request
    from fastapi.responses import JSONResponse

    model_manager = LODModelManager(
        checkpoint,
        lod_config=lod_config or LODServerConfig(),
        attention_mode=attention_mode,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        require_qwen35_fast_path=require_qwen35_fast_path,
    )
    if attention_mode == "lod":
        # Transformers continuous batching owns a conventional paged KV cache
        # and is incompatible with the LOD-owned cache. The LOD manager retains
        # explicitly named conversation caches on one inference thread.
        generation_state = LODSessionGenerationState(
            session_cache_config or LODSessionCacheConfig()
        )
    else:
        from transformers.cli.serving.utils import GenerationState

        generation_state = GenerationState(continuous_batching=False)
    template_kwargs = chat_template_kwargs or {}
    chat_handler = ChatCompletionHandler(
        model_manager=model_manager,
        generation_state=generation_state,
        chat_template_kwargs=template_kwargs,
    )
    app = build_server(
        model_manager,
        chat_handler,
        completion_handler=CompletionHandler(model_manager, generation_state),
        response_handler=ResponseHandler(
            model_manager,
            generation_state,
            chat_template_kwargs=template_kwargs,
        ),
        transcription_handler=TranscriptionHandler(
            model_manager, generation_state
        ),
        generation_state=generation_state,
        enable_cors=enable_cors,
    )

    @app.middleware("http")
    async def lod_session_middleware(request: Request, call_next):
        if attention_mode != "lod":
            return await call_next(request)
        session_id = request.headers.get(LOD_SESSION_HEADER)
        if session_id is None:
            return await call_next(request)
        session_id = session_id.strip()
        if not session_id or len(session_id) > 256:
            return JSONResponse(
                {"error": f"{LOD_SESSION_HEADER} must contain 1-256 characters"},
                status_code=400,
            )
        token = bind_lod_session(session_id)
        try:
            response = await call_next(request)
            response.headers[LOD_SESSION_HEADER] = session_id
            return response
        finally:
            reset_lod_session(token)

    app.state.lod_model_manager = model_manager
    app.state.lod_generation_state = generation_state
    app.state.attention_mode = attention_mode
    return app


__all__ = [
    "LODModelManager",
    "LODServerConfig",
    "LODSessionCacheConfig",
    "build_lod_openai_app",
]

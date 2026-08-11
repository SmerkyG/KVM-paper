#!/usr/bin/env python3
"""Serve an HF model with LOD through Transformers' OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json

import uvicorn

from model.hf_lod_openai_server import (
    LODServerConfig,
    LODSessionCacheConfig,
    build_lod_openai_app,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--attention-mode", choices=("lod", "full"), default="lod"
    )
    parser.add_argument("--open-count", type=int, default=8)
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=512)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--state-min-size", type=int, default=256)
    parser.add_argument("--protected-prefix", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--engine-backend", choices=("torch", "kernel"), default="kernel"
    )
    parser.add_argument(
        "--left-padding-mode",
        choices=("exact", "chunk_aligned"),
        default="chunk_aligned",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default='{"enable_thinking": false}',
        help="JSON object forwarded to apply_chat_template",
    )
    parser.add_argument("--enable-cors", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument("--session-ttl-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--no-auto-session-discovery",
        action="store_true",
        help="Require X-LOD-Session-ID instead of matching retained token prefixes",
    )
    parser.add_argument("--allow-slow-qwen35", action="store_true")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(template_kwargs, dict):
        raise ValueError("--chat-template-kwargs must decode to an object")
    app = build_lod_openai_app(
        args.checkpoint,
        attention_mode=args.attention_mode,
        lod_config=LODServerConfig(
            chunk_size=args.chunk_size,
            local_window=args.local_window,
            state_growth_factor=args.state_growth_factor,
            state_min_size=args.state_min_size,
            protected_prefix=args.protected_prefix,
            open_count=args.open_count,
            page_size=args.page_size,
            kv_bits=args.kv_bits,
            engine_backend=args.engine_backend,
            left_padding_mode=args.left_padding_mode,
        ),
        device=args.device,
        dtype=args.dtype,
        require_qwen35_fast_path=not args.allow_slow_qwen35,
        chat_template_kwargs=template_kwargs,
        session_cache_config=LODSessionCacheConfig(
            max_sessions=args.max_sessions,
            ttl_seconds=args.session_ttl_seconds,
            automatic_discovery=not args.no_auto_session_discovery,
        ),
        enable_cors=args.enable_cors,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

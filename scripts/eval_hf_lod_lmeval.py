#!/usr/bin/env python3
"""Run lm-eval with the model-independent Hugging Face LOD backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from model.hf_pytorch_lod_attention import (
    install_hf_lod_attention,
    pop_hf_lod_dynamic_open_statistics,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig
from scripts.eval_qwen35_lod_lmeval import patch_hotpotqa_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=("full", "lod"), default="lod")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ruler-length", type=int)
    parser.add_argument("--open-count", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=512)
    parser.add_argument("--state-min-size", type=int, default=256)
    parser.add_argument("--state-size-offset", type=int, default=0)
    parser.add_argument("--protected-prefix", type=int, default=1)
    parser.add_argument(
        "--mla-state-key-normalization",
        choices=("none", "latent", "whole", "raw"),
        default="none",
    )
    parser.add_argument(
        "--mla-recursive-page-key-normalization",
        action="store_true",
    )
    parser.add_argument(
        "--state-clustering-policy",
        choices=(
            "manual",
            "qk_norm_aware",
            "rope_aware",
            "rnope_nope_spherical",
            "rnope_rope_spherical",
        ),
        default="manual",
    )
    parser.add_argument(
        "--state-clustering-normalization",
        choices=("none", "leaf_cosine", "centroid_cosine", "cosine", "l2"),
        default="none",
    )
    parser.add_argument("--state-clustering-radial-bias", type=float, default=0.0)
    parser.add_argument(
        "--state-clustering-radial-scope",
        choices=("all", "append", "assignment"),
        default="all",
    )
    parser.add_argument(
        "--state-clustering-centroid-rescale",
        choices=(
            "none",
            "mean_leaf_norm",
            "coherence",
            "spherical_coherence",
            "rope_coherence",
            "direction_l2",
        ),
        default="none",
    )
    parser.add_argument(
        "--state-clustering-centroid-rescale-scope",
        choices=("all", "append", "assignment"),
        default="all",
    )
    parser.add_argument(
        "--state-clustering-query-metric",
        choices=("none", "diagonal", "full"),
        default="none",
    )
    parser.add_argument(
        "--state-clustering-rope-filter",
        choices=("none", "local_window"),
        default="none",
    )
    parser.add_argument(
        "--exact-coherence-matmul",
        action="store_true",
        help="Use the slower two-GEMM BF16 reference for coherence routing.",
    )
    parser.add_argument(
        "--routing-normalization",
        choices=("none", "query", "key", "both", "qk_norm_aware"),
        default="none",
    )
    parser.add_argument(
        "--routing-rope-filter",
        choices=("none", "local_window"),
        default="none",
    )
    parser.add_argument("--routing-rope-cutoff-factor", type=float, default=1.0)
    parser.add_argument("--routing-rope-jensen", action="store_true")
    parser.add_argument("--routing-count-bias", type=float, default=1.0)
    parser.add_argument("--routing-variance-bias", type=float, default=0.0)
    parser.add_argument(
        "--routing-page-mass-candidates",
        type=int,
        choices=(0, 16, 32, 64, 128),
        default=0,
    )
    parser.add_argument(
        "--routing-leaf-mass-candidates",
        type=int,
        choices=(0, 16, 32, 64, 128),
        default=0,
    )
    parser.add_argument(
        "--routing-leaf-mass-objective",
        choices=(
            "exact",
            "additional",
            "deficit",
            "output",
            "rope_jensen",
            "fast_rope_jensen",
            "slow_rope_jensen",
        ),
        default="exact",
    )
    parser.add_argument("--routing-leaf-mass-top-p", type=float)
    parser.add_argument("--routing-leaf-mass-review-top-p", type=float)
    parser.add_argument("--routing-leaf-mass-min-routes", type=int, default=1)
    parser.add_argument(
        "--lod-layer-indices",
        help="comma-separated zero-based attention-layer indices",
    )
    parser.add_argument(
        "--engine-backend", choices=("torch", "kernel"), default="kernel"
    )
    parser.add_argument("--recursive-pages", action="store_true")
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument(
        "--left-padding-mode",
        choices=("exact", "chunk_aligned"),
        default="chunk_aligned",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-gen-toks", type=int)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument(
        "--use-upstream-code",
        action="store_true",
        help="ignore checkpoint auto_map entries when Transformers supports the model",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(
    checkpoint: str,
    device: torch.device,
    *,
    use_upstream_code: bool = False,
):
    trust_remote_code = not use_upstream_code
    composite_config = AutoConfig.from_pretrained(
        checkpoint, trust_remote_code=trust_remote_code
    )
    config = composite_config.get_text_config(decoder=True)
    is_muse_glimmer = getattr(composite_config, "model_type", None) == "muse_glimmer"
    is_qwen35 = type(config).__module__.startswith(
        "transformers.models.qwen3_5."
    )
    if is_qwen35:
        from scripts.probe_qwen35_lod_niah import enable_fla_fast_path

        enable_fla_fast_path(required=True)
    config._attn_implementation = "sdpa"
    load_kwargs = {}
    composite_model_type = getattr(composite_config, "model_type", None)
    if composite_model_type in {"gemma3", "gemma4"} and config is not composite_config:
        # Gemma 4 publishes the text weights under the multimodal
        # ``model.language_model`` subtree. Loading the text-only causal class
        # avoids allocating the unused vision/audio towers, but needs an
        # explicit one-level checkpoint rename.
        from huggingface_hub import hf_hub_download

        index_path = hf_hub_download(
            checkpoint, "model.safetensors.index.json"
        )
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        if composite_model_type == "gemma4":
            source_prefix = "model.language_model."
            load_kwargs["key_mapping"] = {
                key: "model." + key.removeprefix(source_prefix)
                for key in weight_map
                if key.startswith(source_prefix)
            }
        else:
            source_prefix = "language_model."
            load_kwargs["key_mapping"] = {
                key: key.removeprefix(source_prefix)
                for key in weight_map
                if key.startswith(source_prefix)
            }
    if is_muse_glimmer:
        # Muse-Glimmer currently exposes its text decoder only through the
        # multimodal conditional-generation class. Text-only ``input_ids``
        # still take the ordinary causal path; no vision inputs are allocated.
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
        load_config = composite_config
    else:
        model_class = AutoModelForCausalLM
        load_config = config
    model = (
        model_class.from_pretrained(
            checkpoint,
            config=load_config,
            dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
        .to(device)
        .eval()
    )
    if is_qwen35:
        from scripts.probe_qwen35_lod_niah import require_qwen35_acceleration

        acceleration = require_qwen35_acceleration(model)
    else:
        acceleration = None
    return model, acceleration


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.ruler_length is not None and args.ruler_length < 1:
        raise ValueError("RULER length must be positive")
    if args.max_gen_toks is not None and args.max_gen_toks < 1:
        raise ValueError("maximum generated tokens must be positive")
    if not 0 <= args.open_count <= 128:
        raise ValueError("open count must be in [0, 128]")
    if args.kv_bits and not args.recursive_pages:
        raise ValueError("KV quantization requires --recursive-pages")
    if (
        args.routing_leaf_mass_top_p is not None
        or args.routing_leaf_mass_review_top_p is not None
    ) and (
        args.engine_backend != "kernel" or not args.recursive_pages
    ):
        raise ValueError(
            "leaf-mass top-p requires --engine-backend kernel and --recursive-pages"
        )
    lod_layer_indices = None
    if args.lod_layer_indices is not None:
        lod_layer_indices = {
            int(value) for value in args.lod_layer_indices.split(",") if value
        }
        if not lod_layer_indices or any(index < 0 for index in lod_layer_indices):
            raise ValueError("LOD layer indices must be nonnegative integers")
        if args.mode != "lod":
            raise ValueError("LOD layer indices require --mode lod")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=not args.use_upstream_code
    )
    model, acceleration = load_model(
        args.checkpoint,
        device,
        use_upstream_code=args.use_upstream_code,
    )
    installed = []
    if args.mode == "lod":
        config_kwargs = {
            "chunk_size": args.chunk_size,
            "local_window": args.local_window,
            "state_growth_factor": args.state_growth_factor,
            "state_min_size": args.state_min_size,
            "state_size_offset": args.state_size_offset,
            "protected_prefix": args.protected_prefix,
            "mla_state_key_normalization": args.mla_state_key_normalization,
            "mla_recursive_page_key_normalization": (
                args.mla_recursive_page_key_normalization
            ),
            "state_clustering_policy": args.state_clustering_policy,
            "state_clustering_normalization": (
                args.state_clustering_normalization
            ),
            "state_clustering_radial_bias": args.state_clustering_radial_bias,
            "state_clustering_radial_scope": args.state_clustering_radial_scope,
            "state_clustering_centroid_rescale": (
                args.state_clustering_centroid_rescale
            ),
            "state_clustering_centroid_rescale_scope": (
                args.state_clustering_centroid_rescale_scope
            ),
            "state_clustering_query_metric": args.state_clustering_query_metric,
            "state_clustering_rope_filter": args.state_clustering_rope_filter,
            "coherence_single_matmul": not args.exact_coherence_matmul,
            "routing_normalization": args.routing_normalization,
            "routing_rope_filter": args.routing_rope_filter,
            "routing_rope_cutoff_factor": args.routing_rope_cutoff_factor,
            "routing_rope_jensen": args.routing_rope_jensen,
            "routing_count_bias": args.routing_count_bias,
            "routing_variance_bias": args.routing_variance_bias,
            "routing_page_mass_candidates": args.routing_page_mass_candidates,
            "routing_leaf_mass_candidates": args.routing_leaf_mass_candidates,
            "routing_leaf_mass_objective": args.routing_leaf_mass_objective,
            "routing_leaf_mass_review_top_p": (
                args.routing_leaf_mass_review_top_p
            ),
            "routing_leaf_mass_top_p": args.routing_leaf_mass_top_p,
            "routing_leaf_mass_min_routes": args.routing_leaf_mass_min_routes,
            "max_routes": args.open_count,
        }
        config = (
            PagedLODConfig(**config_kwargs, page_size=16, kv_bits=args.kv_bits)
            if args.recursive_pages
            else LODConfig(**config_kwargs)
        )
        installed = install_hf_lod_attention(
            model,
            config=config,
            open_count=args.open_count,
            engine_backend=args.engine_backend,
            left_padding_mode=args.left_padding_mode,
            layer_indices=lod_layer_indices,
        )
    if local_rank == 0:
        if args.mode == "lod":
            print(f"installed HF LOD on {len(installed)} attention layers")
        else:
            print("using native full attention")
        if acceleration is not None:
            print(
                "Qwen3.5 acceleration: "
                + json.dumps(acceleration, sort_keys=True)
            )

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device=str(device),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        enable_thinking=False if args.disable_thinking else None,
    )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from accelerate import Accelerator

        accelerator = Accelerator()
        lm.accelerator = accelerator
        lm._rank = accelerator.process_index
        lm._world_size = accelerator.num_processes
    patch_hotpotqa_download()

    metadata = {
        "pretrained": args.checkpoint,
        "tokenizer": args.checkpoint,
    }
    if args.ruler_length is not None:
        metadata["max_seq_lengths"] = [args.ruler_length]
    gen_kwargs = {}
    if args.max_gen_toks is not None:
        gen_kwargs["max_gen_toks"] = args.max_gen_toks
    evaluation_start = time.perf_counter()
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        bootstrap_iters=0,
        log_samples=args.log_samples,
        gen_kwargs=gen_kwargs or None,
        apply_chat_template=args.apply_chat_template,
        metadata=metadata,
        confirm_run_unsafe_code=True,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    dynamic_open_statistics = (
        pop_hf_lod_dynamic_open_statistics(model)
        if args.mode == "lod"
        and (
            args.routing_leaf_mass_top_p is not None
            or args.routing_leaf_mass_review_top_p is not None
        )
        else {}
    )
    if results is None:
        return

    payload = dict(results)
    payload["lod_evaluation"] = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "ruler_length": args.ruler_length,
        "open_count": args.open_count,
        "state_growth_factor": args.state_growth_factor,
        "chunk_size": args.chunk_size,
        "local_window": args.local_window,
        "state_min_size": args.state_min_size,
        "state_size_offset": args.state_size_offset,
        "protected_prefix": args.protected_prefix,
        "mla_state_key_normalization": args.mla_state_key_normalization,
        "mla_recursive_page_key_normalization": (
            args.mla_recursive_page_key_normalization
        ),
        "state_clustering_policy": args.state_clustering_policy,
        "state_clustering_normalization": args.state_clustering_normalization,
        "state_clustering_radial_bias": args.state_clustering_radial_bias,
        "state_clustering_radial_scope": args.state_clustering_radial_scope,
        "state_clustering_centroid_rescale": (
            args.state_clustering_centroid_rescale
        ),
        "state_clustering_centroid_rescale_scope": (
            args.state_clustering_centroid_rescale_scope
        ),
        "state_clustering_query_metric": args.state_clustering_query_metric,
        "state_clustering_rope_filter": args.state_clustering_rope_filter,
        "coherence_single_matmul": not args.exact_coherence_matmul,
        "routing_normalization": args.routing_normalization,
        "routing_rope_filter": args.routing_rope_filter,
        "routing_rope_cutoff_factor": args.routing_rope_cutoff_factor,
        "routing_rope_jensen": args.routing_rope_jensen,
        "routing_count_bias": args.routing_count_bias,
        "routing_variance_bias": args.routing_variance_bias,
        "routing_page_mass_candidates": args.routing_page_mass_candidates,
        "routing_leaf_mass_candidates": args.routing_leaf_mass_candidates,
        "routing_leaf_mass_objective": args.routing_leaf_mass_objective,
        "routing_leaf_mass_review_top_p": args.routing_leaf_mass_review_top_p,
        "routing_leaf_mass_top_p": args.routing_leaf_mass_top_p,
        "routing_leaf_mass_min_routes": args.routing_leaf_mass_min_routes,
        "dynamic_open_statistics": dynamic_open_statistics,
        "lod_layer_indices": (
            sorted(lod_layer_indices) if lod_layer_indices is not None else None
        ),
        "engine_backend": args.engine_backend,
        "recursive_pages": args.recursive_pages,
        "kv_bits": args.kv_bits,
        "left_padding_mode": args.left_padding_mode,
        "max_gen_toks": args.max_gen_toks,
        "apply_chat_template": args.apply_chat_template,
        "disable_thinking": args.disable_thinking,
        "use_upstream_code": args.use_upstream_code,
        "attention_layers": installed,
        "evaluation_seconds": evaluation_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(make_table(results))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

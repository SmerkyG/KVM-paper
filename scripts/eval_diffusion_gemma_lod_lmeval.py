#!/usr/bin/env python3
"""Run generation-only lm-eval tasks on native or LOD DiffusionGemma."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from lm_eval import evaluator
from lm_eval.utils import make_table
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

from model.hf_diffusion_gemma_lod_attention import (
    install_diffusion_gemma_lod_attention,
)
from model.diffusion_gemma_acceptance_compare import (
    DiffusionGemmaAcceptanceComparator,
    DiffusionGemmaEarlyNativeController,
)
from model.diffusion_gemma_phase_compare import DiffusionGemmaPhaseComparator
from model.diffusion_gemma_consensus_acceptance import (
    DiffusionGemmaConsensusAcceptance,
)
from model.diffusion_gemma_full_attention_review import (
    DiffusionGemmaFullAttentionReviewer,
)
from model.diffusion_gemma_native_entropy_acceptance import (
    DiffusionGemmaNativeEntropyAcceptance,
)
from model.diffusion_gemma_route_mass_compare import (
    DiffusionGemmaRouteMassComparator,
)
from model.lm_eval_diffusion_gemma import DiffusionGemmaARLM, DiffusionGemmaLM
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig
from scripts.eval_qwen35_lod_lmeval import patch_hotpotqa_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="google/diffusiongemma-26B-A4B-it"
    )
    parser.add_argument("--mode", choices=("full", "lod"), required=True)
    parser.add_argument(
        "--generation-mode", choices=("diffusion", "ar"), default="diffusion"
    )
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ruler-length", type=int)
    parser.add_argument("--max-denoising-steps", type=int)
    parser.add_argument("--max-gen-toks", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--open-count", type=int, default=8)
    parser.add_argument("--decoder-open-count", type=int)
    parser.add_argument(
        "--decoder-routing",
        choices=("per_query", "canvas_max", "canvas_cumulative_max"),
        default="per_query",
    )
    parser.add_argument("--compare-native-acceptance", action="store_true")
    parser.add_argument("--compare-attention-phases", action="store_true")
    parser.add_argument(
        "--acceptance-entropy-source",
        choices=("lod", "native_native"),
        default="lod",
        help=(
            "Use either primary LOD entropy or a paired native-encoder/native-"
            "decoder entropy view to choose accepted token positions"
        ),
    )
    parser.add_argument(
        "--compare-route-mass",
        action="store_true",
        help=(
            "Audit native-query exact attention mass and UUID page recall on "
            "the unchanged LOD denoising trajectory"
        ),
    )
    parser.add_argument(
        "--consensus-acceptance",
        choices=("off", "observe", "apply"),
        default="off",
    )
    parser.add_argument("--consensus-probe-open-count", type=int, default=16)
    parser.add_argument(
        "--full-attention-review",
        choices=("off", "observe", "apply"),
        default="off",
    )
    parser.add_argument(
        "--full-review-lod-entropy-threshold", type=float, default=0.001
    )
    parser.add_argument(
        "--full-review-policy",
        choices=("sample_top1", "native_acceptance"),
        default="sample_top1",
    )
    parser.add_argument(
        "--early-native-steps",
        type=int,
        default=0,
        help=(
            "Use native decoder attention for the first N denoising calls of "
            "each canvas while retaining LOD encoder prefill"
        ),
    )
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--local-window",
        type=int,
        default=768,
        help="Exact keys including the 256-token canvas (768 = 512-token prefix lookback)",
    )
    parser.add_argument("--state-min-size", type=int, default=256)
    parser.add_argument("--protected-prefix", type=int, default=1)
    parser.add_argument(
        "--state-clustering-policy",
        choices=("manual", "qk_norm_aware"),
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
        "--routing-normalization",
        choices=("none", "query", "key", "both"),
        default="none",
    )
    parser.add_argument("--routing-count-bias", type=float, default=1.0)
    parser.add_argument("--routing-variance-bias", type=float, default=0.0)
    parser.add_argument(
        "--engine-backend", choices=("torch", "kernel"), default="kernel"
    )
    parser.add_argument(
        "--recursive-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prefill-policy",
        choices=("optimized", "legacy"),
        default="optimized",
        help="Recursive kernel prefill schedule; does not change LOD state size",
    )
    parser.add_argument(
        "--encoder-attention",
        choices=("lod", "native"),
        default="lod",
        help="Encoder output attention; the LOD sidecar is built in either mode",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.ruler_length is not None and args.ruler_length < 1:
        raise ValueError("RULER length must be positive")
    if args.max_denoising_steps is not None and args.max_denoising_steps < 1:
        raise ValueError("max denoising steps must be positive")
    if args.max_gen_toks is not None and args.max_gen_toks < 1:
        raise ValueError("max generated tokens must be positive")
    if args.generation_mode == "ar" and args.max_denoising_steps is not None:
        raise ValueError("max denoising steps do not apply to AR generation")
    if not 0 <= args.open_count <= 8:
        raise ValueError("open count must be in [0, 8]")
    if args.decoder_open_count is not None and args.decoder_open_count < 0:
        raise ValueError("decoder open count cannot be negative")
    if args.early_native_steps < 0:
        raise ValueError("early native steps cannot be negative")
    if args.early_native_steps and args.mode != "lod":
        raise ValueError("early native steps require --mode lod")
    if args.early_native_steps and args.compare_native_acceptance:
        raise ValueError(
            "early native steps cannot be combined with native acceptance comparison"
        )
    if args.compare_attention_phases and args.mode != "lod":
        raise ValueError("attention phase comparison requires --mode lod")
    if args.compare_attention_phases and (
        args.compare_native_acceptance or args.early_native_steps
    ):
        raise ValueError(
            "attention phase comparison cannot be combined with another "
            "denoising-step diagnostic"
        )
    if args.compare_attention_phases and args.encoder_attention != "lod":
        raise ValueError("attention phase comparison requires an LOD encoder trajectory")
    if args.acceptance_entropy_source == "native_native" and args.mode != "lod":
        raise ValueError("native/native entropy acceptance requires --mode lod")
    if (
        args.acceptance_entropy_source == "native_native"
        and args.encoder_attention != "lod"
    ):
        raise ValueError(
            "native/native entropy acceptance requires --encoder-attention lod "
            "for the primary sampling trajectory"
        )
    if args.acceptance_entropy_source == "native_native" and (
        args.compare_native_acceptance
        or args.compare_attention_phases
        or args.early_native_steps
        or args.consensus_acceptance != "off"
        or args.full_attention_review != "off"
        or args.compare_route_mass
    ):
        raise ValueError(
            "native/native entropy acceptance cannot be combined with another "
            "denoising-step diagnostic"
        )
    if args.compare_route_mass and args.mode != "lod":
        raise ValueError("route-mass comparison requires --mode lod")
    if args.compare_route_mass and not args.recursive_pages:
        raise ValueError("route-mass comparison requires recursive pages")
    if args.compare_route_mass and args.engine_backend != "kernel":
        raise ValueError("route-mass comparison requires the kernel backend")
    if args.compare_route_mass and args.encoder_attention != "native":
        raise ValueError(
            "route-mass comparison requires --encoder-attention native so the "
            "native and LOD views share the same encoder trajectory"
        )
    if args.compare_route_mass and (
        args.compare_native_acceptance
        or args.compare_attention_phases
        or args.early_native_steps
        or args.consensus_acceptance != "off"
        or args.full_attention_review != "off"
    ):
        raise ValueError(
            "route-mass comparison cannot be combined with another "
            "denoising-step diagnostic"
        )
    if args.consensus_probe_open_count < 1:
        raise ValueError("consensus probe open count must be positive")
    if args.full_review_lod_entropy_threshold < 0.0:
        raise ValueError("full-review LOD entropy threshold cannot be negative")
    if args.consensus_acceptance != "off" and args.mode != "lod":
        raise ValueError("consensus acceptance requires --mode lod")
    if args.consensus_acceptance != "off" and (
        args.compare_native_acceptance
        or args.compare_attention_phases
        or args.early_native_steps
    ):
        raise ValueError(
            "consensus acceptance cannot be combined with another "
            "denoising-step diagnostic"
        )
    if (
        args.consensus_acceptance != "off"
        and args.consensus_probe_open_count > 8
        and args.engine_backend != "kernel"
    ):
        raise ValueError("a consensus probe above eight routes requires kernel mode")
    if args.full_attention_review != "off" and args.mode != "lod":
        raise ValueError("full-attention review requires --mode lod")
    if args.full_attention_review != "off" and (
        args.compare_native_acceptance
        or args.compare_attention_phases
        or args.early_native_steps
        or args.consensus_acceptance != "off"
    ):
        raise ValueError(
            "full-attention review cannot be combined with another "
            "denoising-step diagnostic"
        )
    if args.local_window < args.chunk_size:
        raise ValueError("local window must include at least the diffusion canvas")
    if args.generation_mode == "ar" and (
        args.compare_native_acceptance
        or args.compare_attention_phases
        or args.early_native_steps
        or args.consensus_acceptance != "off"
        or args.full_attention_review != "off"
        or args.compare_route_mass
        or args.acceptance_entropy_source != "lod"
    ):
        raise ValueError("denoising diagnostics do not apply to AR generation")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    installed: list[int] = []
    acceptance_comparator = None
    early_native_controller = None
    phase_comparator = None
    consensus_controller = None
    full_attention_reviewer = None
    route_mass_comparator = None
    native_entropy_controller = None
    if args.mode == "lod":
        common = {
            "chunk_size": args.chunk_size,
            "local_window": args.local_window,
            "state_growth_factor": args.state_growth_factor,
            "state_min_size": args.state_min_size,
            "protected_prefix": args.protected_prefix,
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
            "routing_normalization": args.routing_normalization,
            "routing_count_bias": args.routing_count_bias,
            "routing_variance_bias": args.routing_variance_bias,
            "max_routes": 8,
            "leaf_dtype": torch.bfloat16,
        }
        config = (
            PagedLODConfig(
                **common,
                page_size=16,
                kv_bits=0,
                quant_group_size=32,
            )
            if args.recursive_pages
            else LODConfig(**common)
        )
        installed = install_diffusion_gemma_lod_attention(
            model,
            config=config,
            open_count=args.open_count,
            engine_backend=args.engine_backend,
            decoder_open_count=args.decoder_open_count,
            decoder_routing=args.decoder_routing,
            prefill_policy=args.prefill_policy,
            encoder_attention_mode=args.encoder_attention,
        )
        print(f"installed DiffusionGemma LOD on {len(installed)} attention layers")
        if args.compare_native_acceptance:
            acceptance_comparator = DiffusionGemmaAcceptanceComparator(model)
            acceptance_comparator.install()
        if args.early_native_steps:
            early_native_controller = DiffusionGemmaEarlyNativeController(
                model, early_steps=args.early_native_steps
            )
            early_native_controller.install()
        if args.compare_attention_phases:
            phase_comparator = DiffusionGemmaPhaseComparator(model)
            phase_comparator.install()
        if args.consensus_acceptance != "off":
            consensus_controller = DiffusionGemmaConsensusAcceptance(
                model,
                probe_open_count=args.consensus_probe_open_count,
                mode=args.consensus_acceptance,
            )
            consensus_controller.install()
        if args.full_attention_review != "off":
            full_attention_reviewer = DiffusionGemmaFullAttentionReviewer(
                model,
                lod_entropy_threshold=args.full_review_lod_entropy_threshold,
                mode=args.full_attention_review,
                policy=args.full_review_policy,
            )
            full_attention_reviewer.install()
        if args.compare_route_mass:
            route_mass_comparator = DiffusionGemmaRouteMassComparator(
                model, tokenizer
            )
            route_mass_comparator.install()
        if args.acceptance_entropy_source == "native_native":
            native_entropy_controller = DiffusionGemmaNativeEntropyAcceptance(model)
            native_entropy_controller.install()
    elif args.compare_native_acceptance:
        raise ValueError("native acceptance comparison requires --mode lod")

    lm_class = (
        DiffusionGemmaARLM
        if args.generation_mode == "ar"
        else DiffusionGemmaLM
    )
    lm = lm_class(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=str(device),
        dtype=torch.bfloat16,
    )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from accelerate import Accelerator

        accelerator = Accelerator()
        lm.accelerator = accelerator
        lm._rank = accelerator.process_index
        lm._world_size = accelerator.num_processes

    patch_hotpotqa_download()
    metadata = {"pretrained": args.checkpoint, "tokenizer": args.checkpoint}
    if args.ruler_length is not None:
        metadata["max_seq_lengths"] = [args.ruler_length]
    gen_kwargs = {}
    if args.max_denoising_steps is not None:
        gen_kwargs["max_denoising_steps"] = args.max_denoising_steps
    if args.max_gen_toks is not None:
        gen_kwargs["max_gen_toks"] = args.max_gen_toks

    started = time.perf_counter()
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        bootstrap_iters=0,
        log_samples=args.log_samples,
        gen_kwargs=gen_kwargs or None,
        apply_chat_template=args.apply_chat_template,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        metadata=metadata,
        confirm_run_unsafe_code=True,
    )
    if results is None:
        return

    payload = dict(results)
    payload["diffusion_gemma_evaluation"] = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "generation_mode": args.generation_mode,
        "tasks": args.tasks,
        "batch_size": args.batch_size,
        "ruler_length": args.ruler_length,
        "max_denoising_steps": args.max_denoising_steps,
        "max_gen_toks": args.max_gen_toks,
        "seed": args.seed,
        "apply_chat_template": args.apply_chat_template,
        "open_count": args.open_count if args.mode == "lod" else None,
        "decoder_open_count": (
            args.decoder_open_count if args.mode == "lod" else None
        ),
        "decoder_routing": args.decoder_routing if args.mode == "lod" else None,
        "state_growth_factor": args.state_growth_factor if args.mode == "lod" else None,
        "chunk_size": args.chunk_size if args.mode == "lod" else None,
        "local_window": args.local_window if args.mode == "lod" else None,
        "state_clustering_policy": (
            args.state_clustering_policy if args.mode == "lod" else None
        ),
        "state_clustering_normalization": (
            args.state_clustering_normalization if args.mode == "lod" else None
        ),
        "state_clustering_radial_bias": (
            args.state_clustering_radial_bias if args.mode == "lod" else None
        ),
        "state_clustering_radial_scope": (
            args.state_clustering_radial_scope if args.mode == "lod" else None
        ),
        "state_clustering_centroid_rescale": (
            args.state_clustering_centroid_rescale
            if args.mode == "lod"
            else None
        ),
        "state_clustering_centroid_rescale_scope": (
            args.state_clustering_centroid_rescale_scope
            if args.mode == "lod"
            else None
        ),
        "routing_normalization": (
            args.routing_normalization if args.mode == "lod" else None
        ),
        "routing_count_bias": (
            args.routing_count_bias if args.mode == "lod" else None
        ),
        "routing_variance_bias": (
            args.routing_variance_bias if args.mode == "lod" else None
        ),
        "recursive_pages": args.recursive_pages if args.mode == "lod" else None,
        "engine_backend": args.engine_backend if args.mode == "lod" else None,
        "prefill_policy": args.prefill_policy if args.mode == "lod" else None,
        "encoder_attention": args.encoder_attention if args.mode == "lod" else None,
        "acceptance_entropy_source": args.acceptance_entropy_source,
        "attention_layers": installed,
        "evaluation_seconds": time.perf_counter() - started,
        "generation_statistics": (
            lm.ar_generation_statistics
            if isinstance(lm, DiffusionGemmaARLM)
            else lm.diffusion_generation_statistics
        ),
        "native_acceptance_comparison": (
            acceptance_comparator.summary()
            if acceptance_comparator is not None
            else None
        ),
        "early_native_attention": (
            early_native_controller.summary()
            if early_native_controller is not None
            else None
        ),
        "attention_phase_comparison": (
            phase_comparator.summary() if phase_comparator is not None else None
        ),
        "consensus_acceptance": (
            consensus_controller.summary()
            if consensus_controller is not None
            else None
        ),
        "full_attention_review": (
            full_attention_reviewer.summary()
            if full_attention_reviewer is not None
            else None
        ),
        "route_mass_comparison": (
            route_mass_comparator.summary()
            if route_mass_comparator is not None
            else None
        ),
        "native_entropy_acceptance": (
            native_entropy_controller.summary()
            if native_entropy_controller is not None
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(make_table(results))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare native and generic-HF LOD causal LM loss on ProLong text."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from model.hf_pytorch_lod_attention import (
    install_hf_lod_attention,
    pop_hf_lod_dynamic_open_statistics,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.slabbed_lod_attention import (
    SlabbedLODConfig,
    SlabbedTwoLevelLODAttention,
)
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.eval_hf_lod_lmeval import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("full", "lod"), required=True)
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--open-count", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=512)
    parser.add_argument("--state-min-size", type=int, default=256)
    parser.add_argument("--protected-prefix", type=int, default=1)
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
        "--engine-backend", choices=("torch", "kernel"), default="kernel"
    )
    parser.add_argument("--recursive-pages", action="store_true")
    parser.add_argument("--slabbed", action="store_true")
    parser.add_argument("--slab-size", type=int, default=4096)
    parser.add_argument("--slots-per-slab", type=int, default=256)
    parser.add_argument("--slab-local-slabs", type=int, default=2)
    parser.add_argument("--slab-routing-chunk-size", type=int, default=4096)
    parser.add_argument("--slab-query-chunk-size", type=int, default=4096)
    parser.add_argument("--slab-merge-group-slabs", type=int, default=0)
    parser.add_argument("--slab-merged-slots-per-group", type=int, default=0)
    parser.add_argument("--slab-merge-budget-growth-factor", type=float, default=0.0)
    parser.add_argument("--slab-coarse-variance-bias", type=float, default=0.0)
    parser.add_argument("--slab-exact-closed-mass-oracle", action="store_true")
    parser.add_argument(
        "--slab-seed-selection",
        choices=("prefix", "strided"),
        default="strided",
    )
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument(
        "--use-upstream-code",
        action="store_true",
        help="ignore checkpoint auto_map entries when Transformers supports the model",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2 or args.samples < 1:
        raise ValueError("sequence length and sample count must be positive")
    if not 0 <= args.open_count <= 128:
        raise ValueError("open count must be in [0, 128]")
    if args.kv_bits and not args.recursive_pages:
        raise ValueError("KV quantization requires --recursive-pages")
    if args.slabbed and args.recursive_pages:
        raise ValueError("--slabbed and --recursive-pages are mutually exclusive")
    if args.slabbed and args.engine_backend != "torch":
        raise ValueError("--slabbed currently requires --engine-backend torch")
    if (
        args.routing_leaf_mass_top_p is not None
        or args.routing_leaf_mass_review_top_p is not None
    ) and (
        args.engine_backend != "kernel" or not args.recursive_pages
    ):
        raise ValueError(
            "leaf-mass top-p requires --engine-backend kernel and --recursive-pages"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=not args.use_upstream_code
    )
    sequences = select_sequences(
        tokenizer,
        args.dataset,
        args.sequence_length,
        args.samples,
        rank,
        world_size,
    )
    model, acceleration = load_model(
        args.checkpoint,
        device,
        use_upstream_code=args.use_upstream_code,
    )
    installed: list[str] = []
    if args.mode == "lod":
        config_kwargs = {
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
            "state_clustering_query_metric": args.state_clustering_query_metric,
            "state_clustering_rope_filter": args.state_clustering_rope_filter,
            "routing_normalization": args.routing_normalization,
            "routing_rope_filter": args.routing_rope_filter,
            "routing_rope_cutoff_factor": args.routing_rope_cutoff_factor,
            "routing_rope_jensen": args.routing_rope_jensen,
            "routing_count_bias": args.routing_count_bias,
            "routing_variance_bias": args.routing_variance_bias,
            "routing_leaf_mass_candidates": args.routing_leaf_mass_candidates,
            "routing_leaf_mass_objective": args.routing_leaf_mass_objective,
            "routing_leaf_mass_review_top_p": (
                args.routing_leaf_mass_review_top_p
            ),
            "routing_leaf_mass_top_p": args.routing_leaf_mass_top_p,
            "routing_leaf_mass_min_routes": args.routing_leaf_mass_min_routes,
            "max_routes": args.open_count,
        }
        if args.slabbed:
            lod_config = SlabbedLODConfig(
                **config_kwargs,
                slab_size=args.slab_size,
                slots_per_slab=args.slots_per_slab,
                local_slabs=args.slab_local_slabs,
                routing_chunk_size=args.slab_routing_chunk_size,
                query_chunk_size=args.slab_query_chunk_size,
                seed_selection=args.slab_seed_selection,
                merge_group_slabs=args.slab_merge_group_slabs,
                merged_slots_per_group=args.slab_merged_slots_per_group,
                merge_budget_growth_factor=args.slab_merge_budget_growth_factor,
                coarse_variance_bias=args.slab_coarse_variance_bias,
                exact_closed_mass_oracle=args.slab_exact_closed_mass_oracle,
            )
        elif args.recursive_pages:
            lod_config = PagedLODConfig(
                **config_kwargs, page_size=16, kv_bits=args.kv_bits
            )
        else:
            lod_config = LODConfig(**config_kwargs)
        installed = install_hf_lod_attention(
            model,
            config=lod_config,
            open_count=args.open_count,
            engine_backend=args.engine_backend,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loss_sum = 0.0
    prediction_tokens = 0
    records = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for sample, sequence in sequences:
            input_ids = sequence.unsqueeze(0).to(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - started
            sample_tokens = int(input_ids.numel()) - 1
            sample_loss = float(result.loss)
            loss_sum += sample_loss * sample_tokens
            prediction_tokens += sample_tokens
            record = {
                "sample": sample,
                "loss": sample_loss,
                "perplexity": math.exp(min(sample_loss, 80.0)),
                "prediction_tokens": sample_tokens,
                "elapsed_seconds": elapsed_seconds,
            }
            records.append(record)
            print(
                f"mode={args.mode} sample={sample} loss={sample_loss:.6f} "
                f"seconds={elapsed_seconds:.3f}",
                flush=True,
            )
            del input_ids, result

    mean_loss = loss_sum / prediction_tokens
    dynamic_open_statistics = (
        pop_hf_lod_dynamic_open_statistics(model)
        if args.mode == "lod"
        and (
            args.routing_leaf_mass_top_p is not None
            or args.routing_leaf_mass_review_top_p is not None
        )
        else {}
    )
    slab_region_statistics = {}
    if args.slabbed:
        for name, module in model.named_modules():
            engine = getattr(module, "_hf_lod_transient_engine", None)
            if isinstance(engine, SlabbedTwoLevelLODAttention):
                slab_region_statistics[name] = engine.last_region_leaf_statistics
    payload = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "dataset": args.dataset,
        "sequence_length": args.sequence_length,
        "samples": args.samples,
        "prediction_tokens": prediction_tokens,
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "attention_layers": installed,
        "lod": {
            "open_count": args.open_count,
            "state_growth_factor": args.state_growth_factor,
            "chunk_size": args.chunk_size,
            "local_window": args.local_window,
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
            "state_clustering_query_metric": args.state_clustering_query_metric,
            "state_clustering_rope_filter": args.state_clustering_rope_filter,
            "routing_normalization": args.routing_normalization,
            "routing_rope_filter": args.routing_rope_filter,
            "routing_rope_cutoff_factor": args.routing_rope_cutoff_factor,
            "routing_rope_jensen": args.routing_rope_jensen,
            "routing_count_bias": args.routing_count_bias,
            "routing_variance_bias": args.routing_variance_bias,
            "routing_leaf_mass_candidates": args.routing_leaf_mass_candidates,
            "routing_leaf_mass_objective": args.routing_leaf_mass_objective,
            "routing_leaf_mass_review_top_p": (
                args.routing_leaf_mass_review_top_p
            ),
            "routing_leaf_mass_top_p": args.routing_leaf_mass_top_p,
            "routing_leaf_mass_min_routes": args.routing_leaf_mass_min_routes,
            "dynamic_open_statistics": dynamic_open_statistics,
            "engine_backend": args.engine_backend,
            "recursive_pages": args.recursive_pages,
            "slabbed": args.slabbed,
            "slab_size": args.slab_size if args.slabbed else None,
            "slots_per_slab": args.slots_per_slab if args.slabbed else None,
            "slab_local_slabs": (
                args.slab_local_slabs if args.slabbed else None
            ),
            "slab_routing_chunk_size": (
                args.slab_routing_chunk_size if args.slabbed else None
            ),
            "slab_query_chunk_size": (
                args.slab_query_chunk_size if args.slabbed else None
            ),
            "slab_seed_selection": (
                args.slab_seed_selection if args.slabbed else None
            ),
            "slab_merge_group_slabs": (
                args.slab_merge_group_slabs if args.slabbed else None
            ),
            "slab_merged_slots_per_group": (
                args.slab_merged_slots_per_group if args.slabbed else None
            ),
            "slab_merge_budget_growth_factor": (
                args.slab_merge_budget_growth_factor if args.slabbed else None
            ),
            "slab_coarse_variance_bias": (
                args.slab_coarse_variance_bias if args.slabbed else None
            ),
            "slab_exact_closed_mass_oracle": (
                args.slab_exact_closed_mass_oracle if args.slabbed else None
            ),
            "slab_region_statistics": slab_region_statistics,
            "kv_bits": args.kv_bits,
        },
        "acceleration": acceleration,
        "use_upstream_code": args.use_upstream_code,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

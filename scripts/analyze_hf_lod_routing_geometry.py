#!/usr/bin/env python3
"""Measure architecture-linked query temperature and LOD centroid coherence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from model import hf_pytorch_lod_attention as hf_lod
from model.hf_pytorch_lod_attention import install_hf_lod_attention
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_attention import TritonLODAttentionCore
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.eval_hf_lod_lmeval import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten().cpu()
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _slope(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / x.square().sum().clamp_min(1e-12))


def _rotary_dims(config: Any, layer_idx: int, head_dim: int) -> int:
    if getattr(config, "model_type", None) == "smollm3":
        if not bool(config.no_rope_layers[layer_idx]):
            return 0
    parameters = getattr(config, "rope_parameters", None)
    if isinstance(parameters, dict):
        factor = float(parameters.get("partial_rotary_factor", 1.0))
    else:
        factor = float(getattr(config, "partial_rotary_factor", 1.0))
    return int(round(head_dim * factor))


def _geometry_record(
    self: TritonLODAttentionCore,
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    local_k: torch.Tensor | None,
    local_v: torch.Tensor | None,
    *,
    state_len: int,
) -> dict[str, Any]:
    valid_counts = counts.detach()[..., :state_len, 0].float()
    protected = self._protected_state_len(state_len)
    valid_counts = valid_counts[..., protected:]
    mean_key = (
        state_k.detach()[..., protected:state_len, :].float()
        / valid_counts.clamp_min(1).unsqueeze(-1)
    )
    query_rms = q.detach().float().square().mean(dim=-1).sqrt()
    if local_k is not None and int(local_k.size(2)):
        reference_rms = (
            local_k.detach().float().square().mean(dim=(-2, -1)).sqrt()
        )
    else:
        singleton = valid_counts.eq(1).unsqueeze(-1)
        reference_rms = (
            mean_key.square().mean(-1, keepdim=True)
            .masked_fill(~singleton, 0)
            .sum(-2)
            / singleton.sum(-2).clamp_min(1)
        ).sqrt()
    centroid_rms = mean_key.square().mean(-1).sqrt()
    coherence = centroid_rms / reference_rms.unsqueeze(-1).clamp_min(1e-12)
    local_key_token_rms = (
        local_k.detach().float().square().mean(dim=-1).sqrt()
        if local_k is not None and int(local_k.size(2))
        else mean_key.new_empty(0)
    )
    log_count = valid_counts.clamp_min(1).log()
    log_coherence = coherence.clamp_min(1e-12).log()
    mean_value = (
        state_v.detach()[..., protected:state_len, :].float()
        / valid_counts.clamp_min(1).unsqueeze(-1)
    )
    if local_v is not None and int(local_v.size(2)):
        reference_value_rms = (
            local_v.detach().float().square().mean(dim=(-2, -1)).sqrt()
        )
    else:
        singleton = valid_counts.eq(1).unsqueeze(-1)
        reference_value_rms = (
            mean_value.square().mean(-1, keepdim=True)
            .masked_fill(~singleton, 0)
            .sum(-2)
            / singleton.sum(-2).clamp_min(1)
        ).sqrt()
    centroid_value_rms = mean_value.square().mean(-1).sqrt()
    value_coherence = centroid_value_rms / reference_value_rms.unsqueeze(
        -1
    ).clamp_min(1e-12)
    record: dict[str, Any] = {
        "query_rms": _summary(query_rms),
        "reference_key_rms": _summary(reference_rms),
        "local_key_token_rms": (
            _summary(local_key_token_rms) if local_key_token_rms.numel() else None
        ),
        "slot_count": _summary(valid_counts),
        "centroid_rms": _summary(centroid_rms),
        "coherence": _summary(coherence),
        "reference_value_rms": _summary(reference_value_rms),
        "centroid_value_rms": _summary(centroid_value_rms),
        "value_coherence": _summary(value_coherence),
        "value_log_coherence_per_log_count_slope": _slope(
            log_count, value_coherence.clamp_min(1e-12).log()
        ),
        "log_coherence_per_log_count_slope": _slope(
            log_count, log_coherence
        ),
        "effective_query_normalized_count_bias": _summary(query_rms),
        "state_len": state_len,
        "protected_slots": protected,
    }
    owner_parts = getattr(self, "_analysis_owner_parts", None)
    key_parts = getattr(self, "_analysis_key_parts", None)
    if owner_parts and key_parts:
        leaf_owner = torch.cat(owner_parts, dim=2).to(state_k.device)
        leaf_key = torch.cat(key_parts, dim=2).to(state_k.device).float()
        valid_leaf = leaf_owner.ge(protected) & leaf_owner.lt(state_len)
        safe_owner = leaf_owner.clamp(min=0, max=max(state_len - 1, 0))
        full_mean_key = (
            state_k.detach()[..., :state_len, :].float()
            / counts.detach()[..., :state_len, :].float().clamp_min(1)
        )
        assigned_centroid = full_mean_key.gather(
            2, safe_owner.unsqueeze(-1).expand(*safe_owner.shape, leaf_key.size(-1))
        )
        leaf_rms = leaf_key.square().mean(-1).sqrt().clamp_min(1e-12)
        assigned_rms = (
            assigned_centroid.square().mean(-1).sqrt().clamp_min(1e-12)
        )
        assignment_cosine = torch.nn.functional.cosine_similarity(
            leaf_key, assigned_centroid, dim=-1
        )[valid_leaf]
        assignment_angular_distance = 1.0 - assignment_cosine
        assignment_log_norm_distance = (
            leaf_rms.log() - assigned_rms.log()
        ).abs()[valid_leaf]
        record["assignment_cosine"] = _summary(assignment_cosine)
        record["assignment_angular_distance"] = _summary(
            assignment_angular_distance
        )
        record["assignment_log_norm_distance"] = _summary(
            assignment_log_norm_distance
        )
        record["mean_balanced_radial_bias"] = float(
            assignment_angular_distance.mean()
            / assignment_log_norm_distance.mean().clamp_min(1e-12)
        )
    rotary_dims = int(getattr(self, "_analysis_rotary_dims", mean_key.size(-1)))
    record["head_dim"] = int(mean_key.size(-1))
    record["rotary_dims"] = rotary_dims
    for name, begin, end in (
        ("rotary", 0, rotary_dims),
        ("unrotated", rotary_dims, int(mean_key.size(-1))),
    ):
        if end <= begin:
            continue
        part_mean = mean_key[..., begin:end]
        if local_k is not None and int(local_k.size(2)):
            part_reference = (
                local_k.detach()[..., begin:end]
                .float()
                .square()
                .mean(dim=(-2, -1))
                .sqrt()
            )
        else:
            part_reference = reference_rms
        part_centroid = part_mean.square().mean(-1).sqrt()
        part_coherence = part_centroid / part_reference.unsqueeze(-1).clamp_min(
            1e-12
        )
        record[f"{name}_coherence"] = _summary(part_coherence)
        record[f"{name}_log_coherence_per_log_count_slope"] = _slope(
            log_count, part_coherence.clamp_min(1e-12).log()
        )
    return record


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    _, sequence = select_sequences(
        tokenizer,
        args.dataset,
        args.sequence_length,
        samples=1,
        rank=0,
        world_size=1,
    )[0]
    model, acceleration = load_model(args.checkpoint, device)
    config = AutoConfig.from_pretrained(
        args.checkpoint, trust_remote_code=True
    ).get_text_config(decoder=True)
    install_hf_lod_attention(
        model,
        config=PagedLODConfig(
            chunk_size=256,
            local_window=512,
            state_growth_factor=16,
            state_min_size=256,
            protected_prefix=1,
            state_clustering_normalization=args.state_clustering_normalization,
            state_clustering_radial_bias=args.state_clustering_radial_bias,
            state_clustering_radial_scope=args.state_clustering_radial_scope,
            state_clustering_centroid_rescale=(
                args.state_clustering_centroid_rescale
            ),
            state_clustering_centroid_rescale_scope=(
                args.state_clustering_centroid_rescale_scope
            ),
            state_clustering_query_metric=args.state_clustering_query_metric,
            page_size=16,
        ),
        open_count=8,
        engine_backend="kernel",
    )

    compatible_modules = [
        module
        for module in model.modules()
        if getattr(module, "_hf_lod_settings", None) is not None
        and isinstance(getattr(module, "layer_idx", None), int)
    ]
    rotary_dims_by_engine = [
        _rotary_dims(config, module.layer_idx, int(module.head_dim))
        for module in compatible_modules
    ]
    original_build = hf_lod._build_engine
    built_engines = 0

    def recording_build(*build_args, **build_kwargs):
        nonlocal built_engines
        engine = original_build(*build_args, **build_kwargs)
        engine._analysis_rotary_dims = rotary_dims_by_engine[built_engines]
        built_engines += 1
        return engine

    original_route = TritonLODAttentionCore._route_top_slots
    original_new = TritonLODAttentionCore._new_page_cache
    original_append = TritonLODAttentionCore._append_page_cache

    def recording_new(self, *new_args, **new_kwargs):
        self._analysis_owner_parts = []
        self._analysis_key_parts = []
        return original_new(self, *new_args, **new_kwargs)

    def recording_append(self, cache, key, value, owners):
        self._analysis_owner_parts.append(owners.detach().cpu())
        self._analysis_key_parts.append(key.detach().cpu())
        return original_append(self, cache, key, value, owners)

    def recording_route(self, q, state_k, state_v, counts, **kwargs):
        if int(q.size(2)) > 1:
            self._analysis_geometry = _geometry_record(
                self,
                q,
                state_k,
                state_v,
                counts,
                kwargs.get("local_k"),
                kwargs.get("local_v"),
                state_len=int(kwargs["state_len"]),
            )
        return original_route(self, q, state_k, state_v, counts, **kwargs)

    hf_lod._build_engine = recording_build
    TritonLODAttentionCore._route_top_slots = recording_route
    TritonLODAttentionCore._new_page_cache = recording_new
    TritonLODAttentionCore._append_page_cache = recording_append
    try:
        with torch.inference_mode():
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        hf_lod._build_engine = original_build
        TritonLODAttentionCore._route_top_slots = original_route
        TritonLODAttentionCore._new_page_cache = original_new
        TritonLODAttentionCore._append_page_cache = original_append

    layers = []
    for module in model.modules():
        engine = getattr(module, "_hf_lod_transient_engine", None)
        layer_idx = getattr(module, "layer_idx", None)
        if engine is None or not isinstance(layer_idx, int):
            continue
        record = getattr(engine, "_analysis_geometry", None)
        if record is None:
            continue
        rotary_dims = _rotary_dims(config, layer_idx, int(record["head_dim"]))
        record = dict(record)
        record.update(
            layer=layer_idx,
            rotary_dims=rotary_dims,
            rotary_fraction=rotary_dims / int(record["head_dim"]),
            has_q_norm=hasattr(module, "q_norm"),
            has_k_norm=hasattr(module, "k_norm"),
            gqa_ratio=int(module.num_key_value_groups),
            attention_scale=float(module.scaling),
        )
        for name in ("q_norm", "k_norm"):
            norm = getattr(module, name, None)
            weight = getattr(norm, "weight", None)
            if isinstance(weight, torch.Tensor):
                gain = weight.detach().float()
                if getattr(config, "model_type", None) == "gemma3":
                    gain = gain + 1.0
                record[f"{name}_gain"] = _summary(gain)
        layers.append(record)

    def layer_mean(path: str) -> float:
        return float(sum(record[path]["mean"] for record in layers) / len(layers))

    payload = {
        "checkpoint": args.checkpoint,
        "sequence_length": args.sequence_length,
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
        "attention_layers": len(layers),
        "acceleration": acceleration,
        "architecture": {
            "head_dim": sorted({record["head_dim"] for record in layers}),
            "gqa_ratio": sorted({record["gqa_ratio"] for record in layers}),
            "q_norm": all(record["has_q_norm"] for record in layers),
            "k_norm": all(record["has_k_norm"] for record in layers),
            "rotary_fractions": sorted(
                {record["rotary_fraction"] for record in layers}
            ),
        },
        "all_layers": {
            "query_rms_mean": layer_mean("query_rms"),
            "reference_key_rms_mean": layer_mean("reference_key_rms"),
            "local_key_token_rms_mean": layer_mean("local_key_token_rms"),
            "coherence_mean": layer_mean("coherence"),
            "value_coherence_mean": layer_mean("value_coherence"),
            "assignment_cosine_mean": layer_mean("assignment_cosine"),
            "assignment_angular_distance_mean": layer_mean(
                "assignment_angular_distance"
            ),
            "assignment_log_norm_distance_mean": layer_mean(
                "assignment_log_norm_distance"
            ),
            "mean_balanced_radial_bias": float(
                sum(record["mean_balanced_radial_bias"] for record in layers)
                / len(layers)
            ),
            "log_coherence_per_log_count_slope_mean": float(
                sum(
                    record["log_coherence_per_log_count_slope"]
                    for record in layers
                )
                / len(layers)
            ),
        },
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["architecture"], sort_keys=True))
    print(json.dumps(payload["all_layers"], sort_keys=True))


if __name__ == "__main__":
    main()

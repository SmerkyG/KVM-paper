#!/usr/bin/env python3
"""Measure how NIAH answer tokens are partitioned into LOD slots and pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from model.hf_pytorch_lod_attention import install_hf_lod_attention
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_attention import TritonLODAttentionCore
from scripts.eval_hf_lod_lmeval import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--sample-indices", type=int, nargs="+", required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
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
            page_size=16,
        ),
        open_count=8,
        engine_backend="kernel",
    )

    original_new = TritonLODAttentionCore._new_page_cache
    original_append = TritonLODAttentionCore._append_page_cache

    def recording_new(self, *new_args, **new_kwargs):
        self._analysis_owner_parts = []
        return original_new(self, *new_args, **new_kwargs)

    def recording_append(self, cache, key, value, owners):
        self._analysis_owner_parts.append(owners.detach().cpu())
        return original_append(self, cache, key, value, owners)

    source = json.loads(args.samples.read_text())["samples"]["niah_single_3"]
    result_samples = []
    TritonLODAttentionCore._new_page_cache = recording_new
    TritonLODAttentionCore._append_page_cache = recording_append
    try:
        for sample_index in args.sample_indices:
            sample = source[sample_index]
            prompt = sample["arguments"][0][0]
            target = sample["target"][0]
            encoded = tokenizer(
                prompt, add_special_tokens=False, return_offsets_mapping=True
            )
            input_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            target_char_starts = [
                begin
                for begin in range(len(prompt))
                if prompt.startswith(target, begin)
            ]
            if len(target_char_starts) != 1:
                raise RuntimeError(
                    f"sample {sample_index} has {len(target_char_starts)} target matches"
                )
            target_char_begin = target_char_starts[0]
            target_char_end = target_char_begin + len(target)
            target_positions = [
                index
                for index, (begin, end) in enumerate(offsets)
                if end > target_char_begin and begin < target_char_end
            ]
            with torch.inference_mode():
                model(
                    input_ids=torch.tensor(input_ids, device=device).unsqueeze(0),
                    use_cache=False,
                )

            layer_records = []
            for module in model.modules():
                engine = getattr(module, "_hf_lod_transient_engine", None)
                layer = getattr(module, "layer_idx", None)
                if engine is None or not isinstance(layer, int):
                    continue
                owner_parts = getattr(engine, "_analysis_owner_parts", None)
                if not owner_parts:
                    continue
                owners = torch.cat(owner_parts, dim=2)
                sink_len = int(engine.sink_len) if engine.separate_sink_cache else 0
                head_records = []
                for head in range(int(owners.size(1))):
                    head_owners = owners[0, head]
                    token_records = []
                    for position in target_positions:
                        offset = position - sink_len
                        if not 0 <= offset < int(head_owners.numel()):
                            continue
                        slot = int(head_owners[offset])
                        slot_rank = int(head_owners[: offset + 1].eq(slot).sum()) - 1
                        token_records.append(
                            {
                                "position": position,
                                "slot": slot,
                                "page": slot_rank // 16,
                                "offset_in_page": slot_rank % 16,
                                "slot_length": int(head_owners.eq(slot).sum()),
                            }
                        )
                    slot_pages = {
                        (record["slot"], record["page"]) for record in token_records
                    }
                    slots = {record["slot"] for record in token_records}
                    pages_per_target_slot: dict[int, set[int]] = {}
                    for record in token_records:
                        pages_per_target_slot.setdefault(record["slot"], set()).add(
                            record["page"]
                        )
                    head_records.append(
                        {
                            "head": head,
                            "archived_target_tokens": len(token_records),
                            "distinct_target_slots": len(slots),
                            "distinct_target_slot_pages": len(slot_pages),
                            "max_target_pages_in_one_slot": max(
                                (len(pages) for pages in pages_per_target_slot.values()),
                                default=0,
                            ),
                            "tokens": token_records,
                        }
                    )
                layer_records.append({"layer": layer, "heads": head_records})
            result_samples.append(
                {
                    "sample_index": sample_index,
                    "target": target,
                    "prompt_tokens": len(input_ids),
                    "target_tokens": len(target_positions),
                    "target_begin": target_positions[0],
                    "layers": layer_records,
                }
            )
    finally:
        TritonLODAttentionCore._new_page_cache = original_new
        TritonLODAttentionCore._append_page_cache = original_append

    payload = {
        "checkpoint": args.checkpoint,
        "state_clustering_normalization": args.state_clustering_normalization,
        "state_clustering_radial_bias": args.state_clustering_radial_bias,
        "state_clustering_radial_scope": args.state_clustering_radial_scope,
        "state_clustering_centroid_rescale": (
            args.state_clustering_centroid_rescale
        ),
        "state_clustering_centroid_rescale_scope": (
            args.state_clustering_centroid_rescale_scope
        ),
        "acceleration": acceleration,
        "samples": result_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"samples": len(result_samples), "output": str(args.output)}))


if __name__ == "__main__":
    main()

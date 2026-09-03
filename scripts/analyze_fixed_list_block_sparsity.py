#!/usr/bin/env python3
"""Measure block sparsity of the fixed-index two-tier decode formulation.

The logical page-size-1 list is rebuilt at a state-update boundary and is
ordered as::

    protected sink, active local window, active coarse centroids,
    valid leaves in centroid-major order

For each real decode query, routes are unioned over the query heads sharing a
KV head.  Sink/local entries are enabled unconditionally, unopened centroids
remain enabled as coarse representatives, and leaves are enabled exactly when
their owner centroid is opened.  There are no allocated-but-unused leaf slots
in this analysis.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
from scripts.eval_vllm_lod_quality import select_niah_s3
from scripts.probe_qwen35_lod_niah import (
    load_text_model,
    require_qwen35_acceleration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--workload", choices=("prolong", "niah_s3"), required=True)
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=65_536)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--leaf-inline-pages-per-slot", type=int, default=128)
    parser.add_argument(
        "--max-open-leaves",
        type=int,
        default=1024,
        help="also report a policy that leaves centroids with >= this many leaves closed",
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenized prompt")
        value = value[0]
    return [int(token) for token in value]


def _find_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("empty sentinel tokenization")
    limit = len(values) - len(needle) + 1
    for index in range(max(limit, 0)):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def _chat_wrapper(tokenizer) -> tuple[list[int], list[int]]:
    """Extract and validate the token-level wrapper for one user message."""
    sentinel = "LOD_FIXED_LIST_CONTENT_SENTINEL_7A91E3"
    rendered = _token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    sentinel_ids = _token_ids(
        tokenizer(sentinel, add_special_tokens=False, return_attention_mask=False)
    )
    begin = _find_subsequence(rendered, sentinel_ids)
    if begin < 0:
        raise RuntimeError("could not isolate user content in the chat template")
    prefix = rendered[:begin]
    suffix = rendered[begin + len(sentinel_ids) :]

    # Prove that replacement at this boundary is exact rather than relying on
    # assumptions about how a particular tokenizer handles adjacent text.
    probe = "A short document used to validate chat-template token splicing."
    probe_ids = _token_ids(
        tokenizer(probe, add_special_tokens=False, return_attention_mask=False)
    )
    actual = _token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": probe}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if prefix + probe_ids + suffix != actual:
        raise RuntimeError("chat-template content cannot be replaced token-exactly")
    return prefix, suffix


def select_prolong_summary_prompts(
    tokenizer,
    dataset_name: str,
    length: int,
    samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build exact-length, two-real-document, chat-formatted summary prompts."""
    question = "\n\nPlease summarize the foregoing documents."
    divider = "\n\n--- End of document; next document follows. ---\n\n"
    prefix, suffix = _chat_wrapper(tokenizer)
    question_ids = _token_ids(
        tokenizer(question, add_special_tokens=False, return_attention_mask=False)
    )
    divider_ids = _token_ids(
        tokenizer(divider, add_special_tokens=False, return_attention_mask=False)
    )
    content_budget = length - len(prefix) - len(suffix) - len(question_ids)
    document_budget = content_budget - len(divider_ids)
    if document_budget < 2:
        raise ValueError("sequence length is too short for the requested prompt")
    left_budget = document_budget // 2
    right_budget = document_budget - left_budget

    dataset = load_dataset(dataset_name, split="train", streaming=True).shuffle(
        seed=42, buffer_size=1_000
    )
    documents: list[tuple[int, list[int]]] = []
    needed = samples * 2
    minimum = max(left_budget, right_budget)
    for dataset_index, document in enumerate(dataset):
        token_count = document.get("length")
        if token_count is not None and int(token_count) < minimum:
            continue
        ids = _token_ids(
            tokenizer(
                document["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=minimum,
                return_attention_mask=False,
            )
        )
        if len(ids) < minimum:
            continue
        documents.append((dataset_index, ids))
        if len(documents) == needed:
            break
    if len(documents) != needed:
        raise RuntimeError(f"found only {len(documents)} sufficiently long documents")

    prompts = []
    for sample in range(samples):
        left_index, left_ids = documents[2 * sample]
        right_index, right_ids = documents[2 * sample + 1]
        prompt_ids = (
            prefix
            + left_ids[:left_budget]
            + divider_ids
            + right_ids[:right_budget]
            + question_ids
            + suffix
        )
        if len(prompt_ids) != length:
            raise AssertionError("ProLong prompt did not reach its exact token budget")
        prompts.append(
            {
                "index": sample,
                "prompt_token_ids": prompt_ids,
                "target": None,
                "source_document_indices": [left_index, right_index],
            }
        )
    metadata = {
        "question": question.strip(),
        "chat_template": True,
        "thinking_disabled": True,
        "documents_per_prompt": 2,
        "chat_prefix_tokens": len(prefix),
        "chat_suffix_tokens": len(suffix),
        "question_tokens": len(question_ids),
        "document_divider_tokens": len(divider_ids),
    }
    return prompts, metadata


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in ("min", "p10", "p50", "p90", "max")}
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, (0.10, 0.50, 0.90))
    return {
        "min": float(array.min()),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


@dataclass
class BlockTotals:
    blocks: int = 0
    zero_blocks: int = 0
    full_blocks: int = 0
    partial_blocks: int = 0
    issued_lanes: int = 0
    active_entries: int = 0
    valid_entries: int = 0
    full_attention_entries: int = 0
    leaf_blocks: int = 0
    zero_leaf_blocks: int = 0
    nonzero_leaf_blocks: int = 0
    event_zero_fraction: list[float] = field(default_factory=list)
    event_issued_lanes: list[float] = field(default_factory=list)

    def add(
        self,
        active: np.ndarray,
        is_leaf: np.ndarray,
        *,
        block: int,
        full_attention_entries: int,
    ) -> None:
        valid = int(active.size)
        padded = math.ceil(valid / block) * block
        active_pad = np.pad(active, (0, padded - valid), constant_values=False)
        leaf_pad = np.pad(is_leaf, (0, padded - valid), constant_values=False)
        counts = active_pad.reshape(-1, block).sum(axis=1)
        leaf_chunks = leaf_pad.reshape(-1, block).any(axis=1)
        nonzero = counts > 0
        zero_count = int((~nonzero).sum())
        block_count = int(counts.size)
        issued = int(nonzero.sum()) * block
        self.blocks += block_count
        self.zero_blocks += zero_count
        self.full_blocks += int((counts == block).sum())
        self.partial_blocks += int(((counts > 0) & (counts < block)).sum())
        self.issued_lanes += issued
        self.active_entries += int(active.sum())
        self.valid_entries += valid
        self.full_attention_entries += full_attention_entries
        self.leaf_blocks += int(leaf_chunks.sum())
        self.zero_leaf_blocks += int((leaf_chunks & ~nonzero).sum())
        self.nonzero_leaf_blocks += int((leaf_chunks & nonzero).sum())
        self.event_zero_fraction.append(zero_count / block_count)
        self.event_issued_lanes.append(float(issued))

    def summary(self, block: int) -> dict[str, Any]:
        nonzero = self.blocks - self.zero_blocks
        return {
            "block_size": block,
            "blocks": self.blocks,
            "zero_blocks": self.zero_blocks,
            "nonzero_blocks": nonzero,
            "zero_block_fraction": self.zero_blocks / self.blocks,
            "full_block_fraction": self.full_blocks / self.blocks,
            "partial_block_fraction": self.partial_blocks / self.blocks,
            "active_entry_fraction": self.active_entries / self.valid_entries,
            "useful_lane_fraction_among_issued": (
                self.active_entries / self.issued_lanes if self.issued_lanes else 0.0
            ),
            "issued_lane_fraction_of_fixed_dense_scan": (
                self.issued_lanes / (self.blocks * block)
            ),
            "issued_lane_ratio_vs_full_attention": (
                self.issued_lanes / self.full_attention_entries
            ),
            "leaf_touching_blocks": self.leaf_blocks,
            "zero_leaf_touching_block_fraction": (
                self.zero_leaf_blocks / self.leaf_blocks if self.leaf_blocks else 0.0
            ),
            "event_zero_block_fraction": _percentiles(self.event_zero_fraction),
            "event_issued_lanes": _percentiles(self.event_issued_lanes),
        }


@dataclass
class PolicyTotals:
    name: str
    block_totals: dict[int, BlockTotals]
    events: int = 0
    fixed_entries: int = 0
    full_attention_entries: int = 0
    sink_entries: int = 0
    local_entries: int = 0
    coarse_entries: int = 0
    active_coarse_entries: int = 0
    leaf_entries: int = 0
    active_leaf_entries: int = 0
    opened_centroids: list[float] = field(default_factory=list)
    opened_leaves: list[float] = field(default_factory=list)

    @classmethod
    def create(cls, name: str, blocks: tuple[int, ...]) -> "PolicyTotals":
        return cls(name=name, block_totals={block: BlockTotals() for block in blocks})

    def add(
        self,
        *,
        active: np.ndarray,
        is_leaf: np.ndarray,
        sink_len: int,
        local_len: int,
        coarse_count: int,
        active_coarse: int,
        leaf_count: int,
        active_leaves: int,
        opened_count: int,
        full_attention_entries: int,
    ) -> None:
        self.events += 1
        self.fixed_entries += int(active.size)
        self.full_attention_entries += full_attention_entries
        self.sink_entries += sink_len
        self.local_entries += local_len
        self.coarse_entries += coarse_count
        self.active_coarse_entries += active_coarse
        self.leaf_entries += leaf_count
        self.active_leaf_entries += active_leaves
        self.opened_centroids.append(float(opened_count))
        self.opened_leaves.append(float(active_leaves))
        for block, totals in self.block_totals.items():
            totals.add(
                active,
                is_leaf,
                block=block,
                full_attention_entries=full_attention_entries,
            )

    def summary(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "mean_fixed_list_entries": self.fixed_entries / self.events,
            "mean_full_attention_entries": self.full_attention_entries / self.events,
            "mean_sink_entries": self.sink_entries / self.events,
            "mean_local_entries": self.local_entries / self.events,
            "mean_coarse_entries": self.coarse_entries / self.events,
            "mean_active_coarse_entries": self.active_coarse_entries / self.events,
            "mean_leaf_entries": self.leaf_entries / self.events,
            "mean_active_leaf_entries": self.active_leaf_entries / self.events,
            "opened_centroids": _percentiles(self.opened_centroids),
            "opened_leaves": _percentiles(self.opened_leaves),
            "blocks": {
                str(block): totals.summary(block)
                for block, totals in self.block_totals.items()
            },
        }


def _snapshot_module(module: Qwen3_5TwoLevelAttention) -> dict[str, Any]:
    state = getattr(module, "_lod_state", None)
    if not isinstance(state, dict):
        raise RuntimeError("LOD state was not initialized")
    page_cache = state.get("page_cache")
    if not isinstance(page_cache, dict):
        raise RuntimeError("paged leaf state was not initialized")
    slot_lengths = page_cache.get("slot_lengths")
    counts = state.get("counts")
    if not isinstance(slot_lengths, torch.Tensor) or not isinstance(counts, torch.Tensor):
        raise RuntimeError("LOD state lacks lengths or counts")
    state_len = int(state["state_len"])
    separate_sink = bool(module.separate_sink_cache)
    sink_len = min(int(module.sink_len), state_len)
    return {
        "slot_lengths": slot_lengths[..., :state_len].detach().cpu().clone(),
        "counts": counts[..., :state_len, 0].detach().cpu().clone(),
        "state_len": state_len,
        "recent_len": int(state["recent_len"]),
        "separate_sink_cache": separate_sink,
        # Conceptually expose the protected state singleton as the sink entry;
        # its coarse slot and leaf are then excluded from the other sections.
        "sink_len": sink_len,
    }


def _fixed_geometry(
    snapshot: dict[str, Any], batch_index: int, kv_head: int
) -> dict[str, Any]:
    lengths = snapshot["slot_lengths"][batch_index, kv_head].numpy().astype(np.int64)
    counts = snapshot["counts"][batch_index, kv_head].numpy()
    protected = 0 if snapshot["separate_sink_cache"] else int(snapshot["sink_len"])
    first_centroid = protected
    active_slots = np.flatnonzero(counts[first_centroid:] > 0.5) + first_centroid
    active_lengths = lengths[active_slots]
    leaf_owners = np.repeat(active_slots, active_lengths)
    sink_len = int(snapshot["sink_len"])
    local_len = int(snapshot["recent_len"])
    coarse_count = int(active_slots.size)
    prefix_count = sink_len + local_len
    fixed_size = prefix_count + coarse_count + int(leaf_owners.size)
    is_leaf = np.zeros(fixed_size, dtype=np.bool_)
    is_leaf[prefix_count + coarse_count :] = True
    return {
        "active_slots": active_slots,
        "active_lengths": active_lengths,
        "leaf_owners": leaf_owners,
        "sink_len": sink_len,
        "local_len": local_len,
        "coarse_count": coarse_count,
        "prefix_count": prefix_count,
        "is_leaf": is_leaf,
        "fixed_size": fixed_size,
        "full_attention_entries": sink_len + local_len + int(leaf_owners.size),
    }


def summarize_runs(
    runs: list[dict[str, Any]], max_open_leaves: int
) -> dict[str, Any]:
    blocks = (16, 64, 256)
    policies = {
        "top8_all": PolicyTotals.create("top8_all", blocks),
        f"top8_lt_{max_open_leaves}_leaves": PolicyTotals.create(
            f"top8_lt_{max_open_leaves}_leaves", blocks
        ),
    }
    layer_event_counts: defaultdict[int, int] = defaultdict(int)

    for run in runs:
        for layer in run["layers"]:
            snapshot = layer["snapshot"]
            routes = layer["top_slots"]
            slot_lengths = snapshot["slot_lengths"]
            batch, kv_heads, _ = slot_lengths.shape
            if not routes:
                raise RuntimeError("no routes were captured for a decode run")
            query_heads = int(routes[0].size(1))
            if query_heads % kv_heads:
                raise RuntimeError("query heads are not divisible by KV heads")
            gqa = query_heads // kv_heads
            geometries = {
                (batch_index, kv_head): _fixed_geometry(snapshot, batch_index, kv_head)
                for batch_index in range(batch)
                for kv_head in range(kv_heads)
            }
            for routed in routes:
                routed = routed[..., 0, :].cpu()
                for batch_index in range(batch):
                    for kv_head in range(kv_heads):
                        geometry = geometries[(batch_index, kv_head)]
                        group = routed[
                            batch_index, kv_head * gqa : (kv_head + 1) * gqa
                        ].reshape(-1)
                        selected = np.unique(
                            group[group >= 0].numpy().astype(np.int64, copy=False)
                        )
                        layer_event_counts[int(layer["model_layer_index"])] += 1
                        for policy_name, policy in policies.items():
                            opened = selected
                            if policy_name != "top8_all" and opened.size:
                                all_lengths = snapshot["slot_lengths"][
                                    batch_index, kv_head
                                ].numpy()
                                opened = opened[all_lengths[opened] < max_open_leaves]
                            state_open = np.zeros(snapshot["state_len"], dtype=np.bool_)
                            state_open[opened] = True
                            coarse_active = ~state_open[geometry["active_slots"]]
                            leaf_active = state_open[geometry["leaf_owners"]]
                            active = np.concatenate(
                                (
                                    np.ones(geometry["prefix_count"], dtype=np.bool_),
                                    coarse_active,
                                    leaf_active,
                                )
                            )
                            policy.add(
                                active=active,
                                is_leaf=geometry["is_leaf"],
                                sink_len=geometry["sink_len"],
                                local_len=geometry["local_len"],
                                coarse_count=geometry["coarse_count"],
                                active_coarse=int(coarse_active.sum()),
                                leaf_count=int(geometry["leaf_owners"].size),
                                active_leaves=int(leaf_active.sum()),
                                opened_count=int(opened.size),
                                full_attention_entries=geometry[
                                    "full_attention_entries"
                                ],
                            )

    return {
        "policies": {name: policy.summary() for name, policy in policies.items()},
        "layer_event_counts": {
            str(key): value for key, value in sorted(layer_event_counts.items())
        },
        "fixed_list_definition": [
            "protected sink entries",
            "active local-window entries",
            "active coarse-centroid entries",
            "valid leaf entries in centroid-major order",
        ],
        "mask_definition": {
            "sink_and_local": "always enabled",
            "coarse": "enabled iff its centroid is unopened",
            "leaf": "enabled iff its owner centroid is opened",
        },
    }


def _group_by_length(prompts: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        groups[len(prompt["prompt_token_ids"])].append(prompt)
    return [groups[length] for length in sorted(groups)]


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.steps <= 0:
        raise ValueError("samples and steps must be positive")
    if args.max_open_leaves <= 0:
        raise ValueError("maximum open leaf count must be positive")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if args.workload == "prolong":
        prompts, prompt_metadata = select_prolong_summary_prompts(
            tokenizer, args.dataset, args.sequence_length, args.samples
        )
    else:
        prompts = select_niah_s3(
            tokenizer,
            args.checkpoint,
            args.sequence_length,
            args.samples,
            sample_offset=args.sample_offset,
            apply_chat_template=True,
            disable_thinking=True,
        )
        prompt_metadata = {
            "chat_template": True,
            "thinking_disabled": True,
            "canonical_task": "RULER NIAH-S3",
        }
    for prompt in prompts:
        prompt["prompt_token_ids"] = _token_ids(prompt["prompt_token_ids"])

    model = load_text_model(
        args.checkpoint,
        "two_level",
        args.two_level_topk,
        args.state_growth_factor,
        device,
        "paged",
        require_fla_fast_path=True,
    )
    acceleration = require_qwen35_acceleration(model)
    modules = [
        module for module in model.modules() if isinstance(module, Qwen3_5TwoLevelAttention)
    ]
    if not modules:
        raise RuntimeError("Qwen3.5 LOD attention modules were not installed")
    for module in modules:
        module.leaf_layout = "query"
        module.leaf_inline_pages_per_slot = args.leaf_inline_pages_per_slot
        # Expose the same route result that the fused decoder normally consumes.
        module.fused_decode_state_route = False

    current_capture: list[list[torch.Tensor]] | None = None
    for layer_index, module in enumerate(modules):
        original = module._route_top_slots

        def captured_route(
            self,
            *method_args,
            __original=original,
            __layer_index=layer_index,
            **method_kwargs,
        ):
            routed = __original(*method_args, **method_kwargs)
            if int(method_args[0].size(2)) == 1 and current_capture is not None:
                current_capture[__layer_index].append(routed.detach().cpu().clone())
            return routed

        module._route_top_slots = types.MethodType(captured_route, module)

    runs: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    prompt_groups = _group_by_length(prompts)
    with torch.inference_mode():
        for group_index, prompt_group in enumerate(prompt_groups):
            prompt_length = len(prompt_group[0]["prompt_token_ids"])
            sequence = torch.tensor(
                [prompt["prompt_token_ids"] for prompt in prompt_group],
                dtype=torch.long,
                device=device,
            )
            current_capture = [[] for _ in modules]
            prefill = model(input_ids=sequence, use_cache=True, logits_to_keep=1)
            cache = prefill.past_key_values
            next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [[int(token)] for token in next_token[:, 0].tolist()]
            layers = []
            for module in modules:
                layers.append(
                    {
                        "model_layer_index": int(module.layer_idx),
                        "snapshot": _snapshot_module(module),
                    }
                )

            position = prompt_length
            output = None
            for _ in range(args.steps):
                output = model(
                    input_ids=next_token,
                    past_key_values=cache,
                    cache_position=torch.tensor([position], dtype=torch.long, device=device),
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.past_key_values
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                for row, token in zip(generated, next_token[:, 0].tolist()):
                    row.append(int(token))
                position += 1
            torch.cuda.synchronize(device)
            if output is None or not bool(torch.isfinite(output.logits).all().item()):
                raise RuntimeError("decode did not produce finite logits")
            for layer, captured in zip(layers, current_capture):
                layer["top_slots"] = captured
                if len(captured) != args.steps:
                    raise RuntimeError(
                        f"captured {len(captured)} routes, expected {args.steps}"
                    )
            runs.append({"prompt_length": prompt_length, "layers": layers})
            for prompt, token_ids in zip(prompt_group, generated):
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                target = prompt.get("target")
                answers.append(
                    {
                        "index": int(prompt["index"]),
                        "prompt_length": prompt_length,
                        "prompt_sha256": hashlib.sha256(
                            np.asarray(prompt["prompt_token_ids"], dtype=np.int32).tobytes()
                        ).hexdigest(),
                        "target": target,
                        "target_found": (
                            str(target).lower() in decoded.lower()
                            if target is not None
                            else None
                        ),
                        "generated_text": decoded,
                        "source_document_indices": prompt.get(
                            "source_document_indices"
                        ),
                    }
                )
            current_capture = None
            if group_index + 1 < len(prompt_groups):
                # NIAH samples differ by a handful of tokens and therefore
                # form several unpadded batches. Release the preceding cache
                # before constructing the next exact prompt-length group.
                prefill = cache = output = next_token = sequence = None
                for module in modules:
                    if hasattr(module, "_lod_state"):
                        delattr(module, "_lod_state")
                gc.collect()
                torch.cuda.empty_cache()

    block_summary = summarize_runs(runs, args.max_open_leaves)
    answers.sort(key=lambda item: item["index"])
    result = {
        "checkpoint": args.checkpoint,
        "workload": args.workload,
        "requested_sequence_length": args.sequence_length,
        "actual_prompt_lengths": _percentiles(
            [float(len(prompt["prompt_token_ids"])) for prompt in prompts]
        ),
        "samples": args.samples,
        "decode_route_steps": args.steps,
        "two_level_topk_per_query_head": args.two_level_topk,
        "state_growth_factor": args.state_growth_factor,
        "max_open_leaves_policy": args.max_open_leaves,
        "prompt_construction": prompt_metadata,
        "prompt_length_groups": [
            {"length": len(group[0]["prompt_token_ids"]), "batch_size": len(group)}
            for group in prompt_groups
        ],
        "qwen35_acceleration": acceleration,
        "niah_targets_found": (
            sum(bool(answer["target_found"]) for answer in answers)
            if args.workload == "niah_s3"
            else None
        ),
        **block_summary,
        "answers": answers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

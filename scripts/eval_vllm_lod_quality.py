#!/usr/bin/env python3
"""Evaluate native or LOD-backed vLLM on ProLong and NIAH-S3."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from vllm_engine_lifecycle import register_llm_shutdown, shutdown_registered_llms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=("full", "lod"), required=True)
    parser.add_argument("--eval", choices=("prolong", "niah_s3"), required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--allow-heterogeneous-global-config", action="store_true")
    parser.add_argument("--muse-native-text-config", action="store_true")
    parser.add_argument(
        "--concatenate-prolong",
        action="store_true",
        help="Concatenate distinct real ProLong documents to reach long contexts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--lod-prefill-route-block-m", type=int)
    parser.add_argument("--lod-prefill-route-num-warps", type=int)
    parser.add_argument("--lod-recursive-page-block-n", type=int)
    parser.add_argument(
        "--lod-recursive-state-route-backend",
        choices=("fused", "resplit"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_prolong_tokens(
    tokenizer,
    length: int,
    samples: int,
    *,
    concatenate: bool = False,
) -> tuple[list[list[int]], list[list[int]]]:
    if concatenate:
        # Streaming shuffle order can depend on which remote/cache shards are
        # visible.  Draw explicit row indices from the materialized dataset so
        # independent configurations always evaluate byte-identical prompts.
        dataset = load_dataset("Seerkfang/prolong-64k-512-new", split="train")
        generator = random.Random(42)
        used_indices: set[int] = set()

        def next_document() -> tuple[int, dict]:
            while True:
                index = generator.randrange(len(dataset))
                if index not in used_indices:
                    used_indices.add(index)
                    return index, dataset[index]

        separator = tokenizer(
            "\n\n--- NEXT PROLONG DOCUMENT ---\n\n",
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        selected = []
        selected_indices = []
        for _ in range(samples):
            token_ids: list[int] = []
            source_indices: list[int] = []
            while len(token_ids) < length:
                source_index, document = next_document()
                document_ids = tokenizer(
                    document["text"],
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                if not document_ids:
                    continue
                source_indices.append(source_index)
                if token_ids:
                    remaining = length - len(token_ids)
                    token_ids.extend(separator[: max(0, remaining - 1)])
                token_ids.extend(document_ids[: length - len(token_ids)])
            selected.append(token_ids)
            selected_indices.append(source_indices)
        return selected, selected_indices
    dataset = load_dataset(
        "Seerkfang/prolong-64k-512-new", split="train", streaming=True
    ).shuffle(seed=42, buffer_size=1_000)
    selected: list[list[int]] = []
    selected_indices: list[list[int]] = []
    for stream_index, document in enumerate(dataset):
        token_count = document.get("length")
        if token_count is not None and int(token_count) < length:
            continue
        token_ids = tokenizer(
            document["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=length,
            return_attention_mask=False,
        )["input_ids"]
        if len(token_ids) == length:
            selected.append(token_ids)
            selected_indices.append([stream_index])
        if len(selected) == samples:
            break
    if len(selected) != samples:
        raise RuntimeError(f"found only {len(selected)} sufficiently long documents")
    return selected, selected_indices


def select_niah_s3(
    tokenizer,
    checkpoint: str,
    length: int,
    samples: int,
    *,
    sample_offset: int = 0,
    apply_chat_template: bool = False,
    disable_thinking: bool = False,
) -> list[dict]:
    random.seed(0)
    np.random.seed(1234)
    try:
        from lm_eval.tasks.ruler.niah_utils import TEMPLATE
        from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack
    except ModuleNotFoundError as error:
        package_root_value = os.environ.get("LMEVAL_PACKAGE_ROOT")
        if not package_root_value:
            raise RuntimeError(
                "NIAH-S3 requires lm_eval, or LMEVAL_PACKAGE_ROOT must point "
                "to its lm_eval package directory"
            ) from error
        package_root = Path(package_root_value).resolve()
        if not (package_root / "tasks" / "ruler" / "niah_utils.py").is_file():
            raise RuntimeError(
                f"LMEVAL_PACKAGE_ROOT is not an lm_eval package: {package_root}"
            ) from error
        dependency_root = str(package_root.parent)
        if dependency_root not in sys.path:
            # Append rather than prepend so serving-environment builds of
            # torch, transformers, and vLLM keep precedence. This path only
            # supplies optional RULER helpers such as wonderwords and nltk.
            sys.path.append(dependency_root)
        optional_dependency_root = os.environ.get("LMEVAL_DEPENDENCY_ROOT")
        if optional_dependency_root:
            optional_dependency_root = str(
                Path(optional_dependency_root).resolve()
            )
            if optional_dependency_root not in sys.path:
                # Keep the serving environment authoritative for shared
                # packages such as torch and transformers. This path only
                # fills optional RULER dependencies absent from that env.
                sys.path.append(optional_dependency_root)
        # The NIAH generator itself has light dependencies, but importing it
        # normally executes lm_eval.tasks.__init__, which pulls the full eval
        # stack into the serving environment. Namespace packages load only the
        # three RULER modules needed to create prompts.
        for name in tuple(sys.modules):
            if name == "lm_eval" or name.startswith("lm_eval."):
                del sys.modules[name]
        for name, path in (
            ("lm_eval", package_root),
            ("lm_eval.tasks", package_root / "tasks"),
            ("lm_eval.tasks.ruler", package_root / "tasks" / "ruler"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        from lm_eval.tasks.ruler.niah_utils import TEMPLATE
        from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

    # Generate only the requested prefix of the canonical 500-example task.
    # This is sample-identical to taking the first ``samples`` records from an
    # isolated lm-eval run at this length, but avoids materializing 500 x 128K
    # tokens merely to evaluate 64 of them.
    documents = generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=length,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="uuids",
        num_samples=sample_offset + samples,
        TOKENIZER=tokenizer,
    )
    selected = []
    for document in documents[sample_offset:]:
        if apply_chat_template:
            # lm-eval represents gen_prefix as a partial assistant turn and
            # asks the chat template to continue that turn. Putting it inside
            # the user message changes instruction-tuned model behavior
            # substantially (especially Muse-Glimmer).
            encoded = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": document["input"]},
                    {"role": "assistant", "content": document["gen_prefix"]},
                ],
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                enable_thinking=not disable_thinking,
            )
            # Transformers 5 returns a BatchEncoding here, while older
            # releases returned the input-id list directly.
            token_ids = (
                encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            )
        else:
            prompt = document["input"] + " " + document["gen_prefix"]
            token_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        selected.append(
            {
                "index": int(document["index"]),
                "prompt_token_ids": token_ids,
                "target": str(document["outputs"][0]),
            }
        )
    return selected


def allow_heterogeneous_global_config(config):
    """Expose Gemma-4's nested text model as a heterogeneous causal LM."""
    text_config = getattr(config, "text_config", config)
    if not hasattr(text_config, "layer_types"):
        return config
    text_config.allow_global_per_layer_attribute_access = True
    full_layers = [
        text_config.per_layer_config[index]
        for index, layer_type in enumerate(text_config.layer_types)
        if layer_type == "full_attention"
    ]
    if full_layers:
        text_config.global_head_dim = max(
            int(layer.head_dim) for layer in full_layers
        )
        text_config.num_global_key_value_heads = min(
            int(layer.num_key_value_heads) for layer in full_layers
        )
    if getattr(text_config, "top_k", None) is None:
        top_k_experts = getattr(text_config, "top_k_experts", None)
        if top_k_experts is not None:
            text_config.top_k = int(top_k_experts)
    text_config.architectures = ["Gemma4ForCausalLM"]
    return text_config


def muse_native_text_config(config):
    """Select the Muse text tower registered by the LOD vLLM plugin."""
    text_config = getattr(config, "text_config", config)
    text_config.architectures = ["MuseGlimmerForCausalLM"]
    return text_config


def make_llm(args: argparse.Namespace):
    from vllm import LLM

    kwargs = {
        "model": args.checkpoint,
        "load_format": os.getenv("VLLM_WEIGHT_CACHE_LOAD_FORMAT", "ipc_cache"),
        "dtype": "bfloat16",
        "max_model_len": args.length + args.max_new_tokens,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": (
            args.max_num_batched_tokens
            or max(args.length, args.batch_size * 2048)
        ),
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if args.mode == "lod":
        kwargs["attention_config"] = {"backend": "CUSTOM"}
    if args.allow_heterogeneous_global_config:
        kwargs["hf_overrides"] = allow_heterogeneous_global_config
    elif args.muse_native_text_config:
        kwargs["hf_overrides"] = muse_native_text_config
    return register_llm_shutdown(LLM(**kwargs))


def inspect_lod_model(model) -> dict[str, object]:
    """Run in the vLLM worker and prove that LOD pools are attached."""
    custom_layers = []
    eligible_layers = []
    pooled_layers = []
    install_count = 0
    direct_prefill_calls = 0
    decode_calls = 0
    catch_up_batches = 0
    catch_up_rows = 0
    batched_cached_prefills = 0
    batched_cached_prefill_rows = 0
    routing_geometries = set()
    recursive_state_route_backends = set()
    prefill_coarse_direct_gqa_configured = set()
    prefill_coarse_direct_gqa_executed = set()
    state_v_abs_max = 0.0
    state_v_nonfinite = 0
    decode_gqa_union_configured = set()
    decode_gqa_union_requested = set()
    decode_gqa_union_score_only = set()
    decode_gqa_union_predicted_mass = set()
    decode_gqa_union_eligible = set()
    decode_gqa_union_hip_configured = set()
    decode_gqa_union_hip_executed = set()
    decode_gqa_union_aiter_final = set()
    decode_gqa_staged_fixed_configured = set()
    decode_gqa_staged_fixed_executed = set()
    decode_gqa_fixed_mask_configured = set()
    decode_gqa_fixed_mask_executed = set()
    decode_gqa_direct_fixed_routes_configured = set()
    decode_gqa_direct_fixed_routes_executed = set()
    decode_centroid_major_hip_configured = set()
    decode_centroid_major_hip_executed = set()
    decode_gqa_static_leaf_aiter_configured = set()
    decode_gqa_static_leaf_aiter_executed = set()
    decode_gqa_static_leaf_caps = set()
    decode_max_open_leaves = set()
    decode_route_cohort = set()
    prefill_route_cohort = set()
    prefill_open_counts = set()
    effective_decode_route_leaf_limits = set()
    decode_open_counts = set()
    static_total_centroids = 0
    static_opened_centroids = 0
    static_total_leaves = 0.0
    static_opened_leaves = 0.0
    static_bin_bounds = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    static_bin_centroids = [0 for _ in range(len(static_bin_bounds) + 1)]
    static_bin_leaves = [0.0 for _ in range(len(static_bin_bounds) + 1)]
    state_count_max = 0.0
    state_split_limits = set()
    split_state_len_max = 0
    split_scheduled_state_len_max = 0
    split_posting_len_max = 0
    state_centroids_at_or_above_open_limit = 0
    selected_route_count_max = 0.0
    gqa_union_epoch_max = 0
    gqa_union_nonempty_sequences = 0
    gqa_union_sequence_count = 0
    gqa_union_centroid_count_sum = 0
    gqa_union_centroid_count_max = 0
    gqa_union_token_count_sum = 0
    gqa_union_token_count_max = 0
    gqa_union_exact_token_count_sum = 0
    gqa_union_exact_token_count_max = 0
    gqa_union_effective_segments = set()
    gqa_union_split_d_reduce = set()
    gqa_union_runtime_sequence_counts = set()
    gqa_union_runtime_kv_heads = set()
    gqa_route_max_mass_values = []
    gqa_route_mass_threshold_hits = {
        denominator: 0 for denominator in (16, 32, 64, 128, 256)
    }
    custom_impl_sliding_windows: dict[str, int] = {}
    for name, module in model.named_modules():
        impl = getattr(module, "impl", None)
        if type(impl).__module__ != "vllm_lod_plugin.backend":
            continue
        custom_layers.append(name)
        impl_window = tuple(int(value) for value in impl.sliding_window)
        window_key = str(impl_window)
        custom_impl_sliding_windows[window_key] = (
            custom_impl_sliding_windows.get(window_key, 0) + 1
        )
        if bool(getattr(impl, "lod_eligible", False)):
            eligible_layers.append(name)
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is not None:
            pooled_layers.append(name)
            install_count += int(pool.install_count)
            direct_prefill_calls += int(pool.direct_prefill_calls)
            decode_calls += int(pool.decode_calls)
            catch_up_batches += int(pool.catch_up_batches)
            catch_up_rows += int(pool.catch_up_rows)
            batched_cached_prefills += int(pool.batched_cached_prefill_calls)
            batched_cached_prefill_rows += int(pool.batched_cached_prefill_rows)
            engine = pool.engine
            state_split_limits.add(
                getattr(pool.settings, "state_split_max_leaves", None)
            )
            split_state_len_max = max(
                split_state_len_max, int(pool.split_state_len_max)
            )
            split_scheduled_state_len_max = max(
                split_scheduled_state_len_max,
                int(pool.split_scheduled_state_len_max),
            )
            split_posting_len_max = max(
                split_posting_len_max, int(pool.split_posting_len_max)
            )
            state_v = pool.state["state_v"]
            state_v_abs_max = max(
                state_v_abs_max,
                float(torch.nan_to_num(state_v).abs().max().item()),
            )
            state_v_nonfinite += int((~torch.isfinite(state_v)).sum().item())
            recursive_state_route_backends.add(
                str(engine.recursive_state_route_backend)
            )
            prefill_coarse_direct_gqa_configured.add(
                (
                    bool(getattr(engine, "prefill_coarse_direct_gqa", False)),
                    int(getattr(engine, "prefill_coarse_max_grouped_rows", 0)),
                    int(getattr(engine, "prefill_coarse_route_block_n", 0)),
                    int(getattr(engine, "prefill_coarse_route_num_warps", 0)),
                )
            )
            executed_direct_gqa = getattr(
                engine, "_lod_prefill_coarse_direct_gqa_executed", None
            )
            if executed_direct_gqa is not None:
                prefill_coarse_direct_gqa_executed.add(
                    tuple(int(value) for value in executed_direct_gqa)
                )
            decode_gqa_union_configured.add(
                bool(getattr(pool.settings, "decode_gqa_union", False))
            )
            decode_gqa_union_hip_configured.add(
                bool(getattr(pool.settings, "decode_gqa_union_hip", False))
            )
            decode_gqa_staged_fixed_configured.add(
                bool(
                    getattr(
                        pool.settings,
                        "decode_gqa_staged_fixed_aiter",
                        False,
                    )
                )
            )
            decode_gqa_fixed_mask_configured.add(
                bool(
                    getattr(
                        pool.settings,
                        "decode_gqa_fixed_mask_aiter",
                        False,
                    )
                )
            )
            decode_gqa_direct_fixed_routes_configured.add(
                bool(
                    getattr(
                        pool.settings,
                        "decode_gqa_fixed_mask_direct_routes",
                        False,
                    )
                )
            )
            decode_centroid_major_hip_configured.add(
                bool(
                    getattr(
                        pool.settings,
                        "decode_centroid_major_hip",
                        False,
                    )
                )
            )
            decode_gqa_static_leaf_aiter_configured.add(
                bool(
                    getattr(
                        pool.settings,
                        "decode_gqa_static_leaf_aiter",
                        False,
                    )
                )
            )
            open_limit = getattr(pool.settings, "decode_max_open_leaves", None)
            decode_max_open_leaves.add(open_limit)
            decode_route_cohort.add(
                bool(getattr(pool.settings, "decode_route_cohort", False))
            )
            prefill_route_cohort.add(
                bool(getattr(pool.settings, "prefill_route_cohort", False))
            )
            prefill_open_counts.add(int(engine.prefill_two_level_topk))
            effective_open_limit = pool._decode_route_leaf_limit()
            effective_decode_route_leaf_limits.add(effective_open_limit)
            decode_open_counts.add(int(pool.settings.open_count))
            static_leaf_cap = getattr(
                pool.settings, "decode_gqa_static_leaf_cap", None
            )
            cap_snapshots = getattr(
                pool, "_static_leaf_cap_value_snapshots", None
            )
            if static_leaf_cap is not None:
                decode_gqa_static_leaf_caps.add(int(static_leaf_cap))
            elif isinstance(cap_snapshots, dict) and cap_snapshots:
                decode_gqa_static_leaf_caps.update(
                    int(value) for value in cap_snapshots.values()
                )
            else:
                decode_gqa_static_leaf_caps.add(None)
            state_counts = pool.state["counts"][..., 0]
            state_count_max = max(
                state_count_max, float(state_counts.max().item())
            )
            if effective_open_limit is not None:
                state_centroids_at_or_above_open_limit += int(
                    (state_counts >= int(effective_open_limit)).sum().item()
                )
            if static_leaf_cap is not None or (
                isinstance(cap_snapshots, dict) and cap_snapshots
            ):
                snapshots = getattr(
                    pool, "_static_leaf_cap_count_snapshots", None
                )
                if isinstance(snapshots, dict) and snapshots:
                    count_cap_pairs = [
                        (
                            leaf_counts,
                            int(static_leaf_cap)
                            if static_leaf_cap is not None
                            else int(cap_snapshots[slot]),
                        )
                        for slot, leaf_counts in snapshots.items()
                        if static_leaf_cap is not None
                        or slot in cap_snapshots
                    ]
                else:
                    page_cache = pool.state.get("page_cache", {})
                    archived_lengths = (
                        page_cache.get("slot_lengths")
                        if isinstance(page_cache, dict)
                        else None
                    )
                    leaf_counts = (
                        archived_lengths
                        if isinstance(archived_lengths, torch.Tensor)
                        else state_counts
                    )
                    count_cap_pairs = (
                        [(leaf_counts, int(static_leaf_cap))]
                        if static_leaf_cap is not None
                        else []
                    )
                # Request release clears live centroid counts, while the fixed
                # posting-list metadata remains available until that cache row
                # is reused. Use the latter to report the quality run's actual
                # exact-leaf distribution.
                for leaf_counts, effective_cap in count_cap_pairs:
                    active_counts = leaf_counts[leaf_counts > 0].float()
                    static_total_centroids += int(active_counts.numel())
                    static_opened_centroids += int(
                        (active_counts <= effective_cap).sum().item()
                    )
                    static_total_leaves += float(active_counts.sum().item())
                    static_opened_leaves += float(
                        active_counts[
                            active_counts <= effective_cap
                        ].sum().item()
                    )
                    lower = 0
                    for index, upper in enumerate(static_bin_bounds):
                        in_bin = (active_counts > lower) & (
                            active_counts <= upper
                        )
                        static_bin_centroids[index] += int(in_bin.sum().item())
                        static_bin_leaves[index] += float(
                            active_counts[in_bin].sum().item()
                        )
                        lower = upper
                    in_bin = active_counts > static_bin_bounds[-1]
                    static_bin_centroids[-1] += int(in_bin.sum().item())
                    static_bin_leaves[-1] += float(
                        active_counts[in_bin].sum().item()
                    )
            decode_storage = pool.decode_buffer_storage
            if isinstance(decode_storage, dict):
                execution_geometry = decode_storage.get(
                    "gqa_union_fixed_execution_geometry"
                )
                if isinstance(execution_geometry, torch.Tensor):
                    gqa_union_effective_segments.add(
                        int(execution_geometry[0].item())
                    )
                    gqa_union_split_d_reduce.add(
                        bool(execution_geometry[1].item())
                    )
                    if execution_geometry.numel() >= 4:
                        gqa_union_runtime_sequence_counts.add(
                            int(execution_geometry[2].item())
                        )
                        gqa_union_runtime_kv_heads.add(
                            int(execution_geometry[3].item())
                        )
                staged_marker = decode_storage.get("gqa_union_destinations")
                if bool(
                    getattr(
                        pool.settings,
                        "decode_gqa_staged_fixed_aiter",
                        False,
                    )
                ) and isinstance(staged_marker, torch.Tensor):
                    decode_gqa_staged_fixed_executed.add(
                        bool(int(staged_marker.reshape(-1)[0].item()) == 1)
                    )
                if bool(
                    getattr(
                        pool.settings,
                        "decode_gqa_fixed_mask_aiter",
                        False,
                    )
                ) and isinstance(staged_marker, torch.Tensor):
                    decode_gqa_fixed_mask_executed.add(
                        bool(int(staged_marker.reshape(-1)[0].item()) == 2)
                    )
                epochs = decode_storage.get("gqa_union_epochs")
                union_counts = decode_storage.get("gqa_union_counts")
                if bool(
                    getattr(pool.settings, "decode_gqa_predicted_mass", False)
                    and getattr(
                        pool.settings,
                        "decode_gqa_fixed_mask_aiter",
                        False,
                    )
                ):
                    # This path clears the live queue in its final reducer so
                    # the next token starts without a reset launch. The saved
                    # previous queue is the route that actually produced the
                    # current output and is therefore the diagnostic source.
                    previous_union_counts = decode_storage.get(
                        "gqa_union_fixed_previous_counts"
                    )
                    if isinstance(previous_union_counts, torch.Tensor):
                        union_counts = previous_union_counts
                union_token_counts = decode_storage.get("gqa_union_token_counts")
                exact_token_counts = decode_storage.get(
                    "gqa_union_hip_context_lens"
                )
                if isinstance(epochs, torch.Tensor):
                    gqa_union_epoch_max = max(
                        gqa_union_epoch_max, int(epochs.max().item())
                    )
                if isinstance(union_token_counts, torch.Tensor):
                    arena_final = bool(
                        getattr(pool.settings, "decode_gqa_union_hip", False)
                        and isinstance(
                            pool.state["page_cache"].get("unified_page1_k"),
                            torch.Tensor,
                        )
                    )
                    combined_token_counts = union_token_counts
                    if arena_final and isinstance(exact_token_counts, torch.Tensor):
                        # The arena call appends local and unopened coarse rows
                        # directly to this AITER context length. It is already
                        # the full final scan length, not a second branch to add.
                        combined_token_counts = exact_token_counts
                        gqa_union_exact_token_count_sum += int(
                            union_token_counts.sum().item()
                        )
                        gqa_union_exact_token_count_max = max(
                            gqa_union_exact_token_count_max,
                            int(union_token_counts.max().item()),
                        )
                    elif isinstance(exact_token_counts, torch.Tensor):
                        combined_token_counts = (
                            combined_token_counts + exact_token_counts
                        )
                        gqa_union_exact_token_count_sum += int(
                            exact_token_counts.sum().item()
                        )
                        gqa_union_exact_token_count_max = max(
                            gqa_union_exact_token_count_max,
                            int(exact_token_counts.max().item()),
                        )
                    gqa_union_sequence_count += int(combined_token_counts.numel())
                    gqa_union_nonempty_sequences += int(
                        (combined_token_counts > 0).sum().item()
                    )
                    gqa_union_token_count_sum += int(
                        combined_token_counts.sum().item()
                    )
                    gqa_union_token_count_max = max(
                        gqa_union_token_count_max,
                        int(combined_token_counts.max().item()),
                    )
                if isinstance(union_counts, torch.Tensor):
                    gqa_union_centroid_count_sum += int(union_counts.sum().item())
                    gqa_union_centroid_count_max = max(
                        gqa_union_centroid_count_max,
                        int(union_counts.max().item()),
                    )
                route_scores = decode_storage.get("route_state_scores")
                route_lse = decode_storage.get("route_full_lse")
                if isinstance(route_scores, torch.Tensor) and isinstance(
                    route_lse, torch.Tensor
                ):
                    route_mass = torch.exp(
                        route_scores.float() - route_lse.float().unsqueeze(-1)
                    )
                    finite_mass = torch.nan_to_num(
                        route_mass, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    gqa_route_max_mass_values.extend(
                        finite_mass.amax(dim=-1).reshape(-1).cpu().tolist()
                    )
                    for denominator in gqa_route_mass_threshold_hits:
                        gqa_route_mass_threshold_hits[denominator] += int(
                            (finite_mass > (1.0 / denominator)).sum().item()
                        )
            for decode_buffers in pool.decode_buffers.values():
                if "gqa_union_last_static_cap_page1" in decode_buffers:
                    decode_gqa_static_leaf_aiter_executed.add(
                        bool(decode_buffers["gqa_union_last_static_cap_page1"])
                    )
                epochs = decode_buffers.get("gqa_union_epochs")
                if "gqa_union_last_requested" in decode_buffers:
                    decode_gqa_union_requested.add(
                        bool(decode_buffers["gqa_union_last_requested"])
                    )
                    decode_gqa_union_score_only.add(
                        bool(decode_buffers["gqa_union_last_score_only"])
                    )
                    decode_gqa_union_predicted_mass.add(
                        bool(
                            decode_buffers.get(
                                "gqa_union_last_predicted_mass", False
                            )
                        )
                    )
                    decode_gqa_union_eligible.add(
                        bool(decode_buffers["gqa_union_last_eligible"])
                    )
                    decode_gqa_union_hip_executed.add(
                        bool(decode_buffers.get("gqa_union_last_hip", False))
                    )
                    decode_gqa_union_aiter_final.add(
                        bool(
                            decode_buffers.get(
                                "gqa_union_last_aiter_final", False
                            )
                        )
                    )
                    if "gqa_union_last_direct_fixed_routes" in decode_buffers:
                        decode_gqa_direct_fixed_routes_executed.add(
                            bool(
                                decode_buffers[
                                    "gqa_union_last_direct_fixed_routes"
                                ]
                            )
                        )
                    if "route_last_centroid_major_hip" in decode_buffers:
                        decode_centroid_major_hip_executed.add(
                            bool(
                                decode_buffers[
                                    "route_last_centroid_major_hip"
                                ]
                            )
                        )
                top_slots = decode_buffers.get("route_top_slots")
                if isinstance(top_slots, torch.Tensor):
                    rows = int(top_slots.size(0))
                    cache_rows = pool.active_indices[:rows].long()
                    row_counts = state_counts.index_select(0, cache_rows)
                    kv_group = int(pool.query_heads // pool.kv_heads)
                    kv_for_query = (
                        torch.arange(
                            pool.query_heads,
                            device=top_slots.device,
                            dtype=torch.long,
                        )
                        // kv_group
                    )
                    query_counts = row_counts[:, kv_for_query, :]
                    slots = top_slots[:, :, 0, :].long()
                    valid = (slots >= 0) & (slots < int(state_counts.size(2)))
                    selected = torch.gather(
                        query_counts,
                        2,
                        slots.clamp(0, int(state_counts.size(2)) - 1),
                    )
                    if bool(valid.any().item()):
                        selected_route_count_max = max(
                            selected_route_count_max,
                            float(selected[valid].max().item()),
                        )
            routing_geometries.add(
                (
                    str(engine.state_clustering_normalization),
                    str(engine.state_clustering_centroid_rescale),
                    str(engine.routing_normalization),
                )
            )
    return {
        "custom_layers": custom_layers,
        "custom_impl_sliding_windows": custom_impl_sliding_windows,
        "eligible_layers": eligible_layers,
        "pooled_layers": pooled_layers,
        "install_count": install_count,
        "direct_prefill_calls": direct_prefill_calls,
        "decode_calls": decode_calls,
        "catch_up_batches": catch_up_batches,
        "catch_up_rows": catch_up_rows,
        "batched_cached_prefills": batched_cached_prefills,
        "batched_cached_prefill_rows": batched_cached_prefill_rows,
        "routing_geometries": sorted(routing_geometries),
        "recursive_state_route_backends": sorted(
            recursive_state_route_backends
        ),
        "prefill_coarse_direct_gqa_configured": sorted(
            prefill_coarse_direct_gqa_configured
        ),
        "prefill_coarse_direct_gqa_executed": sorted(
            prefill_coarse_direct_gqa_executed
        ),
        "decode_gqa_union_configured": sorted(decode_gqa_union_configured),
        "decode_gqa_union_requested": sorted(decode_gqa_union_requested),
        "decode_gqa_union_score_only": sorted(decode_gqa_union_score_only),
        "decode_gqa_union_predicted_mass": sorted(
            decode_gqa_union_predicted_mass
        ),
        "decode_gqa_union_eligible": sorted(decode_gqa_union_eligible),
        "decode_gqa_union_hip_configured": sorted(
            decode_gqa_union_hip_configured
        ),
        "decode_gqa_union_hip_executed": sorted(
            decode_gqa_union_hip_executed
        ),
        "decode_gqa_union_aiter_final": sorted(
            decode_gqa_union_aiter_final
        ),
        "decode_gqa_staged_fixed_configured": sorted(
            decode_gqa_staged_fixed_configured
        ),
        "decode_gqa_staged_fixed_executed": sorted(
            decode_gqa_staged_fixed_executed
        ),
        "decode_gqa_fixed_mask_configured": sorted(
            decode_gqa_fixed_mask_configured
        ),
        "decode_gqa_fixed_mask_executed": sorted(
            decode_gqa_fixed_mask_executed
        ),
        "decode_gqa_direct_fixed_routes_configured": sorted(
            decode_gqa_direct_fixed_routes_configured
        ),
        "decode_gqa_direct_fixed_routes_executed": sorted(
            decode_gqa_direct_fixed_routes_executed
        ),
        "decode_centroid_major_hip_configured": sorted(
            decode_centroid_major_hip_configured
        ),
        "decode_centroid_major_hip_executed": sorted(
            decode_centroid_major_hip_executed
        ),
        "gqa_union_effective_segments": sorted(
            gqa_union_effective_segments
        ),
        "gqa_union_split_d_reduce": sorted(gqa_union_split_d_reduce),
        "gqa_union_runtime_sequence_counts": sorted(
            gqa_union_runtime_sequence_counts
        ),
        "gqa_union_runtime_kv_heads": sorted(gqa_union_runtime_kv_heads),
        "decode_gqa_static_leaf_aiter_configured": sorted(
            decode_gqa_static_leaf_aiter_configured
        ),
        "decode_gqa_static_leaf_aiter_executed": sorted(
            decode_gqa_static_leaf_aiter_executed
        ),
        "decode_gqa_static_leaf_caps": sorted(
            decode_gqa_static_leaf_caps,
            key=lambda value: -1 if value is None else int(value),
        ),
        "static_total_centroids": static_total_centroids,
        "static_total_leaves": static_total_leaves,
        "static_opened_centroids": static_opened_centroids,
        "static_opened_leaves": static_opened_leaves,
        "static_opened_centroid_fraction": (
            static_opened_centroids / max(1, static_total_centroids)
        ),
        "static_opened_leaf_fraction": (
            static_opened_leaves / max(1.0, static_total_leaves)
        ),
        "static_leaf_count_distribution": [
            {
                "leaf_count": (
                    str(upper)
                    if index < 2
                    else f"{static_bin_bounds[index - 1] + 1}-{upper}"
                ),
                "centroid_fraction": centroids / max(1, static_total_centroids),
                "leaf_fraction": leaves / max(1.0, static_total_leaves),
            }
            for index, (upper, centroids, leaves) in enumerate(
                zip(
                    static_bin_bounds,
                    static_bin_centroids[:-1],
                    static_bin_leaves[:-1],
                )
            )
        ]
        + [
            {
                "leaf_count": f">{static_bin_bounds[-1]}",
                "centroid_fraction": static_bin_centroids[-1]
                / max(1, static_total_centroids),
                "leaf_fraction": static_bin_leaves[-1]
                / max(1.0, static_total_leaves),
            }
        ],
        "decode_max_open_leaves": sorted(
            decode_max_open_leaves,
            key=lambda value: -1 if value is None else int(value),
        ),
        "decode_route_cohort": sorted(decode_route_cohort),
        "prefill_route_cohort": sorted(prefill_route_cohort),
        "prefill_open_counts": sorted(prefill_open_counts),
        "effective_decode_route_leaf_limits": sorted(
            effective_decode_route_leaf_limits,
            key=lambda value: -1 if value is None else int(value),
        ),
        "decode_open_counts": sorted(decode_open_counts),
        "state_count_max": state_count_max,
        "state_split_limits": sorted(
            state_split_limits,
            key=lambda value: -1 if value is None else int(value),
        ),
        "split_state_len_max": split_state_len_max,
        "split_scheduled_state_len_max": split_scheduled_state_len_max,
        "split_posting_len_max": split_posting_len_max,
        "state_centroids_at_or_above_open_limit": (
            state_centroids_at_or_above_open_limit
        ),
        "selected_route_count_max": selected_route_count_max,
        # Unlike the configuration booleans above, these values are written by
        # the device union/list kernels. Nonzero values prove that the shared-
        # union execution path actually ran under vLLM (including graph replay).
        "gqa_union_epoch_max": gqa_union_epoch_max,
        "gqa_union_nonempty_sequences": gqa_union_nonempty_sequences,
        "gqa_union_sequence_count": gqa_union_sequence_count,
        "gqa_union_centroid_count_mean": (
            gqa_union_centroid_count_sum / max(1, gqa_union_sequence_count)
        ),
        "gqa_union_centroid_count_max": gqa_union_centroid_count_max,
        "gqa_union_token_count_mean": (
            gqa_union_token_count_sum / max(1, gqa_union_sequence_count)
        ),
        "gqa_union_token_count_max": gqa_union_token_count_max,
        "gqa_union_exact_token_count_mean": (
            gqa_union_exact_token_count_sum / max(1, gqa_union_sequence_count)
        ),
        "gqa_union_exact_token_count_max": gqa_union_exact_token_count_max,
        "gqa_route_max_mass": (
            {
                "minimum": min(gqa_route_max_mass_values),
                "mean": sum(gqa_route_max_mass_values)
                / len(gqa_route_max_mass_values),
                "maximum": max(gqa_route_max_mass_values),
            }
            if gqa_route_max_mass_values
            else None
        ),
        "gqa_route_mass_threshold_hits": gqa_route_mass_threshold_hits,
        "state_v_abs_max": state_v_abs_max,
        "state_v_nonfinite": state_v_nonfinite,
    }


def reset_lod_decode_execution_markers(model) -> int:
    """Reset graph-visible union markers before a measured vLLM decode.

    Clearing the epoch and its stamp table together is important: clearing
    only the epoch could make a later graph replay mistake an old stamp for a
    hit in the new epoch.  A nonzero marker after this reset is device-written
    evidence from the measured generate call, rather than from graph capture,
    warmup, or a different request geometry.
    """
    reset = 0
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        for decode_buffers in pool.decode_buffers.values():
            staged_marker = decode_buffers.get("gqa_union_destinations")
            if isinstance(staged_marker, torch.Tensor):
                staged_marker.reshape(-1)[0].zero_()
            epochs = decode_buffers.get("gqa_union_epochs")
            stamps = decode_buffers.get("gqa_union_seen_stamps")
            token_counts = decode_buffers.get("gqa_union_token_counts")
            exact_token_counts = decode_buffers.get(
                "gqa_union_hip_context_lens"
            )
            launch_lens = decode_buffers.get("gqa_union_hip_launch_lens")
            union_counts = decode_buffers.get("gqa_union_counts")
            execution_geometry = decode_buffers.get(
                "gqa_union_fixed_execution_geometry"
            )
            if not isinstance(epochs, torch.Tensor):
                continue
            epochs.zero_()
            if isinstance(stamps, torch.Tensor):
                stamps.zero_()
            if isinstance(token_counts, torch.Tensor):
                token_counts.zero_()
            if isinstance(exact_token_counts, torch.Tensor):
                exact_token_counts.zero_()
            if isinstance(launch_lens, torch.Tensor):
                launch_lens.zero_()
            if isinstance(union_counts, torch.Tensor):
                union_counts.zero_()
            if isinstance(execution_geometry, torch.Tensor):
                execution_geometry.zero_()
            reset += 1
    return reset


def configure_lod_model(
    model,
    *,
    route_block_m: int | None,
    route_num_warps: int | None,
    page_block_n: int | None,
) -> int:
    """Apply evaluation-only kernel ablations to installed LOD engines."""
    configured = 0
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        if route_block_m is not None:
            pool.engine.prefill_route_block_m = route_block_m
        if route_num_warps is not None:
            pool.engine.prefill_route_num_warps = route_num_warps
        if page_block_n is not None:
            pool.engine.recursive_page_block_n = page_block_n
        configured += 1
    return configured


def evaluate_prolong(args: argparse.Namespace, tokenizer, llm) -> dict:
    from vllm import SamplingParams

    sequences, source_indices = select_prolong_tokens(
        tokenizer,
        args.length,
        args.samples,
        concatenate=args.concatenate_prolong,
    )
    prompts = [{"prompt_token_ids": token_ids} for token_ids in sequences]
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
        flat_logprobs=True,
        detokenize=False,
    )
    started = time.perf_counter()
    outputs = []
    for begin in range(0, len(prompts), args.batch_size):
        outputs.extend(
            llm.generate(
                prompts[begin : begin + args.batch_size],
                params,
                use_tqdm=True,
            )
        )
    elapsed = time.perf_counter() - started
    sample_records = []
    total_nll = 0.0
    total_tokens = 0
    for sample, (token_ids, output, sample_sources) in enumerate(
        zip(sequences, outputs, source_indices)
    ):
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) != len(token_ids):
            raise RuntimeError("vLLM returned incomplete prompt log probabilities")
        nll = 0.0
        for token_id, candidates in zip(token_ids[1:], prompt_logprobs[1:]):
            if candidates is None or token_id not in candidates:
                raise RuntimeError("target token missing from vLLM prompt logprobs")
            nll -= float(candidates[token_id].logprob)
        prediction_tokens = len(token_ids) - 1
        total_nll += nll
        total_tokens += prediction_tokens
        sample_records.append(
            {
                "sample": sample,
                "tokens": len(token_ids),
                "source_indices": sample_sources,
                "token_sha256": hashlib.sha256(
                    np.asarray(token_ids, dtype="<u4").tobytes()
                ).hexdigest(),
                "loss": nll / prediction_tokens,
                "perplexity": math.exp(nll / prediction_tokens),
            }
        )
    loss = total_nll / total_tokens
    return {
        "eval": "prolong",
        "loss": loss,
        "perplexity": math.exp(loss),
        "prediction_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "samples": sample_records,
        "attention_exercised": (
            "direct_lod_prefill"
            if args.mode == "lod"
            and os.environ.get("VLLM_LOD_PREFILL_MODE", "direct") == "direct"
            else "native_prefill"
        ),
    }


def evaluate_niah_s3(args: argparse.Namespace, tokenizer, llm) -> dict:
    from vllm import SamplingParams

    documents = select_niah_s3(
        tokenizer,
        args.checkpoint,
        args.length,
        args.samples,
        apply_chat_template=args.apply_chat_template,
        disable_thinking=args.disable_thinking,
    )
    prompts = [
        {"prompt_token_ids": document["prompt_token_ids"]} for document in documents
    ]
    params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        detokenize=True,
    )
    started = time.perf_counter()
    # Keep each submission within the authoritative LOD pool. Passing all 64
    # prompts to vLLM at once lets the scheduler admit replacement requests as
    # earlier rows finish, which can evict semantic cache rows still needed by
    # in-flight decode and spuriously requests an impossible native fallback.
    outputs = []
    for begin in range(0, len(prompts), args.batch_size):
        outputs.extend(
            llm.generate(
                prompts[begin : begin + args.batch_size],
                params,
                use_tqdm=True,
            )
        )
    elapsed = time.perf_counter() - started
    records = []
    for document, output in zip(documents, outputs):
        response = output.outputs[0].text
        target = document["target"]
        records.append(
            {
                "index": document["index"],
                "input_tokens": len(document["prompt_token_ids"]),
                "target": target,
                "response": response,
                "exact": target.lower() in response.lower(),
            }
        )
    correct = sum(record["exact"] for record in records)
    return {
        "eval": "niah_s3",
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "elapsed_seconds": elapsed,
        "samples": records,
        "attention_exercised": (
            (
                "direct_lod_prefill_and_decode"
                if os.environ.get("VLLM_LOD_PREFILL_MODE", "direct") == "direct"
                else "native_prefill_then_lod_decode"
            )
            if args.mode == "lod"
            else "native_prefill_and_decode"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.lod_recursive_state_route_backend is not None:
        # The worker allocates graph-stable scratch while constructing its LOD
        # pools, so select the route before vLLM starts rather than mutating a
        # live engine after graph capture.
        os.environ["VLLM_LOD_RECURSIVE_STATE_ROUTE_BACKEND"] = (
            args.lod_recursive_state_route_backend
        )
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch size must be positive")
    if args.mode == "lod" and args.batch_size > int(
        os.environ.get("VLLM_LOD_POOL_SIZE", "8")
    ):
        raise ValueError("batch size exceeds VLLM_LOD_POOL_SIZE")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    llm = make_llm(args)
    tuning_requested = any(
        value is not None
        for value in (
            args.lod_prefill_route_block_m,
            args.lod_prefill_route_num_warps,
            args.lod_recursive_page_block_n,
        )
    )
    if args.mode == "lod" and tuning_requested:
        configured = llm.apply_model(
            functools.partial(
                configure_lod_model,
                route_block_m=args.lod_prefill_route_block_m,
                route_num_warps=args.lod_prefill_route_num_warps,
                page_block_n=args.lod_recursive_page_block_n,
            )
        )
        if not configured or not all(value > 0 for value in configured):
            raise RuntimeError("LOD evaluation tuning found no installed layers")
    lod_diagnostics_before = None
    if args.mode == "lod":
        diagnostics = llm.apply_model(inspect_lod_model)
        lod_diagnostics_before = diagnostics[0]
        if not lod_diagnostics_before["custom_layers"]:
            raise RuntimeError("vLLM did not construct any CUSTOM attention layers")
        if (
            lod_diagnostics_before["pooled_layers"]
            != lod_diagnostics_before["eligible_layers"]
        ):
            raise RuntimeError(
                "LOD pools are not attached to every CUSTOM attention layer: "
                f"{lod_diagnostics_before}"
            )
    if args.eval == "prolong":
        result = evaluate_prolong(args, tokenizer, llm)
    else:
        result = evaluate_niah_s3(args, tokenizer, llm)
    lod_diagnostics_after = None
    if args.mode == "lod":
        lod_diagnostics_after = llm.apply_model(inspect_lod_model)[0]
        if (
            os.environ.get("VLLM_LOD_PREFILL_MODE", "direct") == "direct"
            and lod_diagnostics_after["direct_prefill_calls"]
            <= lod_diagnostics_before["direct_prefill_calls"]
        ):
            raise RuntimeError("evaluation completed without invoking direct LOD prefill")
        if args.eval == "niah_s3":
            if not lod_diagnostics_after["decode_calls"]:
                raise RuntimeError("NIAH-S3 completed without invoking LOD decode")
            if (
                lod_diagnostics_after["install_count"]
                <= lod_diagnostics_before["install_count"]
            ):
                raise RuntimeError(
                    "NIAH-S3 completed without installing an LOD cache"
                )
    result.update(
        checkpoint=args.checkpoint,
        mode=args.mode,
        length=args.length,
        requested_samples=args.samples,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        enforce_eager=args.enforce_eager,
        apply_chat_template=args.apply_chat_template,
        disable_thinking=args.disable_thinking,
        allow_heterogeneous_global_config=(
            args.allow_heterogeneous_global_config
        ),
        muse_native_text_config=args.muse_native_text_config,
        concatenate_prolong=args.concatenate_prolong,
        static_cohort_never_readmit=(
            os.environ.get("VLLM_LOD_STATIC_COHORT_NEVER_READMIT", "0") == "1"
            if args.mode == "lod"
            else None
        ),
        static_leaf_cap_divisor=(
            int(os.environ.get("VLLM_LOD_STATIC_LEAF_CAP_DIVISOR", "16"))
            if args.mode == "lod"
            else None
        ),
        prefill_static_leaf_cap_min=(
            int(os.environ.get("VLLM_LOD_PREFILL_STATIC_LEAF_CAP_MIN", "16"))
            if args.mode == "lod"
            else None
        ),
        prefill_route_cohort=(
            os.environ.get("VLLM_LOD_PREFILL_ROUTE_COHORT", "0") == "1"
            if args.mode == "lod"
            else None
        ),
        prefill_open_count=(
            int(
                os.environ.get(
                    "VLLM_LOD_PREFILL_OPEN_COUNT",
                    str(min(3, int(os.environ.get("VLLM_LOD_OPEN_COUNT", "8")))),
                )
            )
            if args.mode == "lod"
            else None
        ),
        state_split_max_leaves=(
            int(os.environ["VLLM_LOD_STATE_SPLIT_MAX_LEAVES"])
            if args.mode == "lod"
            and os.environ.get("VLLM_LOD_STATE_SPLIT_MAX_LEAVES")
            else None
        ),
        decode_static_leaf_cap_min=(
            int(
                os.environ.get(
                    "VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP_MIN", "16"
                )
            )
            if args.mode == "lod"
            else None
        ),
        lod_prefill_route_block_m=args.lod_prefill_route_block_m,
        lod_prefill_route_num_warps=args.lod_prefill_route_num_warps,
        lod_recursive_page_block_n=args.lod_recursive_page_block_n,
        lod_recursive_state_route_backend=(
            args.lod_recursive_state_route_backend
            if args.mode == "lod"
            else None
        ),
        lod_diagnostics_before=lod_diagnostics_before,
        lod_diagnostics_after=lod_diagnostics_after,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}))


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_registered_llms()

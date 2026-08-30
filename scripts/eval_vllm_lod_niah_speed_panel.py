#!/usr/bin/env python3
"""Run a multi-length NIAH-S3 quality and warm speed panel with one model load."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer

from benchmark_vllm_lod_speed import (
    configure_lod_model,
    inspect_attention_memory,
    inspect_full_attention_dispatch,
    inspect_lod_dispatch,
    inspect_lod_model as inspect_lod_speed_model,
    install_full_attention_timers,
    install_lod_phase_timers,
    install_lod_total_timers,
    summarize_full_attention_timers,
    summarize_lod_phase_timers,
    summarize_lod_total_timers,
    timed_generate,
)
from eval_vllm_lod_quality import (
    inspect_lod_model,
    reset_lod_decode_execution_markers,
    select_niah_s3,
)
from vllm_engine_lifecycle import register_llm_shutdown, shutdown_registered_llms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("full", "lod"), required=True)
    parser.add_argument(
        "--lengths",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[8192, 16384, 32768, 65536, 131072],
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--speed-decode-tokens", type=int, default=64)
    parser.add_argument(
        "--speed-prompt-reserve",
        type=int,
        default=0,
        help=(
            "Subtract this many tokens from each ProLong speed prompt. This is "
            "useful at a checkpoint's advertised context boundary, where the "
            "generated decode tokens must also fit within max_position_embeddings."
        ),
    )
    parser.add_argument("--speed-repeats", type=int, default=3)
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help=(
            "Disable vLLM's custom all-reduce and use the distributed backend. "
            "This is required on ROCm configurations where custom_all_reduce_hip "
            "rejects the TP topology during graph-memory profiling."
        ),
    )
    parser.add_argument(
        "--full-attention-backend", default="ROCM_AITER_UNIFIED_ATTN"
    )
    parser.add_argument("--allow-heterogeneous-global-config", action="store_true")
    parser.add_argument("--muse-native-text-config", action="store_true")
    parser.add_argument("--model-impl")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--speed-use-warm-prefix-cache",
        action="store_true",
        help=(
            "Keep the speed prompt in the prefix cache after warmup. This is "
            "a decode-throughput diagnostic: it synchronizes a large request "
            "batch at decode instead of letting long chunked prefills admit "
            "requests in waves."
        ),
    )
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--speed-only", action="store_true")
    parser.add_argument("--profile-lod-phases", action="store_true")
    parser.add_argument("--profile-lod-total", action="store_true")
    parser.add_argument("--profile-full-attention", action="store_true")
    parser.add_argument("--lod-decode-route-group-size", type=int)
    parser.add_argument("--lod-decode-route-num-warps", type=int)
    parser.add_argument("--lod-decode-route-reduce-num-warps", type=int)
    parser.add_argument("--torch-profile-dir", type=Path)
    parser.add_argument("--torch-profile-delay-iterations", type=int, default=0)
    parser.add_argument("--torch-profile-max-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _token_block_diagnostics(token_ids: list[int], block_size: int = 16) -> dict:
    blocks = [
        tuple(token_ids[begin : begin + block_size])
        for begin in range(0, len(token_ids) - block_size + 1, block_size)
    ]
    frequencies: dict[tuple[int, ...], int] = {}
    for block in blocks:
        frequencies[block] = frequencies.get(block, 0) + 1
    unique = len(frequencies)
    return {
        "block_size": block_size,
        "blocks": len(blocks),
        "unique_blocks": unique,
        "unique_block_ratio": unique / max(1, len(blocks)),
        "max_identical_block_occurrences": max(frequencies.values(), default=0),
    }


def make_speed_prompts(
    tokenizer, length: int, batch_size: int, *, streaming: bool = True
) -> tuple[list[dict], dict]:
    """Build exact-length speed prompts from distinct ProLong documents.

    A document is never repeated to fill a request. If one document is too
    short, subsequent shuffled documents are concatenated with a visible
    separator. Documents consumed by one batch row are not reused by another.
    """
    from datasets import load_dataset

    dataset_name = "Seerkfang/prolong-64k-512-new"
    dataset = load_dataset(dataset_name, split="train", streaming=streaming)
    shuffled = (
        dataset.shuffle(seed=20260824, buffer_size=1_000)
        if streaming
        else dataset.shuffle(seed=20260824)
    )
    documents = iter(shuffled)
    separator = tokenizer(
        "\n\n--- NEXT PROLONG DOCUMENT ---\n\n", add_special_tokens=False
    )["input_ids"]
    prompts: list[dict] = []
    records: list[dict] = []
    rejected_records: list[dict] = []
    stream_index = -1
    request_index = 0
    while request_index < batch_size:
        token_ids: list[int] = []
        source_indices: list[int] = []
        source_descriptions: list[dict] = []
        while len(token_ids) < length:
            try:
                document = next(documents)
            except StopIteration as error:
                raise RuntimeError(
                    "ProLong stream ended before all speed prompts were filled"
                ) from error
            stream_index += 1
            document_tokens = tokenizer(
                document["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            if not document_tokens:
                continue
            if token_ids:
                remaining = length - len(token_ids)
                # Preserve at least one token from the next real document;
                # never finish a prompt using only the separator.
                token_ids.extend(separator[: max(0, remaining - 1)])
            remaining = length - len(token_ids)
            token_ids.extend(document_tokens[:remaining])
            source_indices.append(stream_index)
            description = {"stream_index": stream_index}
            for key in ("id", "title", "source", "length"):
                value = document.get(key)
                if value is not None:
                    description[key] = str(value)
            source_descriptions.append(description)
        if len(token_ids) != length:
            raise AssertionError("ProLong speed prompt has the wrong token length")
        digest = hashlib.sha256(
            ",".join(str(token_id) for token_id in token_ids).encode()
        ).hexdigest()
        preview = re.sub(
            r"\s+", " ", tokenizer.decode(token_ids[:128], skip_special_tokens=True)
        ).strip()
        block_diagnostics = _token_block_diagnostics(token_ids)
        # Skip naturally repetitive source documents instead of weakening the
        # guard or aborting the whole matched panel. Consumed documents remain
        # consumed, so accepted batch rows are still distinct and unrepeated.
        if block_diagnostics["unique_block_ratio"] < 0.95:
            rejected_records.append(
                {
                    "prospective_request_index": request_index,
                    "source_documents": source_descriptions,
                    "token_blocks": block_diagnostics,
                    "reason": "unique_block_ratio_below_0.95",
                }
            )
            continue
        prompts.append({"prompt_token_ids": token_ids})
        records.append(
            {
                "request_index": request_index,
                "sha256": digest,
                "source_document_count": len(source_indices),
                "source_documents": source_descriptions,
                "decoded_prefix": preview,
                "token_blocks": block_diagnostics,
            }
        )
        request_index += 1
    hashes = [record["sha256"] for record in records]
    return prompts, {
        "dataset": dataset_name,
        "construction": "distinct shuffled documents, concatenated without repetition",
        "prompt_length": length,
        "all_prompt_hashes_unique": len(set(hashes)) == len(hashes),
        "rejected_candidate_count": len(rejected_records),
        "rejected_candidates": rejected_records,
        "prompts": records,
    }


def allow_heterogeneous_global_config(config):
    """Expose Gemma-4's nested text model as a heterogeneous causal LM."""
    text_config = getattr(config, "text_config", config)
    # vLLM probes hf_overrides once with a skeletal PreTrainedConfig merely to
    # discover whether the override changes model_type.
    if not hasattr(text_config, "layer_types"):
        return config
    text_config.allow_global_per_layer_attribute_access = True
    # Transformers 5 serializes the wide global-attention dimensions only in
    # per_layer_config.  This vLLM Gemma-4 implementation predates that schema
    # and reads the equivalent legacy global attributes.
    full_layers = [
        text_config.per_layer_config[index]
        for index, layer_type in enumerate(text_config.layer_types)
        if layer_type == "full_attention"
    ]
    if full_layers:
        text_config.global_head_dim = max(int(layer.head_dim) for layer in full_layers)
        text_config.num_global_key_value_heads = min(
            int(layer.num_key_value_heads) for layer in full_layers
        )
    # vLLM's generic Transformers MoE adapter currently probes ``top_k``;
    # Gemma-4 names the same architectural quantity ``top_k_experts``.
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


def evaluate_quality(args, tokenizer, llm, length: int) -> dict:
    from vllm import SamplingParams

    documents = select_niah_s3(
        tokenizer,
        args.checkpoint,
        length,
        args.samples,
        sample_offset=args.sample_offset,
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
    outputs = []
    started = time.perf_counter()
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
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "elapsed_seconds": elapsed,
        "input_tokens": sum(record["input_tokens"] for record in records),
        "samples": records,
    }


def evaluate_speed(args, tokenizer, llm, length: int) -> dict:
    from vllm import SamplingParams

    diagnostic_external_empty = os.getenv(
        "VLLM_LOD_DIAGNOSTIC_EXTERNAL_EMPTY_ATTENTION"
    ) in ("skip", "eligible")
    prompt_length = length - args.speed_prompt_reserve
    if prompt_length < 1:
        raise ValueError("speed-prompt-reserve must be smaller than each length")
    prompts, prompt_corpus = make_speed_prompts(
        tokenizer, prompt_length, args.batch_size
    )
    params = SamplingParams(
        temperature=0,
        max_tokens=args.speed_decode_tokens,
        detokenize=False,
        ignore_eos=True,
    )
    warmup = timed_generate(
        llm,
        prompts,
        params,
        return_token_ids=args.enable_prefix_caching,
    )
    warmup_token_ids = warmup[3] if args.enable_prefix_caching else None
    prefix_probe = None
    if args.enable_prefix_caching:
        before = (
            llm.apply_model(inspect_lod_speed_model)[0]
            if args.mode == "lod" and not diagnostic_external_empty
            else None
        )
        elapsed, prefill_elapsed, decode_elapsed, prefix_token_ids = timed_generate(
            llm, prompts, params, return_token_ids=True
        )
        if warmup_token_ids is None:
            raise AssertionError("prefix-output comparison lost the cold tokens")
        matching_prefix_lengths = [
            next(
                (
                    index
                    for index, (cold, cached) in enumerate(
                        zip(cold_row, cached_row)
                    )
                    if cold != cached
                ),
                min(len(cold_row), len(cached_row)),
            )
            for cold_row, cached_row in zip(
                warmup_token_ids, prefix_token_ids, strict=True
            )
        ]
        matching_requests = sum(
            cold_row == cached_row
            for cold_row, cached_row in zip(
                warmup_token_ids, prefix_token_ids, strict=True
            )
        )
        after = (
            llm.apply_model(inspect_lod_speed_model)[0]
            if args.mode == "lod" and not diagnostic_external_empty
            else None
        )
        prefix_probe = {
            "elapsed_seconds": elapsed,
            "prefill_seconds": prefill_elapsed,
            "decode_seconds": decode_elapsed,
            # Greedy output is a useful sensitivity diagnostic, but not a
            # cache-correctness assertion on this backend: Muse full attention
            # itself can choose a different token after a cached-prefix resume.
            # Exact input-token verification and retained-row reuse below are
            # the authoritative cache checks.
            "outputs_match_cold": matching_requests == len(prefix_token_ids),
            "matching_output_requests": matching_requests,
            "output_requests": len(prefix_token_ids),
            "matching_output_prefix_tokens": matching_prefix_lengths,
        }
        if before is not None and after is not None:
            reused = int(after["retained_prefix_reuses"]) - int(
                before["retained_prefix_reuses"]
            )
            prefix_probe["retained_lod_row_reuses"] = reused
            if reused <= 0:
                raise RuntimeError(
                    "vLLM reported prefix caching but no retained LOD row was "
                    f"reused: before={before!r} after={after!r}"
                )
    if args.mode == "lod":
        # Persistent vLLM compile caches do not reliably include out-of-tree
        # attention dispatch in their cache key.  Reset device-written markers
        # after warmup so the measured call proves which decode graph ran.
        llm.apply_model(reset_lod_decode_execution_markers)
    if args.profile_lod_phases:
        installed = llm.apply_model(install_lod_phase_timers)
        if not installed or not all(value > 0 for value in installed):
            raise RuntimeError("LOD phase profiler found no installed layers")
    if args.profile_lod_total:
        installed = llm.apply_model(install_lod_total_timers)
        if not installed or not all(value > 0 for value in installed):
            raise RuntimeError("LOD total profiler found no installed layers")
    if args.profile_full_attention:
        installed = llm.apply_model(install_full_attention_timers)
        if not installed or not all(value > 0 for value in installed):
            raise RuntimeError("full-attention profiler found no global layers")
    if args.torch_profile_dir is not None:
        llm.start_profile("attention_kernel_audit")
    prefills = []
    decodes = []
    totals = []
    for _ in range(args.speed_repeats):
        # Keep the ordinary speed result a cold-prefill measurement even when
        # the separate repeated-prefix probe above is enabled.
        if (
            args.enable_prefix_caching
            and not args.speed_use_warm_prefix_cache
            and not llm.reset_prefix_cache()
        ):
            raise RuntimeError("vLLM refused to reset an idle prefix cache")
        elapsed, prefill_elapsed, decode_elapsed = timed_generate(
            llm, prompts, params
        )
        totals.append(elapsed)
        prefills.append(prefill_elapsed)
        decodes.append(decode_elapsed)
    if args.torch_profile_dir is not None:
        llm.stop_profile()
    prompt_tokens = prompt_length * args.batch_size
    decode_steps = args.speed_decode_tokens - 1
    decode_tokens = args.batch_size * decode_steps
    prefill = statistics.median(prefills)
    decode = statistics.median(decodes)
    result = {
        "prompt_length": prompt_length,
        "prefill_seconds": prefill,
        "prefill_prompt_tokens_per_second": prompt_tokens / prefill,
        "marginal_decode_ms_per_batch_step": (
            1000.0 * decode / decode_steps if decode_steps else None
        ),
        "marginal_decode_tokens_per_second": (
            decode_tokens / decode if decode_steps and decode > 0.0 else None
        ),
        "prefill_timings_seconds": prefills,
        "decode_timings_seconds": decodes,
        "total_timings_seconds": totals,
        "prompt_corpus": prompt_corpus,
    }
    if prefix_probe is not None:
        result["prefix_cache_probe"] = prefix_probe
    if args.mode == "lod":
        execution_audit = llm.apply_model(inspect_lod_model)[0]
        result["lod_decode_execution_audit"] = {
            key: execution_audit[key]
            for key in (
                "decode_gqa_union_configured",
                "decode_gqa_union_requested",
                "decode_gqa_union_score_only",
                "decode_gqa_union_predicted_mass",
                "decode_gqa_union_eligible",
                "decode_gqa_union_hip_configured",
                "decode_gqa_union_hip_executed",
                "decode_centroid_major_hip_configured",
                "decode_centroid_major_hip_executed",
                "decode_gqa_union_aiter_final",
                "decode_gqa_staged_fixed_configured",
                "decode_gqa_staged_fixed_executed",
                "decode_gqa_fixed_mask_configured",
                "decode_gqa_fixed_mask_executed",
                "decode_gqa_overlap_local_sink_configured",
                "decode_gqa_overlap_local_sink_executed",
                "decode_gqa_direct_fixed_routes_configured",
                "decode_gqa_direct_fixed_routes_executed",
                "gqa_union_effective_segments",
                "gqa_union_split_d_reduce",
                "gqa_union_runtime_sequence_counts",
                "gqa_union_runtime_kv_heads",
                "decode_gqa_static_leaf_aiter_configured",
                "decode_gqa_static_leaf_aiter_executed",
                "decode_max_open_leaves",
                "decode_route_cohort",
                "effective_decode_route_leaf_limits",
                "decode_open_counts",
                "gqa_union_epoch_max",
                "gqa_union_nonempty_sequences",
                "gqa_union_sequence_count",
                "gqa_union_centroid_count_mean",
                "gqa_union_centroid_count_max",
                "gqa_union_token_count_mean",
                "gqa_union_token_count_max",
                "gqa_union_exact_token_count_mean",
                "gqa_union_exact_token_count_max",
            )
        }
        if (
            execution_audit["decode_gqa_union_eligible"] == [True]
            and (
                execution_audit["gqa_union_epoch_max"] <= 0
                or execution_audit["gqa_union_nonempty_sequences"] <= 0
            )
            ):
            raise RuntimeError(
                "the measured vLLM decode replayed a graph that did not run "
                "the configured GQA-union kernels; disable or invalidate the "
                "vLLM compile cache"
            )
        if (
            execution_audit["decode_centroid_major_hip_configured"] == [True]
            and execution_audit["decode_centroid_major_hip_executed"] != [True]
        ):
            raise RuntimeError(
                "centroid-major HIP routing was configured but its "
                "device-written execution marker was not observed"
            )
        if (
            execution_audit["decode_gqa_staged_fixed_configured"] == [True]
            and execution_audit["decode_gqa_staged_fixed_executed"] != [True]
        ):
            raise RuntimeError(
                "the measured vLLM decode did not execute the configured "
                "early-fixed staged AITER path"
            )
        if (
            execution_audit["decode_gqa_fixed_mask_configured"] == [True]
            and execution_audit["decode_gqa_fixed_mask_executed"] != [True]
        ):
            raise RuntimeError(
                "the measured vLLM decode did not execute the configured "
                "fixed-list masked AITER path"
            )
        if (
            execution_audit["decode_gqa_overlap_local_sink_configured"]
            == [True]
            and execution_audit["decode_gqa_overlap_local_sink_executed"]
            != [True]
        ):
            raise RuntimeError(
                "the measured vLLM decode did not execute the configured "
                "local/sink overlap path"
            )
        if (
            execution_audit["decode_gqa_static_leaf_aiter_configured"] == [True]
            and True
            not in execution_audit["decode_gqa_static_leaf_aiter_executed"]
        ):
            raise RuntimeError(
                "the measured vLLM decode did not execute the compact static "
                "page-size-one AITER path"
            )
        if (
            execution_audit["decode_gqa_union_hip_configured"] == [True]
            and execution_audit["decode_gqa_static_leaf_aiter_configured"]
            != [True]
            and execution_audit["decode_gqa_union_hip_executed"] != [True]
        ):
            raise RuntimeError(
                "the measured vLLM decode did not execute the configured "
                "AITER HIP page-size-one exact-leaf path"
            )
        if (
            execution_audit["decode_gqa_union_eligible"] == [True]
            and execution_audit["decode_gqa_union_hip_configured"] == [True]
            and execution_audit["decode_gqa_union_aiter_final"] != [True]
        ):
            raise RuntimeError(
                "the measured GQA-union path did not run the unified AITER "
                "leaves/local/coarse final attention"
            )
    if args.profile_lod_phases:
        result["lod_phase_profile"] = llm.apply_model(
            summarize_lod_phase_timers
        )[0]
    if args.profile_lod_total:
        result["lod_total_profile"] = llm.apply_model(
            summarize_lod_total_timers
        )[0]
    if args.profile_full_attention:
        result["full_attention_profile"] = llm.apply_model(
            summarize_full_attention_timers
        )[0]
    return result


def main() -> None:
    args = parse_args()
    if not args.lengths or min(args.lengths) < 2:
        raise ValueError("lengths must contain positive context lengths")
    if args.samples < 1 or args.samples % args.batch_size:
        raise ValueError("samples must be a positive multiple of batch size")
    if args.speed_use_warm_prefix_cache and not args.enable_prefix_caching:
        raise ValueError(
            "--speed-use-warm-prefix-cache requires --enable-prefix-caching"
        )
    if args.sample_offset < 0:
        raise ValueError("sample-offset must be non-negative")
    if args.quality_only and args.speed_only:
        raise ValueError("quality-only and speed-only are mutually exclusive")
    if args.profile_lod_phases and args.mode != "lod":
        raise ValueError("LOD phase profiling requires --mode lod")
    if args.profile_lod_total and args.mode != "lod":
        raise ValueError("LOD total profiling requires --mode lod")
    if args.profile_full_attention and args.mode != "full":
        raise ValueError("full-attention profiling requires --mode full")
    if sum(
        bool(value)
        for value in (
            args.profile_lod_phases,
            args.profile_lod_total,
            args.profile_full_attention,
        )
    ) > 1:
        raise ValueError("attention phase profilers are mutually exclusive")
    if args.mode == "lod" and args.batch_size > int(
        os.environ.get("VLLM_LOD_POOL_SIZE", "8")
    ):
        raise ValueError("batch size exceeds VLLM_LOD_POOL_SIZE")

    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    max_length = max(args.lengths)
    # Speed-only diagnostics may intentionally run a longer decode than the
    # quality panel. Reserve enough model length for whichever path is longer.
    max_model_len = max(
        max_length + args.max_new_tokens + 16,
        max_length - args.speed_prompt_reserve + args.speed_decode_tokens + 16,
    )
    # Keep the scheduler's aggregate token budget independent of the
    # long-prefill classification threshold.  In particular, B1 with a 4K
    # long-prefill threshold must still be able to use the requested 16K
    # aggregate chunk budget.  Coupling these settings silently changed B1
    # panels to 4K chunks while their B8 counterparts retained 16K.
    max_num_batched_tokens = min(
        args.max_num_batched_tokens,
        args.batch_size * max_length,
    )
    long_prefill_token_threshold = min(
        args.long_prefill_token_threshold, max_model_len
    )
    kwargs = {
        "model": args.checkpoint,
        "load_format": os.getenv("VLLM_WEIGHT_CACHE_LOAD_FORMAT", "ipc_cache"),
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "long_prefill_token_threshold": long_prefill_token_threshold,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "disable_log_stats": False,
    }
    if args.allow_heterogeneous_global_config:
        kwargs["hf_overrides"] = allow_heterogeneous_global_config
    elif args.muse_native_text_config:
        kwargs["hf_overrides"] = muse_native_text_config
    if args.model_impl:
        kwargs["model_impl"] = args.model_impl
    if args.language_model_only:
        kwargs["language_model_only"] = True
    if args.torch_profile_dir is not None:
        profile_dir = args.torch_profile_dir.resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": False,
            "delay_iterations": args.torch_profile_delay_iterations,
            "max_iterations": args.torch_profile_max_iterations,
        }
    if args.mode == "lod":
        kwargs["attention_config"] = {"backend": "CUSTOM"}
    elif args.full_attention_backend:
        kwargs["attention_config"] = {"backend": args.full_attention_backend}
    llm = register_llm_shutdown(LLM(**kwargs))
    if args.mode == "lod" and any(
        value is not None
        for value in (
            args.lod_decode_route_group_size,
            args.lod_decode_route_num_warps,
            args.lod_decode_route_reduce_num_warps,
        )
    ):
        configured = llm.apply_model(
            functools.partial(
                configure_lod_model,
                leaf_num_warps=None,
                recursive_page_block_n=None,
                recursive_state_route_backend=None,
                prefill_chunk_len=None,
                prefill_state_update_len=None,
                direct_prefill_route=False,
                decode_route_group_size=args.lod_decode_route_group_size,
                decode_route_num_warps=args.lod_decode_route_num_warps,
                decode_route_reduce_num_warps=(
                    args.lod_decode_route_reduce_num_warps
                ),
                decode_final_reduce_num_warps=None,
                decode_block_n=None,
                decode_num_warps=None,
                decode_use_dot=None,
            )
        )
        if not configured or not all(value > 0 for value in configured):
            raise RuntimeError("LOD decode-route tuning found no installed layers")

    diagnostics_before = None
    if args.mode == "lod":
        diagnostics_before = llm.apply_model(inspect_lod_model)[0]
        if not diagnostics_before["custom_layers"]:
            raise RuntimeError("vLLM did not construct any CUSTOM attention layers")
        diagnostic_external_empty = os.getenv(
            "VLLM_LOD_DIAGNOSTIC_EXTERNAL_EMPTY_ATTENTION"
        ) in ("skip", "eligible")
        if (
            not diagnostic_external_empty
            and diagnostics_before["pooled_layers"]
            != diagnostics_before["eligible_layers"]
        ):
            raise RuntimeError("LOD pools are missing from CUSTOM attention layers")

    result = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "load_format": kwargs["load_format"],
        "lengths": args.lengths,
        "samples_per_length": args.samples,
        "sample_offset": args.sample_offset,
        "batch_size": args.batch_size,
        "speed_use_warm_prefix_cache": args.speed_use_warm_prefix_cache,
        "max_new_tokens": args.max_new_tokens,
        "speed_decode_tokens": args.speed_decode_tokens,
        "speed_prompt_reserve": args.speed_prompt_reserve,
        "speed_repeats": args.speed_repeats,
        "apply_chat_template": args.apply_chat_template,
        "disable_thinking": args.disable_thinking,
        "max_num_batched_tokens": max_num_batched_tokens,
        "long_prefill_token_threshold": long_prefill_token_threshold,
        "tensor_parallel_size": args.tensor_parallel_size,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "enable_prefix_caching": args.enable_prefix_caching,
        "allow_heterogeneous_global_config": (
            args.allow_heterogeneous_global_config
        ),
        "muse_native_text_config": args.muse_native_text_config,
        "model_impl": args.model_impl,
        "language_model_only": args.language_model_only,
        "full_attention_backend": (
            args.full_attention_backend if args.mode == "full" else None
        ),
        "diagnostic_external_empty_attention": os.getenv(
            "VLLM_LOD_DIAGNOSTIC_EXTERNAL_EMPTY_ATTENTION"
        ),
        "lod_diagnostics_before": diagnostics_before,
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for length in args.lengths:
        row = {}
        if not args.speed_only:
            quality = evaluate_quality(args, tokenizer, llm, length)
            row["quality"] = quality
            # Persist expensive quality work before starting the independent
            # speed warmup. This leaves a useful artifact if a third-party
            # attention backend faults on a boundary-length synthetic prompt.
            result["results"][str(length)] = row
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if not args.quality_only:
            row["speed"] = evaluate_speed(args, tokenizer, llm, length)
        result["results"][str(length)] = row
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        summary = {
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "length": length,
        }
        if "quality" in row:
            summary.update(
                correct=row["quality"]["correct"],
                total=row["quality"]["total"],
                quality_seconds=row["quality"]["elapsed_seconds"],
            )
        if "speed" in row:
            summary.update(
                **{
                    key: row["speed"][key]
                    for key in (
                        "prefill_seconds",
                        "prefill_prompt_tokens_per_second",
                        "marginal_decode_ms_per_batch_step",
                    )
                }
            )
        print(json.dumps(summary, sort_keys=True), flush=True)

    if args.mode == "lod":
        result["lod_diagnostics_after"] = llm.apply_model(inspect_lod_model)[0]
        result["lod_speed_diagnostics"] = llm.apply_model(
            inspect_lod_speed_model
        )[0]
        result["lod_dispatch"] = llm.apply_model(inspect_lod_dispatch)[0]
    else:
        result["full_attention_dispatch"] = llm.apply_model(
            inspect_full_attention_dispatch
        )[0]
    result["attention_memory"] = llm.apply_model(inspect_attention_memory)[0]
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_registered_llms()

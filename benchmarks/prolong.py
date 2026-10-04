"""Measure ProLong prompt loss or matched long-context serving speed."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import math
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from ._identity import benchmark_identity
from ._vllm import (
    MODES,
    SCHEDULER_CHUNK,
    close_llm,
    default_gpu_memory_utilization,
    llm_kwargs,
    write_json,
)

DATASET = "Seerkfang/prolong-64k-512-new"
DATASET_REVISION = "97295b7d7fe48dc0aa6ba373af3a8b9d945e505b"
# Frozen raw-dataset rows, jointly checked to have at least 65,536 tokens under
# both release-model tokenizers.  Quality offsets index this list, so different
# tokenizers can never silently substitute different documents.
QUALITY_DOCUMENT_INDICES = (
    0,
    2,
    3,
    5,
    7,
    10,
    11,
    13,
    14,
    19,
    20,
    23,
    24,
    25,
    27,
    28,
)
SPEED_SHUFFLE_SEED = 20_260_824
SEPARATOR = "\n\n--- NEXT PROLONG DOCUMENT ---\n\n"

# Speed panels revisit the beginning of the same deterministically shuffled
# stream for every context length.  Retain its tokenized documents within one
# benchmark process so a long panel does not repeatedly stream and tokenize
# the same source text while all GPUs sit idle.  This changes setup time only;
# prompt token IDs and their recorded digests remain identical.
_SPEED_DOCUMENT_IDS: list[list[int]] = []
_SPEED_DOCUMENTS: Any | None = None
_SPEED_TOKENIZER_ID: int | None = None
_SPEED_TOKEN_CACHE_LOADED = False


def comma_separated_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or min(result) < 2:
        raise argparse.ArgumentTypeError("all context lengths must be at least two")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=("quality", "speed"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=65_536)
    parser.add_argument(
        "--lengths",
        type=comma_separated_ints,
        default=[8_192, 16_384, 32_768, 65_536, 131_072],
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sample-offset", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--speed-samples",
        type=int,
        help=(
            "number of speed prompts (default: batch-size); values larger than "
            "batch-size run the fixed cohort in consecutive execution batches"
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decode-context-parallel-size", type=int, default=1)
    parser.add_argument(
        "--dcp-comm-backend",
        choices=("ag_rs", "a2a"),
        default="ag_rs",
    )
    parser.add_argument("--decode-tokens", type=int, default=1_025)
    parser.add_argument(
        "--synchronized-decode",
        action="store_true",
        help=(
            "hold decode until every request in the speed cohort has finished "
            "prefill, producing a true steady B=N decode measurement"
        ),
    )
    parser.add_argument(
        "--fixed-decode-trace",
        action="store_true",
        help=(
            "teacher-force every decode row with the natural ProLong tokens "
            "immediately following its prompt; this removes model-output "
            "divergence from matched attention-speed comparisons"
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--retain-warmup-allocator",
        action="store_true",
        help=(
            "keep reusable warmup allocations when at least 8 GiB remains "
            "free; apply identically to dense and LoD warm-serving comparisons"
        ),
    )
    parser.add_argument(
        "--diagnostic-prefill-profile",
        action="store_true",
        help=(
            "run an additional instrumented pass after each measured speed "
            "point; profiler durations are diagnostics, never benchmark times"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="generation seed (default: 0; recorded in the result JSON)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help=(
            "vLLM native-cache memory fraction (default: 0.65 for quality, "
            "0.70 for Qwen LoD speed, 0.80 for K2 LoD speed, or 0.90 for "
            "full-attention speed)"
        ),
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        help=(
            "explicit vLLM native-cache reservation per GPU; useful with the "
            "IPC weight daemon when utilization-based sizing leaves too "
            "little transient workspace"
        ),
    )
    parser.add_argument(
        "--full-attention-backend",
        default="ROCM_AITER_UNIFIED_ATTN",
    )
    parser.add_argument(
        "--kimi-gfx942-int4-moe",
        action="store_true",
        help=(
            "use the Kimi-K3 v10 image's explicit lossy MXFP4-to-groupwise-INT4 "
            "MoE conversion on gfx942; apply identically to dense and LoD runs"
        ),
    )
    parser.add_argument(
        "--weight-cache",
        action="store_true",
        help=(
            "load the final post-conversion TP/EP weight shards through the "
            "persistent local HIP-IPC daemon"
        ),
    )
    parser.add_argument(
        "--weight-cache-id",
        default="prolong",
        help="local daemon namespace (default: prolong)",
    )
    parser.add_argument(
        "--weight-cache-dir",
        help="daemon socket directory (default: XDG_RUNTIME_DIR or /tmp)",
    )
    parser.add_argument(
        "--speculative-model",
        help="Qwen3.8 DFlash2 draft checkpoint (for example z-lab/Qwen3.8-27B-DFlash2)",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--speculative-attention-backend", default="TRITON_ATTN")
    parser.add_argument(
        "--dummy-attention",
        action="store_true",
        help=(
            "benchmark-only no-cache/no-attention control used for paired "
            "real-minus-dummy timing"
        ),
    )
    parser.add_argument(
        "--allow-experimental-environment",
        action="store_true",
        help=(
            "allow non-release LOD_KIMI_* overrides; recorded in JSON and "
            "intended only for explicitly labeled development measurements"
        ),
    )
    return parser.parse_args()


# Every ``LOD_KIMI_*`` switch is an implementation experiment.  Release
# geometry is selected by the model-aware defaults in the runtime; allowing an
# old value here would let a nominally canonical result silently execute a
# different cache-construction policy.
_CANONICAL_KIMI_ENVIRONMENT: dict[str, str] = {}


def benchmark_environment() -> dict[str, str]:
    """Return every environment override that can affect benchmark execution."""

    prefixes = (
        "AITER_",
        "CUDA_",
        "FLASH_ATTENTION_",
        "HSA_",
        "HIP_",
        "LOD_",
        "NCCL_",
        "OMP_",
        "PYTORCH_",
        "ROCR_",
        "TORCH_",
        "TRITON_",
        "VLLM_",
    )
    explicit = {
        "HIPBLASLT_TUNING_OVERRIDE_FILE",
        "KIMI_K3_V10_ROOT",
        "LD_LIBRARY_PATH",
        "LOD_REPO_ROOT",
        "PATH",
        "PROLONG_SPEED_TOKEN_CACHE",
        "PYTHONHOME",
        "PYTHONPATH",
        "ROCM_VISIBLE_DEVICES",
        "ROCM_HOME",
        "ROCM_PATH",
        "SAFETENSORS_FAST_GPU",
        "SDK_CORE",
        "SDK_DEV",
        "TOKENIZERS_PARALLELISM",
        "CUDA_VISIBLE_DEVICES",
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in explicit or key.startswith(prefixes)
    }


def validate_release_environment(*, allow_experimental: bool) -> None:
    """Reject silent Kimi path changes in a release-comparable measurement."""

    if allow_experimental:
        return
    violations = []
    for key, value in os.environ.items():
        if not key.startswith("LOD_KIMI_"):
            continue
        expected = _CANONICAL_KIMI_ENVIRONMENT.get(key)
        if expected is None or value != expected:
            violations.append(f"{key}={value!r}")
    if violations:
        raise ValueError(
            "non-release Kimi benchmark environment is active: "
            + ", ".join(sorted(violations))
            + "; remove it or pass --allow-experimental-environment"
        )


def configure_synchronized_decode_environment(
    *, enabled: bool, batch_size: int
) -> None:
    """Configure both halves of the benchmark-only cohort barrier."""

    if enabled:
        if batch_size < 1:
            raise ValueError("synchronized decode batch size must be positive")
        os.environ["LOD_BENCHMARK_SYNCHRONIZED_DECODE"] = "1"
        # Offline LLM.generate submits requests to the asynchronous engine one
        # message at a time. Hold the first scheduler turn until the complete
        # execution batch is visible; otherwise the first long prompt can
        # finish prefill and decode before the final prompts are admitted.
        os.environ["LOD_BENCHMARK_ADMISSION_COHORT"] = str(batch_size)
    else:
        os.environ.pop("LOD_BENCHMARK_SYNCHRONIZED_DECODE", None)
        os.environ.pop("LOD_BENCHMARK_ADMISSION_COHORT", None)


def token_digest(token_ids: list[int]) -> str:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def document_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def select_quality_prompts(
    tokenizer: Any,
    *,
    length: int,
    samples: int,
    sample_offset: int,
) -> tuple[list[dict[str, list[int]]], list[dict[str, Any]]]:
    from datasets import load_dataset

    if samples < 1 or sample_offset < 0:
        raise ValueError("samples must be positive and sample-offset nonnegative")
    stop = sample_offset + samples
    if stop > len(QUALITY_DOCUMENT_INDICES):
        raise ValueError(
            "quality sample range exceeds the frozen shared document cohort: "
            f"requested [{sample_offset}, {stop}), available "
            f"[0, {len(QUALITY_DOCUMENT_INDICES)})"
        )
    selected_indices = QUALITY_DOCUMENT_INDICES[sample_offset:stop]
    dataset = load_dataset(
        DATASET,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
    )
    prompts = []
    metadata = []
    selected = iter(selected_indices)
    target_index = next(selected)
    for dataset_index, document in enumerate(dataset):
        if dataset_index != target_index:
            continue
        text = document["text"]
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=length,
            return_attention_mask=False,
        )["input_ids"]
        if len(token_ids) != length:
            raise RuntimeError(
                f"frozen ProLong document {dataset_index} produced only "
                f"{len(token_ids):,} tokens; {length:,} are required"
            )
        prompts.append({"prompt_token_ids": token_ids})
        metadata.append(
            {
                "dataset_index": dataset_index,
                "document_sha256": document_digest(text),
                "tokens": length,
                "token_sha256": token_digest(token_ids),
            }
        )
        try:
            target_index = next(selected)
        except StopIteration:
            break
    if len(prompts) != samples:
        raise RuntimeError(
            f"dataset ended after finding {len(prompts)} of {samples} frozen documents"
        )
    return prompts, metadata


def _unique_block_ratio(token_ids: list[int], block_size: int = 16) -> float:
    blocks = {
        tuple(token_ids[begin : begin + block_size])
        for begin in range(0, len(token_ids) - block_size + 1, block_size)
    }
    return len(blocks) / max(1, len(token_ids) // block_size)


def _speed_document_ids(tokenizer: Any, index: int) -> list[int]:
    global _SPEED_DOCUMENTS, _SPEED_TOKENIZER_ID, _SPEED_TOKEN_CACHE_LOADED

    tokenizer_id = id(tokenizer)
    if _SPEED_TOKENIZER_ID not in (None, tokenizer_id):
        _SPEED_DOCUMENT_IDS.clear()
        _SPEED_DOCUMENTS = None
        _SPEED_TOKEN_CACHE_LOADED = False
    _SPEED_TOKENIZER_ID = tokenizer_id
    cache_path = os.environ.get("PROLONG_SPEED_TOKEN_CACHE")
    if cache_path and not _SPEED_TOKEN_CACHE_LOADED:
        import torch

        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected = (DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED)
        observed = (
            cached.get("dataset"),
            cached.get("revision"),
            cached.get("seed"),
        )
        if observed != expected:
            raise ValueError(
                "ProLong speed token cache identity differs from this benchmark: "
                f"expected={expected}, observed={observed}"
            )
        documents = cached.get("documents")
        if not isinstance(documents, list) or not all(
            isinstance(document, list) for document in documents
        ):
            raise TypeError("ProLong speed token cache has invalid documents")
        _SPEED_DOCUMENT_IDS.extend(documents)
        _SPEED_TOKEN_CACHE_LOADED = True
    if index < len(_SPEED_DOCUMENT_IDS):
        return _SPEED_DOCUMENT_IDS[index]
    if _SPEED_TOKEN_CACHE_LOADED:
        raise IndexError(
            "ProLong speed token cache is too short: "
            f"requested document {index}, cached {len(_SPEED_DOCUMENT_IDS)}"
        )
    if _SPEED_DOCUMENTS is None:
        from datasets import load_dataset

        _SPEED_DOCUMENTS = iter(
            load_dataset(
                DATASET,
                revision=DATASET_REVISION,
                split="train",
                streaming=True,
            ).shuffle(seed=SPEED_SHUFFLE_SEED, buffer_size=1_000)
        )
    while len(_SPEED_DOCUMENT_IDS) <= index:
        try:
            document = next(_SPEED_DOCUMENTS)
        except StopIteration as exc:
            raise RuntimeError("ProLong ended while filling the speed cache") from exc
        _SPEED_DOCUMENT_IDS.append(
            tokenizer(
                document["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )
    return _SPEED_DOCUMENT_IDS[index]


def _close_speed_document_stream() -> None:
    """Release the streaming dataset before interpreter module teardown."""

    global _SPEED_DOCUMENTS
    documents = _SPEED_DOCUMENTS
    _SPEED_DOCUMENTS = None
    close = getattr(documents, "close", None)
    if callable(close):
        close()


def make_speed_prompts(
    tokenizer: Any,
    *,
    length: int,
    batch_size: int,
    validation_lengths: tuple[int, ...] | None = None,
) -> tuple[list[dict[str, list[int]]], list[dict[str, Any]]]:
    """Build distinct, exact-length prompts without repeating documents."""

    checked_lengths = validation_lengths or (length,)
    if not checked_lengths or min(checked_lengths) < 1 or max(checked_lengths) > length:
        raise ValueError("validation lengths must lie within the generated prompt")
    separator = tokenizer(SEPARATOR, add_special_tokens=False)["input_ids"]
    prompts = []
    metadata = []
    stream_index = -1
    while len(prompts) < batch_size:
        token_ids: list[int] = []
        source_indices = []
        while len(token_ids) < length:
            stream_index += 1
            document_ids = _speed_document_ids(tokenizer, stream_index)
            if not document_ids:
                continue
            if token_ids:
                remaining = length - len(token_ids)
                token_ids.extend(separator[: max(0, remaining - 1)])
            token_ids.extend(document_ids[: length - len(token_ids)])
            source_indices.append(stream_index)
        # Repetitive source can distort expert routing. Match the archived
        # panel's guard by consuming and replacing such a candidate.
        if any(
            _unique_block_ratio(token_ids[:checked_length]) < 0.95
            for checked_length in checked_lengths
        ):
            continue
        prompts.append({"prompt_token_ids": token_ids})
        metadata.append(
            {
                "request_index": len(prompts) - 1,
                "source_stream_indices": source_indices,
                "token_sha256": token_digest(token_ids),
                "unique_16_token_block_ratio": _unique_block_ratio(token_ids),
            }
        )
    return prompts, metadata


def make_speed_trace_panel(
    tokenizer: Any,
    *,
    lengths: list[int],
    batch_size: int,
    decode_tokens: int,
) -> dict[
    int,
    tuple[
        list[dict[str, list[int]]],
        list[dict[str, Any]],
        list[list[int]],
    ],
]:
    """Build one real-token cohort and use nested prefixes at every length."""

    ordered_lengths = sorted(set(lengths))
    if not ordered_lengths or min(ordered_lengths) < 1:
        raise ValueError("speed panel needs positive context lengths")
    panel_prompt_tokens = max(ordered_lengths)
    complete, panel_metadata = make_speed_prompts(
        tokenizer,
        length=panel_prompt_tokens + decode_tokens,
        batch_size=batch_size,
        validation_lengths=tuple(ordered_lengths),
    )
    panel: dict[
        int,
        tuple[
            list[dict[str, list[int]]],
            list[dict[str, Any]],
            list[list[int]],
        ],
    ] = {}
    for length in ordered_lengths:
        prompts: list[dict[str, list[int]]] = []
        metadata: list[dict[str, Any]] = []
        traces: list[list[int]] = []
        for item, panel_record in zip(complete, panel_metadata, strict=True):
            token_ids = item["prompt_token_ids"]
            prompt = token_ids[:length]
            trace = token_ids[length : length + decode_tokens]
            if len(prompt) != length or len(trace) != decode_tokens:
                raise RuntimeError("failed to slice the shared speed panel")
            prompts.append({"prompt_token_ids": prompt})
            traces.append(trace)
            record = dict(panel_record)
            record["panel_token_sha256"] = record.pop("token_sha256")
            record["panel_source_stream_indices"] = record.pop(
                "source_stream_indices"
            )
            record["panel_prompt_tokens"] = panel_prompt_tokens
            record["panel_total_tokens"] = panel_prompt_tokens + decode_tokens
            record.update(
                token_sha256=token_digest(prompt),
                unique_16_token_block_ratio=_unique_block_ratio(prompt),
                trace_token_sha256=token_digest(trace),
                trace_tokens=len(trace),
            )
            metadata.append(record)
        panel[length] = prompts, metadata, traces
    return panel


def evaluate_quality(
    llm: Any,
    tokenizer: Any,
    *,
    length: int,
    samples: int,
    sample_offset: int,
    batch_size: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prompts, metadata = select_quality_prompts(
        tokenizer,
        length=length,
        samples=samples,
        sample_offset=sample_offset,
    )
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
        flat_logprobs=True,
        detokenize=False,
    )
    started = time.perf_counter()
    outputs = []
    for begin in range(0, len(prompts), batch_size):
        outputs.extend(
            llm.generate(prompts[begin : begin + batch_size], params, use_tqdm=True)
        )
    elapsed = time.perf_counter() - started

    total_nll = 0.0
    total_tokens = 0
    records = []
    for prompt, prompt_metadata, output in zip(
        prompts,
        metadata,
        outputs,
        strict=True,
    ):
        token_ids = prompt["prompt_token_ids"]
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) != len(token_ids):
            raise RuntimeError("vLLM returned incomplete prompt log probabilities")
        nll = 0.0
        for token_id, candidates in zip(
            token_ids[1:],
            prompt_logprobs[1:],
            strict=True,
        ):
            if candidates is None or token_id not in candidates:
                raise RuntimeError("target token missing from prompt_logprobs")
            nll -= float(candidates[token_id].logprob)
        predicted_tokens = len(token_ids) - 1
        total_nll += nll
        total_tokens += predicted_tokens
        records.append(
            {
                **prompt_metadata,
                "loss": nll / predicted_tokens,
                "perplexity": math.exp(nll / predicted_tokens),
            }
        )
    loss = total_nll / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "prediction_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "samples": records,
    }


def timed_generate(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    params: Any | list[Any],
) -> tuple[
    float,
    float,
    float,
    tuple[tuple[int, ...], ...],
    dict[str, int],
    dict[str, Any],
]:
    before = speculative_counters(llm)
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    after = speculative_counters(llm)
    request_params = params if isinstance(params, list) else [params] * len(prompts)
    if len(request_params) != len(prompts):
        raise ValueError("sampling-parameter count does not match prompt count")
    expected = [int(item.max_tokens) for item in request_params]
    if any(
        len(output.outputs[0].token_ids) != count
        for output, count in zip(outputs, expected, strict=True)
    ):
        raise RuntimeError("a speed request stopped before max_tokens")
    for output, item in zip(outputs, request_params, strict=True):
        trace = getattr(item, "trace_decode_token_ids", None)
        if trace is not None and list(output.outputs[0].token_ids) != trace:
            raise RuntimeError("vLLM did not reproduce the requested decode trace")
    metrics = [output.metrics for output in outputs]
    if any(metric is None for metric in metrics):
        raise RuntimeError("vLLM did not return per-request timing metrics")
    request_num_preemptions = [
        int(metric.num_preemptions) for metric in metrics
    ]
    if any(request_num_preemptions):
        raise RuntimeError(
            "a speed request was preempted; the measured cohort is invalid: "
            f"{request_num_preemptions!r}"
        )
    request_num_cached_tokens = [
        int(output.num_cached_tokens or 0) for output in outputs
    ]
    if any(request_num_cached_tokens):
        raise RuntimeError(
            "a speed request used prefix-cached tokens; the measured cohort "
            f"is invalid: {request_num_cached_tokens!r}"
        )
    scheduled_times = [float(metric.scheduled_ts) for metric in metrics]
    first_token_times = [float(metric.first_token_ts) for metric in metrics]
    last_token_times = [float(metric.last_token_ts) for metric in metrics]
    scheduled = min(scheduled_times)
    first_token = max(first_token_times)
    last_token = max(last_token_times)
    first_token_spread = max(first_token_times) - min(first_token_times)
    last_token_spread = max(last_token_times) - min(last_token_times)
    all_requests_live_overlap = min(last_token_times) - first_token
    if (
        len(prompts) > 1
        and os.environ.get("LOD_BENCHMARK_SYNCHRONIZED_DECODE", "0") == "1"
    ):
        # first_token_ts is the prefill-produced first sample, so sequential
        # chunked prefills naturally have a large first-token spread.  The
        # cohort barrier applies to subsequent decode.  Validate that all rows
        # remain live after the final prefill and finish together; otherwise a
        # cache-capacity limit may silently turn B=N into consecutive waves.
        decode_window = last_token - first_token
        finish_tolerance = max(0.05, 0.02 * decode_window)
        if all_requests_live_overlap <= 0 or last_token_spread > finish_tolerance:
            raise RuntimeError(
                "synchronized decode cohort did not execute together: "
                f"all-live overlap={all_requests_live_overlap:.6f}s, "
                f"last-token spread={last_token_spread:.6f}s, "
                f"tolerance={finish_tolerance:.6f}s"
            )
    token_ids = tuple(
        tuple(map(int, output.outputs[0].token_ids)) for output in outputs
    )
    counter_delta = {
        name: after[name] - before.get(name, 0)
        for name in after
        if after[name] != before.get(name, 0)
    }
    return (
        elapsed,
        first_token - scheduled,
        last_token - first_token,
        token_ids,
        counter_delta,
        {
            "wall_elapsed_seconds": elapsed,
            "metric_window_seconds": last_token - scheduled,
            "prefill_window_seconds": first_token - scheduled,
            "decode_window_seconds": last_token - first_token,
            "unattributed_wall_seconds": elapsed - (last_token - scheduled),
            "first_token_spread_seconds": first_token_spread,
            "last_token_spread_seconds": last_token_spread,
            "all_requests_live_overlap_seconds": all_requests_live_overlap,
            "request_num_preemptions": request_num_preemptions,
            "request_num_cached_tokens": request_num_cached_tokens,
            "request_prefill_seconds": [
                first - request_scheduled
                for request_scheduled, first in zip(
                    scheduled_times, first_token_times, strict=True
                )
            ],
            "request_decode_seconds": [
                last - first
                for first, last in zip(
                    first_token_times, last_token_times, strict=True
                )
            ],
        },
    )


def speculative_counters(llm: Any) -> dict[str, int]:
    """Read cumulative speculative-decode counters when vLLM exposes them."""

    get_metrics = getattr(llm, "get_metrics", None)
    if not callable(get_metrics):
        return {}
    wanted = {
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
    }
    counters: defaultdict[str, int] = defaultdict(int)
    for metric in get_metrics():
        name = getattr(metric, "name", None)
        value = getattr(metric, "value", None)
        if name in wanted and value is not None:
            counters[name] += int(value)
    return dict(counters)


def timed_generate_cohort(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    params: Any | list[Any],
    *,
    batch_size: int,
) -> tuple[
    float,
    float,
    float,
    tuple[tuple[int, ...], ...],
    dict[str, int],
    list[dict[str, int]],
    list[dict[str, Any]],
]:
    """Run one fixed cohort in execution batches and sum its measurements."""

    elapsed = 0.0
    prefill = 0.0
    decode = 0.0
    token_ids: list[tuple[int, ...]] = []
    counters: defaultdict[str, int] = defaultdict(int)
    batch_counters = []
    batch_timings = []
    for begin in range(0, len(prompts), batch_size):
        batch_params = (
            params[begin : begin + batch_size]
            if isinstance(params, list)
            else params
        )
        batch = timed_generate(
            llm,
            prompts[begin : begin + batch_size],
            batch_params,
        )
        (
            batch_elapsed,
            batch_prefill,
            batch_decode,
            batch_tokens,
            counter_delta,
            batch_timing,
        ) = batch
        elapsed += batch_elapsed
        prefill += batch_prefill
        decode += batch_decode
        token_ids.extend(batch_tokens)
        batch_counters.append(counter_delta)
        batch_timings.append(batch_timing)
        for name, value in counter_delta.items():
            counters[name] += value
    return (
        elapsed,
        prefill,
        decode,
        tuple(token_ids),
        dict(counters),
        batch_counters,
        batch_timings,
    )


def release_worker_allocator_cache(worker: Any, retain: bool = False) -> dict[str, Any]:
    """Return dead warm-up allocations without touching persistent caches.

    Long Kimi prefill has a large transient MoE/attention high-water mark.  On
    ROCm, leaving those inactive blocks reserved by PyTorch can starve RCCL,
    whose launch resources are allocated outside the caching allocator.  The
    speed benchmark's reference generation is deliberately outside the timed
    sample, so normally release only its dead blocks before the measurement.
    An explicit warm-serving experiment may retain them with 8 GiB headroom;
    this policy must be matched across dense and LoD, and is recorded below.
    """

    del worker
    import torch

    torch.cuda.synchronize()
    free_bytes, _ = torch.cuda.mem_get_info()
    reclaimed = not retain or free_bytes < 8 * 1024**3
    if reclaimed:
        torch.cuda.empty_cache()
    return {
        "retention_requested": retain,
        "reclaimed": reclaimed,
        "free_bytes_before": free_bytes,
        "free_bytes_after": torch.cuda.mem_get_info()[0],
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def audit_worker_cohort_capacity(worker: Any) -> dict[str, Any]:
    """Reject a synchronized cohort that cannot all stay resident.

    Otherwise completed rows hold their KV blocks at the decode barrier while
    the final waiting row cannot obtain blocks, silently deadlocking timing.
    Use vLLM's hybrid-cache calculation, not a naive token/block estimate.
    """
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

    runner = worker.model_runner
    return {
        "max_concurrent_requests": get_max_concurrency_for_kv_cache_config(
            runner.vllm_config, runner.kv_cache_config,
        ),
        "num_blocks": int(runner.kv_cache_config.num_blocks),
    }


def validate_cohort_capacity(capacities: list[dict[str, Any]], batch_size: int) -> None:
    if not capacities or min(
        item["max_concurrent_requests"] for item in capacities
    ) < batch_size:
        raise RuntimeError(
            f"synchronized batch {batch_size} exceeds native cache capacity: "
            f"{capacities}; increase --kv-cache-memory-bytes before benchmarking"
        )


def audit_worker_attention_mode(worker: Any) -> dict[str, Any]:
    """Report the actual worker-side attention/runtime selected by vLLM."""

    runner = getattr(worker, "model_runner", None)
    state = getattr(runner, "model_state", None)
    model = getattr(state, "model", None)
    if model is None:
        model = getattr(state, "get_model", lambda: None)()
    runtime = getattr(state, "_vllm_lod_runtime", None)
    pools = getattr(runtime, "pools", {}) if runtime is not None else {}
    engine_configurations: list[dict[str, Any]] = []
    for pool in pools.values():
        engine = pool.engine
        family = getattr(pool, "family", None)
        mode = getattr(getattr(pool, "settings", None), "mode", None)
        engine_configurations.append(
            {
                "family": getattr(family, "value", str(family)),
                "mode": getattr(mode, "value", str(mode)),
                "levels": int(pool.settings.levels),
                "kv_bits": int(pool.settings.kv_bits),
                "prefill_routes": int(
                    engine.prefill_two_level_topk
                    if engine.prefill_two_level_topk is not None
                    else engine.two_level_topk
                ),
                "decode_routes": int(engine.two_level_topk),
                "prefill_chunk_len": int(engine.prefill_chunk_len),
                "prefill_state_update_len": int(
                    engine.prefill_state_update_len
                ),
                "decode_state_update_len": int(engine.decode_state_update_len),
                "local_len": int(engine.local_len),
                "leaf_page_size": int(engine.leaf_page_size),
                "leaf_block_m": int(engine.leaf_block_m),
                "leaf_block_n": int(engine.leaf_block_n),
                "leaf_num_warps": int(engine.leaf_num_warps),
                "prefill_fused_route_coarse": bool(
                    engine.fused_prefill_route_coarse
                ),
                "decode_fused_state_route": bool(engine.fused_decode_state_route),
                "max_open_centroid_leaves": engine.max_open_centroid_leaves,
                "leaf_seal_capacity": engine.leaf_seal_capacity,
            }
        )
    unique_engine_configurations = []
    for configuration in engine_configurations:
        if configuration not in unique_engine_configurations:
            unique_engine_configurations.append(configuration)
    context = getattr(
        getattr(
            getattr(state, "vllm_config", None),
            "compilation_config",
            None,
        ),
        "static_forward_context",
        {},
    )
    attached_pools = sum(
        getattr(layer, "_vllm_lod_pool", None) is not None
        for layer in context.values()
    )
    attention_impl_classes = sorted(
        type(impl).__name__
        for layer in context.values()
        if (impl := getattr(layer, "impl", None)) is not None
    )
    dummy_attention_layers = sum(
        name == "DummyAttentionImpl" for name in attention_impl_classes
    )
    dynamic_moe_layer_count = (
        sum(
            1
            for module in model.modules()
            if "moe" in type(module).__name__.lower()
        )
        if model is not None
        else 0
    )
    loaded_kimi_lod_modules = []
    seen_module_paths: set[Path] = set()
    for module_name, module in sorted(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if "lod_kimi" not in module_name or not module_file:
            continue
        module_path = Path(module_file).resolve()
        if module_path in seen_module_paths or not module_path.is_file():
            continue
        seen_module_paths.add(module_path)
        module_digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
        build_manifest = (
            module_path.parent
            / "build"
            / module_path.stem
            / "build"
            / "build.ninja"
        )
        build_manifest_bytes = (
            build_manifest.read_bytes() if build_manifest.is_file() else None
        )
        route_build_flags = {}
        if build_manifest_bytes is not None:
            manifest_text = build_manifest_bytes.decode(errors="replace")
            route_build_flags = dict(
                re.findall(
                    r"-D(CK_TILE_FMHA_ROUTE_[A-Z_]+)=([^\s]+)",
                    manifest_text,
                )
            )
        loaded_kimi_lod_modules.append(
            {
                "module": module_name,
                "path": str(module_path),
                "size_bytes": module_path.stat().st_size,
                "sha256": module_digest,
                "build_manifest_sha256": (
                    hashlib.sha256(build_manifest_bytes).hexdigest()
                    if build_manifest_bytes is not None
                    else None
                ),
                "route_build_flags": route_build_flags,
            }
        )
    import torch

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    try:
        import vllm.v1.attention.backends.mla.triton_mla as triton_mla

        dense_gluon_decode = bool(
            getattr(
                triton_mla.TritonMLAImpl,
                "_vllm_lod_dense_gluon_installed",
                False,
            )
        )
    except ImportError:
        dense_gluon_decode = False

    # AITER JITs the LoD routing specialization from its installed CK source,
    # which is outside this repository.  Fingerprint the exact source inputs
    # so two nominally identical benchmark arms cannot silently use different
    # route-only semantics (in particular, an output-writing epilogue).
    aiter_route_source_sha256 = None
    try:
        import aiter

        site_packages = Path(aiter.__file__).resolve().parent.parent
        aiter_meta = site_packages / "aiter_meta"
        route_sources = (
            "csrc/py_itfs_ck/mha_fwd_kernels.cu",
            "3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/variants.hpp",
            "3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp",
            "3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp",
            "3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp",
            "3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py",
        )
        digest = hashlib.sha256()
        for relative in route_sources:
            path = aiter_meta / relative
            digest.update(relative.encode())
            digest.update(path.read_bytes())
        aiter_route_source_sha256 = digest.hexdigest()
    except (ImportError, OSError):
        pass
    return {
        "lod_runtime": runtime is not None,
        "lod_pool_count": len(pools),
        "lod_engine_configurations": unique_engine_configurations,
        "lod_cross_layer_prefill_group_size": (
            int(runtime.cross_layer_prefill_group_size)
            if runtime is not None and pools
            else None
        ),
        "attached_lod_layers": attached_pools,
        "attention_layer_count": len(attention_impl_classes),
        "attention_impl_classes": attention_impl_classes,
        "dummy_attention_layers": dummy_attention_layers,
        "dynamic_moe_layer_count": dynamic_moe_layer_count,
        "model_class": type(model).__name__ if model is not None else None,
        "loaded_kimi_lod_modules": loaded_kimi_lod_modules,
        "dense_gluon_decode_installed": dense_gluon_decode,
        "aiter_route_source_sha256": aiter_route_source_sha256,
        "device_name": properties.name,
        "device_arch": getattr(properties, "gcnArchName", None),
        "device_total_memory_bytes": int(properties.total_memory),
        "torch_hip_version": torch.version.hip,
    }


def validate_worker_attention_mode(
    audits: list[dict[str, Any]],
    *,
    mode: str,
    dummy_attention: bool = False,
    require_loaded_kimi_lod: bool = False,
) -> None:
    """Fail before timing when requested and executed attention modes differ."""

    expected_lod = mode != "full"
    invalid = [
        audit
        for audit in audits
        if bool(audit["lod_runtime"]) != expected_lod
        or (int(audit["lod_pool_count"]) > 0) != expected_lod
        or (int(audit["attached_lod_layers"]) > 0) != expected_lod
    ]
    if invalid:
        raise RuntimeError(
            f"requested mode {mode!r} does not match worker attention state: "
            f"{invalid!r}"
        )
    if require_loaded_kimi_lod and expected_lod:
        missing_modules = [
            audit
            for audit in audits
            if "kimi" in str(audit.get("model_class", "")).lower()
            and not audit.get("loaded_kimi_lod_modules")
        ]
        if missing_modules:
            raise RuntimeError(
                "Kimi LoD workers did not expose their loaded AITER JIT "
                f"module identity: {missing_modules!r}"
            )
    if expected_lod:
        expected_mode = mode
        invalid_engines = []
        for audit in audits:
            configurations = audit.get("lod_engine_configurations")
            if not isinstance(configurations, list) or len(configurations) != 1:
                invalid_engines.append(audit)
                continue
            configuration = configurations[0]
            if (
                configuration.get("mode") != expected_mode
                or int(configuration.get("prefill_routes", -1)) != 8
                or int(configuration.get("decode_routes", -1)) != 8
                or int(configuration.get("prefill_chunk_len", -1)) != 16_384
                or int(configuration.get("prefill_state_update_len", -1))
                != 16_384
                or int(configuration.get("decode_state_update_len", -1)) != 256
            ):
                invalid_engines.append(audit)
        if invalid_engines:
            raise RuntimeError(
                "requested release LoD geometry does not match workers: "
                f"{invalid_engines!r}"
            )
    wrong_dummy = [
        audit
        for audit in audits
        if (
            int(audit.get("dummy_attention_layers", 0))
            == int(audit.get("attention_layer_count", 0))
            and int(audit.get("attention_layer_count", 0)) > 0
        )
        != dummy_attention
    ]
    if wrong_dummy:
        raise RuntimeError(
            "requested dummy-attention state does not match workers: "
            f"{wrong_dummy!r}"
        )
    expected_dense_gluon = os.environ.get("VLLM_KIMI_DENSE_GLUON", "0") == "1"
    wrong_dense_decoder = [
        audit
        for audit in audits
        if bool(audit.get("dense_gluon_decode_installed", False))
        != expected_dense_gluon
    ]
    if wrong_dense_decoder:
        raise RuntimeError(
            "requested dense Gluon decoder state does not match workers: "
            f"{wrong_dense_decoder!r}"
        )


def evaluate_speed(
    llm: Any,
    tokenizer: Any,
    *,
    lengths: list[int],
    batch_size: int,
    samples: int,
    decode_tokens: int,
    repeats: int,
    seed: int,
    fixed_decode_trace: bool = False,
    retain_warmup_allocator: bool = False,
    diagnostic_prefill_profile: bool = False,
) -> dict[str, Any]:
    from vllm import SamplingParams

    param_kwargs = {
        "temperature": 0,
        "seed": seed,
        "max_tokens": decode_tokens,
        "detokenize": False,
        "ignore_eos": True,
    }
    if samples % batch_size:
        raise ValueError("speed-samples must be divisible by batch-size")
    cohort_batches = samples // batch_size
    trace_panel = (
        make_speed_trace_panel(
            tokenizer,
            lengths=lengths,
            batch_size=samples,
            decode_tokens=decode_tokens,
        )
        if fixed_decode_trace
        else None
    )
    result = {}
    for length in lengths:
        if fixed_decode_trace:
            assert trace_panel is not None
            prompts, prompt_metadata, traces = trace_panel[length]
            params: Any | list[Any] = [
                SamplingParams(**param_kwargs, trace_decode_token_ids=trace)
                for trace in traces
            ]
        else:
            prompts, prompt_metadata = make_speed_prompts(
                tokenizer,
                length=length,
                batch_size=samples,
            )
            params = SamplingParams(**param_kwargs)
        *_, reference, _, _, warmup_batch_timings = timed_generate_cohort(
            llm,
            prompts,
            params,
            batch_size=batch_size,
        )
        allocator_after_warmup = (
            llm.collective_rpc(release_worker_allocator_cache, args=(True,))
            if retain_warmup_allocator
            else llm.collective_rpc(release_worker_allocator_cache)
        )
        prefill_timings = []
        decode_timings = []
        cohort_prefill_timings = []
        cohort_decode_timings = []
        cohort_wall_timings = []
        measured_batch_timings: list[list[dict[str, Any]]] = []
        speculative_measurements = []
        output_token_sha256: list[list[str]] = []
        first_mismatch_positions: list[list[int | None]] = []
        for _ in range(repeats):
            (
                _cohort_elapsed,
                cohort_prefill,
                cohort_decode,
                token_ids,
                counters,
                batch_counters,
                batch_timings,
            ) = timed_generate_cohort(
                llm,
                prompts,
                params,
                batch_size=batch_size,
            )
            first_mismatch_positions.append(
                [
                    next(
                        (
                            index
                            for index, (expected, actual) in enumerate(
                                zip(reference_row, measured_row, strict=True)
                            )
                            if expected != actual
                        ),
                        None,
                    )
                    for reference_row, measured_row in zip(
                        reference, token_ids, strict=True
                    )
                ]
            )
            prefill_timings.append(cohort_prefill / cohort_batches)
            decode_timings.append(cohort_decode / cohort_batches)
            cohort_prefill_timings.append(cohort_prefill)
            cohort_decode_timings.append(cohort_decode)
            cohort_wall_timings.append(_cohort_elapsed)
            measured_batch_timings.append(batch_timings)
            output_token_sha256.append([token_digest(list(row)) for row in token_ids])
            drafts = counters.get("vllm:spec_decode_num_drafts", 0)
            draft_tokens = counters.get("vllm:spec_decode_num_draft_tokens", 0)
            accepted = counters.get("vllm:spec_decode_num_accepted_tokens", 0)
            if drafts:
                measurement = {
                    "target_cycles": drafts,
                    "draft_tokens": draft_tokens,
                    "accepted_draft_tokens": accepted,
                    "mean_acceptance_length": 1.0 + accepted / drafts,
                    "draft_acceptance_rate": accepted / draft_tokens,
                    "target_cycle_ms": 1_000.0 * cohort_decode / drafts,
                }
                if batch_size == 1:
                    per_request_acceptance = [
                        1.0
                        + item.get("vllm:spec_decode_num_accepted_tokens", 0)
                        / item["vllm:spec_decode_num_drafts"]
                        for item in batch_counters
                        if item.get("vllm:spec_decode_num_drafts", 0)
                    ]
                    if len(per_request_acceptance) != samples:
                        raise RuntimeError(
                            "missing DFlash counters for an isolated speed request"
                        )
                    measurement.update(
                        equal_weight_request_mean_acceptance_length=statistics.mean(
                            per_request_acceptance
                        ),
                        per_request_acceptance_lengths=per_request_acceptance,
                    )
                speculative_measurements.append(measurement)
        prefill = statistics.median(prefill_timings)
        decode = statistics.median(decode_timings)
        decode_steps = decode_tokens - 1
        measurement = {
            "prefill_seconds": prefill,
            "prefill_prompt_tokens_per_second": batch_size * length / prefill,
            "decode_ms_per_batch_step": 1_000.0 * decode / decode_steps,
            "decode_tokens_per_second": batch_size * decode_steps / decode,
            "prefill_timings_seconds": prefill_timings,
            "decode_timings_seconds": decode_timings,
            "cohort_prefill_timings_seconds": cohort_prefill_timings,
            "cohort_decode_timings_seconds": cohort_decode_timings,
            "cohort_wall_timings_seconds": cohort_wall_timings,
            "cohort_unattributed_wall_seconds": [
                wall - prefill_time - decode_time
                for wall, prefill_time, decode_time in zip(
                    cohort_wall_timings,
                    cohort_prefill_timings,
                    cohort_decode_timings,
                    strict=True,
                )
            ],
            "warmup_batch_timings": warmup_batch_timings,
            "measured_batch_timings": measured_batch_timings,
            "greedy_output_identical": all(
                position is None
                for repeat in first_mismatch_positions
                for position in repeat
            ),
            "first_mismatch_positions": first_mismatch_positions,
            "warmup_output_token_sha256": [
                token_digest(list(row)) for row in reference
            ],
            "output_token_sha256": output_token_sha256,
            "speculative_measurements": speculative_measurements,
            "prompts": prompt_metadata,
            "allocator_after_warmup": allocator_after_warmup,
        }
        if speculative_measurements:
            measurement.update(
                speculative_target_cycle_ms=statistics.median(
                    item["target_cycle_ms"] for item in speculative_measurements
                ),
                speculative_mean_acceptance_length=statistics.median(
                    item["mean_acceptance_length"] for item in speculative_measurements
                ),
                speculative_draft_acceptance_rate=statistics.median(
                    item["draft_acceptance_rate"] for item in speculative_measurements
                ),
            )
            if batch_size == 1:
                equal_weight_acceptance = statistics.median(
                    item["equal_weight_request_mean_acceptance_length"]
                    for item in speculative_measurements
                )
                measurement[
                    "speculative_equal_weight_request_mean_acceptance_length"
                ] = equal_weight_acceptance
        result[str(length)] = measurement
        if diagnostic_prefill_profile:
            from benchmarks._prefill_profile import (
                start_prefill_profile,
                stop_prefill_profile,
            )

            # Profiling (and its optional routing-use counters) runs only
            # after the canonical measurement. Never insert profiler events
            # or extra synchronizations into the measured generation itself.
            llm.collective_rpc(start_prefill_profile)
            try:
                timed_generate_cohort(llm, prompts, params, batch_size=batch_size)
            finally:
                measurement["diagnostic_profiles"] = llm.collective_rpc(
                    stop_prefill_profile
                )
    return result


def main() -> None:
    args = parse_args()
    validate_release_environment(
        allow_experimental=args.allow_experimental_environment
    )
    if args.measure == "speed":
        # Attribute Kimi's final background cache-construction tail to prefill
        # rather than the first decode step. Dense attention ignores this
        # common benchmark flag, so matched environments remain identical.
        os.environ["LOD_BENCHMARK_SYNC_PREFILL_CACHE"] = "1"
    elif args.retain_warmup_allocator:
        raise ValueError("--retain-warmup-allocator requires --measure speed")
    if args.diagnostic_prefill_profile and args.measure != "speed":
        raise ValueError("--diagnostic-prefill-profile requires --measure speed")
    run_identity = benchmark_identity()
    # The attention-timing callback is not a msgpack object, so vLLM 0.27
    # requires explicit opt-in before sending it to local workers.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if args.synchronized_decode:
        if args.measure != "speed":
            raise ValueError("--synchronized-decode requires --measure speed")
    configure_synchronized_decode_environment(
        enabled=args.synchronized_decode,
        batch_size=args.batch_size,
    )
    if args.dummy_attention:
        if args.measure != "speed" or args.mode != "full":
            raise ValueError("--dummy-attention requires --measure speed --mode full")
        if args.speculative_model:
            raise ValueError("--dummy-attention does not support speculative decoding")
        os.environ["LOD_BENCHMARK_DUMMY_ATTENTION"] = "1"
    if args.fixed_decode_trace:
        if args.measure != "speed":
            raise ValueError("--fixed-decode-trace requires --measure speed")
        if args.speculative_model:
            raise ValueError(
                "--fixed-decode-trace is incompatible with speculative decoding"
            )
    if min(args.length, args.samples, args.batch_size, args.repeats) < 1:
        raise ValueError("length, samples, batch-size, and repeats must be positive")
    if args.sample_offset < 0:
        raise ValueError("sample-offset must be nonnegative")
    if args.decode_tokens < 2:
        raise ValueError("decode-tokens must be at least two")
    speed_samples = (
        args.batch_size if args.speed_samples is None else args.speed_samples
    )
    if speed_samples < args.batch_size or speed_samples % args.batch_size:
        raise ValueError(
            "speed-samples must be at least batch-size and divisible by it"
        )

    from transformers import AutoTokenizer

    max_length = args.length if args.measure == "quality" else max(args.lengths)
    generation_tokens = 1 if args.measure == "quality" else args.decode_tokens
    gpu_memory_utilization = args.gpu_memory_utilization
    if gpu_memory_utilization is None:
        gpu_memory_utilization = default_gpu_memory_utilization(
            args.checkpoint,
            args.mode,
            quality=args.measure == "quality",
        )
    kwargs = llm_kwargs(
        checkpoint=args.checkpoint,
        mode=args.mode,
        max_model_len=max_length + generation_tokens + 16,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        decode_context_parallel_size=args.decode_context_parallel_size,
        dcp_comm_backend=args.dcp_comm_backend,
        gpu_memory_utilization=gpu_memory_utilization,
        full_attention_backend=args.full_attention_backend,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        speculative_attention_backend=args.speculative_attention_backend,
    )
    if args.dummy_attention:
        kwargs["attention_config"] = {"backend": "CUSTOM"}
        kwargs["additional_config"] = {"lod_benchmark_dummy_attention": True}
    if args.kimi_gfx942_int4_moe:
        if "kimi-k3" not in args.checkpoint.lower():
            raise ValueError("--kimi-gfx942-int4-moe requires a Kimi-K3 checkpoint")
        # The validated gfx942 packed-int4 path uses one expert partition per
        # TP rank.  Besides matching the production geometry (112 of K3's 896
        # experts per MI325X), this bounds the one-time MXFP4 -> INT4
        # conversion workspace and selects AITER's fast EP execution path.
        kwargs["enable_expert_parallel"] = True
        kwargs["disable_custom_all_reduce"] = False
        kwargs["quantization_config"] = {
            "moe": {"weight": "int4_per_group_32"}
        }
        # The full K3 checkpoint is ~1.45 TiB on shared Ceph.  vLLM does not
        # recognize Ceph as a network filesystem, so opt into its rank-sharded
        # background readahead explicitly.  This affects startup only; all
        # benchmark timers begin after model construction and warmup.
        kwargs["safetensors_load_strategy"] = "prefetch"
    if args.weight_cache:
        kwargs["load_format"] = "ipc_cache"
        kwargs["model_loader_extra_config"] = {
            "auto_start": True,
            "cache_id": args.weight_cache_id,
            "cache_dir": args.weight_cache_dir,
            "backing_load_format": "auto",
            # Full K3's cold load includes its one-time MXFP4 -> INT4
            # conversion. Warm clients map those final tensors directly.
            "broker_timeout": 1800.0,
        }
    if args.kv_cache_memory_bytes is not None:
        if args.kv_cache_memory_bytes <= 0:
            raise ValueError("--kv-cache-memory-bytes must be positive")
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.fixed_decode_trace:
        kwargs["enable_trace_replay"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    from vllm import LLM

    llm = LLM(**kwargs)
    try:
        cohort_capacity = None
        if args.synchronized_decode:
            cohort_capacity = llm.collective_rpc(audit_worker_cohort_capacity)
            validate_cohort_capacity(cohort_capacity, args.batch_size)
        worker_attention_audit_before = llm.collective_rpc(
            audit_worker_attention_mode
        )
        validate_worker_attention_mode(
            worker_attention_audit_before,
            mode=args.mode,
            dummy_attention=args.dummy_attention,
        )
        if args.measure == "quality":
            measurements = evaluate_quality(
                llm,
                tokenizer,
                length=args.length,
                samples=args.samples,
                sample_offset=args.sample_offset,
                batch_size=args.batch_size,
            )
        else:
            measurements = evaluate_speed(
                llm,
                tokenizer,
                lengths=args.lengths,
                batch_size=args.batch_size,
                samples=speed_samples,
                decode_tokens=args.decode_tokens,
                repeats=args.repeats,
                seed=args.seed,
                fixed_decode_trace=args.fixed_decode_trace,
                retain_warmup_allocator=args.retain_warmup_allocator,
                diagnostic_prefill_profile=args.diagnostic_prefill_profile,
            )
        worker_attention_audit = llm.collective_rpc(audit_worker_attention_mode)
        validate_worker_attention_mode(
            worker_attention_audit,
            mode=args.mode,
            dummy_attention=args.dummy_attention,
            require_loaded_kimi_lod=True,
        )
        result = {
            "benchmark": "prolong",
            "cohort_capacity": cohort_capacity,
            "benchmark_identity": run_identity,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "measure": args.measure,
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "dummy_attention": args.dummy_attention,
            "timing_protocol": {
                "schema": 10,
                "clock": "vllm-request-metrics-plus-perf-counter-wall-check",
                "prefill_completion": (
                    "final-kimi-cache-event-before-first-token"
                    if args.measure == "speed"
                    else None
                ),
                "cohort_prefill_window": (
                    "earliest-scheduled-to-latest-first-token"
                    if args.measure == "speed"
                    else None
                ),
                "cohort_decode_window": (
                    "latest-first-token-to-latest-last-token"
                    if args.measure == "speed"
                    else None
                ),
                "warmup_generations_per_length": 1,
                "measured_repetitions": args.repeats,
                "warmup_allocator": (
                    "retain-with-8gib-headroom"
                    if args.retain_warmup_allocator
                    else "release-before-measurement"
                ),
                "decode_steps": "generated_tokens_minus_one",
                "preemptions": "recorded-and-rejected",
                "prefix_cache_hits": "recorded-and-rejected",
                "context_panel": (
                    "shared-max-length-prefix-cohort"
                    if args.measure == "speed" and args.fixed_decode_trace
                    else None
                ),
                "decode_inputs": (
                    "prolong-natural-trace-replay"
                    if args.measure == "speed" and args.fixed_decode_trace
                    else "model-greedy"
                    if args.measure == "speed"
                    else None
                ),
            },
            "argv": sys.argv,
            "hostname": platform.node(),
            "benchmark_environment": benchmark_environment(),
            "allow_experimental_environment": args.allow_experimental_environment,
            "decode_routes": None if args.mode == "full" else 8,
            "batch_size": args.batch_size,
            "speed_samples": speed_samples if args.measure == "speed" else None,
            "tensor_parallel_size": args.tensor_parallel_size,
            "decode_context_parallel_size": args.decode_context_parallel_size,
            "dcp_comm_backend": args.dcp_comm_backend,
            "kimi_gfx942_int4_moe": args.kimi_gfx942_int4_moe,
            "weight_cache": args.weight_cache,
            "weight_cache_id": args.weight_cache_id if args.weight_cache else None,
            "gpu_memory_utilization": gpu_memory_utilization,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "full_attention_backend": args.full_attention_backend,
            "effective_attention_config": kwargs["attention_config"],
            "worker_attention_audit": worker_attention_audit,
            "worker_attention_audit_before": worker_attention_audit_before,
            "max_model_len": kwargs["max_model_len"],
            "max_num_seqs": kwargs["max_num_seqs"],
            "enable_prefix_caching": kwargs["enable_prefix_caching"],
            "scheduler_chunk_tokens": SCHEDULER_CHUNK,
            "scheduler_budget_tokens": kwargs["max_num_batched_tokens"],
            "scheduler_cls": kwargs["scheduler_cls"],
            "decode_tokens": args.decode_tokens if args.measure == "speed" else None,
            "decode_input_policy": (
                "prolong-natural-trace-replay"
                if args.measure == "speed" and args.fixed_decode_trace
                else "model-greedy"
                if args.measure == "speed"
                else None
            ),
            "synchronized_decode": (
                args.synchronized_decode if args.measure == "speed" else None
            ),
            "seed": args.seed if args.measure == "speed" else None,
            "speculative_model": args.speculative_model,
            "num_speculative_tokens": (
                args.num_speculative_tokens if args.speculative_model else None
            ),
            "measurements": measurements,
        }
        write_json(args.output, result)
        print(args.output)
    finally:
        _close_speed_document_stream()
        close_llm(llm)


if __name__ == "__main__":
    main()

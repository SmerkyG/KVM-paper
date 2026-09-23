"""Measure ProLong prompt loss or matched long-context serving speed."""

from __future__ import annotations

import argparse
from collections import defaultdict
import functools
import hashlib
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from ._identity import benchmark_identity
from ._vllm import (
    MODES,
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
    parser.add_argument("--decode-tokens", type=int, default=1_025)
    parser.add_argument("--repeats", type=int, default=1)
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
        "--full-attention-backend",
        default="ROCM_AITER_UNIFIED_ATTN",
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
        "--prefill-variant",
        choices=("fast4", "fast8", "generic4", "generic8"),
        help="Benchmark-only K2/Qwen prefill route comparison",
    )
    parser.add_argument(
        "--prefill-exact-mass-coverage",
        type=float,
        help="Benchmark-only target fraction of full attention mass resolved exactly",
    )
    parser.add_argument(
        "--prefill-max-open-leaf-tokens",
        type=int,
        help="Benchmark-only cap on leaves in a centroid eligible for prefill refinement",
    )
    parser.add_argument("--prefill-route-count-bias", type=float, default=1.0)
    parser.add_argument("--prefill-route-large-count-penalty", type=float, default=0.0)
    parser.add_argument("--prefill-route-large-count-threshold", type=float, default=64.0)
    parser.add_argument(
        "--prefill-route-soft-count-pivot",
        type=float,
        help="Benchmark-only pivot in log(n) - log(1 + n/pivot) route score",
    )
    parser.add_argument(
        "--prefill-route-key-spread",
        choices=("total", "per_leaf"),
        help="Benchmark-only count-normalized key-spread routing",
    )
    parser.add_argument(
        "--prefill-route-exclude-singletons",
        action="store_true",
        help="Benchmark-only route filter for centroids with one archived leaf",
    )
    return parser.parse_args()


def configure_prefill_variant(
    model: Any,
    variant: str | None,
    exact_mass_coverage: float | None = None,
    max_open_leaf_tokens: int | None = None,
    route_count_bias: float = 1.0,
    route_large_count_penalty: float = 0.0,
    route_large_count_threshold: float = 64.0,
    route_soft_count_pivot: float | None = None,
    route_key_spread: str | None = None,
    route_exclude_singletons: bool = False,
) -> int:
    """Select a prefill route count without changing decode or tuned leaf tiles."""
    changed = 0
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        if variant is not None:
            pool.engine.prefill_two_level_topk = 8 if variant.endswith("8") else 4
            pool.engine.prefill_aiter_route_coarse = variant.startswith("fast")
        pool.engine.prefill_exact_mass_coverage = exact_mass_coverage
        pool.engine.prefill_max_open_leaf_tokens = max_open_leaf_tokens
        pool.engine.prefill_route_count_bias = route_count_bias
        pool.engine.prefill_route_large_count_penalty = route_large_count_penalty
        pool.engine.prefill_route_large_count_threshold = route_large_count_threshold
        pool.engine.prefill_route_soft_count_pivot = route_soft_count_pivot
        pool.engine.prefill_route_key_spread = route_key_spread
        pool.engine.prefill_route_exclude_singletons = route_exclude_singletons
        changed += 1
    return changed


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


def make_speed_prompts(
    tokenizer: Any,
    *,
    length: int,
    batch_size: int,
) -> tuple[list[dict[str, list[int]]], list[dict[str, Any]]]:
    """Build distinct, exact-length prompts without repeating documents."""

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
    ).shuffle(seed=SPEED_SHUFFLE_SEED, buffer_size=1_000)
    documents = iter(dataset)
    separator = tokenizer(SEPARATOR, add_special_tokens=False)["input_ids"]
    prompts = []
    metadata = []
    stream_index = -1
    while len(prompts) < batch_size:
        token_ids: list[int] = []
        source_indices = []
        while len(token_ids) < length:
            try:
                document = next(documents)
            except StopIteration as exc:
                raise RuntimeError(
                    "ProLong ended before all prompts were filled"
                ) from exc
            stream_index += 1
            document_ids = tokenizer(
                document["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            if not document_ids:
                continue
            if token_ids:
                remaining = length - len(token_ids)
                token_ids.extend(separator[: max(0, remaining - 1)])
            token_ids.extend(document_ids[: length - len(token_ids)])
            source_indices.append(stream_index)
        # Repetitive source can distort expert routing. Match the archived
        # panel's guard by consuming and replacing such a candidate.
        if _unique_block_ratio(token_ids) < 0.95:
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
    params: Any,
) -> tuple[
    float,
    float,
    float,
    tuple[tuple[int, ...], ...],
    dict[str, int],
]:
    before = speculative_counters(llm)
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    after = speculative_counters(llm)
    expected = int(params.max_tokens)
    if any(len(output.outputs[0].token_ids) != expected for output in outputs):
        raise RuntimeError("a speed request stopped before max_tokens")
    metrics = [output.metrics for output in outputs]
    if any(metric is None for metric in metrics):
        raise RuntimeError("vLLM did not return per-request timing metrics")
    scheduled = min(float(metric.scheduled_ts) for metric in metrics)
    first_token = max(float(metric.first_token_ts) for metric in metrics)
    last_token = max(float(metric.last_token_ts) for metric in metrics)
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
    params: Any,
    *,
    batch_size: int,
) -> tuple[
    float,
    float,
    float,
    tuple[tuple[int, ...], ...],
    dict[str, int],
    list[dict[str, int]],
]:
    """Run one fixed cohort in execution batches and sum its measurements."""

    elapsed = 0.0
    prefill = 0.0
    decode = 0.0
    token_ids: list[tuple[int, ...]] = []
    counters: defaultdict[str, int] = defaultdict(int)
    batch_counters = []
    for begin in range(0, len(prompts), batch_size):
        batch = timed_generate(llm, prompts[begin : begin + batch_size], params)
        batch_elapsed, batch_prefill, batch_decode, batch_tokens, counter_delta = batch
        elapsed += batch_elapsed
        prefill += batch_prefill
        decode += batch_decode
        token_ids.extend(batch_tokens)
        batch_counters.append(counter_delta)
        for name, value in counter_delta.items():
            counters[name] += value
    return elapsed, prefill, decode, tuple(token_ids), dict(counters), batch_counters


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
) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0,
        seed=seed,
        max_tokens=decode_tokens,
        detokenize=False,
        ignore_eos=True,
    )
    if samples % batch_size:
        raise ValueError("speed-samples must be divisible by batch-size")
    cohort_batches = samples // batch_size
    result = {}
    for length in lengths:
        prompts, prompt_metadata = make_speed_prompts(
            tokenizer,
            length=length,
            batch_size=samples,
        )
        *_, reference, _, _ = timed_generate_cohort(
            llm,
            prompts,
            params,
            batch_size=batch_size,
        )
        prefill_timings = []
        decode_timings = []
        cohort_prefill_timings = []
        cohort_decode_timings = []
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
            output_token_sha256.append(
                [token_digest(list(row)) for row in token_ids]
            )
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
        }
        if speculative_measurements:
            measurement.update(
                speculative_target_cycle_ms=statistics.median(
                    item["target_cycle_ms"] for item in speculative_measurements
                ),
                speculative_mean_acceptance_length=statistics.median(
                    item["mean_acceptance_length"]
                    for item in speculative_measurements
                ),
                speculative_draft_acceptance_rate=statistics.median(
                    item["draft_acceptance_rate"]
                    for item in speculative_measurements
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
    return result


def main() -> None:
    args = parse_args()
    run_identity = benchmark_identity()
    # This offline benchmark sends only its own callbacks to local vLLM
    # workers. vLLM 0.27 otherwise rejects both the existing route-ablation
    # callbacks and the attention-timing callbacks as non-msgpack objects.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if args.dummy_attention:
        if args.measure != "speed" or args.mode != "full":
            raise ValueError("--dummy-attention requires --measure speed --mode full")
        if args.speculative_model:
            raise ValueError("--dummy-attention does not support speculative decoding")
        os.environ["LOD_BENCHMARK_DUMMY_ATTENTION"] = "1"
    if (
        args.prefill_route_count_bias != 1.0
        or args.prefill_route_large_count_penalty != 0.0
        or args.prefill_route_soft_count_pivot is not None
        or args.prefill_route_key_spread is not None
        or args.prefill_route_exclude_singletons
    ) and args.prefill_variant not in {"generic4", "generic8"}:
        raise ValueError("route count-bias ablations require --prefill-variant generic4/generic8")
    if args.prefill_route_key_spread and args.prefill_route_exclude_singletons:
        raise ValueError("test key spread and singleton exclusion separately")
    if args.prefill_route_key_spread is not None and (
        args.prefill_route_count_bias != 1.0
        or args.prefill_route_large_count_penalty != 0.0
        or args.prefill_route_soft_count_pivot is not None
    ):
        raise ValueError("key-spread routing cannot be combined with count-bias ablations")
    if args.prefill_variant and args.mode == "full":
        raise ValueError("--prefill-variant requires a LoD mode")
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
        gpu_memory_utilization=gpu_memory_utilization,
        full_attention_backend=args.full_attention_backend,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        speculative_attention_backend=args.speculative_attention_backend,
    )
    if args.dummy_attention:
        kwargs["attention_config"] = {"backend": "CUSTOM"}
        kwargs["additional_config"] = {"lod_benchmark_dummy_attention": True}
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    from vllm import LLM

    llm = LLM(**kwargs)
    try:
        if (
            args.prefill_variant
            or args.prefill_exact_mass_coverage is not None
            or args.prefill_max_open_leaf_tokens is not None
        ):
            changed = llm.apply_model(
                functools.partial(
                    configure_prefill_variant,
                    variant=args.prefill_variant,
                    exact_mass_coverage=args.prefill_exact_mass_coverage,
                    max_open_leaf_tokens=args.prefill_max_open_leaf_tokens,
                    route_count_bias=args.prefill_route_count_bias,
                    route_large_count_penalty=args.prefill_route_large_count_penalty,
                    route_large_count_threshold=args.prefill_route_large_count_threshold,
                    route_soft_count_pivot=args.prefill_route_soft_count_pivot,
                    route_key_spread=args.prefill_route_key_spread,
                    route_exclude_singletons=args.prefill_route_exclude_singletons,
                )
            )
            if not changed or not all(count > 0 for count in changed):
                raise RuntimeError("prefill variant found no LoD attention layers")
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
            )
        result = {
            "benchmark": "prolong",
            "benchmark_identity": run_identity,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "measure": args.measure,
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "dummy_attention": args.dummy_attention,
            "prefill_variant": args.prefill_variant or "production",
            "prefill_exact_mass_coverage": args.prefill_exact_mass_coverage,
            "prefill_max_open_leaf_tokens": args.prefill_max_open_leaf_tokens,
            "prefill_route_count_bias": args.prefill_route_count_bias,
            "prefill_route_large_count_penalty": args.prefill_route_large_count_penalty,
            "prefill_route_large_count_threshold": args.prefill_route_large_count_threshold,
            "prefill_route_soft_count_pivot": args.prefill_route_soft_count_pivot,
            "prefill_route_key_spread": args.prefill_route_key_spread,
            "prefill_route_exclude_singletons": args.prefill_route_exclude_singletons,
            "decode_routes": (
                None
                if args.mode == "full"
                else 8 if os.environ.get("LOD_DECODE_TOP8", "1") == "1" else 4
            ),
            "batch_size": args.batch_size,
            "speed_samples": speed_samples if args.measure == "speed" else None,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "scheduler_chunk_tokens": 16_384,
            "scheduler_budget_tokens": kwargs["max_num_batched_tokens"],
            "scheduler_cls": kwargs["scheduler_cls"],
            "decode_tokens": args.decode_tokens if args.measure == "speed" else None,
            "seed": args.seed if args.measure == "speed" else None,
            "speculative_model": args.speculative_model,
            "num_speculative_tokens": (
                args.num_speculative_tokens if args.speculative_model else None
            ),
            "measurements": measurements,
        }
        final_identity = benchmark_identity()
        if final_identity != run_identity:
            raise RuntimeError(
                "benchmark source or runtime identity changed while the run was active; "
                "discard this result and rerun without modifying the environment"
            )
        write_json(args.output, result)
        print(args.output)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

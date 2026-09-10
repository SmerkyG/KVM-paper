"""Measure ProLong prompt loss or matched long-context serving speed."""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import Any

from ._vllm import MODES, close_llm, llm_kwargs, write_json

DATASET = "Seerkfang/prolong-64k-512-new"
DATASET_REVISION = "97295b7d7fe48dc0aa6ba373af3a8b9d945e505b"
QUALITY_SHUFFLE_SEED = 42
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
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=1_025)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--full-attention-backend",
        default="ROCM_AITER_UNIFIED_ATTN",
    )
    return parser.parse_args()


def token_digest(token_ids: list[int]) -> str:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_quality_prompts(
    tokenizer: Any,
    *,
    length: int,
    samples: int,
    sample_offset: int,
) -> tuple[list[dict[str, list[int]]], list[dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
    ).shuffle(seed=QUALITY_SHUFFLE_SEED, buffer_size=1_000)
    prompts = []
    metadata = []
    eligible = 0
    for stream_index, document in enumerate(dataset):
        declared_length = document.get("length")
        if declared_length is not None and int(declared_length) < length:
            continue
        token_ids = tokenizer(
            document["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=length,
            return_attention_mask=False,
        )["input_ids"]
        if len(token_ids) != length:
            continue
        if eligible >= sample_offset:
            prompts.append({"prompt_token_ids": token_ids})
            metadata.append(
                {
                    "stream_index": stream_index,
                    "tokens": length,
                    "token_sha256": token_digest(token_ids),
                }
            )
        eligible += 1
        if len(prompts) == samples:
            break
    if len(prompts) != samples:
        raise RuntimeError(f"found only {len(prompts)} sufficiently long documents")
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
) -> tuple[float, float, float, tuple[tuple[int, ...], ...]]:
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
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
    return elapsed, first_token - scheduled, last_token - first_token, token_ids


def evaluate_speed(
    llm: Any,
    tokenizer: Any,
    *,
    lengths: list[int],
    batch_size: int,
    decode_tokens: int,
    repeats: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0,
        max_tokens=decode_tokens,
        detokenize=False,
        ignore_eos=True,
    )
    result = {}
    for length in lengths:
        prompts, prompt_metadata = make_speed_prompts(
            tokenizer,
            length=length,
            batch_size=batch_size,
        )
        *_, reference = timed_generate(llm, prompts, params)
        total_timings = []
        prefill_timings = []
        decode_timings = []
        for _ in range(repeats):
            elapsed, prefill, decode, token_ids = timed_generate(llm, prompts, params)
            if token_ids != reference:
                raise RuntimeError("greedy output changed across identical speed runs")
            total_timings.append(elapsed)
            prefill_timings.append(prefill)
            decode_timings.append(decode)
        prefill = statistics.median(prefill_timings)
        decode = statistics.median(decode_timings)
        decode_steps = decode_tokens - 1
        result[str(length)] = {
            "prefill_seconds": prefill,
            "prefill_prompt_tokens_per_second": batch_size * length / prefill,
            "decode_ms_per_batch_step": 1_000.0 * decode / decode_steps,
            "decode_tokens_per_second": batch_size * decode_steps / decode,
            "prefill_timings_seconds": prefill_timings,
            "decode_timings_seconds": decode_timings,
            "total_timings_seconds": total_timings,
            "prompts": prompt_metadata,
        }
    return result


def main() -> None:
    args = parse_args()
    if min(args.length, args.samples, args.batch_size, args.repeats) < 1:
        raise ValueError("length, samples, batch-size, and repeats must be positive")
    if args.sample_offset < 0:
        raise ValueError("sample-offset must be nonnegative")
    if args.decode_tokens < 2:
        raise ValueError("decode-tokens must be at least two")

    from transformers import AutoTokenizer

    max_length = args.length if args.measure == "quality" else max(args.lengths)
    generation_tokens = 1 if args.measure == "quality" else args.decode_tokens
    kwargs = llm_kwargs(
        checkpoint=args.checkpoint,
        mode=args.mode,
        max_model_len=max_length + generation_tokens + 16,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        full_attention_backend=args.full_attention_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    from vllm import LLM

    llm = LLM(**kwargs)
    try:
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
                decode_tokens=args.decode_tokens,
                repeats=args.repeats,
            )
        result = {
            "benchmark": "prolong",
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "measure": args.measure,
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "batch_size": args.batch_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "scheduler_chunk_tokens": 16_384,
            "decode_tokens": args.decode_tokens if args.measure == "speed" else None,
            "measurements": measurements,
        }
        write_json(args.output, result)
        print(args.output)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

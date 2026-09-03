#!/usr/bin/env python3
"""Exercise multi-turn LOD requests with varied prompt sizes in one batch."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from transformers import AutoTokenizer

from vllm_engine_lifecycle import register_llm_shutdown, shutdown_registered_llms


MARKERS = (
    "amber falcon seven",
    "copper willow nine",
    "silver otter four",
    "violet cedar eight",
    "golden heron three",
    "indigo maple six",
    "crimson badger two",
    "ivory raven five",
)
INITIAL_LENGTHS = (512, 1024, 2048, 4096, 6144, 8192, 12288, 16384)
TURN_ONE_LENGTHS = (64, 128, 256, 512, 768, 1024, 1536, 2048)
TURN_TWO_LENGTHS = tuple(reversed(TURN_ONE_LENGTHS))
ORDERS = (
    tuple(range(8)),
    (7, 0, 5, 2, 6, 1, 4, 3),
    (3, 6, 1, 7, 0, 4, 2, 5),
)


def inspect_lod_model(model) -> dict[str, int]:
    counters = {
        "layers": 0,
        "installs": 0,
        "direct_prefills": 0,
        "batched_cached_prefills": 0,
        "batched_cached_prefill_rows": 0,
        "decode_calls": 0,
        "catch_up_batches": 0,
        "catch_up_rows": 0,
        "retained_reuses": 0,
    }
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        counters["layers"] += 1
        counters["installs"] += int(pool.install_count)
        counters["direct_prefills"] += int(pool.direct_prefill_calls)
        counters["batched_cached_prefills"] += int(
            pool.batched_cached_prefill_calls
        )
        counters["batched_cached_prefill_rows"] += int(
            pool.batched_cached_prefill_rows
        )
        counters["decode_calls"] += int(pool.decode_calls)
        counters["catch_up_batches"] += int(pool.catch_up_batches)
        counters["catch_up_rows"] += int(pool.catch_up_rows)
        counters["retained_reuses"] += int(pool.retained_reuse_count)
    return counters


def exact_segment(tokenizer, prefix: str, suffix: str, length: int) -> list[int]:
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    remaining = length - len(prefix_ids) - len(suffix_ids)
    if remaining < 0:
        raise ValueError(f"segment length {length} is too short for its framing")
    filler = tokenizer(
        " Routine archive material provides neutral background context.",
        add_special_tokens=False,
    )["input_ids"]
    filler_ids = (filler * ((remaining + len(filler) - 1) // len(filler)))[
        :remaining
    ]
    result = prefix_ids + filler_ids + suffix_ids
    if len(result) != length:
        raise AssertionError("failed to construct an exact-length prompt segment")
    return result


def initial_prompt(tokenizer, marker: str, length: int) -> list[int]:
    return exact_segment(
        tokenizer,
        (
            "System: You are testing independent conversation memory.\n"
            f"User: The private passphrase is {marker}. Remember it exactly.\n"
        ),
        "\nUser: What is my private passphrase? Reply with only the passphrase.\nAssistant:",
        length,
    )


def turn_segment(tokenizer, turn: int, length: int) -> list[int]:
    return exact_segment(
        tokenizer,
        f"\nUser: This is follow-up turn {turn} in the same conversation.\n",
        "\nUser: Repeat my original private passphrase only.\nAssistant:",
        length,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=("full", "lod"), default="lod")
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    prompts = [
        initial_prompt(tokenizer, marker, length)
        for marker, length in zip(MARKERS, INITIAL_LENGTHS)
    ]
    max_length = max(INITIAL_LENGTHS) + sum(
        max(lengths) for lengths in (TURN_ONE_LENGTHS, TURN_TWO_LENGTHS)
    ) + 3 * args.decode_tokens + 32
    llm_kwargs = dict(
        model=args.checkpoint,
        trust_remote_code=True,
        load_format=os.getenv("VLLM_WEIGHT_CACHE_LOAD_FORMAT", "ipc_cache"),
        dtype="bfloat16",
        max_model_len=max_length,
        max_num_seqs=len(MARKERS),
        max_num_batched_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=True,
        disable_log_stats=False,
    )
    if args.mode == "lod":
        llm_kwargs["attention_config"] = {"backend": "CUSTOM"}
    llm = register_llm_shutdown(LLM(**llm_kwargs))
    params = SamplingParams(
        temperature=0,
        max_tokens=args.decode_tokens,
        ignore_eos=True,
    )

    before = llm.apply_model(inspect_lod_model)[0] if args.mode == "lod" else {}
    rounds = []
    turn_lengths = (None, TURN_ONE_LENGTHS, TURN_TWO_LENGTHS)
    for round_index, order in enumerate(ORDERS):
        if round_index:
            lengths = turn_lengths[round_index]
            assert lengths is not None
            prompts = [
                prompt + turn_segment(tokenizer, round_index, lengths[index])
                for index, prompt in enumerate(prompts)
            ]

        started = time.perf_counter()
        outputs = llm.generate(
            [{"prompt_token_ids": prompts[index]} for index in order],
            params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
        round_rows = []
        for output, conversation_index in zip(outputs, order):
            token_ids = list(output.outputs[0].token_ids)
            if len(token_ids) != args.decode_tokens:
                raise RuntimeError("a chat request stopped before max_tokens")
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            marker = MARKERS[conversation_index]
            wrong_markers = [
                candidate
                for candidate in MARKERS
                if candidate != marker and candidate in text.lower()
            ]
            if wrong_markers:
                raise RuntimeError(
                    "conversation output contained another request's marker: "
                    f"conversation={conversation_index}, output={text!r}"
                )
            cached_tokens = int(getattr(output, "num_cached_tokens", 0) or 0)
            prompts[conversation_index].extend(token_ids)
            round_rows.append(
                {
                    "conversation": conversation_index,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "cached_tokens": cached_tokens,
                    "marker_hit": marker in text.lower(),
                    "output": text,
                }
            )
        after_round = (
            llm.apply_model(inspect_lod_model)[0] if args.mode == "lod" else {}
        )
        marker_hits = sum(int(row["marker_hit"]) for row in round_rows)
        if marker_hits < len(MARKERS) - 1:
            raise RuntimeError(
                f"only {marker_hits}/{len(MARKERS)} turn-{round_index} "
                "requests reproduced their retained marker"
            )
        if round_index:
            cache_hits = sum(
                int(row["cached_tokens"] > 0) for row in round_rows
            )
            # Qwen3.5 aligns attention pages with its 544-token recurrent
            # cache pages.  The deliberately short 512-token conversation
            # therefore has no complete reusable block on its first follow-up.
            if cache_hits < len(MARKERS) - 1:
                misses = [
                    int(row["conversation"])
                    for row in round_rows
                    if not row["cached_tokens"]
                ]
                raise RuntimeError(
                    f"only {cache_hits}/{len(MARKERS)} turn-{round_index} "
                    "requests reused native prefix-cache tokens; "
                    f"misses={misses}"
                )
        rounds.append(
            {
                "round": round_index,
                "order": list(order),
                "elapsed_seconds": elapsed,
                "requests": round_rows,
                "lod_counters": after_round,
            }
        )

    if args.mode == "lod":
        after = rounds[-1]["lod_counters"]
        layers = int(after["layers"])
        if layers <= 0:
            raise RuntimeError("vLLM did not attach any LOD pools")
        if (
            int(after["decode_calls"]) <= int(before["decode_calls"])
            and int(after["direct_prefills"]) <= int(before["direct_prefills"])
        ):
            raise RuntimeError("the multi-turn batch never executed LOD attention")
        # Authoritative caches are installed once during the initial prefill and
        # then retained/advanced across turns. Reinstalling on every round would
        # indicate that the LOD cache was discarded and rebuilt from native KV.
        expected_installs = layers * len(MARKERS)
        if int(after["installs"]) - int(before["installs"]) < expected_installs:
            raise RuntimeError(
                "not every initial chat request installed an LOD cache: "
                f"expected at least {expected_installs}, before={before}, after={after}"
            )
        # The 512-token initial request has no complete reusable physical block at
        # Qwen3.5's 544-token hybrid-cache granularity, so its first follow-up must
        # build one new row. Every other transition must reuse the retained row.
        expected_reuses = layers * (len(MARKERS) * (len(ORDERS) - 1) - 1)
        if (
            int(after["retained_reuses"]) - int(before["retained_reuses"])
            < expected_reuses
        ):
            raise RuntimeError(
                "completed LOD cache rows were not retained across chat turns: "
                f"expected at least {expected_reuses}, before={before}, after={after}"
            )

    result = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "initial_lengths": list(INITIAL_LENGTHS),
        "turn_one_lengths": list(TURN_ONE_LENGTHS),
        "turn_two_lengths": list(TURN_TWO_LENGTHS),
        "decode_tokens": args.decode_tokens,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "before": before,
        "rounds": rounds,
        "marker_hits": [
            sum(int(row["marker_hit"]) for row in round_data["requests"])
            for round_data in rounds
        ],
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_registered_llms()

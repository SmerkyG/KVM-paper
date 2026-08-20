#!/usr/bin/env python3
"""Evaluate a LongBench-v2 panel from the single A/B/C/D branch token.

Every request performs the normal long-context prefill, appends the common
assistant prefix ``The correct answer is (``, and restricts the next token to
the model's four single-token A/B/C/D choices.  This exposes a continuous
choice distribution while decoding only one token.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer


PROMPT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""
ASSISTANT_PREFIX = "The correct answer is ("
CHOICES = "ABCD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=262016)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--request-mode",
        choices=("batch", "parallel"),
        default="batch",
        help=(
            "Send each group as one ordered completions request (batch), or as "
            "independent concurrent requests (parallel)."
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def make_prompt(item: dict) -> str:
    return (
        PROMPT.replace("$DOC$", item["context"].strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def truncate_prompt(tokenizer, prompt: str, max_tokens: int) -> tuple[str, int, bool]:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_length = len(input_ids)
    if original_length <= max_tokens:
        return prompt, original_length, False
    left = max_tokens // 2
    input_ids = input_ids[:left] + input_ids[-(max_tokens - left) :]
    return tokenizer.decode(input_ids, skip_special_tokens=True), original_length, True


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(record["_id"]) for record in load_jsonl(path)}


def render_prompt(tokenizer, prompt: str, disable_thinking: bool) -> str:
    kwargs = {}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    return rendered + ASSISTANT_PREFIX


def normalize_choice_logprobs(raw: dict[str, float]) -> dict[str, float]:
    choice_values: dict[str, float] = {}
    for token, value in raw.items():
        stripped = token.strip()
        if stripped in CHOICES:
            choice_values[stripped] = float(value)
    missing = [choice for choice in CHOICES if choice not in choice_values]
    if missing:
        raise RuntimeError(
            f"completion logprobs omitted choices {missing}; returned tokens={list(raw)}"
        )
    return choice_values


def query_batch(
    client: OpenAI,
    args: argparse.Namespace,
    prompts: list[str],
    choice_token_ids: list[int],
) -> tuple[list[dict[str, float]], float]:
    if not prompts:
        raise ValueError("cannot query an empty prompt batch")
    error = None
    for attempt in range(args.retries):
        started = time.perf_counter()
        try:
            completion = client.completions.create(
                model=args.checkpoint,
                prompt=prompts,
                temperature=0.0,
                max_tokens=1,
                logprobs=4,
                extra_body={"allowed_token_ids": choice_token_ids},
            )
            by_index = {}
            for choice in completion.choices:
                top_logprobs = choice.logprobs.top_logprobs
                if not top_logprobs:
                    raise RuntimeError(
                        "completion returned no top-logprob distribution"
                    )
                by_index[int(choice.index)] = normalize_choice_logprobs(
                    top_logprobs[0]
                )
            if sorted(by_index) != list(range(len(prompts))):
                raise RuntimeError(
                    f"completion returned choice indices {sorted(by_index)} for "
                    f"{len(prompts)} prompts"
                )
            return (
                [by_index[index] for index in range(len(prompts))],
                time.perf_counter() - started,
            )
        except Exception as caught:
            error = caught
            if attempt + 1 < args.retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"request failed after {args.retries} attempts: {error}")


def query(
    client: OpenAI,
    args: argparse.Namespace,
    prompt: str,
    choice_token_ids: list[int],
) -> tuple[dict[str, float], float]:
    results, elapsed = query_batch(client, args, [prompt], choice_token_ids)
    return results[0], elapsed


def probabilities(logprobs: dict[str, float]) -> dict[str, float]:
    maximum = max(logprobs.values())
    weights = {choice: math.exp(value - maximum) for choice, value in logprobs.items()}
    denominator = sum(weights.values())
    return {choice: value / denominator for choice, value in weights.items()}


def correct_margin(logprobs: dict[str, float], answer: str) -> float:
    return logprobs[answer] - max(
        value for choice, value in logprobs.items() if choice != answer
    )


def js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    midpoint = {choice: 0.5 * (left[choice] + right[choice]) for choice in CHOICES}

    def kl(first: dict[str, float], second: dict[str, float]) -> float:
        return sum(
            first[choice] * math.log(first[choice] / second[choice])
            for choice in CHOICES
            if first[choice] > 0.0
        )

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def compare_records(reference_records: list[dict], candidate_records: list[dict]) -> dict:
    reference = {str(record["_id"]): record for record in reference_records}
    missing = [
        record["_id"]
        for record in candidate_records
        if str(record["_id"]) not in reference
    ]
    if missing:
        raise ValueError(f"reference output is missing {len(missing)} panel examples")
    pairs = [(reference[str(record["_id"])], record) for record in candidate_records]
    if not pairs:
        raise ValueError("candidate output contains no panel examples")
    divergences = [
        js_divergence(left["choice_probabilities"], right["choice_probabilities"])
        for left, right in pairs
    ]
    margin_deltas = [
        right["correct_margin"] - left["correct_margin"] for left, right in pairs
    ]
    margin_drops = [max(0.0, -value) for value in margin_deltas]
    agreements = sum(left["prediction"] == right["prediction"] for left, right in pairs)
    reference_wins = sum(
        bool(left["correct"]) and not bool(right["correct"])
        for left, right in pairs
    )
    candidate_wins = sum(
        bool(right["correct"]) and not bool(left["correct"])
        for left, right in pairs
    )
    return {
        "count": len(pairs),
        "reference_correct": sum(bool(left["correct"]) for left, _ in pairs),
        "candidate_correct": sum(bool(right["correct"]) for _, right in pairs),
        "prediction_agreement": agreements,
        "prediction_agreement_rate": agreements / len(pairs),
        "reference_correct_candidate_wrong": reference_wins,
        "candidate_correct_reference_wrong": candidate_wins,
        "mean_js_divergence": sum(divergences) / len(divergences),
        "max_js_divergence": max(divergences),
        "mean_correct_margin_delta": sum(margin_deltas) / len(margin_deltas),
        "mean_correct_margin_drop": sum(margin_drops) / len(margin_drops),
        "rms_correct_margin_delta": math.sqrt(
            sum(value * value for value in margin_deltas) / len(margin_deltas)
        ),
    }


def compare(reference_path: Path, candidate_records: list[dict]) -> dict:
    return compare_records(load_jsonl(reference_path), candidate_records)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    selection = load_jsonl(args.selection)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        selection = selection[: args.limit]
    selection_by_id = {str(record["_id"]): record for record in selection}
    if len(selection_by_id) != len(selection):
        raise ValueError("selection contains duplicate IDs")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    choice_token_ids = []
    prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    for choice in CHOICES:
        combined = tokenizer.encode(ASSISTANT_PREFIX + choice, add_special_tokens=False)
        if combined[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"choice {choice} changes tokenization of the answer prefix"
            )
        suffix = combined[len(prefix_ids) :]
        if len(suffix) != 1:
            raise ValueError(f"choice {choice} is not one token after the answer prefix")
        choice_token_ids.append(suffix[0])

    dataset = {
        str(item["_id"]): item
        for item in load_dataset("THUDM/LongBench-v2", split="train")
        if str(item["_id"]) in selection_by_id
    }
    missing = sorted(selection_by_id.keys() - dataset.keys())
    if missing:
        raise ValueError(f"dataset is missing {len(missing)} selected IDs")

    done = completed_ids(args.output)
    prepared = []
    for record in selection:
        record_id = str(record["_id"])
        if record_id in done:
            continue
        item = dataset[record_id]
        prompt, original_tokens, truncated = truncate_prompt(
            tokenizer, make_prompt(item), args.max_input_tokens
        )
        rendered = render_prompt(tokenizer, prompt, args.disable_thinking)
        prepared.append((record, item, rendered, original_tokens, truncated))
    prepared.sort(key=lambda value: int(value[0]["sent_input_tokens"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(args.output) if args.output.exists() else []
    client = OpenAI(
        base_url=args.base_url, api_key="local", timeout=args.request_timeout
    )
    run_started = time.perf_counter()
    with (
        args.output.open("a", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        for begin in range(0, len(prepared), args.workers):
            prepared_batch = prepared[begin : begin + args.workers]
            if args.request_mode == "batch":
                batch_logprobs, batch_elapsed = query_batch(
                    client,
                    args,
                    [entry[2] for entry in prepared_batch],
                    choice_token_ids,
                )
                resolved = [
                    (
                        record,
                        item,
                        original_tokens,
                        truncated,
                        choice_logprobs,
                        batch_elapsed / len(prepared_batch),
                        batch_elapsed,
                    )
                    for (
                        record,
                        item,
                        _,
                        original_tokens,
                        truncated,
                    ), choice_logprobs in zip(prepared_batch, batch_logprobs)
                ]
            else:
                pending = [
                    (
                        record,
                        item,
                        original_tokens,
                        truncated,
                        executor.submit(
                            query, client, args, rendered, choice_token_ids
                        ),
                    )
                    for record, item, rendered, original_tokens, truncated in prepared_batch
                ]
                resolved = []
                for record, item, original_tokens, truncated, future in pending:
                    choice_logprobs, elapsed = future.result()
                    resolved.append(
                        (
                            record,
                            item,
                            original_tokens,
                            truncated,
                            choice_logprobs,
                            elapsed,
                            elapsed,
                        )
                    )

            for (
                record,
                item,
                original_tokens,
                truncated,
                choice_logprobs,
                elapsed,
                batch_elapsed,
            ) in resolved:
                choice_probabilities = probabilities(choice_logprobs)
                prediction = max(CHOICES, key=choice_logprobs.__getitem__)
                result = {
                    "_id": item["_id"],
                    "panel_role": record.get("panel_role"),
                    "domain": item["domain"],
                    "sub_domain": item["sub_domain"],
                    "difficulty": item["difficulty"],
                    "length": item["length"],
                    "answer": item["answer"],
                    "prediction": prediction,
                    "correct": prediction == item["answer"],
                    "choice_logprobs": choice_logprobs,
                    "choice_probabilities": choice_probabilities,
                    "correct_margin": correct_margin(
                        choice_logprobs, item["answer"]
                    ),
                    "original_input_tokens": original_tokens,
                    "sent_input_tokens": min(original_tokens, args.max_input_tokens),
                    "truncated": truncated,
                    "elapsed_seconds": elapsed,
                    "batch_elapsed_seconds": batch_elapsed,
                    "request_mode": args.request_mode,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                existing.append(result)
                print(
                    json.dumps(
                        {
                            "id": item["_id"],
                            "correct": result["correct"],
                            "margin": round(result["correct_margin"], 4),
                            "tokens": result["sent_input_tokens"],
                            "batch_seconds": round(batch_elapsed, 2),
                        }
                    ),
                    flush=True,
                )

    summary = {
        "count": len(existing),
        "correct": sum(bool(record["correct"]) for record in existing),
        "accuracy": (
            sum(bool(record["correct"]) for record in existing) / len(existing)
            if existing
            else None
        ),
        "request_seconds": sum(record["elapsed_seconds"] for record in existing),
        "run_wall_seconds": time.perf_counter() - run_started,
    }
    if args.reference_output is not None:
        summary["comparison"] = compare(args.reference_output, existing)
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

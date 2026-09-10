"""Evaluate an OpenAI-compatible endpoint on official LongBench v2."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DATASET = "THUDM/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
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
ANSWER_PATTERNS = (
    re.compile(r"The correct answer is \(([A-D])\)", re.IGNORECASE),
    re.compile(r"The correct answer is ([A-D])", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=131_072)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--request-timeout", type=float, default=3_600.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument(
        "--sort-by-input-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--guided-answer-choice",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def make_prompt(item: dict[str, Any]) -> str:
    return (
        PROMPT.replace("$DOC$", item["context"].strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def truncate_prompt(
    tokenizer: Any, prompt: str, max_tokens: int
) -> tuple[str, int, bool]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_length = len(token_ids)
    if original_length <= max_tokens:
        return prompt, original_length, False
    left = max_tokens // 2
    retained = token_ids[:left] + token_ids[-(max_tokens - left) :]
    return tokenizer.decode(retained, skip_special_tokens=True), original_length, True


def extract_answer(response: str) -> str | None:
    response = response.replace("*", "")
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(response)
        if match is not None:
            return match.group(1).upper()
    return None


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        correct = int(record["correct"])
        for name in (
            "overall",
            f"difficulty:{record['difficulty']}",
            f"length:{record['length']}",
            f"domain:{record['domain']}",
        ):
            groups[name][0] += correct
            groups[name][1] += 1
    return {
        name: {
            "correct": values[0],
            "count": values[1],
            "accuracy": values[0] / values[1],
        }
        for name, values in sorted(groups.items())
    }


def load_records(output: Path, checkpoint: str) -> list[dict[str, Any]]:
    if not output.exists():
        return []
    records = [
        json.loads(line) for line in output.read_text().splitlines() if line.strip()
    ]
    if any(record.get("checkpoint") != checkpoint for record in records):
        raise ValueError(
            f"{output} contains records for a different checkpoint; "
            "choose a new output path"
        )
    return records


def query(client: Any, args: argparse.Namespace, prompt: str) -> tuple[str, float]:
    from openai import OpenAIError

    error: Exception | None = None
    for attempt in range(args.retries):
        started = time.perf_counter()
        try:
            extra_body: dict[str, Any] = {}
            if args.disable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            if args.guided_answer_choice:
                extra_body["structured_outputs"] = {
                    "choice": [f"The correct answer is ({answer})" for answer in "ABCD"]
                }
            completion = client.chat.completions.create(
                model=args.checkpoint,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=args.max_output_tokens,
                extra_body=extra_body or None,
            )
            content = completion.choices[0].message.content or ""
            return content, time.perf_counter() - started
        except OpenAIError as caught:
            error = caught
            if attempt + 1 < args.retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"request failed after {args.retries} attempts: {error}"
    ) from error


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if min(args.workers, args.retries, args.max_input_tokens) < 1:
        raise ValueError("workers, retries, and max-input-tokens must be positive")
    if args.warmup_batches < 0:
        raise ValueError("warmup-batches must be nonnegative")

    from datasets import load_dataset
    from openai import OpenAI
    from transformers import AutoTokenizer

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    dataset = list(
        load_dataset(
            DATASET,
            revision=DATASET_REVISION,
            split="train",
        )
    )[args.shard_index :: args.num_shards]
    records = load_records(args.output, args.checkpoint)
    completed = {record["_id"] for record in records}
    pending = []
    for position, item in enumerate(dataset, start=1):
        if item["_id"] in completed:
            continue
        prompt, original_tokens, truncated = truncate_prompt(
            tokenizer,
            make_prompt(item),
            args.max_input_tokens,
        )
        pending.append(
            (
                position,
                item,
                prompt,
                original_tokens,
                truncated,
                min(original_tokens, args.max_input_tokens),
            )
        )
    if args.sort_by_input_length:
        pending.sort(key=lambda prepared: prepared[-1])
    if args.limit is not None:
        pending = pending[: args.limit]

    client = OpenAI(
        base_url=args.base_url,
        api_key="local",
        timeout=args.request_timeout,
    )
    if args.warmup_batches and pending:
        warmup = pending[: args.workers]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for _ in range(args.warmup_batches):
                futures = [
                    executor.submit(query, client, args, prepared[2])
                    for prepared in warmup
                ]
                for future in futures:
                    future.result()

    started = time.perf_counter()
    with (
        args.output.open("a", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        for begin in range(0, len(pending), args.workers):
            batch = [
                (*prepared, executor.submit(query, client, args, prepared[2]))
                for prepared in pending[begin : begin + args.workers]
            ]
            for (
                position,
                item,
                _prompt,
                original_tokens,
                truncated,
                sent_tokens,
                future,
            ) in batch:
                response, elapsed = future.result()
                prediction = extract_answer(response)
                record = {
                    "_id": item["_id"],
                    "checkpoint": args.checkpoint,
                    "domain": item["domain"],
                    "sub_domain": item["sub_domain"],
                    "difficulty": item["difficulty"],
                    "length": item["length"],
                    "answer": item["answer"],
                    "prediction": prediction,
                    "correct": prediction == item["answer"],
                    "response": response,
                    "original_input_tokens": original_tokens,
                    "sent_input_tokens": sent_tokens,
                    "truncated": truncated,
                    "elapsed_seconds": elapsed,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(
                    json.dumps(
                        {
                            "progress": f"{position}/{len(dataset)}",
                            "id": item["_id"],
                            "correct": record["correct"],
                            "tokens": sent_tokens,
                            "seconds": round(elapsed, 2),
                        }
                    ),
                    flush=True,
                )

    result = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "checkpoint": args.checkpoint,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "workers": args.workers,
        "new_run_wall_seconds": time.perf_counter() - started,
        "metrics": summarize(records),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

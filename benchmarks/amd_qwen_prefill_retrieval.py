"""Score long-range retrieval from Qwen's first-token prefill logits."""

import argparse
import gzip
import hashlib
import json
import math
import random
import time
from pathlib import Path

from ._identity import benchmark_identity
from ._vllm import close_llm, default_gpu_memory_utilization, llm_kwargs
from .prolong import make_speed_prompts, token_digest

CHECKPOINT = "Qwen/Qwen3.8-27B-FP8"
CONTEXT = 65_536
LABELS = "ABCD"
BLOCKS = (1024, 16384)
LOOKBACK = 256
DISTANCES = {"local-both": 512, "local-16k": 8192, "remote-both": 32768}
WORDS = (
    "amber", "bamboo", "cedar", "delta", "ember", "frost", "granite", "harbor",
    "ivory", "juniper", "lemon", "maple", "onyx", "pebble", "quartz", "velvet",
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def prepare(tokenizer, output, samples, seed):
    if samples < 4 or samples % len(LABELS):
        raise ValueError("Use a multiple of four examples to balance the answer labels")
    haystacks, provenance = make_speed_prompts(tokenizer, length=CONTEXT, batch_size=samples)
    rng = random.Random(seed)
    answers = list(range(len(LABELS))) * (samples // len(LABELS))
    rng.shuffle(answers)
    marker = "INVENTORY_LOOKUP_PROMPT_MARKER"
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    prefix, suffix = chat.split(marker)

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    label_tokens = [encode(label) for label in LABELS]
    if any(len(tokens) != 1 for tokens in label_tokens) or len(set(map(tuple, label_tokens))) != len(LABELS):
        raise ValueError("Expected four distinct single-token labels")
    labels = {label: tokens[0] for label, tokens in zip(LABELS, label_tokens, strict=True)}
    preamble = encode(prefix + "Use the inventory records to answer the final lookup.\n\n")
    records = []
    for index, (haystack, correct) in enumerate(zip(haystacks, answers, strict=True)):
        batches = rng.sample(range(100000, 999999), 2)
        shelves = rng.sample(range(100000, 999999), 2)
        keys = [(batch, shelf) for batch in batches for shelf in shelves]
        rng.shuffle(keys)
        options = rng.sample(WORDS, len(LABELS))
        lookup = [
            f"Batch {batch}, shelf {shelf}, codeword {word}."
            for (batch, shelf), word in zip(keys, options, strict=True)
        ]
        rng.shuffle(lookup)
        evidence = encode("\n\nInventory records\n" + "\n".join(lookup) + "\n\n")
        batch, shelf = keys[correct]
        choices = "\n".join(f"{label}. {word}" for label, word in zip(LABELS, options, strict=True))
        question = encode(
            f"\n\nWhich codeword is listed for batch {batch} and shelf {shelf}?\n{choices}\n"
            "Answer using exactly one letter, A, B, C, or D." + suffix
        )
        filler_len = CONTEXT - len(preamble) - len(evidence) - len(question)
        filler = haystack["prompt_token_ids"][:filler_len]
        for band, distance in DISTANCES.items():
            evidence_end = CONTEXT - distance
            evidence_start = evidence_end - len(evidence)
            position = evidence_start - len(preamble)
            if not 0 <= position <= filler_len:
                raise ValueError("Evidence falls outside the haystack")
            tokens = preamble + filler[:position] + evidence + filler[position:] + question
            if len(tokens) != CONTEXT:
                raise ValueError("Wrong context length")
            local = tuple(evidence_start >= CONTEXT - block - LOOKBACK for block in BLOCKS)
            expected = {"local-both": (True, True), "local-16k": (False, True), "remote-both": (False, False)}
            if local != expected[band]:
                raise ValueError(f"Bad evidence placement for {band}")
            records.append({
                "example": index, "band": band, "gold": LABELS[correct],
                "evidence_start": evidence_start, "evidence_end": evidence_end,
                "question_start": CONTEXT - len(question),
                "token_sha256": token_digest(tokens), "prompt_token_ids": tokens,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with gzip.open(temp, "wt", compresslevel=3) as handle:
        json.dump({"seed": seed, "labels": labels, "records": records, "haystacks": provenance}, handle)
    temp.replace(output)
    print(f"Froze {len(records)} prompts at {output}", flush=True)


def score_cohort(args):
    from vllm import LLM, SamplingParams

    with gzip.open(args.cohort, "rt") as handle:
        cohort = json.load(handle)
    rows = [row for row in cohort["records"] if args.start <= row["example"] < args.stop]
    if args.start < 0 or args.start >= args.stop or len(rows) != (args.stop - args.start) * len(DISTANCES):
        raise ValueError("Incorrect cohort shard")
    settings = llm_kwargs(
        checkpoint=CHECKPOINT, mode=args.mode, max_model_len=CONTEXT + 16, batch_size=1,
        tensor_parallel_size=1, full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
        gpu_memory_utilization=default_gpu_memory_utilization(CHECKPOINT, args.mode),
    )
    labels = cohort["labels"]
    params = SamplingParams(temperature=0, max_tokens=1, seed=0, detokenize=False,
                            logprob_token_ids=list(labels.values()))
    llm = LLM(**settings)
    report = {
        "stage": "running", "mode": args.mode, "checkpoint": CHECKPOINT,
        "benchmark_identity": benchmark_identity(),
        "scheduler_chunk_tokens": settings["long_prefill_token_threshold"],
        "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "records": [],
    }
    try:
        for row in rows:
            started = time.monotonic()
            tokens = row["prompt_token_ids"]
            (result,) = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
            (choice,) = result.outputs
            (probabilities,) = choice.logprobs
            if len(choice.token_ids) != 1 or token_digest(result.prompt_token_ids) != row["token_sha256"]:
                raise ValueError("Unexpected generation length or prompt token IDs")
            scores = {label: probabilities[token].logprob for label, token in labels.items()}
            if not all(math.isfinite(value) for value in scores.values()):
                raise ValueError("Nonfinite answer log probabilities")
            gold = row["gold"]
            prediction = max(scores, key=scores.get)
            margin = scores[gold] - max(score for label, score in scores.items() if label != gold)
            report["records"].append({
                **{key: value for key, value in row.items() if key != "prompt_token_ids"},
                "scores": scores, "prediction": prediction, "gold_margin": margin,
                "correct": prediction == gold, "elapsed_s": time.monotonic() - started,
            })
            write_json(args.output, report)
            print(row["example"], row["band"], prediction, gold, round(margin, 3), flush=True)
        report["stage"] = "complete"
        write_json(args.output, report)
    finally:
        close_llm(llm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--samples", type=int, default=48)
    prep.add_argument("--seed", type=int, default=47283)
    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=("full", "two-tier"), required=True)
    run.add_argument("--cohort", type=Path, required=True)
    run.add_argument("--start", type=int, required=True)
    run.add_argument("--stop", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
        prepare(tokenizer, args.output, args.samples, args.seed)
    else:
        score_cohort(args)


if __name__ == "__main__":
    main()

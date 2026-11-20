"""Freeze Qwen reviewer-ablation inputs before examining predictions."""

import argparse
import hashlib
import json
from pathlib import Path
import random
from collections import defaultdict
from importlib import metadata

from .amd_threshold_prefill_quality_panel import write
from .amd_threshold_ruler_panel import MODELS, TASKS, digest
from .longbench_v2 import DATASET, DATASET_REVISION, make_prompt, truncate_prompt


CHECKPOINT = MODELS["qwen"]
CAP = 131072
COUNT = 32
SEED = 42
MAX_MODEL_LEN = CAP + 1024


def token_hash(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def render(tokenizer, messages, *, assistant_prefix):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, enable_thinking=False, return_dict=False,
        add_generation_prompt=not assistant_prefix, continue_final_message=assistant_prefix,
    )


def ruler_rows(tokenizer, task_name, length):
    import numpy as np
    import requests
    import torch
    from lm_eval.tasks import TaskManager
    from .ruler_lm_eval import OBSOLETE_HOTPOTQA_URL, _HotpotQAResponse

    if metadata.version("lm-eval") != "0.4.13":
        raise ValueError("RULER inputs must use lm-eval 0.4.13")
    original_get = requests.get

    def get_with_hotpotqa_fallback(url, *args, **kwargs):
        if url == OBSOLETE_HOTPOTQA_URL:
            return _HotpotQAResponse()
        return original_get(url, *args, **kwargs)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    manager = TaskManager(metadata={"tokenizer": CHECKPOINT, "max_seq_lengths": [length]})
    requests.get = get_with_hotpotqa_fallback
    try:
        task = manager.load([task_name])["tasks"][task_name]
    finally:
        requests.get = original_get
    config = task.config.generation_kwargs
    rows = []
    for index, doc in enumerate(task.test_docs()):
        if index == COUNT:
            break
        if doc["max_length"] != length:
            raise ValueError(f"Wrong RULER length for {task_name}/{index}")
        messages = [{"role": "user", "content": task.doc_to_text(doc)},
                    {"role": "assistant", "content": doc["gen_prefix"]}]
        ids = render(tokenizer, messages, assistant_prefix=True)
        if len(ids) + config["max_gen_toks"] > MAX_MODEL_LEN:
            raise ValueError(f"Rendered RULER prompt exceeds model capacity {task_name}/{index}")
        rows.append({
            "id": index, "prompt_tokens": ids, "input_sha256": token_hash(ids),
            "targets": doc["outputs"], "max_tokens": config["max_gen_toks"],
            "stop": config.get("until") or None,
        })
    return rows


def longbench_rows(tokenizer, *, untruncated_only=False, count=COUNT, excluded_ids=frozenset()):
    from datasets import load_dataset

    dataset = list(load_dataset(DATASET, revision=DATASET_REVISION, split="train"))
    if len(dataset) != 503:
        raise ValueError("Expected all 503 rows in the pinned LongBench v2 split")
    strata = defaultdict(list)
    original_lengths = {}
    dataset = [item for item in dataset if item["_id"] not in excluded_ids]
    if untruncated_only:
        for start in range(0, len(dataset), 8):
            batch = dataset[start:start + 8]
            tokens = tokenizer([make_prompt(item) for item in batch], add_special_tokens=False, verbose=False)
            original_lengths.update((item["_id"], len(ids))
                                    for item, ids in zip(batch, tokens["input_ids"], strict=True))
    for item in dataset:
        if untruncated_only and original_lengths[item["_id"]] > CAP:
            continue
        strata[item["domain"], item["length"]].append(item)
    rng = random.Random(SEED)
    for rows in strata.values():
        rows.sort(key=lambda row: row["_id"])
        rng.shuffle(rows)
    ranks = sorted(strata, key=lambda key: (-len(strata[key]), key))
    eligible = sum(map(len, strata.values()))
    if len(ranks) > count or eligible < count:
        raise ValueError("Cannot allocate one LongBench example per stratum")
    quotas = {key: count * len(strata[key]) / eligible for key in ranks}
    sizes = {key: len(strata[key]) for key in ranks}
    selected = []
    # Proportional allocation with one example in every nonempty stratum.
    for key in ranks:
        selected.append(strata[key].pop())
    while len(selected) < count:
        key = max((k for k in ranks if strata[k]), key=lambda k: quotas[k] - sizes[k] + len(strata[k]))
        selected.append(strata[key].pop())
    rows = []
    for item in selected:
        prompt = make_prompt(item)
        if untruncated_only:
            original, truncated = original_lengths[item["_id"]], False
        else:
            prompt, original, truncated = truncate_prompt(tokenizer, prompt, CAP)
        ids = render(tokenizer, [{"role": "user", "content": prompt}], assistant_prefix=False)
        if len(ids) + 32 > MAX_MODEL_LEN:
            raise ValueError(f"Rendered LongBench prompt exceeds model capacity {item['_id']}")
        rows.append({
            "id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"], "length": item["length"], "answer": item["answer"],
            "prompt_tokens": ids, "input_sha256": token_hash(ids), "max_tokens": 32,
            "truncated": truncated, "original_tokens": original, "cap_tokens": CAP,
        })
    return sorted(rows, key=lambda row: (len(row["prompt_tokens"]), row["id"])), eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "longbench-v2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ruler-length", type=int, default=CAP)
    parser.add_argument("--untruncated-only", action="store_true")
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--exclude", type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True, local_files_only=True)
    longbench = args.task == "longbench-v2"
    if args.untruncated_only and not longbench:
        parser.error("--untruncated-only requires LongBench v2")
    if not longbench and (args.count != COUNT or args.exclude):
        parser.error("RULER uses the fixed first-32 cohort")
    if longbench:
        if args.exclude:
            excluded = json.loads(args.exclude.read_text())
            excluded_ids = {r["id"] for r in excluded["rows"]}
        else:
            excluded_ids = frozenset()
        rows, eligible = longbench_rows(
            tokenizer, untruncated_only=args.untruncated_only, count=args.count, excluded_ids=excluded_ids,
        )
    else:
        rows = ruler_rows(tokenizer, args.task, args.ruler_length)
    if len(rows) != args.count or len({row["id"] for row in rows}) != args.count:
        raise ValueError(f"Expected {args.count} unique inputs for {args.task}")
    write(args.output, {
        "task": args.task, "checkpoint": CHECKPOINT, "count": args.count, "seed": SEED,
        "context_setting": CAP if longbench else args.ruler_length,
        "protocol": {"dataset": DATASET, "revision": DATASET_REVISION,
                     "truncation": "retain first and last halves of the prompt above 131072 tokens",
                     "sample": "domain by length, at least one per nonempty stratum, then proportional allocation",
                     "untruncated_only": args.untruncated_only, "eligible": eligible,
                     "exclude_sha256": digest(args.exclude) if args.exclude else None}
        if longbench else {"dataset": "lm-eval RULER", "version": "0.4.13", "sample": "first 32 in task order"},
        "rows": rows,
    })
    print(json.dumps({"task": args.task, "count": args.count, "sha256": digest(args.output),
                      "min_tokens": min(len(r["prompt_tokens"]) for r in rows),
                      "max_tokens": max(len(r["prompt_tokens"]) for r in rows)}), flush=True)


if __name__ == "__main__":
    main()

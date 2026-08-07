#!/usr/bin/env python3
"""Paired NIAH evaluation of full GPTAlpha2 and its top-k LOD graft."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from scripts.compare_gptalpha_lod_loss import load_model


def generate_documents(task: str, checkpoint: str, length: int) -> list[dict]:
    random.seed(0)
    np.random.seed(1234)
    from lm_eval.tasks.ruler.niah_utils import (
        niah_single_1,
        niah_single_2,
        niah_single_3,
    )

    generator = {
        "niah_single_1": niah_single_1,
        "niah_single_2": niah_single_2,
        "niah_single_3": niah_single_3,
    }[task]
    standard_lengths = [4096, 8192, 16384, 32768]
    lengths = [value for value in standard_lengths if value <= length]
    dataset = generator(max_seq_lengths=lengths, pretrained=checkpoint)["test"]
    return [doc for doc in dataset if int(doc["max_length"]) == length]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="featherless-ai/kvmpaper_gptalpha_120M"
    )
    parser.add_argument("--mode", choices=("full", "two_level"), required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("niah_single_1", "niah_single_2", "niah_single_3"),
        default=("niah_single_1", "niah_single_2", "niah_single_3"),
    )
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = load_model(args.checkpoint, args.mode, args.two_level_topk, device)

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.mode}_rank_{rank:02d}.jsonl"
    with output_path.open("w") as handle, torch.inference_mode():
        for task in args.tasks:
            documents = generate_documents(task, args.checkpoint, args.length)
            for doc in documents[: args.samples][rank::world_size]:
                prompt = doc["input"] + " " + doc["gen_prefix"]
                input_ids = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(device)
                generated = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                response = tokenizer.decode(
                    generated[0, input_ids.size(1) :],
                    skip_special_tokens=True,
                )
                target = str(doc["outputs"][0])
                record = {
                    "checkpoint": args.checkpoint,
                    "mode": args.mode,
                    "task": task,
                    "index": int(doc["index"]),
                    "length": int(doc["max_length"]),
                    "two_level_topk": (
                        args.two_level_topk if args.mode == "two_level" else None
                    ),
                    "target": target,
                    "response": response,
                    "exact": target.lower() in response.lower(),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"rank={rank} mode={args.mode} task={task} "
                    f"index={doc['index']} exact={record['exact']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

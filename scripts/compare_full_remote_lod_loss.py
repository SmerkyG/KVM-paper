#!/usr/bin/env python3
"""Compare exact full-remote and two-level LOD CE on paired text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("full_remote", "two_level"), required=True)
    parser.add_argument("--dataset", default="SmerkyG/dclm-10B")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(
    checkpoint: Path,
    mode: str,
    device: torch.device,
    two_level_topk: int,
):
    config = MixerConfig.from_pretrained(checkpoint)
    if mode == "full_remote":
        config.token_mixer_class_path = (
            "model.kvm_split_full_attention_mixer.SequenceMixer"
        )
    else:
        config.token_mixer_class_path = "model.kvm_two_level_mixer.SequenceMixer"
        config.kvm_use_merge_gate_keys = 0
        config.kvm_use_merge_gate_values = 0
        config.kvm_use_vlens = 0
    model = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    if mode == "two_level":
        for block in model.transformer.h:
            block.attn.two_level_topk = two_level_topk
    return model


def select_sequences(
    tokenizer,
    dataset_name: str,
    sequence_length: int,
    samples: int,
    rank: int,
    world_size: int,
) -> list[tuple[int, torch.Tensor]]:
    dataset = load_dataset(dataset_name, split="train", streaming=False).shuffle(seed=0)
    minimum_tokens = sequence_length + 1
    selected_texts: list[tuple[int, str]] = []
    for document in dataset:
        token_count = document.get("token_count")
        if token_count is not None and int(token_count) < minimum_tokens:
            continue
        sample = len(selected_texts)
        selected_texts.append((sample, document["text"]))
        if len(selected_texts) == samples:
            break
    if len(selected_texts) != samples:
        raise RuntimeError(
            f"found only {len(selected_texts)} sufficiently long documents"
        )
    shard = selected_texts[rank::world_size]
    encoded = tokenizer(
        [text for _, text in shard],
        add_special_tokens=False,
        truncation=True,
        max_length=minimum_tokens,
        return_attention_mask=False,
    )["input_ids"]
    sequences = [
        (sample, torch.tensor(input_ids, dtype=torch.long))
        for (sample, _), input_ids in zip(shard, encoded, strict=True)
    ]
    if any(int(sequence.numel()) != minimum_tokens for _, sequence in sequences):
        raise RuntimeError("dataset token_count did not match the evaluation tokenizer")
    return sequences


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    sequences = select_sequences(
        tokenizer,
        args.dataset,
        args.sequence_length,
        args.samples,
        rank,
        world_size,
    )
    model = load_model(checkpoint, args.mode, device, args.two_level_topk)

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.mode}_rank_{rank:02d}.jsonl"
    with output_path.open("w") as handle, torch.inference_mode():
        for sample, sequence in sequences:
            sequence = sequence.to(device)
            result = model(
                input_ids=sequence[:-1].unsqueeze(0),
                labels=sequence[1:].unsqueeze(0),
                return_logits=False,
            )
            record = {
                "mode": args.mode,
                "two_level_topk": (
                    args.two_level_topk if args.mode == "two_level" else None
                ),
                "sample": sample,
                "tokens": args.sequence_length,
                "loss": float(result.loss.item()),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"rank={rank} mode={args.mode} sample={sample} "
                f"loss={record['loss']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

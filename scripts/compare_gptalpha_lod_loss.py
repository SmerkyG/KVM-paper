#!/usr/bin/env python3
"""Compare paper GPTAlpha2 with its inference-only top-k LOD graft."""

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
    parser.add_argument(
        "--checkpoint", default="featherless-ai/kvmpaper_gptalpha_120M"
    )
    parser.add_argument("--mode", choices=("full", "two_level"), required=True)
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(checkpoint: str, mode: str, topk: int, device: torch.device):
    config = MixerConfig.from_pretrained(checkpoint)
    if mode == "full":
        config.token_mixer_class_path = "model.gptalpha_mixer.SequenceMixer"
    else:
        config.token_mixer_class_path = (
            "model.gptalpha_two_level_mixer.SequenceMixer"
        )
        config.kvm_value_residual_mode = config.gptalpha_value_residual_mode
        config.kvm_token_shift_mode = config.gptalpha_token_shift_mode
        config.kvm_use_merge_gate_keys = 0
        config.kvm_use_merge_gate_values = 0
        config.kvm_use_head_temps = 0
        config.kvm_use_vlens = 0
        config.state_budget_mode = "power_law"
        config.state_growth_factor = 16.0
        config.state_growth_exponent = 0.5
        config.state_round_down = 1
        config.state_min_len = 256
        config.n_max_d_chunks = 10000
        config.n_bswa_chunks = 2
    model = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    if mode == "two_level":
        for block in model.transformer.h:
            block.attn.two_level_topk = topk
    return model


def select_sequences(
    tokenizer,
    dataset_name: str,
    sequence_length: int,
    samples: int,
    rank: int,
    world_size: int,
) -> list[tuple[int, torch.Tensor]]:
    # Draw from the far end of the paper training corpus's deterministic
    # shuffle.  The 3B-token run consumed the front of this ordering, so this
    # gives a stable, practically held-out long-document slice.
    dataset = load_dataset(dataset_name, split="train", streaming=False).shuffle(
        seed=42
    )
    minimum_tokens = sequence_length + 1
    selected_texts: list[tuple[int, str]] = []
    for dataset_index in range(len(dataset) - 1, -1, -1):
        document = dataset[dataset_index]
        token_count = document.get("length")
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

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    sequences = select_sequences(
        tokenizer,
        args.dataset,
        args.sequence_length,
        args.samples,
        rank,
        world_size,
    )
    model = load_model(args.checkpoint, args.mode, args.two_level_topk, device)

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
                "checkpoint": args.checkpoint,
                "mode": args.mode,
                "sample": sample,
                "tokens": args.sequence_length,
                "two_level_topk": args.two_level_topk if args.mode == "two_level" else None,
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

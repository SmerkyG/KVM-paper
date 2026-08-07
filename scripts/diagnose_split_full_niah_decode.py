#!/usr/bin/env python3
"""Compare cached and recomputed split-full decoding on a RULER NIAH prompt."""

from __future__ import annotations

import argparse
import math
import random
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=("niah_single_2", "niah_single_3"), default="niah_single_3")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument(
        "--mixer",
        choices=("checkpoint", "split_full"),
        default="checkpoint",
    )
    parser.add_argument("--state-temperature-factor", type=float, default=1.0)
    return parser.parse_args()


def generate_document(task: str, checkpoint: Path, length: int, sample: int) -> dict:
    random.seed(0)
    np.random.seed(1234)
    from lm_eval.tasks.ruler.niah_utils import niah_single_2, niah_single_3

    generator = {"niah_single_2": niah_single_2, "niah_single_3": niah_single_3}[task]
    standard_lengths = [4096, 8192, 16384, 32768]
    lengths = [value for value in standard_lengths if value <= length]
    if length not in lengths:
        lengths.append(length)
    dataset = generator(max_seq_lengths=lengths, pretrained=str(checkpoint))["test"]
    documents = [doc for doc in dataset if int(doc["max_length"]) == length]
    return documents[sample]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("decode diagnosis requires a CUDA/ROCm GPU")
    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    config = MixerConfig.from_pretrained(checkpoint)
    if args.mixer == "split_full":
        config.token_mixer_class_path = "model.kvm_split_full_attention_mixer.SequenceMixer"
        config.kvm_use_merge_gate_keys = 0
        config.kvm_use_merge_gate_values = 0
        config.kvm_use_vlens = 0
    model = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    if args.state_temperature_factor != 1.0:
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.state_head_temp.mul_(args.state_temperature_factor)
    document = generate_document(args.task, checkpoint, args.length, args.sample)
    prompt = document["input"] + " " + document["gen_prefix"]
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    sequence = encoded.input_ids.cuda()
    target = str(document["outputs"][0])
    target_begin = prompt.find(target)
    target_end = target_begin + len(target)
    target_positions = [
        index
        for index, (begin, end) in enumerate(encoded.offset_mapping[0].tolist())
        if begin < target_end and end > target_begin
    ]

    attention_stats: dict[int, dict[str, list[float] | list[int]]] = {}
    original_attention_methods = []
    for layer_idx, block in enumerate(model.transformer.h):
        mixer = block.attn
        if not hasattr(mixer, "_split_prefill_attention"):
            continue
        original = mixer._split_prefill_attention
        original_attention_methods.append((mixer, original))

        def wrapped_attention(
            self,
            q,
            local_k,
            remote_k,
            v,
            local_block_mask,
            remote_block_mask,
            *,
            _layer_idx=layer_idx,
            _original=original,
        ):
            seq_len = int(q.size(2))
            local_begin = self._bswa_begin_for_total_len(seq_len)
            if local_begin:
                query = q[..., -1:, :].float()
                scale = 1.0 / math.sqrt(float(self.d_qk_head))
                remote_scores = torch.matmul(
                    query, remote_k[..., :local_begin, :].float().transpose(-1, -2)
                ).squeeze(-2) * scale
                local_scores = torch.matmul(
                    query, local_k[..., local_begin:, :].float().transpose(-1, -2)
                ).squeeze(-2) * scale
                all_scores = torch.cat((remote_scores, local_scores), dim=-1)
                probabilities = torch.softmax(all_scores, dim=-1)
                remote_mass = probabilities[..., :local_begin].sum(-1)
                target_index = torch.tensor(target_positions, device=q.device)
                target_probabilities = probabilities.index_select(-1, target_index)
                target_mass = target_probabilities.sum(-1)
                target_best_score = all_scores.index_select(-1, target_index).max(-1).values
                target_best_rank = (all_scores > target_best_score.unsqueeze(-1)).sum(-1) + 1
                attention_stats[_layer_idx] = {
                    "remote_mass": remote_mass[0].cpu().tolist(),
                    "target_mass": target_mass[0].cpu().tolist(),
                    "target_best_rank": target_best_rank[0].cpu().tolist(),
                    "top_position": all_scores[0].argmax(-1).cpu().tolist(),
                }
            return _original(
                q,
                local_k,
                remote_k,
                v,
                local_block_mask,
                remote_block_mask,
            )

        mixer._split_prefill_attention = types.MethodType(wrapped_attention, mixer)

    with torch.inference_mode():
        cached_result = model(
            input_ids=sequence,
            cache_position=torch.arange(sequence.size(1), device="cuda"),
            use_cache=True,
        )
        for mixer, original in original_attention_methods:
            mixer._split_prefill_attention = original
        cache = cached_result.past_key_values
        cached_logits = cached_result.logits[:, -1]
        generated: list[int] = []
        mismatched_steps: list[int] = []
        largest_delta = 0.0
        for step in range(args.tokens):
            recomputed_logits = model(input_ids=sequence, use_cache=False).logits[:, -1]
            delta = (cached_logits - recomputed_logits).abs()
            largest_delta = max(largest_delta, float(delta.max().item()))
            cached_token = int(cached_logits.argmax(-1).item())
            recomputed_token = int(recomputed_logits.argmax(-1).item())
            if cached_token != recomputed_token:
                mismatched_steps.append(step)
            generated.append(cached_token)
            next_token = torch.tensor([[cached_token]], device="cuda")
            sequence = torch.cat((sequence, next_token), dim=1)
            cached_result = model(
                input_ids=next_token,
                past_key_values=cache,
                cache_position=torch.tensor([sequence.size(1) - 1], device="cuda"),
                use_cache=True,
            )
            cache = cached_result.past_key_values
            cached_logits = cached_result.logits[:, -1]

    generated_text = tokenizer.decode(generated)
    print(
        {
            "task": args.task,
            "length": args.length,
            "sample": args.sample,
            "prompt_tokens": int(sequence.size(1) - len(generated)),
            "target": target,
            "generated": generated_text,
            "target_found": target.lower() in generated_text.lower(),
            "argmax_mismatched_steps": mismatched_steps,
            "largest_logit_max_abs": largest_delta,
            "target_token_positions": target_positions,
            "final_query_attention": attention_stats,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe whether remote/local mass calibration explains split-full NIAH failure."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import model.kvm_split_full_attention_mixer as split_mixer
from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=0)
    parser.add_argument("--decode-only", action="store_true")
    return parser.parse_args()


def generate_documents(checkpoint: Path, length: int, samples: int) -> list[dict]:
    random.seed(0)
    np.random.seed(1234)
    from lm_eval.tasks.ruler.niah_utils import niah_single_3

    standard_lengths = [4096, 8192, 16384, 32768]
    lengths = [value for value in standard_lengths if value <= length]
    if length not in lengths:
        lengths.append(length)
    dataset = niah_single_3(max_seq_lengths=lengths, pretrained=str(checkpoint))["test"]
    return [doc for doc in dataset if int(doc["max_length"]) == length][:samples]


def merge_with_remote_bias(remote_bias: float):
    def merge(local_output, local_lse, remote_output, remote_lse):
        branch_lse = torch.stack((local_lse, remote_lse + remote_bias), dim=-1).float()
        weights = torch.softmax(branch_lse, dim=-1).to(local_output.dtype)
        return (
            local_output * weights[..., 0].unsqueeze(-1)
            + remote_output * weights[..., 1].unsqueeze(-1)
        )

    return merge


def sdpa_with_remote_bias(remote_bias: float, chunk_len: int, bswa_len: int, original):
    def attention(q, k, v, *args, **kwargs):
        if int(q.size(-2)) == 1 and int(k.size(-2)) > bswa_len:
            if kwargs.get("attn_mask") is not None:
                raise AssertionError("diagnostic remote bias expected no decode mask")
            total_len = int(k.size(-2))
            chunk_end = ((total_len + chunk_len - 1) // chunk_len) * chunk_len
            remote_len = max(chunk_end - bswa_len, 0)
            bias = torch.zeros(
                1, 1, 1, total_len, device=q.device, dtype=q.dtype
            )
            bias[..., :remote_len] = remote_bias
            kwargs["attn_mask"] = bias
        return original(q, k, v, *args, **kwargs)

    return attention


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("calibration probe requires a CUDA/ROCm GPU")
    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    config = MixerConfig.from_pretrained(checkpoint)
    model = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint, config=config, torch_dtype=torch.bfloat16
    ).cuda().eval()
    documents = generate_documents(checkpoint, args.length, args.samples)
    cases = []
    for document in documents:
        prompt = document["input"] + " " + document["gen_prefix"]
        input_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.cuda()
        expected_token = tokenizer(
            ": " + str(document["outputs"][0]), add_special_tokens=False
        ).input_ids[0]
        cases.append((input_ids, int(expected_token), str(document["outputs"][0])))

    mixers = [block.attn for block in model.transformer.h]
    original_state_temps = [mixer.state_head_temp.detach().clone() for mixer in mixers]
    original_merge = split_mixer._merge_attention_branches
    variants = [] if args.decode_only else [
        ("baseline", 1.0, 0.0),
        ("temp_0.75", 0.75, 0.0),
        ("temp_1.25", 1.25, 0.0),
        ("temp_1.5", 1.5, 0.0),
        ("temp_2", 2.0, 0.0),
        ("bias_-2", 1.0, -2.0),
        ("bias_-1", 1.0, -1.0),
        ("bias_-0.5", 1.0, -0.5),
        ("bias_0.5", 1.0, 0.5),
        ("bias_1", 1.0, 1.0),
        ("bias_2", 1.0, 2.0),
    ]
    results = []
    try:
        with torch.inference_mode():
            for name, temperature_factor, remote_bias in variants:
                for mixer, original_temperature in zip(
                    mixers, original_state_temps, strict=True
                ):
                    mixer.state_head_temp.copy_(original_temperature * temperature_factor)
                split_mixer._merge_attention_branches = merge_with_remote_bias(remote_bias)
                ranks = []
                predictions = []
                for input_ids, expected_token, _target in cases:
                    logits = model(
                        input_ids=input_ids,
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits[0, -1]
                    expected_logit = logits[expected_token]
                    ranks.append(int((logits > expected_logit).sum().item()) + 1)
                    predictions.append(tokenizer.decode([int(logits.argmax().item())]))
                results.append(
                    {
                        "variant": name,
                        "temperature_factor": temperature_factor,
                        "remote_bias": remote_bias,
                        "expected_first_token_top1": sum(rank == 1 for rank in ranks),
                        "mean_expected_rank": sum(ranks) / len(ranks),
                        "ranks": ranks,
                        "predictions": predictions,
                    }
                )
    finally:
        split_mixer._merge_attention_branches = original_merge
        with torch.no_grad():
            for mixer, original_temperature in zip(
                mixers, original_state_temps, strict=True
            ):
                mixer.state_head_temp.copy_(original_temperature)
    decode_results = []
    if args.decode_tokens:
        original_sdpa = F.scaled_dot_product_attention
        decode_variants = (
            (1.0, 0.0),
            (1.25, 0.0),
            (1.5, 0.0),
            (1.0, 1.0),
            (1.0, 2.0),
        )
        with torch.inference_mode():
            for temperature_factor, remote_bias in decode_variants:
                for mixer, original_temperature in zip(
                    mixers, original_state_temps, strict=True
                ):
                    mixer.state_head_temp.copy_(original_temperature * temperature_factor)
                split_mixer._merge_attention_branches = merge_with_remote_bias(
                    remote_bias
                )
                F.scaled_dot_product_attention = sdpa_with_remote_bias(
                    remote_bias,
                    chunk_len=int(config.chunk_len),
                    bswa_len=int(config.chunk_len) * int(config.n_bswa_chunks),
                    original=original_sdpa,
                )
                successes = []
                generated_texts = []
                for input_ids, _expected_token, target in cases:
                    result = model(
                        input_ids=input_ids,
                        cache_position=torch.arange(input_ids.size(1), device="cuda"),
                        use_cache=True,
                    )
                    cache = result.past_key_values
                    logits = result.logits[:, -1]
                    generated = []
                    for step in range(args.decode_tokens):
                        token = int(logits.argmax(-1).item())
                        generated.append(token)
                        next_token = torch.tensor([[token]], device="cuda")
                        result = model(
                            input_ids=next_token,
                            past_key_values=cache,
                            cache_position=torch.tensor(
                                [input_ids.size(1) + step], device="cuda"
                            ),
                            use_cache=True,
                        )
                        cache = result.past_key_values
                        logits = result.logits[:, -1]
                    generated_text = tokenizer.decode(generated)
                    generated_texts.append(generated_text)
                    successes.append(target.lower() in generated_text.lower())
                decode_results.append(
                    {
                        "temperature_factor": temperature_factor,
                        "remote_bias": remote_bias,
                        "successes": sum(successes),
                        "per_sample": successes,
                        "generated": generated_texts,
                    }
                )
        F.scaled_dot_product_attention = original_sdpa
        split_mixer._merge_attention_branches = original_merge
        with torch.no_grad():
            for mixer, original_temperature in zip(
                mixers, original_state_temps, strict=True
            ):
                mixer.state_head_temp.copy_(original_temperature)
    print(
        {
            "length": args.length,
            "samples": len(cases),
            "results": results,
            "decode_results": decode_results,
        }
    )


if __name__ == "__main__":
    main()

"""Minimal Hugging Face LoD generation example."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lod_attention import LODMode, install


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in LODMode],
        default=LODMode.TWO_TIER.value,
    )
    parser.add_argument("--prompt", default="Explain LoD Attention.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    install(model, mode=args.mode)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

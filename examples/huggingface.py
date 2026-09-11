"""Minimal Hugging Face LoD generation example."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lod_attention import LODMode, install


def load_config(model: str) -> Any:
    """Load a Transformers config, correcting its Qwen3.8 FP8 skip list."""

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    quantization = getattr(config, "quantization_config", None)
    if isinstance(quantization, dict) and quantization.get("quant_method") == "fp8":
        # Transformers 5.15 treats skip-list strings as unanchored regular
        # expressions. Qwen's scalar ``mlp.gate`` exclusion therefore also
        # skips the distinct ``mlp.gate_proj`` linear, leaving an FP8 weight in
        # nn.Linear. The scalar gate is not a Linear and needs no exclusion.
        ignored = quantization.get("modules_to_not_convert")
        if isinstance(ignored, list):
            quantization = dict(quantization)
            quantization["modules_to_not_convert"] = [
                name for name in ignored if not name.endswith(".mlp.gate")
            ]
            config.quantization_config = quantization
    return config


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

    config = load_config(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    install(model, mode=args.mode)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

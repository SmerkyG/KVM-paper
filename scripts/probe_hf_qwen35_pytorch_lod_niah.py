#!/usr/bin/env python3
"""Small NIAH evaluation for Qwen3.5 with the generic HF LOD replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from model.hf_pytorch_lod_attention import (
    replace_qwen35_attention_with_lod,
    reset_hf_lod_caches,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig
from scripts.probe_qwen35_lod_niah import enable_fla_fast_path, generate_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--task",
        choices=("niah_single_1", "niah_single_2", "niah_single_3"),
        required=True,
    )
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--open-count", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--page-size", type=int, default=0)
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NIAH evaluation requires a CUDA or ROCm GPU")
    enable_fla_fast_path(required=True)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    composite_config = AutoConfig.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    config = composite_config.text_config
    config._attn_implementation = "sdpa"
    model = (
        Qwen3_5ForCausalLM.from_pretrained(
            args.checkpoint,
            config=config,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    if args.kv_bits and not args.page_size:
        raise ValueError("--kv-bits=4 requires a positive --page-size")
    config_type = PagedLODConfig if args.page_size else LODConfig
    lod_config = config_type(
        chunk_size=256,
        local_window=512,
        state_growth_factor=args.state_growth_factor,
        state_min_size=256,
        protected_prefix=1,
        max_routes=8,
        **(
            {"page_size": args.page_size, "kv_bits": args.kv_bits}
            if args.page_size
            else {}
        ),
    )
    replaced = replace_qwen35_attention_with_lod(
        model,
        config=lod_config,
        open_count=args.open_count,
    )
    documents = generate_documents(args.task, args.checkpoint, args.length)
    selected = documents[: args.samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with args.output.open("w") as handle:
        for document in selected:
            reset_hf_lod_caches(model)
            prompt = document["input"] + " " + document["gen_prefix"]
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
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(
                generated[0, input_ids.size(1) :],
                skip_special_tokens=True,
            )
            target = str(document["outputs"][0])
            exact = target.lower() in response.lower()
            correct += int(exact)
            record = {
                "checkpoint": args.checkpoint,
                "mode": "hf_pytorch_lod",
                "task": args.task,
                "index": int(document["index"]),
                "length": int(document["max_length"]),
                "input_tokens": int(input_ids.size(1)),
                "replaced_layers": replaced,
                "open_count": args.open_count,
                "state_growth_factor": args.state_growth_factor,
                "page_size": args.page_size or None,
                "kv_bits": args.kv_bits,
                "target": target,
                "response": response,
                "exact": exact,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"task={args.task} index={document['index']} "
                f"tokens={input_ids.size(1)} exact={exact}",
                flush=True,
            )
    print(f"score={correct}/{len(selected)}")


if __name__ == "__main__":
    main()

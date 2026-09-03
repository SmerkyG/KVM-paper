#!/usr/bin/env python3
"""Verify direct LOD prefill falls back safely on a native prefix-cache hit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from vllm_engine_lifecycle import register_llm_shutdown, shutdown_registered_llms


def inspect_lod_counters(model) -> dict[str, int]:
    counters = {
        "layers": 0,
        "installs": 0,
        "direct_prefills": 0,
        "decodes": 0,
        "retained_prefix_reuses": 0,
    }
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        counters["layers"] += 1
        counters["installs"] += int(pool.install_count)
        counters["direct_prefills"] += int(pool.direct_prefill_calls)
        counters["decodes"] += int(pool.decode_calls)
        counters["retained_prefix_reuses"] += int(pool.retained_reuse_count)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--common-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    seed = tokenizer(
        "The retained native cache provides a safe shared prefix. ",
        add_special_tokens=False,
    )["input_ids"]
    common = (seed * ((args.common_tokens + len(seed) - 1) // len(seed)))[
        : args.common_tokens
    ]
    suffix_a = tokenizer(" First continuation.", add_special_tokens=False)["input_ids"]
    suffix_b = tokenizer(" Second distinct continuation.", add_special_tokens=False)[
        "input_ids"
    ]
    llm = register_llm_shutdown(
        LLM(
            model=args.checkpoint,
            load_format=os.getenv("VLLM_WEIGHT_CACHE_LOAD_FORMAT", "ipc_cache"),
            dtype="bfloat16",
            max_model_len=args.common_tokens + 128,
            max_num_seqs=1,
            max_num_batched_tokens=args.common_tokens + 128,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            enable_prefix_caching=True,
            attention_config={"backend": "CUSTOM"},
        )
    )
    params = SamplingParams(temperature=0, max_tokens=2, detokenize=False)
    before = llm.apply_model(inspect_lod_counters)[0]
    first = llm.generate(
        [{"prompt_token_ids": common + suffix_a}], params, use_tqdm=False
    )
    after_first = llm.apply_model(inspect_lod_counters)[0]
    second = llm.generate(
        [{"prompt_token_ids": common + suffix_b}], params, use_tqdm=False
    )
    after_second = llm.apply_model(inspect_lod_counters)[0]

    layers = after_first["layers"]
    if layers <= 0:
        raise RuntimeError("vLLM did not attach any LOD pools")
    first_direct = after_first["direct_prefills"] - before["direct_prefills"]
    if first_direct < layers or first_direct % layers:
        raise RuntimeError(
            "the uncached request did not use direct LOD prefill on every layer: "
            f"before={before}, after={after_first}"
        )
    if (
        after_second["retained_prefix_reuses"]
        - after_first["retained_prefix_reuses"]
        != layers
    ):
        raise RuntimeError("the prefix-cache hit did not reuse the retained LOD row")
    if after_second["installs"] != after_first["installs"]:
        raise RuntimeError("the retained LOD row was rebuilt instead of reused")
    if not first[0].outputs[0].token_ids or not second[0].outputs[0].token_ids:
        raise RuntimeError("prefix-cache verification produced no tokens")

    result = {
        "checkpoint": args.checkpoint,
        "common_tokens": args.common_tokens,
        "before": before,
        "after_first": after_first,
        "after_second": after_second,
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_registered_llms()

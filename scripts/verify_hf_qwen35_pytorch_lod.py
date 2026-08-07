#!/usr/bin/env python3
"""End-to-end Qwen3.5 smoke and cache checks for the HF LOD replacement."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM

from model.hf_pytorch_lod_attention import (
    Qwen3_5FastLODAttention,
    replace_qwen35_attention_with_lod,
    reset_hf_lod_caches,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_engines import (
    KernelCoarseLODAttention,
    KernelRecursivePagedLODAttention,
    KernelTwoLevelLODAttention,
)
from scripts.probe_qwen35_lod_niah import enable_fla_fast_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--exact-length", type=int, default=128)
    parser.add_argument("--lod-length", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--open-count", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=0)
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument(
        "--engine-backend", choices=("torch", "kernel"), default="torch"
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 verification requires a CUDA or ROCm GPU")
    enable_fla_fast_path(required=True)
    device = torch.device("cuda")
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
    generator = torch.Generator(device=device).manual_seed(80)
    exact_ids = torch.randint(
        0,
        config.vocab_size,
        (1, args.exact_length),
        generator=generator,
        device=device,
    )
    baseline_logits = model(input_ids=exact_ids, use_cache=False).logits
    original_state_keys = tuple(model.state_dict())

    if args.kv_bits and not args.page_size:
        raise ValueError("--kv-bits=4 requires a positive --page-size")
    config_type = PagedLODConfig if args.page_size else LODConfig
    paged_options = (
        {"page_size": args.page_size, "kv_bits": args.kv_bits}
        if args.page_size
        else {}
    )
    lod_config = config_type(
        chunk_size=256,
        local_window=512,
        state_growth_factor=16.0,
        state_min_size=256,
        protected_prefix=1,
        max_routes=8,
        **paged_options,
    )
    replaced = replace_qwen35_attention_with_lod(
        model,
        config=lod_config,
        open_count=args.open_count,
        engine_backend=args.engine_backend,
    )
    expected = [
        index
        for index, layer_type in enumerate(config.layer_types)
        if layer_type == "full_attention"
    ]
    if replaced != expected:
        raise AssertionError(f"replaced layers {replaced}, expected {expected}")
    if sum(
        isinstance(module, Qwen3_5FastLODAttention)
        for module in model.modules()
    ) != len(expected):
        raise AssertionError("unexpected number of installed LOD attention modules")
    if args.engine_backend == "kernel":
        expected_engine = (
            KernelCoarseLODAttention
            if args.open_count == 0
            else (
                KernelRecursivePagedLODAttention
                if args.page_size
                else KernelTwoLevelLODAttention
            )
        )
        for module in model.modules():
            if isinstance(module, Qwen3_5FastLODAttention) and not isinstance(
                module.lod_engine, expected_engine
            ):
                raise AssertionError("Qwen adapter installed the wrong LOD engine")
    if tuple(model.state_dict()) != original_state_keys:
        raise AssertionError("attention replacement changed model state-dict keys")

    lod_exact_logits = model(input_ids=exact_ids, use_cache=False).logits
    torch.testing.assert_close(
        lod_exact_logits.float(),
        baseline_logits.float(),
        atol=3e-2,
        rtol=3e-2,
    )
    print(
        f"exact local parity passed: length={args.exact_length} "
        f"layers={replaced}"
    )

    long_ids = torch.randint(
        0,
        config.vocab_size,
        (1, args.lod_length),
        generator=generator,
        device=device,
    )
    long_result = model(input_ids=long_ids, labels=long_ids, use_cache=False)
    if long_result.loss is None or not bool(torch.isfinite(long_result.loss).item()):
        raise AssertionError("LOD model produced a non-finite long-context loss")

    prefix_length = args.lod_length - args.decode_tokens
    reset_hf_lod_caches(model)
    cached = model(input_ids=long_ids[:, :prefix_length], use_cache=True)
    past_key_values = cached.past_key_values
    if past_key_values is None or past_key_values.get_seq_length() != prefix_length:
        raise AssertionError("Qwen cache did not receive the LOD prefill length")
    for layer_index in expected:
        metadata = past_key_values.key_cache[layer_index]
        if metadata is not None and int(metadata.numel()) != 0:
            raise AssertionError("Qwen retained a duplicate ordinary KV cache")
    for token_index in range(prefix_length, args.lod_length):
        cached = model(
            input_ids=long_ids[:, token_index : token_index + 1],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = cached.past_key_values
        if past_key_values is None:
            raise AssertionError("Qwen decode dropped its cache")
    if past_key_values.get_seq_length() != args.lod_length:
        raise AssertionError("Qwen cache length did not advance during decode")
    if not bool(torch.isfinite(cached.logits).all().item()):
        raise AssertionError("LOD cached decode produced non-finite logits")
    if args.engine_backend == "kernel" and args.open_count == 0:
        for module in model.modules():
            if isinstance(module, Qwen3_5FastLODAttention):
                state = module._lod_cache.state
                if (
                    state["owners"].numel()
                    or state["exact_k"].numel()
                    or state["exact_v"].numel()
                ):
                    raise AssertionError("coarse-only engine retained exact leaves")
    print(
        f"long-context and cached decode passed: length={args.lod_length} "
        f"backend={args.engine_backend} loss={float(long_result.loss):.6f}"
    )


if __name__ == "__main__":
    main()

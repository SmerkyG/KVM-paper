#!/usr/bin/env python3
"""End-to-end checkpoint smoke for the registered HF LOD backend."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from model.hf_pytorch_lod_attention import (
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--exact-length", type=int, default=128)
    parser.add_argument("--lod-length", type=int, default=1024)
    parser.add_argument("--turn-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--open-count", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=0)
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument(
        "--engine-backend", choices=("torch", "kernel"), default="torch"
    )
    parser.add_argument(
        "--use-upstream-code",
        action="store_true",
        help="ignore checkpoint auto_map entries when Transformers supports the model",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint verification requires a CUDA or ROCm GPU")
    if args.kv_bits and not args.page_size:
        raise ValueError("--kv-bits=4 requires a positive --page-size")
    device = torch.device("cuda")
    composite_config = AutoConfig.from_pretrained(
        args.checkpoint, trust_remote_code=not args.use_upstream_code
    )
    config = composite_config.get_text_config(decoder=True)
    is_qwen35 = type(config).__module__.startswith(
        ("transformers.models.qwen3_5.", "transformers.models.qwen3_5_moe.")
    )
    if is_qwen35:
        from scripts.probe_qwen35_lod_niah import enable_fla_fast_path

        enable_fla_fast_path(required=True)
    config._attn_implementation = "sdpa"
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=not args.use_upstream_code,
        )
        .to(device)
        .eval()
    )
    generator = torch.Generator(device=device).manual_seed(81)
    exact_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, args.exact_length),
        generator=generator,
        device=device,
    )
    baseline = model(exact_ids, use_cache=False).logits
    original_state_keys = tuple(model.state_dict())

    config_type = PagedLODConfig if args.page_size else LODConfig
    page_options = (
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
        **page_options,
    )
    installed = install_hf_lod_attention(
        model,
        config=lod_config,
        open_count=args.open_count,
        engine_backend=args.engine_backend,
    )
    layer_types = getattr(model.config, "layer_types", None)
    expected_layers = (
        sum(layer_type == "full_attention" for layer_type in layer_types)
        if layer_types is not None
        else int(model.config.num_hidden_layers)
    )
    if len(installed) != expected_layers:
        raise AssertionError(
            f"installed {len(installed)} LOD layers, expected {expected_layers}"
        )
    if tuple(model.state_dict()) != original_state_keys:
        raise AssertionError("LOD installation changed checkpoint state keys")
    lod_exact = model(exact_ids, use_cache=False).logits
    torch.testing.assert_close(
        lod_exact.float(), baseline.float(), atol=3e-2, rtol=3e-2
    )

    long_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, args.lod_length),
        generator=generator,
        device=device,
    )
    long_result = model(long_ids, labels=long_ids, use_cache=False)
    if long_result.loss is None or not bool(torch.isfinite(long_result.loss)):
        raise AssertionError("registered LOD backend produced non-finite loss")

    prefix_length = args.lod_length - args.turn_tokens - args.decode_tokens
    if prefix_length <= 0 or args.turn_tokens <= 1:
        raise ValueError("LOD length must leave room for a multi-token cached turn")
    cache = new_hf_lod_cache(model)
    cached = model(
        long_ids[:, :prefix_length], past_key_values=cache, use_cache=True
    )
    first_turn_end = prefix_length + args.turn_tokens // 2
    cached = model(
        long_ids[:, prefix_length:first_turn_end],
        past_key_values=cache,
        use_cache=True,
    )
    if int(cached.logits.size(1)) != first_turn_end - prefix_length:
        raise AssertionError("cached turn did not return one output per token")
    # A real conversation normally decodes an assistant response before the
    # next user turn. Exercise cached prefill again after that state change.
    cached = model(
        long_ids[:, first_turn_end : first_turn_end + 1],
        past_key_values=cache,
        use_cache=True,
    )
    second_turn_end = prefix_length + args.turn_tokens + 1
    cached = model(
        long_ids[:, first_turn_end + 1 : second_turn_end],
        past_key_values=cache,
        use_cache=True,
    )
    for position in range(second_turn_end, args.lod_length):
        cached = model(
            long_ids[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
    if cache.get_seq_length() != args.lod_length:
        raise AssertionError("LOD-owned HF cache length did not advance")
    lod_layers = (
        cache.lod_layers.values()
        if hasattr(cache, "lod_layers")
        else cache.layers
    )
    if any(layer.keys.numel() or layer.values.numel() for layer in lod_layers):
        raise AssertionError("HF cache retained duplicate ordinary K/V tensors")
    if is_qwen35 and hasattr(cache, "lod_layers"):
        native_layers = getattr(cache.native_cache, "layers", ())
        for layer_index in cache.lod_layers:
            native_layer = native_layers[layer_index]
            if any(
                isinstance(getattr(native_layer, name, None), torch.Tensor)
                and getattr(native_layer, name).numel()
                for name in ("keys", "values")
            ):
                raise AssertionError(
                    "hybrid native cache retained duplicate attention K/V"
                )
    if not bool(torch.isfinite(cached.logits).all()):
        raise AssertionError("registered LOD cached decode produced non-finite logits")
    print(
        f"registered checkpoint smoke passed: checkpoint={args.checkpoint} "
        f"length={args.lod_length} backend={args.engine_backend} "
        f"kv_bits={args.kv_bits} loss={float(long_result.loss):.6f}"
    )


if __name__ == "__main__":
    main()

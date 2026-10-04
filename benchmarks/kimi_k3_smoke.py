"""Small correctness/speed smoke test for the Kimi K3 absorbed-MLA adapter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks._vllm import close_llm, llm_kwargs


MIN_LOD_MODEL_LEN = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="inference-optimization/Kimi-K3-0.40B"
    )
    parser.add_argument(
        "--mode",
        choices=(
            "full",
            "full-hf",
            "two-tier",
            "three-tier-bf16",
            "three-tier-int4",
        ),
        required=True,
    )
    parser.add_argument("--prompt-tokens", type=int, default=3072)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decode-context-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument(
        "--dcp-comm-backend",
        choices=("ag_rs", "a2a"),
        default="ag_rs",
    )
    parser.add_argument(
        "--full-mla-backend",
        choices=("auto", "ROCM_AITER_MLA", "TRITON_MLA"),
        default="auto",
        help="Force the native MLA backend for full-attention controls.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Skip decode CUDA-graph capture for rapid prefill kernel iteration.",
    )
    parser.add_argument(
        "--load-format",
        choices=("auto", "dummy"),
        default="auto",
        help="Use vLLM's deterministic random-weight loader with 'dummy'.",
    )
    parser.add_argument(
        "--weight-cache",
        action="store_true",
        help=(
            "Load the final post-conversion model tensors from the persistent "
            "GPU weight daemon. A cache miss loads and converts the ordinary "
            "checkpoint once; later processes map those tensors through HIP IPC."
        ),
    )
    parser.add_argument("--weight-cache-id", default="kimi-k3-smoke")
    parser.add_argument("--weight-cache-dir")
    parser.add_argument(
        "--kimi-gfx942-int4-moe",
        action="store_true",
        help=(
            "Use K3's validated gfx942 expert-parallel groupwise-INT4 MoE "
            "configuration. This must match the configuration used to "
            "populate a persistent weight cache."
        ),
    )
    parser.add_argument(
        "--skip-tokenizer-init",
        action="store_true",
        help="Generate deterministic token IDs directly for synthetic configs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.prompt_tokens < 1
        or args.decode_tokens < 1
        or args.batch_size < 1
        or args.warmup_runs < 0
    ):
        raise ValueError("token counts must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    tokenizer = None
    if args.skip_tokenizer_init:
        prompt_ids = [3 + (index % 997) for index in range(args.prompt_tokens)]
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, trust_remote_code=True
        )
        seed_text = (
            "Kimi K3 latent attention stores a compressed representation. "
            "This deterministic paragraph exercises a long causal cache.\n"
        )
        seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
        prompt_ids = (
            seed_ids
            * ((args.prompt_tokens + len(seed_ids) - 1) // len(seed_ids))
        )[: args.prompt_tokens]
    max_model_len = max(
        args.prompt_tokens + args.decode_tokens + 8,
        MIN_LOD_MODEL_LEN,
    )
    if args.mode == "full-hf":
        if args.load_format != "auto" or tokenizer is None:
            raise ValueError("full-hf requires real weights and a tokenizer")
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to("cuda")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=args.decode_tokens,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        token_ids = generated.sequences[0, input_ids.size(1) :].tolist()
        chosen_logprobs = [
            float(score[0].log_softmax(dim=-1)[token_id])
            for score, token_id in zip(generated.scores, token_ids, strict=True)
        ]
        result = {
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "prompt_tokens": len(prompt_ids),
            "decode_tokens": len(token_ids),
            "elapsed_seconds": elapsed,
            "token_ids": token_ids,
            "chosen_logprobs": chosen_logprobs,
            "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    from vllm import LLM, SamplingParams

    kwargs = llm_kwargs(
        checkpoint=args.checkpoint,
        mode=args.mode,
        max_model_len=max_model_len,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        decode_context_parallel_size=args.decode_context_parallel_size,
        dcp_comm_backend=args.dcp_comm_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    if args.kimi_gfx942_int4_moe:
        if "kimi-k3" not in args.checkpoint.lower():
            raise ValueError("--kimi-gfx942-int4-moe requires a Kimi-K3 checkpoint")
        kwargs["enable_expert_parallel"] = True
        kwargs["disable_custom_all_reduce"] = False
        kwargs["quantization_config"] = {
            "moe": {"weight": "int4_per_group_32"}
        }
        kwargs["safetensors_load_strategy"] = "prefetch"
    if args.weight_cache:
        if args.load_format != "auto":
            raise ValueError("--weight-cache requires --load-format auto")
        kwargs["load_format"] = "ipc_cache"
        kwargs["model_loader_extra_config"] = {
            "auto_start": True,
            "cache_id": args.weight_cache_id,
            "cache_dir": args.weight_cache_dir,
            "backing_load_format": "auto",
            "broker_timeout": 1800.0,
        }
    else:
        kwargs["load_format"] = args.load_format
    kwargs["skip_tokenizer_init"] = args.skip_tokenizer_init
    kwargs["enforce_eager"] = args.enforce_eager
    if args.kv_cache_memory_bytes is not None:
        if args.kv_cache_memory_bytes <= 0:
            raise ValueError("--kv-cache-memory-bytes must be positive")
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.mode == "full" and args.full_mla_backend != "auto":
        kwargs["attention_config"] = {
            "backend": None,
            "backend_per_kind": {"mla_attention": args.full_mla_backend},
        }
    llm = LLM(**kwargs)
    try:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=args.decode_tokens,
            logprobs=1,
            seed=1234,
            ignore_eos=True,
        )
        prompts = [
            {
                "prompt_token_ids": [
                    3 + ((token_id - 3 + request_index) % 997)
                    for token_id in prompt_ids
                ]
            }
            for request_index in range(args.batch_size)
        ]
        for _ in range(args.warmup_runs):
            llm.generate(prompts, params, use_tqdm=False)
        start = time.perf_counter()
        requests = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        completions = [request.outputs[0] for request in requests]
        metrics = [request.metrics for request in requests]
        if any(metric is None for metric in metrics):
            raise RuntimeError("vLLM did not return per-request timing metrics")
        scheduled = min(float(metric.scheduled_ts) for metric in metrics)
        first_token = max(float(metric.first_token_ts) for metric in metrics)
        last_token = max(float(metric.last_token_ts) for metric in metrics)
        prefill_seconds = first_token - scheduled
        decode_seconds = last_token - first_token
        decode_steps = max(min(len(item.token_ids) for item in completions) - 1, 0)
        token_ids = [list(item.token_ids) for item in completions]
        chosen_logprobs = []
        for completion in completions:
            row = []
            for token_id, candidates in zip(
                completion.token_ids, completion.logprobs or (), strict=True
            ):
                item = candidates[token_id]
                row.append(float(item.logprob))
            chosen_logprobs.append(row)
        result = {
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "dense_decode_backend": (
                "gluon_absorbed_mla"
                if args.mode == "full"
                else None
            ),
            "decode_context_parallel_size": args.decode_context_parallel_size,
            "dcp_comm_backend": args.dcp_comm_backend,
            "weight_cache": args.weight_cache,
            "weight_cache_id": (
                args.weight_cache_id if args.weight_cache else None
            ),
            "kimi_gfx942_int4_moe": args.kimi_gfx942_int4_moe,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batch_size": args.batch_size,
            "prompt_tokens": len(prompt_ids),
            "decode_tokens": min(len(item.token_ids) for item in completions),
            "elapsed_seconds": elapsed,
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": (
                args.batch_size * len(prompt_ids) / prefill_seconds
            ),
            "decode_seconds": decode_seconds,
            "decode_ms_per_batch_step": (
                1_000.0 * decode_seconds / decode_steps if decode_steps else None
            ),
            "decode_tokens_per_second": (
                args.batch_size * decode_steps / decode_seconds
                if decode_steps
                else None
            ),
            "token_ids": token_ids,
            "chosen_logprobs": chosen_logprobs,
            "text": (
                [completion.text for completion in completions]
                if tokenizer is not None
                else None
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

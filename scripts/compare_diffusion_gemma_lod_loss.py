#!/usr/bin/env python3
"""Compare native and recursive-LOD DiffusionGemma denoising CE on ProLong."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

from model.hf_diffusion_gemma_lod_attention import (
    install_diffusion_gemma_lod_attention,
    uninstall_diffusion_gemma_lod_attention,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="google/diffusiongemma-26B-A4B-it"
    )
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--local-window", type=int, default=512)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument(
        "--routing-normalization",
        choices=("none", "query", "key", "both"),
        default="none",
    )
    parser.add_argument("--routing-count-bias", type=float, default=1.0)
    parser.add_argument("--routing-variance-bias", type=float, default=0.0)
    parser.add_argument(
        "--lod-engine", choices=("recursive", "two-level"), default="recursive"
    )
    parser.add_argument(
        "--corruption-rates",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "full", "lod", "lod-prefill"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _distributed_mode(requested: str) -> tuple[int, int, int, str]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if requested == "auto":
        if world_size != 2:
            raise ValueError("--mode auto requires exactly two torchrun ranks")
        mode = ("full", "lod")[rank]
    else:
        mode = requested
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size, mode


def _select_sequences(
    checkpoint: str,
    dataset_name: str,
    sequence_length: int,
    samples: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    sequences = torch.empty(
        samples, sequence_length, dtype=torch.long, device=device
    )
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        dataset = load_dataset(dataset_name, split="train", streaming=True).shuffle(
            seed=42, buffer_size=1_000
        )
        selected = []
        for document in dataset:
            token_count = document.get("length")
            if token_count is not None and int(token_count) < sequence_length:
                continue
            input_ids = tokenizer(
                document["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=sequence_length,
                return_attention_mask=False,
            )["input_ids"]
            if len(input_ids) != sequence_length:
                continue
            selected.append(torch.tensor(input_ids, dtype=torch.long))
            if len(selected) == samples:
                break
        if len(selected) != samples:
            raise RuntimeError(
                f"found only {len(selected)} sufficiently long ProLong documents"
            )
        sequences.copy_(torch.stack(selected).to(device))
    if world_size > 1:
        dist.broadcast(sequences, src=0)
    return sequences


def _corrupt_canvas(
    target: torch.Tensor,
    rate: float,
    *,
    vocabulary_size: int,
    sample: int,
    rate_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17_000 + sample * 101 + rate_index)
    cpu_shape = tuple(target.shape)
    selected = torch.rand(cpu_shape, generator=generator).lt(rate)
    if not bool(selected.any()):
        selected.view(-1)[0] = True
    replacement = torch.randint(
        0, vocabulary_size, cpu_shape, generator=generator, dtype=torch.long
    )
    selected = selected.to(target.device)
    replacement = replacement.to(target.device)
    return torch.where(selected, replacement, target), selected


def _lod_config(
    local_window: int,
    lod_engine: str,
    state_growth_factor: float,
    routing_normalization: str,
    routing_count_bias: float,
    routing_variance_bias: float,
) -> LODConfig:
    common = {
        "chunk_size": 256,
        "local_window": local_window,
        "state_growth_factor": state_growth_factor,
        "state_min_size": 256,
        "protected_prefix": 1,
        "routing_normalization": routing_normalization,
        "routing_count_bias": routing_count_bias,
        "routing_variance_bias": routing_variance_bias,
        "max_routes": 8,
        "leaf_dtype": torch.bfloat16,
    }
    if lod_engine == "recursive":
        return PagedLODConfig(
            **common,
            page_size=16,
            kv_bits=0,
            quant_group_size=32,
        )
    return LODConfig(**common)


def _load_model(
    checkpoint: str,
    mode: str,
    device: torch.device,
    *,
    local_window: int,
    lod_engine: str,
    state_growth_factor: float,
    routing_normalization: str,
    routing_count_bias: float,
    routing_variance_bias: float,
) -> DiffusionGemmaForBlockDiffusion:
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    if mode in ("lod", "lod-prefill"):
        install_diffusion_gemma_lod_attention(
            model,
            config=_lod_config(
                local_window,
                lod_engine,
                state_growth_factor,
                routing_normalization,
                routing_count_bias,
                routing_variance_bias,
            ),
            open_count=8,
            engine_backend="kernel",
        )
    return model


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.sequence_length <= 256:
        raise ValueError("samples must be positive and sequence length must exceed 256")
    if args.local_window < 256 or args.local_window % 256:
        raise ValueError("local window must be a positive multiple of 256")
    if args.state_growth_factor < 0:
        raise ValueError("state growth factor cannot be negative")
    if any(not 0 < rate <= 1 for rate in args.corruption_rates):
        raise ValueError("corruption rates must lie in (0, 1]")
    local_rank, rank, world_size, mode = _distributed_mode(args.mode)
    device = torch.device("cuda", local_rank)
    sequences = _select_sequences(
        args.checkpoint,
        args.dataset,
        args.sequence_length,
        args.samples,
        device,
        rank,
        world_size,
    )
    model = _load_model(
        args.checkpoint,
        mode,
        device,
        local_window=args.local_window,
        lod_engine=args.lod_engine,
        state_growth_factor=args.state_growth_factor,
        routing_normalization=args.routing_normalization,
        routing_count_bias=args.routing_count_bias,
        routing_variance_bias=args.routing_variance_bias,
    )
    canvas_length = int(model.config.canvas_length)
    prefix_length = args.sequence_length - canvas_length
    vocabulary_size = int(model.config.text_config.vocab_size)
    attention_mask = torch.ones(
        1, prefix_length, dtype=torch.long, device=device
    )
    decoder_attention_mask = torch.ones(
        1, args.sequence_length, dtype=torch.long, device=device
    )

    loss_sums = torch.zeros(len(args.corruption_rates), dtype=torch.float64)
    token_counts = torch.zeros(len(args.corruption_rates), dtype=torch.long)
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{mode}_rank_{rank:02d}.jsonl"
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with output_path.open("w") as handle, torch.inference_mode():
        for sample, sequence in enumerate(sequences):
            if mode == "lod-prefill" and sample:
                install_diffusion_gemma_lod_attention(
                    model,
                    config=_lod_config(
                        args.local_window,
                        args.lod_engine,
                        args.state_growth_factor,
                        args.routing_normalization,
                        args.routing_count_bias,
                        args.routing_variance_bias,
                    ),
                    open_count=8,
                    engine_backend="kernel",
                )
            prefix = sequence[:prefix_length].unsqueeze(0)
            target = sequence[prefix_length:].unsqueeze(0)
            encoder = model.model.encoder(
                input_ids=prefix,
                attention_mask=attention_mask,
            )
            if mode == "lod-prefill":
                uninstall_diffusion_gemma_lod_attention(model)
            sample_record = {"sample": sample, "mode": mode, "rates": {}}
            for rate_index, rate in enumerate(args.corruption_rates):
                canvas, selected = _corrupt_canvas(
                    target,
                    rate,
                    vocabulary_size=vocabulary_size,
                    sample=sample,
                    rate_index=rate_index,
                )
                result = model(
                    past_key_values=encoder.past_key_values,
                    decoder_input_ids=canvas,
                    decoder_attention_mask=decoder_attention_mask,
                )
                selected_logits = result.logits[selected]
                selected_target = target[selected]
                loss_sum = F.cross_entropy(
                    selected_logits.float(), selected_target, reduction="sum"
                )
                count = int(selected_target.numel())
                loss_sums[rate_index] += float(loss_sum)
                token_counts[rate_index] += count
                loss = float(loss_sum) / count
                sample_record["rates"][str(rate)] = {
                    "loss": loss,
                    "pseudo_perplexity": math.exp(min(loss, 80.0)),
                    "tokens": count,
                }
                del result
            handle.write(json.dumps(sample_record, sort_keys=True) + "\n")
            handle.flush()
            del encoder
            print(f"rank={rank} mode={mode} sample={sample} complete", flush=True)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    local_statistics = torch.stack(
        (loss_sums.to(device), token_counts.to(device=device, dtype=torch.float64)),
        dim=-1,
    )
    if world_size > 1:
        gathered = [torch.empty_like(local_statistics) for _ in range(world_size)]
        dist.all_gather(gathered, local_statistics)
    else:
        gathered = [local_statistics]

    if rank == 0:
        modes = ("full", "lod") if args.mode == "auto" else (mode,)
        summary = {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "sequence_length": args.sequence_length,
            "local_window": args.local_window,
            "lod_engine": args.lod_engine,
            "state_growth_factor": args.state_growth_factor,
            "routing_normalization": args.routing_normalization,
            "routing_count_bias": args.routing_count_bias,
            "routing_variance_bias": args.routing_variance_bias,
            "prefix_length": prefix_length,
            "canvas_length": canvas_length,
            "samples": args.samples,
            "metric": "corrupted-token denoising cross entropy",
            "modes": {},
        }
        for gathered_mode, statistics in zip(modes, gathered, strict=True):
            statistics = statistics.cpu()
            per_rate = {}
            total_loss = 0.0
            total_tokens = 0
            for rate_index, rate in enumerate(args.corruption_rates):
                rate_loss_sum = float(statistics[rate_index, 0])
                rate_tokens = int(statistics[rate_index, 1])
                loss = rate_loss_sum / rate_tokens
                per_rate[str(rate)] = {
                    "loss": loss,
                    "pseudo_perplexity": math.exp(min(loss, 80.0)),
                    "tokens": rate_tokens,
                }
                total_loss += rate_loss_sum
                total_tokens += rate_tokens
            loss = total_loss / total_tokens
            per_rate["overall"] = {
                "loss": loss,
                "pseudo_perplexity": math.exp(min(loss, 80.0)),
                "tokens": total_tokens,
            }
            summary["modes"][gathered_mode] = per_rate
        if set(summary["modes"]) == {"full", "lod"}:
            full = summary["modes"]["full"]["overall"]
            lod = summary["modes"]["lod"]["overall"]
            summary["difference"] = {
                "loss": lod["loss"] - full["loss"],
                "pseudo_perplexity_ratio": (
                    lod["pseudo_perplexity"] / full["pseudo_perplexity"]
                ),
            }
        summary["elapsed_seconds_rank0"] = elapsed
        summary["peak_vram_bytes_rank0"] = torch.cuda.max_memory_allocated(device)
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

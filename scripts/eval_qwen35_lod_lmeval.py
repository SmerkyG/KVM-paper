#!/usr/bin/env python3
"""Run lm-eval on Qwen3.5 with either full or top-k LOD attention."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table


def patch_hotpotqa_download() -> None:
    """Use the HF mirror because RULER's original HTTP host is unavailable."""
    from lm_eval.tasks.ruler import qa_utils

    # YAML task loading reloads helper modules unless they carry the loader's
    # mtime marker.  Preserve this patched module when qa_hotpot.yaml resolves
    # ``qa_utils.get_hotpotqa`` later.
    qa_utils.__mtime__ = Path(qa_utils.__file__).stat().st_mtime_ns

    def read_hotpotqa_hf():
        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="validation",
        )
        examples = list(dataset)
        documents = sorted(
            {
                f"{title}\n{''.join(sentences)}"
                for example in examples
                for title, sentences in zip(
                    example["context"]["title"],
                    example["context"]["sentences"],
                    strict=True,
                )
            }
        )
        document_indices = {document: index for index, document in enumerate(documents)}
        questions = []
        for example in examples:
            context = [
                document_indices[f"{title}\n{''.join(sentences)}"]
                for title, sentences in zip(
                    example["context"]["title"],
                    example["context"]["sentences"],
                    strict=True,
                )
            ]
            questions.append(
                {
                    "query": example["question"],
                    "outputs": [example["answer"]],
                    "context": context,
                }
            )
        return questions, documents

    qa_utils.read_hotpotqa = read_hotpotqa_hf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=("full", "two_level"), required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ruler-length", type=int)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--recursive-page-lod", action="store_true")
    parser.add_argument("--recursive-page-block-n", type=int, default=16)
    parser.add_argument("--leaf-num-warps", type=int, default=2)
    parser.add_argument("--leaf-key-quant-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument("--leaf-value-quant-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument("--leaf-quant-group-size", type=int, default=32)
    parser.add_argument(
        "--leaf-quant-scale-mode", choices=("max", "l2"), default="max"
    )
    parser.add_argument(
        "--leaf-append-quant-scale-mode", choices=("max", "l2"), default="max"
    )
    parser.add_argument("--page-summary-quant-bits", type=int, choices=(0, 8), default=8)
    parser.add_argument(
        "--page-summary-scale-mode", choices=("max", "l2"), default="l2"
    )
    parser.add_argument("--virtual-page-storage", action="store_true")
    parser.add_argument("--prefill-chunk-length", type=int)
    parser.add_argument("--prefill-local-length", type=int)
    parser.add_argument("--prefill-state-update-length", type=int)
    parser.add_argument("--split-prefill-local-attention", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-start", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.ruler_length is not None and args.ruler_length < 1:
        raise ValueError("RULER length must be positive")
    if (args.sample_start is None) != (args.sample_count is None):
        raise ValueError("sample start and count must be provided together")
    if args.sample_start is not None:
        if args.limit is not None:
            raise ValueError("sample ranges cannot be combined with limit")
        if len(args.tasks) != 1:
            raise ValueError("sample ranges require exactly one task")
        if args.sample_start < 0 or args.sample_count < 1:
            raise ValueError("sample start must be nonnegative and count positive")

    if args.mode == "two_level":
        # RULER's exact-length regrouping produces more than eight legitimate
        # batch shapes.  FlexAttention is compiled fullgraph, so the default
        # Dynamo recompile cap can otherwise fail only on the final batch.
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.cache_size_limit = 64
        # The inference-only LOD graft does not consume padding masks, so its
        # generation batches must be regrouped to equal unpadded lengths.
        import wrap_lmeval  # noqa: F401

    # Import after wrap_lmeval: Qwen's class must inherit the replacement
    # generation mixin when exact-length regrouping is required.
    from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
    from scripts.probe_qwen35_lod_niah import load_text_model

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    model = load_text_model(
        args.checkpoint,
        args.mode,
        args.two_level_topk,
        args.state_growth_factor,
        device,
        "paged",
    )
    if args.mode == "two_level":
        for module in model.modules():
            if not isinstance(module, Qwen3_5TwoLevelAttention):
                continue
            module.recursive_page_lod = args.recursive_page_lod
            module.recursive_page_block_n = args.recursive_page_block_n
            module.leaf_num_warps = args.leaf_num_warps
            module.leaf_key_quant_bits = args.leaf_key_quant_bits
            module.leaf_value_quant_bits = args.leaf_value_quant_bits
            module.leaf_quant_group_size = args.leaf_quant_group_size
            module.leaf_quant_scale_mode = args.leaf_quant_scale_mode
            module.leaf_append_quant_scale_mode = args.leaf_append_quant_scale_mode
            module.page_summary_quant_bits = args.page_summary_quant_bits
            module.page_summary_scale_mode = args.page_summary_scale_mode
            module.virtual_page_storage = args.virtual_page_storage
            # Decode has a fixed 512-token coarse K/V capacity.  Materializing
            # its small bias tensor avoids compiling one FlexAttention graph
            # for every live local length during long generations.
            if args.virtual_page_storage:
                module.coarse_compact_bias = False
            if args.prefill_chunk_length is not None:
                module.prefill_chunk_len = args.prefill_chunk_length
            if args.prefill_local_length is not None:
                module.prefill_local_len = args.prefill_local_length
            if args.prefill_state_update_length is not None:
                module.prefill_state_update_len = args.prefill_state_update_length
            module.split_prefill_local_attention = (
                args.split_prefill_local_attention
            )
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device=str(device),
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from accelerate import Accelerator

        accelerator = Accelerator()
        lm.accelerator = accelerator
        lm._rank = accelerator.process_index
        lm._world_size = accelerator.num_processes
    patch_hotpotqa_download()

    metadata = {
        "pretrained": args.checkpoint,
        "tokenizer": args.checkpoint,
    }
    if args.ruler_length is not None:
        metadata["max_seq_lengths"] = [args.ruler_length]
    samples = None
    if args.sample_start is not None:
        samples = {
            args.tasks[0]: list(
                range(args.sample_start, args.sample_start + args.sample_count)
            )
        }
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        samples=samples,
        bootstrap_iters=0,
        log_samples=False,
        metadata=metadata,
        confirm_run_unsafe_code=True,
    )
    if results is None:
        return

    payload = dict(results)
    payload["lod_evaluation"] = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "sample_start": args.sample_start,
        "sample_count": args.sample_count,
        "ruler_length": args.ruler_length,
        "two_level_topk": (
            args.two_level_topk if args.mode == "two_level" else None
        ),
        "state_growth_factor": (
            args.state_growth_factor if args.mode == "two_level" else None
        ),
        "recursive_page_lod": (
            args.recursive_page_lod if args.mode == "two_level" else None
        ),
        "recursive_page_block_n": (
            args.recursive_page_block_n if args.mode == "two_level" else None
        ),
        "leaf_num_warps": (
            args.leaf_num_warps if args.mode == "two_level" else None
        ),
        "leaf_key_quant_bits": (
            args.leaf_key_quant_bits if args.mode == "two_level" else None
        ),
        "leaf_value_quant_bits": (
            args.leaf_value_quant_bits if args.mode == "two_level" else None
        ),
        "leaf_quant_group_size": (
            args.leaf_quant_group_size if args.mode == "two_level" else None
        ),
        "leaf_quant_scale_mode": (
            args.leaf_quant_scale_mode if args.mode == "two_level" else None
        ),
        "leaf_append_quant_scale_mode": (
            args.leaf_append_quant_scale_mode if args.mode == "two_level" else None
        ),
        "page_summary_quant_bits": (
            args.page_summary_quant_bits if args.mode == "two_level" else None
        ),
        "virtual_page_storage": (
            args.virtual_page_storage if args.mode == "two_level" else None
        ),
        "coarse_compact_bias": (
            not args.virtual_page_storage if args.mode == "two_level" else None
        ),
        "prefill_chunk_length": (
            args.prefill_chunk_length if args.mode == "two_level" else None
        ),
        "prefill_local_length": (
            args.prefill_local_length if args.mode == "two_level" else None
        ),
        "prefill_state_update_length": (
            args.prefill_state_update_length if args.mode == "two_level" else None
        ),
        "split_prefill_local_attention": (
            args.split_prefill_local_attention
            if args.mode == "two_level"
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(make_table(results))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""Llama-3.1-8B CUDA serving smoke checks and matched ProLong timing.

Submit GPU work through cluster-run. Speed prompts use the existing pinned
ProLong dataset and token hashes, with two warmups and repeated measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from benchmarks._identity import benchmark_identity
from benchmarks._vllm import MODES, close_llm, llm_kwargs, write_json


def retrieval_prompt(tokenizer, length, depth, key):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    prefix = encode(
        "Below is an archive. Find the secret access code and report only its six digits.\n"
    )
    needle = encode(f"\nThe secret access code is {key}. Remember this code.\n")
    suffix = encode(
        "\nWhat is the secret access code? Report only its six digits.\nAnswer:"
    )
    filler = encode(
        "The archive contains routine records about weather, roads, and buildings.\n"
    )
    room = length - len(prefix) - len(needle) - len(suffix)
    if room < 0:
        raise ValueError("Retrieval length is too short for the template")
    repeated = (filler * ((room + len(filler) - 1) // len(filler)))[:room]
    cut = int(room * depth)
    ids = prefix + repeated[:cut] + needle + repeated[cut:] + suffix
    assert len(ids) == length
    return dict(prompt_token_ids=ids), dict(
        length=length,
        depth=depth,
        expected=key,
        prompt_token_sha256=hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def prepare_speed_prompts(args, tokenizer, lengths):
    """Freeze the token cohort before model startup and share it across modes."""
    from benchmarks.prolong import (
        make_speed_prompts,
        token_digest,
        DATASET,
        DATASET_REVISION,
    )

    identity = dict(
        tokenizer_sha256=hashlib.sha256(
            (Path(args.checkpoint) / "tokenizer.json").read_bytes()
        ).hexdigest(),
        batch_size=args.batch_size,
        dataset=DATASET,
        dataset_revision=DATASET_REVISION,
        data_files=[
            dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in args.dataset_file
        ],
    )
    if args.prompt_manifest and args.prompt_manifest.exists():
        manifest = json.loads(args.prompt_manifest.read_text())
        if manifest["identity"] != identity:
            raise ValueError(
                "Prompt manifest has a different tokenizer, batch, or dataset source"
            )
    else:
        manifest = dict(identity=identity, contexts={})
        for length in lengths:
            prompts, metadata = make_speed_prompts(
                tokenizer,
                length=length,
                batch_size=args.batch_size,
                data_files=[str(path) for path in args.dataset_file] or None,
            )
            manifest["contexts"][str(length)] = dict(prompts=prompts, metadata=metadata)
        if args.prompt_manifest:
            write_json(args.prompt_manifest, manifest)
    for length in lengths:
        entry = manifest["contexts"][str(length)]
        if (
            len(entry["prompts"]) != args.batch_size
            or len(entry["metadata"]) != args.batch_size
        ):
            raise ValueError("Prompt manifest has a different cohort size")
        for prompt, metadata in zip(entry["prompts"], entry["metadata"]):
            ids = prompt["prompt_token_ids"]
            if len(ids) != length or token_digest(ids) != metadata["token_sha256"]:
                raise ValueError(
                    "Prompt manifest length or token digest does not match"
                )
    return manifest


def worker_state(worker):
    """Describe the actual worker's authoritative LoD pools after generation."""
    runner = worker.model_runner
    found = []
    candidates = [
        runner,
        getattr(runner, "model_state", None),
        getattr(runner, "_model_state", None),
    ]
    for owner in candidates:
        if owner is None:
            continue
        for name, value in vars(owner).items():
            if value.__class__.__name__ == "VLLMLODRuntime":
                found.append(
                    dict(
                        attribute=name,
                        layers=len(value.pools),
                        pool_size=value.pool_size,
                        request_capacity=value.request_capacity,
                        pools=[
                            dict(
                                layer=key,
                                family=pool.family.value,
                                levels=pool.settings.levels,
                                kv_bits=pool.settings.kv_bits,
                                prefill_backend=pool.engine.prefill_local_attention_backend,
                                cuda_routes=pool.engine.prefill_cuda_route_coarse,
                                state_capacity=pool.state_capacity,
                                decode_route_geometry=dict(
                                    group_size=pool.engine.decode_route_group_size,
                                    num_warps=pool.engine.decode_route_num_warps,
                                    reduce_num_warps=pool.engine.decode_route_reduce_num_warps,
                                    parallel_reduce=pool.engine.decode_route_parallel_reduce,
                                ),
                            )
                            for key, pool in list(value.pools.items())[:2]
                        ],
                    )
                )
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=(*MODES, "quest-bf16"), required=True)
    p.add_argument("--quest-budget-tokens", type=int, default=8192,
                   help="QUEST exact-token budget per head, rounded up to pages")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", default="8192,32768,65536")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--decode-tokens", type=int, default=1025)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--memory", type=float, default=0.8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--dataset-file",
        type=Path,
        action="append",
        default=[],
        help="Pinned local ProLong parquet shard; repeat to select more shards",
    )
    p.add_argument(
        "--prompt-manifest",
        type=Path,
        help="Create or reuse this frozen speed-prompt cohort",
    )
    p.add_argument(
        "--retrieval",
        action="store_true",
        help="Run deterministic passkey checks at three depths per length",
    )
    p.add_argument(
        "--retrieval-cohort",
        action="store_true",
        help="Run one concurrent retrieval cohort with distinct keys and mixed context lengths",
    )
    args = p.parse_args()
    if args.mode == "quest-bf16" and (args.batch_size != 1 or args.quest_budget_tokens < 1):
        raise ValueError("QUEST comparison requires batch size one and a positive budget")
    lengths = [int(x) for x in args.lengths.split(",")]
    capacity = args.max_model_len or (max(lengths) + args.decode_tokens + 16)
    if not args.smoke and capacity < max(lengths) + args.decode_tokens:
        raise ValueError(
            "Configured capacity cannot contain the requested context and output"
        )
    if min(args.batch_size, args.repeats, args.warmups) < 1 or args.decode_tokens < 2:
        raise ValueError(
            "Require a positive batch, warmups/repeats, and at least two output tokens"
        )
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    kwargs = llm_kwargs(
        checkpoint=args.checkpoint,
        mode="full" if args.mode == "quest-bf16" else args.mode,
        max_model_len=capacity,
        batch_size=args.batch_size,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.memory,
        full_attention_backend="FLASH_ATTN",
    )
    if args.mode == "full":
        # The dense baseline is the unmodified native Llama implementation.
        os.environ["VLLM_PLUGINS"] = ""
    elif args.mode == "quest-bf16":
        os.environ["VLLM_COMPARISON_BACKEND"] = "quest"
        os.environ["QUEST_BUDGET_TOKENS"] = str(args.quest_budget_tokens)
        kwargs["attention_config"] = {"backend": "CUSTOM"}
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    result = dict(
        benchmark="h100-llama31",
        checkpoint=args.checkpoint,
        mode=args.mode,
        quest_budget_tokens=args.quest_budget_tokens if args.mode == "quest-bf16" else None,
        batch_size=args.batch_size,
        max_model_len=capacity,
        decode_tokens=args.decode_tokens,
        warmups=args.warmups,
        repeats=args.repeats,
        benchmark_identity=benchmark_identity(),
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        node=os.environ["CLUSTER_RUN_NODE"],
        gpus=os.environ["CLUSTER_RUN_GPUS"],
        llm_kwargs=kwargs,
        stage="loading_model",
        measurements={},
    )
    write_json(args.output, result)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    speed_prompts = None
    if not args.smoke and not args.retrieval and not args.retrieval_cohort:
        speed_prompts = prepare_speed_prompts(args, tokenizer, lengths)
        result["prompt_source"] = speed_prompts["identity"]
        result["prompt_manifest_sha256"] = hashlib.sha256(
            json.dumps(speed_prompts, sort_keys=True).encode()
        ).hexdigest()
        write_json(args.output, result)
    started = time.perf_counter()
    llm = LLM(**kwargs)
    result["startup_seconds"] = time.perf_counter() - started
    try:
        questions = [
            "What is the capital of France? Answer with one word.",
            "What is 17 plus 25? Answer with just the number.",
        ]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=32, seed=0),
            use_tqdm=False,
        )
        result["smoke_responses"] = [x.outputs[0].text for x in outputs]
        print("SMOKE_RESPONSES " + json.dumps(result["smoke_responses"]), flush=True)
        if args.smoke:
            result["stage"] = "complete"
        elif args.retrieval_cohort:
            cases = [
                retrieval_prompt(
                    tokenizer,
                    lengths[i % len(lengths)],
                    (0.1, 0.5, 0.9)[i % 3],
                    str(731925 + i * 43),
                )
                for i in range(args.batch_size)
            ]
            outputs = llm.generate(
                [prompt for prompt, _ in cases],
                SamplingParams(temperature=0, max_tokens=32, seed=0),
                use_tqdm=False,
            )
            records = []
            for output, (_, metadata) in zip(outputs, cases):
                text = output.outputs[0].text
                row = metadata | dict(
                    response=text, correct=metadata["expected"] in text
                )
                records.append(row)
                print("RETRIEVAL " + json.dumps(row), flush=True)
            result.update(
                stage="complete",
                retrieval=records,
                retrieval_correct=sum(row["correct"] for row in records),
                retrieval_total=len(records),
            )
        elif args.retrieval:
            records = []
            for length in lengths:
                for depth in (0.1, 0.5, 0.9):
                    key = f"{731925 + int(depth * 1000)}"
                    prompt, metadata = retrieval_prompt(tokenizer, length, depth, key)
                    output = llm.generate(
                        [prompt],
                        SamplingParams(temperature=0, max_tokens=32, seed=0),
                        use_tqdm=False,
                    )[0]
                    text = output.outputs[0].text

                    row = dict(
                        **metadata,
                        response=text,
                        correct=key in text,
                    )
                    records.append(row)
                    result.update(stage="retrieval", retrieval=records)
                    write_json(args.output, result)
                    print("RETRIEVAL " + json.dumps(row), flush=True)
            result.update(
                stage="complete",
                retrieval_correct=sum(row["correct"] for row in records),
                retrieval_total=len(records),
            )
        else:
            from benchmarks.prolong import (
                timed_generate,
                token_digest,
                DATASET,
                DATASET_REVISION,
            )

            result.update(
                dataset=DATASET, dataset_revision=DATASET_REVISION, stage="measuring"
            )
            params = SamplingParams(
                temperature=0,
                max_tokens=args.decode_tokens,
                seed=0,
                ignore_eos=True,
                detokenize=False,
            )
            for length in lengths:
                entry = speed_prompts["contexts"][str(length)]
                prompts, metadata = entry["prompts"], entry["metadata"]
                records = []
                for call in range(args.warmups + args.repeats):
                    phase = "warmup" if call < args.warmups else "measured"
                    print(
                        "TIMING_START "
                        + json.dumps(
                            dict(
                                length=length, call=call, phase=phase, epoch=time.time()
                            )
                        ),
                        flush=True,
                    )
                    elapsed, prefill, decode, tokens, counters = timed_generate(
                        llm, prompts, params
                    )
                    row = dict(
                        phase=phase,
                        elapsed_seconds=elapsed,
                        prefill_seconds=prefill,
                        decode_seconds=decode,
                        decode_ms_per_batch_step=decode
                        * 1000
                        / (args.decode_tokens - 1),
                        output_token_sha256=[token_digest(list(ids)) for ids in tokens],
                        counters=counters,
                    )
                    records.append(row)
                    print(
                        "TIMING_END "
                        + json.dumps(
                            row | dict(length=length, call=call, epoch=time.time())
                        ),
                        flush=True,
                    )
                    result["measurements"][str(length)] = dict(
                        prompts=metadata, records=records
                    )
                    write_json(args.output, result)
                measured = [row for row in records if row["phase"] == "measured"]
                result["measurements"][str(length)].update(
                    prefill_seconds=statistics.median(
                        row["prefill_seconds"] for row in measured
                    ),
                    decode_ms_per_batch_step=statistics.median(
                        row["decode_ms_per_batch_step"] for row in measured
                    ),
                )
                write_json(args.output, result)
            result["stage"] = "complete"
        result["worker_state"] = llm.collective_rpc(worker_state)
        write_json(args.output, result)
        print("RESULT " + str(args.output), flush=True)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

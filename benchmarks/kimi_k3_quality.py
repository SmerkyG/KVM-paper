"""Matched full-K3 LongBench v2 and RULER NIAH-S3 quality evaluation.

Use the native K3 XTML template, with thinking disabled, and the same resident
MoE weights as the ProLong comparison. No speed fixture or owner experiment.
The offline engine uses LongBench's existing prompt/scorer and answer grammar;
it does not include HTTP/frontend overhead in its elapsed time.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import longbench_v2
from ._identity import benchmark_identity
from ._vllm import close_llm, llm_kwargs, write_json
from .niah_s3 import _load_ruler_generator
from .prolong import audit_worker_attention_mode, comma_separated_ints, token_digest
from .prolong import validate_worker_attention_mode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=("full", "two-tier"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", default="niah-s3,longbench-v2")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--weight-cache-id", default="kimi-k3-shared-int4-v6")
    p.add_argument("--kv-cache-memory-bytes", type=int, default=2 * 1024**3)
    p.add_argument("--niah-lengths", type=comma_separated_ints,
                   default=[8192, 16384, 32768, 65536])
    p.add_argument("--niah-samples", type=int, default=128)
    p.add_argument("--longbench-limit", type=int)
    p.add_argument("--max-input-tokens", type=int, default=131072)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Diagnostic only: disable graph replay without changing attention math.")
    p.add_argument("--dense-shadow-decode", action="store_true",
                   help="Diagnostic only: eager dense decode over the same LoD raw archive.")
    return p.parse_args()


def chat_ids(tokenizer: Any, user: str, assistant_prefix: str = "") -> list[int]:
    """K3 ignores continue_final_message: append to an OPEN response instead.

    Structural XTML markers must be encoded by the tokenizer itself. Never
    decode/re-encode the rendered chat or remove markers by string slicing.
    """
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=True,
        add_generation_prompt=True, thinking=False,
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids) + tokenizer.encode(assistant_prefix, add_special_tokens=False)


def evenly_spaced_panel(items: list[Any], limit: int | None) -> list[Any]:
    """A limited smoke panel covers the full length range, not just short rows."""
    if limit is None or limit >= len(items):
        return items
    if limit < 1:
        raise ValueError("longbench-limit must be positive")
    if limit == 1:
        return [items[len(items) // 2]]
    return [items[i * (len(items) - 1) // (limit - 1)] for i in range(limit)]


def prepare_longbench(tokenizer: Any, *, max_input_tokens: int,
                      limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = []
    for item in load_dataset(longbench_v2.DATASET,
                             revision=longbench_v2.DATASET_REVISION, split="train"):
        prompt, original, truncated = longbench_v2.truncate_prompt(
            tokenizer, longbench_v2.make_prompt(item), max_input_tokens,
        )
        ids = chat_ids(tokenizer, prompt)
        rows.append(dict(
            _id=item["_id"], domain=item["domain"], sub_domain=item["sub_domain"],
            difficulty=item["difficulty"], length=item["length"], answer=item["answer"],
            original_input_tokens=original, truncated=truncated,
            sent_input_tokens=len(tokenizer.encode(prompt, add_special_tokens=False)),
            input_tokens=len(ids), token_sha256=token_digest(ids), prompt_token_ids=ids,
        ))
    rows.sort(key=lambda row: row["input_tokens"])
    return evenly_spaced_panel(rows, limit)


def prepare_niah(tokenizer: Any, *, length: int, samples: int) -> list[dict[str, Any]]:
    generate_samples, get_haystack, _ = _load_ruler_generator()
    from lm_eval.tasks.ruler.niah_utils import TEMPLATE

    random.seed(0)
    np.random.seed(1234)
    documents = generate_samples(
        get_haystack(type_haystack="essay"), max_seq_length=length,
        template=TEMPLATE, type_haystack="essay", type_needle_k="words",
        type_needle_v="uuids", num_samples=samples, TOKENIZER=tokenizer,
    )
    rows = []
    for doc in documents:
        ids = chat_ids(tokenizer, doc["input"], doc["gen_prefix"])
        rows.append(dict(index=int(doc["index"]), target=str(doc["outputs"][0]),
                         input_tokens=len(ids), token_sha256=token_digest(ids),
                         prompt_token_ids=ids))
    return rows


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k != "prompt_token_ids"} for row in rows]


def engine_kwargs(args: argparse.Namespace, *, max_model_len: int) -> dict[str, Any]:
    # Free generation ends requests at different steps and can leave a partial
    # cohort. The captured owner layout currently requires all eight live rows.
    os.environ["LOD_KIMI_REQUEST_OWNER_PREFILL"] = "0"
    os.environ["LOD_KIMI_REQUEST_OWNER_DECODE"] = "0"
    kwargs = llm_kwargs(
        checkpoint=args.checkpoint, mode=args.mode, max_model_len=max_model_len,
        batch_size=args.batch_size, tensor_parallel_size=8,
        decode_context_parallel_size=8, dcp_comm_backend="ag_rs",
        gpu_memory_utilization=0.8, full_attention_backend="TRITON_MLA",
    )
    kwargs.update(
        enable_expert_parallel=True, disable_custom_all_reduce=False,
        quantization_config={"moe": {"weight": "int4_per_group_32"}},
        safetensors_load_strategy="prefetch", load_format="ipc_cache",
        model_loader_extra_config=dict(auto_start=True, cache_id=args.weight_cache_id,
            cache_dir=None, backing_load_format="auto", broker_timeout=1800.0),
        kv_cache_memory_bytes=args.kv_cache_memory_bytes, seed=0,
    )
    if getattr(args, "enforce_eager", False):
        kwargs["enforce_eager"] = True
    return kwargs


def evaluate(llm: Any, rows: list[dict[str, Any]], *, task: str, batch_size: int,
             progress: Any) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    params = SamplingParams(temperature=0, seed=0,
        max_tokens=32 if task == "longbench-v2" else 64,
        structured_outputs=(StructuredOutputsParams(
            choice=[f"The correct answer is ({c})" for c in "ABCD"])
            if task == "longbench-v2" else None))
    records = []
    started = time.perf_counter()
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin:begin + batch_size]
        outputs = llm.generate([{"prompt_token_ids": r["prompt_token_ids"]}
                                for r in batch], params, use_tqdm=True)
        for row, output in zip(batch, outputs, strict=True):
            generated = output.outputs[0]
            response = generated.text
            record = {k: v for k, v in row.items() if k != "prompt_token_ids"}
            record.update(response=response, output_tokens=len(generated.token_ids),
                          finish_reason=generated.finish_reason)
            if task == "longbench-v2":
                prediction = longbench_v2.extract_answer(response)
                if prediction is None:
                    raise RuntimeError(f"unparseable constrained LongBench answer: {response!r}")
                record.update(prediction=prediction, correct=prediction == row["answer"])
            else:
                record["correct"] = row["target"].lower() in response.lower()
            records.append(record)
        correct = sum(r["correct"] for r in records)
        result = dict(correct=correct, total=len(records), accuracy=correct/len(records),
                      samples=list(records), elapsed_seconds=time.perf_counter()-started)
        if task == "longbench-v2":
            result["metrics"] = longbench_v2.summarize(records)
        progress(result)
        print("KIMI_QUALITY_PROGRESS " + json.dumps(dict(task=task,
            completed=len(records), requested=len(rows), correct=correct)), flush=True)
    return result


def main() -> None:
    args = parse_args()
    tasks = args.tasks.split(",")
    if not tasks or len(set(tasks)) != len(tasks) or any(
        t not in ("niah-s3", "longbench-v2") for t in tasks
    ):
        raise ValueError("tasks must be niah-s3 and/or longbench-v2")
    if min(args.batch_size, args.niah_samples, args.kv_cache_memory_bytes) < 1:
        raise ValueError("batch size, sample count, and cache reservation must be positive")
    if getattr(args, "dense_shadow_decode", False) and (
            args.mode != "two-tier" or not getattr(args, "enforce_eager", False)):
        raise ValueError("dense shadow decode requires two-tier mode and --enforce-eager")
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    text_config = config.get("text_config", config)
    if int(text_config.get("num_hidden_layers", 0)) != 93:
        raise ValueError("this comparison requires the full trained 93-layer K3 checkpoint")
    forbidden = ("LOD_KIMI_REQUEST_OWNER_PREFILL", "LOD_KIMI_REQUEST_OWNER_DECODE",
                 "LOD_KIMI_DCP_SHARDED_LEAVES", "LOD_KIMI_DCP_LOCAL_PREFILL",
                 "LOD_KIMI_DCP_SHARED_PREFILL")
    if any(os.getenv(k) == "1" for k in forbidden):
        raise ValueError("owner/slice-local/sharded experiments are not this quality comparison")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    panel = {}
    lm_eval_version = None
    if "niah-s3" in tasks:
        _, _, lm_eval_version = _load_ruler_generator()
        for length in args.niah_lengths:
            panel[f"niah-s3:{length}"] = prepare_niah(
                tokenizer, length=length, samples=args.niah_samples)
    if "longbench-v2" in tasks:
        panel["longbench-v2"] = prepare_longbench(tokenizer,
            max_input_tokens=args.max_input_tokens, limit=args.longbench_limit)
    max_input = max(r["input_tokens"] for rows in panel.values() for r in rows)
    kwargs = engine_kwargs(args, max_model_len=max_input + 128)
    result = dict(benchmark="kimi-k3-chat-quality", checkpoint=args.checkpoint,
        mode=args.mode, hostname=platform.node(), argv=sys.argv, batch_size=args.batch_size,
        benchmark_identity=benchmark_identity(), engine_kwargs=kwargs,
        thinking=False, assistant_prefix="open XTML response + ordinary prefix tokens",
        lm_eval_version=lm_eval_version, longbench_dataset=longbench_v2.DATASET,
        longbench_revision=longbench_v2.DATASET_REVISION, max_input_tokens=args.max_input_tokens,
        environment={k:v for k,v in os.environ.items() if k.startswith(("LOD_", "VLLM_", "AITER_"))},
        prompt_manifest={name:public_rows(rows) for name,rows in panel.items()}, results={})
    write_json(args.output.with_suffix(".prompts.json"), result)
    print("KIMI_QUALITY_PREFLIGHT " + json.dumps({
        name:dict(samples=len(rows), min_tokens=min(r["input_tokens"] for r in rows),
                  max_tokens=max(r["input_tokens"] for r in rows))
        for name,rows in panel.items()}), flush=True)
    if args.preflight_only:
        return

    from vllm import LLM
    llm = LLM(**kwargs)
    try:
        if getattr(args, "dense_shadow_decode", False):
            from ._kimi_decode_reference import install_dense_shadow_decoder
            result["dense_shadow_decode_install"] = llm.collective_rpc(install_dense_shadow_decoder)
        before = llm.collective_rpc(audit_worker_attention_mode)
        validate_worker_attention_mode(before, mode=args.mode)
        result["worker_attention_audit_before"] = before
        for name, rows in panel.items():
            def progress(measured: dict[str, Any]) -> None:
                result["results"][name] = measured
                write_json(args.output.with_suffix(".partial.json"), result)
            result["results"][name] = evaluate(llm, rows, task=name.split(":")[0],
                batch_size=args.batch_size, progress=progress)
        audit = llm.collective_rpc(audit_worker_attention_mode)
        validate_worker_attention_mode(audit, mode=args.mode, require_loaded_kimi_lod=True)
        result["worker_attention_audit"] = audit
        if getattr(args, "dense_shadow_decode", False):
            from ._kimi_decode_reference import dense_shadow_audit
            result["dense_shadow_decode_audit"] = llm.collective_rpc(dense_shadow_audit)
        write_json(args.output, result)
        if getattr(args, "dense_shadow_decode", False) and not all(
                any(entry["calls"] for entry in worker["reference_calls"].values())
                for worker in result["dense_shadow_decode_audit"]):
            raise RuntimeError("dense shadow diagnostic never reached every worker's DCP decoder")
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()

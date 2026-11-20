"""Audit BF16 archive consumers during aligned and mixed-length offline prefill."""
import argparse
import json
import os
from pathlib import Path

from benchmarks._vllm import close_llm, llm_kwargs, default_gpu_memory_utilization
from benchmarks.prolong import make_speed_prompts

def counters(model):
    reports = []
    for layer in model.modules():
        pool = getattr(layer, "_vllm_lod_pool", None)
        stats = getattr(pool, "_handoff_audit", None)
        if stats is not None:
            reports.append({**stats, "previous_lengths": sorted(stats["previous_lengths"])})
            del pool._handoff_audit
    return reports

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    prompts, hashes = make_speed_prompts(tokenizer, length=32768, batch_size=8)
    llm = LLM(**llm_kwargs(checkpoint=args.checkpoint, mode=args.mode, max_model_len=32784,
                           batch_size=8, tensor_parallel_size=1, full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
                           gpu_memory_utilization=default_gpu_memory_utilization(args.checkpoint, args.mode)))
    cases = {"mixed": (24576, 31744, 32768, 28672, 20480, 27648, 12288, 16384)}
    results = {}
    try:
        for name,lengths in cases.items():
            batch = [{"prompt_token_ids":p["prompt_token_ids"][:n]} for p,n in zip(prompts,lengths,strict=True)]
            llm.generate(batch, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False),
                         use_tqdm=False)
            (layers,) = llm.apply_model(counters)
            results[name] = {"lengths": lengths, "layers": layers,
                             "checked": sum(a["checked"] for a in layers),
                             "mismatches": sum(a["mismatches"] for a in layers)}
            args.output.write_text(json.dumps({"cases":results,"prompt_sources":hashes},indent=2)+"\n")
            print(name, {k:results[name][k] for k in ("checked","mismatches")}, flush=True)
    finally:
        close_llm(llm)

if __name__ == "__main__":
    main()

# RULER

**Kernel validation:** A race in the shared LoD state-update kernel was
reproduced on H100 and fixed. Reference checks on ROCm passed both before and
after the fix for 128- and 256-dimensional heads at batches 1, 4 and 8; those
checks do not replace a full rerun of the model-quality evaluations below.
[ROCm check records](../results/h100-baselines/provenance/rocm-state-validation/)

This benchmark evaluates all 13 RULER subtasks at a 65,536-token context
length. Each task contains 500 examples. The full-attention and two-tier
baselines use `lm-eval==0.4.13`, greedy generation through vLLM's
OpenAI-compatible chat-completions endpoint, eight concurrent requests,
and the canonical `42,42,42,42` lm-eval seeds.
Chat templating is enabled, model thinking is disabled, and each task's
generation prefix is represented as an assistant-prefill message using
`continue_final_message=True`.

The three-tier BF16 and INT4 columns use optimized top-2 page opening during
decode: eight routed regions and up to two distinct pages per selected region.
They replay the exact archived requests with API seed `1234` and the original
per-task generation budgets. BF16 and INT4 describe KV storage; both models
use FP8 weights. The completed [Qwen panel](../results/qwen-ruler-top2-64k/REPORT.md)
and [K2 panel](../results/k2-ruler-top2-64k/REPORT.md) contain all 13 tasks per
precision, with audited response logs, comparisons, and source provenance.

## Results

Scores are percentages and higher is better. The metric is each task's
`65536,none` RULER score from lm-eval 0.4.13; the replay scorer was validated
against the archived sample scores. All 13 tasks are complete for both models
under full attention, two-tier LoD, and three-tier BF16/INT4 with top-2.
Every task contains exactly 500 logged examples and no evaluator-fatal errors.
Means are unweighted across the 13 task scores. Historical serving revisions,
capacity, and request scheduling differ, so cross-run differences are descriptive.

### Qwen3.8-27B-FP8

| Task | Full attention | Two-tier LoD | Three-tier BF16 (top-2) | Three-tier INT4 (top-2) |
|---|---:|---:|---:|---:|
| NIAH single 1 | 100.00 | 100.00 | 100.00 | 100.00 |
| NIAH single 2 | 100.00 | 99.60 | 99.80 | 99.60 |
| NIAH single 3 | 100.00 | 99.80 | 99.60 | 99.60 |
| NIAH multikey 1 | 100.00 | 99.00 | 100.00 | 99.80 |
| NIAH multikey 2 | 100.00 | 96.20 | 94.40 | 95.20 |
| NIAH multikey 3 | 100.00 | 92.60 | 88.80 | 86.80 |
| NIAH multiquery | 100.00 | 99.90 | 99.85 | 99.80 |
| NIAH multivalue | 100.00 | 99.15 | 97.50 | 97.45 |
| Common-words extraction | 99.98 | 99.90 | 99.96 | 100.00 |
| Frequent-words extraction | 99.40 | 98.87 | 98.47 | 98.33 |
| HotpotQA | 72.00 | 67.20 | 65.60 | 65.60 |
| SQuAD QA | 78.85 | 74.88 | 72.53 | 72.33 |
| Variable tracking | 100.00 | 99.96 | 99.96 | 99.96 |
| **Mean over all 13 tasks** | **96.17** | **94.39** | **93.57** | **93.42** |

Two-tier LoD is 1.78 percentage points below full attention on the unweighted
13-task mean. Three-tier BF16 and INT4 with top-2 score 93.57% and 93.42%,
respectively, 0.82 and 0.97 points below two-tier.

### K2-Horizon-32B-FP8

| Task | Full attention | Two-tier LoD | Three-tier BF16 (top-2) | Three-tier INT4 (top-2) |
|---|---:|---:|---:|---:|
| NIAH single 1 | 100.00 | 100.00 | 100.00 | 100.00 |
| NIAH single 2 | 100.00 | 100.00 | 100.00 | 100.00 |
| NIAH single 3 | 100.00 | 100.00 | 100.00 | 100.00 |
| NIAH multikey 1 | 100.00 | 99.60 | 99.60 | 99.40 |
| NIAH multikey 2 | 99.00 | 93.00 | 80.60 | 78.60 |
| NIAH multikey 3 | 100.00 | 98.20 | 79.80 | 80.80 |
| NIAH multiquery | 100.00 | 100.00 | 99.95 | 100.00 |
| NIAH multivalue | 97.70 | 95.95 | 96.20 | 94.95 |
| Common-words extraction | 92.10 | 94.90 | 94.86 | 94.68 |
| Frequent-words extraction | 85.33 | 85.40 | 85.40 | 84.73 |
| HotpotQA | 60.40 | 57.40 | 57.00 | 56.20 |
| SQuAD QA | 75.30 | 72.08 | 71.37 | 68.07 |
| Variable tracking | 100.00 | 100.00 | 100.00 | 100.00 |
| **Mean over all 13 tasks** | **93.06** | **92.04** | **89.60** | **89.03** |

Two-tier LoD is 1.02 percentage points below full attention on the unweighted
13-task mean. The K2 full-attention versus two-tier category means are 99.59
versus 98.34 for NIAH, 88.72 versus 90.15 for extraction, 67.85 versus 64.74
for QA, and 100.00 versus 100.00 for variable tracking.
Three-tier BF16 and INT4 with top-2 score 89.60% and 89.03%, respectively,
2.44 and 3.01 points below two-tier on the 13-task mean.

## AMD two-tier threshold follow-ups

These experiments use batch one, BF16 KV, normalized-query equal-per-region
**c=16 decode routing**, and the existing top-8 sparse prefill with an exact
first 16,384-token chunk. The full historical tables above retain their
original fixed-eight two-tier results. Fresh top-8 controls retain native
query scores, so the routing comparison changes both the score and the
selection rule.

The [paired 500-example comparison](../results/amd-threshold/full500/REPORT.md)
covers four Qwen tasks and five K2 tasks selected for misses. Threshold routing
matched or improved every task's aggregate score versus fresh top-8 controls.
The selected-task mean rose from 84.44% to 85.06% for Qwen and from 83.22% to
84.42% for K2. These are different task subsets and are not 13-task averages.
Archived dense results on the same requests are included for context.

The [optimized-kernel regression panel](../results/amd-threshold-ruler32-64k/REPORT.md)
contains all 13 tasks, with the first 32 examples per task and model. Compared
with the old c=16 kernel, the initially optimized kernel scored 94.99% to
94.67% on Qwen and 92.17% to 92.32% on K2. The separate
[Qwen precision follow-up](../results/amd-threshold-ruler32-64k/precision-panel/REPORT.md)
scored 95.15%, with no retrieval-task score losses versus the old kernel;
one HotpotQA example still declined. Its attention-only timing was 41.02 µs
versus 39.21 µs for native top-8, and full-model timing with this correction
has not been measured. These small panels do not establish noninferiority
on the 500-example benchmark or replace its results. Frozen requests,
independently audited token hashes, exact commands and successful job exits
are linked from each report. Serving defaults are unchanged.

## AMD three-tier threshold pilot

The [three-tier threshold report](../results/amd-threshold-three-tier/REPORT.md)
adds batch-one c=16 normalized-query region routing with the existing top-two
page frontier, for both models in BF16 and INT4 KV. It includes paired
32-example Multikey 2/3 screens, 1,024-step full-model timings with cache
updates, and same-query attention measurements. Prefill retains top-8 routing
and the exact first 16K chunk. The path is opt-in; serving defaults are unchanged.

These selected retrieval screens are separate from the 500-example, all-13-task
tables above. Current K2 native controls score substantially below the historical
report; that discrepancy remains unresolved. The page-kernel extension matched
the pre-change native outputs bitwise on 12 sampled serving states, and all 24
independent frontier checks plus 66 existing page-opening tests passed.

## AMD threshold prefill pilot

The [Qwen threshold prefill report](../results/amd-threshold-prefill/REPORT.md)
compares top-8 and normalized-query c=16 region routing during BF16 two-tier
prefill. It includes 32 paired examples each of Multikey 2 and Multikey 3,
with the same precise c=16 decoder in both arms, plus 32K/64K full-model
prefill timings against top-8 and dense. The exact first 16K chunk remains
unchanged. These retrieval screens are separate from the full benchmark above.

## Reproduce

These instructions reproduce the full-attention and two-tier baselines with
the corrected chat template and assistant prefill. Exact `cluster-run`/`uv`
commands and frozen source/runtime identities for the three-tier top-2 runs
are recorded in the [Qwen manifest](../results/qwen-ruler-top2-64k/runs.json)
and [K2 manifest](../results/k2-ruler-top2-64k/runs.json).

Install the serving and benchmark dependencies from the repository root and
apply/build the AITER patch described in the root README:

```bash
uv sync --extra vllm --extra benchmarks
```

Start the Qwen two-tier LoD server in one terminal:

```bash
export PYTHONPATH="$PWD:$PWD/integrations/vllm_lod${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=lod_attention
export VLLM_LOD_MODE=two-tier
export VLLM_LOD_POOL_SIZE=8
export VLLM_LOD_MAX_CONTEXT=66560
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run vllm serve Qwen/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --model-impl vllm \
  --renderer-num-workers 8 \
  --language-model-only \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 66560 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16392 \
  --long-prefill-token-threshold 16384 \
  --scheduler-cls vllm_lod_plugin.scheduler.LODChunkAlignedScheduler \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --attention-config '{"backend":"CUSTOM"}'
```

Run all 13 tasks against that server from a second terminal:

```bash
tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multiquery niah_multivalue
  ruler_cwe ruler_fwe ruler_qa_hotpot ruler_qa_squad ruler_vt
)

for task in "${tasks[@]}"; do
  uv run --with tenacity python benchmarks/ruler_lm_eval.py \
    --model local-chat-completions \
    --model_args "model=Qwen/Qwen3.8-27B-FP8,base_url=http://127.0.0.1:8000/v1/chat/completions,tokenizer=Qwen/Qwen3.8-27B-FP8,tokenizer_backend=none,tokenized_requests=false,max_length=65536,num_concurrent=8,max_retries=3,timeout=600" \
    --tasks "$task" \
    --batch_size 1 \
    --limit 500 \
    --seed 42,42,42,42 \
    --metadata '{"max_seq_lengths":[65536]}' \
    --apply_chat_template \
    --gen_kwargs add_generation_prompt=false,continue_final_message=true \
    --output_path "results/ruler-64k-qwen-two-tier/$task" \
    --log_samples
done
```

The wrapper keeps RULER HotpotQA loading reproducible by obtaining the
validation data from the maintained Hugging Face parquet instead of the
obsolete CMU URL. It also supplies K2's required empty reasoning field on the
assistant-prefill message without changing the prefix content.

For the Qwen full-attention control, restart the server in a fresh shell with
`VLLM_PLUGINS`, `VLLM_LOD_MODE`, `VLLM_LOD_POOL_SIZE`, and
`VLLM_LOD_MAX_CONTEXT` unset. Omit the LoD scheduler option and use:

```bash
  --gpu-memory-utilization 0.9 \
  --attention-config '{"backend":"ROCM_AITER_UNIFIED_ATTN"}'
```

For K2, replace the checkpoint everywhere with
`IFM/K2-Horizon-32B-FP8`, omit `--language-model-only`, and use
`--gpu-memory-utilization 0.9` for both LoD and full attention. Keep the same
pool size, request concurrency, scheduler, token budget, task list, and seeds.

## Reproduction requirements

Use the locked repository dependencies, notably vLLM 0.27.1 and
`lm-eval==0.4.13`, the exact model revisions resolved by the checkpoint names,
the patched AITER build from the root README, and an AMD MI325X. Prefix caching
must remain disabled. Preserve eight concurrent requests, `--batch_size 1`,
the 65,536-token task metadata, the 500-example limit, all four seed values,
chat templating, and the assistant-prefill generation arguments.

Run each attention mode in a fresh server process. The task loop may be split
across GPUs for wall-clock throughput, provided every server uses the same
arguments and each task writes to a distinct output directory. Scores are
aggregated as an unweighted arithmetic mean over task-level `65536,none`
metrics. FP8 GEMMs and parallel GPU reductions are not guaranteed to be
bitwise deterministic, so preserve the per-sample logs when comparing reruns.

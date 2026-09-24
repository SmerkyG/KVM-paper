# RULER

This benchmark evaluates all 13 RULER subtasks at a 65,536-token context
length. Each task contains 500 examples. The evaluator uses `lm-eval==0.4.12`,
greedy generation through vLLM's OpenAI-compatible completions endpoint, eight
concurrent requests, and the canonical `42,42,42,42` lm-eval seeds.

## Preliminary results: known-buggy LoD run

**The LoD columns in this section are non-authoritative and must not be cited
as release results.** During this run, a CUDA-graph padding lane could borrow a
dormant LoD cache row and mutate it during ragged decode. That could silently
contaminate later requests or, as happened on Qwen `ruler_vt`, raise a GPU
memory exception. Full attention does not use the LoD row pool and its columns
are unaffected. Corrected LoD reruns are in progress.

Scores are percentages and higher is better. The metric is each task's
`65536,none` result reported by lm-eval. Qwen LoD completed 12 of 13 tasks;
`ruler_vt` is deliberately shown as `crash`, not imputed as zero.

### Qwen3.8-27B-FP8

| Task | Full attention | Two-tier LoD, buggy |
|---|---:|---:|
| NIAH single 1 | 100.00 | 100.00 |
| NIAH single 2 | 100.00 | 100.00 |
| NIAH single 3 | 100.00 | 99.40 |
| NIAH multikey 1 | 100.00 | 98.40 |
| NIAH multikey 2 | 99.80 | 94.80 |
| NIAH multikey 3 | 100.00 | 93.80 |
| NIAH multiquery | 99.95 | 99.75 |
| NIAH multivalue | 99.85 | 98.25 |
| Common-words extraction | 97.94 | 98.96 |
| Frequent-words extraction | 63.73 | 52.60 |
| HotpotQA | 18.00 | 6.20 |
| SQuAD QA | 36.50 | 24.53 |
| Variable tracking | 20.44 | **crash** |
| **Mean over the 12 completed LoD tasks** | **84.65** | **80.56** |
| Mean over all 13 tasks | 79.71 | n/a |

The common-task mean compares exactly the same 12 tasks in both columns. It is
not directly comparable to the 13-task full-attention mean.

### K2-Horizon-32B-FP8

| Task | Full attention | Two-tier LoD, buggy |
|---|---:|---:|
| NIAH single 1 | 100.00 | 100.00 |
| NIAH single 2 | 100.00 | 99.80 |
| NIAH single 3 | 100.00 | 99.80 |
| NIAH multikey 1 | 100.00 | 99.80 |
| NIAH multikey 2 | 99.00 | 87.60 |
| NIAH multikey 3 | 100.00 | 96.00 |
| NIAH multiquery | 100.00 | 99.60 |
| NIAH multivalue | 99.80 | 98.45 |
| Common-words extraction | 88.76 | 89.92 |
| Frequent-words extraction | 82.60 | 80.80 |
| HotpotQA | 55.60 | 51.80 |
| SQuAD QA | 54.60 | 55.17 |
| Variable tracking | 100.00 | 100.00 |
| **Mean over all 13 tasks** | **90.80** | **89.13** |

The K2 LoD run did not raise the exception, but it used the same faulty cache
integration and is therefore also non-authoritative. The completed full-
attention controls remain valid.

## Reproduce

These instructions reproduce the corrected evaluation. They intentionally do
not provide a way to restore the known cache-corruption bug. The fixed plugin
captures every ordinary decode batch size from 1 through 8 and refuses to run
a padded LoD decode row against authoritative cache state.

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
export VLLM_LOD_MAX_CONTEXT=65536
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
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16392 \
  --long-prefill-token-threshold 16384 \
  --scheduler-cls vllm_lod_plugin.scheduler.LODChunkAlignedScheduler \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.7 \
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
    --model local-completions \
    --model_args "model=Qwen/Qwen3.8-27B-FP8,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=Qwen/Qwen3.8-27B-FP8,tokenizer_backend=huggingface,max_length=65536,num_concurrent=8,max_retries=3,timeout=600" \
    --tasks "$task" \
    --batch_size 1 \
    --limit 500 \
    --seed 42,42,42,42 \
    --metadata '{"max_seq_lengths":[65536]}' \
    --output_path "results/ruler-64k-qwen-two-tier/$task" \
    --log_samples
done
```

The wrapper changes only RULER HotpotQA loading: lm-eval 0.4.12 references an
obsolete CMU URL, so `benchmarks/ruler_lm_eval.py` obtains the same validation
data from the maintained Hugging Face parquet instead.

For the Qwen full-attention control, restart the server after unsetting
`VLLM_LOD_MODE` and replace the last two server options with:

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
`lm-eval==0.4.12`, the exact model revisions resolved by the checkpoint names,
the patched AITER build from the root README, and an AMD MI325X. Prefix caching
must remain disabled. Preserve eight concurrent requests, `--batch_size 1`,
the 65,536-token task metadata, the 500-example limit, and all four seed values.

Run each attention mode in a fresh server process. The task loop may be split
across GPUs for wall-clock throughput, provided every server uses the same
arguments and each task writes to a distinct output directory. Scores are
aggregated as an unweighted arithmetic mean over task-level `65536,none`
metrics. FP8 GEMMs and parallel GPU reductions are not guaranteed to be
bitwise deterministic, so preserve the per-sample logs when comparing reruns.

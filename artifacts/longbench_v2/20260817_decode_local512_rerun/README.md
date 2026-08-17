# Qwen3.5-35B-A3B LongBench v2 after the decode-local scan fix

Run date: 2026-08-17

This reruns all 503 LongBench v2 examples in LOD mode after commit `d19f6f5`
reduced the fused decode local-attention scan from the 4,864-entry prefill
backing allocation to the configured 512-token local field. The prior full and
LOD results in `../20260816_latest_fixed_native262k_rerun/` are the comparison
baseline.

## Configuration

- Model: `Qwen/Qwen3.5-35B-A3B`, BF16 weights
- Backend: recursive top-8 LOD with INT4 leaf KV and pool size 8
- vLLM 0.27.1, eight independent one-GPU servers, eight concurrent requests
  per server
- `max_num_batched_tokens=16384`, `long_prefill_token_threshold=4096`
- Native 262,144-token model context; prompts capped at 262,016 tokens
- Thinking disabled; generation constrained to the four answer choices
- Each server processed an unmeasured eight-request 250K--262K-token warm-up
  and an unmeasured short warm-up batch before its evaluator timer started
- Model loading, server startup, tokenization, and both warm-ups are excluded
  from evaluator wall time

## Results

| Metric | Prior LOD | New decode path | Change |
|---|---:|---:|---:|
| Accuracy | 237/503 (47.12%) | 236/503 (46.92%) | -0.20 pp |
| Slowest-shard / eight-GPU wall | 529.93 s | 530.08 s | -0.03% |
| Aggregate shard time | 3,615.47 s | 3,611.84 s | +0.10% |
| Summed request latency | 21,792.59 s | 21,769.55 s | +0.11% |

The new and prior LOD runs agree on 442/503 predictions (87.87%) and 466/503
correctness outcomes (92.64%). Eighteen cases changed from incorrect to correct
and nineteen changed from correct to incorrect, leaving a one-example aggregate
difference. This is quality-stable relative to normal GPU-kernel run-to-run
variation.

Against the unchanged full-attention baseline, the new LOD run scores 236/503
versus 250/503 and remains 2.56x faster by slowest-shard wall time, 2.51x faster
by aggregate shard time, and 2.36x lower in summed request latency.

The decode-local scan optimization does not produce a measurable end-to-end
LongBench speedup because every request generates only 32 tokens and the run is
dominated by long prefill. Its benefit is instead visible in the matched
1,025-token decode sweep recorded under
`../../vllm_lod_speed/decode16k_paired_20260817/`, where LOD decode latency was
8.44% below optimized full attention at 16K.

`comparison.json` contains the paired breakdown against the prior full-attention
run. `timing.json` contains the per-shard and aggregate timing comparison. Raw
per-example outputs and evaluator/server logs are under `lod/`.

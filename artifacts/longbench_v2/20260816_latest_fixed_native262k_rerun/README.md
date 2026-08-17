# Qwen3.5-35B-A3B LongBench v2: latest fixed LOD vs full attention

Run date: 2026-08-16--17

This is a fresh paired evaluation of all 503 LongBench v2 examples. It used the
current fixed LOD working tree based on `6d27cc8`, vLLM 0.27.1, and the model's
native 262,144-token context. Raw prompts were capped at 262,016 tokens to leave
room for chat-template tokens. This left 403 examples untruncated and truncated
only the 100 examples longer than the model context.

## Configuration

- Model: `Qwen/Qwen3.5-35B-A3B`, BF16 weights
- Full backend: `ROCM_AITER_UNIFIED_ATTN`, BF16 KV
- LOD backend: recursive top-8 LOD, INT4 leaf KV, pool size 8
- Eight independent one-GPU servers per mode, balanced four full/four LOD per node
- Eight concurrent requests per server
- `max_num_batched_tokens=16384`, `long_prefill_token_threshold=4096`
- Thinking disabled; generation constrained to the four required answer strings
- Identical dataset shards and length ordering in both modes
- Each server processed an unmeasured long batch (eight 250K--262K prompts) and
  an unmeasured short batch before its evaluator timer started
- Model loading, server startup, tokenization, and both warm-ups are excluded
  from the reported evaluator wall time

## Main results

| Metric | Full attention | Fixed LOD INT4 | LOD vs full |
|---|---:|---:|---:|
| Accuracy | 250/503 (49.70%) | 237/503 (47.12%) | -2.58 pp |
| Slowest-shard / eight-GPU wall | 1,355.02 s | 529.93 s | 2.56x faster |
| Aggregate shard time | 9,080.45 s | 3,615.47 s | 2.51x faster |
| Summed request latency | 51,369.47 s | 21,792.59 s | 2.36x lower |

LOD and full attention produced the same answer on 422/503 examples (83.90%).
The paired correctness table has 217 both correct, 233 both incorrect, 33
full-only correct, and 20 LOD-only correct. The exact two-sided McNemar/binomial
test gives `p=0.0984`. The paired normal 95% interval for LOD minus full accuracy
is approximately -5.42 to +0.25 percentage points. Thus this run does not show
a statistically clear accuracy loss at the 5% level, but it also does not prove
strict equivalence under a predeclared margin.

## Quality breakdown

| Subset | Count | Full | LOD | LOD minus full |
|---|---:|---:|---:|---:|
| Untruncated | 403 | 199 (49.38%) | 191 (47.39%) | -1.99 pp |
| Truncated at 262,016 | 100 | 51 (51.00%) | 46 (46.00%) | -5.00 pp |
| LongBench short | 180 | 94 (52.22%) | 92 (51.11%) | -1.11 pp |
| LongBench medium | 215 | 104 (48.37%) | 98 (45.58%) | -2.79 pp |
| LongBench long | 108 | 52 (48.15%) | 47 (43.52%) | -4.63 pp |
| Easy | 192 | 104 (54.17%) | 93 (48.44%) | -5.73 pp |
| Hard | 311 | 146 (46.95%) | 144 (46.30%) | -0.64 pp |

The largest domain loss is Long In-context Learning (51/81 full versus 43/81
LOD). LOD is slightly better on Code Repository Understanding (24/50 versus
22/50), Long-dialogue History Understanding (20/39 versus 18/39), and
Multi-Document QA (61/125 versus 60/125).

The previous broken 262K LOD run scored 141/503. The fixed implementation
recovers 96 correct answers while retaining essentially the same fast runtime
(the earlier LOD slowest-shard wall was 545.48 s).

Raw per-example outputs and evaluator/server logs are under `full/` and `lod/`.
`comparison.json` contains the complete paired difficulty, length, and domain
breakdown.

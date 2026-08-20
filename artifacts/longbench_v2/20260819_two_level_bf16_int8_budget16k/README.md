# Qwen3.5-35B-A3B two-level LOD at a 16K scheduler budget

Run date: 2026-08-19

This evaluates all 503 LongBench v2 examples with the current uncapped flat
two-level vLLM implementation. Both modes use BF16 model weights. The serving
configuration uses batch size 8, `max_num_batched_tokens=16384`, and
`long_prefill_token_threshold=4096`. The historical full-attention baseline
uses the same scheduler settings, dataset shards, input cap, guided decoding,
and warmup protocol.

## Quality

| Metric | Full attention | Two-level BF16 | Two-level INT8 |
|---|---:|---:|---:|
| All examples | 250/503 (49.70%) | 243/503 (48.31%) | 233/503 (46.32%) |
| Untruncated | 199/403 (49.38%) | 200/403 (49.63%) | 188/403 (46.65%) |
| Truncated | 51/100 (51.00%) | 43/100 (43.00%) | 45/100 (45.00%) |
| LongBench short | 94/180 (52.22%) | 96/180 (53.33%) | 90/180 (50.00%) |
| LongBench medium | 104/215 (48.37%) | 104/215 (48.37%) | 99/215 (46.05%) |
| LongBench long | 52/108 (48.15%) | 43/108 (39.81%) | 44/108 (40.74%) |

BF16 is 1.39 percentage points below full attention. INT8 is 3.38 points below
full attention and 1.99 points below BF16. In the paired comparisons, exact
McNemar/binomial p-values are 0.427 for full versus BF16, 0.030 for full versus
INT8, and 0.132 for BF16 versus INT8.

Changing only the scheduler budget from 128K to 16K changed 54/503 predictions
in each precision mode. BF16 moved from 239 to 243 correct and INT8 moved from
237 to 233; neither paired shift is significant (`p=0.644` and `p=0.608`).
This establishes noticeable numerical/batching variability, so the observed
ten-answer BF16/INT8 gap should not be treated as a precise quantization effect
without repetitions.

## Speed

| Metric | Full attention | Two-level BF16 | Two-level INT8 |
|---|---:|---:|---:|
| Slowest-shard / eight-GPU wall | 1,355.02 s | 528.20 s (2.565x) | 463.40 s (2.924x) |
| Aggregate shard wall | 9,080.45 s | 3,598.79 s (2.523x) | 3,165.55 s (2.869x) |
| Summed request latency | 51,369.47 s | 21,435.10 s (2.397x) | 19,032.77 s (2.699x) |

INT8 is 12.27% lower in slowest-shard wall time, 12.04% lower in aggregate
shard time, and 11.21% lower in summed request latency than BF16. By summed
request time, INT8 is 3.70% slower on the short subset, 9.59% faster on medium,
and 15.35% faster on long examples.

The realistic 16K scheduler cap makes BF16 13.61% slower and INT8 17.16%
slower by slowest-shard wall time than their 128K-budget runs. Despite that,
both remain substantially faster than the historical optimized AITER
full-attention baseline under the same 16K serving budget.

## Configuration and validation

- Model: `Qwen/Qwen3.5-35B-A3B`, BF16 weights
- Full backend: `ROCM_AITER_UNIFIED_ATTN`, BF16 KV
- LOD: flat top-eight two-level routing, raw geometry, `16 sqrt(T)` schedule
- Uncapped physical leaf pages; no leaf sealing or discarding
- BF16 mode: BF16 K/V pages
- INT8 mode: physical signed-INT8 K/V pages with per-token scales, INT8 leaf
  MMA, and automatic long-context INT8 coarse PV
- Direct LOD prefill with 4,096-token chunks and state updates
- Native 262,144-token context; inputs capped at 262,016 tokens
- Eight one-GPU shards per mode, eight resident requests per server
- Guided A-D decoding, thinking disabled, maximum 32 output tokens
- Loading and the long and short warmups are excluded from evaluator timing

All 1,006 scored records completed and all 503 IDs are unique in each mode.
Kernel logs confirm the intended physical BF16 and INT8 page-write paths. Two
local `tee` logs did not retain their final evaluator JSON after cluster-run
cleaned up server descendants; their complete summaries and exit code zero are
retained in the cluster-run logs, which are the source of all wall times here.
Per-request times and quality come directly from the complete JSONL outputs.

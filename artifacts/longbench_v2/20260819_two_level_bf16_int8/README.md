# Qwen3.5-35B-A3B two-level LOD: BF16 versus INT8

Run date: 2026-08-19

This is a matched evaluation of all 503 LongBench v2 examples using the current
uncapped flat two-level vLLM implementation. Both modes use BF16 model weights;
only LOD cache precision and its associated kernels differ.

## Configuration

- Model: `Qwen/Qwen3.5-35B-A3B`, BF16 weights
- vLLM 0.27.1, custom LOD attention backend
- Flat top-eight two-level LOD, raw routing, `16 sqrt(T)` state schedule
- Uncapped physical leaf pages; no leaves are discarded or sealed
- BF16 mode: BF16 K/V pages
- INT8 mode: physical signed-INT8 K/V pages with per-token scales, INT8 leaf
  MMA, and the automatic long-context INT8 coarse-PV path
- Direct prefill with 4,096-token LOD chunks, 4,864-token local backing, and
  4,096-token state updates
- Eight independent one-GPU servers per precision and eight concurrent
  requests per server
- 131,072 aggregate scheduler-token budget, allowing eight 16K request chunks
  to run together instead of serializing the nominal batch
- Native 262,144-token context; inputs capped at 262,016 tokens
- Guided A-D decoding, thinking disabled, maximum 32 output tokens
- An unmeasured 8x262K batch and an unmeasured short batch warmed each server
  before evaluator timing

## Quality

| Metric | Two-level BF16 | Two-level INT8 | INT8 minus BF16 |
|---|---:|---:|---:|
| All examples | 239/503 (47.51%) | 237/503 (47.12%) | -0.40 pp |
| Untruncated | 195/403 (48.39%) | 193/403 (47.89%) | -0.50 pp |
| Truncated | 44/100 (44.00%) | 44/100 (44.00%) | 0.00 pp |
| LongBench short | 96/180 (53.33%) | 92/180 (51.11%) | -2.22 pp |
| LongBench medium | 98/215 (45.58%) | 99/215 (46.05%) | +0.47 pp |
| LongBench long | 45/108 (41.67%) | 46/108 (42.59%) | +0.93 pp |

The modes agree on 449/503 predictions (89.26%). The paired table contains 220
both-correct, 247 both-incorrect, 19 BF16-only correct, and 17 INT8-only
correct examples. The exact two-sided McNemar/binomial p-value is 0.868, so
this run provides no evidence of a systematic accuracy difference.

## Speed

| Metric | Two-level BF16 | Two-level INT8 | INT8 result |
|---|---:|---:|---:|
| Slowest-shard / eight-GPU wall | 464.91 s | 395.51 s | 1.175x; 14.93% lower |
| Aggregate shard wall | 3,179.90 s | 2,696.22 s | 1.179x; 15.21% lower |
| Summed request latency | 24,507.32 s | 20,797.88 s | 1.178x; 15.14% lower |
| Effective prompt throughput | 133,811 tok/s | 157,290 tok/s | 1.175x |

INT8's summed-request speedup increases with context length: 1.054x on the
LongBench short subset, 1.170x on medium, and 1.211x on long. All eight
independent shard-wall ratios fall between 1.166x and 1.195x, so the aggregate
result is not driven by one outlier GPU.

`comparison.json` contains the paired quality breakdown (`full` denotes BF16
and `lod` denotes INT8 in that generic summarizer). `timing.json` contains all
per-shard and aggregate timings. Raw evaluator outputs and server logs are in
the `bf16/` and `int8/` directories.

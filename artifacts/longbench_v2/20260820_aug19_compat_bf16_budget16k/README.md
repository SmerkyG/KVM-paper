# August 19 compatibility LongBench v2 rerun

Run date: 2026-08-20

This is a full 503-example BF16 rerun of the closest reconstructable execution
path to the August 19 two-tier result. The historical run used an uncommitted
working tree, so its exact source cannot be restored. The compatibility preset
keeps the same two-tier state, raw top-eight routing, paging, and prefill
schedule, but selects the known older execution choices:

- fixed eight-way Triton split decode;
- cooperative GQA and HIP decode disabled;
- four-warp leaf-route reduction instead of one warp.

The later kernel audit established that the current profile also used fixed
eight-way Triton decode on this model: its GQA group is eight, while the HIP
specialization requires group four. Thus only the leaf-route reduction warp
count differed in the kernels actually exercised by this comparison.

## Result

| Mode | Correct | Accuracy |
|---|---:|---:|
| Historical full attention | 250/503 | 49.70% |
| Historical August 19 two-tier BF16 | 243/503 | 48.31% |
| August 19 compatibility rerun | 234/503 | 46.52% |
| Current two-tier BF16 run 1 | 230/503 | 45.73% |
| Current two-tier BF16 run 2 | 240/503 | 47.71% |

The compatibility rerun is nine answers below the historical August 19 run,
but almost exactly matches the 235/503 mean of the two current repeats. The
paired historical-versus-compatibility comparison has 12 compatibility-only
wins and 21 historical-only wins (exact two-sided p=0.163), so the difference
does not establish a quality regression beyond the already observed batched
numerical variability.

| Subset | Historical August 19 | Compatibility rerun |
|---|---:|---:|
| Untruncated | 200/403 | 190/403 |
| Truncated | 43/100 | 44/100 |
| Short | 96/180 | 95/180 |
| Medium | 104/215 | 95/215 |
| Long | 43/108 | 44/108 |

The missing historical score was not recovered. The nine-answer difference is
entirely in the medium class; compatibility is one answer better on the long
class. The newer cooperative decode machinery was not exercised and therefore
cannot explain the August 19 quality result.

Prediction agreement is 455/503 (90.46%) with the historical August 19 run,
444/503 (88.27%) with current run 1, and 445/503 (88.47%) with current run 2.
The compatibility path is numerically closer to August 19, but not enough to
reproduce its score.

## Speed

Times exclude model loading, compilation, the eight-request long warmup, and
the evaluator's short warmup.

| Metric | Historical August 19 | Compatibility rerun | Current BF16 mean |
|---|---:|---:|---:|
| Slowest-shard wall | 528.20 s | 528.80 s | 527.95 s |
| Aggregate shard wall | 3,598.79 s | 3,581.91 s | 3,583.18 s |
| Summed request latency | 21,435.10 s | 21,294.44 s | 21,314.59 s |

All three implementations are effectively tied end to end on this workload.
The compatibility path is 0.11% slower than August 19 by slowest-shard wall,
0.47% faster by aggregate wall, and 0.66% faster by summed request latency.

## Configuration and validation

- Model: `Qwen/Qwen3.5-35B-A3B`, BF16 weights and BF16 LOD leaves
- vLLM 0.27.1 on gfx942, eight one-GPU shards, eight resident requests
- `max_num_batched_tokens=16384`, long-prefill threshold 4,096
- Native 262,144-token context; inputs capped at 262,016 tokens
- Direct 4,096-token-chunk two-tier prefill, raw top-eight routing,
  `16 sqrt(T)` state schedule, and uncapped physical leaf pages
- Thinking disabled, guided greedy A-D decoding, at most 32 output tokens

All 503 IDs completed exactly once with exit code zero. Kernel logs confirm
`_split_decode_paged_lod_attention_kernel` and
`_reduce_routed_split_decode_lod_attention_kernel` in both profiles. Raw
records and logs are in `bf16/`; the machine-readable headline values are in
`summary.json`.

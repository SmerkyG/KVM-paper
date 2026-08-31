# Fixed-mask LOD with DFlash2

This panel retests `z-lab/Qwen3.8-27B-DFlash2` against
`Qwen/Qwen3.8-27B-FP8` after the fixed-mask target verifier was generalized to
an arbitrary number of proposal positions. Both arms use TP1/B1 on separate
MI325X GPUs of the same node, seven DFlash2 draft tokens, greedy sampling, 256
emitted tokens, one untimed warmup, three measured repetitions, real
non-repeated ProLong prompts, and 16K aggregate prefill chunks. Decode is the
median marginal complete-model latency per emitted output token.

## Decode

| context | full attention + DFlash2 | fixed-mask LOD + DFlash2 | full / LOD | old piecewise LOD / new |
|---:|---:|---:|---:|---:|
| 8K | **15.722 ms** | 17.571 ms | 0.895x | 1.362x |
| 16K | 9.753 ms | **8.774 ms** | **1.112x** | 2.124x |
| 32K | 15.088 ms | **14.348 ms** | **1.052x** | 1.428x |
| 64K | 19.316 ms | **15.091 ms** | **1.280x** | 1.644x |
| 128K | 22.054 ms | **14.621 ms** | **1.508x** | 1.454x |

DFlash2 acceptance varies with the generated trajectory, so milliseconds per
output are not a pure target-kernel measurement. The fixed-mask device epoch
also records the aggregate number of target-verifier cycles. Its resulting
complete-model time per verifier cycle stays nearly flat:

| context | fixed-mask LOD ms/verifier cycle |
|---:|---:|
| 8K | 40.123 |
| 16K | 39.715 |
| 32K | 39.879 |
| 64K | 40.731 |
| 128K | 41.495 |

At 128K the full-attention repetitions each drafted 623 tokens and accepted
166, corresponding to 89 verifier cycles for 255 marginal output tokens. Its
median 5.624-second marginal decode is therefore about 63.2 ms per verifier
cycle, versus 41.5 ms for fixed-mask LOD, an approximately 1.52x
acceptance-independent advantage. The shorter periodic vLLM metrics combine
parts of adjacent repetitions and are not used for cycle normalization.

## Prefill

| context | full attention | LOD | full / LOD |
|---:|---:|---:|---:|
| 8K | 0.989 s | **0.984 s** | 1.005x |
| 16K | 2.224 s | **2.053 s** | 1.083x |
| 32K | 5.293 s | **4.171 s** | 1.269x |
| 64K | 14.081 s | **8.630 s** | 1.632x |
| 128K | 42.711 s | **18.083 s** | 2.362x |

## Executed path

The post-warmup device audit confirms all of the intended target-side work:

- uniform eight-position speculative verification is flattened into one LOD
  call;
- every position has independent current top-eight routes and causal local
  length;
- direct fixed-route activation, the fixed byte mask, the page-size-one HIP
  final scan, and its stable-LSE reduction all executed;
- the DFlash2 drafter retained its native chronological cache and captured
  graph.

Thus the old statement that DFlash2 requires a piecewise serial LOD target is
obsolete. The earlier corrected LOD panel used the ordinary per-position
top-eight verifier and measured 23.929/18.634/20.483/24.810/21.262 ms from 8K
through 128K. The new fixed-mask verifier is 1.36--2.12x faster than that path.

## Raw records

- `full_current_b1_8k128k_r3_d256.json`
- `lod_fixed_current_b1_8k128k_r3_d256.json`
- `lod_fixed_smoke64_b1_r1_d64.json`
- Full control: cluster run 12224.
- Fixed-mask LOD: cluster run 12223.
- Audited 64K smoke: cluster run 12222.

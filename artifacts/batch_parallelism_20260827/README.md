# Batch-parallelism and low-row decode study (2026-08-27)

This directory tests whether the TP slowdown is fundamentally the same
under-occupancy seen at small batch size.  Runs use Qwen3.5-0.8B at TP1,
65,280-token distinct real ProLong prompts, chat formatting, a 256-token decode,
16K aggregate chunked prefill, synchronized warm-prefix replay, and medians of
five runs unless a filename says `r7`.  The full backend is
`ROCM_AITER_UNIFIED_ATTN`; LOD is the production two-tier top-8 fixed-list
page-size-one HIP path or the production recursive three-tier path.

## Batch-size confirmation

| batch | full decode ms/step | two-tier ms/step | two-tier speedup | three-tier ms/step | three-tier speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.325 | 2.151 | 1.08x | 1.868 | 1.24x |
| 2 | 3.084 | 2.307 | 1.34x | 1.919 | 1.61x |
| 4 | 4.324 | 2.501 | 1.73x | 2.068 | 2.09x |
| 8 | 6.735 | 3.031 | 2.22x | 2.624 | 2.57x |

Two-tier throughput rises from 465 to 2,639 tokens/s between B1 and B8;
three-tier rises from 535 to 3,049 tokens/s.  This confirms that the weak TP
and low-batch result is an occupancy/fixed-cost problem, not more sparse work.
The execution audit confirms the intended fixed-mask page-size-one path on all
six global layers and `12 * batch` audited GQA sequences.

## Occupancy correction

The fixed final-attention scan was already split across many programs, but its
reducer launched only one program per query head and reduced a full
`[segments, D]` FP32 field in that program.  More scan segments therefore made
the scan faster while making this serial, register-heavy reducer much slower.

`_reduce_aiter_page1_segments_with_lse_split_d_kernel` divides the output
dimension into independent 64-channel programs.  Each program cheaply
recomputes the scalar online-softmax correction, so no additional launch or
intermediate scalar field is required.  Its output differs from the original
FP32 reducer by at most `1.1e-8`, with identical LSE.

At D=256/GQA4/B1, six-layer fixed attention with a representative 25% active
block pattern changes as follows:

| scan/reducer | six-layer time |
| --- | ---: |
| 128 segments, original reducer | 0.201 ms |
| 128 segments, split-D64 | 0.192 ms |
| 256 segments, original reducer | 0.230 ms |
| **256 segments, split-D64** | **0.184 ms** |

In production vLLM on the same node/GPU, the B1 control is 2.234 ms and the
256-segment split-D64 version is 2.095 ms, a 6.2% reduction.  The matched full
attention result on that GPU is 2.325 ms, so optimized two-tier is 1.11x faster.
The 128-segment split-D64 control is 2.195 ms, confirming that both halves of
the change matter.  The optimized B1 path scored 8/8 on chat-formatted 64K
NIAH-S3.

The automatic rule is based primarily on local query rows, with a B1
correction for higher-GQA models whose query rows share only a few independent
KV scans:

```text
query_rows_on_rank = batch * kv_heads_on_rank * queries_per_kv
kv_scans_on_rank = batch * kv_heads_on_rank
256 scan segments + split-D64 when query_rows_on_rank <= 8
                              or (batch == 1 and kv_scans_on_rank <= 4)
128 scan segments + original reducer otherwise
```

This also handles tensor-parallel head splitting.  D=128 microbenchmarks support
the geometry rule: GQA4 (eight local query rows) improves by 5.1% with the
256/split-D64 combination, while GQA16 already has 32 query rows and should
remain at 128 segments.

The B1 correction was validated end to end on Qwen3.8-27B-FP8 at 64K. That
model has D=256/GQA6 and only four `(request, KV head)` scans at B1, but the old
24-query-row test excluded it. In a matched seven-repeat real-ProLong run, the
128-segment/original-reducer/direct-union control measured 29.240 ms/step and
the 256-segment/split-D64/direct-route path measured **28.717 ms/step**, a
0.523-ms or **1.79%** reduction. The execution audit records 256 effective
segments, split-D reduction, and direct fixed-route activation. Against the
matched 31.571-ms full-AITER control, optimized two-tier is 1.099x faster
instead of 1.080x. The factorized three-repeat sweep shows that 256 segments
and split-D must be enabled together; direct activation contributes a smaller
additional improvement. Raw results are under `low_sequence_qwen38/`.

### Qwen3.8-27B-FP8 B1 context scaling

A matched five-repeat panel measures complete-model marginal decode latency
from 8K through 256K. Each repetition decodes 256 tokens ending at the stated
context length. Prompts are distinct shuffled ProLong documents concatenated
without repetition and chat formatted; prefill is chunked at 16K. Both arms
run in the same 256K-capable TP1 service configuration on MI325X. Full attention
uses `ROCM_AITER_UNIFIED_ATTN`; LOD uses the audited 256-segment/split-D64/direct
fixed-route path.

| context | full ms/step | LOD ms/step | full / LOD | latency saved |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **28.746** | 28.797 | 0.998x | -0.18% |
| 16K | 29.364 | **28.799** | 1.020x | 1.93% |
| 32K | 30.089 | **28.876** | 1.042x | 4.03% |
| 64K | 31.464 | **29.141** | 1.080x | 7.38% |
| 128K | 34.173 | **29.497** | 1.159x | 13.68% |
| 256K | 39.571 | **30.083** | 1.315x | 23.98% |

The long-capacity service allocates 8,448 coarse slots at every length, which
makes its short-context LOD numbers conservative. The separately configured
64K-only seven-repeat engine above measures 28.717 ms versus 31.571 ms full,
or 1.099x. The scaling panel is the appropriate result for one service that
must accept the entire 8K--256K range. Raw results are under
`qwen38_b1_context_panel/`.

### Qwen3.8-27B-FP8 TP4/B1 context scaling

The same panel was repeated at TP4 with global batch one. No other benchmark
condition changed: each repetition decodes 256 tokens ending at the stated
context, the prompt is made from distinct shuffled ProLong documents without
repetition and is chat formatted, prefill is chunked at 16K, both arms use one
256K-capable service configuration, and each reported latency is the median of
five repetitions. Full attention uses `ROCM_AITER_UNIFIED_ATTN`; LOD uses the
same audited 256-segment/split-D64/direct fixed-route path.

| context | full ms/step | LOD ms/step | full / LOD | latency saved |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **17.100** | 17.381 | 0.984x | -1.64% |
| 16K | 17.452 | **17.375** | 1.004x | 0.44% |
| 32K | 17.651 | **17.408** | 1.014x | 1.38% |
| 64K | 18.326 | **17.638** | 1.039x | 3.76% |
| 128K | 19.426 | **17.994** | 1.080x | 7.37% |
| 256K | 21.821 | **18.695** | 1.167x | 14.33% |

TP4 lowers the complete-model latency floor substantially, so the same sparse
attention saving is a smaller fraction of end-to-end latency than at TP1. LOD
is 1.6% slower at 8K, crosses over by 16K, and reaches a 1.17x speedup at 256K.
Its latency grows by 1.31 ms from 8K to 256K, versus 4.72 ms for full attention.
The saved rank-zero runtime audit at every length records the intended
one-KV-head/six-query-head local TP shape, 256 effective scan segments, split-D
reduction, direct fixed routes, the fixed mask, and HIP union execution. Qwen's
24 query and four KV heads partition evenly here, giving every rank the same
local shape; the service log confirms that all four ranks initialized the
custom attention and AITER modules successfully. Raw results are under
`qwen38_tp4_b1_context_panel/`.

### Gemma-4 and Muse TP1/B1 at 64K

Fresh matched runs test whether the low-row policy generalizes beyond Qwen.
Each arm uses one real, non-repeating, chat-formatted 65,280-token ProLong
prompt followed by 256 timed decode tokens, 16K chunked prefill, warm-prefix
restoration, and five repetitions on the same model-specific MI325X GPU.
Gemma full attention uses `TRITON_ATTN`, because AITER does not support its
D=512 global heads; Muse full attention uses `ROCM_AITER_UNIFIED_ATTN`.

The LOD control uses the former 128-segment scan, complete-D reducer, and
ordinary union construction. The optimized arm uses direct route activation,
256 adaptive scan segments, and split-D64 where supported.

| model | full ms/step | old LOD ms/step | optimized LOD ms/step | new vs old LOD | full / new LOD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma-4-26B-A4B | 9.488 | 6.189 | **5.908** | **4.54% faster** | **1.606x** |
| Muse-Glimmer-30B | **15.876** | 16.542 | 16.277 | **1.60% faster** | 0.975x |

The change is therefore a clear additional win for Gemma and preserves its
large advantage over full attention. Gemma's D=512 path deliberately does not
use split-D; its audit attributes the bundle to direct route activation and
the 128-to-256 segment change. Muse executes all three low-row mechanisms and
does improve, but remains 2.52% slower than its very efficient full-attention
baseline at B1/64K. Four of Muse's five optimized raw decode repetitions are
tightly grouped; one slow outlier does not determine the median result.

All three arms for each model have identical prompt hashes. The LOD audits
confirm fixed-mask and HIP execution; control/optimized effective segment
counts are 128/256, direct-route execution is false/true, and split-D is
false/false for Gemma and false/true for Muse. Raw results are under
`family_b1_64k/`.

### Direct activation at B8

Direct route activation was then tested at B8/64K with the ordinary
128-segment scan. Runs use eight distinct, non-repeating, chat-formatted
65,280-token ProLong prompts, 256 decode tokens, 16K chunked prefill, warm-prefix
restoration, and seven repetitions for the same-GPU route A/Bs. The ordinary
arm first compacts the 8-per-head routes into one union per `(request, KV head)`;
the direct arm applies the same routes to the persistent mask immediately.
Duplicate mask writes are idempotent, so selected centroids and final attention
are unchanged.

| model | ordinary union, 128 segments | direct routes, 128 segments | direct change |
| --- | ---: | ---: | ---: |
| Gemma-4-26B-A4B | 10.837 ms | **10.810 ms** | **0.25% faster** |
| Qwen3.5-0.8B | 3.035 ms | **3.010 ms** | **0.83% faster** |
| Qwen3.8-27B-FP8 | 36.502 ms | **36.497 ms** | **0.01% faster / neutral** |

The Gemma route A/B was repeated on the same physical GPU after the initial
cross-GPU panel differed by only 0.014 ms. Its fresh full-TRITON_ATTN control is
12.727 ms, making direct/128 1.177x faster than full. A forced 256-segment B8
scan measured 10.990 ms, 1.66% slower than direct/128. The execution audits
record 128 effective segments, no split-D reduction, fixed-mask/HIP execution,
and direct activation off/on as intended. All arms have identical eight-prompt
hash lists.

Direct activation is therefore the default for fixed-mask top-eight decode at
all batch sizes; the compact-union path remains only as an explicit control.
Scan geometry stays adaptive and orthogonal: B8 uses 128 segments, while the
validated low-row rule retains 256 segments plus split-D where supported. On
Qwen3.8 B1, direct/128 was 29.204 ms versus 28.765 ms for
direct/256/split-D64, so forcing 128 everywhere would give up about 1.5% for no
algorithmic simplification of route construction. Raw B8 results are under
`family_b8_64k/`.

Enable the tested policy with:

```bash
VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS=256
VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS=1
VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D=64
VLLM_LOD_DECODE_GQA_FIXED_MASK_DIRECT_ROUTES=1
```

## Top-8 result

The existing exact top-8 is not the dominant occupancy opportunity.  A direct
CUDA-graph microbenchmark of its 1,024-candidate reducer is about 9.7 us.  An
exact two-stage partition/global reduction is at least 12.2 us (26% slower),
and a one-kernel tiled scan is at least 13.1 us (35% slower).  Eager production
profiling likewise increased the route-reduce stage from 24.8 to 36.9 us per
LOD layer.  Both alternatives were removed; production retains the simpler
single-program exact top-8.

The key artifacts are `qwen08_{full,two,three}_64k_b*_r5.json`, the
`paired/` same-GPU controls, `micro_fixed_b*_s*_d*.json`, and
`qwen08_two_b1_splitd_niah8.json`.

## Removing top-eight with predicted mass

The corrected predicted-remote-mass path removes the global top-eight reducer.
It still scores the current query against every coarse centroid, but compares
each score with a threshold based on the preceding token's eligible
remote-coarse mass and prepares the fixed-list mask directly. Thus this test
isolates whether the global selection barrier has a scheduling cost much
larger than its directly measured kernel time.

All speed runs below use the same real ProLong 64K corpus and seven-repeat
protocol as the top-eight control in the same job. Predicted mass also uses the
low-row split-D reducer: only D partition zero stores the next remote LSE and
advances the route epoch.

| batch | top-8 ms/step | predicted 1/16 ms/step | change vs top-8 | predicted 1/8 ms/step | change vs top-8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.011 | **1.957** | **2.8% faster** | 1.996 | 0.8% faster |
| 2 | 2.312 | **2.255** | **2.5% faster** | 2.144 | 7.8% faster |
| 8 | **3.016** | 3.033 | 0.6% slower | 2.992 | 0.8% faster |

Fresh matched full-AITER controls are 2.406 ms at B1 and 6.642 ms at B8;
the matching high-quality 1/16 results are therefore 1.23x and 2.19x faster
than full attention. The existing B2 full control is 3.084 ms, making 1/16
1.37x faster there.

At 64K NIAH-S3, 1/16 scores **64/64**, while the more aggressive 1/8
threshold scores **63/64**. The rejected 1/32 policy is also unstable in work:
one B1 sequence selected 214 union centroids, and at B8 it took 3.268 ms versus
3.016 ms for bounded top-eight.

The high-quality B1 path saves 54 us over the six-layer top-eight control.
Six independently timed top-eight reducers cost about 58 us in aggregate, so
removing the barrier recovers approximately its direct cost rather than
unlocking a much larger scheduling gain. At B8, the variable union tail fully
erases that saving. Predicted 1/16 is therefore a valid low-row specialization,
not the new universal default; the next occupancy work should target coarse
scoring and fixed final attention rather than a more elaborate top-k reducer.

The matched artifacts are under `mass_splitd_b1/`, `mass_splitd_b8/`, and the
initial B2 sweep under `mass_ab_b2/`. Quality is recorded in
`qwen08_predmass{16,8}_splitd_niah64.json`; the fresh dense controls are
`qwen08_full_current_64k_b{1,8}_r7.json`.

## Non-top-k low-row work

An eager, event-instrumented B1 profile confirms that the serial top-eight
reducer is not the main remaining cost. Per global LOD layer, coarse scoring
costs about 72 us, fixed-list final attention 149 us, fixed-list preparation
35 us, and top-eight reduction 24 us. The final attention label includes its
scan but not its separately timed 7-us segment reducer. These event timings
carry instrumentation overhead, but they identify the relative targets.

The retained low-row final-attention specialization changes the scan launch
from two warps/two waves per CU to one warp/one wave and uses 256 independent
segments with split-D64 reduction. On six representative D=256/GQA4 layers:

| final scan launch | attention | reduction | total |
| --- | ---: | ---: | ---: |
| 2 warps, 2 waves, 256 segments/split-D64 | 162.4 us | 25.4 us | 187.8 us |
| **1 warp, 1 wave, 256 segments/split-D64** | **124.0 us** | **25.8 us** | **149.9 us** |

This is a 20.2% scan/reduction reduction. Larger wave counts and four-warps
were slower. The launch rule is activated only when the local rank has at most
eight real query rows; the established batch-eight launch remains unchanged.

The fixed-mask activation path can also bypass the separate GQA-union launch
at low row counts. It activates each head-route pair directly; duplicate mask
writes are idempotent, so no compact-list construction or barrier is needed.
An attempted in-program duplicate rejection was removed: its prior-route loads
made fixed preparation 39.5 us/layer versus 38.1 us without rejection, and did
not improve the complete leaf/local stage in the instrumented profile.
The retained N=32/direct-activation code scored 8/8 in a fresh 64K smoke run;
the matching direct-activation algorithm scored 64/64 in the full 64K panel.

### Centroid scorer axis and tile sweep

The proposed query-major vector scorer was tested directly against the
GQA-major M=16 MFMA scorer, including the same exact global top-eight
reduction. Query-major exposes four times as many programs for GQA4, but also
reloads each K slice four times and replaces the matrix dot by scalar vector
work. Best times from each family are:

| batch | centroids | best MFMA score+reduce | best query-major vector | winner |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1,024 | **34.12 us** | 36.06 us | MFMA |
| 1 | 2,048 | **35.26 us** | 36.02 us | MFMA |
| 1 | 4,096 | **33.42 us** | 35.44 us | MFMA |
| 2 | 1,024 | 33.28 us | **32.84 us** | vector by 0.44 us |
| 2 | 2,048 | **34.46 us** | 35.02 us | MFMA |
| 2 | 4,096 | **32.96 us** | 42.12 us | MFMA |
| 8 | 1,024 | **33.76 us** | 40.46 us | MFMA |
| 8 | 2,048 | **35.82 us** | 53.78 us | MFMA |
| 8 | 4,096 | **47.30 us** | 78.04 us | MFMA |

The vector arm changed one of 64 selected random-control slots because its
FP32 scalar reduction rounds differently from BF16 MFMA. A single 1.3% win is
therefore neither robust nor quality-neutral. It is not a production dispatch.
N=16 MFMA similarly does not justify a low-row rule: at 4,096 centroids it
reduces score time only marginally while doubling the candidate field and its
reduction work. Production retains the existing N=32 route geometry.

Artifacts are `qwen08_two_eager_profile_64k_b1_splitphases_r1.json`,
`fixed_scan_tuning/`, `route_occupancy_sweep/`, `low_row_route_scalar.json`,
and `route_axes/`.
The retained-code quality smoke is
`low_row_final/qwen08_b1_n32_direct_current_niah8.json`; the full direct-route
quality run is `direct_route_activation/qwen08_b1_direct_rg16_niah64.json`.

### Centroid-major LDS HIP scorer

A centroid-major HIP mapping fixes the key-reload problem of the rejected
query-major vector experiment. One 256-thread workgroup owns 32 centroids for
one `(batch, KV head)` row. Four complete GQA queries are staged in LDS. Each
eight-lane subgroup then owns one centroid, loads its D=256 BF16 K vector once,
and accumulates all four query scores before an eight-lane reduction. The four
waves transpose scores through LDS and independently emit block top-eight
candidates. Production's exact global top-eight reduction remains unchanged.

The production-equivalent variant divides every key-sum component by its count
and rounds the mean back to BF16 before the dot. Moving division after the dot
changed random-control routes; preserving the original rounding restores 100%
top-eight set agreement in every tested B1/B2/B8, 1,024/2,048/4,096-centroid
case. At 4,096 centroids with INT64 production candidate indices:

| batch | centroid-major HIP score + top-8 | best Triton M16 MFMA + top-8 | change |
| ---: | ---: | ---: | ---: |
| 1 | **23.68 us** | 35.50 us | **33.3% faster** |
| 8 | **43.56 us** | 49.60 us | **12.2% faster** |

The fixed-mask specialization also distributes prefix initialization, prior
route clearing, and current-token arena publication across the score grid. This
avoids the two fallback preparation launches that erased the isolated scorer
win. In the synthetic full fixed-mask call, it measured 0.131 ms versus 0.153
ms for the previous fused Triton route path, with the same selected routes and
the same output error against the reference. In a matched seven-repeat real
65,280-token ProLong B1 vLLM A/B, the old Triton route kernel measured 2.033
ms/decode step and centroid-major HIP measured **1.978 ms/decode step** (2.71%
lower end to end). The device-written audit recorded the HIP specialization as
both configured and executed. A 64K NIAH-S3 smoke was 1/1; random route panels
have exact top-eight agreement, and the integrated output is identical to the
old path.

The same mapping was extended to Qwen3.8-27B-FP8's D=256/GQA6 geometry. Eight
waves score four centroids apiece while reusing each K fragment for all six
queries; six waves then emit the six independent block top-eights. A leaner
six-wave version was exact but slower (114.20 us score+reduce), so it is not
retained. At B8 and 4,096 centroids, the retained eight-wave result is:

| Qwen3.8 B8 route | score + exact global top-8 | change vs production |
| --- | ---: | ---: |
| production Triton M16, two waves | 74.02 us | baseline |
| **centroid-major HIP, eight waves** | **67.40 us** | **8.9% faster** |

The random panel has 100% top-eight row and slot agreement. The isolated
6.62-us saving per LOD layer predicts 105.9 us over Qwen3.8's 16 global layers.
A matched seven-repeat, real-ProLong 64K/B8 vLLM A/B measures 36.548 ms/step
with the prior Triton route and **36.436 ms/step** with centroid-major, a
112-us (0.31%) end-to-end reduction. The device-written audit confirms that
the GQA6 HIP path executed. Historical full AITER attention is 52.030 ms/step,
so the new result is 1.43x faster than full attention. The close agreement
between the predicted and observed absolute saving also shows why the overall
gain is small: routing is already only a small part of the 27B decode step.

The low-occupancy comparison must instead use B1. Two repeated prior-kernel
runs measure 29.290 and 29.274 ms/step, while two centroid-major runs measure
29.373 and 29.433 ms/step. Averaging each pair of medians gives **29.282 ms**
for the prior Triton path and **29.403 ms** for centroid-major: the GQA6
specialization is 0.121 ms or 0.41% slower at B1. A matched full-AITER B1
control is 31.571 ms, so prior LoD is 1.078x faster than full and
centroid-major is 1.074x faster.

This is an integration effect rather than incorrect routing. The eager
complete-call microbenchmark favors centroid-major (0.1374 versus 0.1573 ms),
but replaying the identical calls from CUDA graphs reverses the result: 0.0905
ms for centroid-major versus 0.0887 ms for Triton. The eager measurement was
therefore dominated by launch/enqueue behavior that vLLM's decode graph removes.
Once captured, the 512-thread/eight-wave GQA6 scorer does not beat the existing
MFMA path. This differs from Qwen3.5-0.8B's 256-thread/four-wave GQA4 mapping,
which retained a 2.71% end-to-end B1 benefit. GQA6 support remains opt-in and
should not be the default for Qwen3.8.

The alternative one-wave-per-GQA-query mapping was also implemented as a
controlled HIP benchmark. Four waves share a globally coalesced K-tile load;
the normalized K tile is transposed into padded LDS, and two lanes per wave
score each centroid. Padding, interleaved lane dimensions, and four independent
FMA accumulators remove the avoidable bank-conflict and dependency-chain costs.
It remains slower than centroid-major in every tested regime:

| batch | centroids | centroid-major score + top-8 | one wave/query | change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,024 | **23.52 us** | 26.68 us | 13.4% slower |
| 1 | 2,048 | **23.84 us** | 26.68 us | 11.9% slower |
| 1 | 4,096 | **24.88 us** | 27.52 us | 10.6% slower |
| 2 | 4,096 | **24.12 us** | 29.12 us | 20.7% slower |
| 8 | 4,096 | **43.68 us** | 61.88 us | 41.7% slower |

All query-wave top-eight sets agree exactly with the production MFMA control.
Its two-lane reduction is cheaper, but that does not repay writing every K
component to LDS and reading the tile again in all four query waves. The gap
widens with occupancy, consistent with LDS traffic becoming the bottleneck.
It is retained only as a benchmark variant and is not a production dispatch.

Enable the gfx942 D=256/GQA4-or-GQA6 specialization with
`VLLM_LOD_DECODE_CENTROID_MAJOR_HIP=1`. Dispatch is deliberately restricted to
score-only, grouped, 32-centroid routing; unsupported geometries retain the
existing Triton path. Raw results and the standalone benchmark are under
`centroid_major/` and `scripts/benchmark_centroid_major_route_score.py`.

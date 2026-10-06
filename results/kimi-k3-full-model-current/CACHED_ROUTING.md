# K3 cached-centroid routing and physical-B1 split tuning

October 6 follow-up to [DECODE_PIPELINE.md](DECODE_PIPELINE.md). Two-tier
decode only; no change to prefill, global top-eight selection, the 1,024-leaf
closure policy, separate sink, or global 256-token update cadence.

## Changes

K3 has one physically stored latent KV head, viewed as six virtual heads for
16-query-head tiles. The decode router previously bounded its cached-mean
view using the virtual shape. That falsely required six copies of the means,
so it fell back to dividing the sum keys by counts on every decode step.
The router now aliases the existing, refreshed mean-key cache using its
physical shape and a zero-stride head view. There is no gather, copy, new
cache, or extra allocation. Native GQA layouts retain their existing views;
incompatible/unavailable means retain sum routing.

The count bonus is still calculated in FP32. Reusing the attention arena's
FP16 log-count bias would alter candidate scores and is deliberately avoided.
Frozen random and tied-score checks produce **bit-identical candidate scores
and indices**, including ragged state lengths, reversed row indices, and
counts of zero, 1,024, and 1,025.

The compact consumer uses **32 splits at physical B1**, including one-request
owners, instead of 64. Live B4+ still uses 16; B2/B3 and legacy per-GQA
metadata remain at 64. All views reuse fixed 64-split backing storage. Split
changes preserve the attended KV set, but can change floating-point roundoff.

Code: [allocation-free view](../../lod_attention/kernels/_coarse_route_views.py),
[routing orchestration](../../lod_attention/kernels/paged_decode.py),
[consumer/reducer](../../lod_attention/kernels/kimi_gluon_decode.py).

## Isolated router check

Graph-replay GPU timings on frozen tensors; not serving latency. The original
and cached variants both use the current four-wave router and fresh FP32
count bonus. No collective, projection, update, or other model work is timed.

| Physical batch | Active centroids | Sum-key division (us) | Cached means (us) | Router speedup |
|--:|--:|--:|--:|--:|
| 1 | 512 | 12.145 | 6.786 | 1.79x |
| 1 | 4,096 | 23.954 | 12.866 | 1.86x |
| 8 | 512 | 23.907 | 12.741 | 1.88x |
| 8 | 4,096 | 128.901 | 81.935 | 1.57x |

The compiled router drops from 252 to 128 vector registers, without spills.
Source: [frozen route comparisons](../kimi-k3-mla-stack/oct6-cached-route-means.json).

## Physical-B1 consumer check

Complete consumer plus reduction, graph-replay microseconds. Independent
full-attention output/LSE checks pass. Synthetic compact sets test work
scaling; they do not establish trained-model quality.

| Compact KV rows | 64 splits | 32 splits |
|--:|--:|--:|
| 832 | 26.827 | 15.840 |
| 6,208 | 35.656 | 29.169 |
| 8,256 | 43.860 | 33.709 |
| 24,640 | 85.449 | 75.830 |

Source: [split sweep](../kimi-k3-mla-stack/oct6-single-row-consumer-splits.json).
The 12-layer request-owner fixture changes from 2.41468 to 2.37708 ms/step
with the split change alone, a small 1.6% improvement in a single-pass
comparison. Source: [baseline](../kimi-k3-mla-stack/oct6-owner-pipeline-baseline/b8-owner-lod.json),
[32 splits](../kimi-k3-mla-stack/oct6-owner-pipeline-split32/b8-owner-lod.json).

## Combined captured fixture

Twelve dummy-weight MLA layers, 64K, TP8/DCP8, 1,025 measured decode steps,
four audited updates per request/layer/rank. These fixtures retain real cache
construction and collectives, but are **not** full trained-model timings.

| Layout | Previous pipeline (ms/step) | Cached means + live splits (ms/step) | Latency decrease |
|:--|--:|--:|--:|
| Ordinary B1 | 2.15987 | 1.95529 | 9.5% |
| Eight request owners, physical B1 each | 2.41468 | 2.24648 | 7.0% |

Sources: [ordinary baseline](../kimi-k3-mla-stack/oct6-parallel-decode/b1-lod.json),
[ordinary follow-up](../kimi-k3-mla-stack/oct6-parallel-decode-cached-means/b1-lod.json),
[owner follow-up](../kimi-k3-mla-stack/oct6-owner-pipeline-cached-means/b8-owner-lod.json).
The follow-ups audit cached mean availability and actual 32-split use in
every rank/layer pool. A separate 257-step diagnostic trace reduces owner
router average duration from 26.98 to 17.05 us per layer invocation; compact
attention changes from 33.85 to 31.97 us. These instrumented durations are
not additive wall-time estimates or the source of the fixture latency above.

## Full trained model, 64K

End-to-end milliseconds per batched decode step, TP8/DCP8/EP8, resident
transformed weights and the same frozen ProLong prompts/continuations.
One complete untimed warmup, then 1,025 measured steps containing four
global-256 updates per request/layer/rank. No profiler in this window.
Unchanged dense controls are reused, not retimed solely for a LoD code change.

| Layout | Dense | Previous pipeline LoD | Cached-mean LoD |
|:--|--:|--:|--:|
| Ordinary B1 | 21.845 | 22.025 | 21.626 |
| Ordinary B8 | 34.198 | 32.536 | 32.229 |
| B8 one request's attention per GPU | 34.198 | 31.245 | 30.872 |

B1 is 1.8% lower latency than preceding LoD and approximately tied with dense
(1.0% lower in this single pass). Its loaded audit confirms cached means and
32 splits in all 192 rank/layer pools; frozen inputs, fully live decode and
four actual catch-ups pass the existing full-model validator. B1 uses node 2,
263,186-token capacity and 1 GiB native cache. Prefill was 8.523 s; no prefill
optimization is claimed. Source: [B1 result](oct6-cached-means-decode-b1-64k.json).
Request-owned B8 is 1.2% lower latency than preceding LoD and 9.7% lower than
dense (1.108x dense/LoD). Node 4 retains eight 2K prefill slices, 66,578-token
capacity and 3 GiB native cache. All eight owners execute 1,025 real B8 graph
replays, with four updates and 1,025 decoded tokens in each of 24 MLA layers.
Each pool confirms cached means, 32 splits, 96 local heads and global-256
cadence; prompt/continuation tokens and capacity settings match the preceding
owner run. Owner prefill was 69.093 s, with no algorithmic prefill change.
Source: [owner B8 result](oct6-cached-means-decode-owner-b8-64k.json).
Ordinary B8 is 0.9% lower latency than preceding LoD and 5.8% lower than
dense (1.061x dense/LoD). It uses node 2 with the same B8 capacity/cache sizes;
prefill was 68.345 s. All 192 pools confirm cached routing and 16 consumer
splits, and the full-model input/live-batch/four-update validator passes.
Source: [ordinary B8 result](oct6-cached-means-decode-b8-64k.json).
These small end-to-end improvements are single-pass observations, not a
precision claim about sub-percent differences. No speedup is extrapolated
to unmeasured context lengths.
Previous pipeline and dense sources are listed in [DECODE_PIPELINE.md](DECODE_PIPELINE.md).

## 16K / 128K context follow-up

Same current kernels and full-model four-update protocol, with no profiler,
prefix hits or preemptions. B8 here means **one request's attention per GPU**,
not ordinary DCP8 attention. Each owner executes all 96 heads; native TP8
projections, W_O, MoE and KDA remain distributed. The 64K points below are
unchanged anchors from the preceding run.

| Context | B1 dense (ms/step) | B1 LoD (ms/step) | B8 dense (ms/step) | B8 owner LoD (ms/step) |
|--:|--:|--:|--:|--:|
| 16K | 21.397 | 21.631 | 31.649 | 30.134 |
| 64K | 21.845 | 21.626 | 34.198 | 30.872 |
| 128K | 22.333 | 21.695 | 37.994 | 31.938 |

B1 is 1.1% slower than dense at 16K and 2.9% lower latency at 128K.
The B8 owner is 4.8% lower latency than dense at 16K and 15.9% lower at 128K
(1.190x dense/LoD at 128K). All completed points
verify frozen input hashes, all 192 loaded pool geometries and four actual
global-256 updates. All eight owners execute 1,025 real B8 graph replays.
Dense controls are reused; no code affecting dense has changed.

B1 retains 263,186-token capacity and 1 GiB native cache in the same engine
for both contexts. The 16K owner retains the tested 66,578-token capacity and
3 GiB native cache; the separate 128K owner run provisions 132,114 tokens,
also with 3 GiB native cache. Both owner runs retain eight 2K scheduler row
slices and global per-request 16K semantic prefill construction. Node 2
runs B1, node 4 runs owners; both reuse resident transformed weights.

The first 128K owner attempt completed its full 1,026-token warmup, but
the second generation exhausted memory in an AITER/FlyDSL MoE stage-two
allocation of 1.75 GiB (892 MiB physically free on rank 0), followed by HIP
launch-resource errors on other ranks. It has **no measured timing**.
The [failed artifact](oct6-cached-means-decode-owner-b8-128k.json) is retained.
The retry reuses the owner decode pool during prefill, projects six heads
at a time instead of twelve, enables the existing once-per-prefill-batch
idle allocator pressure check, and uses an expandable allocator with HIP
scratch reclamation. These are storage/execution-policy changes, not a
different KV set or update cadence; they do not change the decode kernels.
The [memory-safe retry](oct6-cached-means-decode-owner-b8-128k-memory-r2.json)
completed its measured generation and all audits. All eight owners ran
1,025 real graph replays, with four updates in every MLA layer and 32
consumer splits; frozen ProLong inputs and the fully live cohort match.
Prefill took **202.915 s/batch**, slower than the reused dense control's
155.750 s. The recorded pressure checks reclaimed idle allocations **56
times per rank during the measured prefill** (zero during warmup). Those
costs are included, not removed from latency. This is a completed decode
speedup, **not** a fast 128K owner-prefill result. No timing comes from warmup.

Prefill latency from these same measurements (no separate timing runs):

| Context | B1 dense (s) | B1 LoD (s) | B8 dense (s) | B8 owner LoD (s) |
|--:|--:|--:|--:|--:|
| 16K | 2.038 | 2.044 | 16.327 | 15.772 |
| 128K | 19.296 | 17.411 | 155.750 | 202.915 |

Sources: [B1 16K/128K](oct6-cached-means-decode-b1-16k128k.json),
[owner B8 16K](oct6-cached-means-decode-owner-b8-16k.json),
[owner B8 128K, memory-safe](oct6-cached-means-decode-owner-b8-128k-memory-r2.json),
[dense B1 controls](oct4-full-b1-decode-power2-four-updates.json),
[dense B8 16K control](oct4-full-b8-decode-16k64k-four-updates.json),
[dense B8 128K control](oct4-full-b8-decode-128k128k-four-updates.json).
The [power-of-two table](DECODE_POWER2.md) now shows only current-kernel LoD
results: ordinary B1 and explicitly labeled row-per-GPU B8 at 16K/64K/128K.
Old ordinary-B8 and one-wave LoD results are removed; unmeasured current
contexts remain blank. The raw historical artifacts are retained.

To reproduce the context follow-up, use the exact environment and trained
weight-cache setup from [the ordinary command](ROUTER_DECODE.md#reproduction)
or [the owner command](OWNER_CAPTURED_DECODE.md#reproduction), with these
arguments. Seed remains 0, repeats remain 1, and output tokens remain 1,026.
Use a new output file; do not overwrite archived controls.

| Layout | Lengths | Max model length | Native cache bytes | Reference-baselines file |
|:--|:--|--:|--:|:--|
| Ordinary B1 | `16384 131072` | `263186` | `1073741824` | `oct6-lod-b1-decode-power2-correct-dcp-four-updates.partial.json` |
| Owner B8 | `16384` | `66578` | `3221225472` | `oct6-lod-b8-decode-16k64k-correct-dcp-four-updates.json` |
| Owner B8 | `131072` | `132114` | `3221225472` | `oct6-lod-b8-decode-128k128k-correct-dcp-four-updates.json` |

Reference paths are under `results/kimi-k3-full-model-current/`. Run the
owner cases sequentially on the same resident-weight node, not concurrently.
Fixed B8/TP8/DCP8 two-tier cohorts now select the owner layout automatically;
`--owner-tp-mla` remains an explicit equivalent. Use `--ordinary-dcp` for the
ordinary-DCP control. Variable-size, freely finishing quality cohorts keep
ordinary DCP rather than claiming to support partial eight-owner batches.
Keep compiled artifacts on local disk and retain the complete-shape warmup.
The 128K memory retry additionally sets:

```bash
export LOD_KIMI_OWNER_POOL_BACKED_PREFILL=1
export LOD_KIMI_OWNER_PREFILL_HEAD_GROUP=6
export LOD_KIMI_OWNER_PRESSURE_CHECK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=0
```

The latest ordinary B1 decoder also passes a matched, freely generated
NIAH-S3 smoke check: **4/4 at nominal 16K and 4/4 at nominal 128K**, as does
dense, with identical prompt hashes and no teacher forcing. This is a small
retrieval check, not a full benchmark score. See
[the quality check](CHAT_QUALITY.md#october-6-cached-routing-b1-check).

## Verification and reproduction

The initial integrated GPU subset passes 133 tests: cached-mean view bounds,
graph replay, owner row permutations, fixed/shared scratch, split reductions,
prefill cache installation and global updates. Three additional GPU cases
pass: random-query serving outputs match sum routing exactly at 32K/64K
capacity, and cached means match the updated FP32 sums/counts after a real
global-256 update. The broad CPU subset passes 411 tests with 71 device-only
skips, before the additional audit-import regression. Isolated router
speedups are not substituted for full-model end-to-end measurements.

Run from the repository root in the pinned K3 v10 environment, keeping
compilation artifacts on local disk:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m pytest -q \
  tests/test_coarse_route_views.py tests/test_kimi_split_reduce.py \
  tests/test_kimi_owner_decode.py tests/test_decode_scratch.py \
  tests/test_kimi_request_prefill.py
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_cached_route_means \
  --output cached-route-means.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_consumer_tune \
  --single-row-only --output single-row-splits.json
```

For full-model runs, retain the exact frozen ProLong inputs, resident weights,
capacity, scheduler and native cache sizes from [DECODE_PIPELINE.md](DECODE_PIPELINE.md).
Warm the complete shape before one measured pass of 1,025 decode steps, with
four audited global-256 updates per request/layer/rank. Dense controls are
unchanged and reused. No profiler runs inside the serving measurement.

The final default-dispatch check ran a two-layer vLLM fixture at B8/16K
**without an owner flag**: all eight owners replayed their graphs for 1,025
steps and observed four updates. This is a dispatch test, not a full-model
speed point. The integrated GPU subset passed **134 tests** in the pinned
image; the broader CPU suite passed **649 tests** with 96 device-only skips,
including default-dispatch, explicit-override and failure-artifact regressions.

## October 6 upstream AITER review

The active K3 v10 image contains **AITER 0.1.22.post1**, Torch
**2.12.0+rocm10.0.0**, and Triton **3.8.0+gitc0a142ff**. The upstream release
tag resolves to `b4d9154d125e09efbe098d986e40fea3549c1244` (September 17).
The wheel is not an untouched checkout: `aiter/mla.py` adds an unsplit-LSE
helper. Its `fused_moe.py` and `ops/triton/gluon/mla_gluon.py` match that tag
byte-for-byte. The inspected upstream main tip is
`e8048a1620f528a27bebd1ca9ada6cbee6d40d99`.

This was a source/compatibility review, **not an upgrade or performance
measurement**. No installed library or compiled kernel was replaced.
Priorities reflect the latest owner profile: compact attention about
31.97 us/layer, routing 17.05 us, split reduction 4.50 us and latent-output
projection 7.06 us. These separate instrumented durations are not additive
full-model latency estimates.

| Priority | New upstream change | Relevance and limitation |
|:--|:--|:--|
| First attention experiment | [`4b12dc51`, gfx942 sparse MLA, #5721](https://github.com/ROCm/aiter/pull/5721) | BF16 512-latent + 64-direct support, gathered KV indices, LSE return and split partials. Uses 32-key tiles/four warps to fit gfx942's 64-KiB LDS. Closest candidate for our compact selected-leaf consumer; not evidence that it beats our existing Gluon kernel. |
| Small code-generation audit | [`8cfa0902`, Triton 3.8 PA register fix, #5909](https://github.com/ROCm/aiter/pull/5909) | Removes a conditional carry of prefetched KV tiles that kept both copies live after compiler if-conversion. Our image uses the affected compiler generation, but our current MLA loop has no equivalent double-buffered conditional: the exact patch is not a drop-in speedup. Use the lesson when adding prefetch. |
| End-to-end K3 floor | [`8253efc4`, fused KDA decode, #4584](https://github.com/ROCm/aiter/pull/4584), and [`446b8e9a`, chunked KDA prefill, #5866](https://github.com/ROCm/aiter/pull/5866) | KDA is recurrent attention, separate from MLA/LoD. The active image **already uses fused HIP KDA decode**; the new Gluon kernel is an alternative to compare against that, not a three-launch-to-one windfall. Prefill currently uses the vendored Triton fallback on gfx942. The new gfx950 Gluon prefill is therefore more interesting, but needs a backport rather than an upgrade alone. Any gain helps both dense and LoD. |
| Fewer model-side launches | [`aa848152`, K3 merged MoE front, #5321](https://github.com/ROCm/aiter/pull/5321) | One merged projection plus a SiTU activation/split epilogue for shared-expert, router and routed-latent branches. Relevant to small-batch overhead outside MLA; requires model/weight integration, not just replacing attention. |
| Graph-safety audit | [`6085899f`, private split-K scratch pool, #5430](https://github.com/ROCm/aiter/pull/5430) | Prevents persistent GEMM scratch from aliasing freed graph intermediates when first allocated during capture. Applicable to FlyDSL consumers and supported by our Torch version. LoD's own shared scratch is already allocated before capture, so this is not automatically a fix for that workspace. |
| Later fusion idea | [`7bfe279d`, persistent MLA split/merge, #6126](https://github.com/ROCm/aiter/pull/6126) | Plans splits from live device-side lengths and merges inside the same launch. Good design reference for removing split/reduction overhead, but the shipped gfx950 FP8, 128-head V4 kernel is not a BF16 gfx942 K3 replacement. |

Important integration constraints for #5721: its public interface has no
per-KV log-mass bias, unlike our combined coarse/exact field, and accepts one
index stream per query rather than our per-16-head-group streams. Preserve
those group-specific selections, the separate sink and coarse replacement;
either add the bias to an adapted kernel or use it only for the exact branch
and correctly merge output/LSE with the remaining coarse branch. Its wrapper
also allocates split scratch, so graph-safe shared buffers need adapting.
gfx942 FP8 is explicitly rejected because its encoding differs from gfx950;
none of this establishes that chronological KV quantization is acceptable.

Additional useful checks:

- [`32fc8813`, >2-GiB MLA load masking, #5648](https://github.com/ROCm/aiter/pull/5648)
  fixes invalid tail reads/undefined values in the upstream gfx950 global-load
  path. That exact async path is not our CDNA3 consumer, but its large-cache
  and partial-split tests are worth reusing for long-context safety.
- [`330f1276`, Qwen3.8-27B TP1 gfx942 GEMM tuning, #5585](https://github.com/ROCm/aiter/pull/5585)
  is directly relevant to the other release model when the corresponding
  A8W8 blockscale GEMM path is used. Do not apply gfx950 K3 tuning rows to
  gfx942 without measuring our actual shapes.
- [`6c179174`, LDS-pipelined MLA, #6098](https://github.com/ROCm/aiter/pull/6098)
  uses gfx1250-specific four-stage transfer machinery. Sliced operand loads
  and prefetch scheduling are useful ideas, but the kernel itself is not
  runnable on MI325X as-is.
- [`1e2230af`, native 12/8-head MLA support, #5695](https://github.com/ROCm/aiter/pull/5695)
  sounds particularly relevant to our 12 heads/rank; the ASM branch added
  here is explicitly gfx950 FP8. It does not remove our gfx942 BF16 padding
  merely by updating AITER.

Recommended order: test/adapt #5721 against frozen real selected-leaf inputs
on the existing small fixture first; then audit register live ranges and
reduce/projection fusion. Investigate fused KDA separately to lower the
full-model floor. Keep the pinned environment and dense controls unchanged
until each proposed change passes correctness and matched timing checks.

### KDA-specific assessment

K3 has **69 KDA layers versus 24 MLA layers**, so recurrent-attention speed
matters even when MLA/LoD is already fast. This count describes the fraction
of the model affected; it does not explain per-layer speedup differences.

The active ROCm model is `vllm.models.kimi_k3.amd.linear`. Full-rank K3 gates
select `KimiK3DeltaAttention`, not the shared Kimi-Linear class. The October 6
full-model startup logs explicitly confirm **fused KDA decode enabled** and
**KDA prefill backend: triton**. The image's HIP decode fuses convolution,
state recurrence and gated normalization on gfx942 already. Its fused HIP
prefill support predicate, however, currently restricts that path to gfx950.
Reading only the shared Kimi-Linear fallback would give the wrong picture of
our actual decoder.

The two most promising KDA directions are:

1. **Prefill kernel/backport:** #5866 merges much of the chunk preparation
   into one launch and follows with a state/output walk; its gfx950 source
   has ordinary buffer operations and MFMA, so a CDNA3 port is plausible,
   not guaranteed. Match the lower-bounded channel gate, FP32 state,
   convolution and sigmoid output gate. The public wrapper computes chunk
   metadata with a device-to-host scalar read unless supplied, and allocates
   several large workspaces: reuse scheduler metadata and fixed scratch
   rather than introducing new synchronization/allocation into each chunk.
2. **Direct paged-state I/O:** [`84daaa65`, FlashKDA #5754](https://github.com/ROCm/aiter/pull/5754)
   lets the prefill kernel read/write the live recurrent state pool and a
   caller-provided output, avoiding gather/scatter and output copies. The
   current gfx942 Triton fallback gathers initial states and copies final
   states/output; the new FlashKDA path itself is architecture-gated, so
   this is a port/integration target, not a benefit automatically obtained
   by installing newer AITER. Respect padded pool strides and never pack a
   writable state pool into a detached contiguous copy.

For decode, #4584 streams smaller state-row tiles with prefetch rather than
necessarily holding the entire head state live. This is worth a matched
kernel comparison against the **existing fused HIP** decoder, using B1/B8,
12 local heads, 128 channels and captured replay. Do not benchmark against
the slower three-operation fallback and call that our current speedup.

Other relevant KDA commits are [`52ffe895`, convolution-history ordering
fix #5619](https://github.com/ROCm/aiter/pull/5619), and [`f32a197d`, speculative
KDA decode #5709](https://github.com/ROCm/aiter/commit/f32a197d).
The former is a useful correctness reference if backporting fused decode.
The latter parallelizes work for multiple speculative tokens and is not an
immediate speed improvement for our ordinary single-token K3 decode sweep.
GDN-specific Qwen kernels use different gates/head topology; they cannot be
substituted for KDA merely because both are delta-rule attention.

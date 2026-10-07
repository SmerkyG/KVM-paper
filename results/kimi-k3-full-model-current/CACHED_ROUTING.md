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

### Isolated upstream KDA tests, October 6

These are **dense-attention fixture tests**, not full trained-K3 latency or
quality scores. Production defaults, installed AITER and vLLM are unchanged.
The benchmark-only MIT backports live in `benchmarks/experimental/kda_gfx942/`.
Only CDNA3 buffer/MFMA substitutions, local helper imports and launch geometry
are changed from upstream #5866/#4584. Compilation uses local `/tmp` caches.

The per-rank prefill probe uses K3's TP8 geometry: 12 heads, 128 channels,
BF16 projections, FP32 V-first recurrent state, the -5 bounded gate and
strided beta/QKV inputs. Metadata is provided before timing. A short FP32
token recurrence checks ragged 63/65-token sequences, strongly decaying gates,
and near-duplicate keys with weak decay. Both the native and new BF16 solvers
have about 1% RMS error in the deliberately difficult duplicate-key case;
this is reported, not hidden behind the ordinary-case 0.8% tolerance.

| 16K KDA prefill kernel | Native image | Backport | Speedup |
|---|---:|---:|---:|
| One state-walk group | 1.698 ms | 1.368 ms | 1.24x |
| Eight state-walk groups | 1.699 ms | 0.976 ms | 1.74x |

The 16K output differs from native by 0.585% relative RMS and the final state
by 0.412%. Changing to eight groups preserves these errors. Timing is one
block of 100 warmed CUDA-graph replays; projection, convolution and final
gated norm are not included in this kernel table. The first CDNA3 walk
configuration required 74,240 bytes of LDS and was rejected before timing.
The working walk uses 16 value rows, one warp and one buffer stage; the
split pass uses 32 rows/two warps. No unsupported gfx950 instructions remain.

The independent decode probe initializes recurrent state from native 16K
prefill, includes conv + recurrence + gated RMSNorm, and compares **the
image's fused HIP decoder**, not its fallback. Native and Gluon have bitwise
equal BF16 output and convolution state; FP32 recurrent state differs by
about 5e-8 relative RMS after one step and 1.1e-6 after 33 recurrent steps.

| KDA decode, 12 local heads | Fused HIP | Gluon backport | HIP/Gluon |
|---|---:|---:|---:|
| B1 | 6.810 us | 7.666 us | 0.888x |
| B8 | 7.903 us | 8.568 us | 0.922x |

One warmed block of 1,025 captured replays is measured for decode, without
profiling. **Keep native HIP decode:** this backport is correct but slower.
The useful candidate so far is the prefill replacement. Full four-layer TP8
validation includes three real KDA modules and one dense MLA module with
7168-wide hidden states, actual projections, conv, gates, norms and TP
collectives; it deliberately omits FFNs and AttnRes mixing equally. Its
initial comparison retains native state gather/scatter and output copies so
direct paged state I/O is not accidentally mixed into the first ablation.

Reproduction (the existing v10 direct launcher, no cluster runner required):

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_kda_upstream_probe \
  --groups 8 --output results/kimi-k3-kda-upstream/prefill-g8.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_kda_decode_upstream_probe \
  --output results/kimi-k3-kda-upstream/decode.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_kda_dense_fixture \
  --output results/kimi-k3-kda-upstream/dense-mixed4.json
```

Run each serially on otherwise idle GPUs. The kernel probes need one MI325X;
the mixed-layer fixture needs eight. The raw accepted kernel measurements
are `results/kimi-k3-kda-upstream/oct6-prefill-g1-16k.json`,
`oct6-prefill-g8-16k-serial.json` and `oct6-decode-16k-serial.json`.
Failed preflight logs are retained separately and are not timing evidence.

### Dense workload and direct state I/O follow-up

The four-layer TP8 fixture's eight-group walk reduced warmed dense prefill
from **27.526 to 24.797 ms** (1.110x). Removing the state gather/scatter and
output copy independently gave **24.727 ms**, essentially unchanged in this
single-pass combined workload. Direct I/O and copied I/O produced bitwise
equal final hidden states on every rank. This fixture has seeded weights,
three real KDA modules, one dense MLA module, and no FFNs or AttnRes mixing;
it is not the trained full model. Raw result:
[`oct6-dense-mixed4-g8-paged-16k.json`](../kimi-k3-kda-upstream/oct6-dense-mixed4-g8-paged-16k.json).

In the isolated KDA kernel, direct pool/output I/O was 1.053x faster for a
single 16K sequence and 1.055x for ragged 16K/4097 sequences. Tests include
padded state-slot strides, mixed initial-state flags and untouched guard
regions. Output and final state were bitwise equal to the copied variant.
Raw result:
[`oct6-direct-state-16k.json`](../kimi-k3-kda-upstream/oct6-direct-state-16k.json).

The trained-model test uses one resident TP8/DCP8/EP8 K3 engine on node 2,
the existing transformed INT4-MoE weight daemon, unchanged native dense MLA,
and the frozen ProLong speed-token corpus. It runs B1 and B8 in that same
engine (eight-request capacity, 16,392 aggregate scheduler tokens, 16K row
chunks, 2 GiB native cache per GPU). Each variant gets one exact-shape warmup
followed by one canonical uninstrumented serving measurement, with one
generated token; these are **prefill timings, not decode measurements**.
There are no prefix hits or preemptions. The first new variant retains all
native state/output copies; the second separately removes them. No installed
library or model default changes. Full results and prompt hashes are in
[`oct6-trained-dense-prefill16k.json`](../kimi-k3-kda-upstream/oct6-trained-dense-prefill16k.json).

| Trained K3 dense prefill, 16K per request | Native KDA | New KDA, same copies | New KDA, direct I/O | Native / direct |
|:--|--:|--:|--:|--:|
| B1 | 2.0437 s | 2.0145 s | 2.0083 s | 1.0176x |
| B8 | 16.3583 s | 16.1398 s | 16.1237 s | 1.0145x |

Both new variants generated the same first token as native for every prompt,
and their tokens also matched each other. That is a sanity check on nine
prompt evaluations, **not a trained-model quality benchmark**. End-to-end
prefill latency fell **1.73% at B1 and 1.43% at B8**; the 1.74x isolated KDA
kernel gain does not imply a comparable whole-model gain. B8 follows the
normal 16K aggregate scheduler budget, not a new 8x16K model batch. The full
model run completed successfully; startup, compilation and warmup are
excluded from the table. No new KDA production default has been promoted.

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_kda_dense_fixture \
  --groups 8 --direct-state --output dense-mixed4-g8-paged.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_kda_state_io_probe \
  --output direct-state-16k.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_kda_dense_prefill \
  --checkpoint /path/to/local/Kimi-K3 --weight-cache-id YOUR_RESIDENT_K3_CACHE \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --length 16384 --batches 1 8 --groups 8 --output trained-dense-prefill16k.json
```

### New sparse MLA with per-key count bias

The benchmark-only adaptation in `benchmarks/experimental/sparse_mla_gfx942/`
vendors the MIT kernel from AITER **4b12dc51 / #5721**, not the image's older
library. It retains the **512 latent + 64 K-only** contraction and returns
FP32 **natural-log LSE**. Optional `key_log_count[cache_row]` adds
`log(count)` **after** scaling QK: the exp2 implementation multiplies this
bias by `log2(e)`, not by the attention scale. Exact/local rows receive zero
bias. This is one BF16 segment; unsupported FP8/two-segment bias launches
are explicitly rejected rather than silently dropping their bias.

Validation covers ragged index lists, -1 sentinels with a NaN-poisoned null
slot, entirely masked rows, 12/16/96 heads, single/split-K execution,
nonzero direct-channel-only queries, and the equality to explicitly
repeating identical keys/values according to integer counts. Finite-row LSE
differs from FP32 by at most **1.1e-6** in the small cases; output RMS error
is 0.057–0.095%. Count replication has 0.189% BF16 output RMS error. Empty
rows give zero output and -infinity LSE; their normalization is guarded.

| Sparse MLA, 16K gathered rows, 12 heads | No bias | Log-count bias |
|:--|--:|--:|
| B1 | 49.69 us | 56.83 us |
| B8 | 66.54 us | 69.11 us |

These are 100 warmed graph replays of the kernel and reduction, not K3
latency or a speed comparison against our production compact consumer. The
minimal wrapper uses per-query CSR lists shared by its query heads and
allocates scratch; production integration still must preserve head-group
selections and reuse scratch. **No production dispatch was changed.** Raw
result: [`oct6-sparse-mla51264-bias.json`](../kimi-k3-kda-upstream/oct6-sparse-mla51264-bias.json).

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_sparse_mla_bias_probe \
  --output sparse-mla51264-bias.json
```

### Triton 3.8 prefetch-register-pressure fix

An isolated in-memory backport of **8cfa0902 / #5909** clones the two affected
Gluon paged-attention JIT functions, removes only their loop-tail conditional
prefetch carry, and keeps separate compile caches for the changed source.
It restores the original functions after the comparison; no installed file
is modified. All three gfx942 BF16 16K cases produced **bitwise equal outputs**.

| Conventional paged attention | Original | Fixed | Speedup | Registers before / after |
|:--|--:|--:|--:|--:|
| B1, 16 Q heads / 1 KV head | 46.05 us | 34.58 us | 1.33x | 289 / 210 |
| B8, 16 Q heads / 1 KV head | 66.43 us | 36.37 us | 1.83x | 289 / 210 |
| B8, 64 Q heads / 8 KV heads | 155.53 us | 142.00 us | 1.10x | 322 / 250 |

Timing includes the unchanged final reduction, using 200 warmed graph
replays. These are **not K3 MLA or end-to-end speedups**: K3 does not dispatch
to these conventional paged-attention kernels. Our existing K3 decoder has
no conditional prefetch carry; the new sparse MLA already carries its
prefetched FP8 tiles unconditionally, and its tested BF16 loop is not
double-buffered. There is consequently no equivalent fix to add to either
K3 loop. Raw result:
[`oct6-pa-prefetch-triton38-fix.json`](../kimi-k3-kda-upstream/oct6-pa-prefetch-triton38-fix.json).

The first PA comparison exposed a second AITER compiler-artifact root used
by its C++ template reduction. That untimed build defaulted to the image's
HOME on Ceph, despite the ordinary AITER/Triton local-cache overrides. The
launcher now additionally sets `AITER_ROOT_DIR=/tmp/dan-agent/aiter-template-k3`
so future template builds also stay on local disk. Explicit overrides remain
supported; the launcher/default-override and experiment CPU tests pass.

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.aiter_pa_prefetch_fix_probe \
  --output pa-prefetch-triton38-fix.json
```

## Further upstream ideas: isolated October 7 tests

All three ideas were tried **independently**, on node 3 / MI325X, using the
same pinned v10 environment. No installed AITER/vLLM file, model weight,
production default, or LoD math was changed. The user approved G8/direct-state
KDA prefill as the baseline for subsequent model comparisons. These initial
tests isolate the affected kernels and do **not** execute KDA, MoE experts,
collectives, or the complete trained model; they cannot establish an
end-to-end speedup or trained-model quality. The earlier full-model KDA gain
is not included in any ratio below.

Inputs are seeded BF16 geometry fixtures, not generated ProLong activations.
Every arm uses identical inputs/weights and preallocated output/workspace.
All compilation and warmup is untimed and uses local `/tmp/dan-agent` caches.
Reported times average two short graph-replay measurements. These are not new
entries for the canonical end-to-end speed panel.

### 1. Query-row packing for fused prefill routing/coarse attention

The lesson from [AITER #4963](https://github.com/ROCm/aiter/pull/4963) was
tested on our **actual fused CK coarse/top-eight operator**, rather than
substituting the FP8 indexer's head-summed/ReLU calculation. Production
already uses 128 query rows per tile. Compare its 128-row/four-wave tile
against the existing isolated 64-row/two-wave tile, with feature-axis steps
32 and 64. Each case has 12 local query heads, 4,096 centroid keys,
Dqk=192/Dv=128 and unchanged log-count bias and global top-eight reduction.
Thus this is a shape-tuning experiment, **not a port of the indexer kernel**.

| Query rows in chunk | Current 128-row tile, step 32 (ms) | 64-row tile, step 32 (ms) | Current/new |
|--:|--:|--:|--:|
| 256 | 1.031 | 0.756 | 1.36x |
| 2,048 | 1.117 | 0.814 | 1.37x |
| 16,384 | 4.652 | 5.006 | 0.93x |

Step 64 has no material additional gain: 0.756/0.814/4.994 ms for the
64-row tile and 1.031/1.118/4.657 ms for the 128-row tile. Every variant
produces **bitwise-identical coarse output, LSE, and route sets** to the
production geometry on the full test inputs. Independent FP32 checks of
32 queries pass: route scores recover global top-eight and coarse-output
RMS error is about 0.61% for every arm, including the original CK baseline.

Conclusion: retain 128 rows for full 16K chunks. A 64-row tile is a promising
candidate for the short 2K row slices in request-owner B8 prefill, but that
complete model/layout has **not** been retimed with it here.
Source: [`oct7-centroid-tiling-oracle.json`](../kimi-k3-kda-upstream/oct7-centroid-tiling-oracle.json).

### 2. One-launch attention plus last-arriver split reduction

Inspired by [AITER #6126](https://github.com/ROCm/aiter/pull/6126), the
experimental Gluon kernel clones production stage-one attention unchanged,
writes its normal split partials, and lets the last completing workgroup
merge them in the same launch. A device-scope release/acquire counter plus
CTA barriers orders the writes. It never spin-waits; counters reset after
every invocation and have dedicated, preallocated, single-stream storage.
No gfx950 code object or FP8 approximation is used.

Graph-replay **consumer plus reduction** times, physical B1/32 splits:

| Query heads | Compact KV rows | Existing two launches (us) | Fused last-arriver (us) | Existing/fused |
|--:|--:|--:|--:|--:|
| 12 | 4,096 | 18.68 | 36.40 | 0.51x |
| 96 | 4,096 | 21.28 | 51.30 | 0.41x |
| 96 | 16,384 | 51.33 | 77.23 | 0.66x |

The 96-head cases represent the attention-owner geometry, not a logical
eight-request model timing. Checks include distinct per-head-group lengths,
the 512+64 key split, log-mass bias, FP32 output/LSE oracle, exact-page
descriptors, poisoned unused cache rows, all-masked/empty inputs, and 100
graph replays without leaking counters. Maximum output disagreement against
the original is 0.00028% RMS and maximum LSE disagreement about 1.9e-6.

This is **not a spill failure**: original stage one uses 284 vector registers
and the fused kernel 280; both use 65,536 bytes of shared memory and zero
spills. The likely tradeoff is the new atomic protocol and merging a whole
16-head tile in one last-arriving workgroup rather than the separate
parallel per-head/channel reduction. That explanation is an inference, not
a separately profiled epilogue measurement. Reject this prototype for
production; saving one launch did not save time.
Source: [`oct7-persistent-merge-resources.json`](../kimi-k3-kda-upstream/oct7-persistent-merge-resources.json).

### 3. Merged K3 MoE front

The [AITER #5321](https://github.com/ROCm/aiter/pull/5321) organization was
adapted to gfx942 using one BF16-input/weight, FP32-output GEMM followed by
a Triton split/SiTU epilogue. It replaces only the three front projections
and shared activation, **not** routed experts or output transforms. Shared
gate/up values round to BF16 before SiTU, router logits remain FP32, and the
routed latent rounds to BF16, matching the native contracts. FP32 router
weights are rejected rather than silently downcast. Eight distinct weight
sets are streamed inside each graph to avoid a single cache-hot matrix
pretending to represent all model layers.

| Per-rank geometry | Tokens in call | Native front (us) | Merged front (us) | Native/merged |
|:--|--:|--:|--:|--:|
| TP8, shared width 96 | 1 | 34.14 | 22.52 | 1.52x |
| TP8, shared width 96 | 8 | 33.89 | 22.92 | 1.48x |
| TP8, shared width 96 | 16,384 | 1,606.99 | 1,994.52 | 0.81x |
| TP1, shared width 768 | 1 | 39.83 | 29.40 | 1.35x |
| TP1, shared width 768 | 8 | 38.18 | 28.98 | 1.32x |
| TP1, shared width 768 | 16,384 | 2,135.18 | 2,313.46 | 0.92x |

K=7,168, router width 896, routed latent width 3,584 in both geometries.
**Geometry correction (October 7):** this first probe used a global shared
width of 768. Official K3 instead has two 3,072-wide shared experts: 6,144
globally and **768 per TP8 rank**. These historical numbers therefore do
not establish the gain on official K3. The corrected probe below uses the
official dimensions; its default is now `--shared-intermediate 6144`.
These are single-GPU, TP-shaped kernel tests, not actual TP1/TP8 serving.
All branch outputs pass native and independent FP32 checks; raw router
top-16 sets match on every tested row. This does not yet test trained
correction-bias/expert dispatch or model predictions. Intermediate 512/2048
shapes were also tested and show smaller gains; use the raw results rather
than extrapolating the decode gain to large prefills.

This initially justified a **decode-sized** model-integration experiment,
not enabling it for 16K prefill. The trained results below supersede that
candidate recommendation. The KDA baseline and attention kernels remain
unchanged in this MoE experiment.
Sources: [`oct7-moe-front-oracle.json`](../kimi-k3-kda-upstream/oct7-moe-front-oracle.json),
[`oct7-moe-front.json`](../kimi-k3-kda-upstream/oct7-moe-front.json).

### Reproduction

From the repository root, run each independently on an otherwise idle gfx942
GPU using the pinned runtime (or an equivalent Torch 2.12/Triton 3.8 runtime
with the patched CK fused-route operator). No trained-model download or
weight daemon is needed. The launcher keeps all compilation artifacts local.

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_centroid_tiling_probe \
  --output centroid-tiling.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_persistent_merge_probe \
  --output persistent-merge.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_moe_front_probe \
  --shared-intermediate 768 --tokens 1 8 16384 --output moe-front.json
```

All device/oracle/graph checks pass. The related CPU reference, launcher,
and previous upstream-probe suite passes **17 tests**. No new production
defaults or official trained-model speed/quality tables were changed.

### October 7: decode-only trained MoE-front integration

The experimental adapter in `benchmarks/experimental/kimi_moe_front.py`
packs immutable input-projection weights once and preallocates output
scratch. It applies to calls of at most the benchmark batch size (1 or 8);
large prefill retains native MoE, but a tiny prefill fragment can use the
candidate. Only decode latency is compared here.
It preserves the native FP32 router/correction bias, selected experts,
expert kernel, shared-expert stream ordering, routed output normalization,
sharded output projection, and reductions. No production default changes.

First, the corrected official geometry gives these **sequential isolated
front-kernel** results (native / merged / native repeat, microseconds):

| TP-shaped rank | Tokens | Native (us) | All-three merged (us) | Native repeat (us) |
|:--|--:|--:|--:|--:|
| TP8, shared width 768 | 1 | 37.51 | 27.28 | 36.71 |
| TP8, shared width 768 | 8 | 38.07 | 27.68 | 35.41 |
| TP1, shared width 6,144 | 1 | 81.56 | 68.82 | 76.66 |
| TP1, shared width 6,144 | 8 | 78.56 | 65.85 | 74.22 |

These tests do not reproduce the native shared-stream overlap, and their
speedup must **not** be reported as a full-model speedup.
Source: [`oct7-moe-front-correct-geometry.json`](../kimi-k3-kda-upstream/oct7-moe-front-correct-geometry.json).

The trained full K3 experiment uses TP8/DCP8, real frozen ProLong token
traces, a 16,384-token prompt, and 1,025 measured batched decode steps.
Each row incurs **four global 256-token LoD updates**. B8 uses the default
one-request-per-GPU LoD layout. The approved G8/direct-state KDA prefill is
fixed for both arms. Weights remain in the resident daemon throughout.
Native and candidate graphs are captured separately and retained in the
same engine; every measured step is audited as a graph replay. Each arm
gets a complete untimed, same-shape warmup. The order is native, merged,
native repeat, with one measured serving window per arm and no profiler
or internal GPU timers in those windows.

| Batch | Native (ms/step) | All-three merged (ms/step) | Native repeat (ms/step) | Mean-native / merged |
|--:|--:|--:|--:|--:|
| 1 | 21.6025 | 22.2766 | 21.5973 | 0.970x |
| 8 | 30.2651 | 30.1163 | 30.2548 | 1.005x |

Merging all three input projections costs **7.415 GiB extra per rank** in
this reversible adapter. The isolated gain does not survive as a useful
serving gain: B1 is about 3.1% slower and B8 only about 0.5% faster. Do not
enable this version by default.

Every trained layer passed the front-output and native corrected-expert-set
checks; full MoE output comparisons passed for the first and last layers.
The timings and cadence audits completed successfully. Both jobs then
aborted during a **separate untimed Kineto/HSA profiling check**, so their
final greedy-output comparison did not complete. This is not a timed-kernel
failure, but these files are not evidence of completed end-to-end quality
validation. Subsequent runs replace that profiler check with a private
scratch canary: poison scratch outside timing, run an untimed greedy
continuation, and check that only the candidate graph wrote it on every
rank. This adds nothing to the measured/captured hot path.

Sources: [`oct7-moe-front-trained-lod-b1-16k-v2.json`](../kimi-k3-kda-upstream/oct7-moe-front-trained-lod-b1-16k-v2.json),
[`oct7-moe-front-trained-lod-b8-16k.json`](../kimi-k3-kda-upstream/oct7-moe-front-trained-lod-b8-16k.json).

The narrower follow-up merges **only router + routed-latent projection**,
leaving shared gate/up, SiTU, and down projection on the native shared
stream. Its two-layer official-geometry graph fixture passed native routing,
MoE-output, and 32-token greedy equality checks; native / merged / native
repeat were 1.0364 / 1.0075 / 1.0361 ms per B8 decode step. This is fixture
evidence, not a trained full-model gain.
Source: [`oct7-moe-front-routed-fixture-b8.json`](../kimi-k3-kda-upstream/oct7-moe-front-routed-fixture-b8.json).

The trained router + latent merge completed the same A/B/A protocol:

| Batch | Native (ms/step) | Router + latent merged (ms/step) | Native repeat (ms/step) | Mean-native / merged |
|--:|--:|--:|--:|--:|
| 1 | 21.5963 | 21.6663 | 21.6005 | 0.997x |
| 8 | 30.2659 | 29.8620 | 30.2634 | 1.013x |

All 736 per-rank/layer corrected expert-set comparisons passed. The native
versus merged 32-token greedy continuations **were not identical** (the
first B1 divergence was at output index 9); passing small numerical errors
does not establish prediction equivalence. Private-scratch checks found
0 native versus 92 candidate front writes on each of the eight ranks.
This version adds **5.506 GiB per rank at B1 / 5.524 GiB at B8**, yields no B1 gain and only about
1.3% B8 gain, so it also remains experimental rather than a default.
Sources: [`oct7-moe-front-routed-trained-b1.json`](../kimi-k3-kda-upstream/oct7-moe-front-routed-trained-b1.json),
[`oct7-moe-front-routed-trained-b8.json`](../kimi-k3-kda-upstream/oct7-moe-front-routed-trained-b8.json).

For future checks, the scratch canary now poisons buffers **immediately
before the first untimed decode graph replay**, rather than before prefill:
a small prefill fragment cannot produce a false-positive graph check.
`--validation-only` checks capture and 32-token greedy equality without
repeating the 1,025-step measurements. No additional GPU work enters the
measured graphs.

Reproduce a trained comparison with an already resident full K3 daemon:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_moe_front_decode \
  --checkpoint /path/to/Kimi-K3 --weight-cache-id YOUR_RESIDENT_CACHE_ID \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --mode two-tier --front routed --length 16384 --batches 8 \
  --decode-tokens 1026 --output moe-front-routed-b8.json
```

Run B1 in its own engine with `--batches 1`; the synchronized admission
barrier is engine-local. `--front all` selects the all-three version.
Compilation artifacts stay on local disk. These experimental commands
do not replace the official decode panel or alter production defaults.

#### Earlier shared-expert enqueue, without weight packing

`--front early` keeps every native projection and SiTU unchanged, but
starts the native shared-expert auxiliary stream before router / routed
down-projection. The native runner's subsequent enqueue reuses that same
invocation; its existing wait, output buffer, and output-tail reductions
remain unchanged. There are no new weights, device buffers, or kernels.
If the backend does not support that auxiliary-stream path, the benchmark
rejects the candidate rather than reporting a no-op as an optimization.

The official-geometry two-layer fixture passed with zero MoE output
disagreement and identical 32-token greedy continuations. Native /
candidate / native repeat were 1.0350 / 0.9998 / 1.0066 ms per B8 step;
the control drift makes that small fixture gain inconclusive.

On the trained full model with the same 16K, G8-KDA, teacher-forced
1,025-step/four-update protocol:

| Batch | Native (ms/step) | Earlier shared enqueue (ms/step) | Native repeat (ms/step) | Mean-native / candidate |
|--:|--:|--:|--:|--:|
| 1 | 21.6153 | 21.9933 | 21.6131 | 0.983x |
| 8 | 30.2609 | 30.5775 | 30.2661 | 0.990x |

Every layer on every rank recorded native early enqueues while constructing
the distinct candidate graph; all measured serving steps replayed their
selected graph. This scheduling-only variant is about 1.8% slower at B1 and
1.0% slower at B8. The first/last standalone trained MoE comparisons have
zero output disagreement, but the B1 greedy continuation differs at index 9.
The eight B8 greedy continuations are all identical between native and early.
Native continuations from the two separate trained experiments also differ
at index 9, so greedy inequality alone cannot be attributed to the candidate
or interpreted as a measured quality regression. No quality gain is claimed.

**Decision:** keep the native production front. Neither merged projection
variant delivers a worthwhile memory/performance tradeoff, and moving
shared work earlier does not improve trained-model latency. The proposed
resource-contention explanation for the scheduling regression is an
inference, not an independently profiled cause. The adapters remain isolated
benchmark experiments; no official decode table or production option changed.

Sources: [`oct7-moe-front-early-shared-fixture.json`](../kimi-k3-kda-upstream/oct7-moe-front-early-shared-fixture.json),
[`oct7-moe-front-early-shared-trained-b1.json`](../kimi-k3-kda-upstream/oct7-moe-front-early-shared-trained-b1.json),
[`oct7-moe-front-early-shared-trained-b8.json`](../kimi-k3-kda-upstream/oct7-moe-front-early-shared-trained-b8.json).

The adapter, graph-switch/canary/cadence, and related upstream/launcher CPU
suite passes **25 tests**. The fixture and both trained early-enqueue jobs
must also report `status: complete` in the raw JSON to establish that the
untimed checks completed; the all-three files above deliberately retain
their profiler-failure status.

## October 7: actual B8/2K prefill query-tile follow-up

The earlier query-tile probe used the generic fused CK operator. This follow-up
uses the current score-only 64-key candidate emitter, 1,024-query-block route
packing, exact top-eight refinement, and unchanged coarse attention. It tests
both the ordinary twelve-head projection group and the memory-saving six-head
group on 2,048 query rows and 4,096 centroid keys. Seeded kernel inputs are
geometry/correctness evidence, not trained-model quality or serving latency.

An initial same-process comparison exposed an AITER registration collision:
`torch_compile_guard` reuses operators by the Python function name, while all
subtile specializations were named `subtile_mha`. Changing the generated shared
library name alone did not change the operator actually called. The q64 arm
therefore silently called q128. `subtile_factory` now includes query tile,
key-subtile width, score-only mode and maximum-reuse mode in the registration
name. No score formula or default tile changes. CPU regressions verify that
the two factories reach AITER with distinct names. The valid GPU run logs
separately import the q128 and q64 libraries; q64 compilation is untimed and
uses local `/tmp` storage.

Do not use `oct7-serving-subtile-query-tile-head12.json`,
`oct7-serving-subtile-chunkpack-query-tile-head12.json`, or
`oct7-serving-subtile-chunkpack-query-tile-head6.json` as tile comparisons:
their apparent equality was the dispatch collision. The initial trained job
`21445-k3-owner-real-q64-prefill` was stopped before making this comparison;
its incomplete artifact is not a benchmark result.
The first distinct-operator trained retry (`21448`) also has no timing: the
new prefill-only harness omitted the existing eight-request admission barrier.
The native scheduler admitted a single row and, with no competing request,
used the full 16,392-token budget rather than a 2K slice. The owner boundary
guard correctly rejected it. The harness now waits for all eight requests
before its first scheduler turn, just like the established fixed-cohort
benchmark. This changes neither the per-row chunk nor either update cadence;
CPU coverage checks that the barrier is always configured.
The admission-fixed attempt (`21449`) failed during engine memory profiling,
before warmup: node 4 had additional roughly 33 GiB allocations on seven
GPUs beyond the resident weight daemon. It has no serving measurements.
The same-engine comparison was moved to node 2, whose already-loaded full
weights occupy approximately 83% VRAM without those additional allocations;
no weight re-materialization or timing-policy change is involved.

Valid isolated graph-replay measurements, average of two short timing blocks:

| Scored heads (isolated controls) | q128 coarse only (ms) | q64 coarse only (ms) | q128 with exact routing (ms) | q64 with exact routing (ms) | Whole-stage speedup |
|--:|--:|--:|--:|--:|--:|
| 12 | 0.3741 | 0.2473 | 0.5133 | 0.3946 | 1.30x |
| 6 | 0.3632 | 0.2251 | 0.4442 | 0.3083 | 1.44x |
| 96 (observed owner geometry) | 1.0218 | 0.8444 | 1.7746 | 1.5942 | 1.11x |

The 12/6-head controls use 4,096 centroids; the observed 96-head owner case
uses 2,048. These are distinct geometry controls, not head-count comparisons.

Coarse output, LSE, and route sets are bitwise identical on both geometries.
Independent FP32 output/LSE/global-top-eight checks also pass; all variants
have the same approximately 0.61% output RMS error versus FP32. Sources:
[`oct7-serving-subtile-unique-op-head12.json`](../kimi-k3-kda-upstream/oct7-serving-subtile-unique-op-head12.json),
[`oct7-serving-subtile-unique-op-head6.json`](../kimi-k3-kda-upstream/oct7-serving-subtile-unique-op-head6.json),
[`oct7-serving-subtile-owner96-states2048.json`](../kimi-k3-kda-upstream/oct7-serving-subtile-owner96-states2048.json).

The trained full-model follow-up uses one resident engine and one warmed
serving measurement per arm, frozen ProLong prompts, B8 request-owner layout,
eight 2K scheduler slices, twelve-head projection groups, and the approved G8
direct-state KDA baseline in both arms. Only the query tile changes. Dispatch
instrumentation records actual query/head/state shapes during warmup and is
removed before serving measurement. First-token equality is a canary, not a
replacement for a corpus loss evaluation. Production retains q128 until the
complete serving comparison supports a change.
The serving audit records `queries=2048,heads=96,states=2048`: the twelve-head
setting controls leaf projection, not coarse scoring. The isolated 12/6-head
controls above therefore do not by themselves quantify request-owner kernel
speed. An additional isolated test uses the observed 96-head/2,048-centroid
geometry; the full trained comparison is authoritative for serving latency.

Completed trained comparison (`21453`, node 2):

| Full K3, TP8/DCP8/EP8, row-per-GPU B8, 32K per prompt | Warmed prefill (s) | First tokens versus q128 |
|:--|--:|:--|
| 128 query rows per CK tile (current) | 33.779946 | Baseline |
| 64 query rows per CK tile | 33.781156 | Identical, 8/8 |

Both warmup audits record 192 coarse calls per rank, each with 2,048 query
rows, 96 heads and 2,048 state entries; the q64 arm separately imports its
own compiled operator. No audit wrapper runs in either measured arm. All
eight request timings are retained in the raw result. The serving difference
is only 0.0036%, so this single warmed comparison shows **no end-to-end gain**
despite the 1.11x isolated stage gain. Keep the q128 production default.
Source: [`oct7-owner-prefill-query64-trained-32k-node2.json`](oct7-owner-prefill-query64-trained-32k-node2.json).
The operator-dispatch, benchmark-corpus/admission and runtime regressions
pass: **19 tests**. No decode settings or cadence were changed by this test.

Reproduction (no weight reload between arms):

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_centroid_tiling_probe \
  --serving-subtile --queries 2048 --heads 12 --output serving-tile-head12.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_centroid_tiling_probe \
  --serving-subtile --queries 2048 --heads 6 --output serving-tile-head6.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_centroid_tiling_probe \
  --serving-subtile --queries 2048 --heads 96 --states 2048 --output serving-tile-owner96.json
TRITON_CACHE_AUTOTUNING=1 VLLM_USE_TRITON_AWQ=1 \
  AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv" \
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_owner_tile_prefill \
  --checkpoint "$CHECKPOINT" --weight-cache-id "$WEIGHT_CACHE_ID" \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --lengths 32768 --output owner-tile-prefill.json
```

For decode, the next low-overhead candidate is enabling the existing fused
global-top-eight/centroid-union builder for K3's **request-owner** compact
descriptor frontier. Ordinary DCP must still perform global distributed
selection before building its local union; fusion must not publish a union
from each rank's unmerged candidates.
It is currently disabled there, so selection, union formation and descriptor
publication are three serial launches. The most recent owner fixture trace
measures about 4.40/4.37/4.40 us per layer for these three kernels, versus
31.97 us for compact attention and 17.05 us for centroid scoring. These are
instrumented durations, not additive wall-time savings. Any fusion must keep
the exact selected union, closed-centroid mass, epoch stamps, graph replay,
global-256 updates and coarse replacement intact; it is not enabled here.

The larger decode target is the compact attention consumer. The current
request-owner launch has six 16-head work tiles and 32 key splits (192
workgroups), and its 512-channel latent output makes register pressure
important. A finer head or output-channel work partition is worth testing
against identical real selected-KV descriptors, not assuming that more
key splits are better: the earlier 64-split test already lost to 32 splits.
Preserve the current union and exact attention calculation. Smaller head
tiles can duplicate KV loads and waste the 16-head MFMA tile; splitting
output channels can duplicate QK work, so neither is a demonstrated gain.

Another, more involved candidate is avoiding the second centroid-QK pass.
The current router scores centroids, while the compact attention consumer
scores the unexpanded centroids again. Retaining coarse output/LSE and
combining it with exact selected-centroid replacement could remove that
duplication, but the router's extra 512-channel value accumulation and a
replacement reduction might erase the saving. Benchmark the whole pipeline
and retain exact softmax accounting rather than infer speed from a removed
pass. These are follow-up proposals, not implemented or promoted results.

### Exact decode fusion and consumer candidates (October 6, 2026)

The three proposals above have now been measured separately on node 3
(MI325X/gfx942, the same pinned v10 runtime). These are fixed-input,
CUDA/HIP-graph microbenchmarks, **not full-model serving timings**. Compilation
and warmup are excluded. The graph contains 32 complete pipeline calls;
five timing samples each replay it ten times. Tables show the median time
per call. The long-replay sampling is only for these small kernel probes;
the serving fixture below uses one measured pass per arm.

Selection/union fusion retains the current exact top eight per head, the
1,024-leaf eligibility cap, and the deduplicated union across each 16-head
group. It resets union counts/epochs in the scoring kernel and atomically
builds the union in the global top-eight reduction. The following descriptor
publication kernel consumes the complete union only after producer kernel
completion. This is enabled only by an explicit private benchmark setting
on a local request-owner pool; distributed DCP must merge candidates across
ranks first. No public option or production default was changed.

| Physical batch per GPU | Active centroids | Separate route/reduce/union (us) | Fused route/reduce+union (us) |
|--:|--:|--:|--:|
| 1 | 2,048 | 13.068 | 12.133 |
| 1 | 4,096 | 19.368 | 18.534 |
| 1 | 512, tied query scores | 12.546 | 10.877 |
| 8 | 2,048 | 54.788 | 55.916 |

The B8 request-owner deployment has **physical B1 per GPU**, not the physical
B8 case in the last row. Score and index tensors are bitwise equal to the
separate implementation. Four replays with changing epoch stamps preserve
the independently deduplicated sets, including reversed request rows,
ragged state lengths, zero counts, singletons, leaf-cap boundaries and ties.
Compiler VGPR counts remain 128 in the scorer and rise by at most one in
the reducer; neither spills. Source:
[`oct7-decode-fused-union.json`](../kimi-k3-mla-stack/oct7-decode-fused-union.json).

The finer-head-work candidate divides the existing 16-head MFMA work unit
into groups of eight or four while preserving its exact selected KV set.
It increases workgroup count, but wastes more of the fixed MFMA tile and
duplicates KV reads. Complete consumer plus split-output-reduction times:

| Physical batch | Centroids | Exact 16-token pages | 16 heads/workgroup (us) | 8 heads/workgroup (us) | 4 heads/workgroup (us) |
|--:|--:|--:|--:|--:|--:|
| 1 | 2,048 | 128 | 25.042 | 41.833 | 58.854 |
| 1 | 4,096 | 256 | 33.876 | 60.137 | 86.949 |
| 1 | 4,096 | 1,024 | 62.993 | 117.601 | 172.973 |
| 8 | 2,048 | 128 | 105.041 | 190.470 | 336.803 |

All variants pass an independent FP32 attention/LSE oracle (maximum output
absolute error 0.000434; maximum LSE error 0.000000954). The current
16-head consumer is retained; subdivisions are **rejected for serving**.
Source: [`oct7-decode-consumer-work-heads.json`](../kimi-k3-mla-stack/oct7-decode-consumer-work-heads.json).

The third probe uses the existing value-producing router and its complete
coarse-output/LSE/top-eight reduction, before any exact-leaf attention,
union formation, replacement correction or communication:

| Physical batch | Centroids | Retained coarse output + selection (us) |
|--:|--:|--:|
| 1 | 2,048 | 30.830 |
| 1 | 4,096 | 61.227 |
| 1 | 512, tied query scores | 26.431 |
| 8 | 2,048 | 154.841 |

Retaining values increases the scorer from 128 to **476 VGPRs**, with
**137 SGPR spills**. For example, 61.227 us at physical B1/4,096 centroids
already exceeds the separate score/union stage plus the full compact
consumer in the nearby 4,096-centroid/256-page control (19.368 + 33.876 us),
before the retained-coarse variant has done any leaf attention. These
controls have distinct synthetic inputs; this is a screening result, not
an additive serving latency estimate. Its output/LSE pass an independent
FP32 reference, and its route tensors match the score-only path exactly.
Further integration of **this existing retain-value implementation** is
rejected; a redesigned lower-register-pressure kernel remains a distinct
future possibility. Source:
[`oct7-decode-retain-coarse.json`](../kimi-k3-mla-stack/oct7-decode-retain-coarse.json).

GPU regression checks: **45 passed** before cleanup, covering owner
lifecycle/global cadence, fixture timing/audits and subdivision
tail-head/reference cases. The rejected subdivision has been removed from
the production kernel and orchestration; its reproducible copy lives only
in `benchmarks/experimental/kimi_decode_work_heads.py`. The production
consumer is back to its original fixed 16-head code, not a dormant branch.
The post-cleanup GPU rerun (`21466`) passes **46 tests**, including the
one-shot oracle's replay restoration/no-double-append behavior and all four
experimental subdivision reference cases; no serving regression was found.

The captured owner-fixture comparison is complete (`21465`, node 3):

| Twelve MLA layers, B8 one-request-per-GPU, 64K prompt | Warmed decode (ms per batch step) |
|:--|--:|
| Separate top-eight selection and union | 2.267567 |
| Exact fused top-eight selection/union | 2.216965 |

The fused arm is **2.23% lower latency (1.0228x)** in this fixture, with one
measured pass per arm and no statistical-significance claim. Both use the
same engine/weights, separately captured graphs, fixed continuations and
1,025 decode replays, including four global-256 updates in every layer/rank.
The capture audit observes fused scorer/reducer launches and **no dedicated
union launch** on all eight ranks. Untimed live-cache checks cover all
12 layers/rank: union sets and counts match, LSE is within the oracle
tolerance, and the maximum output difference is 4.66e-10. Numerical hooks
remove themselves before the measured generation; no profiler is used.
Source: [`oct7-decode-union-fixture.json`](../kimi-k3-mla-stack/oct7-decode-union-fixture.json).

This was a small attention-stack gain, **not a measured trained-full-K3
gain**: the fixture omits MoE and KDA. Fusion remained private until the
following trained comparison; the two substantially losing alternatives
were not promoted or used to trigger a full-model sweep.

Reproduction (no full-model weights or proprietary scheduler required):

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_candidates \
  --kind union --output decode-fused-union.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_candidates \
  --kind consumer --output decode-consumer-work-heads.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_candidates \
  --kind coarse --output decode-retain-coarse.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_union_fixture \
  --layers 12 --length 65536 --output decode-union-fixture.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m pytest -q \
  tests/test_kimi_decode_candidates.py tests/test_kimi_owner_decode.py \
  tests/test_kimi_decode_fixture.py
```

The first three commands and GPU tests need one gfx942 GPU. The captured
serving fixture needs eight GPUs, and uses dummy weights: it is dispatch,
numerical-equivalence, update-cadence and relative-speed evidence only,
**not trained-model prediction quality**. Raw successful run IDs: 21462
(union), 21461 (consumer), 21463 (retained coarse), 21464/21466 (tests)
and 21465 (captured serving fixture).

## October 7 trained union comparison and promotion

The identical frozen B8 cohorts and 1,026-token real ProLong continuations
were replayed in one resident trained K3 engine. Both arms use the approved
G8 direct-state-I/O KDA prefill baseline and packed INT4 MoE weights; each
arm has its own audited model graph and exact-shape warmup.

| Context | Separate union (ms/step) | Fused union (ms/step) | Latency reduction |
|--:|--:|--:|--:|
| 16K | 30.216883 | 30.161214 | 0.184% |
| 64K | 30.930980 | 30.897010 | 0.110% |

All eight ranks execute 1,025 measured graph replays and four global-256
updates in every one of 24 MLA layers. All live-cache selected sets/counts
match exactly; output/LSE equivalence passes floating-point tolerances, not
bitwise identity. The serving gain is small and is not relabeled as the
fixture's 2.23% gain. The safe physical-B1 owner path now defaults to fusion;
ordinary distributed B1 still merges global candidates before deduplication.
Neither the chosen top-eight KV set nor the 1,024-leaf opening cap changes.

Sources: [trained 16K](oct7-trained-decode-union-b8-16k.json) (21468),
[trained 64K](oct7-trained-decode-union-b8-64k.json) (21472).
Reproduce with the fixture command above, adding `--checkpoint`,
`--weight-cache-id`, `--real-token-cache`, and `--reference-baseline` for the
matching archived dense cohort. No weight reload is required between arms.

Post-promotion GPU regressions (21484, node 3): **99 passed**. The original
bitwise storage/mean-reuse comparison retains separate union ordering to
isolate that change; fused union is checked independently for exact sets and
numerical output/LSE agreement. Eager-versus-captured owner attention now
tests both variants. Exact projection grouping at 12, 6, 4 and 2 heads also
passes bitwise output/LSE checks, including compact selected-leaf projection.
All warmup-only membership hooks are removed before measured generation.

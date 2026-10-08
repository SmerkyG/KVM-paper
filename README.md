# LoD Attention

LoD Attention is exact for selected high-mass regions and approximate for the
low-mass remainder. It represents remote context with count-corrected semantic
centroids, refines the eight best regions with exact leaves, and combines those
results with an exact local window and protected sink through log-sum-exp.

This branch is the minimal inference release for the LoD Attention paper. It
contains one fixed production policy, not the research-time tuning matrix.

## Kimi K3 development branch

The local `lod-k3` branch additionally develops absorbed-MLA LoD for Kimi K3.
Its [October 7 prefill/decode sweep](results/kimi-k3-full-model-current/CURRENT_TIMINGS.md),
[benchmark instructions](results/kimi-k3-full-model-current/README.md), and
[1020K memory work](results/kimi-k3-full-model-current/LONG_CONTEXT_MEMORY.md)
are separate from the Qwen/K2 paper-release results below. B1 generation is
validated through 1020K; B8 is validated through 512K. The 1020K/B8 capacity
tests have not completed generation. This branch has not been pushed.

The latest [live-split and allocator fixes](results/kimi-k3-full-model-current/LIVE_SPLITS_ALLOCATOR.md)
decouple dense decode partitioning from maximum context reservation and keep
registered graph communication while retaining expandable large-context
memory. The current sweep above includes only measurements with these fixes;
remaining cells are filled by fresh matched reruns, not older controls.

### GLM5.3-Flash experimental port

This branch also contains an experimental vLLM-only LoD port for
`zai-org/GLM-5.3-Flash`, following its
[vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash). The model's FP8
FFN weights remain FP8; its attention projections and LoD cache remain BF16.
LoD stores exactly the 512-channel latent (no dummy RoPE channels). Prefill
projects centroid means to per-head K256/V256 in one cached, concatenated-weight
GEMM, and uses fused AITER routing/coarse attention. Exact leaves and decode
still use absorbed queries; refined latent values are reduced across routes
before their value projection. GLM now supports two-tier BF16 and three-tier
BF16/INT4, including top-two-page decode. DCP and HF are not yet supported
for GLM. The native comparison retains GLM's learned sparse indexer with its
2,048-token budget; it is **not** all-history dense attention.

`benchmarks/glm53_flash_fixture.py` exercises three KDA layers plus one MLA
layer, retaining GLM's attention geometry and mHC. Vocabulary and FFN width
are reduced and weights are randomly initialized. These are integration and
kernel-correctness tests, **not trained-model quality or full-model timings**.
Raw results are under `results/glm53-flash-fixture/`. The AMD image used here
needs compatibility repairs for FlyDSL shift constants, native indexer page
addressing, and a non-tile-aligned indexer output workspace. No selection rule
is changed by these repairs.

The [trained-model 64K TP4 comparison](results/glm53-flash-full-model/README.md)
uses the full FP8 checkpoint held in a local weight daemon, real ProLong
prompts and 1,025 measured decode steps with full decode graphs. Its LoD
older timings predate the October 8 query-layout repair and are retained as
historical provenance, **not current corrected-kernel speed claims**.
Native sparse timings are unaffected. The corrected B1 rerun gives
**3.132s prefill / 8.519ms decode**, versus native 3.156s / 12.898ms at 64K:
prefill is effectively tied and decode is 1.51x faster. The matched B8 rerun
gives **25.110s cohort prefill / 13.762ms decode**, versus native 25.283s /
17.856ms: prefill is again tied and decode is 1.30x faster. The matched
eight-document ProLong check (65,199 tokens per document, raw text) gives
perplexity **1.4885 LoD versus 1.5617 native sparse**, 4.69% lower on this
cohort. A 16-example, chat-templated LongBench smoke panel scores **9/16 in
both modes**, with identical per-example correctness. This limited 64K
panel is not a full LongBench result; all hashes and methods are in the report.

An opt-in projected-leaf prefill experiment (`LOD_GLM_PROJECTED_LEAVES=1`)
projects selected latent leaves once to K256/V256, then reuses them across
routed queries. After tile/counter tuning, the complete refinement stage is
1.64–1.88x faster in kernel tests, but trained-model TP4/B8 64K prefill improves
only **25.110s → 24.729s (1.52%)**; decode remains **13.762ms**. Its matched
quality checks give PPL **1.48815**, LongBench smoke **11/16**, and NIAH-S3
**8/8**. This is not enabled by default; see the
[experiment and exact commands](results/glm53-flash-full-model/README.md#projected-exact-leaf-prefill-experiment-october-8).

A further [Kimi optimization audit](results/glm53-flash-full-model/README.md#kimi-optimization-audit-and-follow-up-ports-october-8)
ports sorted ordinals to default latent leaves and avoids unused query
absorption/copies. An additional opt-in `LOD_GLM_KDA_PREFILL=1` reuses Kimi's
G8 Gluon KDA arithmetic, preserving GLM's already-activated beta. With both
KDA and projected leaves enabled, the matched TP4/B8/64K result is **23.632 s
prefill / 13.745 ms decode**, versus the prior LoD default **25.110 s /
13.762 ms**. The same KDA change improves native prefill to **24.457 s**;
it is not a LoD-only comparator advantage. Matched quality gives ProLong PPL
**1.487881**, LongBench smoke **11/16**, and NIAH-S3 **8/8**: no aggregate
regression on these small panels, though LongBench swaps one correct answer
for another. KDA/projected leaves remain opt-in; query/latent-dispatch fixes
are enabled by default.

A [128K/256K TP4/B8 speed follow-up](results/glm53-flash-full-model/README.md#updated-longer-context-speed-sweep)
measures LoD **50.116 / 106.586 s prefill** and **14.093 / 14.565 ms decode**,
versus updated native sparse **51.092 / 109.468 s** and **17.713 / 18.172 ms**.
Thus prefill is only 1.9–2.6% shorter; decode is 1.25–1.26x faster. Both lengths
fit with matched 32 GiB native KV reservations, eight live requests and four
global-256 decode updates per request/layer. These new points test speed, not
quality at longer lengths.

The new [MLA top-two-page port](results/mla-three-tier-top2/README.md) supports
three-tier BF16/INT4 on GLM and full-width Kimi, retaining all-leaves prefill.
GLM TP4/B8 at 64K measures **23.743s / 14.408ms** for BF16 and
**24.272s / 14.137ms** for INT4 (prefill / decode). Kimi's
128K MLA-only fixture measures **2.478s / 1.442ms** BF16 and
**2.538s / 1.342ms** INT4 versus dense Gluon **4.055s / 1.804ms**.
Graph/DCP fixture checks pass; these are not full-K3 serving measurements.
The corrected trained GLM three-tier BF16 and INT4 paths both score **8/8**
on matched 64K NIAH-S3, with ProLong PPL **1.488252 / 1.488075** versus
two-tier **1.487881**. LongBench smoke gives **11/16 / 9/16** respectively.
The initial 0/8 was an output-layout bug: GLM's transposed
query layout propagated to its output, while the new final reducer assumed
contiguous storage. Explicit output strides and contiguous adapter scratch
fix it without changing page selection. Shared K/V quantization and summary
writers also now process the single latent record once. The report records
the fixes and independent-reference regression tests.

A separate matched **eight-example NIAH-S3 64K smoke test** scores native
sparse 8/8 (100%), exact all-history attention 8/8 (100%), and corrected
two-tier BF16 LoD **8/8 (100%)**. The repair changes no clustering, routing,
top-eight choices, leaf cap, update cadence, or weights: it packs transposed
queries before the exact-leaf kernel, which reads contiguous head-major rows.
GLM's projected-prefill shortcut had bypassed the common core's preparation.
The former 4/8 result and subsequent uncapped/projected-clustering ablations
used the defective path and cannot establish the quality of those policies.

The trained-tensor check covers all 11 MLA layers and all 64 query heads,
using sixteen final-prefill queries from one real request. Exposing every
remote leaf now matches the dense FP32 reference with **0.431% mean relative
L2 error** (BF16 attention/projection rounding); counts and directory coverage
are exact. The reproduced normal output agrees with its captured serving
output to 0.000451% mean relative L2. Four targeted GPU tests pass, including
transposed/cropped-query dense equivalence. These remain small diagnostics,
not comprehensive ProLong/LongBench quality validation. The linked report
includes raw answers, route-oracle comparisons, reproduction commands and
the native mixed-prefill/decode scheduling workaround.

To reproduce in an installed compatible AMD vLLM environment, keep compiler
caches on local disk, install this repository's vLLM plugin, then run:

```bash
python -m pytest -q tests/test_glm53_flash_lod.py
python -m benchmarks.glm53_flash_fixture --mode full --length 32768 \
  --decode-tokens 1026 --output results/glm53-flash-fixture/native-prefix-32k-control.json
python -m benchmarks.glm53_flash_fixture --mode two-tier --length 32768 \
  --decode-tokens 1026 --output results/glm53-flash-fixture/lod-scalarleaf-32k-current.json
```

The fixture uses one exact-shape untimed warmup, one measured pass, eager
execution, and 1,025 decode steps so four 256-token update boundaries are
included. Its FP8 parameter audit and finite-logprob checks fail explicitly
if the intended execution path is not usable.

The last complete TP1, B1 four-layer fixture panel **before projected
centroids** is retained below as the control, not a new-code measurement:

| Prompt | Method | Prefill | Decode per step |
|---|---|---:|---:|
| 32K | Native learned sparse attention | 0.2345 s | 4.4442 ms |
| 32K | Two-tier LoD, exact initial chunk | 0.2607 s | 3.8257 ms |
| 64K | Native learned sparse attention | 0.4748 s | 4.4393 ms |
| 64K | Two-tier LoD, exact initial chunk | 0.6014 s | 3.7963 ms |

LoD prefill remains 1.11× slower at 32K and 1.27× slower at 64K; decode is
1.16× and 1.17× faster respectively. All four use the default 512-wide
fixture FFN. The latest query-tiled Gluon route/coarse kernel uses 64×64 tiles
and exact max reductions rather than sorting each score tile. It terminates
a tile's selection loop only when no remaining candidate can replace the
current eighth-best route; attention still includes every coarse centroid.
It preserves query normalization, FP32 score/ascending-ID tie ordering, the
top-eight policy and update cadence. The routing microbenchmark improves from
10.65 to 7.49 ms; this is a kernel diagnostic, not a model-latency speedup.
Model prefill improves from the previous 0.2871 to 0.2680 s at 32K, and from
the intermediate Gluon selector's 0.6894 to 0.6309 s at 64K. The latest leaf
change resolves the page directory once per 16-token page in the existing
expert kernel; it further reduces these to 0.2607 and 0.6014 s. The goal of
beating native prefill is **not yet met**. The matched initial-16K ablation
below already shows exact attention is faster than native sparse attention
there; the native indexer budget is not an explanation for that initial chunk.

Both decode measurements include 1,025 steps and therefore four 256-token
LoD update boundaries. Sources are
[32K native](results/glm53-flash-fixture/native-prefix-32k-control.json),
[32K LoD](results/glm53-flash-fixture/lod-scalarleaf-32k-current.json),
[64K native](results/glm53-flash-fixture/native-64k-tp1-current.json), and
[64K LoD](results/glm53-flash-fixture/lod-scalarleaf-64k-current.json).
The latest LoD trials had unrelated workloads on other GPUs of the same node;
the native controls are reused, unchanged measurements. Do not attribute the
small decode-timing change to the page-lookup optimization: it changes only
prefill leaf refinement. The matched page-lookup control is recorded below.
The selector diagnostic is
[here](results/glm53-flash-fixture/gluon-route-early2.json).
Replace `--length 32768` with `--length 65536` for 64K. The earlier seven-step
smokes do not amortize updates and are not the decode comparison. These
eager-mode, random-weight fixture timings are not full-model performance.

#### Projected-centroid prefill and Kimi optimization audit

The new projected path keeps the persistent cache unchanged. One shared
latent mean is prepared per slot, then a single GEMM with concatenated
`[W_UK, W_UV]` columns produces strided K256/V256 views. A source-derived
AITER kernel computes centroid attention and emits native-tile top-eight
candidates in the same pass. Its small candidate reduction obtains the exact
global top-eight, and only afterward closes selected slots above 1,024 leaves.
The final LSE merge removes those selected summaries and includes the exact
refinement, local field, and separate sink. Decode stays latent512.

The projection-inclusive coarse diagnostic improves as follows. Each cell
uses the same random operands, TP1/64 query heads/16K query rows, one untimed
warmup and one measured GPU interval. This is **not model speed or trained
quality evidence**:

| Centroids | Absorbed512 coarse | Projected256 coarse | Speedup |
|---|---:|---:|---:|
| 2,048 | 27.67 ms | 15.15 ms | 1.83× |
| 4,096 | 48.91 ms | 27.76 ms | 1.76× |

Raw data: [matched coarse diagnostic](results/glm53-flash-fixture/projected-coarse-matched.json).
The first 32K whole-fixture prefill is 0.2560 s with projected centroids, and
0.2501 s when the exact local field also uses projected K/V and AITER.
The old 0.2607 s control is reused for that short comparison, so its small
difference is provisional. The **64K comparison below is fresh and sequential
on node 2 GPU 1**, with one untimed warmup and one measured generation per
method, including 1,025 decode steps / four update boundaries:

| 64K TP1/B1 four-layer fixture | Prefill | Decode per step |
|---|---:|---:|
| Absorbed512 control | 0.6036 s | 3.7673 ms |
| Projected centroids; latent local | 0.5667 s | 3.6344 ms |
| Projected centroids and local (new default) | 0.5126 s | 3.6757 ms |

Projecting the centroids alone reduces prefill by 6.1%; including the local
projection reduces it by **15.1% (1.18× faster)**. It is still 8.0% slower than
the reused native sparse 64K control, 0.4748 s, so the native-prefill crossover
goal is not yet met. Decode code is unchanged; do not attribute its small
variation to a prefill kernel optimization. Raw controls are
[absorbed](results/glm53-flash-fixture/projected-control-b1-65536.json),
[projected coarse](results/glm53-flash-fixture/projected-b1-65536.json), and
[projected coarse/local](results/glm53-flash-fixture/projected-local-b1-65536.json).
Both projections keep temporary K/V separate from the persistent latent cache.
The B8/32K nine-token smoke also passes with mixed cached-prefill/decode rows,
finite logprobs and zero preemptions; its eight decode steps are **not** an
update-amortized decode benchmark.

| Kimi optimization | GLM status |
|---|---|
| Projected coarse summaries; fused route/coarse; compact tile candidates | Ported to K256/V256; no duplicate centroid-scoring pass |
| Immutable projection-weight layouts | Cached per layer/source, including source lifetime/version checks; K/V GEMMs combined |
| Zero-copy token/head layout views; 128-key tile padding | Ported; no power-of-two state padding or separate projected K/V copies |
| Separate sink; exact LSE replacement; global 16K/256 update cadence | Preserved; no change to ranking, cap, or cache membership |
| Local/coarse overlap; shared transient scratch; deferred cross-layer construction | Already shared by GLM; new weight packing is on the foreground before either stream can read it |
| Project exact local field through native AITER | Implemented, FP32-reference checked and promoted after the matched speed test |
| Reduce eight exact routes before the value projection | Applied; GLM's exact-leaf search remains latent512, so the full archive is not expanded |
| Compact selected-leaf projection / projected per-head leaf experts | Not yet ported: Kimi's implementation assumes K192/V128 and direct64; this is a remaining GLM refinement candidate, not claimed as inherited |
| G8 direct-state-I/O KDA prefill | Not silently inherited: the Kimi hook targets its AMD KDA module; GLM's caller supplies already-sigmoided beta and a different state-I/O interface. Native GLM KDA is unchanged in these controls |
| Request-per-GPU / DCP layouts, MoE INT4, Kimi-specific 512+64 kernels | Not applicable to the current TP1 NoPE GLM fixture; no model/backend substitution |

Validation: the targeted GPU run passes **58 tests**, including projected
coarse output/LSE/top-eight, ragged 128-key padding, post-rank closing, causal
local offset, separate-sink refinement, adapter algebra and Kimi projected
prefill regressions. The CPU benchmark/config/adapter suite passes **93 tests**
(23 GPU/runtime cases skipped). These are correctness checks, not trained GLM
ProLong/NIAH/LongBench evaluation.

```bash
python -m pytest -q tests/test_glm53_projected_prefill.py tests/test_glm53_flash_lod.py
python -m benchmarks.glm53_projected_coarse \
  --output results/glm53-flash-fixture/projected-coarse-matched.json
python -m benchmarks.glm53_flash_fixture --mode two-tier --length 65536 \
  --decode-tokens 1026 --absorbed-coarse \
  --output results/glm53-flash-fixture/projected-control-b1-65536.json
python -m benchmarks.glm53_flash_fixture --mode two-tier --length 65536 \
  --decode-tokens 1026 --latent-local \
  --output results/glm53-flash-fixture/projected-b1-65536.json
python -m benchmarks.glm53_flash_fixture --mode two-tier --length 65536 \
  --decode-tokens 1026 \
  --output results/glm53-flash-fixture/projected-local-b1-65536.json
```

The matched **TP1/B8 prefill-only** sweep does not improve LoD's relative
performance with the existing 16K total scheduler budget:

| Prompt per request | Native, eight-request prefill | LoD, eight-request prefill | LoD / native |
|---|---:|---:|---:|
| 32K | 1.8209 s | 2.0879 s | 1.147× |
| 64K | 3.7254 s | 4.8091 s | 1.291× |

These are the interval from the first scheduled request to the last request's
first token, **not per-request latency**. LoD throughput is 125,557 tokens/s
at 32K and 109,019 tokens/s at 64K, essentially unchanged from B1. First-token
timestamps show that the scheduler still processes the rows mostly one at a
time: eight requests do not imply eight simultaneous 16K prefill chunks.
Testing occupancy from true concurrent prefill would require shorter
per-request chunks or a larger *matched* total token budget; this sweep changes
neither chunking nor the LoD calculation.

Both modes use a 2 GiB native-cache reservation to avoid capacity preemption,
one exact-shape untimed warmup, one measured eager pass and one generated
token. The native 32K control has ample reserved capacity and no logged
preemptions; all other trials explicitly assert zero request preemptions.
No decode speed or trained-model quality conclusion follows from this sweep.
All runs use node 4 GPU 4 sequentially; unrelated work remains on other GPUs.
Raw files are `b8-prefill-{full,two-tier}-{32768,65536}-current.json` under
`results/glm53-flash-fixture/`.

The first B8 LoD attempt exposed an archive-row selection bug in cross-layer
cached-prefill construction: a one-row centroid update read all eight reserved
BF16 archive rows. The fix uses the same active-row view as the state pack,
without an allocation or extra GPU kernel. Three CPU regression cases cover
rows 0, 3 and 7 across two layers and verify that inactive rows stay unchanged;
the restarted B8 runs above complete successfully. The targeted regression
plus benchmark/config checks pass 71 CPU tests.

```bash
for length in 32768 65536; do
  for mode in full two-tier; do
    python -m benchmarks.glm53_flash_fixture --mode "$mode" --length "$length" \
      --batch-size 8 --kv-cache-mib 2048 --decode-tokens 1 \
      --output "results/glm53-flash-fixture/b8-prefill-${mode}-${length}-current.json"
  done
done
```

Earlier TP4/B1 results (before this TP1-only routing optimization), with a
1,024-wide FFN, were 0.1203 s / 4.8688 ms for native and 0.1313 s / 4.1187 ms
for LoD at 32K. Raw files are `native-prefix-tp4-32k.json` and
`lod-gluon-exact-default-tp4-32k.json`; they are not current-kernel TP4 results.
The routine GLM suite now contains 44 tests, only 19 of which execute GPU
kernels. A 180-case Cartesian Kimi GPU sweep was reduced to six representative
boundary/layout cases, and three redundant GLM GPU combinations were removed.
The page-lookup change passes one targeted FP32 GPU reference test covering
flat, two-level and hashed directories, plus 92 CPU checks (20 GPU/runtime
cases skipped). The broad Kimi and benchmark GPU suite is no longer required
for each GLM optimization iteration.

#### Isolating routing and refinement

The diagnostic `--no-exact-prefix` flag disables only the initial 16K exact
bypass in this fixture; it does not change the default serving policy. At 16K
it retains the ordinary 512-token exact front and runs remote attention for
the other 15,872 queries. With 16K state catch-up there are only 255 initial
singleton regions at that point, so it is **not** a representative large-state
refinement benchmark. Profile shapes are recorded explicitly to expose this.

For a representative short test, `glm53_remote_prefill` builds a 16K LoD
history before timing 16K independent queries against that cache, with no
exact-local attention or overlapping stream. It separately times route/coarse
and leaf refinement, retaining top-eight and the 1,024-leaf closing rule.
Inputs are random; neither diagnostic is trained-model quality evidence or
an end-to-end performance comparison.

```bash
python -m benchmarks.glm53_flash_fixture --mode two-tier --no-exact-prefix \
  --length 16384 --decode-tokens 1 --profile-prefill \
  --output results/glm53-flash-fixture/lod-noexact-16k-profile.json
python -m benchmarks.glm53_remote_prefill --compare-page-lookups --profile-refinement \
  --output results/glm53-flash-fixture/remote16k-scalar-current.json
```

The correctly aliased shared-latent probe has 2,048 centroids. Route/coarse
takes about 23 ms and route-list construction about 0.6 ms. Repeated per-leaf
directory lookup costs about 9.9 ms in the refinement kernel; scalar
per-page lookup reduces it to about 9.0 ms. An explicit-layout Gluon leaf
prototype was no faster than this existing kernel and was removed, together
with its temporary options. Wider query/key tiles were also slower. These
serialized component timings are distinct from overlapping model traces.
The early `remote16k-tiles.json` probe did not alias equal K/V arrays and hence
selected a different routing specialization; use the corrected command above.
The corrected [probe](results/glm53-flash-fixture/remote16k-scalar-current.json)
also checks FP32 coarse output/LSE and the same top-eight indices before timing
the refinement variants.

Increasing the centroid kernel from four to eight waves did not help:
at 4K queries/2,048 centroids it increased latency from 7.30 to 10.87 ms and
compiler-reported spills from 26 to 112. Moving PV ahead of the selector loop
also lost at the default 64×64 tile (7.93 ms). Both changes were removed;
the four-wave kernel and original execution order remain the default. Raw
diagnostics: [wave count](results/glm53-flash-fixture/gluon-route-wave-pressure.json),
[execution order](results/glm53-flash-fixture/gluon-route-pv-before-select.json).

The matched 64K vector-lookup control takes 0.6293 s versus 0.6014 s with
scalar per-page lookup (4.4% less prefill time). Both generate identical
1,026-token outputs and identical first-token logprobs in the random fixture.
The exact-prefix policy, top-eight routes, cap and update sizes are unchanged.
Sources: [vector control](results/glm53-flash-fixture/lod-vectorleaf-64k-matched-control.json)
and [scalar lookup](results/glm53-flash-fixture/lod-scalarleaf-64k-current.json).
Reproduce the control by adding `--vector-page-lookup` to the fixture command.

Before the local-prefill fix, the prefill-only diagnostic traces
([native](results/glm53-flash-fixture/native-32k-prefill-profile.json),
[LoD](results/glm53-flash-fixture/lod-32k-prefill-profile.json)) identify a
missing fast local-attention specialization: the Kimi local MLA path requires
512+64 channels, so GLM's 512-only field falls through to explicit tiled
QK/softmax/PV with materialized scores. Standalone PyTorch copy, pointwise,
softmax and reduction kernels sum to 317.4 ms for LoD versus 7.4 ms for native;
fused centroid route/coarse takes 81.2 ms and exact leaf attention 22.6 ms.
These are diagnostic GPU-work totals, not additive wall-time attribution.
Profiling greatly inflates LoD allocator stalls, so its traced 2.36 s prefill
must not replace the uninstrumented timing. The native model also
selects at most 2,048 attention tokens, while LoD retains its exact 16K front
and large exact local prefill region. The new
`lod_attention/kernels/latent_local_prefill.py` fuses causal QK, online softmax,
PV and optional LSE in a single exact streaming kernel. Its latest Gluon
layout uses a 64×64 tile with four waves, and centroid route/coarse reuses each
key tile across 64 query tokens instead of one. K and V share a single loaded
latent tile; no dummy RoPE tail
or full score matrix is materialized. Cached-turn concatenation preserves
K=V, and separate exact-front/local workspaces avoid overwriting a live front.
Output/LSE, batched strided tensors and causal suffix cases pass GPU reference
tests; the matched random-weight fixture generates the same 1,026 tokens as
before the fix. This is still not evidence of trained-model quality.
Reproduce each trace with the commands above, replacing `--decode-tokens 1026`
with `--decode-tokens 1 --profile-prefill` and choosing separate output files.

#### Native GLM initial-chunk experiment

An opt-in hybrid uses GLM's **learned 2,048-token sparse attention**, not dense
attention, for only the initial uncached scheduler chunk (at most 16K). It
still constructs the normal LoD cache from all of that chunk's latents. Later
chunks, cached turns and decode use LoD exclusively: the indexer performs no
projections, cache writes or selection after the initial chunk. Mixed batches
containing cached rows conservatively fall back to the usual LoD path. No
second native main latent cache is allocated; native indexer/tail and KDA
storage use separate contiguous slabs of one backing allocation.

Matched prefix ablations on the four-layer random fixture show:

| TP / batch | Prompt | Exact-prefix LoD prefill | Native-prefix LoD prefill |
|---|---:|---:|---:|
| TP1 / B1 | 16K | 113.46 ms | 118.32 ms |
| TP1 / B1 | 32K | 287.97 ms | 292.32 ms |
| TP4 / B1 | 32K | 132.84 ms | 138.98 ms |

Native-prefix attention works but is slightly **slower** than the newly
optimized exact prefix, so it is **not the default**. These controls retain the
same indexer weights/cache allocation and change only the initial attention;
TP4 uses the same 1,024-wide fixture FFN in both modes. All entries use one
untimed warmup and one measured pass with 1,025 decode steps. Raw sources are
`lod-exact-prefix-16k-control.json`, `lod-native-prefix-16k-updates.json`,
`lod-exact-prefix-32k-control.json`, `lod-native-prefix-32k.json`,
`lod-exact-prefix-tp4-32k-control.json`, and `lod-native-prefix-tp4-32k.json`
under `results/glm53-flash-fixture/`.

```bash
python -m benchmarks.glm53_flash_fixture --mode two-tier --native-prefix \
  --length 32768 --decode-tokens 1026 \
  --output results/glm53-flash-fixture/lod-native-prefix-32k.json
python -m benchmarks.glm53_flash_fixture --mode two-tier --exact-prefix \
  --length 32768 --decode-tokens 1026 \
  --output results/glm53-flash-fixture/lod-exact-prefix-32k-control.json
```

Add `--tp 4 --ffn-width 1024` to both commands for TP4. In ordinary vLLM calls,
the experimental hybrid is enabled with `hf_overrides={"lod_native_prefix": True}`;
without that override LoD drops the native indexer entirely. GPU tests cover
request-local sparse indices, output/causality, cache addressing and allocation,
prefix-hook cleanup, and the existing Kimi kernels. The final combined suite
passed 395 tests. Matching random-fixture tokens is not trained-model quality
evidence; trained GLM ProLong/NIAH/LongBench have not been run.

## Supported configurations

| Mode | Remote detail | Leaf storage |
|---|---|---|
| `two-tier` | every leaf in each selected centroid | BF16 |
| `three-tier-bf16` | best two semantic pages per selected centroid in decode; all leaves in prefill | BF16 |
| `three-tier-int4` | best two semantic pages per selected centroid in decode; all leaves in prefill | residual INT4 |

All modes use exactly eight routed regions in prefill and decode, a
`16 * sqrt(T)` centroid schedule, a 16K prefill catch-up, a 512-token base
decode window, one separately protected sink, and an exact first 16K prefill
region. Decode catch-up occurs every 256 tokens. Ordinary decode scans every
retained leaf only while the context is at most 2K; INT4
then differs solely by residual-quantization error. DFlash2 stays routed at all
lengths because its one-token and multi-token verifier graphs share one pool.
With vLLM prefix caching
enabled, the exact rollback tail is 1,024
tokens so a retained request can be rewound without restoring native K/V.
Three-tier pages contain 16 leaves. INT4 is applied only to residuals within a
centroid-owned semantic page; sequential K/V blocks are never quantized as if
they were semantically coherent.

The release supports:

| Model family | Hugging Face | vLLM | DFlash2 |
|---|---:|---:|---:|
| Qwen3.8 (`D=256`, GQA 6) | yes | yes | yes |
| K2 Horizon (`D=128`, GQA 8) | yes | yes | no |

Model-specific compatibility code is isolated in
`integrations/vllm_lod/vllm_lod_plugin/models/`. The attention engine and
kernels in `lod_attention/` operate on post-QKV, post-RoPE tensors and do not
own model projections.

## Install

Python 3.12, PyTorch, Transformers 5.15, Triton, and the platform attention
kernels are required. The vLLM integration is validated against vLLM 0.27.1 on
ROCm. Install this project into the environment that already provides the
appropriate accelerator build:

```bash
uv pip install -e .
```

The optimized prefill path requires the AITER change in
`integrations/vllm_lod/patches/aiter-mha-prefill-route8.patch`. Apply it to the
AITER source used by the runtime and rebuild AITER before benchmarking. The
patch provides compile-time normalized and raw routing probes. LoD selects the
normalized specialization for K2 and automatically builds a separately cached
raw specialization for Qwen; neither kernel branches on normalization at run
time. The release specialization is built with
`CK_TILE_FMHA_ROUTE_TOPK=8`.

## Hugging Face

Installation happens after model construction and replaces only global causal
attention layers. The model keeps ownership of projections, RoPE,
normalization, gating, and output projection; LoD owns its K/V cache.

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lod_attention import install

checkpoint = "Qwen/Qwen3.8-27B-FP8"
config = AutoConfig.from_pretrained(checkpoint)
# Transformers 5.15's unanchored FP8 skip patterns accidentally make
# ``mlp.gate`` also match ``mlp.gate_proj``. The former is not a Linear.
quantization = config.quantization_config
quantization["modules_to_not_convert"] = [
    name
    for name in quantization["modules_to_not_convert"]
    if not name.endswith(".mlp.gate")
]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    config=config,
    dtype=torch.bfloat16,
    device_map="auto",
)
install(model, mode="two-tier")

inputs = tokenizer("Explain LoD Attention.", return_tensors="pt").to(model.device)
tokens = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

Select `three-tier-bf16` or `three-tier-int4` with the same `mode` argument.
Generation automatically creates the LoD-owned cache. For direct model calls,
`lod_attention.new_cache(model)` returns an empty cache explicitly.

### Pure PyTorch reference

For clarity and portability, the package also includes a standalone reference
engine in `lod_attention/pytorch_engine.py`. It uses ordinary PyTorch tensor
operations and the same paper methodology as the optimized implementation:
the separate sink, exact first 16K prefill block, 16K prefill catch-up, 256-token
decode catch-up, 512-token decode-local field, `16 * sqrt(T)` semantic state,
architecture-aware region assignment and routing, top-eight refinement, the
1,024-entry region cap, count-corrected summaries, and LSE branch merging.
Two-tier and recursive three-tier BF16 frontiers are supported. INT4 is omitted
because it is a specialized page-storage encoding rather than part of the
attention definition.

Use it through the same Hugging Face adapter:

```python
from lod_attention import install

install(model, mode="two-tier", implementation="pytorch")
# Recursive BF16 is also available:
# install(model, mode="three-tier-bf16", implementation="pytorch")
```

Or call the post-RoPE engine directly:

```python
from lod_attention import PytorchLODAttention

attention = PytorchLODAttention(mode="two-tier")
output, cache = attention(query, key, value, use_cache=True)
```

The reference implementation intentionally materializes remote leaf scores and
uses straightforward Python loops for recursive page selection. It is meant
for reading, testing, and porting the algorithm—not for reproducing the release
kernel speed or memory use.

## vLLM

Installing the package registers the `CUSTOM` attention backend. There are
only three public environment settings:

- `VLLM_LOD_MODE`: one of the three modes above (default `two-tier`).
- `VLLM_LOD_POOL_SIZE`: live or retained request rows per worker (default 8).
- `VLLM_LOD_MAX_CONTEXT`: optional per-row context cap.

Unknown `VLLM_LOD_*` and all old `LOD_DEV_*` tuning flags fail at startup.
Use the included chunk-aligned scheduler with a 16K prefill budget plus one
reserved token per live request. This keeps active decode rows from shaving a
few tokens off a long prefill and forcing a second tiny cache-construction pass:

```bash
VLLM_PLUGINS=lod_attention \
VLLM_LOD_MODE=three-tier-int4 \
VLLM_LOD_POOL_SIZE=8 \
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --attention-backend CUSTOM \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16392 \
  --long-prefill-token-threshold 16384 \
  --scheduler-cls vllm_lod_plugin.scheduler.LODChunkAlignedScheduler \
  --gpu-memory-utilization 0.7 \
  --enable-prefix-caching
```

For speculative decoding, reserve the maximum verification width per live
request instead; batch 8 with seven proposed tokens uses `16448`.

For Qwen, the 0.7 target leaves transient workspace headroom outside vLLM's
native-cache allocator; LoD's authoritative per-request pool is already
included in the model-side allocation. K2's larger model-side 131K pool needs
`--gpu-memory-utilization 0.8` merely to leave vLLM a nonempty native-cache
remainder. Raise either target only after measuring peak memory for the intended
model, mode, context limit, and concurrency.

On vLLM revisions that expose only the structured option, replace
`--attention-backend CUSTOM` with
`--attention-config '{"backend":"CUSTOM"}'`.

The LoD cache is authoritative: compressed remote leaves replace their native
chronological K/V rather than shadowing a full cache. Prefix-cache hits resume
retained LoD rows after exact token-prefix verification. Non-attention and
ineligible local/recurrent layers retain their native vLLM caches.

### Reuse post-load weights

The package also registers the `ipc_cache` model loader. On a cold request, a
node-local broker uses vLLM's ordinary loader and retains the final TP/PP/EP
shards after all loader transformations. Later engines with the same model,
dtype, quantization, attention layout, and parallel topology reconstruct those
tensors through CUDA/HIP IPC. They therefore skip checkpoint reads and costly
post-load conversions such as Kimi-K3 MXFP4-to-groupwise-INT4 conversion; the
serving engine still owns its KV/LoD cache, scheduler, workspaces, and graphs.
Runtime context capacity, scheduler budgets, and batch limits do not partition
the retained weights: engines can reuse one entry at different sequence
lengths as long as the actual model/quantization/parallel tensor layout agrees.
Attention execution mode does not partition an otherwise identical layout
either. In particular, Kimi-K3 DCP engines reuse one canonical resident weight
entry when switching among full attention, two-tier LoD, three-tier BF16, and
three-tier INT4; only the per-engine KV/LoD cache and workspaces are rebuilt.

The loader starts the broker automatically in an ordinary process environment:

```bash
VLLM_PLUGINS=lod_attention \
VLLM_WEIGHT_CACHE_ID=dev \
vllm serve MODEL \
  --load-format ipc_cache \
  --attention-backend CUSTOM
```

Use `--model-loader-extra-config` to choose `cache_id`, `cache_dir`,
`backing_load_format`, or to set `auto_start` to false. A missing entry is
loaded just in time. The default per-GPU retained-weight budget is 60%; an
explicit broker can use a different limit:

```bash
vllm-weight-cache --cache-id dev --max-cache-fraction 0.9
vllm-weight-cache status --cache-id dev
vllm-weight-cache stop --cache-id dev
```

The broker must remain alive while mapped engines run. Batch schedulers that
kill all descendants at job exit (including `cluster-run`) should run the
broker as its own long-lived GPU job and launch clients on those same GPUs with
their scheduler's overlap option. This also keeps the broker's retained VRAM
visible to the scheduler. The broker is single-node and currently requires
DP=1; TP and PP are supported.

## Repository layout

- `lod_attention/`: model-independent HF adapter, PyTorch reference, cache,
  optimized engines, and kernels.
- `integrations/vllm_lod/vllm_lod_plugin/`: vLLM backend and cache lifecycle.
- `integrations/vllm_lod/vllm_lod_plugin/models/`: K2 and Qwen DFlash2 shims.
- `integrations/vllm_lod/patches/`: the required AITER patch.
- `examples/`: minimal HF and vLLM launch examples.
- `benchmarks/`: public LongBench v2, ProLong, and RULER NIAH-S3 runners.
- `tests/`: release-policy and import checks.

## Benchmarks

Each benchmark has a standalone runner, archived results, and commands that use
only public tools:

- [LongBench v2](benchmarks/LONGBENCH_V2.md): end-to-end long-context quality
  and serving wall time.
- [ProLong](benchmarks/PROLONG.md): prompt CE/perplexity and matched prefill and
  1,025-token decode speed sweeps.
- [RULER NIAH-S3](benchmarks/NIAH_S3.md): long-context UUID retrieval.
- [KV-cache VRAM](benchmarks/KV_CACHE_VRAM.md): persistent full-attention BF16
  versus three-tier INT4 cache memory.

The benchmark documents report the current top-8 production results.
The retained-leaf exact decode path is limited to contexts of at most 2,048
tokens, so every published 4K-and-longer result exercises routed LoD.

This implementation is inference-only and does not return dense attention
weights. Sliding-window attention, ALiBi, attention soft caps, DCP/PCP, and
native quantized attention K/V are intentionally rejected instead of silently
falling back to a different LoD calculation.

## License

Apache-2.0. Model compatibility files retain their upstream notices.

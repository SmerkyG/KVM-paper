# K3 half-model residency and prefill batching experiment

Development-only on `lod-k3`; no production defaults changed and not pushed.

Hypothesis: increased true prefill batch size helps LoD more than dense
attention, because dense attention already has better GPU occupancy.
Test one stage of a possible two-node TP8/EP8 pipeline on one node only.
Merely executing half the layers while retaining full-model weights does not
test the intended memory headroom.

## Setup

- First **48 trained layers** of the original 93-layer K3, ending at a
  twelve-layer AttnRes block boundary.
- Keep all 36 real KDA layers, 12 real MLA layers, the dense first FFN,
  all 47 actual EP8 MoEs, and their original trained geometry.
- Separate checkpoint config and weight daemon on **node 2, GPUs 0–7**.
  The original full-model daemon on node 4 is not changed or evicted.
- Only the required original shards are copied from the pinned HF snapshot
  to node-local `/tmp` before weight loading: **51 shards, 746.39 GiB**.
  Tensor data is not rewritten. Some global shards include additional tensors
  on disk, but the truncated model must allocate no language layers 48–92.
- This truncated checkpoint is a **speed fixture**, not a model for quality
  evaluation. Its final head was trained for the full depth. No cross-node
  pipeline speedup can be inferred from a one-stage result alone.

The normal loader/daemon fingerprint distinguishes the 48-layer config;
no fingerprint bypass is needed.

## Status

`21213-kimi-k3-half48-weight-cache` completed all 51 local shard copies and
is serving the dedicated `kimi-k3-first48-int4-v1` broker.
`21215-kimi-k3-half48-preload-audit-r2` completed successfully. All eight ranks
confirmed language layer IDs 0–47, 36 KDA/12 MLA layers, the dense first FFN,
and 47 native EP8 MoEs. Imported storage equals daemon-resident weight storage:
**103.054 GiB/rank**. Device free memory during the preload audit was
129.719–131.678 GiB/rank, including the preload client's small native cache and
startup workspaces. See [residency-audit.json](residency-audit.json).

CPU checkpoint-selection/config/preload, batch-audit and weight-cache tests:
26 passed. The initial waiting
preload client was replaced before loading to supply the native backend
argument required by the shared benchmark configuration.

`21221-kimi-k3-half48-prefill-panel-r6` completed dense and two-tier LoD
**sequentially** at 64K with eight real
ProLong requests, first one 16K row per prefill call, then two 16K rows.
Each arm has one exact-shape warmup and one measured pass. Native cache
reservations are matched at 3 GiB/rank. Actual query-row lengths are inspected
on every rank during warmup; the hooks are removed before measured generation.
Warmup compilation/loading is excluded. Raw timing and command artifacts go
to this directory, with aggregate progress in `prefill-panel.json`.

The first panel (`21216`) completed the one-row dense arm, then the LoD
client failed during startup. The resident daemon had exported its own
`_vllm_lod_absorbed_mla=False` flag, overwriting the LoD client's freshly
initialized flag and preventing pool attachment. Weight imports now preserve
client-local `_vllm_lod_*` dispatch metadata (and new exports omit it).
The retry reuses the completed dense point and resident weights; no daemon
reload or changes to attention mathematics are needed. Regression tests pass
for both dense-to-LoD and LoD-to-dense metadata isolation.

The second attempt (`21217`) exposed a missing canonical-projection view:
the dense backing model had retained native FP8 BMM helpers instead of
`W_UK_T`/`W_UV`. LoD clients now recover these exact BF16 views from the
daemon's original, already-mapped BF16 `kv_b_proj.weight` without copying
or quantizing it. A regression test verifies shared storage, exact tensor
contents, unchanged original weights, and idempotence. Dense clients keep
their original helpers and the completed dense timing remains applicable.

The third and fourth attempts completed LoD warmup but rejected its empty
batch audit. Wrapping `model.forward` did not observe LoD model calls because
the CUDA-graph wrapper's `__call__` invokes its underlying runnable directly.
The fifth attempt caught the image's v2 model runner, which does not have
legacy `_model_forward`. The audit now wraps the pinned v2 runner's actual
`prepare_inputs` result and reads `InputBatch.query_start_loc_np`, independent
of backend metadata and graph wrappers. CPU regression tests verify recording
and complete restoration before measurement. There are no audit hooks or
audit-device synchronizations in the measured pass.

Precision clarification: both arms use **BF16 prefill Q/K/V** and a BF16
latent KV cache. Native prefill uses `kv_b_proj` and the ROCm AITER
FlashAttention prefill backend; prefill query quantization is disabled.
The daemon's FP8 `W_K`/`W_V` helpers are for absorbed-query and value-output
projections in **decode**, not the prefill attention measured here. LoD uses
the original BF16 projection views. Both arms retain the same INT4 MoE
weights; neither arm uses an INT4 or FP8 attention cache in this experiment.

## Comparison policy

Use normal non-owner TP8/DCP8/EP8 dense and two-tier LoD, the same real ProLong
prompts, and the existing global 16K prefill update policy. Compare
one, two, four or eight simultaneous **16K row chunks**, with the same scheduler budget
and row-chunk sizes within each dense/LoD pair. Audit actual scheduled row
counts; a nominal B8 request batch is not proof of a true B2 prefill call.
Measurements exclude warmup and checkpoint copying/loading. Extend batch
size only after the half-residency audit and small smoke checks succeed.

The half-resident weights also make longer contexts and larger real prefill
calls feasible while retaining the trained MoE/KDA layers. The expanded
driver accepts 128K and longer contexts and four/eight simultaneous rows;
it still uses a **16K per-request** prefill chunk/update cadence. Increasing
the total scheduler budget does not change that cadence to 64K or 128K.
Longer-context artifacts have separate filenames so they cannot overwrite
the completed 64K comparisons. Native cache reservations remain matched
within each dense/LoD pair, and measured runs never overlap one another.

## Prefill results

Eight 65,536-token ProLong requests on node 2, TP8/DCP8/EP8. Times are the
warmed **entire request-batch prefill latency**, including final cache
construction and the first output token; not per-request latency, attention
only, or a pipeline throughput measurement. One measured pass per arm.

| Actual prefill call | Dense (s) | Two-tier LoD (s) | Dense / LoD |
|--|--:|--:|--:|
| One 16K row (16K budget) | 35.3505 | 34.3353 | 1.030× |
| Two 16K rows (32K budget) | 33.4816 | 32.3803 | 1.034× |
| Four 16K rows (64K budget) | 32.1491 | 31.0393 | 1.036× |

Sources: [one-row dense](oct5-full-b8-cohort1-prefill.json),
[one-row LoD](oct5-two-tier-b8-cohort1-prefill.json). Their warmup audits
confirmed 32 calls of one 16K row on each rank, exactly 524,288 real tokens.
LoD's loaded-kernel audit passed: 12 attached MLA layers, fused top-8 AITER
route/coarse, global 16K prefill update cadence, cross-layer group size 4.
The one-row comparison is a small **3.0% throughput gain**, not a dramatic
half-model speedup. Two-row batching improves dense throughput by **5.6%**
and LoD by **6.0%** relative to their respective one-row measurements. LoD's
advantage over dense remains small, **3.4%**, rather than showing a major
additional batching benefit. These are single-pass observations, not a
statistical confidence estimate.

The two-row audits confirmed 16 calls of two 16K rows on every rank; see
[two-row dense](oct5-full-b8-cohort2-prefill.json) and
[two-row LoD](oct5-two-tier-b8-cohort2-prefill.json). All four measurements
passed batch and loaded-attention audits. No full-model or two-node pipeline
claim follows from these half-stage timings.

The four-row comparison (`21222-kimi-k3-half48-prefill-b4-64k`) also passed
all eight ranks' audits: eight calls of four 16K rows, 524,288 real tokens
per rank. Sources: [four-row dense](oct5-full-b8-cohort4-prefill.json) and
[four-row LoD](oct5-two-tier-b8-cohort4-prefill.json). Throughput relative
to the one-row call improved by **10.0% dense** and **10.6% LoD**; the
dense/LoD advantage is still small at 64K, **3.6%**. Minimum device free
memory after measured generation was **87.02 GiB dense / 70.82 GiB LoD**
per rank. This is actual device headroom including daemon-resident weights;
the client's PyTorch allocation counter alone does not include those IPC
weight allocations. Post-generation free memory is not a peak-free-memory
measurement.

The eight-row comparison at **128K context** (`21223`) used eight total
ProLong requests and a matched **6 GiB/rank native cache reservation**.
[Dense completed](oct5-full-b8-cohort8-prefill-128k.json) in **70.5547 s**,
with all batch/binary audits passing. The minimum device-free reading
after measured generation was 43.72 GiB/rank. [LoD](oct5-two-tier-b8-cohort8-prefill-128k.json)
completed its 72.2778 s warmup and batch audit, then failed during measured
generation with a ROCm/RCCL resource-allocation error and 12 MB reported
free on the failing GPU. **No valid LoD timing or speedup exists for this
pair.** The warmup time is not substituted for a serving measurement.
The larger scheduler budget does not change the global 16K per-request
update policy. Resident weights remain loaded and unchanged.

## One row per GPU, with local MLA projections

Opt-in experiment (`21225`), not a production default: each GPU computes
one row's full 96-head MLA Q/K/V/O projections and two-tier LoD, then gathers
the hidden-width output back into the native distributed model. All 36 KDA
layers, all 47 MoEs, both residual mixes, and the dense first FFN are retained.
The normal eight-row MoE call is **not microbatched** into separate rows;
the owner's MoE wrapper forwards the complete 131,072-token input unchanged.

Full BF16 MLA projection copies are assembled from already-imported resident
TP shards **once, outside timing**; the daemon's shared weights are never
modified. Real ProLong tokens validate query-head layout and the gated output
projection on every MLA layer and rank before generation. A single grow-only
hidden-output transport arena is shared across the MLA layers. The experiment
is eager and prefill-only; graph-captured serving and concurrent forwards are
not claimed.

The first run checks 32K, then 128K, with eight 16K rows per call, the existing
global 16K update cadence, one exact-shape warmup and one measured pass. Frozen
prompt hashes match the dense cohort above. Native cache reservation remains
6 GiB/rank. Timing includes embeddings, all model layers, owner state updates,
output transfers and first-token production; only startup/projection assembly
and the untimed validation audits are excluded. Both points completed with
all eight ranks' batch and loaded-kernel audits passing.
An initial startup (`21224`) was stopped before timing when its legacy owner
configuration was found to change native norm/IR kernel selection through
`enforce_eager`. The private experiment now keeps the normal startup runtime
configuration. Its real prefills remain uncaptured, as in the dense control;
it never exercises the startup decode graphs with the installed owner hooks.

All eight workers passed the twelve-layer projection checks: maximum relative
L2 difference 0.418% for the gated output projection. Additional gathered
projection storage is **4.852 GiB/rank**. The first 32K point completed in
**14.7120 s**, with all batch/binary audits passing and exactly 48 full-16K
owner MLA calls per GPU across warmup and measurement. Device free memory
after measured generation was at least **38.96 GiB/rank**. The follow-up
half-model dense 32K control (`21227`) completed in **14.9864 s**.
The same-shape dense 128K control is 70.5547 s; owner 128K is **61.3973 s**.
CPU half-model/projection-layout/cache/audit tests: 41 passed.

| Context / eight total requests | Dense prefill (s) | Local-MLA owner LoD prefill (s) | Dense / owner |
|--:|--:|--:|--:|
| 32K | 14.9864 | 14.7120 | 1.019× |
| 128K | 70.5547 | 61.3973 | **1.149×** |

Raw owner result:
[oct5-owner-local-mla-b8-32k128k-prefill.json](oct5-owner-local-mla-b8-32k128k-prefill.json).
At 128K, whole-generation wall time was 61.4233 s, within 27 ms of the prefill
metric; warmup was 61.4122 s. The eight prompt hashes exactly match the dense
128K cohort. All ranks observed eight model calls of eight 16K rows during
warmup; each GPU recorded 240 complete-row MLA calls across both lengths'
warmup and measured passes. This is **13.0% less prefill time / 14.9% higher
throughput** than dense in a single measured pass, not a confidence interval.

At 128K, minimum device-free memory after generation was **36.53 GiB/rank**,
versus 43.72 GiB/rank for dense. Maximum client Torch allocated/reserved memory
was **81.01 / 96.50 GiB/rank**, excluding the daemon's 103.054 GiB resident
weight storage. The owner layout therefore fits comfortably, but does **not**
use less total VRAM than dense at this point; gathered projection copies,
transport storage and attention scratch offset its cache-ownership benefits.
Ordinary eight-row LoD failed its measured pass, so no measured owner-vs-ordinary
LoD speedup or total-VRAM reduction is claimed.

Seven of eight generated first tokens match dense at 128K. This is not a
language-quality score: the half-depth checkpoint is a speed fixture, and LoD
is approximate. It does not establish full-model quality or bitwise equivalence.
This implementation deliberately supports only aligned prefill. It is not an
owner-local Q/K/V/O decode backend or a two-node pipeline throughput result.

### Retain tensor-parallel projections, without the large owner copies

Follow-up `21226` uses the identical aligned eight-row owner attention/cache
layout but leaves the original MLA wrapper and Q/K/V/O/gate projections in
native TP8. Each rank sends its query-head slices to the request owners;
owners return the 128-wide head outputs for native gating and TP W_O.
KDA/MoE, row chunk size, logical update cadence, frozen ProLong prompts,
native cache reservation, norm/operator settings and timing policy are
unchanged. Completed dense/local-projection controls are not rerun.

There are no full Q/gate/O projection copies, removing **4.570 GiB/rank**
from the prior **4.852 GiB/rank** gathered projection storage. The owner still
needs the small full-head latent key/value maps: **0.281 GiB/rank** across all
twelve MLA layers. These are gathered once before warmup, not charged to
measured serving time. Two layer-shared head-transport arenas (**1.125 GiB/rank**,
two allocations total) are reused at the fixed scheduler shape, rather than
the local-projection path's hidden output arena. Both 32K and 128K completed
with all eight ranks' scheduling and loaded-kernel audits passing. Prompt
hashes match the completed controls, and all ranks kept the global 16K
prefill / 256-token decode update configuration. Focused CPU
owner/exchange/cache tests: 74 passed, 2 GPU-only tests skipped.

| Context / eight total requests | Dense prefill (s) | Local-projection owner (s) | Native-TP-projection owner (s) | Dense / TP owner |
|--:|--:|--:|--:|--:|
| 32K | 14.9864 | 14.7120 | 15.1173 | 0.991× |
| 128K | 70.5547 | 61.3973 | 63.0449 | **1.119×** |

Keeping native TP projections costs **2.75% more time at 32K / 2.68% at
128K** than the full local-projection copies, but still saves **10.64%**
of the dense 128K prefill time. The 128K warmup was 63.0283 s and measured
whole-generation wall time was 63.0668 s. These are single measured passes,
not confidence intervals; no decode steps are timed.

Post-generation device free memory was at least **43.06 GiB/rank at 32K**
and **40.63 GiB/rank at 128K**, versus 38.96 / 36.53 GiB/rank with local
projections. The paired per-rank free-memory improvement is **4.098 GiB/rank**
at both lengths. At 128K, maximum client Torch allocated/reserved memory
was **75.81 / 91.97 GiB/rank**, excluding the daemon's resident weights.
This is less memory than the local-copy owner path, but still more than
dense (64.50 / 87.59 GiB/rank; at least 43.72 GiB/rank device-free).
Post-generation free memory and client Torch peaks are distinct metrics;
neither is a measured peak of total device usage including the daemon.

Raw result:
[oct5-owner-tp-mla-b8-32k128k-prefill.json](oct5-owner-tp-mla-b8-32k128k-prefill.json).

The [32K dense control](oct5-full-b8-cohort8-prefill-32k.json) uses the same
eight prompt hashes, native 6 GiB/rank cache reservation, eight 16K rows per
call, and startup/norm/timing policy. All eight ranks passed batch and
attention audits; warmup was 14.9796 s and measured wall time 14.9927 s.
Thus the two owner variants are approximately tied with dense at 32K, not
a demonstrated substantial speedup. This standalone dense engine has a
32K configured maximum; the owner 32K/128K sweep has a 128K maximum.
Memory/capacity comparisons should not conflate those configured maxima.

The original owner decode prototype was eager, not a fair performance control
against normally captured DCP decode. These prefill-only measurements do not
establish decode speed. The captured follow-up is described below.

Reproduce with the owner command below, replacing `--owner-local-mla` with
`--owner-tp-mla` and the output filename with
`results/kimi-k3-half-model/oct5-owner-tp-mla-b8-32k128k-prefill.json`.

## Captured request-owner decode on the split model

Development-only aligned B8/TP8 mode; not a general mixed-batch or prefix-reuse
serving backend. No local full Q/gate/O weight copies are added. KDA/MoE/gating
and W_O retain the normal TP8/EP8 layout. Attention/cache ownership is one
request per GPU, with all 96 heads and a single physical latent KV archive.

- Create the fixed single-owner decode pools and small full-head latent maps
  before startup graph capture. Installing the completed prefill fills those
  buffers without replacing any captured pointers, then releases the source
  prefill cache.
- Reuse native complete-head queries when supplied. Metadata-free synthetic
  startup can omit them; the fallback is a fixed-buffer query all-gather, not
  Python request-wise sends. Both paths are capturable.
- Each owner runs ordinary single-GPU top-eight LoD and W_UV. One fixed-shape
  reduce-scatter returns the twelve-head slices to native TP gating and W_O.
  Rows have disjoint contributors, so no attention-LSE combination is needed.
- Scheduler preprocessing performs global **256-token per-request** catch-ups
  outside the graph. Captured attention itself reads/writes fixed buffers and
  increments the device recent length. The B8 full-model graph is replayed
  between updates; updates are still included in serving-time measurements.

CPU layout/lifecycle tests pass. GPU test `21230` passed exact eager/captured
equivalence, including a changed device request index and fixed buffer/cache
pointers. Initial model startup `21231` rejected the synthetic missing-query
case before timing; its fixed-buffer fallback is covered by a regression
test. No result from that failed startup is used.

The matched split-model dense control (`21228`) completed at B8/32K:
**15.0448 s prefill / 16.7496 ms per batch decode step**. The first captured
owner run (`21232`, generic BMM projections) also completed correctly at
**15.1590 s prefill / 17.5636 ms per batch decode step**: capture alone has
not made it faster than dense at this length. All ranks instantiated a real
B8 model graph with twelve 96-head, DCP=1 owner pools. Every layer/rank
recorded exactly four catch-ups and 1,025 decode tokens in the measured pass.
The follow-up (`21234`) fuses one-token query absorption and value projection
with row selection/output packing. GPU test `21233` passed projection
numerical checks and exact eager/captured attention equivalence. The paired
projection/packing microbenchmark fell from 0.02950 to 0.01130 ms (2.61x),
but this is not an end-to-end speedup. Measured serving decode improved only
to **17.3294 ms per batch step**, still 3.5% slower than dense.

| B8/32K variant | Prefill (s) | Decode (ms/batch step) | Dense/variant decode speed |
| --- | ---: | ---: | ---: |
| Dense native TP8/DCP8 | 15.0448 | 16.7496 | 1.000x |
| Captured owner LoD, generic BMM projections | 15.1590 | 17.5636 | 0.954x |
| Captured owner LoD, fused projections/packing | 15.1740 | 17.3294 | 0.967x |
| Captured owner LoD, fused projections + layer-batched updates | 15.1687 | 17.3104 | 0.968x |

The final follow-up (`21238`) reuses the existing layer-batched state update
for the twelve owner caches. It preserves independent layer state and the
same 256-token global update boundaries; this is not a longer-overflow
experiment. GPU regression `21237` verified exact agreement with independent
updates for centroid keys/counts, the consumer's semantic leaf index list,
and host metadata, with unchanged cache pointers. Physical directory page
IDs/unused page padding are not required to be identical.

The final run completed at **17.3104 ms per batch decode step**, essentially
unchanged from the fused-only variant and still **3.35% slower than dense**.
Its host-only replay audit recorded exactly **1,025 actual B8 model-graph
replays on each of all eight ranks** during the measured pass, not just the
existence of captured graph handles. Every layer/rank also recorded exactly
four state catch-ups and 1,025 decode tokens. No GPU events, synchronization,
or attention replacements were added by that replay counter. The broader CPU
suite passed 364 tests (45 GPU/dependency tests skipped).

This remains an opt-in development ownership path, not a demonstrated decode
speed win or a promoted serving default. There is no evidence from this
32K point alone for its relative decode speed at longer contexts.
Both arms use the same real ProLong prompt/continuation cohort,
one exact-shape warmup, one measured pass, 6 GiB/rank native cache reservation,
and 1,026 outputs (**1,025 timed decode steps**). This continuation cohort is
different from the earlier prefill-only 32K cohort, so those prefill latencies
must not be substituted into this comparison. The truncated model is still
only a speed fixture; trace replay is a timing control, not a quality score.

Dense artifact:
[oct5-full-b8-32k-decode1025.json](oct5-full-b8-32k-decode1025.json).
Initial captured owner artifact:
[oct5-owner-tp-mla-b8-32k-decode1025-bmm.json](oct5-owner-tp-mla-b8-32k-decode1025-bmm.json).
Fused projection artifact:
[oct5-owner-tp-mla-b8-32k-decode1025-fused.json](oct5-owner-tp-mla-b8-32k-decode1025-fused.json).
Final captured/layer-batched artifact:
[oct5-owner-tp-mla-b8-32k-decode1025.json](oct5-owner-tp-mla-b8-32k-decode1025.json).

Reproduce using the owner command below with `--owner-tp-mla`,
`--lengths 32768 --decode-tokens 1026 --reference-decode-trace`, and replace
the reference baseline with
`results/kimi-k3-full-model-current/oct4-full-b8-decode-16k64k-four-updates.json`.
Use output `results/kimi-k3-half-model/oct5-owner-tp-mla-b8-32k-decode1025.json`.
For dense, omit `--owner-tp-mla`, set `--mode full`, and change the output
filename to `oct5-full-b8-32k-decode1025.json`. Keep all other settings identical.

## Reproduction without the cluster runner

From the repo root on the target node, prepare local files and run the broker:

```bash
.venv/bin/python -m benchmarks.kimi_k3_prepare_half_checkpoint \
  --source /home/dan/subusers/agent/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06 \
  --destination /tmp/dan-agent-kimi-k3-first48-f831ab66814297da540d832a5235f8e904f29d06 \
  --layers 48 --copy-workers 4 --serve-cache-id kimi-k3-first48-int4-v1
```

In a second terminal on the same node, trigger and audit the initial load:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_half_preload \
  --checkpoint /tmp/dan-agent-kimi-k3-first48-f831ab66814297da540d832a5235f8e904f29d06 \
  --weight-cache-id kimi-k3-first48-int4-v1 \
  --output results/kimi-k3-half-model/residency-audit.json
```

The prepared v10 userspace and all eight visible MI325X GPUs are required.
The preload client waits for copying to finish, then loads through the new
daemon; it never imports tensors from the full-model daemon on node 4.

Once the residency audit exists, run the matched prefill panel:

```bash
.venv/bin/python -m benchmarks.kimi_k3_half_prefill \
  --checkpoint /tmp/dan-agent-kimi-k3-first48-f831ab66814297da540d832a5235f8e904f29d06 \
  --cache-id kimi-k3-first48-int4-v1
```

Higher concurrency and longer context, on the same broker:

```bash
.venv/bin/python -m benchmarks.kimi_k3_half_prefill \
  --checkpoint /tmp/dan-agent-kimi-k3-first48-f831ab66814297da540d832a5235f8e904f29d06 \
  --cache-id kimi-k3-first48-int4-v1 \
  --cohorts 8 --lengths 131072 --kv-cache-memory-bytes 6442450944
```

Local MLA owner experiment, using the same already-resident half-model broker:

```bash
env \
  LOD_BENCHMARK_ADMISSION_COHORT=8 \
  LOD_BENCHMARK_SYNCHRONIZED_DECODE=1 \
  LOD_BENCHMARK_PREFILL_COHORT=8 \
  LOD_KIMI_SUBTILE64=score LOD_KIMI_CHUNK_TILE_PACK=1 \
  LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 LOD_KIMI_SORT_LEAF_ROUTES=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_PREFILL_MIN_FREE_GIB=4 LOD_KIMI_TILE_REFINE=1 \
  LOD_KIMI_DIRECT_LEAF_RESULT=1 HSA_NO_SCRATCH_RECLAIM=1 \
  TRITON_CACHE_AUTOTUNING=1 VLLM_USE_TRITON_AWQ=1 \
  AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv" \
  bash benchmarks/run_kimi_k3_v10_direct.sh \
    -m benchmarks.kimi_k3_prefill_sweep \
    --checkpoint /tmp/dan-agent-kimi-k3-first48-f831ab66814297da540d832a5235f8e904f29d06 \
    --weight-cache-id kimi-k3-first48-int4-v1 \
    --mode two-tier --owner-local-mla --lengths 32768 131072 \
    --batch-size 8 --decode-tokens 1 --repeats 1 \
    --tensor-parallel-size 8 --decode-context-parallel-size 8 \
    --kv-cache-memory-bytes 6442450944 \
    --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
    --reference-baselines \
      results/kimi-k3-full-model-current/oct4-full-warm-prefill-b8-32k64k.json \
      results/kimi-k3-full-model-current/oct4-full-prefill-scale-b8-5g-128k.json \
    --audit-prefill-batches --report-memory \
    --output results/kimi-k3-half-model/oct5-owner-local-mla-b8-32k128k-prefill.json
```

# Kimi K3 full-model attention timings

The tables below are historical full-model measurements, not a benchmark of
the latest October 4 experimental path. Current **fixture-only** comparisons
and their audits are in [the MLA-stack results](../kimi-k3-mla-stack/README.md),
under "Allocator reuse: current successful fixture path". A full-model
resident-weight prefill check is recorded below; no fixture speedup should be
reported as a full-model speedup. The rejected tile-max-only **v10** route
binary is not valid top-eight evidence; this rejection does not automatically
invalidate other revisions whose audited flags differ.

**October 4 correctness update:** the deferred-query optimization had a sink
bug: it fed a broadcast shape carrier to the final sink score instead of the
real query. The pre-fix October 4 LoD timings below are retained as development
records, **not results for the corrected attention calculation**. Forced trace
checks do not test this. The corrected path projects the tiny sink K/V and
scores it with the real D192 query, retaining all direct-key dimensions and
FP32 score accumulation. It also avoids the unnecessary 216 MiB carrier copy
per 16K/12-head chunk. Dense attention is unaffected. Corrected full-model
timings are in `oct4-lod-correct-sink-scalar-directory-b8.json`.

These are the full-model measurements previously collected for dense full
attention and two-tier BF16 LoD attention. Prefill is total wall time for the
entire request batch. Decode is latency per batched generation step, rather
than per sequence.

## Latest corrected full-model prefill check (October 4)

### Allocator-retention audit

The host-only audit in `oct4-lod-prefill-allocator-audit-b8-64k.json`
(20993) records **eight allocator reclamations per worker during warmup,
and eight more during the measured eight-request prefill**. None of these
runtime calls retained blocks. The minimum physical free memory at the
checks was 4.45--6.60 GiB, below the existing 8 GiB guard. In contrast, the
separate post-warmup check retained blocks on all ranks. Therefore the
post-warmup metadata alone did not establish retention throughout prefill.

This check changes no allocator policy and adds no GPU events or allocations;
the counters reuse the existing memory-pressure query. It takes **71.499 s**
at 64K/B8, versus the matched dense **70.564 s** (0.987x). Prompt and
continuation metadata match the earlier score-only LoD control directly;
there are no preemptions or prefix-cache hits. This identifies a possible
optimization target, not proof that reclamation explains the remaining gap.
Reproduce the preceding score-only configuration with length 65536 and the
same allocator/cache protocol; the counters are recorded automatically.

| Batch | Context | Full prefill | Corrected LoD prefill | Full / LoD |
|---:|---:|---:|---:|---:|
| 8 | 32K | — | 35.111 s | — |
| 8 | 64K | 70.538 s | 72.008 s | 0.980x |

Corrected LoD: `oct4-lod-correct-sink-scalar-directory-b8.json` (20858).
The unchanged dense 64K baseline is `oct4-full-warm-prefill-b8-64k.json`
(20828). Both use the same eight actual ProLong prompts and forced natural
continuation, TP8/DCP8/EP8, 16K scheduler chunks, 3 GiB native cache/rank,
resident weights, one exact-shape warmup and one measured generation, and
retain warm allocator blocks under the same 8 GiB pressure guard. Prompt
metadata and timing protocols were compared directly and match. All eight
LoD workers pass the v13/exact-refinement/top-eight audit and retain allocator
blocks; no requests are preempted or served from prefix cache. Final cache
construction is included before first token. The two-token trace is **not**
an amortized decode or quality result.

The corrected path has **not crossed over on the full model at 64K**. The
32K dense cell is intentionally blank: the older 33.635 s baseline used a
2 GiB reservation and released the warmup allocator, rather than this exact
protocol. Reproduce LoD with the command below, using `--lengths 32768,65536`,
`--kv-cache-memory-bytes 3221225472` and `--retain-warmup-allocator`.

The subsequent independent-projection-overlap check
(`oct4-lod-overlap-projection-b8-64k.json`, 20877) takes **71.924 s** at
64K/B8. It changes only the scheduling of leaf K/V projection relative to
local/coarse completion; expert packing and refinement retain their waits.
Timing protocol and all eight prompt/continuation metadata records match the
dense control above. All workers retain allocator blocks and the binary/path
audit passes. Against 72.008 s without overlap, this is a negligible difference,
not a demonstrated full-model speedup; dense / this candidate is **0.981x**.
The fixture's roughly 2% gain did not transfer. The candidate remains opt-in
(`LOD_KIMI_OVERLAP_LEAF_PROJECTION=1`), and its two-token trace is not a
quality or update-amortized decode result.

The atomics-free 1024-query packing check
(`oct4-lod-chunk1024-b8-64k.json`, 20919) takes **71.925 s** at 64K/B8.
Its prompt/continuation metadata, cache reservation and timing protocol match
the dense baseline above. All eight workers retain allocator blocks and load
the audited exact-refinement binaries; the new fragment-packing kernels are
observed during warmup. The fixture's approximately 2.2% improvement again
does not establish full-model crossover: this is only 0.083 s below the
72.008 s ordinary control and remains about 2% slower than dense. Packing
layout changes neither the 16K per-request prefill cadence nor the global
256-token decode cadence. It remains opt-in, not a new production default.

Reproduce this candidate with the command below, changing the length to
65536, native cache reservation to 3221225472, adding
`--retain-warmup-allocator`, and setting `LOD_KIMI_CHUNK_TILE_PACK=1` plus
`LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024`. As with the other two-token checks,
this is prefill evidence only, not quality or amortized decode evidence.

The combined exact-layout check (`oct4-lod-combined-b8-64k.json`, 20943)
takes **71.297 s** at 64K/B8, versus **72.008 s** for the ordinary corrected
path and **70.538 s** for matched dense attention. This reduces LoD latency by
about 1.0%, but dense / LoD is still **0.989x**: the full model remains about
1.1% slower, not crossed over. It combines 1024-query refinement fragments,
sorted leaf-route ordinals and 64-query/one-warp exact-leaf tiles. Neither
the top-eight rule, 1024-leaf cap nor global 16K/256 cadence changes.

The recorded `prompts` (including continuation metadata), timing protocol
and native cache-byte reservation match the dense baseline directly. All
eight worker audits pass and all warmup allocator blocks are retained. New
fragment, sort and leaf-tile kernels are observed during warmup. The larger
fixture gain does not transfer proportionally to the full model; these are
still two-token **prefill-only** checks, not quality or amortized decode tests.

Reproduce with the same command and reservation as the preceding check,
additionally setting `LOD_KIMI_SORT_LEAF_ROUTES=1`,
`LOD_KIMI_LEAF_BLOCK_M=64` and `LOD_KIMI_LEAF_WARPS=1`. The candidate remains
opt-in until there is stronger full-model evidence.

### Fresh matched dense panel and 64-key subgroup routing

A fresh dense panel (20952, `oct4-full-warm-prefill-b8-32k64k.json`)
uses the same 3 GiB/rank reservation and retained warm allocator protocol
at both lengths. Unlike the historical 32K dense row, it is directly matched
to the corrected October 4 path. The score-only subgroup candidate (20957,
`oct4-lod-subtile-score-b8-32k64k.json`) keeps the native coarse attention
tile at 128 keys but emits one maximum per 64-key half. Selecting eight halves
and rescoring their keys recovers exact global top-eight while halving the
refinement QK work. It also uses the combined fragment/sorted-leaf layout above.

| Batch | Context | Fresh full prefill | Subgroup LoD prefill | Full / LoD |
|---:|---:|---:|---:|---:|
| 8 | 32K | 33.567 s | 34.530 s | 0.972x |
| 8 | 64K | 70.564 s | 71.332 s | 0.989x |

The recorded `prompts`, timing protocol, seed, native cache reservation,
scheduler budget, TP/DCP, trace length and resident weight ID match directly.
All eight candidate workers load the distinct score-only subgroup binary.
This does **not** establish prefill crossover. Compared with the 71.297 s
combined 128-key candidate, the 64K difference is negligible. Its 32K result
is modestly below the earlier corrected ordinary 35.111 s result, but one
measurement does not establish a robust 32K gain. These remain two-token
prefill checks, not quality or amortized decode results.

Reproduce with the command/reservation above, `--lengths 32768,65536`,
`--retain-warmup-allocator`, the combined layout settings, and
`LOD_KIMI_SUBTILE64=score`. For dense, use `--mode full` without the candidate
environment settings. Kernel/fixture results and the subsequent shared-maximum
experiment are documented in the MLA-stack README.

## Historical October 4 matched prefill check (before sink correction)

These fresh controls use the same eight real ProLong prompts, TP8/DCP8/EP8,
16K scheduler chunks, a 2 GiB native-cache reservation per rank, the resident
`kimi-k3-shared-int4-v6` weight daemon, one exact-shape warmup, and one measured
pass. The final LoD construction completion is included before first token.
Prompt and continuation hashes and timing protocols match. Neither run had
preemptions or prefix-cache hits. Both ran sequentially on node 4 with the
weight daemon idle; no timed model processes overlapped.

| Batch | Context | Full prefill | LoD prefill | Full / LoD |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 32K | 33.635 s | 33.951 s | 0.991x |

This is essentially a tie, **not** evidence of a clear full-model speedup at
32K. Sources: `oct4-full-prefill-b8-32k.json` (job 20820) and
`oct4-prefill-reuse-b8-32k.json` (job 20818). The LoD candidate combines exact
tile refinement, direct leaf-result consumption, final-only allocator
reclamation, and pressure-guarded allocator reuse. Its effective construction
group is **four layers**, as recorded by the worker audit (the module's generic
default constant of 12 is not K3's effective group size).

The two generated tokens are a forced natural-text continuation, used solely
to measure prefill cheaply. They do **not** establish quality or amortized
decode performance. In particular, the single decode-step number in these
JSONs is not a 256-token-update-amortized decode benchmark. Model parameters
were mapped from the resident daemon, not reloaded or rematerialized.

Reproduction on a machine with the prepared v10 userspace and resident cache:

```bash
env LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 \
  LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 LOD_KIMI_TILE_REFINE=1 \
  LOD_KIMI_DIRECT_LEAF_RESULT=1 LOD_BENCHMARK_SYNC_PREFILL_CACHE=1 \
  TRITON_CACHE_AUTOTUNING=1 VLLM_USE_TRITON_AWQ=1 \
  PROLONG_SPEED_TOKEN_CACHE="$PWD/results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt" \
  AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv" \
  benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.prolong \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --mode two-tier --measure speed --lengths 32768 \
  --batch-size 8 --speed-samples 8 --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 --dcp-comm-backend ag_rs \
  --decode-tokens 2 --fixed-decode-trace --synchronized-decode \
  --repeats 1 --seed 0 --gpu-memory-utilization 0.8 \
  --kv-cache-memory-bytes 2147483648 --kimi-gfx942-int4-moe \
  --weight-cache --weight-cache-id kimi-k3-shared-int4-v6 \
  --allow-experimental-environment \
  --output results/kimi-k3-full-model-current/oct4-prefill-reuse-b8-32k.json
```

The full-attention control uses the same command with `--mode full`, omits the
four `LOD_KIMI_*` experiment variables, and writes a separate output. Native
AITER handles prefill; the decoder choice must be audited separately in any
future amortized decode panel.

The 64K LoD check completed in 72.297 s
(`oct4-prefill-reuse-b8-64k.json`, job 20821). Its initial dense control with
2 GiB native allocation was cancelled: vLLM reported capacity for **7.21**
65,554-token requests, insufficient for a synchronized eight-request cohort.
Seven completed prefills held their blocks at the decode barrier, leaving the
eighth unable to finish. No timing is reported from that stalled job 20824.
The benchmark now checks vLLM's hybrid-cache concurrency before admitting a
synchronized cohort. The replacement uses 3 GiB for both attention modes.

An explicitly labeled warm-serving comparison additionally uses
`--retain-warmup-allocator` **in both modes**. The ordinary ProLong benchmark
empties idle allocator blocks between warmup and measurement; the fixture
tuner does not. This distinction is now recorded in `timing_protocol` and
per-worker `allocator_after_warmup`, including any pressure-triggered fallback.
Old measurements are not silently relabeled as warm-allocation results.

The replacement matched 64K warm-serving pair completed:

| Batch | Context | Full prefill | LoD prefill | Full / LoD |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 64K | 70.538 s | 72.259 s | 0.976x |

Sources: `oct4-full-warm-prefill-b8-64k.json` (job 20828) and
`oct4-lod-warm-prefill-b8-64k.json` (job 20833). Prompt/continuation metadata
and timing protocols match; both reserve 3 GiB native cache per rank and retain
the warmup allocator with the same 8 GiB pressure guard. All eight workers
retained their allocator blocks, and all mode/binary audits passed. Neither
run had preemption or prefix-cache hits. This is **not a full-model win**;
the fixture improvement has not transferred. The two-token decode caveat
above still applies. Reproduce with the preceding command, changing the
length to 65536, native reservation to 3221225472 bytes, and adding
`--retain-warmup-allocator` identically to both modes.

### Graph-capture and real-input leaf diagnostic

Both matched full-model logs explicitly downgrade `FULL_AND_PIECEWISE` to
`FULL_DECODE_ONLY`: the image's native K3 model supplies neither a compiled
submodule nor breakable CUDA graphs. This affects **both dense and LoD**.
Ordinary decode graph capture succeeds; full prefill is not currently captured.
LoD additionally has host-side per-request cache plans, changing active-state
lengths, and cache-construction completion/reclamation waits. Scratch reuse
alone does not make that entire path graph-replayable.

The subsequent opt-in graph prototype below enables piecewise prefill capture;
the statement above describes the preceding default-path controls, not that
prototype.

A separate diagnostic captured real trained inputs from the first MLA layer
on a 16K-query chunk with 16,127 archived leaves and 2,048 centroids. All eight
routes are open here: approximately 408.54 underlying leaves/query and 51.07
leaves/opened centroid on average. The capture contains actual model tensors
from ProLong, not randomized query/key data. It is a local diagnostic artifact,
not a model checkpoint or a release quality benchmark.

`trained-leaf-tiling.json` (job 20837) tests fixed routes/ownership with different
exact-attention tiles. The current 32x16/one-wave tile wins at 1.407 ms; tested
larger tiles range from 1.602 to 2.255 ms. Output relative L2 errors are below
0.0007 and LSE maximum errors below 0.000002. Leaf projection separately takes
0.248 ms. These are serial warmed single-layer kernel diagnostics, not model
wall times or an additive attention-time attribution.

The same real leaf stage **successfully captures and replays a graph** despite
its ordinary PyTorch temporary-allocation calls. `trained-leaf-fixed-graph.json`
(job 20838) measures 1.409 ms ordinary versus 1.402 ms graph replay: essentially
unchanged. Thus allocation/launch overhead is not a demonstrated major cost
for that warmed leaf stage. This proves only fixed-input leaf-stage capture,
not dynamic serving or whole-model prefill capture.

The full-model GPU profiler attempt (job 20836) fails inside the ROCm profiling
stack with `HSA: read-modify-write on a device resident signal value word is
not supported`. It produced the capture before failing, but **no valid full-model
profiler result**. Do not interpret or report that failed diagnostic as a
canonical speed measurement. The matched 20828/20833 measurements above were
completed separately, without any profiler or capture hooks.

An event-only full-model diagnostic (20851,
`oct4-trained-prefill-stage-diagnostic.json`) subsequently ran without the
crashing GPU profiler. It synchronizes per attention layer and prints phase
events, so its wall-time fields are deliberately **not canonical timings**.
Those intervals also include competing local attention and stream/collective
waits; they cannot be summed or treated as exclusive kernel costs. It predates
the carrier/sink fix above. This limitation prevents using its unusually long
"route" intervals as an attribution of the entire model slowdown.

### Fixed-shape prefill graph trial (not promoted)

The opt-in `LOD_KIMI_GRAPH_PREFILL=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1`
prototype uses vLLM's breakable graphs to capture model work between attention
calls. LoD writes each eager attention result into the caller's static graph
output. Cache plans and construction are re-evaluated on every replay; they
are **not** captured. This retains global per-request 16K prefill / 256 decode
cadences, the 16K scheduler chunk, exact first chunk, top eight, direct-key
channels, and centroid replacement accounting.

| Prefill graph trial | 32K dense (s) | 32K LoD (s) | 64K dense (s) | 64K LoD (s) |
|:--|--:|--:|--:|--:|
| Capture 16,384 and 16,392 token shapes | 33.454 | 37.199 | 70.358 | 74.159 |
| Capture only 16,384 token shape | — | — | — | 74.231 |

Sources: `oct4-full-fixed-prefill-graph-b8.json` (20846),
`oct4-lod-fixed-prefill-graph-b8.json` (20843), and
`oct4-lod-single-prefill-graph-b8-64k.json` (20848). The first row is a
matched graph-enabled pair: eight actual ProLong prompts, TP8/DCP8/EP8,
3 GiB native cache/rank, resident daemon, identical forced continuation,
one exact-shape warmup, one measurement, and final construction completion
included. Jobs ran sequentially with no timed process overlap. Worker audits
verify `FULL_AND_PIECEWISE`, the exact v13/refinement route binary, 24 LoD MLA
layers, and **94 graph segments / 93 eager attention breaks** per captured
large prefill descriptor. All eight output hashes repeat within each mode;
the forced trace is not a quality check or amortized decode measurement.

Capturing only one shape did not recover speed. Its dense cell is deliberately
blank: the dense control captured two shapes, so it is not relabeled as a
one-shape run. The earlier default-path pair remains 70.538 / 72.259 s at 64K.

Both LoD graph trials report less free memory than the default-path control.
For the two-shape 64K run, rank-zero free memory before allocator handling is
6.84 GB, versus 10.13 GB without prefill capture. The graph trial consequently
reclaims blocks under the unchanged 8 GiB headroom guard; the default control
retains them. The one-shape trial still reclaims on seven of eight ranks.
These are observed memory/allocator differences, **not** a proved attribution
of the entire wall-time regression. Graph capture remains experimental.

To reproduce the current one-shape variant, use the warm-serving command above
at 65536 with 3221225472 native bytes, adding the two graph variables before
starting Python. The preceding two-shape trials used the prototype before
removing its additional `16K + batch_size` descriptor. Their explicit sizes
are retained in each JSON's worker audit.

## Matched results

| Batch | Context | Full prefill | LoD prefill | Prefill speedup | Full decode | LoD decode | Decode speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16K | 2.039 s | 2.145 s | 0.95x | 21.38 ms | 20.65 ms | 1.04x |
| 1 | 32K | 4.182 s | 4.398 s | 0.95x | 21.80 ms | 20.64 ms | 1.06x |
| 1 | 64K | 10.489 s | 9.220 s | 1.14x | 22.15 ms | 20.65 ms | 1.07x |
| 1 | 128K | 19.293 s | 18.042 s | 1.07x | 22.29 ms | 21.43 ms | 1.04x |
| 1 | 256K | 45.519 s | 49.999 s | 0.91x | 23.32 ms | 25.03 ms | 0.93x |
| 8 | 16K | 17.240 s | pending | -- | 31.42 ms | pending | -- |
| 8 | 32K | 35.346 s | pending | -- | 32.72 ms | pending | -- |
| 8 | 64K | 73.895 s | 80.079 s | 0.92x | 33.39 ms | 28.22 ms | 1.18x |

Speedup is `full / LoD`, so values greater than one favor LoD.

### Fixed-budget batching probe

A dense full-model control divided the same 16K aggregate scheduler budget
across eight live requests (B8 x 2K per model pass).  Eight 16K-token prompts
completed prefill in 16.032 s, or 8,175 aggregate prompt tok/s.  The ordinary
16K-chunk B8 control took 17.240 s (7,603 tok/s), while the B1 16K control took
2.039 s (8,035 tok/s).  Thus B8 x 2K fits and modestly improves concurrent
throughput, but does not unlock a new full-model throughput regime: the MoE
stack is already well occupied by one 16K flattened-token pass.  The source is
`full-tp8-dcp8-b8x2k-16k.json`.

### B2 x 16K MoE tuning

Two simultaneous 16K chunks give the K3 MoE a 32,768-token flattened input,
which is beyond the largest shape in AMD's shipped K3 tuning table.  The
untuned path therefore used a 64x128x128 stage-one kernel followed by an atomic
stage-two reduction.  A targeted search selected the same 64x128x128 stage-one
geometry with `bnt0`, but changed stage two to the 64x128x128 reduce kernel.
The merged configuration retains every shipped row for smaller shapes.

| Measurement | Untuned | Tuned | Speedup |
| --- | ---: | ---: | ---: |
| Isolated 32K-token FMoE pair | 87.211 ms | 70.292 ms | 1.24x |
| Full model, B2 x 16K prompt | 8.165 s | 5.770 s | 1.42x |
| Full model, B2 x 64K prompt | 17.209 s | 16.744 s | 1.03x |

The 64K figures are medians of three measured passes on the same node, after a
separate warmup pass; the untuned samples were 18.393, 17.200, and 17.209 s,
while the tuned samples were 19.249, 16.725, and 16.744 s.  This resolves the
opposite result from the earlier single-sample comparison: the tuned kernel is
modestly faster once startup variation is removed.  Its smaller end-to-end
gain at 64K is expected because attention and the rest of the model occupy a
larger fraction of the four-cohort request.

The B2 setting does not reduce persistent cache capacity.  Both variants kept
the explicitly requested 5.0 GiB native cache on every GPU, corresponding to
1,179,972 cached tokens or 18 concurrent 65,554-token requests.  A B4 x 16K
probe still failed before cache allocation because the transient MoE workspace
needed another 7 GiB with only 3.70--5.23 GiB free; shrinking the persistent
cache to 1 GiB therefore did not make that shape viable.

The reusable tuner entry points are `benchmarks/tune_kimi_k3_fmoe.py` and
`benchmarks/_kimi_k3_fmoe_tuner.py`.  The selected row and merged runtime table
are `kimi-k3-i4-b2x16k-tuned-fmoe.csv` and
`kimik3_i4_tuned_fmoe_b2x16k_merged.csv`, respectively.

The current update interval is **256 global sequence tokens per request**,
not 256 rank-local records. DCP8 therefore contributes approximately 32 local
records per update, and B8 processes eight requests' shares. A current
1,025-token trace spans four global update periods; DCP must not silently
stretch that interval to 2,048 global tokens. Earlier records using the
rank-local cadence do not validate the current decode-update amortization.
Consult each JSON's protocol and implementation audit rather than assuming
that every historical row used the current rule. The dense 16K--32K controls use
1,025 tokens; some longer dense-only baselines use 258 tokens as identified
below because dense attention has no periodic LoD update. The batch-eight rows
additionally use synchronized decode so request staggering does not change the
effective batch size.

The current 256K LoD prefill entry uses final-only allocator reclamation and a
single cross-layer max-sim/update workspace reused by all six four-layer
groups. The 1,025-step decode entry comes from that same bounded construction
path. Reclamation changes only host synchronization and releases unused
allocator blocks, not the constructed LoD state or decode kernels.

The older LoD 16K--64K measurements are intentionally omitted. Their warmup
request was not removed from the fixed LoD pool by vLLM 0.30's concrete
`DefaultModelState.remove_request` override, so the measured request reused
stale semantic state. The runtime now patches that concrete lifecycle hook.
The replacement 16K--64K rows above use fresh state in one warmed process and
the same 1,025-token fixed decode trace as the 128K and 256K measurements.

## Dense baseline sweep

The dense reference now covers every requested power of two from 16K through
512K at both batch sizes, plus 1,020K at batch one. These are the baselines for
the long-context LoD work below.

| Context | B=1 prefill | B=1 decode | B=8 prefill | B=8 decode |
| ---: | ---: | ---: | ---: | ---: |
| 16K | 2.039 s | 21.38 ms | 17.240 s | 31.42 ms |
| 32K | 4.182 s | 21.80 ms | 35.346 s | 32.72 ms |
| 64K | 10.489 s | 22.15 ms | 73.895 s | 33.39 ms |
| 128K | 19.293 s | 22.29 ms | 163.268 s | 37.08 ms |
| 256K | 45.519 s | 23.32 ms | 380.950 s | 40.90 ms |
| 512K | 123.212 s | 24.68 ms | 1005.590 s | 46.29 ms |
| 1,044,480 (~1.02M) | 352.355 s | 27.37 ms | pending | pending |

The first B=8/1.02M attempt had enough native KV capacity (8.80M tokens for
8.36M requested), but left only 20 MiB free for an RCCL collective and failed
with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. A repeat with a minimally sized
native cache also failed because the 16K scheduler activation left only 12 MiB
free. The table deliberately does not substitute either failed attempt for a
timing. The separate B=8/64K LoD control also exposed that 128 MiB is too small
for eight recurrent-cache rows (six were allocated), while 256 MiB permits
only 4.33 concurrent 64K requests. Its valid retry therefore uses 512 MiB.
The next 1.02M attempt will retain the matched model/cache configuration while
using an 8K scheduler chunk to lower transient activation memory.

## Historical LoD prefill diagnosis

The earlier controlled diagnostic uses the 24-layer MLA-only stack (all
K3 attention layers, with the MoE/FFN work removed).  This is a better proxy
for cross-layer construction and synchronization than the earlier one-layer
fixture, while remaining much faster to iterate than the complete model.

| Batch | Context | Dense MLA-stack prefill | LoD MLA-stack prefill | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 16K | 1.162 s | 1.434 s | 0.81x |
| 8 | 64K | 9.534 s | 9.054 s | 1.05x |
| 1 | 128K | 3.929 s | 3.738 s | 1.05x |
| 1 | 256K | 14.370 s | 9.944 s | 1.45x |

These rows show the intended trend: LoD crosses dense between 16K and 64K in
the true-batch-eight fixture and its advantage grows at longer context.  They
come from `../kimi-k3-mla-stack/full-tp8-dcp8-b8-true-cohort.json`,
`../kimi-k3-mla-stack/two-tier-tp8-dcp8-b8-true-cohort.json`,
`../kimi-k3-mla-stack/full-tp8-dcp8-b1-long.json`, and
`../kimi-k3-mla-stack/two-tier-tp8-dcp8-b1-group4-nomidreclaim-long.json`.

### Historical one-layer geometry diagnostic

A correctness-fixed, full-K3-geometry one-layer fixture initially measured
2.082 s for direct latent-space leaves versus 2.605 s for projected leaves at
256K, but those cold requests included different Triton compilation costs. The
valid warmed comparison reverses that conclusion: direct latent leaves take
1.819 s, the original corrected projected path takes 1.161 s, and a new
coalesced D128+D64 key-packing path takes 1.114 s. The automatic switch to the
compact latent consumer after 64K has therefore been removed. These remain
one-layer diagnostics rather than full-model performance claims; the new path
is being checked end to end below.

The historical batch-eight one-layer sweep also includes the corrected
request-state lifecycle. Earlier warmed fixture runs leaked the completed
request's LoD row into the measured request and are not valid fresh-prefill
measurements. With fresh state, that projected/packed path measured:

| Context | Dense fixture prefill | Current LoD fixture prefill | Speedup |
| ---: | ---: | ---: | ---: |
| 64K | 1.015 s | 2.318 s | 0.44x |
| 256K | 11.213 s | 7.192 s | 1.56x |

This older fixture established improving asymptotic behavior and a crossover
between 64K and 256K. It is retained as a kernel-geometry diagnostic, not as
the current headline result.

A full-model boundary probe then found that allocator reclamation was forcing
the background cross-layer builder to synchronize after every 16K scheduler
chunk. In the measured 64K pass, the first three waits were 1.835 s, 1.954 s,
and 2.001 s; the final wait was already effectively zero because the work had
finished. Retaining the 16K attention/update schedule but reclaiming transient
allocations only at 64K boundaries reduced B=1/64K prefill from 9.411 s to
8.918 s (5.2%). The initial form exhausted transient VRAM at longer contexts;
a 32K compromise reached 256K at 52.555 s.

The current implementation instead reuses one maximum-capacity score/update
workspace across the sequential four-layer groups and drops it at the final
DCP conversion. That bounds allocation growth without periodic reclamation:
the 256K run completes with final-only reclamation in 49.999 s. Profiling shows
ordinary 16K state construction takes about 20--33 ms GPU and 9--12 ms of host
submission across all 24 attention layers. It is therefore not the remaining
multi-second full-model prefill bottleneck.

## Longer but unmatched measurements

| Mode | Batch | Context | Prefill | Decode | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Full attention | 8 | 128K | 163.268 s | 37.08 ms | 258-step synchronized decode |
| Full attention | 8 | 256K | 380.950 s | 40.90 ms | 258-step synchronized decode |
| Full attention | 8 | 512K | 1005.590 s | 46.29 ms | 258-step synchronized decode |
| Full attention | 1 | 512K | 123.212 s | 24.68 ms | 1,025-step decode |
| Three-tier INT4 LoD | 1 | 512K | 146.529 s | 30.52 ms | 2-step fit/correctness timing |
| Full attention | 1 | 1,044,480 (~1.02M) | 352.355 s | 27.37 ms | 1,025-step decode |

The matched table stops at 64K for B=1 and 32K for B=8 because longer LoD runs
were short kernel-development diagnostics, not because these lengths were
shown not to fit. Dense B=1 full attention successfully ran at 512K and about
1.02M; dense B=8 successfully ran through 512K. A matched long-context LoD
sweep therefore remains to be run. The dense B=8 1.02M allocation/timing test
is still pending.

Three-tier INT4 now completes the full-model 512K request. Releasing the shared
state-update/max-sim workspace before final rank-local dequantization removes
the former RCCL out-of-resources failure. After conversion each rank reports
9.006 GB allocated and about 24.4 GB free. This run is a fit and correctness
measurement rather than a competitive timing result: its 146.529 s prefill
and 30.52 ms two-token decode are slower than dense full attention. Longer
decode is still required before quoting a steady-state INT4 decode number.

## 128K-to-256K decode follow-up

The current full-model two-tier artifacts increase from 21.43 ms/step at 128K
to 25.03 ms/step at 256K. A matched 24-layer MLA-only run isolates the entire
LoD attention stack from K3's dynamic MoE/FFN stack: it rises only from
2.8863 ms to 2.9756 ms over the same context change, an increase of 0.0893 ms
(3.1%). Thus the full-model bend is not evidence of a coarse-attention tile or
LoD scan regression. The natural 128K and 256K prefixes produce different
hidden-state and expert-routing traces, while attention itself is only about
3 ms of the 21--25 ms full-model step.

The consumer split screen is consistent with that result: over representative
6,560- and 8,960-row effective sequences, 64 and 128 splits both take about
0.050 ms, while 256 splits are slower. A direct 128-row tile is not viable in
the present absorbed-MLA kernel because it requires 128 KiB of shared memory,
above MI325X's 64 KiB workgroup limit. No short-context-only geometry override
is therefore enabled at 256K.

The construction-memory problem that previously prevented this comparison is
also resolved. Reusing one maximum-capacity state workspace across sequential
layer groups reaches 256K with final-only reclamation. Construction-only
scratch is dropped before the final global-to-rank-local DCP conversion and is
not retained during decode.

## Source artifacts

The matched rows above come from:

- `full-tp8-dcp8-b1-short-v5.json`
- `full-tp8-dcp8-b1-512k-1020k.json`
- `full-tp8-dcp8-b8-16k-32k-sync.json`
- `full-tp8-dcp8-b8-64k-short.json`
- `full-tp8-dcp8-b8-128k-256k-short.json`
- `full-tp8-dcp8-b8-512k-short.json`
- `full-tp8-dcp8-b1-128k-256k-short.json`
- `full-tp8-dcp8-trueb2-16k-64k.json`
- `full-tp8-dcp8-trueb2-16k-64k-moe-tuned.json`
- `full-tp8-dcp8-trueb2-64k-moe-ab-baseline.json`
- `full-tp8-dcp8-trueb2-64k-moe-ab-tuned.json`
- `../kimi-k3-full-geometry/256k-compact-correct.json`
- `../kimi-k3-full-geometry/256k-expanded-correct.json`
- `../kimi-k3-full-geometry/256k-compact-warm-m32n32w4.json`
- `../kimi-k3-full-geometry/256k-expanded-warm-correct.json`
- `../kimi-k3-full-geometry/256k-packed-key-warm.json`
- `../kimi-k3-full-geometry/prefill-b8-dcp8-two-tier-packed-current.json`
- `lod-tp8-dcp8-b1-128k-decode1025-current.json`
- `lod-tp8-dcp8-b1-256k-decode1025-current.json`
- `lod-tp8-dcp8-b8-64k-decode1025-current.json`
- `lod-tp8-dcp8-b1-64k-reclaim-profile.json`
- `lod-tp8-dcp8-b1-64k-reclaim64k.json`
- `lod-tp8-dcp8-b1-256k-reclaim32k.json`
- `schema14-lod-b1-16k-64k-current.json`
- `schema10-lod-b1-128k-group4-decode1025.json`
- `schema10-lod-b1-256k-group4-reused-state-workspace.json`
- `schema13-int4-lod-b1-512k-prefinal-release.json`
- `../kimi-k3-mla-stack/lod-s64-b1-128k-decode1025.json`
- `../kimi-k3-mla-stack/lod-s64-b1-256k-decode1025.json`

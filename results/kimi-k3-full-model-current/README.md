# Kimi K3 full-model attention timings

The opening section contains the current corrected full-model timing checks;
later sections retain historical measurements with their caveats. Current
**fixture-only** comparisons and their audits are in
[the MLA-stack results](../kimi-k3-mla-stack/README.md),
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

## Latest corrected full-model timings (October 4)

### 512K / 1020K B1 extension and decode audit

The requested long B1 extension uses **1,025 generated tokens**, giving
**1,024 timed decode steps** after the prefill-produced first token. The
timing window therefore spans four 256-global-token update periods; it does
not quote the single decode step from the prefill-only sweep as throughput.
Both modes use the same real ProLong prompt/continuation token hashes, one
exact-shape warmup and one measured pass, TP8/DCP8/EP8, and 16K scheduler
chunks. Prefill includes final cache completion before first token. Decode
is measured from first-token to last-token request timestamps, with the
whole-generation wall clock recorded separately for a consistency check.
No profiler or per-layer timing events are inserted into the measurement.
Historical decode rows below are not automatically promoted to this panel.

Two 512K attempts failed during warmup at the second 16K scheduler chunk:
21010 used the current final-only reclamation policy, and 21011 reclaimed
after every 16K chunk. Both reserved 1 GiB/rank for the native descriptor.
RCCL reported zero physical free memory; neither attempt produced a valid
timing. Allocator reclamation alone did not solve the failure.

The replicated prefill shadow initially reserves the entire prompt archive:
a metadata-only allocation audit finds 14.366 GiB/rank at 512K and 28.982
GiB/rank at 1020K across 24 MLA layers, even while only the first 16K tokens
have been processed. These numbers exclude weights, persistent rank-local
decode pools, state-update scratch and model activations. An opt-in fit test
uses `LOD_KIMI_PREFILL_SHADOW_GROW_CHUNK=65536` to allocate this temporary
archive in 64K slabs as needed, bounded by the prompt capacity. This changes
neither state/centroid schedules nor the set of visible keys. Archive growth
preserves shared latent K/V storage instead of allocating a separate V copy;
fixed-address decode pools are unchanged. The default remains full reservation.

The GPU insertion test compares growing and preallocated archives and checks
identical leaf data, page lists and lengths, and latent K/V storage aliasing:
four tests pass (21013). The K3 plus benchmark CPU suites pass 266 tests with
35 GPU skips. The growing-archive full-model test (21014) advanced to 192K
computed tokens, then exhausted memory during warmup. A further test (21015)
reduced the native descriptor to 128 MiB/rank and allowed HSA scratch
reclamation (`HSA_NO_SCRATCH_RECLAIM=0`), but still exhausted physical memory
during warmup. Neither produced a valid prefill or decode result. These are
recorded in `oct4-b1-long-capacity-failures.json`; 512K two-tier LoD has **not**
yet been shown to fit, and 1020K LoD has not been measured.

Fresh dense B1/512K and B1/1020K baselines completed in 21017, using 1,025
generated tokens, the improved Gluon dense decoder, one warmup and one
measurement, and a 4 GiB/rank native cache. The historical long dense rows
below remain historical; the completed audited results are reported here.

The completed source is `oct4-full-b1-512k1020k-decode1025.json`:

| Context | Dense prefill (s) | Dense decode (ms/batch step) | LoD prefill | LoD decode |
|--:|--:|--:|:--|:--|
| 512K | 118.730 | 24.686 | Warmup OOM | Not measured |
| 1020K | 344.493 | 27.378 | Not measured | Not measured |

The 512K point has a 25.278921 s decode window / 1,024 steps and a
144.025411 s whole-generation wall time. Prefill + decode is 144.008690 s;
the remaining 0.016721 s is untimed API/transport overhead. All eight worker
audits confirm real dense attention with the Gluon decoder installed and
`FULL_DECODE_ONLY` graph capture, no dummy attention, zero preemptions and
zero prefix-cache hits. The prompt hash is
`4312e2120861896344fd516cb5f0f94fb885130c879d3f648ed74c5465aabfda`;
the 1,025-token natural continuation hash is
`b621d1d69218d5d4cc63fed239d1f9dd293dd1666d792921cd3e07fb7f6c05cc`.
The 1020K decode window is 28.035058 s / 1,024 steps; its whole-generation
wall time is 372.561775 s, with 0.034157 s outside the request-metric window.
It likewise has zero preemptions/cache hits and real captured dense decode.
The long dense decode means closely reproduce the historical 24.68/27.37 ms
figures, while their prefill times are somewhat lower. This validates these
dense decode points; it does not validate today's LoD
decode or turn the prefill-only sweep into a decode comparison.

### Current 64K decode checks

Fresh 64K B1/B8 comparisons use the corrected sink, the prefill sweep's
runtime retention policy, 1,025 natural-trace output tokens, one full-shape
warmup and one measured pass. Both modes use the same native reservation
within each pair (1 GiB/rank at B1, 3 GiB/rank at B8) and run sequentially on
node 4. Existing per-layer catch-up counters are now read immediately before
and after measured generation through an out-of-band worker RPC. No per-step
instrumentation, GPU events, profiler, or additional synchronization is added
inside the timed generation. The difference is stored in each measurement's
`measured_decode_update_counters`, separately for every worker and layer.

These counters verify that state updates actually ran. The latency itself
still comes from request timestamps, not from summing component timings.
A 1,024-step window spans four periods, but can include three catch-up calls
when its initial state is already caught up and the next boundary is just
past its last input. Observed counters, not an assumed count of four, are the
audit. The counter reader's six unit tests pass; the shared benchmark suite
also passes 33 tests, and an integration test verifies that counter RPCs are
outside generation and that warmup updates are excluded.

The completed matched B1 pair is 21018/21019:
`oct4-lod-b1-64k-decode-update-audit.json` and
`oct4-full-b1-64k-decode-update-audit.json`.

| Context / batch | Dense prefill (s) | LoD prefill (s) | Prefill speedup | Dense decode (ms/step) | LoD decode (ms/step) | Decode speedup |
|:--|--:|--:|--:|--:|--:|--:|
| 64K / B1 | 8.783827 | 8.508550 | 1.032x | 22.173212 | 20.641170 | 1.074x |
| 64K / B8 | 70.467597 | 68.235140 | 1.033x | 34.179869 | 29.363706 | 1.164x |

Each of the 24 MLA layers on all eight LoD workers records three catch-up
batches and three updated request rows during measured generation. The LoD
decode window is 21.136558 s / 1,024 steps; dense is 22.705369 s / 1,024.
Their wall/metric discrepancies are 0.003019 s and 0.003051 s respectively.
Both have zero preemptions and prefix-cache hits. Prompt and natural
continuation hashes, scheduler parameters, native reservation, seed and
timing protocol match exactly. Dense has the improved Gluon decoder
installed, and both use `FULL_DECODE_ONLY` graph capture without dummy
attention. This closely reproduces the earlier ~20.66 ms B1 LoD measurement,
this time with explicit update execution verification.

The matched B8 pair is 21020/21021:
`oct4-lod-b8-64k-decode-update-audit.json` and
`oct4-full-b8-64k-decode-update-audit.json`. Prompt and continuation hashes,
seed, scheduler, native reservation and timing protocol match exactly.
All eight requests are live for the entire measured decode window in each
mode and finish simultaneously: 30.068435 s for LoD and 35.000186 s for dense.
Each of the 24 MLA layers on every rank records three catch-up batches and
24 updated request rows: three per request, not three shared across the
batch. The wall/metric discrepancies are 0.021077 s for LoD and 0.016645 s
for dense, with zero cache hits and preemptions. Both modes use
`FULL_DECODE_ONLY` capture; dense uses the improved Gluon decoder. The B8
warmup decode means are 29.634905 ms for LoD and 34.165425 ms for dense,
close to their measured values. This panel reports one measured pass per
point, not a confidence interval or a guarantee for every serving workload.

**What the audit establishes:** the recent two-token sweeps are prefill-only
evidence, not steady-state decode benchmarks. The fresh B1/B8 pairs measure
1,024 steps and explicitly verify update execution without inserting timing
events inside CUDA graphs. They confirm the earlier ~20.66 ms B1/64K LoD
figure and establish the matched speedups above. Three observed updates in
this finite window are included in the average; a window containing exactly
four updates could differ by one catch-up cost divided by 1,024. Historical
decode rows without this audit remain historical rather than being silently
treated as measurements of the current implementation.

### Reproducing the decode audit

These commands run directly, without `cluster-run`. They require eight
MI325X GPUs, the unpacked K3 v10 image used by the wrapper, the checkpoint
staged on local disk, and the resident packed-weight daemon with cache ID
`kimi-k3-shared-int4-v6`. The exact argument list, effective worker settings,
prompt hashes and timing protocol are also stored in each result JSON.
Run modes sequentially; do not overlap another inference job on these GPUs.

```bash
export LOD_BENCHMARK_SYNC_PREFILL_CACHE=1
export TRITON_CACHE_AUTOTUNING=1 VLLM_USE_TRITON_AWQ=1
export PROLONG_SPEED_TOKEN_CACHE="$PWD/results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt"
export AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv"
k3_bench=(benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.prolong
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06
  --measure speed --tensor-parallel-size 8 --decode-context-parallel-size 8
  --dcp-comm-backend ag_rs --decode-tokens 1025 --fixed-decode-trace
  --synchronized-decode --repeats 1 --seed 0 --gpu-memory-utilization 0.8
  --kimi-gfx942-int4-moe --weight-cache --weight-cache-id kimi-k3-shared-int4-v6
  --allow-experimental-environment --retain-warmup-allocator)

# B8 matched dense and LoD checks; B1 uses batch/samples 1 and 1073741824 bytes.
"${k3_bench[@]}" --mode full --lengths 65536 --batch-size 8 --speed-samples 8 \
  --kv-cache-memory-bytes 3221225472 --output /tmp/k3-full-b8-64k-audit.json
env LOD_KIMI_SUBTILE64=score LOD_KIMI_CHUNK_TILE_PACK=1 \
  LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 LOD_KIMI_SORT_LEAF_ROUTES=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_PREFILL_MIN_FREE_GIB=4 LOD_KIMI_TILE_REFINE=1 \
  LOD_KIMI_DIRECT_LEAF_RESULT=1 \
  "${k3_bench[@]}" --mode two-tier --lengths 65536 --batch-size 8 --speed-samples 8 \
  --kv-cache-memory-bytes 3221225472 --output /tmp/k3-lod-b8-64k-audit.json

# Long B1 dense control. LoD has not completed these lengths.
HSA_NO_SCRATCH_RECLAIM=0 "${k3_bench[@]}" --mode full \
  --lengths 524288,1044480 --batch-size 1 --speed-samples 1 \
  --kv-cache-memory-bytes 4294967296 --output /tmp/k3-full-b1-long-audit.json
```

### B1 / B8 scaling sweep

The current sweep measures **prefill only**, at 16K, 32K, 64K, 128K and
256K, using the corrected sink and the 4 GiB runtime retention reserve.
Each point has one exact-shape warmup and one measured pass on actual
ProLong tokens, with final LoD cache construction included before first
token. The two-token forced continuation is not an amortized decode test.
Full and LoD run sequentially on node 4, sharing the idle packed-weight
daemon, TP8/DCP8/EP8 and the same logical 16K scheduler chunks.

The fresh matched B1 sweep and B8 16K/128K checks are complete.
No estimated timings are included below.

| Context | B1 full prefill (s) | B1 LoD prefill (s) | B1 full / LoD | B8 full prefill (s) | B8 LoD prefill (s) | B8 full / LoD |
|--:|--:|--:|--:|--:|--:|--:|
| 16K | 2.0375 | 2.0368 | 1.000x | 16.3863 | 16.4093 | 0.999x |
| 32K | 4.1805 | 4.1838 | 0.999x | 33.5666 | 33.6356 | 0.998x |
| 64K | 8.7910 | 8.5136 | 1.033x | 70.5637 | 68.4368 | 1.031x |
| 128K | 19.3016 | 17.3786 | 1.111x | 155.4086 | 143.8071 | 1.081x |
| 256K | 45.5186 | 36.0147 | 1.264x | — | Does not fit | — |

B1 sources: `oct4-full-prefill-scale-b1-1g-16k256k.json` (21006) and
`oct4-lod-prefill-scale-retention4g-b1-1g-16k256k.json` (21005). Both reserve
1 GiB/rank for the native cache; all prompt/continuation records and timing
protocol fields match directly. Their single natural-token prompt uses
nested prefixes through 256K. B8 32K/64K sources are the matched
20952/20997 pair below, with 3 GiB/rank and eight prompts through 64K.
Their full prompt/continuation records were checked against the short-panel
generator and match exactly; these baselines are reused, not rerun merely
because the development commit changed. Longer B8 prompts concatenate more
distinct documents and therefore require new matched full/LoD measurements.
The native cache reservation is identical within the B1 and short B8 pairs.
For long B8, each mode's reservation must support eight resident requests:
LoD's native descriptor covers its remaining chronological window, while
remote history resides in its own cache. Reserving the dense pool size for
LoD unnecessarily consumes headroom. At 128K/B8, full reserves 5 GiB/rank
(capacity 10.080 requests) and LoD 1 GiB/rank (capacity 17.333 requests).
These are native-pool reservations, **not total cache VRAM**: LoD also owns
its remote-history cache. Prompt/continuation records, maximum model length,
timing protocol, seed, scheduler budget, TP/DCP and model weights match.
Sources: `oct4-full-prefill-scale-b8-5g-128k.json` (21009) and
`oct4-lod-prefill-scale-retention4g-b8-1g-128k.json` (21008).
The completed B8 16K pair is
`oct4-full-prefill-scale-b8-3g-16k.json` and
`oct4-lod-prefill-scale-retention4g-b8-3g-16k.json` (both 21009), with
identical prompt/continuation metadata, timing protocol and native reservation,
zero preemptions/cache hits, and all eight requests live during decode.

The 256K-capacity B8 LoD engine fails even during its first 128K warmup
request, after 81,920 computed tokens, with zero physical free memory at an
RCCL launch (21007). Sizing the engine for 128K lets the entire B8 pair run.
Thus 256K/B8 is a **capacity failure for this configuration**, not a timing
point, and no fresh 256K/B8 dense rerun is needed to form a nonexistent
speedup. Neither chunking nor attention rules were changed to force a fit.
The B8 128K control is a separate, matched natural-document cohort; the
B8 short and long cohorts are not identical across lengths.

**Capacity caveat:** an initial B1 LoD sweep with an overprovisioned 5 GiB
native pool failed in the 256K warmup. An isolated fresh 256K run also failed
around 192K, reporting zero physical free memory during an RCCL launch.
These are capacity failures, not timings; see
`oct4-prefill-scale-capacity-failures.json`. Reducing the native reservation
to the audited one-request size fixes the B1 failure without changing
attention math, routing, model weights or either update cadence. The matched
1 GiB dense pool has capacity 1.087 requests at the maximum context;
the LoD descriptor has capacity 17.333. The preliminary 5 GiB dense sweep
(`oct4-full-prefill-scale-b1-16k256k.json`, 21002) agrees closely with the
current dense timings but is not the table's comparator.

LoD retains at all ten runtime reclamation checks per worker across the B1
sweep, with minimum observed headroom 4.22 GiB. The separate, common
post-warmup 8 GiB check reclaims on all ranks at 128K and five ranks at 256K;
this remains part of the stated matched protocol, not an unrecorded change.
There are no preemptions or prefix-cache hits. The prefill speedup increases
from a tie at 32K to 1.264x at 256K; no decode-speed claim follows from this
two-token trace. At B8/128K, the runtime pressure guard reclaims at 117 of
128 worker/request checks across warmup and measurement, retaining at only
11; minimum recorded free memory is 2.068 GiB. The common post-warmup check
reclaims on one rank. Both modes keep all eight requests resident through
the decode barrier, with zero preemptions and cache hits. The lower reserve
does not guarantee retention or a fit at arbitrary long-context capacities;
the general default remains 8 GiB.

The benchmark now saves audited completed points to a sibling
`.partial.json` after each length, outside timed generation, so a later
capacity failure does not discard valid shorter results. A completed sweep
still writes the usual final JSON; the partial artifact alone is explicitly
marked as an incomplete sweep. Benchmark unit tests pass 33 cases, including
preservation of a completed point when a later shape raises an exception.

To reproduce B1, use the development command below with
`--lengths 16384,32768,65536,131072,262144 --batch-size 1 --speed-samples 1
--kv-cache-memory-bytes 1073741824`. For the B8 128K pair, use
`--lengths 131072 --batch-size 8 --speed-samples 8`, with cache reservations
`5368709120` for full and `1073741824` for LoD. The B8 16K pair uses
`--lengths 16384`, B8 and `3221225472` in both modes. Change the output path
for each run. Dense uses `--mode full` and omits the LoD-only development
environment variables; the shared image, MoE configuration, warmup policy,
trace and scheduler flags stay the same. The token-cache `.pt` is an optional,
untracked startup optimization: omit `PROLONG_SPEED_TOKEN_CACHE` to tokenize
the same frozen shuffled ProLong dataset directly.

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

The following bounded-headroom experiment
(`oct4-lod-prefill-retention4g-b8-64k.json`, 20994) changes only the runtime
retention reserve from 8 GiB to **4 GiB**, using the development-only
`LOD_KIMI_PREFILL_MIN_FREE_GIB=4`. Completion fences and source release are
unchanged; reclamation still runs below the reserve. Across warmup plus
measurement, seven ranks retain blocks at all sixteen checks and one rank
reclaims once. The run completes without RCCL errors, preemptions or cache
hits; the recorded prompts, continuations, scheduler/cache reservations and
timing protocol match 20993.

Measured prefill is **68.440 s**, versus **71.499 s** with the 8 GiB guard
and **70.564 s** for matched dense attention: a 4.3% LoD latency reduction
and **1.031x** dense / LoD. This is the first corrected 64K/B8 full-model
check in this cohort to beat dense, but it needs repetition and a matched
32K check. The default remains 8 GiB pending those checks. This two-token
trace measures prefill, not amortized decode or new model quality.

The allocator policy tests pass seven cases, including exact reserve
boundaries and invalid reserve settings; the K3/shared-merge CPU suite
passes 230 tests with 39 GPU skips.

The independent matched repeat
(`oct4-lod-prefill-retention4g-b8-32k64k.json`, 20997) confirms this effect:

| Context, B8 | Matched dense (s) | LoD, 8 GiB reserve (s) | LoD, 4 GiB reserve repeat (s) | Dense / repeat LoD |
|--:|--:|--:|--:|--:|
| 32K | 33.5666 | 34.5297 | 33.6356 | 0.998x |
| 64K | 70.5637 | 71.3322 | 68.4368 | 1.031x |

Dense is 20952, the earlier 8 GiB score-only control is 20957. All prompt
and continuation records match directly, and the physical 16392-token
scheduler budget, logical 16K chunk, native cache reservation, seed,
TP8/DCP8 and timing protocol remain identical. The repeated 64K result is
within 0.004 s of 20994; 32K is effectively tied with dense, not a claimed
speedup. Across the repeat's two warmed/measured lengths, seven ranks retain
at all 32 checks and one reclaims once below 4 GiB. There are no preemptions,
prefix-cache hits or RCCL failures.

For the current best **development** configuration, use the score-only
command below with `--lengths 32768,65536`, the 3 GiB reservation,
`--retain-warmup-allocator`, and `LOD_KIMI_PREFILL_MIN_FREE_GIB=4`.
The default reserve remains 8 GiB; a long-context/memory-pressure panel
should precede changing the general serving default. Fixed-shape graph
experiments remain separate and disabled: replayability alone has not yet
demonstrated a transferable speed gain.

Exact current development command (existing local weight daemon and image
setup required; no cluster runner is needed):

```bash
env LOD_KIMI_SUBTILE64=score \
  LOD_KIMI_CHUNK_TILE_PACK=1 LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 \
  LOD_KIMI_SORT_LEAF_ROUTES=1 LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_PREFILL_MIN_FREE_GIB=4 \
  LOD_KIMI_TILE_REFINE=1 LOD_KIMI_DIRECT_LEAF_RESULT=1 \
  LOD_BENCHMARK_SYNC_PREFILL_CACHE=1 TRITON_CACHE_AUTOTUNING=1 VLLM_USE_TRITON_AWQ=1 \
  PROLONG_SPEED_TOKEN_CACHE="$PWD/results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt" \
  AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv" \
  benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.prolong \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --mode two-tier --measure speed --lengths 32768,65536 \
  --batch-size 8 --speed-samples 8 --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 --dcp-comm-backend ag_rs \
  --decode-tokens 2 --fixed-decode-trace --synchronized-decode \
  --repeats 1 --seed 0 --gpu-memory-utilization 0.8 \
  --kv-cache-memory-bytes 3221225472 --kimi-gfx942-int4-moe \
  --weight-cache --weight-cache-id kimi-k3-shared-int4-v6 \
  --allow-experimental-environment --retain-warmup-allocator \
  --output results/kimi-k3-full-model-current/oct4-lod-prefill-retention4g-b8-32k64k.json
```

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

At this checkpoint the corrected path had **not crossed over at 64K**. The
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

## Historical matched results (not the current decode panel)

These rows predate the latest corrected-sink prefill path and allocator
configuration. Several older JSONs also lack the current timing-protocol and
worker-audit fields. They are retained as historical measurements, not a
validated decode-speed claim for today's implementation. The October 4
scaling sweep above uses only two generated tokens and cannot replace them
as an amortized decode benchmark. A fresh long-context pair uses 1,025
generated tokens (1,024 timed decode steps), one exact-shape warmup and one
measurement, with final cache construction charged to prefill and the
256-global-token update cadence unchanged.

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

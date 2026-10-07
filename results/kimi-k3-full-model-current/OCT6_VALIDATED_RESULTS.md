# Kimi K3: October 6 validated results (superseded timing panel)

These measurements use the full trained Kimi K3 model on eight MI325X GPUs,
TP8/DCP8/EP8, resident weights, and real ProLong tokens. Both modes use the
same packed INT4 **MoE weights**; two-tier LoD stores attention K/V in BF16.
Dense decode uses our improved Gluon decoder, not the slower AMD precompiled
decoder. Fixture-only speedups are not included here.

Historical measurements, rejected experiments, and debugging notes have moved
to [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md). Raw artifacts are unchanged.

The node-4 full-model shared-latent centroid comparison on eight matched
65,267-token ProLong documents gives dense PPL **1.33104** versus two-tier
LoD **1.33285** (**+0.1361%**); after the first exact 16K block, the increase
is **+0.1997%**. See [PROLONG_QUALITY.md](PROLONG_QUALITY.md) for raw results,
per-document/position-band scores, audits, and commands. This checks the
existing shared assignments, not head-specific clustering or retrieval quality.
Matched LongBench v2 and RULER NIAH-S3 checks are being prepared in
[CHAT_QUALITY.md](CHAT_QUALITY.md), using K3's native non-thinking chat format.

## Decode

The latest default retains **four-wave centroid routing**, parallel split
reduction and fused distributed-route bookkeeping. It now aliases existing
cached centroid means instead of dividing sum keys on every decode step,
and uses 32 consumer splits at physical B1 / 16 at live B4+. Mean reuse
preserves candidate scores bit-for-bit; splitting can change output roundoff.
Top-eight choices, selected KV sets and global update cadences are unchanged.
The benchmark engine factory now automatically selects **one request's
attention per GPU for fixed B8 K3 two-tier cohorts at TP8/DCP8**. It retains
native TP Q/K/V/O, KDA and MoE, captures the real B8 decode graph, and uses
eight 2K scheduler slices without changing global 16K/256-token construction
cadences. Ordinary B1 remains DCP8. The explicit control is
`LOD_KIMI_REQUEST_OWNER_PREFILL=0` (or `--ordinary-dcp` in the K3 sweep/fixture).
The owner path is still an eight-live-request layout, not ragged continuous
batching; the free-generation quality runner deliberately retains DCP.
Pool-backed prefill and reused transport are enabled by default; long owner
prefill uses six-head projection slices and allocator-pressure checks.
The matched full-model follow-ups are:

| Context | Layout | Dense, ms/step | Current LoD, ms/step |
|--:|:--|--:|--:|
| 16K | B1 ordinary DCP8 | 21.397 | 21.631 |
| 16K | B8 one request's attention per GPU | 31.649 | 30.134 |
| 64K | B1 ordinary DCP8 | 21.845 | 21.626 |
| 64K | B8 ordinary DCP8 | 34.198 | 32.229 |
| 64K | B8 one request's attention per GPU | 34.198 | 30.872 |
| 128K | B1 ordinary DCP8 | 22.333 | 21.695 |
| 128K | B8 one request's attention per GPU | 37.994 | 31.938 |

These are end-to-end batched decode steps including four updates over
1,025 measured steps. At 64K, ordinary DCP is effectively tied with dense at B1
(1.0% lower in this pass) and 5.8% lower latency at B8. Request-owned attention
is 9.7% lower latency than dense and 4.2% lower than ordinary DCP8. Compared
with the preceding pipeline, gains are modest: 1.8% / 0.9% / 1.2% respectively
in these single-pass comparisons. Frozen inputs, actual graph execution and
four updates validate; dense controls are reused. See
[the current matched results and numerical checks](CACHED_ROUTING.md),
[the preceding pipeline results](DECODE_PIPELINE.md) and
[the preceding router results and commands](ROUTER_DECODE.md).
At 16K B1 is 1.1% slower than dense, while request-owned B8 is 4.8% faster.
At 128K B1 is 2.9% faster and request-owned B8 is 15.9% faster.
The first 128K owner run completed full warmup but
exhausted memory in MoE during its second generation; no warmup time is
reported as a measured result. Its completed storage-policy retry is documented in
[the context follow-up](CACHED_ROUTING.md#16k--128k-context-follow-up).
That retry's prefill takes 202.915 s versus dense's 155.750 s, including
56 idle-allocation reclamations per rank; the decoder speedup is not a
prefill speedup claim. The matched freely generated B1 NIAH-S3 check passes
**4/4 at both nominal 16K and 128K**, as does dense; all prompt hashes match.
See [the small retrieval check](CHAT_QUALITY.md#october-6-cached-routing-b1-check).
The [power-of-two panel](DECODE_POWER2.md) shows current kernels only:
ordinary B1 and row-per-GPU B8 at 16K/64K/128K. Old one-wave and ordinary-B8
LoD cells are removed, with unmeasured current contexts left blank.

**Earlier ordinary-LoD decode timings are invalid comparisons.** The October
6 chat audit found both an unwritten BF16 archive-tail read and a missing DCP
dispatch branch. Without replicated Q, decode incorrectly attended to only
one eighth of the history, without merging the sequence shards. Both bugs
are repaired; generation smoke tests passed and the corrected decode panel
has been measured below.
Dense controls, prefill-only measurements, and the teacher-forced ProLong
loss comparison are unaffected. See [CHAT_QUALITY.md](CHAT_QUALITY.md) for
the evidence. The completed single-owner B1 test measures only prefill.

The opt-in request-owner extension completed its B8/256K, 1,025-step
trained-model follow-up `21183-kimi-owner-decode-b8-256k-fixed` on node 4.
It replays the archived dense ProLong prompts/continuation and reuses the
resident daemon. Prefill took **347.569 s** and eager decode **157.650 ms/step**.
The archived dense control took 365.571 s and 47.405 ms/step, respectively.
That older eager owner decode is substantially slower and is not a recommended path.
This is an exploratory comparison: owner execution is eager, while dense
decode is graph-captured. All eight owners and all 24 MLA layers passed the
four-update/1,025-step audit and loaded-attention audit. A numerical test
caught and fixed an empty-softmax-split error; the initial fixture timings
were invalidated and the earlier full warmup cancelled. See
[owner experiment](REQUEST_OWNER_PREFILL.md#owner-decode-extension-october-5)
and [request-centric layout plan](REQUEST_CENTRIC_LAYOUT.md).

The pre-optimization [graph-captured B8 request-owner decoder](OWNER_CAPTURED_DECODE.md)
finishes the matched 16K test at **33.699 ms/step**, versus **37.933 ms**
ordinary LoD and **31.649 ms** dense. It reduces LoD latency by 11.2% but is
still 6.5% slower than dense here. All eight ranks recorded 1,025 actual
eight-row graph replays and four global-256 updates in every MLA layer.
This is a separate, opt-in execution layout; it does not replace the ordinary
DCP8 panel or the historical eager artifact.
The 64K follow-up takes **34.148 ms/step**, versus **38.209 ms** ordinary LoD
and **34.198 ms** dense: 10.6% less LoD latency and effectively tied with
dense. It passed the same actual graph-replay and four-update audits.
Reusing the owner's fixed decode cache for prefill reduces the matched
64K/B8 peak Torch allocation from **27.683 to 25.220 GiB/rank** (2.463 GiB
saved), with effectively unchanged speed. This excludes daemon weights and
driver memory. The first 256K backing-only fit retry still ran out of memory
after all eight rows reached 144K; idle-allocation reclamation advanced to
176K but did not complete. Halving the projection group from twelve to six
then **completed B8/256K prefill and one real decode step**, with no allocator
reclamation. Peak client Torch allocation was 29.814 GiB/rank. This is a
capacity result, not an amortized 256K decode timing; the 16K/64K comparisons
above remain the validated speed results.

A fresh **dense B8/256K** control also fits and completes warmed generation:
**364.752 s/batch prefill** and **48.090 ms/batch decode step**, using the
improved Gluon decoder and the same frozen ProLong cohort. All eight requests
stay live for all 1,025 measured decode steps, with no preemptions or prefix
hits. Dense keeps exact BF16 MLA cache entries sharded with DCP8; no KV
quantization is needed to fit this test. Source:
[fresh dense 256K control](oct6-full-expandable-b8-256k-warm-decode.json).
The per-GPU LoD follow-ups complete full-length warmup but fail when starting
a second generation, including the scratch-reclamation retry. Thus the capacity
success above is not yet a warmed 256K LoD speed comparison. Detailed memory
records and the failed scratch-reclamation retry are in
[the owner follow-up](OWNER_CAPTURED_DECODE.md#256k-warmed-comparison-follow-up).

The [decode table](DECODE_POWER2.md) excludes historical ordinary-LoD results;
the raw artifacts remain available for traceability. Its dense controls remain
usable. The repaired generation smoke tests passed. The corrected LoD panel
has completed B1 through 1020K and B8 through 128K, with four audited updates
per request. The 512K/1020K B1 points use token-sharded prefill as described
below. With the earlier one-wave profile, B1 through 256K was about
25.4--25.6 ms/step and B8 about 37.9--38.4 ms/step,
slower than dense at those matched contexts; the improved 64K cells are above.
B8/128K is nearly tied (38.371 versus 37.994 ms/step).
These superseded timings are historical, not the current default panel.

## Next optimization targets

The latest owner diagnostic, separate from serving timing, measures compact
absorbed attention at **31.97 us/layer**, centroid routing at **17.05 us**,
the split-output reducer at **4.50 us**, and value projection at **7.06 us**.
The first kernel target is therefore compact attention: improve tile/load
reuse and occupancy without changing the selected KV set. Next, consider
combining split-output reduction with the latent value projection/output
packing, removing an intermediate write/read and launch. Preserve the
current BF16 rounding point when validating that fusion. These are proposed
targets, not measured speedups; profiler durations are not additive serving
latency estimates. Source:
[separate 12-layer owner trace summary](../kimi-k3-mla-stack/oct6-owner-pipeline-cached-means/b8-owner-lod.json).

For long owner prefill, the immediate non-attention target is allocator
churn: the completed 128K measurement performed 56 idle-cache reclamations
per rank. Reusing/bounding model and MoE scratch should address that measured
pressure without changing LoD's update cadence or approximation.

The new panel generates **1,026 output tokens**, giving **1,025 timed decode
steps** after the first token produced by prefill. Each LoD request must record
exactly **four 256-global-token catch-ups** in every MLA layer on every rank.
Times are end-to-end milliseconds per **batched step**, including updates,
not attention-only time or milliseconds divided by batch size.

The already validated long dense B1 controls use 1,024 timed steps; dense
has no LoD catch-ups to amortize. The new token-sharded long B1 comparisons below
verifies the whole archived trace and appends one source token for LoD's
1,025-step/four-update protocol:

| Context | Dense prefill (s) | Dense decode (ms/step) |
|--:|--:|--:|
| 512K | 118.730 | 24.686 |
| 1020K | 344.493 | 27.378 |

Source: [long dense B1 controls](oct4-full-b1-512k1020k-decode1025.json).
The live decode table lists its own sources, including individually validated
completed points preserved from incomplete sweeps.

## Prefill

This matched **prefill-only** sweep uses the corrected sink calculation and
the 4 GiB LoD allocator-retention reserve. Prefill is total request-batch
latency, including final cache construction before the first output token.
Its two-token continuation is **not** an amortized decode benchmark.

| Context | B1 dense (s) | B1 LoD (s) | Dense / LoD | B8 dense (s) | B8 LoD (s) | Dense / LoD |
|--:|--:|--:|--:|--:|--:|--:|
| 16K | 2.0375 | 2.0368 | 1.000x | 16.3863 | 16.4093 | 0.999x |
| 32K | 4.1805 | 4.1838 | 0.999x | 33.5666 | 33.6356 | 0.998x |
| 64K | 8.7910 | 8.5136 | 1.033x | 70.5637 | 68.4368 | 1.031x |
| 128K | 19.3016 | 17.3786 | 1.111x | 155.4086 | 143.8071 | 1.081x |
| 256K | 45.5186 | 36.0147 | 1.264x | — | Warmup OOM | — |

Prefill is effectively tied through 32K. The measured B1 speedup grows to
1.264x at 256K; B8 reaches 1.081x at 128K. These are single measured passes,
not confidence intervals or quality evaluations.

The opt-in [compact selected-leaf projection](COMPACT_SELECTED_PREFILL.md)
preserves the existing post-cap top-eight math. It lowers the captured trained
leaf-stage latency by 17.9% and the 128K/24-layer fixture prefill by 8.1%, with
identical outputs on those checks. The full trained-model B1 follow-up is
effectively tied overall: 17.28 versus 17.39 s at 128K and 37.05 versus
37.11 s at 256K. It remains opt-in; fixture gains are not substituted into
this table. It reduces one projection arena, but still reserves its worst-case
capacity and did not lower full-model peak allocation or resolve B8/256K VRAM.

The October 6 [six 16-head owner prefill experiment](SIX_HEAD_OWNER_PREFILL.md)
keeps native TP8/EP8 model execution and improves on the slower B1
single-owner layout, but does not beat ordinary eight-rank LoD: at 128K it
takes 18.506 s, versus 17.391 s ordinary LoD and 19.300 s dense. At 32K it
takes 4.430 s versus 4.189 s ordinary LoD. The actual head transport, latent
replication, global cache coverage and loaded binaries passed validation.
256K warmup failed from GPU resource exhaustion; no decode was tested.
This remains opt-in and is not promoted as the speed default.

Sources, with complete commands and configuration recorded in each JSON:

- B1: [dense](oct4-full-prefill-scale-b1-1g-16k256k.json),
  [LoD](oct4-lod-prefill-scale-retention4g-b1-1g-16k256k.json).
- B8, 16K: [dense](oct4-full-prefill-scale-b8-3g-16k.json),
  [LoD](oct4-lod-prefill-scale-retention4g-b8-3g-16k.json).
- B8, 32K/64K: [dense](oct4-full-warm-prefill-b8-32k64k.json),
  [LoD](oct4-lod-prefill-retention4g-b8-32k64k.json).
- B8, 128K: [dense](oct4-full-prefill-scale-b8-5g-128k.json),
  [LoD](oct4-lod-prefill-scale-retention4g-b8-1g-128k.json).

### B1 token-sharded prefill: long-context capacity follow-up

The opt-in token-sharded path reuses each GPU's fixed BF16 decode backing
for its one-eighth share of the prefill history. Global centroids, top-eight
selection and the 1,024-leaf rule are unchanged; fine attention is combined
across token owners. It does not gather the whole KV history or use request
owners. It successfully completes **512K and 1020K warmup and measured generation**:

| Context, B1 | Dense prefill (s) | Sharded LoD prefill (s) | Dense / LoD | Dense decode (ms/step) | LoD decode (ms/step) |
|--:|--:|--:|--:|--:|--:|
| 512K | 118.730 | 84.443 | 1.406x | 24.686 | 26.393 |
| 1020K | 344.493 | 178.122 | 1.934x | 27.378 | 26.327 |

Both lengths pass all eight ranks' real-attention and four-update audits, with
19.124 / 22.734 GiB/rank peak client Torch allocation, respectively (excluding
daemon weights and driver allocations). The short-context replicated table
above is unchanged. These are warmed single-pass speed tests, not quality tests.
See [TOKEN_SHARDED_PREFILL.md](TOKEN_SHARDED_PREFILL.md) for implementation,
raw results, tests, memory scope and exact reproduction commands.

## Measurement policy and limits

Full and LoD engines run sequentially, with matching prompt/continuation hashes,
seed 0, one exact-shape warmup and one measured pass. The scheduler uses logical
16K prefill chunks. LoD uses exact top-eight routing and the global per-request
16K prefill / 256-token decode update cadences; neither B8 nor DCP divides these
sequence-index cadences. The current development settings are explicit in the
[decode driver](../../benchmarks/kimi_k3_decode_power2.py); they are not all
general serving defaults.

Decode uses captured graphs (`FULL_DECODE_ONLY`) and first-to-last-token request
timestamps, cross-checked against whole-generation wall time. Update counters
are read outside timed generation. Results require real attention, no prefix
cache hits or preemptions, and all B8 requests live throughout the decode window.
No profiling events are inserted into graph replay.

Native cache reservations are matched within B1 and short-B8 pairs. At 128K/B8,
dense reserves 5 GiB/rank and LoD 1 GiB/rank, since LoD owns a separate remote
cache. These reservations are **not total cache VRAM**.

The ordinary replicated two-tier configuration has not been shown to fit
B1/512K or B8/256K; the opt-in request-owner layout does fit B8/256K, as
reported above. Token-sharded B1 now fits and completes measured 512K/1020K generation.
A 256K-capacity replicated B8 LoD engine fails
even during 128K warmup, although the separately sized 128K engine succeeds.
Failures are capacity observations, not timings. Detailed attempts are retained
in the archive and [capacity failure record](oct4-b1-long-capacity-failures.json).

Fresh ordinary **B1/512K** fit probes with shared decode scratch, compact
selected-leaf projection, growing archives and the expandable allocator
also fail during prefill: the first completes a 256K prefix, and a more
aggressive idle-allocation reclamation retry completes 272K, but neither
reaches decode. The latter boundary still has a 10.261 GiB/rank replicated
prefill archive and only 0.979 GiB physically free on rank 0. These are
failed capacity checks, not new speed results. See
[the ordinary 512K retry](PREFILL_VRAM.md#ordinary-dcp8-b1512k-retry-october-6).

The [October 6 prefill memory investigation](PREFILL_VRAM.md) records unique
cache/workspace storage from the corrected B1/256K retry and a failed B8/256K
fit-only probe. Compact projection, growing temporary archives and periodic
allocator reclamation did not make B8/256K fit: it failed during the first
request's prefill, after the last observed completed 32K prefix. The preceding
snapshot shows 7.51 GiB persistent cache, 4.48 GiB decode scratch and only
1.81 GiB physically free per rank; the runtime later reported 86 MB free.
This is a capacity failure, not a new speed measurement.

The subsequent [shared decode workspace](SHARED_DECODE_SCRATCH.md) saves
4.101 GiB/rank (4.482 to 0.380 GiB of decode workspace) while retaining
private layer outputs, LSEs and persistent cache state. Captured GPU
equivalence tests passed. The B8/256K fit retry reached a completed 128K
prefix of its first request but still failed; enabling ROCm scratch
reclamation did not make it fit. No new B8/256K timing is inferred.
The matched captured B8/16K check took 37.933 ms/decode step versus 37.928 ms
with private scratch: latency is unchanged, while the dense control is
31.649 ms. All eight rows remained live and every layer included four
updates per request. This allocation change is not a decode speedup.

The later request-owner prototype below has now completed B8/256K prefill;
the capacity limitation above applies to the original replicated path, not
that prototype. Neither path has a new validated B8/256K LoD decode result.

An opt-in [sharded prefill-archive experiment](../kimi-k3-mla-stack/SHARDED_PREFILL.md)
keeps global centroid/routing math unchanged and reconstructs only the current
layer's exact KVs in temporary workspace. Fixture checks recover most of the
original prefill speed: the matched 1020K fixture takes 41.64 s versus 36.03 s
with the replicated archive (+15.5%), while peak live Torch allocation falls
from 47.60 to 26.48 GiB. The 512K fixture also completes. These are attention-only
fixture results. The trained-model reconstruction probe failed during 512K
warmup with HSA out-of-resources and reported no free device memory; no validated
512K/1020K LoD timing resulted. A current-kernel distributed-leaf consumer also
passes fixture checks: at 512K/B1, prefill is 20.14 s versus 13.74 s replicated,
and peak live Torch allocation is 21.61 versus 29.71 GiB. That is a 27.2%
allocation reduction for 46.6% more fixture prefill time. The trained-model
[512K distributed probe](oct5-distributed-leaf-capacity-failure.json) also failed
warmup at a 376,832-token prefix with no free device memory. It has not solved
full-model capacity, and no valid full-model timing was produced.
This is not yet the default path and is not included in the tables above.

The [October 5 request-owner fixture probe](../kimi-k3-mla-stack/SHARDED_PREFILL.md#october-5-shared-centroid-budget-and-request-owners)
also tests one GPU per request row with bounded fine-attention scratch. At 128K
its eight-owner LoD cohort takes 13.91 s versus 24.64 s dense, with 26.61 versus
27.14 GiB peak Torch allocation per GPU. These are matched **attention-stack
fixture** controls with no MoE or Q/output transfers and a larger aggregate
prefill budget than the TP8 runner. They are not full-model timings and do not
yet prove long-context trained-K3 capacity.

The completed [full trained-model request-owner probe](REQUEST_OWNER_PREFILL.md)
includes actual TP query/output transfers and the MoE layers. Its B8 prefill
is 32.51 / 68.18 / 144.97 s at 32K / 64K / 128K, versus the archived dense
33.57 / 70.56 / 155.41 s. It does not carry over the fixture's speed gain:
64K and 128K are effectively unchanged from previous LoD, with 128K 0.81%
slower. Peak live Torch allocation reaches 26.39 GiB/rank at 128K, excluding
daemon-owned IPC weights; this is not a total-VRAM comparison.
This is an opt-in prefill-only prototype, not a decode replacement. It uses
eight 2K scheduler slices within the same 16K aggregate budget; logical
per-request updates still occur every 16K. It reuses existing dense controls
rather than rerunning them, so the reported comparison does not isolate
ownership from the scheduler change. All eight worker binary audits passed.

The [full-16K-per-owner follow-up](REQUEST_OWNER_PREFILL.md#follow-up-full-16k-blocks-per-owner)
now works on the trained model in four-owner waves: all eight GPUs receive
full 16K blocks, while the TP8/EP8 model runs on all ranks. At 32K/B8 it takes
34.17 s, versus 32.51 s for eight 2K slices and archived dense 33.57 s.
Larger blocks help the attention-stack fixture but have not improved full-model
speed at this length. Eight simultaneous blocks initially exceeded available
VRAM; the four-owner version peaks at 27.54 GiB/rank of live Torch allocations, excluding
daemon weights. It is an experimental prefill-only configuration, not a new
default or a quality-validated serving path.

Token-sharding the native residual bank subsequently enabled all eight GPUs
to process full 16K owner blocks simultaneously. The audited trained-model
32K/64K prefill times are 34.95/69.07 s, versus archived dense 33.57/70.56 s,
with peak live Torch allocations of 26.28/28.78 GiB per rank (daemon weights
excluded). Low-headroom allocator reclamation was needed between warmup and
measurement; this is not an allocator-hot comparison. A two-arena reusable
query/output transport is also verified on the distributed fixture, with
two buffer allocations per rank across 96 attention calls and unchanged
first tokens. On the trained model it gives effectively unchanged 32K time
(34.91 s) while retaining an extra 1.125 GiB/rank; its 64K warmup fails with a
ROCm resource error. Persistent transport arenas remain off by default.

The subsequent **B8/256K request-owner prefill** completes in **340.22 s**,
versus archived dense **365.57 s** (**1.075x**). All eight worker audits pass,
with all eight requests active and unchanged global 16K update boundaries.
This uses 2K query slices, a sharded residual bank, and no per-layer allocator
pressure check. Peak live Torch allocation is 29.25 GiB/rank, excluding daemon
weights. It is a prefill-only measurement: decode and B8/512K are not validated.
Details and raw data are in [REQUEST_OWNER_PREFILL.md](REQUEST_OWNER_PREFILL.md#completed-b8256k-follow-up).
The silent interval included both warmup and measurement; it was incorrectly
suspected to be a hang. The driver now publishes phase changes outside timing.
Duplicate local-field concatenations are removed without retaining extra
buffers. Neither follow-up establishes 256K trained-model capacity.
The [owner-prefill notes](REQUEST_OWNER_PREFILL.md) record the distinctions.

## Reproduction

Run from the repository root on the prepared eight-GPU K3 host. This development
setup requires the unpacked `amdsiloai/vllm:kimi-k3-mi325x-v10` userspace, local
checkpoint `/tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06`,
and resident weight daemon `kimi-k3-shared-int4-v6`. The
[direct wrapper](../../benchmarks/run_kimi_k3_v10_direct.sh) selects the image
runtime; the [orchestrator](../../benchmarks/kimi_k3_decode_power2.py) supplies
the matched full/LoD settings and runs missing measurements sequentially.
No proprietary cluster runner is required.

```bash
# Run missing decode points and update their table:
python -m benchmarks.kimi_k3_decode_power2 --run

# Rebuild the decode table from existing validated files, without inference:
python -m benchmarks.kimi_k3_decode_power2
```

For the prefill-only sweep, use the exact command and environment in the source
JSON for the desired row (`--decode-tokens 2`). Do not use those one-step decode
latencies as throughput. The optional untracked ProLong token cache can be
omitted to tokenize the same frozen dataset directly.

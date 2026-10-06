# Full Kimi K3 request-owner prefill probe (October 5)

This tests the trained full model, not the attention-stack fixture. It assigns
each of eight requests' attention to one GPU while leaving the model weights,
MoE, and output projection in their normal TP8/EP8 layout. Query-head slices
are sent to the request owner and attention outputs sent back. Each owner
keeps one complete latent history and all of that request's centroids. Fine
attention uses 12-head scratch tiles rather than a 96-head scratch allocation.

The measurements below originally used an opt-in **prefill-only prototype**.
An eager owner-decode extension is now being tested (see the final section).
Captured owner serving and prefix reuse are not implemented. No production
default was changed.

## Measurements

Eight MI325X GPUs on node 4; B8, TP8/DCP8/EP8; resident trained INT4 MoE
weights and BF16 two-tier attention. Real ProLong prompt-token hashes are
verified against the archived dense runs before model startup. Each length
gets one exact-shape warmup and one measured pass. Prefill ends at the last
request's first output token and includes final centroid/page construction.
Whole-generation wall time is also recorded as a sanity check. Memory RPCs
run outside the timing window.

| Context | Archived dense (s) | Previous LoD (s) | Request-owner LoD (s) | Dense / owner | Peak live Torch allocation per rank (GiB) |
|--:|--:|--:|--:|--:|--:|
| 32K | 33.5666 | 33.6356 | 32.5086 | 1.033x | 19.496 |
| 64K | 70.5637 | 68.4368 | 68.1821 | 1.035x | 22.406 |
| 128K | 155.4086 | 143.8071 | 144.9714 | 1.072x | 26.393 |

The attention-stack fixture speedup did not carry over: at 64K the owner path is
only 0.37% faster than previous LoD, and at 128K it is 0.81% slower. These
single-pass differences are too small to establish a performance advantage
over previous LoD. The model runs successfully at all three lengths, but this
result does not justify promoting the owner organization for speed alone.

The new run does **not** repeat dense. It reuses these existing controls:

- [32K/64K dense](oct4-full-warm-prefill-b8-32k64k.json).
- [128K dense](oct4-full-prefill-scale-b8-5g-128k.json).
- [32K/64K previous LoD](oct4-lod-prefill-retention4g-b8-32k64k.json).
- [128K previous LoD](oct4-lod-prefill-scale-retention4g-b8-1g-128k.json).

These are archived comparisons, not newly paired same-scheduler measurements.
The owner path schedules eight 2K slices per 16K aggregate budget; the archived
path schedules one 16K slice. The **logical per-request update remains 16K**
and the unfinished logical block stays exact. DCP and batch size do not divide
the sequence-index cadence. The owner path is eager; archived prefill was also
not whole-prefill CUDA-graph captured. Resident weights and the model runtime
are reused, but this comparison does not isolate scheduler or compilation
configuration differences from the ownership change.

Native cache reservation is 1 GiB/rank for the owner path. Archived dense uses
5 GiB/rank at 128K. Torch allocator peaks exclude the daemon's resident IPC
weight storage and therefore are **not total GPU memory peaks**. The JSON also
records reserved allocation and device free memory after generation, which is
not minimum free memory during generation. No matched total-VRAM reduction or
quality score is claimed by this speed probe.

Raw owner measurements: [oct5-full-request-owner-lod-prefill.json](oct5-full-request-owner-lod-prefill.json).
The run completed successfully and the loaded exact-top-eight route/coarse
binary audit passed on all eight workers. Measured whole-generation wall times
were 32.5198, 68.2011, and 144.9974 seconds respectively, agreeing with request
prefill timings within 27 ms. The 128K run's minimum device-free reading
**after** generation was 4.102 GiB/rank; no larger-context capacity is claimed.

## Reproduction

Use the prepared K3 v10 runtime and the already resident
`kimi-k3-shared-int4-v6` weight daemon. The local checkpoint and frozen token
cache must be present. This command runs **only LoD**, with no proprietary
cluster runner required:

```bash
env \
  LOD_KIMI_REQUEST_OWNER_PREFILL=1 \
  LOD_BENCHMARK_ADMISSION_COHORT=8 \
  LOD_KIMI_SUBTILE64=score \
  LOD_KIMI_CHUNK_TILE_PACK=1 \
  LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 \
  LOD_KIMI_SORT_LEAF_ROUTES=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 \
  LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 \
  LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_PREFILL_MIN_FREE_GIB=4 \
  LOD_KIMI_TILE_REFINE=1 \
  LOD_KIMI_DIRECT_LEAF_RESULT=1 \
  HSA_NO_SCRATCH_RECLAIM=0 \
  TRITON_CACHE_AUTOTUNING=1 \
  VLLM_USE_TRITON_AWQ=1 \
  TRITON_CACHE_DIR=/tmp/dan-agent/.triton/cache \
  AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv" \
  benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --mode two-tier --lengths 32768 65536 131072 \
  --batch-size 8 --decode-tokens 1 \
  --kv-cache-memory-bytes 1073741824 \
  --weight-cache-id kimi-k3-shared-int4-v6 \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --reference-baselines \
    results/kimi-k3-full-model-current/oct4-full-warm-prefill-b8-32k64k.json \
    results/kimi-k3-full-model-current/oct4-full-prefill-scale-b8-5g-128k.json \
  --report-memory \
  --output results/kimi-k3-full-model-current/oct5-full-request-owner-lod-prefill.json
```

The opt-in rejects unsupported model geometry and requires B8/TP8/DCP8.
Do not set other experimental local-DCP, shared-centroid, or sharded-leaf
prefill flags together with it. Greedy sampling generates exactly one token;
this is deliberately not a decode-throughput test.

## Correctness checks

`tests/test_kimi_request_prefill.py` checks grouped query/output exchange for
all eight rank positions, global 16K update coverage, block-boundary guards,
and the 2K-sliced owner path against the ordinary 16K engine on GPU. That GPU
comparison gives matching centroid sums, values, counts, leaf counts, and
coverage, with BF16-tolerant attention outputs. The owner TP8 attention-stack
smoke test also runs all 24 MLA layers with actual query/output transfers.
These checks establish the prototype's plumbing, not full-model task quality.
The focused GPU regression suite passed 330 tests initially; its remaining
page-growth test was corrected to compare semantic leaf membership through
both directory levels rather than nondeterministic physical page IDs, then
passed in a separate targeted GPU run. No kernel change was needed for that
test assertion. The owner-specific GPU suite passed all 19 tests.

## Follow-up: full 16K blocks per owner

The independent-owner fixture gave each GPU a complete 16,384-query block,
whereas the full-model probe above gave each owner only 2,048 queries per
scheduler call. Even without MoE, the integrated TP8 owner fixture took
4.5477 seconds at 32K versus 2.038 seconds for independent owners. Thus MoE
alone does not explain why the fixture gain disappeared. Query/output
transfers, smaller attention blocks, and synchronous construction all differ.

The opt-in now accepts `LOD_KIMI_OWNER_QUERY_CHUNK=16384` with
`LOD_BENCHMARK_PREFILL_COHORT=8`: eight full blocks, a 131,080-token scheduler
budget, and unchanged per-request 16K centroid-update boundaries. Native MoE
work is independently sliced into 16,384 tokens by
`LOD_KIMI_OWNER_MOE_CHUNK`; attention does not inherit that smaller aggregate
budget. These settings are experimental, not a new default. A smaller cohort
can bound activation memory while still giving each active owner a full block.
The full model also retains a replicated attention-residual bank: 131K
scheduled tokens require about 14 GiB/rank for that bank alone. Therefore a
successful fixture run is not sufficient evidence that the full model fits.

The initial B8 256K full-model attempt failed in route scatter during warmup,
not with an out-of-memory exception. A 64K attention-stack run reproduced it;
a diagnostic captured inflated route counts that would write beyond the
packed-query allocation. No valid 256K timing or capacity result was produced.
Large-shape isolated route-count and route-packing replay tests pass, so the
integrated failure needed separate validation before claiming support at 256K+.
Separating raw per-chunk route counts from their prefix-offset output removed
the integrated failure: the 64K warmup and measured diagnostic pass completed
with bounds checks on every scatter and all worker binary audits passing.
Diagnostic timings are deliberately not used as serving measurements.

The full-16K TP8 owner fixture also completed with worker audits passing:

| Context | Integrated owners, 16K each (s) | Independent owners, 16K each (s) | Integrated peak Torch allocation/rank (GiB) |
|--:|--:|--:|--:|
| 32K | 3.2956 | 2.038 | 25.478 |
| 64K | 7.7322 | 5.357 | 27.984 |
| 128K | 18.5575 | 13.912 | 31.958 |

At 32K this is 27.5% less time than the earlier integrated 2K-slice smoke
(4.5477 s), but it still trails independent owners. These are fixture results,
not full-model speed or VRAM comparisons. The integrated probe additionally
performs TP query/output transfers and retains replicated model activations;
the independent fixture does neither. A 2K-slice fixture sweep also completed,
with unchanged generated tokens at all shared lengths and worker audits passing:

| Context | Integrated 2K slices (s) | Integrated 16K blocks (s) | 2K / 16K |
|--:|--:|--:|--:|
| 32K | 4.3581 | 3.2956 | 1.322x |
| 64K | 11.7918 | 7.7322 | 1.525x |
| 128K | 32.7734 | 18.5575 | 1.766x |

The 2K engine was sized through 256K, whereas the 16K engine was sized through
128K. This is an observed scheduler/block-size comparison, not an isolated
same-capacity microbenchmark. The 2K fixture also completed 256K in 98.5643 s,
with 28.024 GiB peak Torch allocation/rank; it does not establish full-model
256K fit.

The trained-model eight-simultaneous-block attempts ran out of memory during
32K warmup, both with 16K and with 4K MoE slices. The first failed in a 1.75 GiB
MoE stage-two allocation; the smaller-MoE version failed on a 384 MiB
allocation. Neither produced a valid timing. The four-block cohort now runs:
owners rotate across all eight GPUs, each receiving full 16K query blocks,
but only four attention owners work simultaneously. The native TP8/EP8 model
computation remains on all eight GPUs. This halves the replicated
attention-residual bank from 14 to 7 GiB without changing its arithmetic.
The clean run used `PYTORCH_ALLOC_CONF=expandable_segments:True` to limit
allocator fragmentation. Failed-job workers were removed and device memory
was checked before starting; no dense control was rerun.

| 32K trained-model prefill, B8 | Seconds | Peak live Torch allocation/rank (GiB) |
|:--|--:|--:|
| Archived dense | 33.5666 | Not matched |
| Earlier owner path, eight 2K slices | 32.5086 | 19.496 |
| Four-owner waves, 16K per owner | 34.1701 | 27.542 |

The four-owner run's elapsed wall time was 34.1794 s, within 10 ms of its
prefill metric. All eight workers passed the loaded-binary audit; each GPU
recorded 96 attention-layer calls with exactly 16,384 queries across warmup
and measurement. The larger block works, but does **not** improve trained-model
speed at 32K: it is 5.1% slower than the earlier sliced-owner result and 1.8%
slower than archived dense. These remain archived scheduler comparisons, not
new same-configuration dense pairs. The four-owner run additionally changes
model/MoE aggregate batch geometry; seven of eight generated first tokens
match the earlier owner run. It is not a task-quality evaluation or a claim
of bitwise full-model equivalence.

Raw trained-model result: [four-owner 16K blocks](oct5-full-owner16k-clean4.json).
The model's token-dependent MoE/recurrent work and replicated residual bank
are absent or much smaller in the fixture. Full 16K blocks reduce the fixture's
attention/scheduler cost but cannot establish an end-to-end benefit when
the full model must reduce owner concurrency to fit those larger activations.
No isolated MoE/attention runtime attribution is claimed by these tests.
At that point all eight simultaneous blocks still required further
activation-memory changes; the sharded residual bank below subsequently
enabled audited all-eight-owner trained-model tests at 32K and 64K.

### Token-sharded attention-residual bank

`LOD_KIMI_OWNER_SHARD_RESIDUAL=1` now prototypes that memory change. The
original model forward and decoder layers remain in place. Each rank retains
only its contiguous token slice of the residual bank and applies the native
residual-mixing kernel to that slice. Its output is gathered before ordinary
TP attention/MoE. Prefix-sum updates stay local to their owning token rows;
they are not read on other ranks. This requires eager owner-prefill, PP1, and
no auxiliary hidden-state outputs. It does not change LoD routing or cadence,
and adds an all-gather per residual mix.

The native BF16 kernel is bitwise equivalent in three GPU tests, including
delta updates and block writes. The integrated 24-MLA-layer fixture completed
with all worker binary audits passing and identical generated tokens at
32K, 64K, and 128K:

| Fixture context, eight simultaneous 16K blocks | Original bank (s) | Sharded bank (s) | Original peak Torch (GiB/rank) | Sharded peak Torch (GiB/rank) |
|--:|--:|--:|--:|--:|
| 32K | 3.2956 | 3.3976 | 25.478 | 22.415 |
| 64K | 7.7322 | 7.9412 | 27.984 | 24.920 |
| 128K | 18.5575 | 18.9920 | 31.958 | 28.897 |

The fixture saves 3.06 GiB/rank at a 2.3–3.1% prefill penalty. It has two
residual-bank blocks; the full model has eight, so the corresponding bank
allocation reduction is **12.25 GiB/rank** for a 131K aggregate scheduler
call. That is an allocation calculation, not an isolated measured full-model saving.
The fixture's CPU-metadata audit confirms eight requests in each prefill
call and 16K query blocks on every rank. Raw result:
[sharded-bank fixture](../kimi-k3-mla-stack/oct5-owner16k-sharded-residual.json).
The residual-specific test suite passed 12 tests in the K3 image, including
the three native-kernel GPU checks. The combined owner/residual GPU-image
suite subsequently passed 38 tests, including sliced-versus-full-block engine
equivalence after the storage-ownership fix below.

### 256K capacity attempt and retained-buffer ownership

The single-pass B8/256K trained-model capacity probe did **not** complete.
The scheduler had eight requests at position 192,512 (188K), each receiving
2K, when ROCm reported `HSA_STATUS_ERROR_OUT_OF_RESOURCES` in an RCCL kernel
with 0–152 MB free. This scheduler position is not a completed-prefix
guarantee. No valid timing, capacity success, or binary audit resulted.
The failed client workers were removed; resident daemon weights were retained.
The [failure artifact](oct5-full-owner256k-capacity.json) records its settings.

Investigation also found an ownership issue in unfinished blocks: a contiguous
2K row slice still referenced its whole eight-row backing allocation.
`retain_owner_record` now copies just the owned row, and the 256-token recent
tail also owns its small storage. This changes no K/V values, routing, or update
cadence. A storage-size test verifies that the retained tensor no longer pins
the full batch, and GPU sliced/full-block equivalence passes. The old 256K
attempt predates this fix; **256K capacity with the fix remains unverified**.
An all-eight-owner trained-model probe completed its 32K warm and timed passes
(32.7414 s), but failed in the 64K warmup with an RCCL resource-allocation
error and 1,254 MB reported free. Its worker binary audit had not yet run, so
that partial 32K number is **provisional**, not a validated speed result.
The [failed sweep](oct5-full-eight-owner16k-bankshard.json) is marked accordingly.

The [reserve retry](oct5-full-eight-owner16k-reserve.json) completed both
lengths with all eight loaded-worker binary audits passing. Every GPU received
a full 16K query block, with all eight requests in the same model call.

| Trained-model context, B8 | Archived dense (s) | Eight 2K owner slices (s) | Eight 16K owner blocks, sharded bank (s) | Peak live Torch allocation/rank (GiB) |
|--:|--:|--:|--:|--:|
| 32K | 33.5666 | 32.5086 | 34.9503 | 26.278 |
| 64K | 70.5637 | 68.1821 | 69.0702 | 28.783 |

The elapsed wall times are 34.9596 and 69.0852 s. The first generated tokens
match the 2K-owner run on 8/8 prompts at 32K and 7/8 at 64K; this is not a task
quality evaluation. Large owner blocks fit, but the full-model runtime benefit
does not approach the fixture gain. At 64K the result is 2.2% faster than
archived dense and 1.3% slower than the earlier 2K-owner path.

The benchmark tries to retain the allocator outside timing, but uses an
8 GiB headroom check between warmup and measurement. It reclaimed inactive
allocations on every rank at both lengths; after the 64K warmup, several ranks
had zero reported free memory. The retry also enables proactive allocator
garbage collection. Thus this is not a fully allocator-hot or isolated
comparison of attention block sizes. No per-chunk reclamation was recorded
in this owner path. Peak live Torch allocation excludes daemon-owned weights,
native allocations and dead allocator reservations. **256K capacity with these
changes remains unverified**; these tests do not establish it.

### Reusable request-owner transport storage

Query receive buffers, query layout conversion, output send packing, and the
returned TP-head output previously allocated fresh tensors on every MLA layer.
The opt-in `LOD_KIMI_OWNER_REUSE_TRANSPORT=1` shares two grow-only arenas on the DCP group across
all 24 MLA layers: a rank-major wire arena and an assembled-query arena. Once
attention consumes the assembled queries, the latter arena is reused for the
return output; the wire arena is reused for send packing. Sizes grow only for
a larger scheduler shape, with no tensor-storage allocation in steady-shape
exchanges. Buffers survive request cleanup for reuse on the next request.

RCCL transfers, attention, packing, and subsequent W_O enqueue on the same
current stream. Returned output views must be consumed before the next owner
exchange; this is specific to the eager, sequential prefill-only experiment,
not a concurrency-safe public attention API. Semantic K/V records still own
separate storage and are never overwritten by transport-buffer reuse.

At B8 with 16K queries per owner and BF16 D192, the two arenas occupy
1.125 GiB/rank. This is shared across layers, not multiplied by 24. The audit
reports arena allocation counts/bytes without GPU timers. Repeated eight-rank
exchange tests establish bitwise head ordering and exactly two arena
allocations over 24 iterations. The distributed 24-layer fixture passed all
worker audits and generated the same eight first tokens as the earlier
sharded-bank fixture. Its 32K runtime is 3.4094 s versus 3.3976 s previously,
effectively unchanged in single-pass measurements; peak live Torch allocation
increases from 22.415 to 23.072 GiB/rank because both transport arenas remain
live during attention. Each rank reports exactly two arena allocations across
96 owner-attention calls (warmup plus measurement), demonstrating actual
cross-layer/chunk reuse. Raw result:
[reused transport fixture](../kimi-k3-mla-stack/oct5-owner16k-reused-transport.json).

The full-block path additionally avoids a one-element concatenation and
reuses the already assembled local attention field for the state update,
instead of concatenating the recent tail and new block again.

The [matched trained-model test](oct5-full-owner16k-reused-transport.json)
completed and individually audited 32K at 34.9123 s versus 34.9503 s without
the persistent arenas, with all eight first tokens identical. This is only
0.11% faster in single-pass measurements, not a meaningful measured speed win.
Peak live Torch allocation increased from 26.278 to 27.403 GiB/rank, precisely
the retained arenas' 1.125 GiB. The 64K warmup failed with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` / `hipErrorLaunchOutOfResources`; no 64K
timing was produced. **Persistent arenas are therefore off by default.** The
safe elimination of duplicate concatenations remains on, with no change to
LoD mathematics. Reusing existing kernel scratch, rather than retaining extra
live arenas, would be a preferable next allocation experiment at this capacity.

The reserve retry above started before these changes and **does not** measure
reusable transport buffers. The final owner/residual GPU-image suite passed
55 tests, including engine equivalence after local-field reuse and both
allocating and persistent-arena modes on all rank positions. The CPU-focused
suite passed 86 tests. Failed test-run workers were removed and node4 device
memory was verified back at the resident-daemon-only level; daemon weights
were not unloaded or rematerialized.
These tests count tensor-storage allocation requests, not native driver
allocations; PyTorch's caching allocator can service an allocating call without
a fresh device allocation. No isolated native allocator-cost claim is made.

For a full-model reproduction, keep the command/environment in the earlier
reproduction section and explicitly add the following settings (replace its
output filename; leave persistent transport reuse unset for the working path):

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.1
export LOD_KIMI_OWNER_QUERY_CHUNK=16384
export LOD_KIMI_OWNER_MOE_CHUNK=16384
export LOD_KIMI_OWNER_SHARD_RESIDUAL=1
export LOD_BENCHMARK_PREFILL_COHORT=8
export LOD_BENCHMARK_ADMISSION_COHORT=8
# Run only --lengths 32768 65536 for the verified full-block size range.
# Keep --batch-size 8 --decode-tokens 1 and the same frozen ProLong prompts.
# Only for the arena experiment: export LOD_KIMI_OWNER_REUSE_TRANSPORT=1
# Its 64K trained-model warmup failed; this is not a production recommendation.
```

Raw fixture results: [full 16K owner blocks](../kimi-k3-mla-stack/oct5-owner-tp8-full16k.json).
The [2K-slice sweep](../kimi-k3-mla-stack/oct5-owner-tp8-2k-fixed-prefix.json)
records the repaired 256K fixture run.
Focused verification: 86 CPU tests and 34 owner/routing GPU tests passed, including
2K-sliced versus 16K engine equivalence, exact large-shape route ordinals,
packing prefixes, scheduler budgets, and tokenwise MoE chunking plumbing.

## B8 256K+ follow-up

These trained-model tests retain eight simultaneous 16K query blocks, one
complete request-owned latent archive per GPU, and global-sequence 16K updates.
The eight frozen ProLong prompt hashes match the archived dense 256K control
(365.571 s prefill); weights are taken from the resident daemon, not reloaded.
The owner prototype remains prefill-only: `max_tokens=1` is not a decode test.

| Attempt | MoE microbatch | Outcome |
|--|--:|--|
| [Full-block 256K](oct5-owner16k-b8-256k-speed.json) | 16K | Warmup failed in the first scheduled 16K block; ROCm resource-allocation error, 3,104 MB reported free on the failing GPU. No timing. |
| [Smaller MoE temporary](oct5-owner16k-b8-256k-moe4k.json) | 4K | Warmup failed in the first scheduled block; several GPUs reported zero free memory. No timing. |
| [Allocator-pressure check](oct5-owner16k-b8-256k-pressure.json) | 16K | Warmup also failed in the first scheduled block with zero free memory on the failing GPU. No timing. |

These failures do not establish that B8/256K fits. They also do not measure a
completed 16K prefix: a failure partway through the full model's first forward
can occur after only some layers have processed that block. The failed workers
exited, and device usage was checked back at the daemon-only level between
attempts. The weights were never unloaded or rematerialized.

The request-owner early return bypasses the normal runtime allocator-pressure
hook. The opt-in `LOD_KIMI_OWNER_PRESSURE_CHECK=1` invokes that existing policy
before each real owner MLA call. It releases only inactive allocator blocks
when device headroom is below `LOD_KIMI_PREFILL_MIN_FREE_GIB`; it does not remove
live semantic K/V or change attention/update math. Any reclamation cost belongs
inside the generation timing. The follow-up uses an 8 GiB reserve and ordinary
16K MoE microbatches. This did not establish larger capacity: inactive-cache
reclamation alone was insufficient for eight simultaneous full 16K blocks.

The next attempt halves scheduler query slices to 8K per owner. It still admits
all eight requests together, retains the unfinished 16K logical block exactly,
and constructs centroids only at global 16K boundaries. This is an explicit
activation-memory tradeoff, not a change to update cadence or a test of only
four active requests. Capacity and speed must be measured rather than inferred
from its smaller aggregate 64K forward.

The initial 8K-slice job was cancelled before any measurement because its
scheduler budget still used `16K * cohort`, over-reserving a 128K forward.
Explicit owner-slice budgets now use `max(16K, slice * cohort) + decode reserve`:
eight 8K slices reserve 65,544 tokens, and eight 16K slices still reserve
131,080. Admission remains eight. The owner sizing regression test covers both
settings. The cancelled workers were identified by PID and parent state before
cleanup; they were not allowed to overlap the corrected retry.

The [correctly sized eight-owner 8K-slice run](oct5-owner8k-b8-256k-sized.json)
also failed during warmup, with zero free memory reported on GPU 3. Its failing
scheduler step records 32,768 previously computed tokens on **each** of the
eight requests. This is later than the full-16K failures, but is not a 256K
capacity result and supplies no valid speed point. The next diagnostic uses
the known 64K size rather than repeating an oversized full-model failure.

That [64K diagnostic attempt](oct5-full-owner64k-chunk-diagnostic.json) stalled
during warmup, before the profiler was armed. A read-only health check found
nearly all physical VRAM occupied and GPU activity without completed output.
It was cancelled and its specific client/worker processes cleaned up; daemon
weights were retained. No profiler trace, audit, or timing exists for that
attempt. The experimental per-layer allocator-pressure hook is disabled for
the next retry; the stall's cause has not been established. A smaller 2K-slice
256K run is the next capacity/speed check, with all eight requests active and
the same global 16K logical update boundary.

## Completed B8/256K follow-up

The [2K-slice retry with per-layer pressure checking off](oct5-owner2k-b8-256k-no-pressure.json)
**completed warmup and measurement successfully**, with all eight loaded-worker
binary audits passing. This supersedes the earlier unverified 256K capacity
status for this particular prefill-only configuration.

| Context | Archived dense prefill (s) | Request-owner LoD prefill (s) | Dense / LoD | Peak live Torch allocation/rank (GiB) |
|--:|--:|--:|--:|--:|
| 256K, B8 | 365.5710 | 340.2216 | 1.075x | 29.249 |

Whole-generation wall time was 340.2835 s. Prompt hashes match the archived
dense cohort; all eight workers recorded 6,144 2K MLA owner calls across the
warm and timed passes (24 layers x 128 slices x 2 passes). Every request was
active through the final slice at global position 260,096. The query slice is
2K; global per-request construction still occurs at 16K boundaries. The
sharded residual bank is enabled, native MoE microbatches remain 16K, and
persistent transport buffers and per-layer pressure reclamation are off.

The minimum device-free reading **after** timed generation was 3.555 GiB;
this is not minimum headroom during execution. Peak Torch allocation excludes
daemon-owned IPC weights. No decode timing was measured, and no 512K capacity
is established. The dense control is archived, not a fresh matched-scheduler
run; this single pass is approximately 6.9% less prefill time.

The job was incorrectly suspected to be hung because the driver did not publish
phase changes between startup and completion of **both** warmup and timed
generation. It finished before the cancellation request could apply; the
runner reports exit code zero, and its successful JSON was not modified into
a failure. The approximately eleven-minute execution interval includes both
passes, not eleven minutes of warmup. This was a monitoring mistake, not
evidence of a kernel hang. Phase reporting now explicitly saves and prints
warmup, warmup-complete, measurement, and audit transitions outside the timed
generation window. Forty focused CPU tests pass, including a regression test
for those distinct phases. Worker processes exited and device usage returned
to the daemon-only baseline. A single ENOMEM error from a /proc inspection
does not establish host-memory pressure; the follow-up host check reported
2.6 TiB available and zero recent memory-pressure averages.

## Diagnostic and execution-layout proposal

The diagnostic driver adds `--diagnostic-only`: warm a single requested length,
then profile only its final model chunk on rank zero during a separate pass.
It emits no serving timing. Temporary scopes distinguish local attention,
remote attention, state construction/update, query/output transport, native
MLA projections, recurrent attention, MoE, and residual mixing plus gather.
GPU activities are attributed to their innermost scope using profiler
correlation IDs. Both summed activity durations and interval-union durations
are reported, explicitly not interchangeable with model wall latency. The
instrumented chunk includes profiling overhead; other ranks are not profiled.
This is a diagnosis of stages, not a replacement for the warmed uninstrumented
speed experiment. The harness is first tested on the distributed small fixture.

The user's proposed next layout is promising **if its replicated non-expert
weights fit**:

- Each GPU owns a request's hidden states, all 96 MLA heads, latent cache,
  centroids, recurrent/KDA state, and residual bank. Attention projections and
  residual mixing stay local, not just attention scoring.
- Only routed MoE work uses expert parallelism across all eight GPUs. K3's
  routed expert width is 3,584 versus hidden width 7,168, so its existing
  latent MoE projection can precede expert dispatch.
- Embeddings and the output head can remain vocabulary-sharded; communicate
  at those boundaries rather than redistributing every layer's queries,
  attention outputs and normalized residual inputs. Shared experts also need
  an explicit local-versus-sharded decision.

This differs from the current owner prototype, which still generates queries
with TP8, redistributes their head slices, returns attention outputs for TP8
W_O, and gathers normalized hidden states after every sharded residual mix.
Consequently the current layout retains communication and replicated
full-batch activations that the proposed layout could remove. More replicated
non-expert weights are its main memory tradeoff; with approximately 213 GiB
of daemon-owned weights per GPU, that is a binding constraint, not free space.

[vLLM's documented TP1/DP8/EP8 arrangement](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
provides the closest standard baseline: attention weights are replicated,
experts sharded. Independently keeping embedding/head TP8 requires separate
parallel groups or custom boundary modules, not just setting the global TP
flag. The current TP8 daemon layout also cannot transparently become TP1:
gathering non-expert parameter shards and retaining resident expert shards
would need explicit support. The diagnostic records a metadata-only estimate
of additional non-expert TP parameter bytes; it is a lower-bound capacity
estimate, not a measured serving-memory peak or proof of K3 backend support.

A read-only fetch of the daemon's rank-zero tensor/module metadata quantifies
the main obstacle without reconstructing tensors or moving any weights. Using
**actual input/output partition widths**, not the `tp_size` attribute alone
(replicated linears can also carry `tp_size=8`), removing TP from linear
parameters while retaining expert and embedding/head sharding adds at least:

| Component | Additional parameter storage per GPU (GiB) |
|--|--:|
| MLA projections | 8.490 |
| Recurrent/KDA projections | 50.665 |
| Shared experts | 19.811 |
| Other non-expert linears | 1.184 |
| Total lower bound | 80.150 |

Source: cluster diagnostic `21154-kimi-owner-weight-linear-layout`; these are
parameter-shape calculations, not allocations. Nonlinear sharded parameters,
temporary workspaces, and cache memory are excluded. Classify MLA/KDA from
actual parameter families rather than `layer_index % 4`: the checkpoint's
attention-layer list is one-based. At the existing approximately 213 GiB/rank
resident weight footprint, this fully local layout cannot fit on eight 256 GiB
GPUs even before adding a context cache. Keeping embedding/head TP does not
solve that limitation. Keeping KDA and shared experts sharded, using smaller
attention TP groups, quantizing those currently BF16 non-expert weights, or
using more EP GPUs are alternatives to investigate; none is a measured speed
improvement here. The MLA-only replication cost is much smaller, but would
not eliminate recurrent/residual communication throughout the entire model.

Before building this larger change, compare an integrated fixture with the
trained model at matching query/head geometry, measure residual gathers and
owner transport, and check non-expert weight capacity. Native ROCm latent-MoE
dispatch must be tested for the chosen groups; CUDA-only EP backend examples
must not be assumed compatible with this ROCm image.

## Owner decode extension (October 5)

The owner layout now hands each completed request's cache to a native
single-GPU fixed-address LoD pool on the same owner, configured for all 96
query heads in 16-head tiles. It keeps latent-plus-direct-key storage (576
channels, V aliases the first 512), the separate exact sink, native top-eight
routing, and 256-global-token updates per request. Query/output exchange and
normal TP8/EP8 model execution remain. The old prefill cache is released after
installation; it is not kept as another history archive.

The initial 24-layer, TP8/B8 fixture passed its step/update counters at 32K,
but an additional numerical uniform-attention test found incorrect decode
output. **Its timings are invalid and must not be used.** Counters alone were
not sufficient to establish a correct cache handoff. All eight workers did
execute 24 layers x 1,025 decode calls with four update calls per layer/owner;
the numerical issue was isolated on a single-GPU fixture before retrying.

The compact index table was initialized correctly. The actual fault was the
Gluon softmax's handling of an all-masked tile/split: subtracting `-inf` from
`-inf` produced NaN LSEs. The guard now preserves a zero-mass partial with
zero output and `-inf` LSE. It changes only the empty-field case and does not
mask arbitrary NaNs. The numerical test passed before, immediately before,
and immediately after a real 256-token update: for zero queries, random
latent values must reproduce the full-history mean followed by W_UV. The
first test passed with the fix in 8.70 s on node 3. Old timings remain invalid.

Raw fixture: [owner-decode fixture](../kimi-k3-mla-stack/oct5-owner-decode-fixture-32k.json).
Host tests: 85 passed, one GPU-only test skipped before the distributed fixture.

The trained B8/256K follow-up `21175-kimi-owner-decode-b8-256k` on node 4
was cancelled during warmup after the numerical failure; it provides no
valid timing. The daemon was retained. The intended retry replays the archived
dense ProLong prompt **and continuation hashes**, rather than generating a
different decode/MoE workload. One exact-shape warmup precedes one measured
1,025-step window; no new dense control is run. Audits must show all eight
requests stayed in one decode cohort and four updates occurred on every
owner/layer. This path is **eager**; the archived dense decoder is graph-captured,
so the comparison is explicitly exploratory, not a matched graph performance
claim. The cache-install transition is included, not performed in untimed setup.

The fixed repeat is `21183-kimi-owner-decode-b8-256k-fixed`, on node 4, with
the same frozen dense trace. The distributed 24-layer fixture is repeated
separately as `21182-kimi-owner-decode-fixture-32k-fixed` on node 2. Neither
run reloads or alters the trained weight daemon. Both retain phase reporting
outside timed generation so a quiet warmup is not mistaken for a hang.

Result destination: [trained owner decode, fixed](oct5-owner-decode-b8-256k-fixed.json).
Invalid warmup: [cancelled run](oct5-owner-decode-b8-256k.json).
Layout/weight-capacity plan: [request-centric layout](REQUEST_CENTRIC_LAYOUT.md).

Completed results (fixed softmax, one exact-shape warmup and one measured pass):

| Scope | Prefill (s) | Decode (ms/batched step) |
|:--|--:|--:|
| 24-MLA random fixture, B8/32K, owner/eager | 4.645 | 37.753 |
| Full trained K3, B8/256K, owner/eager | 347.569 | 157.650 |
| Archived full trained dense control, B8/256K, captured decode | 365.571 | 47.405 |

Both fixed owner runs completed. Each of the eight workers recorded 24 MLA
layers, 1,025 decode tokens and four global-256-token updates in every layer.
The full run passed its loaded-attention audit and the frozen trace replay
checks. Peak live Torch allocation was 30.343 GiB/rank (resident IPC daemon
weights excluded). The full-model owner prefill is modestly faster, but its
eager decode is much slower; this is not a production improvement and must
not replace the ordinary decode panel. No performance inference should be
drawn from comparing the small fixture's absolute latency to the full model.

Fixed fixture: [validated counter run](../kimi-k3-mla-stack/oct5-owner-decode-fixture-32k-fixed.json).
Dense control: [archived matched token cohort](oct4-full-b8-decode-256k512k-four-updates.json).

A narrower first prototype could replicate **only MLA projections**, keeping
KDA, shared experts and routed MoE in the existing TP8/EP8 layout. Run each
request's MLA projections, attention and W_O locally, then gather the final
hidden-width attention output for the following unchanged model computation.
Unlike the current owner prototype, this would replace query exchange,
attention-output exchange **and** the normal TP W_O reduction with one
hidden-output gather. That is a communication-volume/launch hypothesis, not a
measured speedup. It still needs the approximately 8.490 GiB/rank replicated
MLA weights and a measured memory budget; it does not make the recurrent or
MoE parts request-local. A full TP1/EP8 layout is not feasible with the current
resident weight footprint. More expert-parallel GPUs or validated non-expert
weight compression would be needed before considering that larger layout.

## Single-owner B1 prefill, full trained model

The next matched probe gives one request to one attention owner, within the
same eight-GPU node. All **93 trained layers** remain present. KDA, MoE,
Q/gate/output projections and model execution stay TP8/EP8; only LoD's full
latent history, centroids and attention evaluation are request-owned. Each
TP rank sends its twelve query heads to the owner, which computes all 96
heads and returns each rank's twelve 128-dimensional head outputs. No
two-node pipeline, omitted FFNs, fake batch rows or full Q/gate/O copies are
used. Small latent head maps are prepared from the resident weights before
warmup. B1 captured owner **decode** is not enabled by this prefill test.

The benchmark now allows `--owner-tp-mla --batch-size 1`. Its owner audit
requires one active attention owner but reports all eight TP workers, rather
than incorrectly requiring eight attention owners for a single request.
CPU tests cover the B1 head exchange, exact 16K per-row scheduling budget,
missing/extra owners and the still-B8-only captured-decode guard. Full-model
results are reported below; the half-model B8 improvement is not a B1 result.

All three arms use the same frozen real ProLong tokens, 16K row chunks,
native 2 GiB/rank cache reservation, full resident INT4-MoE weight cache,
normal non-eager operator selection, and identical configured context
capacity. Each exact shape is warmed before one uninstrumented measured
pass. Warmup-only scheduler audits verify the true token count and remove
their hooks before timing. Loaded-kernel, owner and memory audits occur
outside the measured generation interval. The timing includes LoD cache
construction before the first output token. We measure 32K as a short
runtime check, 128K as the requested comparison, and 256K/512K capacity and
speed where possible. Report seconds per request rather than extrapolating
B8 throughput to B1 latency.

Node 2's split daemon was stopped at the user's request. With explicit
approval, its regenerable 154 GiB Triton compilation cache was cleared;
weights and results were not deleted. The split checkpoint's original shards
are reused while the missing full-model shards are staged from Ceph to local
`/tmp` before loading. The full daemon uses cache ID
`kimi-k3-node2-full-int4-v1`. Node 2 restarted during staging; its local
checkpoint and both jobs were lost. Restaging now checks the remaining shard
bytes against free space and requires at least 128 GiB spare. Per the user's
instruction, the launcher defaults all Triton, TorchInductor, C++ extension,
vLLM, AITER and FlyDSL compilation/autotuning caches to local disk under
`/tmp/dan-agent`. This includes FlyDSL's separate runtime cache, which would
otherwise fall back to the shared image despite the AITER override. Existing
AITER/FlyDSL compiled modules are copied locally once under a lock to avoid
needless recompilation. Explicit caller cache overrides are preserved.
Already-running processes retain their launch environment; the resident
weight daemons are not restarted just to change compiler paths. Compilation
remains untimed.
Job `21267-kimi-k3-full-owner-b1-local-cache` waits for staging job `21266`, then
runs dense, single-owner LoD, and ordinary LoD sequentially on node 2;
quality evaluation continues independently on node 4.

October 6 recovery: all 96 shards have now passed source/destination byte-size
validation, the full 93-layer configuration/index checks passed, and 210 GiB
of local disk remains free. The broker is ready and is loading the full
weights just in time for the first dense client. Later clients reuse those
weights; no repeated full reload is intended. No new timing has completed yet.
All 96 weight shards finished loading on October 6; post-load processing and
export also completed, and clients now reuse the daemon's weights. A separate
launcher/import audit (`21277`)
confirmed all seven compiler cache paths reside under local `/tmp`, on XFS,
including the modules imported by AITER. The launcher/default/override tests
also passed.

The first client (`21267`) failed before measurement: its 1 GiB native-cache
reservation could not serve the configured 512K limit (vLLM required 1.76 GiB).
Job `21278` raised this to 2 GiB uniformly, and dense completed all four lengths.
Single-owner LoD measured 32K, then a benchmark-only audit incorrectly required
the routing binary on all eight ranks. Only rank 0 owns attention for B1;
the other seven correctly ran projections/MoE without loading that binary.
The audit now identifies active owners from independently validated query
counters and checks the binaries on those owners, while retaining all eight
worker reports. Missing/wrong owner binaries still fail. The focused suite
passed 106 tests (two GPU-only skips). Attention kernels and math are unchanged.
The dense control is retained; the owner and ordinary LoD arms resume alone.

| Context | Dense B1 prefill (s) | Single-owner LoD (s) | Ordinary LoD (s) |
|--:|--:|--:|--:|
| 32K | 4.186 | 5.765 | 4.189 |
| 128K | 19.300 | 28.230 | 17.391 |
| 256K | 45.497 | OOM during warmup | 37.112 |
| 512K | 118.586 | Not attempted after OOM | Not scheduled |

The interrupted owner's 32K measurement was 5.769 s, but it did not complete
the binary audit and is not promoted into the completed-result table. Dense
artifact: [oct5-owner-b1-control-full-node2-r2.json](oct5-owner-b1-control-full-node2-r2.json).

The resumed [single-owner run](oct5-owner-b1-tp-mla-prefill-node2-r3.json)
completed and passed owner/binary audits at 32K and 128K. Its prefill is
37.7% and 46.3% slower than dense respectively. At 256K, the owner's memory
pressure left insufficient room for a 1.75 GiB native MoE scratch allocation;
512K was not attempted. The subsequent [ordinary LoD control](oct5-owner-b1-control-two-tier-node2-r2.json)
completed all three scheduled lengths with audits passing. The proposed
[six 16-head owners](SIX_HEAD_OWNER_PREFILL.md) are a separate prefill-only
experiment, not a promotion of the unsuccessful single-owner B1 layout.

Run these sequentially with the K3 v10 environment documented in
[ProLong](PROLONG_QUALITY.md):

```bash
export TRITON_CACHE_DIR=/tmp/dan-agent/.triton/cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/dan-agent/torchinductor_cache
export VLLM_CACHE_ROOT=/tmp/dan-agent/vllm_cache
export AITER_JIT_DIR=/tmp/dan-agent/aiter-jit-k3
for mode in full; do
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_prefill_sweep \
    --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
    --mode "$mode" --lengths 32768 131072 262144 524288 --batch-size 1 --decode-tokens 1 \
    --tensor-parallel-size 8 --decode-context-parallel-size 8 \
    --weight-cache-id kimi-k3-node2-full-int4-v1 --kv-cache-memory-bytes 2147483648 \
    --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
    --reference-baselines results/kimi-k3-full-model-current/oct4-full-prefill-scale-b1-1g-16k256k.json results/kimi-k3-full-model-current/oct4-full-b1-512k1020k-decode1025.json \
    --audit-prefill-batches --report-memory --repeats 1 \
    --output "results/kimi-k3-full-model-current/oct5-owner-b1-control-${mode}-node2.json"
done
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --mode two-tier --owner-tp-mla --lengths 32768 131072 262144 524288 --batch-size 1 --decode-tokens 1 \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --weight-cache-id kimi-k3-node2-full-int4-v1 --kv-cache-memory-bytes 2147483648 \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --reference-baselines results/kimi-k3-full-model-current/oct4-full-prefill-scale-b1-1g-16k256k.json results/kimi-k3-full-model-current/oct4-full-b1-512k1020k-decode1025.json \
  --audit-prefill-batches --report-memory --repeats 1 \
  --output results/kimi-k3-full-model-current/oct5-owner-b1-tp-mla-prefill-node2.json
# Ordinary LoD control: omit --owner-tp-mla; test 32K/128K/256K.
# Its historical replicated archive can require more VRAM at 512K.
```

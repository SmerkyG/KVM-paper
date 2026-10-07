# Kimi K3: current full-model benchmarks

The fresh matched **prefill and decode** results are in
[CURRENT_TIMINGS.md](CURRENT_TIMINGS.md). It includes completed points while
the remaining sweeps run; blank cells are not estimates or older controls.
Run `python -m benchmarks.kimi_k3_current_timings` to refresh it without GPU work.
The preceding panel is preserved in [OCT6_VALIDATED_RESULTS.md](OCT6_VALIDATED_RESULTS.md).

Full trained K3, eight MI325X GPUs, TP8/DCP8/EP8. Dense uses the improved
Gluon decoder. Both modes now use the approved G8 direct-state-I/O KDA prefill
baseline, resident packed INT4 **MoE weights**, and real frozen ProLong tokens.
The supplied v10 image userspace reports vLLM
`0.30.1rc1.dev143+g29468dde8`; the direct runner keeps compilation on local disk.
Attention storage is BF16. B1 LoD uses DCP8; B8 uses one request's attention
per GPU while retaining native TP projections, KDA and MoE. Global per-request
cadences remain 16K prefill / 256 decode, top-eight routing, and the existing
1,024-leaf opening cap. Long B1 uses the [token-sharded archive](TOKEN_SHARDED_PREFILL.md).

Every speed point has one untimed exact-shape warmup, one measured pass,
1,026 output tokens / 1,025 timed decode steps, and four audited LoD updates
per request in every MLA layer. All B8 requests remain live. Compilation,
prefix hits, internal profiler events and failed warmups are excluded.

## October 7 routing fusion

The trained 32K B8 prefill query-tile trial showed no serving gain
(33.77995 s with 128 query rows versus 33.78116 s with 64), so the current
128-row tile is retained. Its dispatch correction and raw trial are recorded
in [CACHED_ROUTING.md](CACHED_ROUTING.md).

The safe request-owner path now fuses exact top-eight reduction and centroid
union construction, removing one launch per MLA layer. Distributed B1 keeps
the required global candidate merge before constructing its union.

| Trained B8 workload | Separate union, ms/step | Fused union, ms/step | Latency reduction |
|--:|--:|--:|--:|
| 16K | 30.217 | 30.161 | 0.18% |
| 64K | 30.931 | 30.897 | 0.11% |

These small one-pass serving gains are **not** the 2.23% fixture-only gain.
Both arms use the same engine, weights, frozen prompts/continuations and KDA
baseline, but separately audited decode graphs. All 192 layer/rank live-cache
checks pass: selected sets/counts agree exactly, outputs and LSE agree within
floating-point tolerance (outputs are not bitwise identical).
Sources: [16K](oct7-trained-decode-union-b8-16k.json),
[64K](oct7-trained-decode-union-b8-64k.json), [kernel/fixture evidence](CACHED_ROUTING.md).

## Current sweeps and capacity work

B1 dense and LoD have completed every point through 1020K. At 1020K, LoD
prefill is **178.284 s versus 342.108 s** dense (1.919×); decode is
**22.611 versus 28.049 ms/step** (1.240×), including four catch-ups.
Fresh dense B8 completes through 512K (939.950 s / 61.130 ms per decode step);
its 1020K warmup fails for lack of device resources and is not a speed result.
Request-owned B8 completes through **512K**: 849.782 s prefill / 33.163 ms
decode, versus 939.950 s / 61.130 ms dense (1.106× / 1.843×). The failed
earlier 512K attempts remain separate from this completed, four-update point.
The completed six-head B8 configuration regressed in prefill and is preserved
in [the superseded-run log](OCT7_B8_UNSHARED.md), not mixed into the replacement
table. Its allocator audit reports pressure-induced reclamation: rank 0
reclaimed on 48 of 64 checks for the 64K warmup and measured pass combined.
Those counters establish pressure, not its exclusive time cost.

With shared construction scratch, the same-engine 32K trial records
**33.9295 s with six heads versus 33.7891 s with twelve** (0.414% lower latency).
All eight first tokens agree. This is a small one-pass grouping difference,
not a statistically established large gain. The two arms have identical
prompts, but this short prefill-only trial's prompt builder differs from the
canonical forced-continuation panel. Its absolute times are therefore not
substituted into that panel or used to attribute the preceding slowdown.
[Raw comparison](oct7-owner-head-groups-32k-cap256k.json).

The default now retains twelve-head projection groups for short live prefixes,
even when the request's capacity is larger. The existing live-leaf memory
bound still reduces groups as needed; the 1020K capacity attempt keeps its
explicit two-head bound. Shared construction scratch and this geometry
have been remeasured on every canonical B8 point through 512K. The 512K
capacity configuration additionally uses compact directories, four-head
fine projection groups, sharded prefill bank and 4K MoE slices. It retains
all selected leaves and changes no routing or update cadence. The 1020K B8
point remains capacity work, not a completed timing result.
The fresh canonical 256K point is **330.056 s / 32.571 ms per decode step**,
versus dense's **360.076 s / 48.105 ms** (1.091× prefill, 1.477× decode). At 16K,
the measured client allocation peak falls from 27.419 to 21.595 GiB; this is
an allocator peak, not a reduction in weight storage or permanent KV size.
The sweep runner saves every completed point before
attempting a longer context. These are fresh measurements, not rerenders of
the previous controls. The [memory investigation](LONG_CONTEXT_MEMORY.md)
separates permanent latent storage, native reservations and reusable scratch.
Completion of a storage-only fixture does not certify million-token generation.

The first B8 retry exposed oversized decode-update score workspace: 256-token
catch-up inherited a 16K prefill reservation. The corrected workspace follows
the actual overflow in 256-token buckets; scores and centroid choices do not
change. That bounded-score configuration completed B8 warmup **and measured
generation** through 256K, including the fresh shared-scratch panel.
Long owner prefills additionally share construction scratch across serial
layers and release it at the decode handoff. Million-token B8 remains an
actual capacity test, not a claimed success before it completes.
The bounded MoE setting for that capacity test is preserved during owner
setup; it no longer gets overwritten by the 16K attention scheduler budget.
The trained million-token engine now initializes with 4K MoE slices and a
startup-only allocator cleanup before the dummy sampler. Generation still
runs out of VRAM: initially in KDA, then in MoE after sharding only the
prefill residual bank. The 2K MoE-slice retry and the subsequent compact
page-directory retry still fail with HIP/HSA device-resource errors in
warmup. The compact directory saves a verified 1.684 GiB/rank without
dropping any leaf or changing membership. A new trained **32K B8** diagnostic
with the complete **1020K reservation** completes prefill and its first
decode step after bounding both local and centroid projection workspaces.
It is a capacity preflight, not a million-token timing or speed improvement
claim. GPU tests retain exact selected sets and numerically equivalent
attention/LSE; the grouped projection paths remain opt-in.
The longer 256K live-prefix control at the same million-token reservation
still fails. Neither reducing the native reservation to 512 MiB nor adding
idle-block reclamation before residual collectives completes that control.
The unsuccessful per-collective reclamation code was removed, and native
cache reservation remains 1 GiB. Reclaiming leaves of permanently closed
centroids is the next storage opportunity, not an implemented saving.
Attention chunks, routing and update cadences do not change; eight-token B8
decode retains its native residual-bank path. Initialization alone is not
a fit claim. See [exact memory trials](LONG_CONTEXT_MEMORY.md).

With the image runtime and a resident full-model daemon, the sequential runner
does not require the proprietary cluster runner:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_refresh_panel \
  --mode two-tier --batch-size 1 --checkpoint /tmp/local/Kimi-K3 \
  --weight-cache-id YOUR_CACHE_ID
```

Use `--mode full` for the dense control and `--batch-size 8` for the B8 panel.
Run only one timing engine per node; compilation artifacts must stay on local
disk. `--block short` measures through 256K for LoD (128K for B8 dense), while
`--block long` measures the remaining lengths. Frozen prompt and continuation
identities are checked against the recorded cohorts. These are timing traces,
not freely generated model-quality scores.
Resuming skips individual audited points, including those saved before a
later capacity failure, without rerunning them or changing their stored times.

The broad CPU regression check passed **697 tests** (99 GPU-only cases skipped):

```bash
python -m pytest -q tests/test_kimi*.py tests/test_benchmarks.py tests/test_attention_timing.py
```

The changed cache/score-workspace paths were also checked on GPU: exact
scores and indices, independent versus shared per-layer construction, and
captured pool-backed decode. The trained fusion A/B above checks actual
live-cache selected sets and numerical output/LSE agreement on every rank.
The projection-group GPU test additionally compares twelve, eight, six, four
and two heads, with bitwise-equal output/LSE for both ordinary and compact
selected-leaf projection (four cases passed).
After the final grouping/startup-reporting changes, the focused benchmark and
policy suite passed 106 tests (three GPU-only cases skipped), including the
profile-only allocator cleanup. Startup allocation failures are now logged as
failed initialization with no invented timing points.
The follow-up capacity-reporting/lifecycle suite passes 53 tests (three
GPU-only cases skipped). Its membership observer now samples a still-live
cache when vLLM returns before cleanup; empty post-cleanup counts are never
used to claim a reclamation opportunity.

Existing quality evidence remains in [PROLONG_QUALITY.md](PROLONG_QUALITY.md)
and [CHAT_QUALITY.md](CHAT_QUALITY.md); it is not relabeled as a new fusion
quality run. Rejected and historical experiments remain in
[HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md).

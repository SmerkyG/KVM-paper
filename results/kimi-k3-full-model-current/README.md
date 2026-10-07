# Kimi K3: current full-model benchmarks

The fresh matched **prefill and decode** results are in
[CURRENT_TIMINGS.md](CURRENT_TIMINGS.md). It includes completed points while
the remaining sweeps run; blank cells are not estimates or older controls.
Run `python -m benchmarks.kimi_k3_current_timings` to refresh it without GPU work.
The preceding panel is preserved in [OCT6_VALIDATED_RESULTS.md](OCT6_VALIDATED_RESULTS.md).

Full trained K3, eight MI325X GPUs, TP8/DCP8/EP8. Dense uses the improved
Gluon decoder. Both modes now use the approved G8 direct-state-I/O KDA prefill
baseline, resident packed INT4 **MoE weights**, and real frozen ProLong tokens.
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
Fresh dense and request-owned B8 sweeps continue on separate nodes.
The fresh B8 prefill points currently regress against dense. This engine's
256K capacity chooses six-head projection groups even at short live lengths,
where the previous trials used twelve. A same-engine 32K comparison of six
and twelve heads, at that same capacity, is queued to isolate this choice;
neither group changes routing or the attention approximation. These partial
points are fresh observations, not yet a claim of the fastest B8 configuration.
The sweep runner saves every completed point before
attempting a longer context. These are fresh measurements, not rerenders of
the previous controls. The [memory investigation](LONG_CONTEXT_MEMORY.md)
separates permanent latent storage, native reservations and reusable scratch.
Completion of a storage-only fixture does not certify million-token generation.

The first B8 retry exposed oversized decode-update score workspace: 256-token
catch-up inherited a 16K prefill reservation. The corrected workspace follows
the actual overflow in 256-token buckets; scores and centroid choices do not
change. The new B8 warmup **and measured generation** complete through 128K so far.
Long owner prefills additionally share construction scratch across serial
layers and release it at the decode handoff. Million-token B8 remains an
actual capacity test, not a claimed success before it completes.
The bounded 8K MoE setting for that capacity test is preserved during owner
setup; it no longer gets overwritten by the 16K attention scheduler budget.

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

Existing quality evidence remains in [PROLONG_QUALITY.md](PROLONG_QUALITY.md)
and [CHAT_QUALITY.md](CHAT_QUALITY.md); it is not relabeled as a new fusion
quality run. Rejected and historical experiments remain in
[HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md).

# Kimi K3: current full-model benchmarks

The current matched **prefill and decode** results are in
[CURRENT_TIMINGS.md](CURRENT_TIMINGS.md). Cells are filled as completed
measurements pass runtime and cohort audits; blanks are not estimates.
The [pre-fix October 7 panel](OCT7_PRE_FIX_TIMINGS.md) and
[investigation/status notes](OCT7_PRE_FIX_STATUS.md) are historical, not
current default timings. October 6 results remain in
[OCT6_VALIDATED_RESULTS.md](OCT6_VALIDATED_RESULTS.md).

## Current setup

Full trained K3, eight MI325X GPUs, TP8/DCP8/EP8. Both modes use the approved
G8 direct-state-I/O KDA prefill baseline, the same resident packed INT4 **MoE
weights**, and real frozen ProLong prompts and teacher-forced continuations.
Attention storage remains BF16; chronological latents are not quantized.
Dense decode uses the improved Gluon kernel, now with **device-side live
context splits**, independent of maximum reservation. Eager caches use
expandable allocation; AITER graph buffers use ordinary registered IPC
allocation. See [validation and diagnosis](LIVE_SPLITS_ALLOCATOR.md).

B1 LoD uses DCP8; B8 places one request's attention per GPU while retaining
native TP projections, KDA and MoE. Global per-request cadences are 16K
prefill / 256 decode, top-eight routing, a 512-token local window, and the
1,024-leaf opening cap. The [long B1 archive](TOKEN_SHARDED_PREFILL.md) is
token-sharded. B8 512K retains compact directories, a sharded prefill bank,
four-head fine projection groups and 4K MoE slices; all leaves remain stored.
These are physical memory/layout choices, not different routing policies.

Each point uses one untimed exact-shape warmup, one measured pass, 1,026
output tokens / **1,025 timed decode steps**, and four audited global-256
LoD updates per request in all 24 MLA layers. All B8 requests remain live
throughout measured decode. Compilation, warmup, prefix hits, preemption,
profiling events and failed capacity attempts are excluded.

The v10 userspace reports vLLM `0.30.1rc1.dev143+g29468dde8`. The direct runner
keeps compilation artifacts on local disk. Resident weight caches are
`kimi-k3-shared-int4-v6` on node 4 and `kimi-k3-node2-scratch0-v2` on node 2;
reuse the existing ID rather than loading another full weight copy. See
[memory work](LONG_CONTEXT_MEMORY.md) for capacity history. Fresh B1 measurements
complete through 1020K and B8 LoD through 512K. Dense B8/512K is finishing;
fresh dense and LoD B8/1020K attempts are queued behind it. B8 1020K has
**not** yet completed generation and is not a timing cell.

## Reproduction

With the documented image runtime, local checkpoint staging and a resident
weight daemon, no proprietary cluster runner is needed:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_refresh_panel \
  --mode two-tier --batch-size 1 --checkpoint /tmp/local/Kimi-K3 \
  --weight-cache-id YOUR_CACHE_ID

bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_refresh_panel \
  --mode full --batch-size 8 --upper-context 524288 \
  --checkpoint /tmp/local/Kimi-K3 --weight-cache-id YOUR_CACHE_ID

python -m benchmarks.kimi_k3_current_timings
```

Run only **one timing engine per node**. Both arms share the frozen token
cache and recorded continuations. The runner uses 16K scheduler chunks, the
shared tuned MoE configuration and the appropriate exact-storage memory
settings automatically. `--block short` measures through 256K for LoD/B1
or 128K for B8 dense; `--block long` measures the remaining lengths.
Resume skips individual audited points, including those saved before a
later capacity failure, without source-hash guards or overwriting times.
The renderer accepts only the post-fix sweep and the explicitly validated
final-policy controls, and requires registered graph communication,
expandable eager allocation and (for dense) live-context splits on all ranks.
Raw JSON records prompts, continuations, timings, policies and memory audits.

Quality evidence remains in [PROLONG_QUALITY.md](PROLONG_QUALITY.md) and
[CHAT_QUALITY.md](CHAT_QUALITY.md). These timing traces are not new freely
generated quality scores. Historical or rejected work remains in
[HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md).

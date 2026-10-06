# K3 decode: parallel reduction and compact routing bookkeeping

This is the preceding pipeline baseline. The later
[cached-centroid / physical-B1 split follow-up](CACHED_ROUTING.md) measures
21.626 / 32.229 / 30.872 ms/step at 64K for ordinary B1 / ordinary B8 /
request-owned B8. The historical comparisons below remain unchanged.

October 6, 2026. These changes optimize the corrected two-tier decoder, not
prefill or the amount of exact attention. The earlier spill-free router remains
enabled. Dense controls use the improved Gluon decoder, not AMD's older
precompiled kernel. Attention caches remain BF16; both model modes use the same
resident transformed INT4 MoE weights.

## Trained-model results

65,536-token real ProLong prompts, TP8/DCP8/EP8. Units are **end-to-end
ms/batched decode step**, not per-layer timings or values divided by batch.

| Layout | Dense | Four-wave router baseline | New pipeline, 64 splits | Current live-split pipeline |
|:--|--:|--:|--:|--:|
| Ordinary DCP8, B1 | 21.845 | 23.004 | 22.025 | 22.025 |
| Ordinary DCP8, B8 | 34.198 | 35.364 | 34.251 | 32.536 |
| One request's attention per GPU, B8 | 34.198 | 31.403 | 31.245 | 31.245 |

The current ordinary pipeline reduces latency by **4.3% at B1 and 8.0% at
B8** versus the preceding four-wave router. It is **0.8% slower than dense
at B1**, effectively close to parity in this one-pass comparison, and **4.9%
lower latency than dense at B8** (1.051x dense/LoD). The much larger isolated
kernel speedups below must not be substituted for these full-model results.

The request-owner layout remains fastest at B8: **4.0% lower latency than
optimized ordinary DCP8 and 8.6% lower than dense** (1.095x dense/LoD).
Its 0.5% change from the preceding owner run is small and is not strong evidence
of an additional serving speedup. Each GPU runs all 96 MLA heads for one
request; native TP projections, W_O, MoE and KDA remain distributed. This is
the existing opt-in attention layout, not whole-model replication. It benefits
from parallel split reduction but does not use distributed top-eight selection
or the B8 consumer-split reduction: each owner executes attention at physical B1.

Every test warms the complete shape and then measures one pass of 1,026 output
tokens / **1,025 timed steps**, including four per-request global-256 updates
in all 24 MLA layers on all eight ranks. Frozen prompt/continuation hashes,
fully live batches, no prefix hits/preemptions, and loaded attention modes
pass the same validator as the archived controls. The final B8 audit also
checks the new rank-major route path and actual 16-split orchestration in
all 192 rank/layer pools. There are no profiler events in the serving window.

B1 and the intermediate B8 64-split run used node 2; current ordinary B8 and
request-owned B8 used node 4.
Nodes are treated as equivalent as requested. Both reuse idle resident weight
daemons, with no competing model workload. B1 retains 263,186-token capacity
and 1 GiB native cache; B8 retains 66,578-token capacity and 3 GiB native cache.
Measured prefill was 8.51 s / 68.32 s at B1/B8; prefill code did not change,
so these observations are not claimed as a prefill optimization.
Owner prefill was 69.13 s with eight 2K row slices, versus ordinary prefill's
single 16K slice. Both retain global per-request 16K semantic construction.
All eight owners report 1,025 actual eight-row graph replays, and each of their
24 MLA layers records four global-256 updates and 1,025 decoded tokens. Frozen
prompts/continuations and capacity/native-cache settings match the preceding
owner run; its audited physical consumer uses 64 splits in every owner pool.

Sources: [new B1](oct6-parallel-decode-b1-64k.json),
[intermediate B8](oct6-parallel-decode-b8-64k.json),
[current B8](oct6-parallel-decode-b8-split16-64k.json),
[current request-owner B8](oct6-parallel-decode-owner-b8-64k.json),
[preceding four-wave results and dense controls](ROUTER_DECODE.md).
The [power-of-two table](DECODE_POWER2.md) updates only these measured 64K
ordinary cells; improvements are not extrapolated to other contexts.

## Implementation

1. **Parallel split-output reduction.** Replace the serial 64-split softmax
   merge with a stable FP32 maximum, weight sum, and weighted value sum. Each
   query head uses two 256-channel workgroups, four waves each. Derive initialized
   splits from that head tile's actual sequence length. Never read unwritten
   suffix outputs or values from all-masked splits. When advancing a request's
   global/DCP ownership lengths, exactly one workgroup does so.
2. **Two-kernel distributed routing bookkeeping.** Pack scores and integer slot
   IDs into fixed FP32 storage; gather along dimension zero; select global
   top-eight candidates and write this rank's owned routes directly into the
   existing buffers. The collective, ranking, ownership and deterministic
   rank-major ties are unchanged. This removes casting, layout rearrangement,
   ownership-mask and copy-back launches. The CPU/reference implementation is
   retained for correctness checks.
3. **Avoid oversplitting the compact consumer at live B4+.** In the 96-head
   native head-tiled path, use 16 rather than 64 attention splits. B8 already
   supplies 768 workgroups; extra splits create masked tiles and intermediate
   traffic. This measurement used 64 at B1--B3 and in legacy per-GQA layouts.
   The later cached-routing follow-up changes physical B1 to 32.
   Strided views reuse the
   preallocated 64-split capacity, so capture/replay requires no new allocation.
   The preceding request-owner measurement below used physical B1 / 64 splits.

No selected KV tokens are dropped. Top-eight selection, 1,024-leaf closure,
separate sink, coarse replacement, global 16K prefill cadence and global
256-token decode cadence are unchanged. Parallel FP32 sums and different split
boundaries can change numerical roundoff; this is not a bit-identical output
claim or a new NIAH/LongBench quality result.

Code: [consumer and reducer](../../lod_attention/kernels/kimi_gluon_decode.py),
[routing bookkeeping](../../lod_attention/kernels/distributed_topk.py),
[decode orchestration](../../lod_attention/kernels/paged_decode.py).

## Isolated checks, not serving timings

| Frozen synthetic test | Before | After |
|:--|--:|--:|
| B1, 64-split reduction | 11.058 us | 3.489 us |
| B8, 64-split reduction | 23.951 us | 8.431 us |
| B1 distributed route bookkeeping, excluding collective | 24.571 us | 5.260 us |
| B8 distributed route bookkeeping, excluding collective | 29.221 us | 5.577 us |

Source: [reducer sweep](../kimi-k3-mla-stack/oct6-parallel-reducer-tune.json),
[route bookkeeping](../kimi-k3-mla-stack/oct6-rank-major-route-plumbing.json).
These are graph-replay GPU timings on frozen tensors, not full-model latency.

The complete compact consumer, including its reducer, is faster at 16 splits
than 64 on the tested B8 sets of 832 / 2,624 / 6,208 KV rows:
39.270 / 68.858 / 135.923 us versus 117.474 / 118.359 / 186.727 us.
See [split sweep](../kimi-k3-mla-stack/oct6-parallel-consumer-splits.json).
The same choice stays faster on much larger synthetic B8 compact sets:
424.59 vs 493.21 us at 24,640 KV rows, and 1,569.35 vs 1,622.51 us at
81,984 rows. Output/LSE reference checks pass (maximum output error below
0.00025). These are attention-work scaling checks, not new trained-model
long-context benchmarks. Source:
[large split check](../kimi-k3-mla-stack/oct6-parallel-consumer-large-splits.json).

## Captured attention-only fixture

Twelve dummy-weight MLA layers, 64K, eight GPUs, corrected DCP dispatch and
tail conversion; one measured pass of 1,025 steps including four updates.
These are not trained-model end-to-end results.

| Batch | Dense | Four-wave router baseline | New pipeline, 64 splits | New pipeline, live splits |
|--:|--:|--:|--:|--:|
| 1 | 2.116 | 2.585 | 2.160 | 2.160 |
| 8 | 3.950 | 4.691 | 4.065 | 3.411 |

Units: ms/batched step. The new B8 fixture is 13.7% lower latency than dense.
Sources: [dense fixture](../kimi-k3-mla-stack/oct6-corrected-decode-profile12/),
[four-wave baseline](../kimi-k3-mla-stack/oct6-router-optimized/),
[new fixture](../kimi-k3-mla-stack/oct6-parallel-decode/).

A separate 257-step diagnostic trace (outside the measured serving window)
shows B8 compact consumer / split reducer averages falling from 108.15 / 14.80 us
to 66.95 / 7.20 us per rank-local layer invocation with 16 splits. Instrumented
durations are diagnostic, not additive estimates of serving wall time.

## Verification and reproduction

The CPU regression subset passes **397 tests**, with **54 device-only skips**.
The final targeted GPU subset passes **42 tests**: poisoned unwritten/empty
splits, broad and uniform attention mass, strided 16-split storage, graph replay,
indirected DCP length advancement exactly once, exact global route ownership,
fixed pointers/shared scratch, and owner decode across updates. Split-sweep
outputs and LSE also match the independent full-attention reference.

Run from the repository root, in the pinned K3 v10 GPU environment:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m pytest -q \
  tests/test_kimi_split_reduce.py tests/test_distributed_topk.py \
  tests/test_decode_scratch.py \
  tests/test_kimi_request_prefill.py::test_owner_decode_preserves_uniform_attention_mass_across_update \
  tests/test_kimi_owner_decode.py::test_pool_backed_owner_prefill_preserves_cache_and_captured_decode

bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_reduce_tune \
  --output reducer.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_route_plumbing \
  --output routing.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_consumer_tune \
  --output consumer.json
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_consumer_tune \
  --large-only --output consumer-large.json
```

For trained-model measurements, use the exact environment exports and B1/B8
commands in [ROUTER_DECODE.md](ROUTER_DECODE.md#reproduction), with distinct output
files. The new pipeline and live-batch split choice are automatic. Reuse the
resident transformed-weight daemon and the same frozen ProLong traces; do not
cold-load/reconvert weights. Keep compiler artifacts on local `/tmp/dan-agent`.
Compile/warm the complete shape before the one measured generation, and never
run another model workload on its GPUs during timing.
For request-owned B8, use [the owner command](OWNER_CAPTURED_DECODE.md#reproduction)
with `--lengths 65536`. Retain its 2K row slices, `--owner-tp-mla`, eight live
rows, 66,578-token capacity and 1,026 output tokens; only the output filename
changes. No extra decoder switch is needed.

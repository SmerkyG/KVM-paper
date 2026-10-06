# K3 decode router: spill-free geometry

This records the router-only baseline. Parallel split reduction, compact
distributed routing bookkeeping and the live-batch split geometry are tested
in [the subsequent decode-pipeline report](../kimi-k3-full-model-current/DECODE_PIPELINE.md).

October 6, 2026, MI325X/gfx942. This changes launch geometry only: the
16-query-head by 64-centroid tile uses four waves instead of one for K3's
512-latent + 64-direct-key geometry. Smaller MLA geometries, other model
families, prefill, top eight, count-corrected scores, the 1,024-leaf closure
rule, and global per-request update cadences are unchanged. Both ordinary
DCP and request-owned decode inherit the same profile.

## Isolated router

CUDA-graph replay of the production router with 96 query heads, six
16-head virtual groups aliasing one KV head, BF16 state sums, and FP32
counts. Compiler warmup and reference checks are outside timing. These
are kernel durations, **not full-model or serving speedups**.

| Batch | Active centroids per request/rank | Old, us | Four waves, us | Router speedup |
|--:|--:|--:|--:|--:|
| 1 | 512 | 103.080 | 12.170 | 8.47x |
| 8 | 512 | 125.470 | 23.906 | 5.25x |
| 1 | 4,096 | 125.575 | 23.952 | 5.24x |
| 8 | 4,096 | 699.046 | 129.540 | 5.40x |

The old kernel uses 512 VGPRs, reports 603 VGPR spills and 130 SGPR
spills, and reserves 1,592 bytes of private scratch per thread. Four waves
use 252 VGPRs, zero VGPR spills, three SGPR-to-VGPR spills and **zero private
scratch**. Two waves avoid private scratch but consume 483 VGPRs and are
slower; eight waves are also slower. The 32-centroid tile helps isolated B1
at 512 centroids but loses at B8 and larger states and emits twice as many
candidate groups. The 128-column tile exceeds the 64 KiB LDS limit. Keep
64 columns and the existing two-stage compilation policy.

Initial sweep: [raw geometry timings](oct6-router-tune.json). The initial
strict FP32-reference comparison rejected two near-tied ranks at B8/4,096
for both old and new geometries; this is not evidence of a regression.
The completed follow-up verifies **bit-identical old/new candidate scores
and indices** at both state sizes and batches, with ragged lengths, row
indirection, zero counts, cap-boundary counts, exact ties and already
materialized mean keys. Maximum discrepancy from the separate torch FP32
reference is 3.34e-6. The two reference-rank differences are the same for
every 64-column geometry; they do not change original router outputs.
See [bitwise validation](oct6-router-tune-validated.json). Materialized
mean keys do not have the original spills; their 512-state B8 corner is
slightly faster at one wave (11.26 versus 12.75 us). Four waves target the
state-sum scoring path actually observed in the slow decoder.

## Captured 12-layer attention-only fixture

65,536-token prompts, TP8/DCP8, all 96 heads, dummy weights, no FFN/MoE/KDA.
One full-shape warmup and one uninstrumented serving measurement; 1,025
timed decode steps contain four global-256 updates of every live request
in every layer/rank. The fixture is a performance diagnostic, not model
quality evidence. Dense controls are the corrected attention-only controls
from [the original diagnosis](DECODE_PROFILE.md); they were not rerun.

| Live batch | Dense, ms/step | Old LoD | Four-wave LoD | LoD latency reduction |
|--:|--:|--:|--:|--:|
| 1 | 2.116 | 3.736 | 2.585 | 30.8% |
| 8 | 3.950 | 5.893 | 4.691 | 20.4% |

The separate 257-step rank-0 B1 trace measures the router at 13.290 us per
layer versus 105.03 us before. Its sum across twelve layers falls from
1.260 to 0.159 ms/step. B8 falls from 130.19 to 28.001 us/layer.
These trace sums explain the serving improvement;
they are not substituted for serving latency or attention-core speedups.
Ordinary LoD is still slower than dense on this fixture.

Sources: [B1](oct6-router-optimized/b1-lod.json),
[B8](oct6-router-optimized/b8-lod.json).

The same optimized fixture with one complete request's attention per GPU
runs at **2.500 ms/B8 step**, compared with 4.691 ms for ordinary DCP and
3.950 ms for dense. All eight ranks verify 1,025 real B8 model graph
replays and four updates in all twelve owner caches, with zero first/last
token spread, cache hits or preemptions. Native TP projections and their
output exchange are included; distributed routing and LSE merging are not
needed in this layout. This is not an old/new owner comparison: there is no
matched pre-optimization 64K owner fixture control. Nor is its 1.58x dense
speedup a full-model claim. See [owner fixture](oct6-router-optimized/b8-owner.json).

GPU correctness checks also pass for uniform attention mass across an
actual 256-token update and for both owner-prefill backing layouts followed
by captured decode (three cases). The CPU regression subset passes 378
tests; 48 device-only cases are skipped in that CPU environment.

The subsequent [full trained-model comparison](../kimi-k3-full-model-current/ROUTER_DECODE.md)
confirms 7--10% lower end-to-end LoD decode latency at 64K, including
the request-owner layout. Those serving results are separate from this fixture.

## Reproduction

From the repo root, in the pinned K3 v10 environment. Compilation artifacts
must use local disk; the launcher defaults to `/tmp/dan-agent`.

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_decode_route_tune --output router-tune.json

# Eight GPUs; ordinary DCP fixture.
bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_decode_fixture --mode two-tier --batch-size 8 \
  --output decode-fixture-b8.json

# Same fixture, one complete request's attention per GPU.
bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_decode_fixture --mode two-tier --batch-size 8 \
  --owner-decode --skip-profile --output decode-fixture-owner-b8.json
```

The diagnostic fixture uses deterministic synthetic prompts and frozen
continuations, seed zero, dummy weights and twelve MLA layers by default.
The optional profiler runs only after the uninstrumented serving pass.

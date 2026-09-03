# Muse-Glimmer two-level prefill subkernel profile

## Setup

This profiles ordinary two-level routed prefill on Muse-Glimmer-30B at 64K,
batch 8, using eight distinct real ProLong documents. LOD storage is BF16,
the scheduler token budget is 16K, and the LOD logical prefill/state-update
chunk is 4K. The leaf-visit cap is disabled.

Despite `VLLM_LOD_OPEN_COUNT=8`, the vLLM pool sets the effective prefill open
count to three. The audited global-layer geometry is 32 query heads, two KV
heads, D=128 (GQA=16), across 13 global NoPE layers.

The event profiler substantially changes scheduler interleaving: its wall
times are 94--99 seconds rather than the uninstrumented 56.508-second LOD
control. Therefore the profile wall times are not speed results. GPU event
durations and their within-pass shares are the useful measurements below.

## Fine LOD breakdown

Times aggregate the complete 8 x 64K instrumented prefill across all 13 global
layers. The disjoint measured LOD total is 17,281.0 GPU-ms: complete two-level
attention plus local attention, state update, and page append.

| component | GPU time | calls | mean/call | share |
|---|---:|---:|---:|---:|
| centroid mean K | 83.2 ms | 2,028 | 41.0 us | 0.5% |
| dense centroid QK logits | 1,026.9 ms | 2,028 | 506.4 us | 5.9% |
| **top-3 scan and selection** | **6,728.5 ms** | 2,028 | **3,317.8 us** | **38.9%** |
| other routing wrapper work | 33.7 ms | 2,028 | 16.6 us | 0.2% |
| centroid mean V | 66.4 ms | 2,028 | 32.8 us | 0.4% |
| **coarse attention/PV** | **3,797.0 ms** | 2,028 | **1,872.3 us** | **22.0%** |
| other coarse wrapper work | 28.1 ms | 2,028 | 13.9 us | 0.2% |
| exact-leaf expert dispatch | 1,123.6 ms | 2,028 | 554.1 us | 6.5% |
| exact-leaf attention | 938.6 ms | 2,028 | 462.8 us | 5.4% |
| exact-leaf route reduction | 168.7 ms | 2,028 | 83.2 us | 1.0% |
| exact-leaf pack/other | 72.4 ms | 2,028 | 35.7 us | 0.4% |
| branch masking/merge and other two-level work | 442.2 ms | 2,028 | 218.1 us | 2.6% |
| AITER exact-local attention | 1,504.5 ms | 2,028 | 741.9 us | 8.7% |
| state update | 886.0 ms | 1,053 | 841.4 us | 5.1% |
| page/leaf archive append | 381.0 ms | 1,118 | 340.8 us | 2.2% |

Exact-leaf dispatch itself breaks down as follows:

| dispatch component | GPU time | mean/call |
|---|---:|---:|
| route preparation | 129.3 ms | 63.8 us |
| expert sort | 410.0 ms | 202.2 us |
| unique-expert grouping | 292.3 ms | 144.1 us |
| block-list construction | 276.0 ms | 136.1 us |

## Native comparison

A matching full-attention instrumentation pass measured 15,993.6 GPU-ms in
the production native global-attention forwards (442 calls, 36.18 ms/call).
The fine LOD pass measured 17,281.0 GPU-ms of disjoint LOD phases. A coarser
LOD event pass, whose lower instrumentation overhead produced fewer scheduler
fragments, measured 15,778.0 GPU-ms. Thus LOD uses approximately as much GPU
time as native global attention on Muse, rather than substantially less.

The scheduler/catch-up structure compounds this. Native full attention was
called 34 times per global layer. LOD's complete two-level attention was
called 94 times per layer in the coarse profile and 156 times per layer in the
fine profile. The exact count changes when event overhead changes prompt/decode
interleaving, but both demonstrate substantially more serial 4K work than the
native path.

For reference, uninstrumented medians remain 56.508 seconds for ordinary LOD
and 51.933 seconds for full attention.

## Diagnosis

The dominant problem is not exact leaves. The dense centroid QK matmul is
efficient (5.9%), but the subsequent scalar/reduction-heavy top-3 scan costs
6.5 times as much as QK and consumes 38.9% of all measured LOD GPU time. The
stable coarse pass then scans the materialized logits and centroid values
again for another 22.0%.

At GQA=16, the routed top-k program groups 16 query positions across all 16
GQA heads, creating 256 logical rows per program while retaining a padded
top-four result for each row. It must reread the complete materialized
query-by-centroid score field and perform an online reduction. This is much
less GPU-efficient than the preceding MFMA QK matmul. The coarse kernel is a
second full centroid-field scan because stable recomputation is enabled.

The result also explains the leaf-cap experiment. Direct exact-leaf attention
is only 5.4% of measured LOD GPU work; dispatch is another 6.5% and is not
removed by truncating posting lists. A 16-leaf visit cap therefore cannot fix
Muse's prefill deficit.

The most promising corrective direction is a tiled hierarchical top-k: emit a
small local top-3 from each regular QK tile, then reduce those candidates. This
avoids one program serially scanning the entire centroid dimension and can
avoid materializing/rereading the complete score tensor. Separately, coarse
attention should use a dense AITER-style centroid attention followed by
selected-centroid subtraction, potentially overlapped with route selection,
rather than the current custom serial recomputation. Exact-leaf dispatch is a
secondary target after those two changes.

## Artifacts

- `muse_top3_64k_b8_phase_r1.json`: coarse LOD phase profile.
- `muse_top3_64k_b8_finephase_r1.json`: fine route/coarse/leaf profile.
- `muse_full_64k_b8_global_phase_r1.json`: native global-attention profile.

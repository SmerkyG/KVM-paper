# Shared-local native MTP verification (2026-08-31)

This follow-up tests whether the two native-MTP target positions can share the
512-token LOD local window on `Qwen/Qwen3.8-27B-FP8`, TP1, batch 1. The speed
record uses a real, non-repeating 64K ProLong document, three repeats, and 256
output tokens per repeat. Times are per emitted output token, so they include
the measured acceptance trajectory.

## Design

Both proposal K/V entries are staged before target attention. For each KV head,
one M=16 program contains the two positions times Qwen's six GQA query heads.
The program scans a disjoint 32-entry slice of the local suffix and loads that
K/V tile once for all 12 rows. Position zero and position one use separate
causal masks, so the latter alone can see the second proposal entry.

The local QK/PV is folded into the first shared coarse-route programs. Centroid
top-eight candidates are computed before the local merge and therefore still
depend only on centroid scores. Local output and mass are merged into the
coarse output/LSE that is later combined with exact leaves. This avoids both a
second copy of the suffix and a separate local-attention launch.

## Result

| 64K target path | ms/output | relative to fused shared-local |
|---|---:|---:|
| full-attention native MTP | 22.761 | 1.136x slower |
| prior parallel LOD, shared route | 20.115 | 1.004x slower |
| separate shared-local launch | 20.385 | 1.018x slower |
| **local fused into shared route** | **20.031** | **1.000x** |

The fused path is 0.42% faster than the prior best consolidated LOD panel and
1.74% faster than the separate shared-local experiment. The gain is modest
because the old per-row local loop was partly hidden behind uneven exact-leaf
posting-list work; sharing eliminates traffic but not the leaf tail.

The chat-formatted 64K NIAH-S3 check scored **8/8**. Its speed phase was run
under `torch.profiler` and must not be used as an end-to-end latency result.
Both the shared-route and shared-local device markers were true in the speed
and quality records.

## Delayed kernel profile

The following numbers are average device time per LOD global-layer call. The
three traces cover different numbers of calls, so averages rather than trace
totals are compared.

| kernel component | prior shared route | separate local | route-fused local |
|---|---:|---:|---:|
| shared two-position coarse route | 18.559 us | 18.328 us | 20.223 us |
| standalone shared local | - | 10.217 us | - |
| exact/remainder split scan | 113.312 us | 106.210 us | 110.331 us |
| top-eight/coarse reduction | 45.923 us | 44.634 us | 45.215 us |
| final stable-LSE reduction | 5.585 us | 5.988 us | 5.543 us |
| proposal K/V preparation | 5.103 us | 5.028 us | 5.397 us |
| cache-length advance | 4.870 us | 6.856 us | 5.393 us |
| **listed LOD total** | **193.352 us** | **197.261 us** | **192.102 us** |

The route-fused trace contains no
`_mtp2_gqa_split_decode_local_attention_kernel`. Adding the shared local tiles
costs only 1.664 us in the already-launched route program, versus a 10.217 us
standalone launch. The exact scan no longer evaluates local entries. The final
reducer remains separate; fusing local does not introduce another reduction.

## Balanced exact posting-list splits

The original exact-leaf kernel assigned each of the eight selected centroid
posting lists to one of eight splits. Its latency was consequently determined
by the largest selected centroid. The corrected MTP path stripes every posting
list across all eight splits: split `s` processes leaf tiles `s`, `s+8`, and so
on for every selected route. This preserves the selected leaves and total QK/PV
work while balancing long and short posting lists across programs.

The apparent acceptance regression was isolated by reconstructing the exact
source tree that produced the original 20.031 ms result. Two independent
process launches of that unchanged source produced 81.6% and 87.5% acceptance,
bracketing the original 85.5%. The prompt hash, model snapshot, command,
temperature-zero sampling, node, and GPU were identical. Repeats within each
process were identical. Thus ms/output was being moved by cross-process
numerical changes to the greedy trajectory, not by a source-code regression.

The verifier-cycle normalization below uses `256 - accepted` target cycles.
Unlike ms/output, it remains stable when a different greedy trajectory accepts
a different number of MTP drafts:

| source and exact-list assignment | ms/output | accepted/drafted | ms/verifier cycle |
|---|---:|---:|---:|
| original historical source, one route/split | 20.031 | 118/138 (85.5%) | 37.013 |
| exact historical-source replay A, one route/split | 20.488 | 115/141 (81.6%) | 37.053 |
| exact historical-source replay B, one route/split | 19.766 | 119/136 (87.5%) | 36.790 |
| **historical source + striped leaves** | **19.522** | 118/137 (86.1%) | **36.073** |

The two historical-source replays differ by 0.722 ms/output but only 0.7% per
verifier cycle. The clean striped run is 2.5% faster per verifier cycle than
the original historical result and is also the best observed absolute result.
The failed coarse-verifier/self-speculation experiment and its runtime/cache
changes were removed before applying the isolated striped-leaf patch.

Relative to the historical delayed trace, the new delayed profile reduced
`_split_decode_paged_lod_attention_kernel` from 110.331 us to **53.041 us per
global-layer call**, a 51.9% kernel reduction.
The complete listed LOD pipeline fell from 192.102 us to approximately
138.094 us per layer. The final reducer grew from 5.543 us to 7.245 us because
all eight partials now commonly contain mass, but that is much smaller than the
57.290 us exact-scan saving.

The chat-formatted 64K NIAH-S3 check remained **8/8**. Striping is enabled by
default whenever the verifier has multiple positions. Set
`VLLM_LOD_SPECULATIVE_STRIPE_ROUTE_LEAVES=0` only to reproduce the legacy
route-per-split control.

## Cross-position and GQA exact-leaf sharing

The next experiment shared each exact page load across both native-MTP target
positions and Qwen3.8-27B's GQA6 query heads. This is already the larger,
higher-GQA Qwen model: a full cooperative block covers 2 positions x 6 query
heads for one KV head. Independent causal masks, selected-route masks, online
softmax state, output, and LSE are retained for every row; only the indexed
K/V page load is shared.

Three launch organizations were evaluated:

1. one fixed candidate program with all 12 rows, plus smaller GQA2/GQA3
   subgroups to recover occupancy;
2. an aggregate GQA2 program that deduplicates four rows' route candidates and
   processes several candidates per workgroup;
3. 32- and 16-partial aggregate layouts trading posting-list parallelism for
   fewer workgroups and synchronizations.

The HIP kernels match the independent reference attention numerically (maximum
synthetic output error `4.77e-7`; aggregate error `2.38e-7`). The end-to-end
comparison uses the matched current-source striped control. Because tiny
floating-point regrouping can move the greedy/MTP acceptance trajectory, the
stable comparison is target-cycle time: marginal decode time covers 255 output
intervals and the number of verifier cycles is `256 - accepted`.

| 64K Qwen3.8-27B target path | ms/output | accepted/drafted | ms/verifier cycle | vs. matched striped |
|---|---:|---:|---:|---:|
| **matched striped leaves** | **19.976** | 115/140 | **36.128** | **1.000x** |
| cooperative GQA2 subgroups | 20.403 | 116/140 | 37.162 | 1.029x slower |
| cooperative GQA3 subgroups | 20.599 | 116/140 | 37.520 | 1.039x slower |
| cooperative full GQA6 | 21.107 | 115/140 | 38.172 | 1.057x slower |
| aggregate GQA2, 4 route x 8 page | 20.254 | 122/134 | 38.544 | 1.067x slower |
| aggregate GQA2, 8 route x 4 page | 21.122 | 118/138 | 39.029 | 1.080x slower |
| aggregate GQA2, 8 route x 2 page | 24.075 | 108/147 | 41.480 | 1.148x slower |

The result is negative even in the favorable GQA6 model. Full 12-row sharing
loses occupancy with a 12-wave workgroup. GQA2 restores occupancy and comes
closest, but its route-union checks and shared-memory barriers still cost more
than the saved page loads. Aggregating candidates avoids the explosion in
mostly empty candidate workgroups, but repeats route bookkeeping within page
splits; reducing page splits then exposes long-posting-list tail latency. The
existing striped kernel's many independent workgroups hide that unevenness
better than the cooperative variants.

Consequently, striped leaves remain the default. The cooperative implementation
is retained only as an explicit experiment and requires
`VLLM_LOD_SPECULATIVE_COOPERATIVE_LEAVES=1`; it is not enabled by default. The
opt-in defaults to the least-slow GQA2 non-aggregate organization. Wider
subgroups and aggregate routing remain available only through their diagnostic
environment controls.

## Current 128K full-MTP comparison

A matched rerun with the current striped implementation used the same 128K
ProLong prompt hash, TP1, batch 1, native one-token MTP, three repeats, and 256
output tokens. Full attention used native AITER; the LOD device audit confirmed
that both the shared two-position route and shared/fused local paths executed.

| 128K target | ms/output | ms/verifier cycle | prefill | prefill tok/s |
|---|---:|---:|---:|---:|
| full-attention MTP | 28.736 | 50.887 | 43.892 s | 2,986 |
| **two-tier striped LOD MTP** | **20.118** | **37.174** | **18.242 s** | **7,185** |

LOD is **1.428x faster per emitted output token**, **1.369x faster per target
verifier cycle**, and **2.406x faster in prefill**. Full attention accepted
112 of 143 drafted tokens; LOD accepted 118 of 138, so verifier-cycle
normalization confirms that the win is not merely an acceptance-rate effect.

## Fixed-mask page-size-one MTP

The non-MTP fixed-mask path is now adapted to the two-position native-MTP
target. Both proposal K/V entries are staged before attention. The shared
route scorer handles the two positions together, while retaining independent
causal rows and top-eight routes. The final AITER-style scan launches eight
sequences (two positions times four KV heads), with Qwen's six GQA query heads
inside each M=16 sequence. It reads the persistent page-size-one arena in
place, fast-fails fixed leaf blocks whose centroid is unopened, and performs
sink, causal-local, coarse, and selected exact-leaf attention in that one scan.
There is no compacted or copied K/V list.

The initial adaptation also launched the older shared-local kernel. That work
was dead: the fixed scan had already included local and sink entries and
returned its complete output without consuming the standalone partials. The
launch is now suppressed for this path. For one-request/two-position MTP, the
default fixed-mask dispatch now uses 256 scan segments; the three-repeat result
was 0.97% faster per target cycle than 128 segments. The MTP-specific adaptive
default can be disabled for diagnostic reproduction.

The matched 128K comparison uses the same real, non-repeating ProLong prompt,
TP1, batch 1, one-token native MTP, three repeats, and 256 emitted tokens per
repeat. Because harmless floating-point regrouping changes the greedy path and
draft acceptance between processes, target-cycle time is the authoritative
kernel comparison. The fixed-mask device epoch counts actual verifier cycles
after warmup.

| 128K LOD-MTP target path | ms/output | ms/target cycle | prefill |
|---|---:|---:|---:|
| striped exact leaves | 20.118 | 37.174 | **18.242 s** |
| fixed mask, 128 segments | 20.569 | 36.869 | 18.740 s |
| **fixed mask, 256 segments** | 21.471 | **36.512** | 18.734 s |

The fixed-mask MTP target is **1.8% faster per verifier cycle** than striped
leaves. The matched full-attention cycle is 50.887 ms, so fixed-mask MTP is
**1.394x faster than full attention** and improves the striped LOD speedup from
1.369x to 1.394x.
Its absolute ms/output is worse in this particular process only because it
accepted fewer drafts; it is not a slower target kernel. Prefill is 2.7% slower
than the striped run, reflecting fixed arena/metadata setup rather than decode
attention.

The final chat-formatted 64K NIAH-S3 check scored **8/8** with the automatic
256-segment geometry. Its device audit confirmed shared two-position route
scoring, direct fixed routes, fixed-mask preparation, the AITER-style unified
final scan, and no shared-local launch.

### Why this is a larger win than non-MTP decode

The August 29 non-MTP two-tier result used a different exact-leaf path: the
fixed-list, fixed-mask page-size-one AITER kernel. A same-prompt current-code
rerun reproduces it. The ordinary flat decoder was tested both with and without
the newer striped posting-list assignment:

| 128K non-MTP target | ms/token | speedup vs full |
|---|---:|---:|
| full AITER | 34.254 | - |
| flat LOD, route assigned | 31.456 | 1.089x |
| flat LOD, striped | 30.578 | 1.120x |
| **fixed-mask page-size-one AITER LOD** | **29.409** | **1.165x** |

Thus striping helps the flat one-token decoder by 2.8%, but does not explain
the MTP result and does not beat the specialized non-MTP AITER path. The MTP
gain comes primarily from processing both target positions in one LOD pipeline:
coarse routing and the local suffix are shared/fused, the two positions provide
better occupancy, and fixed launch/reduction costs are amortized. A full MTP
verifier cycle costs 1.486x one full-attention token, whereas an LOD MTP cycle
costs only 1.216x one flat-striped LOD token. The higher LOD draft acceptance
adds to output-token throughput, but cycle-normalized LOD still wins by 1.369x.

## Historical coarse-only acceptance experiment

The diagnostic implementation described in this section was subsequently
removed; its artifacts are retained as a negative experiment.

`VLLM_LOD_SPECULATIVE_COARSE_ACCEPT=acceptance` suppresses exact-leaf opening
for target row zero, the row whose logits accept or reject the one MTP draft.
The fused coarse branch still includes the 512-token local window and sink
tokens. Row one, which supplies the bonus token after an accepted draft,
continues to use normal top-eight LOD. This is an intentionally approximate
diagnostic: on a rejection, vLLM emits the target-row-zero replacement token,
so this mode emits a *coarse* correction rather than conditionally rerunning
one exact LOD step.

The matched 64K ProLong speed prompt had the same SHA-256 in all arms. On this
prompt, coarse row zero reproduced the normal verifier's complete trajectory
(118 accepted of 138 drafted tokens in every repeat), but barely improved
latency because exact row one's uneven posting-list scan still sets the target
forward tail.

| 64K target path | ms/output | change from exact LOD MTP |
|---|---:|---:|
| normal exact LOD MTP | 20.031 | - |
| coarse acceptance row, exact bonus row | 19.997 | 0.17% faster |
| both rows coarse (speed ceiling) | 18.668 | 6.81% faster |

The retrieval result is decisive against using the diagnostic as a verifier:
the matched chat-formatted 64K NIAH-S3 score fell from **8/8 to 3/8**. The
failed answers contain near-miss UUIDs, exactly the detailed-copy regime in
which the coarse replacement is unsafe. Coarse acceptance rates also diverged
substantially from exact LOD on the NIAH requests, whereas they were identical
on the speed document.

The useful conclusion is narrower than “use coarse logits for rejection.” A
production version needs a conditional two-stage verifier:

1. run the cheap coarse target and accept only a sufficiently safe proposal;
2. when it rejects or the gate detects a detail-sensitive token, run one exact
   LOD target step and emit that correction;
3. never emit the coarse distribution's replacement token.

It also needs protection against a coarse false accept: a high coarse margin is
not automatically evidence that the missing exact leaves agree. The all-coarse
ceiling shows at most 6.8% is available on this 64K setup before paying for
conditional launches, cache rollback, and low-acceptance requests. Therefore
coarse acceptance remains opt-in and is not a default path.

## Raw records

- `lod_fused_localroute_b1_64k_r3_d256.json`: authoritative warm speed result.
- `lod_fused_localroute_niah_profile_b1_64k_n8_r1_d64.json`: 64K NIAH-S3
  correctness record and profiled speed phase.
- `lod_coarse_accept_b1_64k_r3_d256.json`: matched coarse-row-zero speed run.
- `lod_coarse_accept_niah_64k_b1_n8.json`: coarse-row-zero NIAH failure.
- `lod_coarse_all_b1_64k_r3_d256.json`: both-rows-coarse speed ceiling.
- `torch_profile_fused_localroute_64k_delay4/`: delayed fused-path trace.
- `lod_shared_local_fused_m16n32_b1_64k_r3_d256.json`: separate-local control.
- `torch_profile_shared_local_64k_delay4/`: delayed separate-local trace.
- `lod_fused_localroute_striped_b1_64k_r3_d256.json`: balanced exact-list speed.
- `lod_fused_localroute_routeassigned_control_b1_64k_r3_d256.json`: matched
  legacy route-per-split control.
- `lod_fused_localroute_historical_repro_b1_64k_r3_d256.json`: exact historical
  source replay A.
- `lod_fused_localroute_historical_warm_b1_64k_r3_d256.json`: exact historical
  source replay B.
- `lod_fused_localroute_striped_clean_b1_64k_r3_d256.json`: historical source
  plus only the striped exact-leaf update.
- `lod_fused_localroute_striped_niah_profile_b1_64k_n8_r1_d64.json`: balanced
  exact-list 8/8 NIAH result and profiled speed phase.
- `lod_striped_control_current_b1_64k_r3_d256.json`: matched current-source
  control for cooperative exact-leaf tests.
- `lod_cooperative_leaves_gqa2_b1_64k_r3_d256.json`: best cooperative subgroup
  result.
- `lod_cooperative_leaves_gqa3_fixed_b1_64k_r3_d256.json` and
  `lod_cooperative_leaves_gqa6_fixed_b1_64k_r3_d256.json`: wider cooperative
  subgroup results after correcting the wide-block shared-load stride.
- `lod_cooperative_leaves_aggregate_b1_64k_r3_d256.json`,
  `lod_cooperative_leaves_aggregate8x4_b1_64k_r3_d256.json`, and
  `lod_cooperative_leaves_aggregate8x2_b1_64k_r3_d256.json`: aggregate route
  layouts.
- `full_current_b1_128k_r3_d256.json` and
  `lod_striped_current_b1_128k_r3_d256.json`: matched current 128K full-MTP and
  striped-LOD-MTP comparison.
- `full_nomtp_current_b1_128k_r3_d256.json`,
  `lod_nomtp_routeassigned_current_b1_128k_r3_d256.json`,
  `lod_nomtp_striped_current_b1_128k_r3_d256.json`, and
  `lod_nomtp_fixedmask_current_b1_128k_r3_d256.json`: matched current non-MTP
  controls isolating striping and the historical fixed-mask AITER path.
- `lod_mtp_fixedmask_nolocal_b1_128k_r3_d256.json`: optimized 128-segment
  fixed-mask MTP control.
- `lod_mtp_fixedmask_seg256_b1_128k_r3_d256.json`: authoritative 256-segment
  fixed-mask MTP speed result.
- `lod_mtp_fixedmask_niah64_b1_n8.json`: chat-formatted 64K NIAH-S3 8/8
  correctness result.
- `lod_mtp_fixedmask_smoke_audit_b1_8k_d64.json`: graph execution audit smoke
  test.
- `lod_mtp_fixedmask_auto_b1_8k_r1_d64.json`: final automatic 256-segment
  dispatch audit, including the device-counted target-cycle metric.
- `lod_mtp_fixedmask_auto_niah64_b1_n8.json`: final automatic 256-segment,
  chat-formatted 64K NIAH-S3 8/8 result and execution audit.
- `torch_profile_fused_localroute_striped_64k_delay4/`: balanced exact-list
  delayed trace.

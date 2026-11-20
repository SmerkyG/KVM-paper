# K2 and Qwen3.8 threshold prefill penalties on MI325X

Matched 64K ProLong prompts, batch one, two-tier prefill with BF16 KV. Each arm has one warmed full-model timing with GPU events and a separate request to count exact work. Every arm uses packed state max and the same 16K initial exact field. K2 c=16 uses 1,024-region streaming partitions, 16K query scratch tiles, and the fused exact merge. Qwen c=16 uses its faster BLAS threshold path with bitmap packing, 4K query scratch tiles and the standalone tensor-core exact reducer. Top-eight means each model's native serving path.

## Full-request attribution

| Model and routing | Prefill | Routing and coarse | Leaf dispatch and preparation | Token-level kernel | Exact reduction, final merge, and remaining remote work |
|---|---:|---:|---:|---:|---:|
| [K2 top-eight](amd-prefill-penalty-v1/amd-k2-penalty-top8.json) | 13.460 s | 2.041 s | 0.120 s | 0.783 s | 0.238 s |
| [K2 c=16](amd-prefill-penalty-v1/amd-k2-penalty-threshold.json) | 16.635 s | 3.475 s | 0.284 s | 1.417 s | 1.168 s |
| [Qwen top-eight](amd-prefill-penalty-v1/amd-qwen-penalty-top8.json) | 7.899 s | 0.357 s | 0.016 s | 0.144 s | 0.042 s |
| [Qwen c=16](amd-prefill-penalty-v1/amd-qwen-penalty-prior-audited.json) | 8.634 s | 0.565 s | 0.029 s | 0.456 s | 0.209 s |

The last column includes the standalone exact reducer where it is present. It also includes work within the outer remote path outside the separately timed routing and leaf events, such as the final merge and native top-eight eligibility filtering. The four timed columns partition remote attention up to event placement and rounding. They exclude the remaining model work and cache updates, which are in the full-prefill column.

| Contribution to the threshold penalty, relative to each model's top-eight prefill | K2 | Qwen | K2 minus Qwen |
|---|---:|---:|---:|
| Routing and coarse | +10.65 pp | +2.64 pp | +8.01 pp |
| Leaf dispatch and preparation | +1.22 pp | +0.16 pp | +1.05 pp |
| Token-level kernel | +4.71 pp | +3.96 pp | +0.75 pp |
| Exact reduction and merge | +6.91 pp | +2.11 pp | +4.80 pp |
| Other time and measurement residual | +0.09 pp | +0.43 pp | −0.34 pp |
| Full prefill | **+23.59%** | **+9.30%** | **+14.29 pp** |

Routing and combining account for **12.82 of the 14.29 percentage points**. K2 processes 201.3 million remote query-head rows during the prompt, versus 18.9 million for Qwen. This is a 10.7× difference in rows across the timed routing and combining stages, with a 1.70× difference in top-eight wall time. The added routing time averaged over a query-head row is 7.1 ns on K2 and 11.0 ns on Qwen. The corresponding increments in reduction and merge are 4.6 and 8.8 ns. These are throughput-normalized costs from full requests.

## Exact token work

The counters use the query counts per region after grouped dispatch, the number of tokens in each region, and the exact kernel's query and key tile sizes. The padded work is the sum of the scheduled query–entry positions including the final partial query and key tiles. The reuse estimate divides useful query–entry pairs by the sum of entry tile sizes visited by each attention workgroup. It represents logical tiled reuse before device cache effects.

| Model | Exact entries per query head, top-eight → c=16 | Padded / useful pairs, top-eight → c=16 | Useful pairs per tiled KV entry, top-eight → c=16 | Token-kernel time per padded pair, c=16 / top-eight |
|---|---:|---:|---:|---:|
| K2 | 734 → 1,272 | 1.265 → 1.224 | 101.2 → 104.6 | 1.079× |
| Qwen | 412 → 1,243 | 1.186 → 1.156 | 27.0 → 27.7 | 1.081× |

K2 increases useful exact work by 1.73× and Qwen by 3.02×. Slot and key tile utilization improves under threshold in both models. The per-padded-pair slowdown is about 8% in both. This constrains how much of K2's relative penalty can be ascribed to extra exact entries, padding or reduced KV reuse.

## Controls

- **Matched exact workload on K2.** The c=24 diagnostic opens 746 entries per query head, close to 734 for native top-eight. The repeated profile takes **0.791 s** in the token kernel and **15.802 s** overall, against **0.783 s** and **13.460 s** for top-eight. It retains the streaming cutoff, closed coarse projection and variable final merge. The first c=24 profile measured 16.787 s, with 4.155 s in routing and coarse versus 3.431 s on repeat. The c=16 repeat measures 16.642 s. This is a speed diagnostic with a different route rule. Retrieval quality was outside its scope. [Initial c=24 and work counters](amd-prefill-penalty-v1/amd-k2-penalty-c24.json) · [c=24 repeat](amd-prefill-penalty-retime-v1/amd-k2-penalty-retime-c24.json) · [c=16 repeat](amd-prefill-penalty-retime-v1/amd-k2-penalty-retime-c16.json)
- **Fixed selections within the leaf stage.** In the middle saved layer at 4,096 queries, feeding the same top-eight regions through the padded variable leaf path changes K2 leaf time from 1.70 to 3.15 ms and Qwen from 0.94 to 1.70 ms. This includes the standalone reduction in the variable path. Native top-eight combines its outputs later, so the replay is a control for the exact entries held fixed. [K2](amd-prefill-penalty-replay-v1/amd-k2-penalty-replay.json) · [Qwen](amd-prefill-penalty-replay-v1/amd-qwen-penalty-replay-native.json)
- **Same threshold materialization and merge in the replay.** With the same BLAS threshold route and fused merge on both models, median remote time across the three saved layers goes from 5.60 to 15.47 ms on K2 and from 3.74 to 9.24 ms on Qwen. Those replays use the same 4K query tile and 3,547 region slots. The latest streaming route brings the K2 c=16 median down to 11.12 ms. The D256 native AITER build rejected the optional Qwen normalized-top-eight diagnostic, so the Qwen denominator uses its native raw query ranking.
- **Port of K2's fused merge to Qwen.** Qwen c=16 with the same fused merge setting takes 8.813 s. Its exact reduction and merge costs 0.408 s versus 0.209 s with Qwen's separate reducer. The comparison above uses the faster Qwen threshold path. [Qwen fused merge](amd-prefill-penalty-v1/amd-qwen-penalty-threshold.json)
- **Page directory optimization trial.** Hoisting the root lookup from the leaf loop preserves every output bit in the replay. The K2 leaf stage changes from 5.88 / 3.44 / 3.25 ms to 6.01 / 3.65 / 3.34 ms on layers 0 / 32 / 63. The experimental source and runs are in the snapshot. [Directory trial](amd-prefill-directory-hoist-v1/amd-k2-directory-hoist-replay.json)

The 4K K2 c=16 routing trace spends about **1.60 ms computing proxy normalization** and **4.20 ms recomputing QK and accumulating the closed summaries**, with about 0.46 ms combined for bitmap compaction and merging coarse partitions. Native top-eight spends about 3.2 ms on its fused route and coarse pass at this shape. The c=24 frozen replay reduces exact entries while keeping routing and coarse at about the same cost. The full c=16 repeat spends 0.79, 1.13 and 1.55 s on routing and coarse across the three remote chunks with 2,048, 2,896 and 3,547 region slots. The c=24 repeat spends 0.78, 1.12 and 1.53 s at the same region counts. [c=16 and c=24 frozen replay](amd-prefill-penalty-c24-replay-v1/amd-k2-penalty-c24-replay.json)

Matching Qwen's relative penalty on K2 would put the 64K prefill at approximately **14.71 s**. That is **1.92 s** below the measured c=16 run. The combined K2 routing and merge increments total **2.36 s**, compared with **0.63 s** for the extra token-level attention.

The [K2 QK reuse follow-up](K2_PREFILL_QK_REUSE.md) reports three-repetition retimes for the single-QK path, compact exact-output addresses, and final merge changes.

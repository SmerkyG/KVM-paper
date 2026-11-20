# K2 Horizon threshold prefill on MI325X

[Earlier AMD ports](H100_PORTS.md) measured 19.99 s for 64K two-tier c=16 prefill on K2 Horizon, versus 13.46 s with native top-eight. This profile splits the extra selection and grouping cost from the cost of reading more exact KV entries.

[Matrix-free follow-up](K2_PREFILL_STREAMING.md) measures streaming routing and coarse attention without dense score and probability buffers.

[K2 and Qwen threshold penalty audit](AMD_PREFILL_THRESHOLD_PENALTY.md) compares the current paths against native top-eight and counts exact work over the full prompt.

## Same-query profile at 64K

The replay holds the BF16 queries, region state and exact KV storage fixed. Each timing is the median over 4,096 queries from each of layers 0, 32 and 63 in the last 16K chunk of one K2 ProLong prompt. The layers have 64 query heads, eight KV heads, head dimension 128 and 3,547 region slots. Remote attention includes grouping, refinement and merge. Model layers, local attention and cache updates are outside the replay scope and inside the full-model timings.

| Route | Routing and coarse | Exact leaves and reduction | Total remote attention |
|---|---:|---:|---:|
| Native top-eight | 3.21 ms | 2.08 ms | 5.60 ms |
| c=16, bitmap packing | 10.84 ms | 5.24 ms | 16.64 ms |
| c=16, aligned GEMMs and compact dispatch | 7.30 ms | 5.67 ms | 13.52 ms |
| c=16, also share metadata across two queries | 6.06 ms | 5.44 ms | 12.22 ms |

[Baseline stage profile](k2-prefill-breakdown-v1/amd-k2-prefill-stage-profile.json) · [Aligned compact replay](k2-prefill-aligned-speed-v1/amd-k2-compact-aligned-replay.json) · [Two-query selector replay](k2-prefill-grouped-v1/amd-k2-grouped-selector.json)

The native AITER kernel streams routing and coarse attention in one fused pass. The threshold path saves one FP32 query-by-region score matrix to determine the per-query threshold and one BF16 closed-region weight matrix for the value projection. Those two matrices total **5.20 GiB** per tile at the profiled shape. The threshold selector took **5.12 ms**, the two GEMMs took **3.65 ms**, and the bitmap-to-expert metadata kernels took **2.33 ms** combined, using medians across the three layers. Device profiler times and independent graph elapsed times are distinct measurements.

The uncapped c=16 rule also opened **1.47–2.33× as many exact remote entries** as top-eight in the three sampled layers. Average entries per query head were 5,020 vs 2,158, 1,440 vs 618, and 2,080 vs 1,416. That work remains after the selector and grouping changes.

## K2 versus Qwen

Both checkpoints have 64 layers. K2 uses LoD in all 64 attention layers. Qwen3.8 uses LoD in its 16 full-attention layers and leaves the other 48 layers on their linear-attention path. K2 also has 64 query heads per full-attention layer versus 24 for Qwen. At the same query count and region count, K2 has **2.67× more query-head rows per LoD layer** and **10.67× more across the model**. K2's head dimension is 128 versus Qwen's 256, so the per-entry dot products are narrower. The row ratios describe the dimensions of routing and its temporary matrices.

Two-tier routing selects regions and reads all exact entries in those regions. The sampled K2 layers selected fewer regions per head than Qwen on average, but K2's early layer selected larger regions. The table gives the mean per query head at early, middle and late LoD layers from each model's frozen 64K ProLong prompt.

| Model | Open regions with c=16 | Exact KV entries with c=16 | Median routing and coarse | Median exact leaves |
|---|---:|---:|---:|---:|
| K2 Horizon | 26.0 / 21.7 / 14.5 | 5,020 / 1,440 / 2,080 | 10.84 ms | 5.24 ms |
| Qwen3.8 | 24.9 / 31.6 / 21.7 | 1,642 / 1,497 / 2,146 | 4.14 ms | 4.41 ms |

The two replays use 4,096 queries, 3,547 available region slots, bitmap grouping and the models' original attention geometry. Values describe the sampled layers and prompt. [K2 replay](k2-prefill-breakdown-v1/amd-k2-prefill-stage-profile.json) · [Qwen replay](h100-port-checks-v1/amd-h100-replay-qwen-bitmap.json)

## Changes tested

- Padding the region dimension to a multiple of 64 cut time in the two GEMMs from about 3.73 to 1.95 ms on this replay. Dummy regions have zero means, zero probability and no effect on eligibility. An alignment of 256 was similar. [Alignment sweep](k2-prefill-sweep-v1/amd-k2-prefill-alignment.json)
- Compact dispatch groups the selected-region ID lists directly with the existing expert worker. It avoids the three passes that build and scan bitmap-derived grouping metadata. [Dispatch replay](k2-prefill-sweep-v1/amd-k2-prefill-dispatch.json)
- A two-query workgroup shares region sizes and log counts and amortizes selector launch work. On the aligned compact path, the selector fell from about 5.71 to 4.37 ms. Four queries per workgroup and 16 warps for one query were slower. The two-query path is opt-in for region threshold and BF16 coarse weights.
- Moving rank work from selection into the bitmap metadata pass saved 0.4 ms across the three sampled stages. The faster compact path skips that metadata pass. Increasing the number of exact workers gave at most 0.35 ms per layer on the sampled geometry. [Rank sweep](k2-prefill-defer-v1/amd-k2-prefill-deferred.json) · [Exact worker sweep](k2-prefill-sweep-v1/amd-k2-prefill-leaves.json)
- Counting selected regions with atomics inside the two-query selector saved work in dispatch and added more time to the selector. Total remote attention grew by about 0.25–0.32 ms across the three layers. [Fused packing replay](k2-prefill-fused-pack-v1/amd-k2-prefill-fused-pack-replay.json)

The unchanged c=16 rule still uses RMS-normalized routing scores, a 1,024-entry region eligibility limit, uncapped selection and closed-region summaries. The initial exact field stays at 16,384 tokens. The 4,096- or 8,192-query setting is the internal scratch tile.

## Full-model prefill

Batch one on MI325X. FP8 checkpoints and BF16 KV. First-token latency from scheduling, measured after one warmup with three repetitions per arm. Timing includes model execution, local and remote attention and cache updates during prefill. Each model uses identical ProLong token IDs for all its runs at each length. The threshold arms keep the same c=16 route, MMA reducer and packed state max. The 8K scratch tile helped K2 at 64K. The 4K tile was faster at 32K. Qwen showed similar timings with either threshold implementation.

| Model | Prefill | 32K | 64K |
|---|---|---:|---:|
| K2 Horizon | [Dense](k2-prefill-grouped-v1/amd-k2-prefill-speed-dense.json) | 6.352 s | 17.292 s |
| K2 Horizon | [Native top-eight](k2-prefill-aligned-speed-v1/amd-k2-prefill-speed-top8.json) | 5.919 s | 13.378 s |
| K2 Horizon | [Previous c=16, bitmap grouping](k2-prefill-aligned-speed-v1/amd-k2-prefill-speed-reference.json) | 7.435 s | 20.096 s |
| K2 Horizon | [c=16, aligned GEMMs](k2-prefill-aligned-speed-v1/amd-k2-prefill-speed-aligned.json) | 7.339 s | 19.526 s |
| K2 Horizon | [c=16, compact dispatch](k2-prefill-aligned-speed-v1/amd-k2-prefill-speed-compact.json) | 7.132 s | 18.779 s |
| K2 Horizon | [c=16, aligned GEMMs and compact dispatch](k2-prefill-aligned-speed-v1/amd-k2-prefill-speed-combined.json) | 6.981 s | 18.146 s |
| K2 Horizon | [c=16, plus two-query selector, 4K tile](k2-prefill-grouped-v1/amd-k2-prefill-speed-rows2.json) | 6.987 s | 17.911 s |
| K2 Horizon | [c=16, plus two-query selector, 8K tile](k2-prefill-grouped-v1/amd-k2-prefill-speed-rows2-tile8192.json) | 7.160 s | 17.505 s |
| Qwen3.8 | [Dense](k2-prefill-grouped-v1/amd-qwen-prefill-speed-dense.json) | 5.286 s | 14.091 s |
| Qwen3.8 | [Native top-eight](k2-prefill-grouped-v1/amd-qwen-prefill-speed-top8.json) | 3.896 s | 8.035 s |
| Qwen3.8 | [Previous c=16, bitmap grouping](k2-prefill-grouped-v1/amd-qwen-prefill-speed-reference.json) | 4.098 s | 8.613 s |
| Qwen3.8 | [c=16, aligned GEMMs, compact dispatch and two-query selector](k2-prefill-grouped-v1/amd-qwen-prefill-speed-rows2.json) | 4.146 s | 8.642 s |

The experimental K2 settings use `LOD_AMD_PREFILL_REGION_ALIGNMENT=64 LOD_AMD_PREFILL_GROUP_PACK=0 LOD_AMD_PREFILL_SELECT_ROWS=2` over the previous c=16 configuration. The last K2 arm adds `LOD_AMD_PREFILL_QUERY_TILE=8192`. Each GPU run uses `uv` through `cluster-run` with a source snapshot and a sibling `-submission.json`.

At 64K K2 takes 3.20 s longer than Qwen with dense attention and 5.34 s longer with native top-eight. Relative to each model's native top-eight control, the fastest configurations in the table add 4.13 s on K2 and 0.58 s on Qwen. The K2 optimization saves 2.59 s over its prior threshold path and finishes 0.21 s behind dense.

## Why threshold adds more full-prefill time on K2

These follow-up 64K, batch-one measurements use one instrumented repetition after warmup. Each model uses identical prompt token IDs across arms. All four runs enable packed state max. K2 uses the [matrix-free c=16 setting](K2_PREFILL_STREAMING.md) with 1,024-region partitions and 16K query scratch tiles. Qwen uses c=16 with bitmap packing, the MMA leaf reducer and 4K query scratch tiles. Native top-eight uses the fused AITER path in both models. The outer remote-path event includes coarse routing, exact dispatch and reduction, and the merge. The two stage times are nested within it.

| Model | Routing | Full prefill | Remote path | Route and coarse | Exact leaves and dispatch |
|---|---|---:|---:|---:|---:|
| K2 Horizon | [Top-eight](prefill-stage-totals-v1/amd-k2-prefill-totals-top8.json) | 13.439 s | 3.190 s | 2.052 s | 0.897 s |
| K2 Horizon | [Threshold](prefill-stage-totals-v1/amd-k2-prefill-totals-threshold.json) | 17.017 s | 6.638 s | 3.485 s | 2.880 s |
| Qwen3.8 | [Top-eight](prefill-stage-totals-v1/amd-qwen-prefill-totals-top8.json) | 7.904 s | 0.558 s | 0.356 s | 0.160 s |
| Qwen3.8 | [Threshold](prefill-stage-totals-v1/amd-qwen-prefill-totals-threshold.json) | 8.609 s | 1.254 s | 0.563 s | 0.646 s |

Switching to threshold takes the remote path to **2.08×** on K2 and **2.25×** on Qwen. That path accounts for **23.7%** of K2 top-eight prefill and **7.1%** of Qwen top-eight prefill. The different baseline shares explain the end-to-end penalties of about **27%** and **9%**, respectively. The added remote time accounts for **3.45 of 3.58 s** on K2 and **0.70 of 0.70 s** on Qwen.

K2's largest increment is **1.98 s in exact refinement**, including grouped dispatch and reduction. Routing and coarse attention add another **1.43 s** as the threshold path computes the global proxy cutoff and projects only closed summaries. The sampled Qwen layers increased exact entries per query head by a larger factor than the sampled K2 layers. The layer counts and head geometry help explain how much work uses the remote path. The measured remote-operation ratio is larger on Qwen.

### Exact refinement breakdown

In two-tier prefill, opening a region replaces its summary with all its original KV entries. The grouped leaf kernel gathers queries that selected the same region, follows the region's page directory to read the individual keys and values, and returns an attention vector and log partition mass for each query–region pair. The final attention combines these exact contributions with the unopened summaries and the local and sink branches.

The same 64K K2 request was profiled with finer GPU event boundaries. Times are one warmed batch-one repetition. Top-eight combines its eight region outputs inside the final merge. Threshold first reduces its variable number of region outputs to one exact branch, then runs the final merge. The dispatch column includes route grouping and scratch preparation. The last column includes the final merge and other work outside the separately timed routing and leaf stages.

| Routing | Group and prepare | Token-level attention kernel | Separate exact reduction | Final merge and remaining remote work |
|---|---:|---:|---:|---:|
| [Top-eight](k2-prefill-leaf-totals-v1/amd-k2-prefill-leaves-top8.json) | 0.119 s | 0.782 s | Fused into final merge | 0.241 s |
| [Threshold, c=16](k2-prefill-leaf-totals-v1/amd-k2-prefill-leaves-threshold.json) | 0.289 s | 1.433 s | 1.150 s | 0.273 s |

The variable reduction reads a value vector and log mass for every selected region of each query. The small-frontier path runs a tensor-core reduction for up to 32 regions, and larger frontiers use the uncapped fallback. The **1.15 s** for the separate reduction is larger than the **0.65 s** increase in token-level attention.

The [fixed-frontier replay](k2-prefill-leaf-fixed-v1/amd-k2-exact-fixed-frontier.json) holds the selected regions and original KV fixed across the first three columns. All rows use 4,096 saved queries per layer. The variable columns use the threshold leaf worker and its standalone reduction. The threshold-capacity column uses its actual 223-slot row stride. The last column switches to the full c=16 selections.

| Layer | Native top-eight leaf path | Same routes, variable leaf path at width 8 | Same routes, variable leaf path at width 223 | Actual threshold leaf path |
|---:|---:|---:|---:|---:|
| 0 | 2.43 ms | 3.78 ms | 4.08 ms | 8.03 ms |
| 32 | 1.63 ms | 2.75 ms | 3.13 ms | 5.42 ms |
| 63 | 2.07 ms | 3.28 ms | 3.58 ms | 4.98 ms |

The fixed routes demonstrate that some of the overhead comes from the variable implementation even when it visits exactly the same KV entries. This replay excludes the final merge, so use the full-prefill profile above to compare all combining work.

### Fusing the exact reduction with the final merge

The optional fused path passes each selected region's attention vector and log mass directly into the final merge. It preserves the exact branch's BF16 rounding in registers, then combines that branch with the closed summaries, local attention and sink. The leaf worker and its selected KV entries use the same routing rule. Fusion removes the intermediate exact branch buffer and the standalone reducer launches.

Ordinary vector reduction inside the fused kernel was slower in the [middle-layer replay](k2-prefill-fused-merge-v3/amd-k2-fused-merge-replay-2-32-4.json). The faster version uses tensor cores for four query rows per workgroup with up to 64 selected regions per query. The 32-region fused version frequently entered its fallback, with 17.4% of middle-layer rows above 32 regions. Only one of 262,144 middle-layer rows exceeded 64. Larger frontiers continue through an uncapped tiled FP32 reduction. The faster tensor-core tile uses two BF16 components for the FP32 softmax weights. The token-level leaf kernel and the route selection are identical in the frozen replay. [Tensor-core tuning](k2-prefill-fused-merge-v4/amd-k2-merge-mma-4-64-4-2.json)

| Saved K2 layer, 4,096 queries | Leaf stage plus merge before | Leaf stage plus merge fused | Total remote attention before → fused |
|---:|---:|---:|---:|
| 0 | 8.50 ms | 7.45 ms | 14.53 → 13.58 ms |
| 32 | 5.78 ms | 4.93 ms | 11.86 → 11.12 ms |
| 63 | 5.30 ms | 4.72 ms | 11.36 → 10.91 ms |

The stage sums come from independent graph timings on one saved 64K K2 prompt. Every frozen comparison uses the same queries, region selections, and exact KV storage. The fused outputs differ from the separate reducer by up to 0.03125 in BF16. The fused path passed [14 independent PyTorch frontier checks](k2-prefill-fused-merge-v4/amd-k2-mma64-merge-frontier.json), including empty and large frontiers. [Three-layer replay](k2-prefill-fused-merge-v4/amd-k2-mma64-all-layers.json)

Full-model timings are first-token latency at batch one on MI325X, three repetitions after warmup. Both runs use the same ProLong prompts, 16K initial exact field, 16K scratch tiles, streaming region c=16 with 1,024-region partitions, and packed state max.

| Threshold prefill | 32K | 64K |
|---|---:|---:|
| [Separate reducer](k2-prefill-fused-merge-v3/amd-k2-fused-merge-speed-reference.json) | 6.696 s | 16.942 s |
| [Fused merge](k2-prefill-fused-merge-v4/amd-k2-fused-merge-speed-64.json) | 6.639 s | 16.641 s |

The [instrumented fused run](k2-prefill-fused-merge-v4/amd-k2-fused-merge-totals-64.json) records no separate reduction. The combined reduction, final merge and remaining remote work falls from **1.42 s to 1.17 s** against the standalone-reduction profile above. The fused setting matches all eight previous prediction texts and scores on the frozen multikey subset. [Multikey-2](k2-prefill-fused-merge-v4/amd-k2-fused-merge-ruler-niah_multikey_2.json) · [Multikey-3](k2-prefill-fused-merge-v4/amd-k2-fused-merge-ruler-niah_multikey_3.json)

Enable the tested tile with `LOD_AMD_PREFILL_FUSED_MERGE=1`. The opt-in default geometry is four queries, 64 regions and four warps with two BF16 weight components. [Opt-in default replay](k2-prefill-fused-merge-final/amd-k2-merge-final-replay.json)

### Accumulating exact attention without per-region outputs

Three prototypes use the same open regions and BF16 KV. A query-major kernel keeps one running softmax through all its open regions. The adjacent-query variant shares each region's KV loads across up to 16 queries and masks each query's unused regions. The grouped atomic variant keeps the existing region grouping. Each leaf workgroup rescales its region contribution to a common per-query reference and atomically adds its numerator and mass to FP32 per-query accumulators. When a region's log mass differs from the reference by more than 32, the kernel marks that query for a stable online recomputation. All of these variants emit one exact attention branch per query.

On the same middle K2 layer and 4,096-query slice, the grouped kernel with fused merge remains fastest. The exact-stage column excludes the final merge, and the total remote time includes it. Exact-output workspace is reserved memory for leaf outputs, their log masses, and direct accumulators. It excludes the routing, KV, and dispatch buffers.

| Exact path | Exact stage | Total remote | Exact-output workspace |
|---|---:|---:|---:|
| Grouped, fused final merge | 3.40 ms | 11.16 ms | 14,495 MiB |
| [Grouped FP32 atomic accumulation](k2-prefill-exact-accumulate-v1/amd-k2-exact-atomic-replay.json) | 9.37 ms | 15.76 ms | 195 MiB |
| [Adjacent-query groups of 16](k2-prefill-grouped-query-v1/amd-k2-query-group-16-32-4.json) | 64.53 ms | 71.86 ms | 65 MiB |
| [One query per workgroup](k2-prefill-exact-accumulate-v1/amd-k2-exact-query-replay.json) | 112.51 ms | 118.84 ms | 65 MiB |

The atomic variant completed the replay with zero repair rows. Splitting each query's accumulator into [two](k2-prefill-exact-shards-v1/amd-k2-exact-shards-2.json), [four](k2-prefill-exact-shards-v1/amd-k2-exact-shards-4.json), or [eight shards](k2-prefill-exact-shards-v1/amd-k2-exact-shards-8.json) took 9.51, 9.60, and 9.91 ms in the exact stage. The query-owned kernels give up sharing regions across arbitrary queries. The grouped direct-accumulation kernel retains that sharing and adds read-modify-write traffic for each region output channel.

An instrumented replay with the same 608 workers measured **3.03 ms** in the grouped leaf kernel and **8.80 ms** with atomics. Normalization and the empty repair check together took **0.07 ms**. With 4,864 workers, direct accumulation improved only to 9.28 ms for the exact stage and 15.69 ms for the full remote operation. The corresponding grouped fused-merge control took 11.02 ms for the full remote operation. [Kernel profile](k2-prefill-exact-accumulate-v2/amd-k2-atomic-workers-608.json) · [Larger worker grid](k2-prefill-exact-accumulate-v2/amd-k2-atomic-workers-4864.json)

The [single-query](k2-prefill-exact-accumulate-v1/amd-k2-exact-check-query.json), [adjacent-query](k2-prefill-grouped-query-v1/amd-k2-query-group-frontier.json), and [atomic](k2-prefill-exact-accumulate-v1/amd-k2-exact-check-atomic.json) paths each passed 14 independent PyTorch checks, including empty regions and frontiers larger than 64. The [forced-repair test](k2-prefill-exact-accumulate-v1/amd-k2-exact-check-atomic-repair.json) passed all 14 checks with a shifted reference. These are replay and numerical checks on benchmark prototypes.

### Grouping queries by their opened regions

The audit allows queries from all GQA heads that share one physical KV head to group together. It computes a signature from each query's full selected set and checks the actual region IDs inside matched signatures. No collisions occurred in the audited slices. Percentages are weighted by the number of exact query–KV-entry pairs, and the group sizes count queries that open the entire same set.

| K2 layer, 16K queries of a 64K prompt | Work in matching groups of at least 2 | Work in matching groups of at least 16 | Ideal KV reuse in identical-set tiles of up to 16 |
|---:|---:|---:|---:|
| 0 | 55.77% | 32.57% | 1.89× |
| 32 | 1.71% | 0.47% | 1.01× |
| 63 | 19.87% | 8.22% | 1.19× |

Ideal reuse divides the sum of per-query exact entries by the sum of entries visited once per group tile, before hardware page padding or cache hits. With mixed-set tiles of 16, grouping by the largest two selected regions raises the logical reuse from **5.68× to 9.66×** in layer 0, **4.25× to 5.28×** in layer 32, and **7.33× to 8.07×** in layer 63. [16K audit](k2-prefill-frontier-order-v1/amd-k2-frontier-audit-16384.json) · [4K audit](k2-prefill-frontier-order-v1/amd-k2-frontier-audit-4096.json)

The timing prototype orders queries within each KV head and packs them into 16-query workgroups. Sorting by full signature places identical sets together and permits a tile to span multiple sets. Sorting by the two largest regions prioritizes moving fewer KV entries. Each workgroup walks the union of its selected regions, masks attention for queries that excluded a region, and keeps all of each query's exact accumulation in registers. Timing includes ordering, routing, the leaf kernel, and the final merge.

| K2 layer, 4,096 queries | Group by region, fused merge | Adjacent queries | Whole-set signature order | Two-largest-region order |
|---:|---:|---:|---:|---:|
| [0](k2-prefill-frontier-order-v2/amd-k2-frontier-ordered-layer0.json) | 13.52 ms | 129.62 ms | 118.98 ms | 79.22 ms |
| [32](k2-prefill-frontier-order-v2/amd-k2-frontier-ordered-replay.json) | 11.09 ms | 71.36 ms | 115.96 ms | 60.20 ms |
| [63](k2-prefill-frontier-order-v2/amd-k2-frontier-ordered-layer63.json) | 10.91 ms | 50.58 ms | 76.00 ms | 48.88 ms |

With 32 queries per tile, the two-largest-region order reaches **57.59 ms** in the [middle-layer replay](k2-prefill-frontier-order-v2/amd-k2-frontier-largest-32-queries.json). Key generation and sorting took **0.29–0.30 ms** per layer at the measured 4K shape. Working on a union still costs tensor-core work for masked queries. In the middle layer with 16-query tiles, the largest-region order reserves about three query–entry slots in the MMA for every useful one before accounting for page alignment. The region-grouped kernel can group all queries that selected each individual region, regardless of their other selections. [Ordering overhead](k2-prefill-frontier-order-v3/amd-k2-frontier-order-overhead.json)

Both ordered paths passed 14 independent PyTorch frontier checks and the three frozen layer checks. [Signature order](k2-prefill-frontier-order-v2/amd-k2-frontier-signature-check.json) · [Largest-region order](k2-prefill-frontier-order-v2/amd-k2-frontier-largest-check.json)

## Numerical and retrieval checks

The reference and new K2 paths use the same frozen 64K RULER examples and c=16 threshold decode. Both 4K-tile paths scored 100% on eight examples each of multikey-2 and multikey-3. The predictions matched for all 16 examples. The 8K tile was checked on the first four examples of each task and matched all eight reference predictions and scores. These checks cover two retrieval tasks. [Reference](k2-prefill-grouped-v1/amd-k2-prefill-ruler-reference.json) · [4K tile](k2-prefill-grouped-v1/amd-k2-prefill-ruler-rows2.json) · [8K tile multikey-2](k2-prefill-grouped-v1/amd-k2-prefill-ruler-tile8192-niah_multikey_2.json) · [8K tile multikey-3](k2-prefill-grouped-v1/amd-k2-prefill-ruler-tile8192-niah_multikey_3.json)

Alignment preserved all sampled routing selections on the real K2 snapshots. The maximum remote-attention difference from the unaligned BF16 path was 0.015625. Compact dispatch and the grouped selector each gave bitwise-identical outputs on their respective same-alignment replay controls. The final source passed 14 independent PyTorch comparisons per tested setting, including empty regions, odd query counts, large frontiers and graph replay. [Grouped region](k2-prefill-validated-v1/amd-k2-prefill-final-grouped-frontier.json) · [Bitmap region](k2-prefill-validated-v1/amd-k2-prefill-final-bitmap-frontier.json) · [Bitmap per-entry](k2-prefill-validated-v1/amd-k2-prefill-final-entry-frontier.json)

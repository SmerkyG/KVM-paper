# K2 Horizon 64K centroid routing and exact dispatch

Batch one on MI325X, K2 Horizon 32B FP8 with BF16 KV, two-tier LoD, 65,536 identical ProLong tokens, and three measured repetitions after warmup. Full-threshold runs use normalized region mass with c=16 in all three remote chunks. They keep the initial 16K exact prefill, the region clustering schedule, and the same exact KV work for the same selected regions. All rows use the original model projection kernels.

| Prefill path | Median |
|---|---:|
| [Native top-eight](k2-prefill-fast-centroid-v1/amd-k2-fast-centroid-top8.json) | 13.66 s |
| [Previous c=16 wave selector](k2-prefill-fast-centroid-v1/amd-k2-fast-centroid-control.json) | 15.92 s |
| [c=16 with hipBLAS for the centroid QK only](k2-prefill-fast-centroid-v1/amd-k2-fast-centroid-qk-only.json) | 15.66 s |
| [Plus packed log-count and eligibility metadata](k2-prefill-fast-centroid-v1/amd-k2-fast-centroid-combined.json) | 15.59 s |
| [Plus 16-bit slot IDs](k2-prefill-slot-and-tile-v1/amd-k2-narrow-slot-speed.json) | 15.53 s |
| [Plus live-prefix dispatch](k2-prefill-live-prefix-v1/amd-k2-live-prefix-speed.json) | 15.56 s |

The first four rows ran on distinct GPUs on node 2 in the same set of launches. The two slot and dispatch runs followed on node 2. A previous node-2 run recorded 13.44 s for native top-eight and 15.87 s for the wave selector. The small difference between the last two rows falls within the observed variation across runs. [Previous matched runs](K2_PREFILL_64K_PROFILE.md)

The preferred BLAS library is set around the FP32 centroid QK operation and restored before closed-summary projection or model execution continues. With 16K queries at 3,547 regions, that operation fell from 5.71 to 4.13 ms in an isolated replay, both with the same 3,584-column score stride. At 2,048 regions it fell from 4.67 to 2.44 ms. The 2,896-region shape at stride 2,944 fell from 4.84 to 3.84 ms. [2,048 regions](k2-qk-blas-tune-v1/amd-k2-qk-blas-s2048.json) · [2,896 regions](k2-qk-blas-tune-v1/amd-k2-qk-blas-s2896.json) · [3,547 regions](k2-qk-blas-tune-v1/amd-k2-qk-blas-s3547.json)

Packing metadata stores the FP32 log-count magnitude and encodes oversize and empty states in that same load. The selector retains its FP32 cutoff and the same 1,024-entry eligibility limit. On a 16K query replay at 3,547 regions the selector fell from 10.40 to 9.39 ms. Narrow slot IDs save about 1.31 GiB of padded slot storage in the last 16K chunk compared with 64-bit IDs. The live-prefix dispatch visits the valid IDs and still groups queries by region for the exact KV kernel. Its grouped dispatch measured 0.20 s in the full instrumented run versus 0.30 s in the previous wave-selector profile. The exact-leaf computation itself was 1.32 s in that new profile. [Metadata control](k2-prefill-log-size-v1/amd-k2-log-size-final-control.json) · [Packed metadata](k2-prefill-log-size-v1/amd-k2-log-size-final-candidate.json) · [Dispatch profile](k2-prefill-live-prefix-v1/amd-k2-live-prefix-profile.json)

The centroid changes matched prediction text and scores on all 16 archived 64K multikey prompts, including the multikey-2 prompt that native top-eight prefill with c=16 decode previously missed. The original 16K exact field and c=16 decode were used in all quality rows. The narrow-slot and live-prefix kernels both passed 14 independent PyTorch frontier checks. [Multikey-2](k2-prefill-fast-centroid-v1/amd-k2-fast-centroid-ruler-mk2.json) · [Multikey-3](k2-prefill-fast-centroid-v1/amd-k2-fast-centroid-ruler-mk3.json) · [Narrow slots frontier](k2-prefill-slot-and-tile-v1/amd-k2-narrow-slot-check.json) · [Live prefix frontier](k2-prefill-live-prefix-v1/amd-k2-live-prefix-frontier.json)

## Direct fusion trials

One AMD MFMA workgroup computed the whole centroid score row, evaluated the c=16 cutoff in FP32, and wrote the closed probabilities and compact selection bits. Routing scratch fell from 6,014 to 2,430 MiB in the 4,096-query middle-layer replay. Routing and coarse attention took 10.78 ms against 4.76 ms for the separate GEMM and wave selector. The fused producer alone took about 9.88 ms. Other tested layouts exceeded shared memory or failed to compile. [Fused selector replay](k2-prefill-fused-selector-v1/amd-k2-fused-selector-16-8-16-1.json)

The partition prototype computed QK once and used each partition's proxy mass as a lower bound on the whole row. The resulting candidate pruning lost zero selected regions across 98,304 sampled query-head rows from three layers. With 1,024-region partitions, however, the mean candidate list still held 131 to 171 regions per row against only 14 to 26 final opens. The prototype projected each partition, subtracted the opened contributions in FP32 and directly rebuilt unstable rows. The fastest compiled setting took 41.78 ms in the same 4,096-query replay, with 34.18 ms in the producer and 6.86 ms in the closed-summary correction. [Candidate audit](k2-partition-audit-v1/amd-k2-partition-audit.json) · [Partition replay](k2-prefill-partition-v2/amd-k2-partition-128-256-8.json)

The remaining cost includes the exact KV work. The matched frontier audit found about 1,272 opened entries per query head at c=16 versus 734 at top-eight. Grouping preserves KV reuse across queries but leaves the per-region value reductions and final merge. [Opened-entry audit](AMD_PREFILL_THRESHOLD_PENALTY.md)

## Bitwise comparison

For the result-preserving path, use the original QK backend with packed log-count metadata, narrow slot IDs and live-prefix dispatch. On three saved 16K query chunks covering **3,145,728 query-head rows**, this combination preserved every route ID, coarse value, coarse log partition, and final remote output bit. The warmed full 64K prefill median was **15.64 s**. [Layer 0](k2-prefill-exactness-v1/amd-k2-exactness-layer0.json) · [Layer 32](k2-prefill-exactness-v1/amd-k2-exactness-layer32.json) · [Layer 63](k2-prefill-exactness-v1/amd-k2-exactness-layer63.json) · [Result-preserving speed](k2-prefill-exactness-v1/amd-k2-stock-blas-storage-speed.json)

The hipBLAS experiment changed 15 routes across those same rows, despite matching the 16 multikey predictions. Its numbers above are experimental results. The current implementation uses the original QK backend. The shorter merge tile also failed bitwise comparison because the reduction tree changed. Both experimental changes were removed from the active path.

The score workspace can also hold the BF16 probabilities after each selector workgroup finishes reading its own FP32 rows. This removes **7.00 GiB** of routing scratch from the 16K, 3,547-region replay. The grouped bit compactor fills each row's unused tail while packing its live ranks. The two changes independently and together preserved every route ID, closed-summary output, closed log partition, and final remote output bit on all three saved layers. [Layer 0](k2-prefill-storage-reuse-v1/amd-k2-storage-exactness-layer0.json) · [Layer 32](k2-prefill-storage-reuse-v1/amd-k2-storage-exactness-layer32.json) · [Layer 63](k2-prefill-storage-reuse-v1/amd-k2-storage-exactness-layer63.json) · [Scratch replay](k2-prefill-storage-reuse-v1/amd-k2-storage-reused.json)

| Result-preserving configuration | 64K prefill median |
|---|---:|
| [Packed metadata, narrow slot IDs, live-prefix dispatch, original QK](k2-prefill-storage-reuse-v1/amd-k2-bitwise-speed-control.json) | 15.67 s |
| [Plus fused padding and compaction](k2-prefill-storage-reuse-v1/amd-k2-bitwise-speed-fused-pad.json) | **15.60 s** |
| [Score scratch reuse with the original compaction](k2-prefill-storage-reuse-v1/amd-k2-bitwise-speed-reuse.json) | 15.67 s |

All three final speed arms ran on node 2, used the same 65,536 prompt tokens, and produced the same greedy output tokens. Each used three warmed repetitions and the same c=16 routing throughout remote prefill. The 15.60 s configuration is 0.90 s above the original 14.70 s target. The memory-saving configuration has similar speed to the control. The combined scratch and compaction changes passed the independent 14-case frontier check. [Frontier check](k2-prefill-storage-reuse-v1/amd-k2-bitwise-storage-frontier.json) · [Fastest exact launch](k2-prefill-storage-reuse-v1/amd-k2-bitwise-speed-fused-pad-submission.json)

The [FP32 centroid QK follow-up](K2_PREFILL_FP32_QK.md) measures the tiled GEMM, aligned selector, exact-region KV gather, and the combined 64K result.

## Running-prefix candidate audit

The H100 producer uses the mass of all tiles visited so far as a lower bound on the final routing partition. The earlier AMD prototype used each tile's own mass. On the same three saved K2 layers and 98,304 sampled query-head rows, the running-prefix filter with 1,024-region tiles retained a mean of **48–68 candidates** out of 3,547 regions, versus **131–171** for the same-size independent partitions. All finally opened regions survived. An average of 25–33 of 111 bitmap words held at least one candidate. This audit measures candidates and bitmap occupancy. The producer still scores all centroids, and unopened centroids contribute to coarse attention. [Running-prefix audit](k2-prefill-prefix-audit-v1/amd-k2-h100-prefix-audit.json)

The [fused prefix-filter follow-up](K2_PREFILL_FUSED_PREFIX.md) measures the running-prefix and parallel-tile ports at 64K against the current c=16 path.

# Fused running-prefix filter for K2 Horizon prefill

K2 Horizon 32B FP8, BF16 KV, batch one, on MI325X. Full-model runs use the same 65,536-token ProLong prompt, two-tier LoD, c=16 for every remote prefill chunk, and the original 16,384-token exact prefix. Timings are medians of three warmed requests and include cache construction. Model projections and the exact KV path use the existing kernels.

The filter uses the running proxy partition mass as a lower bound on the final mass. A region whose proxy weight falls below the corresponding prefix cutoff cannot pass the final cutoff. Each version retains a floating-point margin near the cutoff and checks the remaining candidates against the final strict cutoff. The final exact KV frontier and closed summaries use the original region weights.

## 64K prefill

| Path | Median prefill |
|---|---:|
| [Current c=16 control on the parallel port’s GPU](k2-prefill-prefix-parallel-v2/amd-k2-prefix-64k-gpu4-control.json) | **15.21 s** |
| [Current c=16 control on the second GPU](k2-prefill-prefix-fused-v1/amd-k2-prefix-64k-control.json) | 15.22 s |
| [Prefix metadata produced by parallel centroid tiles](k2-prefill-prefix-parallel-v2/amd-k2-prefix-parallel-64k.json) | 15.53 s |
| [Running prefix carried by the centroid producer](k2-prefill-prefix-fused-v1/amd-k2-prefix-64k-online.json) | 15.68 s |
| [Native top-eight, earlier baseline](k2-prefill-fast-centroid-v1/amd-k2-fast-centroid-top8.json) | 13.66 s |

The parallel path added 0.32 s (2.1%) on the same GPU. All c=16 runs produced the same greedy output tokens. The parallel tiled path computes each tile’s maximum and proxy mass in the centroid GEMM epilogue. The existing selector combines those statistics and applies the prefix filter as it reads the original scores for coarse attention. The running-prefix path instead emits candidate bits during the centroid GEMM. Both paths compute QK once and have no additional pass over the centroid scores. [Same-GPU control launch](k2-prefill-prefix-parallel-v2/amd-k2-prefix-64k-gpu4-control-submission.json) · [Parallel launch](k2-prefill-prefix-parallel-v2/amd-k2-prefix-parallel-64k-submission.json) · [Running-prefix launch](k2-prefill-prefix-fused-v1/amd-k2-prefix-64k-online-submission.json)

## 128K prefill

Fresh controls for all paths on separate MI325X GPUs on node 2. All runs use the same 131,072-token ProLong prompt, batch one, K2 Horizon FP8, and BF16 KV. The LoD arms use the original 16,384-token exact prefix, and both threshold arms use c=16 for all seven remote chunks with 16,384-query tiles. Timing runs from scheduling to the first token, including cache construction. Each median uses three measured requests after one warmup.

| Path | Median prefill | Measured range |
|---|---:|---:|
| [Dense](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-dense.json) | 52.79 s | 52.74–52.82 s |
| [Two-tier top-8](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-top8.json) | 30.14 s | 30.13–30.15 s |
| [Two-tier c=16](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-c16.json) | 36.85 s | 36.83–36.86 s |
| [Two-tier c=16 with the fused prefix filter](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-c16-prefix.json) | 38.15 s | 38.11–38.17 s |

Unfiltered threshold took 30.2% less time than dense and 22.3% more time than top-8. The prefix filter added 1.30 s (3.5%) on this workload. All four runs had the same prompt and generated the same two greedy tokens on each repetition. [Dense launch](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-dense-submission.json) · [Two-tier top-8 launch](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-top8-submission.json) · [Two-tier c=16 launch](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-c16-submission.json) · [Two-tier c=16 with the fused prefix filter launch](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-c16-prefix-submission.json)

## Routing and coarse attention replay

Graph replay at 16,384 saved queries from layer 32. The 2,048-region row is a slice of that same snapshot.

| Regions | Current | Parallel tiles | Running prefix |
|---:|---:|---:|---:|
| [2,048](k2-prefill-prefix-parallel-v2/amd-k2-prefix-parallel-2048.json) | 9.38 ms | 10.60 ms | [10.82 ms](k2-prefill-prefix-fused-v1/amd-k2-prefix-replay-2048.json) |
| [3,547](k2-prefill-prefix-parallel-v2/amd-k2-prefix-parallel-3547.json) | 16.37 ms | 18.89 ms | [19.87 ms](k2-prefill-prefix-fused-v1/amd-k2-prefix-replay-3547.json) |

The same-GPU device profile at 3,547 regions measured 4.63 → 5.73 ms in the centroid producer and 9.51 → 10.46 ms in the selector with parallel tiles. The producer computes and stores tile statistics. The selector combines them while still reading the full score row to form the closed-summary softmax. The running-prefix producer serializes centroid tiles within each query workgroup and took 6.93 ms. The alternative packed-key MFMA schedule reached 24.15 ms in routing and coarse attention, and the best running-prefix log-domain tuning reached 20.34 ms. [Device profile](k2-prefill-prefix-mfma-v1/amd-k2-prefix-stage-profile.json) · [Packed-key MFMA replay](k2-prefill-prefix-mfma-v1/amd-k2-prefix-mfma-3547.json) · [Log-domain sweep](k2-prefill-prefix-tuning-v1/amd-k2-prefix-log-3547.json)

## Exactness

Across 3,145,728 query-head rows in saved 16K query chunks at layers 0, 32 and 63, the running-prefix port matched the reference bit for bit on opened region IDs, coarse outputs, coarse log partitions and final remote outputs. [Layer 0](k2-prefill-prefix-parallel-v1/amd-k2-prefix-online-final-bitwise-00.json) · [Layer 32](k2-prefill-prefix-parallel-v1/amd-k2-prefix-online-final-bitwise-32.json) · [Layer 63](k2-prefill-prefix-parallel-v1/amd-k2-prefix-online-final-bitwise-63.json)

The parallel-tile port also matched all four fields on the same three layer snapshots. [Layer 0](k2-prefill-prefix-parallel-v2/amd-k2-prefix-parallel-bitwise00.json) · [Layer 32](k2-prefill-prefix-parallel-v2/amd-k2-prefix-parallel-bitwise32.json) · [Layer 63](k2-prefill-prefix-parallel-v2/amd-k2-prefix-parallel-bitwise63.json)

The serving path uses the faster unfiltered kernel. The [archived kernel and replay source](k2-prefill-prefix-mfma-v1/source/) contains the tested variants.

The subsequent 128K attribution attempt completed for [top-eight](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-profile-top8.json). Threshold profiling failed during warmup when an 85 GiB exact-output allocation encountered fragmentation. Its expandable-allocator retry was cancelled after switching further investigation to saved 64K queries. The 128K timing table above uses the separately completed speed runs. [Failed profile launch](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-profile-c16-submission.json) · [Cancelled retry](k2-prefill-prefix-parallel-v2/amd-k2-prefill-128k-profile-c16-expandable-submission.json)

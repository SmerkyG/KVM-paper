# AITER-stage trials for K2 Horizon threshold prefill

The speed runs use one MI325X, batch one, K2 Horizon 32B FP8 with BF16 KV, and the same 65,536-token ProLong prompt. The threshold setting is c=16 in every remote block and the first 16,384 tokens use exact attention. The runs include cache construction and use the original model projection kernels.

## Direct AITER exact-leaf replay

The layer 32 replay freezes 4,096 real queries per head and their original routes. AITER receives each opened region as a variable-length sequence with its existing leaf keys in order. The frozen replay builds its bucket metadata ahead of timing. The full AITER column includes the shared dispatch, KV and query packing, and scattering the values and log partitions into LoD's merge layout. A serving implementation would also need to build its bucket metadata on device.

| Exact-leaf implementation | Attention kernel alone | With dispatch and data movement | With final merge |
|---|---:|---:|---:|
| [Current grouped LoD, CK run's control](k2-prefill-aiter-ck-v1/amd-k2-ck-leaves-layer32.json) | | 2.70 ms | 4.12 ms |
| [AITER CK, buckets by region query length](k2-prefill-aiter-ck-v1/amd-k2-ck-leaves-layer32.json) | 2.13 ms | 4.40 ms | 5.84 ms |
| [AITER CK, split into 128-query groups](k2-prefill-aiter-ck-v1/amd-k2-ck-leaves-layer32.json) | 1.83 ms | 6.98 ms | 8.32 ms |
| [AITER ASM, buckets by region query length](k2-prefill-aiter-leaves-v1/amd-k2-aiter-leaves-layer32.json) | 15.03 ms | 17.57 ms | 18.97 ms |

The CK backend changed BF16 outputs on the same frontier. The largest absolute difference in merged BF16 output was 0.09375. The tests favor reusing LoD's direct read and write paths. An AITER kernel port would need to read the query indices inside the kernel and write into the compact output layout directly to avoid those copies.

## Selector and merge

- A standalone HIP selector with vectorized score loads and wave-local mask packing took **8.05 ms** against **8.68 ms** for Gluon at 16K queries and 3,547 regions. It changed 3 of 1,048,576 selected-region rows, so the Gluon selector remains the basis of the full-model comparison. [HIP selector replay](k2-prefill-hip-selector-v1/amd-k2-hip-wave-selector.json)
- Moving query RMS normalization ahead of the Gluon selector and using the padded metadata's invalid sentinel reduced full routing and coarse time from **16.23 to 15.97 ms** on the saved 16K middle-layer queries. All selected IDs, coarse values, and coarse log partitions matched bit for bit. [RMS and mask replay](k2-prefill-selector-rms-v1/amd-k2-selector-rms-layer32.json)
- The wave ballot preserved output bits, with **17.31 ms** routing and coarse time versus **16.20 ms** for the current integer bit packs. Fetching grouped query tiles on each chiplet and changing probability stores gave little benefit in the two replays. [Ballot replay](k2-prefill-stage-schedule-v1/amd-k2-ballot-selector-test.json) · [Layer 0 scheduling](k2-prefill-selector-chiplet-v1/amd-k2-selector-chart-layer0.json) · [Layer 32 scheduling](k2-prefill-selector-chiplet-v1/amd-k2-selector-chart-layer32.json)
- A 16-rank merge shortcut retained the full-width FP32 mass reduction and preserved the merged output. It took **1.36 ms** against **1.35 ms** for the existing shortcut on the middle-layer query block. Changing the merge workgroup geometry sometimes changed BF16 outputs. [Merge geometry and shortcuts](k2-prefill-stage-schedule-v1/amd-k2-short-merge-tune.json)
- Double buffering QK and selection on separate GPU streams took **16.89 ms** at its best bitwise-identical setting, versus **15.97 ms** for the full-width pass. The reduction and the selected regions were identical in that setting. [Score pipeline replay](k2-prefill-score-pipeline-v1/amd-k2-score-pipeline.json)
- A CK-style 4-by-64 MFMA value reducer with a separate LSE preparation and final merge took **1.23 ms** against **1.35 ms**. It changed 5,049 merged BF16 components on the saved query block and had a maximum absolute difference of 0.03125. [Small-MFMA merge replay](k2-prefill-ck-merge-v1/amd-k2-merge-mfma4.json)

## 64K speed

| Configuration | Warmed 64K prefill median |
|---|---:|
| [Native top-eight reference](k2-prefill-fast-centroid-v1/amd-k2-fast-centroid-top8.json) | 13.66 s |
| [Current threshold, matched leaf-scheduling control](k2-prefill-chiplet-check-v1/amd-k2-chiplet-full-control.json) | 15.21 s |
| [38 adjacent tiles per chiplet and 608 workers](k2-prefill-chiplet-check-v1/amd-k2-chiplet38-full.json) | 15.21 s |
| [76 adjacent tiles per chiplet and 2,432 workers](k2-prefill-chiplet-check-v1/amd-k2-chiplet76-full.json) | 15.14 s |
| [Combined-round threshold control](k2-prefill-style-combined-v1/amd-k2-style-control.json) | 15.12 s |
| [Chiplet scheduling, RMS preparation and padded masking](k2-prefill-style-combined-v1/amd-k2-style-combined.json) | 15.22 s |
| [Combined with streaming probability stores](k2-prefill-style-combined-v1/amd-k2-style-stream-combined.json) | 15.20 s |

Grouping adjacent exact-leaf tiles in launch order keeps each region's key order and encourages reuse of the same region's KV on a chiplet. On the 4,096-query frozen replay, the fastest configuration measured **2.62 ms** against **2.69 ms** and preserved all live exact values and log partitions. [Leaf schedule replay](k2-prefill-stage-schedule-v1/amd-k2-chiplet-leaf-tune.json)

The combined configuration preserved all selected regions, coarse values and log partitions, and every remote output bit across 3,145,728 query-head rows. [Layer 0](k2-prefill-style-combined-v1/amd-k2-style-bitwise-00.json) · [Layer 32](k2-prefill-style-combined-v1/amd-k2-style-bitwise-32.json) · [Layer 63](k2-prefill-style-combined-v1/amd-k2-style-bitwise-63.json)

## Same-GPU crossover

All three configurations shared a single warmed model process and GPU, with the launch order rotated across three measured rounds. The control median was **15.24 s**, chiplet scheduling alone **15.27 s**, and chiplet scheduling plus RMS preparation and padded masking **15.23 s**. All 12 requests returned identical greedy tokens. The best tuned crossover median remains **1.57 s** above 13.66 s. The active implementation retains the previously validated kernel path. [Interleaved timings and launch settings](k2-prefill-one-process-v1/amd-k2-threshold-interleaved.json)

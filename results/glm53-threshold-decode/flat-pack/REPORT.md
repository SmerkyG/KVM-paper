# Flat ID packing in GLM-5.3-Flash threshold decode

## Decode layer at 64K

Each time is in microseconds per MLA layer on one MI325X. The separate flat packer finds the owner of each selected leaf and materializes its physical ID and head mask in an extra kernel. The fused flat packer copies a contiguous slice of IDs from each selected region and writes its head mask alongside the IDs, in the selection kernel. Runs emit one offset, valid length and mask per span into the persistent directory. All three perform the same per-head selection and read the same physical latent rows.

| Decode schedule | Batch | Separate flat | Fused flat | Runs | Flat-to-run gap recovered | Fused flat above runs |
|---|---:|---:|---:|---:|---:|---:|
| Original | 1 | 64.81 | 46.67 | 43.58 | 85.4% | +7.1% |
| Original | 8 | 129.73 | 114.09 | 109.50 | 77.3% | +4.2% |
| Optimized | 1 | 62.17 | 44.92 | 41.10 | 81.9% | +9.3% |
| Optimized | 8 | 128.76 | 112.81 | 107.37 | 74.5% | +5.1% |

The recovered fraction is (separate flat − fused flat) / (separate flat − runs). The original schedule matches the earlier run-width experiment. The optimized schedule defers normalization to the final merge. At B=1 it also combines the closed and exact attention work. All arms within a row use the same schedule. Exact workers use 256 splits at B=1 and 64 at B=8. Runs use widths of 32 and 8 respectively.

## Where the latency goes

| Schedule | Batch | Prepare separate | Prepare fused | Prepare runs | Attention worker separate | Attention worker fused | Attention worker runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 1 | 31.90 | 16.03 | 13.74 | 16.18 | 16.24 | 16.41 |
| Original | 8 | 49.28 | 34.09 | 26.93 | 52.98 | 53.28 | 56.40 |
| Optimized | 1 | 28.54 | 13.87 | 10.19 | 19.13 | 19.08 | 19.20 |
| Optimized | 8 | 49.08 | 32.84 | 26.38 | 52.17 | 52.10 | 54.89 |

Preparation includes a counter reset, region selection, writing the selected output and any separate expansion kernel. The original schedule also does closed-summary attention there. The optimized B=1 schedule does that work in the attention worker. Stage timings use isolated warmed graphs and do not add to the full graph time, which also includes scoring, the append, the final merge and a common local-length reset.

## Implementation and validation

In the initial sweep, moving the 256-lane per-leaf region lookup inside selection changed full-layer latency from 65.21 to 64.13 µs at B=1 and from 130.29 to 129.17 µs at B=8. Copying directly from each selected region avoids that lookup. At B=1, four workgroups copy disjoint portions of each selected region with 32 lanes per region. Only one writes the closed summary and audit mask for the tile. At B=8, one workgroup handles the routing tile and uses 64 lanes per region. The chosen flat configuration is shared across the two trained captures for each batch. The sweeps also tested other copy widths, copying four active regions at a time, expanding temporary run descriptors in registers, and splitting the copy over two to eight workgroups. [Benchmark](../../../benchmarks/glm53_flat_pack.py) · [Selection and exact kernels](../../../lod_attention/kernels/glm_threshold_decode.py)

Each row equally weights the per-capture median of 13 interleaved rounds of 30 warmed GPU graph replays. The captures use 16 query heads on a TP4 rank, one shared 512-channel BF16 latent, a region threshold of 16, and trained queries and KV from GLM layers 3 and 19 after 65,536 and 64,000 prompt tokens. B=8 uses neighboring query positions with eight separate caches. Every candidate was checked against an independent FP32 reference, including region masks, sorted physical leaf IDs and per-leaf membership, outputs, log-normalizers, cache writes and length advancement. The confirmation also checked numerical outputs from the replayed full graphs. The production decode regression suite passed nine tests. Full-model serving and task scores were outside this packing experiment. [Reference and replay harness](../../../benchmarks/glm53_threshold_decode.py) · [Per-stage results and raw capture filenames](summary.json) · [Cluster commands and placement](jobs.json)

The two confirmation panels each ran on four separately allocated GPUs with one job per GPU. An earlier confirmation submission had overlapping GPU assignments and was excluded from all reported confirmation results.

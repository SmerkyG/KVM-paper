# K2 prefill merge and coarse-consumer experiments

The trials use real query, centroid and KV states saved from a 64K K2 Horizon prompt on MI325X. They keep c=16, BF16 KV, and the original 16K exact prefix. Tests replay the remote attention stages at layer 32, with an additional layer-0 merge check. Every experiment uses uv through cluster-run with an archived source snapshot and launch command.

| Trial | Current stage time | Best tested candidate | Result |
|---|---:|---:|---|
| Separate short-frontier merge, 4,096 queries | 1.356 ms | 1.376 ms | Every tested output bit matched |
| Fused selection and coarse-value consumption, 4,096 queries | 4.047 ms | 4.332 ms | Zero changed route rows, changed coarse rounding |
| Fused selection and coarse-value consumption, 16,384 queries | 16.102 ms | 17.239 ms | Four changed route rows out of 1,048,576 |
| Indexed AITER Triton attention with dispatch, KV gathering and merge, 4,096 queries | 4.146 ms | 4.175 ms | Best candidate matched every tested output bit |

The serving implementation uses the preceding validated kernels. These stage timings provide no improvement to the latest full-model medians of 15.21 s at 64K and 36.85 s at 128K. The new variants are retained in their source archives.

## Separate merge kernels

The experiment compiles separate kernels for query groups whose largest frontier is at most 32 or 64 regions and for the remaining groups. Each kernel checks four existing route counts on GPU before loading the attention inputs. The short kernel excludes the long-frontier reduction at compile time while preserving the existing 64-wide FP32 mass reduction and the shorter value MMA. Every query group has exactly one writer.

| Layer | Original merge | Split at 32 | Split at 64 | Split at 32 and 64 |
|---:|---:|---:|---:|---:|
| 0 | 1.403 ms | 2.328 ms | 1.415 ms | 2.314 ms |
| 32 | 1.356 ms | 1.880 ms | 1.376 ms | 1.905 ms |

All variants matched the final remote outputs bit for bit on 524,288 query-head rows across these two saved blocks. In the middle layer, 31.6% of four-query groups exceed 32 regions, while 0.0015% exceed 64. The additional launches did not produce a speed benefit. [Layer 0](k2-prefill-separate-merge-v1/amd-k2-split-merge00.json) · [Layer 32](k2-prefill-separate-merge-v1/amd-k2-split-merge32.json) · [Source and runner](k2-prefill-separate-merge-v1/source/benchmarks/amd_k2_prefill_merge_split.py)

## Fused coarse consumer

The parallel BF16-input, FP32-output centroid GEMM also emits per-tile routing maxima and masses. A small reduction obtains the final cutoff. A new consumer reads the saved scores once, selects regions and accumulates closed-summary values with tensor cores. It writes selection bits and small partial outputs in place of the dense probability matrix. A row-batched final reduction combines the coarse partitions. QK is computed once.

The first implementation took 4.577 ms against 4.053 ms. Batching the final reduction and tuning query/region tiles improved the best 4K replay to 4.332 ms against 4.047 ms. On the full saved 16K query block, the best setting took 17.239 ms against 16.102 ms. FP16 coarse products and preparing query RMS once did not improve the result. The last RMS trial took 17.265 ms against 16.103 ms. [Initial sweep](k2-prefill-fused-consumer-v1/amd-k2-consumer32.json) · [4K tuning](k2-prefill-fused-consumer-v2/amd-k2-consumer-v2-4k.json) · [16K tuning](k2-prefill-fused-consumer-v2/amd-k2-consumer-v2-16k.json) · [RMS reuse](k2-prefill-fused-consumer-v3/amd-k2-consumer-v3-16k.json)

Tiled normalization changes the floating-point reduction order. The 4K query slice had identical selections, and the full 16K slice changed four selected-region rows. Online coarse accumulation also changes BF16 rounding, with a maximum coarse-output difference of 0.03125. These candidates failed the strict selection-equivalence check and were slower. [Consumer source](k2-prefill-fused-consumer-v3/source/lod_attention/kernels/threshold_prefill_fused_consumer.py)

## Indexed AITER attention

The adapter calls AITER's released Triton attention inner loop, gathers Q directly through the existing GPU route indices, and writes directly to compact query-region output addresses. The timed full path includes the existing GPU route grouping and region KV materialization. The benchmark uses the same saved routes and the same original KV order. The earlier CK-wrapper experiment's query packing and output scattering are eliminated in this Triton adapter.

The best bitwise-matching setting used 128 queries, 32 keys, four waves and 2,432 persistent workers with preloaded values and pipelining. Its attention loop took 2.317 ms. Including grouping and KV materialization took 2.754 ms, against 2.714 ms for the existing LoD path. Including the final merge took 4.175 ms, against 4.146 ms. The 64-key variants were slower and changed some rounded outputs. [Corrected sweep](k2-prefill-aiter-indexed-v3/amd-k2-aiter-indexed32-corrected.json) · [Adapter source](k2-prefill-aiter-indexed-v3/source/benchmarks/amd_k2_prefill_aiter_indexed.py)

The first adapter launch failed a validation assertion that also inspected unused KV slots. The second launch had an incorrect tail-mask argument and included padded keys in attention. Both are invalid experiments. The corrected sweep above checks only used KV entries and supplies AITER's required tail count. Its 32-key variants match the merged output bit for bit. [Initial assertion failure launch](k2-prefill-aiter-indexed-v1/amd-k2-aiter-indexed32-submission.json) · [Invalid tail-mask run](k2-prefill-fused-consumer-v3/amd-k2-aiter-indexed32-v2.json)

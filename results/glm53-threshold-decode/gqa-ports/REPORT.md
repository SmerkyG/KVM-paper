# GQA kernel optimizations in GLM-5.3-Flash threshold decode

Both arms use the original GLM per-head c=16 region-mass rule and descriptor-based leaf access. We ported three execution choices from the Qwen/K2 kernels. Workers can pass an FP32 value numerator, maximum and softmax mass to the final merge, allowing it to perform the normalization once. An exact-attention worker can also accumulate a tile of closed summaries and emit a single combined partial. Finally, workers can take consecutive blocks of selected leaves. Each query head keeps its own membership mask in all variants. [GLM kernel](../../../lod_attention/kernels/glm_threshold_decode.py) · [Reference and benchmark](../../../benchmarks/glm53_threshold_decode.py)

## Full-model decode at 64K

| Batch | Previous c=16 (ms/step) | Ported c=16 (ms/step) | Latency reduction | Previous repeats (ms) | Ported repeats (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 8.153 | 8.114 | +0.48% | 8.152 / 8.153 | 8.114 / 8.113 |
| 8 | 13.524 | 13.517 | +0.05% | 13.524 / 13.523 | 13.516 / 13.517 |

The batch-one saving was 39 µs per full-model step. At batch eight the two implementations were effectively tied. The mean difference was 7 µs per step, despite a repeatable attention-layer improvement on both captures. These serving measurements use one GPU allocation and two repetitions per arm.

The B=1 port combines closed and exact partials, gives each worker a contiguous range and defers normalization. The B=8 port defers normalization and keeps the previous separate closed projection and interleaved exact workers. Each arm used the same TP4 group of MI325Xs, native FP8 weights, BF16 KV, top-eight prefill and 65,536-token ProLong prompts. After a complete warmup there were two measured generations per arm, each with 1,025 timed steps, all requests live, and four 256-token cache updates per request/layer/rank. Every generated token matched the prescribed trace. The threshold and merge switches were audited on all 11 MLA layers on all four ranks. [B1 control](serving-b1-c16.json) · [B1 port](serving-b1-ported.json) · [B8 control](serving-b8-c16.json) · [B8 port](serving-b8-ported.json)

## Controlled kernel comparison

| Implementation | B=1 (µs/layer) | B=8 (µs/layer) |
|---|---:|---:|
| Previous threshold kernel | 43.76 | 106.56 |
| Normalize at final merge | 42.89 | 104.00 |
| Combine closed and exact attention | 42.73 | 127.13 |
| Combine attention and normalize at merge | 41.38 | 125.63 |
| Contiguous distribution of leaf tiles | 43.31 | 106.48 |
| Normalize at merge and use contiguous tiles | 42.24 | 104.46 |
| All three changes | 41.14 | 124.11 |

A separate two-arm confirmation measured 43.51 → 41.06 µs at batch one and 105.77 → 103.75 µs at batch eight. Each value equally weights two trained GLM captures after 65,536 and 64,000 tokens. The benchmark uses final-prefill queries as decode inputs and batch eight uses adjacent query positions and eight separately constructed caches. The same exact entries and head masks were verified for every candidate. Latent outputs and log normalizers passed an independent FP32 reference. [All timing rounds and source commands](jobs.json) · [Summary](summary.json)

In the serving kernel this schedule is opt-in with `LOD_GLM_THRESHOLD_DEFERRED_MERGE=1` and `LOD_GLM_DECODE_THRESHOLD=16`. The tests exercise empty remote fields, over-cap regions, 32 query heads, graph replay on different slots and changed query/KV inputs. The experiment measures speed at fixed routing policy. Downstream quality was not scored. [GPU regression tests](../../../tests/test_glm_threshold_decode.py) · [Prior GLM native and top-eight comparison](../REPORT.md)

# Paired cost of Qwen probability precision correction

Qwen3.8-27B-FP8, nominal 64K archived real-model query/KV snapshots on one MI325X. Both settings use the same selected entries, cached state, local/sink attention, materialized-index/GQA-shared kernel settings and local-length rollback. Only `LOD_AMD_THRESHOLD_PRECISE` changes. Every GPU invocation uses uv through cluster-run with the frozen inference source.

Each layer runs flags **0 → 1 → 1 → 0**, with graph warmup and three timing repetitions per configuration. The table averages the two configuration measurements per flag. All selected masks matched and BF16-aware output checks passed. The full model and cache catch-up are outside this scope.

| Sampled LoD layer | Correction off | Correction on | Extra per call | Attention overhead |
|---|---:|---:|---:|---:|
| 0 | 34.94 µs | 40.98 µs | 6.03 µs | 17.3% |
| 8 | 33.56 µs | 35.56 µs | 2.00 µs | 6.0% |
| 15 | 38.87 µs | 40.86 µs | 1.99 µs | 5.1% |
| **Mean** | **35.79 µs** | **39.13 µs** | **3.34 µs** | **9.3%** |

The correction costs time at every sampled layer. These are three fixed queries at one context length, not a universal overhead estimate or an end-to-end throughput result. The earlier 35.59 µs and 41.02 µs figures came from different generated queries and must not be subtracted to estimate correction cost.

Qwen has 16 LoD attention pools in the serving probe (which samples pool indices 0, 8 and 15). If these sampled costs represent all 16, the projected extra time is **0.053 ms/token**, about **0.2%** of the earlier uncorrected 28.44 ms/token full-model timing. This is a projection; corrected full-model speed has not been measured.

[Exact command](job.json), [configurations](configs.json), [raw timings and input hashes](qwen.json), [summary](REPORT.json), [parent precision/quality panel](../precision-panel/REPORT.md).

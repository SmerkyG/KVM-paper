# Qwen precision follow-up: all 13 RULER tasks

Qwen3.8-27B-FP8, batch-one two-tier LoD, BF16 KV, equal-per-region c=16 decode routing and nominal 64K archived RULER requests. The first [32-example/task panel](../REPORT.md) found some changed answers with the optimized BF16-probability path. This follow-up changes only `LOD_AMD_THRESHOLD_PRECISE=1`: the exact-entry value product adds a second BF16 dot product for the probability-rounding residual. KV storage, routing, local/sink attention, prefill, prompts and generation settings are unchanged. It uses the same frozen first 32 examples of every task, not just examples with changed scores.

| Task | Old c=16 | Optimized | Precision correction | Correction vs old, points | Improved / declined vs old |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_single_2 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_single_3 | 100.00 | 96.88 | 100.00 | +0.00 | 0 / 0 |
| niah_multikey_1 | 100.00 | 96.88 | 100.00 | +0.00 | 0 / 0 |
| niah_multikey_2 | 96.88 | 100.00 | 100.00 | +3.12 | 1 / 0 |
| niah_multikey_3 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_multiquery | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_multivalue | 100.00 | 99.22 | 100.00 | +0.00 | 0 / 0 |
| ruler_cwe | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| ruler_fwe | 97.92 | 100.00 | 100.00 | +2.08 | 2 / 0 |
| ruler_qa_hotpot | 65.62 | 59.38 | 62.50 | -3.12 | 0 / 1 |
| ruler_qa_squad | 74.48 | 78.39 | 74.48 | +0.00 | 0 / 0 |
| ruler_vt | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| **13-task mean** | **94.99** | **94.67** | **95.15** | **+0.16** | **3 / 1** |

Paired bootstrap 95% interval for correction minus old: [-0.56, +0.88] points. Thirty-two examples per task cannot certify noninferiority on the full benchmark. Fractional RULER recall scores are retained.

## Attention timing

| Old c=16 | Optimized with precision correction | Native top-8 |
|---:|---:|---:|
| 86.94 µs | 41.02 µs | 39.21 µs |

These are graph-captured attention calls on identical actual query/cache states from the first, middle and last layers. The scope includes routing, selected entries, coarse/local/sink attention and a common local-length rollback; it excludes the rest of the model and cache catch-up. It is not a full-model speed measurement. The earlier uncorrected timing used different generated queries and must not be subtracted from this result to estimate correction overhead. Native top-8 retains its own routing rule and refinement budget, so this is not a work-matched comparison.

A separate [paired precision-cost check](../precision-paired-timing/REPORT.md) runs the correction off/on on identical saved queries and KV states; use that check to quantify precision overhead.

[Commands and settings](jobs.json), [all job statuses and placement](provenance.json), [kernel probe](kernel-probe.json), [raw summary](REPORT.json), [changed predictions](changed-examples.json), [parent token audit](../token-audit.json). This experiment leaves serving defaults unchanged.

# H100 threshold-prefill normalization and packing

BF16 Llama-3.1-8B-Instruct, batch 1, entry-16 two-tier prefill. The existing 16,384-token exact initial prefill chunk is unchanged. Every timed result is a full-model prefill median of three runs after one warmup, on the same H100 and prompt.

The producer computes normalized-query routing scores and partial log-normalizers together, then selection reduces those partials and reads the score matrix once. Bitmask packing uses one bit per region. Direct packing writes selected region IDs to bounded per-query buffers and retains packed entry indices; it avoids the selection mask.

| Variant | 32K prefill | 64K prefill |
|---|---:|---:|
| Baseline, matched source | 2.509 s | 7.998 s |
| Producer, matched source | 2.460 s | 7.592 s |
| Producer, initial snapshot | 2.456 s | 7.586 s |
| Bitmask | 2.566 s | 8.114 s |
| Producer + bitmask | 2.552 s | 7.783 s |
| Direct region packing | 2.528 s | 7.920 s |
| Producer + direct packing | 2.513 s | 7.554 s |
| Producer + direct packing, 64-region tiles | 2.462 s | 8.673 s |

The bitmask and direct-packing variants were tested in separate logged source snapshots. The baseline and producer rows are repeated with the same snapshot; other rows should be read as separate short-run measurements. Earlier [top-8 LoD timings](../prefill-next/REPORT.md) were 1.656 s at 32K and 4.189 s at 64K; this experiment has not closed that gap.
At 32K, synchronized diagnostic stage times summed over layers: routing score generation 0.152 s baseline versus 0.177 s producer; selection plus coarse 0.473 s versus 0.382 s. These include diagnostic synchronization and launch overhead and are not end-to-end GPU event measurements.

Real-state checks:

- producer: 3 RULER examples, 49,152 query-heads, 8 selection differences from the independent reference (0 outside tolerance); largest output error 0.015625.
- bitmask: 3 RULER examples, 49,152 query-heads, 1 selection differences from the independent reference (0 outside tolerance); largest output error 0.015625.
- combined: 3 RULER examples, 49,152 query-heads, 8 selection differences from the independent reference (0 outside tolerance); largest output error 0.015625.
- direct: 3 RULER examples, 49,152 query-heads, 1 selection differences from the independent reference (0 outside tolerance); largest output error 0.015625.
- producer-direct: 3 RULER examples, 49,152 query-heads, 8 selection differences from the independent reference (0 outside tolerance); largest output error 0.015625.
- Producer + direct packing, 64-region tiles: 3 examples, 49,152 query-heads, 0 selection differences (0 outside tolerance); largest output error 0.015625.

For producer-enabled variants, the check also compares their complete routing-logit tile against the original matmul bit-for-bit. Selected-region differences within 2e-5 of the decision boundary are reported, not hidden.

| Paired RULER task | Examples | Baseline | Producer | Improved | Declined |
|---|---:|---:|---:|---:|---:|
| qa_1 | 32 | 68.75% | 71.88% | 1 | 0 |
| niah_multikey_2 | 32 | 71.88% | 81.25% | 3 | 0 |

Scores stop at the first end-of-turn or end-of-text token. Paired quality runs use identical execution source files; their snapshots differ only in the submission helper. These two tasks are a limited quality sample, not the full RULER panel. Results, run commands, and source hashes are retained in this directory. The initial direct-packing checks failed because the diagnostic reader first treated a flat workspace as shaped and then inferred its current width from a larger retained allocation; dependent runs were skipped and resubmitted after that reader was corrected.

| Mask-free paired RULER task | Examples | Baseline | Producer + direct, 64-region tiles | Improved | Declined |
|---|---:|---:|---:|---:|---:|
| qa_1 | 32 | 68.75% | 71.88% | 1 | 0 |
| niah_multikey_2 | 32 | 71.88% | 81.25% | 3 | 0 |

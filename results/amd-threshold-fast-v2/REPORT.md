# AMD two-tier threshold decode optimization

Final setting: equal-per-region **c=16**, batch one, 65,536-token ProLong input, BF16 KV and FP8 model weights on MI325X. The router retains normalized-query ranking and the existing 1,024-entry eligibility limit. No top-k cap was added. The exact first prefill chunk remains 16,384 tokens, and later prefill retains top-8 routing. The per-entry rule is outside this result.

## Full-model decode, including cache updates

Each model uses one GPU and serial **top-8 → optimized c=16 → top-8** runs. Each run warms up with one full generation, then measures two generations of 1,024 decode steps. The final top-8 control checks baseline drift; overlapping repetition ranges are treated as near parity. “Less time” compares optimized latency to the median of all four native-control repetitions. These are measurements on one prompt; autoregressive continuations can differ, and the methods are not quality- or work-budget matched.

| Model | Top-8 before, ms/token | Optimized c=16 | Top-8 after | Less time vs control median |
|---|---:|---:|---:|---:|
| QWEN | [28.473](speed-v4/amd-fast-speed-v4-qwen-top8.json) | [28.442](speed-v4/amd-fast-speed-v4-qwen-fast.json) | [27.456](speed-v4/amd-fast-speed-v4-qwen-top8-after.json) | -1.7% |
| K2 | [39.240](speed-v4/amd-fast-speed-v4-k2-top8.json) | [39.060](speed-v4/amd-fast-speed-v4-k2-fast.json) | [39.225](speed-v4/amd-fast-speed-v4-k2-top8-after.json) | +0.5% |

QWEN: timing ranges overlap or favor top-8; a consistent speedup is not established.
K2: both optimized repetitions are below all four top-8 control repetitions in this run.

## Implementation

The chosen experimental path has five stages:

1. Share summary-key reads across GQA heads, produce native and normalized proxy logits plus tile normalizers, and append the current KV entry without advancing its length.
2. Compute the c=16 threshold; evaluate closed summaries and pack selected regions, lengths and per-head membership bits inside each routing tile.
3. Use parallel prefix-scan shards to compact physical KV indices and membership bits into a shared table. No KV data is copied. The number of shards is fixed, so packing work is linear in routing-tile count plus the selected union’s entry count.
4. Use AITER-shaped segmented attention over the compact table, sharing KV reads across GQA heads while retaining each head’s own frontier. Local, sink and current entries are included.
5. Merge partial attention results and advance the local length.

The selected settings use 64-region routing tiles, 128 exact segments, 64 entries per attention block, eight index-preparation shards, one warp per preparation worker, four warps per exact worker and a 32-dimension merge block. The tensor-core value product uses native-style BF16 attention probabilities; stored KV remains BF16. Counts remain FP32 when forming means. There is no top-k cap, sorting, selected-count CPU readback, change to c=16, or removal of local/sink attention.

The persistent index already exists in Qwen’s native path. The opt-in K2 path maintains it at normal cache catch-up boundaries. **That extra K2 update cost is included in the full-model timings.**

## Work audit

Cached-summary QK scores are computed once per call and reused for routing and coarse attention. The chosen index-packing stage uses prefix scans and a fixed number of workers; it does not compare every region with every other region. Its flat tables have linear remote-token capacity, but exact attention visits only the selected union plus local/sink/current entries. The per-tile coarse stage still repeats the reduction of all tile normalizers: with T routing tiles, that metadata arithmetic is O(T²), while summary matrix products are linear in region count at fixed head dimension. Persistent-index catch-up visits the archive. Prefill still uses its existing top-8 implementation after the exact first chunk and has not been audited as subquadratic here. **No complete-execution subquadratic claim follows from these measurements.**

## Same-query kernel checks

The first, middle and last attention layers use actual model queries and serving caches after a three-token generation. Old threshold, optimized threshold and native top-8 calls see identical query/cache state, with the same local-length rollback. Native top-8 uses its own routing rule and refinement budget. The table averages three sampled layers. This scope excludes projections, FFNs, scheduling and cache catch-up and must not be read as a full-model speedup.

| Model | Old c=16, µs/call | Optimized c=16 | Native top-8 | Less time vs old c=16 | Max attention-component difference |
|---|---:|---:|---:|---:|---:|
| QWEN | 87.98 | 35.59 | 39.04 | 59.5% | 0.0625 |
| K2 | 127.49 | 39.17 | 36.83 | 69.3% | 0.015625 |

All sampled old/optimized threshold selected-region masks matched exactly. Tensor results are checked with BF16-aware tolerances (atol=0.004, rtol=0.008) plus independent PyTorch frontier tests; absolute differences are reported above. The chosen settings passed 12 independent PyTorch frontier cases over both head geometries, repeated under both model configurations. These cover empty selections, a 1,024-entry region, exclusion of a 1,025-entry region, and repeated fixed-cache calls.

## Retrieval smoke check

Eight examples per task and model: four historical misses plus four successes. Requests and reference answers are paired exactly. This deliberately enriched 32-example total checks implementation regression and is **not a full RULER panel or an accuracy estimate**.

| Model | Task | Old c=16 | Optimized c=16 |
|---|---|---:|---:|
| QWEN | niah_multikey_2 | 75.00% | 75.00% |
| QWEN | niah_multikey_3 | 100.00% | 100.00% |
| K2 | niah_multikey_2 | 62.50% | 62.50% |
| K2 | niah_multikey_3 | 87.50% | 87.50% |

- QWEN native before/after controls generated different token sequences despite matching prompt hashes. The bracketing spread cannot be attributed to clock drift alone.
- QWEN retrieval: 1 improved, 1 declined; text changed on 7/16 requests.
- QWEN greedy repeat checks against each run’s warmup: top8: identical, fast: identical, top8-after: identical.
- K2 native before/after controls generated different token sequences despite matching prompt hashes. The bracketing spread cannot be attributed to clock drift alone.
- K2 retrieval: 0 improved, 0 declined; text changed on 0/16 requests.
- K2 greedy repeat checks against each run’s warmup: top8: identical, fast: first mismatch [[101], [101]], top8-after: first mismatch [[52], [3]].

## Reproduction and exclusions

Every GPU run uses `uv` through `cluster-run`. Sources are frozen per phase and hash-validated. The report requires successful completion of the final speed, serving-cache, reference and quality jobs. Earlier alternatives and failed attempts remain in the provenance but contribute no rows to the final tables: grouped GQA, small GQA subgroups, whole-frontier consumers, region-owned workers, balanced scalar workers, active-block workers, fused tile-local tables, resource-limited configurations, a compiler-rejected pointer-select reduction, and initial numerical-validation failures. The absolute-only 0.04 validation gate rejected a one-BF16-ULP difference (0.0625 at a large output component); the final gate uses the explicit dtype-aware tolerance above and reports maximum absolute errors. The first K2 serving attempt failed at initialization because its native path did not allocate the persistent index; this was fixed and rerun with its maintenance cost included.

[Raw summary](REPORT.json), [chosen settings](settings-v4.json), [all jobs, placement and exit status](provenance.json), [retrieval cohort](quality-v4/cohort-manifest.json), [Qwen serving-cache checks](verify-model-v4/amd-fast-verify-model-v4-qwen.json), [K2 serving-cache checks](verify-model-v4/amd-fast-verify-model-v4-k2.json).

Enable this experiment with `LOD_AMD_THRESHOLD_MULTIPLIER=16` and the environment settings in `settings-v4.json`. The serving default remains the existing native router. After freezing these measurements, CPU dispatch guards were added for the tested two-tier BF16 head geometries; the independent frontier checks were rerun, with no change to the measured GPU kernels.

## Subsequent 32-example/task RULER validation

The [complete regression panel](../amd-threshold-ruler32-64k/REPORT.md) tests all 13 tasks at nominal 64K with 32 paired examples per task and model, retaining the same prefill and decode settings as the speed measurements above. Qwen scored 94.99% with the old c=16 kernel and 94.67% with the optimized kernel (six examples improved, five declined). K2 scored 92.17% and 92.32%, respectively (four improved, one declined). The requests were selected by their first 32 IDs, without enriching failures. Archived dense scores, actual prompt-length ranges, independent prompt-token verification and all successful job exits are included in the report.

The [Qwen precision follow-up](../amd-threshold-ruler32-64k/precision-panel/REPORT.md) enables the existing exact-entry probability-residual correction with `LOD_AMD_THRESHOLD_PRECISE=1`. On the same complete 13-task panel it scored 95.15%, versus 94.99% for the old kernel, with three examples improved and one declined. All eight retrieval-task scores matched or improved; one HotpotQA example still declined. These small, reused cohorts do not certify unchanged quality on the full benchmark.

For further Qwen quality runs, prefer this correction; K2's tested optimized setting retains `PRECISE=0`. The combined opt-in environment settings are saved in [settings-ruler32.json](../amd-threshold-ruler32-64k/settings-ruler32.json). On a fresh same-query/cache attention probe, corrected Qwen took **41.02 µs**, compared with **86.94 µs** for old c=16 and **39.21 µs** for native top-8. Thus it was 52.8% faster than the old threshold kernel and 4.6% slower than top-8 in this attention-only scope. **The full-model speed table above measures the uncorrected setting; corrected full-model speed has not been measured.** Serving defaults are unchanged.

A subsequent [paired precision-cost test](../amd-threshold-ruler32-64k/precision-paired-timing/REPORT.md) compares correction off/on using identical saved queries and KV states in order 0 → 1 → 1 → 0. Across three sampled layers, the mean rises from **35.79 to 39.13 µs**, an extra **3.34 µs per call (9.3%)**. This directly isolates the flag's attention cost; the earlier 35.59 and 41.02 µs measurements use different queries. Projecting the paired difference across Qwen's 16 LoD attention pools gives about 0.053 ms/token, or 0.2% of the earlier 28.44 ms/token full-model timing, if the samples are representative. This projection is not a corrected full-model timing measurement.

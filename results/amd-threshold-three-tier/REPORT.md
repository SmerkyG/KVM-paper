# AMD three-tier threshold decode

Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8, batch one, nominal 64K context, BF16 or INT4 KV. The experiment selects every eligible region with normalized-query proxy mass above 16 times the mean region mass. Each selected region opens the existing two highest-scoring pages, plus one disjoint residual summary. Regions over 1,024 entries remain coarse, as in serving. Page ranking and INT4 reconstruction use the native implementation. Prefill retains top-8 routing and the exact first 16,384-token chunk. The implementation is opt-in; serving defaults are unchanged.

## Amortized full-model decode

One GPU per model/precision runs native top-8 → threshold → native top-8 serially. Each variant warms up, then measures two 1,024-step generations, including periodic cache catch-up and quantization maintenance. Time change compares threshold to the median of all four native-control repetitions. Prompts match; generated continuations can differ.

| Model | KV | Native before | Threshold | Native after | Time change vs control median |
|---|---|---:|---:|---:|---:|
| K2 | BF16 | 40.939 ms | 41.726 ms | 40.875 ms | +2.0% |
| K2 | INT4 | 41.235 ms | 42.899 ms | 41.035 ms | +4.5% |
| QWEN | BF16 | 29.419 ms | 28.251 ms | 29.525 ms | -4.1% |
| QWEN | INT4 | 28.658 ms | 29.865 ms | 28.691 ms | +4.2% |

The native before/after controls generated different continuations despite matching prompt hashes for K2/BF16, K2/INT4, QWEN/BF16, QWEN/INT4. The bracket therefore measures both timing and continuation variability; small differences should not be read as an isolated kernel-cost effect. Raw repetition hashes and timing ranges are retained.

## Same-query/cache attention

Graph-captured complete attention calls on actual queries and retained serving states from first/middle/last LoD layers. All methods include the same local-length rollback. Model projections, FFNs and cache catch-up are excluded. Native top-8 opens at most eight regions; threshold uses its actual variable region budget. This is not a work-matched comparison.

| Model | KV | Native top-8 | Threshold | Mean regions/head | Exact remote-entry bound/head |
|---|---|---:|---:|---:|---:|
| QWEN | BF16 | 98.81 µs | 79.18 µs | 38.26 | ≤1224.4 |
| QWEN | INT4 | 108.56 µs | 104.67 µs | 37.75 | ≤1208.0 |
| K2 | BF16 | 57.34 µs | 80.75 µs | 29.11 | ≤931.7 |
| K2 | INT4 | 61.21 µs | 97.31 µs | 28.92 | ≤925.3 |

The entry bound is 32 times the opened-region count, averaged across sampled heads/layers; partial or singleton pages reduce actual work. It is not a measured exact-entry count. Closed and residual summaries are additional frontier items.

## Retrieval screen

The first 32 examples of Multikey 2 and Multikey 3 per model, with identical frozen requests, generation settings and seed 1234 across methods and KV precisions. These are selected retrieval tasks, not a full 13-task RULER score. Dense uses the same archived requests in an older runtime and is descriptive. Token hashes are checked against the earlier independent CPU token audit.

| Model | KV | Task | Archived dense | Fresh top-8 | Threshold | Difference, points | Improved / declined |
|---|---|---|---:|---:|---:|---:|---:|
| K2 | BF16 | niah_multikey_2 | 100.00 | 62.50 | 90.62 | +28.12 | 9 / 0 |
| K2 | BF16 | niah_multikey_3 | 100.00 | 18.75 | 96.88 | +78.12 | 26 / 1 |
| K2 | INT4 | niah_multikey_2 | 100.00 | 50.00 | 81.25 | +31.25 | 10 / 0 |
| K2 | INT4 | niah_multikey_3 | 100.00 | 21.88 | 84.38 | +62.50 | 20 / 0 |
| QWEN | BF16 | niah_multikey_2 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| QWEN | BF16 | niah_multikey_3 | 100.00 | 84.38 | 100.00 | +15.62 | 5 / 0 |
| QWEN | INT4 | niah_multikey_2 | 100.00 | 96.88 | 100.00 | +3.12 | 1 / 0 |
| QWEN | INT4 | niah_multikey_3 | 100.00 | 87.50 | 100.00 | +12.50 | 4 / 0 |

The fresh batch-one K2 top-8 controls are substantially below the historical 500-example three-tier report, which used a different serving revision and schedule. This discrepancy is unresolved, so gains here should not be read as reproducing an improvement over that historical baseline. A separate [pre-extension native-kernel audit](native-audit-v1/jobs.json) found bitwise-identical full native attention outputs on all 12 sampled real-query/cache cases across both models and KV precisions.

## Prefill timing

| Model | KV | Native before | Threshold | Native after |
|---|---|---:|---:|---:|
| K2 | BF16 | 13.922 s | 13.940 s | 13.918 s |
| K2 | INT4 | 14.732 s | 14.736 s | 14.717 s |
| QWEN | BF16 | 8.071 s | 8.334 s | 8.078 s |
| QWEN | INT4 | 8.310 s | 8.480 s | 8.317 s |

## Enable the experiment

Use the existing `three-tier-bf16` or `three-tier-int4` model mode and add `LOD_AMD_THRESHOLD_MULTIPLIER=16` to the cluster-run environment. The new path defaults to 64 persistent page workers, normalized-query per-region mass and the precision correction. The selected settings JSON below records all flags explicitly. The current experiment supports batch-one ordinary decode on the two tested AMD model geometries; it rejects other routing rules or speculative/batched decode. The default path without the multiplier remains native top-8.

## Correctness and work audit

All 24 independent PyTorch frontier cases and all 66 existing page-opening tests passed. Checks cover both model geometries and KV precisions, partial/singleton pages, empty selections and empty state, region/page ties, overflow hashing, oversized-region exclusion, multiple regions per worker, KV append, and repeated graph replay.

The router computes QK scores once. Selected region IDs use one prefix scan and one scatter, with linear packing work; KV is shared in place. A fixed worker count processes only valid selected regions on the device, without sorting, CPU count reads or a top-k cap. Closed-summary value products use the probability-residual precision correction; selected-page attention and residuals retain native FP32 accumulation. The coarse stage still repeats the tile-normalizer reduction for each routing tile, so that metadata arithmetic is quadratic in tile count. Prefill and the existing archive catch-up implementation are preserved. These measurements do not establish subquadratic work for complete execution.

[Selected settings](serving-v2/settings.json), [serving commands](serving-v2/jobs.json), [kernel probes](probe-v3/jobs.json), [frozen cohorts](serving-v2/cohort-manifest.json), [job status and placement](provenance.json), [raw summary](REPORT.json), [changed predictions](changed-examples.json). Earlier tuning phases remain separately recorded and do not replace the selected measurements above. The [superseded eight-worker candidates](serving-v1/superseded.json) include deliberately canceled runs; their partial quality and speed outputs are excluded. Native quality controls from that phase are retained with their original commands and frozen source.

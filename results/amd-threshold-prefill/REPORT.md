# AMD Qwen threshold prefill

Qwen3.8-27B-FP8 weights, BF16 two-tier KV, batch one, MI325X. The opt-in router opens every eligible region whose normalized-query proxy mass exceeds **16/R**, where R counts nonempty regions. Regions larger than 1,024 entries remain coarse. This report measures the equal-per-region rule; it does not measure entry-16 routing.

## Full-model prefill

Each GPU allocation runs top-8 → threshold → top-8 → dense serially. Each variant has one full warmup and three measured runs on identical nonrepeating ProLong prompts. The exact first chunk, scheduler chunks and prefill cache-update schedule remain 16,384 tokens. The existing local window and sink attention are retained. Later remote attention uses 4,096-query scratch tiles. Decode remains native top-8 in every speed run. Timings are first-token latency from scheduling; initialization and compilation are outside the measurements.

| Context | Top-8 before | Threshold | Top-8 after | Change vs native median | Dense | Dense / threshold |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 3.903 s | 4.130 s | 3.909 s | +5.7% | 5.282 s | 1.28× |
| 64K | 8.068 s | 8.990 s | 8.074 s | +11.4% | 14.100 s | 1.57× |

The following completion times include the first decode step. This charges any deferred final prefill cache update to the comparison, since decode waits for its completion. These are two-token request latencies, not steady-state decode measurements.

| Context | Top-8 before | Threshold | Top-8 after | Dense |
|---|---:|---:|---:|---:|
| 32K | 3.931 s | 4.158 s | 3.937 s | 5.312 s |
| 64K | 8.095 s | 9.017 s | 8.102 s | 14.132 s |

## Retrieval check

First 32 frozen examples per task at nominal 64K; identical prompts, token hashes, targets and seed. Both arms use the previously tested precise c=16 decoder. The variable is prefill routing. This is a 64-example retrieval pilot, not a full RULER panel.

| Task | Top-8 prefill | Threshold prefill | Improved | Declined | Actual input tokens |
|---|---:|---:|---:|---:|---|
| niah_multikey_2 | 100.00% | 100.00% | 0 | 0 | 64,920–65,140 |
| niah_multikey_3 | 100.00% | 100.00% | 0 | 0 | 63,515–63,820 |

## Identical-query remote attention

These graph-captured diagnostics replay three actual Qwen layers from the last 16K chunk of the same 64K native-prefill prompt. They include routing, coarse attention, grouping, exact leaves, sink and final merge. They exclude local-attention computation, the rest of the model and cache updates; the full-model table includes those costs. The methods open different amounts of work.

| Layer | Method | Route/coarse | Leaves, packing and reduction | Complete remote call | Regions/head | Entries/head |
|---|---|---:|---:|---:|---:|---:|
| 0 | top8 | 9.01 ms | 3.92 ms | 13.73 ms | 7.87 | 566.5 |
| 0 | blas | 16.10 ms | 16.60 ms | 33.94 ms | 22.41 | 1660.8 |
| 8 | top8 | 9.28 ms | 3.02 ms | 13.51 ms | 8.00 | 318.3 |
| 8 | blas | 16.66 ms | 19.08 ms | 38.05 ms | 31.41 | 1504.1 |
| 15 | top8 | 11.18 ms | 4.47 ms | 16.62 ms | 8.00 | 647.0 |
| 15 | blas | 17.40 ms | 22.08 ms | 41.09 ms | 21.86 | 2088.4 |

## Implementation and validation

A grouped BLAS product computes query–mean-key scores once with FP32 output, reading shared GQA KV storage directly. One row pass normalizes routing, computes closed-summary probabilities and packs selected IDs. The value product uses BF16 probabilities. Bitmasks and integer ranks feed deterministic query-tile counts and prefix scans; packing has linear work and no global counter contention. Fixed persistent leaf workers read the actual GPU work count. The final reduction uses 32/64-route fast paths and a general uncapped fallback. Selection never truncates to eight or sixteen regions. No selected-count CPU readback occurs.

All 14 independent PyTorch frontier checks passed, including empty selections, uneven sizes, eligibility limits, random page ownership, query-tile boundaries, more than 64 opened regions, both D/GQA geometries and graph replay. The native page/hash regressions also passed. Real-state validation independently checks selection and closed coarse attention on sampled actual queries; exact frontier/reference tolerances and raw validation results are retained in the JSON artifacts.

The first exact chunk and local attention are unchanged. Cache initialization and amortized semantic/page updates use the existing code. The complete execution has not been given a new asymptotic bound by this work. The attention score matrix and grouping scans scale with queries × regions; exact work depends on selected region sizes and the existing eligibility limit.

Enable the tested prefill path with `LOD_AMD_PREFILL_THRESHOLD=16`. This option does not enable or change threshold decode. The complete experimental settings are in [settings.json](settings.json). Serving defaults remain unchanged. Model validation here covers BF16 two-tier Qwen at batch one.

## Provenance and exclusions

Final commands, frozen source hashes and outputs: [final-v15/jobs.json](final-v15/jobs.json). All GPU runs use uv through cluster-run. [REPORT.json](REPORT.json) records every submitted experiment, its completion status, and which runs were selected. Failed or cancelled tuning runs are excluded from the tables. These include the initial whole-chunk allocation failures, an incorrect atomic grouping prototype, its debugging checks and superseded routing probes. A fused full-coarse experiment was slower than BLAS and was removed from the final implementation. Wider leaf tile candidates that failed the comparison tolerance were rejected.

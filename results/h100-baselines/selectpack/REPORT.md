# H100 threshold selection and packing optimization

BF16 Llama-3.1-8B-Instruct, batch one, one H100. c=16 per-region routing, the 1,024-entry eligible-region limit, exact initial 16,384-token prefill, local/sink attention and cache updates remain unchanged. No sorting, selection cap or selected-count CPU readback is introduced. Experiments use uv and source-snapshotting cluster-run.

## Selected implementation

The uncapped, per-region fused threshold path now automatically uses **32-query × 256-region tiles, four warps and device prefix scans** from 4,096 active regions onward. Below that point it retains the original full-row pass. Empty packing tiles skip mask unpacking and prefix work. This changes the selection implementation within the opt-in threshold adapter; vLLM serving retains its existing routing defaults.

| Context tokens | Original packing | Optimized packing | Prefill reduction |
|---:|---:|---:|---:|
| 32,768 | 1.911 s | 1.909 s | Unchanged |
| 65,536 | 5.384 s | 5.386 s | Unchanged |
| 128,000 | 14.269 s | 13.437 s | 5.8% |

At 128,000 tokens, full-model selection/packing GPU time falls from **2.886 s to 2.081 s (27.9% less)**. The isolated heavy live fixture improves by 48.5%; that number is not the full-prefill speedup.

The scan reads each saved score once, writes a compact selection mask and tile counts, computes query/expert prefixes, then packs selected pairs from the mask. It introduces no expert atomics. Work is linear in the query–region matrix size; this does not establish a subquadratic bound for complete prefill or amortized cache maintenance.

## Live selection fixture

The probe intercepts real prefill scores in memory and intentionally stops model execution after tuning. It saves no Q/K/V snapshot. These are selection timings, not full-model latency or quality. Each candidate is checked against the original full-row pass: complete selection bits, counts, entries, selected region ordering, expert counts and unique gap-free expert offsets.

Fixture `threshold-next-selectpack-bias-v2-live-probe-128k.json`: scores [32, 4096, 5016], 42.17 mean selected regions/query-head.

| Backend | Query tile | Region tile | Warps | Median | Reduction vs full-row |
|---|---:|---:|---:|---:|---:|
| full-row | — | — | 8 | 5.217 ms | 0.0% |
| scan | 8 | 128 | 4 | 4.041 ms | 22.5% |
| scan | 16 | 128 | 4 | 3.442 ms | 34.0% |
| scan | 32 | 128 | 4 | 4.109 ms | 21.2% |
| scan | 64 | 128 | 4 | 4.347 ms | 16.7% |
| scan | 16 | 256 | 4 | 3.928 ms | 24.7% |
| scan | 32 | 256 | 4 | 3.605 ms | 30.9% |
| bias | — | — | 4 | 4.827 ms | 7.5% |
| bias | — | — | 8 | 4.862 ms | 6.8% |
| bias | — | — | 16 | 4.809 ms | 7.8% |

Fixture `threshold-next-selectpack-guard-v2-live-probe-128k.json`: scores [32, 4096, 5016], 42.17 mean selected regions/query-head.

| Backend | Query tile | Region tile | Warps | Median | Reduction vs full-row |
|---|---:|---:|---:|---:|---:|
| full-row | — | — | 8 | 5.239 ms | 0.0% |
| scan | 8 | 128 | 4 | 3.575 ms | 31.8% |
| scan | 16 | 128 | 4 | 2.928 ms | 44.1% |
| scan | 32 | 128 | 4 | 3.192 ms | 39.1% |
| scan | 64 | 128 | 4 | 3.087 ms | 41.1% |
| scan | 16 | 256 | 4 | 2.991 ms | 42.9% |
| scan | 32 | 256 | 4 | 2.699 ms | 48.5% |
| hybrid | 16 | 128 | 4 | 2.789 ms | 46.8% |
| hybrid | 16 | 64 | 4 | 4.274 ms | 18.4% |
| hybrid | 16 | 128 | 8 | 4.793 ms | 8.5% |
| hybrid | 32 | 128 | 8 | 3.754 ms | 28.3% |
| hybrid | 32 | 256 | 8 | 3.300 ms | 37.0% |
| hybrid | 64 | 128 | 8 | 3.960 ms | 24.4% |
| hybrid | 32 | 256 | 16 | 4.670 ms | 10.9% |
| bias | — | — | 4 | 4.831 ms | 7.8% |
| bias | — | — | 8 | 4.862 ms | 7.2% |
| bias | — | — | 16 | 4.812 ms | 8.1% |

Fixture `threshold-next-selectpack-hybrid-v2-live-probe-128k.json`: scores [32, 4096, 5016], 42.17 mean selected regions/query-head.

| Backend | Query tile | Region tile | Warps | Median | Reduction vs full-row |
|---|---:|---:|---:|---:|---:|
| full-row | — | — | 8 | 5.247 ms | 0.0% |
| scan | 8 | 128 | 4 | 4.042 ms | 23.0% |
| scan | 16 | 128 | 4 | 3.445 ms | 34.3% |
| scan | 32 | 128 | 4 | 4.115 ms | 21.6% |
| scan | 64 | 128 | 4 | 4.347 ms | 17.1% |
| scan | 16 | 256 | 4 | 3.926 ms | 25.2% |
| scan | 32 | 256 | 4 | 3.605 ms | 31.3% |
| hybrid | 16 | 128 | 4 | 3.306 ms | 37.0% |
| hybrid | 32 | 128 | 4 | 4.042 ms | 23.0% |
| hybrid | 32 | 256 | 4 | 3.561 ms | 32.1% |
| bias | — | — | 4 | 4.827 ms | 8.0% |
| bias | — | — | 8 | 4.863 ms | 7.3% |
| bias | — | — | 16 | 4.812 ms | 8.3% |

Fixture `threshold-next-selectpack-v2-live-probe-128k.json`: scores [32, 4096, 5016], 42.17 mean selected regions/query-head.

| Backend | Query tile | Region tile | Warps | Median | Reduction vs full-row |
|---|---:|---:|---:|---:|---:|
| full-row | — | — | 8 | 5.215 ms | 0.0% |
| scan | 8 | 128 | 4 | 4.021 ms | 22.9% |
| scan | 16 | 128 | 4 | 3.438 ms | 34.1% |
| scan | 32 | 128 | 4 | 4.100 ms | 21.4% |
| scan | 64 | 128 | 4 | 4.333 ms | 16.9% |
| scan | 16 | 256 | 4 | 3.914 ms | 24.9% |
| scan | 32 | 256 | 4 | 3.590 ms | 31.2% |

## Full-model prefill

One warmup plus three measured repetitions; median wall time, including final cache completion and excluding model loading, initial cache allocation and graph capture. Every arm uses expandable allocation segments and 8,192-token MLP chunks. Internal attention scratch tiles remain 4,096 queries.

| Paired series | Tokens | Backend | Control | Candidate | Reduction | Sampled output IDs match |
|---|---:|---|---:|---:|---:|---|
| final-v1 | 128,000 | scan | 14.269 s | 13.437 s | 5.8% | True |
| final-v1 | 32,768 | scan | 1.911 s | 1.909 s | 0.1% | True |
| final-v1 | 65,536 | scan | 5.384 s | 5.386 s | -0.0% | True |
| scan-v1 | 128,000 | scan | 14.258 s | 13.703 s | 3.9% | True |
| scan-v1 | 32,768 | scan | 1.910 s | 1.940 s | -1.6% | True |

The two teacher-forced output tokens are only a sampled model check. They do not establish RULER quality. A negative reduction is a regression; microbenchmarks are not substituted for full-model results.

## GPU stage profiles

One measured instrumented repetition after warmup; summed CUDA-event time over all layers. These are diagnostic stage costs.

| Series | Backend | Tokens | Selection/packing | Exact including dispatch/reduction | Coarse producer | Merge |
|---|---|---:|---:|---:|---:|---:|
| final-v1 | control | 128,000 | 2.886 s | 2.777 s | 2.366 s | 0.959 s |
| final-v1 | scan | 128,000 | 2.081 s | 2.785 s | 2.361 s | 0.965 s |
| scan-v1 | control | 128,000 | 2.884 s | 2.773 s | 2.365 s | 0.958 s |
| scan-v1 | scan | 128,000 | 2.314 s | 2.781 s | 2.390 s | 0.965 s |

## Validation and limits

24 synthetic fixture configurations passed, covering query and region tails, empty/oversized regions, concentrated mass, and empty selection.

The scan variant uses tiled score evaluation, query and expert prefix scans, and a second mask-only packing pass. The hybrid instead reserves one expert range per query tile, while keeping query-prefix packing. The bias variant precomputes region-only log-counts and threshold shifts and reuses query RMS from the producer, retaining the original atomic expert packing.

Automatic-dispatch 128,000-token reference check: **32 layers, 224 sampled frontiers**, 3 selection differences (0 outside boundary tolerance), maximum absolute attention-output difference 0.015625. Same-cache eager/CUDA-graph checks also pass. Instrumented verification time is excluded from latency tables.

Paired RULER correctness check: **8/8 first-response scores unchanged**, **6/8 first-response token sequences identical**, and 3/8 complete fixed-length sequences identical for `threshold-next-selectpack-final-v1-scan-ruler-check.json`. The first response ends at the checkpoint's EOS/EOT tokens; later continuation differences are excluded from scoring. First-response differences: cwe (2 examples). This is a two-example-per-task correctness panel, not a new full RULER accuracy estimate.

| Task | Examples | Original first-response score | Scan first-response score |
|---|---:|---:|---:|
| niah_multikey_2 | 2 | 100.00% | 100.00% |
| niah_multikey_3 | 2 | 100.00% | 100.00% |
| cwe | 2 | 45.00% | 45.00% |
| qa_1 | 2 | 100.00% | 100.00% |

## Provenance

Raw outputs, commands, immutable per-run source hashes and logs are retained in this directory. source-objects/ stores content-verified executed sources. SUMMARY.json records the SHA-256 of every analyzed raw output. Failed and partial runs are excluded from all tables.

0 pending jobs; 1 excluded failed jobs.

- `threshold-next-selectpack-v1-live-probe-128k`: exit 2; see its provenance log.

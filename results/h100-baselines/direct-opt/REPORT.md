# Direct-summary prefill optimization

Llama-3.1-8B-Instruct, BF16 model and KV cache, batch one, H100 80GB. Uncapped equal-per-region c=16, 1,024-entry eligibility, exact initial 16,384-token field, local/sink attention and cache completion are fixed. GPU experiments use uv and immutable cluster-run snapshots with a serial GPU lock.

## Implementation

The selected candidate computes final threshold decisions and the unnormalized closed weights from the same saved FP32 score pass. It stores exact selection bits, compacts nonempty bitmap words, and packs only their set bits without rereading routing scores. The score producer no longer builds conservative prefix candidates. Query scaling explicitly rounds before adding log-counts, matching the legacy sparse selector; contracting those operations into an FMA flipped boundary decisions and reduced CWE. The single FP16 coarse projection, FP32 accumulation and division after projection are unchanged. Exact KV attention retains its existing tile and worker settings. No selected-count CPU readback, selection cap or exact-prefill-size change is introduced.

Within the existing opt-in direct scaled-FP16 configuration, the corrected selector is automatic. Set `LOD_H100_DIRECT_FUSE_SELECT=0` to restore the previous selector. Other prefill paths and serving defaults are unchanged.

## Uninstrumented full-model timings

One warmup and three measured repetitions; median wall time includes deferred cache completion, excludes initial cache allocation and graph capture. All arms use 8,192-token MLP chunks and the same frozen prompt at each length. Prefill is not whole-model graph replay. Controls are the completed final-v1 arms; the corrected selector is rerun as final-v2. The correction does not affect the control paths, and their input identities are checked.

| Context tokens | Method | Prefill | Peak PyTorch allocation |
|---:|---|---:|---:|
| 32,768 | Optimized direct FP16 | 1.760 s | 33.43 GiB |
| 32,768 | Dense | 1.562 s | 20.51 GiB |
| 32,768 | Previous direct FP16 | 1.830 s | 33.43 GiB |
| 32,768 | Top-eight | 1.654 s | 24.00 GiB |
| 65,536 | Optimized direct FP16 | 4.748 s | 41.06 GiB |
| 65,536 | Dense | 4.749 s | 25.27 GiB |
| 65,536 | Previous direct FP16 | 5.009 s | 41.06 GiB |
| 65,536 | Top-eight | 4.168 s | 30.12 GiB |
| 128,000 | Optimized direct FP16 | 11.816 s | 66.44 GiB |
| 128,000 | Dense | 15.698 s | 34.36 GiB |
| 128,000 | Previous direct FP16 | 12.268 s | 66.44 GiB |
| 128,000 | Top-eight | 9.531 s | 41.30 GiB |

<details>
<summary>Instrumented 128K trials, including regressions</summary>

One measured repetition after warmup; these diagnose costs and are not final timing medians. Score production/selection/coarse is an aggregate of successive stages, without double-counting their nested timers. For fused-merge trials, the final merge includes exact-route reduction. Top-eight’s final merge is not separately instrumented. Regressing trials are retained as experiments, not promoted implementations.

| Trial | Wall time | Scores + selection + coarse | Exact KV kernel | Separate exact reduction | Final merge | Mean opened regions | Mean exact entries |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline-v1-baseline | 12.339 s | 4.119 s | 2.007 s | 0.684 s | 0.084 s | 22.67 | 1460.5 |
| baseline-v1-top8 | 9.599 s | 3.101 s | 0.717 s | Not separated | Not separated | 7.98 | 522.9 |
| bitmap-split-v1-select (rejected: FMA routing) | 11.804 s | 3.592 s | 2.009 s | 0.687 s | 0.084 s | 22.59 | 1458.8 |
| bitmap-v1-select (rejected: FMA routing) | 11.896 s | 3.674 s | 2.011 s | 0.686 s | 0.084 s | 22.59 | 1459.4 |
| bitmap-v1-combined (rejected: FMA routing) | 11.994 s | 3.654 s | 1.997 s | In final merge | 0.884 s | 22.54 | 1454.1 |
| bitmap-w4-v1-select (rejected: FMA routing) | 11.949 s | 3.734 s | 2.010 s | 0.687 s | 0.084 s | 22.64 | 1457.7 |
| fixed-v2-select (corrected rounding) | 11.891 s | 3.665 s | 2.008 s | 0.688 s | 0.084 s | 22.63 | 1453.1 |
| fuse-hoist-v1-select (rejected: FMA routing) | 14.438 s | 6.239 s | 2.003 s | 0.683 s | 0.084 s | 22.59 | 1459.4 |
| fuse-v1-select (rejected: FMA routing) | 14.418 s | 6.254 s | 1.997 s | 0.682 s | 0.083 s | 22.60 | 1456.2 |
| fuse-v1-hoist | 12.337 s | 4.110 s | 2.010 s | 0.686 s | 0.084 s | 22.67 | 1460.5 |
| leaf128x32-v1-baseline | 12.604 s | 4.114 s | 2.272 s | 0.683 s | 0.084 s | 22.67 | 1459.5 |
| leaf128x64-v1-baseline | 12.813 s | 4.101 s | 2.497 s | 0.679 s | 0.083 s | 22.73 | 1461.3 |
| leaf16x64-v1-baseline | 15.139 s | 4.087 s | 4.867 s | 0.673 s | 0.083 s | 22.61 | 1454.6 |
| leaf32x32-v1-baseline | 13.599 s | 4.112 s | 3.281 s | 0.681 s | 0.084 s | 22.71 | 1459.9 |
| leaf32x64-v1-baseline | 13.276 s | 4.099 s | 2.965 s | 0.677 s | 0.084 s | 22.61 | 1454.5 |
| leaf64-w8-v1-select (rejected: FMA routing) | 13.464 s | 3.655 s | 3.611 s | 0.680 s | 0.084 s | 22.59 | 1459.4 |
| leaf64x64-v1-baseline | 12.762 s | 4.104 s | 2.445 s | 0.680 s | 0.084 s | 22.73 | 1461.5 |
| merge-v1-merge | 12.422 s | 4.111 s | 1.996 s | In final merge | 0.881 s | 22.56 | 1454.9 |
| workers1056-v1-select (rejected: FMA routing) | 11.926 s | 3.672 s | 2.044 s | 0.685 s | 0.084 s | 22.59 | 1459.3 |
| workers528-v1-select (rejected: FMA routing) | 11.925 s | 3.671 s | 2.041 s | 0.685 s | 0.084 s | 22.59 | 1459.3 |

The flat fused selector, hoisted preparation, fused final reducer, and alternative exact tiles/worker counts are diagnostic trials. The selected implementation uses sparse bitmap packing and retains the original exact kernel.

</details>

## Correctness and limited quality checks

Independent frontier checks compare selection and attention against the reference. Same-cache eager/graph decode logits must match exactly. These sample frontiers; they do not validate every prefill token independently.

- `threshold-directopt-fixed-v2-select-32768-verify.json`: 32 sampled frontiers, zero selection differences outside tolerance, exact eager/graph equality; maximum reference error 0.0078125.
- `threshold-directopt-fixed-v2-select-65536-verify.json`: 96 sampled frontiers, zero selection differences outside tolerance, exact eager/graph equality; maximum reference error 0.0078125.
- `threshold-directopt-fixed-v2-select-128000-verify.json`: 224 sampled frontiers, zero selection differences outside tolerance, exact eager/graph equality; maximum reference error 0.0156250.
- `threshold-directopt-winner-v2-select-32768-verify.json`: 32 sampled frontiers, zero selection differences outside tolerance, exact eager/graph equality; maximum reference error 0.0078125.
- `threshold-directopt-winner-v2-select-65536-verify.json`: 96 sampled frontiers, zero selection differences outside tolerance, exact eager/graph equality; maximum reference error 0.0156250.
- `threshold-directopt-winner-v2-select-128000-verify.json`: 224 sampled frontiers, zero selection differences outside tolerance, exact eager/graph equality; maximum reference error 0.0156250.
- `threshold-directopt-rounding-v1-select-0-rounding.json`: 1024 FP32 cases match independent eager PyTorch exactly; the fused-FMA negative control changes 318 results and flips 165 strict decisions.
- `threshold-directopt-same-input-v1-select-32768-numerics.json`: 1 live calls; no differences in producer, selection, coarse or exact outputs.
- `threshold-directopt-trace-v1-select-32768-numerics.json`: 3 live calls; a difference was found.
- `threshold-directopt-trace-v2-select-32768-numerics.json`: 3 live calls; a difference was found.
- `threshold-directopt-trace-v3-select-32768-numerics.json`: 128 live calls; no differences in producer, selection, coarse or exact outputs.

Quality uses the frozen 32K cohort and ends at the first EOS/EOT. CWE is compared against the fresh previous-selector control; retrieval/QA retains the earlier identical-input reference. A CWE task score or six-example retrieval/QA check is not a 13-task macro average. The earlier previous-selector CWE result was 32.81%; the fresh control is 34.38%, with one example changing. That variation is not credited as a quality improvement from the optimization.

| Run | Method | Task | Examples | Score |
|---|---|---|---:|---:|
| threshold-directopt-baseline-cwe-v2-baseline-32768-quality.json | Previous selector control | cwe | 32 | 34.38% |
| threshold-directopt-winner-cwe-v1-select-32768-quality.json | Rejected fused-FMA selector | cwe | 32 | 30.31% |
| threshold-directopt-winner-fixed-cwe-v1-select-32768-quality.json | Corrected optimized selector | cwe | 32 | 34.38% |
| threshold-directopt-winner-fixed-panel-v1-select-32768-quality.json | Corrected optimized selector | niah_multikey_2 | 2 | 100.00% |
| threshold-directopt-winner-fixed-panel-v1-select-32768-quality.json | Corrected optimized selector | niah_multikey_3 | 2 | 100.00% |
| threshold-directopt-winner-fixed-panel-v1-select-32768-quality.json | Corrected optimized selector | qa_1 | 2 | 100.00% |
| threshold-directopt-winner-panel-v1-select-32768-quality.json | Rejected fused-FMA selector | niah_multikey_2 | 2 | 100.00% |
| threshold-directopt-winner-panel-v1-select-32768-quality.json | Rejected fused-FMA selector | niah_multikey_3 | 2 | 100.00% |
| threshold-directopt-winner-panel-v1-select-32768-quality.json | Rejected fused-FMA selector | qa_1 | 2 | 100.00% |

32/32 individual scores unchanged from the previous FP16 candidate.


22/32 individual scores unchanged from the previous FP16 candidate.


30/32 individual scores unchanged from the previous FP16 candidate.


6/6 individual scores unchanged from the previous FP16 candidate.


6/6 individual scores unchanged from the previous FP16 candidate.


## Provenance and status

0 pending; 3 failed. Failed and partial results are excluded. Commands and logs are in provenance/; every executed-source file is saved by verified SHA-256 in source-objects/. SUMMARY.json hashes analyzed inputs. No complete-execution subquadratic bound or amortized decode timing is inferred.

The initial three winner-v1 model checks failed before inference because a configuration import loaded the serving Transformers integration too early. The dispatch setting was moved to a dependency-free module; the winner-v2 checks supersede those attempts. No timing or quality result is inferred from them.

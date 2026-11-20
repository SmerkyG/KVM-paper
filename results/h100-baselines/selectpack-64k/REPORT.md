# H100 threshold prefill at 64K

Llama-3.1-8B-Instruct, BF16, batch one, H100 80GB. The c=16 per-region rule opens every eligible region, with the existing 1,024-entry limit. The exact initial 16,384-token field, local/sink attention, model/MLP chunks and cache updates remain fixed. Experiments use uv and source-snapshotting cluster-run, serially on one GPU.

## Selected result

Fresh matched runs: one warmup and three measured repetitions per method. This is an opt-in threshold-prefill implementation; serving defaults are unchanged.

| Method | 65,536-token prefill |
|---|---:|
| Previous threshold | 5.366 s |
| Selection-only optimization | 5.237 s |
| Direct-closed candidate (experimental) | 4.989 s |
| Dense | 4.750 s |
| Top-eight | 4.159 s |

The direct-closed candidate is **7.0% faster** than the control and **5.0% slower than dense**. The 16K exact initial field and uncapped c=16 selections are retained.

The selection-only implementation is **2.4% faster** than the control and **10.3% slower than dense**. It keeps the previous coarse-attention and correction arithmetic. Within the opt-in fast threshold path, automatic dispatch now uses this prefix filter at 2,048–4,095 active region slots; larger problems retain the optimized prefix-scan packer. Set `LOD_H100_FAST_PREFIX_SELECT=0` to disable the filter.

**Quality limitation:** these are eight-example 32K RULER checks, not full-panel estimates.
Direct-closed candidate: 7/8 first-response scores unchanged; 6/8 first-response token sequences identical.
cwe source row 14: **90% → 70%**. The direct-closed candidate remains experimental.
Selection-only: 8/8 first-response scores unchanged; 6/8 first-response token sequences identical.

The experimental direct-closed path uses conservative prefix filtering, direct closed-summary normalization with shared FP32 log-counts, and cuBLAS BF16-to-FP32 value projection without KV duplication. Producer tiles are 64 queries × 128 regions with four warps; normalization handles one query at a time. Exact leaves retain the existing wide reducer. Full enable flags and commands are in `threshold-select64-final-v1-models-runs.json`.


The [coarse-arithmetic ablation and combination follow-up](COMBINATION.md) extends CWE to all 32 frozen examples: conservative 30.00%, direct-closed 27.19%, online-compatible combination 30.31%. The combination recovers the eight checked scores but takes 5.535 s at 64K; a single-projection version takes 5.210 s and retains the original CWE example's regression. Neither is promoted. These task-specific scores are not a full RULER average.


**Subsequent result:** the [normalization and offset follow-up](../closed-offset/REPORT.md) keeps direct closed-summary attention using one FP16 coarse projection and FP32 normalization afterward: **5.019 s at 64K**, **32.81% CWE recall on all 32 frozen examples at 32K**. Model/KV storage stay BF16. This supersedes the arithmetic combination's speed/quality tradeoff; it is not a full RULER average or a serving-default change.

## Additional automatic-dispatch checks

One warmup and three measured repetitions, using the same frozen prompt manifest and full cache-completion scope. These are additional context checks; dense/top-eight controls were rerun only at 64K in this follow-up.

| Context tokens | Automatic threshold prefill |
|---:|---:|
| 32,768 | 1.899 s |
| 128,000 | 13.246 s |

<details>
<summary>Full tuning history and instrumented stage pilots</summary>

## Full-model timings

One warmup and three measured repetitions; median wall-clock prefill includes deferred LoD cache completion and excludes model loading, initial cache allocation and decode graph capture. MLP chunks are 8,192 tokens for every method; internal attention scratch remains 4,096 queries. All arms use expandable allocation segments.

| Series | Context tokens | Method | Prefill | Peak PyTorch allocation |
|---|---:|---|---:|---:|
| auto-v1 | 128,000 | auto | 13.246 s | 65.51 GiB |
| auto-v1 | 32,768 | auto | 1.899 s | 32.93 GiB |
| combine-log-v1 | 65,536 | prefix | 5.231 s | 40.19 GiB |
| combine-online-single-v1 | 65,536 | prefix-closed-tuned | 5.210 s | 41.06 GiB |
| combine-online-v1 | 65,536 | prefix-closed-tuned | 5.535 s | 41.92 GiB |
| combine-tile-v1 | 65,536 | prefix-tuned | 5.515 s | 40.19 GiB |
| dense-v3 | 65,536 | dense | 4.764 s | 25.27 GiB |
| final-v1 | 65,536 | control | 5.366 s | 40.14 GiB |
| final-v1 | 65,536 | dense | 4.750 s | 25.27 GiB |
| final-v1 | 65,536 | prefix-closed-tuned | 4.989 s | 41.06 GiB |
| final-v1 | 65,536 | top8 | 4.159 s | 30.12 GiB |
| prefix-safe-v1 | 65,536 | prefix | 5.237 s | 40.19 GiB |
| v2 | 65,536 | control | 5.366 s | 40.14 GiB |
| v2 | 65,536 | fused | 5.490 s | 40.14 GiB |
| v2 | 65,536 | fused-tuned | 5.519 s | 40.14 GiB |
| v2 | 65,536 | top8 | 4.168 s | 30.12 GiB |

Full-model results are kept separate from instrumented pilots and isolated kernel probes. Earlier invalid fusion timings are excluded even when the timing job exited successfully.

## Instrumented stage pilots

One measured repetition after warmup, summed CUDA-event time over all layers. These diagnose costs and are not the final timing medians.

| Series | Method | Prefill | Producer | Selection/packing | Closed projection | Exact including dispatch/reduction | Correction/merge |
|---|---|---:|---:|---:|---:|---:|---:|
| bitmap-m16-v3 | bitmap | 5.337 s | 0.791 s | 0.633 s | 0.000 s | 0.880 s | 0.381 s |
| bitmap-m16-v3 | bitmap-fused | 5.381 s | 0.791 s | 0.633 s | 0.000 s | 0.582 s | 0.726 s |
| bitmap-m32-v3 | bitmap-fused | 5.378 s | 0.791 s | 0.632 s | 0.000 s | 0.582 s | 0.725 s |
| bucket-reduce-v3 | prefix-closed-tuned | 5.021 s | 0.462 s | 0.466 s | 0.499 s | 0.909 s | 0.037 s |
| direct-blas-v1 | prefix-closed-tuned | 5.060 s | 0.462 s | 0.465 s | 0.572 s | 0.878 s | 0.037 s |
| direct-tuned-v1 | prefix-closed-tuned | 5.189 s | 0.523 s | 0.447 s | 0.651 s | 0.880 s | 0.037 s |
| direct-v1 | prefix-closed | 5.405 s | 0.728 s | 0.473 s | 0.650 s | 0.877 s | 0.037 s |
| grouped-v1 | prefix-closed-tuned | 5.439 s | 0.461 s | 0.464 s | 0.741 s | 1.085 s | 0.037 s |
| merge-m1-v3 | fused | 5.256 s | 0.770 s | 0.713 s | 0.000 s | 0.583 s | 0.544 s |
| packed-v1 | prefix-closed-tuned | 5.395 s | 0.461 s | 0.465 s | 0.742 s | 1.044 s | 0.037 s |
| prefix-full-v1 | prefix-full | 5.434 s | 0.856 s | 0.678 s | 0.000 s | 0.875 s | 0.380 s |
| prefix-full-v1 | prefix-full-fused | 5.303 s | 0.857 s | 0.679 s | 0.000 s | 0.581 s | 0.541 s |
| prefix-v1 | bucket | 5.500 s | 0.766 s | 0.709 s | 0.000 s | 0.579 s | 0.798 s |
| prefix-v1 | prefix | 5.231 s | 0.857 s | 0.472 s | 0.000 s | 0.878 s | 0.380 s |
| prefix-v1 | prefix-bucket | 5.339 s | 0.856 s | 0.472 s | 0.000 s | 0.579 s | 0.797 s |
| recompute-individual-v3 | recompute-tuned | 5.112 s | 0.239 s | 1.326 s | 0.000 s | 0.872 s | 0.037 s |
| recompute-v1 | recompute-tuned | 6.563 s | 0.237 s | 2.786 s | 0.000 s | 0.866 s | 0.037 s |
| shared-log-m1-v1 | prefix-closed-tuned | 4.993 s | 0.463 s | 0.466 s | 0.499 s | 0.880 s | 0.037 s |
| shared-log-v1 | prefix-closed-tuned | 5.229 s | 0.461 s | 0.465 s | 0.742 s | 0.877 s | 0.037 s |
| v1 | control | 5.388 s | 0.767 s | 0.711 s | 0.000 s | 0.875 s | 0.392 s |
| v1 | scan | 5.429 s | 0.792 s | 0.714 s | 0.000 s | 0.880 s | 0.393 s |
| v3 | fused | 5.428 s | 0.768 s | 0.710 s | 0.000 s | 0.582 s | 0.724 s |

</details>

## Implementations

- Bitmap ranks replace boolean-tile scans with population counts and word prefixes. They preserve complete selection and ascending region order.
- Sparse tile staging packs only selected pairs after tile counts, but reserves extra scratch capacity; it did not outperform the scan on the initial live fixture.
- Fused route merging combines exact-route reduction and coarse-summary replacement. Its rare numerical fallback uses 64-bit route indexing and checks groups of 128 flags before doing repair work. Bucketed merging uses 32/64/128-route fast paths and an uncapped fallback.
- Prefix filtering reuses the producer's running routing partition. Since prefix mass is no greater than final mass, a region that fails the prefix threshold cannot pass the final threshold. A conservative FP32 boundary guard retains extra candidates. The final comparison uses the original saved FP32 score and final normalizer. Packing compacts nonempty bitmap words and reads only their candidate scores. No selection cap or CPU selected-count readback is introduced.
- Direct closed-summary projection omits full-coarse value attention in the producer, then normalizes only unopened summaries and projects their values with shared-GQA Triton or cuBLAS BF16-to-FP32 multiplication. It avoids subtraction and its numerical fallback. The final local/sink merge is unchanged.
- The recomputed-QK prototype omits the saved score matrix and fuses selection/packing with a second QK pass and closed-summary attention. Its timing is retained as a separate pilot, not substituted for the selected implementation.
- Separate query-count reducers compact query IDs into 32/64/128-route buckets and retain a general uncapped reducer. Counts stay on the GPU. The standalone oracle records BF16 output rounding separately from differences against the BF16-rounded PyTorch reference.

These are implementation changes within the experimental threshold path. They do not change the exact initial field or establish subquadratic complete prefill/cache maintenance. The producer still visits every query–region pair; prefix filtering has no improved worst-case candidate bound.

## Correctness

49 successful independent synthetic/merge fixtures are retained. The dedicated cancellation fixture tests 1,560 repaired queries for each merge tile, and separately addresses a route value beyond 2^31 elements.

- `threshold-select64-auto-v1-auto-65536-verify.json`: 96 sampled frontiers across 32 layers, maximum absolute attention-output difference 0.015625, 0 threshold-boundary differences and 0 outside tolerance. Same-cache eager/graph logits match exactly.
- `threshold-select64-combine-log-v1-prefix-65536-verify.json`: 96 sampled frontiers across 32 layers, maximum absolute attention-output difference 0.015625, 0 threshold-boundary differences and 0 outside tolerance. Same-cache eager/graph logits match exactly.
- `threshold-select64-combine-online-single-v1-prefix-closed-tuned-65536-verify.json`: 96 sampled frontiers across 32 layers, maximum absolute attention-output difference 0.007812, 0 threshold-boundary differences and 0 outside tolerance. Same-cache eager/graph logits match exactly.
- `threshold-select64-combine-online-v1-prefix-closed-tuned-65536-verify.json`: 96 sampled frontiers across 32 layers, maximum absolute attention-output difference 0.015625, 0 threshold-boundary differences and 0 outside tolerance. Same-cache eager/graph logits match exactly.
- `threshold-select64-combine-tile-v1-prefix-tuned-65536-verify.json`: 96 sampled frontiers across 32 layers, maximum absolute attention-output difference 0.007812, 0 threshold-boundary differences and 0 outside tolerance. Same-cache eager/graph logits match exactly.
- `threshold-select64-final-v1-prefix-closed-tuned-65536-verify.json`: 96 sampled frontiers across 32 layers, maximum absolute attention-output difference 0.007812, 0 threshold-boundary differences and 0 outside tolerance. Same-cache eager/graph logits match exactly.
- `threshold-select64-prefix-safe-v1-prefix-65536-verify.json`: 96 sampled frontiers across 32 layers, maximum absolute attention-output difference 0.015625, 0 threshold-boundary differences and 0 outside tolerance. Same-cache eager/graph logits match exactly.
- `threshold-select64-v2-combined-65536-verify.json`: 96 sampled frontiers across 32 layers, maximum absolute attention-output difference 0.015625, 0 threshold-boundary differences and 0 outside tolerance. Same-cache eager/graph logits match exactly.
- `threshold-select64-v2-fused-65536-verify.json`: 96 sampled frontiers across 32 layers, maximum absolute attention-output difference 0.015625, 2 threshold-boundary differences and 0 outside tolerance. Same-cache eager/graph logits match exactly.

Paired 32K RULER correctness panel `threshold-select64-final-v1-prefix-closed-tuned-ruler.json`: 7/8 first-response scores unchanged; 6/8 first-response token sequences identical. Two examples/task is a correctness panel, not a new full RULER estimate.

Paired 32K RULER correctness panel `threshold-select64-prefix-safe-v1-prefix-ruler.json`: 8/8 first-response scores unchanged; 6/8 first-response token sequences identical. Two examples/task is a correctness panel, not a new full RULER estimate.

## Live-input kernel probes

These intercept real layer-zero inputs in GPU memory, then intentionally stop execution after validation/tuning. No Q/K/V snapshot is saved. They exclude the rest of the model, exact attention and cache updates. Each selection candidate checks complete masks/counts, ascending selected IDs and unique gap-free expert offsets against the original pass.

- `threshold-select64-bitmap-large-v2-live-65536.json`: see raw fixture/records for every tested geometry and its validation.
- `threshold-select64-bitmap-small-v2-live-65536.json`: see raw fixture/records for every tested geometry and its validation.
- `threshold-select64-bitmap-v2-live-65536.json`: see raw fixture/records for every tested geometry and its validation.
- `threshold-select64-direct-v1-producer-65536.json`: see raw fixture/records for every tested geometry and its validation.
- `threshold-select64-prefix-wide-v1-producer-65536.json`: see raw fixture/records for every tested geometry and its validation.
- `threshold-select64-recompute-tune-v1-recompute-producer-65536.json`: see raw fixture/records for every tested geometry and its validation.
- `threshold-select64-v1-live-65536.json`: see raw fixture/records for every tested geometry and its validation.
- `threshold-select64-v1-producer-65536.json`: see raw fixture/records for every tested geometry and its validation.

## Provenance and exclusions

Raw results and immutable executed-source hashes are retained alongside provenance logs and content-verified source-objects/. SUMMARY.json hashes every analyzed raw output. Failed, partial and known-invalid runs are excluded from timing tables.

0 pending; 13 failed; 1 additional invalid completed timings.

- `threshold-select64-v1-combined-65536-profiles`: exit 1; see its provenance log.
- `threshold-select64-v1-fused-65536-verify`: exit 1; see its provenance log.
- `threshold-select64-bitmap-v1-bitmap-m16-n128-c2`: exit 1; see its provenance log.
- `threshold-select64-bitmap-v1-bitmap-fused-m16-n128-c2`: exit 1; see its provenance log.
- `threshold-select64-v2-dense-65536-models`: exit 1; see its provenance log.
- `threshold-select64-bucket-reduce-v1-reduce-check`: exit 1; see its provenance log.
- `threshold-select64-bucket-reduce-v1-prefix-closed-tuned-65536-profiles`: exit 1; see its provenance log.
- `threshold-select64-bucket-reduce-v2-reduce-check`: exit 1; see its provenance log.
- `threshold-select64-recompute-individual-v2-recompute-tuned-m32-n256-c2`: exit 1; see its provenance log.
- `threshold-select64-recompute-individual-v2-recompute-tuned-m32-n256-c512`: exit 1; see its provenance log.
- `threshold-select64-recompute-individual-v2-recompute-tuned-65536-profiles`: exit 1; see its provenance log.
- `threshold-select64-combine-numerics-v1-coarse-numerics-65536`: exit 2; see its provenance log.
- `threshold-select64-combine-numerics-v2-coarse-numerics-65536`: exit -15; see its provenance log.
- `threshold-select64-v1-fused-65536-profiles.json`: Invalid: later full-model reference check failed; 32-bit fallback indexing overflowed.

The initial fusion exposed 32-bit fallback indexing overflow; its timing is invalid. The first bitmap implementation hit a compiler gather layout assertion; broadcast/reshape fixes it. The initial dense retry incorrectly requested LoD-specific cache completion and is excluded; the corrected dense run uses the native cache.


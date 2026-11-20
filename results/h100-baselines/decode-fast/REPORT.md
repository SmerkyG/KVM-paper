# Faster uncapped H100 threshold decode

BF16 Llama-3.1-8B-Instruct, batch 1, H100 80GB, RetroInfer runtime. All GPU experiments use uv and the self-documenting cluster-run adapter. Threshold rules and selected-region budgets are unchanged: there is no cap of eight or sixteen regions.

The per-region rule opens every eligible region with proxy mass greater than 16 times the average; its existing 1,024-entry eligibility limit remains. Entry-16 opens regions with proxy mass per entry greater than 1/(16 sqrt(n)), with the existing deterministic strict entry-work guard. Routing still uses the same normalized-query scores and BF16 mean arithmetic.

## Complete model, including amortized cache updates

Every timing uses complete-model CUDA graphs, 512 untimed decode steps and 1,024 measured steps, with context growing throughout. The LoD rows retain top-eight sparse prefill with the unchanged 16,384-token exact initial chunk; dense controls use dense prefill. Cache catch-up remains included for LoD. One warmup repetition precedes two measured repetitions. Values are medians; initial prefill, allocation and graph capture are excluded from decode time.

### Dense and top-eight controls

Fresh dense and top-eight controls use the same H100, checkpoint, BF16 precision, batch, replayed prompt tokens, graph policy and 512+1,024-step decode window. Their shared source files exactly match the threshold runs; the only added file is the submission helper. All LoD measured repetitions include 128 layer cache-update events.

| Method | 32K decode, ms/token | 64K decode, ms/token |
|---|---:|---:|
| Dense | [8.706](threshold-decode-baselines-dense-32k.json) | [10.100](threshold-decode-baselines-dense-64k.json) |
| Two-tier top-8 | [8.321](threshold-decode-baselines-top8-32k.json) | [8.704](threshold-decode-baselines-top8-64k.json) |
| Optimized per-region c=16 | [7.541](threshold-decode-v4-region-fast-32k.json) | [7.952](threshold-decode-v4-region-fast-64k.json) |
| Optimized entry-16 | [7.573](threshold-decode-v4-entry-fast-32k.json) | [7.960](threshold-decode-v4-entry-fast-64k.json) |

| Method | 32K prefill, seconds | 64K prefill, seconds |
|---|---:|---:|
| Dense | 1.568 | 4.784 |
| Two-tier top-8 | 1.654 | 4.182 |
| Optimized per-region c=16 | 1.656 | 4.200 |
| Optimized entry-16 | 1.658 | 4.198 |

Dense uses dense prefill. All LoD rows retain top-eight sparse prefill and the same exact initial chunk; threshold changes affect decode only here. Decode excludes prefill and capture, but includes normal KV appends and LoD cache catch-up. This is a timing comparison, not a quality-matched comparison.

### Previous versus optimized thresholds

| Initial context | Routing rule | Prior ms/token | Optimized ms/token | Latency reduction | Layer updates per measured window |
|---|---|---:|---:|---:|---|
| 32K | Per-region c=16 | [7.817](threshold-decode-v4-region-legacy-32k.json) | [7.541](threshold-decode-v4-region-fast-32k.json) | 3.5% | [128, 128] |
| 32K | Entry-16 | [7.994](threshold-decode-v4-entry-legacy-32k.json) | [7.573](threshold-decode-v4-entry-fast-32k.json) | 5.3% | [128, 128] |
| 64K | Per-region c=16 | [8.297](threshold-decode-v4-region-legacy-64k.json) | [7.952](threshold-decode-v4-region-fast-64k.json) | 4.2% | [128, 128] |
| 64K | Entry-16 | [8.251](threshold-decode-v4-entry-legacy-64k.json) | [7.960](threshold-decode-v4-entry-fast-64k.json) | 3.5% | [128, 128] |

## What changed

The optimized router fuses thresholding, region compaction and entry offsets into one prefix-scan kernel. Exact workers then traverse actual selected entries, finding each entry through its compact region offsets and the existing page directory. This avoids the separate page-descriptor expansion and skips padding in short pages. Coarse attention still directly excludes selected summaries, so this path does not use full-coarse subtraction or its cancellation fallback.

The indexed variant retains the existing local/current/sink branch and uses six kernel launches per attention call instead of nine. A combined variant includes those exact entries in the compact exact workers, using five launches. There is no host readback of selected counts, no new GQA expansion, and no change to the cache update schedule. Both reuse the original score producer; no extra score simplification is applied.

| Rule and context | Optimized implementation | Exact worker splits |
|---|---|---:|
| region, 32K | indexed | 64 |
| entry, 32K | combined | 16 |
| region, 64K | indexed | 64 |
| entry, 64K | indexed | 32 |

## Fixed-query diagnostics

These graph timings cover a complete attention call plus the same local-length rollback, on identical real queries/KV states from layers 0, 15 and 31. They omit model layers and cache maintenance. The table reports arithmetic means of the three per-layer medians, not full-model latency. Each median has three event-timed repetitions of 40 graph replays with 16 calls per replay.

| Context | Rule | Prior attention, us | Selected optimized attention, us | Reduction |
|---|---|---:|---:|---:|
| 32K | region | 42.79 | 35.47 | 17.1% |
| 32K | entry | 49.40 | 32.27 | 34.7% |
| 64K | region | 61.64 | 50.42 | 18.2% |
| 64K | entry | 59.57 | 44.57 | 25.2% |

Packing fusion alone gave little benefit and regressed the 64K probe: the large single-head prefix scan offset the saved launches. Compact entry attention produced the useful gain. Combining local attention helped entry-16 at 32K but was slower than retaining its separate kernel for the other chosen configurations.

## Correctness and scope

| Optimized graph versus legacy eager on identical cache | Checked steps | Different next-token choices | Maximum logit difference |
|---|---:|---:|---:|
| region, 32K | 17 | 0 | 0.476562 |
| entry, 32K | 17 | 0 | 0.406250 |
| region, 64K | 17 | 0 | 1.191406 |
| entry, 64K | 17 | 0 | 0.375000 |

The independent PyTorch frontier suite passed all eleven cases for each implementation, covering uneven sizes, empty selections, regions above 1,024 entries in entry mode, entry multiples 16/24/32, the strict work guard, and local/sink append. Maximum component error was 0.000482. All 144 real-query tuning checks selected the same regions as legacy; maximum attention component difference was 0.000488. Whole-model checks include sampled steps around cache updates and compare the optimized graph to the legacy implementation on the same cache, rather than comparing unrelated generated trajectories.

| Paired complete-model repetition | Predicted-token disagreements, warmup / measured 1 / measured 2 |
|---|---|
| region, 32K | [0, 0, 0] out of 1,537 per repetition |
| entry, 32K | [0, 0, 0] out of 1,537 per repetition |
| region, 64K | [1, 1, 0] out of 1,537 per repetition |
| entry, 64K | [0, 0, 0] out of 1,537 per repetition |

These are implementation and limited next-token checks. The full RULER quality panel was not rerun. Attention reductions differ at floating-point precision; those differences can propagate through later layers and threshold decisions. Full-model logits are not bitwise equivalent: the largest sampled difference was 1.191406 in the 64K per-region check, and its paired timing runs had one predicted-token disagreement in two repetitions. Quality equivalence has not been established by these speed tests. This work does not add a proof of subquadratic complete prefill or cache maintenance.

The optimization is opt-in through `LOD_H100_THRESHOLD_IMPLEMENTATION=indexed` or `combined`, with `LOD_H100_THRESHOLD_SPLITS` as listed above. Existing threshold enable/rule flags remain required. Serving defaults remain unchanged.

Failed preliminary probes are retained: the first tuning harness used the wrong integer type in reporting, and the initial full-model validation helper lacked a local import. Dependent timings were skipped after those failures; failed or partial trials are excluded here. All final model comparisons share identical source hashes and GPU identity. Raw commands, logs, timings and validation remain in the JSON files and provenance directory.

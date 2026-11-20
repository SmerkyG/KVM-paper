# AMD-derived three-tier threshold decode on H100

Llama-3.1-8B-Instruct, BF16 weights, batch one, H100 80GB HBM3, RetroInfer runtime. All GPU experiments use uv and self-documenting cluster-run snapshots. These runs use the corrected cache-update kernel.

## Complete model, including amortized cache updates

Latency in ms/token. All arms graph the complete model decode step. Each run has one warmup repetition and two measured repetitions; each repetition has 512 untimed steps followed by 1,024 timed steps. Top-eight decode rows average the medians of fresh controls before and after the candidate; the separate controls remain in SUMMARY.json. LoD timed windows each include 128 layer cache-update events, including the cached-mean refresh when enabled. Prompts and teacher-forced decode inputs are identical across arms; fresh dense controls use the same GPU.

| Context | Dense | Top-8 BF16 | Threshold BF16 | Top-8 INT4 | Threshold INT4 |
|---|---:|---:|---:|---:|---:|
| 32K | 8.725 | 7.828 | 7.715 | 7.780 | 7.562 |
| 64K | 10.087 | 8.268 | 7.943 | 8.122 | 7.876 |

The threshold rule is normalized-query proxy mass > 16 / number of nonempty regions. Regions above the existing 1,024-entry eligibility limit stay summarized. Region counts are uncapped. Each opened region contributes its top two exact 16-token pages plus one disjoint residual summary. This differs from opening the fixed top eight regions; it is a routing-policy comparison as well as a performance comparison.

## Prefill and maintenance

Full-model prefill in seconds, including explicit final cache catch-up. Prefill retains top-eight region routing and opens all leaves in those regions. Its exact initial 16,384-token chunk, local attention and sink attention are unchanged. Top-two page selection applies to decode. Cache allocation and graph capture are outside prefill and decode timings, consistently with the existing harness; raw records retain their separate costs.

| Context | Dense | Top-8 BF16 | Threshold BF16 | Top-8 INT4 | Threshold INT4 |
|---|---:|---:|---:|---:|---:|
| 32K | 1.557 | 1.770 | 1.741 | 1.910 | 1.907 |
| 64K | 4.777 | 4.458 | 4.479 | 4.897 | 4.884 |

Persistent BF16-rounded region means are refreshed at the serving pool's existing install/catch-up boundaries. No leaf-index rebuild, exact-KV copy, GQA expansion, selected-count host readback or selected-region cap is added. The uncached controls in SUMMARY.json use the same selected settings and frontier and include the same amortized state updates.

The following full-model ms/token comparison isolates mean caching: region selections, routing tiles, page workers, precision and all other attention settings match between each pair. The cost of maintaining the cached means is included.

| Context | KV | Threshold without caching | Threshold with caching | Reduction |
|---|---|---:|---:|---:|
| 32K | BF16 | 7.761 | 7.715 | 0.6% |
| 32K | INT4 | 7.599 | 7.562 | 0.5% |
| 64K | BF16 | 8.025 | 7.943 | 1.0% |
| 64K | INT4 | 8.010 | 7.876 | 1.7% |

## Validation and retrieval

The independent PyTorch frontier suite passed 24 BF16/INT4 cases covering uneven region sizes, overflow page tables, partial pages, selections opening every eligible region or none, empty remote context, ties, graph replay and current-KV appends. Both cached and uncached H100 arithmetic were checked. Real-query tuning uses layers 0, 15 and 31; every threshold candidate selects the same regions as the uncached control. Complete-model graph checks compare cached graph logits with uncached eager logits on the identical cache, including steps around cache updates. All sampled logits agree exactly.

The limited 32K retrieval panel uses 16 examples per task, 32 examples per precision, with greedy generation and the existing task token limits. Top-eight controls were rerun. Dense and RetroInfer scores reuse matching examples from the frozen panels; token hashes and answers are checked. Within each precision, the initial prefill prediction matches between top-eight and threshold arms on every example. This is not a 13-task RULER average or a 64K quality evaluation.

### First response

These scores stop at the first EOS/EOT token, matching the main 13-task quality comparisons. The saved generations were rescored on CPU; `first-response-audit.json` records input and tokenizer hashes, per-example scores and stop positions, with the analysis in `first-response-audit.py`.

| Retrieval task | Examples | Dense | RetroInfer | Top-8 BF16 | Threshold BF16 | Top-8 INT4 | Threshold INT4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_3 | 16 | 100.00 | 100.00 | 43.75 | 93.75 | 31.25 | 87.50 |
| niah_multivalue | 16 | 98.44 | 98.44 | 100.00 | 100.00 | 100.00 | 100.00 |

BF16 improves eight examples with no regressions; INT4 improves ten and regresses one. The earlier INT4 multikey score of 93.75% counted all generated text; first-response accuracy is 87.50%. The BF16 top-eight baseline similarly changes from 50.00% over full output to 43.75% on the first response. Full-output scores remain below for transparency.

### Full fixed-length output

These scores can award credit for text generated after the first EOS/EOT token.

| Retrieval task | Examples | Dense | Top-8 BF16 | Threshold BF16 | Top-8 INT4 | Threshold INT4 |
|---|---:|---:|---:|---:|---:|---:|
| niah_multikey_3 | 16 | 100.00 | 50.00 | 93.75 | 31.25 | 93.75 |
| niah_multivalue | 16 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

For full-output scoring, BF16 improves seven examples and regresses none. INT4 improves eleven examples and regresses one (`niah_multikey_3`, source row 187), for a net gain of ten. BF16's remaining full-output miss is source row 59. Both source rows were rerun in both precisions without mean caching; every generated token matches the corresponding cached run. Mean caching leaves these mistakes and the INT4 regression unchanged. The full panel and this single regression are retained explicitly; thresholds remain opt-in.

## Implementation and reproduction

`benchmarks/h100_threshold_three_tier.py` adapts the shared AMD threshold producer and page consumer to Llama GQA (32 query / 8 KV heads, dimension 128). The producer computes native and RMS-normalized proxy logits, combines tile normalizers, selects regions while computing closed-summary attention with a probability-residual precision correction, and packs region IDs with prefix scans. Persistent page workers consume device-resident actual counts. A dimension-blocked merge combines coarse, page/residual and local/sink fields and appends the current KV exactly once.

Enable `LOD_H100_THRESHOLD_MULTIPLIER=16`, `LOD_H100_THRESHOLD_IMPLEMENTATION=three-tier`, and `LOD_H100_THREE_TIER_CACHED_MEANS=1` before creating the cache. Use `three-tier-bf16` or `three-tier-int4` mode. The measured tile/worker/warp/merge settings are in SUMMARY.json and three-tier-v2-model-runs.json. They correspond to `LOD_H100_THRESHOLD_TILE`, `LOD_H100_THREE_TIER_WORKERS`, `LOD_H100_THREE_TIER_PAGE_WARPS`, and `LOD_H100_THRESHOLD_MERGE_BLOCK`. This path supports batch one and per-region thresholds. Serving defaults are unchanged.

The threshold page-summary scan uses `engine.leaf_block_n`: 32 pages for BF16 and 16 for INT4. The raw `lod_route_geometry.page_block` field records the unchanged production `decode_block_n` attribute. The batch adapter explicitly rejects H100 threshold settings; its configuration guard was checked before GPU setup and is outside the timed batch-one path.

Fixed-query graph timings cover the complete attention call plus the same local-length rollback for every arm; they exclude model layers and cache maintenance. Their raw tuning trials and selected averages are kept separately from complete-model results.

Attention-call latency below is in microseconds, averaged over real queries from layers 0, 15 and 31. The uncached column uses the original 64-region-tile / 64-worker setting; the cached column also uses its selected tuning settings. Full-model uncached controls instead use exactly the candidate settings to isolate mean caching. Raw `mean_opened_region_entries` describes the total sizes of selected regions; exact remote attention reads at most two 16-token pages per selected region.

| Context | KV | Top-eight | Threshold, uncached | Threshold, cached and tuned |
|---|---|---:|---:|---:|
| 32K | BF16 | 37.17 | 37.57 | 36.84 |
| 32K | INT4 | 36.22 | 33.56 | 32.89 |
| 64K | BF16 | 51.86 | 47.09 | 45.20 |
| 64K | INT4 | 47.82 | 45.63 | 43.24 |

Packing uses linear prefix scans. The current coarse producer combines all tile normalizers in each coarse tile, so that normalization step still has a quadratic-in-tile-count term. Prefill and existing cache construction/update paths have no new subquadratic-work guarantee. This report claims measured latency only.

Commands, runtime flags, GPU identity, exit states and source hashes are retained in provenance/. The report audits identical executed LoD/adapter code and hardware across final arms and verifies the collected source/ snapshot. The first frontier attempt failed on the old Transformers import facade; unfinished initial tuning runs were superseded to expand page-worker counts. Failed/cancelled/partial trials remain recorded and are excluded from the final comparison.

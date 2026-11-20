# AMD-derived two-tier threshold decode on H100

Llama-3.1-8B-Instruct, BF16 weights and KV, batch one, RetroInfer runtime, H100 80GB HBM3. Both arms use the same normalized-query per-region c=16 threshold, including the existing 1,024-entry eligibility limit; selections remain uncapped. All GPU experiments use uv and cluster-run.

## Full model with amortized cache updates

Complete-model CUDA graphs, 512 untimed decode steps followed by 1,024 measured steps, one warmup repetition and two measured repetitions per run. Each candidate is bracketed by fresh controls on the same GPU. Decode time includes normal KV appends, cache catch-up, persistent-index rebuilds and mean refreshes. Each measured repetition includes 128 layer cache-update events. Initial prefill and graph capture are outside decode time.

| Initial context | Current optimized H100 | AMD-derived pipeline | Latency reduction |
|---|---:|---:|---:|
| 32K | 7.529 ms/token | 7.236 ms/token | 3.9% |
| 64K | 7.950 ms/token | 7.504 ms/token | 5.6% |

Both arms retain top-eight sparse prefill and the unchanged 16,384-token exact initial chunk. The candidate also prepares its persistent index during prefill; this cost is included below.

| Context | Control prefill before / after | Candidate prefill | Control decode before / after |
|---|---:|---:|---:|
| 32K | 1.651 / 1.655 s | 1.655 s | 7.529 / 7.527 ms/token |
| 64K | 4.184 / 4.195 s | 4.192 s | 7.950 / 7.952 ms/token |

## Implementation

The selected implementation reuses the AMD materialized four-stage pipeline: (1) shared-GQA proxy scores and partial normalizers; (2) threshold selection, closed-summary attention and per-tile union packing; (3) compact physical indices plus per-head membership bits; (4) tensor-core attention over that frontier, including local/current/sink entries, followed by a dimension-blocked final merge. K/V storage is shared directly; only indices and membership bits are compacted.

Region means and the region-ordered leaf index are maintained at normal cache-update boundaries. The H100 adapter explicitly retains the old BF16-rounded count division for means. No selected-count CPU readback, selected-region cap, GQA expansion or smaller exact-prefill chunk is introduced. The entry-16 rule is outside this candidate; it continues to use its existing implementation.

## Checks and fixed-query diagnostics

The independent PyTorch frontier suite passed six per-region cases, including empty selections, uneven region sizes, all-open stress and the 1,024-entry boundary. The winning settings were selected from identical-query graph measurements on layers 0, 15 and 31. These diagnostics time the complete attention call plus identical local-length rollback and exclude model layers and cache updates.

All 84 real-query comparisons selected the same regions as the indexed control. In the raw diagnostics, `mean_union_entries` includes local/current/sink entries for the materialized pipeline; the earlier direct shared-entry prototype records remote entries only. Their complete attention timings both include local and sink attention.

| Context | Current attention | Selected candidate attention | Sampled model steps | Different next-token choices | Maximum logit difference |
|---|---:|---:|---:|---:|---:|
| 32K | 35.49 µs | 24.01 µs | 17 | 0 | 1.875000 |
| 64K | 49.96 µs | 30.36 µs | 17 | 0 | 0.421875 |

Checks compare the candidate graph to legacy eager decode on the identical cache, including steps around cache updates. Floating-point reductions differ. These component and next-token checks do not establish full RULER quality equivalence; RULER has not been rerun.

| Context | Predicted-token disagreements vs control, warmup / measured 1 / measured 2 |
|---|---|
| 32K | [0, 0, 0] out of 1,537 per repetition |
| 64K | [1, 1, 0] out of 1,537 per repetition |

## Reproduction and limits

Enable `LOD_H100_THRESHOLD_IMPLEMENTATION=materialized` before cache allocation, together with `LOD_H100_THRESHOLD_MULTIPLIER=16` and `LOD_H100_THRESHOLD_RULE=region`. Exact measured settings, commands, GPU identity and source hashes are in `threshold-transfer-v1-runs.json`, `SUMMARY.json` and `provenance/`. The reusable operators retain their `LOD_AMD_THRESHOLD_*` tuning options; the measured runs explicitly set them. Serving defaults remain unchanged.

Both lengths select 32 exact-worker splits, 128-entry attention tiles, four attention warps, 32-dimension merge blocks and the probability-residual precision correction. Set `LOD_H100_THRESHOLD_SPLITS=32`, `LOD_AMD_THRESHOLD_ENTRY_BLOCK=128`, `LOD_AMD_THRESHOLD_ENTRY_WARPS=4`, `LOD_AMD_THRESHOLD_MERGE_BLOCK=32` and `LOD_AMD_THRESHOLD_PRECISE=1` with the enable flags above. The physical-index compactor uses its tested defaults of four shards and one warp. In raw full-model JSON, `decode_threshold.entry_block` describes the indexed implementation and the `shared_*` fields describe the direct shared-entry prototype; the materialized implementation reads the `LOD_AMD_THRESHOLD_*` values recorded in each run command.

Per-step packing uses linear prefix scans with fixed worker counts. The persistent index refresh visits the remote cache at update boundaries. Partial routing normalizers are currently combined in each coarse tile; complete prefill, routing and cache maintenance have no new subquadratic complexity guarantee. Timings report measured costs, including maintenance.

A direct port of shared exact-entry attention alone was approximately tied at 32K and 64K and slower for entry-16. Dimension-blocking the old final merge alone gave only a small improvement. The measured winner combines cached means, the persistent index, fused tiled selection and shared tensor-core attention. The first materialized fixture run failed on the old Transformers import facade; its dependent probes were skipped. The corrected fixture uses the existing RetroInfer namespace adapter. Failed and partial runs are retained and excluded from comparisons.

All final full-model controls, candidates and checks use identical frozen source files and GPU identity. `source/` is validated against the recorded hashes. Raw runs and all tuning trials remain in this directory.

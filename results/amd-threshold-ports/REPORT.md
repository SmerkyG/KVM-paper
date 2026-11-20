# AMD LoD threshold ports on MI325X

The vLLM threshold path accepts ordinary one-token requests at batch one and batch eight. The kernels use a GPU-resident map into persistent per-request KV storage. Three-tier routing supports fixed page budgets in BF16 and INT4 KV and adaptive pages in BF16 KV.

## 64K RULER retrieval checks

Each row uses the same eight frozen prompts per task for that model. The checkpoints have FP8 weights. Every row below uses BF16 KV. Except for the prefill comparisons, the runs use top-eight prefill. Three-tier region routing uses c=16. Adaptive pages use c=0.5 and RMS-normalized query scores.

| Model | Prefill and decode | Multikey-2 | Multikey-3 |
|---|---|---:|---:|
| Qwen3.8 | [Top-eight prefill, two-tier region c=16](initial-v1/amd-port-qwen-two-region.json) | 100.00% | 100.00% |
| Qwen3.8 | [Top-eight prefill, two-tier entry multiplier 16](initial-v1/amd-port-qwen-two-entry.json) | 100.00% | 100.00% |
| K2 Horizon | [Top-eight prefill, two-tier region c=16](initial-v1/amd-port-k2-two-region.json) | 87.50% | 100.00% |
| K2 Horizon | [Top-eight prefill, two-tier entry multiplier 16](initial-v1/amd-port-k2-two-entry.json) | 75.00% | 87.50% |
| K2 Horizon | [Fixed one page](initial-v1/amd-port-k2-page-1.json) | 75.00% | 62.50% |
| K2 Horizon | [Fixed two pages](initial-v1/amd-port-k2-page-2.json) | 87.50% | 100.00% |
| K2 Horizon | [Fixed four pages](initial-v1/amd-port-k2-page-4.json) | 87.50% | 100.00% |
| K2 Horizon | [Adaptive BF16 pages, four warps](initial-v1/amd-port-k2-page-adaptive.json) | 87.50% | 100.00% |
| K2 Horizon | [Adaptive BF16 pages, two warps](tune-audit-v1/amd-port-k2-page-adaptive-w2.json) | 87.50% | 100.00% |
| K2 Horizon | [Region threshold prefill, two-tier region c=16](initial-v1/amd-port-k2-region-prefill.json) | 100.00% | 100.00% |
| K2 Horizon | [Region threshold prefill with scaled FP16, MMA reduction and packed max, two-tier region c=16](prefill-v1/amd-port-k2-opt-prefill.json) | 100.00% | 100.00% |
| K2 Horizon | Entry threshold prefill, two-tier region c=16 ([multikey-2](entry-memory-v2/amd-port-k2-entry-prefill-tiled.json), [multikey-3](entry-memory-v2/amd-port-k2-entry-prefill-mk3.json)) | 100.00% | 87.50% |

Per-entry prefill uses a strict remote exact-entry bound with multiplier 16. The first K2 launch exceeded device memory in padded exact-route scratch. The successful launch bounds internal query tiles by scratch size and keeps the 16,384-token initial exact field. It remains much slower than region threshold prefill on this cohort. The original tiled job was stopped after finishing multikey-2. The independent multikey-3 shard ran to completion.

The Qwen3.8 entry-prefill smoke passed 4/4 prompts across the same two tasks. [Qwen entry prefill](final-shapes-v2/amd-port-qwen-entry-prefill-smoke.json).

## Batch-eight end-to-end decode

K2 Horizon at 32,768 input tokens, eight simultaneous requests on one MI325X, and 1,024 timed decode steps. Timings are the median of two measured repetitions after warmup. The prefill policy is top-eight across this table. Full serving timings include the normal cache update schedule.

| Threshold decode | ms/batch step | Tokens/s across batch |
|---|---:|---:|
| [Two-tier region, legacy](serving-v1/amd-port-speed-k2-two-reference.json) | 60.55 | 132.1 |
| [Two-tier region, batched materialized](serving-v1/amd-port-speed-k2-two-materialized.json) | 50.64 | 158.0 |
| [Three-tier BF16, fixed two pages](serving-v1/amd-port-speed-k2-pages-fixed.json) | 55.89 | 143.1 |
| [Three-tier INT4, fixed two pages](serving-v1/amd-port-speed-k2-pages-int4.json) | 55.42 | 144.3 |
| [Three-tier BF16 adaptive, four warps](serving-v1/amd-port-speed-k2-pages-adapt-w4.json) | 58.60 | 136.5 |
| [Three-tier BF16 adaptive, two warps](serving-v1/amd-port-speed-k2-pages-adapt-w2.json) | 55.11 | 145.2 |

Adaptive c=0.5 now defaults to two warps for K2 geometry. The two- and four-warp launches scored the same on the matched retrieval cohort. [Default graph check](tuned-default-v1/amd-port-final-adaptive-default-b8.json) · [Default frontier check](tuned-default-v1/amd-port-final-adaptive-default-frontier.json).

The batch-eight timing runs showed token differences across repeated generations. The raw files record first mismatch positions. These are timings, and retrieval quality is measured separately with the frozen batch-one cohort.

The existing top-eight serving path also showed repeated-generation differences at batch eight. [Repeatability control](tune-audit-v1/amd-port-k2-top8-repeatability.json).

## Threshold prefill

K2 Horizon, 65,536 tokens, batch one, three measured repetitions after warmup. The two-token generation keeps this comparison focused on prefill. The context, model and BF16 KV are matched. The region arms use c=16 and the entry arm uses multiplier 16 with the deterministic work bound.

| Prefill path | Seconds |
|---|---:|
| [BF16 coarse projection, bucket reducer](prefill-speed-v1/amd-port-k2-prefill-speed-bf16-bucket.json) | 21.674 |
| [MMA frontier reducer](prefill-speed-v1/amd-port-k2-prefill-speed-tc.json) | 20.559 |
| [Packed state max/index](prefill-speed-v1/amd-port-k2-prefill-speed-packed.json) | 21.902 |
| [MMA reducer and packed max/index](prefill-speed-v1/amd-port-k2-prefill-speed-tc-packed.json) | 20.389 |
| [Per-entry prefill with memory-bounded query tiles](prefill-speed-v1/amd-port-k2-prefill-speed-entry.json) | 47.226 |

Fixed-query replay on three saved Qwen3.8 layers. Median of the three layer timings in milliseconds. This scope includes remote attention and excludes full-model prefill and cache updates. [Per-layer timings and output checks](opt-profile-v2/amd-port-prefill-real-profile-v2.json).

| Remote path | Route and coarse | Exact leaves and reduction | Remote attention |
|---|---:|---:|---:|
| BF16 and bucket | 16.49 | 18.06 | 35.63 |
| Scaled FP16 | 17.00 | 18.08 | 36.00 |
| MMA reduction | 16.44 | 16.12 | 33.92 |
| Scaled FP16 and MMA reduction | 16.94 | 16.12 | 34.40 |

The packed BF16 state max/index was checked against PyTorch for ties, signed zeros, infinities and NaNs. For a 4,096 by 2,880 score matrix the packed kernel took 7.73 µs versus 16.17 µs. The AMD guard uses the packed kernel up through 4,096 columns. [Maximum/index checks](final-check-v1/amd-port-state-max-guarded.json).

## Correctness and reproduction

- Fixed [one](initial-v1/amd-port-fixed1-verified.json) and [four](initial-v1/amd-port-fixed4-verified.json) pages match a PyTorch frontier on Qwen and K2 geometry in BF16 and INT4. [Adaptive BF16](initial-v1/amd-port-adaptive-verified.json) also matches the independent page and entry counters.
- Graph replays at batch eight compare with independent batch-one requests on both head geometries, including changed row maps, mixed active lengths and verification of current-token KV writes. [Region](batch-v1/amd-port-b8-two.json), [entry](batch-v1/amd-port-b8-entry.json), [materialized](final-shapes-v2/amd-port-b8-materialized-shapes-v2.json), [fixed BF16](batch-v1/amd-port-b8-three-bf16.json), [fixed INT4](final-shapes-v2/amd-port-b8-int4-shapes-v2.json), [adaptive two warps](final-shapes-v2/amd-port-b8-adaptive-w2-shapes-v2.json) and [adaptive four warps](final-shapes-v2/amd-port-b8-adaptive-shapes-v2.json).
- [Entry prefill](prefill-v1/amd-port-prefill-entry-check.json) and the [tight-bound variant](final-check-v1/amd-port-prefill-entry-tight-bound.json) match PyTorch selection and output. [Scaled FP16 and MMA reduction](prefill-v1/amd-port-prefill-tc-check.json) also passed the independent fixture.

GPU runs use `uv` and `cluster-run`. Each result has a sibling `-submission.json` with the full command and cluster run ID. Each phase retains its source snapshot and SHA-256 manifest.

| Setting | Environment |
|---|---|
| Region threshold decode | `LOD_AMD_THRESHOLD_MULTIPLIER=16` |
| Fixed recursive page budget | `LOD_AMD_THRESHOLD_PAGES=1`, `2`, or `4` |
| Adaptive recursive BF16 pages | `LOD_AMD_THRESHOLD_PAGE_MULTIPLIER=0.5` |
| Timed K2 batched materialized region routing | `LOD_AMD_THRESHOLD_IMPLEMENTATION=materialized LOD_AMD_THRESHOLD_SPLITS=128 LOD_AMD_THRESHOLD_ENTRY_WARPS=4 LOD_AMD_THRESHOLD_PRECISE=1` |
| Per-entry prefill | `LOD_AMD_PREFILL_THRESHOLD=16 LOD_AMD_PREFILL_RULE=entry` |
| Optional prefill fast paths | `LOD_AMD_PREFILL_REDUCE=tc`, `LOD_AMD_STATE_PACKED_MAX=1` |
| Scaled FP16 coarse projection | `LOD_AMD_PREFILL_PROBABILITY=scaled-fp16` |

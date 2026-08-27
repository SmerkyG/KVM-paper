# INT8 on the current best two- and three-tier paths

This directory records the 2026-08-27 batch-eight precision comparison on
gfx942. Every speed pair uses the same eight distinct 64K ProLong prompts,
16K scheduler chunks, 64 generated tokens, three warm repetitions, automatic
routing geometry, and the current kernel dispatch. Times are matched medians;
cache is the plugin-reported persistent LOD cache and excludes decode scratch.

## Result

| model and path | format | prefill | decode / B8 step | LOD cache |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B, three-tier | BF16 | 5.767 s | 2.794 ms | 7.870 GiB |
| | INT8 | **5.469 s** (-5.2%) | 2.823 ms (+1.0%) | **4.510 GiB** (-42.7%) |
| Qwen3.8-27B-FP8, three-tier | BF16 | **70.042 s** | **35.100 ms** | 41.972 GiB |
| | INT8 | 72.968 s (+4.2%) | 35.305 ms (+0.6%) | **24.053 GiB** (-42.7%) |
| Gemma-4-26B-A4B, three-tier | BF16 | **19.022 s** | 9.549 ms | 12.947 GiB |
| | INT8 | 20.684 s (+8.7%) | **9.485 ms** (-0.7%) | **7.345 GiB** (-43.3%) |
| Muse-Glimmer-30B, three-tier | BF16 | **55.444 s** | **19.165 ms** | 8.745 GiB |
| | INT8 | 55.474 s (+0.05%) | 19.204 ms (+0.20%) | **5.109 GiB** (-41.6%) |
| Qwen3.5-0.8B, generic two-tier | BF16 | **4.436 s** | **3.816 ms** | 8.018 GiB |
| | INT8 | 4.703 s (+6.0%) | 3.847 ms (+0.8%) | **5.019 GiB** (-37.4%) |

Native recursive INT8 is therefore the memory-balanced three-tier format for
Qwen, Gemma, and Muse. It removes 41.6--43.3% of persistent LOD cache while
leaving decode within 1.1%. Qwen prefill ranges from 5.2% faster on the small
model to 4.2% slower on the 27B model. Gemma pays a larger 8.7% prefill penalty,
so BF16 remains its latency-first format and INT8 is its low-memory serving
format. Muse INT8 is effectively tied with BF16 in both prefill and decode.

The fastest two-tier BF16 result uses the fixed-list page-size-one path:
4.517 s prefill, 2.885 ms decode, and 8.481 GiB cache. That path currently
requires BF16 persistent metadata. Comparing its 2.885 ms with generic INT8's
3.847 ms would incorrectly attribute a 33% algorithm/pathway gap to
quantization. The matched generic kernels show that on-demand INT8 itself adds
only 0.8% decode latency. Keep the production two-tier path BF16 until the
fixed-list consumer reads INT8 directly.

Muse confirms the same choice. A one-repeat screen of its generic two-tier
INT8 path measured 57.201 s prefill, 20.941 ms decode, and 5.843 GiB cache.
Recursive three-tier INT8 is faster in both phases (55.474 s and 19.204 ms)
and smaller (5.109 GiB), so there is no reason to promote the generic
two-tier INT8 fallback on Muse.

## Quality

| model | check | BF16 | INT8 |
|---|---|---:|---:|
| Qwen3.5-0.8B | 64K NIAH-S3 | 64/64 | 64/64 |
| Qwen3.5-0.8B | 8 x 8K ProLong token CE | 1.923225 | 1.923071 |
| Qwen3.5-0.8B | 8 x 8K ProLong perplexity | 6.842989 | 6.841939 |
| Qwen3.8-27B-FP8 | 64K NIAH-S3 smoke | established 64/64 | 8/8 |
| Gemma-4-26B-A4B | 64K NIAH-S3 smoke | 8/8 | 8/8 |
| Muse-Glimmer-30B | 64K NIAH-S3 | established 64/64 | 64/64 |

The Qwen CE difference is -0.000153 in favor of INT8 and is noise-sized; it is
not evidence that quantization improves the model. The Gemma check is a smoke
test, not a replacement for its established larger quality panel.

## Why not stage INT8 into BF16?

The matched generic two-tier result already answers the kernel question:
on-demand INT8 decode is only 0.8% slower than generic BF16. The large gap is
that the fixed-list BF16 algorithm does not yet accept INT8, not that its
on-demand arithmetic is slow. Expanding the full cache before attention would
also add at least one INT8 read plus one BF16 write: approximately 66 GiB per
Qwen3.8 batch step and 20 GiB per Gemma batch step across their LOD layers,
before attention reads the BF16 data again. Persisting the expansion would
give back the memory saving. Per-query compaction would reintroduce route-list
construction and a serialized copy. None is preferable to direct INT8 reads.

## Cold-centroid diagnostic

`qwen08_prolong64k_ever_opened.json` tracks the union of top-eight centroid
routes over every prefill query in one real 64K ProLong document, collapsed
across the four GQA query heads for each KV head. Across Qwen3.5-0.8B's six LOD
layers, 87.95% of live centroids and 92.77% of their leaves were selected at
least once. The per-layer selected-leaf coverage ranges from 89.34% to 95.61%.
Thus only 7.23% of leaf bytes are cold even under this hindsight oracle.

Evicting those leaves is unsafe because a future query can select them, and an
online policy cannot know the oracle set in advance. Compressing only that
subset from INT8 to INT4 could save at most another 3.6% of leaf bytes, before
mixed-format metadata and kernels. Cold-centroid eviction or tiered precision
is therefore not part of the recommended design.

## Configuration policy

Use native recursive INT8 for the memory-balanced Qwen/Gemma/Muse three-tier
mode:

```bash
VLLM_LOD_LEVELS=3
VLLM_LOD_KV_BITS=8
VLLM_LOD_ROUTING_GEOMETRY=auto
```

Use `VLLM_LOD_KV_BITS=0` for latency-first Gemma prefill and for the current
best two-tier fixed-list path. No model-wide automatic default was changed:
Phi's fastest recursive prefill consumer is BF16-only, and OLMo was not part
of this precision validation.

The primary files are the paired `*_64k_b8_r3.json` speed records, the Qwen
`*_niah64_64k.json` and `*_prolong8k_s8.json` quality records, the Gemma
`*_niah8_64k.json` smoke records, and the cold-centroid profile named above.

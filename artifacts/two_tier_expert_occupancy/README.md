# Two-tier GQA expert occupancy diagnostic

This diagnostic asks whether flat two-tier LOD creates sufficiently large
attention experts to make INT8 matrix instructions useful.  An expert is one
`(request, KV head, routed centroid)` tuple.  All query heads routed to that
tuple share its exact leaf list, so its ideal QK shape is `M x D` by `D x N`.

Configuration: `Qwen/Qwen3.5-0.8B`, distinct ProLong prompts, batch 8, top 8,
state schedule `16 sqrt(T)`, 64 greedy decode steps, six LOD layers.  Qwen3.5
0.8B has eight query heads and two KV heads, hence only four queries can ever
share an expert.  MMA occupancy below assumes a `16 x 32` output tile.

| Metric | 8K | 64K |
|---|---:|---:|
| Mean queries per expert | 1.549 | 1.613 |
| Experts with `M >= 2` | 34.09% | 36.33% |
| Experts with `M = 4` | 5.40% | 7.34% |
| Route pairs belonging to `M >= 2` experts | 57.46% | 60.53% |
| Route pairs belonging to `M = 4` experts | 13.94% | 18.19% |
| Ideal leaf reuse from expert grouping | 1.820x | 1.782x |
| Ideal leaf-read reduction | 45.07% | 43.90% |
| Mean exact leaf tokens per query | 169.0 | 891.3 |
| Exact work relative to recursive three-tier (128 tokens) | 1.32x | 6.96x |
| Useful fraction after only `M=16` padding | 11.38% | 11.14% |
| Useful fraction after `M=16, N=32` padding | 5.20% | 9.62% |

The centroid leaf lists are long enough to improve N occupancy at 64K, but M
is the hard limitation.  Most experts have only one query and no expert can
exceed four queries.  Batch size eight does not increase M because different
requests have different KV caches.  A standard 16-row INT8 MMA therefore
computes roughly nine times more rows than are useful even before N padding.

Flat two-tier also opens far more exact leaves than the recursive three-tier
path: 1.32x at 8K and 6.96x at 64K.  The selected centroids are substantially
larger than an average centroid, so the simple `T / state_size` estimate
understates this cost.

Conclusion: expert grouping is a useful memory-traffic optimization, but on
this model it does not create an attractive conventional INT8-MMA shape and
does not offset the rapidly growing exact work of flat two-tier LOD.  It may
be worth repeating on a model with a GQA group of at least 16; the conclusion
cannot be extrapolated to such a model from this one.

Raw results:

- `qwen35_08b_b8_8k.json`
- `qwen35_08b_b8_64k.json`

Reproduction:

```bash
.venv/bin/python -m scripts.diagnose_two_tier_expert_occupancy \
  --sequence-length 8192 --batch-size 8 --steps 64 \
  --state-growth-factor 16 \
  --output artifacts/two_tier_expert_occupancy/qwen35_08b_b8_8k.json
```

## INT8 coarse-region summaries

The current recursive decode router already shares each coarse K/V region tile
across a GQA group and evaluates QK and probability-times-V with BF16 MFMA.  On
Qwen3.5-0.8B its four real query rows are padded to the native 16-row MFMA
shape.  Quantizing region summaries therefore does not improve M occupancy; it
only changes summary bandwidth and MFMA datatype.

The measured fused decode breakdown is:

| Metric | 8K | 64K |
|---|---:|---:|
| End-to-end decode batch step | 4.398 ms | 4.326 ms |
| Coarse-region scan (`route_groups`) | 0.115 ms | 0.118 ms |
| Route candidate reduction | 0.132 ms | 0.255 ms |
| Region scan fraction of decode | 2.62% | 2.72% |

Even deleting the region scan entirely could therefore improve decode by only
about 2.7%.  INT8 K with BF16 V saves only one quarter of the combined K/V
payload, giving a bandwidth-only ceiling around 0.7%.  An ideal INT8 K/V path
that halved the whole scan would save about 0.06 ms, or 1.4% end to end.  The
candidate reducer, now the larger routing cost at 64K, is unaffected.

Prefill is more promising because it supplies many query rows and fully
occupies matrix tiles.  In the 64K profile, route and coarse-attention events
were 0.472 s and 0.808 s respectively out of a 5.556 s prefill.  Their combined
ideal 2x ceiling is about an 11.5% prefill improvement, before charging query
quantization and region re-quantization after state updates.

An efficient prototype should first quantize only region K with one scale per
region and one scale per query row.  Groupwise channel scales would split a
128-channel dot into several small MMAs and recreate the leaf-kernel problem.
INT8 V requires quantizing the softmax probabilities as well, which is more
complex and affects every closed region's coarse contribution.  Region K also
directly determines the top-8 routing boundary, so quality risk is higher than
for quantized leaves and should be checked with a route-set mismatch diagnostic
before an end-to-end kernel is pursued.

Raw 64K profile: `region_summary_route_profile_64k_b8.json`.

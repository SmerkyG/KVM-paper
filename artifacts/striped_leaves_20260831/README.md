# Striped exact-leaf load balancing (2026-08-31)

This experiment ports the MTP exact-leaf balancing change to ordinary
two-tier decode's portable Triton fallback. The selected top-eight centroids,
leaf set, attention arithmetic, and final LSE merge are unchanged. Only the
assignment of work changes:

- legacy: split `s` owns complete route `s`, so the largest selected centroid
  determines tail latency;
- striped: all splits process disjoint stripes of every route, balancing the
  total selected leaves across the eight workers.

Both speed tests use 64K real ProLong text, 16K chunked prefill, the same
captured vLLM path, BF16 LOD storage, top-eight routing, and a 1,024-leaf route
limit. The AITER fixed-list/union and GQA-cooperative backends were disabled to
force the portable fallback under test. Times are median marginal decode
milliseconds per batch step across five 513-token (Qwen) or 257-token (Muse)
generations.

| Model | Batch | Legacy | Striped | Improvement |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 8 | 3.690 ms | **3.435 ms** | **6.9%** |
| Muse-Glimmer-30B | 1 | 17.117 ms | **16.572 ms** | **3.2%** |

The striped Qwen fallback also scores **64/64** on chat-formatted 64K
NIAH-S3. Since striping neither adds nor removes an attention term, the result
is expected to be numerically equivalent apart from normal parallel reduction
rounding.

The production page-size-1 AITER fixed-list path already concatenates all
selected posting lists and splits the flat sequence evenly. The ordinary
GQA-cooperative backend similarly partitions each route across page-list
splits. Those primary paths therefore already have the load-balancing property
that this patch adds to the fallback.

Three-tier decode has a different dependency: each routed centroid must first
choose exactly one globally best page. Splitting its page summaries among
workers requires a partial-max/global-max stage before exact-page attention;
simply applying the two-tier stripe would incorrectly open one page per split.
The existing opt-in materialized page-score backend is the correct parallel
alternative when its GQA reuse outweighs writing the complete score field.

For completeness, a correct two-stage page-stripe prototype was implemented
and tested before being removed from the production code. It used a first
kernel to find a partial top page in each stripe, then reduced those candidates
inside the existing residual/exact-page kernel. It reused route scratch and
therefore added no cache or state memory. At Qwen 0.8B, 64K, batch 8, five
repeats, the results were:

| Page stripes | Decode |
|---:|---:|
| 0 | 2.595 ms |
| 2 | 2.581 ms (-0.54%) |
| 4 | 2.611 ms (+0.60%) |
| 8 | 2.986 ms (+15.0%) |

The two-stripe result was only a noise-sized gain. On the intended
large-centroid counterexample, Muse at 64K/batch 8, two stripes were slightly
slower: 20.143 ms versus the 20.058 ms three-tier baseline. The additional
selection launch cancels the load-balancing benefit, so this experimental path
is not retained. Three-tier continues to use its existing selector or the
GQA-cooperative materialized-score option.

Artifacts:

- `qwen08_generic_legacy_64k_b8_r5_d513.json`
- `qwen08_generic_striped_64k_b8_r5_d513.json`
- `qwen08_generic_striped_niah_64k_b8_n64.json`
- `muse_generic_legacy_64k_b1_r5_d257.json`
- `muse_generic_striped_64k_b1_r5_d257.json`
- `qwen08_recursive_stripe{0,2,4,8}_64k_b8_r5_d513.json`
- `muse_recursive_stripe0_64k_b8_r3_d257.json`
- `muse_recursive_stripe2_64k_b8_r1_d513.json`

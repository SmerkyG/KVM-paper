# Monotone static-cohort eviction

## Result

Static cohorts make most archived leaf K/V unnecessary. At 64K, where the
scheduled cap is 16 for the entire request, only 17.36%-34.06% of leaf tokens
belong to the final exact cohort across the five-model panel. Counts only grow,
so this final cohort is already the monotone (never re-admitted) cohort.

The current implementation still allocates chronological leaf K/V for every
token. The table therefore reports two projections from the measured working
sets rather than claiming that the current allocator has already released the
memory:

* **Leaf-compacted** removes evicted leaf K/V but leaves the existing page
  summaries and page-directory machinery allocated.
* **Static arena** also removes the unused 16-token page summaries, page
  counts, virtual-page directory, overflow directory, and page-index table;
  shrinks the persistent AITER index and bias arrays with the leaf arena; and
  adds a conservative two INT32 words per retained token plus one INT32 head
  per centroid for allocation/list metadata. State K/V, counts, local K/V,
  coarse K/V, and other measured pool allocations remain.

All memory values are GiB per measured worker/GPU. Phi-4 is TP5, so its row is
the per-rank footprint.

| 64K model | Retained leaf K/V | Current LoD cache | Leaf-compacted | Static arena | Static-arena saving |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | 31.74% | 45.10 | 23.08 | 17.05 | 28.05 GiB (62.19%) |
| Gemma-4-26B-A4B | 34.06% | 13.77 | 7.12 | 5.53 | 8.23 GiB (59.80%) |
| Phi-4 TP5 | 19.06% | 29.49 | 13.17 | 8.15 | 21.33 GiB (72.35%) |
| Muse-Glimmer-30B | 29.79% | 9.58 | 4.98 | 3.37 | 6.21 GiB (64.82%) |
| OLMo-3-32B | 17.36% | 47.18 | 20.53 | 12.48 | 34.70 GiB (73.55%) |

The retention measurements come from the B8 64K real-prompt static-prefill
panel in `artifacts/static_prefill_panel_20260826`. The allocation geometry is
the recorded worker-side `lod_dispatch`, and the baseline is the measured
`attention_memory.lod_cache_bytes` in each artifact.

## Why never re-admit matters at 128K

At 128K the default schedule grows from cap 16 to cap 23. Ordinary static
routing uses the current cap and therefore re-admits centroids with 17-23
leaves. A diagnostic run tracked a terminal state per centroid (`unseen`,
`exact`, or `evicted`) over real ProLong B8 prefill:

| Qwen3.5-0.8B, 128K | Exact leaves / all archived leaves | Current 15.01-GiB pool: leaf-compacted | Projected static arena |
|---|---:|---:|---:|
| Current-cap membership (can re-admit) | 32.95% | 6.93 GiB | 4.95 GiB |
| Monotone, never re-admit | 2,948,881 / 12,558,336 = **23.48%** | **5.79 GiB** | **3.79 GiB** |

Thus the requested monotone rule removes another 9.47% of all archived leaf
tokens at 128K. In the clean static-arena projection that is another 1.16 GiB
per GPU beyond current-cap static membership, and 11.22 GiB (74.75%) below the
currently allocated LoD cache.

The diagnostic artifact is
`artifacts/static_cohort_eviction_20260826/qwen08_128k_b8_final_diag.json`.
It exercised 6 LoD layers x 8 requests x 2 KV heads and observed scheduled
caps 16 through 23. The run only measured membership; it did not alter model
attention or physically compact the cache.

## Quality of never re-admitting

The production attention-membership rule was enabled with
`VLLM_LOD_STATIC_COHORT_NEVER_READMIT=1` and tested at 128K, where the cap
actually grows from 16 to 23. At shorter tested lengths the cap stays at its
16-entry floor, so current-cap and never-readmitted static routing are
identical. These tests change exact-versus-coarse membership but do not yet
physically release the evicted K/V allocation.

The ProLong panel uses the same eight deterministic 131,072-token streams in
all four arms. Each stream concatenates distinct shuffled real documents from
`Seerkfang/prolong-64k-512-new`; no document or synthetic text is repeated to
fill the context. Loss covers 1,048,568 next-token predictions in each arm.
BF16 LoD, B8, 16K scheduler chunks, and 4K state updates are used throughout.

| Qwen3.5-0.8B, ProLong 128K | Cross entropy | Perplexity | CE delta vs full | PPL delta vs full |
|---|---:|---:|---:|---:|
| Full attention | **1.934835** | **6.922904** | - | - |
| Two-tier top-8 | 1.961644 | 7.111010 | +0.026809 | +2.717% |
| Static, current-cap re-admission | 1.984229 | 7.273441 | +0.049394 | +5.063% |
| Static, never re-admit | 1.985693 | 7.284092 | +0.050857 | +5.217% |

Never re-admission therefore adds **0.001463 CE** over ordinary static routing,
or **0.1465% perplexity** relative to ordinary static. Across the eight paired
documents the mean CE delta is +0.001463 with standard error 0.000661 (range
-0.000596 to +0.004094). This is a small but measurable cost: only about 3% of
the much larger static-versus-full CE gap.

The matched 64-example NIAH-S3 retrieval panel did not show a regression:

| Qwen3.5-0.8B, NIAH-S3 128K | Score | Exact-leaf fraction |
|---|---:|---:|
| Static, current-cap re-admission | 63/64 | 32.67% at the final cap |
| Static, never re-admit | **64/64** | **26.63% permanent** (32.64% would satisfy the final cap) |

The policies differed on one example: current-cap static made a one-character
GUID error that never-readmitted static did not. This single result is not
evidence that eviction improves quality, but it rules out a visible retrieval
penalty in this panel. Combined with ProLong, the conclusion is that monotone
eviction preserves downstream retrieval while imposing a very small average
language-modeling loss.

Artifacts:

* `qwen08_prolong_ce_full_128k_b8_s8.json`
* `qwen08_prolong_ce_top8_128k_b8_s8.json`
* `qwen08_prolong_ce_static_128k_b8_s8.json`
* `qwen08_prolong_ce_monotone_128k_b8_s8.json`
* `qwen08_current_static_128k_b8_s64.json`
* `qwen08_monotone_128k_b8_s64.json`

## Recommended implementation

Use a page-size-one compact leaf arena, not the existing 16-token leaf pages.
The latter wastes nearly a full page for every underfull centroid and largely
erases the benefit on models such as OLMo, where many centroids have very few
leaves.

Each centroid has a terminal one-byte state:

```
0 unseen -> 1 exact -> -1 evicted
                    (terminal)
```

Coarse `state_k`, `state_v`, and `count` continue to update in all states. New
tokens assigned to an evicted centroid update only those coarse aggregates.
When an exact centroid first exceeds the cap, return its exact tokens to the
arena free list and never archive leaves for it again. Crucially, both sides of
attention must consult the terminal state: the exact AITER list includes only
state `1`, while the coarse branch includes state `-1` even if a later, larger
scheduled cap would otherwise make its current count eligible.

The least complicated efficient metadata is:

* one compact BF16 K arena and V arena;
* one INT32 linked-list/free-list word per physical token (a separate free
  stack is optional, not required);
* one INT32 list head and the existing length/status per centroid;
* the persistent page-size-one AITER index list already used by static decode.

Prepending tokens makes append O(1); leaf order is irrelevant to softmax.
Freeing a centroid traverses at most the cohort cap (16 at 64K, 23 at 128K),
and happens only at a state update. The final AITER list is rebuilt once per
4K state update and reused, so decode keeps the current one-attention-call
shape.

For a 4K prefill update, the clean two-pass order is:

1. Route/update the coarse state and emit each new token's centroid ID.
2. Update terminal cohort states, release newly evicted lists, compact only
   tokens whose centroid remains exact, and rebuild the persistent AITER list.

This avoids writing K/V that will be evicted at the same update. To implement
"ever outside" exactly when the scheduled cap changes inside a 4K chunk, split
the update at the rare schedule boundary. Adjacent cap boundaries are already
more than 4K tokens apart in the relevant range, so there is at most one such
split per update and most updates are unchanged.

## Allocation caveat and optimal ownership

There is no sublinear worst-case capacity bound: `16 sqrt(T)` centroids can all
contain exactly `sqrt(T)/16` leaves, whose product is T. A conventional fixed
PyTorch tensor must therefore either use an empirical capacity/headroom or
reserve the original T-token physical allocation, in which case eviction does
not return VRAM.

The simple production version should use one pointer-stable arena shared by
all LoD layers on a GPU, rather than one separately provisioned arena per
layer. Global ownership averages variation across layers, KV heads, and B8
requests; indices already carry arbitrary page-size-one locations. Provision
it from a measured live-token high-water mark plus explicit headroom, and fail
loudly on exhaustion rather than silently changing routing.

The fully robust version keeps the same design but reserves a large virtual
address range and maps/unmaps HIP physical-memory blocks as the live arena
grows and shrinks. That preserves the K/V base pointer required by captured
vLLM graphs while making physical VRAM follow live leaves. It is more involved
than the fixed shared arena, so the latter is the appropriate first
implementation and benchmark target.

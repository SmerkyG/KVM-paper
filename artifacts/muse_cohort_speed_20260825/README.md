# Muse unrestricted top-8 versus static cohort decode (2026-08-25)

## Question

At 64K and 128K, is ordinary query-dependent top-eight routing faster than
opening every leaf belonging to the small-centroid cohort?

The static cohort is inclusive and uses

`max(16, ceil(sqrt(T) / 16))`,

which gives a posting-list cap of 16 at 64K and 23 at 128K.  "Unrestricted"
means that top-eight is not limited to that cohort; it retains the ordinary
1024-leaf safety guard.

## Method

- checkpoint: `meta-models/Muse-Glimmer-30B`, native text configuration;
- batch 8, BF16 two-level LOD, raw routing geometry;
- distinct real ProLong documents, with identical prompt hashes between arms;
- 16K maximum batched prefill, 64 generated tokens, median of three measured
  cold-prefill/decode repetitions after warmup;
- top-eight: persistent centroid-major fixed list, route-prepared mask, and
  M16/N64 AITER-shaped final attention;
- static cohort: one compact persistent sink + exact-small + coarse-large +
  local list, with no query-dependent routing or mask.

The execution audits confirm that the fixed-mask top-eight and compact static
AITER paths respectively executed at both lengths.

## Results

| context | unrestricted top-8 decode | every cohort leaf decode | static speedup | top-8 prefill | static prefill |
|---:|---:|---:|---:|---:|---:|
| 64K | 19.349 ms | **18.696 ms** | **3.49%** | 56.508 s | 55.040 s |
| 128K | 19.072 ms | **18.503 ms** | **3.08%** | 120.744 s | 121.794 s |

Prefill is algorithmically the same between the two decode policies; its
opposite-signed differences are run noise rather than an effect of selection.

For context, the post-cache-fix matched controls were:

| context | full attention | prior static cap 15 | current formula static |
|---:|---:|---:|---:|
| 64K | 19.215 ms | 18.647 ms | 18.696 ms (+0.26%) |
| 128K | 19.487 ms | 17.996 ms | 18.503 ms (+2.81%) |

The older 21.198-ms static result predates the custom-cache slowdown fix and
must not be used as the current baseline.

## Interpretation

For Muse, removing query-dependent coarse scoring, top-eight selection, and
fixed-mask preparation is worth about 3% end to end, even though the static
policy evaluates every eligible small-centroid leaf.  Increasing the 128K cap
from 15 to 23 adds enough exact work to slow static decode by 2.81%, but it is
still 3.08% faster than unrestricted top-eight and 5.32% faster than the recent
matched full-attention control.

This supports the serial-barrier hypothesis, but only modestly: the static
formulation is the faster one on Muse, while the benefit is a few percent, not
a large multiple.

## Artifacts

- `muse_unrestricted_top8_b8_r3.json`
- `muse_all_cohort_leaves_b8_r3.json`
- post-cache-fix references:
  `../static_lod_muse_20260825/muse_current_static_cap15_64k_128k_b8_r3.json`
  and `../static_lod_muse_20260825/muse_current_full_64k_128k_b8_r3.json`

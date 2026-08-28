# Small-centroid cohort routing on Qwen3.5-0.8B (2026-08-25)

## Question

Can two-level decode select routes only from centroids whose posting-list size
is at most

`max(16, ceil(sqrt(T) / 16))`

and then reduce top-eight to top-four/six, or replace top-k with a predicted
mass cutoff, without losing the quality of ordinary top-eight routing?

The cohort policy overrides the older `DECODE_MAX_OPEN_LEAVES` guard rather
than intersecting with it.  At 64K the inclusive cohort cap is 16, represented
by the route kernels' audited exclusive limit of 17.  Larger centroids remain
live, keep receiving updates, and contribute through their coarse entry; they
are simply ineligible to have their exact leaves opened.

## Method

- model: `Qwen/Qwen3.5-0.8B`, BF16 LOD state, raw routing geometry;
- context: 64K, batch 8, chat template enabled, thinking disabled;
- quality: NIAH-S3 GUID retrieval, greedy decoding;
- speed: distinct real ProLong documents, 65 timed decode tokens, median of
  five repeats;
- final attention: fixed-list M16/N64 page-size-one AITER-shaped path with 128
  segments and block-fast-fail masks.

An initial sweep accidentally reused the old static-cap diagnostic mask, which
opened every small centroid after query-dependent routing.  Those artifacts do
not measure the requested policy and are intentionally excluded below.  The
corrected path uses the cap only while forming routes.

## Results

| policy | 64K NIAH-S3 | decode ms / batch step | mean union centroids | result |
|---|---:|---:|---:|---|
| historical full AITER attention | not rerun | 5.093 | n/a | reference |
| unrestricted top-8 | **64/64** | **3.077** | 19.61 | quality baseline |
| unrestricted predicted mass, 1/32 | 63/64 | not timed | not collected | one GUID character omitted |
| cohort top-8 | 62/64 | 3.091 | 19.89 | no speed gain; quality loss |
| cohort top-6 | 4/8 screen | 2.773 | 15.83 | reject |
| cohort top-4 | 44/64 | 2.935 | 10.64 | reject |
| cohort predicted mass, 1/16 | 7/8 screen | 2.783 | 9.30 | reject |
| cohort predicted mass, 1/32 | 62/64 | **2.953** | 12.60 | faster, but not lossless |
| static: open every cohort leaf | **64/64** | 4.309 | n/a | accurate but slower |

The cohort top-eight and 1/32 mass policies miss the same examples, 25 and 38.
This localizes the remaining error to the cohort restriction rather than the
particular top-k versus mass selector.  Opening every small-centroid leaf
recovers 64/64, so the evidence is present in the cohort, but query-dependent
routing does not identify all of the small centroids needed for exact GUID
reconstruction.  Conversely, unrestricted top-eight can open a larger posting
list when its coarse score warrants it and recovers 64/64.

The additional unrestricted predicted-mass 1/32 control reaches 63/64 and
misses example 6 by dropping one character from the GUID.  Its audited route
has no cohort restriction, so predicted mass itself is slightly less robust
than fixed top-eight on this panel.  Removing the cohort recovers one net case
relative to 62/64, although the failure set changes rather than being a strict
subset (example 6 versus examples 25 and 38).

## Conclusion

Restricting dynamic routing to this cohort is not a free speed optimization on
Qwen3.5-0.8B.  Fixed top-eight performs the same amount of route construction
and nearly the same final masked scan, so the restriction is effectively tied
with unrestricted top-eight in speed while dropping two NIAH cases.  On the
matched GPU-0 speed-only controls, removing the top-k barrier with predicted
mass is a real 4.0% decode speedup, but the cohort restriction caps quality at
62/64 in this panel.  The earlier combined quality/speed run on another GPU
reported 2.785 ms; it is retained as an artifact but is not used for the
matched comparison.

The safe result is therefore to keep unrestricted top-eight as the primary
policy.  A promising follow-up would retain the small-centroid cohort as a fast
first choice but permit a small score- or mass-triggered escape set of larger
centroids; the current experiment says a hard cohort boundary is the wrong
uniform rule.

## Artifacts

- `qwen08_top8_nocohort_64k_b8_r5.json`
- `qwen08_top8_nocohort_64k_b8_s64.json`
- `qwen08_predmass32_nocohort_64k_b8_s64.json`
- `qwen08_cohort_top8_fixed_64k_b8_s8_r5.json`
- `qwen08_cohort_top8_fixed_64k_b8_s64.json`
- `qwen08_cohort_top6_fixed3_64k_b8_s8_r5.json`
- `qwen08_cohort_top4_fixed2_64k_b8_s8_r5.json`
- `qwen08_cohort_top4_fixed_64k_b8_s64.json`
- `qwen08_cohort_predmass16_fixed_64k_b8_s8_r5.json`
- `qwen08_cohort_predmass32_fixed_64k_b8_s8_r5.json`
- `qwen08_cohort_predmass32_fixed_64k_b8_speed_gpu0_r5.json`
- `qwen08_cohort_predmass32_fixed_64k_b8_s64.json`

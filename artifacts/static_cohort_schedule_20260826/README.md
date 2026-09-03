# Static-cohort schedule comparison

## Result

At 128K, changing the inclusive static-cohort cap from
`max(16, ceil(sqrt(T) / 16))` (cap 16 through 23) to
`max(32, ceil(sqrt(T) / 8))` (cap 32 through 46) materially improves language
modeling quality, while approximately doubling the leaves that must remain
exact. Both schedules retain NIAH-S3 retrieval quality under monotone
never-readmission.

All ProLong arms use the same eight deterministic 131,072-token streams,
formed by concatenating distinct explicitly indexed rows from
`Seerkfang/prolong-64k-512-new`. Every artifact records per-stream source row
indices and SHA-256 hashes; the hashes match across all six arms. Each row
covers 1,048,568 next-token predictions with Qwen3.5-0.8B, BF16, B8, 16K vLLM
scheduler chunks, and 4K LoD state updates.

| ProLong 128K | Cross entropy | Perplexity | CE delta vs full | PPL delta vs full | Final-cap exact leaves |
|---|---:|---:|---:|---:|---:|
| Full attention | **2.511656** | **12.325328** | - | - | 100% |
| Two-tier top-8 | 2.538752 | 12.663861 | +0.027096 | +2.747% | query-dependent |
| Old static, re-admitting | 2.547977 | 12.781224 | +0.036321 | +3.699% | 31.40% |
| Old static, never re-admit | 2.549034 | 12.794735 | +0.037377 | +3.808% | 31.57% current-cap membership |
| New static, re-admitting | 2.532484 | 12.584730 | **+0.020828** | **+2.105%** | 55.12% |
| New static, never re-admit | 2.532110 | 12.580017 | **+0.020453** | **+2.066%** | 55.21% current-cap membership |

On paired documents, the new schedule improves CE by 0.015493 +/- 0.002484
standard error versus old re-admitting static, and by 0.016924 +/- 0.002282
versus old never-readmitted static. Its perplexity is respectively 1.54% and
1.68% lower than the matching old schedule. The new static mean is also
0.00627 CE below top-8, although that smaller difference is not resolved by
only eight documents (paired standard error 0.00600).

Never-readmission costs +0.001057 CE under the old schedule but -0.000375 CE
under the new schedule. The latter is noise-sized (paired standard error
0.000625): with the larger initial cohort, terminal eviction has no detected
ProLong penalty.

## NIAH-S3 and retention

The matched 64-example 128K NIAH-S3 panel used chat formatting and the same
B8/16K scheduler setup. The recorded cap histories and kernel-dispatch audits
prove that the requested schedules ran in both prefill and decode.

| Schedule | Re-admitting | Never re-admit | Final-cap exact leaves | Permanent exact leaves |
|---|---:|---:|---:|---:|
| Old: `max(16, ceil(sqrt(T)/16))` | 63/64 | 64/64 | 32.6% | 26.63% |
| New: `max(32, ceil(sqrt(T)/8))` | **64/64** | **64/64** | 57.5% | 51.35% |

The new schedule removes the old re-admitting arm's single-character GUID
error, but the old never-readmitted schedule was already perfect. The robust
gain is therefore the ProLong loss reduction, not NIAH accuracy. Its cost is
large: permanent exact retention rises from 26.63% to 51.35% on NIAH, and
current-cap exact retention rises from about 31% to 55% on the deterministic
ProLong panel. This will reduce the achievable compact-cache memory saving and
increase exact-leaf attention work.

## Recommendation

The new schedule is the better quality-oriented static default candidate: it
closes about 43% of the old static-versus-full CE gap and makes monotone
eviction effectively free in this panel. It should not simply replace the old
schedule when maximum compression or speed is the objective, because it keeps
roughly twice as many permanent leaves. Treat the two as quality and
compression operating points until matched speed measurements establish the
end-to-end cost.

## Artifacts

The authoritative deterministic ProLong artifacts begin with
`qwen08_prolong_det_`; the earlier `qwen08_prolong_ce_` files in this directory
used a streaming shuffle whose row order differed from the prior online run
and are excluded from every comparison above.

* `qwen08_prolong_det_full_128k_b8_s8.json`
* `qwen08_prolong_det_top8_128k_b8_s8.json`
* `qwen08_prolong_det_static_sqrt16_min16_128k_b8_s8.json`
* `qwen08_prolong_det_monotone_sqrt16_min16_128k_b8_s8.json`
* `qwen08_prolong_det_static_sqrt8_min32_128k_b8_s8.json`
* `qwen08_prolong_det_monotone_sqrt8_min32_128k_b8_s8.json`
* `qwen08_niah_static_sqrt8_min32_128k_b8_s64.json`
* `qwen08_niah_monotone_sqrt8_min32_128k_b8_s64.json`

The old-schedule NIAH controls are in
`artifacts/static_cohort_eviction_20260826`.

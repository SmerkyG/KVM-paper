# Two-level routed-prefill leaf-visit cap diagnostic

## Question

Is ordinary two-level routed prefill slow because a few selected centroids
have very long posting lists? This diagnostic leaves routing unchanged but
limits exact attention to the first 16 leaves of each selected centroid.

This is a speed-only experiment. It removes the selected centroid's coarse
contribution in the usual way but does not restore a coarse residual for the
truncated leaves, so the capped output is not quality-valid.

## Important routing correction

The vLLM pool currently sets
`prefill_two_level_topk = min(3, settings.open_count)`. Therefore, despite
historical artifact names containing `top8` and runs setting
`VLLM_LOD_OPEN_COUNT=8`, ordinary prefill opens three centroids. Decode still
opens eight. Both arms below have an audited effective prefill open count of
three.

## 64K batch-8 ProLong results

All runs use real distinct ProLong documents, 16K scheduler chunks, 4K LOD
prefill catch-up chunks, BF16 LOD storage, and the median of three measured
repetitions. Lower is better. The uncapped large-model controls are the
post-cache-fix controls from the immediately preceding 64K panel.

| model | uncapped routed top-3 | cap 16 | cap delta | full attention | cap vs full |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 4.535 s | 4.617 s | +1.8% | 9.930 s | -53.5% |
| Phi-4 TP5 | 32.930 s | 32.092 s | -2.5% | 28.119 s | +14.1% |
| Muse-Glimmer-30B | 56.508 s | 54.748 s | -3.1% | 51.933 s | +5.4% |
| OLMo-3-1125-32B | 74.457 s | 72.364 s | -2.8% | 67.892 s | +6.6% |

Measured repeats:

| model | uncapped repeats (s) | cap-16 repeats (s) |
|---|---|---|
| Qwen3.5-0.8B | 4.535, 4.524, 4.546 | 5.424, 4.617, 4.542 |
| Phi-4 TP5 | 32.948, 32.917, 32.930 | 32.063, 32.092, 32.135 |
| Muse-Glimmer-30B | 56.598, 56.508, 54.764 | 56.427, 54.748, 54.720 |
| OLMo-3-1125-32B | 76.094, 74.358, 74.457 | 72.418, 72.316, 72.364 |

## Conclusion

The cap saves only about 2.5--3.1% on the three slow large-model paths and
nothing on Qwen. Muse's repeat distributions overlap particularly strongly.
Even the nominal capped medians remain slower than full attention on Phi,
Muse, and OLMo. Oversized selected-centroid leaf scans therefore contribute a
small amount of time, but do not explain the main routed-prefill deficit.

## Implementation and artifacts

`VLLM_LOD_PREFILL_LEAF_VISIT_CAP=16` clamps the temporary per-expert lengths
passed to exact leaf attention. It does not change routing or authoritative
stored posting-list lengths. The benchmark records both the configured cap
and the effective prefill open count in its dispatch audit.

- `qwen08_top3_uncapped_64k_b8_r3.json`
- `qwen08_top3_visitcap16_64k_b8_r3.json`
- `phi4_tp5_top3_visitcap16_64k_b8_r3.json`
- `muse_top3_visitcap16_64k_b8_r3.json`
- `olmo_top3_visitcap16_64k_b8_r3.json`

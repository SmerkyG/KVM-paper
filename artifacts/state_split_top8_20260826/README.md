# Two-level top-8 with bounded posting-list splits

Date: 2026-08-26

## Experiment

This tests an overcomplete two-level LOD state on
`Qwen/Qwen3.5-0.8B`:

- literal top-8 in prefill and decode;
- split a posting list when a state entry would exceed 16 leaves;
- continue appending the same number of ordinary state entries prescribed by
  the original `16*sqrt(T)` schedule, independently of split-created entries;
- batch 8, BF16 leaf KV, 16K vLLM scheduler budget, 4K LOD state updates;
- coherence-aware assignment geometry selected by the existing `auto` policy.

`VLLM_LOD_STATE_SPLIT_MAX_LEAVES=16` enables the experiment and
`VLLM_LOD_PREFILL_OPEN_COUNT=8` requests literal prefill top-8.

## Split construction

Already-published leaves stay in their existing posting list. New assignments
that would exceed its capacity are gathered as follows:

1. Sort the assignments by affinity to the existing parent. The closest keys
   fill any capacity remaining in that parent.
2. Use the least parent-like incoming key as a directional pivot.
3. Sort the overflow by similarity to the pivot and gather consecutive groups
   of at most 16 into child centroids.

The pivot step makes sibling centers differ in key direction. It replaces the
earlier scalar-affinity bands, which could differ only in distance from the
parent while remaining directionally blurry. It does not relocate old leaf
pages.

The implementation tracks `scheduled_state_len` separately from actual
`state_len`. Split children only increase the latter; each later update still
appends every entry required to advance the former along the ordinary
schedule.

## 128K ProLong quality

Matched deterministic concatenated ProLong, 8 examples and 1,048,568 scored
tokens:

| method | CE | perplexity | delta vs full | evaluator elapsed |
|---|---:|---:|---:|---:|
| Full attention | 2.511656 | 12.325328 | - | 72.32 s |
| Unsplit literal top-8 | **2.530030** | **12.553884** | +0.018374 | 18.53 s |
| Split-16, scalar parent-affinity bands | 2.534353 | 12.608270 | +0.022697 | 80.14 s |
| Split-16, directional pivot gathering | 2.534265 | 12.607165 | +0.022609 | 117.25 s |
| Prior production prefill top-k | 2.538752 | 12.663861 | +0.027096 | 17.55 s |

Directional gathering improves only 0.000088 CE over scalar bands and remains
0.004235 CE worse than the exact unsplit top-8 control. The elapsed column is
an end-to-end evaluator diagnostic, not a warmed kernel benchmark, but the
large regression is unambiguous: this torch-based split construction and the
overcomplete coarse scan are not speed viable.

At 128K the directional run reached 14,288 actual state slots while the
independent ordinary schedule reached 5,724. The largest posting list was
exactly 16, so both the leaf cap and schedule-independence invariants held.

## NIAH-S3

The directional implementation scored 8/8 at both 8K and 128K. At 8K it
reached 1,512 actual state slots versus 1,378 ordinary scheduled slots. At
128K it reached 14,995 actual slots versus 5,775 scheduled slots. The maximum
posting list was 16 in both runs. The 128K screen took 130.61 seconds; a full
64-example run was not warranted after the matched ProLong regression.

## Conclusion

The experiment establishes that bounded posting lists and independent normal
schedule growth are implementable and that child centroids can be gathered
with directionally distinct centers. It does **not** beat unsplit literal
top-8: finer centroids make top-8 cover much less total exact mass, while the
overcomplete coarse field costs more to scan. Center separability is therefore
not the main remaining issue. This should stay an opt-in diagnostic rather
than replace the primary two-level path.

## Artifacts

- `qwen08_prolong_control_top8_128k_b8_s8.json`
- `qwen08_prolong_split16_top8_128k_b8_s8.json`
- `qwen08_prolong_split16_pivot_top8_128k_b8_s8.json`
- `qwen08_niah_split16_top8_128k_b8_s8.json`
- `qwen08_niah_split16_pivot_top8_8k_b8_s8.json`
- `qwen08_niah_split16_pivot_top8_128k_b8_s8.json`

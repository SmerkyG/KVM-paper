# Split-N long-expert prefill, batch 8

Qwen3.5-0.8B, 128K context, BF16 virtual leaf storage, top-8 two-level
routing, `16*sqrt(T)` state schedule, gfx942.

The implementation adds a final compound-sort bucket only for posting lists
above a configurable leaf threshold. Each long expert's query blocks execute
against 2/4/8 contiguous leaf ranges, write compact FP32 output/LSE partials,
and a second kernel performs the exact log-sum-exp merge. Ordinary experts do
not allocate partials and remain on the original kernel. Persistent LOD state
is unchanged.

## Exact-leaf phase

| Leaf threshold | Splits | Total exact-leaf time |
|---:|---:|---:|
| Disabled, matched run | 1 | 1748.37 ms |
| 2048 | 2 | 1812.28 ms |
| 2048 | 4 | 1805.61 ms |
| 4096 | 2 | 1810.12 ms |
| 4096 | 4 | 1803.71 ms |
| 4096 | 8 | 1800.41 ms |
| 8192 | 2 | 1752.04 ms |
| 8192 | 4 | 1737.09 / 1750.49 ms |
| 8192 | 8 | 1742.07 ms |

The >8K tail is effectively neutral (the matched split-4 repeat is 1750.49
ms versus 1748.37 ms unsplit). Broader thresholds regress by roughly 3--4%.
Prefill already exposes many independent M=16 query blocks for a long expert,
so splitting N adds query reloads, programs, FP32 transient storage, and a
reduction without materially improving occupancy. This is unlike decode,
where M is only 1--4 and N splitting is essential.

## Correctness and decision

A focused long-list comparison against unsplit attention measured maximum
absolute output error `4.8828125e-4` and LSE error `1.9073486e-6` for split
2, 4, and 8. Integrated 128K model runs produced finite logits.

The implementation remains available behind `long_expert_threshold` and
`long_expert_splits`, but defaults off because it does not improve prefill
speed on this workload.

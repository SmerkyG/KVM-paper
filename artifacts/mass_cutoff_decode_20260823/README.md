# Mass-cutoff decode experiment (2026-08-23)

## Question

Can two-tier LoD decode replace its ordered top-8 routing dependency with a
per-centroid mass cutoff, and does attending fewer exact leaves improve decode
latency?

## Implementations

Two decode variants were implemented.

1. **Predictive parallel cutoff** (`--decode-route-mass-fraction 0.03125`): use
   the preceding token's observed state-partition LSE as the threshold for the
   current token. A GQA-grouped state scan simultaneously computes current
   coarse attention and materializes every centroid above the cutoff. A compact
   pass writes at most eight routes per query. This is the variant that removes
   the ordered top-8 dependency.
2. **Exact-current top-8 filter** (also pass `--decode-mass-top8-filter`): retain
   the production fused top-8 router, then discard any of its routes below
   `current_state_lse + log(1/32)` inside the existing route/coarse reduction.
   This does not remove the top-8 dependency; it is a control that isolates the
   benefit of doing less exact-leaf work.

An atomic append implementation and overlap of coarse reduction with leaf work
were also measured. Both were slower than deterministic candidate
materialization plus compaction, so they are not retained in the active path.

## Setup

- Checkpoint: `Qwen/Qwen3.5-0.8B`
- Hardware: gfx942 cluster GPU
- Batch: 8
- Contexts: 8k, 16k, 32k, and 64k
- Decode: 16 warm-up tokens, then 257 timed tokens, five repeats
- Two-tier BF16 LoD cache and BF16 page summaries
- Baseline: current production top-8 two-tier decode
- Cutoff: 1/32 of estimated state attention mass
- Speed route cap: 8

## Whole-model decode latency

These are paired on the same GPU at each context length. The median is more
stable than the mean, but repeat-to-repeat cluster noise remains substantial
(one 16k baseline repeat was 17.21 ms while the other four were 13.96--14.69
ms). Treat small differences as neutral.

| Context | Top-8 median (ms/token) | Exact filter median (ms/token) | Latency change |
|---:|---:|---:|---:|
| 8k | 14.083 | 14.463 | 2.70% slower |
| 16k | 14.437 | 13.479 | 6.64% faster (noisy) |
| 32k | 14.532 | 14.312 | 1.52% faster / effectively neutral |
| 64k | 14.549 | 14.060 | 3.36% faster |

## Isolated LoD phase timing

CUDA events cover state routing, route reduction, exact leaf/local attention,
and final reduction across the model's six full-attention layers. Event
instrumentation perturbs absolute timing, but identifies where work changed.

| Context | Top-8 LoD phases (ms/token) | Exact filter LoD phases (ms/token) | LoD phase change | Leaf/local time saved |
|---:|---:|---:|---:|---:|
| 8k | 1.164 | 1.167 | 0.23% slower | 0.031 ms |
| 16k | 1.280 | 1.236 | 3.46% faster | 0.090 ms |
| 32k | 1.420 | 1.426 | 0.47% slower | 0.107 ms |
| 64k | 1.727 | 1.485 | 13.98% faster | 0.273 ms |

At 64k, the exact filter reduced selected leaf tokens from 4,176 to 1,170 per
sampled routing group (72% fewer) and reduced leaf/local time from 0.866 to
0.593 ms/token. Routing was unchanged (0.783 vs 0.781 ms/token), as expected.
Final reduction rose by 0.033 ms, leaving a net 0.241 ms LoD-phase saving.

At 32k, selected leaf tokens fell by 57%, but the 0.107 ms leaf saving was
almost exactly canceled elsewhere. Therefore the useful speed crossover on
this implementation is near 64k, not at short context.

## Predictive cutoff result

The predictive path really removes top-8 serialization, but it did not improve
speed. Median whole-model latency was slower than top-8 by 5.55%, 1.08%, 3.56%,
and 0.96% at 8k, 16k, 32k, and 64k respectively. Its grouped state producer is
as fast as the production top-8 score producer at 8k (0.369 vs 0.368 ms/token
across the six layers), but candidate compaction and coarse reduction erase the
saved leaf work. An atomic route append was slower still.

## Quality

On 32k NIAH-S3 with eight examples:

- Exact-current 1/32 filter: 8/8 exact.
- Predictive 1/32 cutoff with a 16-route cap: 8/8 exact.
- Predictive 1/32 cutoff with the speed-matched 8-route cap: 7/8 exact.
- Predictive 1/16 and 1/64 cutoffs: 7/8 exact each.

## Conclusion

Mass pruning itself becomes useful at 64k, where it gives a measurable leaf
kernel and end-to-end decode saving. The current predictive implementation does
not yet turn removal of top-8 serialization into a speedup: parallel threshold
selection still needs a GPU-friendly way to emit a compact ragged route list
without an additional global compaction dependency. The exact-current filter is
worth keeping as a low-complexity long-context optimization and as the control
for future predictive-router work.

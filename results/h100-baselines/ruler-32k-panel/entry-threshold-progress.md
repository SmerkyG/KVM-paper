# Per-entry LoD threshold investigation: in progress

This is a progress report. The full 32K RULER quality run is complete; the
paired 64K retrieval runs are still pending. Do not interpret the
three-example pilot as a quality result.

The [full 32K comparison](threshold-entry-32k-comparison.md) gives a
first-response 13-task average of 81.55% for entry multiple 16, versus 87.04%
for the region control, with 32 matched examples per task.

Follow-up tests of entry multiples 24 and 32 are complete for single-needle 3,
multi-key 2, and multi-key 3, with matched inputs and separate timings. They
loosen the deterministic bound to 24 or 32 times `sqrt(n)` respectively.
The 24 run scored 100%, 93.75%, and 93.75% on those three tasks, respectively,
versus 96.875%, 68.75%, and 71.875% for entry multiple 16. The region control
scored 100% on all three. The 32 run scored 100%, 93.75%, and 100%, with
the same two multi-key 2 failures as 24. It opens an average of 789 versus 504
entries/head on multi-key 2 and 775 versus 529 on multi-key 3 at the sampled
final step. These use 32 examples per task. The matched H100 whole-model
decode timings are 8.404 and 8.821 ms/token at 32K, and 8.714 and 9.070
ms/token at 64K, respectively; the region control is 7.817 and 8.302. Each
value is based on two measured repetitions. The
[retrieval and timing report](threshold-entry-multipliers-32k-report.md)
contains the complete comparison.
[Entry 24 outputs](threshold-entry-24-ruler32k-retrieval.json) ·
[Entry 32 outputs](threshold-entry-32-ruler32k-retrieval.json).

Additional experiments are testing entry cutoffs 24 and 32 with the original
strict `16 sqrt(n)` work cap. When candidates exceed that cap, this modified
rule keeps a deterministic prefix in region order. Its correctness check
passed. The 24 cutoff with the cap scored 100%, 93.75%, and 93.75% on the
three selected retrieval tasks, matching the uncapped 24 scores; 9 of the 96
full generated sequences differed. The capped 32 cutoff scored 100%,
96.875%, and 100% on those tasks, compared with 100%, 93.75%, and 100%
without the cap. These are small selected samples, not a full RULER panel;
the capped timings and results are in the [cap comparison](threshold-entry-cap16-32k-report.md).
[Entry 24 with cap outputs](entry-24-cap16-ruler32k-retrieval.json) ·
[Entry 32 with cap outputs](entry-32-cap16-ruler32k-retrieval.json).

For each query head, the entry rule uses the normalized proxy distribution
over nonempty remote regions and opens region r when
`p_r / n_r > 1 / (16 sqrt(n))`. Here n is the current number of remote
entries, not the total prompt length. In exact arithmetic, summing the
inequality over opened regions gives fewer than `16 sqrt(n)` opened remote
entries. Unopened regions remain represented by their summaries.
The proof counts logical KV entries. The paged worker processes 16-slot
pages with masks, so padded lanes and per-page bookkeeping add work; they do
not turn this into a bound of exactly `16 sqrt(n)` hardware operations.

The experimental H100 and AMD kernels use a one-part-per-million conservative
log threshold and an integer guard against actual page-list lengths. In the
unusual event that floating-point rounding would otherwise exceed the limit,
the guard closes a suffix of candidates in region order. Thus the numerical
implementation can be slightly more conservative than the ideal rule. It does
not sort. The entry path can open a region larger than 1,024 entries; the
existing per-region threshold has a 1,024-entry limit. Synthetic H100 checks
matched an independent frontier calculation, including a selected 1,025-entry
region, and checked scan sizes up to 513 groups. AMD synthetic checks passed
for the per-head and shared-KV paths; their entry-rule performance and quality
are not established by those checks.

The H100 full-model timings below use the same BF16 Llama 3.1 weights,
batch-one teacher-forced inputs, graph-captured decode, one warmup and two
measured repetitions. Each repetition has 512 untimed and 1,024 timed decode
steps. The decode window includes four cache catch-ups per layer; prefill is
timed separately. Two repetitions cannot establish the significance of a
small difference.

| Prompt tokens | Per-region c=16 | Per-entry | Per-region prefill | Per-entry prefill |
|---:|---:|---:|---:|---:|
| 32,768 | 7.817 ms/token | 7.997 ms/token | 1.663 s | 1.661 s |
| 65,536 | 8.302 ms/token | 8.255 ms/token | 4.195 s | 4.203 s |

[32K region](threshold-region-32768-guard-amortized.json) ·
[32K entry](threshold-entry-32768-guard-amortized.json) ·
[64K region](threshold-region-65536-guard-amortized.json) ·
[64K entry](threshold-entry-65536-guard-amortized.json).

The compacting step now scans fixed 32-group tiles once with a carried prefix,
then packs once, instead of having every tile rescan all prior tiles. Its work
is linear in the number of tiles. This change has not yet been established as a
speedup: the new 32K per-region time is 7.817 ms/token versus 7.745 ms/token in
the earlier measurement, which also differs in instrumentation and run.

The fresh per-region audit covers 32 examples each of multi-key 2, multi-key 3,
multi-value, multi-query, and QA 1. At the final decode step, averaged across
all layers and query heads, it opened respectively 805, 524, 460, 446, and
589 remote entries per head. Its maximum observed opened-entry count divided
by `16 sqrt(n)` was 3.153, 2.392, 1.225, 1.308, and 2.270 for those tasks.
This is a final-step sample, not a maximum over every generated token. The
first-response scores match the previous per-region run on all 160 examples;
19 full generated sequences differ. Paired entry-rule work measurements for
these tasks are in the full report.
[Per-region audit output](threshold-region-guard-ruler-32k-audit5.json).

The first three completed per-entry tasks are single-needle 1, 2, and 3:
100%, 100%, and 96.875% respectively, compared with 100% on all three for
the earlier per-region run. Single-needle 3 has one missed UUID in 32
examples. Multi-key 1 and multi-key 2 have also completed: 100% and 68.75%,
respectively, compared with 100% for each in the matched per-region control.
On multi-key 2, the final-step sample averages 277 opened remote entries/head
and 26.5% opened proxy mass for the entry rule, versus 805 entries/head and
27.9% proxy mass for the region rule. Multi-key 3 scores 71.875% versus 100%;
entry opens 305 entries/head and 32.9% proxy mass, versus 524 entries/head and
27.6% for the region control. These are partial results; seven of the 13 tasks
were pending when that partial comparison was made; full-panel results are
now in the linked report.

The bound covers opened **remote** entries per query head, not local and sink
attention, summary scoring, prefill, or cache updates. Prefill still uses its
own fixed top-eight routes and 1,024-entry region cap. The ordinary unsplit
centroid schedule grows approximately as `16 sqrt(T)`, and catch-up assigns
fixed-size overflow blocks to the current state, but page-table maintenance,
allocation, fallback paths, prefill, and the model's other layers also require
an audit. The timings include catch-up but do not constitute an end-to-end
asymptotic proof.

## Prefill and cache-update audit

For the H100 Llama adapter, prefill uses the production pool: it processes
16,384-token chunks with a 16,640-token local field, computes an exact initial
field, routes subsequent chunks by a separate top-eight rule, and builds the
semantic page cache. It does not use the experimental per-entry decode rule.
The fixed-size exact first field and local attention have substantial real cost;
the full prompt also incurs routing, clustering, and page construction.
The separately measured prefill times above include this work.

Decode catch-up occurs after fixed-size batches leave the recent field. Its
state update assigns the batch against the current centroids; on the ordinary
unsplit schedule their count grows like `sqrt(T)`. It also publishes page
ordinals, updates counts and summaries, and moves the remaining local suffix.
The generic page-ordinal path sorts each update batch by owner; page and leaf
storage can grow and copy, and overflow checks can scan slot lengths and
synchronize with the host. Alternate configurations include additional
splitting and sorting paths. The timed decode windows include four catch-up
events per layer, but those short windows do not characterize rare growth,
rehashing, or all fallback paths.

Consequently, the proof here establishes a per-query-head bound on *logical
exact remote entries* for decode and linear-work packing in the number of
route groups. It does not establish a deterministic subquadratic work bound
for complete prefill, cache construction, all amortized updates, or every
runtime configuration. In particular, the bounded opening count alone is not
evidence for such a claim.

## Experimental prefill thresholds

An opt-in H100 Llama BF16 prototype applies equal-region c=16 or the entry
rule to the remote part of each prefill query, while retaining exact local
and sink attention. Entry 16 and the entry-24 cutoff with the strict 16 cap
are included. Its variable-length routes use tiled PyTorch selection and the
existing grouped paged-leaf path; it is a correctness-first implementation,
not a production-speed kernel. CPU selection checks and the independent GPU
frontier check passed (maximum absolute output error 0.0034), including after
padding route lists to reusable compilation sizes. Three matched RULER smoke
tests, work audits, and warmed timing runs have completed. The remaining
not-yet-started 64K follow-ups were requeued after
these tests; the run ID mapping is recorded in
[threshold-requeue-map.json](threshold-requeue-map.json). The pilots are
deliberately limited and cannot establish a full-panel quality result.
An additional audit covers eight matched examples each on multikey 2 and
multikey 3, per prefill rule. All three variants scored 100% on both tasks.
On multikey 2, region c=16 opened 361.6 entries in 10.1 regions per remote
query head, versus 215.8 in 69.0 regions for entry 16 and 421.8 in 116.4
regions for entry 24 with cap 16. Region c=16 reached 2.095 times the
16 sqrt(n) reference bound; both entry variants stayed below their strict cap. Work is
averaged over remote prefill queries and all layers, excluding local and sink.
This remains a small, selected quality sample. See the
[prefill report](threshold-prefill-progress.md) for results and timing scope.

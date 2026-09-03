# Static small-centroid page-size-one decode (2026-08-25)

## Method

This experiment removes query-dependent routing from two-level LOD decode.
At each state update, it builds one persistent index list per KV head:

1. protected sink entries;
2. every exact leaf of a centroid with fewer than 16 leaves (implemented as
   `max_exact_leaves=15`), or one `log(count)`-biased coarse entry for a larger
   centroid; and
3. a fixed-address local-window suffix.

Each decoded token only prepares the current local K/V and context lengths,
runs one AITER-shaped page-size-one indexed attention scan, and reduces its
split-K outputs. There is no coarse scoring, top-k, union construction, or
mask construction. All speed prompts are distinct real ProLong documents;
results use batch 8, 64K prompts, 64 decoded timing tokens, and three repeats.

## Results

| model | NIAH-S3 screen | mean scan rows | exact remote leaves | static decode ms | historical full ms | best prior LOD ms |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 8/8 | 20,794 | 30.35% | 4.309 | 5.093 | 3.644 |
| Qwen3.8-27B-FP8 | 7/8 | 20,233 | 29.19% | 40.321 | 52.030 | 36.776 |
| Muse-Glimmer-30B | 8/8 | 17,741 | 26.63% | 21.198 | 19.242 | 21.453 |
| OLMo-3-1125-32B | 7/8 | 17,940 | 24.83% | 46.639 | 30.481 | 34.566 |
| Phi-4 (TP=5) | 0/8 | 6,357 | 18.48% | 6.603 | 9.970 | 9.682 |

Phi-4 full attention also scores 0/64 at 64K, so its 64K NIAH result is not a
static-cutoff regression signal. Gemma-4-26B-A4B uses D=512 on its eligible
global layers, outside this page-size-one kernel's current D=128/256 support;
it therefore fell back to ordinary LOD and is not included as a static result.

The static scan beats full attention on both Qwens and Phi, but not Muse or
OLMo. Removing routing is therefore insufficient by itself. This kernel still
launches one M16 attention row group per KV head: OLMo's GQA=5 underfills M16
and its eight KV heads create 64 independent batch-8 sequences and reductions.
Muse's GQA=16 fills M16, but its dense-attention baseline is efficient enough
that reducing the scanned rows does not recover the remaining LOD overhead.
The 7/8 Qwen3.8 and OLMo screens also show that a universal cutoff is not a
quality-safe replacement for query-dependent routing.

### Muse regression diagnosis

Muse is not a balanced-centroid counterexample.  At 64K, 70.94% of its
centroids have at most 16 leaves but contain only 26.63% of all leaves; the
remaining 29.06% contain 73.37%.  The compact list therefore scans about
17.7K entries rather than 64K.

Two earlier interpretations were wrong.  Muse's 13 global layers are NoPE
(`layer_rope_theta=0`); RoPE is applied only on its 39 native sliding-window
layers.  Muse's vLLM adapter also instantiates the dense `AfmoeMLP`, not
`AfmoeMoE`, so there is no data-dependent expert selection.  Zero-output
controls are therefore subtractable when their cache append is retained.

The first no-attention control incorrectly skipped the whole static function,
including its K/V append and `local_lens` update.  The corrected `prepare-only`
control performs the identical append and advances the cache, then omits only
QK, PV, and split reduction.  A first three-repeat pair measured 20.546 ms
prepare-only and 21.333 ms compute-then-zero.  The final scheduler-locked pairs
below supersede that 0.787-ms delta.  The eager phase profile remains useful
internally: prepare, indexed attention, and reduction average 6.89, 67.05, and
6.88 us/layer, respectively.

To remove vLLM's four-at-a-time prompt admission from the attribution, the
final controls used a 2K per-request threshold under the same 16K total prefill
budget, keeping all eight requests active together.  Results were:

| scheduler-locked control | ms/step |
|---|---:|
| native, no global attention, async on | 18.406 |
| native, global attention then zero, async on | 19.884 |
| CUSTOM delegates native, no LOD pool, async on | 18.298 |
| native, no global attention, async off | 18.809 |
| no LOD pool, forced tiny placeholder spec, async off | 20.458 |
| LOD prepare-only, tiny placeholder, async off | 21.208 |
| LOD compute-then-zero, tiny placeholder, async off | 22.301 |
| LOD prepare-only, tiny placeholder, diagnostic async on | 20.154 |
| LOD prepare-only, full-width bounded staging, async off | 19.122 |
| LOD compute-then-zero, full-width bounded staging, async off | 19.981 |

The native attention delta is 1.478 ms.  The locked LOD attention delta is
1.092 ms with the placeholder and 0.859 ms with full-width staging.  This
yields an exact decomposition of the 2.417-ms placeholder regression:

| contribution | delta |
|---|---:|
| authoritative LOD disables vLLM async scheduling | +0.403 ms |
| 1-head x 1-channel heterogeneous placeholder cache spec | +1.649 ms |
| remaining LOD runtime/pool work | +0.750 ms |
| static attention replacing native attention | -0.386 ms |
| **net** | **+2.417 ms** |

Direct timing of LOD's additional `runtime.preprocess` accounts for roughly
0.41--0.52 ms of the 0.750-ms pool term.  The static prepare kernel versus the
native cache write differs by only about 0.02 ms.  Merely selecting the CUSTOM
backend is neutral; the 18.298-ms delegate control matches native.

The dominant problem is therefore the tiny placeholder cache, not attention,
RoPE, centroid balance, or CUSTOM dispatch.  It creates a heterogeneous hybrid
cache specification and makes vLLM's per-step cache/attention metadata path
materially slower even when no LOD pool exists.  Setting
`VLLM_LOD_NATIVE_PLACEHOLDER_CACHE=0` retains a bounded 1024-token full-width
staging specification rather than a 64K chronological cache.  It increases
native cache allocation from 5.898 to 6.244 GB (+0.346 GB), while reducing
prepare-only latency by 2.086 ms.  Its directly measured compute-then-zero
latency is 19.981 ms versus 19.884 ms native, a remaining difference of only
0.097 ms.  In additive form, full-width staging pays +0.403 ms for safe
async-off scheduling and +0.313 ms for remaining runtime/pool work, then saves
0.619 ms in attention.

Forcing async scheduling on LOD is only a diagnostic: the general authoritative
implementation updates rows in place, so preparing the next model batch before
the current graph settles is not yet correctness-safe.  The full-width bounded
staging option is already supported and does not have that caveat.

For Qwen3.5-0.8B, 32, 64, and 128 split-K segments measured 4.309, 4.413, and
4.330 ms respectively (the 32-segment number includes the quality run). The
differences are noise-scale: the indexed attention work, not the reduction,
sets the runtime here.

## Cutoff schedule

A fixed 16 is not comparable across context lengths. With the state schedule
`N_state ~= 16 sqrt(T)`, the expected posting-list occupancy is

`T / N_state ~= sqrt(T) / 16`.

Thus a principled first schedule is to open centroids whose leaf count is below
`sqrt(T)/16`. For the implementation's inclusive integer cap, that is
`ceil(sqrt(T)/16) - 1`; it equals 15 at 64K. An even more adaptive equivalent
is the observed mean occupancy, `T_remote / N_nonempty`, optionally multiplied
by one global constant. This avoids making the 8K policy much denser than the
64K policy: fixed cap 15 opened 72.61% of remote leaves at 8K but only 30.35%
at 64K on Qwen3.5-0.8B.

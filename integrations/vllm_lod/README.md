# LOD Attention for vLLM

This out-of-tree plugin makes semantic LOD state the authoritative cache for
eligible global-attention layers. Initial and cached prefill build that state
directly, and decode advances the same fixed-address rows. Authoritative LOD
layers bind no chronological native attention K/V, including when vLLM prefix
caching is enabled. Prefix hashes come from token/block metadata; a hit resumes
an exactly matched retained LOD row. Recurrent/Mamba state and attention layers
that are not LOD-compatible remain native.

Completed LOD rows remain available for content-matched prefix reuse. When
vLLM resumes a prompt at a physical block boundary, the plugin verifies the
exact token prefix. A hit inside the unclustered tail is a metadata-only
rollback. For an older shared-prefix boundary, two-level LOD rebuilds its
semantic centroids from the row's chronological exact-leaf archive; it never
tries to invert clustered history. No native attention-cache fallback is used
for an external LOD layer.

External semantic ownership is the only supported cache mode. The plugin does
not expose the older dual-cache, bounded-staging, or 1x1/1x1x1 placeholder
paths, so a CUSTOM attention run cannot silently select their slower geometry.

The default `VLLM_LOD_PROFILE=production` path is the current flat BF16
implementation shared with the Hugging Face benchmark. Exact leaves are stored
once in a dense chronological pool; compact per-centroid page tables index that
pool. It opens all exact pages in each of the top-eight centroids. Production
attention settings are locked: setting any `VLLM_LOD_*` tuning variable causes
startup to fail rather than silently selecting a different implementation.
Only resource sizing through `VLLM_LOD_POOL_SIZE` and `VLLM_LOD_MAX_CONTEXT`
remains configurable. Historical kernels and tuning controls are available only
after the explicit research opt-in `VLLM_LOD_PROFILE=experimental`.
Production also rejects scheduler budgets below the resolved prefill chunk,
because smaller incoming slices change when the LOD state is updated. That
minimum is 16K for Qwen/K2 and 4K for Gemma; a common launch should use
`--max-num-batched-tokens 16384` or larger and either
`--long-prefill-token-threshold 0` or a threshold of at least 16384.

When GQA-union decode and its AITER path are enabled, one indexed M16/N64
attention call consumes exact leaves, the local window, protected sinks, and
unopened coarse entries from a shared page-size-one arena. Coarse entries carry
an FP16 logit bias. Every centroid occupies a fixed suffix position: opened or
inactive entries receive `-inf`, while unopened entries receive `log(count)`.
This preserves represented token mass without compacting a per-query centroid
list, a separate coarse value pass, or a final branch merge.

`VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER=1` selects the fixed-list variant. At
each 256-token state-update boundary it stores one persistent list per KV head:
local positions, sinks, every coarse position, then all valid leaves in
centroid-major order. Decode changes only an epoch-stamped opened-centroid mask.
A parallel mask-preparation kernel resolves owners and route epochs into one
byte per list entry and one byte per 64-entry block. The M16/N64 page-size-one
kernel first reads the block byte; a fully inactive tile exits without loading
its lane mask or K/V and without issuing QK or PV MFMA. Partially active tiles
retain the ordinary AITER-shaped online softmax, including `log(count)` on
unopened coarse entries. The production fixed scan uses 256 segments for D=256
heads and 128 for D=512 heads. K2's D=128/GQA8 and Qwen3.5-0.8B's D=256/GQA4
geometries instead build the compact union of the eight routes from each
related query head and feed only those exact leaves, the local field, sinks,
and unopened coarse entries to the same page-size-one AITER kernel. This avoids
scanning their nearly dense fixed owner masks. Qwen3.8's D=256/GQA6 geometry
retains fixed-mask scan because compact-union leaf attention is slower there.
Both paths require two-level BF16 GQA-union decode. The controls described below
are experimental-profile diagnostics and are not accepted by production.
`VLLM_LOD_DECODE_GQA_FIXED_MASK_BLOCK_N` selects the experimental fast-fail
tile width (16, 64, or 128; default 64), and
`VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS` selects 8--512 split segments
(default 128). For low-row decode,
`VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS=1` selects 256 segments when
the local rank has at most eight real query rows and 128 otherwise.
The two-position fixed-mask MTP path enables its measured adaptive geometry by
default: batch-1 MTP uses 256 segments and larger batches use 128. Set
`VLLM_LOD_SPECULATIVE_FIXED_MASK_ADAPTIVE_SEGMENTS=0` to disable this MTP-only
default and honor the ordinary fixed segment setting directly.
`VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D=64` splits the formerly
one-program-per-query-head FP32 segment reduction across output dimensions; it
is applied only in that same at-most-eight-row regime. The two settings are
intended to be enabled together. The split also applies to predicted-mass
decode: only D partition zero persists the remote-coarse LSE and advances the
next-token route epoch, while all partitions reduce disjoint output channels.
In the low-row regime, the fixed scan also automatically uses one warp and one
wave per EU; the batch-eight launch remains at the configured two/two default.
Fixed-mask top-eight decode uses direct route activation by default. Its route
programs update the persistent mask directly; duplicate writes are idempotent,
so the separate compact GQA-union launch and barrier are unnecessary at any
batch size. Set `VLLM_LOD_DECODE_GQA_FIXED_MASK_DIRECT_ROUTES=0` only for an
old-path control. Direct activation is exact and does not change centroid
selection.

The fixed-mask scorer reads BF16 centroid means already materialized in the
page-size-one arena at state-update boundaries. Authoritative state still
stores sums, but decode no longer repeats a component-wise division for every
query. Candidate scores and routes are bitwise identical to the former hot-path
division. At 64K/B1 on Qwen3.5-0.8B this reduced the stable median from 1.984
to 1.962 ms/step. FP8 mean storage is not enabled: it saved no time at the
4,096-centroid target and perturbed route candidates. Likewise, cached FP32
`log(count)` added a load without reducing score time. Details are in
`artifacts/decode_local_overlap_20260829/README.md`.

`VLLM_LOD_DECODE_ROUTE_COHORT=1` restricts dynamic top-k or mass-cutoff
routing to centroids in the same small-posting-list cohort used by the static
variant: an inclusive `max(16, ceil(sqrt(T) / 16))` leaf cap.  This policy
replaces `VLLM_LOD_DECODE_MAX_OPEN_LEAVES`; it is not intersected with that
legacy guard.  `VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP` remains an optional fixed
override for controlled single-length experiments.  `VLLM_LOD_OPEN_COUNT`
selects one through eight routes for top-k experiments.  Predicted-mass decode
instead uses `VLLM_LOD_DECODE_GQA_PREDICTED_MASS=1` and
`VLLM_LOD_DECODE_GQA_MASS_FRACTION`; it applies the current query to the
eligible centroids while reusing only the preceding token's total mass as the
cutoff denominator. On Qwen3.5-0.8B at 64K, `0.0625` (1/16) retained 64/64
NIAH-S3 and was useful only in the low-row regime; bounded top-eight remains
the general default because mass routing can form long-tail unions and did not
improve the batch-eight result.

`VLLM_LOD_DECODE_GQA_PILOT_Z=1` is the no-top-k threshold-routing experiment.
During prefill it calibrates each layer and query head from 64 sampled queries,
retaining the minimum standardized eighth-best centroid score. Decode scores
64 evenly spaced pilot centroids to estimate the current query's score mean
and standard deviation, then directly stamps every centroid above the
calibrated threshold into the fixed mask. It therefore uses the current query
without a global top-eight reduction or a preceding-token route. Set
`VLLM_LOD_DECODE_GQA_PILOT_Z_ROUTE_COUNT` to change the calibrated order
statistic (default 8; 128 was tested for adjacent-T/4 groups), and set
`VLLM_LOD_DECODE_GQA_PILOT_Z_MARGIN` to a nonnegative standardized-score
margin. Zero is the fastest measured pilot setting and scored 64/64 on 64K
NIAH-S3. After fusing its queue reset into the final reduction it measured
1.998 ms/step at 64K/B1, versus 1.984 ms for current top-eight: removing the
serial top-k is therefore nearly latency-neutral, but not yet faster on Qwen.
This option currently requires two-level BF16 GQA-union HIP decode, supports
both fixed-mask and compact selected-list consumers, and is mutually exclusive
with predicted-mass routing. The adjacent-T/4/route-128 transfer was negative:
at 64K/B1 the fixed-list form measured 2.253 ms/step and the compact form 3.537
ms/step, versus 1.998 ms for ordinary pilot LOD. The compact four-head union
selected about 11,062 of 15,424 live adjacent groups, so this is not a usable
N/4 default even though the fixed-list path passed an 8/8 chat-formatted 64K
NIAH-S3 smoke test; details are in
`artifacts/fixed_adjacent4_top128_20260828/README.md`.

`VLLM_LOD_DECODE_GQA_STATIC_LEAF_AITER=1` selects the routing-free compact
variant. At a state update it builds one persistent page-size-one list per KV
head: sinks, every leaf whose centroid count is at most
the active cap, one `log(count)`-biased coarse entry for each larger centroid,
and the fixed local-window suffix. By default the inclusive cap for a request
of length `T` is `max(16, ceil(sqrt(T) / 16))`. Set
`VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP` to use a fixed experimental override, or
`VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP_MIN` to change the default floor.
`VLLM_LOD_STATIC_LEAF_CAP_DIVISOR` changes the shared prefill/decode divisor
(default 16), so a floor of 32 and divisor of 8 select
`max(32, ceil(sqrt(T) / 8))`. `VLLM_LOD_STATIC_COHORT_NEVER_READMIT=1` makes
cohort eviction terminal: a centroid that ever exceeds the active cap remains
coarse-only even if a later cap increase would otherwise re-admit its leaves.
Decode
then performs a single indexed attention scan over that list; it has no
coarse-score, top-k, union, or mask-construction dependency. It shares the
split count selected by `VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS` and requires
two-level BF16 GQA-union decode with
`VLLM_LOD_DECODE_GQA_UNION_HIP=1`.

## Best measured 64K batch-8 prefill and decode

This is the canonical speed table as of 2026-08-27. It compares full attention
with the fastest measured **quality-conscious production configuration** for
two-tier and recursive three-tier LOD. Static cohorts and route variants that
won only a speed screen are excluded. In particular, unrestricted
query-dependent top-eight remains the two-tier decode reference on every
model, including Phi, Muse, and OLMo.

All rows use context 65,536, batch 8, eight distinct real ProLong documents,
16,384 maximum batched prefill tokens, a 64-token prompt reserve, BF16 LOD
state, and warm runs. Prefill is elapsed seconds for all eight prompts. Decode
is milliseconds per batch-eight decoding step. Parentheses report
`full-attention time / LOD time`; values below 1.0 mean that LOD is slower.
Bold marks the fastest of full, two-tier, and three-tier in that row.
Phi uses TP5; the other rows use TP1. Qwen, Gemma, Phi, and Muse use their
validated chat formatting (plus Muse's native text configuration), while OLMo
uses its raw base-model prompt.

### Prefill

| model | full attention | best two-tier | best three-tier |
|---|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | 9.909 s | 5.168 s (1.92x) | **4.323 s (2.29x)** |
| `Qwen/Qwen3.8-27B-FP8` | 110.565 s | **65.062 s (1.70x)** | 66.912 s (1.65x) |
| `google/gemma-4-26B-A4B-it` | 40.063 s | **18.110 s (2.21x)** | 19.175 s (2.09x) |
| `microsoft/phi-4` (TP5) | **28.119 s** | 43.216 s (0.65x) | 34.788 s (0.81x) |
| `meta-models/Muse-Glimmer-30B` | 51.933 s | **51.514 s (1.008x)** | 54.009 s (0.962x) |
| `allenai/Olmo-3-1125-32B` | **67.892 s** | 69.286 s (0.980x) | 74.795 s (0.908x) |

The two-tier prefill rows use routed top-three selection. Qwen3.8 and OLMo use
native-GQA coarse packing; Muse uses the hierarchical selector plus branch
overlap; Phi uses the current automatic spherical geometry and hierarchical
selector; Gemma retains its D=512 path. Qwen3.5-0.8B's later kernel changes do
not dispatch on its two-tier geometry, so its latest applicable routed result
remains 5.168 seconds.

The selected three-tier rows use hierarchical prefill selection and the
current automatic route backend. Qwen3.5-0.8B uses local/LOD overlap and
re-split routing. Qwen3.8 uses re-split routing. Gemma, Muse, and OLMo retain
the fused route. Phi uses 4,096-token updates and the D128/GQA4 expert/MFMA
complete-centroid prefill consumer; its decode still performs ordinary
recursive page routing.

### Decode

| model | full attention | best two-tier top-8 | best three-tier top-8 |
|---|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | 5.822 ms | 3.077 ms (1.89x) | **2.369 ms (2.46x)** |
| `Qwen/Qwen3.8-27B-FP8` | 52.030 ms | 36.334 ms (1.43x) | **34.883 ms (1.49x)** |
| `google/gemma-4-26B-A4B-it` | 11.694 ms | 10.235 ms (1.14x) | **9.365 ms (1.25x)** |
| `microsoft/phi-4` (TP5) | **9.970 ms** | 11.198 ms (0.89x) | 9.998 ms (0.997x) |
| `meta-models/Muse-Glimmer-30B` | 19.215 ms | 19.153 ms (1.003x) | **19.133 ms (1.004x)** |
| `allenai/Olmo-3-1125-32B` | 30.481 ms | **28.769 ms (1.06x)** | 29.125 ms (1.05x) |

Two-tier decode is unrestricted top-eight with the page-size-one HIP/AITER
final scan. Current automatic dispatch uses the segmented route producer only
on Muse and the grouped producer elsewhere. Three-tier uses re-split routing
on Qwen3.5-0.8B, Qwen3.8, and Phi, and fused routing on Gemma, Muse, and OLMo.
The faster re-split speed screens for Gemma and OLMo are not promoted because
they regressed their matched quality checks.

The quality record remains part of this table's interpretation. Qwen3.5-0.8B,
Qwen3.8, and Muse reached 64/64 NIAH-S3. Gemma two-tier scored 62/64; the
three-tier fused route corrected the single miss observed in its 63/64
re-split run on a matched block. OLMo two-tier and selected three-tier both
scored 54/64 versus full attention's 64/64, so it retains a broader LOD quality
gap. Phi's NIAH-S3 task is not discriminative because full attention also
scores 0/64; its corrected three-tier prefill was instead checked with ProLong
CE loss and was neutral within 0.013%.

The full-attention controls are the latest accepted matched records. The
Qwen3.5-0.8B control is the current seven-repeat AITER run. The five larger
models reuse the historical native-attention controls because the subsequent
cache and custom-kernel fixes do not execute in native full attention. OLMo
uses the newer eight-document 67.892-second control, not the older
68.537-second panel value.

### Phi-4 TP1 diagnostic

A matched TP1 rerun uses a 4,096-token scheduler aggregate because the current
QH40 LOD selector workspace does not fit the canonical 16,384-token aggregate.
With that aggregate applied to every row, full / two-tier / three-tier prefill
is **56.095 / 228.145 / 189.468 seconds**, while decode is **14.796 / 12.826 /
14.179 milliseconds**. Thus two-tier wins TP1 decode by 1.154x, but native full
attention wins prefill by 4.07x over two-tier and 3.38x over three-tier. The
full configuration and raw records are in
`artifacts/phi_tp1_tier_panel_20260827/README.md`.

This is a speed diagnostic only. Phi-4 has a native 16K position limit without
configured RoPE scaling, so its 64K quality output is non-discriminative. The
other large-model rows in the canonical table (Qwen3.8, Gemma, Muse, and OLMo)
are TP1; Phi was the only TP5 row.

### TP4 parallelism diagnostic

Matched 64K/B8 TP4 runs confirm that splitting the attention heads over four
GPUs narrows LOD's relative advantage without reversing it. Qwen full / two /
three prefill is **43.696 / 32.233 / 32.608 seconds**, and decode is **26.111 /
23.036 / 22.213 milliseconds**. Gemma full / two / three prefill is **15.793 /
10.406 / 10.146 seconds**, and decode is **12.432 / 11.264 / 10.812
milliseconds**. Qwen's two-tier decode speedup therefore falls from 1.43x at
TP1/B8 to 1.13x at TP4/B8; Gemma's falls from 1.14x to 1.10x.

Higher batch restores the missing per-GPU parallelism. In a synchronized
TP4/B32 Qwen decode test, full attention takes **47.226 ms** per batch step and
two-tier takes **28.096 ms**, a **1.68x** LOD speedup. The LOD audit reports all
32 requests on every eligible layer. Comparing this synchronized prefix-cache
replay with the cold B8 panel is directional because the Qwen Mamba cache mode
also changes. The detailed methodology, collective and full-backend caveats,
and raw artifact names are in
`artifacts/tp4_tier_panel_20260827/README.md`.

### TP1 batch-size and occupancy diagnostic

A fresh cold-prefill batch-one context panel for Qwen3.5-0.8B is recorded in
`artifacts/batch1_established_20260829/README.md`. With real ProLong prompts and
16K chunking, full / two-tier / recursive-three-tier prefill at
8K, 16K, 32K, 64K, and 128K is respectively
`0.059/0.075/0.068`, `0.153/0.157/0.149`, `0.413/0.338/0.306`,
`1.285/0.733/0.649`, and `4.421/1.599/1.450` seconds. Matched decode is
`1.833/1.768/1.698`, `1.936/1.792/1.708`, `2.083/1.839/1.743`,
`2.331/1.893/1.766`, and `2.840/2.036/1.795` milliseconds per token. Thus the
primary two-tier prefill crosses over between 16K and 32K, while recursive
three-tier is at parity by 16K and reaches 3.05x prefill and 1.58x decode
speedup at 128K. The recent overlap, staged-attention, pilot-threshold,
FP8-routing, and centroid-major experiments were disabled.

The matched Qwen3.8-27B-FP8 TP1/B1 cold-prefill panel in the same artifact
finds that two-tier, rather than recursive three-tier, is fastest at every
measured point. Full / two-tier prefill from 8K through 128K is
`0.968/0.955`, `2.154/1.968`, `5.231/4.085`, `14.046/8.553`, and
`42.534/18.083` seconds. Full / two-tier decode is `28.683/28.522`,
`29.405/28.651`, `30.047/28.668`, `31.404/28.877`, and `34.181/29.251`
milliseconds. Two-tier is therefore approximately tied at 8K and reaches
2.35x prefill and 1.17x decode speedup at 128K. Recursive three-tier is close
in prefill but adds roughly 0.4--0.7 ms of decode overhead on this model.

The matched TP1/B8 panel intentionally uses a 16,384-token aggregate scheduler
budget across the entire batch. At 8K/16K/32K/64K/128K, full versus two-tier
prefill is `7.553/7.584`, `17.218/15.926`, `41.812/33.052`,
`112.544/69.120`, and `341.448/146.699` seconds. Full versus recursive
three-tier decode is `37.929/35.241`, `41.399/35.276`, `46.049/35.319`,
`54.956/35.382`, and `71.781/35.542` milliseconds per batch step. Thus
two-tier reaches 2.33x prefill speedup and three-tier reaches 2.02x decode
speedup at 128K. The complete combined B1/B8 tables and raw record names are
in `artifacts/batch1_established_20260829/README.md`.

A synchronized TP1 64K sweep on Qwen3.5-0.8B confirms that per-rank parallelism
is the limiting variable. Two-tier speedup over full attention rises from
1.08x / 1.34x / 1.73x to 2.22x at batch 1 / 2 / 4 / 8; recursive three-tier
rises from 1.24x / 1.61x / 2.09x to 2.57x. Prompts are distinct real ProLong
documents, not repeated synthetic text.

The fixed attention scan itself benefits from more splits at B1, but its old
reducer handled the complete `[segments, D]` FP32 field in one program per
query head. The new split-D64 reducer makes the scan/reduction pair parallel:
on the same GPU, Qwen3.5-0.8B two-tier B1 falls from 2.234 to 2.095 ms, 6.2%
faster, and is 1.11x faster than the 2.325-ms full-attention result. It is
numerically equal within 1.1e-8 and scored 8/8 on 64K NIAH-S3. Exact
partitioned and tiled top-8 alternatives were 26% and 35% slower than the
existing 9.7-us reducer, so they were not retained.

Low occupancy is determined by the number of independent KV scans as well as
the raw query-row count. Qwen3.8-27B-FP8 B1 has four KV scans at D=256/GQA6 but
24 query rows, so the original eight-query-row rule incorrectly left it on the
128-segment serial-reducer path. The B1 correction enables 256 segments,
split-D64, and direct fixed-route activation when `batch == 1` and local KV
scans are at most four. A matched seven-repeat 64K real-ProLong run falls from
29.240 to **28.717 ms/step** (1.79%); the matched 31.571-ms full-AITER result
makes optimized two-tier 1.099x faster. Full methodology and raw artifact names
are in `artifacts/batch_parallelism_20260827/README.md`.

The exact common LOD settings for the two-tier decode reference are installed
by the production profile. A normal launch should specify only resource sizes:

```bash
VLLM_LOD_POOL_SIZE=8
VLLM_LOD_MAX_CONTEXT=131200
VLLM_LOD_PROFILE=production
```

For gfx942 D=256/GQA4 or GQA6 layers using grouped 32-centroid score-only routing,
`VLLM_LOD_DECODE_CENTROID_MAJOR_HIP=1` selects the low-row centroid-major HIP
scorer. It stages the GQA queries in LDS, loads each centroid K vector once for
all scores, preserves production's BF16 mean-key rounding, and emits
the same eight candidates per route block. With fixed-mask attention it also
performs prefix-mask maintenance and old-route clearing in the score grid, so
no preparation launch is added. Other dimensions, GQA ratios, route widths,
and non-score-only paths fall back to the existing Triton implementation.
Execution is reported as `decode_centroid_major_hip_{configured,executed}` in
the decode audit and as `centroid_major_vector` in the dispatch manifest.
The option remains experimental and opt-in. In particular, Qwen3.8's GQA6
mapping was 0.41% slower than the prior path in repeated 64K/B1 vLLM runs even
though it was faster in eager microbenchmarks; CUDA-graph replay removes that
apparent launch-overhead advantage. The GQA4 Qwen3.5-0.8B B1 path retained a
2.71% end-to-end gain.

The production profile hard-codes unrestricted top-eight decode, a 1024-leaf
guard, no route cohort or predicted-mass approximation, and fixed-mask AITER.
None of those settings should be repeated in a production launch.

Leave `VLLM_LOD_DECODE_HIERARCHICAL_ROUTE` unset so geometry dispatch selects
Muse's validated segmented route and the grouped route on the other rows.
Production always uses the current automatic spherical/coherence-aware policy.
The archived `raw` geometry is available only after selecting the experimental
profile explicitly.

`VLLM_LOD_ROUTING_POSITIVE_DOT_STATS=1` is a diagnostic-only switch that
records positive routing-dot density and the number of 64-centroid tiles with
fewer than `k` positive entries. The extra reductions are not intended for
timing runs. Setting `VLLM_LOD_ROUTING_CUTOFF_STATS_MIN_STATE` to a positive
state length additionally simulates periodically reused top-n score cutoffs
near that state size. `VLLM_LOD_ROUTING_CUTOFF_STATS_ROUTE_COUNT` restricts
that simulation to the requested routing width. Setting
`VLLM_LOD_ROUTING_CUTOFF_STATS_NORMALIZATION` accepts `raw`, diagnostic-only
full-state `lse`, `pilot64_lse`, or `pilot64_z`. The pilot variants normalize
against 64 evenly sampled centroid scores; `pilot64_z` uses their mean and
standard deviation. They model a practical one-tile calibration rather than a
second full coarse scan. The report includes both single-query refresh periods
and cutoffs derived from the minimum/low quantiles of the preceding
state-update segment. These diagnostics perform repeated population scans and
their wall-clock times are invalid as speed measurements.

The optional static 64K speed diagnostic instead adds:

```bash
VLLM_LOD_PROFILE=experimental
VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER=0
VLLM_LOD_DECODE_GQA_STATIC_LEAF_AITER=1
VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP=16
```

Leaving `VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP` unset selects the length-aware
inclusive `max(16, ceil(sqrt(T) / 16))` schedule; it also evaluates to 16 at
64K. This mode is not part of the high-quality reference.

The archived raw-routing top-eight panel measured 36.247 ms on Qwen,
10.421 ms on Gemma, 10.913 ms on Phi, 19.349 ms on Muse, and 28.878 ms on
OLMo. The current table instead uses the later matched automatic-geometry
route-only rerun. It found grouped/segmented latencies of
36.334/36.478 ms on Qwen, 10.235/10.300 ms on Gemma, and
11.198/11.163 ms on Phi.  Muse's historical 19.349-ms fast path already used
the segmented schedule, and the current rerun is 19.153 ms.  OLMo measured
28.769/28.436 ms, but its requested tuned arm retained a one-tile grouped
producer and only changed grouped tile/reducer geometry; it is not a
hierarchical-route comparison.  Automatic segmented routing is therefore
limited to Muse.  It is exact with respect to the top-eight route set, but it
is not a speed win on Qwen or Gemma and its Phi delta is only 0.32%.

The reference artifacts are:

- `artifacts/static_vs_top8_30b_20260825/README.md` for the large-model full
  controls and archived two-tier controls;
- `artifacts/static_prefill_20260825/README.md` for Qwen3.5-0.8B two-tier
  prefill and `artifacts/cohort_routing_20260825/README.md` for its decode;
- `artifacts/prefill_route_hierarchical_20260826/README.md` and
  `artifacts/prefill_direct_gqa_20260826/README.md` for the current large-model
  two-tier rows; and
- `artifacts/three_tier_refresh_qwen08_20260826/README.md`,
  `artifacts/three_tier_resplit_family_20260826/README.md`, and
  `artifacts/three_tier_phi_prefill_20260826/README.md` for three-tier.

### Memory-balanced precision policy

The speed table above remains BF16 so it compares the latency-first paths.
For Qwen, Gemma, and Muse recursive three-tier serving,
`VLLM_LOD_KV_BITS=8` is the validated memory-balanced format. At 64K/B8 it
reduces persistent LOD cache by 42.7% on both Qwen models, 43.3% on Gemma, and
41.6% on Muse. Relative to matched BF16 runs, decode changes by +1.0% on
Qwen3.5-0.8B, +0.6% on Qwen3.8-27B, -0.7% on Gemma, and +0.2% on Muse. Prefill
changes by -5.2%, +4.2%, +8.7%, and +0.05%, respectively. Gemma should
therefore retain BF16 when prefill latency is the priority and use INT8 when
capacity is the priority; Muse INT8 is tied with BF16 for both phases.

Qwen3.5-0.8B retained 64/64 NIAH-S3 in both formats and its eight-example 8K
ProLong CE was 1.923225 BF16 versus 1.923071 INT8. Gemma retained 8/8 in the
matched 64K NIAH-S3 smoke check, and Qwen3.8 INT8 also passed its 8/8 smoke
check. Muse INT8 retained the full 64/64 score of its BF16 reference. Enable
the memory-balanced recursive path explicitly:

```bash
VLLM_LOD_PROFILE=experimental
VLLM_LOD_LEVELS=3
VLLM_LOD_KV_BITS=8
VLLM_LOD_ROUTING_GEOMETRY=auto
```

The current fastest two-tier fixed-list page-size-one path remains BF16. Its
persistent metadata does not yet accept INT8; generic two-tier INT8 is only
0.8% slower than generic BF16 in matched decode, but falls onto a pathway that
is 33% slower than fixed-list BF16. Expanding INT8 into BF16 before attention
would add a full-cache read and write, lose the bandwidth saving, and either
consume the saved memory persistently or reintroduce per-query compaction.
Direct INT8 support in the fixed-list consumer is the appropriate future
two-tier change.

A hindsight profile over every prefill query in one real 64K Qwen3.5-0.8B
ProLong document found that 92.77% of leaf bytes belonged to centroids opened
at least once. Evicting the remaining 7.23% cannot be decided safely in
advance; compressing only that subset from INT8 to INT4 would save at most a
further 3.6% of leaf bytes before mixed-format overhead. Cold-centroid
eviction is therefore not enabled. Full paired measurements and the profile
are in `artifacts/int8_best_tiers_20260827/README.md`.

The route-pair
harness `scripts/run_vllm_lod_decode_route_pair.sh` fails after a run unless
the audit confirms fixed-mask execution, HIP execution, page-size-one final
attention, and the requested grouped or segmented route producer.

Treat this section as a release record and keep both tables current. A future
configuration replaces a row only after a matched real-text B8 rerun, a quality
result appropriate for that model, and a dispatch audit proving that the
intended kernels executed. Update the date, value, policy description, quality
note, and artifact pointer in the same commit. Record the checkpoint, TP,
prompt formatting, all non-default environment overrides, and timing protocol
in the artifact. Do not infer the active kernel from requested flags, combine
the best individual repetition from different runs, or promote an isolated
kernel, static cohort, or speed-only route result into these tables.

## Install and run

Install the plugin editable into the environment that provides vLLM:

```bash
uv add --editable /absolute/path/to/code/integrations/vllm_lod
```

The integration is currently validated against vLLM 0.27.1 for ROCm. K2
Horizon support landed upstream after vLLM 0.28.0, so the plugin includes the
upstream `K2HorizonForCausalLM` implementation from
[vLLM commit `1f76efaa`](https://github.com/vllm-project/vllm/commit/1f76efaa2195485b92cb04215aba6fb8f5fe523d)
for older installations. It registers that backport lazily and automatically
uses vLLM's native model when a newer installation already provides it. Thus
`IFM/K2-Horizon-0.9B` works with the same ordinary and `CUSTOM` LOD launch
commands below. The published checkpoint currently requires vLLM's
`--trust-remote-code` option to load its configuration.

### Reuse loaded weights while developing

The package also registers an `ipc_cache` model loader. A long-lived, GPU-light
broker loads and post-processes each exact model/TP/PP configuration on its
first request, then fresh vLLM workers map the retained final parameters,
buffers, and tensor attributes through CUDA/HIP IPC. The client constructs the
module tree on `meta`, so it neither rereads the checkpoint nor briefly
allocates a second copy of the weights.

The first vLLM process using `--load-format ipc_cache` automatically starts the
broker if it is not already running. Startup is serialized by an owner-only
filesystem lock, so concurrent TP ranks share one broker. The broker then stays
alive for subsequent vLLM processes on that node and GPU allocation. To choose
non-default eviction limits up front, it can still be started manually:

```bash
VLLM_PLUGINS=lod_attention \
vllm-weight-cache --cache-id dev
```

Fresh LOD or ordinary vLLM processes on the same node and physical GPUs can
then use it as follows:

```bash
VLLM_PLUGINS=lod_attention \
VLLM_WEIGHT_CACHE_ID=dev \
vllm serve MODEL \
  --tensor-parallel-size 8 \
  --load-format ipc_cache \
  --attention-backend CUSTOM
```

Use `VLLM_PLUGINS=weight_cache` instead when testing normal attention without
installing the LOD hooks. `VLLM_WEIGHT_CACHE_DIR` selects a non-default
owner-only socket directory. Equivalent per-run settings can be passed through
`--model-loader-extra-config '{"cache_id":"dev","cache_dir":"..."}'`. The
`auto_start` setting defaults to true; set it to false or export
`VLLM_WEIGHT_CACHE_AUTO_START=0` when an absent broker should be an error. Broker
startup logs are written to `broker.log` in the selected cache namespace. The
client sends its exact vLLM configuration to the broker; the backing load
defaults to `auto` and can be changed with `backing_load_format` and
`backing_loader_extra_config` in that same object. A fingerprint mismatch or an
incomplete meta mapping is a hard error rather than a silent disk fallback.

The broker tracks live vLLM worker PIDs as leases. Its default cache budget is
60% of each GPU's memory; after a miss it LRU-evicts only resident models whose
workers have exited. Use `--max-cache-fraction` or
`--max-cache-gb-per-gpu` to change that budget. If an uncached model hits OOM
while inactive models are resident, the broker evicts them and retries once.

The daemon must remain alive for the lifetime of every mapped engine. Check or
stop one cache namespace with:

```bash
vllm-weight-cache status --cache-id dev
vllm-weight-cache stop --cache-id dev
```

On `cluster-run`, keep the daemon as a detached job and use `--overlap-own` for
development jobs that must share its allocated GPUs. The `ipc_cache` plugin
accounts for weights that were resident before vLLM's normal memory snapshot,
so ordinary `--gpu-memory-utilization` settings continue to include the mapped
model weights. On a cold miss, an LRU eviction may legitimately increase free
memory after vLLM takes that snapshot. The loader rebases only the byte range
reported by the broker as evicted; a larger unexplained release still trips
vLLM's ordinary concurrent-process guard. The broker is single-node and DP=1;
TP and PP are supported and discovered from each requesting vLLM engine.

The repository's current vLLM benchmark, quality, prefix-cache, chat-batch, and
NIAH panel entry points select `ipc_cache` by default. They share the namespace
from `VLLM_WEIGHT_CACHE_ID` (`dev` in the shell launchers). Set
`VLLM_WEIGHT_CACHE_LOAD_FORMAT=auto` only for an explicit uncached control.

Then select the registered custom backend:

```bash
VLLM_PLUGINS=lod_attention \
VLLM_LOD_PROFILE=production \
VLLM_LOD_POOL_SIZE=8 \
vllm serve MODEL \
  --attention-backend CUSTOM \
  --kv-cache-dtype bfloat16 \
  --max-num-batched-tokens 16384 \
  --long-prefill-token-threshold 16384 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --max-num-seqs 8
```

Prefix caching does not allocate chronological K/V for authoritative LOD
layers. The retained LOD-row pool must be large enough for the live requests
and prefixes that should remain reusable.

`VLLM_LOD_POOL_SIZE` is the number of simultaneous or retained request rows on
each worker. It defaults to 8 and must be at least `--max-num-seqs`. The pool is
independent of vLLM's stable request indices; a persistent device indirection
table maps active batch rows to LOD rows without changing captured tensor
addresses. Graph padding never borrows a scheduled row. It may temporarily use
an unscheduled or completed row after restoring that row's real tail length;
the dummy append is placed just beyond the retained tail and discarded before
the row is observed again.

The optional `VLLM_LOD_MAX_CONTEXT` caps each LOD row. It defaults to vLLM's
`max_model_len`. Production fixes 256-token state chunks, a 512-token decode
local window, a `16 * sqrt(T)` state schedule, dense BF16 leaves, top-eight
GQA-union decode, and automatic spherical/coherence routing. Qwen3.8 and Gemma
use the persistent fixed-mask union; Qwen3.5-0.8B's D=256/GQA4 and K2's
D=128/GQA8/KV8 heads use the compact selected union. Qwen uses top-three 16K
exact-first prefill, Gemma's D=512 heads use their validated top-three 4K
schedule, and K2 uses top-four 16K exact-first prefill. These choices are
resolved from attention geometry and audited during pool construction. The
remaining tuning discussion in this document describes the explicit
`VLLM_LOD_PROFILE=experimental` research surface.
Routed two-level D=128/GQA=16 prefill automatically overlaps its independent
coarse, exact-leaf, and exact-local branches on separate GPU streams after
routing. Static-cohort prefill is excluded because keeping its larger AITER
exact-leaf working set live beside materialized coarse logits exceeds the
normal transient-memory allowance.
Recursive Qwen3.5-0.8B prefill overlaps only its independent local branch with
the dependent coarse/page LOD branch; the final merge explicitly waits on that
stream even when recursive page attention takes its dense-page early return.
`VLLM_LOD_PREFILL_OVERLAP_COARSE_LEAF=0` and
`VLLM_LOD_PREFILL_OVERLAP_LOCAL_LOD=0` disable the respective overlaps for
diagnostics; setting either to `1` enables it on other supported geometries.
Top-three prefill routing uses an exact two-stage selector on the measured
large-model geometries where it improves or preserves end-to-end speed:
independent wide centroid tiles emit their local candidates, then a small
reduction selects and orders the global three. The automatic geometry set is
D128/GQA16/KV2, D128/GQA5/KV8, D128/GQA4/KV2, D256/GQA6/KV4, and
D512/GQA8/KV2. The former selector remains automatic elsewhere. Qwen3.5-0.8B's
D256/GQA4/KV2 geometry remains on the former selector for two-level prefill,
where the extra launch loses, but uses the exact two-stage selector for
recursive three-tier prefill, where it improves 64K/B8 latency by 5.4%. Set
`VLLM_LOD_PREFILL_HIERARCHICAL_ROUTE={0,1}` to override automatic dispatch.
`VLLM_LOD_PREFILL_HIERARCHICAL_ROUTE=0` or `1` overrides geometry selection;
both selectors use the same scores and return the same ordered routes.
Expert-major BF16 prefill normally radix-sorts route rows by selected centroid.
The measured D256/GQA4/KV2 two-level top-three geometry instead constructs
exact expert buckets with a histogram and prefix scatter. On Qwen3.5-0.8B this reduces the
isolated top-three exact-leaf stage by 12.2% at B1 and 8.2% at B8, and reduces
a matched real-ProLong 64K/B8 prefill from 4.168 to 4.128 seconds (0.96%). It
does not change selected leaves or attention math and passed 8/8 64K NIAH-S3.
This is deliberately not automatic on Muse: its concentrated routes make the
atomic count/scatter 4.1x slower than radix dispatch in the production vLLM
profile. Set `VLLM_LOD_PREFILL_DIRECT_EXPERT_BUCKETS=0` to disable the measured
Qwen policy or `1` to force it for a BF16 expert-layout diagnostic. The full
route-concentration and overlap-stage analysis is in
`artifacts/prefill_direct_buckets_20260828/README.md`.
The coarse-attention consumer independently folds native GQA directly into
the matrix-row dimension on the measured irregular-ratio winners:
D128/GQA5/KV8 uses M128/N16/W8, and D256/GQA6/KV4 uses M64/N16/W8. This avoids
padding the head group to the next power of two and is automatic for both
two-level and recursive LOD. Power-of-two GQA4/GQA16 already fill their former M64 tiles and retain
that path; the D512/GQA8 GEMM alternative also remains unchanged because a
packed-PV experiment regressed end-to-end Gemma prefill. Set
`VLLM_LOD_PREFILL_COARSE_DIRECT_GQA=0` to disable automatic direct packing, or
set it to `1` and use `VLLM_LOD_PREFILL_COARSE_GROUPED_ROWS`,
`VLLM_LOD_PREFILL_COARSE_BLOCK_N`, and
`VLLM_LOD_PREFILL_COARSE_NUM_WARPS` for an explicit diagnostic geometry. The
64K/B8 production A/B and dispatch audit are in
`artifacts/prefill_direct_gqa_20260826/README.md`.
`VLLM_LOD_PREFILL_STATIC_LEAF_AITER=1` replaces query-dependent prefill top-k
with the static small-centroid cohort. After each 4,096-token state catch-up it
rebuilds one page-size-one AITER list per KV head containing every leaf whose
centroid has at most `max(16, ceil(sqrt(T) / 16))` leaves; the list stays fixed
for that query chunk. Larger centroids remain represented by their biased
coarse entries. `VLLM_LOD_PREFILL_STATIC_LEAF_CAP_MIN` changes the floor, and
`VLLM_LOD_STATIC_LEAF_CAP_DIVISOR` changes the shared schedule divisor. This
experimental prefill mode requires two-level BF16 dense leaf storage.
It shares the native-GQA coarse consumer above. At 64K/B8 this reduces matched
static Qwen3.8 prefill from 79.422 to 69.648 seconds (-12.31%) and static OLMo
from 72.053 to 70.901 seconds (-1.60%). These current numbers, the superseded
original panel, the dispatch audit, and a static phase profile are recorded in
`artifacts/static_prefill_recent_20260826/README.md`.
`VLLM_LOD_PREFILL_ROUTE_COHORT=1` instead retains dynamic routing but restricts
eligible routes to the same scheduled small-centroid cohort; larger centroids
remain in the count-corrected coarse residual. The production prefill route
count remains `min(3, VLLM_LOD_OPEN_COUNT)`. Set
`VLLM_LOD_PREFILL_OPEN_COUNT=1..8` only when an explicit prefill route count is
needed. At 128K, the enlarged floor-32/divisor-8 cohort was quality-negative
when combined with top-k, even with literal prefill top-8; results are in
`artifacts/static_cohort_top8_20260826/README.md`.
`VLLM_LOD_LEAF_SEAL_CAPACITY` is an opt-in diagnostic; it is unset by default,
and the reported two-tier benchmarks retain every leaf. The two-tier cache accepts
`VLLM_LOD_KV_BITS=0` (BF16) or `8` (signed INT8 K/V with one BF16 scale per
token); its routing state and page summaries remain BF16.
`VLLM_LOD_DENSE_LEAF_STORAGE` defaults to true and
removes the physical page-fragmentation overhead without discarding leaves.
`VLLM_LOD_PREFILL_INT8_ROUTE_MMA=1` enables the experimental Sage-style
fused centroid QK path: queries and centroid keys are scaled per row, QK uses
INT8 MMA, and route selection plus stable coarse attention consume the scores
without materializing a query-by-centroid tensor. It remains opt-in because
the current Triton value accumulator reduces occupancy on gfx942; the regular
materialized route GEMM is faster in the measured Qwen3.5-0.8B configuration.
For leaf attention, the Sage-style path quantizes Q and uses INT8 QK MMA.
Below 64K it dequantizes V for BF16 PV MMA; at 64K and above it also quantizes
the probabilities for INT8 PV because the longer posting-list scan amortizes
that fixed work. `VLLM_LOD_PREFILL_INT8_PV_MMA=0` or `1` overrides this
automatic crossover. Batch-8 INT8 leaf kernels use two warps by default; one
warp under-occupies the true batched workload.

### Validated recursive prefill refresh

On Qwen3.5-0.8B at 64K/B8, recursive BF16 LOD retains its historical
1,280-token state-update cadence. Moving updates to 4,096 tokens was 1.4%
slower in the initial three-repeat screen, so that change was rejected. With
seven measured repetitions over eight distinct real ProLong documents, exact
two-stage route selection reduced median prefill from 4.646 to 4.396 seconds;
overlapping the independent local branch reduced it further to 4.348 seconds.
The matched AITER full-attention result was 9.909 seconds. Decode was
2.657 ms per batch step for the final LOD run versus 5.822 ms for full
attention. The automatic final configuration scored 8/8 on chat-formatted
64K NIAH-S3, and its dispatch audit recorded the hierarchical selector,
recursive indexed residual-page kernel, AITER local attention, local/LOD
overlap, and the unchanged 1,280-token update length. Results and all raw
samples are in `artifacts/three_tier_refresh_qwen08_20260826/README.md`.

Replacing the grouped recursive state-route kernel with the allocation-free
re-split implementation subsequently reduced steady batch-8 decode from
2.657 to 2.369 ms per step (10.8%) without a steady-state prefill regression.
The re-split path scored 64/64 on the same chat-formatted 64K NIAH-S3 panel.
The subsequent five-family panel generalized automatic dispatch by measured
attention geometry and allocated request capacity; see the routing section
below and `artifacts/three_tier_resplit_family_20260826/README.md`.

Recursive three-tier storage accepts 0, 4, or 8 bits;
quantized storage requires the same precision for K and V. External ownership
forces `VLLM_LOD_PREFILL_MODE=direct`.
On the measured recursive `D=128/GQA=4/KVH=2` geometry (Phi-4 at TP5),
prefill automatically reuses the two-tier expert/MFMA exact-leaf consumer.
It evaluates all leaves of each selected centroid while the update path still
constructs the recursive page archive; one-token decode therefore retains
ordinary centroid-to-page routing. This is faster than Phi's query-major
one-page residual kernel despite evaluating more leaves. Set
`VLLM_LOD_RECURSIVE_PREFILL_ALL_LEAVES=0` to retain literal page-selecting
recursive prefill, or `1` to request the hybrid on another BF16 geometry.
The matched speed and ProLong CE results are recorded in
`artifacts/three_tier_phi_prefill_20260826/README.md`.
`VLLM_LOD_PREFIX_ROLLBACK_TOKENS` (1024) controls the exact local tail retained
for inexpensive metadata-only prefix rollback; it is semantic LOD state, not a
native cache. Eligible layers always retain a scheduler-visible virtual
full-history group with the original block geometry and are restored as
worker-only attention groups, so they receive ordinary metadata while binding
no native K/V tensor. With prefix caching enabled, that virtual group stores
chained token hashes in a bounded CPU sentinel table, but it never consumes GPU
tensor bytes or IDs from vLLM's shared physical block pool. Physical cache
sizing therefore uses only the model's remaining native groups, while hybrid
coordination keeps the full-attention topology that vLLM uses to build its fast
local-attention metadata path. This is an unconditional cache-ownership
invariant for every LOD-eligible CUSTOM layer, not a model- or mode-specific
option: scheduler specs and model markers must match exactly, and startup fails
if an eligible layer has an unsupported spec, loses its virtual group, or
appears in any native GPU K/V tensor. The plugin retains and exactly verifies
the corresponding semantic LOD row before accepting a prefix hit.
Production uses automatic routing: it selects coherence-aware state routing for
attention modules with normalized keys and spherical routing for unnormalized
keys. Under the experimental profile, `VLLM_LOD_ROUTING_GEOMETRY` may select
`raw`, `spherical`, or `coherence` for diagnostics.

At 32K and above, two-tier decode can group one or more efficient native-width
centroid tiles behind each program and reduce the shorter candidate and
online-softmax fields in parallel. This is the decode form of the same
hierarchical routing idea; it preserves the top-eight route set and retains
the established BF16-rounded centroid-mean arithmetic. Automatic segmented
decode dispatch is enabled only for the validated Muse geometry
`D=128/GQA=16/KVH=2`. Other geometries retain the grouped producer unless the
diagnostic override below is set; a forced override must not be described as
segmented unless the dispatch audit records more than one effective route
segment and `_segments_kernel` as the producer.
`VLLM_LOD_DECODE_HIERARCHICAL_ROUTE={0,1}` force-disables/enables only this
route schedule for matched diagnostics. The older
`VLLM_LOD_DECODE_GEOMETRY_TUNING=0` setting disables all decode geometry
tuning, including unrelated leaf and dot-product choices, and therefore must
not be used as a route-only A/B control.

For recursive three-tier decode,
`VLLM_LOD_RECURSIVE_STATE_ROUTE_BACKEND=auto` selects the measured backend by
geometry. `fused` retains the existing grouped state-route kernel, while
`resplit` selects the allocation-free score-materialization
path. The latter keeps score generation, softmax, split value accumulation,
and value reduction independently testable. It reuses each score-table tile
load for both tile top-eight and tile LSE, then reduces the top-eight candidates
and LSE partials together. This conservative fusion is the default for the
`resplit` backend; the benchmark reproducer can still execute every stage
separately. `resplit` is not a universal speed default. At batch eight,
automatic dispatch uses it for Phi-4 TP5 at every measured capacity, and after
the measured request-capacity crossover for Qwen3.5-0.8B (`65,536`) and
Qwen3.8-27B (`22,528`). Muse-Glimmer's `D=128/GQA=16/KVH=2` geometry remains
grouped through 128K. OLMo's re-split route was 5.98% faster at 64K, but scored
7/8 on a matched NIAH-S3 smoke where grouped routing and full attention both
scored 8/8. The completed grouped OLMo panel scored 54/64 versus full
attention's 64/64, so grouped avoids an additional route regression but does
not close that model's broader LOD-quality gap. Gemma re-split was 4.54% faster
but scored 63/64, while grouped
routing passed the matched eight-example block containing its lone miss.
Both automatic defaults therefore remain grouped; `resplit` is still
available as an explicit speed/diagnostic override.
The decision uses the pool's allocated request capacity, not the momentary
prompt length, because the graph-safe state field and decode scratch are fixed
when the pool is constructed. Unmeasured geometries retain grouped routing.
Explicit `fused` and `resplit` overrides remain available for reproducible
comparisons.
The score table is FP32 and the count-corrected probability table is FP16; this
combination removed score-ordering loss and was more accurate against exact
coarse attention than either the initial BF16-probability path or the existing
grouped route in the Qwen control.

`VLLM_LOD_AUG19_COMPAT=1` selects the closest reconstructable execution path
to the August 19, 2026 BF16 LongBench run: fixed eight-way Triton decode,
cooperative GQA/HIP decode disabled, and a four-warp leaf-route reduction.
The historical run used an uncommitted working tree, so this is an execution
compatibility preset rather than a byte-exact source restoration. The current
optimized path remains the default. `VLLM_LOD_LEAF_REDUCE_NUM_WARPS` controls
the route-reduction warp count directly when the compatibility preset is off.

For the supplied LongBench launcher, the seventh argument selects the profile:
`current` (default) or `aug19`. On Qwen3.5-35B-A3B both profiles use the generic
split decoder because its 16 query heads and two KV heads form GQA groups of
eight. The only exercised kernel difference between those two LongBench
profiles is therefore the one- versus four-warp prefill leaf-route reduction.

## Execution contract

- The default authoritative path runs direct LOD prefill. Eligible global
  layers bind no native K/V tensor. Prefix caching keeps only scheduler token
  hashes plus retained semantic LOD rows. There is no alternate cache-ownership
  mode or native rebuild path.
- Pure one-token decode uses fixed-address LOD pools and stable request-row
  indirection, so the Triton decode launches can be captured in CUDA graphs.
- vLLM asynchronous scheduling remains enabled. Model forwards and all LOD
  state mutations are submitted in worker order on the main GPU stream, while
  only sampling-output copies overlap on a separate stream. Consequently the
  large semantic state remains single-buffered and graph addresses stay fixed.
- The benchmark panel uses a 16,384-token aggregate scheduler budget across
  the entire batch. `long_prefill_token_threshold` may remain 16,384, but
  `max_num_batched_tokens` must also remain 16,384 for this methodology.
- Two-tier mode protects the sink inside the state, matching the current HF
  implementation. Recursive compatibility mode retains its separate sink
  branch.
- State/page pools and one maximum-batch decode workspace are reserved before
  vLLM computes its native KV-block budget. Smaller captured batches use stable
  views of that workspace, so decode replay does not allocate or change tensor
  addresses.
- State/page catch-up runs in `ModelState.preprocess_state`, between graph
  replays, in 256-token batches. Decode itself only appends to a fixed local
  tail and advances one integer length per active row.
- Portable two-tier decode uses one fixed eight-way Triton split kernel for
  exact leaves and local tokens, followed by one stable-LSE reduction. This is
  the path for GQA-8 Qwen3.5-35B-A3B and for every unsupported geometry.
- On gfx942 only, H=256/GQA-4 decode may use one specialized HIP kernel that
  loads a routed leaf tile once for the four query heads sharing its KV head.
  A small Triton kernel handles the local branch. Both BF16 and signed INT8
  leaf storage are supported. There is deliberately no second cooperative
  Triton fallback; disabling or missing this specialization selects the
  generic split decoder.
- A direct-prefill mixed batch uses LOD only when every request can advance an
  exact authoritative prefix. A missing prefix is an error because there is
  deliberately no native attention fallback. Retained two-level rows can
  reconstruct an older matched prefix from their chronological leaves.
  Ordinary decode tokens update only host metadata between state boundaries,
  so eager state-maintenance launches occur once per update interval rather
  than once per token.
- Sliding-window, encoder, ALiBi, attention-sink, soft-capped, quantized-native
  KV, and DCP paths remain native or are rejected when a lossless fallback is
  not available. DFlash and DFlash2 speculative decode are supported: their
  draft caches stay native, while target verification appends proposed tokens
  to LOD and restores the scheduler's committed prefix before the next proposal
  when necessary.
  Tensor parallelism and hybrid recurrent layers do not alter the per-layer
  LOD contract.

### Original DFlash on Gemma 4

The public `z-lab/gemma-4-26B-A4B-it-DFlash` checkpoint is the original
16-position DFlash design (one anchor plus 15 proposed tokens), not DFlash2.
The plugin registers its `DFlashDraftModel` on the pinned vLLM revision and
retains Gemma's target embedding scale and final-logit soft cap.  Draft layers
do not inherit the target's multimodal-prefix mask.  The input-preparation
kernel masks rejected target suffixes out of the draft cache and handles a
fully rejected row without reading an invalid last context position.

Recursive target verification stages all 16 proposal positions together. The
conservative D=512 default bounds one routing/attention launch to 32 flattened
rows, so B1 stays one parallel launch while B8 uses four four-position chunks,
rather than 16 serial target calls. Individually validated 32K, 64K, and 128K
B8 profiles raise that bound to 64 rows and use two eight-position chunks.
Every position still receives its own current route and causal recent length;
routes are neither shared across chunks nor lagged. The panel wrapper defaults
Gemma DFlash to prefill top four and decode top eight. The validated 128K
high-throughput profile explicitly uses prefill top three; a mid-prompt
top-four-to-top-three switch was rejected because it scored 7/8 on NIAH-S3
even though either fixed profile scored 8/8. Complete TP1 B1/B8 speed and
NIAH-S3 results are recorded in
`artifacts/gemma_dflash_20260901/README.md`.
At B8, the selected LOD profiles reduce prefill from 40.135 to 22.839 seconds
at 64K (1.76x) and from 138.830 to 46.109 seconds at 128K (3.01x). Their
end-to-end speculative decode results are 10.773 versus 41.769 ms at 64K and
12.655 versus 55.819 ms at 128K, though those ratios include acceptance
trajectory; the LOD target cycle is a steadier 25.972/25.351 ms.

### DFlash2 on the pinned vLLM revision

The plugin registers `DFlash2DraftModel` and the path-aware DFlash2 candidate
selector that are present in newer vLLM releases but absent from the pinned
ROCm benchmark environment. The large target remains resident in the IPC
weight cache; the much smaller nested draft model uses vLLM's ordinary model
loader so it cannot be mistaken for the target by the daemon protocol. Set
`VLLM_LOD_PANEL_SPECULATIVE_MODEL=z-lab/Qwen3.8-27B-DFlash2` when using
`scripts/run_vllm_lod_niah_speed_panel.sh`; the wrapper selects the V2 runner,
seven draft tokens, and `TRITON_ATTN` for the draft unless overridden.

Multi-token DFlash2 target verification can now use the same captured
fixed-mask path as deeper native MTP. DFlash2 selects one linear seven-token
path before target verification, so all eight target positions are staged and
flattened into one LOD launch. Every position retains its own current top-eight
routes and causal local length; routes are not shared or lagged. The DFlash2
drafter keeps its native chronological cache and captured graph. Long-prompt
prefill retains the measured top-three route count, while verification uses
decode's top eight. Using prefill top three for the verifier remains invalid:
it can accept a token that sequential top-eight LOD would reject.

On Qwen3.8-27B-FP8 TP1/B1, the current full/fixed-mask-LOD DFlash2 decode
medians are **15.722/17.571 ms at 8K**, **9.753/8.774 ms at 16K**,
**15.088/14.348 ms at 32K**, **19.316/15.091 ms at 64K**, and
**22.054/14.621 ms at 128K**. Thus LOD is slower at 8K, but 1.11x, 1.05x,
1.28x, and 1.51x faster from 16K through 128K. Its complete-model verifier
cycle stays near 40--41.5 ms across the panel; at 128K full attention takes
about 63.2 ms per verifier cycle, making the target-side comparison 1.52x
independent of acceptance. The device audit confirms direct fixed routes,
fixed-mask execution, the page-size-one HIP scan, and the final LSE reduction.
The matched protocol, prefill table, and raw records are in
`artifacts/dflash2_qwen38_20260831/README.md`; the older serial verifier and its
quality controls remain documented in
`artifacts/dflash2_qwen38_20260830/README.md`.

Recursive three-tier DFlash2 verification uses the same flattened
eight-position contract, but every target position performs an independent
current centroid and page route against the shared immutable recursive
archive. Proposal K/V is staged before the launch and logical per-position
recent lengths enforce causality. The wide-GQA local QK/PV kernels must index
those logical lengths rather than the repeated physical cache row; fixing that
distinction made both BF16 and INT4 score 8/8 on batch-eight NIAH-S3 at 64K and
128K. The audit confirms one flattened verifier rather than eight serial
calls.

For speculative recursive verification only, the grouped (`fused`) state
router is the safe default. The materialized re-split route currently causes
an HSA memory fault at 128K under speculative verification, including eager
and serial-verifier controls, while ordinary non-speculative re-split decode
passes. This does not change the ordinary recursive auto policy. Set
`VLLM_LOD_SPECULATIVE_RECURSIVE_STATE_ROUTE_BACKEND=resplit` only for targeted
diagnosis.

On Qwen3.8-27B-FP8 TP1, three-tier BF16/INT4 target-verifier cycles are
41.313/41.760 ms at 64K/B1 and 42.085/42.064 ms at 128K/B1. At B8 they are
62.444/61.133 ms and 58.765/55.036 ms. Thus recursive INT4 has no measured
decode penalty and is 2.1--6.3% faster in the B8 screen, because only selected
16-token residual pages are dequantized. INT4 prefill is 7.8--11.2% slower.
Allocated semantic LOD cache falls from 9.748 to 3.716 GiB at B1 and 77.984 to
29.731 GiB at B8 (61.9%). The implementation, complete speed tables, quality
results, and raw records are in
`artifacts/dflash2_three_tier_20260831/README.md`.
Three-tier BF16 is not yet an unconditional replacement for fixed-mask
two-tier: two-tier wins 64K/B8 end-to-end decode (23.766 versus 25.957 ms),
whereas three-tier wins at 128K/B8 (17.745 versus 19.746 ms). Use three-tier
for recursive INT4 storage and treat the BF16 crossover as the current TP1
decision pending a repeated acceptance-matched panel.

The short-context TP1/B1 recursive verifier now pairs adjacent DFlash target
positions in the native M16 state-route tile.  Qwen's eight verifier positions
become four independent M12 groups instead of eight underfilled M6 scans;
every position retains its own current route and causal local length.  The same
programs absorb pairwise local attention.  This lowers the three-tier target
cycle from 40.693/40.885/40.886/41.195 ms to
39.754/39.242/39.757/39.676 ms at 8/16/32/64K.  It is now 0.3--2.7% faster
than the matched two-tier verifier at every one of those lengths.  The
route-only ablation shows that paired state routing supplies most of the gain;
local fusion contributes up to another 0.66 ms.

Recursive BF16 prefill on TP1's `(D, GQA, KVH) = (256, 6, 4)` now uses the
regular complete-expert MFMA consumer for requests whose total prompt is at
most 64K; longer requests use one-page recursive selection for every chunk.
The recursive archive is built throughout, so decode and memory semantics do
not change.  The bound is the natural page crossover: under the 16*sqrt(T)
schedule, average posting length `sqrt(T)/16` reaches the 16-token page size at
64K context and 4096 state entries.  This reduces three-tier prefill from
1.001/2.061/4.212/8.701 seconds
to 0.994/2.055/4.154/8.531 seconds at 8/16/32/64K.  The automatic path scores
8/8 on NIAH-S3 at both 8K and 64K, and its device audit confirms pairwise route
and fused-local execution.  Expert MFMA and recursive-page attention retain
independent two-wave/one-wave launch geometry; automatic 128K prefill is
18.316 seconds versus 18.338 seconds for explicit page-only.  Full diagnosis,
ablations, and raw records are in
`artifacts/dflash2_three_tier_short_20260831/README.md`.

The matched TP4 DFlash2 panel uses six local query heads, one local KV head,
head dimension 256, three measured speed repetitions, and the same 16K
aggregate prefill budget.  Full/recursive-BF16 decode is 9.777/9.219 ms at
64K/B1, 11.702/9.144 ms at 128K/B1, 17.255/13.399 ms at 64K/B8, and
20.888/13.099 ms at 128K/B8: recursive LOD is 1.06x--1.59x faster.  Prefill is
5.367/4.401, 14.583/9.447, 43.562/36.954, and 120.436/79.846 seconds in the
same order, a 1.18x--1.54x speedup.  Full and recursive BF16 both score 8/8
on NIAH-S3 at 64K and 128K.
The complete protocol and raw records are in
`artifacts/dflash2_three_tier_tp4_20260831/README.md`.

The final post-optimization three-repeat matrix retests three-tier BF16 at
TP1/TP4 and B1/B8 over 8--128K.  Target-verifier cycles are 39.33--40.56 ms
at TP1/B1, 50.22--57.15 ms at TP1/B8, 25.07--25.48 ms at TP4/B1, and
30.33--32.25 ms at TP4/B8.  At 64K/128K this improves the previous recursive
cycles by 3.6%/3.6%, 15.2%/14.5%, 3.8%/3.1%, and 5.2%/5.1%, respectively.
Against hash-matched historical full-attention controls, current end-to-end
LOD speedups are 1.51x/2.08x, 1.83x/2.56x, 1.10x/1.28x, and 1.40x/1.63x in
the same geometry order.  The full prefill, decode, and cycle tables plus
execution audits are in
`artifacts/dflash2_three_tier_matrix_20260831/README.md`.

On ROCm, multi-GPU DFlash graph capture must not run alongside PyTorch's
ProcessGroupNCCL watchdog: its background HIP-event query is illegal during
capture even when model collectives use vLLM's custom all-reduce.  The panel
wrapper therefore defaults TP DFlash runs to blocking wait, with asynchronous
error handling and monitoring disabled; explicit caller overrides remain
available.  Recursive catch-up also uses an overlap-safe suffix shift when a
speculative update boundary leaves exact recent tokens.

### Native MTP target verification

One-token native MTP produces a two-position target verification. The old LOD
adapter transposed those positions and invoked the complete M=1 LOD decoder
twice serially. That erased most of the target-attention saving: on
Qwen3.8-27B-FP8 TP1/B1, full/old-LOD MTP took 22.761/22.331 ms per emitted
token at 64K.

This is a batch-1 comparison. The matched non-speculative batch-1 control at
64K is 31.404/28.877 ms for full/two-tier LOD (1.087x); the much larger 1.43x
historical Qwen3.8 win is the batch-8 panel and is not the proper MTP control.

The default two-tier path now stages both proposed K/V entries and verifies
both positions with one flattened LOD launch. Both positions route against the
same immutable remote state, while per-position logical recent lengths enforce
causality: position zero sees its own K/V and position one sees both proposed
entries. Qwen's two GQA-6 position groups are packed as 12 useful rows in one
M=16 coarse-scoring tile, so a centroid K/V tile is loaded once. The same
programs also consume disjoint tiles of the recent suffix. Thus 512 of the
513/514 visible local entries are loaded once for all 12 verifier rows; only
the causal mask differs for the newest proposal entry. Local scores never enter
centroid top-eight selection: they are merged only into the coarse output and
LSE. This removes a separate local-attention launch without changing routing.
The normal prefix-restore path truncates any rejected proposal suffix before
the next verification; routes are never lagged.

The original corrected 64K time was **20.031 ms (1.136x over full MTP)**,
versus 22.761 ms for full MTP. The pre-local-sharing 128K result remains **21.742 ms
(1.306x)** versus 28.400 ms for full MTP. It scores 8/8 on chat-formatted
NIAH-S3 at both 8K and 64K. Device execution markers for both shared routing
and shared local attention let the benchmark fail rather than silently time a
different captured path. Set `VLLM_LOD_SPECULATIVE_PARALLEL=0` only to
reproduce the old serial verifier, or `VLLM_LOD_SPECULATIVE_SHARED_ROUTE=0` to
retain the parallel artificial batch while scoring its positions
independently. The original matched panel is in
`artifacts/mtp_qwen38_20260830/README.md`; the local-sharing implementation,
profile, and raw records are in `artifacts/mtp_qwen38_20260831/README.md`.

The subsequent balanced exact-list update stripes every selected centroid's
leaves across all eight exact-attention splits instead of assigning one whole
centroid to each split. After reconstructing the exact source of the 20.031 ms
run and applying only this update, the matched 64K result was **19.522
ms/output**, with 86.1% draft acceptance. Its estimated verifier-cycle cost
was 36.073 ms versus 37.013 ms for the original result, a 2.5% improvement.
The exact-leaf kernel fell from 110.331 to **53.041 us per global-layer call**
in the delayed profile, and the prior striped check retained 8/8 NIAH-S3.
Striping is the default for multi-position target verification;
`VLLM_LOD_SPECULATIVE_STRIPE_ROUTE_LEAVES=0` is the legacy control.

The same balancing is now the default in the portable two-tier decode
fallback. Instead of assigning one complete selected centroid posting list to
each split, every split walks a disjoint stripe of every selected posting
list. This removes the longest-centroid tail without changing which leaves are
read or how their output/LSE is merged. Set
`VLLM_LOD_DECODE_STRIPE_ROUTE_LEAVES=0` only to reproduce the legacy fallback.
The production page-size-1 AITER fixed-list path already flattens all selected
leaves before partitioning them, and the GQA-cooperative backend already
partitions every route by page-list split, so neither needs a separate striped
variant. Matched Qwen and Muse measurements are in
`artifacts/striped_leaves_20260831/README.md`.

Independent launches of the exact historical source produced 81.6% and 87.5%
acceptance despite identical temperature-zero input and configuration. Their
per-verifier-cycle costs remained within 0.7%, so speculative speed comparisons
must report both acceptance and a cycle-normalized latency; ms/output alone can
look regressed when the greedy trajectory changes. The unsuccessful
coarse-verifier/self-speculation implementation was removed rather than kept as
another production path.

Every eligible global layer uses external LOD for prefill and decode, including
short requests. A captured graph never swaps in native attention and no native
chronological cache exists for such a layer.

## Paper-oriented kernel surface

The primary two-tier implementation has one prefill route and two decode
backends. Precision changes are compile-time storage specializations, not
different LOD algorithms.

Prefill processes a 4,096-token chunk in three stages:

1. A fused routing/coarse-attention kernel scores state centroids, retains the
   top eight routes, and produces the unopened-centroid residual branch.
2. The expert-major leaf kernel attends each routed query group to the exact
   posting list of its centroid, without compacting or copying K/V.
3. A stable-LSE reduction merges the eight exact-route results with the coarse,
   local-window, and protected-token branches.

Decode first performs the same centroid routing/coarse calculation. It then
uses either the generic split-8 exact/local kernel or the optional gfx942
H=256/GQA-4 HIP specialization described above, and finally performs one
stable-LSE merge with the coarse branch. The compatibility profile does not
duplicate these algorithms; it only freezes older dispatch/reduction settings.

The paper implementation intentionally removed the unused combined
cooperative Triton kernel and the slower cooperative Triton fallback. This
deleted over 900 lines of kernel and fallback-dispatch code. Generic decode
scratch no longer reserves the
specialized GQA partial buffers. A 64K, batch-8 Qwen3.5-0.8B validation after
the deletion produced identical top-1 outputs in both modes; after compilation,
the specialized and generic full-model decode steps were 14.14 ms and 14.55 ms,
respectively. The focused numerical verifier matched specialized BF16 output
to the generic/reference result within 7.4e-4 maximum absolute error and INT8
within 8.6e-4.

## Current memory behavior

External semantic ownership replaces full chronological K/V for eligible global
layers. With prefix caching, an all-global model retains its real logical cache
geometry in the scheduler for token hashes, while the worker removes the GPU
allocation and attention path entirely. Hybrid models reuse an existing native
group's hashes and need no tracker. Without prefix caching, no scheduler group
or worker tensor is needed for those layers. Recurrent and sliding-window caches
remain native. Semantic leaves are
quantized only after region assignment; sequential native blocks are never
treated as quantizable pages.

For Qwen3.5-0.8B at 64K, batch eight, INT4 LOD uses 2.817 GB of semantic cache,
0.221 GB of native scheduler/recurrent state, and a 0.063 GB shared decode
workspace. Persistent cache is therefore 3.037 GB, 55.6% below full attention's
6.845 GB. The earlier Transformers figure of 2.758 GB counted semantic
attention state but not Qwen's recurrent GDN cache; adding the same 0.221 GB
puts it within 2.0% of the vLLM result.

`dual` mode intentionally retains both representations and should not be used
to assess memory savings.

### Recursive INT4 quality default

`VLLM_LOD_KV_BITS=4` now defaults to four-channel page-wide groups for leaf
residuals and INT8 page summaries, with an L2-refined leaf scale during both
prefill conversion and decode appends. The former quality default shared one
scale across 256 residual values, so an outlier could waste a material fraction
of INT4's range for 16 channels. The new leaf layout covers 64
values per scale while preserving the inexpensive broadcast scale load in
attention. It uses 4.25 effective bits per leaf value, 4.62% more than the
former G16 layout and 73.44% less than a BF16 leaf payload.

On the fixed Qwen3.5-0.8B 48-example LongBench-v2 damage panel, mean choice
distributions from three G4 L2 runs agreed with the mean of four BF16 LOD runs
on 45/48 choices (93.75%). A single G4 run also had the best tested
distribution metrics (mean JS 0.001910 and RMS correct-margin drift 0.2626).
Repeats matter because
the four BF16 runs themselves had only 90.28% pairwise prediction agreement.
The G4 eight-by-8K ProLong CE was 1.925282, compared with 1.925134 for BF16.
This is a rapid proxy result, not evidence that the historical full 503-example
INT4 gap is closed. G4 L2 also retained 64/64 Qwen3.5-0.8B NIAH-S3 at 8K,
batch eight. Detailed results and rejected scale formats are in
`artifacts/int4_quality_recovery_20260827/README.md`.

The current matched Qwen3.8-27B-FP8 speed panel uses both a 16,384-token
aggregate scheduler budget and a 16,384-token per-request long-prefill
threshold. No retained timing uses the rejected 4K per-request cap. At B1,
two-tier BF16 is the fastest prefill and decode choice from 8K through 128K.
At B8, full attention narrowly wins 8K prefill, two-tier BF16 wins prefill from
16K onward, and recursive three-tier BF16 wins decode at every length. At
64K/B8, two-tier BF16 prefill is 69.120 seconds; recursive BF16/INT4 prefill is
69.793/76.597 seconds, and recursive BF16/INT4 decode is 35.340/35.751 ms.
Matched full attention is 112.544 seconds and 54.956 ms.

The INT4 prefill optimization changes neither cache format nor approximation.
Relative to recursive BF16, INT4 prefill is at most 9.6% slower at B1 and
10.3% slower at B8; decode remains within 1.4%. Allocated recursive LOD cache
storage falls from 10.449 GB to 3.985 GB at B1 and from 83.592 GB to 31.881 GB
at B8 (61.9%) after scale metadata, page summaries, and routing state are
included. The implementation uses a one-wave, four-group page quantizer,
grouped cached-prefill requantization, no finalized-cache BF16 fallback, and a
shared page-mean factorization in leaf QK/PV. Final code retained 64/64
Qwen3.5-0.8B NIAH-S3. Two-tier INT4 is not reported because the flat two-tier
cache currently supports BF16 and INT8 storage only. Kernel profiles and
correctness checks are in
`artifacts/int4_prefill_optimization_20260829/README.md`; the authoritative
full/two-tier/three-tier B1/B8 decision tables and raw records are in
`artifacts/int4_context_panel_20260829/README.md`.

The matched Muse-Glimmer-30B TP1 panel uses the same 16,384-token aggregate
budget and per-request threshold. Full attention wins prefill through 32K at
B1 and through 64K at B8. Two-tier BF16 crosses at 64K/B1 and 128K/B8; at
128K it takes 13.478/109.322 seconds versus 14.946/120.432 seconds for full
attention. Full attention wins decode through 128K/B1 and 64K/B8. Recursive
three-tier BF16 crosses at 128K/B8, taking 20.041 ms versus 20.919 ms for full
attention. Recursive INT4 reduces allocated LOD cache by 60.7%, keeps decode
within 1.0% of recursive BF16, and costs up to 6.8% in prefill. The complete
tables and raw records are in
`artifacts/muse_tier_int4_context_panel_20260830/README.md`.

The matched Gemma-4-26B-A4B-it TP1 panel uses the same 16,384-token aggregate
budget and per-request threshold, with the validated native `TRITON_ATTN`
D=512 control. Full attention wins both phases only at 8K. From 16K onward,
two-tier BF16 wins prefill and recursive three-tier BF16 wins decode. At
128K/B8, prefill is 137.626/41.845/44.628 seconds for full/two-tier/three-tier
BF16, while decode is 15.542/11.303/9.812 ms. Recursive INT4 reduces allocated
LOD cache by 62.5%, but its D=512 prefill is up to 41.4% slower than recursive
BF16, so it is a capacity rather than latency choice on Gemma. The complete
tables and raw records are in
`artifacts/gemma_tier_int4_context_panel_20260830/README.md`.

Set `VLLM_LOD_QUANT_GROUP_SIZE`, `VLLM_LOD_LEAF_QUANT_SCALE_MODE`, and
`VLLM_LOD_LEAF_APPEND_QUANT_SCALE_MODE` explicitly to override the precision
policy. Group size 16 plus `l2` reproduces the former quality default; group
size 32 plus `max` reproduces the legacy INT4 layout.
`VLLM_LOD_QUANT_TOKEN_GROUP_SIZE` remains 16 by default; smaller
values are experimental because they add per-read token-axis scale traffic.

## Quality validation

The authoritative integration was checked with Qwen3.5-0.8B on vLLM 0.27.1
for ROCm. NIAH used eight 8K examples at batch size eight; the initial ProLong
check used two 8K documents.

| Evaluation | Native vLLM | LOD backend |
| --- | ---: | ---: |
| ProLong token CE | 2.125621 | 2.128387 |
| ProLong perplexity | 8.378100 | 8.401305 |
| NIAH-S3 exact match | 8/8 | 8/8 |

ProLong prompt log-probabilities exercise direct LOD prefill; its CE increase in
this paired check was 0.002766 (0.13%). NIAH-S3 exercises direct prefill and INT4
recursive LOD decode. The final NIAH run used CUDA graphs. The worker recorded
48 real authoritative cache installations: six global-attention layers for
each of eight requests.

The current flat two-tier BF16 port was separately checked after enabling raw
routing and the 4,096/4,864/4,096 prefill schedule. It also scored 8/8 on the
same 8K NIAH-S3 batch, with direct LOD prefill and captured LOD decode both
exercised. See
`artifacts/vllm_lod_quality/qwen35_0p8b_two_tier_raw_bf16_niah_s3_8k_s8.json`.

Full 503-example LongBench v2 runs used identical guided A-D decoding for full
and LOD attention. After fixing cached-prefill finalization, stable page
chronology, and unused INT4-page summaries, Qwen3.5-35B-A3B LOD scored 229/503
(45.53%) versus full attention's 245/503 (48.71%). Qwen3.8-27B-FP8 LOD scored
256/503 (50.89%) versus full attention's 269/503 (53.48%).

| Model and subset | Full attention | LOD attention |
| --- | ---: | ---: |
| Qwen3.5 overall | 48.71% | 45.53% |
| Qwen3.5 short / medium / long | 52.78% / 48.84% / 41.67% | 49.44% / 46.05% / 37.96% |
| Qwen3.8 overall | 53.48% | 50.89% |
| Qwen3.8 short / medium / long | 55.00% / 53.95% / 50.00% | 53.89% / 49.77% / 48.15% |

All four runs truncated the same 205 prompts to the model's 131,072-token
limit. The remaining LOD gaps are 3.18 percentage points on Qwen3.5 and 2.58
points on Qwen3.8; NIAH success alone is not sufficient validation for this
approximation.

## Warm serving performance

The flat physical-page INT8 path now allocates the persistent vLLM pool as
signed INT8 and retains its per-token K/V scales. Before this fix, only the
transient prefill cache was INT8: installation copied its integer codes into a
BF16 destination and discarded the scales. Results from that broken path did
not measure valid INT8 attention.

The corrected uncapped Qwen3.5-0.8B batch-8 measurements below use 16K chunks
per request, a 128K aggregate scheduler budget, M=16/N=32 leaf tiles, two
warps, and 4,096-token state updates. The 32K pair disables asynchronous
scheduling so both precisions execute the same 78 direct-prefill calls. The
64K pair has identical scheduler diagnostics. INT8 uses BF16 PV below 64K and
INT8 PV at 64K, selected automatically.

| Context | BF16 prefill | INT8 prefill | INT8 change | BF16 LOD cache | INT8 LOD cache |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K | 1.791 s | 1.798 s | 0.4% slower | 9.05 GB | 5.03 GB |
| 64K | 4.089 s | 3.952 s | 3.3% faster | 14.56 GB | 7.95 GB |

Thus the remaining 32K difference is effectively noise-sized, while the
longer posting-list scan amortizes probability quantization at 64K. Cache
storage falls by 44.4% and 45.4%, respectively; BF16 routing state and page
metadata prevent the total from reaching exactly 50%. The optimized INT8
kernel retains NIAH-S3 accuracy at 64/64. On eight 8K ProLong examples it has
token CE 1.924265 and perplexity 6.850114, versus 1.923501--1.923557 for the
matched BF16 checks.

Recursive cache-native INT8 uses signed 8-bit page-mean residuals for both K and V, with
one BF16 scale per page and 32 channels. At 8K and batch eight on
Qwen3.5-0.8B, five warm 1,025-token runs measured 0.418 s prefill and 3.677 ms
per decode batch step. That is within 0.3% of BF16 decode, 3.1% faster than
INT4 decode, and 7.6% faster than INT4 prefill. Its 1.341 GB LOD cache was
30.2% smaller than BF16. See
`artifacts/vllm_lod_speed/INT8_8K_B8_20260817.md` for the paired results.

The following Qwen3.5-0.8B measurements use batch size eight, CUDA graphs, a
65,536-token scheduler budget, an 8,192-token long-prefill threshold, 1,025
generated tokens, and the speed-oriented BF16 LOD leaf cache
(`VLLM_LOD_KV_BITS=0`). They report warm medians from five to seven runs. The
exact full backend uses `ROCM_AITER_UNIFIED_ATTN`, the fastest working exact
backend tested on this ROCm 7.2 system, and a tightly sized native block pool.
The 1,024 measured decode intervals include four 256-token LOD state updates,
so their costs are amortized rather than represented by a single boundary.

| Context | Full prefill | LOD prefill | Full decode step | LOD decode step |
| --- | ---: | ---: | ---: | ---: |
| 16K | 0.898 s | 1.056 s | 3.09 ms | 3.47 ms |
| 64K | 8.039 s | 6.177 s | 5.11 ms | 4.26 ms |

At 64K this is a 1.30x prefill speedup and a 1.20x decode-step speedup. At 16K,
exact full attention remains 1.18x faster in prefill and 1.12x faster in decode.
INT4 remains the memory-oriented mode. The current low-memory 64K result is
reported in the memory section above. Total device use is not reported as a
cache metric because model weights, compiled kernels, and runtime workspaces
remain resident and vary with warmup history.

vLLM's automatic ROCm selection chose `ROCM_ATTN` for this hybrid model. Its
native paged kernel supports only 16- and 32-token blocks, while Qwen3.5's
hybrid recurrent cache forces a 544-token attention block. It therefore fell
back to the generic Triton chunked-paged path, inflating the 16K decode step to
15.93 ms and the 64K decode step to 56.78 ms. Those timings are backend fallback
diagnostics, not a fair exact-attention baseline.

Cached-prefill state boundaries are rounded down to the 256-token update grid.
This keeps at most one extra partial chunk exact and prevents arbitrary request
lengths from producing an unbounded family of page-update specializations.
The fused routing, coherence-update, final-reduction, and sparse page-transfer
kernels keep ragged tensor extents and their batch/head strides as runtime
arguments. Prefill routing uses 128-query tiles, coarse attention uses 32x64
tiles, recursive page attention uses two-page tiles, and the local branch uses
AITER. The final merge now honors arbitrary output strides, so packed prefill
writes directly into vLLM's token-major output rather than allocating and
copying a second full output tensor. An unseen device/kernel configuration
still incurs ordinary Triton JIT work; production images should preserve or
pre-populate their Triton compilation cache. Warm medians above exclude setup
time but can still contain isolated shape-specific compilation outliers.

The paired evaluator is `scripts/eval_vllm_lod_quality.py`. Its LOD NIAH check
requires both an executed decode path and real cache installation, preventing
graph-capture warmups from being mistaken for an LOD result.

Warm batch throughput can be reproduced with:

```bash
VLLM_LOD_PROFILE=production python scripts/benchmark_vllm_lod_speed.py \
  --mode lod --length 8192 --batch-size 8 --decode-tokens 1025 --repeats 5 \
  --max-num-batched-tokens 65536 --long-prefill-token-threshold 8192 \
  --output artifacts/vllm_lod_speed/lod_b8_8k_d1025.json
```

Use at least 1,025 decode tokens when comparing amortized serving performance
so the measurement spans four state-update boundaries.

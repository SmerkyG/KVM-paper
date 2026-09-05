# K2 Horizon 0.9B: full attention vs. LOD attention

Checkpoint: `IFM/K2-Horizon-0.9B`

Date: 2026-09-03

Hardware: one AMD Instinct MI325X per process. Software: PyTorch 2.9.1 ROCm and
Transformers 5.15.0. Full attention uses the model's SDPA backend. LOD uses the
generic Hugging Face adapter with the recursive paged kernel engine, BF16 leaf
KV, 16-token pages, top-8 routing, a `16 * sqrt(T)` state schedule (minimum
256), a 512-token local window, and 256-token state updates. The
`qk_norm_aware` policy resolves to cosine clustering without centroid rescaling
because K2 Horizon does not normalize its keys.

## Quality

ProLong cross-entropy is token-weighted over identical deterministic samples.
Lower is better.

| Length | Samples | Full CE | LOD CE | Delta | Full PPL | LOD PPL |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8 | 2.92380 | 2.92234 | -0.00146 | 18.6118 | 18.5847 |
| 32K | 4 | 2.81613 | 2.80521 | -0.01091 | 16.7120 | 16.5306 |
| 64K | 2 | 2.86667 | 2.85566 | -0.01101 | 17.5784 | 17.3859 |

NIAH-S3 exact-match uses the same eight generated cases in each matched pair.

| Length | Batch | Full | LOD |
|---:|---:|---:|---:|
| 8K | 8 | 8/8 | 8/8 |
| 32K | 8 | 5/8 | 6/8 |
| 64K | 1 | 4/8 | 8/8 |

The eight-case NIAH panels are deliberately small smoke panels, so the apparent
LOD improvement should not be treated as a precise model-quality estimate.
The important result is that the approximation did not damage retrieval on
these cases. Full attention scored 2/8 at 64K with batch 8; the matched batch-1
control above is used in the table because the model's result was numerically
batch-sensitive.

## Warm speed

Each row is a matched process-level comparison. Prefill is seconds for the
whole batch. Decode is milliseconds per batched autoregressive step, measured
over 1,025 generated tokens so four 256-token state-update boundaries are
included. Values are medians of three measured repetitions after warmup.

| Context | Batch | Full prefill (s) | LOD prefill (s) | Full decode (ms) | LOD decode (ms) | Decode speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8 | 0.502 | 1.548 | 16.395 | 16.301 | 1.01x |
| 16K | 8 | 1.375 | 4.329 | 27.845 | 17.664 | 1.58x |
| 32K | 8 | 4.284 | 12.424 | 50.579 | 16.258 | 3.11x |
| 64K | 4 | 7.433 | 16.824 | 85.887 | 16.135 | 5.32x |

The current generic Hugging Face LOD path is compelling for decode but not for
prefill: prefill is 2.26-3.15x slower than SDPA in this sweep. Its transient
BF16 prefill working set is also large; batch 8 at 64K exceeded the 256 GiB
device, so that matched row uses batch 4. These peak allocations are not a
measurement of the vLLM INT4 persistent-cache implementation. The released
vLLM 0.27.1 build does not contain K2 Horizon, so this historical panel used
the model-independent HF backend. The vLLM LOD plugin now backports the
upstream architecture on older vLLM releases and defers to native support when
available.

## vLLM backport validation

The conditional backport was exercised with vLLM `0.27.1+rocm723`. Native
attention resolved `K2HorizonForCausalLM` and scored 8/8 on the 8K NIAH-S3
smoke panel. LOD installed its cache on all 28 attention layers, ran both
direct prefill and decode, and scored 6/8 on the same panel. A retained-prefix
test reused all 28 semantic cache rows on the second request without rebuilding
the shared prefix. On a fixed 46-token prompt, native vLLM and Hugging Face
produced mean cross-entropies of 3.20680 and 3.21104 respectively. The small
NIAH panel is a compatibility smoke test rather than a new quality result; the
historical Hugging Face results above remain the more complete matched
comparison.

The native vLLM and BF16 two-level LOD paths were also compared on the pinned
64K ProLong documents 8--15 (524,280 predicted tokens). Both modes consumed
the same source indices and K2 token hashes.

| 64K subset | Full CE | LOD CE | CE delta | Full PPL | LOD PPL |
|---|---:|---:|---:|---:|---:|
| ProLong 8--15 | 1.03174 | 1.03259 | +0.00085 | 2.80593 | 2.80833 |

The largest per-document CE increase was 0.00459; three of eight documents
improved under LOD. This run used direct LOD prefill with three opened regions,
top-8 decode routing, the `16 * sqrt(T)` state schedule, and 16K vLLM scheduler
chunks.

### vLLM 64K prefill routing sweep

The following sweep separates warm prefill timing from model startup and kernel
compilation. Each timing is the median of three measured batch-8 passes after a
warmup, over the same eight distinct 65,536-token ProLong prompts (524,288 input
tokens total). Quality uses the pinned ProLong documents 8--15 described above.
All LOD arms use BF16 leaves, direct prefill, 16K scheduler chunks, a
`16 * sqrt(T)` state schedule, and top-8 decode routing.

| Prefill configuration | Median (s) | Tokens/s | CE | CE delta | PPL delta |
|---|---:|---:|---:|---:|---:|
| Full attention | 13.290 | 39,451 | 1.03174 | -- | -- |
| Two-tier, top-2 regions | 22.930 | 22,865 | 1.03416 | +0.00242 | +0.242% |
| Two-tier, top-3 regions | 25.945 | 20,207 | 1.03259 | +0.00085 | +0.086% |
| Three-tier, top-3 regions | 28.631 | 18,312 | 1.03505 | +0.00331 | +0.332% |

Two-tier top-2 is 13.2% faster than two-tier top-3 while retaining a small
average quality delta. Two-tier top-3 is the more conservative quality choice.
The tested three-tier page-selection path is dominated: it is 10.3% slower
than two-tier top-3 and has a larger loss delta. Native full attention remains
1.73x faster than the fastest LOD arm on this unusually small, head-dimension-64,
GQA-4 model, so K2 Horizon is not a prefill speed win at 64K with the current
kernels.

The 8K and 16K timing files use a full 1,025-token warmup. The 32K and 64K
records used a 257-token warmup but were retained because all three measured
repetitions were already stable. The benchmark helper now defaults to warming
the full requested decode length.

## K2 Horizon 32B FP8: corrected vLLM 64K timing

The 32B FP8 checkpoint exposed two independent implementation problems in the
initial vLLM measurements. A one-token decode fallback reserved its routing
workspace at the 4,096-token prefill capacity (about 4.25 GiB per layer), and
the weighted AITER coarse experiment materialized a broadcast count bias plus
packed transposes. Decode routing now reserves one query row, and CK attention
consumes the native outer strides directly. Flat BF16 two-tier pools also omit
unused recursive page-summary tensors.

The production comparison below uses eight 64K ProLong prompts, a 16K aggregate
scheduler budget, 4K per-request chunks, top-2 prefill routing, top-8 decode,
the `16 * sqrt(T)` state schedule, and BF16 leaves. Prefill is measured after a
warm pass. Decode generates 1,025 tokens, so four 256-token state updates are
included rather than hidden outside the measurement.

| 64K / batch 8 | Full attention | LOD attention | Relative result |
|---|---:|---:|---:|
| Prefill | 138.704 s | 129.417 s | LOD 1.072x throughput |
| Decode batch step | 68.855 ms | 56.228 ms | LOD 1.225x throughput |
| Decode tokens/s | 116.19 | 142.28 | LOD +22.5% |

The fast decode result requires the unified GQA exact-leaf/local/coarse AITER
path. The ordinary non-union LOD decode took 73.977 ms per batch step and is
therefore not the selected configuration. A phase-profiled union run attributed
about 20.4 ms/step to centroid routing, 19.9 ms/step to exact-leaf plus local
attention, about 4 ms/step amortized to the four state updates, and roughly
12 ms/step to the rest of the model.

For prefill, the profiled LOD work was dominated by routing (25.0 s), coarse
attention (15.7 s), exact leaves and their dispatch (12.7 s), local attention
(7.1 s), and state maintenance. Full attention spent about 74.7 s in attention.
View-only CK local inputs saved another 1.0 s end to end and were bit-exact with
packed inputs in the focused output/LSE check. Replacing the coarse branch with
weighted AITER remained slower at 148.185 s, and direct streaming route plus
AITER did not finish a warm pass within 200 s, so neither is selected.

Increasing the aggregate scheduler budget from 16K to 32K did not improve full
attention (138.928 s) and made LOD decisively slower: the B8-wide routing tensor
doubled transient traffic and did not finish its warm pass within 170 s. Thus
the apparent underfill was not the bottleneck; B4 waves are the better point for
this 32B geometry. The stride-only production change does not alter attention
math; its focused packed-versus-strided output and LSE check was bit-exact.

## K2 Horizon 32B FP8: recursive BF16 and residual INT4

The current three-tier implementation uses the same K2 TP1 geometry as the
two-tier path: 64 query heads, eight KV heads, head dimension 128, cosine
centroid assignment, a `16 * sqrt(T)` state schedule, and a separate protected
sink. Prefill opens the top three centroids and attends to every leaf in them.
At 64K, average posting length is one 16-token page; at 128K it is about 1.4
pages, but complete-centroid attention remains faster than paying for another
page-routing stage. Decode still performs genuine recursive routing: it opens
the best page in each of the top eight centroids.

The residual INT4 path stores each leaf as a four-bit residual from its page
mean, with four-channel L2-refined scales and INT8 page summaries. The new
expert-major kernel consumes that representation directly during complete-leaf
prefill instead of materializing a BF16 fallback. K2 uses an M64/N16/four-warp
INT4 tile, amortizes decode cache updates over 512 tokens while retaining the
pending tail exactly, and distributes the coarse value reduction over D=32
tiles. BF16 retains the faster M32/N16/two-warp geometry and 256-token update
interval. The page archive and recursive decode behavior are unchanged by the
complete-centroid prefill consumer.

The table below uses eight distinct ProLong prompts, TP1, a 16K aggregate vLLM
scheduler budget, full prompt lengths, one warm pass, and 1,025 measured decode
tokens. Four ordinary BF16 updates or two INT4 updates are therefore included
in every decode average. Timings are one measured repetition; the matched full
attention controls were rerun with the same zero-reserve prompt convention.

| Context | Full prefill (s) | Three-tier BF16 (s) | Three-tier INT4 (s) | Full decode (ms/step) | BF16 decode | INT4 decode |
|---:|---:|---:|---:|---:|---:|---:|
| 16K | 20.933 | 24.165 | 24.624 | 49.658 | 52.304 | 52.890 |
| 32K | 51.147 | 64.902 | 67.806 | 56.562 | 54.179 | 54.448 |
| 64K | 140.101 | 156.377 | 167.274 | 69.732 | 56.234 | 56.433 |

At 16K, constructing the recursive cache costs BF16 15.4% in prefill and 5.3%
in decode versus native full attention. The decode crossover occurs between
16K and 32K: BF16/INT4 are respectively 4.4%/3.9% faster at 32K and 24.0%/23.6%
faster at 64K. Their decode time rises by only 3.93/3.54 ms from 16K to 64K,
while native full attention rises by 20.07 ms. Prefill is not yet a throughput
win, but its BF16 deficit contracts from 26.9% at 32K to 11.6% at 64K; INT4
pays additional inline reconstruction work in exchange for the smaller cache.

At 128K/B8, complete-centroid INT4 prefill takes 408.777 s versus 522.569 s
for the one-page-per-centroid consumer, a 21.8% reduction in elapsed time (or
27.8% greater throughput). Decode is effectively identical at 59.036 versus
58.932 ms/step. The complete-centroid path is therefore the automatic K2
policy through the model's 128K range. From 64K to 128K, its prefill time grows
2.44x while sequence length doubles, and decode grows by only 4.6%. Full
attention and recursive BF16 do not fit this B8/128K memory point on one
MI325X; INT4 uses 126.451 GB of semantic cache and 176.083 GB total after
reclaim.

At 64K/B8, the INT4 semantic cache is 73.737 GB versus 178.551 GB for
recursive BF16 and 183.375 GB for the native full-attention KV allocation. It
therefore reduces semantic cache storage by 58.7% versus BF16 recursive LOD and
59.8% versus full attention. No chronological native attention cache or BF16
leaf shadow is retained. Total post-reclaim device use is 121.922 GB for INT4
LOD versus 245.631 GB for full attention.

Quality was measured on the same pinned 64K ProLong documents 8--15 used by
the earlier K2 panel (524,280 predicted tokens), plus the canonical eight-case
64K NIAH-S3 smoke panel.

| 64K mode | ProLong CE | PPL | Delta from full | NIAH-S3 |
|---|---:|---:|---:|---:|
| Full attention | 0.521554 | 1.684644 | -- | not rerun |
| Three-tier BF16 | 0.524174 | 1.689063 | +0.002620 | 8/8 |
| Three-tier INT4 | 0.524734 | 1.690009 | +0.003179 | 8/8 |

INT4 adds only 0.000560 CE over the BF16 approximation. The complete-centroid
consumer also improves BF16's CE by 0.00430 over the older one-page-prefill
three-tier run. The focused kernel check reconstructs exactly the same output
and LSE as explicit dequantization (output MSE 0.0); its error against original
BF16 leaves was 0.000585 MSE.

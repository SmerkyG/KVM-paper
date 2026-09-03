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

The 8K and 16K timing files use a full 1,025-token warmup. The 32K and 64K
records used a 257-token warmup but were retained because all three measured
repetitions were already stable. The benchmark helper now defaults to warming
the full requested decode length.

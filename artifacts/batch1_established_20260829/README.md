# Established LOD batch-one context panels (2026-08-29)

This panel returns to the established high-quality BF16 kernels after the
local/sink-overlap, staged-attention, pilot-threshold, FP8-routing, and
centroid-major experiments. It compares native full attention, primary
two-tier LOD, and recursive three-tier LOD at batch one.

## Qwen3.5-0.8B

All arms use one MI325X GPU, real non-repeating ProLong text, chat formatting,
thinking disabled, a 256-token prompt reserve, 256 timed decode tokens, three
repetitions, cold prefill, and a 16,384-token scheduler/prefill chunk limit.
Full attention uses `ROCM_AITER_UNIFIED_ATTN`. Times are complete-model prompt
prefill and complete-model marginal decode latency, not isolated attention
kernel timings. Lower latency is better; each speedup is `full / LOD`.

| Context | Full prefill | Two-tier prefill | Full / two | Three-tier prefill | Full / three | Full decode | Two-tier decode | Full / two | Three-tier decode | Full / three |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | **0.059 s** | 0.075 s | 0.78x | 0.068 s | 0.86x | 1.833 ms | 1.768 ms | 1.04x | **1.698 ms** | **1.08x** |
| 16K | 0.153 s | 0.157 s | 0.97x | **0.149 s** | **1.03x** | 1.936 ms | 1.792 ms | 1.08x | **1.708 ms** | **1.13x** |
| 32K | 0.413 s | 0.338 s | 1.22x | **0.306 s** | **1.35x** | 2.083 ms | 1.839 ms | 1.13x | **1.743 ms** | **1.20x** |
| 64K | 1.285 s | 0.733 s | 1.75x | **0.649 s** | **1.98x** | 2.331 ms | 1.893 ms | 1.23x | **1.766 ms** | **1.32x** |
| 128K | 4.421 s | 1.599 s | 2.77x | **1.450 s** | **3.05x** | 2.840 ms | 2.036 ms | 1.39x | **1.795 ms** | **1.58x** |

Two-tier uses routed top-three prefill with the expert-major exact-leaf path,
then unrestricted top-eight decode with direct route activation, the fixed
M16/N64 page-size-one HIP scan, 256 adaptive segments at B1, and split-D64
reduction. The execution audit records both fixed-mask and HIP GQA-union
execution.

Three-tier uses hierarchical top-three direct prefill with indexed recursive
residual-page attention. Decode uses the established re-split/materialized-GQA
state route and indexed residual-page kernel; it does not inherit the
two-tier-only fixed-list switch. The execution audit records the intended
`resplit` backend.

At batch one, two-tier cold prefill crosses over between 16K and 32K. Recursive
three-tier is at parity by 16K and is the fastest established LOD path at every
measured length. At 128K, three-tier is 3.05x faster for prefill and 1.58x
faster for decode. This is a speed-only rerun of previously qualified
high-quality configurations; NIAH was not repeated in this panel.

Raw records:

- `qwen08_full_b1_r3.json`
- `qwen08_two_b1_r3.json`
- `qwen08_three_b1_r3.json`

## Qwen3.8-27B-FP8

The same panel was repeated on Qwen3.8-27B-FP8 at TP1. The model weights are
FP8, while the LOD K/V state is BF16. All other methodology is unchanged:
real non-repeating ProLong text, chat formatting, thinking disabled, cold 16K
chunked prefill, 256 timed decode tokens, three repetitions, and native
`ROCM_AITER_UNIFIED_ATTN` for full attention. Each speedup is `full / LOD`.

| Context | Full prefill | Two-tier prefill | Full / two | Three-tier prefill | Full / three | Full decode | Two-tier decode | Full / two | Three-tier decode | Full / three |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 0.968 s | **0.955 s** | **1.013x** | 0.965 s | 1.003x | 28.683 ms | **28.522 ms** | **1.006x** | 29.250 ms | 0.981x |
| 16K | 2.154 s | **1.968 s** | **1.094x** | 1.981 s | 1.087x | 29.405 ms | **28.651 ms** | **1.026x** | 29.310 ms | 1.003x |
| 32K | 5.231 s | **4.085 s** | **1.280x** | 4.110 s | 1.273x | 30.047 ms | **28.668 ms** | **1.048x** | 29.350 ms | 1.024x |
| 64K | 14.046 s | **8.553 s** | **1.642x** | 8.579 s | 1.637x | 31.404 ms | **28.877 ms** | **1.088x** | 29.406 ms | 1.068x |
| 128K | 42.534 s | **18.083 s** | **2.352x** | 18.205 s | 2.336x | 34.181 ms | **29.251 ms** | **1.169x** | 29.648 ms | 1.153x |

Unlike the 0.8B model, two-tier is the fastest established path for both
phases at every measured length on the 27B model. The two LOD prefills are
close, but recursive decode pays roughly 0.4--0.7 ms more fixed overhead per
token. Two-tier is already approximately tied with full attention at 8K and
reaches 2.35x prefill and 1.17x decode speedup at 128K.

The two-tier audit records the intended expert-major prefill, fixed M16/N64
page-size-one HIP scan, 256 adaptive batch-one segments, split-D64 reduction,
and direct route activation. The recursive audit records hierarchical
residual-page prefill and the `resplit` decode backend.

Raw records:

- `qwen38_full_b1_r3.json`
- `qwen38_two_b1_r3.json`
- `qwen38_three_b1_r3.json`

## Qwen3.8-27B-FP8 batch-eight rerun

> **Superseded speed table:** the results below preserve the original August
> 29 established panel, before the final INT4 prefill optimizations. The
> authoritative current-code comparison, including full attention, two-tier
> BF16, three-tier BF16, and three-tier INT4 under a matched 16K aggregate and
> 16K per-request scheduler configuration, is in
> `../int4_context_panel_20260829/README.md`.

The matched B8 panel intentionally limits the entire scheduler iteration to
16,384 prompt tokens. It does not admit a separate 16K chunk for every
request. All arms use `max_num_batched_tokens=16,384`, a 16,384-token
per-request threshold, real non-repeating ProLong text, chat formatting,
thinking disabled, a 256-token prompt reserve, 256 timed decode tokens, and
three repetitions. The eight prompt hashes are distinct within each length
and identical across full, two-tier, and three-tier arms.

### Combined B1/B8 prefill, including INT4

Times are complete-model cold-prefill wall time in seconds. The B1 and B8
columns share the same 16K aggregate scheduler limit. `three INT4` uses the
recursive G4/L2 format: four-channel page-wide INT4 leaf groups, L2-refined
scales for prefill conversion and decode appends, and INT8 page summaries.

| Context | B1 full | B1 two BF16 | B1 three BF16 | B1 three INT4 | B8 full | B8 two BF16 | B8 three BF16 | B8 three INT4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 0.968 | **0.955** | 0.965 | 0.994 | **7.553** | 7.584 | 7.723 | 8.758 |
| 16K | 2.154 | **1.968** | 1.981 | 2.025 | 17.218 | **15.926** | 16.144 | 19.376 |
| 32K | 5.231 | **4.085** | 4.110 | 4.774 | 41.812 | **33.052** | 33.528 | 41.787 |
| 64K | 14.046 | **8.553** | 8.579 | 10.526 | 112.544 | **69.120** | 70.018 | 87.744 |
| 128K | 42.534 | **18.083** | 18.205 | 22.724 | 341.448 | **146.699** | 148.615 | 185.843 |

### Combined B1/B8 decode, including INT4

Times are complete-model marginal decode latency in milliseconds per batch
step. A B8 step generates eight tokens concurrently.

| Context | B1 full | B1 two BF16 | B1 three BF16 | B1 three INT4 | B8 full | B8 two BF16 | B8 three BF16 | B8 three INT4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 28.683 | **28.522** | 29.250 | 29.734 | 37.929 | 35.933 | **35.241** | 36.230 |
| 16K | 29.405 | **28.651** | 29.310 | 29.945 | 41.399 | 36.257 | **35.276** | 36.015 |
| 32K | 30.047 | **28.668** | 29.350 | 29.989 | 46.049 | 36.385 | **35.319** | 36.213 |
| 64K | 31.404 | **28.877** | 29.406 | 30.067 | 54.956 | 36.812 | **35.382** | 36.188 |
| 128K | 34.181 | **29.251** | 29.648 | 30.162 | 71.781 | 37.587 | **35.542** | 36.325 |

At B8, two-tier is the fastest prefill path from 16K onward and reaches a
2.33x speedup at 128K. Recursive three-tier is the fastest decode path at
every length and reaches a 2.02x speedup at 128K. At B1, two-tier remains the
fastest Qwen3.8 path in both phases, although the recursive prefill numbers are
close.

INT4 is a memory-oriented option on this kernel set. Relative to recursive
BF16, its prefill is 3.0--24.8% slower at B1 and 13.4--25.3% slower at B8;
decode is 1.7--2.3% slower at B1 and 2.1--2.8% slower at B8. At 128K/B8 it
still beats full attention by 1.84x in prefill and 1.98x in decode. The
allocated recursive LOD cache falls from 10.449 GB to 3.985 GB at B1 and from
83.592 GB to 31.881 GB at B8, a 61.9% reduction including scale metadata,
summaries, and routing state.

There is no two-tier INT4 column because the flat vLLM two-tier cache and its
fast page-size-one path currently support BF16 and INT8 only; configuration
validation rejects `VLLM_LOD_LEVELS=2 VLLM_LOD_KV_BITS=4`. The table does not
mislabel a recursive fallback as two-tier INT4.

Raw B8 records:

- `../batch8_established_20260829/qwen38_full_b8_r3.json`
- `../batch8_established_20260829/qwen38_full_b8_128_r3.json`
- `../batch8_established_20260829/qwen38_two_b8_r3.json`
- `../batch8_established_20260829/qwen38_three_b8_r3.json`
- `qwen38_three_int4_b1_r3.json`
- `../batch8_established_20260829/qwen38_three_int4_b8_r3.json`

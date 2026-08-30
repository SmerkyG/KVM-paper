# Current Qwen3.8 tier and precision context panel (2026-08-30)

This is the current Qwen3.8-27B-FP8 TP1 decision panel for native full
attention, two-tier BF16 LOD, recursive three-tier BF16 LOD, and recursive
three-tier G4/L2 INT4 LOD. Lower latency is better; the fastest entry in each
row is bold.

Every retained arm uses one MI325X, real distinct ProLong text, chat
formatting, thinking disabled, a 256-token prompt reserve, 256 timed decode
tokens, one warmup, and three measured repetitions. Both
`max_num_batched_tokens` and `long_prefill_token_threshold` are 16,384, so
there is no 4K per-request scheduler cap. LOD's internal 4,096-token state
update granularity is unchanged and is not a scheduler limit. Prefill is
complete-model cold-prefill wall time for the whole batch. Decode is marginal
complete-model latency per batch step; a B8 step emits eight tokens. Values
are medians of the three measured repetitions.

## Batch one

### Prefill

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 0.968 s | **0.955 s** | 0.966 s | 0.974 s |
| 16K | 2.154 s | **1.968 s** | 1.983 s | 1.982 s |
| 32K | 5.231 s | **4.085 s** | 4.120 s | 4.343 s |
| 64K | 14.046 s | **8.553 s** | 8.615 s | 9.310 s |
| 128K | 42.534 s | **18.083 s** | 18.215 s | 19.964 s |

### Decode

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 28.683 ms | **28.522 ms** | 29.282 ms | 29.538 ms |
| 16K | 29.405 ms | **28.651 ms** | 29.367 ms | 29.641 ms |
| 32K | 30.047 ms | **28.668 ms** | 29.405 ms | 29.684 ms |
| 64K | 31.404 ms | **28.877 ms** | 29.456 ms | 29.759 ms |
| 128K | 34.181 ms | **29.251 ms** | 29.528 ms | 29.768 ms |

Two-tier BF16 is the latency choice for both phases at every B1 context.
Three-tier BF16 prefill is within 1.1% of two-tier, but its additional routing
stage costs roughly 0.3--0.8 ms per decode step.

## Batch eight

### Prefill

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **7.553 s** | 7.584 s | 7.742 s | 7.968 s |
| 16K | 17.218 s | **15.926 s** | 16.132 s | 17.153 s |
| 32K | 41.812 s | **33.052 s** | 33.494 s | 36.391 s |
| 64K | 112.544 s | **69.120 s** | 69.793 s | 76.597 s |
| 128K | 341.448 s | **146.699 s** | 148.098 s | 163.317 s |

### Decode

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 37.929 ms | 35.933 ms | **35.208 ms** | 35.496 ms |
| 16K | 41.399 ms | 36.257 ms | **35.247 ms** | 35.598 ms |
| 32K | 46.049 ms | 36.385 ms | **35.369 ms** | 35.741 ms |
| 64K | 54.956 ms | 36.812 ms | **35.340 ms** | 35.751 ms |
| 128K | 71.781 ms | 37.587 ms | **35.406 ms** | 35.868 ms |

At B8, full attention narrowly wins 8K prefill by 0.4%. Two-tier BF16 is the
prefill choice from 16K onward. Three-tier BF16 is the decode choice at every
length, beating two-tier by 0.7--2.2 ms per step and becoming increasingly
advantageous with context length.

INT4 is the memory-oriented choice. Relative to three-tier BF16, current INT4
prefill ranges from parity to 9.6% slower at B1 and from 2.9% to 10.3% slower
at B8. Decode remains within 1.4% in every row. Allocated recursive LOD cache
storage falls from 10.449 GB to 3.985 GB at B1 and from 83.592 GB to 31.881 GB
at B8, a 61.9% reduction including scales, summaries, and routing state.

There is no two-tier INT4 column: the flat two-tier cache and page-size-one
fast path currently support BF16 and INT8, not INT4. The panel does not label
a recursive fallback as two-tier INT4.

## Validation and raw records

All arms contain three raw prefill and three raw decode timings per row. Prompt
hashes match across full, two-tier, and both three-tier arms at every length;
all eight B8 prompts are distinct. The recursive execution audit records
three levels, hierarchical top-three prefill, the `resplit` state-route
backend, indexed residual-page attention, BF16 or INT4 leaves as requested,
and no two-tier fixed-list fallback.

Retained records:

- `../batch1_established_20260829/qwen38_full_b1_r3.json`
- `../batch1_established_20260829/qwen38_two_b1_r3.json`
- `qwen38_three_bf16_b1_th16k_r3.json`
- `qwen38_three_int4_b1_th16k_r3.json`
- `../batch8_established_20260829/qwen38_full_b8_r3.json`
- `../batch8_established_20260829/qwen38_full_b8_128_r3.json`
- `../batch8_established_20260829/qwen38_two_b8_r3.json`
- `qwen38_three_bf16_b8_th16k_r3.json`
- `qwen38_three_int4_b8_th16k_r3.json`

The older JSON files in this directory without `th16k` used a 4K per-request
threshold (and two earlier B1 diagnostics also accidentally used a 4K
aggregate budget). They are retained only as rejected diagnostics and do not
contribute to any table above. The canceled `qwen38_two_bf16_*` 4K reruns are
also excluded; the complete established two-tier records already use the
required 16K/16K scheduler configuration.

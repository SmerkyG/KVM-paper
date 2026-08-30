# Current Gemma tier and precision context panel (2026-08-30)

This is the current `google/gemma-4-26B-A4B-it` TP1 decision panel for native
full attention, two-tier BF16 LOD, recursive three-tier BF16 LOD, and
recursive three-tier G4/L2 INT4 LOD. Lower latency is better; the fastest
entry in each row is bold.

Every arm uses one MI325X, the language model only, real distinct ProLong
text, chat formatting, thinking disabled, a 256-token prompt reserve, 256
timed decode tokens, one warmup, and three measured repetitions. Both
`max_num_batched_tokens` and `long_prefill_token_threshold` are 16,384, so
there is no 4K per-request scheduler cap. LOD's internal 4,096-token state
update granularity is unchanged and is not a scheduler limit. Prefill is
complete-model cold-prefill wall time for the whole batch. Decode is marginal
complete-model latency per batch step; a B8 step emits eight tokens. Values
are medians of the three measured repetitions. The native Gemma control uses
the validated `TRITON_ATTN` D=512 backend.

## Batch one

### Prefill

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **0.230 s** | 0.246 s | 0.245 s | 0.251 s |
| 16K | 0.575 s | **0.493 s** | 0.506 s | 0.514 s |
| 32K | 1.618 s | **1.063 s** | 1.097 s | 1.385 s |
| 64K | 5.064 s | **2.301 s** | 2.398 s | 3.260 s |
| 128K | 17.345 s | **5.040 s** | 5.394 s | 7.482 s |

### Decode

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **5.768 ms** | 6.099 ms | 6.292 ms | 6.430 ms |
| 16K | 7.115 ms | 6.313 ms | **6.141 ms** | 6.472 ms |
| 32K | 8.107 ms | 6.651 ms | **6.336 ms** | 6.495 ms |
| 64K | 9.321 ms | 6.938 ms | **6.198 ms** | 6.550 ms |
| 128K | 12.130 ms | 7.106 ms | **6.426 ms** | 6.524 ms |

Full attention is the B1 choice only at 8K. Two-tier BF16 wins prefill from
16K onward and is 3.44x faster than full attention at 128K. Three-tier BF16
wins decode from 16K onward and is 1.89x faster at 128K.

## Batch eight

### Prefill

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **1.765 s** | 1.902 s | 1.907 s | 2.234 s |
| 16K | 4.575 s | **4.084 s** | 4.230 s | 5.823 s |
| 32K | 12.927 s | **8.758 s** | 9.117 s | 12.868 s |
| 64K | 40.481 s | **19.021 s** | 20.143 s | 28.485 s |
| 128K | 137.626 s | **41.845 s** | 44.628 s | 62.785 s |

### Decode

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **9.157 ms** | 10.127 ms | 9.693 ms | 10.087 ms |
| 16K | 10.452 ms | 10.473 ms | **9.809 ms** | 10.030 ms |
| 32K | 11.556 ms | 10.635 ms | **9.666 ms** | 10.005 ms |
| 64K | 12.783 ms | 11.197 ms | **10.023 ms** | 10.084 ms |
| 128K | 15.542 ms | 11.303 ms | **9.812 ms** | 9.973 ms |

The B8 decision is the same: full attention at 8K, two-tier BF16 prefill from
16K onward, and three-tier BF16 decode from 16K onward. At 128K, two-tier
prefill is 3.29x faster than full attention and three-tier decode is 1.58x
faster.

INT4 is a capacity option on Gemma, not a latency option. Relative to
three-tier BF16, INT4 prefill is up to 38.7% slower at B1 and 41.4% slower at
B8; decode is within 5.7% at B1 and 4.1% at B8. Allocated recursive LOD cache
storage falls from 3.235 GB to 1.214 GB at B1 and from 25.878 GB to 9.713 GB
at B8, a 62.5% reduction including scales, summaries, and routing state.

There is no two-tier INT4 column: the flat two-tier cache and page-size-one
fast path currently support BF16 and INT8, not INT4. The panel does not label
a recursive fallback as two-tier INT4.

## Validation and raw records

All arms contain three raw prefill and three raw decode timings per row. Prompt
hashes match across full, two-tier, and both three-tier arms at every length;
all eight B8 prompts are distinct. Full attention reports `TRITON_ATTN`. LOD
reports five eligible global-attention layers and the geometry-selected
`fused` state-route backend, with BF16 or INT4 leaves as requested and no
attention fallback. The runtime was capped at 131,152 tokens for this panel;
Gemma's text configuration advertises 262,144 positions.

Retained records:

- `gemma_full_b1_r3.json`
- `gemma_two_bf16_b1_r3.json`
- `gemma_three_bf16_b1_r3.json`
- `gemma_three_int4_b1_r3.json`
- `gemma_full_b8_r3.json`
- `gemma_two_bf16_b8_r3.json`
- `gemma_three_bf16_b8_r3.json`
- `gemma_three_int4_b8_r3.json`

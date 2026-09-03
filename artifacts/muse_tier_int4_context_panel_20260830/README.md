# Current Muse tier and precision context panel (2026-08-30)

This is the current `meta-models/Muse-Glimmer-30B` TP1 decision panel for
native full attention, two-tier BF16 LOD, recursive three-tier BF16 LOD, and
recursive three-tier G4/L2 INT4 LOD. Lower latency is better; the fastest
entry in each row is bold.

Every arm uses one MI325X, real distinct ProLong text, Muse's native text
configuration, chat formatting, thinking disabled, a 256-token prompt
reserve, 256 timed decode tokens, one warmup, and three measured repetitions.
Both `max_num_batched_tokens` and `long_prefill_token_threshold` are 16,384,
so there is no 4K per-request scheduler cap. LOD's internal 4,096-token state
update granularity is unchanged and is not a scheduler limit. Prefill is
complete-model cold-prefill wall time for the whole batch. Decode is marginal
complete-model latency per batch step; a B8 step emits eight tokens. Values
are medians of the three measured repetitions.

## Batch one

### Prefill

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **0.695 s** | 0.753 s | 0.765 s | 0.769 s |
| 16K | **1.387 s** | 1.486 s | 1.533 s | 1.548 s |
| 32K | **2.960 s** | 3.103 s | 3.242 s | 3.348 s |
| 64K | 6.490 s | **6.477 s** | 6.856 s | 7.206 s |
| 128K | 14.946 s | **13.478 s** | 14.706 s | 15.703 s |

### Decode

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **15.548 ms** | 16.331 ms | 16.397 ms | 16.435 ms |
| 16K | **15.584 ms** | 16.605 ms | 16.446 ms | 16.400 ms |
| 32K | **15.636 ms** | 16.761 ms | 16.464 ms | 16.441 ms |
| 64K | **15.755 ms** | 17.095 ms | 16.443 ms | 16.412 ms |
| 128K | **16.003 ms** | 17.268 ms | 16.471 ms | 16.601 ms |

Full attention is the B1 decode choice at every length and the prefill choice
through 32K. Two-tier BF16 prefill is essentially tied at 64K and wins by
1.11x at 128K.

## Batch eight

### Prefill

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **5.458 s** | 6.058 s | 6.116 s | 6.197 s |
| 16K | **11.379 s** | 12.440 s | 12.746 s | 13.199 s |
| 32K | **24.029 s** | 25.537 s | 26.454 s | 27.594 s |
| 64K | **52.212 s** | 52.724 s | 55.339 s | 58.219 s |
| 128K | 120.432 s | **109.322 s** | 118.006 s | 125.519 s |

### Decode

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | **18.831 ms** | 20.371 ms | 20.030 ms | 20.095 ms |
| 16K | **18.828 ms** | 20.555 ms | 20.039 ms | 20.099 ms |
| 32K | **19.252 ms** | 21.014 ms | 20.120 ms | 20.117 ms |
| 64K | **19.846 ms** | 21.489 ms | 20.085 ms | 20.060 ms |
| 128K | 20.919 ms | 21.933 ms | **20.041 ms** | 20.060 ms |

At B8, full attention wins prefill through 64K; two-tier BF16 crosses at 128K
and is 1.10x faster. Full attention wins decode through 64K. Recursive
three-tier BF16 crosses at 128K, where it is 1.04x faster than full attention
and 1.09x faster than two-tier.

INT4 is the memory-oriented recursive choice rather than the latency choice.
Relative to three-tier BF16, INT4 prefill is 0.5--6.8% slower at B1 and
1.3--6.4% slower at B8, while decode stays within 1.0% in every row. Allocated
recursive LOD cache storage falls from 2.162 GB to 0.850 GB at B1 and from
17.298 GB to 6.800 GB at B8, a 60.7% reduction including scales, summaries,
and routing state.

There is no two-tier INT4 column: the flat two-tier cache and page-size-one
fast path currently support BF16 and INT8, not INT4. The panel does not label
a recursive fallback as two-tier INT4.

## Validation and raw records

All arms contain three raw prefill and three raw decode timings per row. Prompt
hashes match across full, two-tier, and both three-tier arms at every length;
all eight B8 prompts are distinct. Full attention reports the
`ROCM_AITER_UNIFIED_ATTN` backend. The recursive Muse dispatch reports the
geometry-selected `fused` state-route backend, BF16 or INT4 leaves as
requested, and no attention fallback. The 128K prompt is 130,816 tokens, so
the 256-token timed decode ends exactly at Muse's advertised 131,072-token
limit.

Retained records:

- `muse_full_b1_r3.json`
- `muse_two_bf16_b1_r3.json`
- `muse_three_bf16_b1_r3.json`
- `muse_three_int4_b1_r3.json`
- `muse_full_b8_r3.json`
- `muse_two_bf16_b8_r3.json`
- `muse_three_bf16_b8_r3.json`
- `muse_three_int4_b8_r3.json`

# AITER page-size-1 M sweep (2026-08-23)

## Setup

- GPU: AMD Instinct MI325X / gfx942
- AITER operator: `mha_batch_prefill_func`
- KV page size: 1
- BF16 Q/K/V, D=128, noncausal
- Every sequence owns a distinct physical KV stream
- M: 16, 64, 128 query rows sharing each sequence's KV
- Seven or nine timing trials after warm-up; medians reported

The main long-context sweep uses 8--128 independent sequences and K=4K,
16K, and 64K. Short sweeps cover K=16--4K.

## Fixed number of independent KV sequences

The table below uses 64 independent sequences. Increasing M therefore
increases useful attention work while leaving physical KV traffic unchanged.

| K | M16 latency | M64 latency | M128 latency | M64 useful throughput / M16 | M128 useful throughput / M16 |
|---:|---:|---:|---:|---:|---:|
| 16 | 14.16 us | 14.27 us | 14.38 us | 3.97x | 7.88x |
| 64 | 14.17 us | 14.25 us | 14.28 us | 3.98x | 7.94x |
| 128 | 14.15 us | 14.35 us | 14.43 us | 3.95x | 7.85x |
| 256 | 14.23 us | 14.21 us | 14.32 us | 4.01x | 7.95x |
| 512 | 20.39 us | 21.38 us | 22.54 us | 3.81x | 7.24x |
| 1K | 38.49 us | 39.60 us | 40.17 us | 3.89x | 7.67x |
| 4K | 176.43 us | 177.61 us | 179.14 us | 3.97x | 7.83x |
| 16K | 808.44 us | 810.67 us | 813.02 us | 3.99x | 7.96x |
| 64K | 3.203 ms | 3.207 ms | 3.217 ms | 4.00x | 7.97x |

M64 and M128 process four and eight times as many useful QK/PV pairs at
almost unchanged latency. Thus the page-size-1 AITER path is overwhelmingly
KV/dispatch limited in this range, and M16 uses only about one quarter of the
useful-row throughput available at M64 and one eighth of M128 throughput.

K=16--256 all take approximately 14 us. Underfilled N therefore does not make
individual calls cheaper: it consumes the same launch/minimum-tile cost while
doing proportionally less useful work.

## Fixed total query rows

This comparison holds useful QK/PV work fixed at 1,024 query rows:

- M16: 64 sequences
- M64: 16 sequences
- M128: 8 sequences

| K | M16 | M64 | M128 | M64 latency change | M128 latency change |
|---:|---:|---:|---:|---:|---:|
| 128 | 14.24 us | 14.39 us | 14.41 us | 1.0% slower | 1.2% slower |
| 256 | 14.23 us | 14.41 us | 14.42 us | 1.3% slower | 1.3% slower |
| 512 | 20.39 us | 21.15 us | 21.18 us | 3.8% slower | 3.9% slower |
| 1K | 38.49 us | 37.82 us | 37.83 us | 1.7% faster | 1.7% faster |
| 4K | 176.43 us | 164.23 us | 154.02 us | 6.9% faster | 12.7% faster |
| 16K | 808.44 us | 688.62 us | 685.53 us | 14.8% faster | 15.2% faster |
| 64K | 3.203 ms | 3.181 ms | 2.793 ms | 0.7% faster | 12.8% faster |

The latency benefit becomes much smaller when larger M is obtained by
reducing the number of independent sequences, because occupancy falls. This
is the relevant caveat for AR decode: unrelated batch rows or KV heads cannot
be converted into a true M128 attention problem because their K/V differs.

## Implications

1. The user's M-efficiency concern is confirmed. With the same independent KV
   streams, M128 obtains roughly 7.8--8.0x M16's useful throughput and M64
   obtains roughly 3.8--4.0x.
2. Very short or underfilled K lists sit on a roughly 14-us floor and waste
   nearly all of their potential throughput.
3. Increasing M does not automatically reduce end-to-end latency when it also
   removes independent sequences. At fixed useful work, the measured gain was
   generally 0--17%.
4. LoD should therefore concatenate valid selected KV across centroid/page
   boundaries to fill N tiles, but it cannot recover true M64/M128 reuse in
   GQA16 AR decode without additional query rows that genuinely share the
   same KV sequence.

Raw results:

- `m16_m64_m128_d128_seqpanel.json`: K=4K--64K and sequence-count sweep.
- `m16_m64_m128_d128_shortk.json`: K=128--4K.
- `m16_m64_m128_d128_tinyk.json`: K=16--128.

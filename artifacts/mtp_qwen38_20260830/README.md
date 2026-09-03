# Native MTP target-attention study (2026-08-30)

This directory compares one-token native MTP on
`Qwen/Qwen3.8-27B-FP8`, TP1, batch 1, using real, non-repeating ProLong
documents. Decode latency is the median of three 256-output-token runs and is
reported per emitted output token. Long prefill uses the repository's 16K
aggregate scheduler budget. Full attention uses native AITER; LOD is the
two-tier BF16, unrestricted-top-8 target path.

## Result

The original LOD integration verified the MTP proposal by invoking the whole
M=1 LOD decoder twice in a Python loop. Full attention instead verified the
two target positions in one native M=2 AITER call. The corrected path stages
both proposed K/V entries, represents the two positions as one artificial
batch, and runs one routed LOD pipeline. The second position sees the first
proposal through its exact causal local suffix; both positions route against
the same immutable remote state. Rejected suffix entries are restored to the
scheduler's committed prefix before the next proposal.

| context | full MTP | old serial LOD MTP | parallel LOD MTP | parallel vs full | parallel vs old |
|---:|---:|---:|---:|---:|---:|
| 8K | 18.872 ms | 21.319 ms | **18.628 ms** | **1.013x** | **1.144x** |
| 16K | **19.003 ms** | 21.093 ms | 19.322 ms | 0.983x | **1.092x** |
| 32K | **20.561 ms** | 21.291 ms | 20.814 ms | 0.988x | **1.023x** |
| 64K | 22.761 ms | 22.331 ms | **20.115 ms** | **1.132x** | **1.110x** |
| 128K | 28.400 ms | 22.898 ms | **21.742 ms** | **1.306x** | **1.053x** |

These are output-token times, so acceptance changes affect them. The logged
64K mean draft acceptance was 84.8% for full attention, 80.9% for old serial
LOD, and 88.2% for corrected LOD. Their approximate target cycles were
42.06, 40.40, and 37.86 ms, respectively. At 128K, full/old/new acceptance
was 80.9/80.9/75.9%, corresponding to approximately 51.37/41.42/38.24 ms per
cycle. The corrected kernel therefore improves actual target-cycle latency as
well as output-normalized throughput. At 32K it has a small target-cycle win
but a lower acceptance rate makes it 1.2% slower per emitted token than full
MTP; verifier sparsity affects both compute and acceptance.

A route-group sweep from 32 to 64 changed 64K output latency from 20.286 to
20.232 ms (0.27%), within the range where the extra specialization is not
worth making the default.

## Why the old path was slow

The premise also mixed batching regimes. The often-quoted Qwen3.8 64K result
is the batch-8 panel (52.030 ms full versus 36.334 ms two-tier LOD, 1.43x).
This native-MTP study is batch 1. In the matched non-speculative batch-1
records, full/LOD were 31.404/28.877 ms at 64K, only a 1.087x end-to-end win.
MTP did expose an additional, real integration problem:

1. It serialized two complete M=1 LOD calls, repeating coarse QK/PV, top-8
   reduction, exact/local attention, and final LSE reduction.
2. Qwen's six query heads per KV head were padded into an M=16 route tile
   separately for each target position. The corrected scorer packs both
   positions as 12 useful rows and loads each centroid K/V tile once.
3. The draft proposer, sampling, non-attention target work, and MTP book-
   keeping are unchanged by LOD. With only one drafted token, those fixed
   costs substantially dilute the saving from sparse target attention.
4. The old 64K LOD run also had lower draft acceptance than full attention,
   further hiding its target-cycle advantage in output-token latency.

## Matched profile

The 64K `torch.profiler` trace covers ten speculative target cycles. Profiling
adds substantial overhead, so the trace is used only to apportion LOD work,
not as an end-to-end speed record.

| corrected LOD component | time per cycle | share of LOD kernels |
|---|---:|---:|
| exact leaves + 512-token local suffix | 1.926 ms | 58.6% |
| top-8 route/coarse reduction | 0.781 ms | 23.8% |
| shared two-position coarse QK/PV | 0.316 ms | 9.6% |
| final LSE merge | 0.095 ms | 2.9% |
| proposal K/V preparation | 0.087 ms | 2.6% |
| cache-length advance | 0.083 ms | 2.5% |
| **total** | **3.287 ms** | **100%** |

The main optimization opportunity is now exact leaf/local attention, followed
by the small-program top-8 reduction. Coarse scoring is no longer the dominant
problem. The raw trace and summary are under `torch_profile_shared_64k/`.

## Correctness and execution audit

- Corrected parallel LOD MTP scores 8/8 on chat-formatted NIAH-S3 at both 8K
  and 64K; the 64K run logged 100% draft acceptance.
- The speed and quality harnesses carry a device execution marker and fail if
  the captured graph does not execute the shared two-position route scorer.
- The parallel path preserves exact causal visibility and performs no lagged
  routing. It is enabled only for the supported two-position, two-tier target
  geometry; unsupported configurations retain the serial correctness path.

## Raw records

- `full_b1_8k_128k_r3_d256.json`: native full-attention MTP control.
- `lod_b1_8k_128k_r3_d256.json`: original serial LOD-MTP control.
- `lod_parallel_sharedroute_b1_8k_128k_r3_d256.json`: consolidated corrected
  speed record, with prompt hashes matched to the full and old-LOD panels.
- `lod_parallel_sharedroute_b1_64k_r3_d256.json` and
  `lod_parallel_sharedroute_b1_128k_r3_d256.json`: isolated diagnostics.
- `lod_parallel_sharedroute_group64_b1_64k_r3_d256.json`: route-group-64
  diagnostic.
- `lod_parallel_sharedroute_niah_s3_64k_b1_n8.json`: 64K quality record.
- `torch_profile_shared_64k/`: corrected target trace and profiler table.

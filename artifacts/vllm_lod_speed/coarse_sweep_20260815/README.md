# Ragged prefill phase profile and coarse fusion

Qwen3.5-35B-A3B, batch 8, vLLM 0.27.1, MI325X, INT4 recursive LOD,
16K scheduler token budget. Timings exclude model load and JIT warmup.

## Phase profile

The ragged exact-local branch is not the prefill bottleneck:

| Prompt bucket | Measured prefill | Exact local | Share |
| --- | ---: | ---: | ---: |
| 11-18K | 5,426.9 ms | 139.1 ms | 2.56% |
| 19-27K | 8,414.3 ms | 212.9 ms | 2.53% |
| 33-54K | 20,124.0 ms | 399.1 ms | 1.98% |

The fine 19-27K profile measured 8,839.4 ms total. Its important LOD phases
were coarse remainder 2,504.8 ms (28.34%), exact recursive leaves 1,233.9 ms
(13.96%), state update 746.7 ms (8.45%), routing 361.1 ms (4.09%), and direct
pool/orchestration residual 349.7 ms (3.96%). Exact local was 205.6 ms (2.33%).

Sources: `../phase_profile_20260815/35b_short-a.json`,
`../phase_profile_20260815/35b_short-b.json`,
`../phase_profile_20260815/35b_medium.json`, and
`../phase_profile_20260815/fine_35b_short_b.json`.

## Optimization

Fusing top-k route selection with the coarse remainder and using a bounded
M16/N32/W8 tile avoids a second scan of routing logits. The prior split
M32/N64 path remains useful as a fallback, while a fused M32/N64 tile exceeded
the MI325X shared-memory limit.

On the 19-27K bucket, the first no-profiler repeat-3 confirmation was 6.280 s
versus 8.724 s for the old default (28.0% faster). A default-only confirmation
was 6.617 s (24.2% faster); the difference tracks vLLM ragged scheduler packing
(480 versus 510 layer-level direct-prefill calls across warmup plus repeats).
The optimized full-attention AITER control was 6.139 s. Thus optimized LOD was
2.3-7.8% slower in these two scheduler realizations, rather than 42% slower.

On the 11-18K bucket, a same-node no-profiler comparison measured 4.474 s
fused versus 5.213 s split (14.2% faster). Optimized full AITER was 3.646 s, so
LOD still trails full attention at this shorter range, as expected.

Do not compare against `full_control_r3.json`: that run selected ROCM_ATTN and
took 15.98 s. The fair full baseline is `full_aiter_control_r3.json`, which
explicitly selects `ROCM_AITER_UNIFIED_ATTN`.

Kernel verification after the residual-local-LSE fix produced exact dynamic
route sets and opening counts, and the fixed-pool BF16 and INT4 parity tests
both passed.

# Recursive INT4 prefill optimization (2026-08-29)

The authoritative three-repeat 8K--128K panel uses a 16K aggregate scheduler
budget and a 16K per-request threshold. At 64K/B8, current recursive
BF16/INT4 prefill is 69.793/76.597 seconds (9.7% INT4 overhead), and decode is
35.340/35.751 ms (1.2%). See `../int4_context_panel_20260829/README.md` for the
full native/two-tier/three-tier B1/B8 decision tables. No authoritative timing
uses the rejected 4K per-request scheduler cap.

This experiment optimized the recursive three-tier INT4 path without changing
its cache format, quantization policy, routing, or opened leaves.  GPU runs used
`cluster-run`, real distinct ProLong document text with chat formatting, batch
size eight, and a 16,384-token aggregate scheduler budget.

## Development diagnostic

The following single-run 64K measurements used a 4K per-request threshold
during kernel development. They isolate the optimization sequence but are not
serving results and are not used in the authoritative context panel. In that
diagnostic, INT4 prefill is 73.606 seconds versus
67.959 seconds for BF16 LOD: an 8.3% remaining penalty.  The previous
established INT4 result was 87.744 seconds, so the optimized path is 16.1%
faster.  It remains 1.50x faster than the matched historical 110.565-second
full-attention prefill.  Decode did not regress.

| Recursive three-tier mode | Prefill | Decode batch step | Notes |
| --- | ---: | ---: | --- |
| BF16 control | 67.959 s | 33.867 ms | current code, 4K long-prefill threshold |
| INT4, matched call geometry | 75.460 s | 34.599 ms | before the final shared-anchor factoring |
| INT4, final current code | 73.606 s | 34.679 ms | includes all optimizations below |
| Previous established INT4 | 87.744 s | 36.188 ms | older 16K-threshold panel configuration |
| Historical full attention | 110.565 s | 52.030 ms | matched 64K/B8 full-attention panel |

The current INT4 and BF16 end-to-end runs naturally produced different cached
prefill batching under asynchronous scheduling.  The strictly call-matched
BF16/INT4 pair is 67.959/75.460 seconds (an 11.0% INT4 penalty); the final
shared-anchor change was then measured independently on identical small-model
call geometry.  The current end-to-end row is the serving result, while the
call-matched row isolates the preceding kernel changes.

## Changes

1. The four-channel INT4 page quantizer now uses one wave.  Its old four-wave
   launch reduced only 64 values per program and left most lanes idle.
2. Final conversion and cached-prefill append quantize four adjacent channel
   groups per program.  At the representative B1/KVH2/4,096-page/D256
   geometry, final conversion fell from 0.8514 ms to 0.2429 ms (3.50x) while
   producing bit-identical codes and scales.
3. Finalized quantized caches no longer retain a dead mixed-BF16 fallback in
   residual-page attention.  Conversion and append publish complete changed
   pages before attention observes them.
4. Residual-page attention factors the shared page mean out of its 16-token
   QK and PV matrices.  On identical Qwen3.5-0.8B 64K/B8 call geometry, this
   reduced exact-leaf time by 9.9%, the inclusive two-level phase by 5.1%, and
   end-to-end prefill by 2.3%.

The launch choices can still be swept with
`VLLM_LOD_INT4_QUANT_NUM_WARPS` and
`VLLM_LOD_INT4_QUANT_GROUPS_PER_PROGRAM`.

## Correctness and quality

- Grouped final conversion is bit-identical to the legacy single-group kernel
  for codes, scales, and published counts.
- The four-group append path is exact across codes, scales, summaries, page
  counts, and page metadata in the cached-prefill append checker.
- Qwen3.5-0.8B retained 64/64 NIAH-S3 at 8K/B8 with final current code.
- The storage format and established G4 L2 quantization quality policy are
  unchanged.

Primary artifacts:

- `int4_quant_group_sweep_v2.json`
- `int4_append_group_check.json`
- `qwen08_int4_algebra_64k_b8_r1.json`
- `qwen08_int4_final_niah64_8k.json`
- `qwen38_bf16_control_64k_b8_r1.json`
- `qwen38_int4_current_64k_b8_r1.json`

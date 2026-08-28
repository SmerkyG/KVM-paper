# TP4 parallelism panel

This panel measures whether tensor parallelism reduces per-GPU attention
parallelism enough to erode LOD's advantage. All primary rows are vLLM,
batch eight, 64K real distinct ProLong prompts with the checkpoint chat
template, 16K aggregate scheduler chunks, 64 decode tokens, and medians of
three runs. Full attention and LOD use the same TP topology and collective
backend.

vLLM's ROCm custom all-reduce failed during graph-memory profiling on this
four-GPU topology for both full attention and LOD. Every valid TP4 row below
therefore uses PYNCCL. Qwen full attention uses
`ROCM_AITER_UNIFIED_ATTN`. Gemma full attention uses `TRITON_ATTN`: the
available AITER-unified D512 TP4 configuration requested 128 KiB of shared
memory on hardware with a 64 KiB limit and could not start.

## Batch-eight TP1 versus TP4

Times are seconds for prefill of the whole batch and milliseconds per batch
decode step. Parentheses are full-attention time divided by LOD time on the
same topology; larger is better for LOD.

| Model / topology | Full prefill | Two-tier prefill | Three-tier prefill | Full decode | Two-tier decode | Three-tier decode |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B-FP8, TP1 | 110.565 s | 65.062 s (1.70x) | 66.912 s (1.65x) | 52.030 ms | 36.334 ms (1.43x) | 34.883 ms (1.49x) |
| Qwen3.8-27B-FP8, TP4 | 43.696 s | 32.233 s (1.36x) | 32.608 s (1.34x) | 26.111 ms | 23.036 ms (1.13x) | 22.213 ms (1.18x) |
| Gemma-4-26B-A4B, TP1 | 40.063 s | 18.110 s (2.21x) | 19.175 s (2.09x) | 11.694 ms | 10.235 ms (1.14x) | 9.365 ms (1.25x) |
| Gemma-4-26B-A4B, TP4 | 15.793 s | 10.406 s (1.52x) | 10.146 s (1.56x) | 12.432 ms | 11.264 ms (1.10x) | 10.812 ms (1.15x) |

The concern is real: TP4 narrows LOD's relative advantage at batch eight on
both models. It does not reverse it. Qwen's local full-attention geometry is
QH6/KVH1/GQA6 at TP4. Gemma's replicated-KV geometry is QH4/KVH1/GQA4 per
rank, rather than global QH16/KVH2/GQA8. The smaller per-rank query grid is
therefore a plausible cause, especially for LOD's multi-stage kernels.

The executed-path audit confirms two-tier's fixed-mask/union HIP path and
three-tier's selected recursive route. Gemma three-tier uses the
quality-selected fused route; Qwen uses re-split.

## Qwen TP4 synchronized batch-32 decode

Cold 64K prefill with a 16K aggregate scheduler budget admits 32 requests in
waves, so it does not provide 32 simultaneous decode rows. For this diagnostic
the same real ProLong prompts are first prefetched once, then replayed from
their vLLM prefix cache. The measured calls have all 32 requests active and
use medians of five runs. This section is a steady-state decode comparison;
its cached-prefix prefill time is not comparable with the cold-prefill table.

| Batch | Full decode | Two-tier decode | LOD speedup | Full throughput | Two-tier throughput |
|---:|---:|---:|---:|---:|---:|
| 8, cold panel | 26.111 ms | 23.036 ms | 1.13x | 306 tok/s | 347 tok/s |
| 32, synchronized | 47.226 ms | 28.096 ms | **1.68x** | 678 tok/s | 1,139 tok/s |

Thus batch 32 more than makes up the Qwen TP4/B8 loss: full-attention
throughput scales 2.21x from B8 to B32, while two-tier LOD scales 3.28x. The
two-tier execution audit reports 512 sequences per step, exactly 32 requests
times 16 eligible LOD layers, and confirms the fixed-mask and page-size-one
HIP/AITER paths executed.

The 1.68x B32 full-versus-LOD ratio is fully matched. The B8-to-B32 scaling
figures are directional rather than a strict one-variable A/B because the
synchronized replay enables vLLM prefix caching (and its Qwen Mamba `align`
cache mode), whereas the canonical cold B8 panel does not.

Three-tier synchronized B32 is unavailable because both the ordinary and an
832-token-aligned replay exposed a partial-prefix restore case in the external
three-tier state. It is not folded into the successful matched full/two-tier
result.

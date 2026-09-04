# Gemma 4 cache-maintenance validation

These checks use `google/gemma-4-26B-A4B-it`, BF16 two-tier LOD, batch 8,
top-3 prefill routing, top-8 decode routing, a `16 * sqrt(T)` state schedule,
and 16K aggregate vLLM scheduler batches. Only Gemma's five global-attention
layers are converted; its 25 sliding-window layers remain native.

The selected 4K direct-prefill defaults scored 8/8 on NIAH-S3 at 64K. On
ProLong documents 8--15, that schedule had cross-entropy 3.92848 over 524,280
prediction tokens. The experimental 16K exact-first/deferred-cache schedule
slightly improved cross-entropy to 3.89334 (-0.03514), but its prompt-logprob
evaluation took 62.19 seconds versus 24.07 seconds. The 512-dimensional Gemma
global heads make that larger exact/local field unattractive, so it is retained
as an opt-in path rather than selected for Gemma.

The same 16K experiment scored 7/8 on the small NIAH-S3 check. The selected 4K
schedule exactly reproduced all eight GUIDs, matching the first eight samples
of the earlier 61/64 BF16 Gemma result.

The selected schedule's warm batch-8 64K timing (three measured repetitions,
with a 256-token decode that includes one state update) was 18.402 seconds for
prefill and 10.354 ms per decode batch step. The prior Gemma BF16 LOD record was
19.021 seconds and 11.197 ms, so the current implementation lowers latency by
3.3% and 7.5%, respectively. Against the recorded full-attention baseline of
40.481 seconds and 12.783 ms, current LOD is 2.20x faster for prefill and 1.23x
faster for decode.

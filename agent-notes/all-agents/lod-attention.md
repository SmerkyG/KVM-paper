# Lod Attention

Topic hints: Gemma D512 favors 4K LOD prefill blocks

## Lessons

- On Gemma 4 26B's 512-dimensional global-attention heads, the 16K exact-first/deferred-cache prefill schedule improved 64K ProLong CE slightly but made prompt-logprob evaluation about 2.6x slower. Keep the 4K direct-prefill schedule as the selected default for this geometry unless the large exact/local field is optimized; the current shared code still improved 64K batch-8 latency to 18.402 s prefill and 10.354 ms decode while preserving 8/8 NIAH-S3.

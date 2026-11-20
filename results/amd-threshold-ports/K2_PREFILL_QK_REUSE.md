# K2 Horizon threshold prefill with QK reuse

Batch one on AMD MI325X, K2 Horizon 32B FP8 weights, BF16 KV, two-tier LoD, and a fixed 65,536-token ProLong prompt. Threshold runs retain the normalized equal-per-region rule with c=16, the 1,024-token region eligibility limit, all eligible opened regions, and the initial 16,384-token exact field. Scratch tiles only partition the implementation of the remote branch. Full-prefill timings are medians of three repetitions after one warmup and include model execution, local attention and cache updates.

## Full prefill

| Routing and merge | Query scratch tile | Median |
|---|---:|---:|
| [Native top-eight](k2-prefill-final-v2/amd-k2-final-top8-speed.json) | Native | 13.42 s |
| [Streaming c=16 with fused merge](k2-select-bitset-v1/amd-k2-bitset-speed-stream.json) | 16K | 16.64 s |
| [One FP32 QK, bitset selection and BLAS coarse projection](k2-select-bitset-v1/amd-k2-bitset-speed-bits16k.json) | 16K | 16.53 s |
| [One FP32 QK with compact exact-output addresses](k2-pack-exact-v1/amd-k2-pack-exact-speed-blas.json) | 16K | 16.46 s |
| [Plus overlap between exact-local attention and remote refinement](k2-prefill-overlap-v1/amd-k2-overlap-speed-blas.json) | 16K | 16.22 s |
| [Plus one BF16 component for exact merge weights](k2-merge-direct-v1/amd-k2-merge-mma1-speed.json) | 16K | 16.18 s |
| [Final source with exact workspace growth fix](k2-prefill-final-v2/amd-k2-final-16k-speed.json) | 16K | **16.17 s** |
| [Same kernels with 8K query scratch](k2-prefill-final-v1/amd-k2-final-8k-speed.json) | 8K | 16.31 s |

The final median is **0.48 s faster** than the streaming c=16 control and **1.47 s above** the 14.70 s target. The final and control runs produced identical greedy output token hashes on the timed prompt. The kernel options are opt-in. [Final launch command](k2-prefill-final-v2/amd-k2-final-16k-speed-submission.json)

## Same-query replay

The middle layer replay uses 4,096 queries from the final 16K chunk, 64 query heads, and 3,547 region slots. Timings use GPU graphs. Remote attention includes routing, dispatch, exact leaves, and merge. The exact-local branch is outside this replay.

| Routing | Route and coarse | Remote attention |
|---|---:|---:|
| Streaming, recomputing QK | 6.13 ms | 11.13 ms |
| [Streaming with row-major FP32 scores](k2-save-qk-v2/amd-k2-save-qk-pv-128-64-1024-4-1-m0.json) | 8.36 ms | 13.34 ms |
| [Streaming with tile-major FP32 scores](k2-save-qk-v2/amd-k2-save-qk-pv-128-64-1024-4-1-m1.json) | 7.29 ms | 12.41 ms |
| [One FP32 BLAS QK with two selector rows and eight warps](k2-select-bitset-v1/amd-k2-select-bitset-2-8.json) | 5.21 ms | 10.67 ms |

Writing scores from the small streaming tiles cost more than recomputing their QK. Saving FP16 scores changed decisions near the cutoff and failed the replay output tolerance. The faster path computes one FP32 score matrix with BLAS, reads it once to evaluate both proxy and native logits, and records compact selection bitsets. The closed-summary projection uses BF16 probabilities and BLAS. In the 64K full-model profile, the route and coarse time fell from **3.47 to 3.31 s**. The 3,547-region chunk accounted for the improvement. [Old stage profile](amd-prefill-penalty-retime-v1/amd-k2-penalty-retime-c16.json) · [New stage profile](k2-select-merge-v1/amd-k2-bitset-full-attribution.json)

Compact exact-output addresses use an exclusive device prefix of the per-query route counts. They preserve canonical route order and the region-grouped KV reads. Leaf time fell from **3.44 to 3.17 ms** in the middle layer, while the merge took about **1.49 ms** with either layout. The workspace still reserves the padded worst-case capacity. [Packed-output replay](k2-pack-exact-v1/amd-k2-pack-exact-middle.json)

The fused merge still uses FP32 partition masses and normalization. For the tensor-core value sum alone, one BF16 weight component replaces the two-component expansion. Merge time fell from **1.49 to 1.39 ms** in the middle layer and by 0.11 ms in both other saved layers. The widest frontiers keep their existing FP32 fallback. Smaller width buckets and carrying the exact partition mass directly gave little end-to-end improvement. [Layer 0](k2-merge-direct-v1/amd-k2-merge-direct-layer0.json) · [Layer 32](k2-merge-direct-v1/amd-k2-merge-direct-layer32.json) · [Layer 63](k2-merge-direct-v1/amd-k2-merge-direct-layer63.json) · [Width buckets](k2-select-merge-v1/amd-k2-merge-buckets-replay.json)

The full instrumented BLAS run spends 3.31 s in routing and coarse attention, 1.43 s in exact token attention, 0.29 s in dispatch, and 1.17 s in the merge and other remote work, before compact addresses and overlap. The fixed-Q replay savings are largest at the longest region axis and translate to a smaller full-model reduction. [Stage attribution](k2-select-merge-v1/amd-k2-bitset-full-attribution.json)

## Validation and memory

The final kernels passed 14 independent PyTorch frontier checks covering empty regions, odd query counts, D128 and D256, and frontiers above 64 regions. With the 8K scratch tile, all eight comparison prompts from the archived 64K RULER multikey subset matched the streaming baseline's prediction text and score. Both tasks scored 4/4. The inputs and targets were verified by hash. [Frontier checks](k2-prefill-final-v1/amd-k2-final-frontier.json) · [Multikey-2](k2-prefill-final-v1/amd-k2-final-8k-ruler-mk2.json) · [Multikey-3](k2-prefill-final-v1/amd-k2-final-8k-ruler-mk3.json)

The 16K scratch tile can require up to 54.97 GiB for exact leaf output alone at this shape. Earlier 16K RULER runs ran out of memory when growing that workspace between prompts. The compact-address variant now releases the cached allocation before requesting a larger one. The 8K scratch tile halves the largest exact-output reservation while keeping the 16K exact field.

The 16K scratch retest with the workspace growth fix and expandable allocator segments completed all four multikey-2 prompts. It matched the previous prediction texts and scores. [16K scratch multikey-2](k2-prefill-final-v2/amd-k2-final-16k-ruler-mk2.json) · [Launch command](k2-prefill-final-v2/amd-k2-final-16k-ruler-mk2-submission.json)

The [64K attention-only profile](K2_PREFILL_64K_PROFILE.md) has per-kernel attribution for both routing passes and the timings for the AMD wave selector.

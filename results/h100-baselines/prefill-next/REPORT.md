# H100 two-tier entry-16 prefill optimization

Llama-3.1-8B-Instruct, BF16, batch 1. LoD keeps the existing 16,384-token prefill chunks, including the exact first chunk. The 512-query value in the run configurations is an internal routing tile, not a change to the prefill chunk size. The strict entry-16 limit is unchanged.

Timings measure full model prefill and the corresponding cache work in the RetroInfer runtime. Cache initialization and graph capture are timed separately and excluded. The dense reference uses dense prefill and its own cache; its cache work differs from LoD. Each cell is the median of two warmed runs, in seconds; this small sample does not establish small performance differences.

| Method | 32K prefill (s) | 64K prefill (s) |
|---|---:|---:|
| Dense | [1.564](threshold-prefill-next-dense-32768-timing.json) | [4.788](threshold-prefill-next-dense-65536-timing.json) |
| Fixed top-8 LoD | [1.656](threshold-prefill-next-top8-32768-timing.json) | [4.189](threshold-prefill-next-top8-65536-timing.json) |
| Entry-16, previous GPU-to-CPU count readback | [2.627](threshold-prefill-next-control-32768-timing.json) | [9.159](threshold-prefill-next-control-65536-timing.json) |
| Entry-16, reuse routing data | [2.565](threshold-prefill-next-reuse-entry-16-tile512-32768-timing.json) | — |
| Entry-16, device-only padded buffers | [3.149](threshold-prefill-next-nosync-entry-16-tile512-32768-timing.json) | — |
| Entry-16, fused packing into padded buffers | [3.175](threshold-prefill-next-pack-entry-16-tile512-32768-timing.json) | — |
| Entry-16, paired-query KV reuse | [2.892](threshold-prefill-next-pair64-entry-16-tile512-32768-timing.json) | — |
| Entry-16, device-only packed buffers | [2.588](threshold-prefill-next-packed-entry-16-tile512-32768-timing.json) | — |
| Entry-16, packed buffers and routing-data reuse | [2.565](threshold-prefill-next-packedreuse-entry-16-tile512-32768-timing.json) | [9.281](threshold-prefill-next-packedreuse-65536-timing.json) |
| Entry-16, packed and streamed region expansion | [3.026](threshold-prefill-next-packedstream-entry-16-tile512-32768-timing.json) | — |
| Entry-16, packed and wider selection | [2.501](threshold-prefill-next-packed128-entry-16-tile512-32768-timing.json) | [7.984](threshold-prefill-next-packed128-65536-timing.json) |

The packed path computes offsets on the GPU and stores selected regions and entries contiguously. It removes the per-tile host readback of route counts; it still reserves buffers from the worst-case work bound. The other device-only variants were tested separately and are not implicitly combined with it. These paths are experimental, controlled by environment variables, and do not change serving defaults.

For the fastest tested option, use `LOD_H100_PREFILL_PACKED=1`, `LOD_H100_PREFILL_REUSE_MEAN=1`, and `LOD_H100_PREFILL_SELECT_N=128` together with the fused entry-16 prefill settings. The packed path implies no host count readback.

A separate warmed diagnostic explicitly synchronizes the GPU between stages. Across all 32 layers it measured route logits 0.139s, select and coarse 0.529s, compact 0.082s, expand indices 0.230s, exact leaves 0.508s, merge 0.098s ([profile](threshold-prefill-next-packedreuse-32768-profile.json)). These are diagnostic stage timings, not a replacement for the end-to-end timings above.

## Correctness and retrieval

On three 32K RULER examples, checks compared 49,152 query-head frontiers with the reference: 0 selection mismatches. The largest attention output absolute difference was 0.01562. Reused routing logits were checked bit-for-bit against the [original routing computation](threshold-prefill-next-packedreuse-entry-16-ruler32k-verify.json).

| 32K multikey retrieval | Multikey 2 | Multikey 3 |
|---|---:|---:|
| [Previous entry-16](../ruler-32k-panel/threshold-prefill-default-entry-16-cap16-ruler32k-audit8.json) | 100.0% | 100.0% |
| [Routing-data reuse](threshold-prefill-next-reuse-entry-16-ruler32k-audit8.json) | 100.0% | 100.0% |
| [Packed and reuse](threshold-prefill-next-packedreuse-entry-16-ruler32k-audit8.json) | 100.0% | 100.0% |
| [Packed, reuse, wider selection](threshold-prefill-next-packed128-entry-16-ruler32k-audit8.json) | 100.0% | 100.0% |

Each score covers eight matched examples per task, not the full RULER panel. The strict bound concerns logically opened remote KV entries per query head; it does not establish a subquadratic bound for complete prefill or amortized cache updates. Run commands, source hashes, and logs are in the provenance directory alongside these results.

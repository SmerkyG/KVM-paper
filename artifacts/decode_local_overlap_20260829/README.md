# Batch-one coarse routing and local-attention overlap

This panel investigates two independent ways to reduce the batch-one decode
critical path on Qwen3.5-0.8B at 64K:

1. precompute centroid means at state-update boundaries instead of dividing
   every K component by `count` on every decoded token; and
2. overlap query-dependent coarse routing with independent local/sink
   attention.

All end-to-end rows use real ProLong text, batch one, a 65,280-token prompt,
256 requested decode tokens, seven repetitions, prefix caching, 4K chunked
prefill, two-tier unrestricted top-eight routing, and the fixed-list page-size-
one AITER path. The device audit verifies the requested custom kernels.

| decode path | median ms/step | result |
| --- | ---: | --- |
| established current top-eight control | 1.984 | reference |
| immediate serial control before cached means | 2.041 | noisy matched control |
| **serial, update-cached BF16 centroid means** | **1.962** | retained |
| vector local/sink overlap with separate merge | 2.276 | rejected |
| page-one overlap, combined reduction, unshortened remote range | 2.152 | rejected |
| page-one overlap, combined reduction, shortened remote range | 2.075 | rejected |
| staged coarse/local/top-eight experiment | 2.445 | rejected |

The cached means are exact with respect to the old route calculation. State K
continues to store sums for update semantics; the existing persistent
page-size-one arena receives BF16-rounded means only at install/catch-up
boundaries. The hot fixed-mask route scorer reads that arena. A synthetic route
panel found bitwise-equal candidate scores and indices, while the real 64K
median is 1.1% below the stable 1.984-ms control. The larger apparent change
against the immediate 2.041-ms run includes run-to-run variance.

The first overlap result also exposed avoidable traversal: the remote fixed
scan masked the local prefix but still partitioned the complete address range.
Starting it after the aligned 512-position local reservation improves 2.152 to
2.075 ms/step. The corrected local scan addresses fixed-position sinks without
building a compact table, and its 8K cached-versus-cold probe matches all 64
generated tokens. At 64K, however, hiding eight local 64-token tiles per layer
does not repay the extra page-one launch, CU contention, and two-field final
reduction: it remains 5.8% slower than the 1.962-ms serial cached-mean path.
The implementation is opt-in through
`VLLM_LOD_DECODE_GQA_OVERLAP_LOCAL_SINK=1` and remains disabled by default.

## Coarse-score floor and FP8 storage

The isolated centroid-major score kernel is flat at about 13.4 us for 1,024,
2,048, and 4,096 centroids. It reaches 16.92 us at 8,192, 22.12 us at 16,384,
and 30.16 us at 32,768. Thus the 4,096-centroid 64K geometry is a launch/
occupancy floor, not a key-bandwidth regime.

The complete fixed-prepare route kernel was then run with sum K, cached BF16
mean K, and cached unscaled E4M3FNUZ mean K. FP8 is loaded from half-sized
storage and promoted to BF16 before the existing MFMA; this isolates the
storage-bandwidth benefit without also quantizing Q.

| centroids | sum K | BF16 mean K | FP8 mean K | FP8 block-candidate agreement |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 41.50 us | **40.96 us** | 41.00 us | 99.17% |
| 8,192 | 46.98 us | 44.88 us | **44.60 us** | 99.14% |
| 16,384 | 58.72 us | 55.54 us | **54.96 us** | 99.32% |

FP8 saves nothing at the target 4K-centroid geometry. Even at 16K centroids it
saves only 0.58 us over cached BF16 means and changes 0.68% of the block-level
candidate slots before global top-eight. Per-block scales would improve route
fidelity but add metadata traffic. FP8 coarse storage is therefore not a
batch-one latency default; it remains potentially useful for memory or much
larger centroid tables.

Precomputing `log(count)` was also rejected. An FP32 cached bias preserves
scores exactly but was no faster than evaluating `log` in the score kernel;
the extra load slightly lost at all three sizes. FP16 cached bias was at most
0.18 us faster and retained only about 98% of block candidate slots.

Artifacts in this directory contain the end-to-end results and the longer
centroid-major score measurements. `scripts/benchmark_precomputed_mean_route.py`
reproduces the cached-mean, FP8-storage, and cached-bias microbenchmarks.

# INT4 LoD cache VRAM

This experiment measures the persistent attention-cache memory used by a
conventional full-attention BF16 K/V cache and by production three-tier INT4
LoD. The reported reduction always uses the **full-attention cache** as its
denominator and includes LoD's centroids, exact local field, page summaries,
quantization scales, and indexing metadata.

## Result

At 128K context, three-tier INT4 reduces persistent attention-cache memory by
**56.5%** for Qwen3.8-27B-FP8 and **49.2%** for K2-Horizon-32B-FP8. Their
equal-model mean is **52.9%**. Using unique tensor payload rather than allocator
deltas gives 56.8% and 50.0%, respectively, or a 53.4% equal-model mean.

### Qwen3.8-27B-FP8

Qwen has 16 full-attention layers with four 256-wide K/V heads. Its 48
recurrent layers are intentionally excluded because their recurrent state is
the same in full-attention and LoD runs.

| Context | Full BF16 K/V | Three-tier INT4 | Reduction |
|---:|---:|---:|---:|
| 8K | 0.500 GiB | 0.440 GiB | 11.9% |
| 16K | 1.000 GiB | 0.725 GiB | 27.5% |
| 32K | 2.000 GiB | 1.150 GiB | 42.5% |
| 64K | 4.000 GiB | 1.969 GiB | 50.8% |
| 128K | 8.000 GiB | 3.478 GiB | 56.5% |

At 128K, the INT4 unique tensor payload is 3.455 GiB, a 56.8% reduction from
the 8.000 GiB full-attention payload.

### K2-Horizon-32B-FP8

K2 has 64 full-attention layers with eight 128-wide K/V heads.

| Context | Full BF16 K/V | Three-tier INT4 | Reduction |
|---:|---:|---:|---:|
| 8K | 2.000 GiB | 2.349 GiB | -17.4% |
| 16K | 4.000 GiB | 3.630 GiB | 9.2% |
| 32K | 8.000 GiB | 5.650 GiB | 29.4% |
| 64K | 16.000 GiB | 9.551 GiB | 40.3% |
| 128K | 32.000 GiB | 16.250 GiB | 49.2% |

At 128K, the INT4 unique tensor payload is 16.002 GiB, a 50.0% reduction from
the 32.000 GiB full-attention payload.

The 128K K2 INT4 unique payload breaks down as follows:

| Component | GiB | Share |
|---|---:|---:|
| Packed 4-bit leaf K/V | 8.016 | 50.1% |
| Quantized page summaries | 1.721 | 10.8% |
| Page-summary scales | 0.860 | 5.4% |
| Leaf scales | 0.860 | 5.4% |
| Centroid sum K/V | 1.438 | 9.0% |
| Attention-ready centroid/local arena | 1.625 | 10.2% |
| Inline page directory | 0.719 | 4.5% |
| Page indices | 0.430 | 2.7% |
| Overflow hash | 0.250 | 1.6% |
| Other persistent tensors | 0.083 | 0.5% |

Thus 8.016 GiB is the packed leaf archive itself. The largest reducible
overheads are the duplicated sum/attention-ready centroid representations,
page summaries and scales, and the 128-entry inline page directory.

For batch size eight, unique payload scales linearly with the fixed request
rows: 64.000 to 27.641 GiB for Qwen and 256.000 to 128.014 GiB for K2.

## Measurement method

This is a capacity-allocation test, so it uses no text corpus and does not
populate the cache with model activations. Persistent tensor sizes depend only
on model geometry, cache organization, request count, and configured context
capacity. ProLong is used separately for the speed and quality experiments.

The measurements were made on an AMD Instinct MI325X on 2026-09-25 using the
release checkout with the standardized 256-token decode update interval. For
each model and context length, the experiment:

1. computed the conventional full-attention BF16 K/V payload as
   `tokens * layers * 2(K,V) * kv_heads * head_dim * 2 bytes`;
2. constructed a three-tier INT4 `VLLMLayerLODPool` with
   `VLLMLODSettings.production()`, one request row, BF16 model dtype, and the
   production model geometry;
3. synchronized the GPU and recorded the change in
   `torch.cuda.memory_allocated()` caused by one layer's persistent pool;
4. independently enumerated unique tensor storages reachable from the pool,
   deduplicating aliased views by storage pointer; and
5. multiplied the per-layer quantities by the number of full-attention layers.

GiB means 2^30 bytes. Because every tested context is an exact multiple of the
vLLM cache block size, the formula is also the exact full-cache tensor payload;
there is no partially filled final block. The INT4 allocator-delta tables are
the primary result. Unique-storage totals cross-check allocator rounding and do
not double-count aliased views.

The LoD measurement includes persistent state centroids, the exact local K/V
tail, the protected sink, leaf residual storage, page summaries, INT4 scales,
page ownership tables, and overflow-hash metadata. Both sides exclude model
weights, temporary attention workspaces, and model state unrelated to the
full-attention cache. The vLLM native cache associated with converted LoD
layers is metadata-only and does not hold a second native K/V tensor.

At 128K, both modes used leaf capacity 131,328, page capacity 14,096, and state
capacity 5,888. Both models and both three-tier modes used local capacity 768:
the 512-token K2 INT4 exception has been removed from the release profile.

## Interpreting the percentage

INT4 does not reduce the whole cache by the theoretical 75% because only the
page-local leaf residual payload is packed to four bits. Centroids, the exact
local field, the sink, page summaries or scales, and indexing metadata retain
wider representations. These fixed and lower-order costs are a larger fraction
at short contexts: K2 INT4 is 17.4% larger than full attention at 8K, crosses
below it by 16K, and saves 49.2% at 128K. Qwen saves 11.9% at 8K and 56.5% at
128K.

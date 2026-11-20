# H100 LoD optimizations on AMD MI325X

Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8 with BF16 KV. The new paths are opt-in. Experiments use `uv` through `cluster-run` and retain the source snapshot and exact submission command.

## Two-tier prefill

[K2 threshold prefill profiles and retimes](K2_PREFILL_PROFILE.md) include the aligned and compact selection experiments and fresh dense controls.

Full-model first-token latency at batch one, with the same tokenized ProLong prompt for each model and context. Each run warms up and then measures three repetitions. Top-eight uses its serving defaults. All threshold arms use c=16, BF16 coarse projection, the MMA reducer and packed state max. The initial exact field is 16,384 tokens across the table. Bitmap ranks reconstruct each selected region’s position from the prefix and bits of its 32-region word. The automatic tile sizes depend on padded frontier capacity.

| Model | Prefill | 32K | 64K |
|---|---|---:|---:|
| K2 Horizon | [Native top-eight](h100-port-prefill-v1/amd-h100-prefill-k2-top8.json) | 5.939 s | 13.463 s |
| K2 Horizon | [Region threshold](h100-port-prefill-v1/amd-h100-prefill-k2-threshold.json) | 7.557 s | 20.444 s |
| K2 Horizon | [Bitmap ranks, 4,096 queries](h100-port-prefill-v1/amd-h100-prefill-k2-bitmap.json) | 7.337 s | 19.991 s |
| K2 Horizon | [Bitmap ranks, automatic query tiles](h100-port-prefill-v1/amd-h100-prefill-k2-auto.json) | 7.449 s | 20.280 s |
| Qwen3.8 | [Native top-eight](h100-port-prefill-v1/amd-h100-prefill-qwen-top8.json) | 3.805 s | 7.877 s |
| Qwen3.8 | [Region threshold](h100-port-prefill-v1/amd-h100-prefill-qwen-threshold.json) | 4.003 s | 8.718 s |
| Qwen3.8 | [Bitmap ranks, 4,096 queries](h100-port-prefill-v1/amd-h100-prefill-qwen-bitmap.json) | 3.959 s | 8.623 s |
| Qwen3.8 | [Bitmap ranks, automatic query tiles](h100-port-prefill-v1/amd-h100-prefill-qwen-auto.json) | 3.987 s | 8.660 s |

### Same-query prefill replay

Each entry is the median over three real layers from the final 64K chunk. Each replay uses a fixed 4,096-query slice, the same KV, and the same c=16 region work for both threshold paths. Remote attention includes the leaf worker and merge and excludes model and cache update time. The two threshold variants agree bitwise on the replay outputs.

| Model | Rank and bitmap scratch, before → after | Route and coarse, before → after | Remote attention, before → after |
|---|---:|---:|---:|
| [K2 Horizon](h100-port-checks-v1/amd-h100-replay-k2-bitmap.json) | 1884 → 166 MiB | 11.138 → 11.032 ms | 17.005 → 16.927 ms |
| [Qwen3.8](h100-port-checks-v1/amd-h100-replay-qwen-bitmap.json) | 707 → 62 MiB | 4.188 → 4.139 ms | 8.885 → 8.817 ms |

## Three-tier decode

32K, batch eight, BF16 KV, top-eight prefill, and 1,024 decode steps per request. The numbers are full-model ms per batch step from two measured repetitions after warmup with serving cache updates. Both models have fresh top-eight controls. Each threshold arm keeps c=16 and fixed top-two pages. The tuned K2 arm combines cached region means, the short eight-page scan, 128-region tiles and 24 page workers per query head.

| Model | Decode | Median | Repetitions |
|---|---|---:|---|
| K2 Horizon | [Native top-eight](h100-port-decode-v1/amd-h100-b8-k2-top8.json) | 50.117 ms | 50.168 / 50.066 ms |
| K2 Horizon | [Region threshold](h100-port-decode-v1/amd-h100-b8-k2-threshold.json) | 55.967 ms | 55.958 / 55.976 ms |
| K2 Horizon | [Cached region means](h100-port-decode-v1/amd-h100-b8-k2-means.json) | 54.839 ms | 54.750 / 54.929 ms |
| K2 Horizon | [Specialized fixed-two-page kernel](h100-port-decode-v1/amd-h100-b8-k2-fixed.json) | 52.867 ms | 52.854 / 52.881 ms |
| K2 Horizon | [Specialized pages and cached means](h100-port-decode-v1/amd-h100-b8-k2-fixed-means.json) | 51.202 ms | 51.213 / 51.191 ms |
| K2 Horizon | [Specialized pages, cached means and tuned launch](h100-port-decode-v1/amd-h100-b8-k2-tuned.json) | 51.323 ms | 51.305 / 51.341 ms |
| Qwen3.8 | [Native top-eight](h100-port-decode-v1/amd-h100-b8-qwen-top8.json) | 34.909 ms | 34.918 / 34.900 ms |
| Qwen3.8 | [Region threshold](h100-port-decode-v1/amd-h100-b8-qwen-threshold.json) | 35.882 ms | 35.887 / 35.876 ms |
| Qwen3.8 | [Specialized fixed-two-page kernel](h100-port-decode-v1/amd-h100-b8-qwen-fixed.json) | 34.570 ms | 34.576 / 34.564 ms |

Some batch-eight generations differ between repetitions. The run JSONs record first mismatch positions.

## Correctness and retrieval

The bitmap paths pass all independent PyTorch frontier checks with both routing rules, including the strict entry-work bound. The fixed-page specialization passes independent PyTorch checks including ties and partial pages. Cached means and the combined fixed-page path pass batch-eight replay against the original single-request kernels, with distinct cache rows and current-token KV writes.

| Model | Change | Multikey-2 before → after | Multikey-3 before → after |
|---|---|---:|---:|
| K2 Horizon | Bitmap ranks and auto tiling ([reference](h100-port-checks-v1/amd-h100-retrieval-k2-ref.json), [candidate](h100-port-checks-v1/amd-h100-retrieval-k2-port.json)) | 100% → 100% | 100% → 100% |
| Qwen3.8 | Bitmap ranks and auto tiling ([reference](h100-port-checks-v1/amd-h100-retrieval-qwen-ref.json), [candidate](h100-port-checks-v1/amd-h100-retrieval-qwen-port.json)) | 100% → 100% | 100% → 100% |
| K2 Horizon | Tuned fixed pages and cached means ([reference](h100-port-checks-v1/amd-h100-fixed-k2-retrieval-ref.json), [candidate](h100-port-checks-v1/amd-h100-fixed-k2-retrieval-tuned.json)) | 75% → 75% | 100% → 100% |

Retrieval checks use the same four frozen examples per task at nominal 64K, batch one.

- [Region bitmap check](h100-port-prefill-v1/amd-h100-bitmap-frontier.json) and [per-entry bitmap check](h100-port-prefill-v1/amd-h100-bitmap-entry-frontier.json).
- [Fixed-page PyTorch check](h100-port-decode-v1/amd-h100-port-fixed-frontier.json) and [fixed-page graph check](h100-port-decode-v1/amd-h100-port-fixed-batch.json).
- [Cached-means graph check](h100-port-checks-v1/amd-h100-means-vs-generic.json) and [combined graph check](h100-port-checks-v1/amd-h100-fixed-means-vs-generic.json).

### Experimental settings

| Optimization | Environment |
|---|---|
| Bitmap ranks | `LOD_AMD_PREFILL_BITMAP_RANKS=1` |
| Automatic region query tile | `LOD_AMD_PREFILL_QUERY_TILE=auto` |
| Specialized fixed-two-page BF16 refinement | `LOD_AMD_THRESHOLD_FIXED_SCAN=8` |
| Reuse K2 region means | `LOD_AMD_THRESHOLD_CACHED_MEANS=1` |
| Tuned K2 launch | `LOD_AMD_THRESHOLD_TILE=128 LOD_AMD_THRESHOLD_PAGE_WORKERS=24` |

Each result has a sibling `-submission.json` with its full invocation and cluster log path.

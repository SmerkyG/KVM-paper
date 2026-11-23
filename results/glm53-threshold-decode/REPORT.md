# GLM-5.3-Flash per-head threshold decode on MI325X

The [Qwen/K2 optimization follow-up](gqa-ports/REPORT.md) holds GLM’s c=16 routing fixed and measures deferred normalization, combined closed and exact attention, and leaf-tile scheduling at B=1 and B=8.

The [Qwen and K2 decode-kernel transfer study](gqa-transfer/REPORT.md) compares worker geometry, region routing, partial-output layouts and full-model serving on the GLM run-based decoder.

The [flat ID packing comparison](flat-pack/REPORT.md) measures how much of the descriptor advantage remains when selection and flat packing share a kernel.

## Trained full-model serving at 64K

One TP4 group of four MI325X GPUs served the pinned GLM-5.3-Flash checkpoint at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`. All 45 language layers ran, including the 11 MLA layers. The two LoD arms used the same FP8 model weights, BF16 two-tier KV, top-eight prefill, 16K exact prefix, projected prefill leaves, Gluon KDA prefill and captured decode. The threshold arm changes decode only. As a separate reference we also timed native GLM, with its learned indexer and 2,048-token budget, on the same prompts and GPU group. The common weight broker loaded the complete native checkpoint first and supplied the four TP shards to each client. [Serving harness](serving.py) · [Paired results](summary.json)

| Batch | Native learned sparse (ms/step) | Existing top-eight (ms/step) | Threshold c=16 (ms/step) | Latency reduction | Top-eight repeats (ms) | Threshold repeats (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12.897 | 8.446 | 8.168 | +3.29% | 8.444 / 8.447 | 8.164 / 8.171 |
| 8 | 17.813 | 13.751 | 13.543 | +1.51% | 13.756 / 13.746 | 13.542 / 13.544 |

At c=16, full-model decode latency fell by 3.3% at batch one and 1.5% at batch eight relative to the existing top-eight LoD path. Threshold was faster in both measured repetitions at each batch. Native GLM took 1.58× and 1.32× as long per step, respectively. Only 11 of the 45 language layers use MLA, so the isolated attention gain translates into a smaller change in total latency.

Each arm has a full warmup followed by two measured generations on a fixed ProLong continuation. Each request had a 65,536-token prompt and generated 1,026 tokens, yielding 1,025 timed decode steps. Batch eight used eight distinct prompts and kept all eight live over the entire measured interval. The per-step metric includes the whole model, query absorption, output projection, and, in both LoD arms, four 256-token cache updates per request on every MLA layer and rank. Initial prefill cache construction is charged to prefill. All eight prompt and continuation records match the archived GLM comparison. Worker audits verify dispatch. [B1 native](serving-b1-native.json) · [B8 native](serving-b8-native.json) · [B1 top-eight](serving-b1-top8.json) · [B1 threshold](serving-b1-c16.json) · [B8 top-eight](serving-b8-top8.json) · [B8 threshold](serving-b8-c16.json)

## Captured attention layer timings

| Batch | Top-eight (µs) | c=8 (µs) | c=16 (µs) | c=32 (µs) |
|---|---:|---:|---:|---:|
| 1 | 69.32 | 46.82 | 43.81 | 42.64 |
| 8 | 119.26 | 132.12 | 108.52 | 90.83 |

Entries are equal-weight means of the median GPU-graph times on two different trained requests and layers. Each replay uses one TP4 rank with 16 absorbed query heads, one shared 512-channel KV latent, and the serving pool built from the real prefix. The sources are the final prefill queries from layer 3 of a 65,536-token prompt and layer 19 of a 64,000-token prompt, replayed as decode inputs. At batch eight the replay uses eight neighboring positions from each source with eight separate prefix caches. Each arm resets the valid local length and includes region scoring, exact and closed attention, local and sink, the current KV append, and reduction. Timing uses 11 interleaved rounds of 20 graph replays. Model projections, other layers and periodic cache rebuilding are included in the full-model table above. [Layer 3 B1](confirm-d1-b1.json) · [Layer 19 B1](confirm-d2-b1.json) · [Layer 3 B8](confirm-d1-b8.json) · [Layer 19 B8](confirm-d2-b8.json)

| Batch | Existing top-eight exact remote leaves/head | c=16 exact remote leaves/head | c=16 distinct remote leaves read per 16-head group |
|---|---:|---:|---:|
| 1 | 8,800 | 7,162 | 19,086 |
| 8 | 8,180 | 6,682 | 17,266 |

The existing two-tier GLM decoder takes the union of top-eight regions across each 16-head tile. Every head in that tile attends to the entire union. Threshold decode compares each head’s population-corrected region mass to c divided by its nonempty region count. It preserves the 1,024-leaf eligibility cap. The kernel loads each selected shared latent for the tile and uses a bit mask to give it exact attention only on the heads that selected its region. The remaining heads retain the corresponding closed summary. Consequently these serving arms represent different routing policies. Task accuracy remains unmeasured on these forced continuations.

## Implementation and validation

The first kernel computes native GLM centroid scores and tile log normalizers while appending the current latent. The second uses the saved scores to select regions, project closed summaries and pack short runs pointing into the existing region-to-leaf directory. The third evaluates the exact remote, local and sink fields with per-head membership. The fourth merges FP32 partials, returns the latent output and log normalizer, and advances the physical cache row. The timed configuration uses 32-entry descriptors and 256 exact-attention splits for batch one, and 8-entry descriptors and 64 splits for batch eight. Each KV latent is stored once and shared across the heads. Enable this path with `LOD_GLM_DECODE_THRESHOLD=16` on the `lod-k3` branch. [Kernel](../../lod_attention/kernels/glm_threshold_decode.py) · [vLLM dispatch](../../integrations/vllm_lod/vllm_lod_plugin/pool.py)

All 65 GPU regression checks passed. The independent FP32 replay checks every selected physical leaf and head membership, output, log normalizer and KV append. On the four c=16 replays relative latent-output error was below 0.18%. GPU integration tests also exercise graph replay with changed queries and cache slots, 32 heads, empty remote fields, no opens, and centroids above the population cap. [Reference and timing harness](../../benchmarks/glm53_threshold_decode.py) · [Regression tests](../../tests/test_glm_threshold_decode.py)

All experiments used uv and cluster-run. The compressed patch restores the preliminary per-leaf packing and backend sweeps on top of the final source. [Cluster commands](jobs.json) · [Tuning variants](tuning-variants.patch.gz)

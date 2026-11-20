# ClusterKV on H100

Llama-3.1-8B-Instruct, 32,768-token ProLong prefix, native Llama 3.1 RoPE, and the same RetroInfer model runtime. Decode uses 256 untimed steps and 1,024 timed teacher-forced steps. Each row is the median of two repetitions after one warmup. The CUDA graph table uses device-resident planning for ClusterKV. The earlier eager table uses ClusterKV’s released planner, which reads GPU metadata on the CPU.

ClusterKV uses its released cosine K-means, retrieval, and paged-attention kernels with 400 initial centroids per KV head, a 1,024-token budget, two dense layers, 16 sink tokens, and a 320-token recent window. The 1,024-token budget covers 1,023 retrieved entries and the current token. It adds eight centroids per window update, following the released benchmark default. The paper describes four. Full KV stays on the GPU using the release’s resident path. The adapter supplies native Llama 3.1 RoPE. The CUDA port corrects centroid counts, sorts all live centroids, fixes the block-scan carry, and maps each query head’s selected indices to its shared KV head. The adapter gives the current token its own final page and indexes a completed window after its last exact attention step. [Paper](https://arxiv.org/html/2412.03213v1#S5), [author source](https://github.com/sjtu-zhao-lab/ClusterKV/tree/c7380819588771e8c5dadacdbd33942d7c53adea).

Prefill includes index construction and any deferred cache completion. Model loading and initial cache allocation are outside the timing windows. The decode window contains four ClusterKV updates and one RetroInfer index rebuild. Peak allocated memory is the PyTorch allocator peak, including model weights. The CUDA extension’s own allocations are outside this memory counter.

## Optimized LoD at batch one and eight

All four variants use the c=16 region cutoff, BF16 model weights, and top-eight prefill with cache completion. Each row times full-model CUDA graph decode, including 128 layer cache catch-ups per repetition at B=1 and 1,024 at B=8. A decode step emits one token per request. B=8 uses independent cache state for eight copies of the same prompt and prefills them sequentially. The B=1 runs explicitly enable the optimized kernels and settings. The adaptive-page variant is enabled by selecting the 0.5 page threshold.

| LoD decode | Remote KV | B=1 prefill (s) | B=1 decode (ms/step) | B=8 prefill (s) | B=8 decode (ms/step) | B=8 output tokens/s |
|---|---|---:|---:|---:|---:|---:|
| Two-tier | BF16 | 1.657 | [7.420](clusterkv-llama32k-b1-lod-optimized-two-tier-bf16-time.json) | 13.324 | [9.719](clusterkv-llama32k-b8-lod-optimized-two-tier-bf16-time.json) | 823.1 |
| Three-tier, fixed top-two pages | BF16 | 1.744 | [7.836](clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-bf16-time.json) | 14.106 | [10.318](clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-bf16-time.json) | 775.3 |
| Three-tier, fixed top-two pages | INT4 | 1.919 | [7.645](clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-int4-time.json) | 15.356 | [10.867](clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-int4-time.json) | 736.2 |
| Three-tier, adaptive page c=0.5 | BF16 | 1.746 | [8.044](clusterkv-llama32k-b1-lod-optimized-adaptive-three-tier-bf16-time.json) | 14.132 | [10.821](clusterkv-llama32k-b8-lod-page05-v2-residual-final-time.json) | 739.3 |

Batch-one retimes of the earlier kernel settings use the same prompt, graph, precision, and cache update schedule.

| LoD decode | Remote KV | Earlier B=1 settings (ms/step) | B=8 tuning at B=1 (ms/step) | Latency change |
|---|---|---:|---:|---:|
| Two-tier | BF16 | [7.255](clusterkv-llama32k-b1-lod-optimized-two-tier-bf16-control.json) | 7.420 | +2.3% |
| Three-tier, fixed top-two pages | BF16 | [7.723](clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-bf16-control.json) | 7.836 | +1.5% |
| Three-tier, fixed top-two pages | INT4 | [7.575](clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-int4-control.json) | 7.645 | +0.9% |
| Three-tier, adaptive page c=0.5 | BF16 | [8.502](clusterkv-llama32k-b1-lod-optimized-adaptive-three-tier-bf16-control.json) | 8.044 | -5.4% |

The earlier settings give the lowest B=1 latency for two-tier and both fixed-page variants. The optimized adaptive-page kernel gives the lowest B=1 latency for that variant.

## Batch-one eager comparison

| Decoder | Model and KV precision | Prefill (s) | Decode (ms/token) | Peak allocated (GiB) |
|---|---|---:|---:|---:|
| [ClusterKV](clusterkv-llama32k-eager-v10-retro-clusterkv.json) | FP16 | 2.193 | 10.013 | 20.78 |
| [Dense, ClusterKV attention kernels](clusterkv-llama32k-eager-v10-retro-clusterkv-full.json) | FP16 | 1.662 | 8.903 | 20.63 |
| [Dense, RetroInfer attention kernels](clusterkv-llama32k-eager-v1-retro-full.json) | FP16 | 1.597 | 9.502 | 20.63 |
| [Dense, RetroInfer attention kernels](clusterkv-llama32k-eager-bf16-v1-retro-full.json) | BF16 | 1.567 | 9.661 | 20.63 |
| [RetroInfer GPU-only](clusterkv-llama32k-eager-bf16-v1-retro.json) | BF16 | 1.754 | 11.682 | 19.55 |
| [LoD two-tier, c=16 threshold decode](clusterkv-llama32k-eager-bf16-v2-retro-lod-bf16.json) | BF16 | 1.660 | 11.785 | 24.14 |

ClusterKV decode takes 5.4% more time per token than the FP16 RetroInfer dense control and 12.5% more than its own FP16 dense path.

The FP16 dense controls match ClusterKV’s precision. The BF16 dense control matches LoD’s precision. LoD uses materialized threshold decode with top-eight prefill.

The two FP16 dense implementations chose identical next tokens at every step in all three timing repetitions. ClusterKV’s two measured repetitions differed on 1 of 1,281 predicted tokens with the same seed and teacher-forced input. Centroid accumulation uses FP16 atomics.

## Matched CUDA graph retime

Both BF16 paths use the same prompt, H100, and timing window as the eager table. Each graph captures the full model decode step. LoD’s cache catch-up runs outside the graph and inside the timing window.

| Method | Eager (ms/token) | CUDA graph (ms/token) | Latency reduction |
|---|---:|---:|---:|
| Dense BF16 | [9.535](clusterkv-llama32k-ab-eager-retro-full.json) | [8.710](clusterkv-llama32k-ab-graph-retro-full.json) | 8.7% |
| LoD two-tier BF16, c=16 threshold decode | [12.686](clusterkv-llama32k-ab-eager-retro-lod-bf16.json) | [7.288](clusterkv-llama32k-ab-graph-retro-lod-bf16.json) | 42.5% |

Both LoD paths include 128 layer cache updates per repetition and choose identical next tokens on the teacher-forced sequence. LoD’s eager path submits five attention kernels per layer per token, including proxy scoring, selection, index packing, exact attention and the merge. Whole-model graph replay avoids issuing those launches individually from Python. The batch-one eager comparison uses eager execution for every arm.

## Whole-model CUDA graph comparison

Same 32K prompt, H100, batch size of one, and 1,024-token timed window as the eager table. The new ClusterKV path computes split offsets and the current KV page on the GPU using the position, prompt length, and retrieval budget. It calls the same centroid selector and split-KV attention kernels with persistent metadata. It uses fixed split counts for dense and sparse layers. The full-model graph covers KV append, retrieval, attention, and next-token selection. Window clustering runs between replays and is included in timed decode. Each ClusterKV repetition contains four updates. Initial workspace setup and initial graph capture are outside timed decode.

| Method | Precision | Execution | Prefill (s) | Decode (ms/token) |
|---|---|---|---:|---:|
| [ClusterKV](clusterkv-llama32k-graph-v1.json) | FP16 | CUDA graph | 2.204 | 9.125 |
| [ClusterKV, device planning in eager mode](clusterkv-llama32k-device-eager-v1.json) | FP16 | Eager | 2.205 | 9.904 |
| [Dense, ClusterKV attention kernels](clusterkv-llama32k-graph-full-v1.json) | FP16 | CUDA graph | 1.665 | 8.408 |
| [Dense, RetroInfer attention kernels](clusterkv-llama32k-graph-dense-fp16-v1.json) | FP16 | CUDA graph | 1.601 | 8.672 |
| [Dense, RetroInfer attention kernels](clusterkv-llama32k-graph-dense-bf16-v1.json) | BF16 | CUDA graph | 1.574 | 8.721 |
| [RetroInfer GPU-only sparse decode](clusterkv-llama32k-graph-retro-sparse-v1.json) | BF16 | CUDA graph | 1.755 | 9.862 |
| [LoD two-tier, c=16 threshold decode](clusterkv-llama32k-graph-lod-bf16-v1.json) | BF16 | CUDA graph | 1.664 | 7.284 |
| [LoD three-tier, c=16 threshold decode](clusterkv-llama32k-b1-lod-three-c16-bf16-v1.json) | BF16 | CUDA graph | 1.746 | 7.721 |
| [LoD three-tier, c=16 threshold decode](clusterkv-llama32k-b1-lod-three-c16-int4-v1.json) | BF16 weights / INT4 KV | CUDA graph | 1.918 | 7.575 |
| [LoD three-tier, region c=16, adaptive page c=0.5](clusterkv-llama32k-b1-lod-optimized-adaptive-three-tier-bf16-time.json) | BF16 | CUDA graph | 1.746 | 8.044 |

RetroInfer uses dense prefill and its GPU-only sparse decoder with 1.8% exact cluster retrieval and 23.2% cluster estimation. Every timed repetition includes one index rebuild and the ensuing graph recapture. The [same-cache check](clusterkv-llama32k-graph-retro-sparse-check-v1.json) matches eager logits at 27 sampled steps around and between updates.

The first measured graph repetition differs from the earlier released-planner repetition on 3 of 1,281 predicted tokens. The split geometry differs, and prefill centroid sums use FP16 atomics.

[Same-cache verification](clusterkv-llama32k-graph-check-v2.json) found identical graph and eager logits at 25 sampled steps, including both sides of two window boundaries. The [synthetic fixture](provenance/1794547874683084330-clusterkv-h100-graph-fixture-v1/output.log) checks the released planner, exact attention and retrieval indices, and passes CUDA memcheck. The RULER table below uses the earlier released-planner run.

The adaptive BF16 row uses the optimized 0.5 page threshold. It includes 128 layer cache catch-ups per timed repetition. The other LoD rows use the original batch-one settings.

Fixed-page three-tier uses the same c=16 region cutoff as two-tier and opens up to two pages per selected region. In the targeted BF16 RULER evaluation at 32K, fixed two-page refinement scored 90.63% on multikey-3 versus 100.00% for two-tier (32 matched prompts). [Three-tier retrieval diagnosis](../three-tier-repair/REPORT.md#why-two-pages-fail). The full RULER table below reports two-tier LoD.

## Matched 64K batch-one timing

Same H100, FP16 checkpoint, 65,536-token prefix, and teacher-forced sequence for both arms. Prefill includes index construction. After 256 untimed decode steps, a window of 1,024 steps includes four ClusterKV updates. Both methods use whole-model graphs and report medians of two repetitions after one warmup.

| Method | Prefill (s) | Decode (ms/token) |
|---|---:|---:|
| [Dense FP16](llama64k-b1-clusterkv-matched-dense-fp16.json) | 4.835 | 10.072 |
| [ClusterKV FP16](llama64k-b1-clusterkv-matched-clusterkv.json) | 5.914 | 9.387 |

The [same-cache check](llama64k-b1-clusterkv-matched-graph-check.json) matched logits exactly at all 37 sampled steps, including cache-update boundaries. [Dense FP16 command](provenance/1794794328890682766-llama64k-b1-clusterkv-matched-dense-fp16/run.json) · [ClusterKV command](provenance/1794794327676770288-llama64k-b1-clusterkv-matched-clusterkv/run.json) · [Graph-check command](provenance/1794794306574489329-llama64k-b1-clusterkv-matched-graph-check/run.json).

The full 64K RULER panel uses ClusterKV's released eager planner with the same budgets. Its 13-task mean is 58.125%, versus 85.99% for FP16 dense. The graph planner reproduced four sampled retrieval misses. [64K RULER results](../ruler-64k-panel/RULER_64K.md#fp16-precision-matched-comparison).

The [batch-one context sweep](../llama-context-scaling/REPORT.md#matched-full-model-decode) compares the fixed 1,024-token budget with ⌊16√N⌋ at 16,384 through 128,000 prompt tokens, retaining 400 initial centroids in both arms.

## Batch-eight CUDA graph comparison

Eight copies of the same 32K prompt, with independent KV state per request. The H100 runs BF16 weights, sequential per-request prefill, and a whole-model CUDA graph for decode. After 256 untimed steps, 1,024 teacher-forced batch steps are timed. Each row gives the median of two repetitions after one warmup. Decode includes periodic cache maintenance and RetroInfer graph recapture. Each batch step emits eight output tokens.

| Method | KV precision | Prefill (s) | Decode (ms/batch step) | Output tokens/s |
|---|---|---:|---:|---:|
| [Dense](clusterkv-llama32k-b8-dense-v1.json) | BF16 | 12.639 | 18.463 | 433.3 |
| [RetroInfer GPU-only sparse decode](clusterkv-llama32k-b8-retro-v1.json) | BF16 | 14.115 | 11.804 | 677.8 |
| [LoD two-tier, c=16 threshold decode](clusterkv-llama32k-b8-lod-optimized-two-tier-bf16-time.json) | BF16 | 13.324 | 9.719 | 823.1 |
| [LoD three-tier, c=16 threshold decode](clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-bf16-time.json) | BF16 | 14.106 | 10.318 | 775.3 |
| [LoD three-tier, c=16 threshold decode](clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-int4-time.json) | INT4 | 15.356 | 10.867 | 736.2 |
| [LoD three-tier, region c=16, adaptive page c=0.5](clusterkv-llama32k-b8-lod-page05-v2-residual-final-time.json) | BF16 | 14.132 | 10.821 | 739.3 |
| [Multipole, 512-token budget](clusterkv-llama32k-b8-multipole-v2.json) | BF16 | 18.179 | 22.892 | 349.5 |

RetroInfer takes 36.1% less time per batch step than dense at batch eight.

Adaptive page routing uses RMS-normalized queries and opens pages whose proxy mass exceeds half the mean page mass within the selected region. On the earlier batch-one BF16 RULER run, this configuration scored 100% on multikey-3 and 86.73% across all 13 tasks. [Three-tier RULER results](../three-tier-repair/REPORT.md#full-ruler-panel).

Multipole uses its fixed-position routing query and Triton 3.1. The other rows use Triton 3.7.1.

ClusterKV supports one request per decode step in this implementation. The RULER scores below use the batch-one configurations. Our QUEST and Squeezed adapters in this harness invoke each request’s attention serially.

Two-tier uses materialized c=16 region threshold decode at both batch sizes. All LoD rows use fixed top-eight prefill. The [shared-centroid report](../llama-shared-index/REPORT.md#batch-eight-whole-model-decode) uses optimized threshold prefill, a 30,720-token prefix and 1,536 timed steps.

Each RetroInfer repetition contains one index rebuild and graph recapture.

Each LoD repetition includes 1,024 layer-by-request cache updates.

Each Multipole repetition includes 8 online cluster updates.

Same-cache graph and eager checks matched the next token for every request. [RetroInfer GPU-only sparse decode BF16](clusterkv-llama32k-b8-retro-check-v1.json), [LoD two-tier, c=16 threshold decode BF16](clusterkv-llama32k-b8-lod-two-port-optimized-check.json), [LoD three-tier, c=16 threshold decode BF16](clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-check-fixed-route128-v3.json), [LoD three-tier, c=16 threshold decode INT4](clusterkv-llama32k-b8-lod-int4-opt-scales-w24-check.json), [LoD three-tier, region c=16, adaptive page c=0.5 BF16](clusterkv-llama32k-b8-lod-page05-v2-residual-logit-check.json), [Multipole, 512-token budget BF16](clusterkv-llama32k-b8-multipole-check-v1.json).

Multipole’s maximum sampled absolute logit difference was 2.438.

### Optimized BF16 two-tier threshold decode

The optimized kernel computes the global cutoff once per query head and packs indices for four selected regions at a time. Routing uses 128 regions per tile.

The port reduces batch-step latency by 2.9% at region c=16. Both runs include 1,024 layer-by-request cache updates per timed repetition and give identical predicted tokens across the timed sequence.

| Configuration | Prefill (s) | Decode (ms/batch step) | Output tokens/s |
|---|---:|---:|---:|
| [Reference](clusterkv-llama32k-b8-lod-two-port-control-time.json) | 13.343 | 10.009 | 799.3 |
| [Optimized](clusterkv-llama32k-b8-lod-two-port-optimized-time.json) | 13.340 | 9.715 | 823.5 |

The graph matches eager logits at 17 sampled steps. The independent-request test checks distinct cache rows, the 1,024-entry routing eligibility boundary, and KV appends. The maximum absolute attention-output difference was 0.00781. [Model graph check](clusterkv-llama32k-b8-lod-two-port-optimized-check.json). [Independent-request check](clusterkv-llama32k-b8-lod-two-port-optimized-v2-fixture.json).

The [timing repeat](clusterkv-llama32k-b8-lod-two-port-optimized-v2-time.json) measures 9.716 ms per batch step.

On the 32 matching batch-one RULER multikey-3 prompts, first-response accuracy was 100.00% with the [reference](clusterkv-llama32k-b8-lod-two-port-control-ruler.json) and 100.00% with the [optimized kernels](clusterkv-llama32k-b8-lod-two-port-optimized-ruler.json). The complete generated token sequences matched on 21 of 32 prompts.

These settings are the H100 RetroInfer harness default for Llama-3.1-8B-Instruct at 32K and batch eight in this mode. The [default timing](clusterkv-llama32k-b8-lod-page05-v2-two-tier-no-overrides-time.json) measures 9.718 ms per batch step and matches the explicit configuration’s token predictions. Set `LOD_H100_TWO_TIER_OPTIMIZED=0` to opt out. [Graph check](clusterkv-llama32k-b8-lod-page05-v2-two-tier-no-overrides-check.json).

### Optimized INT4 fixed two-page refinement

The timed configuration uses 128 regions per routing tile, 24 page workers per head, and 16 pages per scan tile. The region cutoff is computed once per query head, region packing uses a fused prefix scan, and local KV loads are shared within each GQA group. Page scales are applied once to the query and after the weighted value reduction. The H100 RetroInfer harness defaults to these settings for Llama-3.1-8B-Instruct at 32K and batch eight with three-tier INT4, region c=16, and two pages per opened region. Set `LOD_H100_INT4_TUNED_DEFAULTS=0` to restore the previous launch defaults.

INT4 decode takes **10.848 ms per batch step** (737.5 tokens/s), down 9.0% from 11.925 ms. Both timings include 1,024 layer-by-request cache updates per repetition. [Reference](clusterkv-llama32k-b8-lod-int4-opt-control-time.json). [Optimized](clusterkv-llama32k-b8-lod-int4-opt-scales-w24-v2-time.json).

Next-token predictions differ at 0 of 16,384 timed positions and 16 of 4,112 preceding positions in the measured repetitions. The optimized graph matches its eager logits at 17 sampled steps. The maximum absolute output difference against independent reference requests was 0.00006. [Model graph check](clusterkv-llama32k-b8-lod-int4-opt-scales-w24-check.json). [Independent-request check](clusterkv-llama32k-b8-lod-int4-opt-scales-w24-fixture.json).

First-response accuracy on 32 matching RULER multikey-3 prompts was 84.38% with the [reference](clusterkv-llama32k-b8-lod-int4-opt-control-ruler.json) and 84.38% with the [optimized configuration](clusterkv-llama32k-b8-lod-int4-opt-scales-w24-ruler.json). 0 prompts changed first-response score.

The [timing repeat](clusterkv-llama32k-b8-lod-int4-opt-scales-w24-final-time.json) measures 10.850 ms per batch step.

The [default launch](clusterkv-llama32k-b8-lod-int4-opt-default-time.json) measures 10.848 ms per batch step. Its predicted tokens match the explicit optimized run. [Default graph check](clusterkv-llama32k-b8-lod-int4-opt-default-check.json). [Opt-out graph check](clusterkv-llama32k-b8-lod-int4-opt-default-off-no-overrides-check.json).

### Three-tier BF16 and INT4 with matched page tiles

The original BF16 timing used two warps and a 32-page scan tile. The original INT4 timing used one warp and a 16-page scan tile. These reruns use one warp, 16 pages per scan tile and 64 page workers in both arms. Model precision is BF16. The timing window and cache updates match the table above.

| KV precision | Prefill (s) | Decode (ms/batch step) | Output tokens/s |
|---|---:|---:|---:|
| [BF16](clusterkv-llama32k-b8-lod-audit-bf16-w1-s64-p16-time-v1.json) | 14.123 | 11.712 | 683.1 |
| [INT4](clusterkv-llama32k-b8-lod-audit-int4-w1-s64-p16-time-v1.json) | 15.363 | 11.926 | 670.8 |

INT4 takes 1.8% more decode time than BF16 with matched page tiles. BF16’s original timing improves by about 1 ms per batch step.

The [fresh RetroInfer sparse control](clusterkv-llama32k-b8-lod-audit-retro-time-v1.json) takes 11.783 ms per batch step including one rebuild and graph recapture per repetition.

### Optimized BF16 fixed two-page refinement

Region routing uses c=16. Each opened region contributes up to two exact pages and its residual summary. The specialized kernel reads both pages in one tile and uses a shorter summary scan for small regions. The local attention kernel shares KV loads across query heads in each GQA group. Region packing uses one linear prefix scan and copy per head. The timed configuration uses 128-region tiles and 16 page workers per query head.

The [full-model timing](clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-time-fixed-route128-v3.json) is **10.316 ms per batch step** (775.5 tokens/s), including 1,024 layer-by-request cache updates per repetition. The [model graph check](clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-check-fixed-route128-v3.json) matches eager logits at 17 steps. The [independent-request check](clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-fixture-fixed-route128-v3.json) compares distinct caches and queries against the existing page decoder (maximum absolute attention-output difference 0.00195).

Predicted tokens match the BF16 control with the same page tile and warp count across all requests and repetitions.

The retime reduces fixed-two-page latency by 19.0% from the original BF16 run. Two-tier takes 9.715 ms per batch step with the same timing window.

The [independent timing repeat](clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-time-fixed-route128-repeat-v4.json) measures 10.305 ms per batch step.

These settings are the H100 RetroInfer harness default for Llama-3.1-8B-Instruct at 32K and batch eight in this mode. The [default timing](clusterkv-llama32k-b8-lod-page05-v2-fixed-bf16-default-time.json) measures 10.303 ms per batch step and matches the explicit configuration’s token predictions. Set `LOD_H100_BF16_TUNED_DEFAULTS=0` to opt out. [Graph check](clusterkv-llama32k-b8-lod-page05-v2-fixed-bf16-default-check.json).

On 32 matched batch-one RULER multikey-3 prompts, the optimized fixed-two-page kernel with 64-region tiles and 32 page workers scored 84.38% versus 90.62% for the original fixed-two-page kernel. [RULER rerun](clusterkv-llama32k-b8-lod-audit-bf16-w1-s32-fixed-ruler-multikey-v1.json).

| Current batch-one kernel | Multikey-3 |
|---|---:|
| [Reference kernel, original launch settings](clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-original-geometry-v2.json) | 90.62% |
| [Reference kernel, 16-page tile and one warp](clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-control-v2.json) | 87.50% |
| [Specialized page kernel, 16-page tile and one warp](clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-specialized-only-v2.json) | 87.50% |
| [Optimized kernel, selected batch-eight launch settings](clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-selected-full-v4.json) | 90.62% |

### Adaptive BF16 page threshold optimization

The kernel scans summaries in a tile of 8 pages when the region fits, and uses a full scan otherwise. It reuses the scanned page IDs and counts and processes selected entries 2 pages at a time. The measured launch uses 24 workers per query head, 2 warp(s) per worker, and 128 regions per routing tile. The 0.5 cutoff applies to all candidate pages in the region. The residual logit reuses the page query-key dot products.

The explicit tuning run used `LOD_H100_ADAPTIVE_RESIDUAL_LOGIT=1 LOD_H100_ADAPTIVE_SHORT_SCAN=8 LOD_H100_ADAPTIVE_REUSE_DIRECTORY=1 LOD_H100_ADAPTIVE_EXACT_PAGES=2 LOD_H100_ADAPTIVE_PAGE_FAST=1`.

The [optimized decoder](clusterkv-llama32k-b8-lod-page05-v2-residual-logit-time.json) takes **10.821 ms per batch step** (739.3 tokens/s), down 19.5% from the [retimed control](clusterkv-llama32k-b8-lod-page05-v2-fast-control-time.json) at 13.446 ms. Both timings include 1,024 layer-by-request cache updates per repetition.

Next-token predictions differ at 0 of 16,384 timed positions. Graph and eager logits match at 17 steps. The independent-request fixture compares route membership, exact page and entry counts, and attention output. [Graph check](clusterkv-llama32k-b8-lod-page05-v2-residual-logit-check.json). [Fixture](clusterkv-llama32k-b8-lod-page05-v2-residual-logit-fixture.json).

On 32 matched RULER multikey-3 prompts, first-response accuracy was 100.00% for the [control](clusterkv-llama32k-b8-lod-page05-v2-fast-control-ruler.json) and 100.00% for the [optimized path](clusterkv-llama32k-b8-lod-page05-v2-residual-logit-ruler.json). The full generated sequence matched on 20 prompts.
At the final generated step, mean exact remote entries per layer and query head were 452.7 for the control and 457.0 for the optimized path.

The [256-step trace](clusterkv-llama32k-b8-lod-page05-v2-residual-final-profile-kernels.csv) measures 60.3 µs per layer for adaptive page refinement.

The [timing repeat](clusterkv-llama32k-b8-lod-page05-v2-residual-final-time.json) measures 10.821 ms per batch step.

Three-tier BF16 now uses the optimized kernel by default when `LOD_H100_PAGE_THRESHOLD=0.5` selects the mean-page rule on H100. Set `LOD_H100_ADAPTIVE_TUNED_DEFAULTS=0` to use the earlier defaults. [B=1 graph check](clusterkv-llama32k-b1-lod-adaptive-default-graph-check.json) · [B=8 graph check](clusterkv-llama32k-b8-lod-adaptive-default-graph-check.json).

A separate 256-step CUDA trace shows mean GPU time per layer for each page-refinement kernel call. The trace records the original adaptive path and the matched fixed-page launches.

| Page refinement | Mean GPU time (µs/layer) |
|---|---:|
| [Fixed two pages, BF16](clusterkv-llama32k-b8-lod-audit-bf16-w1-s64-p16-profile-v1-kernels.csv) | 63.6 |
| [Fixed two pages, INT4](clusterkv-llama32k-b8-lod-audit-int4-w1-s64-p16-profile-v1-kernels.csv) | 68.8 |
| [Adaptive pages, BF16](clusterkv-llama32k-b8-lod-audit-adaptive-bf16-w2-s32-profile-v1-kernels.csv) | 128.4 |
| [Optimized fixed two pages, INT4](clusterkv-llama32k-b8-lod-int4-opt-scales-w24-final-profile-kernels.csv) | 60.6 |

The longer adaptive page kernel accounts for 2.07 ms per batch step across 32 layers in the trace. Region scoring, packing, and local attention have similar kernel times across all three traces.

Three-tier also matched eight independent single-request runs on distinct queries, cache contents, and cache lengths in both precisions. [Independent-request check](clusterkv-llama32k-b8-lod-three-c16-independent-check-v2.json).

Adaptive page routing also matched eight independent single-request runs with distinct queries, cache contents, and cache lengths. [Adaptive-page independent-request check](clusterkv-llama32k-b8-lod-three-c16-page05-fixture-v1.json).

## 32K RULER quality

[ClusterKV generations](clusterkv-llama32k-ruler13-n32-v11-clusterkv.json) use the [frozen 32-example-per-task cohort](../quality-ruler-32k-frozen-13x32.json.gz). The 416 tokenized inputs match the controls. Scores stop at the first EOS/EOT and the overall mean weights all 13 tasks equally. ClusterKV uses FP16 and its native dense prefill. RetroInfer uses dense prefill and GPU-only sparse decode in BF16. LoD uses BF16 and a c=16 region threshold.

The [adaptive three-tier run](clusterkv-llama32k-lod-adaptive-default-ruler13-n32.json) uses sparse top-eight prefill and the optimized H100 default for the 0.5 page threshold. Graph and eager logits matched for the sampled step on each task.

The [dense-prefill LoD run](ruler-llama32k-dense-prefill-materialized-c16-full13-v1.json) uses dense causal attention outputs in every prefill layer and builds the usual LoD cache from those layers’ activations. Its first token matches dense BF16 on all 416 prompts. The adapter also computes sparse prefill while constructing the cache and discards its attention outputs. The comparison measures generation quality.

The [adaptive three-tier dense-prefill run](clusterkv-llama32k-lod-adaptive-dense-prefill-ruler13-n32.json) uses the same c=16 region and c=0.5 page thresholds as the sparse-prefill run. Its first token matches dense BF16 on all 416 prompts. This quality run used a separate H100 of the same model and driver version.

The [adaptive three-tier threshold-prefill panel](../adaptive-threshold-prefill/RULER_32K.md) uses c=16 region routing during prefill and adaptive c=0.5 page refinement during decode. It scores **88.09%** on the same 416 prompts.

| Task | ClusterKV FP16 | Dense FP16 | Dense BF16 | RetroInfer sparse BF16 | LoD BF16, sparse prefill | LoD BF16, dense prefill | LoD adaptive three-tier, sparse prefill | LoD adaptive three-tier, dense prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 90.62% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 93.75% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 96.88% | 100.00% | 96.88% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 96.88% | 100.00% | 96.88% |
| niah_multikey_2 | 96.88% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_3 | 81.25% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 88.28% | 96.09% | 96.88% | 98.44% | 99.22% | 96.09% | 99.22% | 96.88% |
| niah_multiquery | 95.31% | 99.22% | 99.22% | 98.44% | 97.66% | 97.66% | 97.66% | 95.31% |
| vt | 98.12% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| fwe | 76.04% | 90.62% | 90.62% | 86.46% | 88.54% | 88.54% | 88.54% | 88.54% |
| cwe | 20.62% | 37.50% | 37.50% | 36.25% | 29.38% | 29.69% | 27.50% | 28.12% |
| qa_1 | 78.12% | 81.25% | 81.25% | 81.25% | 68.75% | 81.25% | 68.75% | 81.25% |
| qa_2 | 50.00% | 50.00% | 50.00% | 50.00% | 46.88% | 50.00% | 46.88% | 50.00% |
| NIAH mean | 93.26% | 99.41% | 99.51% | 99.61% | 99.61% | 98.44% | 99.61% | 98.24% |
| **13-task mean** | **82.23%** | **88.82%** | **88.88%** | **88.53%** | **86.96%** | **87.46%** | **86.81%** | **87.22%** |

Dense-prefill LoD changes the 13-task mean by +0.50 points versus sparse-prefill LoD (95% paired bootstrap interval -0.97 to +2.00) and by -1.07 points versus RetroInfer (-2.35 to +0.15). The bootstrap resamples matched prompts within each task.

Dense prefill changes the adaptive three-tier mean by +0.41 points (95% paired interval -1.09 to +1.92).

The optimized adaptive kernel changes the 13-task mean by +0.08 points versus the earlier adaptive kernel at the same thresholds (86.73%, 95% paired bootstrap interval -0.29 to +0.54). [Earlier adaptive panel](../three-tier-repair/three-tier-repair-normpage-full13-summary.md).

[First-response and full-output comparisons](clusterkv-llama32k-lod-adaptive-default-ruler13-n32-summary.md).

[BF16 quality controls](../three-tier-repair/REPORT.md).

## Reproduction

Runs use `uv` and `cluster-run` on a single H100 80 GB. [ClusterKV source patch](clusterkv-h100.patch).

[Device-planning source patch](clusterkv-h100-graph.patch).

- H100 extension build [command](provenance/1794542190664415926-clusterkv-h100-build-v9/run.json) · [log](provenance/1794542190664415926-clusterkv-h100-build-v9/output.log)
- Selector and attention verification [command](provenance/1794542190742470154-clusterkv-h100-selector-check-v9/run.json) · [log](provenance/1794542190742470154-clusterkv-h100-selector-check-v9/output.log)
- `clusterkv-llama32k-eager-v10-retro-clusterkv.json` [command](provenance/1794542190821968665-clusterkv-llama32k-eager-v10-retro-clusterkv/run.json) · [log](provenance/1794542190821968665-clusterkv-llama32k-eager-v10-retro-clusterkv/output.log)
- `clusterkv-llama32k-eager-v10-retro-clusterkv-full.json` [command](provenance/1794542190902859934-clusterkv-llama32k-eager-v10-retro-clusterkv-full/run.json) · [log](provenance/1794542190902859934-clusterkv-llama32k-eager-v10-retro-clusterkv-full/output.log)
- `clusterkv-llama32k-eager-v1-retro-full.json` [command](provenance/1794539940017913335-clusterkv-llama32k-eager-v1-retro-full/run.json) · [log](provenance/1794539940017913335-clusterkv-llama32k-eager-v1-retro-full/output.log)
- `clusterkv-llama32k-eager-bf16-v1-retro-full.json` [command](provenance/1794540498748509987-clusterkv-llama32k-eager-bf16-v1-retro-full/run.json) · [log](provenance/1794540498748509987-clusterkv-llama32k-eager-bf16-v1-retro-full/output.log)
- `clusterkv-llama32k-eager-bf16-v1-retro.json` [command](provenance/1794540860737184124-clusterkv-llama32k-eager-bf16-v1-retro/run.json) · [log](provenance/1794540860737184124-clusterkv-llama32k-eager-bf16-v1-retro/output.log)
- `clusterkv-llama32k-eager-bf16-v2-retro-lod-bf16.json` [command](provenance/1794541451993144890-clusterkv-llama32k-eager-bf16-v2-retro-lod-bf16/run.json) · [log](provenance/1794541451993144890-clusterkv-llama32k-eager-bf16-v2-retro-lod-bf16/output.log)
- `clusterkv-llama32k-ab-eager-retro-full.json` [command](provenance/1794547178447646719-clusterkv-llama32k-ab-eager-retro-full/run.json) · [log](provenance/1794547178447646719-clusterkv-llama32k-ab-eager-retro-full/output.log)
- `clusterkv-llama32k-ab-graph-retro-full.json` [command](provenance/1794547178370940600-clusterkv-llama32k-ab-graph-retro-full/run.json) · [log](provenance/1794547178370940600-clusterkv-llama32k-ab-graph-retro-full/output.log)
- `clusterkv-llama32k-ab-eager-retro-lod-bf16.json` [command](provenance/1794547178217049743-clusterkv-llama32k-ab-eager-retro-lod-bf16/run.json) · [log](provenance/1794547178217049743-clusterkv-llama32k-ab-eager-retro-lod-bf16/output.log)
- `clusterkv-llama32k-ab-graph-retro-lod-bf16.json` [command](provenance/1794547178293312735-clusterkv-llama32k-ab-graph-retro-lod-bf16/run.json) · [log](provenance/1794547178293312735-clusterkv-llama32k-ab-graph-retro-lod-bf16/output.log)
- `clusterkv-llama32k-graph-v1.json` [command](provenance/1794547905152707218-clusterkv-llama32k-graph-v1/run.json) · [log](provenance/1794547905152707218-clusterkv-llama32k-graph-v1/output.log)
- `clusterkv-llama32k-device-eager-v1.json` [command](provenance/1794547905234757314-clusterkv-llama32k-device-eager-v1/run.json) · [log](provenance/1794547905234757314-clusterkv-llama32k-device-eager-v1/output.log)
- `clusterkv-llama32k-graph-full-v1.json` [command](provenance/1794547905317380479-clusterkv-llama32k-graph-full-v1/run.json) · [log](provenance/1794547905317380479-clusterkv-llama32k-graph-full-v1/output.log)
- `clusterkv-llama32k-graph-dense-fp16-v1.json` [command](provenance/1794547905399213091-clusterkv-llama32k-graph-dense-fp16-v1/run.json) · [log](provenance/1794547905399213091-clusterkv-llama32k-graph-dense-fp16-v1/output.log)
- `clusterkv-llama32k-graph-dense-bf16-v1.json` [command](provenance/1794547905481185335-clusterkv-llama32k-graph-dense-bf16-v1/run.json) · [log](provenance/1794547905481185335-clusterkv-llama32k-graph-dense-bf16-v1/output.log)
- `clusterkv-llama32k-graph-lod-bf16-v1.json` [command](provenance/1794547905563285922-clusterkv-llama32k-graph-lod-bf16-v1/run.json) · [log](provenance/1794547905563285922-clusterkv-llama32k-graph-lod-bf16-v1/output.log)
- `clusterkv-llama32k-graph-check-v2.json` [command](provenance/1794547905067886752-clusterkv-llama32k-graph-check-v2/run.json) · [log](provenance/1794547905067886752-clusterkv-llama32k-graph-check-v2/output.log)
- `clusterkv-llama32k-graph-retro-sparse-v1.json` [command](provenance/1794548927002309018-clusterkv-llama32k-graph-retro-sparse-v1/run.json) · [log](provenance/1794548927002309018-clusterkv-llama32k-graph-retro-sparse-v1/output.log)
- `clusterkv-llama32k-graph-retro-sparse-check-v1.json` [command](provenance/1794548926922648094-clusterkv-llama32k-graph-retro-sparse-check-v1/run.json) · [log](provenance/1794548926922648094-clusterkv-llama32k-graph-retro-sparse-check-v1/output.log)
- `clusterkv-llama32k-b1-lod-three-c16-bf16-v1.json` [command](provenance/1794611139355519002-clusterkv-llama32k-b1-lod-three-c16-bf16-v1/run.json) · [log](provenance/1794611139355519002-clusterkv-llama32k-b1-lod-three-c16-bf16-v1/output.log)
- `clusterkv-llama32k-b1-lod-three-c16-int4-v1.json` [command](provenance/1794611140497978039-clusterkv-llama32k-b1-lod-three-c16-int4-v1/run.json) · [log](provenance/1794611140497978039-clusterkv-llama32k-b1-lod-three-c16-int4-v1/output.log)
- `clusterkv-llama32k-b1-lod-optimized-two-tier-bf16-time.json` [command](provenance/1794624406140105186-clusterkv-llama32k-b1-lod-optimized-two-tier-bf16-time/run.json) · [log](provenance/1794624406140105186-clusterkv-llama32k-b1-lod-optimized-two-tier-bf16-time/output.log)
- `clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-bf16-time.json` [command](provenance/1794624406213509628-clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-bf16-time/run.json) · [log](provenance/1794624406213509628-clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-bf16-time/output.log)
- `clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-int4-time.json` [command](provenance/1794624406288865761-clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-int4-time/run.json) · [log](provenance/1794624406288865761-clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-int4-time/output.log)
- `clusterkv-llama32k-b1-lod-optimized-adaptive-three-tier-bf16-time.json` [command](provenance/1794624406363361591-clusterkv-llama32k-b1-lod-optimized-adaptive-three-tier-bf16-time/run.json) · [log](provenance/1794624406363361591-clusterkv-llama32k-b1-lod-optimized-adaptive-three-tier-bf16-time/output.log)
- `clusterkv-llama32k-b8-lod-optimized-two-tier-bf16-time.json` [command](provenance/1794624734804139920-clusterkv-llama32k-b8-lod-optimized-two-tier-bf16-time/run.json) · [log](provenance/1794624734804139920-clusterkv-llama32k-b8-lod-optimized-two-tier-bf16-time/output.log)
- `clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-bf16-time.json` [command](provenance/1794624734876898517-clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-bf16-time/run.json) · [log](provenance/1794624734876898517-clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-bf16-time/output.log)
- `clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-int4-time.json` [command](provenance/1794624734950637060-clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-int4-time/run.json) · [log](provenance/1794624734950637060-clusterkv-llama32k-b8-lod-optimized-fixed-three-tier-int4-time/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-residual-final-time.json` [command](provenance/1794622354600549361-clusterkv-llama32k-b8-lod-page05-v2-residual-final-time/run.json) · [log](provenance/1794622354600549361-clusterkv-llama32k-b8-lod-page05-v2-residual-final-time/output.log)
- `clusterkv-llama32k-b1-lod-optimized-two-tier-bf16-control.json` [command](provenance/1794624568993638272-clusterkv-llama32k-b1-lod-optimized-two-tier-bf16-control/run.json) · [log](provenance/1794624568993638272-clusterkv-llama32k-b1-lod-optimized-two-tier-bf16-control/output.log)
- `clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-bf16-control.json` [command](provenance/1794624569066442608-clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-bf16-control/run.json) · [log](provenance/1794624569066442608-clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-bf16-control/output.log)
- `clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-int4-control.json` [command](provenance/1794624569141754100-clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-int4-control/run.json) · [log](provenance/1794624569141754100-clusterkv-llama32k-b1-lod-optimized-fixed-three-tier-int4-control/output.log)
- `clusterkv-llama32k-b1-lod-optimized-adaptive-three-tier-bf16-control.json` [command](provenance/1794624587012660979-clusterkv-llama32k-b1-lod-optimized-adaptive-three-tier-bf16-control/run.json) · [log](provenance/1794624587012660979-clusterkv-llama32k-b1-lod-optimized-adaptive-three-tier-bf16-control/output.log)
- `clusterkv-llama32k-b8-dense-v1.json` [command](provenance/1794549654276923207-clusterkv-llama32k-b8-dense-v1/run.json) · [log](provenance/1794549654276923207-clusterkv-llama32k-b8-dense-v1/output.log)
- `clusterkv-llama32k-b8-retro-v1.json` [command](provenance/1794549654349614447-clusterkv-llama32k-b8-retro-v1/run.json) · [log](provenance/1794549654349614447-clusterkv-llama32k-b8-retro-v1/output.log)
- `clusterkv-llama32k-b8-lod-threshold-v2.json` [command](provenance/1794610660592495181-clusterkv-llama32k-b8-lod-threshold-v2/run.json) · [log](provenance/1794610660592495181-clusterkv-llama32k-b8-lod-threshold-v2/output.log)
- `clusterkv-llama32k-b8-lod-three-c16-bf16-v1.json` [command](provenance/1794611141627806723-clusterkv-llama32k-b8-lod-three-c16-bf16-v1/run.json) · [log](provenance/1794611141627806723-clusterkv-llama32k-b8-lod-three-c16-bf16-v1/output.log)
- `clusterkv-llama32k-b8-lod-three-c16-int4-v1.json` [command](provenance/1794611142736095101-clusterkv-llama32k-b8-lod-three-c16-int4-v1/run.json) · [log](provenance/1794611142736095101-clusterkv-llama32k-b8-lod-three-c16-int4-v1/output.log)
- `clusterkv-llama32k-b8-lod-three-c16-page05-timing-v1.json` [command](provenance/1794612160728013143-clusterkv-llama32k-b8-lod-three-c16-page05-timing-v1/run.json) · [log](provenance/1794612160728013143-clusterkv-llama32k-b8-lod-three-c16-page05-timing-v1/output.log)
- `clusterkv-llama32k-b8-multipole-v2.json` [command](provenance/1794549830068482393-clusterkv-llama32k-b8-multipole-v2/run.json) · [log](provenance/1794549830068482393-clusterkv-llama32k-b8-multipole-v2/output.log)
- `clusterkv-llama32k-b8-retro-check-v1.json` [command](provenance/1794549654423873553-clusterkv-llama32k-b8-retro-check-v1/run.json) · [log](provenance/1794549654423873553-clusterkv-llama32k-b8-retro-check-v1/output.log)
- `clusterkv-llama32k-b8-lod-threshold-check-v1.json` [command](provenance/1794610593288751418-clusterkv-llama32k-b8-lod-threshold-check-v1/run.json) · [log](provenance/1794610593288751418-clusterkv-llama32k-b8-lod-threshold-check-v1/output.log)
- `clusterkv-llama32k-b8-lod-three-c16-bf16-check-v1.json` [command](provenance/1794611128992367928-clusterkv-llama32k-b8-lod-three-c16-bf16-check-v1/run.json) · [log](provenance/1794611128992367928-clusterkv-llama32k-b8-lod-three-c16-bf16-check-v1/output.log)
- `clusterkv-llama32k-b8-lod-three-c16-int4-check-v1.json` [command](provenance/1794611130237313377-clusterkv-llama32k-b8-lod-three-c16-int4-check-v1/run.json) · [log](provenance/1794611130237313377-clusterkv-llama32k-b8-lod-three-c16-int4-check-v1/output.log)
- `clusterkv-llama32k-b8-lod-three-c16-page05-model-check-v1.json` [command](provenance/1794612111016271126-clusterkv-llama32k-b8-lod-three-c16-page05-model-check-v1/run.json) · [log](provenance/1794612111016271126-clusterkv-llama32k-b8-lod-three-c16-page05-model-check-v1/output.log)
- `clusterkv-llama32k-b8-multipole-check-v1.json` [command](provenance/1794549719826590363-clusterkv-llama32k-b8-multipole-check-v1/run.json) · [log](provenance/1794549719826590363-clusterkv-llama32k-b8-multipole-check-v1/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-w1-s64-p16-time-v1.json` [command](provenance/1794612616755198899-clusterkv-llama32k-b8-lod-audit-bf16-w1-s64-p16-time-v1/run.json) · [log](provenance/1794612616755198899-clusterkv-llama32k-b8-lod-audit-bf16-w1-s64-p16-time-v1/output.log)
- `clusterkv-llama32k-b8-lod-audit-int4-w1-s64-p16-time-v1.json` [command](provenance/1794612616943048917-clusterkv-llama32k-b8-lod-audit-int4-w1-s64-p16-time-v1/run.json) · [log](provenance/1794612616943048917-clusterkv-llama32k-b8-lod-audit-int4-w1-s64-p16-time-v1/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-time-fixed-route128-v3.json` [command](provenance/1794614737720619635-clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-time-fixed-route128-v3/run.json) · [log](provenance/1794614737720619635-clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-time-fixed-route128-v3/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-check-fixed-route128-v3.json` [command](provenance/1794615109898913306-clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-check-fixed-route128-v3/run.json) · [log](provenance/1794615109898913306-clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-check-fixed-route128-v3/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-fixture-fixed-route128-v3.json` [command](provenance/1794615109942990584-clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-fixture-fixed-route128-v3/run.json) · [log](provenance/1794615109942990584-clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-fixture-fixed-route128-v3/output.log)
- `clusterkv-llama32k-b8-lod-two-port-control-time.json` [command](provenance/1794616829101482488-clusterkv-llama32k-b8-lod-two-port-control-time/run.json) · [log](provenance/1794616829101482488-clusterkv-llama32k-b8-lod-two-port-control-time/output.log)
- `clusterkv-llama32k-b8-lod-two-port-optimized-time.json` [command](provenance/1794617749751028474-clusterkv-llama32k-b8-lod-two-port-optimized-time/run.json) · [log](provenance/1794617749751028474-clusterkv-llama32k-b8-lod-two-port-optimized-time/output.log)
- `clusterkv-llama32k-b8-lod-two-port-optimized-check.json` [command](provenance/1794617748566264430-clusterkv-llama32k-b8-lod-two-port-optimized-check/run.json) · [log](provenance/1794617748566264430-clusterkv-llama32k-b8-lod-two-port-optimized-check/output.log)
- `clusterkv-llama32k-b8-lod-two-port-optimized-v2-fixture.json` [command](provenance/1794617884435330343-clusterkv-llama32k-b8-lod-two-port-optimized-v2-fixture/run.json) · [log](provenance/1794617884435330343-clusterkv-llama32k-b8-lod-two-port-optimized-v2-fixture/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-control-time.json` [command](provenance/1794618181368163738-clusterkv-llama32k-b8-lod-int4-opt-control-time/run.json) · [log](provenance/1794618181368163738-clusterkv-llama32k-b8-lod-int4-opt-control-time/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-scales-w24-v2-time.json` [command](provenance/1794619209368840975-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-v2-time/run.json) · [log](provenance/1794619209368840975-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-v2-time/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-scales-w24-check.json` [command](provenance/1794619328986032594-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-check/run.json) · [log](provenance/1794619328986032594-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-check/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-scales-w24-fixture.json` [command](provenance/1794619327798313081-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-fixture/run.json) · [log](provenance/1794619327798313081-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-fixture/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-control-ruler.json` [command](provenance/1794618335853605459-clusterkv-llama32k-b8-lod-int4-opt-control-ruler/run.json) · [log](provenance/1794618335853605459-clusterkv-llama32k-b8-lod-int4-opt-control-ruler/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-scales-w24-ruler.json` [command](provenance/1794619330260336030-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-ruler/run.json) · [log](provenance/1794619330260336030-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-ruler/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-default-time.json` [command](provenance/1794619940931133028-clusterkv-llama32k-b8-lod-int4-opt-default-time/run.json) · [log](provenance/1794619940931133028-clusterkv-llama32k-b8-lod-int4-opt-default-time/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-default-check.json` [command](provenance/1794619942193242089-clusterkv-llama32k-b8-lod-int4-opt-default-check/run.json) · [log](provenance/1794619942193242089-clusterkv-llama32k-b8-lod-int4-opt-default-check/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-default-off-no-overrides-check.json` [command](provenance/1794620052707414278-clusterkv-llama32k-b8-lod-int4-opt-default-off-no-overrides-check/run.json) · [log](provenance/1794620052707414278-clusterkv-llama32k-b8-lod-int4-opt-default-off-no-overrides-check/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-two-tier-no-overrides-time.json` [command](provenance/1794620538970582743-clusterkv-llama32k-b8-lod-page05-v2-two-tier-no-overrides-time/run.json) · [log](provenance/1794620538970582743-clusterkv-llama32k-b8-lod-page05-v2-two-tier-no-overrides-time/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-two-tier-no-overrides-check.json` [command](provenance/1794620441235430218-clusterkv-llama32k-b8-lod-page05-v2-two-tier-no-overrides-check/run.json) · [log](provenance/1794620441235430218-clusterkv-llama32k-b8-lod-page05-v2-two-tier-no-overrides-check/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-fixed-bf16-default-time.json` [command](provenance/1794620288891512766-clusterkv-llama32k-b8-lod-page05-v2-fixed-bf16-default-time/run.json) · [log](provenance/1794620288891512766-clusterkv-llama32k-b8-lod-page05-v2-fixed-bf16-default-time/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-fixed-bf16-default-check.json` [command](provenance/1794620455827726205-clusterkv-llama32k-b8-lod-page05-v2-fixed-bf16-default-check/run.json) · [log](provenance/1794620455827726205-clusterkv-llama32k-b8-lod-page05-v2-fixed-bf16-default-check/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-fast-control-time.json` [command](provenance/1794620291233636036-clusterkv-llama32k-b8-lod-page05-v2-fast-control-time/run.json) · [log](provenance/1794620291233636036-clusterkv-llama32k-b8-lod-page05-v2-fast-control-time/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-residual-logit-time.json` [command](provenance/1794621848302995389-clusterkv-llama32k-b8-lod-page05-v2-residual-logit-time/run.json) · [log](provenance/1794621848302995389-clusterkv-llama32k-b8-lod-page05-v2-residual-logit-time/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-residual-logit-check.json` [command](provenance/1794622319198569474-clusterkv-llama32k-b8-lod-page05-v2-residual-logit-check/run.json) · [log](provenance/1794622319198569474-clusterkv-llama32k-b8-lod-page05-v2-residual-logit-check/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-residual-logit-fixture.json` [command](provenance/1794622317998832200-clusterkv-llama32k-b8-lod-page05-v2-residual-logit-fixture/run.json) · [log](provenance/1794622317998832200-clusterkv-llama32k-b8-lod-page05-v2-residual-logit-fixture/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-fast-control-ruler.json` [command](provenance/1794620454807370267-clusterkv-llama32k-b8-lod-page05-v2-fast-control-ruler/run.json) · [log](provenance/1794620454807370267-clusterkv-llama32k-b8-lod-page05-v2-fast-control-ruler/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-residual-logit-ruler.json` [command](provenance/1794622320398350532-clusterkv-llama32k-b8-lod-page05-v2-residual-logit-ruler/run.json) · [log](provenance/1794622320398350532-clusterkv-llama32k-b8-lod-page05-v2-residual-logit-ruler/output.log)
- `clusterkv-llama32k-b1-lod-adaptive-default-graph-check.json` [command](provenance/1794625613963528301-clusterkv-llama32k-b1-lod-adaptive-default-graph-check/run.json) · [log](provenance/1794625613963528301-clusterkv-llama32k-b1-lod-adaptive-default-graph-check/output.log)
- `clusterkv-llama32k-b8-lod-adaptive-default-graph-check.json` [command](provenance/1794625614039834237-clusterkv-llama32k-b8-lod-adaptive-default-graph-check/run.json) · [log](provenance/1794625614039834237-clusterkv-llama32k-b8-lod-adaptive-default-graph-check/output.log)
- `clusterkv-llama32k-b8-lod-two-port-control-ruler.json` [command](provenance/1794617455146179578-clusterkv-llama32k-b8-lod-two-port-control-ruler/run.json) · [log](provenance/1794617455146179578-clusterkv-llama32k-b8-lod-two-port-control-ruler/output.log)
- `clusterkv-llama32k-b8-lod-two-port-optimized-ruler.json` [command](provenance/1794617750884286496-clusterkv-llama32k-b8-lod-two-port-optimized-ruler/run.json) · [log](provenance/1794617750884286496-clusterkv-llama32k-b8-lod-two-port-optimized-ruler/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-original-geometry-v2.json` [command](provenance/1794615259709996040-clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-original-geometry-v2/run.json) · [log](provenance/1794615259709996040-clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-original-geometry-v2/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-control-v2.json` [command](provenance/1794615259723964011-clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-control-v2/run.json) · [log](provenance/1794615259723964011-clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-control-v2/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-specialized-only-v2.json` [command](provenance/1794615259659709477-clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-specialized-only-v2/run.json) · [log](provenance/1794615259659709477-clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-specialized-only-v2/output.log)
- `clusterkv-llama32k-b8-lod-audit-adaptive-bf16-w2-s32-time-opt-v1.json` [command](provenance/1794612839803396341-clusterkv-llama32k-b8-lod-audit-adaptive-bf16-w2-s32-time-opt-v1/run.json) · [log](provenance/1794612839803396341-clusterkv-llama32k-b8-lod-audit-adaptive-bf16-w2-s32-time-opt-v1/output.log)
- `clusterkv-llama32k-b8-lod-audit-retro-time-v1.json` [command](provenance/1794612889536729174-clusterkv-llama32k-b8-lod-audit-retro-time-v1/run.json) · [log](provenance/1794612889536729174-clusterkv-llama32k-b8-lod-audit-retro-time-v1/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-time-fixed-route128-repeat-v4.json` [command](provenance/1794615222279967630-clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-time-fixed-route128-repeat-v4/run.json) · [log](provenance/1794615222279967630-clusterkv-llama32k-b8-lod-audit-bf16-w1-s16-p16-time-fixed-route128-repeat-v4/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-w1-s32-fixed-ruler-multikey-v1.json` [command](provenance/1794614852129047993-clusterkv-llama32k-b8-lod-audit-bf16-w1-s32-fixed-ruler-multikey-v1/run.json) · [log](provenance/1794614852129047993-clusterkv-llama32k-b8-lod-audit-bf16-w1-s32-fixed-ruler-multikey-v1/output.log)
- `clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-selected-full-v4.json` [command](provenance/1794615717389815846-clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-selected-full-v4/run.json) · [log](provenance/1794615717389815846-clusterkv-llama32k-b8-lod-audit-bf16-fixed-ruler-selected-full-v4/output.log)
- `clusterkv-llama32k-b8-lod-two-port-optimized-v2-time.json` [command](provenance/1794617954611785545-clusterkv-llama32k-b8-lod-two-port-optimized-v2-time/run.json) · [log](provenance/1794617954611785545-clusterkv-llama32k-b8-lod-two-port-optimized-v2-time/output.log)
- `clusterkv-llama32k-b8-lod-int4-opt-scales-w24-final-time.json` [command](provenance/1794619377700641535-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-final-time/run.json) · [log](provenance/1794619377700641535-clusterkv-llama32k-b8-lod-int4-opt-scales-w24-final-time/output.log)
- `clusterkv-llama32k-b8-lod-page05-v2-residual-final-time.json` [command](provenance/1794622354600549361-clusterkv-llama32k-b8-lod-page05-v2-residual-final-time/run.json) · [log](provenance/1794622354600549361-clusterkv-llama32k-b8-lod-page05-v2-residual-final-time/output.log)
- `clusterkv-llama32k-b8-lod-three-c16-independent-check-v2.json` [command](provenance/1794611410465125913-clusterkv-llama32k-b8-lod-three-c16-independent-check-v2/run.json) · [log](provenance/1794611410465125913-clusterkv-llama32k-b8-lod-three-c16-independent-check-v2/output.log)
- `clusterkv-llama32k-b8-lod-three-c16-page05-fixture-v1.json` [command](provenance/1794612152406586958-clusterkv-llama32k-b8-lod-three-c16-page05-fixture-v1/run.json) · [log](provenance/1794612152406586958-clusterkv-llama32k-b8-lod-three-c16-page05-fixture-v1/output.log)
- `clusterkv-llama32k-ruler13-n32-v11-clusterkv.json` [command](provenance/1794542190982239145-clusterkv-llama32k-ruler13-n32-v11-clusterkv/run.json) · [log](provenance/1794542190982239145-clusterkv-llama32k-ruler13-n32-v11-clusterkv/output.log)
- `clusterkv-llama32k-ruler13-n32-v6-dense-fp16.json` [command](provenance/1794542212700119542-clusterkv-llama32k-ruler13-n32-v6-dense-fp16/run.json) · [log](provenance/1794542212700119542-clusterkv-llama32k-ruler13-n32-v6-dense-fp16/output.log)
- `ruler-llama32k-dense-prefill-materialized-c16-full13-v1.json` [command](provenance/1794550629616366216-ruler-llama32k-dense-prefill-materialized-c16-full13-v1/run.json) · [log](provenance/1794550629616366216-ruler-llama32k-dense-prefill-materialized-c16-full13-v1/output.log)
- `clusterkv-llama32k-lod-adaptive-default-ruler13-n32.json` [command](provenance/1794625533583590349-clusterkv-llama32k-lod-adaptive-default-ruler13-n32/run.json) · [log](provenance/1794625533583590349-clusterkv-llama32k-lod-adaptive-default-ruler13-n32/output.log)
- `clusterkv-llama32k-lod-adaptive-dense-prefill-ruler13-n32.json` [command](provenance/1794786415046858623-clusterkv-llama32k-lod-adaptive-dense-prefill-ruler13-n32/run.json) · [log](provenance/1794786415046858623-clusterkv-llama32k-lod-adaptive-dense-prefill-ruler13-n32/output.log)

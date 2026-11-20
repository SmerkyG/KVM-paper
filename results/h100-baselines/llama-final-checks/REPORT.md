# Llama 3.1 H100 timing with the final threshold recipes

Llama-3.1-8B-Instruct on H100 80 GB HBM3. Every arm uses the same prompt at a given context, teacher-forced tokens, whole-model CUDA graphs, and the median of two repetitions after one warmup.

Prefill includes index construction and LoD cache completion. Initial cache allocation and graph setup are reported in the raw records. The timed decode covers 1,024 batch steps after 256 untimed steps. Each timed RetroInfer repetition includes one index rebuild and graph recapture. Each LoD repetition includes 128 per-layer catch-ups at B=1 and 1,024 layer-by-request catch-ups at B=8.

Both LoD methods use c=16 threshold prefill and decode, a 16,384-token initial exact field, and BF16 KV. Adaptive three-tier uses a c=0.5 page threshold on RMS-normalized queries during decode. The same routing settings are in the [32K quality panel](../adaptive-threshold-prefill/RULER_32K.md) and [64K quality panel](../ruler-64k-panel/RULER_64K.md).

## 32K, batch 1

| Method | Prefill (s) | Decode (ms/batch step) | Output tokens/s |
|---|---:|---:|---:|
| [Dense BF16](llama32k-b1-final-predecode-dense.json) | 1.585 | 8.732 | 114.5 |
| [RetroInfer GPU-only BF16](llama32k-b1-final-predecode-retro.json) | 1.795 | 9.790 | 102.1 |
| [LoD two-tier BF16](llama32k-b1-final-predecode-two-tier.json) | 1.659 | 7.237 | 138.2 |
| [LoD adaptive three-tier BF16](llama32k-b1-final-predecode-adaptive.json) | 1.752 | 8.076 | 123.8 |

## 64K, batch 1

| Method | Prefill (s) | Decode (ms/batch step) | Output tokens/s |
|---|---:|---:|---:|
| [Dense BF16](llama64k-b1-final-predecode-dense.json) | 4.813 | 10.118 | 98.8 |
| [RetroInfer GPU-only BF16](llama64k-b1-final-predecode-retro.json) | 5.331 | 10.678 | 93.7 |
| [LoD two-tier BF16](llama64k-b1-final-predecode-two-tier.json) | 4.527 | 7.481 | 133.7 |
| [LoD adaptive three-tier BF16](llama64k-b1-final-predecode-adaptive.json) | 4.879 | 8.534 | 117.2 |

## 32K, batch 8

| Method | Prefill (s) | Decode (ms/batch step) | Output tokens/s |
|---|---:|---:|---:|
| [Dense BF16](llama32k-b8-final-predecode-dense.json) | 12.747 | 18.516 | 432.1 |
| [RetroInfer GPU-only BF16](llama32k-b8-final-predecode-retro.json) | 14.704 | 11.815 | 677.2 |
| [LoD two-tier BF16](llama32k-b8-final-predecode-two-tier-tile1024.json) | 14.115 | 9.841 | 812.9 |
| [LoD adaptive three-tier BF16](llama32k-b8-final-predecode-adaptive-tile1024.json) | 15.157 | 10.818 | 739.5 |

Batch-eight LoD uses 1,024-query scratch tiles and sequential per-request prefill. Automatic scratch sizing ran out of memory for both LoD modes. The smaller tiles preserve the routing thresholds and the initial exact field.

For the repeated timing prefix, every B=8 request produced the same 1,281 next-token predictions as its B=1 run for both LoD modes and dense, across all three repetitions.

## 64K, batch 4

Four copies of the 65,536-token timing prompt with independent decode caches. Each step emits four tokens. Every arm uses the same teacher-forced inputs, 256 untimed steps and 1,024 timed steps with whole-model CUDA graphs. The table uses the median of two measured repetitions after one warmup.

| Method | Prefill (s) | Decode (ms/batch step) | Output tokens/s | Peak PyTorch (GiB) |
|---|---:|---:|---:|---:|
| [Dense BF16](llama64k-b4-final-predecode-dense.json) | 19.332 | 18.259 | 219.1 | 49.90 |
| [RetroInfer GPU-only BF16](llama64k-b4-final-predecode-retro.json) | 21.816 | 11.962 | 334.4 | 50.27 |
| [LoD two-tier BF16](llama64k-b4-final-predecode-two-tier.json) | 19.583 | 9.201 | 434.7 | 64.13 |
| [LoD adaptive three-tier BF16](llama64k-b4-final-predecode-adaptive.json) | 22.368 | 10.015 | 399.4 | 65.24 |
| [Dense FP16](llama64k-b4-final-predecode-dense-fp16.json) | 19.647 | 18.268 | 219.0 | 49.90 |
| [QUEST FP16](llama64k-b4-final-predecode-quest.json) | 20.387 | 19.754 | 202.5 | 54.25 |
| [Multipole BF16 (512 exact tokens)](../llama-context-scaling/llama-denseprefill-b4-65536-t4096.json) | Dense prefix + setup | 20.530 | 194.8 | 70.83 |
| [Squeezed BF16 (p90)](../llama-context-scaling/llama-denseprefill-b4-65536-t4096.json) | Dense prefix + setup | 17.376 | 230.2 | 49.89 |

The first six arms prefill one request at a time and use 8,192-token MLP chunks. Both LoD modes use 1,024-query scratch tiles, the same 16,384-token initial exact field and 512 layer-by-request catch-ups per timed repetition. RetroInfer includes one rebuild and graph recapture. QUEST uses four serial author attention calls per layer inside the batched model graph with the 2,048-token budget. ClusterKV’s adapter supports batch one.

Multipole and Squeezed use dense causal prefill on the frozen prompt, then populate four independent decode caches from its KV tensors. They use 4,096 centroids per KV head, matching the batch-one width. Index fitting, Squeezed calibration, and cache installation occur before decode timing. The Multipole window includes eight full-batch tail refinements. The reported peaks cover the decode worker. [Dense-prefill reproduction](../llama-context-scaling/DENSE_PREFILL.md#decode-timing).

### 64K batch-four verification

Each same-cache check compares graph and eager results within batch four. The batch-one comparison uses all 15,372 predicted tokens across the three repetitions and four requests, including the token predicted after prefill.

| Method | Graph/eager steps with matching tokens | Max absolute logit difference | Predictions differing from B=1 |
|---|---:|---:|---:|
| [RetroInfer GPU-only BF16](llama64k-b4-final-predecode-retro-graph-check.json) | 27/27 | 0 | 5 |
| [LoD two-tier BF16](llama64k-b4-final-predecode-two-tier-graph-check.json) | 27/27 | 0 | 8 |
| [LoD adaptive three-tier BF16](llama64k-b4-final-predecode-adaptive-graph-check.json) | 27/27 | 0 | 4 |

Dense BF16, dense FP16 and QUEST differ from their batch-one controls on 0, 0 and 0 predictions, respectively. Multipole and Squeezed ran a same-cache graph and eager check on the first decode step of each repetition.

### 64K batch-four reproduction

- Dense BF16. [Results](llama64k-b4-final-predecode-dense.json) · [Command](provenance/1794802557932661125-llama64k-b4-final-predecode-dense/run.json) · [Log](provenance/1794802557932661125-llama64k-b4-final-predecode-dense/output.log).
- RetroInfer GPU-only BF16. [Results](llama64k-b4-final-predecode-retro.json) · [Command](provenance/1794802601713923918-llama64k-b4-final-predecode-retro/run.json) · [Log](provenance/1794802601713923918-llama64k-b4-final-predecode-retro/output.log).
- LoD two-tier BF16. [Results](llama64k-b4-final-predecode-two-tier.json) · [Command](provenance/1794802601704358691-llama64k-b4-final-predecode-two-tier/run.json) · [Log](provenance/1794802601704358691-llama64k-b4-final-predecode-two-tier/output.log).
- LoD adaptive three-tier BF16. [Results](llama64k-b4-final-predecode-adaptive.json) · [Command](provenance/1794802601734979564-llama64k-b4-final-predecode-adaptive/run.json) · [Log](provenance/1794802601734979564-llama64k-b4-final-predecode-adaptive/output.log).
- Dense FP16. [Results](llama64k-b4-final-predecode-dense-fp16.json) · [Command](provenance/1794802601716199393-llama64k-b4-final-predecode-dense-fp16/run.json) · [Log](provenance/1794802601716199393-llama64k-b4-final-predecode-dense-fp16/output.log).
- QUEST FP16. [Results](llama64k-b4-final-predecode-quest.json) · [Command](provenance/1794802601692677815-llama64k-b4-final-predecode-quest/run.json) · [Log](provenance/1794802601692677815-llama64k-b4-final-predecode-quest/output.log).
- Multipole and Squeezed. [Dense-prefill results](../llama-context-scaling/llama-denseprefill-b4-65536-t4096.json) · [Command](../llama-context-scaling/provenance/1794809449450553342-llama-denseprefill-b4-65536-t4096/run.json) · [Log](../llama-context-scaling/provenance/1794809449450553342-llama-denseprefill-b4-65536-t4096/output.log).
- RetroInfer GPU-only BF16 graph check. [Results](llama64k-b4-final-predecode-retro-graph-check.json) · [Command](provenance/1794802815819255801-llama64k-b4-final-predecode-retro-graph-check/run.json) · [Log](provenance/1794802815819255801-llama64k-b4-final-predecode-retro-graph-check/output.log).
- LoD two-tier BF16 graph check. [Results](llama64k-b4-final-predecode-two-tier-graph-check.json) · [Command](provenance/1794802815936176324-llama64k-b4-final-predecode-two-tier-graph-check/run.json) · [Log](provenance/1794802815936176324-llama64k-b4-final-predecode-two-tier-graph-check/output.log).
- LoD adaptive three-tier BF16 graph check. [Results](llama64k-b4-final-predecode-adaptive-graph-check.json) · [Command](provenance/1794802815802247583-llama64k-b4-final-predecode-adaptive-graph-check/run.json) · [Log](provenance/1794802815802247583-llama64k-b4-final-predecode-adaptive-graph-check/output.log).

## Batch-four context scaling

Four independent caches for copies of the frozen timing prompt on one H100 80 GB. Each arm replays the final 1,280 prompt tokens, with 256 untimed steps and 1,024 timed CUDA-graph steps. Each step emits four tokens. The table uses the median of two measured repetitions after one warmup. The 64K rows reuse the runs above. [Batch-four plot](../llama-context-scaling/decode-latency-b4.pdf) · [Plot data](../llama-context-scaling/context-timings-b4.csv).

RetroInfer uses its dense fallback throughout the 8K window, so its sparse curve starts at 16K. At 16K it builds the first index during the first 69 warmup steps. Every measured sparse RetroInfer window includes one rebuild. QUEST runs four serial attention calls per layer inside the batched model graph. The ClusterKV adapter supports batch one.

Both LoD modes, Multipole and Squeezed start the timed window with exactly 1,448, 2,048, 2,896 and 4,096 centroid slots per KV head at 8K, 16K, 32K and 64K. This is the same matching rule as batch one. LoD can add slots during decode. Multipole and Squeezed use dense prefill and keep their shared decoder index width fixed, with a 512-token exact budget and eight timed tail refinements for Multipole and a p90 cutoff calibrated on that index for Squeezed. [Batch-one protocol](../llama-context-scaling/REPORT.md#matched-full-model-decode).

| Method | 8K | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| Dense BF16 | [8.722](llama8k-b4-final-predecode-dense.json) | [9.909](llama16k-b4-final-predecode-dense.json) | [12.755](llama32k-b4-final-predecode-dense.json) | [18.259](llama64k-b4-final-predecode-dense.json) |
| Dense FP16 | [8.786](llama8k-b4-final-predecode-dense-fp16.json) | [9.937](llama16k-b4-final-predecode-dense-fp16.json) | [12.856](llama32k-b4-final-predecode-dense-fp16.json) | [18.268](llama64k-b4-final-predecode-dense-fp16.json) |
| RetroInfer BF16 | — | [9.951](llama16k-b4-final-predecode-retro.json) | [10.527](llama32k-b4-final-predecode-retro.json) | [11.962](llama64k-b4-final-predecode-retro.json) |
| LoD adaptive two-tier BF16 | [7.800](llama8k-b4-final-predecode-two-tier.json) | [8.165](llama16k-b4-final-predecode-two-tier.json) | [8.548](llama32k-b4-final-predecode-two-tier.json) | [9.201](llama64k-b4-final-predecode-two-tier.json) |
| LoD adaptive three-tier BF16 | [8.110](llama8k-b4-final-predecode-adaptive.json) | [8.550](llama16k-b4-final-predecode-adaptive.json) | [9.096](llama32k-b4-final-predecode-adaptive.json) | [10.015](llama64k-b4-final-predecode-adaptive.json) |
| QUEST FP16 | [11.301](llama8k-b4-final-predecode-quest.json) | [12.293](llama16k-b4-final-predecode-quest.json) | [14.928](llama32k-b4-final-predecode-quest.json) | [19.754](llama64k-b4-final-predecode-quest.json) |
| Multipole BF16 (512 exact tokens) | [17.948](../llama-context-scaling/llama-denseprefill-b4-8192-t1448.json) | [17.730](../llama-context-scaling/llama-denseprefill-b4-16384-t2048.json) | [19.302](../llama-context-scaling/llama-denseprefill-b4-32768-t2896.json) | [20.530](../llama-context-scaling/llama-denseprefill-b4-65536-t4096.json) |
| Squeezed BF16 (p90) | [10.432](../llama-context-scaling/llama-denseprefill-b4-8192-t1448.json) | [11.670](../llama-context-scaling/llama-denseprefill-b4-16384-t2048.json) | [13.767](../llama-context-scaling/llama-denseprefill-b4-32768-t2896.json) | [17.376](../llama-context-scaling/llama-denseprefill-b4-65536-t4096.json) |

Decode latency in ms per batch step.

### Fixed-2,900-centroid ablation

The 32K LoD-matched count is 2,896. The 64K rows show the same fixed index against the larger matched width. At each length, both settings use the same GPU, dense prefill, source snapshot and decode window. Multipole keeps its 512-token exact budget and Squeezed calibrates p90 for each index.

[Batch-four ablation plot](../llama-context-scaling/fixed-centroid-ablation-b4.pdf).

| Context | Method | LoD-matched slots | Fixed 2,900 (ms/step) | Matched (ms/step) |
|---:|---|---:|---:|---:|
| 32K | Multipole BF16 (512 exact tokens) | 2,896 | [20.559](../llama-context-scaling/llama-denseprefill-b4-32768-t2900.json) | [19.302](../llama-context-scaling/llama-denseprefill-b4-32768-t2896.json) |
| 32K | Squeezed BF16 (p90) | 2,896 | [13.751](../llama-context-scaling/llama-denseprefill-b4-32768-t2900.json) | [13.767](../llama-context-scaling/llama-denseprefill-b4-32768-t2896.json) |
| 64K | Multipole BF16 (512 exact tokens) | 4,096 | [20.803](../llama-context-scaling/llama-denseprefill-b4-65536-t2900.json) | [20.530](../llama-context-scaling/llama-denseprefill-b4-65536-t4096.json) |
| 64K | Squeezed BF16 (p90) | 4,096 | [16.365](../llama-context-scaling/llama-denseprefill-b4-65536-t2900.json) | [17.376](../llama-context-scaling/llama-denseprefill-b4-65536-t4096.json) |

### Centroid alignment at 32K, batch 4

**Historical LoD-prefill diagnostic.** The main comparison above uses dense prefill.

The two pairs change the fitted index by one slot across a multiple of 64 or 128. Each pair uses one GPU and the same source, prompt, 512-token exact budget for Multipole, and 1,024-step decode window. All four widths use 23 centroid lookup tiles of 128 slots. Each width fits its own index and calibrates Squeezed’s p90 cutoff on that index.

| Centroids per KV head | Alignment | Multipole (ms/step) | Squeezed p90 (ms/step) |
|---:|---|---:|---:|
| [2,880](llama32k-b4-final-predecode-alignment-2880.json) | 64 | 19.325 (18.901, 19.749) | 13.777 (13.771, 13.784) |
| [2,881](llama32k-b4-final-predecode-alignment-2881.json) | Partial tile | 20.372 (19.942, 20.802) | 13.759 (13.763, 13.755) |
| [2,943](llama32k-b4-final-predecode-alignment-2943.json) | Partial tile | 20.971 (20.540, 21.403) | 13.745 (13.791, 13.700) |
| [2,944](llama32k-b4-final-predecode-alignment-2944.json) | 128 | 19.335 (18.907, 19.763) | 13.818 (13.822, 13.815) |

Each cell gives the median followed by the two measured repetitions. These are whole-model times, including the eight timed Multipole tail refinements. The earlier [Qwen lookup-only diagnostic](../qwen-long/qwen-long-matched-centroids-squeezed-kernels-decode32-v1-shared-32768-shared.json) pads the same index with empty slots and measures the lookup separately.

- 2,880 slots. [Results](llama32k-b4-final-predecode-alignment-2880.json) · [Command](provenance/1794804433382277367-llama32k-b4-final-predecode-alignment-2880/run.json) · [Log](provenance/1794804433382277367-llama32k-b4-final-predecode-alignment-2880/output.log).
- 2,881 slots. [Results](llama32k-b4-final-predecode-alignment-2881.json) · [Command](provenance/1794804436969679463-llama32k-b4-final-predecode-alignment-2881/run.json) · [Log](provenance/1794804436969679463-llama32k-b4-final-predecode-alignment-2881/output.log).
- 2,943 slots. [Results](llama32k-b4-final-predecode-alignment-2943.json) · [Command](provenance/1794804433203406509-llama32k-b4-final-predecode-alignment-2943/run.json) · [Log](provenance/1794804433203406509-llama32k-b4-final-predecode-alignment-2943/output.log).
- 2,944 slots. [Results](llama32k-b4-final-predecode-alignment-2944.json) · [Command](provenance/1794804436760953883-llama32k-b4-final-predecode-alignment-2944/run.json) · [Log](provenance/1794804436760953883-llama32k-b4-final-predecode-alignment-2944/output.log).

### Squeezed 8K padding check

**Historical LoD-prefill diagnostic.** This probe and the timing-shift follow-up use KV from a LoD prompt pass. The current performance comparison uses dense prefill.

The earlier 1,472-centroid run had a 10.11 ms/step warmup repetition and two 10.45 ms/step measured repetitions. The 1,448-centroid run stayed near 10.09 ms/step. Both selected about 1,036 prefix tokens per head at the end of the window. [Earlier 1,448 run](llama8k-b4-final-predecode-shared-index.json) · [Earlier 1,472 run](llama8k-b4-final-predecode-alignment-1472.json).

In those same runs, Multipole measured 17.533 ms/step at the exact-match width of 1,448 and 17.072 ms/step at the 64-aligned width of 1,472.

The control below fits the 1,448-slot and 1,472-slot indexes once from the same prefix. The padded arm copies the 1,448-slot index and adds 24 centroids with zero assigned tokens, preserving the p90 cutoff. All three arms time independent batch-four decode caches in one worker, alternating the order across repetitions. The window retains 256 untimed and 1,024 timed steps. The first repetition per arm is warmup. GPU clocks are sampled every 250 ms.

| Index | Slots | First repetition (ms/step) | Measured repeats (ms/step) | Median (ms/step) | Median SM clocks (MHz) |
|---|---:|---:|---:|---:|---:|
| Original | 1,448 | 10.081 | 10.439, 10.435 | 10.437 | 1980, 1980 |
| Same index with empty padding | 1,472 | 10.103 | 10.447, 10.453 | 10.450 | 1980, 1980 |
| Refitted | 1,472 | 10.102 | 10.448, 10.451 | 10.450 | 1980, 1980 |

All three arms shift from approximately 10.1 to 10.45 ms/step after the first round. Median reported clocks are 1,980 MHz for SMs and 2,619 MHz for memory in each repetition. The follow-up below traces the shift to GPU idle time inside the decode graph.

The padded and original arms differed on 0 of 15,372 predicted token IDs. [Results](llama8k-b4-squeezed-padding-probe.json) · [Command](provenance/1794805531321094858-llama8k-b4-squeezed-padding-probe/run.json) · [Log](provenance/1794805531321094858-llama8k-b4-squeezed-padding-probe/output.log).

### Squeezed timing shift

These follow-up runs replay the same Llama 3.1 8B BF16 batch-four 8K window, with 256 untimed and 1,024 timed steps. In workers alternating the three index configurations above, the unchanged 1,448-centroid case slows by 3.3–3.5% on two H100s. CUDA events agree with synchronized host timings. The CPU finished submitting the window about one second before the GPU completed it in both rounds. All sampled clock-event masks were zero, with median SM and memory clocks of 1,980 and 2,619 MHz in both rounds. The selected prefix counts at the end of the window match, and all 5,124 recorded output IDs match across rounds for the 1,448-centroid index.

| Mixed-width worker, 1,448 slots | First round host / CUDA (ms/step) | Second round host / CUDA (ms/step) |
|---|---:|---:|
| [H100 GPU 0](llama8k-b4-squeezed-round-mixed-gpu0.json) | 10.084 / 10.084 | 10.438 / 10.438 |
| [H100 GPU 3](llama8k-b4-squeezed-round-mixed-events.json) | 10.113 / 10.113 | 10.450 / 10.450 |

Nsight Systems traced 16 consecutive replays at the same token positions in each round on GPU 1. Both trace slices have the same kernels, grid and block dimensions, and recorded copy sizes. There are 1,029 captured kernels per step plus six outside the graph. The table measures the union of GPU activity intervals and classifies gaps by whether both adjacent operations belong to the same replay. Absolute timings come from the event-only runs because trace setup perturbs the enclosing wall and CUDA-event windows. [Nsight Systems documentation](https://docs.nvidia.com/nsight-systems/UserGuide/).

| GPU time in traced slice (ms/step) | First round | Second round | Change |
|---|---:|---:|---:|
| Active kernels and memory ops | 10.251 | 10.170 | −0.081 |
| Gaps inside each CUDA graph | 0.084 | 0.454 | +0.370 |
| Gaps outside the graphs | 0.037 | 0.043 | +0.007 |
| Total span | 10.372 | 10.668 | +0.296 |

The increased gaps appear throughout the graph, including between MLP, normalization, rotary-position, and attention kernels. The centroid scorer itself used 1.271 and 1.252 ms/step in the two slices. The GPU spends more time waiting between already queued operations. This is a real decode latency penalty from graph execution overhead. [Mixed-width GPU trace breakdown](llama8k-b4-squeezed-round-mixed-trace-analysis.json) · [Trace command](provenance/1794806070426661772-llama8k-b4-squeezed-round-mixed-trace/run.json).

With only the 1,448-centroid index in the worker, six recaptures stayed within 10.082–10.086 ms/step. Reusing one graph stayed within 10.090–10.094 ms/step over six rounds and a seventh after a 30-second pause. The fixed-width trace control showed 0.087 and 0.088 ms/step of gaps inside the graphs. [Recapture results](llama8k-b4-squeezed-round-events.json) · [Graph reuse results](llama8k-b4-squeezed-round-reuse.json) · [Fixed-width GPU trace breakdown](llama8k-b4-squeezed-round-trace-analysis.json).

Keeping the original 1,448-centroid graph and its KV buffers alive while timing the other widths still shifted that same graph from 10.092 to 10.439 and 10.440 ms/step. Fresh 1,448-centroid graphs in those later rounds took 10.442 and 10.447 ms/step. In a separate mixed-width run, the refitted arm switched back to the fast cadence within a single timed window. The shift can therefore occur without changing a captured graph or its layout. The GPU timelines establish where the extra latency occurs. The specific CUDA runtime trigger remains unresolved. [Retained-graph results](llama8k-b4-squeezed-round-retained.json) · [Command](provenance/1794806341987904001-llama8k-b4-squeezed-round-retained/run.json).

### Batch-four context scaling reproduction

All runs used `uv` and `cluster-run`.

The [dense-prefill report](../llama-context-scaling/DENSE_PREFILL.md#timing-commands) lists the current Multipole and Squeezed timing runs. The original LoD-prefill runs below are historical.

- llama8k-b4-final-predecode-dense. [Results](llama8k-b4-final-predecode-dense.json) · [Command](provenance/1794803366083976821-llama8k-b4-final-predecode-dense/run.json) · [Log](provenance/1794803366083976821-llama8k-b4-final-predecode-dense/output.log).
- llama8k-b4-final-predecode-dense-fp16. [Results](llama8k-b4-final-predecode-dense-fp16.json) · [Command](provenance/1794803366062652629-llama8k-b4-final-predecode-dense-fp16/run.json) · [Log](provenance/1794803366062652629-llama8k-b4-final-predecode-dense-fp16/output.log).
- llama8k-b4-final-predecode-two-tier. [Results](llama8k-b4-final-predecode-two-tier.json) · [Command](provenance/1794803335687356608-llama8k-b4-final-predecode-two-tier/run.json) · [Log](provenance/1794803335687356608-llama8k-b4-final-predecode-two-tier/output.log).
- llama8k-b4-final-predecode-adaptive. [Results](llama8k-b4-final-predecode-adaptive.json) · [Command](provenance/1794803366031849553-llama8k-b4-final-predecode-adaptive/run.json) · [Log](provenance/1794803366031849553-llama8k-b4-final-predecode-adaptive/output.log).
- llama8k-b4-final-predecode-quest. [Results](llama8k-b4-final-predecode-quest.json) · [Command](provenance/1794803366031370831-llama8k-b4-final-predecode-quest/run.json) · [Log](provenance/1794803366031370831-llama8k-b4-final-predecode-quest/output.log).
- llama8k-b4-final-predecode-shared-index. [Results](llama8k-b4-final-predecode-shared-index.json) · [Command](provenance/1794803416204732848-llama8k-b4-final-predecode-shared-index/run.json) · [Log](provenance/1794803416204732848-llama8k-b4-final-predecode-shared-index/output.log).
- llama16k-b4-final-predecode-dense. [Results](llama16k-b4-final-predecode-dense.json) · [Command](provenance/1794803366084256333-llama16k-b4-final-predecode-dense/run.json) · [Log](provenance/1794803366084256333-llama16k-b4-final-predecode-dense/output.log).
- llama16k-b4-final-predecode-dense-fp16. [Results](llama16k-b4-final-predecode-dense-fp16.json) · [Command](provenance/1794803366036941198-llama16k-b4-final-predecode-dense-fp16/run.json) · [Log](provenance/1794803366036941198-llama16k-b4-final-predecode-dense-fp16/output.log).
- llama16k-b4-final-predecode-retro. [Results](llama16k-b4-final-predecode-retro.json) · [Command](provenance/1794803366084910151-llama16k-b4-final-predecode-retro/run.json) · [Log](provenance/1794803366084910151-llama16k-b4-final-predecode-retro/output.log).
- llama16k-b4-final-predecode-two-tier. [Results](llama16k-b4-final-predecode-two-tier.json) · [Command](provenance/1794803335565433099-llama16k-b4-final-predecode-two-tier/run.json) · [Log](provenance/1794803335565433099-llama16k-b4-final-predecode-two-tier/output.log).
- llama16k-b4-final-predecode-adaptive. [Results](llama16k-b4-final-predecode-adaptive.json) · [Command](provenance/1794803366083793198-llama16k-b4-final-predecode-adaptive/run.json) · [Log](provenance/1794803366083793198-llama16k-b4-final-predecode-adaptive/output.log).
- llama16k-b4-final-predecode-quest. [Results](llama16k-b4-final-predecode-quest.json) · [Command](provenance/1794803366081218054-llama16k-b4-final-predecode-quest/run.json) · [Log](provenance/1794803366081218054-llama16k-b4-final-predecode-quest/output.log).
- llama16k-b4-final-predecode-shared-index. [Results](llama16k-b4-final-predecode-shared-index.json) · [Command](provenance/1794803416223728975-llama16k-b4-final-predecode-shared-index/run.json) · [Log](provenance/1794803416223728975-llama16k-b4-final-predecode-shared-index/output.log).
- llama32k-b4-final-predecode-dense. [Results](llama32k-b4-final-predecode-dense.json) · [Command](provenance/1794803366088701350-llama32k-b4-final-predecode-dense/run.json) · [Log](provenance/1794803366088701350-llama32k-b4-final-predecode-dense/output.log).
- llama32k-b4-final-predecode-dense-fp16. [Results](llama32k-b4-final-predecode-dense-fp16.json) · [Command](provenance/1794803366079534142-llama32k-b4-final-predecode-dense-fp16/run.json) · [Log](provenance/1794803366079534142-llama32k-b4-final-predecode-dense-fp16/output.log).
- llama32k-b4-final-predecode-retro. [Results](llama32k-b4-final-predecode-retro.json) · [Command](provenance/1794803366007671366-llama32k-b4-final-predecode-retro/run.json) · [Log](provenance/1794803366007671366-llama32k-b4-final-predecode-retro/output.log).
- llama32k-b4-final-predecode-two-tier. [Results](llama32k-b4-final-predecode-two-tier.json) · [Command](provenance/1794803335597297883-llama32k-b4-final-predecode-two-tier/run.json) · [Log](provenance/1794803335597297883-llama32k-b4-final-predecode-two-tier/output.log).
- llama32k-b4-final-predecode-adaptive. [Results](llama32k-b4-final-predecode-adaptive.json) · [Command](provenance/1794803377160125028-llama32k-b4-final-predecode-adaptive/run.json) · [Log](provenance/1794803377160125028-llama32k-b4-final-predecode-adaptive/output.log).
- llama32k-b4-final-predecode-quest. [Results](llama32k-b4-final-predecode-quest.json) · [Command](provenance/1794803377183281703-llama32k-b4-final-predecode-quest/run.json) · [Log](provenance/1794803377183281703-llama32k-b4-final-predecode-quest/output.log).
- llama32k-b4-final-predecode-shared-index. [Results](llama32k-b4-final-predecode-shared-index.json) · [Command](provenance/1794803416255779397-llama32k-b4-final-predecode-shared-index/run.json) · [Log](provenance/1794803416255779397-llama32k-b4-final-predecode-shared-index/output.log).
- llama64k-b4-final-predecode-dense. [Results](llama64k-b4-final-predecode-dense.json) · [Command](provenance/1794802557932661125-llama64k-b4-final-predecode-dense/run.json) · [Log](provenance/1794802557932661125-llama64k-b4-final-predecode-dense/output.log).
- llama64k-b4-final-predecode-dense-fp16. [Results](llama64k-b4-final-predecode-dense-fp16.json) · [Command](provenance/1794802601716199393-llama64k-b4-final-predecode-dense-fp16/run.json) · [Log](provenance/1794802601716199393-llama64k-b4-final-predecode-dense-fp16/output.log).
- llama64k-b4-final-predecode-retro. [Results](llama64k-b4-final-predecode-retro.json) · [Command](provenance/1794802601713923918-llama64k-b4-final-predecode-retro/run.json) · [Log](provenance/1794802601713923918-llama64k-b4-final-predecode-retro/output.log).
- llama64k-b4-final-predecode-two-tier. [Results](llama64k-b4-final-predecode-two-tier.json) · [Command](provenance/1794802601704358691-llama64k-b4-final-predecode-two-tier/run.json) · [Log](provenance/1794802601704358691-llama64k-b4-final-predecode-two-tier/output.log).
- llama64k-b4-final-predecode-adaptive. [Results](llama64k-b4-final-predecode-adaptive.json) · [Command](provenance/1794802601734979564-llama64k-b4-final-predecode-adaptive/run.json) · [Log](provenance/1794802601734979564-llama64k-b4-final-predecode-adaptive/output.log).
- llama64k-b4-final-predecode-quest. [Results](llama64k-b4-final-predecode-quest.json) · [Command](provenance/1794802601692677815-llama64k-b4-final-predecode-quest/run.json) · [Log](provenance/1794802601692677815-llama64k-b4-final-predecode-quest/output.log).
- llama64k-b4-final-predecode-shared-index. [Results](llama64k-b4-final-predecode-shared-index.json) · [Command](provenance/1794802732770121724-llama64k-b4-final-predecode-shared-index/run.json) · [Log](provenance/1794802732770121724-llama64k-b4-final-predecode-shared-index/output.log).
- llama8k-b4-final-predecode-alignment-1472. [Results](llama8k-b4-final-predecode-alignment-1472.json) · [Command](provenance/1794805136236588739-llama8k-b4-final-predecode-alignment-1472/run.json) · [Log](provenance/1794805136236588739-llama8k-b4-final-predecode-alignment-1472/output.log).
- llama32k-b4-final-predecode-alignment-2880. [Results](llama32k-b4-final-predecode-alignment-2880.json) · [Command](provenance/1794804433382277367-llama32k-b4-final-predecode-alignment-2880/run.json) · [Log](provenance/1794804433382277367-llama32k-b4-final-predecode-alignment-2880/output.log).
- llama32k-b4-final-predecode-shared-index-2900. [Results](llama32k-b4-final-predecode-shared-index-2900.json) · [Command](provenance/1794803366084053517-llama32k-b4-final-predecode-shared-index-2900/run.json) · [Log](provenance/1794803366084053517-llama32k-b4-final-predecode-shared-index-2900/output.log).
- llama64k-b4-final-predecode-shared-index-2900. [Results](llama64k-b4-final-predecode-shared-index-2900.json) · [Command](provenance/1794803173690645270-llama64k-b4-final-predecode-shared-index-2900/run.json) · [Log](provenance/1794803173690645270-llama64k-b4-final-predecode-shared-index-2900/output.log).
- llama65536-b4-two-tier-centroid-count-check. [Results](llama65536-b4-two-tier-centroid-count-check.json) · [Command](provenance/1794806745743877891-llama65536-b4-two-tier-centroid-count-check/run.json).
- llama65536-b4-adaptive-centroid-count-check. [Results](llama65536-b4-adaptive-centroid-count-check.json) · [Command](provenance/1794806745739954466-llama65536-b4-adaptive-centroid-count-check/run.json).
- llama8192-b4-adaptive-centroid-count-check. [Results](llama8192-b4-adaptive-centroid-count-check.json) · [Command](provenance/1794806745574068155-llama8192-b4-adaptive-centroid-count-check/run.json).
- llama16384-b4-adaptive-centroid-count-check. [Results](llama16384-b4-adaptive-centroid-count-check.json) · [Command](provenance/1794806769027470056-llama16384-b4-adaptive-centroid-count-check/run.json).
- llama32768-b4-adaptive-centroid-count-check. [Results](llama32768-b4-adaptive-centroid-count-check.json) · [Command](provenance/1794806745617407400-llama32768-b4-adaptive-centroid-count-check/run.json).

## Related Llama comparisons

- [Batch-one context scaling](../llama-context-scaling/REPORT.md) covers 16K through 128,000 tokens, including the shared-index comparison.
- [ClusterKV](../clusterkv/REPORT.md) uses FP16 and a precision-matched dense control for 32K quality and 32K/64K timing.
- [QUEST](QUEST.md) uses an FP16 dense control and measures the shared-KV 2,048-token recipe at 32K and 64K.
- [Multipole and Squeezed p90 32K speed and quality](../llama-shared-index/REPORT.md) uses the shared LoD prefill and shared index.

## Reproduction

All runs used `uv` and `cluster-run`.

- 32K B=1 Dense BF16. [Command](provenance/1794792691260011804-llama32k-b1-final-predecode-dense/run.json) · [Log](provenance/1794792691260011804-llama32k-b1-final-predecode-dense/output.log).
- 32K B=1 RetroInfer GPU-only BF16. [Command](provenance/1794792691359874825-llama32k-b1-final-predecode-retro/run.json) · [Log](provenance/1794792691359874825-llama32k-b1-final-predecode-retro/output.log).
- 32K B=1 LoD two-tier BF16. [Command](provenance/1794792692002667738-llama32k-b1-final-predecode-two-tier/run.json) · [Log](provenance/1794792692002667738-llama32k-b1-final-predecode-two-tier/output.log).
- 32K B=1 LoD adaptive three-tier BF16. [Command](provenance/1794792691270792166-llama32k-b1-final-predecode-adaptive/run.json) · [Log](provenance/1794792691270792166-llama32k-b1-final-predecode-adaptive/output.log).
- 64K B=1 Dense BF16. [Command](provenance/1794792710571209897-llama64k-b1-final-predecode-dense/run.json) · [Log](provenance/1794792710571209897-llama64k-b1-final-predecode-dense/output.log).
- 64K B=1 RetroInfer GPU-only BF16. [Command](provenance/1794792710598819057-llama64k-b1-final-predecode-retro/run.json) · [Log](provenance/1794792710598819057-llama64k-b1-final-predecode-retro/output.log).
- 64K B=1 LoD two-tier BF16. [Command](provenance/1794792710628299762-llama64k-b1-final-predecode-two-tier/run.json) · [Log](provenance/1794792710628299762-llama64k-b1-final-predecode-two-tier/output.log).
- 64K B=1 LoD adaptive three-tier BF16. [Command](provenance/1794792710572967226-llama64k-b1-final-predecode-adaptive/run.json) · [Log](provenance/1794792710572967226-llama64k-b1-final-predecode-adaptive/output.log).
- 32K B=8 Dense BF16. [Command](provenance/1794792873552503918-llama32k-b8-final-predecode-dense/run.json) · [Log](provenance/1794792873552503918-llama32k-b8-final-predecode-dense/output.log).
- 32K B=8 RetroInfer GPU-only BF16. [Command](provenance/1794792873715068432-llama32k-b8-final-predecode-retro/run.json) · [Log](provenance/1794792873715068432-llama32k-b8-final-predecode-retro/output.log).
- 32K B=8 LoD two-tier BF16. [Command](provenance/1794793001341515711-llama32k-b8-final-predecode-two-tier-tile1024/run.json) · [Log](provenance/1794793001341515711-llama32k-b8-final-predecode-two-tier-tile1024/output.log).
- 32K B=8 LoD adaptive three-tier BF16. [Command](provenance/1794793001340952735-llama32k-b8-final-predecode-adaptive-tile1024/run.json) · [Log](provenance/1794793001340952735-llama32k-b8-final-predecode-adaptive-tile1024/output.log).
- Batch-eight automatic scratch trial. [Command](provenance/1794792873685108323-llama32k-b8-final-predecode-adaptive/run.json) · [Out-of-memory log](provenance/1794792873685108323-llama32k-b8-final-predecode-adaptive/output.log).
- Batch-eight automatic scratch trial. [Command](provenance/1794792873705271819-llama32k-b8-final-predecode-two-tier/run.json) · [Out-of-memory log](provenance/1794792873705271819-llama32k-b8-final-predecode-two-tier/output.log).

# Llama 3.1 H100 comparison figures

| Measurement | Published settings | LoD-matched settings |
|---|---|---|
| Decode latency and RULER quality, batch 1 | [Figure](decode-speed-quality.pdf) · [CSV](decode-speed-quality.csv) | [Figure](decode-speed-quality-lod-matched.pdf) · [CSV](decode-speed-quality-lod-matched.csv) |
| Decode latency by context, batches 1 and 4 | [Figure](decode-latency-paper.pdf) | [Figure](decode-latency-lod-matched.pdf) |
| Prefill latency by context, batches 1 and 4 | [Both settings](prefill-latency.pdf) | [Timing data and sources for both settings](comparison-timings.csv) |

The speed–quality points combine batch-one full-model decode timing with the mean across all 13 RULER tasks, 32 frozen prompts per task, at 32K and 64K. The decode curves use 16K through 128,000 tokens at batch one and 8K through 64K at batch four. Each decode timing covers 1,024 steps after 256 untimed steps with the same teacher-forced tokens. The median of two repetitions follows one warmup repetition. Timed decode includes KV writes, periodic maintenance and any required graph recapture. One step emits one token per request.

In the five figures, Squeezed uses dense causal prefill and fits its own index on normalized native-RoPE keys with Multipole's batched GPU k-means routine in place of the released cuML fitter. It calibrates a p80 cutoff. In the fixed-width ablation, Multipole and Squeezed share the dense-prefill KV tensors, and Squeezed reuses Multipole's initial centroids and token assignments with its own p90 cutoff. [Shared-index ablation](DENSE_PREFILL.md).

## Operating points

| Method | Published operating point used here | LoD-matched operating point |
|---|---|---|
| Dense BF16 and FP16 | Full causal attention | Same reference |
| LoD two-tier and adaptive three-tier, BF16 | Region threshold c=16. Three-tier also uses a c=0.5 page threshold in decode | Same reference |
| Multipole BF16 | One centroid per 16 indexed tokens, 512 remote exact tokens | Match initial centroids per KV head, retaining the 512-token budget and the released update interval |
| Squeezed BF16 | 5% centroids and p80 | Match initial centroids per KV head, retaining independent fitting and p80 |
| RetroInfer BF16 | GPU-only, 1.8% retrieval and 23.2% estimation, approximately one prefill centroid per 16 tokens | Match initial centroids per KV head, retaining the published retrieval and estimation fractions |
| QUEST FP16 | 2,048 exact tokens including the current page | Match LoD two-tier's mean exact-token count on the sparse layers, using whole 16-token pages |
| ClusterKV FP16 | 400 initial centroids and a 1,024-token budget | Keep 400 centroids and adjust the token budget to match LoD two-tier's mean exact-token count, including the exact sink and recent window |

Multipole's 512-token operating point comes from its Qwen3 experiments. QUEST and ClusterKV retain their two dense layers. [Multipole and Squeezed settings](PAPER_SETTINGS.md) · [RetroInfer paper](https://arxiv.org/pdf/2505.02922) · [QUEST paper](https://proceedings.mlr.press/v235/tang24l.html) · [ClusterKV paper](https://arxiv.org/pdf/2412.03213).

At the LoD-matched index widths, Squeezed retains about 6,200 and 12,300 prefix tokens per query head at the final timed decode step at 32K and 64K. Multipole caps exact remote attention at 512 and represents the remaining regions by centroids. [Paired score audit, held-out CWE, and Multipole budget check](SQUEEZED_WIDTH_AUDIT.md).

## LoD matching targets

Both LoD tiers have the same starting centroid count on these timing prefixes. For exact attention, the diagnostic samples every 64 steps of the 1,024-step window at batch one. It averages selected remote tokens, current token, local field and sink across query heads and layers 3–32. QUEST reserves one page for the current partial page, whose mean used length is 8.5 entries. ClusterKV's configured budget includes the current token, with the sink and active generated-token window counted in addition. Batch four uses the same per-request budget as batch one. The RULER runs use the corresponding nominal-context settings.

| Prompt tokens | LoD starting centroids per KV head | LoD mean exact entries per query head | QUEST configured budget | ClusterKV configured budget |
|---:|---:|---:|---:|---:|
| [8,192](llama-lod-exact-audit-8192.json) | 1,448 | 599.75 | 608 | 416 |
| [16,384](llama-lod-exact-audit-16384.json) | 2,048 | 847.24 | 848 | 664 |
| [32,768](llama-lod-exact-audit-32768.json) | 2,896 | 1,168.90 | 1,184 | 985 |
| [65,536](llama-lod-exact-audit-65536.json) | 4,096 | 1,885.31 | 1,888 | 1,702 |
| [128,000](llama-lod-exact-audit-128000.json) | 5,724 | 2,888.79 | 2,896 | 2,705 |

RetroInfer uses the nearest admissible prefill width of 5,728 at 128,000 and rounds each RULER request to the nearest width allowed by its segment count. Its 16K path builds the initial index during warmup with 2,048 matched centroids. At 8K it uses its dense fallback, so the sparse curve starts at 16K. Batch-four QUEST uses four independent author attention calls per layer within the model graph. The ClusterKV adapter supports batch one and uses the released eager planner for quality. [LoD-matched quality results and runs](RULER_LOD_MATCHED.md).

## Prefill

Prefill latency is wall time per batch for the method's prompt pass and required initial cache construction. LoD uses its threshold prefill and completes its deferred cache work. The prior methods use dense causal prefill and their own indexes. Four-request prefill processes four separate requests, using copies of the same timing prompt. Multipole fits each request and Squeezed builds and calibrates four independent indexes. At 16K, the RetroInfer total includes the separately timed initial index build during decode warmup. Initial cache allocation, model loading, and decode graph capture are outside the prefill totals. [Raw sources in the timing CSV](comparison-timings.csv).

### Reusing Multipole's index for Squeezed

Both measurements below use the LoD-matched centroid count at batch one. The independent Squeezed path fits normalized native-RoPE keys with the batched GPU solver and calibrates p80. The shared-index experiment uses Multipole's fitted index and calibrates p90. Its calibration and installation column is the incremental cost when the Multipole index already exists. A cold shared-index run also incurs the Multipole setup column. The dense prompt pass precedes these setup costs.

| Prompt tokens | Centroids per KV head | Independent Squeezed setup (s) | Shared-index calibration and install (s) | Multipole setup (s) |
|---:|---:|---:|---:|---:|
| 16,384 | 2,048 | [0.598](llama-figure-prefill-matched-b1-16384-squeezed.json) | [0.227](llama-denseprefill-b1-16384-t2048.json) | 0.508 |
| 32,768 | 2,896 | [1.341](llama-figure-prefill-matched-b1-32768-squeezed.json) | [0.313](llama-denseprefill-b1-32768-t2896.json) | 1.001 |
| 65,536 | 4,096 | [3.400](llama-figure-prefill-matched-b1-65536-squeezed.json) | [0.817](llama-denseprefill-b1-65536-t4096.json) | 1.648 |
| 128,000 | 5,724 | [8.785](llama-figure-prefill-matched-b1-128000-squeezed.json) | [1.565](llama-denseprefill-b1-128000-t5724.json) | 3.238 |

The 2,900-centroid runs are available as a fixed-width ablation. [Speed–quality ablation](decode-speed-quality-fixed2900.pdf) · [Batch-one width ablation](fixed-centroid-ablation-b1.pdf) · [Batch-four width ablation](fixed-centroid-ablation-b4.pdf).

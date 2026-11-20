# Llama 3.1 H100 context scaling

The [comparison figure index](COMPARISON_FIGURES.md) links the published-setting and LoD-matched speed–quality and decode plots and the separate prefill plot.

Batch one on H100 80 GB. Context points are 16,384, 32,768, 65,536 and 128,000 tokens. The 16K prompt uses the first 16,384 tokens of the archived 32K timing prompt. The other prompts come from the frozen timing manifests. [Timing tokens](prompts-context-scaling.json).

At 128,000 tokens the historical full-model batched comparison ran at batch two. Dense ran out of memory at batch four. At batch eight the BF16 KV alone requires 125 GiB before weights and scratch. [Historical 128K batch-two runs](../../../benchmarks/RETRO_LOD.md#historical-llama-full-model-amortized-retroinfer-comparison) · [32K batch-eight runs](../llama-final-checks/REPORT.md#32k-batch-8).

[Earlier decode latency plot with the shared p90 index](decode-latency.pdf) · [Plot data](context-timings.csv).

[Earlier batch-four plot with the shared p90 index](decode-latency-b4.pdf) · [Batch-four timings and 32K ablation](../llama-final-checks/REPORT.md#batch-four-context-scaling).

## Decode latency and RULER quality

The batch-one figure uses common axes for 32K and 64K. Each point combines the median whole-model decode latency on the frozen timing prefix with the equally weighted mean over 13 RULER tasks, 32 examples per task. The CSV records the precision, prefill policy, starting centroid count where applicable, attention budget or routing threshold, and timing and quality sources for each point. [Figure](decode-speed-quality.pdf) · [Plot data and sources](decode-speed-quality.csv).

LoD uses each tier's own threshold prefill. The main comparison runs Multipole with one centroid per 16 indexed tokens and a 512-token remote exact budget, and Squeezed with 5% centroids and its p80 cutoff. Both use dense causal prefill. [Paper-setting results and run commands](PAPER_SETTINGS.md).

The fixed-2,900-centroid experiment uses dense causal prefill, Multipole's 512-token budget, and a p90 Squeezed cutoff calibrated on the same initial index. [Fixed-count speed–quality ablation](decode-speed-quality-fixed2900.pdf) · [Plot data](decode-speed-quality-fixed2900.csv) · [Per-task scores and commands](DENSE_PREFILL.md).

ClusterKV uses 400 initial centroids and a 1,024-token budget in both measurements. Quality uses the released eager planner, and speed uses CUDA-graph capture with device planning. At 64K, the device planner reproduced all four sampled retrieval misses. QUEST uses the same 2,048-token budget for speed and quality. [ClusterKV verification](../ruler-64k-panel/RULER_64K.md#clusterkv-graph-rollout-check) · [QUEST results](../llama-final-checks/QUEST.md).

## Matched full-model decode

At each context, every method starts from the full listed prompt and replays the same final 1,280 prompt token IDs as decode inputs. Each arm uses 256 untimed steps and 1,024 graph-captured timed steps. Timings are medians of two repetitions after a warmup. Prefill includes cache completion and any index construction that takes place during prefill. RetroInfer at 16K builds its initial index during the first 69 untimed decode steps. At every length, the timed window includes a rebuild. QUEST uses a 2,048-token budget. The two ClusterKV curves keep 400 initial centroids and use a fixed 1,024-token budget or ⌊16√N⌋, where N is the initial prompt length. Both arms were retimed on the same GPU at each length. Each timed repetition includes four index updates.

LoD adaptive two-tier selects a variable number of regions using the c=16 mass threshold in both prefill and decode. Adaptive three-tier uses the same region threshold and adds a c=0.5 page threshold in decode.

At 128,000 tokens, LoD uses 4,096-query prefill scratch tiles and the same 16,384-token initial exact field and attention thresholds as at shorter contexts.

Multipole and Squeezed use dense causal prefill and their Triton 3.1 decoders. Their initial decoder index has the same number of slots per KV head as both LoD modes at the start of the timed window. This is also the matching rule for batch four. LoD adds slots during decode. Multipole and Squeezed keep the chosen width for the run. Multipole uses a 512-token exact remote budget and includes eight tail refinements per measured repetition. Squeezed calibrates its p90 cutoff on the index for each run. Timed decode includes KV writes and maintenance. Initial index fitting, calibration and graph capture are outside the timed window. [Dense-prefill runs and commands](DENSE_PREFILL.md#decode-timing).

| Method | 16K | 32K | 64K | 128,000 |
|---|---:|---:|---:|---:|
| Dense BF16 | [7.846](llama-context-16384-dense.json) | [8.732](../llama-final-checks/llama32k-b1-final-predecode-dense.json) | [10.118](../llama-final-checks/llama64k-b1-final-predecode-dense.json) | [12.723](llama-context-128000-dense.json) |
| RetroInfer BF16 | [9.513](llama-context-16384-retro-deferred-index.json) | [9.790](../llama-final-checks/llama32k-b1-final-predecode-retro.json) | [10.678](../llama-final-checks/llama64k-b1-final-predecode-retro.json) | [12.271](llama-context-128000-retro.json) |
| LoD adaptive two-tier BF16 | [7.129](llama-context-16384-two-tier.json) | [7.237](../llama-final-checks/llama32k-b1-final-predecode-two-tier.json) | [7.481](../llama-final-checks/llama64k-b1-final-predecode-two-tier.json) | [7.784](llama-context-128000-two-tier.json) |
| LoD adaptive three-tier BF16 | [7.705](llama-context-16384-adaptive.json) | [8.076](../llama-final-checks/llama32k-b1-final-predecode-adaptive.json) | [8.534](../llama-final-checks/llama64k-b1-final-predecode-adaptive.json) | [9.150](llama-context-128000-adaptive.json) |
| Dense FP16 | [7.813](llama-context-16384-dense-fp16.json) | [8.669](../llama-final-checks/llama32k-b1-quest-matched-dense-fp16.json) | [10.056](../llama-final-checks/llama64k-b1-quest-matched-dense-fp16.json) | [12.670](llama-context-128000-dense-fp16.json) |
| QUEST FP16 | [7.753](llama-context-16384-quest.json) | [8.516](../llama-final-checks/llama32k-b1-quest-matched-quest.json) | [9.687](../llama-final-checks/llama64k-b1-quest-matched-quest.json) | [12.036](llama-context-128000-quest.json) |
| ClusterKV FP16 (1,024) | [8.795](llama-context-16384-clusterkv-fixed.json) | [9.084](llama-context-32768-clusterkv-fixed.json) | [9.373](llama-context-65536-clusterkv-fixed.json) | [9.897](llama-context-128000-clusterkv-fixed.json) |
| ClusterKV FP16 (16√N) | [9.053](llama-context-16384-clusterkv-sqrt.json) | [9.654](llama-context-32768-clusterkv-sqrt.json) | [10.737](llama-context-65536-clusterkv-sqrt.json) | [12.670](llama-context-128000-clusterkv-sqrt.json) |
| Multipole BF16 (512) | [15.626](llama-denseprefill-b1-16384-t2048.json) | [16.493](llama-denseprefill-b1-32768-t2896.json) | [17.505](llama-denseprefill-b1-65536-t4096.json) | [21.652](llama-denseprefill-b1-128000-t5724.json) |
| Squeezed BF16 (p90) | [10.852](llama-denseprefill-b1-16384-t2048.json) | [11.859](llama-denseprefill-b1-32768-t2896.json) | [12.871](llama-denseprefill-b1-65536-t4096.json) | [15.642](llama-denseprefill-b1-128000-t5724.json) |

Decode latency in ms/token.

| Prompt tokens | Both LoD modes at timed start | Two-tier LoD at timed end | Multipole and Squeezed slots |
|---:|---:|---:|---:|
| [16,384](llama-context-16384-two-tier-centroid-count.json) | 2,048 | 2,111 | 2,048 |
| [32,768](llama-context-32768-two-tier-centroid-count.json) | 2,896 | 2,941 | 2,896 |
| [65,536](llama-context-65536-two-tier-centroid-count.json) | 4,096 | 4,127 | 4,096 |
| [128,000](llama-context-128000-two-tier-centroid-count.json) | 5,724 | 5,747 | 5,724 |

The starting count matches on all 32 layers for both LoD modes. All eight timed Multipole updates refine the tail. The full block-layout rebuild falls outside these windows. [Adaptive three-tier count checks for 16K](../llama-final-checks/llama16384-b1-adaptive-centroid-count-check.json), [32K](../llama-final-checks/llama32768-b1-adaptive-centroid-count-check.json), [64K](../llama-final-checks/llama65536-b1-adaptive-centroid-count-check.json) and [128,000 tokens](../llama-final-checks/llama128000-b1-adaptive-centroid-count-check.json).

The scaled ClusterKV budgets are 2,048, 2,896, 4,096, 5,724 tokens in order of prompt length. The budget is set once per run, using the same sink and exact recent window in both arms. From 16K to 128,000, decode latency rises by 3.62 ms/token with the scaled budget and 1.10 ms/token with the fixed budget. The [5,724-token graph check](llama-context-128000-clusterkv-sqrt-graph-check.json) matched same-cache eager logits on 37 steps. [Command](provenance/1794798656823816955-llama-context-128000-clusterkv-sqrt-graph-check/run.json).

## Fixed-2,900-centroid ablation

Both widths use the same GPU, dense prefill, teacher-forced decode window, and 512-token exact remote budget for Multipole. Squeezed recalibrates p90 for each index.

[Batch-one ablation plot](fixed-centroid-ablation-b1.pdf) · [Batch-four ablation plot](fixed-centroid-ablation-b4.pdf).

| Prompt tokens | Multipole fixed | Multipole matched | Squeezed fixed | Squeezed matched |
|---:|---:|---:|---:|---:|
| 16,384 | [17.497](llama-denseprefill-b1-16384-t2900.json) | [15.626](llama-denseprefill-b1-16384-t2048.json) | [11.496](llama-denseprefill-b1-16384-t2900.json) | [10.852](llama-denseprefill-b1-16384-t2048.json) |
| 32,768 | [17.116](llama-denseprefill-b1-32768-t2900.json) | [16.493](llama-denseprefill-b1-32768-t2896.json) | [11.845](llama-denseprefill-b1-32768-t2900.json) | [11.859](llama-denseprefill-b1-32768-t2896.json) |
| 65,536 | [17.077](llama-denseprefill-b1-65536-t2900.json) | [17.505](llama-denseprefill-b1-65536-t4096.json) | [12.008](llama-denseprefill-b1-65536-t2900.json) | [12.871](llama-denseprefill-b1-65536-t4096.json) |
| 128,000 | [17.184](llama-denseprefill-b1-128000-t2900.json) | [21.652](llama-denseprefill-b1-128000-t5724.json) | [13.399](llama-denseprefill-b1-128000-t2900.json) | [15.642](llama-denseprefill-b1-128000-t5724.json) |

Decode latency in ms/token.

## Prefill time in seconds

| Method | 16K | 32K | 64K | 128,000 |
|---|---:|---:|---:|---:|
| Dense BF16 | 0.582 | 1.585 | 4.813 | 15.746 |
| RetroInfer BF16 | 0.594 | 1.795 | 5.331 | 16.415 |
| LoD adaptive two-tier BF16 | 0.639 | 1.659 | 4.527 | 11.693 |
| LoD adaptive three-tier BF16 | 0.675 | 1.752 | 4.879 | 13.382 |
| Dense FP16 | 0.703 | 1.578 | 4.791 | 15.866 |
| QUEST FP16 | 0.603 | 1.614 | 4.866 | 15.982 |
| ClusterKV FP16 (1,024) | 0.928 | 2.229 | 5.911 | 17.496 |
| ClusterKV FP16 (16√N) | 0.899 | 2.227 | 5.913 | 17.519 |

## Multipole and Squeezed setup in seconds

The dense prompt pass is shared within each timing run. Cache and index costs are medians of the measured repetitions. Squeezed reuses the initial Multipole index and calibrates its cutoff.

| Prompt tokens | Dense prefix preparation | Multipole cache and fitting | Squeezed calibration | Squeezed cache install |
|---:|---:|---:|---:|---:|
| 16,384 | 0.876 | 0.508 | 0.217 | 0.010 |
| 32,768 | 1.852 | 1.001 | 0.298 | 0.016 |
| 65,536 | 5.062 | 1.648 | 0.786 | 0.031 |
| 128,000 | 15.998 | 3.238 | 1.504 | 0.060 |

The [16K RetroInfer graph check](llama-context-16384-retro-deferred-graph-check.json) matched eager logits on all 17 sampled steps after the first index build. [Command](provenance/1794795967541361030-llama-context-16384-retro-deferred-graph-check/run.json).

The [dense-prefill RULER panel](DENSE_PREFILL.md) uses the fixed 2,900-slot index for its Multipole and Squeezed results.

## Reproduction

GPU runs use `uv` and `cluster-run`. Each command records the kernel settings, prompt and timing window.

The [dense-prefill report](DENSE_PREFILL.md#timing-commands) lists the Multipole and Squeezed commands behind the current plots. The original LoD-prefill runs below are historical.

- llama-context-16384-dense. [Results](llama-context-16384-dense.json) · [Command](provenance/1794795662953271453-llama-context-16384-dense/run.json) · [Log](provenance/1794795662953271453-llama-context-16384-dense/output.log).
- llama-context-16384-retro-deferred-index. [Results](llama-context-16384-retro-deferred-index.json) · [Command](provenance/1794795907523572934-llama-context-16384-retro-deferred-index/run.json) · [Log](provenance/1794795907523572934-llama-context-16384-retro-deferred-index/output.log).
- llama-context-16384-two-tier. [Results](llama-context-16384-two-tier.json) · [Command](provenance/1794795663052907598-llama-context-16384-two-tier/run.json) · [Log](provenance/1794795663052907598-llama-context-16384-two-tier/output.log).
- llama-context-16384-adaptive. [Results](llama-context-16384-adaptive.json) · [Command](provenance/1794795662942316253-llama-context-16384-adaptive/run.json) · [Log](provenance/1794795662942316253-llama-context-16384-adaptive/output.log).
- llama-context-16384-dense-fp16. [Results](llama-context-16384-dense-fp16.json) · [Command](provenance/1794795687112586113-llama-context-16384-dense-fp16/run.json) · [Log](provenance/1794795687112586113-llama-context-16384-dense-fp16/output.log).
- llama-context-16384-quest. [Results](llama-context-16384-quest.json) · [Command](provenance/1794795687004875640-llama-context-16384-quest/run.json) · [Log](provenance/1794795687004875640-llama-context-16384-quest/output.log).
- llama-context-16384-clusterkv-fixed. [Results](llama-context-16384-clusterkv-fixed.json) · [Command](provenance/1794798660547456123-llama-context-16384-clusterkv-fixed/run.json) · [Log](provenance/1794798660547456123-llama-context-16384-clusterkv-fixed/output.log).
- llama-context-16384-clusterkv-sqrt. [Results](llama-context-16384-clusterkv-sqrt.json) · [Command](provenance/1794798656880303354-llama-context-16384-clusterkv-sqrt/run.json) · [Log](provenance/1794798656880303354-llama-context-16384-clusterkv-sqrt/output.log).
- llama32k-b1-final-predecode-dense. [Results](../llama-final-checks/llama32k-b1-final-predecode-dense.json) · [Command](../llama-final-checks/provenance/1794792691260011804-llama32k-b1-final-predecode-dense/run.json) · [Log](../llama-final-checks/provenance/1794792691260011804-llama32k-b1-final-predecode-dense/output.log).
- llama32k-b1-final-predecode-retro. [Results](../llama-final-checks/llama32k-b1-final-predecode-retro.json) · [Command](../llama-final-checks/provenance/1794792691359874825-llama32k-b1-final-predecode-retro/run.json) · [Log](../llama-final-checks/provenance/1794792691359874825-llama32k-b1-final-predecode-retro/output.log).
- llama32k-b1-final-predecode-two-tier. [Results](../llama-final-checks/llama32k-b1-final-predecode-two-tier.json) · [Command](../llama-final-checks/provenance/1794792692002667738-llama32k-b1-final-predecode-two-tier/run.json) · [Log](../llama-final-checks/provenance/1794792692002667738-llama32k-b1-final-predecode-two-tier/output.log).
- llama32k-b1-final-predecode-adaptive. [Results](../llama-final-checks/llama32k-b1-final-predecode-adaptive.json) · [Command](../llama-final-checks/provenance/1794792691270792166-llama32k-b1-final-predecode-adaptive/run.json) · [Log](../llama-final-checks/provenance/1794792691270792166-llama32k-b1-final-predecode-adaptive/output.log).
- llama32k-b1-quest-matched-dense-fp16. [Results](../llama-final-checks/llama32k-b1-quest-matched-dense-fp16.json) · [Command](../llama-final-checks/provenance/1794794813903968401-llama32k-b1-quest-matched-dense-fp16/run.json) · [Log](../llama-final-checks/provenance/1794794813903968401-llama32k-b1-quest-matched-dense-fp16/output.log).
- llama32k-b1-quest-matched-quest. [Results](../llama-final-checks/llama32k-b1-quest-matched-quest.json) · [Command](../llama-final-checks/provenance/1794794813766694006-llama32k-b1-quest-matched-quest/run.json) · [Log](../llama-final-checks/provenance/1794794813766694006-llama32k-b1-quest-matched-quest/output.log).
- llama-context-32768-clusterkv-fixed. [Results](llama-context-32768-clusterkv-fixed.json) · [Command](provenance/1794798656932924160-llama-context-32768-clusterkv-fixed/run.json) · [Log](provenance/1794798656932924160-llama-context-32768-clusterkv-fixed/output.log).
- llama-context-32768-clusterkv-sqrt. [Results](llama-context-32768-clusterkv-sqrt.json) · [Command](provenance/1794798660606245701-llama-context-32768-clusterkv-sqrt/run.json) · [Log](provenance/1794798660606245701-llama-context-32768-clusterkv-sqrt/output.log).
- llama64k-b1-final-predecode-dense. [Results](../llama-final-checks/llama64k-b1-final-predecode-dense.json) · [Command](../llama-final-checks/provenance/1794792710571209897-llama64k-b1-final-predecode-dense/run.json) · [Log](../llama-final-checks/provenance/1794792710571209897-llama64k-b1-final-predecode-dense/output.log).
- llama64k-b1-final-predecode-retro. [Results](../llama-final-checks/llama64k-b1-final-predecode-retro.json) · [Command](../llama-final-checks/provenance/1794792710598819057-llama64k-b1-final-predecode-retro/run.json) · [Log](../llama-final-checks/provenance/1794792710598819057-llama64k-b1-final-predecode-retro/output.log).
- llama64k-b1-final-predecode-two-tier. [Results](../llama-final-checks/llama64k-b1-final-predecode-two-tier.json) · [Command](../llama-final-checks/provenance/1794792710628299762-llama64k-b1-final-predecode-two-tier/run.json) · [Log](../llama-final-checks/provenance/1794792710628299762-llama64k-b1-final-predecode-two-tier/output.log).
- llama64k-b1-final-predecode-adaptive. [Results](../llama-final-checks/llama64k-b1-final-predecode-adaptive.json) · [Command](../llama-final-checks/provenance/1794792710572967226-llama64k-b1-final-predecode-adaptive/run.json) · [Log](../llama-final-checks/provenance/1794792710572967226-llama64k-b1-final-predecode-adaptive/output.log).
- llama64k-b1-quest-matched-dense-fp16. [Results](../llama-final-checks/llama64k-b1-quest-matched-dense-fp16.json) · [Command](../llama-final-checks/provenance/1794793594169491551-llama64k-b1-quest-matched-dense-fp16/run.json) · [Log](../llama-final-checks/provenance/1794793594169491551-llama64k-b1-quest-matched-dense-fp16/output.log).
- llama64k-b1-quest-matched-quest. [Results](../llama-final-checks/llama64k-b1-quest-matched-quest.json) · [Command](../llama-final-checks/provenance/1794793594092536896-llama64k-b1-quest-matched-quest/run.json) · [Log](../llama-final-checks/provenance/1794793594092536896-llama64k-b1-quest-matched-quest/output.log).
- llama-context-65536-clusterkv-fixed. [Results](llama-context-65536-clusterkv-fixed.json) · [Command](provenance/1794798660767925427-llama-context-65536-clusterkv-fixed/run.json) · [Log](provenance/1794798660767925427-llama-context-65536-clusterkv-fixed/output.log).
- llama-context-65536-clusterkv-sqrt. [Results](llama-context-65536-clusterkv-sqrt.json) · [Command](provenance/1794798656971076913-llama-context-65536-clusterkv-sqrt/run.json) · [Log](provenance/1794798656971076913-llama-context-65536-clusterkv-sqrt/output.log).
- llama-context-128000-dense. [Results](llama-context-128000-dense.json) · [Command](provenance/1794797803997464863-llama-context-128000-dense/run.json) · [Log](provenance/1794797803997464863-llama-context-128000-dense/output.log).
- llama-context-128000-retro. [Results](llama-context-128000-retro.json) · [Command](provenance/1794795967431477083-llama-context-128000-retro/run.json) · [Log](provenance/1794795967431477083-llama-context-128000-retro/output.log).
- llama-context-128000-two-tier. [Results](llama-context-128000-two-tier.json) · [Command](provenance/1794796997617316134-llama-context-128000-two-tier/run.json) · [Log](provenance/1794796997617316134-llama-context-128000-two-tier/output.log).
- llama-context-128000-adaptive. [Results](llama-context-128000-adaptive.json) · [Command](provenance/1794795967452936579-llama-context-128000-adaptive/run.json) · [Log](provenance/1794795967452936579-llama-context-128000-adaptive/output.log).
- llama-context-128000-dense-fp16. [Results](llama-context-128000-dense-fp16.json) · [Command](provenance/1794796997740678306-llama-context-128000-dense-fp16/run.json) · [Log](provenance/1794796997740678306-llama-context-128000-dense-fp16/output.log).
- llama-context-128000-quest. [Results](llama-context-128000-quest.json) · [Command](provenance/1794795967559991506-llama-context-128000-quest/run.json) · [Log](provenance/1794795967559991506-llama-context-128000-quest/output.log).
- llama-context-128000-clusterkv-fixed. [Results](llama-context-128000-clusterkv-fixed.json) · [Command](provenance/1794798657012290445-llama-context-128000-clusterkv-fixed/run.json) · [Log](provenance/1794798657012290445-llama-context-128000-clusterkv-fixed/output.log).
- llama-context-128000-clusterkv-sqrt. [Results](llama-context-128000-clusterkv-sqrt.json) · [Command](provenance/1794798660804798717-llama-context-128000-clusterkv-sqrt/run.json) · [Log](provenance/1794798660804798717-llama-context-128000-clusterkv-sqrt/output.log).
- llama-context-16384-shared-matched-lod-centroids. [Results](llama-context-16384-shared-matched-lod-centroids.json) · [Command](provenance/1794801781003679586-llama-context-16384-shared-matched-lod-centroids/run.json) · [Log](provenance/1794801781003679586-llama-context-16384-shared-matched-lod-centroids/output.log).
- llama-context-32768-shared-matched-lod-centroids. [Results](llama-context-32768-shared-matched-lod-centroids.json) · [Command](provenance/1794801781099484899-llama-context-32768-shared-matched-lod-centroids/run.json) · [Log](provenance/1794801781099484899-llama-context-32768-shared-matched-lod-centroids/output.log).
- llama-context-65536-shared-matched-lod-centroids. [Results](llama-context-65536-shared-matched-lod-centroids.json) · [Command](provenance/1794801781048960502-llama-context-65536-shared-matched-lod-centroids/run.json) · [Log](provenance/1794801781048960502-llama-context-65536-shared-matched-lod-centroids/output.log).
- llama-context-128000-shared-matched-lod-centroids. [Results](llama-context-128000-shared-matched-lod-centroids.json) · [Command](provenance/1794801781153378155-llama-context-128000-shared-matched-lod-centroids/run.json) · [Log](provenance/1794801781153378155-llama-context-128000-shared-matched-lod-centroids/output.log).
- llama-context-16384-shared-matched-v2. [Results](llama-context-16384-shared-matched-v2.json) · [Command](provenance/1794800394180520009-llama-context-16384-shared-matched-v2/run.json) · [Log](provenance/1794800394180520009-llama-context-16384-shared-matched-v2/output.log).
- llama-context-32768-shared-matched-v2. [Results](llama-context-32768-shared-matched-v2.json) · [Command](provenance/1794800394274652655-llama-context-32768-shared-matched-v2/run.json) · [Log](provenance/1794800394274652655-llama-context-32768-shared-matched-v2/output.log).
- llama-context-65536-shared-matched-v2. [Results](llama-context-65536-shared-matched-v2.json) · [Command](provenance/1794800394190513305-llama-context-65536-shared-matched-v2/run.json) · [Log](provenance/1794800394190513305-llama-context-65536-shared-matched-v2/output.log).
- llama-context-128000-shared-matched-v2. [Results](llama-context-128000-shared-matched-v2.json) · [Command](provenance/1794800394201460513-llama-context-128000-shared-matched-v2/run.json) · [Log](provenance/1794800394201460513-llama-context-128000-shared-matched-v2/output.log).
- llama-context-16384-two-tier-centroid-count. [Results](llama-context-16384-two-tier-centroid-count.json) · [Command](provenance/1794801733768566081-llama-context-16384-two-tier-centroid-count/run.json) · [Log](provenance/1794801733768566081-llama-context-16384-two-tier-centroid-count/output.log).
- llama-context-32768-two-tier-centroid-count. [Results](llama-context-32768-two-tier-centroid-count.json) · [Command](provenance/1794801733633589049-llama-context-32768-two-tier-centroid-count/run.json) · [Log](provenance/1794801733633589049-llama-context-32768-two-tier-centroid-count/output.log).
- llama-context-65536-two-tier-centroid-count. [Results](llama-context-65536-two-tier-centroid-count.json) · [Command](provenance/1794801733749131510-llama-context-65536-two-tier-centroid-count/run.json) · [Log](provenance/1794801733749131510-llama-context-65536-two-tier-centroid-count/output.log).
- llama-context-128000-two-tier-centroid-count. [Results](llama-context-128000-two-tier-centroid-count.json) · [Command](provenance/1794801733634807675-llama-context-128000-two-tier-centroid-count/run.json) · [Log](provenance/1794801733634807675-llama-context-128000-two-tier-centroid-count/output.log).

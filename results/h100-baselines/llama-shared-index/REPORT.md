# Llama 3.1 8B shared-centroid comparison at 32K

**Historical experiment.** These runs initialized every decoder from a LoD prompt pass. The paper comparison uses dense prefill for Multipole and Squeezed. See the [current comparison](../llama-context-scaling/REPORT.md#decode-latency-and-ruler-quality).

Llama-3.1-8B-Instruct, BF16, one H100 80 GB. LoD, dense, Multipole, and Squeezed p90 start from the same optimized two-tier LoD prefill and first generated token. LoD uses c=16 per-region thresholds in prefill and decode, a 16,384-token initial exact field, fused saved-score selection, scaled-FP16 coarse projection, tensor-core exact reduction, fused Llama cache-key RMS normalization, and materialized shared-GQA decode.

Multipole fits 2,900 centroids per physical KV head, uses a 512-token exact budget and value-centroid replacement, and refines its final clustering block every 128 tokens. Squeezed uses its released kernels with shared GQA storage. It calibrates a p90 cutoff from 100 prefill queries at routing position 2,048 using Multipole’s initial index, then holds the index fixed. Squeezed keeps the sink, unclustered prefix tail, and generated tokens exact. LoD uses its usual centroid schedule.

## Batch-one whole-model decode

The four CUDA graphs use the same 30,720-token Llama prefix and the same teacher-forced tokens from the [timing manifest](../../h100-llama31/raw/prompts-b1.json). After 512 warmup steps, the run measures three consecutive 512-step windows through 32,768 tokens. Timings include KV writes, launches, and periodic cache maintenance. Each decoder passed the initial same-cache graph/eager logit check.

| Decoder | Centroid slots per KV head | Median ms/token | Window ms/token |
|---|---:|---:|---|
| LoD two-tier, c=16 | 2,816–2,884 | **7.694** | 7.636, 7.694, 7.702 |
| Dense | — | 8.622 | 8.608, 8.622, 8.641 |
| Multipole, 512 | 2,900 | 17.855 | 17.842, 17.872, 17.855 |
| Squeezed p90 on the Multipole index | 2,900 | 11.809 | 11.798, 11.809, 11.853 |

Each window includes 64 LoD layer catch-ups and four Multipole index refinements. At the ends of the windows, Squeezed selected a mean of 2,664–3,387 exact prefix entries per query head, including the exact sink and prefix tail.

Initial setup is outside the decode timer. Multipole’s cache installation and index build took 3.21 s. Reusing that index took a further 0.53 s for Squeezed calibration and 0.02 s for Squeezed cache installation. [Raw timings](llama-shared-index-2900-decode-v1.json) · [Run command](provenance/1794603798692478792-llama-shared-index-2900-decode-v1/run.json).

## Batch-eight whole-model decode

Eight copies of the same 30,720-token prefix, with independent KV and updates for each request. All methods use the batch-one timing prompt and the same 512 warmup and three 512-step timed windows. Each step emits eight tokens. LoD prefill KV was checked bit for bit across all requests and layers against the shared prefix used by the other decoders.

| Decoder | Median ms/batch step | Aggregate tokens/s | Throughput vs B=1 | Window ms/batch step |
|---|---:|---:|---:|---|
| LoD two-tier, c=16 | **10.024** | **798.1** | 6.14× | 10.013, 10.024, 10.097 |
| Dense | 17.896 | 447.0 | 3.85× | 17.728, 17.896, 18.063 |
| Multipole, 512 | 23.576 | 339.3 | 6.06× | 23.432, 23.576, 23.730 |
| Squeezed p90 on the Multipole index | 16.858 | 474.6 | 5.60× | 16.855, 16.858, 17.043 |

Multipole fits one initial 2,900-centroid index for the repeated prompt, then copies it into each request’s cache and refines each request every 128 tokens. Its clustering distance calculation remains compiled, with internal CUDA graph caching disabled to fit the growing temporary buffers on one H100. The whole-model decode graph stays captured. Squeezed takes the same initial index and p90 threshold and runs the released lookup and attention across flattened request and head axes. The sparse kernel directly addresses each head’s fixed range of blocks. The one-time CPU staging of prefill KV between runtimes, initial index construction, and calibration are outside the decode timer. [Squeezed addressing patch](squeezed-shared-batch.patch).

Each measured window includes 512 LoD layer-by-request cache catch-ups and four full-batch Multipole refinements. Multipole’s initial fitting and cache setup took 3.23 s. Squeezed calibration took 0.54 s and cache setup took 0.57 s. All four initial same-cache graph/eager checks matched logits. A separate test with distinct requests compared the batched LoD and Squeezed kernels against eight independent controllers. The previous full-window timing was within 1% for all four decoders, and its raw LoD update counter omitted batch updates. [Raw timings](llama-shared-index-2900-b8-decode-v4.json) · [Run command](provenance/1794609542946202225-llama-shared-index-2900-b8-decode-v4/run.json) · [Batch addressing checks](provenance/1794609510411586611-llama-shared-index-b8-addressing-v2/output.log).

## 32K RULER, batch one

The [64K evaluation](../ruler-64k-panel/RULER_64K.md#bf16-shared-prefill-comparison) uses the same shared-index decoder settings. The [batch-one decode curves](../llama-context-scaling/REPORT.md#matched-full-model-decode) cover 16K through 128,000 tokens with a common timing window.

The quality run uses all 416 prompts from the [frozen Llama cohort](../quality-ruler-32k-frozen-13x32.json.gz), with 32 examples per task. Scores stop at the first EOS/EOT. The prompt hashes match the prior Llama comparison on all 416 examples. Shared LoD prefill matches the first token from fully dense prefill on 409 of 416 prompts. All four decoders passed the initial same-cache graph/eager check on every prompt. [Generations](llama-shared-index-2900-ruler-full-v1.json) · [Run command](provenance/1794603800006103380-llama-shared-index-2900-ruler-full-v1/run.json).

| Task | Examples | LoD | Dense decode | Multipole | Squeezed p90 |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 32 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 32 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 32 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 32 | 96.88% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 32 | 100.00% | 100.00% | 93.75% | 100.00% |
| niah_multikey_3 | 32 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 32 | 99.22% | 96.09% | 98.44% | 99.22% |
| niah_multiquery | 32 | 98.44% | 98.44% | 99.22% | 97.66% |
| vt | 32 | 100.00% | 100.00% | 95.62% | 94.38% |
| fwe | 32 | 88.54% | 91.67% | 87.50% | 94.79% |
| cwe | 32 | 36.25% | 31.88% | 22.19% | 44.38% |
| qa_1 | 32 | 78.12% | 78.12% | 78.12% | 78.12% |
| qa_2 | 32 | 46.88% | 46.88% | 46.88% | 46.88% |
| **13-task mean** | **416** | **88.02%** | **87.93%** | **86.29%** | **88.88%** |

Relative to dense decode from the shared prefill, LoD changes the macro mean by +0.10 percentage points (95% paired bootstrap interval −1.08 to +1.26), Multipole by −1.64 points (−3.11 to −0.24), and Squeezed p90 by +0.95 points (+0.08 to +1.88). The bootstrap resamples prompts within each equally weighted task.

The quality run took 70.4 minutes of prompt processing. Median prompt time was 9.80 s, including 1.75 s for shared prefill, 3.47 s for Multipole cache setup and fitting, and 0.50 s for Squeezed calibration.

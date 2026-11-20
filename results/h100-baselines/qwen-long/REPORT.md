# Qwen3-8B long-context quality and decode comparison

BF16, batch one, one H100 80GB per run. All GPU runs use uv and immutable cluster-run snapshots. Corrected runs use 2x YaRN with max_position_embeddings=65536 at 64K and 4x YaRN with max_position_embeddings=131072 near 128K.

**Prefill policy.** Panels in this archive that share a LoD-generated prefix across decoders are historical. Current baseline comparisons use independent method-appropriate prefill. [Current Llama comparison](../llama-context-scaling/REPORT.md#decode-latency-and-ruler-quality).

**Correction to the previous long-context results.** The earlier 128K runs left max_position_embeddings at 40,960. Transformers 4.51.3 consequently used 1.25x YaRN despite factor=4 in the configuration. The native runtime and independent Transformers control shared this error. Those results are quarantined. The 32K native-RoPE panel remains valid.

Qwen documents 32,768 native tokens and 131,072 tokens with YaRN. The correction follows its [Transformers long-context configuration](https://github.com/QwenLM/Qwen3/blob/main/docs/source/inference/transformers.md).

## Full-model quality

Dense and LoD share the frozen prompts at each length. Each length uses a separate cohort with the same task generators, seed and Qwen chat format. Thinking is disabled and scoring stops at the first EOS/EOT. LoD uses uncapped per-region c=16 threshold attention in prefill and decode with a 16,384-token initial exact field.

### Near 64K

Prompts span 61,417–65,297 tokens.

| Task | Examples | Dense | Optimized LoD |
|---|---:|---:|---:|
| niah_multikey_2 | 8 | 75.00% | 62.50% |
| niah_multikey_3 | 8 | 87.50% | 75.00% |
| cwe | 8 | 67.50% | 62.50% |
| qa_1 | 8 | 50.00% | 50.00% |

On paired examples, 21 scores tie, 4 favor LoD, and 7 favor dense. Fixed-length generations match exactly on 9 of the paired prompts. [Dense outputs](qwen-long-yarnfix-64k-v1-dense-65536-quality.json), [LoD outputs](qwen-long-yarnfix-64k-v1-lod-lod-65536-quality.json).

### Near 128,000 tokens

Prompts span 124,179–127,811 tokens.

| Task | Examples | Dense | Optimized LoD |
|---|---:|---:|---:|
| niah_multikey_2 | 8 | 75.00% | 62.50% |
| niah_multikey_3 | 8 | 25.00% | 37.50% |
| cwe | 8 | 28.75% | 21.25% |
| qa_1 | 8 | 50.00% | 50.00% |

On paired examples, 21 scores tie, 5 favor LoD, and 6 favor dense. Fixed-length generations match exactly on 1 of the paired prompts. [Dense outputs](qwen-long-yarnfix-v1-dense-128000-quality.json), [LoD outputs](qwen-long-yarnfix-v2-lod-128000-quality.json).


The [frequency regression check](qwen-long-yarnfix-v1-rope-check-128000-rope-check.json) evaluates the installed Transformers implementation directly.

| Configuration | max_position_embeddings | Declared factor | Effective factor | Attention scaling |
|---|---:|---:|---:|---:|
| Archived | 40,960 | 4.00 | 1.25 | 1.022314355 |
| Corrected | 131,072 | 4.00 | 4.00 | 1.138629436 |

The [frequency regression check](qwen-long-yarnfix-64k-v1-rope-check-65536-rope-check.json) evaluates the installed Transformers implementation directly.

| Configuration | max_position_embeddings | Declared factor | Effective factor | Attention scaling |
|---|---:|---:|---:|---:|
| Archived | 40,960 | 2.00 | 1.25 | 1.022314355 |
| Corrected | 65,536 | 2.00 | 2.00 | 1.069314718 |

The independent [Transformers reference](qwen-long-yarnfix-64k-v1-reference-65536-reference.json) uses the same YaRN and prompts with a 65,536-token context limit.

| Task | Source row | Input tokens | RetroInfer dense score | Transformers dense score | First three tokens agree |
|---|---:|---:|---:|---:|---|
| niah_multikey_2 | 0 | 65,006 | 100% | 100% | True |
| niah_multikey_3 | 0 | 64,625 | 100% | 100% | False |

The independent [Transformers reference](qwen-long-yarnfix-v2-reference-128000-reference.json) uses the same YaRN and prompts with a 131,072-token context limit.

| Task | Source row | Input tokens | RetroInfer dense score | Transformers dense score | First three tokens agree |
|---|---:|---:|---:|---:|---|
| niah_multikey_2 | 0 | 127,624 | 100% | 100% | True |
| niah_multikey_3 | 0 | 127,436 | 0% | 0% | True |

## Full 32K RULER panel

Qwen3-8B BF16 with native RoPE and thinking disabled. All three decoders share one optimized LoD prefill per prompt. The cohort has 32 examples per task. The mean weights the 13 tasks equally.

Measured prompt time sums to 88.6 minutes on one H100. [Generations](qwen-long-ruler13-32k-quality-n32-v6-shared-32768-shared.json), [frozen cohort](qwen-long-ruler13-32k-n32-v2-cohort-32768-prepare.json).

Scoring stops at the first EOS/EOT. Generation checks for EOS/EOT every eight tokens after the initial fixed-length segment. All runs use the same task caps.

The run preserved 164 prompts from the [earlier checkpoint](qwen-long-ruler13-32k-quality-n32-v5-shared-32768-shared.json).

After the retry, the shared prefill used 4096-query workspace tiles. The context limit, exact prefill field, and routing policy were the same throughout the panel.

| RULER task | Threshold LoD | Dense | Multipole 1:16, 512 |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 96.88% | 100.00% | 100.00% |
| niah_multikey_2 | 90.62% | 93.75% | 93.75% |
| niah_multikey_3 | 100.00% | 100.00% | 96.88% |
| niah_multivalue | 97.66% | 97.66% | 98.44% |
| niah_multiquery | 98.44% | 96.88% | 96.09% |
| vt | 99.38% | 99.38% | 99.38% |
| fwe | 94.79% | 92.71% | 93.75% |
| cwe | 77.50% | 78.75% | 83.12% |
| qa_1 | 59.38% | 62.50% | 62.50% |
| qa_2 | 46.88% | 46.88% | 46.88% |
| **13-task mean** | **89.35%** | **89.88%** | **90.06%** |

Whole-model decode includes periodic cache updates. [Matched 32K timing data](qwen-long-ruler13-32k-decode-v6-shared-32768-shared.json).

| Decoder | Median ms/token |
|---|---:|
| Threshold LoD | 7.836 |
| Dense | 9.254 |
| Multipole 1:16, 512 | 18.939 |


## 32K RULER with a shared 2,900-centroid index

Qwen3-8B BF16 on H100, with 32 prompts per task and batch-one decoding. All four decoders share one optimized LoD prefill per prompt. Multipole uses a 512-token exact budget, value-centroid replacement, and 128-token online refinement at fixed centroid count. Squeezed p90 reuses the same initial Multipole centroids and assignments, calibrates on the final 100 prefill queries at routing position 2,048, and holds its prefix index fixed. The sink, unclustered prefix tail, and generated tokens are exact. [Generations](qwen-long-ruler13-32k-multipole-p90-k2900-n32-v9-shared-32768-shared.json), [frozen cohort](qwen-long-ruler13-32k-n32-v2-cohort-32768-prepare.json).

The final run resumed from the [296-prompt checkpoint](qwen-long-ruler13-32k-multipole-p90-k2900-n32-v8-shared-32768-shared.json). Only fully scored prompts carried forward.

| RULER task | Examples | Threshold LoD | Dense | Multipole, 512 | Squeezed p90 |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 32 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 32 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 32 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 32 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 32 | 90.62% | 93.75% | 93.75% | 93.75% |
| niah_multikey_3 | 32 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 32 | 98.44% | 97.66% | 98.44% | 99.22% |
| niah_multiquery | 32 | 98.44% | 96.88% | 96.88% | 97.66% |
| vt | 32 | 99.38% | 99.38% | 99.38% | 99.38% |
| fwe | 32 | 93.75% | 92.71% | 92.71% | 92.71% |
| cwe | 32 | 77.81% | 78.75% | 85.31% | 84.06% |
| qa_1 | 32 | 59.38% | 62.50% | 62.50% | 59.38% |
| qa_2 | 32 | 46.88% | 46.88% | 46.88% | 46.88% |
| **13-task mean** | **416** | **89.59%** | **89.88%** | **90.45%** | **90.23%** |

The panel took 63.0 minutes of prompt processing. Median prompt time was 8.12 s, including 1.99 s for shared prefill, 1.85 s for Multipole cache setup and fitting, and 0.40 s for p90 calibration.


## 64K comparison on the same 128 prompts

LoD and dense share a prefill execution within each prompt. Multipole recomputes the same optimized LoD prefill recipe in a separate run. Prompt tokens, task labels, and the 2× YaRN configuration match. Multipole uses the author’s 1:16 clustering, centroid replacement, and 512-token exact budget.

The independently executed prefills produced the same first token on 124 of 128 prompts.

[LoD and dense generations](qwen-long-lod-dense-quality64-n32-resumed-v1-shared-65536-shared.json), [Multipole generations](qwen-long-multipole-quality64-n32-ipc-v1-shared-65536-shared.json).

The LoD and dense run reused 98 prompts from the [pre-restart checkpoint](qwen-long-lod-dense-quality64-n32-resident-v1-shared-65536-shared.json). The replacement H100 finished that panel and ran all Multipole prompts.

All decode latencies come from a separate matched three-way run on the replacement H100, including periodic updates. [Timing data](qwen-long-multipole-shared-weights-decode64-v1-shared-65536-shared.json).

| Method | Examples/task | Multikey-2 | Multikey-3 | CWE | QA-1 | Four-task mean | Decode ms/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Threshold LoD | 32 | 75.00% | 56.25% | 62.81% | 59.38% | 63.36% | 8.195 |
| Dense | 32 | 78.12% | 59.38% | 64.69% | 56.25% | 64.61% | 10.777 |
| Multipole 1:16, 512 | 32 | 78.12% | 37.50% | 67.19% | 53.12% | 58.98% | 21.214 |


## Same-prefill decode quality

All rows within a panel start with the same optimized LoD prefill, original per-layer activations, and first output token. Dense decode therefore diagnoses what the shared prefill retained. The released Squeezed port uses its cuML fit, global percentile calibration, and released lookup, sparse attention, and reduction kernels with shared GQA KV. Each physical KV head is fitted once, then its centers and labels are shared across its query heads. The port uses deterministic two-pass packing of the same selected set for graph/eager parity. Multipole uses one centroid per 16 tokens, centroid replacement, and a 512-token exact budget. The Multipole authors’ Squeezed subset uses 1:8 clustering and a token budget. It uses the authors’ windowed routing, with centroid replacement disabled. Each completed arm passes an exact graph/eager logit check at the same cache state.

[qwen-long-released-quality32-v2-shared-32768-shared](qwen-long-released-quality32-v2-shared-32768-shared.json), context limit 32,768. Actual input lengths 31,147–32,497.

| Method | Examples/task | Multikey-2 | Multikey-3 | CWE | QA-1 | Four-task mean | Decode ms/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Threshold LoD | 2 | 50.00% | 100.00% | 70.00% | 50.00% | 67.50% | 7.802 |
| Dense | 2 | 50.00% | 100.00% | 75.00% | 50.00% | 68.75% | 9.284 |
| Released Squeezed port, p90 | 2 | 50.00% | 100.00% | 80.00% | 50.00% | 70.00% | 10.688 |
| Released Squeezed port, p80 | 2 | 50.00% | 100.00% | 85.00% | 50.00% | 71.25% | 10.866 |

[qwen-long-released-quality64-v1-shared-65536-shared](qwen-long-released-quality64-v1-shared-65536-shared.json), context limit 65,536. Actual input lengths 63,538–65,240.

| Method | Examples/task | Multikey-2 | Multikey-3 | CWE | QA-1 | Four-task mean | Decode ms/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Threshold LoD | 2 | 50.00% | 50.00% | 65.00% | 50.00% | 53.75% | 8.187 |
| Dense | 2 | 50.00% | 50.00% | 80.00% | 50.00% | 57.50% | 10.812 |
| Released Squeezed port, p90 | 2 | 50.00% | 50.00% | 85.00% | 50.00% | 58.75% | 12.150 |
| Released Squeezed port, p80 | 2 | 50.00% | 50.00% | 85.00% | 50.00% | 58.75% | 12.536 |

Full dense prefill and decode on these same prompts scores niah_multikey_2 50.00%, niah_multikey_3 100.00%, cwe 80.00%, qa_1 50.00%. [Reference generations](qwen-long-yarnfix-64k-v1-dense-65536-quality.json).

[qwen-long-multipole-quality64-pilot-v1-shared-65536-shared](qwen-long-multipole-quality64-pilot-v1-shared-65536-shared.json), context limit 65,536. Actual input lengths 65,003–65,006.

| Method | Examples/task | Multikey-2 | Multikey-3 | CWE | QA-1 | Four-task mean | Decode ms/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Threshold LoD | 2 | 50.00% | — | — | — | — | 7.998 |
| Dense | 2 | 50.00% | — | — | — | — | 10.516 |
| Multipole 1:16, 512 | 2 | 0.00% | — | — | — | — | 21.607 |

[qwen-long-yarnfix-quality-v2-shared-128000-shared](qwen-long-yarnfix-quality-v2-shared-128000-shared.json), context limit 128,000. Actual input lengths 124,179–127,639.

| Method | Examples/task | Multikey-2 | Multikey-3 | CWE | QA-1 | Four-task mean | Decode ms/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Threshold LoD | 2 | 100.00% | 0.00% | 15.00% | 50.00% | 41.25% | 8.696 |
| Dense | 2 | 100.00% | 0.00% | 30.00% | 50.00% | 45.00% | 13.849 |
| Squeezed subset 1:8, 512 | 2 | 100.00% | 0.00% | 20.00% | 50.00% | 42.50% | 34.081 |
| Squeezed subset 1:8, 2,048 | 2 | 100.00% | 0.00% | 10.00% | 50.00% | 40.00% | 33.938 |

Full dense prefill and decode on these same prompts scores niah_multikey_2 100.00%, niah_multikey_3 0.00%, cwe 25.00%, qa_1 50.00%. [Reference generations](qwen-long-yarnfix-v1-dense-128000-quality.json).

<a id="quality-run-wall-time-at-64k"></a>

## Quality-run wall time

These short profiles use frozen RULER prompts with shared LoD prefill. Scoring stops at the first EOS/EOT. The run commands record the generation limits. The full prompt time includes cache setup, activation handoff, and graph checks. It excludes loading the main model once and cleanup after saving the prompt.

The GPU-resident path assembles native dense KV from the LoD sink, chronological BF16 archive, and live exact tail. The verification profile compares every layer with the original model activations. These short profiles measure runtime.

### Multipole worker transport

CUDA IPC shares GPU allocations across the two Triton runtimes. The worker uses the same compiled author clustering, attention kernels and 128-token updates. The prompt time covers shared prefill and all three decoders. The Multipole arm includes worker launch, index construction, graph validation and generation.

| KV transfer | Worker weights | Prompts | Median prompt (s) | CPU KV export (s) | Multipole arm (s) | Worker setup (s) | Index setup (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| [CPU and tmpfs](qwen-long-multipole-host-profile64-v2-shared-65536-shared.json) | Loaded per prompt | 2 | 42.27 | 6.89 | 26.35 | 3.16 | 6.54 |
| [CUDA IPC](qwen-long-multipole-ipc-profile64-v2-shared-65536-shared.json) | Loaded per prompt | 2 | 25.04 | 0.00 | 15.89 | 3.24 | 4.61 |
| [CUDA IPC](qwen-long-multipole-shared-weights-profile64-v1-shared-65536-shared.json) | Shared on GPU | 2 | 20.84 | 0.00 | 11.70 | 0.14 | 4.57 |

65,536-token limit. [Dense KV via CPU](qwen-long-lod-dense-profile64-v1-shared-65536-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 18.42 | 13.55 | — | — | 1.68 | 2.36 |
| niah_multikey_3 | 0 | 16.91 | 12.44 | — | — | 1.68 | 2.32 |
| cwe | 0 | 17.08 | 12.73 | — | — | 1.72 | 2.18 |
| qa_1 | 0 | 15.46 | 12.72 | — | — | 1.72 | 0.55 |

65,536-token limit. [Dense KV via CPU](qwen-long-multipole-profile64-v1-shared-65536-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 53.28 | 17.32 | — | — | 1.72 | 5.33 |
| niah_multikey_2 | 1 | 51.99 | 16.39 | — | — | 1.70 | 5.31 |

| Source row | Multipole arm (s) | Worker model setup (s) | Index setup (s) | Graph and check (s) | Generation (s) | Handoff and other overhead (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31.06 | 4.04 | 7.79 | 0.58 | 2.98 | 15.67 |
| 1 | 31.13 | 4.20 | 7.74 | 0.58 | 2.98 | 15.63 |

65,536-token limit. [Dense KV via CPU](qwen-long-lod-dense-profile64-v2-shared-65536-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 18.50 | 13.59 | 7.32 | 4.61 | 1.71 | 2.35 |
| niah_multikey_2 | 1 | 17.21 | 12.70 | 7.32 | 3.91 | 1.69 | 2.36 |

65,536-token limit. [GPU-resident dense KV checked against the original activations](qwen-long-dense-resident64-verify-v1-shared-65536-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 18.62 | 13.61 | 7.36 | 4.61 | 1.75 | 2.36 |
| niah_multikey_3 | 0 | 17.03 | 12.52 | 7.36 | 3.69 | 1.72 | 2.32 |
| cwe | 0 | 17.26 | 12.83 | 7.41 | 3.94 | 1.73 | 2.19 |
| qa_1 | 0 | 15.62 | 12.76 | 7.36 | 3.93 | 1.72 | 0.55 |

65,536-token limit. [GPU-resident dense KV](qwen-long-dense-resident64-device-v1-shared-65536-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 9.93 | 6.38 | 0.00 | 4.72 | 0.35 | 2.36 |
| niah_multikey_3 | 0 | 8.41 | 5.27 | 0.00 | 3.80 | 0.37 | 2.32 |
| cwe | 0 | 8.48 | 5.50 | 0.00 | 4.02 | 0.35 | 2.18 |
| qa_1 | 0 | 6.73 | 5.39 | 0.00 | 3.91 | 0.35 | 0.55 |

65,536-token limit. [Dense KV via CPU](qwen-long-dense-resident64-host-v1-shared-65536-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 18.60 | 13.68 | 7.42 | 4.61 | 1.73 | 2.35 |
| niah_multikey_3 | 0 | 17.02 | 12.52 | 7.36 | 3.68 | 1.72 | 2.32 |
| cwe | 0 | 17.22 | 12.84 | 7.42 | 3.94 | 1.73 | 2.18 |
| qa_1 | 0 | 15.35 | 12.62 | 7.32 | 3.83 | 1.69 | 0.55 |

65,536-token limit. [GPU-resident dense KV](qwen-long-multipole-ipc-profile64-v2-shared-65536-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 26.55 | 6.31 | 0.00 | 4.65 | 0.30 | 5.84 |
| niah_multikey_2 | 1 | 23.53 | 5.51 | 0.00 | 4.01 | 0.32 | 5.30 |

| Source row | Multipole arm (s) | Worker model setup (s) | Index setup (s) | Graph and check (s) | Generation (s) | Handoff and other overhead (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16.82 | 3.18 | 4.57 | 2.03 | 3.43 | 3.60 |
| 1 | 14.96 | 3.30 | 4.64 | 0.51 | 2.91 | 3.60 |

65,536-token limit. [GPU-resident dense KV](qwen-long-multipole-host-profile64-v2-shared-65536-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 42.91 | 13.09 | 6.89 | 4.54 | 0.30 | 5.32 |
| niah_multikey_2 | 1 | 41.63 | 12.27 | 6.88 | 3.90 | 0.29 | 5.29 |

| Source row | Multipole arm (s) | Worker model setup (s) | Index setup (s) | Graph and check (s) | Generation (s) | Handoff and other overhead (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 26.37 | 3.15 | 6.58 | 0.49 | 2.91 | 13.24 |
| 1 | 26.33 | 3.16 | 6.50 | 0.50 | 2.91 | 13.25 |

65,536-token limit. [GPU-resident dense KV](qwen-long-multipole-shared-weights-profile64-v1-shared-65536-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 21.45 | 6.30 | 0.00 | 4.64 | 0.31 | 5.32 |
| niah_multikey_2 | 1 | 20.24 | 5.51 | 0.00 | 4.00 | 0.31 | 5.30 |

| Source row | Multipole arm (s) | Worker model setup (s) | Index setup (s) | Graph and check (s) | Generation (s) | Handoff and other overhead (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11.70 | 0.14 | 4.56 | 0.50 | 2.91 | 3.60 |
| 1 | 11.70 | 0.14 | 4.57 | 0.50 | 2.91 | 3.57 |

32,768-token limit. [GPU-resident dense KV](qwen-long-squeezed-online-profile32-v1-shared-32768-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 21.64 | 4.94 | 0.00 | 3.92 | 0.21 | 0.54 |
| niah_multikey_3 | 0 | 16.06 | 3.02 | 0.00 | 2.27 | 0.21 | 3.21 |
| cwe | 0 | 13.01 | 2.00 | 0.00 | 1.26 | 0.21 | 1.59 |
| qa_1 | 0 | 9.81 | 1.92 | 0.00 | 1.21 | 0.21 | 0.21 |

32,768-token limit. [Quality runtime](qwen-long-squeezed-online-alone-profile32-v1-shared-32768-shared.json)

| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 0 | 11.05 | 2.74 | 0.00 | 1.84 | — | 0.25 |
| niah_multikey_3 | 0 | 10.85 | 1.94 | 0.00 | 1.19 | — | 1.20 |
| cwe | 0 | 10.37 | 1.96 | 0.00 | 1.22 | — | 0.81 |
| qa_1 | 0 | 9.49 | 1.87 | 0.00 | 1.15 | — | 0.10 |

## Decode-only speed

Full-model graph replay on one shared real-text prefill per context, with identical teacher-forced tokens. Timed windows include per-step KV writes, host launch and required ongoing cache/index updates. Prefill, initial index construction, cache allocation, graph capture, and warmup are excluded. Consecutive windows are not independent prefills. Quality is scored on the frozen RULER prompts. The timing workload uses a separate real-text prefix.

### Matched centroid count at 32K

Qwen3-8B BF16, batch one, native RoPE on one H100. The methods share a 30,720-token LoD prefill, then decode 512 warmup and 1,536 timed tokens. [Timing data](qwen-long-matched-centroids-decode32-v1-shared-32768-shared.json).

The two Multipole-based methods use 2,900 centroid slots per KV head and a 512-token exact budget per KV head. Both refine their final clustering block every 128 tokens with the released sequential K-means kernel and zero new centroid seeds. Multipole also uses value-centroid replacement for the unselected clusters. LoD keeps its production centroid schedule.

| Method | Centroid slots per KV head | Median ms/token | Window ms/token | Index updates (ms/token) |
|---|---:|---:|---|---:|
| Threshold LoD | 2,816–2,884 | 7.784 | 7.762, 7.784, 7.810 | — |
| Dense | — | 9.251 | 9.237, 9.251, 9.270 | — |
| Squeezed with Multipole clustering | 2,900 | 17.859 | 17.850, 17.859, 17.944 | 1.583 |
| Multipole | 2,900 | 18.734 | 18.719, 18.734, 18.777 | 0.849 |

Centroid counts are from the last layer at the measured window boundaries. Total latency includes all required cache updates. The update column shows the separately instrumented time for the two Multipole-based methods.

### Kernel swap at matched centroid count

Both subset arms start from the same 2,900-centroid index and use the same grouped 512-token selection and 128-token update interval. The swap uses the released Squeezed lookup, sparse attention, and reduction with shared head addressing. Its per-query-head scores are summed before the common budget sort and index packing.

| First kernel set | Kernels | Median ms/token | Index updates (ms/token) | Decode excluding profiled updates (ms/token) |
|---|---|---:|---:|---:|
| [Multipole](qwen-long-matched-centroids-squeezed-kernels-decode32-v1-shared-32768-shared.json) | Multipole | 17.855 | 1.567 | 16.282 |
| [Multipole](qwen-long-matched-centroids-squeezed-kernels-decode32-v1-shared-32768-shared.json) | Squeezed | 16.890 | 0.771 | 16.120 |
| [Squeezed](qwen-long-matched-centroids-squeezed-kernels-decode32-reverse-v1-shared-32768-shared.json) | Squeezed | 17.727 | 1.586 | 16.142 |
| [Squeezed](qwen-long-matched-centroids-squeezed-kernels-decode32-reverse-v1-shared-32768-shared.json) | Multipole | 17.003 | 0.766 | 16.239 |

The final-layer diagnostic replays each phase in a separate CUDA graph. For the alignment check, lookup uses the first 2,816 slots or pads the 2,900-slot index with empty entries to reach 2,944.

| First kernel set | Kernels | Lookup at 2,816 (µs) | Lookup at 2,900 (µs) | Lookup at 2,944 (µs) | Selection and pack (µs) | Sparse attention and merge (µs) |
|---|---|---:|---:|---:|---:|---:|
| Multipole | Multipole | 75.47 | 82.87 | 75.40 | 108.47 | 12.24 |
| Multipole | Squeezed | 77.71 | 79.33 | 79.02 | 109.43 | 12.64 |
| Squeezed | Squeezed | 77.71 | 77.98 | 77.63 | 111.09 | 12.73 |
| Squeezed | Multipole | 75.44 | 82.32 | 75.62 | 111.63 | 12.33 |

Both lookups selected identical tokens on each profiled final-layer input.

### Released Squeezed centroid count at 32K

Qwen3-8B BF16 at batch one on H100, using shared GQA centroid storage. Both indexes use the same 30,720-token prefill and 512 warmup and 1,536 timed teacher-forced decode tokens. Each index has its own p90 threshold fitted on the same 100 observation tokens. [Timing data](qwen-long-squeezed-p90-centroid-count-decode32-v1-shared-32768-shared.json).

| Centroids per KV head | Decode ms/token | Window ms/token | Lookup only (ms/token) | Mean exact prefix tokens per query head | Build and calibrate (s) |
|---|---:|---|---:|---:|---:|
| 1,531 | 10.577 | 10.544, 10.577, 10.636 | 0.979 | 2,656 | 60.2 |
| 2,900 | 11.387 | 11.348, 11.387, 11.445 | 1.753 | 2,427 | 107.7 |

The exact prefix count averages heads and layers at the end of each timed window. The lookup-only diagnostic times one graph-captured centroid lookup per layer. Both widths use the same observation queries, with the smaller width taking the first 1,531 centroids of the larger index. Build and calibration are outside decode timing.

### Squeezed with Multipole clustering at 32K

The online variant uses Multipole’s block clustering at one centroid per eight tokens, a 512-token exact budget, and centroid replacement disabled. It updates the index every 128 tokens. The released port uses cuML clustering and percentile calibration to select from a fixed prefix index.

| Decoder | Initial index and cache setup (s) | Decode ms/token |
|---|---:|---:|
| [Squeezed subset 1:8, 512](qwen-long-squeezed-online-decode32-v1-shared-32768-shared.json) | 3.18 | 19.559 |
| [Released Squeezed port, p90](qwen-long-squeezed-calibrated-decode32-matched-v1-shared-32768-shared.json) | 64.01 | 10.623 |

The online setup measurement includes cache installation and clustering. The released-port setup measurement includes clustering and calibration.

Online index updates account for 1.96 ms/token. The remaining decode step takes 17.60 ms/token.

Online Squeezed alone took a median 10.61 seconds per prompt across 4 RULER pilot prompts. Median online setup was 3.36 seconds. [Pilot records](qwen-long-squeezed-online-alone-profile32-v1-shared-32768-shared.json).

LoD, dense, and online Squeezed together took a median 14.53 seconds per prompt across 4 RULER pilot prompts. Median online setup was 3.39 seconds. [Pilot records](qwen-long-squeezed-online-profile32-v1-shared-32768-shared.json).

### Matched 64K whole-model decode

Each run prefills 63,488 tokens, warms up for 512 decode steps, and measures 1,536 teacher-forced steps on one H100. [LoD, dense, and Multipole timing data](qwen-long-multipole-shared-weights-decode64-v1-shared-65536-shared.json).

The Squeezed run used the same prompt, teacher-forced tokens, and prefill recipe. Its LoD and dense controls measured 8.20 and 10.80 ms/token. [Squeezed timing data](qwen-long-squeezed-decode64-matched-ipc-v1-shared-65536-shared.json).

| Decoder | Median ms/token | Tokens/s | 32-per-task quality panel |
|---|---:|---:|---|
| Threshold LoD | 8.195 | 122.0 | Complete |
| Dense | 10.777 | 92.8 | Complete |
| Multipole 1:16, 512 | 21.214 | 47.1 | Complete |
| Released Squeezed port, p90 | 12.197 | 82.0 | Speed only |

### Initial index setup outside the decode timer

| Method | Seconds | Measured work |
|---|---:|---|
| Multipole | 3.90 | Cache installation and initial clustering |
| Released Squeezed port | 192.24 | Initial clustering and percentile calibration |

LoD builds its region state during the common prefill.

32,768 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. GPU GPU-e990762e-21ba-1072-8311-be210c4e4762. [Timing data](qwen-long-released-decode32-v2-shared-32768-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 7.802 | 7.833, 7.802, 7.773 | 72, 72, 72 |
| Dense | 9.284 | 9.260, 9.284, 9.300 | 0, 0, 0 |
| Released Squeezed port, p90 | 10.688 | 10.681, 10.689, 10.688 | 0, 0, 0 |
| Released Squeezed port, p80 | 10.866 | 10.866, 10.882, 10.859 | 0, 0, 0 |

The released port built 1,633 centroids per physical KV head with a 100-token observation window. Index build and calibration took 65.7 seconds outside the decode timer.

30,720 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 512 decode steps. GPU GPU-b3722181-849c-94e5-0f28-ab4d273c96ce. [Timing data](qwen-long-ruler13-32k-decode-v6-shared-32768-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 7.836 | 7.836, 7.815, 7.848 | 72, 72, 72 |
| Dense | 9.254 | 9.239, 9.254, 9.275 | 0, 0, 0 |
| Multipole 1:16, 512 | 18.939 | 18.939, 18.805, 32.056 | 4, 4, 4 |

Multipole’s profiled online updates account for 2.173 ms/token. The remaining whole-model decode time is 16.854 ms/token.

30,720 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 512 decode steps. GPU GPU-b3722181-849c-94e5-0f28-ab4d273c96ce. [Timing data](qwen-long-squeezed-online-decode32-v1-shared-32768-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 7.816 | 7.816, 7.810, 7.860 | 72, 72, 72 |
| Dense | 9.254 | 9.238, 9.254, 9.274 | 0, 0, 0 |
| Squeezed subset 1:8, 512 | 19.559 | 19.559, 19.546, 19.616 | 4, 4, 4 |

30,720 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 512 decode steps. GPU GPU-b3722181-849c-94e5-0f28-ab4d273c96ce. [Timing data](qwen-long-squeezed-calibrated-decode32-matched-v1-shared-32768-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 7.801 | 7.801, 7.799, 7.835 | 72, 72, 72 |
| Dense | 9.237 | 9.222, 9.237, 9.256 | 0, 0, 0 |
| Released Squeezed port, p90 | 10.623 | 10.584, 10.623, 10.679 | 0, 0, 0 |

The released port built 1,531 centroids per physical KV head with a 100-token observation window. Index build and calibration took 64.0 seconds outside the decode timer.

63,992 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 8 decode steps. GPU GPU-e990762e-21ba-1072-8311-be210c4e4762. [Timing data](qwen-long-released-decode64-v1-shared-65536-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 8.187 | 8.374, 8.161, 8.187 | 72, 72, 72 |
| Dense | 10.812 | 10.793, 10.812, 10.834 | 0, 0, 0 |
| Released Squeezed port, p90 | 12.150 | 12.145, 12.150, 12.164 | 0, 0, 0 |
| Released Squeezed port, p80 | 12.536 | 12.504, 12.536, 12.586 | 0, 0, 0 |

The released port built 3,194 centroids per physical KV head with a 100-token observation window. Index build and calibration took 192.5 seconds outside the decode timer.

63,992 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 8 decode steps. GPU GPU-9401dbfd-51b8-9bb8-d0d2-6b7008ad85c6. [Timing data](qwen-long-released-decode64-n32-v1-shared-65536-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 8.023 | 8.080, 7.985, 8.023 | 72, 72, 72 |
| Dense | 10.516 | 10.493, 10.516, 10.540 | 0, 0, 0 |
| Released Squeezed port, p90 | 11.920 | 11.906, 11.920, 11.952 | 0, 0, 0 |
| Released Squeezed port, p80 | 12.381 | 12.350, 12.381, 12.440 | 0, 0, 0 |

The released port built 3,194 centroids per physical KV head with a 100-token observation window. Index build and calibration took 206.4 seconds outside the decode timer.

63,488 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 512 decode steps. GPU GPU-9401dbfd-51b8-9bb8-d0d2-6b7008ad85c6. [Timing data](qwen-long-multipole-decode64-v1-shared-65536-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 7.998 | 7.983, 7.998, 8.001 | 72, 72, 72 |
| Dense | 10.516 | 10.493, 10.516, 10.538 | 0, 0, 0 |
| Multipole 1:16, 512 | 21.607 | 21.497, 21.607, 21.801 | 4, 4, 4 |

63,488 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 512 decode steps. GPU GPU-9401dbfd-51b8-9bb8-d0d2-6b7008ad85c6. [Timing data](qwen-long-dense-resident64-decode-v2-shared-65536-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 8.000 | 7.999, 8.000, 8.014 | 72, 72, 72 |
| Dense | 10.513 | 10.491, 10.513, 10.535 | 0, 0, 0 |

63,488 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 512 decode steps. GPU GPU-b3722181-849c-94e5-0f28-ab4d273c96ce. [Timing data](qwen-long-lod-dense-decode64-resumed-host-v1-shared-65536-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 8.174 | 8.174, 8.169, 8.204 | 72, 72, 72 |
| Dense | 10.803 | 10.784, 10.803, 10.825 | 0, 0, 0 |

63,488 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 512 decode steps. GPU GPU-b3722181-849c-94e5-0f28-ab4d273c96ce. [Timing data](qwen-long-multipole-cost64-v1-shared-65536-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 8.171 | 8.169, 8.171, 8.215 | 72, 72, 72 |
| Dense | 10.799 | 10.780, 10.799, 10.821 | 0, 0, 0 |
| Multipole 1:16, 512 | 21.395 | 21.311, 21.441, 21.395 | 4, 4, 4 |

Multipole’s profiled online updates account for 2.096 ms/token. The remaining whole-model decode time is 19.291 ms/token.

63,488 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 512 decode steps. GPU GPU-b3722181-849c-94e5-0f28-ab4d273c96ce. [Timing data](qwen-long-multipole-shared-weights-decode64-v1-shared-65536-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 8.195 | 8.195, 8.178, 8.226 | 72, 72, 72 |
| Dense | 10.777 | 10.758, 10.777, 10.802 | 0, 0, 0 |
| Multipole 1:16, 512 | 21.214 | 21.139, 21.214, 21.233 | 4, 4, 4 |

Multipole’s profiled online updates account for 1.946 ms/token. The remaining whole-model decode time is 19.276 ms/token.

63,488 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. Warmup uses 512 decode steps. GPU GPU-b3722181-849c-94e5-0f28-ab4d273c96ce. [Timing data](qwen-long-squeezed-decode64-matched-ipc-v1-shared-65536-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 8.196 | 8.185, 8.196, 8.231 | 72, 72, 72 |
| Dense | 10.802 | 10.782, 10.802, 10.823 | 0, 0, 0 |
| Released Squeezed port, p90 | 12.197 | 12.177, 12.197, 12.252 | 0, 0, 0 |

The released port built 3,169 centroids per physical KV head with a 100-token observation window. Index build and calibration took 192.2 seconds outside the decode timer.

128,000 prefix tokens. The 3 consecutive windows measure 512, 512, 512 decode steps. GPU GPU-e990762e-21ba-1072-8311-be210c4e4762. [Timing data](qwen-long-yarnfix-decode-v2-shared-128000-shared.json).

| Method | Median ms/token | Window ms/token | Update events by window |
|---|---:|---|---|
| Threshold LoD | 8.696 | 8.666, 8.699, 8.696 | 72, 72, 72 |
| Dense | 13.849 | 13.831, 13.849, 13.870 | 0, 0, 0 |
| Squeezed subset 1:8, 512 | 34.081 | 34.054, 34.081, 34.184 | 4, 4, 4 |
| Squeezed subset 1:8, 2,048 | 33.938 | 33.894, 33.938, 33.999 | 4, 4, 4 |

LoD counts cache catch-up events per layer. Multipole and its Squeezed subset count index-update boundaries for the whole model. Calibrated Squeezed keeps the prefix index fixed and attends exactly to generated tokens. The percentiles set score thresholds calibrated on the observation window. The subset runs its compiled clustering and decode graph in its compatible Triton environment, using the exact saved prefill activations. Each subset budget builds its own initial index using the worker’s seed-0 RNG stream. The one-time host handoff is outside the decode window.

## Validation and archived attempts

The archive contains 5 completed model runs with the incorrect long-context RoPE configuration. SUMMARY.json retains their files and exclusion reasons. The prior 64K/128K prefill and 128K real-state graph reports also require corrected reruns.

The released port’s original atomic compaction produced graph/eager logit differences. Deterministic packing retains the same selected entries and gives exact graph/eager logits in the reported runs. The packing check also matches PyTorch selected indices at 4K, 32K, and 128K.

Earlier setup attempts encountered missing generator dependencies, incompatible Inductor/Triton versions, and an unavailable offline wheel requirement. The completed reruns use the installed compatible clustering environment and retain the authors’ compilation.

The run archive has 0 pending, 24 failed, 1 interrupted, and 18 cancelled jobs. Tables use completed panels. See [SUMMARY.json](SUMMARY.json) and provenance/ for hashes, commands and logs.

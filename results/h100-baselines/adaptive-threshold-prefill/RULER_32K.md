# Llama 3.1 32K RULER with threshold prefill and adaptive three-tier decode

Llama-3.1-8B-Instruct with BF16 weights and KV. Each method uses the same 416 tokenized inputs across 13 tasks. Scores stop at the first EOS/EOT and weight each task equally.

The [dense-prefill Multipole and Squeezed results](../llama-context-scaling/DENSE_PREFILL.md#32k-ruler) use this same cohort and match the dense BF16 first token on all 416 prompts.

The new LoD run uses c=16 per-region prefill with exact attention to the selected regions. The BF16 cache feeds c=16 region routing and c=0.5 RMS-normalized page routing during decode. The run uses the optimized threshold-prefill settings from the [two-tier run](../direct-opt-v3/RULER_32K.md) and the adaptive decode settings from the [dense and top-eight prefill controls](../clusterkv/REPORT.md#32k-ruler-quality).

| Task | Dense BF16 | RetroInfer BF16 | Two-tier, threshold prefill | Adaptive three-tier, top-eight prefill | Adaptive three-tier, dense prefill | Adaptive three-tier, threshold prefill |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 96.88% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 96.88% | 100.00% | 96.88% | 100.00% |
| niah_multikey_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 96.88% | 98.44% | 99.22% | 99.22% | 96.88% | 99.22% |
| niah_multiquery | 99.22% | 98.44% | 98.44% | 97.66% | 95.31% | 95.31% |
| vt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| fwe | 90.62% | 86.46% | 88.54% | 88.54% | 88.54% | 87.50% |
| cwe | 37.50% | 36.25% | 35.31% | 27.50% | 28.12% | 35.00% |
| qa_1 | 81.25% | 81.25% | 78.12% | 68.75% | 81.25% | 81.25% |
| qa_2 | 50.00% | 50.00% | 46.88% | 46.88% | 50.00% | 46.88% |
| NIAH mean | 99.51% | 99.61% | 99.32% | 99.61% | 98.24% | 99.32% |
| **13-task mean** | **88.88%** | **88.53%** | **87.95%** | **86.81%** | **87.22%** | **88.09%** |

- Versus Adaptive three-tier, top-eight prefill, the new run changes the mean by +1.28 points (paired 95% interval -0.03 to +2.65).
- Versus Adaptive three-tier, dense prefill, the new run changes the mean by +0.87 points (paired 95% interval -0.32 to +2.10).
- Versus Two-tier, threshold prefill, the new run changes the mean by +0.14 points (paired 95% interval -0.76 to +1.08).
- Versus RetroInfer BF16, the new run changes the mean by -0.44 points (paired 95% interval -1.95 to +1.03).
- Versus Dense BF16, the new run changes the mean by -0.79 points (paired 95% interval -2.22 to +0.57).

## Verification

The prefill audit checked 32,768 head-query pairs on two prompts and found zero selections outside the threshold rounding tolerance. The maximum absolute prefill output difference was 0.015625. The full panel matched graph and eager decode logits for the sampled step on all 13 tasks. Three-tier and two-tier threshold prefill gave the same first token on 414 of 416 prompts.

[Frozen cohort](../quality-ruler-32k-frozen-13x32.json.gz), SHA-256 `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.

- [llama32k-adaptive-threshold-prefill-smoke.json](llama32k-adaptive-threshold-prefill-smoke.json) · [Run command](provenance/1794790263501767278-llama32k-adaptive-threshold-prefill-smoke/run.json) · [Log](provenance/1794790263501767278-llama32k-adaptive-threshold-prefill-smoke/output.log).
- [llama32k-adaptive-threshold-prefill-ruler13-n32-gpu0.json](llama32k-adaptive-threshold-prefill-ruler13-n32-gpu0.json) · [Run command](provenance/1794790640789765462-llama32k-adaptive-threshold-prefill-ruler13-n32-gpu0/run.json) · [Log](provenance/1794790640789765462-llama32k-adaptive-threshold-prefill-ruler13-n32-gpu0/output.log).
- [llama32k-adaptive-threshold-prefill-ruler13-n32-gpu2.json](llama32k-adaptive-threshold-prefill-ruler13-n32-gpu2.json) · [Run command](provenance/1794790640698123268-llama32k-adaptive-threshold-prefill-ruler13-n32-gpu2/run.json) · [Log](provenance/1794790640698123268-llama32k-adaptive-threshold-prefill-ruler13-n32-gpu2/output.log).
- [llama32k-adaptive-threshold-prefill-ruler13-n32-gpu3.json](llama32k-adaptive-threshold-prefill-ruler13-n32-gpu3.json) · [Run command](provenance/1794790640844108157-llama32k-adaptive-threshold-prefill-ruler13-n32-gpu3/run.json) · [Log](provenance/1794790640844108157-llama32k-adaptive-threshold-prefill-ruler13-n32-gpu3/output.log).
- [llama32k-adaptive-threshold-prefill-ruler13-n32-host1.json](llama32k-adaptive-threshold-prefill-ruler13-n32-host1.json) · [Run command](provenance/1794790332572263574-llama32k-adaptive-threshold-prefill-ruler13-n32-host1/run.json) · [Log](provenance/1794790332572263574-llama32k-adaptive-threshold-prefill-ruler13-n32-host1/output.log).

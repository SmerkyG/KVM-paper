# Llama 3.1 32K RULER with optimized threshold prefill

Llama-3.1-8B-Instruct with BF16 weights and KV. The optimized LoD recipe uses the c=16 region threshold for two-tier prefill and decode. The source snapshot and launch environment match the accepted 32,768-token prefill timing. The four quality shards ran one request per GPU on a four-H100 80 GB HBM3 host. Each task uses the same 32 frozen prompts and tokenized inputs across methods. Scores stop at the first EOS/EOT. Each task contributes equally to the mean.

| Task | Dense BF16 | RetroInfer BF16 | Earlier LoD with dense prefill | Optimized threshold LoD |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 96.88% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 96.88% | 96.88% |
| niah_multikey_2 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 96.88% | 98.44% | 96.09% | 99.22% |
| niah_multiquery | 99.22% | 98.44% | 97.66% | 98.44% |
| vt | 100.00% | 100.00% | 100.00% | 100.00% |
| fwe | 90.62% | 86.46% | 88.54% | 88.54% |
| cwe | 37.50% | 36.25% | 29.69% | 35.31% |
| qa_1 | 81.25% | 81.25% | 81.25% | 78.12% |
| qa_2 | 50.00% | 50.00% | 50.00% | 46.88% |
| **13-task mean** | **88.88%** | **88.53%** | **87.46%** | **87.95%** |

- Versus Dense BF16, -0.93 points (paired 95% interval -2.33 to +0.43).
- Versus RetroInfer BF16, -0.57 points (paired 95% interval -2.01 to +0.81).
- Versus Earlier LoD with dense prefill, +0.49 points (paired 95% interval -0.60 to +1.57).

The earlier dense-prefill LoD run uses a different source snapshot. Its paired score comparison measures the two full recipes. The [earlier matched prefill matrix](../ruler-32k-prefill-matrix/REPORT.md) holds the decoder and source snapshot fixed.

Frozen cohort SHA-256 `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.

- GPU 0, 128 prompts. [Generations](threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu0-v2.json) · [Run command](provenance/1794787527799947485-threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu0-v2/run.json) · [Log](provenance/1794787527799947485-threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu0-v2/output.log).
- GPU 1, 96 prompts. [Generations](threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu1-v2.json) · [Run command](provenance/1794787527914292001-threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu1-v2/run.json) · [Log](provenance/1794787527914292001-threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu1-v2/output.log).
- GPU 2, 96 prompts. [Generations](threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu2-v2.json) · [Run command](provenance/1794787528034217746-threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu2-v2/run.json) · [Log](provenance/1794787528034217746-threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu2-v2/output.log).
- GPU 3, 96 prompts. [Generations](threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu3-v2.json) · [Run command](provenance/1794787528182940588-threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu3-v2/run.json) · [Log](provenance/1794787528182940588-threshold-prefill-accepted-v3-llama32k-ruler13-n32-gpu3-v2/output.log).

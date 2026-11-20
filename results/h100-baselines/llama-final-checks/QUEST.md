# Shared-KV QUEST on Llama 3.1 8B

QUEST uses the released CUDA kernels and the shared-GQA addressing patch with FP16 weights and KV, two dense layers, and a 2,048-token exact budget that includes the current 16-token page. Dense uses the same FP16 checkpoint. Both use whole-model CUDA graphs for the timing runs.

The adapter seeds the first decode page with the partial prefill KV and its bounds. The graph/eager check restores the captured path’s cache before continuing that generation.

The PyTorch reference fixture verified all 16 starting page offsets over 576 graph steps, with matching selected pages and a maximum attention output error of 0.000513. It includes a rollover between two page-bounds blocks and graph/eager checks at page boundaries.

## 32K RULER

All 416 tokenized prompts and first generated tokens match the FP16 dense control. 373 prompts start decode with a partial page. Scores stop at the first EOS/EOT.

| Task | Dense FP16 | QUEST 2,048 |
|---|---:|---:|
| niah_single_1 | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 84.38% |
| niah_multivalue | 96.09% | 96.88% |
| niah_multiquery | 99.22% | 98.44% |
| vt | 100.00% | 98.12% |
| fwe | 90.62% | 89.58% |
| cwe | 37.50% | 20.94% |
| qa_1 | 81.25% | 81.25% |
| qa_2 | 50.00% | 50.00% |
| **13-task mean** | **88.82%** | **86.12%** |

QUEST changes the mean by -2.70 points (paired 95% interval -4.09 to -1.45). Graph and eager chose the same next token at the sampled step on each task. The maximum sampled absolute logit difference was 0.144531.

## 64K RULER

All 416 tokenized prompts and first generated tokens match the FP16 dense control. 387 prompts start decode with a partial page. Scores stop at the first EOS/EOT.

| Task | Dense FP16 | QUEST 2,048 |
|---|---:|---:|
| niah_single_1 | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 46.88% |
| niah_multivalue | 98.44% | 98.44% |
| niah_multiquery | 100.00% | 100.00% |
| vt | 100.00% | 95.00% |
| fwe | 87.50% | 83.33% |
| cwe | 3.75% | 1.88% |
| qa_1 | 75.00% | 75.00% |
| qa_2 | 53.12% | 50.00% |
| **13-task mean** | **85.99%** | **80.81%** |

QUEST changes the mean by -5.18 points (paired 95% interval -6.72 to -3.63). Graph and eager chose the same next token at the sampled step on each task. The maximum sampled absolute logit difference was 0.450195.

## Matched batch-one timing

The timed window uses the same prompt and teacher-forced tokens within each context, with 256 warmup decode steps and 1,024 measured steps. Prefill includes cache population. Graph capture and initial cache allocation are recorded separately. Each row gives the median of two runs after one warmup.

| Context | Method | Prefill (s) | Decode (ms/token) |
|---|---|---:|---:|
| 32K | [Dense FP16](llama32k-b1-quest-matched-dense-fp16.json) | 1.578 | 8.669 |
| 32K | [QUEST](llama32k-b1-quest-matched-quest.json) | 1.614 | 8.516 |
| 64K | [Dense FP16](llama64k-b1-quest-matched-dense-fp16.json) | 4.791 | 10.056 |
| 64K | [QUEST](llama64k-b1-quest-matched-quest.json) | 4.866 | 9.687 |

## Reproduction

All new checks and runs used `uv` and `cluster-run`.

- [32K dense FP16 control](../clusterkv/clusterkv-llama32k-ruler13-n32-v6-dense-fp16.json).
- Partial-page reference fixture. [Output](llama-quest-author-shared-state-preservation-check.json) · [Command](provenance/1794793390965818487-llama-quest-author-shared-state-preservation-check/run.json) · [Log](provenance/1794793390965818487-llama-quest-author-shared-state-preservation-check/output.log).
- 32K QUEST. [Generations](llama32k-quest-state-ruler13-n32-gpu0-quest.json) · [Command](provenance/1794793549923602071-llama32k-quest-state-ruler13-n32-gpu0-quest/run.json) · [Log](provenance/1794793549923602071-llama32k-quest-state-ruler13-n32-gpu0-quest/output.log).
- 32K QUEST. [Generations](llama32k-quest-state-ruler13-n32-gpu1-quest.json) · [Command](provenance/1794793550061809686-llama32k-quest-state-ruler13-n32-gpu1-quest/run.json) · [Log](provenance/1794793550061809686-llama32k-quest-state-ruler13-n32-gpu1-quest/output.log).
- 32K QUEST. [Generations](llama32k-quest-state-ruler13-n32-gpu2-quest.json) · [Command](provenance/1794793549923447192-llama32k-quest-state-ruler13-n32-gpu2-quest/run.json) · [Log](provenance/1794793549923447192-llama32k-quest-state-ruler13-n32-gpu2-quest/output.log).
- 32K QUEST. [Generations](llama32k-quest-state-ruler13-n32-gpu3-quest.json) · [Command](provenance/1794793549933052464-llama32k-quest-state-ruler13-n32-gpu3-quest/run.json) · [Log](provenance/1794793549933052464-llama32k-quest-state-ruler13-n32-gpu3-quest/output.log).
- 32K QUEST. [Generations](llama32k-quest-state-ruler13-n32-host1-quest.json) · [Command](provenance/1794793554418833674-llama32k-quest-state-ruler13-n32-host1-quest/run.json) · [Log](provenance/1794793554418833674-llama32k-quest-state-ruler13-n32-host1-quest/output.log).
- 64K QUEST. [Generations](llama64k-quest-state-ruler13-n32-gpu0-quest.json) · [Command](provenance/1794793578304209882-llama64k-quest-state-ruler13-n32-gpu0-quest/run.json) · [Log](provenance/1794793578304209882-llama64k-quest-state-ruler13-n32-gpu0-quest/output.log).
- 64K QUEST. [Generations](llama64k-quest-state-ruler13-n32-gpu1-quest.json) · [Command](provenance/1794793578276489149-llama64k-quest-state-ruler13-n32-gpu1-quest/run.json) · [Log](provenance/1794793578276489149-llama64k-quest-state-ruler13-n32-gpu1-quest/output.log).
- 64K QUEST. [Generations](llama64k-quest-state-ruler13-n32-gpu2-quest.json) · [Command](provenance/1794793578294048883-llama64k-quest-state-ruler13-n32-gpu2-quest/run.json) · [Log](provenance/1794793578294048883-llama64k-quest-state-ruler13-n32-gpu2-quest/output.log).
- 64K QUEST. [Generations](llama64k-quest-state-ruler13-n32-gpu3-quest.json) · [Command](provenance/1794793578273851589-llama64k-quest-state-ruler13-n32-gpu3-quest/run.json) · [Log](provenance/1794793578273851589-llama64k-quest-state-ruler13-n32-gpu3-quest/output.log).
- 64K QUEST. [Generations](llama64k-quest-state-ruler13-n32-host1-quest.json) · [Command](provenance/1794793578145693839-llama64k-quest-state-ruler13-n32-host1-quest/run.json) · [Log](provenance/1794793578145693839-llama64k-quest-state-ruler13-n32-host1-quest/output.log).
- 64K dense FP16. [Generations](llama64k-quest-ruler13-n32-gpu0-dense.json) · [Command](provenance/1794793086215839603-llama64k-quest-ruler13-n32-gpu0-dense/run.json) · [Log](provenance/1794793086215839603-llama64k-quest-ruler13-n32-gpu0-dense/output.log).
- 64K dense FP16. [Generations](llama64k-quest-ruler13-n32-gpu1-dense.json) · [Command](provenance/1794793086218103082-llama64k-quest-ruler13-n32-gpu1-dense/run.json) · [Log](provenance/1794793086218103082-llama64k-quest-ruler13-n32-gpu1-dense/output.log).
- 64K dense FP16. [Generations](llama64k-quest-ruler13-n32-gpu2-dense.json) · [Command](provenance/1794793086359306694-llama64k-quest-ruler13-n32-gpu2-dense/run.json) · [Log](provenance/1794793086359306694-llama64k-quest-ruler13-n32-gpu2-dense/output.log).
- 64K dense FP16. [Generations](llama64k-quest-ruler13-n32-gpu3-dense.json) · [Command](provenance/1794793086248518837-llama64k-quest-ruler13-n32-gpu3-dense/run.json) · [Log](provenance/1794793086248518837-llama64k-quest-ruler13-n32-gpu3-dense/output.log).
- 64K dense FP16. [Generations](llama64k-quest-ruler13-n32-host1-dense.json) · [Command](provenance/1794793086280586958-llama64k-quest-ruler13-n32-host1-dense/run.json) · [Log](provenance/1794793086280586958-llama64k-quest-ruler13-n32-host1-dense/output.log).
- 32K Dense FP16 timing. [Command](provenance/1794794813903968401-llama32k-b1-quest-matched-dense-fp16/run.json) · [Log](provenance/1794794813903968401-llama32k-b1-quest-matched-dense-fp16/output.log).
- 32K QUEST timing. [Command](provenance/1794794813766694006-llama32k-b1-quest-matched-quest/run.json) · [Log](provenance/1794794813766694006-llama32k-b1-quest-matched-quest/output.log).
- 64K Dense FP16 timing. [Command](provenance/1794793594169491551-llama64k-b1-quest-matched-dense-fp16/run.json) · [Log](provenance/1794793594169491551-llama64k-b1-quest-matched-dense-fp16/output.log).
- 64K QUEST timing. [Command](provenance/1794793594092536896-llama64k-b1-quest-matched-quest/run.json) · [Log](provenance/1794793594092536896-llama64k-b1-quest-matched-quest/output.log).

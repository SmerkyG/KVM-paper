# Llama 3.1 64K RULER with optimized threshold prefill

Llama-3.1-8B-Instruct with BF16 weights and KV. The 13 tasks each use 32 frozen examples. Tokenized prompts span 59,017–65,531 tokens and match across every arm. The score stops at the first EOS/EOT and weights each task equally.

LoD uses the c=16 region threshold in both prefill and two-tier decode. The source snapshot and kernel settings match the accepted 65,536-token prefill timing. Dense and RetroInfer both use dense prefill. RetroInfer uses GPU-only sparse decode with 1.8% retrieval and 23.2% estimation budgets. All three methods decode at batch one with CUDA graphs on H100 80 GB HBM3 GPUs. One dense shard and one adaptive three-tier shard ran on a separate H100 of the same model. Each shard checks graph and eager logits on one prompt per task.

Adaptive three-tier uses the same c=16 whole-region threshold prefill. During BF16 decode it opens regions at c=16 and refines pages at c=0.5 using RMS-normalized query scores. The [32K panel](../adaptive-threshold-prefill/RULER_32K.md) uses the same kernel settings.

| Task | Dense BF16 | RetroInfer BF16 | LoD two-tier c=16 prefill and decode | LoD adaptive three-tier c=16 prefill and decode |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 96.88% | 96.88% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 96.88% | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 100.00% | 90.62% | 90.62% |
| niah_multivalue | 98.44% | 97.66% | 95.31% | 98.44% |
| niah_multiquery | 100.00% | 100.00% | 97.66% | 97.66% |
| vt | 100.00% | 98.75% | 100.00% | 100.00% |
| fwe | 87.50% | 75.00% | 81.25% | 79.17% |
| cwe | 3.75% | 0.62% | 3.44% | 3.44% |
| qa_1 | 75.00% | 75.00% | 78.12% | 78.12% |
| qa_2 | 53.12% | 53.12% | 56.25% | 56.25% |
| NIAH mean | 99.80% | 99.32% | 97.56% | 97.95% |
| **13-task mean** | **85.99%** | **84.39%** | **84.58%** | **84.66%** |

- LoD two-tier c=16 prefill and decode versus Dense BF16 changes the mean by -1.41 points (paired 95% interval -2.73 to -0.14).
- LoD two-tier c=16 prefill and decode versus RetroInfer BF16 changes the mean by +0.19 points (paired 95% interval -1.19 to +1.56).
- LoD adaptive three-tier c=16 prefill and decode versus LoD two-tier c=16 prefill and decode changes the mean by +0.08 points (paired 95% interval -0.32 to +0.54).
- LoD adaptive three-tier c=16 prefill and decode versus Dense BF16 changes the mean by -1.33 points (paired 95% interval -2.64 to -0.07).
- LoD adaptive three-tier c=16 prefill and decode versus RetroInfer BF16 changes the mean by +0.27 points (paired 95% interval -1.07 to +1.57).

The [FP16 QUEST comparison](../llama-final-checks/QUEST.md#64k-ruler) uses the same 416 prompts with a precision-matched dense control.

## Complete 64K method evaluation

All arms below use the same 416 tokenized Llama 3.1 8B prompts with 32 samples per task. Scoring stops at the first EOS/EOT.

The [dense-prefill Multipole and Squeezed results](../llama-context-scaling/DENSE_PREFILL.md#64k-ruler) use this same cohort and match the dense BF16 first token on all 416 prompts.

### FP16 precision-matched comparison

ClusterKV uses its released eager planner with 400 initial centroids and a 1,024-token exact budget. QUEST uses 2,048 tokens. The full-model [ClusterKV graph timing](../clusterkv/REPORT.md#matched-64k-batch-one-timing) uses the same ClusterKV budgets. ClusterKV and FP16 dense emitted the same first token on all 416 prompts.

| Task | Dense FP16 | QUEST 2,048 | ClusterKV |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 62.50% |
| niah_single_2 | 100.00% | 100.00% | 65.62% |
| niah_single_3 | 100.00% | 100.00% | 75.00% |
| niah_multikey_1 | 100.00% | 100.00% | 87.50% |
| niah_multikey_2 | 100.00% | 100.00% | 68.75% |
| niah_multikey_3 | 100.00% | 46.88% | 21.88% |
| niah_multivalue | 98.44% | 98.44% | 39.06% |
| niah_multiquery | 100.00% | 100.00% | 50.00% |
| vt | 100.00% | 95.00% | 84.38% |
| fwe | 87.50% | 83.33% | 78.12% |
| cwe | 3.75% | 1.88% | 0.94% |
| qa_1 | 75.00% | 75.00% | 71.88% |
| qa_2 | 53.12% | 50.00% | 50.00% |
| **13-task mean** | **85.99%** | **80.81%** | **58.12%** |

- QUEST 2,048 changes the mean versus Dense FP16 by -5.18 points (paired 95% interval -6.72 to -3.63).
- ClusterKV changes the mean versus Dense FP16 by -27.86 points (paired 95% interval -31.18 to -24.61).

### BF16 shared-prefill comparison

**Historical experiment.** This section uses a LoD prompt pass for all decoders. The paper comparison uses dense prefill for Multipole and Squeezed. See the [current comparison](../llama-context-scaling/REPORT.md#decode-latency-and-ruler-quality).

All four decoders start from the same optimized c=16 LoD prefill and first token. Multipole has a 512-token exact budget and fits 2,900 centroids per physical KV head. Squeezed reuses the initial index and applies a p90 cutoff. Multipole refines its index during decode, while Squeezed keeps it fixed. The shared first token agrees with fully dense BF16 prefill on 395 of 416 prompts. Median shared evaluation time was 15.00 seconds per prompt.

The shared runner uses fixed 4,096-query prefill scratch tiles. The standalone two-tier run above uses automatic scratch sizing. The two LoD runs matched on 411 of 416 first tokens and 398 first-response token sequences.

| Task | LoD | Dense decode | Multipole 512 | Squeezed p90 |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 96.88% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 96.88% | 96.88% | 96.88% |
| niah_multikey_2 | 96.88% | 100.00% | 96.88% | 100.00% |
| niah_multikey_3 | 90.62% | 93.75% | 43.75% | 93.75% |
| niah_multivalue | 95.31% | 97.66% | 94.53% | 95.31% |
| niah_multiquery | 97.66% | 100.00% | 99.22% | 98.44% |
| vt | 100.00% | 100.00% | 92.50% | 92.50% |
| fwe | 79.17% | 85.42% | 83.33% | 85.42% |
| cwe | 3.44% | 3.75% | 0.62% | 5.94% |
| qa_1 | 78.12% | 78.12% | 78.12% | 78.12% |
| qa_2 | 56.25% | 56.25% | 56.25% | 56.25% |
| **13-task mean** | **84.18%** | **85.52%** | **80.16%** | **84.82%** |

- LoD changes the mean versus Dense decode by -1.35 points (paired 95% interval -2.51 to -0.24).
- Multipole 512 changes the mean versus Dense decode by -5.36 points (paired 95% interval -6.99 to -3.79).
- Squeezed p90 changes the mean versus Dense decode by -0.71 points (paired 95% interval -1.35 to -0.04).

#### ClusterKV graph rollout check

The graph path reproduced all four sampled eager retrieval misses. The first-response token sequences matched on one of the four prompts. [Check prompts](../llama-context-scaling/clusterkv-64k-graph-check-prompts.json).

| Task and source row | Eager score | Graph score |
|---|---:|---:|
| [niah_single_1, row 94](../llama-context-scaling/llama-context-64915-clusterkv-ruler64-graph-check.json) | 0.00% | 0.00% |
| [niah_single_2, row 86](../llama-context-scaling/llama-context-65314-clusterkv-ruler64-graph-check.json) | 0.00% | 0.00% |
| [niah_multikey_3, row 4](../llama-context-scaling/llama-context-63929-clusterkv-ruler64-graph-check.json) | 0.00% | 0.00% |
| [niah_multikey_3, row 36](../llama-context-scaling/llama-context-63961-clusterkv-ruler64-graph-check.json) | 0.00% | 0.00% |

[QUEST full results](../llama-final-checks/QUEST.md#64k-ruler) · [Context timing sweep](../llama-context-scaling/REPORT.md).

#### Run provenance

- ClusterKV FP16. [Generations](../llama-context-scaling/llama-context-64k-clusterkv-ruler-gpu0.json) · [Command](../llama-context-scaling/provenance/1794795844663928127-llama-context-64k-clusterkv-ruler-gpu0/run.json) · [Log](../llama-context-scaling/provenance/1794795844663928127-llama-context-64k-clusterkv-ruler-gpu0/output.log).
- ClusterKV FP16. [Generations](../llama-context-scaling/llama-context-64k-clusterkv-ruler-gpu1.json) · [Command](../llama-context-scaling/provenance/1794795944935424265-llama-context-64k-clusterkv-ruler-gpu1/run.json) · [Log](../llama-context-scaling/provenance/1794795944935424265-llama-context-64k-clusterkv-ruler-gpu1/output.log).
- ClusterKV FP16. [Generations](../llama-context-scaling/llama-context-64k-clusterkv-ruler-gpu2.json) · [Command](../llama-context-scaling/provenance/1794795844622016949-llama-context-64k-clusterkv-ruler-gpu2/run.json) · [Log](../llama-context-scaling/provenance/1794795844622016949-llama-context-64k-clusterkv-ruler-gpu2/output.log).
- ClusterKV FP16. [Generations](../llama-context-scaling/llama-context-64k-clusterkv-ruler-gpu3.json) · [Command](../llama-context-scaling/provenance/1794795844756775039-llama-context-64k-clusterkv-ruler-gpu3/run.json) · [Log](../llama-context-scaling/provenance/1794795844756775039-llama-context-64k-clusterkv-ruler-gpu3/output.log).
- Shared prefill BF16. [Generations](../llama-context-scaling/llama-context-64k-shared-ruler-gpu0.json) · [Command](../llama-context-scaling/provenance/1794795944944743964-llama-context-64k-shared-ruler-gpu0/run.json) · [Log](../llama-context-scaling/provenance/1794795944944743964-llama-context-64k-shared-ruler-gpu0/output.log).
- Shared prefill BF16. [Generations](../llama-context-scaling/llama-context-64k-shared-ruler-gpu1-balanced.json) · [Command](../llama-context-scaling/provenance/1794796046964897852-llama-context-64k-shared-ruler-gpu1-balanced/run.json) · [Log](../llama-context-scaling/provenance/1794796046964897852-llama-context-64k-shared-ruler-gpu1-balanced/output.log).
- Shared prefill BF16. [Generations](../llama-context-scaling/llama-context-64k-shared-ruler-gpu2-balanced.json) · [Command](../llama-context-scaling/provenance/1794796047116341766-llama-context-64k-shared-ruler-gpu2-balanced/run.json) · [Log](../llama-context-scaling/provenance/1794796047116341766-llama-context-64k-shared-ruler-gpu2-balanced/output.log).
- Shared prefill BF16. [Generations](../llama-context-scaling/llama-context-64k-shared-ruler-gpu3.json) · [Command](../llama-context-scaling/provenance/1794795944930118974-llama-context-64k-shared-ruler-gpu3/run.json) · [Log](../llama-context-scaling/provenance/1794795944930118974-llama-context-64k-shared-ruler-gpu3/output.log).
- Shared prefill BF16. [Generations](../llama-context-scaling/llama-context-64k-shared-ruler-gpusolo-balanced.json) · [Command](../llama-context-scaling/provenance/1794796047250356231-llama-context-64k-shared-ruler-gpusolo-balanced/run.json) · [Log](../llama-context-scaling/provenance/1794796047250356231-llama-context-64k-shared-ruler-gpusolo-balanced/output.log).
- Shared prefill BF16. [Generations](../llama-context-scaling/llama-context-64k-shared-ruler-gpusolo.json) · [Command](../llama-context-scaling/provenance/1794795784996132552-llama-context-64k-shared-ruler-gpusolo/run.json) · [Log](../llama-context-scaling/provenance/1794795784996132552-llama-context-64k-shared-ruler-gpusolo/output.log).
- ClusterKV graph check. [Generations](../llama-context-scaling/llama-context-64915-clusterkv-ruler64-graph-check.json) · [Command](../llama-context-scaling/provenance/1794796876757860839-llama-context-64915-clusterkv-ruler64-graph-check/run.json) · [Log](../llama-context-scaling/provenance/1794796876757860839-llama-context-64915-clusterkv-ruler64-graph-check/output.log).
- ClusterKV graph check. [Generations](../llama-context-scaling/llama-context-65314-clusterkv-ruler64-graph-check.json) · [Command](../llama-context-scaling/provenance/1794796876846918358-llama-context-65314-clusterkv-ruler64-graph-check/run.json) · [Log](../llama-context-scaling/provenance/1794796876846918358-llama-context-65314-clusterkv-ruler64-graph-check/output.log).
- ClusterKV graph check. [Generations](../llama-context-scaling/llama-context-63929-clusterkv-ruler64-graph-check.json) · [Command](../llama-context-scaling/provenance/1794796876796350827-llama-context-63929-clusterkv-ruler64-graph-check/run.json) · [Log](../llama-context-scaling/provenance/1794796876796350827-llama-context-63929-clusterkv-ruler64-graph-check/output.log).
- ClusterKV graph check. [Generations](../llama-context-scaling/llama-context-63961-clusterkv-ruler64-graph-check.json) · [Command](../llama-context-scaling/provenance/1794796876710353117-llama-context-63961-clusterkv-ruler64-graph-check/run.json) · [Log](../llama-context-scaling/provenance/1794796876710353117-llama-context-63961-clusterkv-ruler64-graph-check/output.log).

## Verification and reproduction

The adaptive prefill audit checked 98,304 head-query pairs on two 64K prompts. Region selection matched the reference, and the maximum absolute prefill output difference was 0.015625.

[Frozen cohort](../quality-ruler-64k-frozen-13x32.json.gz), SHA-256 `bf80f84fb9b17e30ef154117b8165b352e70cb07e5b925d32b6498be51005c2b`.

- Adaptive three-tier prefill audit. [Output](llama64k-adaptive-threshold-prefill-smoke.json) · [Run command](provenance/1794791867192001144-llama64k-adaptive-threshold-prefill-smoke/run.json) · [Log](provenance/1794791867192001144-llama64k-adaptive-threshold-prefill-smoke/output.log).
- Dense BF16, separate H100. [Generations](llama64k-ruler13-n32-dense-gpu0.json) · [Run command](provenance/1794789877066123640-llama64k-ruler13-n32-dense-gpu0/run.json) · [Log](provenance/1794789877066123640-llama64k-ruler13-n32-dense-gpu0/output.log).
- Dense BF16, GPU 1. [Generations](llama64k-ruler13-n32-dense-gpu1.json) · [Run command](provenance/1794789314242446831-llama64k-ruler13-n32-dense-gpu1/run.json) · [Log](provenance/1794789314242446831-llama64k-ruler13-n32-dense-gpu1/output.log).
- Dense BF16, GPU 2. [Generations](llama64k-ruler13-n32-dense-gpu2.json) · [Run command](provenance/1794789314369204728-llama64k-ruler13-n32-dense-gpu2/run.json) · [Log](provenance/1794789314369204728-llama64k-ruler13-n32-dense-gpu2/output.log).
- Dense BF16, GPU 3. [Generations](llama64k-ruler13-n32-dense-gpu3.json) · [Run command](provenance/1794789314480988629-llama64k-ruler13-n32-dense-gpu3/run.json) · [Log](provenance/1794789314480988629-llama64k-ruler13-n32-dense-gpu3/output.log).
- RetroInfer BF16, GPU 0. [Generations](llama64k-ruler13-n32-retroinfer-gpu0.json) · [Run command](provenance/1794789314595121977-llama64k-ruler13-n32-retroinfer-gpu0/run.json) · [Log](provenance/1794789314595121977-llama64k-ruler13-n32-retroinfer-gpu0/output.log).
- RetroInfer BF16, GPU 1. [Generations](llama64k-ruler13-n32-retroinfer-gpu1.json) · [Run command](provenance/1794789314708518173-llama64k-ruler13-n32-retroinfer-gpu1/run.json) · [Log](provenance/1794789314708518173-llama64k-ruler13-n32-retroinfer-gpu1/output.log).
- RetroInfer BF16, GPU 2. [Generations](llama64k-ruler13-n32-retroinfer-gpu2.json) · [Run command](provenance/1794789314819752876-llama64k-ruler13-n32-retroinfer-gpu2/run.json) · [Log](provenance/1794789314819752876-llama64k-ruler13-n32-retroinfer-gpu2/output.log).
- RetroInfer BF16, GPU 3. [Generations](llama64k-ruler13-n32-retroinfer-gpu3.json) · [Run command](provenance/1794789314930037214-llama64k-ruler13-n32-retroinfer-gpu3/run.json) · [Log](provenance/1794789314930037214-llama64k-ruler13-n32-retroinfer-gpu3/output.log).
- LoD two-tier c=16 prefill and decode, GPU 0. [Generations](llama64k-ruler13-n32-threshold-gpu0.json) · [Run command](provenance/1794789313594910692-llama64k-ruler13-n32-threshold-gpu0/run.json) · [Log](provenance/1794789313594910692-llama64k-ruler13-n32-threshold-gpu0/output.log).
- LoD two-tier c=16 prefill and decode, GPU 1. [Generations](llama64k-ruler13-n32-threshold-gpu1.json) · [Run command](provenance/1794789313710772354-llama64k-ruler13-n32-threshold-gpu1/run.json) · [Log](provenance/1794789313710772354-llama64k-ruler13-n32-threshold-gpu1/output.log).
- LoD two-tier c=16 prefill and decode, GPU 2. [Generations](llama64k-ruler13-n32-threshold-gpu2.json) · [Run command](provenance/1794789313832705610-llama64k-ruler13-n32-threshold-gpu2/run.json) · [Log](provenance/1794789313832705610-llama64k-ruler13-n32-threshold-gpu2/output.log).
- LoD two-tier c=16 prefill and decode, GPU 3. [Generations](llama64k-ruler13-n32-threshold-gpu3.json) · [Run command](provenance/1794789313972113027-llama64k-ruler13-n32-threshold-gpu3/run.json) · [Log](provenance/1794789313972113027-llama64k-ruler13-n32-threshold-gpu3/output.log).
- LoD adaptive three-tier c=16 prefill and decode, GPU 0. [Generations](llama64k-adaptive-threshold-prefill-ruler13-n32-gpu0.json) · [Run command](provenance/1794791927605886555-llama64k-adaptive-threshold-prefill-ruler13-n32-gpu0/run.json) · [Log](provenance/1794791927605886555-llama64k-adaptive-threshold-prefill-ruler13-n32-gpu0/output.log).
- LoD adaptive three-tier c=16 prefill and decode, GPU 1. [Generations](llama64k-adaptive-threshold-prefill-ruler13-n32-gpu1.json) · [Run command](provenance/1794791927464222442-llama64k-adaptive-threshold-prefill-ruler13-n32-gpu1/run.json) · [Log](provenance/1794791927464222442-llama64k-adaptive-threshold-prefill-ruler13-n32-gpu1/output.log).
- LoD adaptive three-tier c=16 prefill and decode, GPU 2. [Generations](llama64k-adaptive-threshold-prefill-ruler13-n32-gpu2.json) · [Run command](provenance/1794791927611424839-llama64k-adaptive-threshold-prefill-ruler13-n32-gpu2/run.json) · [Log](provenance/1794791927611424839-llama64k-adaptive-threshold-prefill-ruler13-n32-gpu2/output.log).
- LoD adaptive three-tier c=16 prefill and decode, GPU 3. [Generations](llama64k-adaptive-threshold-prefill-ruler13-n32-gpu3.json) · [Run command](provenance/1794791927513953905-llama64k-adaptive-threshold-prefill-ruler13-n32-gpu3/run.json) · [Log](provenance/1794791927513953905-llama64k-adaptive-threshold-prefill-ruler13-n32-gpu3/output.log).
- LoD adaptive three-tier c=16 prefill and decode, separate H100. [Generations](llama64k-adaptive-threshold-prefill-ruler13-n32-host1.json) · [Run command](provenance/1794791931769742923-llama64k-adaptive-threshold-prefill-ruler13-n32-host1/run.json) · [Log](provenance/1794791931769742923-llama64k-adaptive-threshold-prefill-ruler13-n32-host1/output.log).

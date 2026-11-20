# Qwen3-8B real-state attention comparison on H100

**128K configuration correction.** The archived 128K model states used
effective 1.25x YaRN in Transformers 4.51.3 because max_position_embeddings
stayed at 40,960. Correct factor-four YaRN also requires this field to be
131,072. The 128K real-state measurements require recapture and retiming.
The [corrected long-context investigation](../results/h100-baselines/qwen-long/REPORT.md)
records the regression and corrected checks. The 32K native-RoPE rows remain valid.

These are **batch-one, single-layer, graph-only** timings on model-produced
queries and KV states. They are not full-model inference, amortized generation
timings, or a quality-matched comparison. Results use BF16 Q/K/V on one H100
80 GB. LoD three-tier INT4 quantizes its retained remote page storage.

The [Qwen3 optimization transfer report](../results/h100-baselines/qwen-opt/REPORT.md)
covers full-model prefill, complete cache updates, and the separate 32K quality panel.
The [long-context quality and connected decode report](../results/h100-baselines/qwen-long/REPORT.md)
uses matched 64K/128K RULER cohorts and measures whole-model decode from shared optimized LoD prefill states.
Its [matched 64K decode table](../results/h100-baselines/qwen-long/REPORT.md#matched-64k-whole-model-decode)
includes periodic cache updates and lists initial index setup separately.

## Inputs and timing boundary

The real input comes from layer 18 (zero-indexed) of `Qwen/Qwen3-8B`. We run
the model through this layer in 4,096-token prefill chunks on actual text from
the cached `Seerkfang/prolong-64k-512-new` dataset, then teacher-force the next
token to obtain the query. Each timed method uses that same layer's query and
prefix keys and values. The 32K capture uses native RoPE. The archived 128K
capture used effective 1.25x YaRN. The query is at position 32,768 or 131,072,
respectively, so each test processes one query token after the stated prefix.
The 128K prompt concatenates two dataset documents; the 32K prompt uses one.
The full model is loaded initially, and layers after layer 18 are released
before prefill because they cannot affect that layer's activations.

Q/K normalization and the model's position transform are included in the
captured activations. Sampled keys and values matched the live model KV cache
exactly. The states are transferred through an anonymous, memory-backed file
descriptor between the model process and the kernel environments; no Q/K/V
snapshot is written to disk. The author and LoD kernels use Triton 3.1.0 and
3.7.1 respectively, as in the synthetic comparison. Model execution, prompt
processing, cluster/index construction, CUDA graph capture, per-token KV
writes and all update costs are outside the timed graphs.

Dense uses the author's FlashDecoding graph over the whole prefix. The fixed
LoD columns open up to eight whole regions or, for three-tier, two pages per
selected region. The new two-tier threshold column opens regions above c=16
times the average normalized proxy mass, with the existing 1,024-entry region
eligibility limit and no cap on the number selected. Multipole and its Squeezed variant use the paper-benchmark
lookup and sparse-attention kernels. Multipole additionally times centroid
replacement; the Squeezed variant omits it. As in the synthetic paper-component
benchmark, the replacement result and lookup result are **not wired into** the
timed sparse-attention operation: the sparse fixture uses a selection computed
before graph capture. All fixtures here are derived from the same real Q/K/V,
but the Multipole graph is not a connected computation of its full output.
The Squeezed variant uses **1:8** (one key centroid per eight remote tokens);
Multipole uses **1:16** (one key/value centroid pair per sixteen remote tokens).
Clustering is initialized with the authors' block k-means on
the model's position-transformed keys, and cluster key/value summaries and
counts are reconstructed from the resulting assignments. This common native
RoPE input differs from the authors' windowed-RoPE generation implementation.

For local/sink rows, both sparse methods include one sink and 256 recent tokens
and use the remainder as remote; dense attends to the whole prefix. In
remote-only rows, LoD excludes its one sink and 256 recent tokens, while the
Multipole and Squeezed fixtures use the full stated prefix as remote. Thus the
remote-only rows have slightly different remote lengths, consistent with the
synthetic report. Budgets, cache representations, quality and measured work
also differ across methods.

## Graph timings

Values are milliseconds per batch step, median of two 512-replay graph
measurements. These figures do **not** include update addends.

| Prefix | Attention fields | Dense | Multipole 1:16, 512 | Squeezed 1:8, 512 | LoD two-tier BF16, top-eight | LoD three-tier BF16 | LoD three-tier INT4 | LoD two-tier BF16, threshold |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 32K | Remote only | 0.069 | 0.161 | 0.103 | 0.049 | 0.042 | 0.042 | **0.029** |
| 32K | Local + sink | 0.069 | 0.192 | 0.139 | 0.051 | 0.046 | 0.044 | **0.028** |
| 128K, incorrect YaRN | Remote only | 0.256 | 0.214 | 0.132 | 0.075 | 0.065 | 0.062 | 0.059 |
| 128K, incorrect YaRN | Local + sink | 0.256 | 0.256 | 0.175 | 0.076 | 0.066 | 0.061 | 0.060 |

This table uses the fresh matched rerun with the threshold implementation.
With local and sink attention at 32K, threshold two-tier is 1.80× faster than
fixed top-eight and 2.44× faster than dense. Threshold routing opens an average
of 29.44 regions and 1,394 remote entries per query head at 32K.
These are kernel ratios on one fixed real query, not full-model speedups
or an accuracy claim. The implementation runs read-only: it appends no dummy
current token, performs no cache writes, and preserves local lengths through
graph replay. Complete frontier references over all query heads, routing checks,
eager/replay checks and local/sink perturbation checks pass.

The [threshold follow-up report](../results/h100-baselines/threshold-next/REPORT.md)
keeps the separate Qwen quality and full-model prefill experiments alongside
these timings. Its raw records include the fresh
[32K author](../results/h100-baselines/threshold-next/threshold-next-real-v2-real-qwen-32768.json.author.json),
[32K LoD](../results/h100-baselines/threshold-next/threshold-next-real-v2-real-qwen-32768.json.lod.json),
[32K threshold](../results/h100-baselines/threshold-next/threshold-next-real-v2-real-qwen-32768.json.threshold.json),
[128K author](../results/h100-baselines/threshold-next/threshold-next-real-v2-real-qwen-131072.json.author.json),
[128K LoD](../results/h100-baselines/threshold-next/threshold-next-real-v2-real-qwen-131072.json.lod.json)
and [128K threshold](../results/h100-baselines/threshold-next/threshold-next-real-v2-real-qwen-131072.json.threshold.json).

The dense column is repeated for convenience: it always includes the whole
prefix. These Multipole and Squeezed measurements are slower than dense at
32K under this paper-component boundary.

| Prefix | Attention fields | Multipole 1:16, 5% | Squeezed 1:8, 5% | Multipole 1:16, 10% | Squeezed 1:8, 10% |
|:---|:---|---:|---:|---:|---:|
| 32K | Remote only | 0.187 | 0.125 | 0.202 | 0.136 |
| 32K | Local + sink | 0.219 | 0.163 | 0.268 | 0.177 |
| 128K, incorrect YaRN | Remote only | 0.254 | 0.175 | 0.264 | 0.186 |
| 128K, incorrect YaRN | Local + sink | 0.298 | 0.223 | 0.331 | 0.225 |

The budget is an upper bound: the author lookup masks entries at its threshold,
so the number of valid selected entries can be lower. Counts are recorded in
the raw results. At 128K and 5%, for example, the budget is 6,553 remote
entries without local/sink and 6,540 with them.

## Checks and provenance

All timed graphs matched eager execution and repeated stably. Dense attention
was checked against an independent reference for one KV head; selected sparse
attention was also checked against an independent reference for one KV head
and its four query heads, including local/sink when present. The LoD tests
checked that perturbing local and sink values affected the local/sink output
and did not affect the remote-only output. These tests do not establish
end-to-end output quality, especially because the Multipole component graph
does not merge replacement into sparse attention.

At 128K the largest device-memory usage observed at a graph-setup sampling
point was 35.12 GiB, with the model process and selected-layer benchmark
coexisting. This is a sampled reading, not a continuous peak measurement;
the model was truncated after layer 18 for this experiment.

The existing [synthetic Multipole report](MULTIPOLE.md) includes separately
measured synthetic update addends. They are not included here and do not
measure updates on these real activations. Raw result records are
[32K author](../results/h100-baselines/qwen3-real-graph-b1-32768.json.author.json),
[32K LoD](../results/h100-baselines/qwen3-real-graph-b1-32768.json.lod.json),
[128K author](../results/h100-baselines/qwen3-real-graph-b1-131072.json.author.json)
and [128K LoD](../results/h100-baselines/qwen3-real-graph-b1-131072.json.lod.json).
The [cluster-run records](../results/h100-baselines/provenance/qwen3-real-state/)
contain commands, source hashes and logs. The in-memory model-state and graph
implementations are [qwen3_real_capture.py](qwen3_real_capture.py) and
[qwen3_real_graph.py](qwen3_real_graph.py).

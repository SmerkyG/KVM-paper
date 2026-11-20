# Llama 3.1 8B Instruct at published Multipole and Squeezed settings

Both methods use BF16, dense causal prefill, and Llama's native RoPE on H100 80 GB. Each prompt supplies the same dense KV states and the same first sampled token to the two decoders.

- **Multipole** uses one centroid per 16 indexed tokens per physical KV head, 8,192-token clustering blocks, and a 512-token remote exact budget in addition to the sink and recent field. The 512-token operating point comes from the paper's Qwen3 evaluation. [Multipole Attention, Section 5 and Appendix C](https://proceedings.neurips.cc/paper_files/paper/2025/file/359f442022ef31bcd9e41959576107ef-Paper-Conference.pdf).
- **Squeezed** uses one level of clustering with 5% centroids per physical KV head and the p80 cutoff from the Llama 3.1 experiment. The final 100 prefix tokens supply native-RoPE queries for cutoff calibration and are always exact. Calibration sets a score threshold, so the realized fraction selected can vary. [Squeezed Attention, Section 6.1 and Appendix G](https://aclanthology.org/2025.acl-long.1568.pdf).

Squeezed fits its own index on normalized, native-RoPE prompt keys and recomputes centroids as means of the original keys. We use a batched GPU solver with random initialization and 10 Lloyd iterations in place of the released cuML fit. Index construction is outside decode timing. Both decoders run their respective Triton kernels inside the same Llama runtime.

[Main speed–quality figure](decode-speed-quality.pdf) · [Plot data](decode-speed-quality.csv) · [Fixed-count ablation](DENSE_PREFILL.md).

## 32K RULER

All 416 prompt hashes and first tokens match the archived dense BF16 run. The table uses the first response through EOS/EOT and averages equally across the 13 tasks.

| Task | Dense BF16 | Multipole | Squeezed |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 96.88% | 100.00% |
| niah_multikey_3 | 100.00% | 93.75% | 93.75% |
| niah_multivalue | 96.88% | 99.22% | 98.44% |
| niah_multiquery | 99.22% | 98.44% | 99.22% |
| vt | 100.00% | 95.62% | 98.75% |
| fwe | 90.62% | 86.46% | 90.62% |
| cwe | 37.50% | 25.62% | 38.12% |
| qa_1 | 81.25% | 81.25% | 81.25% |
| qa_2 | 50.00% | 50.00% | 50.00% |
| **13-task mean** | **88.88%** | **86.71%** | **88.47%** |

Initial centroids per KV head range from 1,760 to 2,032 for Multipole and 1,412 to 1,632 for Squeezed because RULER prompt lengths vary. Median index and cache setup takes 3.10 s and 1.83 s, respectively.

- Shard 0. [Generations](llama-paper-32768-ruler-shard0.json) · [Command](provenance/1794813914338293942-llama-paper-32768-ruler-shard0/run.json) · [Log](provenance/1794813914338293942-llama-paper-32768-ruler-shard0/output.log).
- Shard 1. [Generations](llama-paper-32768-ruler-shard1.json) · [Command](provenance/1794813914327427354-llama-paper-32768-ruler-shard1/run.json) · [Log](provenance/1794813914327427354-llama-paper-32768-ruler-shard1/output.log).
- Shard 2. [Generations](llama-paper-32768-ruler-shard2.json) · [Command](provenance/1794813914382785487-llama-paper-32768-ruler-shard2/run.json) · [Log](provenance/1794813914382785487-llama-paper-32768-ruler-shard2/output.log).
- Shard 3. [Generations](llama-paper-32768-ruler-shard3.json) · [Command](provenance/1794813993224886692-llama-paper-32768-ruler-shard3/run.json) · [Log](provenance/1794813993224886692-llama-paper-32768-ruler-shard3/output.log).

## 64K RULER

All 416 prompt hashes and first tokens match the archived dense BF16 run. The table uses the first response through EOS/EOT and averages equally across the 13 tasks.

| Task | Dense BF16 | Multipole | Squeezed |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 93.75% | 100.00% |
| niah_multikey_3 | 100.00% | 46.88% | 87.50% |
| niah_multivalue | 98.44% | 99.22% | 92.97% |
| niah_multiquery | 100.00% | 100.00% | 100.00% |
| vt | 100.00% | 93.75% | 99.38% |
| fwe | 87.50% | 84.38% | 84.38% |
| cwe | 3.75% | 0.62% | 6.25% |
| qa_1 | 75.00% | 75.00% | 75.00% |
| qa_2 | 53.12% | 56.25% | 53.12% |
| **13-task mean** | **85.99%** | **80.76%** | **84.51%** |

Initial centroids per KV head range from 3,672 to 4,080 for Multipole and 2,945 to 3,271 for Squeezed because RULER prompt lengths vary. Median index and cache setup takes 3.80 s and 4.92 s, respectively.

- Shard 0 (10 tasks). [Generations](llama-paper-65536-ruler-shard0.json) · [Command](provenance/1794814254138906549-llama-paper-65536-ruler-shard0/run.json) · [Log](provenance/1794814254138906549-llama-paper-65536-ruler-shard0/output.log).
- Shard 0 (CWE and QA). [Generations](llama-paper-65536-ruler-tail-shard0.json) · [Command](provenance/1794814254190013777-llama-paper-65536-ruler-tail-shard0/run.json) · [Log](provenance/1794814254190013777-llama-paper-65536-ruler-tail-shard0/output.log).
- Shard 1 (10 tasks). [Generations](llama-paper-65536-ruler-shard1.json) · [Command](provenance/1794814254280959822-llama-paper-65536-ruler-shard1/run.json) · [Log](provenance/1794814254280959822-llama-paper-65536-ruler-shard1/output.log).
- Shard 1 (CWE and QA). [Generations](llama-paper-65536-ruler-tail-shard1.json) · [Command](provenance/1794815890751673330-llama-paper-65536-ruler-tail-shard1/run.json) · [Log](provenance/1794815890751673330-llama-paper-65536-ruler-tail-shard1/output.log).
- Shard 2 (10 tasks). [Generations](llama-paper-65536-ruler-shard2.json) · [Command](provenance/1794814254073199584-llama-paper-65536-ruler-shard2/run.json) · [Log](provenance/1794814254073199584-llama-paper-65536-ruler-shard2/output.log).
- Shard 2 (CWE and QA). [Generations](llama-paper-65536-ruler-tail-shard2.json) · [Command](provenance/1794814254146610519-llama-paper-65536-ruler-tail-shard2/run.json) · [Log](provenance/1794814254146610519-llama-paper-65536-ruler-tail-shard2/output.log).
- Shard 3. [Generations](llama-paper-65536-ruler-shard3.json) · [Command](provenance/1794813993268153462-llama-paper-65536-ruler-shard3/run.json) · [Log](provenance/1794813993268153462-llama-paper-65536-ruler-shard3/output.log).

## Decode timing

Each arm starts from the frozen full prefix, replays 256 untimed steps, and measures the next 1,024 steps. The chart uses the median of two measurements after a warmup. Timed decode includes KV writes and eight Multipole index updates. Dense prefill, initial index fitting, calibration, and graph capture precede the timed window. Squeezed's selected fraction is the final timed step, averaged over layers and query heads and aggregated across the two repetitions.

| Context | Multipole initial centroids per KV head | Squeezed centroids per KV head | Multipole (ms/token) | Squeezed (ms/token) | Squeezed selected prefix |
|---|---:|---:|---:|---:|---:|
| 32K | 2,032 | 1,633 | [16.487](llama-paper-b1-32768-paired.json) | [10.634](llama-paper-b1-32768-paired.json) | 18.2% |
| 64K | 4,080 | 3,271 | [20.399](llama-paper-b1-65536-paired.json) | [12.919](llama-paper-b1-65536-paired.json) | 18.3% |

- 32K. [Command](provenance/1794813921305544492-llama-paper-b1-32768-paired/run.json) · [Log](provenance/1794813921305544492-llama-paper-b1-32768-paired/output.log).
- 64K. [Command](provenance/1794813914206399638-llama-paper-b1-65536-paired/run.json) · [Log](provenance/1794813914206399638-llama-paper-b1-65536-paired/output.log).

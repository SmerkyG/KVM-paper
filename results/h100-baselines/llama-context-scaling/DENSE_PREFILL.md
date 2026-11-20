# Llama 3.1 8B with dense prefill for Multipole and Squeezed

Both methods use BF16 on H100 80 GB. Each prompt is processed once with dense causal attention. The decoders receive the same per-token KV tensors and the quality runs also use the same first generated token. Multipole fits the initial index at 2,900 centroids per KV head and uses a 512-token remote exact budget with its sink and recent tail. Squeezed reuses the initial index and calibrates its p90 cutoff. Two-tier and adaptive three-tier LoD use their own threshold prefill in the paper comparison.

[Fixed-2,900 speed–quality ablation](decode-speed-quality-fixed2900.pdf) · [Plot data and sources](decode-speed-quality-fixed2900.csv) · [Paper-setting comparison](PAPER_SETTINGS.md).

## 32K RULER

The four shards cover all 416 frozen prompts, with 32 examples in each of 13 tasks. All input-token hashes match the existing dense BF16 cohort, and 416 of 416 first tokens match the reference dense run. The scorer stops at the first EOS/EOT.

| Task | Dense BF16 | Multipole | Squeezed |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 96.88% | 100.00% |
| niah_multivalue | 96.88% | 97.66% | 97.66% |
| niah_multiquery | 99.22% | 99.22% | 98.44% |
| vt | 100.00% | 96.25% | 96.25% |
| fwe | 90.62% | 88.54% | 91.67% |
| cwe | 37.50% | 22.81% | 54.69% |
| qa_1 | 81.25% | 81.25% | 81.25% |
| qa_2 | 50.00% | 50.00% | 50.00% |
| **13-task mean** | **88.88%** | **87.12%** | **90.00%** |

Squeezed gains 1.11 percentage points on the 13-task mean (95% paired bootstrap interval +0.25 to +2.04, with 50,000 resamples within each task). Common-word extraction (CWE) contributes +1.32 points to that mean. Across the other 12 tasks, Squeezed averages 92.94% and dense averages 93.16%.

Both arms were rescored from token IDs through the first EOS/EOT with identical prompt hashes, target words, and first tokens. On CWE, Squeezed recovers 175 of 320 target words and dense recovers 120. A whole-word check gives 174 and 120. On source row 407, dense listed ten words absent from the numbered input lists, including “the” and “and.” Squeezed returned nine of the ten target words. The RULER score measures target-word recovery. [Example generations](llama-32k-denseprefill-ruler-shard2.json).

A separate evaluation sampled 128 disjoint 32K CWE examples using the same decoder settings. [Held-out results](CWE_32K_128.md).

At 16K, the corresponding 32 dataset rows yield 93.44% for dense and 94.06% for Squeezed. Both recover target words on every example. [16K comparison](CWE_16K_32.md).

A 32K replay with float64 RoPE table construction found the same zero-hit prompts as the native float32 arm. [Context-length breakdown and precision check](CWE_ROPE_PRECISION.md).

- Shard 0. [Generations](llama-32k-denseprefill-ruler-shard0.json) · [Command](provenance/1794808599422945886-llama-32k-denseprefill-ruler-shard0/run.json) · [Log](provenance/1794808599422945886-llama-32k-denseprefill-ruler-shard0/output.log).
- Shard 1. [Generations](llama-32k-denseprefill-ruler-shard1.json) · [Command](provenance/1794808599503213985-llama-32k-denseprefill-ruler-shard1/run.json) · [Log](provenance/1794808599503213985-llama-32k-denseprefill-ruler-shard1/output.log).
- Shard 2. [Generations](llama-32k-denseprefill-ruler-shard2.json) · [Command](provenance/1794808599328940459-llama-32k-denseprefill-ruler-shard2/run.json) · [Log](provenance/1794808599328940459-llama-32k-denseprefill-ruler-shard2/output.log).
- Shard 3. [Generations](llama-32k-denseprefill-ruler-shard3.json) · [Command](provenance/1794808599327280309-llama-32k-denseprefill-ruler-shard3/run.json) · [Log](provenance/1794808599327280309-llama-32k-denseprefill-ruler-shard3/output.log).

## 64K RULER

The four shards cover all 416 frozen prompts, with 32 examples in each of 13 tasks. All input-token hashes match the existing dense BF16 cohort, and 416 of 416 first tokens match the reference dense run. The scorer stops at the first EOS/EOT.

| Task | Dense BF16 | Multipole | Squeezed |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 96.88% | 100.00% |
| niah_multikey_3 | 100.00% | 43.75% | 93.75% |
| niah_multivalue | 98.44% | 96.09% | 96.88% |
| niah_multiquery | 100.00% | 99.22% | 100.00% |
| vt | 100.00% | 92.50% | 92.50% |
| fwe | 87.50% | 85.42% | 85.42% |
| cwe | 3.75% | 0.62% | 4.69% |
| qa_1 | 75.00% | 75.00% | 75.00% |
| qa_2 | 53.12% | 53.12% | 53.12% |
| **13-task mean** | **85.99%** | **80.20%** | **84.72%** |

- Shard 0. [Generations](llama-64k-denseprefill-ruler-shard0.json) · [Command](provenance/1794809660128988195-llama-64k-denseprefill-ruler-shard0/run.json) · [Log](provenance/1794809660128988195-llama-64k-denseprefill-ruler-shard0/output.log).
- Shard 1. [Generations](llama-64k-denseprefill-ruler-shard1.json) · [Command](provenance/1794809662113436747-llama-64k-denseprefill-ruler-shard1/run.json) · [Log](provenance/1794809662113436747-llama-64k-denseprefill-ruler-shard1/output.log).
- Shard 2. [Generations](llama-64k-denseprefill-ruler-shard2.json) · [Command](provenance/1794809655018876563-llama-64k-denseprefill-ruler-shard2/run.json) · [Log](provenance/1794809655018876563-llama-64k-denseprefill-ruler-shard2/output.log).
- Shard 3. [Generations](llama-64k-denseprefill-ruler-shard3.json) · [Command](provenance/1794809661503807163-llama-64k-denseprefill-ruler-shard3/run.json) · [Log](provenance/1794809661503807163-llama-64k-denseprefill-ruler-shard3/output.log).

## Decode timing

The timing runs use the frozen prompt at each context, 256 untimed steps, and 1,024 timed CUDA-graph steps. Every entry reports the median of two measured repetitions after a warmup. Timings include KV writes and periodic maintenance. Dense prefill, initial fitting, calibration, and graph capture are accounted for outside decode.

| Batch | Prompt tokens | Centroids per KV head | Multipole (ms/step) | Squeezed (ms/step) |
|---:|---:|---:|---:|---:|
| 1 | 16,384 | 2,048 | [15.626](llama-denseprefill-b1-16384-t2048.json) | [10.852](llama-denseprefill-b1-16384-t2048.json) |
| 1 | 16,384 | 2,900 | [17.497](llama-denseprefill-b1-16384-t2900.json) | [11.496](llama-denseprefill-b1-16384-t2900.json) |
| 1 | 32,768 | 2,896 | [16.493](llama-denseprefill-b1-32768-t2896.json) | [11.859](llama-denseprefill-b1-32768-t2896.json) |
| 1 | 32,768 | 2,900 | [17.116](llama-denseprefill-b1-32768-t2900.json) | [11.845](llama-denseprefill-b1-32768-t2900.json) |
| 1 | 65,536 | 2,900 | [17.077](llama-denseprefill-b1-65536-t2900.json) | [12.008](llama-denseprefill-b1-65536-t2900.json) |
| 1 | 65,536 | 4,096 | [17.505](llama-denseprefill-b1-65536-t4096.json) | [12.871](llama-denseprefill-b1-65536-t4096.json) |
| 1 | 128,000 | 2,900 | [17.184](llama-denseprefill-b1-128000-t2900.json) | [13.399](llama-denseprefill-b1-128000-t2900.json) |
| 1 | 128,000 | 5,724 | [21.652](llama-denseprefill-b1-128000-t5724.json) | [15.642](llama-denseprefill-b1-128000-t5724.json) |
| 4 | 8,192 | 1,448 | [17.948](llama-denseprefill-b4-8192-t1448.json) | [10.432](llama-denseprefill-b4-8192-t1448.json) |
| 4 | 16,384 | 2,048 | [17.730](llama-denseprefill-b4-16384-t2048.json) | [11.670](llama-denseprefill-b4-16384-t2048.json) |
| 4 | 32,768 | 2,896 | [19.302](llama-denseprefill-b4-32768-t2896.json) | [13.767](llama-denseprefill-b4-32768-t2896.json) |
| 4 | 32,768 | 2,900 | [20.559](llama-denseprefill-b4-32768-t2900.json) | [13.751](llama-denseprefill-b4-32768-t2900.json) |
| 4 | 65,536 | 2,900 | [20.803](llama-denseprefill-b4-65536-t2900.json) | [16.365](llama-denseprefill-b4-65536-t2900.json) |
| 4 | 65,536 | 4,096 | [20.530](llama-denseprefill-b4-65536-t4096.json) | [17.376](llama-denseprefill-b4-65536-t4096.json) |

### Timing commands

- [llama-denseprefill-b1-16384-t2048](provenance/1794810015207668834-llama-denseprefill-b1-16384-t2048/run.json) · [Log](provenance/1794810015207668834-llama-denseprefill-b1-16384-t2048/output.log).
- [llama-denseprefill-b1-16384-t2900](provenance/1794810139338903702-llama-denseprefill-b1-16384-t2900/run.json) · [Log](provenance/1794810139338903702-llama-denseprefill-b1-16384-t2900/output.log).
- [llama-denseprefill-b1-32768-t2896](provenance/1794808995645075022-llama-denseprefill-b1-32768-t2896/run.json) · [Log](provenance/1794808995645075022-llama-denseprefill-b1-32768-t2896/output.log).
- [llama-denseprefill-b1-32768-t2900](provenance/1794808714348370939-llama-denseprefill-b1-32768-t2900/run.json) · [Log](provenance/1794808714348370939-llama-denseprefill-b1-32768-t2900/output.log).
- [llama-denseprefill-b1-65536-t2900](provenance/1794808851735914406-llama-denseprefill-b1-65536-t2900/run.json) · [Log](provenance/1794808851735914406-llama-denseprefill-b1-65536-t2900/output.log).
- [llama-denseprefill-b1-65536-t4096](provenance/1794809132237435925-llama-denseprefill-b1-65536-t4096/run.json) · [Log](provenance/1794809132237435925-llama-denseprefill-b1-65536-t4096/output.log).
- [llama-denseprefill-b1-128000-t2900](provenance/1794810467375850838-llama-denseprefill-b1-128000-t2900/run.json) · [Log](provenance/1794810467375850838-llama-denseprefill-b1-128000-t2900/output.log).
- [llama-denseprefill-b1-128000-t5724](provenance/1794810274764243896-llama-denseprefill-b1-128000-t5724/run.json) · [Log](provenance/1794810274764243896-llama-denseprefill-b1-128000-t5724/output.log).
- [llama-denseprefill-b4-8192-t1448](provenance/1794810633686577299-llama-denseprefill-b4-8192-t1448/run.json) · [Log](provenance/1794810633686577299-llama-denseprefill-b4-8192-t1448/output.log).
- [llama-denseprefill-b4-16384-t2048](provenance/1794810768608920705-llama-denseprefill-b4-16384-t2048/run.json) · [Log](provenance/1794810768608920705-llama-denseprefill-b4-16384-t2048/output.log).
- [llama-denseprefill-b4-32768-t2896](provenance/1794809281485085261-llama-denseprefill-b4-32768-t2896/run.json) · [Log](provenance/1794809281485085261-llama-denseprefill-b4-32768-t2896/output.log).
- [llama-denseprefill-b4-32768-t2900](provenance/1794809649747012014-llama-denseprefill-b4-32768-t2900/run.json) · [Log](provenance/1794809649747012014-llama-denseprefill-b4-32768-t2900/output.log).
- [llama-denseprefill-b4-65536-t2900](provenance/1794809819026036255-llama-denseprefill-b4-65536-t2900/run.json) · [Log](provenance/1794809819026036255-llama-denseprefill-b4-65536-t2900/output.log).
- [llama-denseprefill-b4-65536-t4096](provenance/1794809449450553342-llama-denseprefill-b4-65536-t4096/run.json) · [Log](provenance/1794809449450553342-llama-denseprefill-b4-65536-t4096/output.log).

## Benchmark prefill policy check

A single-prompt mixed LoD and dense run verified the native dense cache against the captured dense-prefill KV at every layer. Its LoD and dense prompt passes produced different first tokens. Dense matched the frozen dense reference, and both decode graphs matched their own eager logits. [Output](llama-denseprefill-policy-check.json) · [Command](provenance/1794810948609494028-llama-denseprefill-policy-check/run.json).

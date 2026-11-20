# Qwen3-8B: Llama optimization transfer on H100

**Long-context configuration correction.** Earlier 64K and 128,000-token runs used effective 1.25x YaRN in Transformers 4.51.3. max_position_embeddings remained 40,960 and overrode the declared factor of four. This report excludes those runs. The 32K native-RoPE timings and quality panel remain valid. See the [corrected long-context investigation](../qwen-long/REPORT.md).

BF16 weights and KV, batch one, one H100 80GB. All GPU experiments use uv and immutable cluster-run snapshots.

## Full-model prefill

One warmup and three measured repetitions per arm. Medians include complete cache updates/completion, local/sink attention and the unchanged 16,384-token initial exact field. Initial cache allocation, loading and decode graph capture are excluded consistently. Prefill is eager with 8,192-token MLP chunks in every arm. The single decode step after prefill is a check; no amortized decode timing is reported.

| Context tokens | Dense | Top-eight | Previous threshold | Optimized threshold | Optimized versus dense |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 1.714 s | 1.847 s | 2.119 s | 1.965 s | 14.7% more time |

Same frozen real-text Qwen-tokenized ProLong prefix across arms. Native Q/K norms are retained. 32K uses native RoPE. Corrected longer runs use factor-four YaRN for every arm. Threshold routing remains uncapped equal-per-region c=16 with the existing 1,024-entry eligibility limit; it is not the per-entry bounded rule.

## Ported implementation

- Save native QK in FP32; share strict threshold selection and closed-weight preparation; compact and pack only the selected bitmap bits, without sorting or count readbacks.
- Retain explicitly rounded proxy scaling and the globally shifted FP16 coarse projection with FP32 division. Model QK and exact KV remain BF16.
- Use the three-part BF16 tensor-core exact-output reducer with an uncapped scalar fallback for larger frontiers.
- Reuse the adaptive producer stages and scratch policy at 32K/64K; retain 4,096-query scratch throughout the longest context for memory headroom.
- Reuse the exact packed max/index reduction on the original cuBLAS state scores. Fuse Qwen’s own coherence division instead of applying Llama’s cosine key normalization.
- The decode attention kernel already shares the optimized materialized GQA implementation. These prefill times do not update the separate read-only graph or amortized decode tables.

| Context | Optimized peak PyTorch allocation | Measured samples |
|---:|---:|---|
| 32,768 | 65.60 GiB | 1.9650 s, 1.9626 s, 1.9653 s |

Peak allocation includes weights and excludes external driver/library allocations. Serving defaults and the routing/work policy are unchanged. No complete-execution subquadratic bound is claimed.

## Quality at 32K

The same four frozen tasks, 32 examples each, thinking disabled. Scores stop at the first EOS/EOT. This is not the complete 13-task RULER panel or a new RULER macro average. The [long-context follow-up](../qwen-long/REPORT.md) reports matched 128K quality separately.

| Method | Multikey-2 | Multikey-3 | CWE | QA-1 |
|---|---:|---:|---:|---:|
| Dense | 93.75% | 100.00% | 79.38% | 62.50% |
| Fixed top-eight prefill and decode | 87.50% | 93.75% | 73.75% | 56.25% |
| Threshold decode, top-eight prefill | 90.62% | 100.00% | 71.56% | 59.38% |
| Previous threshold prefill and decode | 90.62% | 100.00% | 78.44% | 59.38% |
| Optimized threshold prefill and decode | 90.62% | 100.00% | 76.88% | 59.38% |

## Correctness and scope

Complete measurement and check panel: **False**.

Task changes from the previous threshold run: niah_multikey_2 +0.00 pp, niah_multikey_3 +0.00 pp, cwe -1.56 pp, qa_1 +0.00 pp.

The Qwen cache rescale matches independent PyTorch arithmetic bit for bit over 8,785,920 BF16 outputs, including strided inputs. Native-RoPE model prefill and five decode steps match the Hugging Face reference argmax.

At 32,768 tokens: 36 sampled frontiers, zero selection differences outside tolerance, maximum reference attention error 0.062500, at most 1 adjacent BF16 step where error exceeds 0.04, and exact same-cache eager/graph logits.

The initial Qwen validator stopped at an absolute difference of 0.0625 and its dependent quality job did not execute. The follow-up check retains the existing 0.04 absolute tolerance and additionally permits exactly one adjacent representable BF16 value for larger activations. Raw records retain both absolute errors and BF16 step counts. Failed or partial runs do not contribute timings or quality scores.

## Reducer tuning

These are instrumented one-repetition diagnostics. The 64-route tensor-core block was slower than the 32-route block in the 32K runs. Final timings and quality use 32, with the uncapped fallback.

| Context | TC routes | Prefill wall time | Exact-output reduction | Regions/query/head | Entries/query/head |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 32 | 1.969 s | 0.059 s | 14.49 | 638.9 |
| 32,768 | 64 | 2.010 s | 0.100 s | 14.48 | 638.6 |

The [32K run commands](qwen-opt-v1-models-runs.json) record the opt-in configuration. The [archived long-context commands](qwen-opt-scale-v1-models-runs.json) retain the original configuration.

The [raw summary](SUMMARY.json), commands/logs in provenance/, and content-verified source objects retain all inputs. This panel has 0 pending, 2 failed and 2 cancelled jobs.

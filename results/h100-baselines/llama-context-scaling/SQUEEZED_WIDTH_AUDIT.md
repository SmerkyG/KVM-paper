# Squeezed Attention width and Multipole budget audit

Llama 3.1 8B Instruct, BF16 on H100. Squeezed uses dense causal prefill in all arms. The 5% arm uses the paper's single-level centroid ratio and the Llama adapter's p80 calibration. The LoD-matched arm uses 2,896 centroids per KV head at 32K and 4,096 at 64K and recalibrates p80 for its own index. Both use independent fits on normalized native-RoPE keys with a 10-iteration batched Lloyd solver. [Matched RULER panel](RULER_LOD_MATCHED.md) · [Published-setting panel](PAPER_SETTINGS.md) · [Squeezed Attention](https://aclanthology.org/2025.acl-long.1568.pdf).

Each fit recomputes assignments, centroids, and the cutoff. P80 sets a score threshold from the final 100 prompt queries, using the distribution of token proxy scores averaged over those queries. The response-decoding threshold applies to each new query, so the number of tokens selected can vary even when both indexes use p80.

The Multipole paper's head-to-head accounts for the exact-token budget and explicitly uses an online Squeezed variant with its clustering and windowed RoPE. Their 1:8 Squeezed index matches the metadata size of Multipole's 1:16 index because Multipole stores both key and value centroids. The LoD-matched chart controls centroid count. Its plotted Multipole arm still opens up to 512 remote tokens and uses centroid replacement for the other clusters. In our Llama integration, Multipole shares its selected tokens across a GQA group while Squeezed selects separately per query head. [Multipole Attention, Section 5.1 and Appendix D](https://proceedings.neurips.cc/paper_files/paper/2025/file/359f442022ef31bcd9e41959576107ef-Paper-Conference.pdf).

## Plotted RULER results

We aligned the archived generations on input-token hash, source row, prompt hash, answers, and dense-prefill first token and recomputed each score through the first EOS/EOT. Intervals use 50,000 paired bootstrap resamples within each task. Intervals quantify prompt sampling with these runs' fitted indexes.

| Context | Dense BF16 | Squeezed 5% p80 | Squeezed LoD-count p80 | LoD-count minus dense (points, 95% interval) | LoD-count minus 5% (points, 95% interval) | LoD-count minus dense outside CWE (points) |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 88.88% | 88.47% | 89.66% | +0.78 (-0.13 to +1.73) | +1.19 (+0.24 to +2.27) | +0.20 |
| 64K | 85.99% | 84.51% | 86.08% | +0.10 (-0.92 to +1.11) | +1.57 (+0.39 to +2.83) | -0.57 |

| Context | Dense CWE | 5% CWE | LoD-count CWE | Whole-word hits, dense / 5% / LoD-count | Zero-hit prompts, dense / 5% / LoD-count |
|---|---:|---:|---:|---:|---:|
| 32K | 37.50% | 38.12% | 45.31% | 120 / 122 / 145 out of 320 | 15 / 13 / 11 |
| 64K | 3.75% | 6.25% | 11.88% | 10 / 18 / 36 out of 320 | 28 / 26 / 24 |

At 32K, CWE accounts for 0.60 of the 0.78-point lead over dense. At 64K the CWE gain contributes 0.625 points while the remaining 12 tasks average below dense. The increase over 5% also includes multikey-3 at both lengths and multivalue at 64K. CWE accounts for 2.07 of Squeezed's 2.54-point lead over 512-token Multipole at 32K. At 64K, multikey-3 and CWE contribute 2.64 and 0.87 points to the 4.10-point gap. The prior 128-example p90 test uses a shared Multipole index. [Earlier held-out p90 test](CWE_32K_128.md).

## Exact attention on the timing prompts

The values are averages over layers and query heads at the last measured teacher-forced step, and over the two measured repetitions. Multipole's budget refers to exact remote tokens, with a separate exact sink and recent field. Squeezed's counts cover its fixed prefix.

| Context | Squeezed 5% p80 | Squeezed LoD-count p80 | Multipole remote exact cap |
|---|---:|---:|---:|
| 32K | [5,979](llama-paper-b1-32768-paired.json) | [6,203](llama-lodmatch-b1-32768-paired.json) | 512 |
| 64K | [12,009](llama-paper-b1-65536-paired.json) | [12,313](llama-lodmatch-b1-65536-paired.json) | 512 |

## Held-out width check

### 32K held-out CWE

128 prompts disjoint from the plotted cohort. All three arms share dense prefill and the first token. Both Squeezed arms reset the fitter seed to zero before each fit. The retained prefix count averages the first decode step and every eighth step through the first response, across layers, query heads and prompts.

| Method | CWE | Whole-word hits (out of 1,280) | Zero-hit prompts | Mean retained prefix tokens per query head |
|---|---:|---:|---:|---:|
| Dense BF16 | 43.67% | 555 | 47 | Full prefix |
| Squeezed 5% p80 | 47.42% | 606 | 27 | 6,095 |
| Squeezed LoD-count p80 | 50.47% | 644 | 33 | 5,793 |

The dense arm reproduces the earlier held-out dense first-response text and score on 128 of 128 prompts.

LoD-count p80 minus dense is +6.80 points (+2.42 to +11.56, paired 95% interval), with 31 wins, 34 losses and 63 ties.

LoD-count p80 minus 5% p80 is +3.05 points (-1.72 to +7.73, paired 95% interval), with 43 wins, 25 losses and 60 ties.

LoD-count p80 recovered 89 target-word hits on prompts where dense scored zero. Its net change across all prompts was +87 hits. Among responses with at least eight parseable numbered entries, five or more of the first ten entries began with a word that did not start any source entry in 55/128 dense, 45/128 5% p80, and 43/128 LoD-count p80 responses.

On source row 448, dense and 5% p80 both start their answers with generic words such as *the*, *and*, and *a*. LoD-count p80 starts with *principle*, *communication*, and *stove* and includes nine of the ten target words. RULER CWE scores target-word recovery.

- [Generations](llama-squeezed-width-32768-seed0-shard0.json) · [Command](provenance/1794824125660692594-llama-squeezed-width-32768-seed0-shard0/run.json).
- [Generations](llama-squeezed-width-32768-seed0-shard1.json) · [Command](provenance/1794824125806229888-llama-squeezed-width-32768-seed0-shard1/run.json).
- [Generations](llama-squeezed-width-32768-seed0-shard2.json) · [Command](provenance/1794824125704669214-llama-squeezed-width-32768-seed0-shard2/run.json).
- [Generations](llama-squeezed-width-32768-seed0-shard3.json) · [Command](provenance/1794824125810172330-llama-squeezed-width-32768-seed0-shard3/run.json).

### 64K held-out CWE

128 prompts disjoint from the plotted cohort. All three arms share dense prefill and the first token. Both Squeezed arms reset the fitter seed to zero before each fit. The retained prefix count averages the first decode step and every eighth step through the first response, across layers, query heads and prompts.

| Method | CWE | Whole-word hits (out of 1,280) | Zero-hit prompts | Mean retained prefix tokens per query head |
|---|---:|---:|---:|---:|
| Dense BF16 | 6.87% | 88 | 106 | Full prefix |
| Squeezed 5% p80 | 8.83% | 113 | 101 | 10,939 |
| Squeezed LoD-count p80 | 8.12% | 104 | 100 | 10,717 |

LoD-count p80 minus dense is +1.25 points (-1.17 to +3.91, paired 95% interval), with 11 wins, 6 losses and 111 ties.

LoD-count p80 minus 5% p80 is -0.70 points (-3.36 to +1.95, paired 95% interval), with 8 wins, 9 losses and 111 ties.

LoD-count p80 recovered 31 target-word hits on prompts where dense scored zero. Its net change across all prompts was +16 hits. Among responses with at least eight parseable numbered entries, five or more of the first ten entries began with a word that did not start any source entry in 112/128 dense, 110/128 5% p80, and 112/128 LoD-count p80 responses.

[Frozen holdout](llama-squeezed-width-cwe64k-holdout.json.gz) · [Cohort command](provenance/1794824630947464321-llama-squeezed-width-cwe64k-holdout/run.json).

- [Generations](llama-squeezed-width-65536-seed0-shard0.json) · [Command](provenance/1794824737732901698-llama-squeezed-width-65536-seed0-shard0/run.json).
- [Generations](llama-squeezed-width-65536-seed0-shard1.json) · [Command](provenance/1794824737536616705-llama-squeezed-width-65536-seed0-shard1/run.json).
- [Generations](llama-squeezed-width-65536-seed0-shard2.json) · [Command](provenance/1794824737785462392-llama-squeezed-width-65536-seed0-shard2/run.json).
- [Generations](llama-squeezed-width-65536-seed0-shard3.json) · [Command](provenance/1794824763804645028-llama-squeezed-width-65536-seed0-shard3/run.json).

## 64K Multipole budget check

The two Multipole arms reuse the same fitted 4,096-centroid index per prompt, with the chart's 512-token remote exact cap and a 12,000-token remote exact cap. Squeezed uses an independent 4,096-centroid fit with the seed reset to zero per prompt. Dense is the archived reference. All use the same prompt and first token. The 12,000-token cap approximates the Squeezed workload on the timing prefix. Realized Squeezed selection varies by prompt and decode step.

| Task | Dense BF16 | Multipole 512 | Multipole 12,000 | Squeezed p80 | Effect of widening Multipole (points, paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| niah_multikey_3 | 100.00% | 59.38% | 100.00% | 93.75% | +40.62 (+25.00 to +56.25) |
| cwe | 3.75% | 0.62% | 2.19% | 5.00% | +1.56 (+0.00 to +4.69) |

Squeezed counts cover the fixed prefix across sampled steps and layers. Multipole counts cover remote exact entries in the last layer and at the last generated step logged by the harness.

| Task | Mean Squeezed prefix tokens | Mean Multipole 512 remote tokens | Mean Multipole 12,000 remote tokens |
|---|---:|---:|---:|
| niah_multikey_3 | 10,227 | 502 | 11,986 |
| cwe | 10,629 | 505 | 11,980 |

- [Generations](llama-squeezed-budget-65536-shard0.json) · [Command](provenance/1794824396454303334-llama-squeezed-budget-65536-shard0/run.json).
- [Generations](llama-squeezed-budget-65536-shard1.json) · [Command](provenance/1794824396656786980-llama-squeezed-budget-65536-shard1/run.json).
- [Generations](llama-squeezed-budget-65536-shard2.json) · [Command](provenance/1794824396599346036-llama-squeezed-budget-65536-shard2/run.json).
- [Generations](llama-squeezed-budget-65536-shard3.json) · [Command](provenance/1794824396785905705-llama-squeezed-budget-65536-shard3/run.json).

### 64K CWE fit initialization

The same 32 plotted prompts give 11.88% for the archived fit, 5.00% when resetting to seed zero per prompt, and 7.81% when resetting to seed one per prompt. The archived fit used the worker's RNG stream after the Multipole arm. Each index uses 4,096 centroids and its own p80 cutoff.

- [Generations](llama-squeezed-width-65536-seed1-original-cwe.json) · [Command](provenance/1794825038831364175-llama-squeezed-width-65536-seed1-original-cwe/run.json).

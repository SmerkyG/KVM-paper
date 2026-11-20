# Threshold prefill and real Qwen follow-up

Llama-3.1-8B-Instruct and Qwen3-8B, BF16, batch one, one H100 80GB. GPU experiments use uv and source-snapshotting cluster-run; commands, logs and source hashes are retained in provenance/ and source-objects/. Serving defaults are unchanged.

The latest Llama prefill scaling and its 32K RULER panel are presented first. Earlier Llama and Qwen runs retain their original implementations.

## Latest optimized threshold prefill

The new matched retime beats dense at 64K and 128,000 tokens and narrows the gap at 32K. These are eager full-model prefill medians from one warmup and three measured repetitions, including complete cache updates and completion. Initial cache allocation, loading and decode graph capture are excluded consistently. These are not attention-only or graph-replay timings.

| Context tokens | Dense | Top-eight | Previous optimized threshold | New threshold | New versus dense |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 1.553 s | 1.622 s | 1.756 s | 1.663 s | 7.1% more time |
| 65,536 | 4.749 s | 4.101 s | 4.737 s | 4.518 s | 4.9% less time |
| 128,000 | 15.697 s | 9.420 s | 11.816 s | 11.528 s | 26.6% less time |

Same BF16 Llama-3.1-8B-Instruct checkpoint, H100 80GB, batch one and frozen prompts. Uncapped equal-per-region c=16, 1,024-entry eligibility limit, the initial exact 16,384-token field, local/sink attention and cache-update schedule are unchanged. This is not the per-entry bounded rule.

The saved-score selector and globally shifted FP16 coarse projection remain intact. Small exact frontiers now reduce with tensor cores using three BF16 parts of each FP32 weight, with the original uncapped reducer above 32 routes. Fused cache-key normalization reproduces PyTorch 2.5 rounding, and a packed max/index reduction consumes the original cuBLAS state scores. The fresh top-eight controls use the same cache improvements. Adaptive scratch tiles use 16,384/8,192/4,096 queries as region counts grow; producer pipelining uses two stages through 2,048 regions and one above that. At 128,000 tokens the scratch tile is explicitly 4,096 throughout for memory headroom; the initial exact field remains 16,384 tokens.

The full 32-example CWE check scores **35.94%**, versus **34.38%** for the fresh previous-selector control. Two examples each of multikey-2, multikey-3 and QA-1 retain 100%. Faster online-coarse and tiled-cache variants lost quality and were rejected.

Independent checks match 4.4 million BF16 normalized values and all tested state max/index results (NaN payload bits excluded). Sampled model frontiers match selection outside tolerance at all tested lengths, and same-cache eager/graph decode logits match exactly.

Peak PyTorch allocation is 64.20 GiB at 32K and 70.68 GiB at 64K, including weights and excluding external library/driver memory. The 32K result remains slower than dense. The new 128,000-token configuration peaks at 66.44 GiB and has not had a new 128K quality panel. The longest prompt is 128,000, not 131,072 tokens. Serving defaults are unchanged. No subquadratic bound for complete prefill or amortized cache updates is claimed.

See the [optimization retime report](../direct-opt-v3/REPORT.md) for the full configuration, raw samples, quality checks, rejected trials and immutable source provenance.

The [full 32K RULER panel](../direct-opt-v3/RULER_32K.md) uses the same source snapshot and kernel settings as the accepted prefill timing. Optimized threshold LoD scored **87.95%** on all 416 prompts, versus **88.53%** for RetroInfer and **88.88%** for dense BF16.

The [full 64K RULER panel](../ruler-64k-panel/RULER_64K.md) scores **84.58%** for two-tier threshold LoD, **84.66%** for adaptive three-tier, **84.39%** for RetroInfer, and **85.99%** for dense BF16. All four methods use the same 416 tokenized inputs.

The [matched whole-model retimes](../llama-final-checks/REPORT.md) cover both optimized threshold-prefill recipes at 32K and 64K, with periodic decode cache updates included.

The [32K adaptive three-tier threshold-prefill panel](../adaptive-threshold-prefill/RULER_32K.md) scores **88.09%** with the c=16 optimized threshold prefill and c=0.5 adaptive page refinement during decode.

<details>
<summary>Previous selector optimization, including the earlier 128K retime</summary>

## Previous selector optimization

The optimized direct-FP16 path takes **11.816 s at 128,000 tokens**, **3.7% less time** than the freshly rerun previous direct-FP16 path. It remains 24.0% slower than top-eight. These are uninstrumented full-model measurements, including cache completion.

Final threshold decisions and closed weights now share one saved-FP32-score pass. The kernel writes exact selection bits; separate word compaction and sparse packing read those bits without rescanning routing scores. Query scaling explicitly rounds before adding log-counts to preserve the old selector’s boundary decisions. The FP16 coarse projection and FP32 division afterward remain unchanged. The original exact-attention geometry and worker count were faster than the alternatives tested and are retained.

Same Llama-3.1-8B-Instruct checkpoint, BF16 model/KV, batch one, H100 80GB, frozen prompt per length, one warmup and three measured repetitions; medians below. Uncapped equal-per-region c=16, 1,024-entry region eligibility, the 16,384-token initial exact field, local/sink attention and cache-update scope are unchanged. Initial allocation, model loading and graph capture are excluded for every arm; MLP chunks remain 8,192. The longest prompt is 128,000 tokens.

| Context tokens | Dense | Top-eight | Previous direct FP16 | Optimized direct FP16 | Threshold time reduction |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 1.562 s | 1.654 s | 1.830 s | 1.760 s | 3.8% |
| 65,536 | 4.749 s | 4.168 s | 5.009 s | 4.748 s | 5.2% |
| 128,000 | 15.698 s | 9.531 s | 12.268 s | 11.816 s | 3.7% |

The new 32K quality checks score **34.38% on all 32 CWE examples** and 100% on two examples each of multikey-2, multikey-3 and QA-1. 36/38 individual scores are unchanged from the previous FP16 candidate. The fresh previous-selector CWE control scores 34.38%; its earlier 32.81% result differs on one example. That variation is not attributed to the optimization. **These checks are not a new full 13-task RULER average.**

At 128K, peak PyTorch allocation is 66.44 GiB (including weights; excluding external driver/library allocations). Independent frontier checks at all three lengths have zero selection differences outside tolerance and exact same-cache eager/graph logits. Prefill itself is not whole-model graph replay; two decode tokens are checks rather than amortized decode timings.

The optimized selector is automatic within the experimental direct scaled-FP16 path; `LOD_H100_DIRECT_FUSE_SELECT=0` restores the previous selector. Other prefill paths and serving defaults are unchanged. The [profiling and optimization report](../direct-opt/REPORT.md) records stage timings, regressing trials, validation and source provenance; [SUMMARY.json](../direct-opt/SUMMARY.json) records the matched raw inputs. No complete-execution subquadratic bound is inferred.

</details>

<details>
<summary>Previous direct-summary precision and scaling panel</summary>

## Previous direct-summary precision panel

The previous direct closed-summary candidate takes **1.824 s at 32K**, **5.019 s at 64K** and **12.258 s at 128,000 tokens**. At the longest context, this is 21.9% less prefill time than dense and 7.3% less than conservative threshold. Top-eight remains faster. The candidate is experimental and opt-in; serving defaults are unchanged.

These Llama runs use two-tier **equal-per-region c=16**, uncapped selection of eligible regions up to 1,024 entries, and the unchanged 16,384-token initial exact prefill field, local/sink attention and cache-update schedule. They do not use the per-entry work-bound rule. Model weights and KV storage remain BF16.

The candidate saves native QK scores in FP32, uses a conservative prefix candidate filter followed by exact threshold selection, and computes closed summaries directly. It forms unnormalized exp(logit − closed maximum) weights in FP32, uses FP16 operands for one coarse value projection with FP32 accumulation/output, then divides by the FP32 denominator. Query/KV tensors and exact attention remain BF16, and GQA heads share KV storage. There is no full-coarse-minus-opened subtraction or selected-count CPU readback.

### Full-model prefill scaling

Matched frozen prompts within each context, one warmup and three measured repetitions; median wall time includes deferred cache completion. Initial cache allocation, model loading and graph capture are excluded consistently. MLP chunks are 8,192 and attention scratch tiles are 4,096 queries. These are full-model prefill timings, not whole-model CUDA graph replay. The two-token decode checks are not amortized decode measurements. The longest prompt is 128,000, not 131,072 tokens.

| Context tokens | Dense | Top-eight | Conservative threshold | Previous direct BF16 | New direct, FP16 coarse operands |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 1.560 s | 1.652 s | 1.899 s | 1.820 s | 1.824 s |
| 65,536 | 4.752 s | — | 5.225 s | 4.996 s | 5.019 s |
| 128,000 | 15.703 s | 9.540 s | 13.226 s | 12.219 s | 12.258 s |

The 64K row reuses the matched direct-summary precision panel; top-eight was not rerun at 64K in that series. The 32K/128K conservative controls use automatic selection/packing dispatch; both direct variants use prefix filtering with the same producer tile. The precision change adds less than 0.5% time relative to previous direct BF16 at every length. Threshold remains slower than dense at 32K and 64K.

| Context tokens | Candidate peak PyTorch allocation | Sampled frontiers across 32 layers | Maximum reference attention error |
|---:|---:|---:|---:|
| 32,768 | 33.43 GiB | 32 | 0.0078125 |
| 65,536 | 41.06 GiB | 96 | 0.0078125 |
| 128,000 | 66.44 GiB | 224 | 0.0156250 |

Peak allocation includes model weights and benchmark execution, but excludes driver/library memory outside PyTorch. All sampled frontiers have zero selection differences outside tolerance; same-cache eager/graph logits match exactly at every length. These checks sample frontiers rather than independently validating every token.

### Latest quality checks at 32K

CWE uses all 32 examples from the same frozen cohort as the full panel below, scored only before the first EOS/EOT. Baselines are earlier quality runs on identical inputs; their speed controls above are rerun. These are task scores, not RULER macro averages.

| Method | CWE, 32 examples |
|---|---:|
| Dense | 37.50% |
| Conservative prefix threshold | 30.00% |
| Previous direct BF16 | 27.19% |
| Unnormalized BF16, FP32 division | 27.50% |
| New direct, FP16 coarse operands | 32.81% |

The new candidate also scores 100% on six retrieval/QA examples: two each from multikey-2, multikey-3 and QA-1. All six scores are unchanged from conservative threshold. This small check does not establish full retrieval quality.

**The new direct-summary candidate has no full 13-task RULER average.** The older 87.52% average below belongs to a different prefill implementation. Its 32.81% CWE score happens to equal the new candidate’s CWE score; the remaining tasks have not been rerun for this candidate. No new Qwen or 128K quality panel is implied.

### Provenance and scope

The [direct-summary report](../closed-offset/REPORT.md) contains the precision/offset ablations and numerical audit. Its [machine-readable summary](../closed-offset/SUMMARY.json), raw inputs, and [run commands and logs](../closed-offset/provenance/) remain in that directory. This report verifies their input hashes and rechecks the new quality outputs against the tokenizer and frozen cohort; SUMMARY.json records the imported panel separately under latest_direct.

Latest panel: 0 pending, 0 failed, 2 cancelled jobs. The two optional probability-offset expansions were cancelled while queued; neither contributes a full-task quality score. Cache completion is included in speed measurements, but these results do not establish a subquadratic bound for complete prefill or amortized cache updates.

</details>

<details>
<summary>Historical prefill experiments and stage audits</summary>

These tables describe earlier implementations, not the direct-summary candidate above. The subsequent [selection/packing optimization](../selectpack/REPORT.md) reduces 128,000-token fused threshold prefill from 14.269 s to 13.437 s in a fresh matched pair, keeping c=16 selection and the exact initial 16,384-token field. The tables below preserve the preceding experiments.

## Historical prefill optimization

The exact initial prefill chunk remains 16,384 tokens. Selection remains uncapped per-region c=16, with native-query exact attention, closed summaries, local and sink fields and cache completion. There is no selected-count CPU readback. These experiments do not change the mathematical routing rule or establish subquadratic complete execution.

Grouped atomic reservations, smaller query/region tiles, a direct closed-summary streaming pass, and a dimension-split exact-output reduction passed the independent frontier checks but did not improve full-model prefill. Reducing the closed selector from eight to four warps gave a small pilot improvement; the matched scaling panel checks its repeatability. The longer-context fused producer keeps eight warps.

| Pilot, one warmup + two timed repetitions | 32K prefill | 64K prefill |
|---|---:|---:|
| Existing control | 1.874 s | 5.392 s |
| Four selection warps | 1.840 s | 5.569 s |
| Grouped reservations, four queries | 1.928 s | 5.630 s |
| Tiled selection, 32 queries | 1.945 s | 5.484 s |
| Streaming closed-summary pass, 32 queries | 1.925 s | 6.093 s |
| Dimension-split exact-output reduction | 1.892 s | 5.493 s |

## Historical full-model prefill scaling

Same frozen prompt within each context, same checkpoint and BF16 runtime. One warmup and three measured repetitions; median wall time, including LoD deferred cache completion but excluding initial cache allocation for every method. MLP chunks are 8,192 for every arm at every length. The longest full-model prompt is 128,000 tokens; the Qwen graph fixture below is separately 131,072. Two-token generation is used only to finalize prefill and capture; its decode timing is not used as an amortized performance result.

| Tokens | Dense | Top-eight prefill | Closed threshold | Fused threshold | Best completed threshold / dense |
|---|---:|---:|---:|---:|---:|
| 32,768 | 1.554 s | 1.647 s | 1.842 s | 1.912 s | 1.185× |
| 65,536 | 4.751 s | 4.167 s | 5.432 s | 5.369 s | 1.130× |
| 128,000 | 15.695 s | 9.535 s | 14.555 s | 14.277 s | 0.910× |

The final column is a latency ratio: below 1 means faster than dense. The initial 128,000-token threshold jobs, including the smaller-scratch retries, completed their warmup prefill but exhausted memory on the second repetition. Benchmark cleanup inside run_once ran while local pool/engine references still kept callback cycles reachable. The cleanup rerun collects after that frame returns and before allocating the next cache, outside timing, and records per-repetition allocated and peak memory. It restores the original 4,096-query scratch tile; the 16,384-token exact prefill field, routing rule and update schedule are unchanged. Closed threshold now completes all repetitions. Fused threshold initially failed after cleanup: a 16 GiB allocation could not fit despite 13.30 GiB of reserved unused memory, consistent with allocator fragmentation. The separate expandable-segment retry now completes one warmup and three measured repetitions at 128K; it changes memory allocation, not attention selection or arithmetic. Native dense/top-eight controls retain their default allocator, as recorded in SUMMARY.json. Both threshold implementations are attempted at all lengths; failed runs contribute no time. Choosing the lower completed time does not change exact-prefill size or routing budget.

### 128K memory audit

Allocated PyTorch memory includes model weights and the entire benchmark repetition; it excludes driver/library allocations outside PyTorch. Peak is the largest of the three measured repetitions. Collection happens between repetitions, outside prefill and decode timing. A return to baseline confirms that the previous cache is released.

| Method | Peak allocated | After collection |
|---|---:|---:|
| Dense | 34.36 GiB | 15.08 GiB |
| Top-eight | 41.35 GiB | 15.08 GiB |
| Closed threshold | 63.47 GiB | 15.08 GiB |
| Fused threshold | 65.38 GiB | 15.08 GiB |

### Historical stage and capacity audit

These diagnostics profile the older fused producer and coarse-correction path. The new direct-summary candidate has no opened-summary subtraction/correction; these stage times must not be attributed to it. The instrumented diagnostics count every remote prefill query head across all 32 layers, excluding the exact initial field and local/sink. They do not replace the uninstrumented full-model times above. Counts are accumulated on the GPU and copied to the CPU after model execution. Both audit arms use expandable segments.

| Tokens | Method | Mean opened regions | Mean exact entries | Reserved route width | Largest leaf-output capacity |
|---|---|---:|---:|---:|---:|
| 32,768 | Top-eight | 8.00 | 261.6 | 8 | 1.00 GiB |
| 32,768 | Fused threshold | 10.02 | 312.6 | 256 | 8.00 GiB |
| 128,000 | Top-eight | 7.98 | 523.7 | 8 | 1.00 GiB |
| 128,000 | Fused threshold | 22.62 | 1465.6 | 512 | 16.00 GiB |

The reserved route width is allocation capacity, not the number actually opened. Leaf dispatch and variable reduction follow actual counts. The existing per-region c=16 rule has a cutoff proportional to 1/R, so its opened region count may grow with R; it does not maintain a fixed eight-region budget. The fused implementation also writes a full saved-score matrix and launches separate threshold packing, exact result reduction and coarse correction. Top-eight keeps the fixed selection within its coarse producer and merges eight route results directly.

| Tokens | Method | Coarse producer + routing | Separate selection | Exact KV kernel | Separate route reduction | Threshold correction/merge |
|---|---|---:|---:|---:|---:|---:|
| 32,768 | Top-eight | 0.228 s | Included | 0.079 s | In final merge | Not separately measured |
| 32,768 | Fused threshold | 0.176 s | 0.139 s | 0.101 s | 0.095 s | 0.114 s |
| 128,000 | Top-eight | 3.101 s | Included | 0.718 s | In final merge | Not separately measured |
| 128,000 | Fused threshold | 2.376 s | 2.897 s | 1.999 s | 0.681 s | 0.963 s |

GPU-event stage times are summed over all layers. The top-eight final merge is not separately instrumented, so this is not a complete subtraction of the two end-to-end times. It separates increased exact KV work from selection, reduction and correction overhead. Extra work-count reductions slightly perturb the profiles.

</details>

## Llama full RULER quality — earlier implementation

First response ends at the first EOS/EOT token. A full average requires all 13 tasks and 32 matched examples/task. Historical 88.02% threshold-prefill quality predates the decoder used in this panel and is not substituted here. The 87.52% result is the earlier combined threshold prefill/decode implementation, not the new direct-summary candidate above. A full rerun is required before assigning that candidate an average.

| Task | Dense | RetroInfer | Earlier threshold prefill + decode |
|---|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 |
| niah_single_2 | 100.00 | 100.00 | 100.00 |
| niah_single_3 | 100.00 | 100.00 | 100.00 |
| niah_multikey_1 | 100.00 | 100.00 | 96.88 |
| niah_multikey_2 | 100.00 | 100.00 | 100.00 |
| niah_multikey_3 | 100.00 | 100.00 | 100.00 |
| niah_multivalue | 96.88 | 98.44 | 99.22 |
| niah_multiquery | 99.22 | 98.44 | 95.31 |
| vt | 100.00 | 100.00 | 100.00 |
| fwe | 90.62 | 86.46 | 88.54 |
| cwe | 37.50 | 36.25 | 32.81 |
| qa_1 | 81.25 | 81.25 | 78.12 |
| qa_2 | 50.00 | 50.00 | 46.88 |
| **Average** | **88.88** | **88.53** | **87.52** |

## Qwen3-8B optimized threshold prefill

The [Qwen transfer report](../qwen-opt/REPORT.md) records the port, same-cache and numerical checks, and the separate reducer tuning trials. Qwen keeps its native Q/K norms and coherence-based clustering. These are three-run eager full-model prefill medians including cache completion, on matched BF16, batch-one prompts. 32K uses native RoPE. The archived 64K and 128,000-token runs used incorrect YaRN scaling and are excluded. The initial exact field stays 16,384 tokens.

| Context tokens | Dense | Top-eight | Previous threshold | Optimized threshold | Time reduction versus previous threshold |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 1.714 s | 1.847 s | 2.119 s | 1.965 s | 7.3% |

## Qwen 32K quality

The optimized threshold row uses the newly transferred prefill kernels. The other rows use their earlier implementations. Four selected tasks on the existing frozen Qwen cohort, thinking disabled. This is not the complete 13-task panel and has no full RULER average. Values use first-response scoring. All LoD configurations here use sparse prefill. Connected author methods use native dense prefill plus cluster construction, windowed routing position 2048, ten sink tokens and 128-token cluster updates. Their graph/eager checks must pass exactly. The Squeezed subset is the Multipole authors’ no-replacement variant at 1:8, not the separate calibrated Squeezed release. Stable-order selected-index packing is enabled only in these connected author quality runs; their latency is not substituted for the untouched paper-component graphs.

| Method | Examples/task | Multikey-2 | Multikey-3 | CWE | QA-1 |
|---|---:|---:|---:|---:|---:|
| Dense | 32 | 93.75 | 100.00 | 79.38 | 62.50 |
| Fixed top-eight prefill and decode | 32 | 87.50 | 93.75 | 73.75 | 56.25 |
| Threshold decode, top-eight prefill | 32 | 90.62 | 100.00 | 71.56 | 59.38 |
| Threshold decode and prefill | 32 | 90.62 | 100.00 | 78.44 | 59.38 |
| Multipole 1:16, budget 512 | 2 | 50.00 | 100.00 | 85.00 | 50.00 |
| Multipole 1:16, budget 2048 | 2 | 50.00 | 100.00 | 80.00 | 50.00 |
| Squeezed subset 1:8, budget 512 | 2 | 100.00 | 100.00 | 90.00 | 50.00 |
| Squeezed subset 1:8, budget 2048 | 2 | 50.00 | 100.00 | 85.00 | 50.00 |
| Optimized threshold prefill and decode | 32 | 90.62 | 100.00 | 76.88 | 59.38 |

## Qwen real-state graph timings

Same layer-18 model-produced query and KV states, BF16, graph-only paper-component boundary. Each time is one attention call in milliseconds; model weights and projections, cache writes, cluster construction and amortized updates are excluded. Local/sink rows cover the whole prefix. Remote-only LoD excludes one sink and 256 recent entries; author remote-only fixtures include the full prefix. Multipole lookup/replacement results are not wired into its timed sparse fixture. Squeezed uses 1:8, Multipole 1:16, both budget 512. These are not quality-matched whole-model timings. The archived 128K captures used incorrect YaRN scaling and are excluded. See the [configuration correction](../qwen-long/REPORT.md).

| Tokens | Scope | Dense | Multipole | Squeezed subset | LoD top-eight | LoD threshold |
|---|---|---:|---:|---:|---:|---:|
| 32,768 | remote | 0.06906 | 0.16081 | 0.10256 | 0.04924 | 0.02899 |
| 32,768 | local | 0.06906 | 0.19195 | 0.13897 | 0.05081 | 0.02826 |

All threshold graph rows pass full-frontier PyTorch checks across every query head, routing checks, eager/replay equality and local/sink perturbation checks. Read-only replay leaves cache lengths unchanged and adds no dummy token.

## Historical run status and exclusions

The original threshold-next series has 0 queued/running jobs; 24 excluded failed jobs. These counts are separate from the latest direct-summary panel above. No partial or failed generation run contributes a score above. Raw run status and commands are retained; SUMMARY.json records every scored input SHA-256.

- `threshold-next-v1-qwen-top8-prefill` (exit 1); inspect its provenance log.
- `threshold-next-v1-qwen-threshold-prefill` (exit 1); inspect its provenance log.
- `threshold-next-real-v1-real-qwen-4096` (exit 1); inspect its provenance log.
- `threshold-next-reduce-v1-closed-split-check` (exit 1); inspect its provenance log.
- `threshold-next-reduce-v1-fast-split-check` (exit 1); inspect its provenance log.
- `threshold-next-reduce-v1-closed-split-32768` (exit 1); inspect its provenance log.
- `threshold-next-reduce-v1-fast-split-65536` (exit 1); inspect its provenance log.
- `threshold-next-author-v1-qwen-multipole-512` (exit 1); inspect its provenance log.
- `threshold-next-author-v1-qwen-multipole-2048` (exit 1); inspect its provenance log.
- `threshold-next-author-v1-qwen-squeezed-multipole-512` (exit 1); inspect its provenance log.
- `threshold-next-author-v1-qwen-squeezed-multipole-2048` (exit 1); inspect its provenance log.
- `threshold-next-scaling-v1-closed-m1-128000` (exit 1); inspect its provenance log.
- `threshold-next-scaling-v1-fast-m1-128000` (exit 1); inspect its provenance log.
- `threshold-next-scaling-memory-v2-closed-m1-128000` (exit 1); inspect its provenance log.
- `threshold-next-scaling-memory-v2-fast-m1-128000` (exit 1); inspect its provenance log.
- `threshold-next-scaling-cleanup-v3-fast-m1-128000` (exit 1); inspect its provenance log.
- `threshold-next-author-full-v1-qwen-multipole-512` (exit 1); inspect its provenance log.
- `threshold-next-author-full-v1-qwen-multipole-2048` (exit 1); inspect its provenance log.
- `threshold-next-author-full-v1-qwen-squeezed-multipole-512` (exit 1); inspect its provenance log.
- `threshold-next-author-full-v1-qwen-squeezed-multipole-2048` (exit 1); inspect its provenance log.
- `threshold-next-scaling-audit-v1-top8-m1-32768` (exit 1); inspect its provenance log.
- `threshold-next-scaling-audit-v1-fast-m1-32768` (exit 1); inspect its provenance log.
- `threshold-next-scaling-audit-v1-top8-m1-128000` (exit 1); inspect its provenance log.
- `threshold-next-scaling-audit-v1-fast-m1-128000` (exit 1); inspect its provenance log.

Early Qwen attempts used an environment without Qwen3 support; corrected runs use isolated Transformers 4.51.3. The first read-only graph validation had a reference adapter error, fixed in v2. The first split reduction checks had a compile-time variable shape error, fixed and rechecked in v2. Initial connected author runs failed the exact graph/eager check and are excluded; no quality conclusion is drawn from them.

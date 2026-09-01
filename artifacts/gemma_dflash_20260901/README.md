# Gemma 4 original DFlash with recursive LOD

This is the matched Gemma 4 speculative panel using recursive three-tier BF16
LOD target attention. The target is `google/gemma-4-26B-A4B-it`. The only
published Gemma checkpoint, `z-lab/gemma-4-26B-A4B-it-DFlash`, is **original
DFlash**, not DFlash2: it supplies one anchor plus 15 proposed tokens in a
16-position target verification.

The matched TP4 follow-up is in
`../gemma_dflash_tp4_20260901/README.md`. Its quality-selected top-16 prefill
profile matches full attention at 8/8 NIAH-S3 throughout 8--128K. At 128K,
verifier-cycle speedup is 5.62x at B1 and 5.27x at B8.

The matched TP2 follow-up is in `../gemma_tp2_20260901/README.md`. TP2 keeps
the original GQA-8 geometry without KV-head replication. The exact TP1 route
profiles score 8/8 throughout. At 128K, recursive LOD reduces the TP2 DFlash
verifier cycle from 69.804 to 17.922 ms at B1 (3.90x) and from 57.982 to
16.308 ms at B8 (3.56x). The same artifact also contains ordinary greedy
full/two-tier/three-tier TP2 results.

Five global layers use LOD with `(D, QH, KVH, GQA) = (512, 16, 2, 8)`;
Gemma's 25 local layers remain native. Recursive verification keeps all 16
proposal K/V positions staged together. B1 uses one 16-position launch. At B8,
the conservative 32-row bound uses four four-position chunks; an explicitly
selected 64-row profile uses two eight-position chunks. Both avoid 16 serial
calls, and each proposal position retains its own current route and causal
local length.

## Validated profiles

- Decode always uses top eight.
- B1 uses prefill top four throughout the 8--128K panel.
- B8 8K and 16K use prefill top four with the conservative 32-row bound.
- B8 32K and 64K use prefill top four with the 64-row bound.
- B8 128K uses fixed prefill top three with the 64-row bound. Fixed top three
  is the high-throughput long-context profile; it scored 8/8.
- Do not switch from top four to top three partway through a prompt. That
  seemingly natural adaptive variant scored only 7/8 at 128K even though each
  fixed profile scored 8/8. Prefill attention changes downstream hidden states,
  so the mixed execution is a third approximation rather than a cheap blend of
  two validated ones.

The panel wrapper defaults Gemma DFlash to prefill top four and the backend's
conservative D=512 row bound. Set
`VLLM_LOD_SPECULATIVE_PARALLEL_MAX_ROWS=64` for the validated 32K/64K B8
profiles. At 128K B8, also set `VLLM_LOD_PREFILL_OPEN_COUNT=3` before starting
the server or benchmark.

## NIAH-S3 quality

These are batch-eight, chat-formatted, greedy results using the same GUID
targets in both modes. Full attention and the selected fixed LOD profile score
8/8 at every length.

| context | full DFlash | recursive LOD DFlash | LOD profile |
|---:|---:|---:|:---|
| 8K | 8/8 | 8/8 | top-4, 32-row bound |
| 16K | 8/8 | 8/8 | top-4, 32-row bound |
| 32K | 8/8 | 8/8 | top-4, 64-row bound |
| 64K | 8/8 | 8/8 | top-4, 64-row bound |
| 128K | 8/8 | 8/8 | fixed top-3, 64-row bound |

The 64-row top-four control scored 7/8 at 16K, so it is not a universal Gemma
replacement for the conservative 32-row split. The initial top-three panel
also scored 7/8 at 32K and 64K, which isolated the
short-context difference to prefill routing rather than speculative acceptance.
Serial and non-speculative controls scored 8/8. Top four restored 8/8 at 32K
and 64K with the 64-row verifier. Fixed top three independently scored 8/8 at
128K. A conservative top-four/32-row 128K control also scored 8/8, but its B8
posting-list tail was too slow to use as the production profile.

## Speed protocol

All rows are TP1 on one MI325X. Prompts are distinct, non-repeated real
documents from `Seerkfang/prolong-64k-512-new`, chat formatted with a final
request to summarize the documents. Full and LOD prompt hashes match at every
length. The scheduler has a 16,384-token aggregate prefill budget, prompts
reserve 256 positions for generation, and sampling is greedy. Each point has
one warmup and three measured repetitions of 256 output tokens.

B1 uses a 0.80 vLLM memory fraction. B8 uses 0.86 through 64K and 0.80 at 128K;
the lower 128K fraction only prevents vLLM from reserving an unnecessary
140.5-GiB native cache in addition to the LOD arena. Both target and draft use
`TRITON_ATTN`: stock AITER's D=512 launch requests more LDS than gfx942
provides, so AITER is not a valid Gemma full-attention control.

Decode milliseconds are per emitted token at B1 and per emitted batch step at
B8. End-to-end DFlash decode includes proposal acceptance and can therefore be
strongly non-monotonic. Verifier-cycle milliseconds divide measured decode wall
time by scheduler iterations that actually verify draft tokens. This uses the
same denominator for full attention and LOD, excludes the initial no-draft
target invocation, and removes the direct dependence on emitted-token count.
It includes the drafter and scheduler work surrounding the target invocation
rather than claiming to isolate an attention kernel. At B8 the average launch
cost still reflects the changing active-row mix as requests finish on different
cycles; the raw records therefore retain both launch and active-row counts.

### Batch one

| context | full prefill (s) | LOD prefill (s) | prefill speedup | full decode (ms) | LOD decode (ms) | decode speedup | full verifier cycle (ms) | LOD verifier cycle (ms) | cycle speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 0.230 | 0.249 | 0.92x | 1.764 | 2.425 | 0.73x | 14.474 | 15.877 | 0.91x |
| 16K | 0.587 | 0.519 | 1.13x | 8.831 | 5.124 | 1.72x | 18.266 | 15.836 | 1.15x |
| 32K | 1.661 | 1.226 | 1.35x | 3.131 | 6.590 | 0.48x | 25.926 | 16.666 | 1.56x |
| 64K | 5.073 | 2.677 | 1.89x | 23.381 | 5.906 | 3.96x | 41.203 | 17.635 | 2.34x |
| 128K | 17.268 | 6.074 | 2.84x | 55.749 | 4.397 | 12.68x | 72.210 | 20.243 | 3.57x |

The 3.57x B1 verifier-cycle gain at 128K is plausible from the work geometry.
Each full target invocation evaluates 16 verification positions against about
131K cached positions in each of Gemma's five global `D=512` attention layers:
about 10.5 million query-position/cache-position pairs, or 167.8 million
`D=512` dot products after accounting for 16 query heads.
Three-tier LOD instead starts from about `16*sqrt(T) = 5,793` coarse entries,
keeps 512 exact-local positions, and visits page summaries and exact leaves
only inside routed regions. That is more than a 20x reduction in the regular
history scan before the selected-page work.

The reduction cannot become a 20x whole-cycle speedup. The other 25
sliding-attention layers, projections, MoE blocks, normalization, logits,
DFlash proposal, rejection/sampling, and scheduler work remain. LOD also adds
coarse and page routing, reductions, and less regular indexed leaf loads. The
measured 72.210-to-20.243-ms change says that replacing the five long global
attention paths removes at least 51.967 ms, or 72% of the full cycle; the
remaining roughly 20 ms is the unchanged work plus LOD's own routing cost.

### Batch eight

| context | full prefill (s) | LOD prefill (s) | prefill speedup | full decode (ms) | LOD decode (ms) | decode speedup | full verifier cycle (ms) | LOD verifier cycle (ms) | cycle speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 1.808 | 2.136 | 0.85x | 10.541 | 9.825 | 1.07x | 17.357 | 23.549 | 0.74x |
| 16K | 4.898 | 4.947 | 0.99x | 13.799 | 11.575 | 1.19x | 20.920 | 20.337 | 1.03x |
| 32K | 12.921 | 10.777 | 1.20x | 13.844 | 9.428 | 1.47x | 26.570 | 20.724 | 1.28x |
| 64K | 40.135 | 22.839 | 1.76x | 41.769 | 10.773 | 3.88x | 40.226 | 20.617 | 1.95x |
| 128K | 138.830 | 46.109 | 3.01x | 55.819 | 12.655 | 4.41x | 57.787 | 19.415 | 2.98x |

The very large B1 decode ratios at 64K and 128K, and the B8 ratios at 64K and
128K, are serving outcomes from favorable greedy/acceptance trajectories, not
direct claims of equivalent kernel speedups. Prefill and the verifier-cycle
columns are the cleaner attention-side measurements. The B8 cycle ratio is
still workload-dependent because the two paths can leave different numbers of
active request rows in a launch; see the per-repeat active-row counts in the
raw cycle-accounted records.

## Compatibility work

- Target-only Gemma config overrides no longer rewrite the Qwen-shaped DFlash
  draft config into a nonexistent architecture.
- The draft preserves Gemma embedding scaling and final-logit soft-capping,
  while disabling the target's multimodal-prefix mask on draft layers.
- Rejected target suffix positions write PAD draft-cache slots. Fully rejected
  rows use a safe next-position anchor.
- Mixed prefill/decode scheduling may recover host metadata from device-local
  exact K/V only when every LOD layer proves the required suffix is present.
- Descriptive weight-cache IDs that would overflow `sockaddr_un` are compacted
  to a readable prefix plus a stable digest.
- Execution audits distinguish the configured panel batch from smaller row
  signatures that legitimately execute while a speculative batch drains.

## Raw records

- Full speed: `full_tp1_b1_8k128k_r3_d256_final.json` (cluster 12333) and
  `full_tp1_b8_8k128k_r3_d256_final.json` (cluster 12331).
- Cycle-accounted full speed:
  `full_tp1_b1_8k128k_r3_d256_cycles.json` (cluster 12369) and
  `full_tp1_b8_8k128k_r3_d256_cycles.json` (cluster 12375).
- LOD speed: `lod3_top4_tp1_b1_8k128k_r3_d256_final.json` (cluster 12351),
  `lod3_top4_tp1_b8_8k128k_r3_d256_final.json` for B8 8K/16K (cluster 12352),
  `lod3_top4_rows64_tp1_b8_8k64k_r3_d256_final.json` for B8 32K/64K
  (cluster 12356), and
  `lod3_top3_rows64_tp1_b8_128k_r3_d256_final.json` (cluster 12360).
- Cycle-accounted LOD speed:
  `lod3_top4_tp1_b1_8k128k_r3_d256_cycles.json` (cluster 12374),
  `lod3_top4_tp1_b8_8k16k_r3_d256_cycles.json` (cluster 12377),
  `lod3_top4_rows64_tp1_b8_32k64k_r3_d256_cycles.json` (cluster 12378), and
  `lod3_top3_rows64_tp1_b8_128k_r3_d256_cycles.json` (cluster 12379).
- Full NIAH-S3: `full_tp1_b8_niah8_8k128k.json` (cluster 12319).
- Selected LOD NIAH-S3: `lod3_top4_tp1_b8_niah8_8k128k_final.json` for 8K and
  16K (cluster 12349), `lod3_top4_rows64_tp1_b8_niah8_32k64k.json` for 32K and
  64K (cluster 12355), and `lod3_chunk64_tp1_b8_niah8_128k.json` for fixed
  top-three 128K.
- Rejected 64-row short-context control:
  `lod3_top4_rows64_tp1_b8_niah8_8k16k.json` (cluster 12364).
- Rejected adaptive control: `lod3_adaptive_rows64_tp1_b8_niah8_128k_final.json`
  (cluster 12359).
- Conservative 128K quality control:
  `lod3_top4_tp1_b8_niah8_128k_final.json` (cluster 12353).

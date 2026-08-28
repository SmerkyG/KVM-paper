# Fixed-list block-masked page-size-one decode

This experiment implements the fixed centroid-major index-list design for
two-tier LoD decode. The physical list is rebuilt only with the 256-token state
update and contains local/sink entries, every coarse entry, and every valid
leaf. Decode routing changes masks, not indices.

## Kernel path

1. Existing grouped coarse scoring and top-8 GQA-union routing stamp opened
   centroids.
2. `_materialize_aiter_fixed_route_mask_kernel` resolves those stamps in
   parallel into one `uint8` lane mask per fixed-list entry and one `uint8`
   nonempty flag per N=64 block.
3. `kernel_page1_attention_3d_bias_fixed_mask` runs the AITER-shaped M16/N64
   online-softmax path over 128 independent segments. It reads the block flag
   first and skips lane-mask, index, K/V, QK, and PV work for empty blocks.
4. The existing segment/LSE reducer writes the final result. No per-token leaf
   compaction, opened-coarse subtraction, separate local branch, or branch
   merge is used.

## Correctness

The production Qwen geometry (`B=8`, two KV heads, GQA4, `D=256`, 4,352 state
slots, irregular/underfull posting lists) differs from a direct attention
reference by at most `3.30e-4` and `4.49e-5` mean absolute error:
`micro_qwen64_geometry_prepared_mask_seg128.json`.

The real vLLM Qwen3.5-0.8B run, with chat formatting, scored 8/8 on NIAH-S3 at
64K: `qwen08_vllm_niah_s3_64k_s8.json`. Its execution audit records both the
fixed-mask and unified AITER paths as configured and executed.

## Speed

All decode numbers below are batch 8 at 64K on gfx942 with real ProLong text.

| Path | ms / batch decode step | Relative to full attention |
|---|---:|---:|
| Full AITER attention | 5.093 | 1.00x |
| Existing compact-list LoD | 3.797 | 1.34x |
| Fixed list, owner lookup inside attention | 8.959 | 0.57x |
| Fixed list, prepared masks + 128 segments | **3.784** | **1.35x** |

The isolated production-geometry kernel test explains the improvement. Moving
owner/stamp traversal out of the serial attention loop reduced the attention
kernel from 1,061 us to 219 us with 32 segments; mask preparation cost 39 us.
Increasing the long fixed scan to 128 segments reduced the complete synthetic
LoD call from 1.116 ms to 0.166 ms. The final vLLM result is effectively tied
with the compact-list path while eliminating its dynamic leaf-list generation.

## N=16 fast-fail sweep

N=16 was tested as a true execution tile: surviving blocks issue
`tl.dot([16,D], [D,16])`, rather than merely OR-ing four N=16 flags before an
N=64 dot. Mask preparation remains grouped over 256 logical entries.

| Fixed attention geometry | Isolated complete LoD call | 64K vLLM decode |
|---|---:|---:|
| N64, 128 segments | **0.166 ms** | **3.784 ms/step** |
| N16, 128 segments | 0.171 ms | not run |
| N16, 256 segments | **0.151 ms** | 3.836 ms/step |
| N16, 512 segments | 0.226 ms | not run |

N16/256 is 9% faster in the isolated production-geometry call, but 1.4%
slower in the complete vLLM graph. Its finer block sparsity does not repay the
additional tile and segment-reduction overhead end to end, so N64/128 remains
the default. N16 remains selectable with
`VLLM_LOD_DECODE_GQA_FIXED_MASK_BLOCK_N=16` and
`VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS=256`.

## Muse-Glimmer-30B at 64K

The same fixed-list path was exercised on Muse (`D=128`, GQA16) at batch 8
with real ProLong documents. The execution audit confirms that both the fixed
mask and unified AITER paths executed.

| Path | ms / batch decode step | Relative to full attention |
|---|---:|---:|
| Full AITER attention (historical, 3 repeats) | **19.180** | **1.00x** |
| Prior compact/unified LoD (best historical run) | 21.453 | 0.89x |
| Fixed list, N64 / 128 segments | 21.818 | 0.88x |
| Fixed list, N16 / 256 segments | 21.999 | 0.87x |

N16 is 0.8% slower than N64 on Muse and 14.7% slower than full attention.
The finer mask therefore does not solve Muse's 64K decode deficit; as on
Qwen, N64/128 remains the better production default. These are single-run
fixed-list measurements (each timed over 64 decode steps), so the sub-1%
N16/N64 difference should be treated as effectively tied rather than a robust
regression.

## Qwen final-attention-only ablation

The fixed final attention was also isolated from routing, union construction,
and mask preparation using the production Qwen geometry (`B=8`, two KV heads,
GQA4, `D=256`, fixed length 70,144). Exactly one quarter of execution tiles
were enabled, coherently at the fast-fail tile boundary. Each measurement is a
500-replay CUDA-graph average.

| Geometry | Active fraction | Attention only | Attention + segment reduction |
|---|---:|---:|---:|
| N64 / 128 segments | 0% | 0.021 ms | 0.031 ms |
| N64 / 128 segments | 25% | 0.105 ms | **0.113 ms** |
| N64 / 128 segments | 100% | 0.324 ms | 0.332 ms |
| N16 / 256 segments | 25% | 0.091 ms | **0.109 ms** |

A matched native AITER dense-attention measurement (`B=8`, QH8/KVH2,
`D=256`, 64K) takes 0.266 ms with its production-relevant page size 16.
Thus the N64 final branch with 25% active tiles is 2.36x faster than dense
attention, including its reduction. N16 reduces attention time slightly but
spends more on its 256-way reduction, leaving essentially the same total.
This ablation shifts the primary optimization target to the serialized coarse
score/top-k/union/mask-preparation chain rather than the final attention.

Primary artifacts:

- `qwen08_vllm_prolong_64k_b8_prepared_mask_r3.json`
- `qwen08_vllm_prolong_64k_b8_r3.json`
- `qwen08_vllm_compact_prolong_64k_b8_phase_r1.json`
- `micro_qwen64_geometry.json`
- `micro_qwen64_geometry_prepared_mask.json`
- `micro_qwen64_geometry_prepared_mask_seg128.json`
- `micro_qwen64_geometry_n16_seg128.json`
- `micro_qwen64_geometry_n16_seg256.json`
- `micro_qwen64_geometry_n16_seg512.json`
- `qwen08_vllm_prolong_64k_b8_n16_seg256_r3.json`
- `qwen08_vllm_niah_s3_64k_s8.json`
- `muse30b_vllm_prolong_64k_b8_n64_seg128_r1.json`
- `muse30b_vllm_prolong_64k_b8_n16_seg256_r1.json`
- `qwen64_final_only_n64_quarter.json`
- `qwen64_final_only_n16_quarter.json`
- `qwen64_full_aiter_matched.json`

## Up-front reset and predicted remote mass

Resetting the previous fixed mask on a side stream before routing was correct
but slower (`0.1457 ms` versus `0.1339 ms`) because stream synchronization cost
more than the overlap saved. The production replacement distributes prefix
preparation and previous-union clearing across the existing M16/N64 route-score
grid. For top-8 this reduced the isolated chain to `0.1286 ms`, a 4.0% win.

The predicted-mass route now compares each current centroid against the prior
token's *eligible remote-coarse* LSE, rather than the final attention LSE that
also includes local, sink, and exact-leaf mass. Its first-token bootstrap keeps
one current-query winner per N64 tile and therefore does not reintroduce a
global top-k barrier. It scored 64/64 on Qwen3.5-0.8B NIAH-S3 at 8K; the old
denominator scored 5/8 in the matching smoke test.

Two more serialized launches were removed after that quality result:

1. The current remote LSE is reduced and stored inside the final segment
   reducer instead of by a separate kernel.
2. The final reducer clears the route queue and advances its epoch for the next
   token. The next route therefore starts with a prepared queue and no reset
   launch.

| Qwen production-geometry chain | Isolated ms |
|---|---:|
| Fused top-8 reset/preparation | 0.1286 |
| Corrected predicted mass, separate LSE store/reset | 0.1108 |
| Predicted mass, fused LSE store | 0.1103 |
| Predicted mass, next queue prepared by final reducer | **0.1051** |

The final isolated path has four launches: predicted route plus fixed-mask
preparation (`23.0 us`), current-union activation (`2.7 us`), fixed masked
attention (`72.8 us`), and final output/remote-LSE/next-epoch reduction
(`6.6 us`). Its direct-attention reference error is unchanged at `3.20e-4`
maximum and `4.39e-5` mean absolute error.

With real ProLong text at 64K, batch 8, the same audited vLLM path takes
**3.610 ms per decode step**. This is faster than the immediately preceding
corrected-predictor run (`3.656 ms`), the historical recursive best
(`3.637 ms`), and full AITER attention (`5.093 ms`). The post-lifecycle-change
NIAH smoke test remains 8/8.

Additional artifacts:

- `micro_qwen64_preroute_reset_v1.json`
- `micro_qwen64_fused_preroute_v2.json`
- `micro_qwen64_predremote_fixed_v1.json`
- `micro_qwen64_predremote_fixed_fusedreduce_v2.json`
- `micro_qwen64_predremote_fixed_nostartreset_v4.json`
- `qwen08_predremote_fixed_niah_s3_8k_s64.json`
- `qwen08_predremote_fixed_nostartreset_niah_s3_8k_s8.json`
- `qwen08_predremote_fixed_nostartreset_64k_d129_r3.json`

## Muse and Qwen3.8-27B-FP8 panel

The final predicted-remote-mass path was tested at 64K on the two larger
models with batch 8, chat formatting, 16K chunked prefill, real distinct
ProLong documents for speed, and three speed repeats over 129 decoded tokens.
The execution audits confirm that predicted-mass routing, the HIP fixed-mask
attention, and the unified AITER final path all executed. NIAH-S3 was run in
two disjoint pieces (offsets 0--7 and 8--63) and combined below.

| Model | NIAH-S3 | Prefill | Decode | Historical full decode | Decode vs full |
|---|---:|---:|---:|---:|---:|
| Muse-Glimmer-30B | **64/64** | 54.507 s (9,619 tok/s) | 21.781 ms/step | 19.180 ms/step | 0.881x |
| Qwen3.8-27B-FP8 | **64/64** | 81.203 s (6,457 tok/s) | 37.254 ms/step | 52.030 ms/step | **1.397x** |

For reference, the historical matched full-prefill times were 52.363 s for
Muse and 110.565 s for Qwen, so the observed prefill rates are 0.961x and
1.362x full respectively. Predicted-mass fixed-list routing only changes
decode; these prefill differences must not be attributed to that mechanism.
The historical dense baselines used 64 timed decode tokens rather than 129,
but otherwise match the 64K/batch-8/chat/16K-chunk/three-repeat setup.

The new Muse result is effectively tied with the prior fixed top-8 result
(21.818 ms, 0.17% slower than the new result) and remains slower than full
attention. The Qwen result is 1.30% slower than the prior best LoD panel
(36.776 ms), although it remains substantially faster than full attention.
On the matched ProLong speed prompts the predictor opened 28.79 centroids on
average for Muse versus 8.58 for Qwen. Thus the 1/16 cutoff approximates top-8
on Qwen, while on Muse its larger union gives back the benefit of removing the
serialized top-8 selection in additional exact-leaf work.

A matched Muse speed-only run tightened the cutoff from 1/16 to 1/8:

| Muse cutoff | Mean opened centroids | Decode |
|---|---:|---:|
| 1/16 | 28.79 | 21.781 ms/step |
| 1/8 | 15.71 | **21.687 ms/step** |

The 1/8 cutoff reduces the opened-centroid union by 45.4% but improves decode
latency by only 0.43%. It remains 13.1% slower than the 19.180 ms full-attention
baseline. This shows that excess exact-leaf work is not the dominant remaining
Muse cost in this fixed-list formulation; the mostly fixed routing, mask,
indexed-scan, and reduction costs dominate the cutoff-sensitive portion.

Panel artifacts:

- `muse30b_predremote_fixed_64k_s8_d129_r3.json`
- `muse30b_predremote_fixed_niah_s3_64k_s56_offset8.json`
- `muse30b_predremote_mass8_fixed_64k_d129_r3.json`
- `qwen38_27b_predremote_fixed_64k_s8_d129_r3.json`
- `qwen38_27b_predremote_fixed_niah_s3_64k_s56_offset8.json`

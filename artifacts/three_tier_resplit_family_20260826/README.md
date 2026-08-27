# Three-tier re-split family panel

> **Phi prefill update (2026-08-26):** the 40.564-second Phi prefill row below
> is superseded. Recursive D128/GQA4/KVH2 prefill now uses 4,096-token state
> updates and the expert/MFMA complete-centroid consumer, measuring 34.788
> seconds through final automatic dispatch. Decode still uses the re-split
> recursive page route. See
> `artifacts/three_tier_phi_prefill_20260826/README.md`.

This panel decides when the allocation-free re-split state route should replace
the grouped/fused state route in recursive three-tier LOD. All primary speed
runs use vLLM, batch eight, real distinct ProLong text, 16K scheduler chunks,
BF16 LOD state, top eight, and the current production prefill/page/local/leaf
kernels. The route override is the only intended decode difference.

## Full-model 64K result

Times are medians of three runs. Prefill is seconds for eight 64K prompts;
decode is milliseconds per batch-eight step.

| Model | Fused prefill | Re-split prefill | Fused decode | Re-split decode | Decode change | Automatic choice |
|---|---:|---:|---:|---:|---:|---|
| Gemma-4-26B-A4B | 19.175 s | 19.864 s | 9.365 ms | **8.940 ms** | **-4.54%** | fused (quality) |
| Qwen3.8-27B-FP8 | 67.203 s | 66.912 s | 35.985 ms | **34.883 ms** | **-3.06%** | re-split |
| OLMo-3-1125-32B | 74.795 s | 74.493 s | 29.125 ms | **27.382 ms** | **-5.98%** | fused (quality) |
| Muse-Glimmer-30B | **54.009 s** | 54.057 s | **19.133 ms** | 19.255 ms | +0.64% | fused |
| Phi-4 TP5 | 40.737 s | 40.564 s | 11.509 ms | **9.998 ms** | **-13.13%** | re-split |

The state-route backend is decode-only, so the small prefill differences are
run-to-run variation rather than a causal prefill change. Gemma's prefill
samples were visibly less stable; no prefill policy is selected from that
column. Phi's fused control was independently repeated at 11.557 ms. Its older
hierarchical grouped schedule reached 10.130 ms, still slower than re-split.

OLMo is one case where latency alone does not select the default: on the
same eight 64K NIAH-S3 examples, re-split scored 7/8 while fused and full
attention both scored 8/8. The automatic policy therefore retains fused
routing; re-split remains available as an explicit speed/diagnostic override.
Gemma has the same sensitivity: re-split scored 63/64 and missed sample 43,
while a matched fused run over samples 40-47 scored 8/8 and passed that case.
It therefore also retains fused routing automatically.

Historical matched 64K full-attention decode latencies were 11.694 ms for
Gemma, 52.030 ms for Qwen3.8, 30.481 ms for OLMo, 19.215 ms for Muse, and
9.970 ms for Phi. Thus the selected three-tier route is faster than full
attention on the first three, essentially tied on Muse, and essentially tied
on Phi. Historical full-attention prefill times were 40.063, 110.565, 67.892,
51.933, and 28.119 seconds in the same model order.

| Model | Historical full decode | Selected three-tier decode | Speedup |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 5.822 ms | 2.369 ms re-split | **2.46x** |
| Gemma-4-26B-A4B | 11.694 ms | 9.365 ms fused | **1.25x** |
| Qwen3.8-27B-FP8 | 52.030 ms | 34.883 ms re-split | **1.49x** |
| OLMo-3-1125-32B | 30.481 ms | 29.125 ms fused | **1.05x** |
| Muse-Glimmer-30B | 19.215 ms | 19.133 ms fused | **1.00x** |
| Phi-4 TP5 | 9.970 ms | 9.998 ms re-split | **1.00x** |

## Two-tier comparison

The following compares the selected three-tier route with the fastest
high-quality two-tier reference at 64K/B8. A negative final column favors
three-tier. Unrestricted top-eight is used uniformly; lower-latency static
cohorts are excluded because their quality evidence is not comparable.

| Model | High-quality two-tier decode | Selected three-tier decode | Three-tier versus two-tier |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 3.077 ms, unrestricted top-8 | **2.369 ms, re-split** | **-23.01%** |
| Qwen3.8-27B-FP8 | 36.247 ms, unrestricted top-8 | **34.883 ms, re-split** | **-3.76%** |
| Gemma-4-26B-A4B | 10.421 ms, unrestricted top-8 | **9.365 ms, fused** | **-10.13%** |
| Phi-4 TP5 | 10.913 ms, unrestricted top-8 | **9.998 ms, re-split** | **-8.39%** |
| Muse-Glimmer-30B | 19.349 ms, unrestricted top-8 | **19.133 ms, fused** | **-1.12%** |
| OLMo-3-1125-32B | **28.878 ms, unrestricted top-8** | 29.125 ms, fused | +0.86% |

Muse's unrestricted two-tier
top-eight route scored 64/64 at 19.349 ms, while selected three-tier scored
64/64 at 19.133 ms, making three-tier 1.12% faster. OLMo unrestricted
two-tier and selected three-tier both scored 54/64; their latencies were
28.878 and 29.125 ms respectively, making three-tier 0.86% slower. The faster
OLMo static result had only a 7/8 screen and is not the high-quality reference.
Qwen0.8 and Qwen3.8 are directly
quality matched at 64/64 and favor three-tier. Phi's task fails under full
attention too. Gemma two-tier scored 62/64; three-tier re-split scored 63/64,
and the selected fused route corrected that run's sole miss in the matched
sample block.

## Capacity policy

Re-split has a nearly fixed launch floor, while the grouped producer grows with
the allocated state field. `VLLM_LOD_RECURSIVE_STATE_ROUTE_BACKEND=auto` uses
the following measured batch-eight policy:

| Per-rank attention geometry | Representative model | Re-split at request capacity | Otherwise |
|---|---|---:|---|
| D128 / GQA4 / KVH2 | Phi-4 TP5 | all measured capacities | fused override remains available |
| D128 / GQA5 / KVH8 | OLMo-3-32B | never automatically (quality guard) | fused |
| D256 / GQA4 / KVH2 | Qwen3.5-0.8B | >=65,536 | fused |
| D256 / GQA6 / KVH4 | Qwen3.8-27B | >=22,528 | fused |
| D512 / GQA8 / KVH2 | Gemma-4-26B-A4B | never automatically (quality guard) | fused |
| D128 / GQA16 / KVH2 | Muse-Glimmer-30B | not through 128K | fused |

The decision is made from allocated request capacity when the graph-safe pool
is constructed. It deliberately is not switched per token. Explicit `fused`
and `resplit` settings bypass the table.

The isolated production-kernel screens cover 8K, 16K, 18K, 20K, 22K, 24K,
32K, 48K, 64K, 96K, and 128K. They also verify exact top-eight selection and
show that fused tile top-k/LSE plus the combined reducer is the best re-split
fusion on every geometry. The existing score-QK schedules remain optimal:
N64 for D128, N32 for D256, and N16 for D512. Eight normalized-PV splits remain
the uniform setting; Gemma's isolated four-split advantage is only about
3.4 microseconds per LOD layer and is not large enough to justify another
production option.

## Quality and integration checks

- Qwen3.5-0.8B re-split: 64/64 on chat-formatted 64K NIAH-S3.
- Qwen3.8-27B-FP8 re-split: 64/64 on chat-formatted 64K NIAH-S3.
- Muse's selected fused route scored 64/64 on chat-formatted 64K NIAH-S3.
- Gemma's completed re-split panel scored 63/64; the matched fused
  samples 40-47 scored 8/8 and corrected its lone miss, so automatic dispatch
  now keeps Gemma fused.
- OLMo re-split scored 7/8 on the raw-prompt 64K smoke, versus 8/8 for both
  fused routing and full attention, so automatic dispatch keeps it fused.
  The completed fused panel scored 54/64 versus the historical full-attention
  64/64; fused avoids an additional route regression but does not cure OLMo's
  broader LOD quality gap.
- Phi re-split scored 0/8 on chat-formatted 64K NIAH-S3, consistent with the
  historical 0/64 full-attention result; this task is not a useful Phi quality
  discriminator, so its automatic choice is based on speed and route parity.
- Every isolated geometry preserves the exact top-eight route set; coarse
  output differences are only the expected BF16 reduction-order noise.
- The vLLM external-cache contract passes with the new automatic table.
- Gemma exposed a separate IPC weight-cache issue: unquantized MoE weights were
  mapped after daemon-side conversion but the client's small Python runtime
  kernel object was absent. The loader now reconstructs only that object,
  without re-shuffling shared weights, and the cached Gemma smoke run passes.

Raw results are the JSON files in this directory. The `isolated_all_*` files
contain the route stage and fusion sweeps; `*_fused_64k_b8_r3.json` and
`*_resplit_64k_b8_r3.json` are the matched full-model runs.

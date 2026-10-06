# Corrected K3 decode diagnosis: 12 MLA layers

This is the one-wave baseline diagnosis. The subsequent
[four-wave router optimization](ROUTER_OPTIMIZATION.md) removes its private
scratch spills and reduces captured-fixture decode latency by 31%/20% at
B1/B8 without changing the attention calculation.

October 6, 2026. Node 3, eight MI325X GPUs, TP8/DCP8, 65,536-token
prompts, 12 consecutive native K3 MLA layers, no KDA or FFN/MoE. Geometry
is unchanged: 96 query heads, 512 latent plus 64 direct-key channels, and
512 latent value channels. Dummy weights and deterministic synthetic token
IDs make this a **performance diagnostic, not a quality benchmark**.

## Serving measurements

One complete shape warmup, then one uninstrumented measured pass. Each
request produces 1,026 frozen continuation tokens: 1,025 timed decode
steps. Every LoD layer on every rank records four **global per-request
256-token updates**, including four updates of all eight rows at B8.
Batch rows finish together with no preemptions or prefix-cache hits.
Decode uses `FULL_DECODE_ONLY` CUDA graphs, not an eager fallback.

| Live batch | Dense Gluon, ms/step | Ordinary two-tier LoD, ms/step | LoD / dense latency |
|--:|--:|--:|--:|
| 1 | 2.116 | 3.736 | 1.766x |
| 8 | 3.950 | 5.893 | 1.492x |

Dense must use the same attention-only fixture. The initial `b1-full.json`
and `b8-full.json` controls accidentally retained the fixture's tiny
128-channel FFN and second AttnRes branch; **do not use them as matched
controls**. Dense registration now installs the config-gated fixture
wrapper too. This does not remove FFNs from ordinary checkpoints or change
their archived dense timings. The replacement controls have the
`full-attention-only` filename suffix and an eight-worker fixture audit.

## What is slow

The largest avoidable cost is centroid routing, not cache updates. These
are **rank-0 GPU kernel durations from a separate 257-step trace**, divided
by the actual replay count. They are diagnostic averages, not subtractive
attention-core latency or additional serving measurements. The profiler
is absent from the canonical measurement above.

| LoD stage, summed kernel ms per 12-layer step | B1 | B8 |
|:--|--:|--:|
| Centroid scoring / local candidates | 1.260 | 1.562 |
| Exact leaves + closed summaries + local attention | 0.283 | 1.300 |
| Attention split reduction | 0.138 | 0.330 |
| Local top-eight reduction | 0.052 | 0.061 |
| Gathered-rank top-eight reduction | 0.051 | 0.052 |
| Per-GQA selected-centroid union | 0.056 | 0.055 |
| Compact page-descriptor construction | 0.058 | 0.078 |

`_decode_route_coarse_gqa_groups_kernel` alone averages **105.03 us per
layer at B1 and 130.19 us at B8**. Its K3 geometry is one wave, 16 query
heads, and 64 centroid columns. The current compiled variant contains
**512 VGPRs, 603 reported VGPR spills, 130 SGPR spills, and 1,592 bytes of
private scratch per thread**, with 32 KiB LDS. The artifact compiled at
20:36:14 UTC during this run uses the real 512+64 products, not a padded
1,024-channel score. See [compiled resource record](oct6-corrected-decode-profile12/route-codegen.json).

The measured hot kernel plus this resource pressure makes the router's
tile/layout the first optimization target. Register spilling is confirmed;
how much latency it causes still requires a matched geometry experiment.
Changing only allocation/reuse cannot remove this kernel's on-device work.

There is also an extra top-eight exchange per layer: the LoD trace has
37 all-gathers per step versus 25 in dense (12 layers plus final sampling).
Query gathering and LSE merging are required by ordinary DCP; the third
per-layer exchange combines local routing winners. Torch casts, packing,
localization and reductions add small kernels around that exchange.

The trace's update boundary increases the inter-graph launch interval from
a typical 4.017 to 28.257 ms at B1, and 6.181 to 33.654 ms at B8. Its excess
is roughly 24/27 ms once per 256 steps: about 0.095/0.107 ms amortized,
**only a diagnostic estimate**, including the instrumented host interval.
Ordinary graph steps are already slow, so updates do not explain the main
regression. Rare update-associated GPU kernels total about 1.96/2.58 ms
over the traced update, not milliseconds per generated token.

At B8, the live fixture snapshot has 512 occupied centroids per row/rank,
roughly 16 leaves per centroid, maxima 253--256, and no centroid above
the 1,024-leaf cap. The final layer's shared union scratch contains up to
eight selected centroids / 2,048 leaves per GQA tile. This workload is
not representative of trained key distributions, but it rules out
oversized-centroid closure as the explanation for this fixture's floor.
The B1 post-request population snapshot was taken after row release and
is empty; it is **not** evidence of an empty cache during measurement.

## One request per GPU at B8

This has already been tested on the full trained model, with the corrected
queries, captured decode, 1,025 steps and all four updates:

| Context | Dense | Ordinary DCP8 LoD | One-request-per-GPU LoD |
|--:|--:|--:|--:|
| 16K | 31.649 | 37.933 | 33.699 |
| 64K | 34.198 | 38.209 | 34.148 |

All entries are ms per batched step. Owners improved LoD by about 11%,
but remained slower than dense at 16K and effectively tied dense at 64K.
They avoid distributed top-eight/LSE combination by attending to the
owner's full history, while retaining query/output transport and native
TP/MoE/KDA. The centroid router is still used. These full-model numbers
are reused observations, **not new runs or fixture extrapolations**.
Details: [captured owner decoder](../kimi-k3-full-model-current/OWNER_CAPTURED_DECODE.md).

## Reproduce

From the repo root in the pinned K3 v10 environment, with eight GPUs.
No weight daemon or full-model checkpoint is needed. Compilation stays
on local `/tmp/dan-agent` storage. Repeat for B1/B8 and `full`/`two-tier`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_CACHE_AUTOTUNING=1 \
HSA_NO_SCRATCH_RECLAIM=0 \
bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_decode_fixture \
  --mode two-tier --batch-size 8 --layers 12 --length 65536 \
  --output results/kimi-k3-mla-stack/decode-fixture-b8-lod.json
```

The benchmark validates the fixture structure, four updates, frozen output
tokens, and 257 actual diagnostic graph replays on each of eight ranks.
The profiler starts at the first **real decode replay**, after prefill and
capture, and stops before taking the live cache-population snapshot.
Chrome traces are separate local diagnostic artifacts; overlapping kernel
durations and enclosing annotation ranges are never summed as wall time.

Host tests: 274 passed, 40 GPU-only tests skipped. All four GPU benchmark
cases completed, including the corrected dense fixture audits. This turn
changes diagnostic tooling and the private fixture hookup only; it does
not promote an unmeasured production router optimization or change LoD math.

## Raw measurements

- [B1 LoD](oct6-corrected-decode-profile12/b1-lod.json).
- [B8 LoD](oct6-corrected-decode-profile12/b8-lod.json).
- [Corrected B1 dense](oct6-corrected-decode-profile12/b1-full-attention-only.json).
- [Corrected B8 dense](oct6-corrected-decode-profile12/b8-full-attention-only.json).

# Compact selected-leaf projection (October 6)

This is an opt-in prefill optimization on `lod-k3`, not a different LoD
approximation. It projects the union of leaves from the **actual post-cap
top-eight routes**, once per head and centroid, then gives the existing
expert-major attention kernel a contiguous range for each selected centroid.
The centroid assignments, leaf order, routing scores, 1,024-leaf closure
rule, softmax/LSE replacement and global 16K/256-token cadences are unchanged.
Decode does not use this projection path.

## Implementation

1. Reuse the route counts that also pack the queries for expert attention.
2. On the GPU, prefix-scan the lengths of selected centroids. Empty or
   unselected centroids receive an empty range; centroid ranges have no padding.
3. Project tiles from that compact list. A tile may cross centroid boundaries
   within a head but never crosses heads, so it uses one head's K/V matrices.
4. Read those ranges directly during fine attention, eliminating its virtual
   page lookup and chronological leaf-index gather.

The projection is D512 latent to D128 key/value, retaining the D64 direct-key
part. It uses FP32 accumulation and BF16 output, like the previous GEMMs.
There is no CPU read of selected counts, dynamic host-sized output allocation,
or projection once per query. The reusable output arena still reserves the
worst-case `B * heads * archived_tokens` rows; only its selected prefix is
written. This is **not selected-only VRAM allocation**.

The previous projection retained 192+128+128 BF16 channels per leaf/head;
the compact projection retains 192+128. That reduces this projection arena's
capacity by 28.6%, not total attention-cache VRAM. Replicated prefill archives,
rank-local persistent decode pools and other workspaces remain separate costs.

## Checks and measured results

GPU tests cover distinct batches/heads, strided inputs, empty and uneven
centroids, dense and paged directories, all leaves selected, buffer reuse,
and graph replay after changing the selection. An independent dense reference
caught an existing batch-stride bug in the ordinary projected-leaf consumer:
it incorrectly derived a batch offset from the head stride. The consumer now
uses independent batch and head strides. B1 addressing is unchanged.

### Captured trained-model leaf stage

Real late-prefill inputs: `[B,H,Q,D] = [1,12,16384,192]`, 16,127 archived
latents, 2,048 active centroids. The complete measured stage includes route
counting, projection, query packing, exact leaf attention and LSE reduction.
It excludes the rest of the model. A/B/A ordinary controls bracket the
candidate; five kernel samples check variance, not full-model repetitions.

| Method | Complete leaf stage (ms) | Output max error | LSE max error |
|:--|--:|--:|--:|
| Ordinary, before | 1.2050 | — | — |
| Compact, 64-row projection tile | 0.9877 | 0 | 0 |
| Ordinary, after | 1.2025 | — | — |

This is 1.219x stage throughput, or 17.9% less stage latency. It is **not an
18% full-model prefill gain**. The selected union contains 55,584 of 193,524
leaf/head pairs (28.7%). Outputs and LSE are bitwise identical on this capture.
The output allocation still has room for all 193,524 pairs (118.1 MiB).
The separate 0.1994 ms count-plus-projection diagnostic is not an additive
model-time attribution.

Source: [unpadded trained leaf replay](../kimi-k3-mla-stack/oct6-compact-selected-trained-unpadded-leaf-stage.json).
The earlier padded prototype is superseded and is not a release candidate.

### 24-layer MLA-stack fixture

TP8/DCP8, B1, one warmed measured pass per point and A/B/A controls. This
fixture has no MoE/FFN or KDA; its synthetic prompts are not a quality test.

| Context | Ordinary before (s) | Compact (s) | Ordinary after (s) |
|--:|--:|--:|--:|
| 32K | 0.4108 | 0.4051 | 0.4123 |
| 64K | 0.9767 | 0.9363 | 0.9806 |
| 128K | 2.3466 | 2.1580 | 2.3474 |

The 128K fixture latency is 8.1% lower. Generated tokens match all three
variants, and the loaded top-eight routing binaries and compact calls passed
the eight-rank audit. Because this A/B/A process retains both projection
workspaces, its memory readings are **not** an isolated VRAM comparison.

Source: [fixture A/B/A](../kimi-k3-mla-stack/oct6-compact-selected-fixture-b1-aba.json).

### Full trained model, node 2

B1, TP8/DCP8/EP8, resident full-model weights, real frozen ProLong prompts,
one shape-matched warmup and one measured pass, 2 GiB native cache per rank,
524,297-token engine capacity. The ordinary/dense controls are reused from
the completed node-2 comparison with matching prompt hashes and configuration.
All three generated first tokens match the ordinary LoD control; this is a
numerical smoke check, not a retrieval/quality evaluation.

| Context | Dense (s) | Ordinary LoD (s) | Compact LoD (s) |
|--:|--:|--:|--:|
| 32K | 4.1858 | 4.1891 | 4.1836 |
| 128K | 19.2996 | 17.3906 | 17.2793 |
| 256K | 45.4965 | 37.1120 | 37.0476 |

The compact version is effectively tied overall: its largest gain here is
0.64% at 128K, not the 8.1% fixture gain. There is no demonstrated reduction
in full-model peak live Torch allocation: ordinary/compact peaks are
16.49/17.41, 20.70/21.34 and 26.10/26.37 GiB at 32K/128K/256K respectively.
These client-allocation readings exclude daemon-owned IPC weights and
driver scratch, and do not establish a total-VRAM regression or saving.
The smaller projection arena does not by itself solve B8/256K capacity.
The compact path therefore remains opt-in, not a new default.

Sources: [compact full-model check](oct6-compact-selected-b1-prefill-node2.json),
[ordinary control](oct5-owner-b1-control-two-tier-node2-r2.json),
[dense control](oct5-owner-b1-control-full-node2-r2.json).

## Reproduction

From the repository root, with the K3 v10 environment already installed:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m pytest -q \
  tests/test_kimi_compact_projection.py

bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_compact_projection \
  --input results/kimi-k3-full-model-current/trained-prefill-leaf-input.pt \
  --projection-tiles 16 32 64 --repeats 5 \
  --output results/kimi-k3-mla-stack/compact-selected-replay.json

bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_owner_tune \
  --checkpoint tests/fixtures/kimi-k3-mla-stack \
  --lengths 32768 65536 131072 --batch-size 1 \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --kv-cache-memory-bytes 2147483648 \
  --variants current current_compact current \
  --output results/kimi-k3-mla-stack/compact-selected-fixture-aba.json
```

For a full-model run, retain the ordinary runner's settings and set
`LOD_KIMI_COMPACT_SELECTED_PROJECTION=1`. Use the same daemon, logical context
capacity, native-cache reservation, frozen prompt hashes and warmup policy as
the control. Compiler caches must stay on local disk; the direct launcher
sets their paths under `/tmp/dan-agent` unless explicitly overridden.

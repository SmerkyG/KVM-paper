# K3 first-24 attention-stage fixture

This matches the first 24 decoder layers' attention placement and dimensions
in the downloaded official Kimi-K3 config: each 12-layer stage contains nine
KDA and three MLA layers, with MLA at one-based layers 4, 8, 12, 16, 20, 24.
KDA retains its width-four convolutions, full-rank gates, 96 128-d heads,
recurrent cache, and gated output normalization. MLA retains 96 heads, a
1536-d query latent, a 512+64 key record, and 128-d projected values. The
attention-side AttnRes block size is 12.

The benchmark initializes random normalized weights, not trained K3 weights.
KDA uses `A_log=0` and `dt_bias=0` in both layouts. Its native cache reserves
row 0 for padding; real fixture request states start at row 1.
Feed-forward branches, embedding execution, and LM head execution are excluded
in both layouts. The tiny structurally required MLP is replaced at construction
by the existing `lod_attention_only_fixture` mechanism. Results are layout and
attention-stage performance evidence, not model-quality or global EP8 throughput
evidence. No persistent full-model daemon weights are loaded or changed.

Run on eight otherwise idle MI325X GPUs in the v10 userspace:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LOD_KIMI_SUBTILE64=score LOD_KIMI_CHUNK_TILE_PACK=1 \
LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 LOD_KIMI_SORT_LEAF_ROUTES=1 \
LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
LOD_KIMI_TILE_REFINE=1 LOD_KIMI_DIRECT_LEAF_RESULT=1 \
LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
HSA_NO_SCRATCH_RECLAIM=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_two_stage_prefill \
  --tensor-parallel-size 8 \
  --lengths 32768 65536 --batches 1 8 \
  --output results/kimi-k3-mla-stack/first24-two-stage.json
```

The speed control is **full attention under TP8**, using native AITER causal
FlashAttention on expanded D192 keys/queries and D128 values. Every prior
MLA token remains in the dense field; there are no centroids or LoD routing
in this baseline. It uses replicated latent history, not DCP. Compare
it with sequential owner-local LoD stages and two-slot pipelined owner-local
stages. Each owner keeps complete projections only for its 12-layer stage.
Only ranks 0 and 1 execute the two owner stages; the other six ranks remain
inactive during the owner comparison. The full eight-rank control still runs
all 24 layers on every rank. Original TP8 shards remain resident; peak VRAM therefore does
not represent a final repartitioned-weight memory saving. No CUDA graphs are
used in any variant. Weight assembly and metadata construction are untimed.

The receiver retains the original input rows, which are exactly bank entry 0.
Only the completed first-block hidden state is transferred. Pass
`--include-input-bank` to additionally send bank entry 0 (448 instead of
224 MiB per 16K chunk) without changing arithmetic. This first-boundary
optimization does not imply that later stages can skip unseen residual banks.

Outputs must be finite. An **untimed** TP8 LoD numerical control isolates
weight repartitioning/rounding from the intentional full-to-LoD approximation;
the predeclared 5% relative-L2 bound applies to that same-LoD arithmetic check,
not to full-attention equivalence. Differences from the dense output are
reported separately and are not trained-model quality evidence. The native
dense kernel's continued-prefill causal alignment is checked against a small
FP32 oracle before timing. Pipelined and sequential owner outputs must be
bitwise equal. Every MLA cache is reported with its final per-row sequence
length and state coverage. Timing uses one exact-shape warmup and one measured
pass, synchronized only at the interval boundaries. The reported elapsed time
is the maximum across all TP workers, including fill/drain and cache reset and
construction. Preallocated GPU events record each completed 16K microbatch;
the interval average after the first completion is reported separately from
cohort wall time. This is finite-cohort post-fill spacing at increasing context,
not a fixed-context infinite-pipeline rate. Logical all-reduce input bytes are counted, but must not be
described as measured network wire bytes.

`--tensor-parallel-size 2` is an explicit alternate TP geometry, not the
baseline for the proposed alternative to TP8. The archived initial TP2 pilot
compared LoD against LoD; its timings are not dense-control results.

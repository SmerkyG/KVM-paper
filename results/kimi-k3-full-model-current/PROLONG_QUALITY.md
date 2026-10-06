# Kimi K3 shared-latent centroid quality check

## Question and scope

Does the existing full-model two-tier LoD implementation preserve ProLong
next-token quality when its centroid assignments are shared across MLA
query heads? The October 5 per-request-owner speed experiment is **not** used
for this comparison. Neither is the truncated 48-layer speed fixture.

The implementation clusters the common 512-dimensional normalized latent plus
64 direct-key channels using the existing key-similarity criterion. Each
query head scores/routes the resulting centroids independently. For a fixed
leaf set, projecting its latent mean is equivalent to averaging its projected
keys/values because the projection maps are linear. This does **not** imply
that shared latent-space clustering is equally good for every head: different
head projections can emphasize different directions, and softmax is nonlinear.
Per-head or query-metric-aware clustering remains a possible alternative, not
something proven necessary or implemented by this experiment.

## Protocol

- Full trained 93-layer Kimi K3 checkpoint: 24 MLA and 69 KDA layers.
- Node 4, eight MI325X GPUs, TP8/DCP8/EP8, resident weight namespace
  `kimi-k3-shared-int4-v6`. The same converted INT4 **MoE weights** are used
  in dense and LoD modes; attention records are BF16, not INT4.
- Existing ordinary two-tier LoD: shared latent assignments, per-head
  top-eight routing, separate sink, 16K global prefill update cadence.
  No request-owner, slice-local-centroid, or head-specific-clustering variant.
- Frozen dataset `Seerkfang/prolong-64k-512-new`, revision
  `97295b7d7fe48dc0aa6ba373af3a8b9d945e505b`.
- Quality cohort offsets 8–15, corresponding to raw document indices
  **14, 19, 20, 23, 24, 25, 27, 28**. Each prompt is exactly **65,267** tokenizer
  tokens. Both arms must match document and token SHA256 digests.
- The initial 65,536-token run (`21239`) stopped before inference because raw
  document 19 has only 65,267 Kimi tokens (document 28 has 65,460). CPU preflight
  `21241` checked all eight documents. Both arms therefore use the shortest
  common length, 65,267, without padding, concatenation, or substitutions.
  This is 0.41% shorter than 64K, not an exact 65,536-token evaluation.
- Raw-document teacher forcing, without a chat template, batch size one,
  temperature zero, seed zero, one generated token. Score all 65,266 observed
  next-token probabilities per document; do not replace them with a forced
  output trace or score only the generated token.
- Also report 16K bands by the **query position** that predicts the target,
  plus the pooled loss after query position 16,383. This separates the first
  exact attention block from the region using LoD approximation.
- Run dense and LoD sequentially on the same node. Quality-run elapsed time
  includes prompt-logprob work and cold kernels; it is **not** a warm serving
  speed benchmark.

## Results

Dense completed in job `21243-kimi-k3-full-prolong-quality-8docs-65267-r2`.
The matched LoD job `21244-kimi-k3-two-tier-prolong-quality-8docs-65267`
also completed on the same node. Complete results:
[dense](oct5-full-prolong-quality-8docs-65267tokens.json) and
[two-tier LoD](oct5-two-tier-prolong-quality-8docs-65267tokens.json).
The first 65,267-token attempt (`21242`) scored one document, then stopped
because the new progress printer was missing its `json` import. The saved
[one-document partial](oct5-full-prolong-quality-65267tokens-firstdoc-reporting-failure.json)
is not used as the eight-document baseline. A regression test now executes
the complete preflight/progress/final-output path before the restart.

| Scored query region | Dense loss | LoD loss | Dense PPL | LoD PPL | Relative PPL change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole 65,267-token prompt | 0.285961252 | 0.287321475 | 1.331040879 | 1.332852623 | +0.1361% |
| After the first exact 16K block | 0.277305618 | 0.279300656 | 1.319569593 | 1.322204813 | +0.1997% |

The whole-prompt comparison scores **522,128** next-token targets; the
post-prefix comparison scores **391,056**. Losses are token-weighted pooled
negative log probabilities, and PPL is their exponential, not the mean of
document perplexities.

### Position bands

Indices are zero-based attention-query positions, with an exclusive upper
bound. Each query predicts the following corpus token.

| Query positions | Targets across eight documents | Dense loss | LoD loss | Dense PPL | LoD PPL | Relative PPL change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–16,383 | 131,072 | 0.311785516 | 0.311251750 | 1.365861705 | 1.365132851 | −0.0534% |
| 16,384–32,767 | 131,072 | 0.274822778 | 0.276018310 | 1.316297377 | 1.317871994 | +0.1196% |
| 32,768–49,151 | 131,072 | 0.262239160 | 0.263900019 | 1.299837374 | 1.301998015 | +0.1662% |
| 49,152–65,265 | 128,912 | 0.295148965 | 0.298296684 | 1.343326452 | 1.347561529 | +0.3153% |

The first exact block is not bitwise equivalent between the two execution
paths: its pooled PPL differs by −0.0534%. Consequently this experiment
measures the existing implementations end to end, not an isolated causal
effect of centroid sharing. The post-prefix loss increase is 0.00199504 nats
per token.

### Paired documents

| Raw document index | Dense PPL | LoD PPL | Relative PPL change |
| --- | ---: | ---: | ---: |
| 14 | 1.633629455 | 1.640389691 | +0.4138% |
| 19 | 1.147248363 | 1.147973627 | +0.0632% |
| 20 | 1.303876988 | 1.305007247 | +0.0867% |
| 23 | 1.190884137 | 1.191134644 | +0.0210% |
| 24 | 1.469711285 | 1.470949324 | +0.0842% |
| 25 | 1.195941497 | 1.197725230 | +0.1491% |
| 27 | 1.273638464 | 1.276386261 | +0.2157% |
| 28 | 1.512257990 | 1.513098831 | +0.0556% |

All eight document/token digests match. Final audits passed on all eight
ranks: 24 MLA layers in each arm, all 24 attached to LoD only in the LoD arm,
no dummy attention, two-tier BF16 records, eight routes in prefill/decode,
16,384-token global prefill updates and 256-token global decode updates.
Model, dataset revision, seed, cohort, scheduler budget, MoE weight namespace,
and runtime environment match except the intended LoD-enable and dense-Gluon
decode switches. Loaded LoD module manifests are retained in the final JSON.

**Interpretation:** shared-latent centroids incur a small ProLong degradation
on this cohort: +0.1361% pooled PPL, with every document between +0.0210% and
+0.4138%. This supports their use for this measured prefill setting, but does
not establish equivalence for individual heads, retrieval tasks, decode
teacher forcing, or longer contexts. No head-specific clustering control was
run, and no clustering/routing/model implementation was changed for this
comparison.

Files are written incrementally after each completed document. A `.partial.json`
is not a completed eight-document score or a passed final worker audit.

## Reproduction without the cluster runner

On the prepared node with the same unpacked Kimi v10 userspace and resident
weight daemon, run from the repository root. See the main README for the
daemon/checkpoint setup. The checkpoint is staged on local `/tmp` disk.

```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_TRITON_AWQ=1
export TRITON_CACHE_AUTOTUNING=1
export HSA_NO_SCRATCH_RECLAIM=1
export LOD_BENCHMARK_SYNC_PREFILL_CACHE=1
export AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv"

# Established full-model development kernel settings, unchanged between arms.
export LOD_KIMI_SUBTILE64=score
export LOD_KIMI_CHUNK_TILE_PACK=1
export LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024
export LOD_KIMI_SORT_LEAF_ROUTES=1
export LOD_KIMI_LEAF_BLOCK_M=64
export LOD_KIMI_LEAF_WARPS=1
export LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0
export LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1
export LOD_KIMI_PREFILL_MIN_FREE_GIB=4
export LOD_KIMI_TILE_REFINE=1
export LOD_KIMI_DIRECT_LEAF_RESULT=1

# Do not enable the separate owner/sharded/slice-local experiments.
unset LOD_KIMI_REQUEST_OWNER_PREFILL LOD_KIMI_REQUEST_OWNER_DECODE
unset LOD_KIMI_DCP_SHARDED_LEAVES LOD_KIMI_DCP_LOCAL_PREFILL
unset LOD_KIMI_DCP_SHARED_PREFILL

for mode in full two-tier; do
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.prolong \
    --measure quality \
    --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
    --mode "$mode" --length 65267 --samples 8 --sample-offset 8 \
    --batch-size 1 --tensor-parallel-size 8 --decode-context-parallel-size 8 \
    --dcp-comm-backend ag_rs --seed 0 \
    --gpu-memory-utilization 0.8 --kv-cache-memory-bytes 1073741824 \
    --kimi-gfx942-int4-moe --weight-cache --weight-cache-id kimi-k3-shared-int4-v6 \
    --allow-experimental-environment \
    --output "results/kimi-k3-full-model-current/oct5-${mode}-prolong-quality-8docs-65267tokens.json"
done
```

The development-environment allowance is explicit because these are the
existing measured K3 kernel settings, not the Qwen/K2 paper release defaults.
The final JSON includes effective worker/engine audits and the environment.

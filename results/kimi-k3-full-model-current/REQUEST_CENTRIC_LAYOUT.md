# K3 request-centric execution plan

Status: design and opt-in prototype, not a release default. Branch `lod-k3`.

## Keep ordinary tensor parallelism first

There is no requirement to replicate non-MoE weights to make attention caches
request-local. The current TP8/EP8 prototype already keeps the native tensor
partitioning of Q/K/V projections, W_O, KDA, shared experts, embeddings, and
the output head. Only MLA attention/cache computation moves to a request
owner. TP ranks send their 12-head query slice to that owner, which computes
all 96 heads against one cache, then returns each rank's output slice for
normal TP W_O. Routed experts remain distributed over all eight GPUs.

This saves duplicated cache construction and summaries. It does not remove
the projection collectives, query/output exchanges, native TP output reduction,
KDA collectives, or routed-expert communication. It is not eight independent
model replicas and does not change expert routing or attention arithmetic.

Decode can retain this layout: one fixed-address native single-GPU LoD pool
per owned request/layer, 16-head attention tiles, all 96 heads, exact separate
sink, top-eight routing, and updates every 256 **global tokens per request**.
There is no distributed top-k or attention LSE merge because one GPU has the
entire request's MLA cache. The initial prototype is eager, not whole-model
graph-captured; results must say so. A prefill cache is copied into the native
decode arena once and the old copy is released. That transition belongs in
the measured latency/memory envelope, not an untimed hidden setup.

## Weight capacity limits

Daemon metadata was inspected read-only, without mapping or loading another
copy of the trained weights. Replication factors were inferred from actual
partition widths, not a module's `tp_size` field (which can exist on replicated
modules too). Estimated additional BF16 linear-weight storage per GPU for
full TP1 non-MoE execution, keeping routed experts distributed:

| Weight category | Additional GiB/rank |
|:--|--:|
| MLA projections | 8.490 |
| KDA/recurrent projections | 50.665 |
| Shared experts | 19.811 |
| Other nonexpert linear weights | 1.184 |
| Total | 80.150 |

These are lower bounds: nonlinear parameters, biases, additional embedding/
head replication, activations, and caches are excluded. The daemon already
holds roughly 213 GiB/rank; an additional 80 GiB does not fit a 256-GiB GPU.
Retaining TP for these weights is therefore a practical capacity choice, not
an incompatibility with request-local attention. Normal vLLM DP+EP would
replicate the nonexpert weights; using it without a capacity calculation would
not solve this problem.

## Preferred next layout: attention stages, globally distributed MoE

User clarification (October 5): keep several attention layers, including
their Q/K/V/O weights, on one GPU; keep the intervening MoE execution over
the ordinary EP/TP group. This is **not** whole-decoder-layer pipeline
parallelism, and it does not require replicating every attention projection
on every request owner.

Partition attention layers into balanced stages. A stage owner stores the
complete projection weights and LoD caches for its assigned layers. For a
microbatch, it computes normalization/AttnRes, Q/K/V, LoD attention, and W_O
locally. It then supplies the MoE input to the distributed expert group and
receives the combined MoE result before executing its next attention layer.
The hidden state and required residual-bank state move to the next stage
owner only when the microbatch leaves the stage. Expert placement and routing
arithmetic need not change.

Compared with the current request-owner prototype, this removes TP attention
projection collectives, the exchange of head-partitioned queries and outputs,
and distributed attention routing/LSE merging. It replaces those with an
owner-to-MoE interface on every layer and a stage-boundary interface less
frequently. It does **not** eliminate expert dispatch/combine traffic or the
dependency on an intervening MoE result. B8 prefill has independent rows and
chunks that might overlap attention with other rows' expert work; an explicit
schedule is required, not an assumption that all transfer latency disappears.

Weight capacity is different from full per-request projection replication:
each attention layer's weights exist once across the owners, rather than once
on every GPU. Balanced aggregate projection storage should therefore be
similar to TP8, but stage imbalance, temporary daemon-shard gathering, KV
placement, and recurrent-attention state must be measured. The existing daemon
contains TP8 shards; changing execution layout alone does not make complete
Q/K/V/O tensors available on a stage owner.

K3-specific constraints verified in the image's native `amd/linear.py`:

- AttnRes is a bank of previous block states, not an ordinary single residual.
  Native PP transmits `residual` with shape
  `[tokens, ceil(stage_end / attn_res_block_size), hidden_size]` alongside the
  hidden state. Aligning stage boundaries with block boundaries avoids splitting
  a partial block but does not remove the previous-block bank.
- Keep that bank and both attention-side and MoE-side AttnRes mixes on the
  attention owner while it owns a stage. Experts need the mixed MoE input and
  return their output; they need not receive the bank for every MoE invocation.
- The present opt-in token-sharded bank implementation explicitly supports
  PP1 only. It cannot be reused unchanged by passing a PP flag.
- Native EP accepts its established token/collective layout. Owner-local
  tokens require an adapter; merely giving other TP ranks empty tensors is
  not a valid replacement. Shared-expert TP and latent MoE input/output
  transforms also need an explicit, equivalent interface.
- KDA layers are not LoD MLA layers. Their placement and state must be chosen
  explicitly. Keeping them TP8 is a conservative first option but introduces
  additional layout changes; putting them on stages requires balancing their
  much larger projection weights.

First test a small two-stage attention fixture with real distributed MoE,
fixed microbatch buffers, and a fixed collective order. Check numerical
equivalence, cache continuation, global 16K prefill/256-token decode cadences,
and peak VRAM before considering a full-model weight repartition. Then measure
B1 and B8 separately: pipeline throughput gains do not imply lower B1 latency.
Do not interrupt the currently running trained owner-decode benchmark or
reload its resident weight daemon to implement this design.

The first non-pipelined two-layer fixture is complete:
[attention-stage experiment](../kimi-k3-mla-stack/ATTENTION_STAGE.md).
It includes native EP8/TP8 MoE and excludes embeddings/LM head. Owner-local
Q/K/V/O is 3.5–8.1% slower than TP projections at 32K/64K with 2K/16K
chunks. This does not demonstrate a standalone speed benefit. Expert choices
are frozen to isolate ordinary BF16 W_O reduction roundoff from near-tied
random routing; it is not trained-model quality evidence. No full-model stage
layout or pipeline implementation has been promoted.

The subsequent [first-24 attention-stage pilot](../kimi-k3-mla-stack/FIRST24_TWO_STAGE.md)
uses K3's real first two 12-layer blocks (nine KDA, three MLA per owner),
without FFNs/MoE in either layout. It includes the full attention and
projection computation, not just relocating projections around owner-local
attention. At B8 it pipelines in 7.24 s at 32K and 15.38 s at 64K, versus
9.24/20.10 s for its TP2 control. Pipelined and sequential owner outputs are
bitwise equal. This is a two-GPU resource-matched pilot, **not** the required
full-attention TP8 comparison or a measured eight-stage/global-MoE throughput result. The
interval is total time including fill/drain, not steady-state-only timing.
The corrected fixture now defaults to a native AITER full-attention TP8
control and separately records post-fill chunk-completion intervals. The
historical pilot compared LoD to LoD and cannot serve as that dense baseline.

The corrected panel has now completed: first-24 dense TP8 versus the two
owner-local LoD stages takes 2.882 versus 7.279 s at B8/32K, and 6.559 versus
15.355 s at B8/64K. Post-first chunk spacing is 180.966 versus 434.787 ms,
and 206.214 versus 471.352 ms respectively. These remain attention-only
diagnostics, not a full-layout comparison. The six idle owner ranks and
omitted per-layer MoE dispatch/combine make their communication contention
unrepresentative. Do not extrapolate a production throughput claim from them.

### Required MoE-inclusive follow-up

The next layout test must retain both attention-side and MoE-side AttnRes,
the first-24 KDA/MLA placement, and native MoE execution on all eight GPUs.
Full-attention TP8/PP1 is the speed control. The owner candidate must send
each layer's mixed MoE input to the EP/TP ranks and return the expert result
to its owner **before that row advances to the next attention layer**.
Do not move MoE to the end of a 12-layer block or add a separately timed
MoE duration to this attention-only measurement.

Use one deterministic global service order for MoE invocations across
pipeline stages and rows, with fixed buffers and explicit completion events.
Overlap another row's owner-local attention with the active row's native
distributed MoE where dependencies permit. Include the real expert
dispatch/combine, shared experts, latent down/up transforms, owner-interface
transfers, bank handling, and pipeline endpoints in the timed interval.
Validate the sequential MoE-inclusive owner layout before enabling overlap.

The existing two-MLA-layer MoE pilot used only 32 experts/4 selected experts,
so it is not an official-K3 bandwidth/performance control either. Any reduced
fixture must state its expert count, active count, dimensions, quantization,
and communication backend. Production conclusions require the official
16-active/896-expert geometry and production INT4 MoE path (not the BF16
Triton stand-in), or a demonstrated match of the relevant bottlenecks.
The resident trained weights daemon should not be reloaded or re-quantized
to perform these checks.

## Alternative: replicated owner-local MLA projections

Replicate only MLA's projection weights on the owners (about 8.5 GiB/rank
extra). For MLA layers, each GPU produces its request's queries, evaluates
attention, and applies W_O locally. Gather the resulting hidden-state rows
before the next normally tensor-parallel component. KDA, shared experts,
embeddings and the output head can remain TP8; routed experts remain EP8.

This replaces the two attention head-layout exchanges plus MLA's TP output
reduction with a hidden-state layout exchange. It avoids a potentially large
96-head output transfer: the native hidden width is 7,168, while concatenated
MLA value heads have width 12,288. It still needs communication; speed gains
are a hypothesis, not a measured result. It also reduces memory available for
long-context KV caching, so it cannot be promoted based on speed alone.

The already gathered W_UK/W_UV head maps are small and are not the same as
replicating the full Q/input and W_O projection weights. They do not eliminate
the current exchanges.

## Implementation and validation order

1. Finish owner-prefill to native-owner-decode continuity. Test the attention
   fixture first, then trained B8/256K with 1,026 output tokens (1,025 measured
   decode steps). Audit four 256-token updates on every owner and MLA layer.
   Use the frozen ProLong prompts and dense continuation trace. No dense rerun
   is needed for this exploratory measurement; label the eager/graph difference.
2. Profile the narrow owner-layout exchanges separately from attention and
   native model work. Do not infer a communication bottleneck merely from
   missing FFNs in the fixture or by summing overlapping GPU intervals.
3. If exchanges justify it, test the attention-stage layout above on a fixture.
   Replicated MLA-only projections remain an alternative if capacity allows,
   not a prerequisite. Keep the daemon intact and use no new weight conversion.
   Compare identical prompts and update policy before a full-model experiment.
4. Preallocate query/output arenas and decode scratch, then make the fixed
   owner schedule graph-replayable. Keep semantic updates on the correct
   per-request boundaries. Prefix reuse, mixed owner occupancy, preemption,
   and overlapping forwards require explicit lifetime tests before serving.

Request pipelining can hide communication only if other independent rows
have work ready. It cannot remove the sequential layer dependency of one
request. Whole-model graph capture also cannot be assumed while cache
transitions allocate or CPU metadata controls update boundaries.

# Eight-stage attention pipeline, shared native MoE

Experimental on `lod-k3`; not pushed, no production defaults changed.
Implementation: [kimi_k3_eight_stage.py](../../benchmarks/kimi_k3_eight_stage.py).

## Question

Can owner-local KDA recover the throughput lost in earlier non-pipelined or
two-owner experiments when all eight GPUs are attention-stage owners?

Compare three layouts with identical trained weights and frozen ProLong tokens:

1. Native full-attention TP8, no attention pipeline.
2. Eight stages with full-head KDA and MLA/LoD on their layer owners.
3. The same eight stages and MLA/LoD, with KDA served under native TP8.

All retain the dense first FFN, every native EP8 INT4 MoE, latent/shared expert
transforms, both AttnRes operations, and the actual intervening dependencies.
No expert routing is frozen and no separately timed MoE duration is added.

Full-model stages are contiguous groups of twelve layers (the final group
uses the remaining layers), matching AttnRes block boundaries. The initial
24-layer smoke instead uses eight groups of three layers and is strictly a
correctness check, not a full-pipeline throughput measurement.

## Execution

Each rank owns one attention stage and uses a compute stream for its local
attention and residual operations. A separate, consistently ordered shared
stream invokes the native distributed FFNs. After a stage's FFN completes,
its next attention can run while the expert service handles another stage.
TP8 KDA uses that same globally ordered service, whereas owner KDA does not.
Each original request's chunks stay chronological at every layer.

Transfers include the running residual prefix, pending FFN delta, and every
valid AttnRes bank entry. Two preallocated slots per stage bound in-flight
memory. Native embedding/all-reduce, service broadcasts, stage handoffs,
FFNs, attention, and global 16K-per-row LoD updates are inside measurement.
Weight assembly, engine startup, final output normalization, LM head,
sampling, and serving scheduler overhead are outside.

Complete stage-owned attention weights are gathered outside measurement.
The resident daemon is not reloaded or modified. Native TP shards are also
retained in this prototype; this is not its final memory layout.

## Validation and timing

- CPU schedule tests visit every layer of every chunk exactly once, preserve
  per-layer chronological cache order, and exercise all eight stages.
- Before timing, every first-512-token output across all rows and stages is
  compared against sequential execution of the same layout. This includes
  continued KDA state and natural trained MoE routing.
- All MLA caches are audited for owner placement, complete global sequence
  history, and the final 256-token exact tail.
- One exact-shape warmup and one synchronized pass per layout. Cohort wall
  time uses the slowest of all eight workers.
- Final-stage completion events separately report completion spacing after
  fill and before drain. Short streams without enough completions do not
  receive a steady-state throughput claim. Context changes remain explicit;
  this is a finite-stream estimate, not an infinite constant-cost pipeline.
- Admission-to-final-chunk completion latency is reported per row, separately
  from throughput. No factor-of-eight extrapolation is made.

## Status

CPU tests pass (18 across the pipeline and related owner probes). Trained 24-layer,
eight-stage scheduling validation passed as
`21207-kimi-eight-stage-trained-validation` on node 4, GPUs 0–7, reusing
`kimi-k3-shared-int4-v6`. Both owner-KDA and TP8-KDA pipelines produced
bitwise-equal outputs to sequential same-layout execution for all eight
512-token prompts, including continued KDA state and native routed MoE.
Raw: [validation](oct5-eight-stage-trained-validation.json).

The full 93-layer check also passed in
`21208-kimi-eight-stage-full-trained-panel`, including two cohorts and slot
recycling, but its dense 16K-row warmup OOMed in native MoE while the prototype
held both owner and TP KDA state. There are no accepted timings from that run.
Inactive KDA state is now released between layouts, outside measurement.
The active-cache retry `21209` passed both full-layer checks again but still
OOMed in dense warmup: each rank also held two full eight-block residual banks
that the serial control did not use. The control now allocates one bank,
and owners allocate only the bank entries needed by their own stage, lazily.
No chunk-size or mathematical change was made in that allocation fix. The next
comparison was `21210-kimi-eight-stage-full-sized-banks`, same node and daemon. It streamed
two cohorts of eight distinct 32K frozen ProLong prompts through eight cache
slots, using one 16K row slice per microbatch: 32 microbatches total. Cache
slots are recycled independently at each layer only after the prior request
has passed that layer. Before timing, the full-layer probe repeats the
schedule-equivalence check with two short cohorts, including slot recycling.
The 16K dense control completed in `21210`
([saved control summary](oct5-eight-stage-full-16k-dense-summary.json)):
**68.858891 s** for all sixteen
32K requests (524,288 tokens), with post-first completion spacing equivalent
to **7,776.29 tokens/s**. All eight worker timing records completed and cache
audits passed. The owner-KDA warmup then exhausted device launch resources
while KDA and MoE overlapped; ROCm reported zero available free VRAM. No owner
timing was accepted; the failed benchmark was cancelled, not the daemon.

`21211-kimi-eight-stage-full-4k` completed the same full-layer comparison at **4K
slices in every layout**, still using global 16K LoD updates. Its stream has
128 microbatches. Accepted worker results are now saved after each variant,
outside measurement, so a later failure cannot discard the control.
Both full-layer short correctness checks passed again. All eight worker timing
records, finite-output checks, and long-context cache audits completed for all
three layouts. The job exited successfully. There are sixteen distinct real
ProLong prompts, each 32,768 tokens, in two cohorts through eight reusable slots:

| Layout, 4K slices | Cohort wall time | Post-fill/pre-drain tokens/s | Mean row latency | Row latency range |
| --- | ---: | ---: | ---: | ---: |
| Full attention, native TP8 KDA | 76.189 s | 6,879.06 | 33.906 s | 33.577–34.237 s |
| Owner MLA/LoD + owner KDA | 125.969 s | 4,164.23 | 61.428 s | 58.979–62.622 s |
| Owner MLA/LoD + native TP8 KDA | 103.405 s | 5,072.12 | 50.418 s | 48.107–51.439 s |

The owner-KDA layout is **1.653× slower** than its matched dense control,
including after pipeline fill; this is not just its drain latency. The completed
16K-slice dense control is faster still (68.859 s), so these results do not claim
a gain by restricting the baseline to smaller slices. Row latency excludes
waiting before admission and is distinct from cohort wall time.

Retaining TP8 KDA improves the pipeline by **1.218×** relative to owner KDA,
but this layout is still **1.357× slower** than the matched dense control.
Both pipelines lose in measured post-fill/pre-drain throughput as well, so
amortizing their startup is insufficient in this implementation. This result
does not establish that an optimized owner pipeline could never win: the
prototype retains extra weights and required 4K slices because the 16K
owner-KDA attempt exhausted VRAM. It does establish that moving attention
to eight owners alone, while keeping the shared native EP8 FFN service,
does not recover the proposed gain here. The FFN service uses all eight GPUs;
owner attention can overlap it but also competes for those same devices.
An overlap trace would be needed to attribute the exact losses to compute,
communication, or allocation rather than inferring them from wall time.

Raw results and all eight worker records:
[completed panel](oct5-eight-stage-full-trained-panel-4k.json).
Accepted partial records also remain in
`oct5-eight-stage-full-trained-panel-4k.json.rank{0..7}.partial.json`.
No production path is promoted and nothing on `lod-k3` was pushed.

After collection, the parent aggregation was adjusted to recompute cohort
tokens/s from the slowest-worker time (rather than retaining rank 7's derived
rate). This does not change any recorded wall time, completion event, or table
entry. Future runs also print warmup/measured phase markers outside timing.

## Reproduction

Use the existing v10 image wrapper and the same environment recorded in
[MLA_ONLY_OWNERS.md](MLA_ONLY_OWNERS.md). Replace its module/arguments with:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_eight_stage \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --weight-cache-id kimi-k3-shared-int4-v6 \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --lengths 32768 --batch 8 --request-cohorts 2 --chunk-size 4096 --layer-count 0 \
  --output results/kimi-k3-mla-stack/oct5-eight-stage-full-trained-panel-4k.json
```

For the initial correctness smoke use `--layer-count 24 --chunk-size 2048
--validation-only`. It does not execute or publish the long-context timings.

# First-24 K3 attention-stage experiment

Status: initial two-GPU LoD-versus-LoD pilot and corrected full-attention TP8
B1/B8 panel complete. **Attention-only diagnostic, not evidence for choosing
the full-model execution layout:** native EP8 MoE is missing.
Full attention under TP8, not TP2 or TP8 LoD, is the relevant speed baseline.
The superseded TP8 LoD smoke `21197` was
cancelled before any accepted timings after this clarification.
Branch `lod-k3`, not pushed. See the
[fixture instructions](../../tests/fixtures/kimi-k3-first24/README.md).

The follow-up [MLA-only owner experiment](MLA_ONLY_OWNERS.md) keeps KDA on
TP8 and includes the trained native EP8 MoE, instead of moving KDA to
owners or omitting FFNs. It is a separate hybrid execution probe; the
attention-only numbers below are not its baseline.

The fixture contains the actual first 24 attention layers, not 24 MLA layers.
Each of two 12-layer owners has nine KDA layers and three MLA layers. Full
projections, native KDA convolution/recurrence and gated norm, MLA LoD cache
construction, and attention-side AttnRes are included. MoE, embeddings, and
LM head are excluded equally. The historical two-GPU LoD control uses TP2 with
replicated latent history and head-parallel attention. It does not use DCP.

The proposed eight-stage execution still needs a separate global EP8 MoE
interface experiment. An attention-only pipeline cannot establish its full
model latency, throughput, or collective ordering.

## Corrected full-attention TP8 control

Raw smoke: [32K B1](oct5-first24-full-tp8-smoke.json), run
`21198-kimi-first24-full-tp8-smoke`, node 2 GPUs 0–7. The control executes
all first-24 layers on TP8; the candidate executes two owner-local 12-layer
LoD stages on ranks 0–1, with ranks 2–7 inactive during candidate execution.

| 32K B1, first 24 attention layers | Cohort total s | Post-first-completion interval ms |
|:--|--:|--:|
| Full attention, TP8, native AITER | 0.357665 | 192.541 |
| Owner-local LoD, sequential | 1.661484 | 929.601 |
| Owner-local LoD, two-stage pipeline | 1.288105 | 544.289 |

The candidate is **3.60x slower** in cohort time for this short smoke test.
Only two microbatches are present, so the interval column is a single spacing,
not a reliable steady-state throughput estimate or an eight-stage result.
No speed win is established.

The native dense causal kernel passed an untimed FP32 check with different
query/key lengths (continued-prefill bottom-right alignment). Dense cache
audits show every MLA layer's full 32K history, with zero centroid updates.
Owner LoD and sequential owner arithmetic are bitwise equal; relative L2
to an untimed same-LoD TP8 control is 2.2903%, below the declared 5% numerical
repartition bound. Relative L2 versus dense is 4.8438%. These random-weight
differences are not trained-model quality results. The full control records
84 GiB summed logical all-reduce inputs; owners have none and hand off
448 MiB across the two chunks.

These corrected dense-baseline results supersede the historical TP2
LoD-versus-LoD speed comparison below for evaluating the proposed layout.

### Completed corrected panel

Raw: [32K/64K B1/B8](oct5-first24-full-tp8-panel.json), run
`21199-kimi-first24-full-tp8-panel`, node 2 GPUs 0–7. All 96 worker/point/
layout cache audits passed. Pipeline and sequential owner outputs are
bitwise equal at every point. Same-LoD repartition relative L2 is 2.31–2.37%.

| Context | Batch | Dense TP8 total s | LoD pipeline total s | Dense post-first spacing ms | LoD pipeline post-first spacing ms |
|--:|--:|--:|--:|--:|--:|
| 32K | 1 | 0.359788 | 1.299015 | 194.402 | 542.032 |
| 32K | 8 | 2.881796 | 7.278731 | 180.966 | 434.787 |
| 64K | 1 | 0.821925 | 2.369868 | 218.317 | 537.071 |
| 64K | 8 | 6.559187 | 15.355343 | 206.214 | 471.352 |

Post-first spacing is `(last completion - first completion)/(chunks-1)`
on the final stage's GPU timeline. It excludes the initial fill, but includes
the transition between context lengths and other finite-cohort effects; it
is not an infinite steady-state rate. Total time additionally includes
reset, allocation, and pipeline endpoints, using the slowest TP worker.

The missing MoE changes both contention and dependencies: all eight GPUs
must dispatch/evaluate/combine experts between attention layers, not merely
once after a complete 12-layer attention stage. The current candidate has
six inactive ranks and only one inter-stage boundary. It therefore does
not recreate the proposed full model's bandwidth/collective schedule.
Neither these results nor the reduced handoff-byte estimates establish its
full-model latency or throughput. No further attention-only panel is needed
for that decision; an MoE-inclusive test is required.

## Residual-bank transfer

The native block-write kernel stores the unnormalized prefix into the bank.
At layer 0, bank entry 0 is therefore exactly the original input hidden state.
For this fixture both GPUs already have identical original input rows. The
second owner can seed bank entry 0 from those rows, eliminating its transfer
without changing AttnRes arithmetic. Its first layer is layer 12, which writes
bank entry 1 from the completed first-block output before attending to the
two bank/prefix sources. Pipelined outputs are checked against the sequential
owner layout and the untimed same-LoD TP control.

For a BF16 16K chunk and hidden width 7,168:

| Handoff payload | MiB |
|:--|--:|
| Completed block output | 224 |
| Original input bank entry | 224 |
| Conventional complete first-boundary transfer | 448 |
| Receiver retains original input bank entry | 224 |

Later owners must receive bank entries they have not seen. Old entries are
immutable, but that does not mean cumulative banks can always be omitted.
The fixture bounds in-flight handoff buffers to two slots per rank; those
buffers are preallocated and safely reused with stream events.

## Comparison with TP8 projection traffic

Every KDA and MLA layer's row-parallel output projection produces a
`[16384, 7168]` BF16 tensor: 224 MiB. There is one TP output all-reduce per
attention layer. Thus twelve layers reduce 2.625 GiB of logical output
payload. This number is not the total transmitted network data.

With a conventional ring all-reduce over eight ranks, each rank sends
`2 * (8-1)/8` times that payload (and receives the same amount):

| Estimate for twelve layers, one 16K chunk | GiB |
|:--|--:|
| Sent per rank | 4.59375 |
| Received per rank | 4.59375 |
| Sent across all eight ranks | 36.75 |
| One complete first-boundary stage handoff | 0.4375 |
| One handoff with original input retained | 0.21875 |

The two handoff sizes are respectively 84 and 168 times smaller than
aggregate TP8 transmitted payload for those twelve output reductions.
These are ring-algorithm estimates, not measured RCCL/xGMI link counters.
Receive bytes are not added again when counting unique aggregate transfers.
DCP attention traffic, MoE dispatch/combine, and the additional owner-to-MoE
interface required by the proposed full layout are excluded. The ratios are
not latency speedups and do not establish a whole-model gain.

The fixture records logical all-reduce input bytes and explicit
point-to-point handoff payloads alongside synchronized wall time. All layouts
are eager and process 16K chunks, with one warmed measured pass; startup and
weight assembly are excluded. LoD layouts use the same per-request global
16K prefill update cadence; the full-attention baseline has no LoD updates.

## Accepted initial smoke test

Raw: [32K B1](oct5-first24-two-stage-smoke.json), run
`21194-kimi-first24-two-stage-valid-states`, node 2 GPUs 0–1. Two 16K
microbatches include cache construction at both global boundaries.

| 32K context, B1, first 24 layers | Prefill seconds |
|:--|--:|
| TP2 head/projection parallelism | 1.153584 |
| Two owner-local stages, sequential | 1.649606 |
| Two owner-local stages, pipelined | 1.285485 |

Pipeline and sequential owner results are bitwise equal. Relative L2 versus
TP2 is 0.0204413. The TP2 LoD control's summed all-reduce input payload is
21 GiB, with 24 reductions per microbatch per rank. Owner layouts have no
all-reduces and send 448 MiB total across the two chunks, with the original
input bank retained. These are logical payload counters, not RCCL link counters.

Only two microbatches are available here, so pipeline fill/drain is material.
The pipelined owner path is still 11.43% slower than TP2 in this initial B1
smoke; this is not a speed win.

## Complete two-GPU pilot

Raw: [32K/64K B1/B8](oct5-first24-two-stage-panel.json), run
`21195-kimi-first24-two-stage-panel`, node 2 GPUs 0–1.

| Context | Batch | 16K microbatches | TP2 total s | Sequential owners total s | Pipelined owners total s | TP2 / pipeline |
|--:|--:|--:|--:|--:|--:|--:|
| 32K | 1 | 2 | 1.152464 | 1.641240 | 1.282921 | 0.898x |
| 32K | 8 | 16 | 9.236730 | 13.187868 | 7.240428 | 1.276x |
| 64K | 1 | 4 | 2.497550 | 3.658278 | 2.331008 | 1.071x |
| 64K | 8 | 32 | 20.101208 | 29.347929 | 15.384879 | 1.307x |

All four pipelined outputs are bitwise equal to sequential owner outputs.
Relative L2 versus TP2 is 2.05–2.11%, below the declared 5% bound. Every
MLA layer has all of its row caches present with the final total length and
coverage `length-256`; state length is 2,896 at 32K and 4,096 at 64K.

These are **total** warmed elapsed times for the entire cohort through both
stages, including fill/drain, state/cache reset, updates, and handoff. They
are not independently measured steady-state completion spacing. B8 averages
are 452.527 ms per microbatch at 32K and 480.777 ms at 64K, but those averages
still include the pipeline endpoints and changing context-dependent costs.
The slower producer/consumer maximum is used; the producer's earlier finish
does not define the result.

TP2 was chosen to compare equal two-GPU resources in the initial pilot. It
does **not** establish latency against the intended TP8 layout. The next
relevant comparison is the same first-24 fixture under native **full-attention TP8** versus
two owner-local stages, with the remaining six attention ranks inactive,
using identical weights within that comparison. The benchmark now defaults
to full-attention TP8. `--tensor-parallel-size 2` changes TP geometry, but
does not reproduce the archived LoD-versus-LoD baseline.
Preallocated GPU events record each completed microbatch; completion spacing after
pipeline fill should be reported alongside total time, before extrapolating
to eight attention stages or adding the global MoE service.

The full control uses native AITER causal FlashAttention (D192 Q/K, D128 V)
and retains all chronological MLA tokens. It has no LoD state updates or
routing; cache writes and history expansion are timed. Native KDA and
attention-side AttnRes are unchanged. An untimed TP8 LoD run is used only
for a numerical repartition check against owner LoD, not as a speed baseline.

The optional full-bank transfer variant also passed:
[32K B8, bank transmitted](oct5-first24-bank-transfer.json), run
`21196-kimi-first24-bank-transfer-check`, node 3 GPUs 0–1. TP2/sequential/
pipeline totals were 9.202706/13.148811/7.187893 s; the pipeline is bitwise
equal to sequential owners and has 2.0523% relative L2 versus its TP2 control.
It sends 7 GiB instead of the retained-bank pilot's 3.5 GiB. This checks
both handoff arithmetic variants, but is not a strict paired bank-latency
ablation: the runs have different maximum-context-shaped random cohorts.

### Rejected development attempts

- `21189`: inherited the unrelated B8/TP8 request-owner serving flag;
  rejected by setup before model execution. Removed from the direct TP2 probe.
- `21190`: reused a 48-head bootstrap engine for 96 owner heads without
  updating its GQA group geometry; rejected by the refinement shape check.
- `21191`/`21192`/`21193`: used KDA cache index 0 for a real request. Native
  vLLM reserves 0 as `NULL_BLOCK_ID`; causal convolution skipped writing those
  outputs. The diagnostic identified non-finites in the TP2 control itself.
  Request IDs now map to state rows 1 onward, with row 0 reserved and a CPU
  regression test. A temporary gate-bias hypothesis did not fix it and was
  reverted. No timings from these failed runs are accepted.

No production kernel, trained weight, persistent daemon, or release default
was changed to correct these private fixture errors.

# Two-layer owner-local Q/K/V/O prefill experiment

Branch `lod-k3`, October 5, 2026. No production default or full-model weight
layout was changed. No pipeline overlap was implemented.

## Result

Moving full Q/K/V/O to the LoD attention owner is **not faster on its own**
in this fixture. It removes the query/head-output exchange and TP W_O
all-reduce, but concentrates projection GEMMs on one GPU and must still
broadcast the projected hidden output to native distributed MoE.

| Model chunk | Context | TP Q/K/V/O (ms) | Owner Q/K/V/O (ms) | Owner latency change |
|--:|--:|--:|--:|--:|
| 2K | 32K | 307.75 | 322.74 | +4.87% |
| 2K | 64K | 874.87 | 905.64 | +3.52% |
| 16K | 32K | 199.59 | 215.85 | +8.14% |
| 16K | 64K | 482.32 | 518.56 | +7.51% |

These are full prefill intervals through **two** native MLA+MoE decoder
layers, including norms/AttnRes, router and expert computation, expert/TP
communication, owner transport, and final LoD construction. No embeddings,
LM head, token sampling, scheduler work, startup, or weight gathering is
timed. The result is not per-layer attention-only time or trained K3 latency.

The non-pipelined layout is viable at roughly comparable latency, but there
is no standalone speed gain to promote. Pipeline overlap across independent
requests could use GPUs idle during owner attention; that benefit has not
been measured. Do not claim that communication is the dominant bottleneck
or that overlapping it will necessarily erase the extra latency.

## Test and validation

- Eight MI325X GPUs per run. Native TP8/EP8 MoE, one attention owner on rank 0.
  2K chunks ran on node 2 and 16K chunks on node 3; compare variants within
  each run, not as a perfectly paired cross-node chunk-size study.
- Full MLA geometry: hidden 7168, 96 query heads, Q low rank 1536, latent
  512 plus direct key 64, expanded K/V 128 per head, output gating retained.
- Two real native K3 decoder layers, including AttnRes. Expert count reduced
  to 32/four active; expert latent 4096, intermediate 2048, one shared expert.
  Random normalized weights and seed-1234 hidden inputs; no KDA layers.
- Both variants use the same single-owner LoD kernels, top-eight routing,
  separate sink, and global 16K construction boundaries. Owner Q/K/V/O uses
  complete matrices assembled from the control's TP shards before timing.
  The prototype retains the original TP shards and is not a final VRAM test.
- Native vLLM Triton BF16 SITU MoE is used in both variants. The image's
  unquantized AITER generator cannot build `situv2`; it differs from the full
  trained model's INT4 AITER MoE kernel. No activation was substituted.
- Expert choices are frozen from an untimed control pass, while router GEMMs,
  top-k, expert kernels, and communication still execute. In the initial
  natural-routing diagnostic, first-layer Q/K/V and attention were identical;
  BF16 TP W_O partial rounding versus a full GEMM produced about 0.4% output
  difference. Nearly tied random expert choices amplified that to 7.5% final
  error. Freezing choices isolates layout and is **not quality evidence**.
- Both complete fixed-route prefills pass the predeclared 2.5% relative-L2
  output check; observed error is about 1.00% across two layers. All eight
  workers agree and report real native EP size 8. Final state metadata agrees
  across variants: at 32K, coverage 32512/state 2896/archived leaves 32511;
  at 64K, coverage 65280/state 4096/archived leaves 65279, in both layers.
- One exact-shape warmup and one measured pass per variant. CPU-group start
  barrier plus GPU synchronization at both boundaries; report the maximum
  wall interval over all eight workers. No profiler events or graph replay.
  Intermediate tracing is diagnostic-only and is disabled in reported runs.
- Host checks: 52 passed, two GPU-only tests skipped in the targeted suite.
  The actual eight-GPU prefills provide the fixture's numerical/run checks.

## Code, commands, and raw data

Probe: [kimi_k3_attention_stage_probe.py](../../benchmarks/kimi_k3_attention_stage_probe.py).
Fixture and non-proprietary reproduction command:
[two-layer MLA+MoE fixture](../../tests/fixtures/kimi-k3-attention-moe/README.md).
Run the command once with `--chunk-size 2048` and once with
`--chunk-size 16384`, using lengths 32768 and 65536. Both require eight GPUs
and the unpacked K3 v10 image/userspace. No model checkpoint download or
resident trained-weight daemon is needed.

- [2K raw results](oct5-attention-stage-2layer-2k.json), job
  `21187-kimi-attention-stage-2layer-fixed-routes`.
- [16K raw results](oct5-attention-stage-2layer-16k.json), job
  `21188-kimi-attention-stage-2layer-16k-fixed-routes`.
- Natural-routing diagnostic: job `21186-kimi-attention-stage-equivalence`;
  its timings were not accepted or placed in the table.

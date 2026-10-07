# Live-context decode splits and IPC-safe graph allocation

## Diagnosis

Two independent configuration changes slowed the October 7 decode refresh.
First, enabling `expandable_segments` globally made AITER disable registered
graph communication and use staging copies. Second, dense decode selected its
compile-time split count from vLLM's capture metadata, which contains maximum
model reservation, not the request's live context.

Matched full-model controls at 16K, with frozen ProLong prompts and
continuations, isolated the effects:

| Configuration | B1 decode (ms/step) | B8 decode (ms/step) |
|:--|--:|--:|
| Expandable allocator, increased reservation | 22.241 | 32.975 |
| Ordinary allocator, same increased reservation | 21.654 | 32.376 |
| Ordinary allocator, previous smaller reservation | 21.435 | 31.639 |

Those controls used the previous reservation-based split selector. Raw files:
`oct7-dense-decode-allocator-b{1,8}-{True,False}.json` and
`oct7-dense-decode-old-cap-b{1,8}.json`.

## Implementation

Dense decode now reserves 128 splits, but chooses the active partition count
from each row's **device-side live DCP length**. No host length read,
synchronization, allocation, recapture, or context-length specialization is
required during replay. The policy retains vLLM's power-of-two 512-keys/split
length rule, with a batch/head occupancy floor. For the real 96-head gathered
K3 query on gfx942, that floor is 32 at B1 and 4 at B8. Longer rows can use
more splits independently of other rows and maximum reservation.

Unused workgroups return before query/KV loads. The reducer derives the same
active partition count and never reads stale unused outputs. Empty padded
rows produce zero output and negative-infinite LSE. The dense partition and
reduction preserve full attention mathematically; only FP32 summation order
can change. LoD selection, attention math, its global 16K prefill / 256-token
decode update cadence, and its BF16 latent archive are unchanged.

`vllm_lod_plugin.graph_allocator` uses ordinary allocations only while
initializing AITER's IPC communicator and capturing/registering its graph
buffers. It restores the exact caller allocator configuration afterward,
including other allocator settings and on exceptions. Existing expandable
allocations are not copied or converted. Eager prefill and large persistent
caches remain expandable, and no allocator toggles run on graph replay.
AITER's unsupported-architecture/transport guards and explicit registration
opt-outs remain authoritative; the adapter never forces its registration flag.

## Validation

- GPU graph replay with ragged rows, page sizes 1 and 768, partial head tiles,
  changing live lengths, split-boundary crossings, and NaN-poisoned unused
  scratch matches independent FP32 attention and LSE references.
- A two-GPU AITER fixture exports ordinary captured communication buffers,
  replays correct sums after input changes, and verifies that history before
  capture and eager growth afterward still use expandable segments.
- CPU tests check nesting, exact setting restoration, exceptions,
  idempotence, and preservation of AITER safety opt-outs.
- Full trained-model checks use the resident weights, real frozen ProLong
  prompts, one untimed warmup and one measured pass, and 1,026 outputs /
  1,025 decode steps. The loaded-communicator audit unwraps vLLM's
  `AiterCustomAllreduce` before reporting AITER's actual registration state.

### Full trained-model results

| Mode | Batch | Live context | Previous refresh decode (ms/step) | Fixed decode (ms/step) |
|:--|:--|--:|--:|--:|
| Dense | B1 | 16K | 22.241 | 21.311 |
| Dense | B1 | 128K | 22.998 | 22.227 |
| Dense | B8 | 16K | 32.975 | 31.665 |
| LoD | B8 | 16K | 30.753 | 30.134 |
| LoD | B8 | 128K | 31.962 | 31.339 |
| LoD (sharded prefill) | B1 | 512K | 22.431 | 21.827 |

Dense B1 reserves 1,045,514 tokens while running the 16K and 128K rows. B8
dense reserves 132,106 tokens. The first B8 dense check predates adding the
occupancy floor, but its live partition is unchanged: both policies use four
splits throughout that 16K continuation. Its first communicator audit looked
at vLLM's wrapper and incorrectly reported `false`; successful graph-buffer
registration in the log is authoritative, and subsequent audits inspect the
underlying AITER instance. Neither artifact nor its timing was rewritten.

Prefill was unaffected by the fix: dense B1 is 2.006 s / 18.988 s at 16K /
128K, and LoD B8 is 16.226 s / 150.572 s at those lengths. These are
verification points, not a fresh complete power-of-two panel.

Raw fixed checks: `oct7-live-splits-floor-full-b1.json`,
`oct7-live-splits-graph-allocator-full-b8.json`, and
`oct7-graph-allocator-lod-b8.json`.

The trained **512K B1** check completed warmup and measured generation with
a 1,045,514-token reservation, expandable eager allocation, registered graph
communication, and all four required decode updates. Prefill is **83.278 s**
(previous refresh 83.521 s), decode is **21.827 ms/step**. It uses the existing
token-sharded prefill history, shared scratch, and 1-GiB native cache budget;
no leaves were dropped or quantized. Raw result:
`oct7-graph-allocator-lod-b1-long-fit.json`.
Rank 0's measured client Torch peak is 20.807 GiB; physical free memory after
generation is 5.348–7.281 GiB across the eight ranks. Resident daemon weights
are outside the client's Torch peak, so that peak is not total device usage.

The B8 reservation control also completed. The same 16K prompts and
continuations take **31.665 ms/step** with a 132,106-token reservation versus
**31.743 ms/step** with a 66,578-token reservation (0.25% difference). Both
choose four active splits per row, rather than the old reservation-selected
32 versus 16. Raw small-reservation control:
`oct7-live-splits-full-b8-small-reservation.json`. Its loaded audit confirms
registered capture and expandable eager allocation on all eight ranks.

The fresh B8/512K LoD check has now completed both passes after the allocator
fix: **849.877 s prefill / 32.570 ms per decode step**, including four updates
per request in all 24 MLA layers. The dense 512K control also completed:
**939.860 s prefill / 60.436 ms per decode step**. LoD is **1.106× faster for
prefill and 1.856× for decode**. Do not infer million-token B8 support from a capacity reservation
or from the small distributed fixture. Fresh dense and LoD B8/1020K attempts
follow each mode's 512K engine, with no simultaneous engines on one node.
The complete rerun progress is tracked in
[CURRENT_TIMINGS.md](CURRENT_TIMINGS.md); the preceding complete panel is
preserved separately in [OCT7_PRE_FIX_TIMINGS.md](OCT7_PRE_FIX_TIMINGS.md).

The focused CPU set passed 350 tests with 46 accelerator skips. The full
CPU suite passed 824 with 137 skips and one pre-existing reachability failure:
`benchmarks/kimi_k3_decode_power2.py:load_sharded_result` has only test callers,
which the release-surface test does not count. That untouched historical
renderer helper is unrelated to these fixes. The live-split GPU cases and
distributed IPC fixture passed separately. The final GPU run, including live
16K/32K/64K split-boundary crossings and the existing LoD kernel checks,
passed all 14 tests.

## Reproduction (no cluster runner)

Run the fixture on a machine with two available gfx942 GPUs in the documented
K3 v10 environment:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run --standalone --nproc-per-node 2 \
  -m benchmarks.kimi_k3_graph_allocator_probe
python -m pytest -q tests/test_graph_allocator.py tests/test_kimi_dense_live_splits.py
```

The kernel occupancy probe is `python -m benchmarks.kimi_dense_split_probe`.
It measures individual captured kernels, **not** serving latency. Full-model
serving uses `benchmarks.kimi_k3_prefill_sweep` and the same resident trained
INT4-MoE daemon and frozen token cache as `CURRENT_TIMINGS.md`; attention
caches are BF16. Keep one active engine per node. Compilation artifacts and
weight staging stay on local disk, and initialization/warmup are excluded.

After starting the resident daemon according to the existing benchmark
instructions, reproduce the dense B1 check as follows. Set `K3_CHECKPOINT`
to the staged local checkpoint and `K3_WEIGHT_CACHE_ID` to that node's daemon
ID. This does not invoke a proprietary scheduler:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
TRITON_CACHE_AUTOTUNING=1 VLLM_USE_TRITON_AWQ=1 \
AITER_CONFIG_FMOE=results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv \
python -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint "$K3_CHECKPOINT" --weight-cache-id "$K3_WEIGHT_CACHE_ID" \
  --mode full --batch-size 1 --lengths 16384 131072 \
  --max-model-len 1045514 --decode-tokens 1026 --reference-decode-trace \
  --repeats 1 --report-memory --kv-cache-memory-bytes 5368709120 \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --reference-baselines results/kimi-k3-full-model-current/oct4-full-b1-decode-power2-four-updates.json \
  --output results/kimi-k3-full-model-current/local-live-splits-b1.json
```

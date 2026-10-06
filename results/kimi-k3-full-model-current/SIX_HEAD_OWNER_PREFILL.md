# Six 16-head K3 prefill owners

Experimental B1 prefill on the `lod-k3` branch. The model remains TP8/EP8:
all trained layers, native Q/gate/O projections, KDA, and MoE are unchanged.
Only MLA attention is repartitioned into six independent contiguous ranges
of 16 heads, on ranks 0–5. Ranks 6–7 still execute their native model shards.
No production default changes and no decode speed claim.

Each attention owner retains the full latent history and the same global
centroids. Routing still opens top-8 centroids, uses the existing 1,024-leaf
cap, and updates at 16K **global sequence** boundaries. There is no history
sharding, distributed top-k, or cross-owner attention/LSE reduction. Native
latent/direct records are already replicated; the preparation audit checks
this numerically before measurement, so no extra KV broadcast is needed.

The TP projection output is regrouped from eight 12-head shards to six
16-head groups, then projected head outputs return to their original shards.
Contiguous packed RCCL packets and two grow-only workspaces are shared across
MLA layers. Three owners have an overlapping native head range, leaving 72
of 96 heads transported each direction, versus 84 in the B1 single-owner
layout. There are nine remote packets instead of seven; fewer bytes alone
do not guarantee lower communication latency.

## Validation

- CPU tests cover all eight ranks, both transport directions, exact head
  ordering, workspace reuse, complete 96-head coverage, and owner audits.
- GPU arithmetic comparison checks 16-head groups against native 12-head
  groups, with identical latent keys, centroid sums/counts, and coverage.
- Before warmup, every MLA layer verifies actual RCCL query regrouping and
  output return against native Q projections, and checks KV replication.
- Warmup-only scheduler audits ensure full 16K row chunks. All timing is
  uninstrumented, with one warmup and one measured pass. Cache construction
  is included through the first output token; audits are outside timing.

The 19-test GPU suite passed on node 3, including arithmetic equivalence.
The TP8, 24-MLA-layer fixture also completed: 0.632 s at 32K and 1.447 s at
64K, with all six owners, all 96 head ranges, loaded kernels, and actual
RCCL round trips audited successfully. KV replication and transport errors
were exactly zero in every layer/rank. These are synthetic-token fixture
results, not full-model performance or task quality. Raw results:
[fixture JSON](../kimi-k3-mla-stack/oct6-six-head-owners-prefill.json).
Full-model real-token timings completed at 32K and 128K on node 2, reusing
resident weights. Compilation artifacts
are on local `/tmp/dan-agent` storage, never the shared image/Ceph directory.

## Full trained-model results

B1, all 93 layers (24 MLA, 69 KDA), TP8/DCP8/EP8, warmed real ProLong tokens,
one measured pass. The new experiment has six attention owners; all other
model execution remains on eight GPUs. Times are end-to-end prefill seconds
through the first output token, including centroid/page construction.

| Context | Dense | Ordinary eight-rank LoD | Previous single-owner LoD | Six 16-head owners |
|--:|--:|--:|--:|--:|
| 32K | 4.186 | 4.189 | 5.765 | 4.430 |
| 128K | 19.300 | 17.391 | 28.230 | 18.506 |
| 256K | 45.497 | 37.112 | OOM in warmup | Resource exhaustion in warmup |

Six owners reduce single-owner latency by 23.2% at 32K and 34.4% at 128K,
but are 5.7% and 6.4% slower than ordinary eight-rank LoD. At 128K they are
1.043x faster than dense. The measured points passed all six active-owner
binary/head-range audits; every MLA layer on all six owners reported the
same final global coverage, state length, and 16K update cadence. Real prompt
hashes exactly match the retained controls, and the first generated token
matches both existing LoD layouts at each completed length. This is not a
full quality benchmark or evidence that 16-head prefill tiles always help.

At 256K, warmup aborted with ROCm `HSA_STATUS_ERROR_OUT_OF_RESOURCES` in an
RCCL kernel on GPU 3. The runtime reported only **62 MB free** on that GPU.
No valid 256K timing was produced; 512K was not attempted. This is a
memory/resource-pressure failure, not a demonstrated head-ordering error.
After 128K generation, the minimum free reading was 5.223 GiB; the maximum
live client Torch allocation during that pass was 23.221 GiB. Client Torch
peaks exclude the daemon's resident weight allocation and are not total VRAM.

The full sweep's top-level status is failed because of 256K; its separately
audited 32K and 128K points are complete and valid. Raw artifacts:

- [Six-owner run](oct6-six-head-owners-b1-prefill-node2.json).
- [Dense control](oct5-owner-b1-control-full-node2-r2.json).
- [Ordinary LoD control](oct5-owner-b1-control-two-tier-node2-r2.json).
- [Single-owner comparison](oct5-owner-b1-tp-mla-prefill-node2-r3.json).

Do not promote this layout as a speed default: the ordinary path remains
faster. Decode has not been implemented or measured for six head owners.

## Full-model reproduction

Use the existing K3 v10 runtime, resident full-weight daemon on node 2, and
the frozen real ProLong tokens. Set the same kernel environment as in
[the B1 control experiment](REQUEST_OWNER_PREFILL.md#single-owner-b1-prefill-full-trained-model).
Run directly without proprietary scheduling tools:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --mode two-tier --head-owner-mla --lengths 32768 131072 262144 \
  --max-model-len 524297 \
  --batch-size 1 --decode-tokens 1 --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 --weight-cache-id kimi-k3-node2-full-int4-v1 \
  --kv-cache-memory-bytes 2147483648 \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --reference-baselines results/kimi-k3-full-model-current/oct4-full-prefill-scale-b1-1g-16k256k.json \
  --audit-prefill-batches --report-memory --repeats 1 \
  --output results/kimi-k3-full-model-current/oct6-six-head-owners-b1-prefill-node2.json
```

The experiment explicitly rejects batched/mixed decode. `max_tokens=1`
measures prefill only. The native cache reservation, real prompt hashes,
operator selection, and row chunk sizes match the existing dense controls;
no dense rerun is necessary for this initial layout comparison.

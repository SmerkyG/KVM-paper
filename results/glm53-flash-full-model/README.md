# GLM5.3-Flash trained-model TP4 comparison

Requested October 7, 2026. The four-layer random fixture is **not** used for
these results. The full `zai-org/GLM-5.3-Flash` FP8 checkpoint is pinned to
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`: all 45 language layers, including
11 NoPE MLA layers and the original KDA, mHC, and 288-expert MoE geometry.
Language-only serving omits the unused vision tower and MTP is disabled.

**October 8 correction:** the projected GLM prefill shortcut bypassed the
common core's contiguous-query preparation. Its exact-leaf consumer reads
flattened head-major query rows, but GLM supplied a transposed token-major
view. The consumer now packs noncontiguous queries at its public boundary;
already-contiguous callers incur no copy. The explicitly historical LoD
tables below predate this fix and must not be interpreted as current quality or speed.
Native and exact-all-history controls did not use the defective leaf path.
See the trained-tensor audit below for the isolated correctness result.
The latest [longer-context sweep](#updated-longer-context-speed-sweep) enables
the projected-leaf and Gluon KDA follow-up ports in addition to this fix.

Corrected B1/B8 speed reruns (job 21849) and matched native/LoD ProLong plus
LongBench smoke checks (job 21852) are complete. Quality evaluation started
only after timing finished, without competing GPU execution. Their protocol,
preflight and results are documented at the end of this report.

The native comparator uses the model's **learned sparse attention with a
2,048-token budget**, not all-history dense attention. LoD uses projected
K256/V256 centroids and local prefill, latent512 exact leaves/decode, top-eight
refinement, the separate sink, 1,024-leaf closure, and global-sequence
16K-prefill / 256-decode update cadence. Neither arm modifies the native FP8
weights. KDA remains native in both arms.

## Protocol and results

Four MI325X GPUs on one node; TP4/EP4/DCP1. Checkpoint files and compilation
artifacts are staged on local disk. One node-local daemon retains the loaded
post-conversion FP8 TP/EP shards; native and LoD clients reuse them rather
than reloading or converting shared weights. Only one timing engine runs on
the node at a time.

Real ProLong prompts, frozen shuffle seed `20260824`, generation seed `1234`.
The same natural continuation is teacher-forced for both arms. B1 uses the
first prompt of the B8 cohort. Each point uses an untimed exact-shape warmup,
then **one** measured generation: 1,026 outputs / 1,025 decode steps, including
four 256-token decode updates per LoD row/layer. The final cache-construction
wait belongs to prefill, not decode. No profiling events in the timed path.
Prefill uses a 16K scheduler chunk and a 16K plus decode-reserve aggregate
budget in both arms. B8 decode waits for all eight prefills, retaining eight
live requests throughout. Prefix hits, preemptions, missing layers, and wrong
FP8 weights fail the run rather than entering the table.

### Current corrected-kernel measurements

The unchanged native controls are reused, not rerun solely because the LoD
query-layout fix changed source files. Prompt and forced-continuation hashes,
scheduler budget, chunk size, seed and cache reservation are identical.

| Batch | Native prefill (s) | Corrected LoD prefill (s) | Native decode (ms/step) | Corrected LoD decode (ms/step) |
|---|---:|---:|---:|---:|
| B1 | 3.156 | 3.132 | 12.898 | 8.519 |
| B8 | 25.283 | 25.110 | 17.856 | 13.762 |

The [corrected B1 result](packed-query-fix-two-tier-tp4-b1-65536.json)
records four updates on all eleven MLA layers/all four ranks. Its untimed
warmup and measured decode agree at 8.522 / 8.519 ms per step. Prefill is
effectively tied with native; decode is 1.51x faster. The
[corrected B8 result](packed-query-fix-two-tier-tp4-b8-65536.json) has identical
native-control prompt/trace hashes, eight live requests throughout the full
1,025-step decode interval and 32 updated rows on every MLA layer/rank
(four per request). B8 decode is 1.30x faster; its warmup and measured decode
are 13.757 / 13.762 ms per step. B8 prefill is the total cohort time and remains
effectively tied with native, not a substantial speed improvement.

### Projected exact-leaf prefill experiment (October 8)

The opt-in `LOD_GLM_PROJECTED_LEAVES=1` changes only the remote exact-leaf
prefill consumer. Coarse routing, top-eight selection, post-ranking closure
above 1,024 leaves, centroid assignment, local attention, sink treatment and
decode remain unchanged. Persistent K/V stays shared latent512. Each selected
(head, centroid) leaf range is projected once to K256/V256 and reused by all
its queries; the attention output therefore no longer needs a post-leaf
W_UV projection. Projection uses the immutable combined UK/UV matrix, device
prefix scans, and a reusable compact arena. That arena has worst-case
capacity for all leaf/head pairs: this experiment does **not** claim
selected-only allocated VRAM.

Seven targeted GPU checks pass, covering independent dense references,
non-contiguous source/query views, ragged and empty centroids, separate sinks,
post-ranking closure, short/long route counting and changed selections during
graph replay. Existing
CPU checks pass (87 passed, 20 GPU skips). Projection versus absorbed-query
arithmetic differs only in BF16 rounding, not in selected regions.

| Kernel diagnostic geometry | Latent refinement (ms) | Projected refinement (ms) | Speedup |
|---|---:|---:|---:|
| 16 heads (one TP4 rank), 16K query slab | 3.705 | 2.551 | 1.45x |
| 64 heads (TP1), 16K query slab | 13.068 | 7.426 | 1.76x |

These are **random-input, remote-refinement-only** GPU intervals, not model
latency or quality. Both consume identical routes from the current projected
coarse kernel. Each interval includes dispatch, union projection, exact leaf
attention and route reduction, plus final W_UV for the latent control. Five
short iterations are used only for the kernel diagnostic; model measurements
still use one measured generation. The latent control brackets the tile
sweep. The 49,152-token cache build leaves 32,767 remote leaves because the
remaining exact local tail is not part of refinement. Raw kernel results:
[TP4 geometry](../glm53-flash-fixture/projected-leaves-h16-48k-16k-r2.json),
[TP1 geometry](../glm53-flash-fixture/projected-leaves-h64-48k-16k.json).
The first sweep's unsupported 128-row attention tile exceeded LDS capacity;
it was not selected or used for the model run.

Additional tuning reuses Kimi's atomics-free route ordinals: sort fixed
2,048-item route fragments, scan per-region counts, then scatter queries.
It changes only metadata order, never routing or attention. Slabs of at least
2,048 queries use this counter; short slabs retain the small atomic counter.
The complete refinement stage improves further to **2.246 ms** at 16 heads
(1.64x vs its bracketed 3.686 ms latent control), and **6.945 ms** at 64 heads
(1.88x vs 13.054 ms). The latter is a matched node-4 GPU-4 kernel comparison;
the overlapping node-3 attempt is not used. Raw data:
[16-head counter tuning](../glm53-flash-fixture/projected-leaves-h16-counts.json),
[64-head counter tuning](../glm53-flash-fixture/projected-leaves-h64-counts.json).
Tuning the latent control's tile size/wave count did not beat its default.
Those kernel speedups are not substituted for full-model speedups.

The initial trained TP4/B1 64K result is **3.092 s prefill / 8.527 ms decode**, versus
the unchanged corrected latent-leaf result **3.132 s / 8.519 ms** and native
control **3.156 s / 12.898 ms**. The model-level prefill difference is only
1.3%, much smaller than the isolated refinement gain; decode is unchanged
within 0.1%. The initial B8 result is **24.786 s / 13.770 ms**, versus latent
**25.110 s / 13.762 ms**, again only 1.3% shorter prefill. These initial model
timings precede the sorted-counter optimization. The candidate remains opt-in,
not the production default.
[Raw projected B1 result](projected-leaves-two-tier-tp4-b1-65536.json), job
21873, uses the same real ProLong prompt/forced continuation and timing
protocol as the current corrected-kernel panel. JIT occurs only in its
untimed warmup. No full-model profiling events are inserted.

The final optimized B8 run (job 21897, same node/GPUs, warmed canonical
1,025-step decode) measures **24.729 s cohort prefill / 13.762 ms decode**:
**1.52% shorter prefill** than corrected latent-leaf LoD, **2.19% shorter**
than the native control, and unchanged decode. Prompt manifests and forced
output hashes match the latent control exactly; all eight requests remain
live throughout decode, with four updates per row/layer, zero preemptions
and zero prefix hits. This is a modest whole-model gain, not the 1.64x
kernel-stage speedup.
[Final optimized B8 result](projected-leaves-optimized-two-tier-tp4-b8-65536.json).

The optimized path's trained quality check (job 21879, node 3 GPUs 0–3) uses
the same frozen document/example hashes and generation setup as the corrected
quality panel below. All eleven MLA layers on every rank report 163 projected
leaf calls. Final targeted GPU checks: 7 passed in 12.52 seconds (job 21893).

| Quality check | Native sparse | Corrected latent leaves | Projected leaves |
|---|---:|---:|---:|
| ProLong PPL, eight common 65,199-token prefixes | 1.561707 | 1.488467 | 1.488150 |
| LongBench-v2 metadata-selected smoke panel | 9/16 | 9/16 | 11/16 |
| NIAH-S3 64K smoke panel | 8/8 | 8/8 | 8/8 |

ProLong changes by **−0.0213% relative to latent-leaf LoD**, consistent with
the changed BF16 rounding boundary, not a material loss regression. The
LongBench difference consists of two formerly wrong answers becoming correct
and one wrong letter changing to another wrong letter; no formerly correct
answer becomes wrong. Sixteen examples are insufficient to claim a general
accuracy improvement. All NIAH responses stop naturally; the same eight
prompt hashes/UUID targets match the corrected diagnostic control.
[Complete projected quality result](projected-leaves-quality-two-tier-64k.json).

Reproduce the kernel diagnostic on one GPU with the supported image/runtime:

```bash
OMP_NUM_THREADS=2 bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.glm53_projected_leaves --heads 16 \
  --history 49152 --queries 16384 \
  --output results/glm53-flash-fixture/projected-leaves-reproduced.json
```

For the trained model, prepend `LOD_GLM_PROJECTED_LEAVES=1` to the existing
64K LoD speed/quality commands. Omit it (or use `0`) for the latent-leaf
control. Do not compare a timed cold start with a warmed control.

The combined quality check reuses one loaded model for all three panels:

```bash
LOD_GLM_PROJECTED_LEAVES=1 OMP_NUM_THREADS=2 \
LM_EVAL_PACKAGE_ROOT="$PWD/.venv/lib/python3.12/site-packages/lm_eval" \
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.glm53_flash_full \
  --checkpoint /tmp/dan-agent/models/GLM-5.3-Flash \
  --mode two-tier --tp 4 --batch-size 8 --length 65536 \
  --measure quality --include-niah-s3 --weight-cache-id glm53-flash-tp4 \
  --output results/glm53-flash-full-model/projected-leaves-quality-reproduced.json
```

The trained commands require the resident FP8 daemon described below; model
tests use the matched TP4/EP4/DCP1 setup. No cluster runner is required by the
benchmark itself. Keep all JIT artifacts on local disk as the launcher does.

### Kimi optimization audit and follow-up ports (October 8)

Already shared with Kimi: projected coarse/local prefill, fused route/coarse
scoring, concatenated immutable UK/UV weights, separate sinks, deferred
cross-layer construction, shared decode scratch, 16-head Gluon decode tiles,
and live-length rather than reservation-sized decode splitting. The selected
leaf projection and sorted projected-leaf dispatch were ported in the
preceding experiment; they are not newly discovered omissions.

Three remaining omissions were addressed:

1. **Sorted ordinals for latent leaves too.** Long GLM slabs now use Kimi's
   fragment sort/prefix counter even with the default latent512 leaf consumer.
   Short slabs retain atomic counting. A matched random-input TP4-geometry
   A/B/A probe measures **3.861 / 3.527 / 3.795 ms** for atomic/sorted/atomic
   complete remote refinement, a **7.85%** reduction against the bracketed
   control. Routes and sampled outputs/LSE match. This is not model latency.
   [Raw counter probe](../glm53-flash-fixture/latent-sorted-counts.json), job 21902.
2. **Avoid redundant query absorption/copies.** Like Kimi, projected GLM
   prefill now passes a shape-only latent view, instead of running a full
   Q256-to-Q512 BMM solely to supply tensor dimensions. This applies to the
   exact front by default and to later chunks with projected leaves enabled.
   The tiny separate sink key is projected to K256 for the final merge.
   Mixed one-token decode rows still absorb their real queries. Absorbed/local
   and native-prefix diagnostic controls retain real absorbed queries. NoPE
   decode also returns the BMM's view directly rather than concatenating an
   empty positional-key tail and copying it. Seven targeted GPU tests pass
   (job 21900), including poisoned shape-only queries, sinks and closure.
3. **Kimi's G8 Gluon KDA arithmetic.** The opt-in
   `LOD_GLM_KDA_PREFILL=1` adapts the same validated gfx942 kernel to GLM's
   already-sigmoided FP32 beta; it does **not** apply sigmoid twice or recover
   logits. Initial/final V-first FP32 state, convolution, normalization,
   gather/scatter and decode remain unchanged. It applies to the native
   comparator as well as LoD. Unsupported input contracts fall back to the
   original function. Singleton/ragged sequences and nonzero initial state
   match an independent FP32 recurrence with 0.44% output and 0.30–0.32%
   final-state relative RMS error, comparable to native's 0.44–0.45% and
   0.33–0.34%. The isolated 16K kernel stage falls from **2.089 to 1.254 ms**
   at 16 heads (TP4 geometry), and **5.643 to 4.793 ms** at 64 heads (TP1).
   [Raw KDA probe](../glm53-flash-fixture/kimi-kda-port.json), successful job
   21899. The first probe passed correctness but its reporting code treated
   a timing dictionary as a number; its failed timing output is not used.

The KDA experiment deliberately retains GLM's native cached ragged metadata
helper and state gather/scatter. Kimi's direct paged-state I/O is **not yet
ported**: GLM's caller owns those operations rather than passing cache indices
to the chunk function. The optional Kimi prefill graph experiments also remain
K3-geometry-specific, and K3's DCP8/request-owner layout is not enabled on this
DCP1 port. These are still gaps, not claimed completed optimizations.

Full-model matched validation of the new changes is recorded below as it
completes; the earlier table remains the earlier measured code, not a timing
claim for the new ports. Native and LoD KDA arms must use the same KDA setting.

The KDA-only full-model control (job 21901, TP4/B8/64K) completes with
**24.457 s prefill / 17.837 ms decode**, versus the unchanged native
**25.283 s / 17.856 ms**: **3.27% shorter prefill**, unchanged decode within
0.11%. Prompt and forced-continuation hashes are identical, all eight requests
remain live for the complete 1,025-step interval, and no preemption/prefix hit
occurs. The full-model control changes no MLA calculation or cache policy.
[Raw native KDA result](kimi-kda-native-tp4-b8-65536.json).

The combined LoD run (job 21903, same node/GPUs and protocol) completes with
**23.632 s prefill / 13.745 ms decode**. That is **5.89% shorter prefill**
than the former latent-leaf default (25.110 s), and **4.43% shorter** than the
previous optimized projected-leaf experiment (24.729 s). Decode changes by
less than 0.13%. Against the **new native KDA baseline** of 24.457 s /
17.837 ms, LoD prefill is **3.37% shorter** and decode is **1.30x faster**.
These are whole-model gains, not isolated attention speedups.

All four ranks report 2,210 Gluon KDA calls and all eleven MLA layers report
1,048,576 deferred query tokens (the two complete B8/64K passes) and 48
projected-leaf calls. Prompt/trace hashes match the previous controls; all
eight requests remain live for the whole decode interval. Four global-256
updates per request/layer are audited, with no prefix hits or preemptions.
[Raw combined LoD result](kimi-ports-two-tier-tp4-b8-65536.json).

The conservative default retains latent leaves, now with sorted long-slab
dispatch and deferred exact-front absorption. Projected leaves and the G8 KDA
port remain explicit opt-ins; the completed validation below supports using
them together but does not test all context lengths/parallel layouts.

The matched trained quality check (job 21904, same node/GPUs) also completes.
Document/prompt manifests match the preceding projected-leaf quality run
exactly. All four ranks confirm actual Gluon KDA execution; all eleven MLA
layers confirm projected leaves and unchanged top-eight/1,024-leaf policy.

| Quality check | Previous projected leaves | New Kimi ports + projected leaves |
|---|---:|---:|
| ProLong loss, 521,584 predictions | 0.397534 | 0.397353 |
| ProLong perplexity | 1.488150 | 1.487881 |
| LongBench-v2 smoke panel | 11/16 | 11/16 |
| NIAH-S3 64K smoke panel | 8/8 | 8/8 |

Perplexity changes by **−0.0181%**, negligible at this BF16 rounding scale.
LongBench is not bit-identical: example `6704fe26bb02136c067ce670` changes
from correct C to wrong A, while `6708ae87bb02136c067d1847` changes from wrong
A to correct C. Every other LongBench prediction is unchanged. Neither the
small answer panels nor the lower ProLong number establish a general quality
improvement. All responses stop naturally and parse correctly. The targeted
CPU suite passes **173 tests** (66 GPU/runtime skips); seven GPU attention
checks and the independent KDA recurrence checks pass separately.
[Raw matched quality result](kimi-ports-quality-two-tier-64k.json).

Reproduce the combined trained speed point and quality smoke panel:

```bash
LOD_GLM_KDA_PREFILL=1 LOD_GLM_PROJECTED_LEAVES=1 OMP_NUM_THREADS=2 \
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.glm53_flash_full \
  --checkpoint /local/models/GLM-5.3-Flash --weight-cache-id glm53-flash-tp4 \
  --mode two-tier --tp 4 --batch-size 8 --length 65536 \
  --measure speed --decode-tokens 1026 \
  --output results/glm53-flash-full-model/kimi-ports-speed-reproduced.json

LOD_GLM_KDA_PREFILL=1 LOD_GLM_PROJECTED_LEAVES=1 OMP_NUM_THREADS=2 \
LM_EVAL_PACKAGE_ROOT="$PWD/.venv/lib/python3.12/site-packages/lm_eval" \
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.glm53_flash_full \
  --checkpoint /local/models/GLM-5.3-Flash --weight-cache-id glm53-flash-tp4 \
  --mode two-tier --tp 4 --batch-size 8 --length 65536 \
  --measure quality --include-niah-s3 \
  --output results/glm53-flash-full-model/kimi-ports-quality-reproduced.json
```

For the matched native speed control, retain `LOD_GLM_KDA_PREFILL=1`, omit
`LOD_GLM_PROJECTED_LEAVES`, and use `--mode full` with a different output
filename. Use the same resident FP8 daemon and scheduler/cache settings.

### Updated longer-context speed sweep

The 128K/256K follow-up uses the same complete model, TP4/EP4/DCP1, B8,
16K prefill chunks, top-eight LoD, 1,024-leaf closure, and global-sequence
16K/256 update cadences. Both arms enable the G8 KDA port; LoD additionally
enables projected exact leaves and the query/dispatch optimizations above.
Native attention remains the learned 2,048-token sparse comparator, not dense
all-history attention. Prefill is the total time for the eight-request cohort;
decode is milliseconds per batch step, with all eight requests live.

Both longer-context arms reserve **32 GiB** of native KV cache per rank, rather
than the 16 GiB used at 64K, to keep B8/256K resident without preemption. Each
point still uses one untimed exact-shape warmup and one measured generation
of 1,026 outputs / 1,025 decode steps. Completed job 21905 executed the four clients
sequentially on node 3 GPUs 0–3, reusing the resident weight daemon. Compilation
artifacts stay on local disk and startup/data preparation are not timed.

| Context | Native prefill (s) | LoD prefill (s) | Prefill speedup | Native decode (ms/step) | LoD decode (ms/step) | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|
| 64K (previous measured point) | 24.457 | 23.632 | 1.03x | 17.837 | 13.745 | 1.30x |
| 128K | 51.092 | 50.116 | 1.02x | 17.713 | 14.093 | 1.26x |
| 256K | 109.468 | 106.586 | 1.03x | 18.172 | 14.565 | 1.25x |

The 128K pair has identical prompt/forced-output hashes, no preemptions or
prefix hits, and eight live requests throughout decode. All eleven LoD MLA
layers on all four ranks record 32 updated rows (four updates per request).
Warmup/measured LoD decode agrees at 14.091 / 14.093 ms per step. Prefill is
only 1.91% shorter than native; this is not a substantial prefill speedup.
Raw results: [128K native](kimi-ports-full-tp4-b8-131072.json),
[128K LoD](kimi-ports-two-tier-tp4-b8-131072.json).

The 256K pair also matches prompt/forced-output hashes, keeps eight live
requests throughout decode, and records four updates per request on all eleven
MLA layers/all four ranks. There are zero preemptions or prefix hits. LoD
warmup/measured decode agrees at 14.564 / 14.565 ms per step. Every rank reports
8,738 actual G8 KDA calls, and every MLA layer reports 4,194,304 deferred query
tokens (two full B8/256K passes) and 240 projected-leaf calls. Monitored JIT
compilation finishes during untimed warmup, before the measured pass.
Raw results: [256K native](kimi-ports-full-tp4-b8-262144.json),
[256K LoD](kimi-ports-two-tier-tp4-b8-262144.json).

The longer-context prefill gain is modest: **1.91% shorter at 128K and 2.63%
shorter at 256K**, not a large or consistently increasing advantage. Decode
is **1.26x / 1.25x faster**, respectively. These are end-to-end model timings,
not attention-only timings. Both lengths fit; no additional quality tests
were performed at these lengths, so the quality evidence remains the 64K
panel above.

Reproduce each point with the existing daemon and compatible image runtime:

```bash
LOD_GLM_KDA_PREFILL=1 OMP_NUM_THREADS=2 \
TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
TORCH_NCCL_ENABLE_MONITORING=0 \
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.glm53_flash_full \
  --checkpoint /local/models/GLM-5.3-Flash --weight-cache-id glm53-flash-tp4 \
  --mode full --tp 4 --batch-size 8 --length 131072 \
  --measure speed --decode-tokens 1026 --kv-cache-gib 32 \
  --output results/glm53-flash-full-model/kimi-ports-full-tp4-b8-131072.json
# Repeat with LOD_GLM_PROJECTED_LEAVES=1, --mode two-tier, and a distinct output.
# Then repeat both arms with --length 262144 and corresponding output paths.
```

Reproduce the isolated comparisons without loading weights:

```bash
OMP_NUM_THREADS=2 bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.glm53_kda_prefill \
  --output results/glm53-flash-fixture/kimi-kda-port-reproduced.json
OMP_NUM_THREADS=2 bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.glm53_projected_leaves --heads 16 \
  --history 49152 --queries 16384 --latent-counter-probe \
  --output results/glm53-flash-fixture/latent-sorted-counts-reproduced.json
```

### Historical pre-fix measurements

| Batch | Native prefill (s) | LoD prefill (s) | Native decode (ms/step) | LoD decode (ms/step) |
|---|---:|---:|---:|---:|
| B1 | 3.156 | 3.110 | 12.898 | 8.522 |
| B8 | 25.283 | 25.108 | 17.856 | 13.799 |

This table is retained as **pre-fix provenance**, not a corrected-kernel
speed claim. Quality-generation or tensor-replay wall times are not
substitutes for the current canonical measurements above.

B8 prefill is the total time for the eight-request cohort, not the time for
one row. Decode is milliseconds per batch step with all eight rows live.

Download completed on node 3 in 139 seconds. The daemon is running with
namespace `glm53-flash-tp4`, GPUs 0–3.
The first cold load completed in 95 seconds at about 75.72 GiB per rank.
Startup then encountered an AITER shared `/tmp/aiter_configs` tuning-table
permission error. The benchmark now explicitly selects the shipped
`configs/tuned_fmoe.csv` and `configs/a8w8_blockscale_tuned_gemm.csv`, bypassing
the shared-directory merge without changing kernel math. The warm-daemon
native B1 retry completed. LoD's IPC loader now omits both the unused native
indexer and its separate `indexer_rope_emb` buffer, matching the indexer-free
LoD constructor; unrelated missing modules still fail loudly. The loader
also keeps the LoD client's `is_sparse`/`use_sparse`/`is_v32` dispatch flags
instead of importing the native daemon's choice. This changes no weights,
ranking, or attention math. The loader regression tests pass. All four
comparisons completed successfully from the same resident weights. The
daemon remains running. These measurements are speed traces, not quality
scores.

B1 prompt and continuation digests match exactly between arms. Prefill is
effectively tied (1.015x); decode is **1.51x faster** with LoD. Every MLA layer
on all four ranks recorded four decode updates. LoD's warmup/measurement
decode intervals agree at 8.525/8.522 ms per step. Its new tuple-keyed
projection layouts are also supported by the untimed memory inventory,
without counting the retained daemon-owned source views as client scratch.
The targeted CPU regressions pass: 116 passed, 20 runtime/GPU cases skipped.

B8 prompt and continuation digests also match exactly. Prefill is again
effectively tied (1.007x); LoD decode is **1.29x faster**. The entire measured
decode interval has eight live requests with zero completion-time spread,
and all 11 MLA layers on each of four ranks record 32 updated rows (four
updates per request). Native/LoD warmup decode is 17.801/13.792 ms per step,
consistent with the measured 17.856/13.799 ms. No new JIT builds occur during
the final measured passes. The same first prompt/continuation appears in B1
and B8, with eight distinct prompts in B8.

The completed [native B1 result](native-tp4-b1-65536.json) records 20.35x native
cache concurrency headroom, zero preemptions/prefix hits, and its exact token
and continuation digests. Startup/warmup and data preparation are excluded.
The [LoD B1 result](two-tier-tp4-b1-65536.json) also records its complete update
counters, memory inventory and zero preemptions/prefix hits.
The [native B8 result](full-tp4-b8-65536.json) records eight live requests for
the entire 18.303-second measured decode window and simultaneous completion,
with zero preemptions/prefix hits.
The [LoD B8 result](two-tier-tp4-b8-65536.json) records the matching cohort,
14.144-second decode window, update counters and memory inventory.

This is **not a substantial prefill speed win at 64K**: the less-than-2%
differences are within ordinary single-pass variation. Decode improves by
33.9% in latency at B1 and 22.7% at B8 against the native sparse comparator.
The node was running only this GPU timing engine plus its idle resident
weight daemon. Compilation artifacts are local, and initial setup/builds
are excluded. Timing-job provenance: native B1 job 21783; the successful
LoD B1/native B8/LoD B8 sequence is job 21796, both on node 3 GPUs 0–3.

## NIAH-S3 64K quality check

Current corrected-kernel result (job **21846**, October 8): **8/8 exact
UUID answers**, matching the existing native and dense controls on identical
prompt hashes. Default latent clustering, top-8/top-8, 1,024-leaf closure,
separate sink and global update cadences are unchanged. All answers stop
naturally in 28–32 output tokens. The
[corrected raw result](trained-routing-diagnostic-packed-fix-64k-8.json)
also includes every worker/cache audit and the trained-tensor diagnostics.

The table and ablation discussions below are **historical pre-fix controls**;
in particular their small differences cannot establish whether removing the
cap or projected-key clustering improves a correct LoD implementation.

Matched **eight-example smoke test**, TP4/B8, completed October 7 on node 3
GPUs 0–3 (job 21803). This is not a comprehensive NIAH result.

| Attention | Correct / total | Accuracy |
|---|---:|---:|
| Native learned sparse | 8 / 8 | 100% |
| Two-tier BF16 LoD | 4 / 8 | 50% |
| Exact all-history MLA (indexer disabled) | 8 / 8 | 100% |
| Exact all-history prefill + unchanged LoD decode | 8 / 8 | 100% |
| Ordinary LoD prefill, except exact final row + LoD decode | 5 / 8 | 62.5% |
| Uncapped top-eight LoD (ordinary latent clustering) | 5 / 8 | 62.5% |
| Projected-key clustering, capped top-eight LoD | 5 / 8 | 62.5% |

All eight prompt hashes, targets, sample indices and lengths match exactly.
The 65,536-token RULER generator produces chat-formatted lengths of
65,092–65,098 tokens. `lm-eval` 0.4.13 supplies the standard essay/word-key/UUID
generator, Python seed `0`, NumPy seed `1234`, indices 0–7. Both arms use the
checkpoint's native chat template and assistant answer-prefix continuation,
with a closed `<think></think>` block, greedy generation, engine seed `1234`
and a 64-token output limit. All responses end naturally with `finish_reason`
`stop`; outputs use 27–38 tokens. LoD misses indices 1, 3, 4 and 6, returning
incorrect UUID-like answers rather than truncated responses. The current
LoD port therefore has a clear quality regression on this small panel.

The exact-all-history ablation (job 21806) uses the same eight prompts and
the same resident weights, TP4/B8, closed-thinking chat template, output
limit, scheduler chunk and cohort barrier. All 11 MLA layers on every rank
are audited as indexer-free, non-sparse, and without LoD pools. Prefill uses
vLLM's exact AITER FlashAttention path, including exact LSE combination over
history chunks; decode uses vLLM's dense Triton MLA path. No centroid
approximation, leaf cap or sparse token budget is applied. KDA, mHC and MoE
are unchanged. All eight responses are correct and stop naturally, using
28–32 output tokens. This **does not reproduce LoD's failures**: the change
from native sparse to full-history attention alone is not sufficient to
explain the 4/8 regression on this panel. It points toward the LoD
approximation/routing or an implementation issue; it does not establish
which of those is responsible.

The [dense result](niah-s3-64k-cohort-exact-all-history-r2-8.json) includes
the complete audits and answers. All three arms have identical prompt
manifests, daemon endpoints and resident weight byte counts. The preceding
dense attempt (job 21805) failed during initialization because native
sparse daemon metadata replaced the dense client's `prefill_backend` with
`None`; it produced no quality score. The benchmark-only dense worker now
keeps its constructor's backend, sparse-dispatch and head-padding choices
instead of importing them from the native sparse daemon. Production native
and LoD imports are unchanged. A four-layer 4K prefill/decode smoke test
(job 21804) and CPU metadata regressions cover the ablation setup.

The initial run (job 21802) failed in the **native** GLM pooled indexer's
mixed-prefill/decode path with a 426-versus-427-row mismatch. No score is
reported for that failed run. The successful comparison uses the same
existing synchronized-prefill cohort barrier as the speed panel: complete
all eight prefills before starting decode, identically in both arms. This
avoids the broken native mixed path without changing prompts, ranking or
attention math; it is a scheduling workaround, not a repair of that upstream
exception. Responses may finish independently after prefill.

Raw [native](niah-s3-64k-cohort-full-8.json) and
[LoD](niah-s3-64k-cohort-two-tier-8.json) records include prompt hashes,
targets, generated text and finish reasons. Their elapsed times include
first-use JIT work and **are not canonical speed measurements**.

After starting the daemon as below, reproduce the quality check with:

```bash
for mode in full two-tier; do
  HIP_VISIBLE_DEVICES=0,1,2,3 \
    TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
    TORCH_NCCL_ENABLE_MONITORING=0 \
    python -m benchmarks.glm53_flash_full \
    --checkpoint /tmp/local/GLM-5.3-Flash \
    --mode "$mode" --tp 4 --length 65536 --batch-size 8 \
    --measure niah-s3 --niah-samples 8 --max-new-tokens 64 \
    --weight-cache-id glm53-flash-tp4 \
    --output "results/glm53-flash-full-model/niah-s3-64k-cohort-${mode}-8.json"
done
```

Install `lm-eval==0.4.13` and the repository's benchmark dependencies in the
serving environment. Local runs here expose that package through
`LM_EVAL_PACKAGE_ROOT` / `LM_EVAL_VERSION`, because the AMD image does not
bundle the harness. The targeted CPU regression suite, including the dense
ablation regressions, passes (97 passed, 20 GPU/runtime-dependent tests skipped).

Reproduce the full-history ablation without altering checkpoint config or
reloading the resident weights:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
  TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
  TORCH_NCCL_ENABLE_MONITORING=0 \
  python -m benchmarks.glm53_flash_full \
  --checkpoint /tmp/local/GLM-5.3-Flash \
  --mode full --exact-all-history --tp 4 --length 65536 --batch-size 8 \
  --measure niah-s3 --niah-samples 8 --max-new-tokens 64 \
  --weight-cache-id glm53-flash-tp4 \
  --output results/glm53-flash-full-model/niah-s3-64k-cohort-exact-all-history-r2-8.json
```

The ablation is isolated in `benchmarks/_glm53_dense_attention.py`; it is
never selected by normal native or LoD serving. Its first-use elapsed time
includes an AITER LSE kernel compilation and is not a canonical speed point.

### Isolating prefill from decode

Job **21816**, node 3 GPUs 0–3, applies exact causal all-history attention
to **every prefill row**, including the final query row, while building the
ordinary LoD cache from those exact-prefill representations. Decode uses
the unmodified two-tier BF16 LoD kernels, top-eight selection and the
**1,024-leaf cap**. It recovers all four previous misses (indices 1, 3, 4, 6):
**8/8**, with 28–32 output tokens, all stopping naturally.

All four arms have identical prompt manifests, resident daemon endpoints
and resident weight byte counts. Each of the 11 MLA layers on all four
ranks records exactly 520,769 exact-prefill tokens in 39 ragged scheduler
calls. The temporary request-local latent histories are empty before the
post-generation audit; decode cannot read them. The same audit confirms
top-8/top-8 and the unchanged 1,024 cap. The
[complete result](niah-s3-64k-cohort-exact-prefill-lod-decode-8.json) retains
answers, prompt hashes and both initialization and post-generation audits.

This shows that **LoD decode can succeed with exact-prefill representations**
on this panel. It does not yet distinguish errors in background prefill
representations from retrieval on the final query row, which produces the
first generated answer token. Nor does it establish that capped decode
would succeed with the original LoD-prefill representations. The next
quality investigation should focus on prefill/final-query approximation;
the subsequently completed uncapped follow-up is described below.
Production defaults remain unchanged.

The benchmark-only implementation is
`benchmarks/_glm53_lod_ablation.py`. It computes ordinary LoD prefill to
retain identical cache-construction machinery, then replaces its output
with projected K256/V256 exact causal attention over a temporary full latent
history. It intentionally computes extra attention and is **not a speed
candidate**. Its 34.83-second generation elapsed time includes first-use
work and must not replace the canonical speed table above. A four-layer
20K two-chunk prefill/decode smoke test (job 21815) passed before the trained
run. CPU regressions pass: 100 passed, 20 GPU/runtime-dependent skips.

Reproduce from the same resident daemon:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
  TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
  TORCH_NCCL_ENABLE_MONITORING=0 \
  python -m benchmarks.glm53_flash_full \
  --checkpoint /tmp/local/GLM-5.3-Flash \
  --mode two-tier --exact-prefill --tp 4 --length 65536 --batch-size 8 \
  --measure niah-s3 --niah-samples 8 --max-new-tokens 64 \
  --weight-cache-id glm53-flash-tp4 \
  --output results/glm53-flash-full-model/niah-s3-64k-cohort-exact-prefill-lod-decode-8.json
```

The separate `--uncapped` quality-only control disables closure before
cache construction without altering the ranking or the selected top-eight.
Its trained-model result is described below.

### Final-row-only exact attention

Job **21821**, node 3 GPUs 0–3, tests whether the last prefill lookup alone
explains the regression. All earlier prefill rows use the ordinary LoD
policy (including its existing exact first 16K chunk). Only the final
sequence position of each request receives exact causal all-history
attention in each MLA layer. The request's temporary archive contains the
latents generated by this otherwise-normal LoD prefill, not the latents
from the earlier all-exact-prefill experiment. Decode keeps the same
top-eight kernels and 1,024-leaf cap.

Result: **5/8 (62.5%)**. Index **3** is recovered; indices **1, 4, 6** still
fail, and no previously correct sample becomes incorrect. All answers stop
naturally, with 18–39 generated tokens. All five comparison arms have
identical prompt manifests, daemon endpoints and resident weight byte
counts. On all 11 MLA layers of all four ranks, the audit confirms exactly
**eight real rows replaced in eight calls**, empty temporary histories,
top-8/top-8 and the unchanged cap. The
[raw result](niah-s3-64k-cohort-exact-final-row-8.json) includes all responses
and post-generation audits.

The final-row lookup contributes to the error but is **not a sufficient
explanation**: all-exact prefill obtains 8/8 whereas this control obtains
5/8. This does not prove the remaining errors originate in the background
essay. Earlier question/assistant-prefix rows can also require retrieval,
and decode now reads a cache produced by mostly-approximate prefill. Also,
the first generated token may be punctuation or formatting, with the first
actual UUID token retrieved in a later decode step. The next narrow control
would make the entire final question/assistant-prefill region exact while
leaving the background on LoD. That follow-up has **not** been run; subsequent
tests instead change the approximation globally, not one prompt region.

Both a direct single-query GPU smoke test (job 21819) and the equivalent
causal-kernel smoke test (job 21820) passed at 20K. AITER dispatches a single
query to a separate unmasked library, which incurred a cold build. The full
test reuses the existing causal library by evaluating a second, discarded
query at position N-2 and keeping only the real query at N-1. The real row
attends to all N keys, with no extra key, extra softmax mass or persistent
cache entry. CPU regressions verify causal alignment, preservation of every
non-final output row, independent ragged request histories and decode
pass-through: **105 passed, 20 GPU/runtime-dependent skips**. Production
defaults are unchanged. The 26.04-second quality-generation elapsed time is
not a canonical speed result and does not change the speed table.

Reproduce from the same resident daemon:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
  TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
  TORCH_NCCL_ENABLE_MONITORING=0 \
  python -m benchmarks.glm53_flash_full \
  --checkpoint /tmp/local/GLM-5.3-Flash \
  --mode two-tier --exact-final-row --tp 4 --length 65536 --batch-size 8 \
  --measure niah-s3 --niah-samples 8 --max-new-tokens 64 \
  --weight-cache-id glm53-flash-tp4 \
  --output results/glm53-flash-full-model/niah-s3-64k-cohort-exact-final-row-8.json
```

### Uncapped top-eight

Job **21823**, node 3 GPUs 0–3, removes the 1,024-leaf closure from both
prefill and decode, before allocating or constructing the cache. Everything
else remains unchanged: latent-space cosine clustering, shared latent sums,
top-eight ranking with count bias, the exact first 16K chunk, and ordinary
LoD attention on every subsequent prefill and decode row.

Result: **5/8 by the unchanged RULER substring scorer**, versus 4/8 capped.
Index 3 becomes correct; index 2 becomes incorrect. Index 6 is newly scored
correct but its response is `4e6a1a40b-f031-44b9-bb6b-0095ac7b7ab2`, with an
extra leading `4` before the target UUID. Thus it is **not a clean exact
UUID recovery**; requiring an exact extracted UUID leaves 4/8. Indices 1
and 4 remain incorrect. All responses stop naturally, using 18–39 output
tokens. Removing closure is not sufficient to resolve the regression.

The [raw result](niah-s3-64k-cohort-uncapped-top8-8.json) matches the capped
run's prompt manifest, daemon endpoints and weight byte counts exactly.
All 44 pool audits confirm top-8/top-8 and `max_open_centroid_leaves=null`.
Its 27.54-second quality-generation elapsed time is not a canonical speed
measurement. This diagnostic is not promoted to production; unbounded
refinement abandons the existing worst-case work bound.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
  TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
  TORCH_NCCL_ENABLE_MONITORING=0 \
  python -m benchmarks.glm53_flash_full \
  --checkpoint /tmp/local/GLM-5.3-Flash \
  --mode two-tier --uncapped --tp 4 --length 65536 --batch-size 8 \
  --measure niah-s3 --niah-samples 8 --max-new-tokens 64 \
  --weight-cache-id glm53-flash-tp4 \
  --output results/glm53-flash-full-model/niah-s3-64k-cohort-uncapped-top8-8.json
```

### Projected-key clustering diagnostic

The benchmark-only `--projected-clustering` control uses the actual learned
`W_UK` key spaces instead of raw latent cosine for append selection and
leaf assignment. Each TP rank gathers all 64 heads' immutable key-projection
weights once, before real cache construction. For each leaf and centroid,
it projects the original latent/latent mean into K256 for each head,
normalizes each head independently, and concatenates the resulting vectors.
Their dot product, up to a fixed positive factor, is **mean per-head cosine**.
This avoids letting high-norm heads dominate the assignment objective.

Persistent centroids still hold raw latent sums; projecting their means is
exactly the same as averaging the corresponding expanded keys, by linearity.
Values, coarse attention, routing scores, count bias, top-eight selection,
and exact replacement arithmetic are unchanged. This does **not** introduce
64 independent centroid sets, nor does it normalize keys used for attention.
The initial test retains the 1,024-leaf cap to isolate clustering from closure.

The literal implementation in `benchmarks/_glm53_projected_clustering.py`
materializes a 64×256-dimensional transient vector per clustering key and
disables native-geometry streaming scans and cross-layer updates. It is an
accuracy diagnostic, **not a proposed fast serving implementation**. A
four-layer 20K, two-chunk GPU prefill/decode fixture passes (job **21824**);
the targeted CPU suite passes **110 tests**, with 20 GPU/runtime-dependent
skips, including a real state-update check of projected owners and unchanged
latent/value sums.

Trained-model job **21825**, node 3 GPUs 0–3, completes with **5/8** under
both the unchanged harness scorer and exact UUID extraction. It recovers
index **3** and retains all four previous successes; indices **1, 4, 6**
still fail. All responses stop naturally, using 27–39 output tokens. The
[raw result](niah-s3-64k-cohort-projected-clustering-top8-8.json) has the same
prompt manifest, daemon endpoints and resident weight byte counts as the
ordinary capped run. All 44 pools confirm projected clustering was actually
called, all 64 heads were included, top-8/top-8, and the unchanged 1,024 cap.

This is a modest quality improvement, **not a resolution** of the 8/8 versus
4/8 regression. It does not rule out a head-specific clustering problem,
because this diagnostic still chooses one shared membership by averaging
head similarities. Nor does it establish that projected geometry is the
main cause: it recovers the same single example as final-row exact attention.
Its 29.97-second generation elapsed time includes first-use work and is
**not a controlled performance comparison**. No production defaults change.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
  TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
  TORCH_NCCL_ENABLE_MONITORING=0 \
  python -m benchmarks.glm53_flash_full \
  --checkpoint /tmp/local/GLM-5.3-Flash \
  --mode two-tier --projected-clustering --tp 4 --length 65536 --batch-size 8 \
  --measure niah-s3 --niah-samples 8 --max-new-tokens 64 \
  --weight-cache-id glm53-flash-tp4 \
  --output results/glm53-flash-full-model/niah-s3-64k-cohort-projected-clustering-top8-8.json
```

## Reproduction without the cluster runner

In the compatible AMD vLLM environment, install the repository's plugin and
keep all JIT caches on local disk. Download the pinned checkpoint locally:

```bash
hf download zai-org/GLM-5.3-Flash \
  --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a \
  --local-dir /tmp/local/GLM-5.3-Flash

HIP_VISIBLE_DEVICES=0,1,2,3 VLLM_PLUGINS=lod_attention \
  python -m vllm_lod_plugin.weight_cache_daemon serve \
  --cache-id glm53-flash-tp4 --max-cache-gb-per-gpu 115
```

Leave the broker running, then run the following clients **sequentially**
on the same GPUs. Both use 16 GiB per-rank native-cache reservation, with
LoD's authoritative semantic cache separately owned by its engine.

```bash
for batch in 1 8; do
  for mode in full two-tier; do
    HIP_VISIBLE_DEVICES=0,1,2,3 \
      TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
      TORCH_NCCL_ENABLE_MONITORING=0 \
      python -m benchmarks.glm53_flash_full \
      --checkpoint /tmp/local/GLM-5.3-Flash \
      --mode "$mode" --batch-size "$batch" --tp 4 --length 65536 \
      --decode-tokens 1026 --weight-cache-id glm53-flash-tp4 \
      --output "results/glm53-flash-full-model/${mode}-tp4-b${batch}-65536.json"
  done
done
```

This branch's `benchmarks/run_kimi_k3_v10_direct.sh` selects the documented
AMD image userspace and local compiler caches; it is not required on an
already compatible installed runtime. Graph mode is full-decode-only. A
deliberately eager diagnostic must use `--enforce-eager` identically in both
arms and is not mixed with graph timings. Speed traces are not free-generation
quality scores.

## October 8: trained-tensor correctness and route-oracle audit

The requested all-exposed check found a real implementation bug, rather
than proving a defect in latent clustering. GLM's adapter supplies
`[B,H,Q,512]` queries as a transposed `[B,Q,H,512]` view. The paged expert
kernel addresses flattened contiguous query rows. GLM's early return into
projected prefill skipped the normal core's `q.contiguous()`, so exact-leaf
attention read another head/token's query. Coarse and local attention used
explicit strides and did not have this problem. Decode already packs its
queries and is unchanged.

The fix is one unconditional `q = q.contiguous()` at the public
`paged_leaf_attention` boundary. It is a no-op for already-packed inputs;
GLM copies its transposed prefill queries. No keys, values, centroids,
normalization, routing scores, opening policies or model weights change.
The matched 64K smoke test improves from **4/8 to 8/8**, with eight clean
UUIDs, no output truncation, and the original 1,024 cap retained. Native
sparse and exact all-history attention were already 8/8. Their prompt
manifests and the corrected run's manifests are byte-for-byte equal.

For the diagnostic, capture sixteen final-prefill queries from logical
request slot 1 in all 11 MLA layers on all four TP ranks: **11,264 real
head/query observations**, not random vectors. Each capture has 49,333
remote leaves, 15,763 exact local keys and a separate one-token sink.
Independent CPU enumeration finds every remote leaf in exactly one
directory posting, and state counts match leaf counts exactly. Raw centroid
sums have 0.251% mean relative L2 difference from independent FP32 sums,
consistent with their stored BF16 accumulation.

The all-exposed control groups the remote leaves into eight superregions
and opens every region through the existing indexed leaf consumer and
coarse-replacement kernel. It does not use a separate dense output to
replace the kernel result. The original directory is audited before this
regrouping. An independent FP32 reference covers all remote leaves, causal
local keys and the sink. It reproduces the serving path's absorbed-BF16
remote query and projected-BF16 local field; a second expanded-BF16 dense
reference measures the absorption/projection rounding difference.

On the **same saved first-layer tensors**, the all-exposed output error
drops from **58.097% to 0.442% relative L2** after the layout fix. Its exact
remote LSE error drops from **2.32049 to 0.000000954**. On the new, corrected
full-model captures, all-exposed output error averages **0.431%**, ranging
from **0.360% to 0.506%** across 44 layer/rank captures. Maximum remote LSE
error is **0.00000572**. Replaying ordinary routing matches the captured
serving output to **0.000451% mean relative L2** (maximum 0.003335%). These
checks put the remaining arithmetic discrepancy at BF16 attention/value
projection rounding, not an unaccounted softmax branch or missing leaves.

Then, on those same corrected tensors, change which eight original regions
are refined. Subtraction always uses their **original centroid scores**,
never oracle scores. The maximum-leaf oracle ranks a region by its highest
actual leaf score; the mass oracle ranks by actual leaf-score log-sum-exp.
Oracles are uncapped to isolate ranking from closure; ordinary uncapped
routing is included separately. All other regions retain their summaries.

| Selection | Mean output relative L2 vs dense | Top remote leaf's region opened | Mean opened KV tokens |
|---|---:|---:|---:|
| Ordinary top-eight, cap 1,024 | 14.460% | 73.01% | 684 |
| Ordinary top-eight, uncapped | 14.444% | 73.01% | 756 |
| Actual maximum-leaf-score oracle | 10.992% | 100.00% | 151 |
| Actual region-mass oracle | 10.584% | 85.47% | 844 |

Output errors are arithmetic means of each layer/rank capture's global
relative L2, not perplexity or answer accuracy. Oracle kernel outputs also
match independent FP32 mixed exact/coarse references to approximately
0.54% mean relative L2. These comparisons show some room for better region
selection, but **they are not oracle NIAH generation runs**, and the normal
corrected policy already scores 8/8 on this small cohort. The earlier
uncapped/projected-clustering comparisons must not be used to assess those
ideas without rerunning them on correct kernels.

Provenance: pre-fix 44-capture audit job **21842**, paired saved-tensor
replay **21844**, corrected full-model generation and 44-capture audit
**21846**, four GPU regression tests **21847**. All ran on node 3. A first
diagnostic attempt (**21840**) captured successfully but its rebuilt control
omitted mandatory virtual-cache source arguments; it has no successful
diagnostic result. That benchmark-only setup error was fixed before 21842.
A first new GPU-test setup omitted virtual-page mode; the corrected test
then passed with the other three projected-prefill tests. The CPU suite
passes **101 tests**, with 20 GPU/runtime cases skipped on the CPU host.

Raw results: [pre-fix all-layer audit](trained-routing-diagnostic-64k-8-r2.json),
[pre-fix paired replay](trained-routing-first-layer-components.json),
[fixed paired replay](trained-routing-first-layer-packed-query-fix.json), and
[corrected NIAH plus all-layer audit](trained-routing-diagnostic-packed-fix-64k-8.json).
Large captured tensors stay on node-local disk, not in git. These diagnostic
generation/replay times are **not canonical speed measurements**.

Reproduce using the same resident daemon and compatible installed runtime:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
  TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
  TORCH_NCCL_ENABLE_MONITORING=0 \
  python -m benchmarks.glm53_flash_full \
  --checkpoint /tmp/local/GLM-5.3-Flash \
  --mode two-tier --tp 4 --length 65536 --batch-size 8 \
  --measure niah-s3 --niah-samples 8 --max-new-tokens 64 \
  --weight-cache-id glm53-flash-tp4 --routing-diagnostic \
  --diagnostic-save-dir /tmp/glm53-trained-routing \
  --output results/glm53-flash-full-model/trained-routing-diagnostic-packed-fix-64k-8.json
```

Replay one captured layer without loading model weights:

```bash
HIP_VISIBLE_DEVICES=0 python -m benchmarks._glm53_routing_diagnostic \
  /tmp/glm53-trained-routing/rank0-language_model.model.layers.3.self_attn.mla_attn.mla_attn.pt \
  --output /tmp/glm53-routing-replay.json
```

The diagnostic performs CPU copies and synchronizations, so do not use it
for speed measurements. Compiler artifacts remain on local disk; the image
runner automatically chooses the same local compilation-cache directories.

## Corrected-kernel broader quality validation (October 8)

The small follow-up panel compares **native learned sparse attention** and
ordinary corrected two-tier BF16 LoD, using the same resident FP8 weights,
TP4/EP4/DCP1, B8, seed 1234 and cohort schedule. It is not a full LongBench
run or an all-history dense control. No exact-prefill ablation, projected-key
clustering, uncapping, or trained-tensor capture is enabled.

ProLong uses frozen raw dataset rows **14, 19, 20, 23, 24, 25, 27, 28** from
`Seerkfang/prolong-64k-512-new`, revision
`97295b7d7fe48dc0aa6ba373af3a8b9d945e505b`. These are the usual frozen cohort's
offsets 8–15, not a tokenizer-dependent replacement selection. Under GLM's
tokenizer row 19 has only 65,199 tokens, and rows 25/27 are slightly below
65,536. All eight documents therefore use their **common 65,199-token prefix**,
with no padding, duplication or substitutions. As in the other ProLong
experiments, these are raw continuation-loss inputs, **not chat prompts**.
The raw result records document/token hashes, individual losses and four
position bands, separating the exact first 16K from later LoD chunks.

LongBench uses 16 metadata-selected examples from the pinned official
`THUDM/LongBench-v2` revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`: six long, five medium and five
short examples, spread across five domains. Selection is independent of model
outputs or attention mode. These use the native GLM chat template with an
explicit empty-thinking assistant prefix, greedy decoding and the existing
four-choice answer grammar, capped at 32 output tokens. Content beyond
65,408 tokens is truncated with the standard first/last-half rule; 11 of the
16 examples are truncated and the rendered chat stays within 64K. Fifteen
examples extend beyond the exact 16K prefix. This is a smoke test of matched
inputs, **not a full-benchmark accuracy estimate**.

The completed [input preflight](packed-query-fix-quality-preflight-r2.json)
records all token hashes before model execution. Both GPU arms completed
successfully in job 21852 on node 3 GPUs 0–3. The
[native raw result](packed-query-fix-quality-full-64k.json) and
[corrected LoD raw result](packed-query-fix-quality-two-tier-64k.json)
contain identical document/prompt hashes and identical resident-daemon weight
identities across all four ranks. Every LoD pool uses top-eight in both phases
and the 1,024-leaf closure. No model layer is removed or reinitialized.

| Measure | Native learned sparse | Corrected two-tier BF16 LoD |
|---|---:|---:|
| ProLong loss (521,584 predictions) | 0.445780 | 0.397747 |
| ProLong perplexity | 1.561707 | 1.488467 (-4.690%) |
| LongBench smoke accuracy | 9/16 (56.25%) | 9/16 (56.25%) |
| LongBench short / medium / long correct | 5/5 / 3/5 / 1/6 | 5/5 / 3/5 / 1/6 |

LoD perplexity is lower on **all eight documents**, not driven by a single
outlier. The individual reductions range from 1.190% to 7.964%.

| Prediction-query position band | Native perplexity | LoD perplexity | Relative change |
|---|---:|---:|---:|
| 0–16,383 | 1.602368 | 1.580735 | -1.350% |
| 16,384–32,767 | 1.508926 | 1.453759 | -3.656% |
| 32,768–49,151 | 1.514377 | 1.426366 | -5.812% |
| 49,152–65,197 | 1.625905 | 1.497719 | -7.884% |

For the 390,512 predictions **after the exact first 16K**, perplexity is
1.548292 native versus 1.458721 LoD. Thus the overall result is not hiding a
later-context loss regression behind the exact prefix. The comparator is
native **learned sparse** attention, not dense attention: even LoD's exact
first-16K band differs from native's 2,048-token selection. Lower perplexity
on this cohort does not establish numerical equivalence to native attention,
an approximation-error bound against dense attention, or a general quality
improvement on other datasets.

LongBench correctness agrees **on every example**: nine both correct and
seven both wrong. Three of the shared wrong examples choose different wrong
letters; the answers are not all bitwise identical. All 32 responses stop
naturally and parse correctly, with no truncated generation or thinking
overrun. The small, input-truncated panel cannot establish full-LongBench
equivalence, but it finds no additional LoD failures on these inputs.

Generation/logprob elapsed times include shape warmup and scoring and are
not canonical speed measurements. The current speed table above is the
separately warmed, 1,025-step comparison. The related CPU benchmark/adapter
suite passes **87 tests**, with 20 GPU/runtime tests skipped locally; the
four targeted projected-prefill GPU tests were already passed by the repair.

With the compatible vLLM image/plugin installed and the documented FP8 weight
daemon resident, run the modes **sequentially**:

```bash
python -m benchmarks.glm53_flash_full \
  --checkpoint /local/models/GLM-5.3-Flash \
  --mode full --tp 4 --length 65536 --batch-size 8 \
  --measure quality --quality-samples 8 --sample-offset 8 \
  --longbench-samples 16 --weight-cache-id glm53-flash-tp4 \
  --output results/glm53-flash-full-model/packed-query-fix-quality-full-64k.json
# Then repeat with --mode two-tier and a distinct output file.
```

Add `--preflight-only` to check corpus lengths and chat rendering without
loading model weights. The default generic ProLong selector remains strict;
only this GLM panel explicitly permits shorter documents, then clips the
whole frozen cohort to a recorded common length.

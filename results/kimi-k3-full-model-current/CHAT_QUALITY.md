# Full Kimi K3: LongBench v2 and RULER NIAH-S3

This comparison complements [ProLong quality](PROLONG_QUALITY.md), which found
a +0.1361% pooled PPL increase with shared latent centroids. It tests the
existing full-model two-tier implementation, not the 48-layer speed fixture
or the request-owner/slice-local variants. Matched smoke job
`21250-kimi-k3-longbench-niah-matched-smoke` runs dense, then LoD, sequentially
on node 4. Full evaluations are gated on these checks.

## October 6 cached-routing B1 check

The latest cached-mean/32-split decoder was checked with four freely
generated NIAH-S3 responses each at nominal 16K and 128K. These use ordinary
B1 TP8/DCP8 attention, not request owners. The prompts, native non-thinking
chat format and seed are matched between dense and two-tier LoD. Unlike the
speed panel, **no teacher-forced continuation or trace replay is enabled**.
Generation is greedy and unconstrained, with a 64-token maximum. This is a
quick correctness check, not a full benchmark or a warmed timing run.

| Nominal context | Dense | Latest two-tier LoD |
|--:|--:|--:|
| 16K | 4/4 | 4/4 |
| 128K | 4/4 | 4/4 |

Job `21391-kimi-cached-means-niah-s3-b1-16k128k` completed LoD and dense
sequentially on node 2, reusing its resident transformed full-model weights.
Both arms passed the final eight-worker attention-mode audit; all eight
prompt-token hashes, targets and chat settings match exactly.

LoD completed all eight responses and passed the final loaded-LoD audit on
all eight workers. The actual input lengths are 15,702–15,717 and
130,729–130,735 tokens. Every response contains its entire target UUID,
not just its first token. Source:
[latest B1 LoD NIAH result](oct6-cached-means-niah-s3-b1-16k128k-two-tier.json),
[matched dense result](oct6-cached-means-niah-s3-b1-16k128k-full.json).
This confirms long-context retrieval on this small panel; it does not prove
all workloads are correct or establish a population accuracy estimate.

Run from the repository root with the same kernel environment as
[cached routing](CACHED_ROUTING.md):

```bash
for mode in two-tier full; do
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_quality \
    --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
    --weight-cache-id kimi-k3-node2-full-int4-v1 \
    --mode "$mode" --tasks niah-s3 --batch-size 1 \
    --niah-lengths 16384,131072 --niah-samples 4 \
    --kv-cache-memory-bytes 1073741824 \
    --output "results/kimi-k3-full-model-current/oct6-cached-means-niah-s3-b1-16k128k-$mode.json"
done
```

## Matched protocol

- Full trained 93-layer K3, TP8/DCP8/EP8 on node 4's eight MI325X GPUs.
- Same resident `kimi-k3-shared-int4-v6` groupwise-INT4 **MoE weights** in
  dense and LoD; attention cache records are BF16 in both arms.
- Ordinary two-tier LoD, eight routes, global 16K prefill updates and global
  256-token decode updates. Dense uses the improved Gluon decoder.
- Native K3 XTML chat rendering with **`thinking=False`**. Passing only
  `enable_thinking=False` to this tokenizer does not disable thinking.
- K3's tokenizer ignores `continue_final_message`. For RULER, render a user
  message with an **open assistant response**, then append `gen_prefix` as
  ordinary tokens. Do not render a completed assistant message or re-tokenize
  XTML structural markers as text. The CPU preflight verified this behavior.
- Greedy generation, model seed 0, no prefix caching. Dense/LoD prompt-token
  hashes must match. Both arms use concurrency two to leave headroom beside
  the resident full model, not the Qwen/K2 single-GPU concurrency settings.
- LongBench uses the existing official prompt, first/last-half input truncation
  at 131,072 **raw prompt** tokens, 32 output tokens, and the constrained
  choices `The correct answer is (A)` through `(D)`. Chat markers are added
  afterward and included in the engine context allowance.
- All 503 LongBench examples come from `THUDM/LongBench-v2`, revision
  `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`.
- NIAH-S3 uses the installed `lm-eval==0.4.13` RULER generator, essay haystack,
  word key, UUID value, and its canonical assistant prefix. Python/NumPy
  setup seeds are 0/1234, with the generator's unchanged default seed 42.
  Correctness means the generated response contains the target UUID.
- The runner is **offline vLLM**, not HTTP. It reuses the LongBench prompt,
  truncation, answer grammar, and scorer, but does not time the HTTP frontend.
  Elapsed times include cold kernels; they are operational run times, not
  the canonical warm-serving attention-speed benchmark.

## Preflight and smoke

The smoke panel is eight NIAH-S3 examples each at 32K/64K and eight LongBench
examples selected evenly across the sorted input lengths, including truncated
131K inputs. It is not a full benchmark score. Complete runs will use 128
NIAH examples per length at 8K, 16K, 32K, 64K and all 503 LongBench examples.

CPU preflight passed: the nominal 32K NIAH inputs contain 32,196–32,206 chat
tokens, the nominal 64K inputs 64,861–64,867, and the LongBench smoke inputs
10,090–131,096. RULER's nominal length is a generator budget, not an exact
number of input tokens; its unchanged generator reserves output space and
fits whole essay units.

| Smoke task | Dense | Two-tier LoD |
| --- | ---: | ---: |
| NIAH-S3, nominal 32K | 8/8 | Pending |
| NIAH-S3, nominal 64K | 8/8 | Pending |
| LongBench v2, representative eight | 4/8 | Pending |

The dense smoke completed all eight worker audits. Its generation times were
36.09 s (32K NIAH), 73.23 s (64K NIAH), and 97.60 s (LongBench panel), excluding
dataset preparation and model startup. All eight constrained LongBench answers
parsed. These times are operational context, not an attention-speed comparison.
The complete [dense smoke artifact](oct5-chat-smoke-full.json) retains every
response and prompt hash.

The first LoD smoke failed before scoring with `Kimi expanded local query has
incompatible geometry`. A ragged scheduler slice crossed the exact-first 16K
boundary: the latent query was sliced for the exact prefix, while its expanded
192-dimensional query still covered the entire scheduler slice. The cached
prefill exact branch now slices both identically and clears the temporary on
failure. Two focused CPU regression cases pass. Corrected LoD smoke job
`21251-kimi-k3-chat-lod-smoke-r2` reuses the completed dense control; failed
job `21250` is not a LoD quality result. The matched prompt hashes will be
checked before proceeding to the complete runs.

The second smoke completed the two needle panels but scored 0/8 at each
length, then failed during the first mixed LongBench scheduler batch. These
are failed diagnostic results, not completed benchmark scores. Two additional
implementation faults were found:

1. Mixed prefill/decode rows used rank-local decode without gathering all TP
   query heads and LSE-combining the eight DCP sequence slices. They now use
   ordinary distributed decode; four CPU cases exercise packed/nonpacked
   rows and eager/lazy query absorption.
2. Cache-only construction of a short ragged DCP prefix could archive beyond
   its explicitly requested final boundary. Initial construction now respects
   that boundary; four CPU regression cases cover short/long prefixes.

Job `21257-kimi-k3-chat-lod-smoke-r3` matched all dense prompt hashes but
still scored 0/8 on each needle panel. LongBench reached 2/4 before a GPU
resource failure (`_decode_route_coarse_gqa_groups_kernel`, reported free
memory 0 MB); this is not a completed LongBench score. A separate batch-one
check (`21263`) completed but scored 0/4 at both 8K and 32K. The 8K prefill
is exact, and some responses start with the right UUID prefix before
diverging. Thus the unresolved failure is not explained solely by batching
or approximate long prefill. The real DCP cache/decoder is being checked
numerically against dense attention before more full-model evaluations.

The numerical check isolated a BF16 shadow-conversion bug: the reserved
archive suffix was read as if it held the live recent tokens. That suffix is
not authoritative; recent tokens must come from `recent_k`. Both individual
and layer-batched conversion now concatenate the protected sink, archived
prefix only through `coverage`, and live recent tail, retaining static DCP
ownership strides and the latent K/V alias. Twenty-four CPU cases poison
the unused archive with NaNs and cover every DCP rank. The real GPU decoder
now matches a dense 128-token reference on all 96 heads, both cache rows,
and all eight physical slices (partial LSE maximum error < 2e-6).
Before the fix that same GPU reference failed with output error up to 0.286.

Job `21270-kimi-k3-chat-quality-fixed-tail` first reruns batch-one 8K/32K
needles, then runs the matched batch-two smoke panel only if the short check
passes. It still scored **0/4 at both lengths**, so the larger panel was not
started. The tail repair is necessary but not sufficient. An explicitly
eager 8K rerun (`21271`, CUDA graphs and compilation disabled) also scored
**0/4**. Thus graph replay alone does not explain the remaining failure.
An eager-only diagnostic now replaces the LoD decode result with dense FP32
attention over its same raw DCP archive and recent tail; prefill and weights
are unchanged. This is a correctness diagnostic, not a timing or LoD score.
No full suite is launched based solely on unit tests.

The dispatch trace (`21273`) found the missing case: this full model's Q
projection is **not replicated across DCP ranks** (`qrep=False`). The adapter
incorrectly treated that optional optimization as a prerequisite for DCP
decode. It ran 12 local heads against only this rank's history, without the
eight-way LSE merge. The dense-reference hook consequently received zero
calls in `21272`; that run was **not** a dense-shadow test. Ordinary DCP
decode now gathers the absorbed local-head queries when a replicated query
is absent, evaluates all 96 heads over each owned slice, and LSE-combines
and scatters to the 12 head owners. Already replicated queries keep their
existing no-gather path. A regression test fails on the old nonreplicated
branch and both query variants pass with the repair. The first graph-mode
startup (`21274`) exposed a missing host-side dummy-row map; warmup now
initializes both host and GPU identity maps, with a CPU regression test.
Job `21275-kimi-k3-correct-dcp-gated-smoke` runs four eager 8K needles first,
then graph-mode 8K/32K needles, then the matched batch-two smoke, with an
accuracy gate between stages. All four eager 8K needles succeeded (4/4).
The graph-mode checks also passed 4/4 at 8K and 4/4 at 32K; the matched
batch-two checks are next. These small checks do not replace the full
benchmark. The focused CPU launcher/quality/K3 regression suite passed
269 tests, with 40 GPU-only skips.

The standalone GPU kernel checks passed 314 tests, and the mixed-DCP/owner
regression suite passed another 90 tests on node 3. These unit results do
not establish end-to-end quality. Large quality runs remain gated on the
repaired smoke results.

The runtime needed `wonderwords==2.2.0` plus NLTK's `punkt_tab` resource before
the generator could run; these were installed during CPU preflight, before
loading any model weights. The generator/runtime imports are tested separately
from model startup.

## Reproduction

Use the same prepared K3 v10 userspace and resident weight daemon as
[ProLong](PROLONG_QUALITY.md#reproduction-without-the-cluster-runner), including
its environment/kernel settings. From the repository root:

```bash
# CPU-only prompt/dataset preflight; does not construct a model.
bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_quality \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --mode full --preflight-only \
  --niah-lengths 32768,65536 --niah-samples 8 --longbench-limit 8 \
  --output results/kimi-k3-full-model-current/chat-preflight.json

# Matched smoke tests; run sequentially on the daemon's eight GPUs.
for mode in full two-tier; do
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_quality \
    --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
    --mode "$mode" --batch-size 2 \
    --niah-lengths 32768,65536 --niah-samples 8 --longbench-limit 8 \
    --output "results/kimi-k3-full-model-current/chat-smoke-${mode}.json"
done

# Full tests: omit --longbench-limit and use 128 NIAH examples per length.
for mode in full two-tier; do
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_quality \
    --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
    --mode "$mode" --batch-size 2 \
    --niah-lengths 8192,16384,32768,65536 --niah-samples 128 \
    --output "results/kimi-k3-full-model-current/chat-full-${mode}.json"
done
```

`--tasks niah-s3` or `--tasks longbench-v2` runs only that benchmark. Each
`.prompts.json` records IDs, lengths, targets, and hashes before GPU startup;
`.partial.json` records every completed batch. Only the final JSON includes
the final worker/kernel audit. Failed runs and partial scores must not be
presented as completed benchmark results.

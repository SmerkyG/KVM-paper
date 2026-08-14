# LOD Attention development snapshot

This branch collects the inference-time and training prototypes, optimized
kernels, evaluation scripts, and experiment outputs developed for two-level
LOD attention.

## Main implementations

- `model/pytorch_lod_attention.py`: model-independent PyTorch reference with
  both coarse-only and exact-leaf LOD attention
- `model/pytorch_lod_attention_fast.py`: inference-oriented PyTorch backend
  using SDPA, separately compiled FlexAttention, and packed FlashAttention
- `model/hf_pytorch_lod_attention.py`: registered model-independent Hugging
  Face backend and uniformly LOD-owned cache
- `model/hf_qwen35_lod_attention.py`: Qwen3.5 hybrid-cache compatibility adapter
- `model/hf_diffusion_gemma_lod_attention.py`: isolated DiffusionGemma
  encoder/denoiser adapter
- `model/lm_eval_diffusion_gemma.py`: generation-only lm-eval wrapper for
  DiffusionGemma
- `model/triton_lod_engines.py`: generic kernel-backed post-QKV engines
- `model/triton_lod_attention.py`: optimized post-QKV LOD runtime
- `model/kernels/lod_kernels.py`: state update and routing Triton kernels
- `model/kernels/paged_leaf_attention.py`: paged, virtual-page, recursive-page,
  and quantized leaf attention kernels
- `model/qwen35_two_level_attention.py`: legacy Qwen3.5 graft compatibility shim
- `model/kvm_two_level_mixer.py`: pure-PyTorch training prototype
- `model/gptalpha_two_level_mixer.py`: GPTAlpha2 inference approximation
- `model/kvm_split_full_attention_mixer.py`: full-remote baseline

The corresponding `scripts/` entry points cover ProLong loss, NIAH, RULER and
lm-eval evaluation, profiling, kernel verification, and full-remote comparison.

The serving design for converting an already cached full-attention BF16 prefix
into LOD state is recorded in
[`docs/lod_full_cache_conversion.md`](docs/lod_full_cache_conversion.md). In
particular, INT4 pages are semantic region-owned pages, never arbitrary
consecutive physical-cache blocks.

The first vLLM integration is an editable out-of-tree plugin under
`integrations/vllm_lod`. It registers `--attention-backend CUSTOM`, preserves
native prefill and prefix caching, converts native BF16 K/V into semantic LOD
pages outside graph replay, and uses fixed-pool recursive decode with stable
request indirection. Pool and graph-scratch memory is reserved before vLLM
sizes its native cache, and captured padding uses distinct resettable rows.
Its README lists launch settings and current limitations.

## Model-independent PyTorch API

`CoarseLODAttention` keeps only the low-LOD state and exact local window;
there is no full-history leaf archive. `TwoLevelLODAttention` additionally
keeps the original remote K/V tensors in BF16, routes each query to at most
eight state regions, replaces every opened region with an independently
normalized exact attention, and combines all branches using their LSEs.

Both modules accept post-projection, post-RoPE tensors and leave the usual HF
head flattening, output gating, and output projection to the caller:

```python
from model.pytorch_lod_attention import LODConfig, TwoLevelLODAttention

lod = TwoLevelLODAttention(LODConfig(max_routes=8))
attention_output, lod_cache = lod(
    query,                         # [batch, query_heads, length, key_dim]
    key,                           # [batch, KV_heads, length, key_dim]
    value,                         # [batch, KV_heads, length, value_dim]
    cache=lod_cache,               # omit for prefill
    use_cache=True,
    open_count=8,                  # or [batch, query_heads, length]
)
```

`open_count` is clamped to the number of state regions that actually exist,
so the normal setting means “open up to eight.” Set it to a smaller integer or
a per-query tensor to dynamically reduce exact work. The implementation uses
ordinary PyTorch matmuls, masks, softmax, and LSE merging as a readable
correctness reference; it has no Transformers, Qwen, Triton, or custom-kernel
dependency. Run its focused CPU checks with:

```bash
PYTHONPATH=. uv run python scripts/verify_pytorch_lod_attention.py
```

For inference, replace the class names with `FastCoarseLODAttention` or
`FastTwoLevelLODAttention` from `model.pytorch_lod_attention_fast`. Their API
and cache objects are identical. The fast two-level path uses one fused
FlexAttention operation for the coarse state plus local field, and asks it for
that field's LSE. The exact leaves are dispatched together using posting lists
and produce their own LSE; the two results are then renormalized exactly. SDPA
is used only while the entire attention field is local, where no cross-branch
LSE is needed.

The fast exact-leaf implementation selects between two PyTorch paths: direct
gathering for small routed sets (normally decode), and packed variable-length
FlashAttention for larger prefill work. Posting lists are cached until the
owner table changes. Verify and benchmark it with:

```bash
PYTHONPATH=. uv run python scripts/verify_pytorch_lod_attention_fast.py
PYTHONPATH=. uv run python scripts/benchmark_pytorch_lod_attention_fast.py
```

## Hugging Face replacement

`model/hf_pytorch_lod_attention.py` registers a model-independent backend with
Hugging Face's `AttentionInterface`. The model retains its own projections,
normalization, positional encoding, output gating, and output projection. The
same backend has been checked with Llama, Mistral, and Qwen3 decoder models:

```python
from transformers import AutoModelForCausalLM
from model.hf_pytorch_lod_attention import (
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention_paged import PagedLODConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype="bfloat16"
).cuda().eval()
install_hf_lod_attention(
    model,
    config=PagedLODConfig(
        chunk_size=256,
        local_window=512,
        state_growth_factor=16.0,
        state_min_size=256,
        max_routes=8,
        page_size=16,
        kv_bits=0,  # change only this storage policy to 4 for INT4 leaves
    ),
    open_count=8,
    engine_backend="kernel",  # required for region-owned INT4 pages
)
lod_cache = new_hf_lod_cache(model)
output = model.generate(
    input_ids,
    past_key_values=lod_cache,
    max_new_tokens=128,
)
```

`HFLODCache` owns the exact leaves, low-LOD state, ownership metadata, and local
window in both BF16 and INT4 modes. Its HF `keys` and `values` members are empty
typed sentinels, so the model does not retain a second ordinary KV cache. Cache
updates stage the new post-RoPE K/V block; the registered attention backend
then consumes it in the correct causal order. Beam expansion and reordering are
implemented, while partial cache rollback, padding, and non-causal attention
are rejected explicitly.

The pure-PyTorch paged reference intentionally rejects `kv_bits=4`: its older
physical-page implementation grouped consecutive positions before ownership
was known. The kernel recursive engine is the supported INT4 implementation;
it computes anchors and residual scales only after leaves have been assigned
to semantic state-region pages.

Qwen3.5 interleaves softmax and recurrent linear-attention layers and therefore
still uses `model/hf_qwen35_lod_attention.py` as a compatibility adapter. It
owns its attention K/V in the same way and uses Qwen's hybrid cache only for
linear state and length bookkeeping.

Run the generic multi-model checks, Qwen3.5 compatibility checks, and NIAH
smoke evaluation with:

```bash
PYTHONPATH=. uv run python scripts/verify_hf_lod_attention.py
PYTHONPATH=. uv run python scripts/verify_hf_lod_checkpoint.py \
  --checkpoint Qwen/Qwen3-0.6B --engine-backend kernel
PYTHONPATH=. uv run python scripts/verify_hf_qwen35_pytorch_lod.py
PYTHONPATH=. uv run python scripts/probe_hf_qwen35_pytorch_lod_niah.py \
  --task niah_single_3 --length 8192 --samples 8 --output /tmp/niah3.jsonl
```

## Experiment outputs

The LOD-specific `artifacts/` subdirectories contain JSON, JSONL, and text log
outputs for the full-remote, GPTAlpha2, Qwen3.5, dynamic-opening, prefill,
recursive-page, virtual-page, and INT4 experiments. These are result records,
not model weights.

Model checkpoints and local model caches are intentionally excluded. In
particular, the source worktree's `hf-models/` directory is not part of this
branch.

## DiffusionGemma evaluation

Do not report the corrupted-canvas pseudo-perplexity as ordinary AR
perplexity. It is useful for diagnosing attention approximation, but it is
conditional on a chosen corruption distribution. For a comparable headline
quality metric, run the same generation task, prompts, output budget, sampler,
and random seed with native and LOD attention. The DiffusionGemma wrapper
intentionally rejects lm-eval likelihood requests because HFLM's next-token
factorization is not valid for block diffusion.

For example, matched RULER runs at 8K are:

```bash
uv run python -m scripts.eval_diffusion_gemma_lod_lmeval \
  --mode full --tasks ruler --ruler-length 8192 --batch-size 1 \
  --seed 1234 --output artifacts/diffusion_gemma_lod/ruler8k/full.json

uv run python -m scripts.eval_diffusion_gemma_lod_lmeval \
  --mode lod --tasks ruler --ruler-length 8192 --batch-size 1 \
  --seed 1234 --local-window 768 --recursive-pages \
  --output artifacts/diffusion_gemma_lod/ruler8k/lod.json
```

Add `--apply-chat-template` only when the comparison baseline uses the same
instruction template. The result JSON records wall time and DiffusionGemma's
model-reported mean tokens per forward in addition to task metrics. Run the
CPU wrapper and adapter checks with:

```bash
uv run python -m scripts.verify_diffusion_gemma_lmeval
uv run python -m scripts.verify_hf_diffusion_gemma_lod
```

Decoder routing experiments can keep prompt-state construction fixed while
changing only the exact page budget:

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_diffusion_gemma_lod_lmeval.py \
  --mode lod --tasks niah_single_3 --ruler-length 8192 --limit 64 \
  --batch-size 4 --open-count 8 --decoder-open-count 32 \
  --decoder-routing per_query --output /tmp/dg-route32.json
```

`--decoder-routing` also accepts `canvas_max` (one route set selected by the
maximum score over all canvas queries) and `canvas_cumulative_max` (the same
scores accumulated over denoising calls). These modes do not alter the LOD
state size. The matched 64-example results are recorded in
`artifacts/diffusion_gemma_lod/niah3_8k_routes/summary.json`.

To compare native and LOD logits on the same LOD-controlled denoising and
acceptance trajectory, add `--compare-native-acceptance`. The diagnostic runs
an extra native decoder pass on the identical canvas, cache, self-conditioning
logits, and denoising step, but only the LOD pass controls sampling. It reports
accepted top-1 disagreements, entropy bins, native NLL, position prefixes, and
unique disagreement locations. The 64-example NIAH-3 result is in
`artifacts/diffusion_gemma_lod/acceptance_compare/summary.json`.

To diagnose whether nearly random high-noise canvas queries cause bad LOD
routing, `--early-native-steps N` uses native decoder attention for the first
N denoising calls of every canvas, then returns to LOD. The causal encoder
prefill remains LOD and the LOD state configuration is unchanged.

The matched 64-example NIAH-3 sweep at 8K is recorded in
`artifacts/diffusion_gemma_lod/early_native_sweep/summary.json`. On that build,
k=0/1/2/4 scored 47/49/52/53 out of 64. These are historical top-3-denoiser
results: before the decoder/prefill phase-boundary fix below, the optimized
prefill route budget leaked into the multi-token denoising canvas. The trend
still diagnoses the high-noise phase, but the absolute scores are not top-8
decoder controls. Use a same-build legacy-policy control only with that caveat.
Select that control with `--prefill-policy legacy`; the default is
`optimized`. This changes prefill scheduling only, not the LOD state budget.
For phase-isolation diagnostics, `--encoder-attention native` returns native
encoder attention while still building the same LOD sidecar for an optional
LOD decoder. Combine it with `--early-native-steps 48` for native attention in
both phases.

The historical attention-phase 2x2 is recorded in
`artifacts/diffusion_gemma_lod/attention_phase_factorial/summary.json`.
LOD/LOD, LOD/native, native/LOD, and native/native scored 55/52/56/59 out of
64. Its LOD decoder arms used the accidentally inherited top-3 budget, while
native decoder arms are unaffected. Native/native reproduced all 64 original
full-attention responses exactly; neither native phase alone recovered the
baseline on that sparse trajectory, indicating a coupled phase interaction.
The legacy/optimized LOD-prefill control is in
`artifacts/diffusion_gemma_lod/prefill_decoder_factorial/summary.json` and
is confounded: legacy used decoder top-8 while optimized inherited top-3, so it
must not be used to estimate the encoder-prefill-policy effect.

Add `--compare-attention-phases` to evaluate LOD/LOD, LOD/native,
native/LOD, and native/native logits on every identical step of one
LOD/LOD-controlled trajectory. The diagnostic maintains a native-encoder
shadow cache and reports top-1 repair categories and acceptance-mask changes;
only the LOD/LOD branch samples, accepts, renoises, or self-conditions.
The 64-example result is recorded in
`artifacts/diffusion_gemma_lod/coupled_phase_compare/summary.json`. LOD/LOD
and native/native acceptance masks differed on 13.0% of active positions.
Among 42 reference-accepted top-1 disagreements with LOD entropy below
0.001, 33 required both encoder and decoder attention to be native. This run
also predates the boundary fix and therefore characterizes a top-3 LOD
denoiser, not the intended top-8 configuration.

`--consensus-acceptance observe|apply` runs a wider sparse decoder probe
(top-16 by default) while keeping top-8 logits authoritative. Apply mode keeps
the original per-sequence acceptance count when possible, re-ranks positions
by the maximum primary/probe entropy, and vetoes primary/probe top-1
disagreements. Observe mode executes the identical probe but retains native
DiffusionGemma acceptance as a scheduling/numerical control.

DiffusionGemma's 256-token denoising canvas is a decoder operation even though
its query length is greater than one. The adapter explicitly disables the
generic engine's query-length-selected prefill route budget during these calls,
so optimized encoder prefill remains top-3 while the primary and consensus
decoder views really use their requested top-8 and top-16 budgets.

The corrected 64-example NIAH-3 results are in
`artifacts/diffusion_gemma_lod/consensus_acceptance/summary.json`. Ordinary
top-8, top-8 plus an observe-only top-16 probe, and applied consensus scored
50/64, 44/64, and 48/64. On the observe trajectory, top-16 entropy on tokens
accepted by top-8 averaged 0.193 versus 0.00111 for top-8, a 174x mismatch;
this strongly supports the sparse-confidence diagnosis. The acceptance rule
did not improve over the ordinary top-8 run, however, and the probe roughly
doubled wall time. Keep it experimental and disabled by default. The no-probe
and observe controls also diverged despite a matched seed, so separate-run
DiffusionGemma trajectory variation limits fine-grained score attribution.

`--full-attention-review observe|apply` runs the primary top-8 decoder first,
then reviews top-8-accepted tokens whose entropy is at most 0.001 with a native
decoder pass on the identical LOD-encoder cache and canvas. A reviewed sample
survives when it matches native top-1. Apply mode does not backfill vetoed
positions and uses native logits for self-conditioning only at vetoed
positions. `--full-review-policy native_acceptance` additionally requires the
position to appear in native DiffusionGemma's entropy-bound acceptance mask;
that broader diagnostic policy is not the default. This adds no LOD state and
retains linear-in-prefix decoder attention complexity.

The 64-example NIAH-3 result is recorded in
`artifacts/diffusion_gemma_lod/full_attention_review/summary.json`. The
ordinary top-8 and native-review observe controls both scored 50/64. Exact
native decoder attention disagreed with only 33 of 121,729 reviewed samples on
the observe trajectory. The targeted apply run vetoed 15 of 136,282 reviewed
samples and scored 48/64, so this review rule did not help. The broader native
acceptance-mask rule vetoed thousands of positions and scored 47/64. Decoder
review uses the LOD-computed encoder cache and therefore cannot repair
encoder-side sparse-attention damage, consistent with the coupled-phase result
that most low-entropy repairs required both encoder and decoder attention to be
native. Keep both policies experimental and disabled by default.

## Autoregressive DiffusionGemma diagnostic

DiffusionGemma also contains a causal encoder. Transformers documents its
encoder output as supporting an autoregressive training loss, but does not
expose an AR generation method. `DiffusionGemmaARLM` in
`model/lm_eval_diffusion_gemma.py` greedily decodes from those causal encoder
states through the checkpoint's tied LM head. Select it with
`--generation-mode ar`; `--max-gen-toks` can override a task's output budget.

The 8K NIAH-3 results are recorded in
`artifacts/diffusion_gemma_ar/summary.json`. With a matched batch size of 16
and 64-token output cap, native AR scored 51/64 while optimized-prefill LOD AR
scored 38/64. Thus the checkpoint supports coherent AR generation, but LOD
still loses 13 examples with no diffusion acceptance, canvas decoder, or
decoder routing in the path. This localizes an additional failure to the
causal encoder attention path (prompt prefill and/or incremental AR decode),
not acceptance alone. On the first 16 examples, legacy top-8 prefill scored
12/16 versus 13/16 for optimized top-3 prefill, so the new fast-prefill route
budget alone does not explain the regression. No LOD state setting changed.

## Cross-family causal-LM diagnostic

To separate a DiffusionGemma-specific failure from broader model-family
sensitivity, the generic Hugging Face adapter was evaluated on several
ordinary causal language models. The retrieval test here is **only RULER
NIAH-3 at 8K**, not the complete RULER suite. Each full/LOD pair uses the same
checkpoint, tokenizer, evaluator, 64 examples, and 64-token generation cap.
Instruction/chat checkpoints are evaluated with `--apply-chat-template`; the
Qwen3 and SmolLM3 thinking modes are disabled so the short output budget is
used for the answer. The earlier generic-HF runs omitted the chat template and
mixed a base/hybrid Qwen control with instruction-tuned models. Their NIAH
numbers are superseded and must not be used for cross-family conclusions.

Every LOD run uses the same fixed state configuration: 256-token chunks, a
512-token local window, growth factor 16, minimum state size 256, one protected
prefix token, eight open routes, 16-token recursive pages, and BF16 KV state.
No state-size increase or model-specific quality tuning was used. Only native
global-attention layers are replaced; recurrent or sliding-window layers stay
native.

| Model | Attention layout | LOD layers | Full NIAH-3 | LOD NIAH-3 | Paired full-only / LOD-only | ProLong full -> LOD PPL | PPL increase |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 28 dense global | 28 | 64/64 | 60/64 | 4 / 0 | 22.708 -> 22.856 | 0.65% |
| Gemma 3 1B IT | 22 sliding + 4 global | 4 | 61/64 | 47/64 | 14 / 0 | 37.575 -> 37.972 | 1.06% |
| SmolLM3-3B | 36 dense global | 36 | 61/64 | 43/64 | 19 / 1 | 13.870 -> 13.960 | 0.65% |
| Gemma 4 26B-A4B IT | 25 sliding + 5 global | 5 | 64/64 | 64/64 | 0 / 0 | 931.222 -> 1012.722 | 8.75% |

The table's LOD column uses recursive one-page-per-routed-region attention.
A matched flat two-level ablation, which opens all exact pages in each routed
region without changing the coarse state size or eight-route budget, gives:

| Model | Full | Recursive pages | Flat two-level | Two-level paired full-only / LOD-only |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 64/64 | 60/64 | 60/64 | 4 / 0 |
| SmolLM3-3B | 61/64 | 43/64 | 50/64 | 14 / 3 |

Qwen3's aggregate score is unchanged; two examples improve and two different
examples regress relative to recursive pages. SmolLM3 recovers seven net
examples, but flat two-level LOD still trails full attention by 11. Recursive
page selection therefore contributes to the SmolLM3 regression but does not
explain most of the cross-family difference.

A separate Qwen3 formatting ablation omits `--apply-chat-template` while
leaving the checkpoint, examples, generation limit, BF16 precision, and
recursive LOD policy unchanged. Native attention and LOD both score 62/64.
The paired outcomes are 60 both-correct, two native-only, two LOD-only, and
zero both-incorrect. Thus removing the template eliminates Qwen3's aggregate
64-to-60 LOD gap, but partly by lowering the native baseline from 64 to 62;
it does not make the two attention modes behaviorally identical.

SmolLM3 has a NoPE layer every fourth block. A selective-layer ablation uses
the same recursive LOD policy while leaving all other attention layers native:

| SmolLM3 LOD subset | Layer indices | LOD layers | NIAH-3 |
| --- | --- | ---: | ---: |
| NoPE only | 3, 7, 11, 15, 19, 23, 27, 31, 35 | 9 | 48/64 |
| Depth-matched RoPE only | 2, 6, 10, 14, 18, 22, 26, 30, 34 | 9 | 64/64 |
| All RoPE only | all remaining indices | 27 | 8/64 |
| All layers | 0--35 | 36 | 43/64 |

The nine-layer comparison is particularly clean: all 16 discordant examples
favor the depth-matched RoPE subset, with none favoring the NoPE subset. This
supports the hypothesis that SmolLM3's specialized NoPE layers are more
sensitive to LOD than nearby RoPE layers. The 8/64 all-RoPE score separately
shows severe degradation accumulated over many dense layers. Because adding
LOD at the nine NoPE layers to that 27-layer condition improves rather than
monotonically worsens the score, these effects interact strongly and the
subset score drops must not be added as an independent causal decomposition.

The SmolLM3 result is not a left-padding artifact. Its rendered chat prompts
span 7,463--7,483 tokens, with at most 18 left-padding tokens in a batch of
eight. Removing batching entirely gives 64/64 for native attention and 44/64
for recursive LOD, compared with 61/64 and 43/64 at batch size eight. The
paired batch-size-one result has 44 both-correct and 20 native-only examples.
At batch size eight, switching from chunk-aligned masking to exact physical
padding removal scores 39/64 under LOD, below the 43/64 chunk-aligned result.
Thus batching changes a few borderline generations in both modes, but cannot
account for the large SmolLM3 LOD regression. The small ProLong delta is also
not evidence against this conclusion: ProLong is evaluated one document at a
time without padding, yet token-average loss is simply much less sensitive
than exact UUID retrieval.

Qwen3 is the dense Qwen control: LOD replaces all 28 attention layers, unlike
the earlier hybrid Qwen3.5 control. The route kernel now also supports the
12-query-head / 2-KV-head GQA ratio used by Qwen2.5-1.5B-Instruct; its matched
batch-size-one results are included in the routing analysis below.

ProLong is measured as ordinary causal next-token loss on the same eight
deterministically selected 8,192-token documents for both modes (65,528
predicted tokens per model). It intentionally measures raw causal next-token
loss rather than chat-formatted generation, so the NIAH formatting correction
does not invalidate these paired deltas. Gemma 4's absolute perplexity on
unformatted raw text is unusually high, so its paired loss change is more
informative than the absolute value. All tested checkpoints, including
DiffusionGemma, declare `tie_word_embeddings=true`; embedding tying therefore
does not distinguish the successful controls from the regressing models.

Chat formatting completely eliminates the apparent Gemma 4 regression: both
full attention and LOD score 64/64. This correction invalidates the strongest
earlier evidence for a Gemma-wide problem. The remaining results still show
model-family sensitivity. Dense Qwen3 is robust but not exact at 60/64 under
LOD, while Gemma 3 and SmolLM3 fall to 47/64 and 43/64. Thus applying LOD to
all dense layers is not sufficient to explain the discrepancy: Qwen3 loses
four examples, whereas all-dense SmolLM3 loses 18 net examples.

Average perplexity is not an adequate proxy for this failure mode. Qwen3 and
SmolLM3 both have about a 0.65% ProLong perplexity increase, but lose four and
18 net NIAH examples respectively. Gemma 4 has no NIAH regression despite the
largest average-loss degradation. Exact long-range retrieval and
token-averaged language-model loss should therefore both be reported.

The complete machine-readable summary and per-run outputs are under
`artifacts/hf_lod_family_diagnostic/summary.json`. Use
`scripts/eval_hf_lod_lmeval.py --mode full|lod --apply-chat-template` for the
matched NIAH test and
`scripts/compare_hf_lod_loss.py --mode full|lod` for ProLong. Gemma 4 is loaded
as its text-only causal model with an explicit checkpoint key mapping, avoiding
allocation of unused multimodal towers. Its 512-dimensional GQA heads also
required a route-kernel tiling safeguard: the adapter caps the coarse-route
query block to the available shared-memory budget and selects the logits route
path. This changes kernel execution only, not LOD state, routes, or outputs.

## Post-hoc routing-mass diagnostics on causal LMs

The original closed-slot routing score is
`beta * dot(q, mean_k) + log(count)`, using the model's native attention scale.
A 64-example, batch-size-one NIAH-S3
sweep at 8K tested whether model-family sensitivity comes from a mismatched
query temperature, slot-count prior, or the mass omitted when keys cancel in a
mean. These experiments use the same BF16 state and eight open routes as the
cross-family diagnostic; no run increases the state size.

`--routing-count-bias alpha` changes only route selection to
`beta * dot(q, mean_k) + alpha * log(count)`, where `beta` is the native
attention scale. Coarse attention still uses
the mathematically required coefficient of one, so this does not distort the
closed-slot attention mass after routes have been chosen.
`--routing-normalization query` RMS-normalizes queries for routing only. The
best results were:

| Model | Full | Original LOD | Query norm | Count-only | Query + count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B Instruct | 64/64 | 39/64 | 57/64 | 57/64 (`alpha=2`) | 57/64 (`alpha=1.25`) |
| SmolLM3-3B | 64/64 | 44/64 | 59/64 | 60/64 (`alpha=1.5`) | 59/64 (`alpha=1.25`) |
| Gemma 3 1B IT | 61/64 | 50/64 | 51/64 | 50--53/64 (`alpha=2`) | 53/64 (`alpha=1.5`) |
| Phi-4 | 64/64 | 61/64 | 64/64 | not run | not run |

The repeated Gemma count-only runs straddle three borderline examples, while
the query-plus-count run is the best result from the final path. Relative to
the original LOD outputs, the coefficient-one query-direction formula alone
recovers 18 Qwen2.5 examples, 15 SmolLM3 examples, and one Gemma example. It
still trails native attention by 7, 5, and 10 examples respectively, so it is
a correction for one identified mechanism rather than a claim that all LOD
error is removed. Qwen3.5-0.8B remains a robust control: query normalization
and both `alpha=0.5` and `alpha=2` retain 64/64.

On the same eight ProLong 8K documents, the route changes do not trade away
ordinary next-token loss:

| Model | Full PPL | Original LOD | Query direction (`alpha=1`) | Count-only | Query + fitted count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B Instruct | 15.291 | not run | 15.345 | 15.354 | 15.349 |
| SmolLM3-3B | 13.870 | 13.960 | 13.949 | 13.953 | 13.948 |
| Gemma 3 1B IT | 37.575 | 37.972 | 37.901 | 37.916 | 37.878 |

The coefficient-one query-direction policy is the best LOD variant for
Qwen2.5, is within `0.001` PPL of the fitted combination for SmolLM3, and
improves the original Gemma route without matching its fitted combination.
On Qwen3.5-0.8B it also slightly improves 25.947 to 25.940 PPL (full attention
is 25.843). These results are consistent with a route correction rather than
a general change to the model's attention distribution.

Several more literal corrections to centroid cancellation were also tested.
A scalar second-order variance correction estimates the missing log-mass from
`E[||k||^2] - ||mean_k||^2`; it scores 42/64, 53/64, and 47/64 on Qwen2.5,
SmolLM3, and Gemma 3. Page-centroid log-sum-exp reranking over the best 16/32
candidate slots scores 42/49, 51/48, and 47/49. Neither is competitive with
the much cheaper routing temperature/count changes.

For a definitive cancellation test, `--routing-leaf-mass-candidates 16|32`
computes exact token-level log-sum-exp from the existing leaf pages before
choosing the final eight routes. Both materialized and virtual-page Triton
kernels agree with a direct PyTorch calculation within `4.8e-7`. It adds no
persistent state and, for a fixed candidate count and square-root region size,
keeps prefill `O(n^1.5)`. It is currently an unfused diagnostic and is slow.
On the first 16 NIAH examples, exact total-mass reranking with 16/32 candidates
scores 7/0 for Qwen2.5, 9/1 for SmolLM3, and 13/13 for Gemma 3. Ranking exact
missing mass, `log(Z_exact - Z_centroid)`, scores 1/0, 2/1, and 11/12.

Thus cancellation is real but is not by itself the correct routing objective.
Opening the slots with the largest exact or omitted probability mass can evict
semantically useful routes, because partition-function error does not measure
the corresponding attention-output error; the value vectors matter too. Keep
page/leaf-mass reranking experimental. For the two models without architectural
Q/K normalization, query normalization with the ordinary coefficient-one count
prior is the best shared policy found. Count-only `alpha=1.5` is one example
better when optimizing SmolLM3 NIAH alone, but is a fitted model-wide constant
rather than the mechanistic correction below. The Q/K-normalized controls
motivate the architecture dispatch derived below.

A value-aware follow-up computed the exact single-slot attention-output change
for each of the 32 candidates and ranked its reduction in squared error toward
the output with all 32 candidates opened. Its CUDA implementation matches the
direct formula within `3.0e-7`, but scores only 6/16, 1/16, and 12/16. This
rules out the simple "mass versus value utility" explanation as a complete
fix: independent per-slot utilities still miss interactions among the eight
simultaneously opened regions and can optimize small attention-output changes
that are irrelevant to retrieval. It remains diagnostic and is not part of
the recommended route policy.

### Mechanistic correction: route by query direction, not query temperature

The count sweep does not imply that each architecture needs a manually tuned
`alpha`. Let `beta` be the layer's native attention scale and let

```
r(q) = sqrt(mean(q**2))
```

for one query head and token. Query-normalized routing is

```
score_s = beta * dot(q / r(q), mean_k_s) + log(count_s).
```

Multiplying all candidate scores by the positive scalar `r(q)` does not change
their order, so this is exactly rank-equivalent to

```
score_s' = beta * dot(q, mean_k_s) + r(q) * log(count_s).
```

The formerly tuned count coefficient is therefore replaced by a quantity
computed per query, head, token, and layer. It is deliberately not inferred
from a model-family name or a config constant: Q/K normalization is a useful
architecture-level predictor, but its learned gain and the incoming hidden
state make the actual temperature activation-dependent. On an 8K ProLong sequence, the
mean `r(q)` is 1.975 for Qwen2.5-1.5B and 1.334 for SmolLM3, closely matching
their best fixed count coefficients of 2.0 and 1.5. The layer means span
1.18--2.75 and 0.88--1.55, respectively, explaining why a single model-wide
coefficient is brittle. These two architectures do not have per-head Q/K
normalization. Gemma 3, Qwen3, and Qwen3.5 do; their mean query RMS values are
1.309, 1.503, and 1.359. The architecture therefore tells us whether query
magnitude is the raw projection scale or the result of explicit normalization
and a learned gain. In the former case, `r(q)` supplies the exact adaptive
coefficient for the current activation.

This is a routing/search correction, not an alternate attention law. Query
norm is the softmax temperature of native attention, but it is a nuisance
scale when choosing which regions will be made visible. Removing it prevents
a large-norm, high-inverse-temperature but wrong centroid match from
overwhelming the region-size prior. Closed-region attention still uses the
native query and the exact coefficient-one count correction. Raw key norms
are retained because they can encode trained salience; normalizing keys or
both Q and K is less consistent than query-only normalization.

The failure is specific to routing from lossy centroids. For a leaf key
`k_i` represented by its slot centroid `mean_k_s`, the hidden routing-logit
error is

```
delta_i = beta * dot(q, k_i - mean_k_s),
abs(delta_i) <= beta * norm(q) * norm(k_i - mean_k_s).
```

Query magnitude therefore sharpens the centroid ranking and amplifies its
unobserved error by the same factor. A high native-attention temperature is
not evidence that the correct leaf is represented by the winning centroid;
under sparse visibility it can produce a confidently wrong route. Directional
routing removes that unsupported confidence only while selecting visibility.
[QK normalization](https://arxiv.org/abs/2010.04245) previously identified
vector-norm-driven softmax saturation in ordinary attention; the distinction
here is that normalization is confined to the discrete visibility search, so
the pretrained model's attention temperature and outputs are left unchanged.

Centroid cancellation does not explain this result. Mean-key coherence is
0.886 for Qwen2.5, 0.876 for SmolLM3, 0.937 for Gemma 3, and 0.906 for
Qwen3.5. Its dependence on slot count is nearly flat for the two models helped
most by query normalization. Qwen3.5 has the strongest negative dependence
yet scores 64/64. Exact leaf-level partition-mass reranking also performs much
worse than query-direction routing. The original fixed `alpha` was
compensating for query temperature in a search objective, not repairing a
biased estimate of the softmax partition function.

Phi-4 is a held-out confirmation rather than a model used to choose the rule.
It has no Q/K normalization, its mean 8K query RMS is 1.379, and its
log-coherence/log-count slope is only -0.002. The architecture-aware query
route improves it from 61/64 to 64/64 at the original batch size of eight.
This is the predicted direction even though cancellation is essentially
independent of slot size, strengthening the query-temperature explanation.

### Architecture-specific residual sensitivity

Query direction explains the shared correction, while architecture determines
how much residual LOD error is tolerated:

| Model | Q/K norm | LOD-modified attention | Other relevant structure | Mean last-token attention-output relative error | Final-logit relative error |
| --- | --- | ---: | --- | ---: | ---: |
| Qwen2.5-1.5B | no | 28/28 | all RoPE | 0.052 | 0.019 |
| SmolLM3-3B | no | 36/36 | 9 NoPE layers | 0.092 | 0.043 |
| Gemma 3 1B | yes | 4/26 | post-attention branch RMSNorm before residual | 0.071 | 0.017 |
| Qwen3.5-0.8B | yes | 6/24 | gated attention output; remaining layers are linear attention | 0.051 | 0.012 |

The errors are from matched native/LOD forwards on the same 8K ProLong
sequence and are diagnostic, not a task metric. Query normalization reduces
mean attention-output error from 0.052 to 0.047 on Qwen2.5, from 0.092 to
0.087 on SmolLM3, and from 0.071 to 0.060 on Gemma 3. A single token's logit
distance need not move monotonically with downstream task accuracy, which is
why NIAH and ProLong remain the acceptance tests.

SmolLM3 isolates a second mechanism. With the original routes, applying LOD to
only its nine NoPE layers scores 48/64; nine depth-matched RoPE layers score
64/64, while all 27 RoPE layers score 8/64. Query-direction routing changes
those results to 57/64, 64/64, and 63/64. Thus it almost completely removes
the error accumulated across the 27 ordinary RoPE layers, but seven NoPE-only
failures remain. The layer-drift measurement agrees: mean attention-output
error is 0.149 in NoPE layers versus 0.072 in RoPE layers (0.142 versus 0.068
with query normalization). NoPE sensitivity is therefore a distinct
representation/separability problem, not a different count coefficient.
Indeed, mean query RMS is lower in the NoPE layers (1.175) than in the RoPE
layers (1.386), so their roughly twofold output sensitivity cannot be blamed
on a larger query temperature.
Normalizing both query and centroid directions improves the NoPE-only slice
from 57/64 to 59/64, but an architecture-dispatched combination (both in the
nine NoPE layers, query-only elsewhere) scores 58/64, below the 59/64 global
query-only result. The subset rule is therefore non-compositional evidence,
not a justified architecture heuristic, and is not exposed as an automatic
policy.

The dense Qwen3 control prevents turning query normalization into another
universal knob. Its original batch-size-one route scores 63/64, while query
normalization scores 62/64 (full attention is 64/64). Qwen3 has explicit Q/K
normalization, whereas Qwen2.5 and SmolLM3, the two large recoveries, do not.
For Q/K-normalized modules the post-normalization gain is trained as part of
the attention geometry; treating its magnitude as a nuisance is unjustified.

The architecture-derived policy is therefore
`--routing-normalization qk_norm_aware --routing-count-bias 1`, equivalently

```
alpha(q, module) = r(q),  if the attention module has no query normalization
                   1,     if it has explicit Q/K normalization.
```

The first branch is implemented as query-direction routing and the second is
the native route. Across the measured models this selects Qwen2.5 57/64,
SmolLM3 59/64, and Phi-4 64/64, while preserving Gemma 3 50/64, Qwen3 63/64,
Qwen3.5 64/64, Gemma 4 64/64, and OLMo 3 32B 64/64. It sacrifices Gemma 3's one-example query-normalization
gain rather than fitting a policy to a noisy exception (native Gemma 3 itself
varies around 61/64). The rule has no model-family constant and does not
increase persistent state. NoPE layers, the number of modified layers,
attention-output gates, and branch normalization predict residual sensitivity;
they should not be collapsed into the routing coefficient because they act at
different points in the computation.

| Architecture | Query norm | Policy branch | Architecture-specific interpretation |
| --- | --- | --- | --- |
| Qwen2.5 | no | query direction | raw query RMS averages 1.975; adaptive coefficient matches the fixed `alpha=2` optimum |
| SmolLM3 | no | query direction | fixes ordinary RoPE-layer accumulation; NoPE layers remain unusually sensitive |
| Phi-4 | no | query direction | held-out 61/64 -> 64/64 confirmation at mean RMS 1.379 |
| Qwen3 | yes | native route | normalization hurts 63/64 -> 62/64, so its relearned scale must be preserved |
| Gemma 3 | yes | native route | only 4 global layers are replaced, but branch RMSNorm amplifies residual errors |
| Qwen3.5 | yes | native route | only 6 softmax layers are replaced and the attention output is gated |
| Gemma 4 | yes | native route | only 5 global layers are replaced; both policies remain 64/64 |
| OLMo 3 | yes | native route | 16 of 64 layers are global/LOD; the original route is already 64/64 |

Two larger/cross-generation controls support that separation. Gemma 4
26B-A4B, which has only five global-attention layers modified by LOD, remains
64/64 with both the original and query-direction routes. Dense Qwen3-1.7B has
Q/K normalization and an 8K mean query RMS of 1.503, but is already 63/64 in
the matched batch-size-one original route. The negative query-normalization
result is evidence that raw `r(q)` alone cannot decide whether a correction is
appropriate: the architecture must first say whether that scale has already
been explicitly normalized and relearned. The amount and placement of
approximate attention then predict whether residual route errors turn into a
task failure.

### Fast-RoPE channel routing diagnostic

RoPE is not automatically friendly to a mean-key hierarchy. For rotary pair
`j` in a rotary subspace of width `d_r`, define

```
omega_j  = theta ** (-2*j/d_r)
lambda_j = 2*pi / omega_j.
```

If otherwise similar keys fill a position interval of length `L`, averaging
their rotated representation attenuates that pair by approximately

```
A_j(L) = abs(sin(L*omega_j/2) / (L*sin(omega_j/2))).
```

The fast pairs therefore cancel or acquire an unstable phase in a centroid.
They can still be useful for exact nearby attention and can retain trained
long-range evidence, but they are plausible confounders for which remote slot
should be opened. This differs from generic key-norm cancellation because the
known position frequency supplies a physically meaningful reference scale.

`--routing-rope-filter local_window` forces the diagnostic rule

```
route_mask_j = 0 if lambda_j <= gamma * W_local else 1,
```

on routing queries and mean keys only. The exact BSWA branch already covers
that region, but does not prove that `gamma=1` is optimal. The diagnostic
exposes `gamma` as `--routing-rope-cutoff-factor` and defaults it to one.
The filtered query is rescaled to its pre-mask RMS, preventing the mask from
also changing centroid-logit temperature relative to `log(slot_count)`.
Actual closed/open attention is unchanged, as are state size and leaf pages.
NoPE layers receive no mask. With partial RoPE, the unrotated tail is always
retained. At `W_local=512` and `gamma=1`, this excludes 19 of 64
rotary pairs for SmolLM3, 21 of 64 for Qwen2.5, and 9 of Qwen3.5's 32 rotary
pairs; Qwen3.5's other 192 head dimensions are untouched.

The forced-filter results show that wavelength alone is not a sufficient
policy:

| Model / LOD slice | NIAH-S3 before | With fast-pair filter | ProLong perplexity before -> filtered | Mean attention-output error before -> filtered |
| --- | ---: | ---: | ---: | ---: |
| SmolLM3, all 36 layers | 59/64 | 61/64 | 13.9485 -> 13.9459 | 0.08686 -> 0.08525 |
| SmolLM3, 27 RoPE layers | 63/64 | 63/64 | - | RoPE layers: 0.06835 -> 0.06620 |
| Qwen2.5-1.5B, all RoPE | 57/64 | 56/64 | 15.3448 -> 15.3443 | 0.04711 -> 0.04850 |
| Qwen3.5-0.8B, partial RoPE control | 64/64 | 64/64 | - | - |

Gemma3-1B makes the cutoff sensitivity explicit. Its four LOD-modified global
layers use full 256-dimensional RoPE with `theta=1e6`. A matched
scale-preserving sweep gives:

| `gamma` | Maximum removed wavelength | Removed dimensions | NIAH-S3 |
| ---: | ---: | ---: | ---: |
| 0 (unfiltered) | - | 0/256 | 50/64 |
| 0.25 | 128 | 56/256 | 47/64 |
| 0.5 | 256 | 70/256 | 51/64 |
| 1 | 512 | 82/256 | 48/64 |
| 2 | 1024 | 96/256 | 48/64 |
| 4 | 2048 | 108/256 | 47/64 |

The original `gamma=1` implementation without RMS preservation happens to tie
the unfiltered 50/64, confirming that frequency removal and routing
temperature had previously been mixed. The best corrected point is only one
example above baseline, and its ProLong perplexity is slightly worse
(37.972 -> 37.991), so it is not a recommended hyperparameter. More
importantly, the curve is non-monotone: even within one full-RoPE model,
removing fast channels can help or hurt depending on the boundary. Qwen3.5
remains 64/64 after RMS preservation, but its `gamma=1` mask removes only
18/256 dimensions, versus Gemma's 82/256, so that control cannot establish a
partial/full-RoPE distinction.

SmolLM's NoPE-layer error is unchanged (0.14238 -> 0.14240), localizing the
small gain to the RoPE path rather than accidentally retuning NoPE routing.
Qwen2.5 shows that trained fast pairs can still supply useful routing evidence
despite their phase instability. The near-identical ProLong perplexities show
that this is a sparse retrieval/routing effect, not a broad language-model
quality change. SmolLM's direction is consistent with the RNoPE division of
labor: NoPE supports global retrieval while RoPE favors local/recency behavior
([RNoPE](https://arxiv.org/abs/2501.18795)); SmolLM3 deliberately places a
NoPE layer every fourth layer ([model report](https://huggingface.co/blog/smollm3)).

Partial versus full RoPE is not itself a causal decision rule. An unrotated
subspace provides a possible fallback but does not prove that the trained fast
pairs are expendable; full RoPE likewise does not prove that they are useful.
Qwen3.5's unchanged 64/64 is a safety control, not evidence that partial RoPE
causes the mask to work. The wavelength mask therefore remains an explicit
diagnostic and is not architecture-dispatched automatically.

A justified adaptive rule would need to measure the reliability of each
frequency in the actual LOD representation. Candidate statistics include the
within-slot phase coherence

```
C_sj = abs(sum(i in slot_s, exp(i*omega_j*p_i))) / count_s
```

and agreement between that band's centroid contribution and its constituent
leaf scores. These can be computed transiently from existing leaf pages and
token indices without increasing persistent state. They distinguish “fast but
still predictive” from “fast and confounding” directly; the presence of NoPE
or unrotated dimensions does not.

#### Leaf-derived Jensen-gap diagnostic

The automatic attenuation is important when interpreting the count term. For
slot `s`, let `x_si = scale * q dot k_si`. The ordinary centroid route is

```
coarse_s = log(n_s) + mean_i(x_si),
```

whereas the exact aggregate log-mass is

```
exact_s = logsumexp_i(x_si)
        = log(n_s) + mean_i(x_si) + J_s,
J_s     = log(mean_i(exp(x_si))) - mean_i(x_si) >= 0.
```

Fast-RoPE phase cancellation already drives its contribution to
`mean_i(x_si)` toward zero as a slot gains leaves. It does **not** imply that
`log(n_s)` should shrink: the count is the correct zeroth-order mass when the
band is phase-incoherent. The only omitted term in this mass formula is the
Jensen gap `J_s`. If a
fast band is merely noise, however, restoring its slot-dependent Jensen gap
can make routing worse by rewarding high-variance rather than relevant slots.

Two implementations test that distinction. `--routing-rope-jensen` estimates
per-plane variance from the exact local field and the slot centroid. This
requires no leaf-page scan, but is the wrong estimator: the local field mixes
semantic variation from unrelated keys with within-slot variation. Its
NIAH-S3 results are sharply negative (Gemma3 50 -> 47, Qwen2.5 57 -> 19,
SmolLM3 59 -> 58; Qwen3.5 remains 64).

The more faithful path transiently scans the existing leaves of a fixed number
of coarse candidates (16, 32, or 64) and then selects the final top 8. It adds
`max(exact_band - coarse_band, 0)` to each candidate's original full routing
score. `--routing-leaf-mass-objective rope_jensen` uses the whole rotary
subspace; `fast_rope_jensen` uses only pairs with
`lambda_j <= gamma * W_local`, while leaving the ordinary centroid score and
all trained channels intact; and `slow_rope_jensen` uses the complementary
long-wavelength pairs. Non-power-of-two selected widths are zero-padded only
in this temporary dot-product view. None of these modes enlarges persistent
state. With a constant candidate count, the additional leaf work is
`O(sqrt(T))` per query under the current hierarchy and preserves the
subquadratic decode bound.

The matched 8K NIAH-S3 results are:

| Model | Baseline | Whole-RoPE gap | Fast-band gap (`gamma=1`) |
| --- | ---: | ---: | ---: |
| Gemma3-1B | 50/64 | 53/64 | 48/64 |
| Qwen2.5-1.5B | 57/64 | 58/64 | 58/64 |
| SmolLM3-3B | 59/64 | 56/64 | 60/64 |
| Qwen3.5-0.8B | 64/64 | - | 64/64 |

This confirms the self-averaging point but rejects a universal Jensen
correction. Restoring all rotary variance helps the two full-RoPE controls and
hurts hybrid SmolLM; restoring only the fast-band gap helps SmolLM slightly,
is neutral-to-small on Qwen2.5, and harms Gemma. Therefore the count bias
should remain `log(n_s)` rather than being coherence-scaled, and neither
Jensen mode is enabled automatically. The architecture-dependent sign also
explains why a wavelength cutoff alone cannot be promoted into a corrective
formula. Qwen3.5's perfect fast-band control is especially useful here: the
new candidate refinement does not disturb the small hybrid model where the
ordinary LOD route was already exact on this task.

More fundamentally, exact attention mass is not necessarily the best opening
objective. A slot with many mutually cancelling or low-value leaves can have
large mass without carrying the answer, while a lower-mass slot can have much
larger attention-output consequence. The mixed exact-gap results therefore do
not indicate estimator error: they show that mass correction and retrieval
quality are different objectives.

#### Gemma/Qwen full-RoPE follow-up

Gemma's small positive result is repeatable and identifies a more specific
failure than a generic RoPE cutoff. Because Gemma3's four modified global
layers use full-head RoPE, the whole-RoPE correction algebraically reduces to

```
original full-head centroid score
  + (exact full-head leaf mass - centroid full-head mass)
= exact leaf attention mass
```

for every shortlisted slot. This is not a fitted variance coefficient. It
corrects the Jensen error in replacing `sum_i exp(q dot k_i)` by
`n * exp(q dot mean_i(k_i))`. The latter can badly underestimate a slot that
contains one sharp content match plus many irrelevant leaves.

On matched 500-example Gemma3-1B NIAH-S3 runs:

| Route policy | Correct | Accuracy | Gain over centroid |
| --- | ---: | ---: | ---: |
| Centroid baseline | 372/500 | 74.4% | - |
| Slow-band exact gap, 16 candidates | 415/500 | 83.0% | +8.6 points |
| Whole-RoPE exact mass, 16 candidates | 420/500 | 84.0% | +9.6 points |
| Slow-band exact gap, 32 candidates | 440/500 | 88.0% | +13.6 points |
| Whole-RoPE exact mass, 32 candidates | 441/500 | 88.2% | +13.8 points |
| Whole-RoPE exact mass, 64 candidates | 455/500 | 91.0% | +16.6 points |

The same 64 logged prompts provide causal pairing rather than just aggregate
scores. Whole-RoPE top-16 changes four baseline failures to successes with no
regressions. Top-32 changes eight failures to successes with no regressions.
Four of the additional rescues therefore came from slots that the centroid
placed outside the top-16 shortlist but exact leaf mass promoted into the
opened top eight. The baseline failures include truncated UUIDs and copies of
unrelated nearby links, consistent with a sharp remote match being diluted in
the mean key.

The frequency ablation localizes most of the gain to content-bearing slow
channels. On the 64-example slice, fast-only top-16 is 49/64 and has two gains
but one regression relative to the 48/64 paired baseline. Slow-only top-16 is
52/64, equal to whole-RoPE top-16. At 32 candidates, slow-only gives 54--57/64
across repeated kernel runs, versus a stable 56/64 for whole-RoPE. The
slow-band cutoff sweep peaks when wavelengths below roughly 256--512 tokens
are excluded from the correction; excluding through 1024 or 2048 tokens falls
to 55/64. Thus fast RoPE is not the source of the Gemma improvement, although
the full band contributes about one point on the larger sample.

The effect generalizes to another full-RoPE architecture but not to RNoPE:

| Model | Centroid baseline | Slow-band exact gap, 32 candidates |
| --- | ---: | ---: |
| Gemma3-1B | 48--50/64 | 54--57/64 |
| Qwen2.5-1.5B | 57/64 | 63/64 |
| SmolLM3-3B | 59/64 | 53/64 |
| Qwen3.5-0.8B | 64/64 | 64/64 |

SmolLM3's RoPE layers are not its only global-retrieval path: every fourth
layer is NoPE. Exact attention mass in its RoPE layers can therefore promote
positional/recency mass that is not useful content, while its NoPE layers
already carry global retrieval. The architecture-level rule suggested by
these controls is to consider slow-band or whole-head exact-mass reranking
when full-RoPE layers themselves must perform global retrieval, not simply
whenever a model has RoPE.

Finally, the improvement transfers weakly to ordinary text. On the same
8-by-8K ProLong slice, baseline LOD perplexity is 37.972, slow-band top-32 is
37.865, whole-RoPE top-32 is 37.857, and full attention is 37.575. The
reranker closes about 29% of the LOD-to-full perplexity gap without increasing
persistent state.

#### Corrected-mass dynamic opening

The monotonic 16-to-32-to-64 candidate gain indicates a centroid-recall
failure: Gemma state means are sufficiently diffuse that a sharp matching leaf
can fall outside a short centroid shortlist. Merely applying the old top-p
rule to eight shortlisted centroid logits cannot address that failure. The
generic kernel backend therefore supports a two-budget policy:

1. Review a bounded candidate pool using the leaf/page hierarchy and replace
   each reviewed centroid's coarse log mass with its refined log mass.
2. Sort those refined masses and open the shortest prefix that explains a
   requested fraction of the estimated *complete remote-state mass*, subject
   to a route cap.

If `c_s` is the coarse mass of state slot `s`, `r_s` is the refined mass for a
reviewed candidate set `C`, the denominator is

```
Z_corrected = sum_s exp(c_s) - sum_{s in C} exp(c_s)
              + sum_{s in C} exp(r_s).
```

This is different from normalizing top-p only over `C`: unreviewed state mass
remains in the denominator. It adds no persistent state entries. The evaluator
exposes it as `--routing-leaf-mass-top-p`; `--open-count` is the maximum actual
route count, while `--routing-leaf-mass-candidates` is the transient review
pool. `--routing-leaf-mass-min-routes` can impose a fixed route floor. The
evaluator records opened-route histograms and means.

The encouraging 64-example result did not survive the full evaluation. On 500
matched Gemma3-1B NIAH-S3 prompts:

| Opening policy | Mean decode pages/head | NIAH-S3 |
| --- | ---: | ---: |
| Fixed 8 | 8.00 | 455/500 |
| Dynamic 90%, cap 16, no floor | 9.83 | 447/500 |
| Dynamic 90%, cap 16, floor 8 | 12.16 | 456/500 |

Without a floor, 42% of route rows open fewer than eight pages and accuracy
falls by eight examples. Adding the floor recovers nine examples, but the
extra 4.16 pages/head beyond fixed eight buy only one example. Nearly 49% of
decode route rows hit the cap of 16. Corrected mass is therefore useful for
estimating how diffuse attention is, but it is not a useful quality signal for
allocating extra final routes on Gemma. In particular, low sparse-attention
entropy can still be confidently wrong when the matching centroid was never
reviewed.

#### Dynamic centroid review

There are two different dynamic budgets. The experiment above varies how many
final regions contribute exact attention *after* a fixed candidate pool has
been refined. The monotonic candidate-width result instead suggests varying
how many coarse centroids are inspected by the finer page/leaf estimator while
keeping eight final routes fixed. The kernel now exposes this independently as
`--routing-leaf-mass-review-top-p`: sort the coarse remote-state masses, review
the shortest prefix reaching the requested fraction of complete coarse mass,
and clamp that prefix to `[final route count, candidate cap]`.

On the matched 64-example Gemma slice with a 64-centroid cap:

| Coarse mass target | Mean centroids reviewed at decode | NIAH-S3 |
| --- | ---: | ---: |
| 75% | 31.68 | 49/64 |
| 90% | 41.43 | 53/64 |
| 95% | 47.05 | 52/64 |
| 99% | 52.57 | 56/64 |
| Fixed 64 | 64.00 | 57/64 |

The 90% policy on Qwen3.5-0.8B reviews only 31.87 centroids at decode and still
scores 64/64. Gemma needs a 99% target and most of the fixed cap merely to
approach fixed-64 quality. This is stronger evidence for centroid blur than
route entropy alone: the useful Gemma region can have both a low coarse rank
and low assigned coarse mass, then be promoted by its page/leaf summaries.
Consequently, a coarse-mass top-p policy cannot reliably know when to stop
searching. A useful adaptive replacement needs an uncertainty signal about
the centroid approximation itself--for example within-slot page-score spread
or mean-key norm cancellation--rather than confidence computed from the
coarse logits it is trying to correct.

The 8-by-8K ProLong control for corrected final-route mass gives perplexity
37.869, versus 37.972 for baseline LOD, 37.857 for fixed 32-candidate
refinement, and 37.575 for full attention. This is neutral-to-positive on
ordinary text, but does not rescue dynamic routing as an NIAH improvement.
Neither dynamic mechanism adds persistent state entries.

#### Centroid formation: spherical assignment

Broad leaf review diagnoses a bad coarse shortlist but does not repair the
state that produced it. A fixed-state alternative now changes only the
geometry used while assigning overflow leaves to state entries. The stored
state remains the exact full-key/full-value sums and counts, and final coarse
and opened attention are unchanged. In spherical mode the online assignment
and farthest-leaf append scores use

```
u(k) = k / rms(k)
score(leaf i, slot s) = u(k_i) dot u(mean_k_s).
```

This removes both leaf-length and centroid-coherence scale from the clustering
decision. It is not equivalent to normalizing the keys used by attention.
Only transient clustering tensors are normalized; state capacity, page count,
and the eight final routes are identical. The evaluator exposes this as
`--state-clustering-normalization cosine`.

On the matched 64-example Gemma3-1B chat-templated NIAH-S3 slice:

| State assignment | NIAH-S3 |
| --- | ---: |
| Raw dot-product baseline | 48/64 |
| Ignore fast RoPE pairs | 49/64 |
| Normalize leaves only | 49/64 |
| Normalize centroids only | 40/64 |
| Normalize both (spherical) | **58/64** |
| Diagonal shared-query metric, raw dot | 44/64 |
| Diagonal shared-query metric + spherical | 58/64 |
| Ordinary Euclidean/L2 | 37/64 |
| Full query-covariance Mahalanobis/L2 | 27/64 |

The paired spherical result rescues 12 baseline failures and regresses two,
leaving only four prompts wrong in both runs. The one-sided controls are
important: the gain cannot be described as merely dividing out the mean-key
norm. The destination comparison and the cross-leaf append comparison must
both live in the same unit-sphere geometry. Otherwise the scores used to
decide which leaves become new centroids are not calibrated with the scores
used to decide which existing centroid absorbs the others.

The larger confirmation is 438/500 (87.6%) with spherical construction,
versus 372/500 (74.4%) for the same raw-centroid configuration. The older raw
run did not retain sample logs, so the 13.2-point difference can only be
checked as an aggregate comparison; the paired raw-versus-spherical evidence
is the 64-example result above. It also exceeds the 420/500 result from
inspecting 16 final candidates, while still opening only eight routes. The
improvement therefore comes from a better fixed-size state, not from reviewing
a larger fraction of it.

Matched full attention is 480/500 (96.0%) on exactly the same 500 prompt
hashes as the spherical run. Of those prompts, 436 are correct under both, 44
are full-only, two are spherical-LOD-only, and 18 are wrong under both. Thus
the remaining gap is not solely an LOD failure--full attention itself misses
20 prompts--but spherical LOD remains 8.4 points below the measured model
ceiling.

The state itself becomes measurably less diffuse. On the same 8K ProLong
geometry probe, mean centroid resultant length/coherence rises from 0.9367 to
0.9540 at exactly the same mean 3.8066 leaves per slot. This supports a causal
chain from spherical assignment to tighter directional clusters to improved
coarse recall; it is not just a lucky final-routing hyperparameter.

MQA is a plausible reason that Gemma3-1B is unusually sensitive: its single KV
head must serve four query heads, whereas Gemma3-4B has four KV heads. It is
not, however, a sufficient explanation. Qwen3.5-0.8B has the same four queries
per KV head and works under ordinary LOD. More directly, weighting clustering
by the diagonal covariance of all query heads sharing a KV head hurts, and the
exact full-covariance expected-logit-error objective hurts much more. Averaging
the query geometries therefore does not repair Gemma's centroids. The useful
signal is key direction itself; MQA may make that angular code more
multimodal, but it does not yield a special covariance correction.

Nor are Gemma's raw centroids simply lower-norm than Qwen's by this coherence
measure: the comparable raw 8K probe gives mean key coherence 0.9367 for
Gemma3-1B and 0.9062 for Qwen3.5-0.8B. Cross-family coherence magnitudes are
not directly a task-margin measurement, but this reverses the prediction of a
literal "MQA makes the means blurrier" explanation. Gemma is more sensitive
to which directions are mixed, rather than exhibiting more cancellation in
the aggregate.

As a non-regression control, Qwen3.5-0.8B remains 64/64 with spherical state
construction. Thus the correction preserves the architecture where ordinary
LOD was already perfect, even though Qwen has the same 4:1 query-to-KV group
ratio.

The KVM-style interpretation is supported by a wider spherical-only sweep.
These runs change state construction only: routing uses the original queries,
stored key/value sums remain unnormalized, and final coarse/opened attention
is unchanged. All rows contain 64 NIAH-S3 examples at 8K:

| Model | Raw state | Spherical state | Spherical rescues / regressions |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B Instruct | 39/64 | **63/64** | 24 / 0 |
| Qwen2.5-7B Instruct | 51/64 | **64/64** | 13 / 0 |
| SmolLM3-3B | 44/64 | **63/64** | 20 / 1 |
| Phi-4 | 61/64 | **64/64** | 3 / 0 |
| Qwen3-1.7B, no chat template | 62/64 | 62/64 | 2 / 2 |
| Qwen3.5-0.8B, no chat template | 64/64 | 64/64 | 0 / 0 |
| Gemma3-1B IT | 48/64 | **58/64** | 12 / 2 |
| Gemma3-4B IT | **64/64** | 61/64 | 0 / 3 |
| Gemma 4 26B-A4B IT | 64/64 | 64/64 | 0 / 0 |
| Muse-Glimmer-30B | 64/64 | 64/64 | 0 / 0 |
| OLMo3-32B | 64/64 | 64/64 | 0 / 0 |

Prompt hashes and batch sizes are matched within every row. The relevant
mechanism is consequently not specific to Gemma or MQA.
It is the same basic effect as renormalizing keys in KVM: averaging introduces
a cancellation-dependent norm that is not part of individual-token QK
normalization, and letting that norm control state formation can make clusters
poorly separated.

The older query-normalized routing is not responsible for these gains. Clean
factorial controls give:

| Model | Raw state + raw query | Raw state + normalized query | Spherical state + raw query | Spherical state + normalized query |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B | 39/64 | 57/64 | **63/64** | **63/64** |
| Qwen2.5-7B | 51/64 | 58/64 | **64/64** | **64/64** |
| SmolLM3-3B | 44/64 | 59/64 | **63/64** | **63/64** |
| Phi-4 | 61/64 | 64/64 | **64/64** | **64/64** |

For Qwen2.5-1.5B, Qwen2.5-7B, and Phi-4, adding query normalization to the
spherical state changes no correctness outcomes. SmolLM swaps one correct and
one incorrect example with no aggregate change. Query normalization was a
partial compensation for a poor raw state; spherical construction directly
fixes the larger error source on this retrieval slice.

This is not yet a universal default. On the matched 8-by-8K ProLong slice,
spherical construction changes perplexity from 37.972 to 38.106 (full
attention is 37.575). Six of eight sequences regress slightly and two improve.
Thus spherical assignment is the first direct centroid-side NIAH improvement,
and is dramatically cheaper than inspecting 64 centroids, but its small
ordinary-text tradeoff should be retained in any architecture policy rather
than hidden by automatically enabling it everywhere.

The geometry probe narrows but does not fully explain that tradeoff. All four
global layers become more directionally coherent, but spherical assignment
makes occupancy more skewed; for example layer 11's largest slot grows from
136 to 495 leaves. A direct value-side check rejects the simplest payload-blur
explanation: mean value coherence also rises in every layer, from 0.9190 to
0.9449 overall. The remaining ProLong cost therefore correlates with the much
coarser occupancy/mass distribution, not with ordinary key or value norm
cancellation. A follow-up should measure within-slot attention-score/Jensen
dispersion or regularize spherical occupancy, rather than returning to broad
centroid review or adding more state entries.

### Preserving radial key distinctions during spherical state construction

Pure spherical construction deliberately identifies keys that have the same
direction. That can discard a real distinction when post-QK-normalization key
length varies: two nearly collinear keys can still produce different attention
logits because `q dot k` retains key magnitude. Gemma3-4B makes this concern
concrete. The token-to-token RMS coefficient of variation in its global
layers averages about 10.4%, compared with about 3.7% in Gemma3-1B, and its
learned `k_norm` gains are much more anisotropic in later layers.

The experimental radial correction uses the scale-free clustering score

```
u(k) = k / rms(k)
score(k, c) = u(k) dot u(c) / d
              - lambda * abs(log(rms(k) / rms(c))).
```

Using a log ratio rather than `abs(rms(k)-rms(c))` is important: it gives the
same answer after a layer-wide rescaling and treats `2x` and `1/2x` norm
changes symmetrically. The implementation appends `log(rms)` only to the
transient routing tensor. Stored key/value sums, counts, state capacity, and
page capacity are unchanged. The CLI controls are
`--state-clustering-radial-bias` and
`--state-clustering-radial-scope {all,append,assignment}`; a nonzero bias
requires spherical (`cosine`) construction.

On the matched Gemma3-4B 64-example NIAH-S3 slice, the coefficient sweep is:

| State construction | NIAH-S3 |
| --- | ---: |
| Raw dot product | 64/64 |
| Spherical, no radial term | 61/64 |
| Radial lambda=0.125 | 63/64 |
| Radial lambda=0.25 | **64/64** |
| Radial lambda=0.5 | 63/64 |
| Radial lambda=1.0 | 63/64 |
| Radial lambda=2.0 | 63/64 |

At lambda 0.25 the radial run rescues exactly the three spherical failures
(indices 28, 42, and 60) and introduces no regression. Applying that term only
when selecting new centroid seeds (`append`), only when assigning leaves to
centroids (`assignment`), or in both places produces the same 64/64
correctness vector. Thus either use of norm is sufficient to perturb the 4B
state away from the bad spherical partition on this slice; this result does
not uniquely localize the gain to seed selection.

The Gemma3-1B control prevents treating lambda 0.25 as a universal constant:

| Gemma3-1B construction | NIAH-S3 |
| --- | ---: |
| Spherical | **58/64** |
| Radial 0.25, both roles | 56/64 |
| Radial 0.25, assignment only | 56/64 |
| Radial 0.25, append only | 49/64 |
| Radial 0.5, both roles | **58/64** |
| Radial 1.0, both roles | **58/64** |

The equal 58/64 aggregates at 0.5 and 1.0 hide paired swaps, so they are
controls rather than evidence of invariance. More importantly, seed-only norm
separation is actively harmful on 1B even though it is sufficient to repair
the 4B slice.

Three cross-family controls with the same spherical-plus-radial lambda 0.25
configuration reinforce that restriction:

| Model | Spherical | Radial 0.25 |
| --- | ---: | ---: |
| Qwen2.5-1.5B Instruct | 63/64 | 63/64 |
| Phi-4 | 64/64 | 64/64 |
| SmolLM3-3B | **63/64** | 58/64 |

The correction is therefore neutral on two families but materially harmful on
SmolLM3. Gemma3-4B is not enough evidence to dispatch this by the presence of
QK normalization, partial/full RoPE, or dense/hybrid attention alone.

Needle membership traces also refine the original "blurry centroid" account.
For the three 4B spherical failures, the 33 UUID tokens occupy 27.27 distinct
slots per layer/head on average under spherical construction, versus 18.70
under raw assignment; the median total size of a slot containing a target
token falls from three leaves to one. The successful radial-0.25 state is only
slightly less fragmented at 26.50 slots and median size one. The spherical
failure is therefore not caused by putting the UUID into a few unusually large
and blurry target slots. It is a downstream routing-coherence failure after
aggressive fragmentation. Radial separation changes which fragments coexist
and which page summaries win, rather than broadly reducing slot size.

A first attempt to derive lambda from aggregate within-cluster residuals does
not work. The measured
`mean(1-cosine)/mean(abs(delta log RMS))` is about 0.95 for both Gemma3-1B and
Gemma3-4B, so it cannot explain their different response to the radial term.
This option remains experimental until the meaningful learned norm variation
can be separated from norm loss caused by directional cancellation in a
centroid. Enabling it architecture-wide based only on the 4B lambda sweep
would reproduce the kind of unexplained per-model tuning that the geometry
work is intended to eliminate.

#### Restoring the mean constituent-key norm

A more literal way to separate genuine key length from cancellation is to
keep, for each slot, the running sum of its constituent leaf-key RMS norms.
Immediately before using a centroid for state construction, form

```
r_bar_s = sum_i rms(k_i) / n_s
c_route_s = c_s / rms(c_s) * r_bar_s.
```

This preserves the direction of the ordinary vector-sum centroid while giving
it the average radial length of the underlying keys. The stored key and value
sums are unchanged. It adds no state slots or KV channels: the only persistent
metadata is one FP32 scalar per centroid (less than 0.2% of K+V state bytes for
a 128-dimensional key and value). The experimental switch is
`--state-clustering-centroid-rescale mean_leaf_norm`; it is disabled by
default and is incompatible with the spherical and explicit radial-penalty
modes so that its effect can be identified cleanly.

The matched 64-example NIAH-S3 sweep is negative:

| Model | Raw dot-product state | Spherical state | Mean-leaf-norm rescale |
| --- | ---: | ---: | ---: |
| Gemma3-4B IT | **64/64** | 61/64 | 60/64 |
| Gemma3-1B IT | 48/64 | **58/64** | 30/64 |
| Qwen2.5-1.5B Instruct | 39/64 | **63/64** | 54/64 |
| Phi-4 | 61/64 | **64/64** | 63/64 |
| SmolLM3-3B | 44/64 | **63/64** | 41/64 |

This reveals why the direct correction is not principled even though its
statistic is useful. The ordinary centroid radius can be decomposed as

```
rms(mean_i k_i) = mean_i(rms(k_i)) * directional_coherence_s,
```

where the equality defines a weighted directional-coherence factor. Direct
rescaling sets that factor to one. It therefore makes a slot whose keys point
in many different directions look just as radially strong as a tight slot
with the same constituent norms, promoting incoherent summaries. The broad
regression shows that cancellation attenuation is not merely an accidental
length bias; it carries useful evidence about whether a centroid is a good
representative.

Consequently the running norm sum should not replace centroid length. If it is
retained for future work, it should expose the two quantities separately:
mean constituent norm captures genuine token-scale variation, while their
ratio to the resultant centroid norm captures coherence. Any corrective rule
must use both rather than erasing the latter. The direct rescale remains only
as a diagnostic implementation.

#### Coherence-preserving, position-aware construction

The useful quantity in the norm accumulator is the resultant length

```
rho_s = rms(mean_i(k_i)) / mean_i(rms(k_i)).
```

Dividing a centroid only by its mean constituent norm gives the assignment
score

```
c_route_s = mean_i(k_i) / mean_i(rms(k_i))
score(k, s) = k dot c_route_s.
```

For a fixed leaf, its own norm is common to every destination. The comparison
therefore retains `cosine(k,c) * rho_s`: it removes genuine average token
scale from the destination while retaining cancellation as evidence that a
centroid is a poor representative. Applying this only during leaf assignment
and retaining direction-dominated farthest-leaf seeding is exposed as
`--state-clustering-centroid-rescale coherence
--state-clustering-centroid-rescale-scope assignment`.

A superficially cleaner normalized-key k-means objective does not work. If
`m_s = sum(k_i) / sum(rms(k_i))`, exact nearest-centroid assignment of a unit
key would maximize

```
unit(k) dot m_s - 0.5 * ||m_s||^2.
```

The `direction_l2` diagnostic implements that expression without a fitted
coefficient. Its radius penalty over-corrects: it scores 52/64 on Gemma3-1B,
62/64 on Gemma3-4B, 62/64 on Qwen2.5-1.5B, 64/64 on Phi-4, and 56/64 on
SmolLM3. Thus the origin of normalized-key space is not a meaningful shared
prototype; centroid coherence must influence assignment monotonically rather
than appear as a Euclidean radius penalty.

The initial positional-design explanation for the SmolLM3 exception is
falsified. LOD states are layer-local, so there is no requirement for RoPE and
NoPE layers to learn matching centroid partitions. Hybrid GDN decoders are
also a direct counterexample to the proposed need for one clustering geometry
throughout a jointly trained model.

The missing SmolLM3 factorial confirms that the two layer classes contribute
independently:

| SmolLM3 construction | NIAH-S3 |
| --- | ---: |
| coherence on every layer | 59/64 |
| spherical NoPE, coherence RoPE | 61/64 |
| coherence NoPE, spherical RoPE | 61/64 |
| spherical on every layer | **63/64** |

Both layer types benefit from spherical construction by two points; neither
is uniquely responsible, and there is no interaction attributable to
cross-layer co-adaptation.

The architecture variable that does predict the useful geometry is explicit
K normalization. With head-wise QK normalization, the key entering attention
has the form

```
k_hat = gain * k / rms(k).
```

The token-dependent global activation scale has therefore been removed before
the fixed learned channel gain. `rho_s` is then a calibrated measure of
directional cancellation. Without explicit K normalization, `rho_s` is a
norm-weighted resultant: raw activation magnitude changes which leaves
dominate both the centroid direction and its apparent coherence. Spherical
construction removes that unreliable radial evidence from state routing.

The `spherical_coherence` control separates centroid admission from
assignment:

```
admission score = cosine(k, c)
assignment score = cosine(k, c) * rho_s
```

It remains 64/64 on Gemma3-4B. On SmolLM3 it reaches only 60/64, compared with
59/64 for ordinary coherence and 63/64 for fully spherical construction:
normalizing leaf admission repairs one error, while retaining uncalibrated
`rho_s` costs three. Gemma3-1B shows the inverse result: spherical coherence
scores 59/64, between spherical at 58/64 and ordinary coherence at 61/64.
There, calibrated `rho_s` repairs one error and retaining the normalized
model's radial admission signal repairs two more. This crossed control ties
the policy to K-scale calibration rather than positional layout.

`--state-clustering-policy qk_norm_aware` consequently resolves every routed
layer independently:

1. If the attention module explicitly normalizes K, use coherence-aware
   assignment.
2. Otherwise, use spherical construction.

This is not a model-name or hybrid-layout dispatch and has no fitted threshold
or coefficient. The state keeps the same slots and stored K/V channels;
coherence-aware layers add only the existing FP32 norm sum per centroid.

The batched implementation does not reconstruct those route keys or allocate
one-hot norm updates on every chunk. A Triton preparation kernel builds a
transient spherical/coherence route-key cache once, subsequent updates refresh
only the appended and merge-destination slots, and the existing fused state
update atomically accumulates the FP32 key-norm sum. The cache changes neither
the number of LOD state slots nor the stored attention K/V representation.

For batch eight with 256 overflow leaves and 1,448 active state slots on
MI325X, the exact cached MFMA route timings are:

| KV geometry | D=128 | D=256 |
| --- | ---: | ---: |
| Raw construction | 0.0643 ms | 0.1195 ms |
| Spherical, cached | 0.0471 ms | 0.0874 ms |
| Coherence, cached | 0.0653 ms | 0.1362 ms |
| Spherical, plus 256-slot refresh | 0.0538 ms | 0.0965 ms |
| Coherence, plus 256-slot refresh | 0.0730 ms | 0.1476 ms |

The 256-slot refresh is a conservative upper bound because an 8K chunk usually
changes fewer unique destinations. Assignment indices, top-16 append sets, and
BF16 scores are exact against dense reconstruction in both the full-build and
sparse-refresh verifiers. A 64x64/8-warp score-streaming variant remains
available when avoiding the transient score matrix matters more than latency.

On the matched 64-example 8K NIAH-S3 slice, the resulting single policy
preserves the best observed result in every diagnostic family:

| Model | K norm | Raw | Spherical | Coherence assignment | QK-norm-aware policy |
| --- | :---: | ---: | ---: | ---: | ---: |
| Gemma3-1B IT | yes | 48/64 | 58/64 | **61/64** | **61/64** |
| Gemma3-4B IT | yes | **64/64** | 61/64 | **64/64** | **64/64** |
| Qwen2.5-1.5B Instruct | no | 39/64 | **63/64** | **63/64** | **63/64** |
| Phi-4 | no | 61/64 | **64/64** | **64/64** | **64/64** |
| SmolLM3-3B (RNoPE) | no | 44/64 | **63/64** | 59/64 | **63/64** |
| Qwen3-1.7B base | yes | 62/64 | 62/64 | **64/64** | **64/64** |
| Qwen3.5-0.8B (partial RoPE, GDN hybrid) | yes | **64/64** | **64/64** | **64/64** | **64/64** |

Merely increasing final opened pages does not produce this uniformity.
Spherical construction with 16 rather than eight pages remains 58/64 on
Gemma3-1B, reaches only 62/64 on Gemma3-4B, and regresses SmolLM3 from 63/64
to 60/64. The policy repairs the coarse partition instead of masking it with a
larger downstream review budget.

The selected geometries do not exchange retrieval quality for ordinary-text
loss on the matched 8-by-8K ProLong controls already run:

| Model | Full attention | Previous best LOD control | Selected geometry |
| --- | ---: | ---: | ---: |
| Gemma3-1B | 37.5749 | **37.9014** | 37.9108 |
| Qwen2.5-1.5B | 15.2915 | 15.3448 | **15.3313** |
| SmolLM3-3B | 13.8700 | 13.9485 | **13.9214** |
| Qwen3.5-0.8B | 25.8431 | **25.9396** | 25.9407 |

The Gemma and Qwen3.5 differences from their previous query-normalized
controls are respectively 0.025% and 0.004% in perplexity, while Qwen2.5 and
SmolLM3 improve. This also avoids the Gemma spherical-state regression to
38.1056.

##### Partial-RoPE coherence control

For a partial-RoPE key, whole-key coherence mixes cancellation in the rotated
and unrotated subspaces. A stricter positional hypothesis would use

```
rho_rope_s = rms(mean_i(k_i[rope])) / mean_i(rms(k_i[rope]))
```

as the scalar reliability of the full centroid direction. The
`rope_coherence` diagnostic implements exactly this while reusing the same
single FP32 accumulator; it adds no state relative to ordinary coherence.

Two 25%-RoPE controls do not support changing the default. Both Qwen3.5-0.8B
and Gemma 4 26B-A4B remain 64/64 on the matched NIAH-S3 slice. On ProLong,
Qwen3.5 changes from 25.94072 with whole-key coherence to 25.94035 with
RoPE-only coherence. Gemma 4 changes from 1038.949 to 1038.993. These deltas
are negligible compared with the choice of whether to apply coherence at all.

The interpretation is that resultant length is not only a positional-phase
statistic. Cancellation among unrotated content channels also measures how
well one direction represents the slot. Partial RoPE therefore determines
which channels may exhibit phase cancellation, but it does not make the other
channels irrelevant to centroid reliability. `rope_coherence` remains a
diagnostic rather than the architecture policy.

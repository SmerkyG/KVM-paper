# LoD Attention

LoD Attention is exact for selected high-mass regions and approximate for the
low-mass remainder. It represents remote context with count-corrected semantic
centroids, refines the four best regions with exact leaves, and combines those
results with an exact local window and protected sink through log-sum-exp.

This branch is the minimal inference release for the LoD Attention paper. It
contains one fixed production policy, not the research-time tuning matrix.

## Supported configurations

| Mode | Remote detail | Leaf storage |
|---|---|---|
| `two-tier` | every leaf in each selected centroid | BF16 |
| `three-tier-bf16` | best semantic page in each selected centroid | BF16 |
| `three-tier-int4` | best semantic page in each selected centroid | residual INT4 |

All modes use exactly four routed regions in prefill and decode, a
`16 * sqrt(T)` centroid schedule, a 16K prefill catch-up, a 512-token base
decode window, one separately protected sink, and an exact first 16K prefill
region. Decode catch-up occurs every 256 tokens, except that K2 INT4 uses a
fixed 512-token interval to amortize quantized-page maintenance. Ordinary
decode scans every retained leaf only while the context is at most 2K; INT4
then differs solely by residual-quantization error. DFlash2 stays routed at all
lengths because its one-token and multi-token verifier graphs share one pool.
With vLLM prefix caching
enabled, the exact rollback tail is 1,024
tokens so a retained request can be rewound without restoring native K/V.
Three-tier pages contain 16 leaves. INT4 is applied only to residuals within a
centroid-owned semantic page; sequential K/V blocks are never quantized as if
they were semantically coherent.

The release supports:

| Model family | Hugging Face | vLLM | DFlash2 |
|---|---:|---:|---:|
| Qwen3.8 (`D=256`, GQA 6) | yes | yes | yes |
| K2 Horizon (`D=128`, GQA 8) | yes | yes | no |

Model-specific compatibility code is isolated in
`integrations/vllm_lod/vllm_lod_plugin/models/`. The attention engine and
kernels in `lod_attention/` operate on post-QKV, post-RoPE tensors and do not
own model projections.

## Install

Python 3.12, PyTorch, Transformers 5.15, Triton, and the platform attention
kernels are required. The vLLM integration is validated against vLLM 0.27.1 on
ROCm. Install this project into the environment that already provides the
appropriate accelerator build:

```bash
uv pip install -e .
```

The optimized prefill path requires the AITER change in
`integrations/vllm_lod/patches/aiter-mha-prefill-route4.patch`. Apply it to the
AITER source used by the runtime and rebuild AITER before benchmarking. The
patch provides compile-time normalized and raw routing probes. LoD selects the
normalized specialization for K2 and automatically builds a separately cached
raw specialization for Qwen; neither kernel branches on normalization at run
time.

## Hugging Face

Installation happens after model construction and replaces only global causal
attention layers. The model keeps ownership of projections, RoPE,
normalization, gating, and output projection; LoD owns its K/V cache.

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lod_attention import install

checkpoint = "Qwen/Qwen3.8-27B-FP8"
config = AutoConfig.from_pretrained(checkpoint)
# Transformers 5.15's unanchored FP8 skip patterns accidentally make
# ``mlp.gate`` also match ``mlp.gate_proj``. The former is not a Linear.
quantization = config.quantization_config
quantization["modules_to_not_convert"] = [
    name
    for name in quantization["modules_to_not_convert"]
    if not name.endswith(".mlp.gate")
]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    config=config,
    dtype=torch.bfloat16,
    device_map="auto",
)
install(model, mode="two-tier")

inputs = tokenizer("Explain LoD Attention.", return_tensors="pt").to(model.device)
tokens = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

Select `three-tier-bf16` or `three-tier-int4` with the same `mode` argument.
Generation automatically creates the LoD-owned cache. For direct model calls,
`lod_attention.new_cache(model)` returns an empty cache explicitly.

## vLLM

Installing the package registers the `CUSTOM` attention backend. There are
only three public environment settings:

- `VLLM_LOD_MODE`: one of the three modes above (default `two-tier`).
- `VLLM_LOD_POOL_SIZE`: live or retained request rows per worker (default 8).
- `VLLM_LOD_MAX_CONTEXT`: optional per-row context cap.

Unknown `VLLM_LOD_*` and all old `LOD_DEV_*` tuning flags fail at startup.
Use a 16K scheduler budget so scheduler slicing cannot silently change the
state-update policy:

```bash
VLLM_PLUGINS=lod_attention \
VLLM_LOD_MODE=three-tier-int4 \
VLLM_LOD_POOL_SIZE=8 \
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --attention-backend CUSTOM \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --long-prefill-token-threshold 16384 \
  --gpu-memory-utilization 0.7 \
  --enable-prefix-caching
```

For Qwen, the 0.7 target leaves transient workspace headroom outside vLLM's
native-cache allocator; LoD's authoritative per-request pool is already
included in the model-side allocation. K2's larger model-side 131K pool needs
`--gpu-memory-utilization 0.8` merely to leave vLLM a nonempty native-cache
remainder. Raise either target only after measuring peak memory for the intended
model, mode, context limit, and concurrency.

On vLLM revisions that expose only the structured option, replace
`--attention-backend CUSTOM` with
`--attention-config '{"backend":"CUSTOM"}'`.

The LoD cache is authoritative: compressed remote leaves replace their native
chronological K/V rather than shadowing a full cache. Prefix-cache hits resume
retained LoD rows after exact token-prefix verification. Non-attention and
ineligible local/recurrent layers retain their native vLLM caches.

## Repository layout

- `lod_attention/`: model-independent HF adapter, cache, engines, and kernels.
- `integrations/vllm_lod/vllm_lod_plugin/`: vLLM backend and cache lifecycle.
- `integrations/vllm_lod/vllm_lod_plugin/models/`: K2 and Qwen DFlash2 shims.
- `integrations/vllm_lod/patches/`: the required AITER patch.
- `examples/`: minimal HF and vLLM launch examples.
- `benchmarks/`: public LongBench v2, ProLong, and RULER NIAH-S3 runners.
- `tests/`: release-policy and import checks.

## Benchmarks

Each benchmark has a standalone runner, archived results, and commands that use
only public tools:

- [LongBench v2](benchmarks/LONGBENCH_V2.md): end-to-end long-context quality
  and serving wall time.
- [ProLong](benchmarks/PROLONG.md): prompt CE/perplexity and matched prefill and
  1,025-token decode speed sweeps.
- [RULER NIAH-S3](benchmarks/NIAH_S3.md): long-context UUID retrieval.

The documents report finalized top-4 measurements from the release checkout.
The retained-leaf exact decode path is limited to contexts of at most 2,048
tokens, so every published 4K-and-longer result exercises routed LoD.

This implementation is inference-only and does not return dense attention
weights. Sliding-window attention, ALiBi, attention soft caps, DCP/PCP, and
native quantized attention K/V are intentionally rejected instead of silently
falling back to a different LoD calculation.

## License

Apache-2.0. Model compatibility files retain their upstream notices.

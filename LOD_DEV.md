# LOD Attention development snapshot

This branch collects the inference-time and training prototypes, optimized
kernels, evaluation scripts, and experiment outputs developed for two-level
LOD attention.

## Main implementations

- `model/pytorch_lod_attention.py`: model-independent PyTorch reference with
  both coarse-only and exact-leaf LOD attention
- `model/pytorch_lod_attention_fast.py`: inference-oriented PyTorch backend
  using SDPA, separately compiled FlexAttention, and packed FlashAttention
- `model/qwen35_two_level_attention.py`: inference-time Qwen3.5 attention graft
- `model/kernels/qwen35_lod_kernels.py`: state update and routing kernels
- `model/kernels/paged_leaf_attention.py`: paged, virtual-page, recursive-page,
  and quantized leaf attention kernels
- `model/kvm_two_level_mixer.py`: pure-PyTorch training prototype
- `model/gptalpha_two_level_mixer.py`: GPTAlpha2 inference approximation
- `model/kvm_split_full_attention_mixer.py`: full-remote baseline

The corresponding `scripts/` entry points cover ProLong loss, NIAH, RULER and
lm-eval evaluation, profiling, kernel verification, and full-remote comparison.

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

`model/hf_pytorch_lod_attention.py` supplies the thin model-specific layer
around the generic post-RoPE implementation. For Qwen3.5 it preserves the
loaded projection, normalization, gate, and output-projection weights and
changes only the attention calculation in each `full_attention` layer:

```python
from transformers import Qwen3_5ForCausalLM
from model.hf_pytorch_lod_attention import replace_qwen35_attention_with_lod
from model.pytorch_lod_attention import LODConfig

model = Qwen3_5ForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B", dtype="bfloat16"
).cuda().eval()
replaced_layers = replace_qwen35_attention_with_lod(
    model,
    config=LODConfig(
        chunk_size=256,
        local_window=512,
        state_growth_factor=16.0,
        state_min_size=256,
        max_routes=8,
    ),
    open_count=8,
)
```

The adapter uses Qwen's hybrid cache only to report sequence length; exact KV,
LOD state, and the local window live in the LOD cache instead of being copied
into the ordinary full-attention cache. It currently assumes unpadded causal
input and single-beam generation. The same pattern can support another HF
model by copying its projection/RoPE/output shell and calling
`FastTwoLevelLODAttention` at the post-RoPE boundary.

Run the Qwen adapter checks and NIAH smoke evaluation with:

```bash
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

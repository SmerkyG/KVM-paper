# LOD Attention development snapshot

This branch collects the inference-time and training prototypes, optimized
kernels, evaluation scripts, and experiment outputs developed for two-level
LOD attention.

## Main implementations

- `model/pytorch_lod_attention.py`: model-independent PyTorch reference with
  both coarse-only and exact-leaf LOD attention
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

## Experiment outputs

The LOD-specific `artifacts/` subdirectories contain JSON, JSONL, and text log
outputs for the full-remote, GPTAlpha2, Qwen3.5, dynamic-opening, prefill,
recursive-page, virtual-page, and INT4 experiments. These are result records,
not model weights.

Model checkpoints and local model caches are intentionally excluded. In
particular, the source worktree's `hf-models/` directory is not part of this
branch.

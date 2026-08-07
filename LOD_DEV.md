# LOD Attention development snapshot

This branch collects the inference-time and training prototypes, optimized
kernels, evaluation scripts, and experiment outputs developed for two-level
LOD attention.

## Main implementations

- `model/qwen35_two_level_attention.py`: inference-time Qwen3.5 attention graft
- `model/kernels/qwen35_lod_kernels.py`: state update and routing kernels
- `model/kernels/paged_leaf_attention.py`: paged, virtual-page, recursive-page,
  and quantized leaf attention kernels
- `model/kvm_two_level_mixer.py`: pure-PyTorch training prototype
- `model/gptalpha_two_level_mixer.py`: GPTAlpha2 inference approximation
- `model/kvm_split_full_attention_mixer.py`: full-remote baseline

The corresponding `scripts/` entry points cover ProLong loss, NIAH, RULER and
lm-eval evaluation, profiling, kernel verification, and full-remote comparison.

## Experiment outputs

The LOD-specific `artifacts/` subdirectories contain JSON, JSONL, and text log
outputs for the full-remote, GPTAlpha2, Qwen3.5, dynamic-opening, prefill,
recursive-page, virtual-page, and INT4 experiments. These are result records,
not model weights.

Model checkpoints and local model caches are intentionally excluded. In
particular, the source worktree's `hf-models/` directory is not part of this
branch.

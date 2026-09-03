#!/usr/bin/env bash
set -euo pipefail

length="$1"
output_dir="artifacts/dense_page_prefill/tiny_long_context"
mkdir -p "$output_dir"

for layout in expert expert_tiny; do
  PYTHONPATH=. .venv/bin/python scripts/profile_qwen35_lod_exact_phases.py \
    --sequence-length "$length" \
    --batch-size 8 \
    --state-growth-factor 16 \
    --prefill-two-level-topk 8 \
    --block-m 16 \
    --block-n 32 \
    --num-warps 2 \
    --layout "$layout" \
    --tiny-expert-max 8 \
    --tiny-max-context 65536 \
    --tiny-block-m 8 \
    --tiny-num-warps 1 \
    --reduce-num-warps 1 \
    --virtual-page-storage \
    --output "$output_dir/phase_${layout}_${length}_b8.json"

  PYTHONPATH=. .venv/bin/python scripts/profile_qwen35_prefill_total.py \
    --mode two_level \
    --sequence-length "$length" \
    --batch-size 8 \
    --two-level-topk 8 \
    --prefill-two-level-topk 8 \
    --state-growth-factor 16 \
    --leaf-attention-backend paged \
    --leaf-layout "$layout" \
    --virtual-page-storage \
    --leaf-block-m 16 \
    --leaf-block-n 32 \
    --leaf-num-warps 2 \
    --tiny-expert-max 8 \
    --tiny-max-context 65536 \
    --tiny-block-m 8 \
    --tiny-num-warps 1 \
    --reduce-num-warps 1 \
    --repeats 2 \
    --output "$output_dir/e2e_${layout}_${length}_b8.json"
done

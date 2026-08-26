#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?checkpoint required}
output_prefix=${2:?output prefix required}
apply_chat_template=${3:-1}
tensor_parallel_size=${4:-1}
repo=${5:-/home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code}

export VLLM_LOD_PANEL_SPEED_ONLY=1
export VLLM_LOD_PANEL_SPEED_DECODE_TOKENS=${VLLM_LOD_PANEL_SPEED_DECODE_TOKENS:-64}
export VLLM_WEIGHT_CACHE_LOAD_FORMAT=${VLLM_WEIGHT_CACHE_LOAD_FORMAT:-auto}
# Hold prefill constant while comparing only the decode route/coarse schedule.
export VLLM_LOD_PREFILL_HIERARCHICAL_ROUTE=1
export VLLM_LOD_PREFILL_OVERLAP_COARSE_LEAF=0
export VLLM_LOD_PREFILL_OVERLAP_LOCAL_LOD=0
decode_tokens=${VLLM_LOD_PANEL_SPEED_DECODE_TOKENS}
speed_repeats=${VLLM_LOD_PANEL_SPEED_REPEATS:-1}

export VLLM_LOD_DECODE_GEOMETRY_TUNING=1
for tuned in ${VLLM_LOD_DECODE_ROUTE_ORDER:-0 1}; do
  # Toggle only route/coarse segmentation. Leaf, local, dot-product, and all
  # other decode geometry choices remain identical in both arms.
  export VLLM_LOD_DECODE_HIERARCHICAL_ROUTE=$tuned
  suffix=grouped
  if [[ $tuned == 1 ]]; then
    suffix=hierarchical
  fi
  "$repo/scripts/run_vllm_lod_niah_speed_panel.sh" \
    "$checkpoint" \
    lod \
    "${output_prefix}_${suffix}_64k_b8_r${speed_repeats}_d${decode_tokens}.json" \
    65536 \
    8 \
    "$speed_repeats" \
    "$apply_chat_template" \
    "$repo" \
    "$tensor_parallel_size"
done

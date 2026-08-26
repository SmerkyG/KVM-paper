#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?checkpoint required}
output_prefix=${2:?output prefix required}
apply_chat_template=${3:-1}
tensor_parallel_size=${4:-1}
repo=${5:-/home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code}

export VLLM_LOD_PANEL_SPEED_ONLY=1
# The second token makes the panel record marginal decode without spending
# time on a full quality decode during a prefill-only A/B measurement.
export VLLM_LOD_PANEL_SPEED_DECODE_TOKENS=2
export VLLM_WEIGHT_CACHE_LOAD_FORMAT=${VLLM_WEIGHT_CACHE_LOAD_FORMAT:-ipc_cache}
# Isolate route selection from the independently tuned Muse branch overlap.
export VLLM_LOD_PREFILL_OVERLAP_COARSE_LEAF=0
export VLLM_LOD_PREFILL_OVERLAP_LOCAL_LOD=0

for hierarchical in 0 1; do
  export VLLM_LOD_PREFILL_HIERARCHICAL_ROUTE=$hierarchical
  suffix=grouped
  if [[ $hierarchical == 1 ]]; then
    suffix=hierarchical
  fi
  "$repo/scripts/run_vllm_lod_niah_speed_panel.sh" \
    "$checkpoint" \
    lod \
    "${output_prefix}_${suffix}_64k_b8_r3.json" \
    65536 \
    8 \
    3 \
    "$apply_chat_template" \
    "$repo" \
    "$tensor_parallel_size"
done

#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?checkpoint required}
output_prefix=${2:?output prefix required}
apply_chat_template=${3:-1}
tensor_parallel_size=${4:-1}
repo=${5:-/home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code}

export VLLM_LOD_PANEL_SPEED_ONLY=1
# The panel derives marginal decode time from tokens after the first sampled
# token, so two is the minimum that still writes its prefill measurements.
export VLLM_LOD_PANEL_SPEED_DECODE_TOKENS=2
export VLLM_WEIGHT_CACHE_LOAD_FORMAT=${VLLM_WEIGHT_CACHE_LOAD_FORMAT:-ipc_cache}

for overlap in 0 1; do
  export VLLM_LOD_PREFILL_OVERLAP_COARSE_LEAF=$overlap
  export VLLM_LOD_PREFILL_OVERLAP_LOCAL_LOD=$overlap
  suffix=off
  if [[ $overlap == 1 ]]; then
    suffix=on
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

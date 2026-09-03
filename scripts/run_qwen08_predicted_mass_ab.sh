#!/usr/bin/env bash
set -euo pipefail

batch=${1:?batch size required}
output_dir=${2:?output directory required}
repeats=${3:-7}
repo=${4:-/home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code}

mkdir -p "$output_dir"
cd "$repo"

export VLLM_WEIGHT_CACHE_LOAD_FORMAT=auto
export VLLM_LOD_PANEL_BATCH_SIZE="$batch"
export VLLM_LOD_PANEL_MAX_CONTEXT=65616
export VLLM_LOD_PANEL_SPEED_ONLY=1
export VLLM_LOD_PANEL_SPEED_PROMPT_RESERVE=256
export VLLM_LOD_PANEL_SPEED_DECODE_TOKENS=256
export VLLM_LOD_PANEL_MAX_BATCHED_TOKENS=16384
export VLLM_LOD_PANEL_LONG_PREFILL_THRESHOLD=4096
export VLLM_LOD_PANEL_GPU_MEMORY_UTILIZATION=0.8
export VLLM_LOD_PANEL_SPEED_USE_WARM_PREFIX_CACHE=1
export VLLM_LOD_PANEL_LEVELS=2
export VLLM_LOD_PANEL_ROUTING_GEOMETRY=auto
export VLLM_LOD_OPEN_COUNT=8
export VLLM_LOD_PREFILL_OPEN_COUNT=3
export VLLM_LOD_DECODE_MAX_OPEN_LEAVES=1024
export VLLM_LOD_DECODE_ROUTE_COHORT=0
export VLLM_LOD_DECODE_GQA_STATIC_LEAF_AITER=0
export VLLM_LOD_DECODE_GQA_UNION=1
export VLLM_LOD_DECODE_GQA_UNION_HIP=1
export VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER=1
export VLLM_LOD_DECODE_GQA_FIXED_MASK_BLOCK_N=64
export VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS=128
export VLLM_LOD_DECODE_SPLIT_KV=8
export VLLM_LOD_DECODE_GEOMETRY_TUNING=1
export VLLM_LOD_LEAF_BLOCK_N=32

run_case() {
    local label=$1
    local predicted_mass=$2
    local mass_fraction=$3
    local adaptive_segments=$4
    local reduce_block_d=$5
    VLLM_LOD_DECODE_GQA_PREDICTED_MASS="$predicted_mass" \
    VLLM_LOD_DECODE_GQA_MASS_FRACTION="$mass_fraction" \
    VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS="$adaptive_segments" \
    VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D="$reduce_block_d" \
        bash scripts/run_vllm_lod_niah_speed_panel.sh \
        Qwen/Qwen3.5-0.8B lod \
        "$output_dir/qwen08_two_64k_b${batch}_${label}_r${repeats}.json" \
        65536 "$batch" "$repeats" 1 "$repo" 1
}

# Both selectors use the low-row occupancy policy. The predicted-mass reducer
# confines its remote-LSE and next-queue side effects to D partition zero.
run_case top8 0 0 1 64
run_case predmass16 1 0.0625 1 64
run_case predmass32 1 0.03125 1 64
run_case predmass8 1 0.125 1 64

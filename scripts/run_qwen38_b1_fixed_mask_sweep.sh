#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?output directory required}
repeats=${2:-3}
repo=${3:-/home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code}
case_filter=${4:-all}

mkdir -p "$output_dir"
cd "$repo"

export VLLM_WEIGHT_CACHE_LOAD_FORMAT=ipc_cache
export VLLM_LOD_PANEL_BATCH_SIZE=1
export VLLM_LOD_PANEL_MAX_CONTEXT=65744
export VLLM_LOD_PANEL_SPEED_ONLY=1
export VLLM_LOD_PANEL_SPEED_PROMPT_RESERVE=64
export VLLM_LOD_PANEL_SPEED_DECODE_TOKENS=256
export VLLM_LOD_PANEL_MAX_BATCHED_TOKENS=4096
export VLLM_LOD_PANEL_LONG_PREFILL_THRESHOLD=4096
export VLLM_LOD_PANEL_GPU_MEMORY_UTILIZATION=0.9
export VLLM_LOD_PANEL_SPEED_USE_WARM_PREFIX_CACHE=1
export VLLM_LOD_PANEL_LEVELS=2
export VLLM_LOD_PANEL_ROUTING_GEOMETRY=auto
export VLLM_LOD_OPEN_COUNT=8
export VLLM_LOD_PREFILL_OPEN_COUNT=3
export VLLM_LOD_PREFILL_HIERARCHICAL_ROUTE=1
export VLLM_LOD_DECODE_MAX_OPEN_LEAVES=1024
export VLLM_LOD_DECODE_ROUTE_COHORT=0
export VLLM_LOD_DECODE_GQA_PREDICTED_MASS=0
export VLLM_LOD_DECODE_GQA_STATIC_LEAF_AITER=0
export VLLM_LOD_DECODE_GQA_UNION=1
export VLLM_LOD_DECODE_GQA_UNION_HIP=1
export VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER=1
export VLLM_LOD_DECODE_GQA_FIXED_MASK_BLOCK_N=64
export VLLM_LOD_DECODE_SPLIT_KV=8
export VLLM_LOD_DECODE_GEOMETRY_TUNING=1
export VLLM_LOD_DECODE_CENTROID_MAJOR_HIP=0
export VLLM_LOD_LEAF_BLOCK_N=16

run_case() {
    local label=$1
    if [[ "$case_filter" != all && ",$case_filter," != *",$label,"* ]]; then
        return
    fi
    local adaptive_segments=$2
    local reduce_block_d=$3
    local direct_routes=$4
    local segments=$5
    VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS="$adaptive_segments" \
    VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D="$reduce_block_d" \
    VLLM_LOD_DECODE_GQA_FIXED_MASK_DIRECT_ROUTES="$direct_routes" \
    VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS="$segments" \
        bash scripts/run_vllm_lod_niah_speed_panel.sh \
        Qwen/Qwen3.8-27B-FP8 lod \
        "$output_dir/qwen38_b1_64k_${label}_r${repeats}.json" \
        65536 1 "$repeats" 1 "$repo" 1
}

run_case baseline 0 0 0 128
run_case segments256 1 0 0 256
run_case splitd64 0 64 0 128
run_case direct 0 0 1 128
run_case segments256_splitd64 1 64 0 256
run_case optimized 1 64 1 256

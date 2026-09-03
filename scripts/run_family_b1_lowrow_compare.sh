#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?checkpoint required}
output_prefix=${2:?output prefix required}
full_backend=${3:?full backend required}
apply_chat_template=${4:-1}
repeats=${5:-5}
repo=${6:-/home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code}
arm=${7:-all}

cd "$repo"
mkdir -p "$(dirname "$output_prefix")"

export VLLM_WEIGHT_CACHE_LOAD_FORMAT=${VLLM_WEIGHT_CACHE_LOAD_FORMAT:-ipc_cache}
export VLLM_LOD_PANEL_BATCH_SIZE=${VLLM_LOD_PANEL_BATCH_SIZE:-1}
panel_samples=${VLLM_LOD_PANEL_SAMPLES:-$VLLM_LOD_PANEL_BATCH_SIZE}
export VLLM_LOD_PANEL_MAX_CONTEXT=65792
export VLLM_LOD_PANEL_SPEED_ONLY=1
export VLLM_LOD_PANEL_SPEED_PROMPT_RESERVE=256
export VLLM_LOD_PANEL_SPEED_DECODE_TOKENS=256
export VLLM_LOD_PANEL_MAX_BATCHED_TOKENS=16384
export VLLM_LOD_PANEL_LONG_PREFILL_THRESHOLD=16384
export VLLM_LOD_PANEL_GPU_MEMORY_UTILIZATION=${VLLM_LOD_PANEL_GPU_MEMORY_UTILIZATION:-0.9}
export VLLM_LOD_PANEL_SPEED_USE_WARM_PREFIX_CACHE=1
export VLLM_LOD_PANEL_FULL_BACKEND="$full_backend"
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

run_panel() {
    local mode=$1
    local output=$2
    bash scripts/run_vllm_lod_niah_speed_panel.sh \
        "$checkpoint" "$mode" "$output" 65536 "$panel_samples" "$repeats" \
        "$apply_chat_template" "$repo" 1
}

if [[ "$arm" == all || "$arm" == full ]]; then
    run_panel full "${output_prefix}_full_r${repeats}.json"
fi

# Pre-change low-row control: build the ordinary centroid union, scan with the
# established 128 segments, and reduce the complete D-vector in one program.
if [[ "$arm" == all || "$arm" == control || "$arm" == direct_ab ]]; then
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS=128
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS=0
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D=0
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_DIRECT_ROUTES=0
    run_panel lod "${output_prefix}_lod_control_r${repeats}.json"
fi

# Current low-row path: direct route activation, 256 scan segments, and a
# split-D64 reducer where the head geometry supports it.
if [[ "$arm" == all || "$arm" == optimized ]]; then
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS=256
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS=1
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D=64
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_DIRECT_ROUTES=1
    run_panel lod "${output_prefix}_lod_optimized_r${repeats}.json"
fi

if [[ "$arm" == direct128 || "$arm" == direct_ab ]]; then
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS=128
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS=0
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D=0
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_DIRECT_ROUTES=1
    run_panel lod "${output_prefix}_lod_direct128_r${repeats}.json"
fi

if [[ "$arm" == direct256 ]]; then
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS=256
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS=0
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D=64
    export VLLM_LOD_DECODE_GQA_FIXED_MASK_DIRECT_ROUTES=1
    run_panel lod "${output_prefix}_lod_direct256_r${repeats}.json"
fi

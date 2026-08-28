#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?output directory required}
repeats=${2:-5}
repo=${3:-/home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code}
tensor_parallel_size=${4:-1}

mkdir -p "$output_dir"
cd "$repo"

export VLLM_WEIGHT_CACHE_LOAD_FORMAT=ipc_cache
export VLLM_LOD_PANEL_BATCH_SIZE=1
export VLLM_LOD_PANEL_MAX_CONTEXT=262400
export VLLM_LOD_PANEL_SPEED_ONLY=1
export VLLM_LOD_PANEL_SPEED_PROMPT_RESERVE=256
export VLLM_LOD_PANEL_SPEED_DECODE_TOKENS=256
export VLLM_LOD_PANEL_MAX_BATCHED_TOKENS=16384
export VLLM_LOD_PANEL_LONG_PREFILL_THRESHOLD=16384
export VLLM_LOD_PANEL_GPU_MEMORY_UTILIZATION=0.9
export VLLM_LOD_PANEL_SPEED_USE_WARM_PREFIX_CACHE=1
export VLLM_LOD_PANEL_FULL_BACKEND=ROCM_AITER_UNIFIED_ATTN
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
export VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS=256
export VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS=1
export VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D=64
export VLLM_LOD_DECODE_GQA_FIXED_MASK_DIRECT_ROUTES=1
export VLLM_LOD_DECODE_SPLIT_KV=8
export VLLM_LOD_DECODE_GEOMETRY_TUNING=1
export VLLM_LOD_DECODE_CENTROID_MAJOR_HIP=0
export VLLM_LOD_LEAF_BLOCK_N=16

lengths=8192,16384,32768,65536,131072,262144
panel_prefix=qwen38_b1
if [[ "$tensor_parallel_size" != 1 ]]; then
    panel_prefix="qwen38_tp${tensor_parallel_size}_b1"
fi

bash scripts/run_vllm_lod_niah_speed_panel.sh \
    Qwen/Qwen3.8-27B-FP8 full \
    "$output_dir/${panel_prefix}_full_r${repeats}.json" \
    "$lengths" 1 "$repeats" 1 "$repo" "$tensor_parallel_size"

bash scripts/run_vllm_lod_niah_speed_panel.sh \
    Qwen/Qwen3.8-27B-FP8 lod \
    "$output_dir/${panel_prefix}_lod_optimized_r${repeats}.json" \
    "$lengths" 1 "$repeats" 1 "$repo" "$tensor_parallel_size"

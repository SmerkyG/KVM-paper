#!/usr/bin/env bash
set -euo pipefail

model=${1:?usage: vllm.sh MODEL [two-tier|three-tier-bf16|three-tier-int4]}
mode=${2:-two-tier}
gpu_memory_utilization=0.7
case ${model,,} in
  *k2*horizon*) gpu_memory_utilization=0.8 ;;
esac

VLLM_PLUGINS=lod_attention \
VLLM_LOD_MODE="$mode" \
VLLM_LOD_POOL_SIZE=8 \
vllm serve "$model" \
  --attention-backend CUSTOM \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --long-prefill-token-threshold 16384 \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --enable-prefix-caching \
  --trust-remote-code

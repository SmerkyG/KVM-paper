#!/usr/bin/env bash
set -euo pipefail

precision=${1:?precision (bf16 or int8) required}
shard=${2:?shard index required}
output=${3:?output path required}
repo=${4:-/home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code}
max_num_batched_tokens=${5:-131072}
long_prefill_token_threshold=${6:-16384}
profile=${7:-current}
port=${CLUSTER_RUN_PORT_0:-${MASTER_PORT:?}}
server_log=${output%.jsonl}.server.log
eval_log=${output%.jsonl}.eval.log
warm_output=${output%.jsonl}.warmup-long.jsonl
evaluator=/home/dan/subusers/agent/kvm-paper-dg/branches/lod-logical-prefill-fix/code/scripts/eval_longbench_v2_openai.py
vllm_root=/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1

case "$precision" in
  bf16)
    kv_bits=0
    ;;
  int8)
    kv_bits=8
    ;;
  *)
    echo "precision must be bf16 or int8, got: $precision" >&2
    exit 2
    ;;
esac

case "$profile" in
  current)
    aug19_compat=0
    ;;
  aug19)
    aug19_compat=1
    ;;
  *)
    echo "profile must be current or aug19, got: $profile" >&2
    exit 2
    ;;
esac

cd "$repo"
mkdir -p "$(dirname "$output")"
rm -f "$output" "$warm_output"

env \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  VLLM_PLUGINS=lod_attention \
  PYTHONPATH="$repo/integrations/vllm_lod:$repo" \
  VLLM_LOD_CACHE_OWNERSHIP=lod \
  VLLM_LOD_AUG19_COMPAT="$aug19_compat" \
  VLLM_LOD_POOL_SIZE=8 \
  VLLM_LOD_LEVELS=2 \
  VLLM_LOD_KV_BITS="$kv_bits" \
  VLLM_LOD_MAX_CONTEXT=262144 \
  VLLM_LOD_STATE_GROWTH_FACTOR=16 \
  VLLM_LOD_DENSE_LEAF_STORAGE=0 \
  VLLM_LOD_PREFILL_MODE=direct \
  VLLM_LOD_ROUTING_GEOMETRY=raw \
  VLLM_LOD_PREFILL_CHUNK_SIZE=4096 \
  VLLM_LOD_PREFILL_LOCAL_WINDOW=4864 \
  VLLM_LOD_PREFILL_STATE_UPDATE_SIZE=4096 \
  VLLM_LOD_PREFILL_INT8_COARSE_NUM_WARPS=8 \
  "$vllm_root/bin/vllm" serve \
    Qwen/Qwen3.5-35B-A3B \
    --attention-config '{"backend":"CUSTOM"}' \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --long-prefill-token-threshold "$long_prefill_token_threshold" \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --host 127.0.0.1 \
    --port "$port" >"$server_log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 900); do
  if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -300 "$server_log"
    exit 1
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
  tail -300 "$server_log"
  exit 1
fi

# Compile every long-context two-tier specialization outside the measured run.
"$vllm_root/bin/python" "$evaluator" \
  --base-url "http://127.0.0.1:${port}/v1" \
  --checkpoint Qwen/Qwen3.5-35B-A3B \
  --output "$warm_output" \
  --max-input-tokens 262016 \
  --max-output-tokens 32 \
  --prompt-token-min 250000 \
  --limit 8 \
  --workers 8 \
  --sort-by-input-length \
  --disable-thinking \
  --guided-answer-choice >/dev/null

# The evaluator starts its wall timer after this additional short warm-up batch.
"$vllm_root/bin/python" "$evaluator" \
  --base-url "http://127.0.0.1:${port}/v1" \
  --checkpoint Qwen/Qwen3.5-35B-A3B \
  --output "$output" \
  --max-input-tokens 262016 \
  --max-output-tokens 32 \
  --workers 8 \
  --warmup-batches 1 \
  --sort-by-input-length \
  --shard-index "$shard" \
  --num-shards 8 \
  --disable-thinking \
  --guided-answer-choice | tee "$eval_log"

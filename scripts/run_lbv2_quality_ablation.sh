#!/usr/bin/env bash
set -euo pipefail

variant=${1:?variant name required}
output=${2:?output path required}
repo=${3:-/home/dan/subusers/agent/kvm-paper-dg/.runtime/lod-dev-nospecialize-20260815}
limit=${4:-8}
workers=${5:-1}
prompt_token_min=${6:-120000}
port=${CLUSTER_RUN_PORT_0:-${MASTER_PORT:?}}
log=${output%.jsonl}.server.log

cd "$repo"
mkdir -p "$(dirname "$output")"
env \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  VLLM_PLUGINS=lod_attention \
  PYTHONPATH="$repo/integrations/vllm_lod:$repo" \
  VLLM_LOD_CACHE_OWNERSHIP=lod \
  VLLM_LOD_POOL_SIZE=8 \
  VLLM_LOD_KV_BITS=4 \
  VLLM_LOD_MAX_CONTEXT=131200 \
  /home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/vllm serve \
    Qwen/Qwen3.5-35B-A3B \
    --attention-config '{"backend":"CUSTOM"}' \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --max-model-len 131200 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --long-prefill-token-threshold 4096 \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --host 127.0.0.1 \
    --port "$port" >"$log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 900); do
  if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -300 "$log"
    exit 1
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
  tail -300 "$log"
  exit 1
fi

/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python \
  /home/dan/subusers/agent/kvm-paper-dg/branches/lod-logical-prefill-fix/code/scripts/eval_longbench_v2_openai.py \
  --base-url "http://127.0.0.1:${port}/v1" \
  --checkpoint Qwen/Qwen3.5-35B-A3B \
  --output "$output" \
  --max-input-tokens 131072 \
  --max-output-tokens 32 \
  --workers "$workers" \
  --sort-by-input-length \
  --selection-records-dir \
    /home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code/artifacts/longbench_v2/qwen35_35b_a3b_vllm_current_full_guided \
  --prompt-token-min "$prompt_token_min" \
  --limit "$limit" \
  --disable-thinking \
  --guided-answer-choice

#!/usr/bin/env bash
set -euo pipefail

mode=${1:?mode (full or lod) required}
shard=${2:?shard index required}
output=${3:?output path required}
repo=${4:-/home/dan/subusers/agent/kvm-paper-dg/.runtime/lod-dev-nospecialize-20260815}
port=${CLUSTER_RUN_PORT_0:-${MASTER_PORT:?}}
server_log=${output%.jsonl}.server.log
eval_log=${output%.jsonl}.eval.log
warm_output=${output%.jsonl}.warmup-long.jsonl
evaluator=/home/dan/subusers/agent/kvm-paper-dg/branches/lod-logical-prefill-fix/code/scripts/eval_longbench_v2_openai.py
vllm_root=/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1

case "$mode" in
  lod)
    plugin=lod_attention
    backend=CUSTOM
    ;;
  full)
    plugin=
    backend=ROCM_AITER_UNIFIED_ATTN
    ;;
  *)
    echo "mode must be full or lod, got: $mode" >&2
    exit 2
    ;;
esac

cd "$repo"
mkdir -p "$(dirname "$output")"
rm -f "$output" "$warm_output"

env \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  VLLM_PLUGINS="$plugin" \
  PYTHONPATH="$repo/integrations/vllm_lod:$repo" \
  VLLM_LOD_POOL_SIZE=8 \
  VLLM_LOD_KV_BITS=4 \
  VLLM_LOD_MAX_CONTEXT=262144 \
  "$vllm_root/bin/vllm" serve \
    Qwen/Qwen3.5-35B-A3B \
    --attention-config "{\"backend\":\"$backend\"}" \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --long-prefill-token-threshold 4096 \
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

# Compile/capture the long-context path without including it in timed results.
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

# The evaluator starts its wall timer after this short-context warm-up batch.
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

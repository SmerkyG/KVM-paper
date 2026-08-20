#!/usr/bin/env bash
set -euo pipefail

kv_bits=${1:?LOD KV bits (0 for BF16, 4 for INT4, or 8 for INT8) required}
output=${2:?output JSONL path required}
reference_output=${3:-}
repo=${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
selection=${5:-$repo/artifacts/longbench_v2/quant_margin_sentinel8_qwen35_0p8b_20260817/selection.jsonl}
port=${CLUSTER_RUN_PORT_0:-${MASTER_PORT:?}}
server_log=${output%.jsonl}.server.log
eval_log=${output%.jsonl}.eval.log
vllm_root=/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1
checkpoint=${VLLM_MODEL:-Qwen/Qwen3.5-0.8B}
max_context=${LOD_MARGIN_MAX_CONTEXT:-262144}
max_input_tokens=${LOD_MARGIN_MAX_INPUT_TOKENS:-$((max_context - 128))}
gpu_memory_utilization=${LOD_MARGIN_GPU_MEMORY_UTILIZATION:-0.8}

case "$kv_bits" in
  0|4|8) ;;
  *)
    echo "kv_bits must be 0 (BF16), 4 (INT4), or 8 (INT8), got: $kv_bits" >&2
    exit 2
    ;;
esac

cd "$repo"
mkdir -p "$(dirname "$output")"

env \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  VLLM_PLUGINS=lod_attention \
  PYTHONPATH="$repo/integrations/vllm_lod:$repo" \
  VLLM_LOD_CACHE_OWNERSHIP=lod \
  VLLM_LOD_POOL_SIZE=8 \
  VLLM_LOD_KV_BITS="$kv_bits" \
  VLLM_LOD_KEY_BITS="${LOD_MARGIN_KEY_BITS:-$kv_bits}" \
  VLLM_LOD_VALUE_BITS="${LOD_MARGIN_VALUE_BITS:-$kv_bits}" \
  VLLM_LOD_MAX_CONTEXT="$max_context" \
  "$vllm_root/bin/vllm" serve \
    "$checkpoint" \
    --attention-config '{"backend":"CUSTOM"}' \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --max-model-len "$max_context" \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --long-prefill-token-threshold 4096 \
    --gpu-memory-utilization "$gpu_memory_utilization" \
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

reference_args=()
if [[ -n "$reference_output" ]]; then
  reference_args=(--reference-output "$reference_output")
fi

eval_args=(
  --base-url "http://127.0.0.1:${port}/v1"
  --checkpoint "$checkpoint"
  --selection "$selection"
  --max-input-tokens "$max_input_tokens"
  --workers 8
  --disable-thinking
)

"$vllm_root/bin/python" scripts/eval_longbench_choice_margins.py \
  "${eval_args[@]}" \
  --output "$output" \
  "${reference_args[@]}" | tee "$eval_log"

if [[ -n "${LOD_MARGIN_REPEAT_OUTPUT:-}" ]]; then
  repeat_log=${LOD_MARGIN_REPEAT_OUTPUT%.jsonl}.eval.log
  "$vllm_root/bin/python" scripts/eval_longbench_choice_margins.py \
    "${eval_args[@]}" \
    --output "$LOD_MARGIN_REPEAT_OUTPUT" | tee "$repeat_log"
fi

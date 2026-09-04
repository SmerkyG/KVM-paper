#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
checkpoint=${1:?checkpoint required}
mode=${2:?mode (full or lod) required}
output=${3:?output path required}
lengths=${4:-8192,16384,32768,65536,131072}
samples=${5:-64}
speed_repeats=${6:-3}
apply_chat_template=${7:-1}
repo=${8:-$(cd -- "$script_dir/.." && pwd)}
tensor_parallel_size=${9:-1}
vllm_root=/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1
batch_size=${VLLM_LOD_PANEL_BATCH_SIZE:-8}
speed_prompt_reserve=${VLLM_LOD_PANEL_SPEED_PROMPT_RESERVE:-}
if [[ -z "$speed_prompt_reserve" ]]; then
  # OLMo advertises exactly 64K positions, so its 64K panel must leave room
  # for the 64 generated timing tokens. Other panel checkpoints either have a
  # larger advertised limit or already use an explicit override.
  if [[ "$checkpoint" == google/gemma-4-* ]]; then
    # Gemma DFlash may finish the last accepted speculative block slightly
    # beyond the requested output count. Keep the established panel's 256
    # prompt-token reserve so the verifier never becomes ragged at the model
    # boundary (for example, a final 6-position block instead of 16).
    speed_prompt_reserve=256
  elif [[ "$checkpoint" == allenai/Olmo-* ]]; then
    speed_prompt_reserve=64
  else
    speed_prompt_reserve=0
  fi
fi

if [[ "$mode" != full && "$mode" != lod ]]; then
  echo "mode must be full or lod, got: $mode" >&2
  exit 2
fi
if [[ "$apply_chat_template" != 0 && "$apply_chat_template" != 1 ]]; then
  echo "apply_chat_template must be zero or one" >&2
  exit 2
fi

cd "$repo"
mkdir -p "$(dirname "$output")"

args=(
  "$repo/scripts/eval_vllm_lod_niah_speed_panel.py"
  --checkpoint "$checkpoint"
  --mode "$mode"
  --lengths "$lengths"
  --samples "$samples"
  --sample-offset "${VLLM_LOD_PANEL_SAMPLE_OFFSET:-0}"
  --batch-size "$batch_size"
  --max-new-tokens 64
  --speed-decode-tokens "${VLLM_LOD_PANEL_SPEED_DECODE_TOKENS:-64}"
  --speed-prompt-reserve "$speed_prompt_reserve"
  --speed-repeats "$speed_repeats"
  --max-num-batched-tokens "${VLLM_LOD_PANEL_MAX_BATCHED_TOKENS:-16384}"
  --long-prefill-token-threshold "${VLLM_LOD_PANEL_LONG_PREFILL_THRESHOLD:-16384}"
  --gpu-memory-utilization "${VLLM_LOD_PANEL_GPU_MEMORY_UTILIZATION:-0.8}"
  --tensor-parallel-size "$tensor_parallel_size"
  --output "$output"
)
if [[ "${VLLM_LOD_PANEL_QUALITY_ONLY:-0}" == 1 ]]; then
  args+=(--quality-only)
fi
if [[ "${VLLM_LOD_PANEL_SPEED_ONLY:-0}" == 1 ]]; then
  args+=(--speed-only)
fi
if [[ "${VLLM_LOD_PANEL_PROFILE_PHASES:-0}" == 1 ]]; then
  args+=(--profile-lod-phases)
fi
if [[ "${VLLM_LOD_PANEL_PROFILE_LOD_TOTAL:-0}" == 1 ]]; then
  args+=(--profile-lod-total)
fi
if [[ "${VLLM_LOD_PANEL_PROFILE_FULL_ATTENTION:-0}" == 1 ]]; then
  args+=(--profile-full-attention)
fi
if [[ -n "${VLLM_LOD_PANEL_TORCH_PROFILE_DIR:-}" ]]; then
  args+=(
    --torch-profile-dir "$VLLM_LOD_PANEL_TORCH_PROFILE_DIR"
    --torch-profile-delay-iterations "${VLLM_LOD_PANEL_TORCH_PROFILE_DELAY:-0}"
    --torch-profile-max-iterations "${VLLM_LOD_PANEL_TORCH_PROFILE_MAX:-0}"
  )
fi
if [[ "${VLLM_LOD_PANEL_ENFORCE_EAGER:-0}" == 1 ]]; then
  args+=(--enforce-eager)
fi
if [[ "${VLLM_LOD_PANEL_PREFIX_CACHING:-0}" == 1 ]]; then
  args+=(--enable-prefix-caching)
fi
if [[ "${VLLM_LOD_PANEL_SPEED_USE_WARM_PREFIX_CACHE:-0}" == 1 ]]; then
  args+=(--enable-prefix-caching --speed-use-warm-prefix-cache)
fi
if [[ "${VLLM_LOD_PANEL_DISABLE_CUSTOM_ALL_REDUCE:-0}" == 1 ]]; then
  args+=(--disable-custom-all-reduce)
fi
if [[ -n "${VLLM_LOD_PANEL_SPECULATIVE_MODEL:-}" ]]; then
  args+=(
    --speculative-model "$VLLM_LOD_PANEL_SPECULATIVE_MODEL"
    --speculative-method "${VLLM_LOD_PANEL_SPECULATIVE_METHOD:-dflash}"
    --num-speculative-tokens "${VLLM_LOD_PANEL_SPECULATIVE_TOKENS:-7}"
    --speculative-attention-backend \
      "${VLLM_LOD_PANEL_SPECULATIVE_ATTENTION_BACKEND:-TRITON_ATTN}"
  )
fi
if [[ -n "${VLLM_LOD_PANEL_DECODE_ROUTE_GROUP_SIZE:-}" ]]; then
  # Speculative target verification is captured during LLM construction.
  # Configure graph-shaping route geometry before capture as well as through
  # the post-construction diagnostic hook below.
  common_route_group_size="$VLLM_LOD_PANEL_DECODE_ROUTE_GROUP_SIZE"
  args+=(
    --lod-decode-route-group-size
    "$common_route_group_size"
  )
fi
if [[ -n "${VLLM_LOD_PANEL_DECODE_ROUTE_NUM_WARPS:-}" ]]; then
  common_route_num_warps="$VLLM_LOD_PANEL_DECODE_ROUTE_NUM_WARPS"
  args+=(
    --lod-decode-route-num-warps
    "$common_route_num_warps"
  )
fi
if [[ -n "${VLLM_LOD_PANEL_DECODE_ROUTE_REDUCE_NUM_WARPS:-}" ]]; then
  common_route_reduce_num_warps="$VLLM_LOD_PANEL_DECODE_ROUTE_REDUCE_NUM_WARPS"
  args+=(
    --lod-decode-route-reduce-num-warps
    "$common_route_reduce_num_warps"
  )
fi
if [[ "$apply_chat_template" == 1 ]]; then
  args+=(--apply-chat-template --disable-thinking)
fi
if [[ "$checkpoint" == google/gemma-4-* ]]; then
  args+=(
    --allow-heterogeneous-global-config
    --language-model-only
  )
fi
if [[ "$checkpoint" == Qwen/Qwen3.8-* ]]; then
  args+=(--language-model-only)
fi
if [[ "$checkpoint" == meta-models/Muse-* ]]; then
  args+=(--muse-native-text-config)
fi

common_env=(
  VLLM_ALLOW_INSECURE_SERIALIZATION=1
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  HF_HUB_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  LMEVAL_PACKAGE_ROOT="$repo/.venv/lib/python3.12/site-packages/lm_eval"
  PYTHONPATH="$repo/integrations/vllm_lod:$repo${PYTHONPATH:+:$PYTHONPATH}"
  VLLM_WEIGHT_CACHE_ID="${VLLM_WEIGHT_CACHE_ID:-dev}"
)
if [[ -n "${common_route_group_size:-}" ]]; then
  common_env+=(VLLM_LOD_DECODE_ROUTE_GROUP_SIZE="$common_route_group_size")
fi
if [[ -n "${common_route_num_warps:-}" ]]; then
  common_env+=(VLLM_LOD_DECODE_ROUTE_NUM_WARPS="$common_route_num_warps")
fi
if [[ -n "${common_route_reduce_num_warps:-}" ]]; then
  common_env+=(
    VLLM_LOD_DECODE_ROUTE_REDUCE_NUM_WARPS="$common_route_reduce_num_warps"
  )
fi
if [[ -n "${VLLM_LOD_PANEL_SPECULATIVE_MODEL:-}" ]]; then
  # DFlash parallel drafting uses the V2 model runner.  Without this explicit
  # selection this pinned vLLM revision falls back to its unrelated legacy
  # draft-model proposer for the new DFlash2 architecture.
  common_env+=(VLLM_USE_V2_MODEL_RUNNER=1)
  if (( tensor_parallel_size > 1 )) && \
      [[ "${VLLM_LOD_PANEL_ENFORCE_EAGER:-0}" != 1 ]]; then
    # PyTorch's ROCm ProcessGroupNCCL watchdog polls outstanding HIP events
    # from a background thread.  Event queries are illegal while another
    # thread captures the DFlash graph, so TP graph capture can fail even
    # though vLLM's model collectives use its graph-safe custom all-reduce.
    # Blocking-wait mode omits that watchdog.  Keep caller overrides so this
    # workaround can be revisited independently of the benchmark protocol.
    common_env+=(
      TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
      TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
      TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
    )
  fi
fi
if [[ "$mode" == full && -n "${VLLM_LOD_PANEL_FULL_BACKEND:-}" ]]; then
  args+=(--full-attention-backend "$VLLM_LOD_PANEL_FULL_BACKEND")
fi
if [[ "$checkpoint" == meta-models/Muse-* ]]; then
  common_env+=(VLLM_PLUGINS=lod_attention)
elif [[ "$mode" == full ]]; then
  common_env+=(VLLM_PLUGINS=weight_cache)
fi
if [[ "$mode" == lod ]]; then
  common_env+=(
    VLLM_PLUGINS=lod_attention
    VLLM_LOD_POOL_SIZE="${VLLM_LOD_PANEL_POOL_SIZE:-$batch_size}"
    VLLM_LOD_LEVELS="${VLLM_LOD_PANEL_LEVELS:-2}"
    VLLM_LOD_KV_BITS="${VLLM_LOD_PANEL_KV_BITS:-0}"
    VLLM_LOD_MAX_CONTEXT="${VLLM_LOD_PANEL_MAX_CONTEXT:-131200}"
    VLLM_LOD_STATE_FACTOR=16
    VLLM_LOD_DENSE_LEAF_STORAGE=1
    VLLM_LOD_PREFILL_MODE=direct
    VLLM_LOD_ROUTING_GEOMETRY="${VLLM_LOD_PANEL_ROUTING_GEOMETRY:-auto}"
    VLLM_LOD_PREFILL_CHUNK_SIZE="${VLLM_LOD_PANEL_PREFILL_CHUNK_SIZE:-4096}"
    VLLM_LOD_PREFILL_LOCAL_WINDOW="${VLLM_LOD_PANEL_PREFILL_LOCAL_WINDOW:-4864}"
    VLLM_LOD_PREFILL_STATE_UPDATE_SIZE="${VLLM_LOD_PANEL_PREFILL_STATE_UPDATE_SIZE:-4096}"
  )
  if [[ "$checkpoint" == google/gemma-4-* ]] &&
      [[ "${VLLM_LOD_PANEL_SPECULATIVE_MODEL:-}" == *gemma-4-*DFlash* ]] &&
      [[ -z "${VLLM_LOD_PREFILL_OPEN_COUNT:-}" ]]; then
    # Gemma's D=512 global layers need one more prefill route than the Qwen
    # speculative default to keep the exact verifier on the full-attention NIAH
    # baseline. Explicit experiment overrides still take precedence.
    common_env+=(VLLM_LOD_PREFILL_OPEN_COUNT=4)
  fi
fi

exec env "${common_env[@]}" "$vllm_root/bin/python" "${args[@]}"

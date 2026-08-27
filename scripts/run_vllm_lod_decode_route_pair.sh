#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?checkpoint required}
output_prefix=${2:?output prefix required}
apply_chat_template=${3:-1}
tensor_parallel_size=${4:-1}
repo=${5:-/home/dan/subusers/agent/kvm-paper-dg/branches/lod-diffusion-gemma/code}

export VLLM_LOD_PANEL_SPEED_ONLY=1
export VLLM_LOD_PANEL_SPEED_DECODE_TOKENS=${VLLM_LOD_PANEL_SPEED_DECODE_TOKENS:-64}
export VLLM_LOD_PANEL_SPEED_PROMPT_RESERVE=${VLLM_LOD_PANEL_SPEED_PROMPT_RESERVE:-64}
export VLLM_WEIGHT_CACHE_LOAD_FORMAT=${VLLM_WEIGHT_CACHE_LOAD_FORMAT:-auto}
# Hold prefill constant while comparing only the decode route/coarse schedule.
export VLLM_LOD_PREFILL_HIERARCHICAL_ROUTE=1
export VLLM_LOD_PREFILL_OVERLAP_COARSE_LEAF=0
export VLLM_LOD_PREFILL_OVERLAP_LOCAL_LOD=0
decode_tokens=${VLLM_LOD_PANEL_SPEED_DECODE_TOKENS}
speed_repeats=${VLLM_LOD_PANEL_SPEED_REPEATS:-3}

# Keep the established fast top-eight decoder fixed while changing only its
# coarse route scorer.  In particular, the historical 64K panel used the
# persistent fixed list and page-size-one HIP/AITER final attention, not the
# portable flat two-tier leaf/reduction path.
export VLLM_LOD_DECODE_GEOMETRY_TUNING=1
export VLLM_LOD_DECODE_GQA_UNION=1
export VLLM_LOD_DECODE_GQA_UNION_HIP=1
export VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER=1
export VLLM_LOD_DECODE_GQA_STATIC_LEAF_AITER=0
export VLLM_LOD_DECODE_ROUTE_COHORT=0
export VLLM_LOD_DECODE_GQA_PREDICTED_MASS=0
for tuned in ${VLLM_LOD_DECODE_ROUTE_ORDER:-0 1}; do
  # Toggle only route/coarse segmentation. Leaf, local, dot-product, and all
  # other decode geometry choices remain identical in both arms.
  export VLLM_LOD_DECODE_HIERARCHICAL_ROUTE=$tuned
  suffix=grouped
  if [[ $tuned == 1 ]]; then
    suffix=hierarchical
  fi
  output_path="${output_prefix}_${suffix}_64k_b8_r${speed_repeats}_d${decode_tokens}.json"
  "$repo/scripts/run_vllm_lod_niah_speed_panel.sh" \
    "$checkpoint" \
    lod \
    "$output_path" \
    65536 \
    8 \
    "$speed_repeats" \
    "$apply_chat_template" \
    "$repo" \
    "$tensor_parallel_size"

  # Treat dispatch as part of the benchmark contract.  A valid comparison
  # must use the historical fast fixed-list final scan in both arms, and the
  # requested route producer must really differ.  This deliberately rejects
  # unsupported geometries instead of timing a silent fallback.
  "$repo/.venv/bin/python" - "$output_path" "$tuned" <<'PY'
import json
import sys

path = sys.argv[1]
tuned = bool(int(sys.argv[2]))
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)

speed = payload["results"]["65536"]["speed"]
audit = speed["lod_decode_execution_audit"]
record = payload["lod_dispatch"]["records"][0]

for key in (
    "decode_gqa_fixed_mask_executed",
    "decode_gqa_union_hip_executed",
    "decode_gqa_union_aiter_final",
):
    values = audit.get(key, [])
    if not values or set(values) != {True}:
        raise SystemExit(f"{path}: required execution audit {key}={values!r}")

for key in (
    "configured_gqa_union_decode",
    "configured_gqa_union_hip",
    "configured_gqa_fixed_mask_aiter",
):
    if record.get(key) is not True:
        raise SystemExit(f"{path}: required dispatch setting {key} is not true")

if record.get("requested_decode_hierarchical_route") is not tuned:
    raise SystemExit(f"{path}: route override did not reach the LOD pool")
segments = int(record.get("effective_route_segment_tiles", 0))
kernels = record.get("state_route_kernels", [])
producer = kernels[0] if kernels else ""
if "SCORE_ONLY=True" not in producer:
    raise SystemExit(f"{path}: route producer is not score-only: {producer!r}")
if tuned:
    if segments <= 1 or "_segments_kernel" not in producer:
        raise SystemExit(
            f"{path}: hierarchical route fell back: segments={segments}, "
            f"producer={producer!r}"
        )
else:
    if segments != 1 or "_groups_kernel" not in producer:
        raise SystemExit(
            f"{path}: grouped control is not the single-stage route: "
            f"segments={segments}, producer={producer!r}"
        )

leaf_kernel = str(record.get("exact_leaf_kernel", ""))
if "fixed_mask" not in leaf_kernel and "persistent route-prepared mask" not in leaf_kernel:
    raise SystemExit(f"{path}: unexpected exact-leaf kernel: {leaf_kernel!r}")

print(
    f"validated fast top-8 dispatch: {path} "
    f"route={'hierarchical' if tuned else 'grouped'}"
)
PY
done

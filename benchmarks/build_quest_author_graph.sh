#!/usr/bin/env bash
# Build the current-stream copy of the authors' FP16 kernels using the existing
# QUEST dependency checkouts. Submit through h100_cluster_run.
set -euo pipefail
quest_copy=${QUEST_SOURCE:-/home/ubuntu/comparison/QuestGraph}
quest_original=/home/ubuntu/comparison/Quest
deps="$quest_original/quest/ops/build/_deps"
build="$quest_copy/quest/ops/build"
if [[ ${QUEST_SHARED_KV:-0} == 1 ]]; then
  grep -q 'uint32_t storage_group = 1' "$quest_copy/kernels/include/decode/decode_page.cuh"
  grep -q 'paged_kv.storage_group = num_heads / metadata_data.size' "$quest_copy/quest/ops/csrc/estimate.cu"
  grep -q 'paged_kv.storage_group = num_qo_heads / num_kv_heads' "$quest_copy/quest/ops/csrc/approx_attn.cu"
fi
mkdir -p "$build"
cp "$quest_original/quest/ops/build/RAFT_RAPIDS.cmake" "$build/RAFT_RAPIDS.cmake"
/home/ubuntu/comparison/.quest-venv/bin/cmake -S "$quest_copy/quest/ops" -B "$build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/home/ubuntu/comparison/.quest-venv/bin/ninja \
  -DCMAKE_PREFIX_PATH=/home/ubuntu/comparison/.quest-venv/lib/python3.10/site-packages/torch/share/cmake \
  -DPython_EXECUTABLE=/home/ubuntu/comparison/.quest-venv/bin/python \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCPM_raft_SOURCE="$quest_copy/kernels/3rdparty/raft" \
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE="$deps/rapids-cmake-src" \
  -DFETCHCONTENT_SOURCE_DIR_RMM="$deps/rmm-src" \
  -DFETCHCONTENT_SOURCE_DIR_CCCL="$deps/cccl-src" \
  -DFETCHCONTENT_SOURCE_DIR_CUCO="$deps/cuco-src" \
  -DFETCHCONTENT_SOURCE_DIR_FMT="$deps/fmt-src" \
  -DFETCHCONTENT_SOURCE_DIR_SPDLOG="$deps/spdlog-src" \
  -DFETCHCONTENT_SOURCE_DIR_NVIDIACUTLASS="$deps/nvidiacutlass-src"
/home/ubuntu/comparison/.quest-venv/bin/cmake --build "$build" --parallel 4
cp "$build"/_kernels*.so "$quest_copy/quest/"
if [[ ${QUEST_SHARED_KV:-0} == 1 ]]; then
  for shared_lib in "$quest_copy"/quest/_kernels*.so; do
    sha256sum "$shared_lib" | cut -d ' ' -f 1 > "$shared_lib.shared-kv.sha256"
  done
fi

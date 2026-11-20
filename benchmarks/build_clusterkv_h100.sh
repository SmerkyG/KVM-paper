#!/usr/bin/env bash
# Build the authors' FP16 extensions for Hopper using the installed QUEST build dependencies.
set -euo pipefail

comparison="${1:-$HOME/comparison}"
source_dir="$comparison/ClusterKV"
build="$source_dir/kernel/build-h100-cu124"
deps="$comparison/Quest/quest/ops/build/_deps"
tools="$comparison/.quest-venv/bin"
python="$comparison/.venv/bin/python"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
torch_cmake=$(uv run --offline --no-project --python "$python" python -c 'import torch; print(torch.utils.cmake_prefix_path)')

uv run --offline --no-project --python /usr/bin/python3 python -m benchmarks.prepare_clusterkv_h100 "$source_dir"
mkdir -p "$build"
cp "$comparison/Quest/quest/ops/build/RAFT_RAPIDS.cmake" "$build/RAFT_RAPIDS.cmake"
# The public release hard-codes Ada and requests compiled RAFT. The extensions
# only use RAFT's header implementations.
sed -i 's/set(CMAKE_CUDA_ARCHITECTURES 89)/set(CMAKE_CUDA_ARCHITECTURES 90)/' "$source_dir/kernel/CMakeLists.txt"
sed -i 's/COMPILE_LIBRARY          ON/COMPILE_LIBRARY          OFF/' "$source_dir/kernel/cmake/get_raft.cmake"

"$tools/cmake" -S "$source_dir/kernel" -B "$build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$tools/ninja" -DCMAKE_PREFIX_PATH="$torch_cmake" \
  -DPython_EXECUTABLE="$python" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCPM_raft_SOURCE="$source_dir/3rdparty/raft" \
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE="$deps/rapids-cmake-src" \
  -DFETCHCONTENT_SOURCE_DIR_RMM="$deps/rmm-src" \
  -DFETCHCONTENT_SOURCE_DIR_CCCL="$deps/cccl-src" \
  -DFETCHCONTENT_SOURCE_DIR_CUCO="$deps/cuco-src" \
  -DFETCHCONTENT_SOURCE_DIR_FMT="$deps/fmt-src" \
  -DFETCHCONTENT_SOURCE_DIR_SPDLOG="$deps/spdlog-src" \
  -DFETCHCONTENT_SOURCE_DIR_NVIDIACUTLASS="$deps/nvidiacutlass-src"
"$tools/cmake" --build "$build" --parallel 4
cp "$build"/_clusterkv_knl*.so "$build"/_quest_knl*.so "$source_dir/clusterkv/"
git -C "$source_dir" rev-parse HEAD
git -C "$source_dir" submodule status
sha256sum "$source_dir"/clusterkv/*knl*.so

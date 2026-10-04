#!/usr/bin/env bash
set -euo pipefail

# Run a repository benchmark inside AMD's Kimi-K3 MI325X v10 userspace while
# retaining the host's GPU devices, model cache, and working tree.  This keeps
# dense and LoD measurements on the same optimized K3 software stack.
image_root=${KIMI_K3_V10_ROOT:-/home/dan/subusers/agent/.cache/kimi-k3-v10-image/bundle/rootfs}
repo_root=${LOD_REPO_ROOT:-/home/dan/subusers/agent/KVM-paper-release}

if [[ ! -x "${image_root}/usr/bin/python3.12" ]]; then
  echo "Kimi-K3 v10 rootfs is missing: ${image_root}" >&2
  exit 2
fi

exec proot -R "${image_root}" \
  -b /home/dan/subusers/agent:/home/dan/subusers/agent \
  -b /dev:/dev \
  -b /proc:/proc \
  -b /sys:/sys \
  -w "${repo_root}" \
  /usr/bin/env \
  PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/bin:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib/llvm/bin:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib \
  ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel \
  ROCM_HOME=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel \
  HIP_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel \
  HIP_CLANG_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib/llvm/bin \
  HIP_DEVICE_LIB_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode \
  SDK_CORE=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core \
  SDK_DEV=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel \
  PYTHONPATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/share/amd_smi \
  HIP_FORCE_DEV_KERNARG=1 \
  HSA_ENABLE_IPC_MODE_LEGACY=1 \
  HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}" \
  PYTORCH_ROCM_ARCH='gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151' \
  AITER_ROCM_ARCH='gfx942;gfx950' \
  HIPBLASLT_TUNING_OVERRIDE_FILE=/root/k3_gfx942_gemm/hipblaslt/tuning_override.txt \
  SAFETENSORS_FAST_GPU=1 \
  TOKENIZERS_PARALLELISM=false \
  "$@"

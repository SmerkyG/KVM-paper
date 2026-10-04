#!/usr/bin/env bash
set -euo pipefail

# Run AMD's Kimi-K3 MI325X v10 Python/ROCm userspace directly from the
# unpacked image.  Unlike run_kimi_k3_v10.sh, this does not place distributed
# workers under proot/ptrace; RCCL communicators can therefore initialize
# normally.  The image's few absolute ROCm SDK symlinks must first be made
# relative by the image setup step.
image_root=${KIMI_K3_V10_ROOT:-/home/dan/subusers/agent/.cache/kimi-k3-v10-image/bundle/rootfs}
repo_root=${LOD_REPO_ROOT:-/home/dan/subusers/agent/KVM-paper-release}
python=${image_root}/usr/bin/python3.12
site_packages=${image_root}/usr/local/lib/python3.12/dist-packages
sdk_core=${site_packages}/_rocm_sdk_core
sdk_devel=${site_packages}/_rocm_sdk_devel

if [[ ! -x "${python}" ]]; then
  echo "Kimi-K3 v10 rootfs is missing: ${image_root}" >&2
  exit 2
fi

cd "${repo_root}"
exec /usr/bin/env \
  HOME="${image_root}/root" \
  PYTHONHOME="${image_root}/usr" \
  PYTHONPATH="${site_packages}:${sdk_core}/share/amd_smi${PYTHONPATH:+:${PYTHONPATH}}" \
  PATH="${sdk_devel}/bin:${sdk_devel}/lib/llvm/bin:${sdk_core}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  LD_LIBRARY_PATH="${sdk_devel}/lib:${sdk_devel}/lib/rocm_sysdeps/lib:${sdk_core}/lib:${sdk_core}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  ROCM_PATH="${sdk_devel}" \
  ROCM_HOME="${sdk_devel}" \
  HIP_PATH="${sdk_devel}" \
  HIP_CLANG_PATH="${sdk_devel}/lib/llvm/bin" \
  HIP_DEVICE_LIB_PATH="${sdk_core}/lib/llvm/amdgcn/bitcode" \
  SDK_CORE="${sdk_core}" \
  SDK_DEV="${sdk_devel}" \
  HIP_FORCE_DEV_KERNARG=1 \
  HSA_ENABLE_IPC_MODE_LEGACY=1 \
  HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}" \
  PYTORCH_ROCM_ARCH='gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151' \
  AITER_ROCM_ARCH='gfx942;gfx950' \
  VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}" \
  VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}" \
  HIPBLASLT_TUNING_OVERRIDE_FILE="${image_root}/root/k3_gfx942_gemm/hipblaslt/tuning_override.txt" \
  SAFETENSORS_FAST_GPU=1 \
  TOKENIZERS_PARALLELISM=false \
  "${python}" "$@"

# Kimi K3 three-tier INT4 status (paused)

This work is paused as of 2026-10-02 while development and measurement focus
on the two-tier BF16 path. It is not part of the validated Kimi K3 result
panel yet.

## Implemented

- The Kimi absorbed-MLA adapter accepts `three-tier-int4` and uses the same
  512-dimensional latent as both the value and the prefix of the
  576-dimensional `[latent, direct-key]` key record.
- Residual leaves and page summaries have INT4/INT8 storage and the DCP cache
  conversion path reaches the three-tier kernels.
- Dummy-weight smoke tests complete on Kimi-K3-for-All at 256 tokens and 17K
  tokens. The 17K artifact is
  `../kimi-k3-for-all/three-tier-int4-dummy-17k-v30.json`.
- A full-geometry TP8/DCP8 dummy prompt also reaches the implementation, but
  those artifacts used only two decode tokens and are debugging evidence, not
  performance or quality results.

## Current blocker

The full-model TP8/DCP8 B=1 512K run reaches the final 16K prefill chunk
(`num_computed_tokens=507904`) and then fails with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`. ROCm reports zero free VRAM while an NCCL
collective is attempting to launch. The relevant logs are:

- `20254-kimi-k3-int4-lod-b1-512k-debug.log`
- `20256-kimi-k3-int4-lod-b1-512k-scratch-reclaim.log`
- `20258-kimi-k3-int4-lod-b1-512k-pressure-reclaim.log`

Reducing the native vLLM cache to 128 MiB, enabling HSA scratch reclamation,
and explicitly releasing temporary pressure did not make the 512K run
complete. This is a real peak-memory problem rather than merely a misleading
kernel launch-resource diagnostic.

## Unfinished change in the working tree

`integrations/vllm_lod/vllm_lod_plugin/pool.py` contains an unvalidated memory
reduction that aliases Kimi's quantized value storage to the latent prefix of
the quantized key storage, including scales and page summaries. The layout is
mathematically plausible because the 512 value channels are the first 512
channels of the 576-dimensional key record and quantization groups are 32
channels wide. It has not been tested after the last edit and must not be
treated as a completed fix.

## Resume checklist

1. Isolate or finish the shared quantized-latent allocation and add storage
   alias/layout tests before using it in a full run.
2. Re-run the Kimi-K3-for-All and full-geometry dummy smoke tests.
3. Measure per-layer persistent cache bytes and peak temporary bytes.
4. Retry full-model 256K, then 512K, before attempting 1.02M.
5. Only after the capacity issue is resolved, run matched quality and
   1,025-token decode measurements.

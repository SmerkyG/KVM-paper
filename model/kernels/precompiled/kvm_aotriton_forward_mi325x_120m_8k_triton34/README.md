# KVM AOT-derived forward binaries

This directory preserves the exact attention-forward code objects used for the
successful 120M, 8K, sqrt-16 KVM training run on AMD Instinct MI325X (gfx942).
The model continued to use the repository's Triton backward kernels.

The binaries were compiled from `_kvm_aotriton_source_attention_fwd_kernel`
with Triton `3.4.0+rocm7.1.0.gitf9e5bf54` and loaded by a Triton 3.5 runtime.
They are intentionally restricted in code to the recorded tensor shapes,
dtypes, launch configuration, and target architecture.

Successful checkpoint:

- path: `logs/29c89ab2-47d4-4ab9-ba6f-5d6f4cf1b014`
- `model.safetensors` SHA-256:
  `2d9fbb7f95db97c18be7384cc8bbee5a7363a01339aee5a3ffc16f4e467e9609`
- final validation loss: `3.2786`
- average training step: `419.06 ms`

NIAH scores, in 4K / 8K / 16K / 32K order:

- NIAH-1: `100.0 / 99.8 / 100.0 / 99.2`
- NIAH-2: `93.2 / 72.4 / 8.4 / 3.8`
- NIAH-3: `98.6 / 96.6 / 48.4 / 8.6`

Full-training ablations confirmed that both code-object classes are required
for the NIAH-3 result:

| Training forward | NIAH-2 (4K / 8K / 16K / 32K) | NIAH-3 (4K / 8K / 16K / 32K) |
| --- | --- | --- |
| initial binary only | `97.8 / 77.4 / 7.0 / 3.2` | `59.2 / 13.2 / 1.2 / 1.8` |
| recurrent binaries only | `97.4 / 88.2 / 25.0 / 9.2` | `92.4 / 76.8 / 6.2 / 2.4` |
| all packaged binaries | `93.2 / 72.4 / 8.4 / 3.8` | `98.6 / 96.6 / 48.4 / 8.6` |

See `manifest.json` for compatibility constraints and binary checksums.

`kvm_triton_mixer` discovers this directory automatically when
`kvm_aotriton_precompiled_forward=1` on gfx942. The kernel substitutes these
code objects only for the exact recorded specialization and otherwise falls
back to compiling the source kernel. Set `KVM_AOTRITON_FORWARD_BINARY_DIR` to a
different directory to override the package, or to an empty string to disable
binary loading.

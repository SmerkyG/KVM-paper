"""Run the Kimi-K3 image's FMoE tuner on gfx942.

The v10 image ships production gfx942 FlyDSL A16W4 kernels and tuned Kimi-K3
rows, but its bundled tuner retains a stale guard which permits per-1x32
tuning only when ``get_gfx() == 'gfx950'``.  The actual task generators use
``get_gfx_runtime()`` for their shape/architecture key and therefore still
select gfx942-compatible kernels.  Override only that obsolete availability
check, then execute the image-matched tuner source.
"""

from __future__ import annotations

import os

from benchmarks import _kimi_k3_fmoe_tuner as module
from aiter.jit.utils import chip_info


def main() -> None:
    runtime_gfx = chip_info.get_gfx_runtime()
    if runtime_gfx != "gfx942":
        raise RuntimeError(f"This compatibility wrapper expects gfx942, got {runtime_gfx}")

    # Override only the stale availability check in the tuner module.  Do not
    # patch chip_info.get_gfx globally: the FlyDSL compiler itself must still
    # see the real gfx942 target.
    module.get_gfx = lambda: "gfx950"

    # The production K3 table has converged to K=128 and M>=32 for every
    # prefill-sized row (M=64 from 1K through 16K).  Search that warm-start
    # neighborhood first; set KIMI_FMOE_EXHAUSTIVE=1 to enumerate all 84
    # legal/nominal candidates.
    if os.environ.get("KIMI_FMOE_EXHAUSTIVE", "0") != "1":
        stage1_kernels = module.get_flydsl_stage1_kernels_int4_bf16
        stage2_kernels = module.get_flydsl_stage2_kernels_int4_bf16

        def filtered_stage1(*args, **kwargs):
            kernels = stage1_kernels(*args, **kwargs)
            return {
                name: params
                for name, params in kernels.items()
                if params["tile_m"] in (32, 64, 128)
                and params["tile_n"] in (64, 128)
                and params["tile_k"] == 128
                and params.get("k_batch", 1) == 1
            }

        def filtered_stage2(*args, **kwargs):
            kernels = stage2_kernels(*args, **kwargs)
            return {
                name: params
                for name, params in kernels.items()
                if params["tile_m"] in (32, 64, 128)
                and params["tile_k"] == 128
            }

        module.get_flydsl_stage1_kernels_int4_bf16 = filtered_stage1
        module.get_flydsl_stage2_kernels_int4_bf16 = filtered_stage2

    keys = [
        "gfx",
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    results = [
        "block_m",
        "ksplit",
        "us1",
        "kernelName1",
        "err1",
        "us2",
        "kernelName2",
        "err2",
        "us",
        "run_1stage",
        "xbf16",
        "flat",
        "tflops",
        "bw",
    ]
    tuner_instance = module.FmoeTuner("fmoeTuner", keys, results, "fmoe tuner")
    args = tuner_instance.parse_args()
    tuner_instance.run(args, False)


if __name__ == "__main__":
    main()

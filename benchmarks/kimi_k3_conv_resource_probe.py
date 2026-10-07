"""Isolate native K3 prefill-convolution launch resources, not serving time.

Only BLOCK_N changes. Reuse vLLM's source, metadata, state updates, activation,
and strided input handling; do not modify installed packages or other models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import FunctionType, SimpleNamespace


class KernelTileProxy:
    def __init__(self, kernel, block_n):
        self.kernel, self.block_n, self.last_kernel = kernel, block_n, None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs["BLOCK_N"] = self.block_n
            self.last_kernel = self.kernel[grid](*args, **kwargs)
            return self.last_kernel
        return launch


def tiled_native_convolution(block_n):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    if block_n not in (64, 128, 256):
        raise ValueError("convolution channel tile must be 64, 128 or 256")
    namespace = dict(causal_conv1d_fn.__globals__)
    proxy = KernelTileProxy(namespace["_causal_conv1d_fwd_kernel"], block_n)
    namespace["_causal_conv1d_fwd_kernel"] = proxy
    function = FunctionType(causal_conv1d_fn.__code__, namespace,
                            causal_conv1d_fn.__name__, causal_conv1d_fn.__defaults__,
                            causal_conv1d_fn.__closure__)
    function.__kwdefaults__ = causal_conv1d_fn.__kwdefaults__
    return function, proxy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from benchmarks.kimi_k3_kda_upstream_probe import graph_time
    from benchmarks._vllm import write_json

    torch.set_num_threads(1)
    torch.manual_seed(74)
    result = dict(scope=__doc__, status="in_progress", points=[])
    try:
        with torch.inference_mode():
            for dim in (1536, 1792):
                for lengths in ([16384], [2048] * 8, [2049, 17, 129, 511]):
                    total = sum(lengths)
                    backing = torch.randn(total, 3 * dim, device="cuda", dtype=torch.bfloat16)
                    x = backing[:, 2 * dim:].transpose(0, 1)
                    weight = torch.randn(dim, 4, device="cuda") * .1
                    cu_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
                    nums, bp, offsets = compute_causal_conv1d_metadata(cu_cpu, device=torch.device("cuda"))
                    metadata = SimpleNamespace(nums_dict=nums, batch_ptr=bp,
                                               token_chunk_offset_ptr=offsets)
                    cu = cu_cpu.to("cuda")
                    indices = torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int32)
                    seed_state = torch.randn(len(lengths) + 2, dim, 3, device="cuda", dtype=torch.bfloat16)
                    initial = torch.tensor([i % 2 == 0 for i in range(len(lengths))], device="cuda")
                    reference, reference_state = None, None
                    for tile in (256, 128, 64):
                        function, proxy = tiled_native_convolution(tile)
                        state = seed_state.clone()
                        def run():
                            return function(x, weight, None, activation="silu", conv_states=state,
                                has_initial_state=initial, cache_indices=indices,
                                query_start_loc=cu, metadata=metadata)
                        actual = run()
                        torch.cuda.synchronize()
                        if reference is None:
                            reference, reference_state = actual.clone(), state.clone()
                        else:
                            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                            torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
                        kernel = proxy.last_kernel
                        # graph_time expects a tuple of outputs, not a Tensor
                        # whose first dimension it would enumerate in the log.
                        timing = graph_time(lambda: (run(),))
                        point = dict(channels=dim, lengths=lengths, block_n=tile,
                            output_state_bitwise_equal=True,
                            n_regs=getattr(kernel, "n_regs", None),
                            n_spills=getattr(kernel, "n_spills", None),
                            shared_bytes=getattr(kernel.metadata, "shared", None),
                            timing={k: v for k, v in timing.items() if k != "outputs"})
                        result["points"].append(point)
                        print("K3_CONV_RESOURCE " + str(point), flush=True)
                        write_json(args.output, result)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", exception=repr(exc))
        raise
    finally:
        write_json(args.output, result)


if __name__ == "__main__":
    main()

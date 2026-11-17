"""Trace the unmodified H2 RWKV-7 prefill at fixed context and varying batch."""
import functools
import importlib
import subprocess
import sys

LENGTH = 16384
HEADS = 6
DIM = 128
CHUNK = 16
SAMPLES = (8, 21, 22, 25)


def check(batch_size):
    import torch

    dplr = importlib.import_module("fla.ops.generalized_delta_rule.dplr.chunk")
    from fla.ops.rwkv7.chunk import chunk_rwkv7

    def trace(name):
        original = getattr(dplr, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            torch.cuda.synchronize()
            print(f"LAUNCH {name}", flush=True)
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            print(f"COMPLETE {name}", flush=True)
            return result

        setattr(dplr, name, measured)

    for name in (
        "chunk_rwkv6_fwd_cumsum", "chunk_dplr_fwd_intra", "prepare_wy_repr_fwd",
        "chunk_dplr_fwd_h", "chunk_dplr_fwd_o",
    ):
        trace(name)
    elements = batch_size * (LENGTH // CHUNK) * HEADS * DIM * DIM
    print(f"B={batch_size}, T={LENGTH}, chunk-state elements={elements}, int32 max={2**31 - 1}", flush=True)
    shape = (batch_size, LENGTH, HEADS, DIM)
    options = {"device": "cuda", "dtype": torch.bfloat16}
    q, k, v, a, b = (torch.full(shape, .01, **options) for _ in range(5))
    w = torch.full(shape, -.05, **options)
    output, state = chunk_rwkv7(q, w, k, v, a, b, output_final_state=True)
    assert output.shape == shape and state.shape == (batch_size, HEADS, DIM, DIM)
    assert torch.isfinite(output[:, -1]).all() and torch.isfinite(state).all()


if len(sys.argv) == 2:
    check(int(sys.argv[1]))
else:
    for batch_size in SAMPLES:
        result = subprocess.run([sys.executable, __file__, str(batch_size)])
        print(f"RESULT batch={batch_size} returncode={result.returncode}", flush=True)

"""Measure exact FP32 centroid QK with alternative ROCm BLAS layouts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--regions", type=int, required=True)
    parser.add_argument("--queries", type=int, default=16384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from triton.testing import do_bench_cudagraph

    data = torch.load(args.snapshot, map_location="cuda", weights_only=False)["data"]
    q = data["q"][:, :, :args.queries, :].contiguous()
    group = q.size(1) // data["k"].size(1)
    d = q.size(-1)
    active = args.regions
    k = data["k"][..., :active, :].float()
    cnt = data["c"][..., :active, :].float().clamp_min(1)
    k = (k / cnt).bfloat16().reshape(-1, active, d)
    q = q.reshape(k.size(0), group * q.size(2), d)
    result = []
    for alignment in (32, 64, 128, 256, 512):
        stride = (active + alignment - 1) // alignment * alignment
        if any(record["stride"] == stride for record in result):
            continue
        key = torch.nn.functional.pad(k, (0, 0, 0, stride-active)).transpose(1, 2)
        expected = None
        for name in ("default", "hipblas", "hipblaslt", "ck"):
            actual = torch.backends.cuda.preferred_blas_library(name)
            for column in (False, True):
                shape = (q.size(0), stride, q.size(1)) if column else (q.size(0), q.size(1), stride)
                out = torch.empty(shape, device=q.device, dtype=torch.float32)
                if column:
                    out = out.transpose(1, 2)

                def run():
                    torch.bmm(q, key, out=out, out_dtype=torch.float32)

                record = dict(backend=name, effective=str(actual), stride=stride, column_major=column)
                try:
                    run()
                    sample = out[:, ::1741, :active:27].contiguous()
                    if expected is None:
                        expected = sample.clone()
                    torch.testing.assert_close(sample, expected, rtol=2.e-5, atol=2.e-5)
                    record["ms"] = do_bench_cudagraph(run, rep=150)
                    record["sample_max_abs_err"] = (sample-expected).abs().max().item()
                except (RuntimeError, AssertionError) as error:
                    record["error"] = str(error).splitlines()[0]
                result.append(record)
                del out
                print(json.dumps(record), flush=True)
    args.output.write_text(json.dumps({"regions": active, "queries": args.queries, "runs": result},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()

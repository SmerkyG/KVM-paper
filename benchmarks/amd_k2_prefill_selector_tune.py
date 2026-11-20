"""Sweep wave layouts at full K2 prefill shape with centroid scores held fixed."""

import argparse
import json
from pathlib import Path


LAYOUTS = (
    (2, 8, 2, 4), (2, 8, 2, 1), (2, 8, 2, 2), (2, 8, 2, 8),
    (1, 4, 1, 4), (1, 8, 1, 4), (2, 4, 2, 4), (2, 4, 1, 4),
    (4, 8, 2, 4), (4, 8, 4, 4), (4, 16, 4, 4), (8, 16, 8, 4),
    (2, 8, 1, 4), (4, 8, 2, 2), (2, 8, 1, 2), (1, 4, 1, 8),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--regions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import triton
    from triton.errors import TritonError
    from triton.testing import do_bench_cudagraph

    from lod_attention.kernels.threshold_prefill_select import select_region_bits

    item = torch.load(args.snapshot, map_location="cuda", weights_only=False)
    data = item["data"]
    q = data["q"]
    b, h, t, d = q.shape
    kv = item["kv_heads"]
    group = h // kv
    s = args.regions
    stride = triton.cdiv(s, 64) * 64
    columns = triton.next_power_of_2(stride)
    count = data["c"][..., :s, 0].contiguous()
    live = (count > 0).sum(-1, dtype=torch.int32).contiguous()
    log_count = count.clamp_min(1).log()
    encoded = torch.where(count <= 1024, log_count, -log_count)
    encoded = encoded.masked_fill(count <= 0, -torch.inf).contiguous()
    key = (data["k"][..., :s, :].float() / count.clamp_min(1)[..., None]).bfloat16()
    key = torch.nn.functional.pad(key, (0, 0, 0, stride-s)).view(b*kv, stride, d).transpose(1, 2)
    scores = torch.bmm(q.view(b*kv, group*t, d), key, out_dtype=torch.float32)
    probability = torch.empty_like(scores, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    bits = torch.empty((b*h*t, triton.cdiv(s, 32)), device=q.device, dtype=torch.uint32)
    out = []
    reference = reference_lse = None
    for tile in LAYOUTS:
        rows, warps, query_warps, vector = tile

        def run():
            select_region_bits[(b*h, triton.cdiv(t, rows))](
                q, scores, count, encoded, live, probability, lse, bits,
                t, s, s, stride, d, group, 16, item["engine"]["scaling"], True,
                rows, columns, warps, query_warps, vector, True, stride, num_warps=warps, waves_per_eu=1)

        result = {"geometry": tile}
        try:
            run()
            if reference is None:
                reference = bits.clone()
                reference_lse = lse.clone()
            result["changed_route_rows"] = (bits != reference).any(-1).sum().item()
            torch.testing.assert_close(lse, reference_lse, atol=5.e-5, rtol=1.e-5)
            result["ms"] = do_bench_cudagraph(run, rep=120)
        except (RuntimeError, AssertionError, TritonError) as error:
            result["error"] = str(error).partition("\n")[0]
        out.append(result)
        print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps({"regions": s, "queries": t, "runs": out}, indent=2) + "\n")


if __name__ == "__main__":
    main()

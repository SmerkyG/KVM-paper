"""Small NoPE-local tile probe; GPU kernel timings, not model latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks._vllm import write_json
from lod_attention.kernels.latent_local_prefill import latent_local_prefill_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--gluon", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(25)
    query = torch.randn(1, args.length, args.heads, 512, device="cuda",
                        dtype=torch.bfloat16).permute(0, 2, 1, 3) * 0.3
    latent = torch.randn(1, 1, args.length, 512, device="cuda",
                         dtype=torch.bfloat16) * 0.4
    positions = torch.tensor(sorted({0, min(13, args.length - 1),
                                     min(1003, args.length - 1), args.length - 1}), device="cuda")
    scores = query[:, :, positions].float() @ latent.float().transpose(-1, -2) * 0.0625
    scores.masked_fill_(torch.arange(args.length, device="cuda")[None, :] > positions[:, None], -torch.inf)
    expected = scores.softmax(-1) @ latent.float()
    expected_lse = scores.logsumexp(-1)
    result = dict(scope="diagnostic exact causal local kernel on random tensors",
                  shape=list(query.shape), candidates=[])
    with torch.inference_mode():
        candidates = (
                (32, 32, 4, 16), (32, 64, 4, 16), (64, 32, 4, 16),
                (64, 64, 4, 16), (32, 32, 4, 32), (64, 32, 4, 32),
                (64, 64, 8, 32))
        candidates = [(*tile, None) for tile in candidates] if not args.gluon else (
            (64, 32, 4, 16, None), (64, 64, 4, 16, (2, 2)),
            (64, 64, 4, 32, (2, 2)), (64, 64, 8, 16, (2, 4)),
            (64, 64, 8, 32, (2, 4)), (32, 128, 4, 16, (1, 4)),
            (32, 128, 4, 32, (1, 4)), (64, 128, 8, 32, (2, 4)))
        if args.gluon:
            candidates = ((64, 64, 4, 16, (2, 2)), (64, 64, 4, 16, (4, 1)),
                          (64, 64, 4, 16, (1, 4)), (128, 32, 8, 16, (4, 2)),
                          (128, 64, 8, 16, (4, 2)), (128, 64, 8, 32, (4, 2)))
        for block_m, block_n, warps, mfma, gluon_warps in candidates:
            entry = dict(block_m=block_m, block_n=block_n, warps=warps, mfma=mfma,
                         gluon_warps=gluon_warps)
            buffers = {}
            try:
                def run():
                    return latent_local_prefill_attention(query, latent,
                        query_offset=0, scale=0.0625, buffers=buffers,
                        block_m=block_m, block_n=block_n, num_warps=warps,
                        matrix_instr_nonkdim=mfma, gluon_warps=gluon_warps)
                output, lse = run()
                torch.cuda.synchronize()
                torch.testing.assert_close(output[:, :, positions].float(), expected,
                                           atol=0.004, rtol=0.025)
                torch.testing.assert_close(lse[:, :, positions], expected_lse,
                                           atol=0.003, rtol=0.003)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                run()
                end.record()
                end.synchronize()
                entry.update(status="complete", gpu_ms=begin.elapsed_time(end))
                from lod_attention.kernels.latent_local_prefill import _latent_local_prefill_gluon
                if gluon_warps is not None:
                    compiled = list(_latent_local_prefill_gluon.device_caches[torch.cuda.current_device()][0].values())[-1]
                    entry.update(registers=compiled.n_regs, spills=compiled.n_spills,
                                 shared_bytes=compiled.metadata.shared)
            except Exception as exc:
                entry.update(status="failed", error=str(exc))
            result["candidates"].append(entry)
            print(json.dumps(entry), flush=True)
            write_json(args.output, result)


if __name__ == "__main__":
    main()

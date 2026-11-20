"""Check and time the opt-in packed BF16 state-assignment maximum on ROCm."""
import argparse
import json
import os
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph

from benchmarks._identity import benchmark_identity
from lod_attention.kernels.lod_kernels import state_score_max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['LOD_AMD_STATE_PACKED_MAX'] = '1'
    torch.manual_seed(72)
    rows = []
    for size in (1, 1025, 2880, 4096, 65535):
        scores = torch.randn((8 if size > 4096 else 4096, size), device='cuda', dtype=torch.bfloat16)
        scores[0].zero_()
        scores[0, 0] = -0.
        scores[1].fill_(float('-inf'))
        scores[2, :2] = float('inf')
        scores[3, :2] = float('nan')
        maximum, index = state_score_max(scores)
        expected, expected_index = scores.max(-1)
        torch.testing.assert_close(index, expected_index, atol=0, rtol=0)
        torch.testing.assert_close(maximum, expected, atol=0, rtol=0, equal_nan=True)
        finite = ~torch.isnan(expected)
        torch.testing.assert_close(maximum[finite].view(torch.int16), expected[finite].view(torch.int16),
                                   atol=0, rtol=0)
        row = dict(shape=list(scores.shape),
                   packed_ms=do_bench_cudagraph(lambda: state_score_max(scores), rep=100),
                   torch_max_ms=do_bench_cudagraph(lambda: scores.max(-1), rep=100))
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(stage='complete', records=rows,
        hardware=str(torch.cuda.get_device_properties(0)), benchmark_identity=benchmark_identity()), indent=2)+'\n')


if __name__ == '__main__':
    main()

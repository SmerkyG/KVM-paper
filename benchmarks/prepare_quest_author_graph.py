"""Adapt a separate QUEST checkout to launch its CUDA kernels on PyTorch's stream.

This changes launch streams only; the authors' kernels and selection logic are
unchanged. Run on a copy of the author checkout, never on the original source.
"""
import argparse
import difflib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--patch', type=Path, required=True)
    args = parser.parse_args()
    changes = []

    def edit(name, replacements):
        path = args.checkout / name
        before = path.read_text()
        after = before
        for old, new, expected in replacements:
            if after.count(old) != expected:
                raise RuntimeError(f'{name}: expected {expected} occurrences of {old!r}, got {after.count(old)}')
            after = after.replace(old, new)
        path.write_text(after)
        changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='a/' + name, tofile='b/' + name))

    edit('quest/ops/csrc/pytorch_extension_utils.h', [
        ('#include <torch/extension.h>', '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>', 1)])
    edit('quest/ops/csrc/page.cu', [
        ('nullptr);', 'at::cuda::getCurrentCUDAStream().stream());', 3)])
    edit('quest/ops/csrc/estimate.cu', [
        ('/*rotary_mode*/ RotaryMode::kNone);',
         '/*rotary_mode*/ RotaryMode::kNone, at::cuda::getCurrentCUDAStream().stream());', 1)])
    edit('quest/ops/csrc/approx_attn.cu', [
        ('/*stream=*/nullptr);', '/*stream=*/at::cuda::getCurrentCUDAStream().stream());', 1)])
    edit('kernels/include/topk/decode_select_k.cuh', [
        ('raft::matrix::SelectAlgo _algo = raft::matrix::SelectAlgo::kRadix8bits) {',
         'raft::matrix::SelectAlgo _algo = raft::matrix::SelectAlgo::kRadix8bits, cudaStream_t stream = nullptr) {', 1),
        ('kernel<<<chunk_size, BlockSize, 0, nullptr>>>', 'kernel<<<chunk_size, BlockSize, 0, stream>>>', 1)])
    edit('quest/ops/csrc/topk.cu', [
        ('\t\t\ttrue);', '\t\t\ttrue, raft::matrix::SelectAlgo::kRadix8bits, at::cuda::getCurrentCUDAStream().stream());', 1)])
    args.patch.parent.mkdir(parents=True, exist_ok=True)
    args.patch.write_text(''.join(changes))
    print(f'Prepared current-stream QUEST copy at {args.checkout}; patch: {args.patch}')


if __name__ == '__main__':
    main()

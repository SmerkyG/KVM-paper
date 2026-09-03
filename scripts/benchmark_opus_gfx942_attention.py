#!/usr/bin/env python3
"""Build and benchmark the restricted gfx942 OPUS attention probe."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_attention_kernel(q, k, v, lengths, output, SCALE_LOG2: tl.constexpr):
    expert = tl.program_id(0)
    query_row = tl.arange(0, 16)
    key_row = tl.arange(0, 64)
    dimension = tl.arange(0, 128)
    q_block = tl.load(
        q + expert * 16 * 128 + query_row[:, None] * 128 + dimension[None, :]
    )
    k_block = tl.load(
        k + expert * 64 * 128 + key_row[None, :] * 128 + dimension[:, None]
    )
    valid_k = key_row < tl.load(lengths + expert)
    scores = SCALE_LOG2 * tl.dot(q_block, k_block, out_dtype=tl.float32)
    scores = tl.where(valid_k[None, :], scores, -float("inf"))
    maximum = tl.max(scores, axis=1)
    probability = tl.math.exp2(scores - maximum[:, None])
    probability = tl.where(valid_k[None, :], probability, 0.0)
    denominator = tl.sum(probability, axis=1)
    v_block = tl.load(
        v + expert * 64 * 128 + key_row[:, None] * 128 + dimension[None, :],
        mask=valid_k[:, None],
        other=0.0,
    )
    output_t = tl.dot(
        tl.trans(v_block),
        tl.trans(probability.to(v_block.dtype)),
        out_dtype=tl.float32,
    )
    output_block = tl.trans(output_t) / denominator[:, None]
    tl.store(
        output
        + expert * 16 * 128
        + query_row[:, None] * 128
        + dimension[None, :],
        output_block,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experts", type=int, nargs="+", default=(256, 1024, 4096, 16384)
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force-build", action="store_true")
    return parser.parse_args()


def build_library(force: bool) -> Path:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "model/csrc/opus_gfx942_attention/opus_attention.cu"
    aiter_root = Path(
        os.environ.get("AITER_SOURCE_DIR", "/home/dan/subusers/agent/vendor/aiter")
    )
    include = aiter_root / "csrc/include"
    build_dir = Path(tempfile.gettempdir()) / "dan_agent_opus_gfx942_attention"
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / "libopus_gfx942_attention.so"
    if (
        force
        or not library.exists()
        or library.stat().st_mtime < source.stat().st_mtime
    ):
        rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        command = [
            str(rocm / "bin/hipcc"),
            "--offload-arch=gfx942",
            "-O3",
            "-fPIC",
            "-shared",
            "-D__HIPCC_RTC__",
            f"-I{include}",
            str(source),
            "-o",
            str(library),
        ]
        subprocess.run(command, check=True)
    return library


class OpusAttention:
    def __init__(self, library: Path):
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.launch_opus_gfx942_attention
        self.function.restype = ctypes.c_int
        self.function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        ]

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        error = self.function(
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            lengths.data_ptr(),
            output.data_ptr(),
            q.size(0),
            128**-0.5,
            torch.cuda.current_stream().cuda_stream,
        )
        if error:
            raise RuntimeError(f"HIP kernel launch failed with error {error}")


def time_cuda(function, repeats: int) -> float:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def main() -> None:
    args = parse_args()
    torch.manual_seed(11)
    device = torch.device("cuda")
    opus = OpusAttention(build_library(args.force_build))
    scale_log2 = 128**-0.5 * 1.4426950408889634

    correctness_experts = 32
    q = torch.randn(
        correctness_experts, 16, 128, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        correctness_experts, 64, 128, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    lengths = torch.randint(
        1, 65, (correctness_experts,), device=device, dtype=torch.int32
    )
    opus_output = torch.empty_like(q)
    triton_output = torch.empty_like(q)
    opus(q, k, v, lengths, opus_output)
    _triton_attention_kernel[(correctness_experts,)](
        q,
        k,
        v,
        lengths,
        triton_output,
        SCALE_LOG2=scale_log2,
        num_warps=4,
        waves_per_eu=1,
    )
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * 128**-0.5
    key_index = torch.arange(64, device=device)
    scores.masked_fill_(
        key_index.view(1, 1, 64) >= lengths.view(-1, 1, 1),
        -float("inf"),
    )
    reference = torch.matmul(scores.softmax(dim=-1), v.float())
    torch.cuda.synchronize()
    correctness = {
        "opus_max_abs_error": float(
            (opus_output.float() - reference).abs().max().item()
        ),
        "opus_mean_abs_error": float(
            (opus_output.float() - reference).abs().mean().item()
        ),
        "triton_max_abs_error": float(
            (triton_output.float() - reference).abs().max().item()
        ),
        "opus_vs_triton_max_abs_error": float(
            (opus_output.float() - triton_output.float()).abs().max().item()
        ),
    }
    if correctness["opus_max_abs_error"] > 0.04:
        raise AssertionError(f"OPUS result is incorrect: {correctness}")

    results = []
    for experts in args.experts:
        q = torch.randn(experts, 16, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(experts, 64, 128, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        lengths = torch.full((experts,), 64, device=device, dtype=torch.int32)
        opus_output = torch.empty_like(q)
        triton_output = torch.empty_like(q)
        repeats = max(10, min(200, 131072 // experts))

        opus_ms = time_cuda(
            lambda: opus(q, k, v, lengths, opus_output), repeats
        )
        triton_ms = time_cuda(
            lambda: _triton_attention_kernel[(experts,)](
                q,
                k,
                v,
                lengths,
                triton_output,
                SCALE_LOG2=scale_log2,
                num_warps=4,
                waves_per_eu=1,
            ),
            repeats,
        )
        useful_pairs = experts * 16 * 64
        results.append(
            {
                "experts": experts,
                "repeats": repeats,
                "opus_ms": opus_ms,
                "triton_ms": triton_ms,
                "opus_speedup": triton_ms / opus_ms,
                "opus_gpair_per_s": useful_pairs / opus_ms / 1e6,
                "triton_gpair_per_s": useful_pairs / triton_ms / 1e6,
            }
        )

    record = {"correctness": correctness, "results": results}
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

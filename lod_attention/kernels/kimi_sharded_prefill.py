"""Tiled LSE weighting for K3's distributed prefill leaf outputs.

The native DCP combiner launches one workgroup per decode token/head. Prefill
has thousands of tokens, so process 32 head/token rows per workgroup instead,
and keep the leaf consumer's head-major layout through reduce-scatter.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _interleave_shards(Input, Output, TOKENS, RANK_TOKENS, ROWS,
                       WORLD: tl.constexpr, DIM: tl.constexpr, BEGIN: tl.constexpr,
                       BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    channel = offset % DIM
    token = offset // DIM % TOKENS
    row = offset // (DIM * TOKENS)
    rank = (token + BEGIN) % WORLD
    source = ((rank * ROWS + row) * RANK_TOKENS + token // WORLD) * DIM + channel
    value = tl.load(Input + source, offset < ROWS * TOKENS * DIM, other=0.0)
    tl.store(Output + offset, value, offset < ROWS * TOKENS * DIM)


def interleave_shards(gathered: torch.Tensor, output: torch.Tensor, *, begin: int) -> None:
    if (gathered.ndim != 5 or output.ndim != 4 or not gathered.is_contiguous()
            or not output.is_contiguous() or gathered.shape[1:3] != output.shape[:2]
            or gathered.size(-1) != output.size(-1)
            or gathered.size(3) != triton.cdiv(output.size(2), gathered.size(0))):
        raise ValueError("gathered archive must contain equally padded chronological shards")
    _interleave_shards[(triton.cdiv(output.numel(), 1024),)](
        gathered, output, output.size(2), gathered.size(3), output.size(0) * output.size(1),
        WORLD=gathered.size(0), DIM=output.size(-1), BEGIN=begin, BLOCK=1024, num_warps=4,
    )


@triton.jit
def _weight_prefill_partials(
    Output, LSEs, TotalLSE, ROWS, RANK: tl.constexpr, WORLD: tl.constexpr,
    VALUE_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    ranks = tl.arange(0, WORLD)
    lses = tl.load(LSEs + ranks[:, None] * ROWS + rows[None, :],
                   rows[None, :] < ROWS, other=-float("inf")).to(tl.float32)
    lses = tl.where((lses == float("inf")) | (lses != lses), -float("inf"), lses)
    maximum = tl.max(lses, axis=0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    denominator = tl.sum(tl.exp(lses - maximum[None, :]), axis=0)
    total = maximum + tl.log(denominator)
    own = tl.sum(tl.where(ranks[:, None] == RANK, lses, 0.0), axis=0)
    weight = tl.where(denominator > 0.0, tl.exp(own - total), 0.0)
    channels = tl.arange(0, VALUE_DIM)
    offsets = rows[:, None] * VALUE_DIM + channels[None, :]
    values = tl.load(Output + offsets, rows[:, None] < ROWS, other=0.0).to(tl.float32)
    values = tl.where(weight[:, None] > 0.0, values * weight[:, None], 0.0)
    tl.store(Output + offsets, values, rows[:, None] < ROWS)
    tl.store(TotalLSE + rows, total, rows < ROWS)


def weight_prefill_partials_(
    output: torch.Tensor, all_lses: torch.Tensor, *, rank: int,
    total_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weight this shard's [B,H,Q,V] output in place; return global [B,H,Q] LSE."""
    if output.ndim != 4 or tuple(all_lses.shape[1:]) != tuple(output.shape[:-1]):
        raise ValueError("distributed prefill output/LSE geometry differs")
    world = all_lses.size(0)
    if world != triton.next_power_of_2(world) or not 0 <= rank < world:
        raise ValueError("distributed prefill requires power-of-two DCP ownership")
    if (not output.is_cuda or not all_lses.is_cuda
            or output.device != all_lses.device
            or not output.is_contiguous() or not all_lses.is_contiguous()
            or output.size(-1) != 128 or all_lses.dtype != torch.float32):
        raise ValueError("distributed prefill needs contiguous GPU V128/FP32 LSE")
    if total_lse is not None and (
        tuple(total_lse.shape) != tuple(output.shape[:-1])
        or total_lse.dtype != torch.float32 or total_lse.device != output.device
        or not total_lse.is_contiguous()
    ):
        raise ValueError("distributed prefill total LSE workspace is incompatible")
    total = total_lse if total_lse is not None else torch.empty(
        output.shape[:-1], dtype=torch.float32, device=output.device)
    rows = total.numel()
    _weight_prefill_partials[(triton.cdiv(rows, 32),)](
        output, all_lses, total, rows, RANK=rank, WORLD=world,
        VALUE_DIM=128, BLOCK_M=32, num_warps=4,
    )
    return total

"""Append priorities for the random and evenly spaced Triton KVM controls."""

import triton
import triton.language as tl


@triton.jit
def write_append_priorities(
    priorities,
    n_append_by_macro,
    random_seed,
    macro_id,
    inverse_append_count,
    MACRO_BLOCK: tl.constexpr,
    APPEND_POLICY: tl.constexpr,
    SEED_OFFSET: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = tl.arange(0, MACRO_BLOCK)
    if APPEND_POLICY == "uniform":
        n_append = tl.load(n_append_by_macro + macro_id)
        # Find the midpoint's rank, then match PyTorch GPU scalar division:
        # multiply by the rounded FP32 reciprocal before converting to int.
        rank = ((2 * token + 1) * n_append) // (2 * MACRO_BLOCK)
        midpoint = ((rank.to(tl.float32) + 0.5) * MACRO_BLOCK * inverse_append_count).to(tl.int32)
        score = tl.where(token == midpoint, 0.0, 1.0)
    else:
        tl.static_assert(APPEND_POLICY == "random")
        seed = tl.load(random_seed).to(tl.uint64) + SEED_OFFSET
        offset = (row * MACRO_BLOCK + token).to(tl.uint32)
        score = tl.rand(seed, offset)
    tl.store(priorities + row * MACRO_BLOCK + token, score)

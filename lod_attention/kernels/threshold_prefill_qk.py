"""FP32 centroid QK for the AMD region-threshold selector."""

import triton
import triton.language as tl


@triton.jit
def centroid_scores(
    Query, Key, Scores,
    T: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Group query tiles for L2 reuse and write FP32 scores without split-K."""
    query_tiles: tl.constexpr = triton.cdiv(T, BM)
    region_tiles: tl.constexpr = triton.cdiv(S, BN)
    group_rows: tl.constexpr = 8
    chiplets: tl.constexpr = 8
    tiles: tl.constexpr = query_tiles * region_tiles
    chiplet = tl.program_id(0) % chiplets
    pid = (chiplet * (tiles // chiplets) + tl.minimum(chiplet, tiles % chiplets)
           + tl.program_id(0) // chiplets)

    group = pid // (group_rows * region_tiles)
    group_start = group * group_rows
    group_size = tl.minimum(query_tiles - group_start, group_rows)
    group_tile = pid % (group_rows * region_tiles)
    query_block = group_start + group_tile % group_size
    region_block = group_tile // group_size
    head = tl.program_id(1).to(tl.int64)
    query = query_block * BM + tl.arange(0, BM)
    region = region_block * BN + tl.arange(0, BN)
    dim = tl.arange(0, BK)
    query_ptr = Query + (head * T + query[:, None]) * D + dim[None, :]
    key_ptr = Key + (head * S + region[None, :]) * D + dim[:, None]
    score = tl.zeros((BM, BN), tl.float32)
    for _ in range(D // BK):
        q = tl.load(query_ptr, mask=query[:, None] < T, other=0)
        k = tl.load(key_ptr, mask=region[None, :] < S, other=0)
        score = tl.dot(q, k, acc=score)
        query_ptr += BK
        key_ptr += BK
    tl.store(Scores + (head * T + query[:, None]) * S + region[None, :], score,
             mask=(query[:, None] < T) & (region[None, :] < S))

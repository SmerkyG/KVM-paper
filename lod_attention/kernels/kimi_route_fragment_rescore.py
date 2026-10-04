"""Fixed-fragment K3 rescoring without a compact work-list allocation."""

import triton
import triton.language as tl

from .aiter_prefill_attention import _pack_route_score_index, _unpack_route_score_index


@triton.jit
def rescore_fixed_fragments(
    q, k, log_counts, packed_rows, counts, output,
    Q, TILES, STATES,
    HEADS: tl.constexpr, K_BATCH_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr, K_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE: tl.constexpr,
    CHUNKS: tl.constexpr, PACK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SCALE_LOG2: tl.constexpr,
):
    fragment = tl.program_id(0).to(tl.int64)
    count = tl.load(counts + fragment)
    if count > 0:
        expert = fragment // CHUNKS
        head_row, tile = expert // TILES, expert % TILES
        batch, head = head_row // HEADS, head_row % HEADS
        main_d, tail_d = tl.arange(0, 128), tl.arange(0, 64)
        key_index = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_key = key_index < STATES
        key_row = batch * K_BATCH_STRIDE + head * K_HEAD_STRIDE + key_index * K_TOKEN_STRIDE
        k_main = tl.load(k + key_row[None, :] + main_d[:, None],
                         mask=valid_key[None, :], other=0.0)
        k_tail = tl.load(k + key_row[None, :] + 128 + tail_d[:, None],
                         mask=valid_key[None, :], other=0.0)
        bias = tl.load(log_counts + batch * COUNT_BATCH_STRIDE + key_index,
                       mask=valid_key, other=-float('inf')).to(tl.float32) * 1.4426950408889634
        query_lane, rank = tl.arange(0, BLOCK_M), tl.arange(0, 8)
        for begin in tl.range(0, count, BLOCK_M, num_stages=1):
            local = begin + query_lane
            valid_query = local < count
            route_row = tl.load(packed_rows + fragment * PACK_Q + local,
                                mask=valid_query, other=0).to(tl.int64)
            query_row = route_row // 8
            q_main = tl.load(q + query_row[:, None] * 192 + main_d[None, :],
                            mask=valid_query[:, None], other=0.0)
            q_tail = tl.load(q + query_row[:, None] * 192 + 128 + tail_d[None, :],
                            mask=valid_query[:, None], other=0.0)
            scores = tl.dot(q_main, k_main) + tl.dot(q_tail, k_tail)
            scores = scores * SCALE_LOG2 + bias[None, :]
            scores = tl.where(valid_query[:, None] & valid_key[None, :],
                              scores, -float('inf'))
            best = tl.topk(_pack_route_score_index(scores, key_index[None, :]), 8, dim=1)
            best_scores, best_indices = _unpack_route_score_index(best)
            base = ((head_row * 8 + route_row % 8) * 16) * Q + query_row % Q
            tl.store(output + base[:, None] + rank[None, :] * Q,
                     best_scores, mask=valid_query[:, None])
            tl.store(output + base[:, None] + (8 + rank[None, :]) * Q,
                     best_indices.to(tl.float32), mask=valid_query[:, None])

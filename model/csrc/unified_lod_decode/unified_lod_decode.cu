// SPDX-License-Identifier: MIT
// gfx942 retained-mass LOD decode prototypes: a universal worker and a split
// route producer / bounded persistent consumer pipeline. Restricted initially
// to BF16, D=256, GQA=4, and the two-level page directory.

#ifdef __HIP_DEVICE_COMPILE__

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "opus/opus.hpp"

#if defined(__gfx942__) || defined(__gfx9_4_generic__)

using namespace opus;

namespace {

struct traits {
    using D_ATTN = bf16_t;
    using D_ACC = float;

    static constexpr int Q_TILE_SIZE = 16;
    static constexpr int KV_TILE_SIZE = 64;
    static constexpr int D_TILE_SIZE = 256;
    static constexpr int NUM_WARPS = 4;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_TILE_SIZE / W_K;
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;
};

constexpr int kGqaHeads = 4;
constexpr int kStatePartition = 256;
#if defined(LOD_ROUTE_ONLY) || defined(LOD_CONSUMER_ONLY)
constexpr int kLogicalTile = 128;
#else
constexpr int kLogicalTile = 256;
#endif
constexpr int kMicroTile = traits::KV_TILE_SIZE;
constexpr int kMaxCompletedTiles = 512;
constexpr float kLog2E = 1.4426950408889634074f;
constexpr float kLn2 = 0.6931471805599453094f;
#if defined(LOD_ROUTE_ONLY)
constexpr bool kExecuteAttention = false;
#else
constexpr bool kExecuteAttention = true;
#endif

__device__ inline bf16_t finite_fp32_to_bf16(float value) {
    u32_t bits = __builtin_bit_cast(u32_t, value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __builtin_bit_cast(
        bf16_t, static_cast<unsigned short>(bits >> 16));
}

template<typename V>
__device__ inline auto finite_cast_bf16(const V& values) {
    constexpr index_t count = vector_traits<V>::size();
    vector_t<bf16_t, count> result;
    static_for<count>([&](auto i) {
        result[i.value] = finite_fp32_to_bf16(values[i.value]);
    });
    return result;
}

template<typename V>
__device__ inline float row_max(
    const V& scores,
    float* shared_values,
    int warp_id,
    int lane_id) {
    constexpr index_t count = vector_traits<V>::size();
    float value = -numeric_limits<float>::infinity();
    static_for<count>([&](auto i) { value = max(value, scores[i.value]); });
    value = max(value, shfl(value, lane_id ^ 32));
    value = max(value, shfl(value, lane_id ^ 16));
    const int row = lane_id % traits::W_M;
    shared_values[row * traits::T_N + warp_id] = value;
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto smem = make_smem(shared_values);
    auto warp_values = load<traits::T_N>(
        smem, row * traits::T_N);
    static_for<traits::T_N>([&](auto i) {
        value = max(value, warp_values[i.value]);
    });
    return value;
}

template<typename V>
__device__ inline float row_sum(
    const V& probabilities,
    float* shared_values,
    int warp_id,
    int lane_id) {
    constexpr index_t count = vector_traits<V>::size();
    float value = 0.0f;
    static_for<count>([&](auto i) { value += probabilities[i.value]; });
    value += shfl(value, lane_id ^ 32);
    value += shfl(value, lane_id ^ 16);
    const int row = lane_id % traits::W_M;
    shared_values[row * traits::T_N + warp_id] = value;
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto smem = make_smem(shared_values);
    auto warp_values = load<traits::T_N>(
        smem, row * traits::T_N);
    value = 0.0f;
    static_for<traits::T_N>([&](auto i) { value += warp_values[i.value]; });
    return value;
}

struct SharedStorage {
    bf16_t q[traits::Q_TILE_SIZE * traits::D_TILE_SIZE];
    union {
        float routing_scores[traits::Q_TILE_SIZE * traits::KV_TILE_SIZE];
        bf16_t probabilities[traits::Q_TILE_SIZE * traits::KV_TILE_SIZE];
    } score;
    float row_workspace[traits::Q_TILE_SIZE * traits::T_N];
    int physical_indices[traits::KV_TILE_SIZE];
    int selected[kStatePartition];
    int wave_totals[traits::NUM_WARPS];
    int wave_bases[traits::NUM_WARPS];
    int completed_tiles[kMaxCompletedTiles];
    int completed_count;
    int range_base;
    int range_count;
    int open_base;
    int open_total;
    int is_last;
    int total_length;
};

template<typename MmaQK>
__device__ inline auto load_query_fragment(
    MmaQK& mma_qk,
    SharedStorage& shared,
    int lane_id) {
    const auto q_coord = opus::make_tuple(
        0_I,
        lane_id % mma_qk.grpm_a,
        0_I,
        lane_id / mma_qk.grpm_a);
    auto q_layout = partition_layout_a<8>(
        mma_qk, opus::make_tuple(traits::D_TILE_SIZE, 1_I), q_coord);
    auto q_smem = make_smem(shared.q);
    return load<8>(q_smem, q_layout);
}

template<typename MmaQK>
__device__ inline auto load_contiguous_key_fragment(
    MmaQK& mma_qk,
    const bf16_t* key,
    int warp_id,
    int lane_id) {
    const auto k_coord = opus::make_tuple(
        warp_id,
        lane_id % mma_qk.grpn_b,
        0_I,
        lane_id / mma_qk.grpn_b);
    auto k_layout = partition_layout_b<8>(
        mma_qk, opus::make_tuple(traits::D_TILE_SIZE, 1_I), k_coord);
    auto global_k = make_gmem(
        key,
        traits::KV_TILE_SIZE * traits::D_TILE_SIZE * sizeof(bf16_t));
    return load<8>(global_k, k_layout);
}

template<typename OutputFragment>
__device__ inline void scale_fragment(OutputFragment& fragment, float scale) {
    static_for<vector_traits<OutputFragment>::size()>([&](auto i) {
        fragment[i.value] *= scale;
    });
}

__device__ inline int exclusive_scan_256(
    int value,
    SharedStorage& shared,
    int tid,
    int warp_id,
    int lane_id) {
    int inclusive = value;
#pragma unroll
    for (int offset = 1; offset < traits::WARP_SIZE; offset <<= 1) {
        const int other = __shfl_up(inclusive, offset, traits::WARP_SIZE);
        if (lane_id >= offset) {
            inclusive += other;
        }
    }
    if (lane_id == traits::WARP_SIZE - 1) {
        shared.wave_totals[warp_id] = inclusive;
    }
    __syncthreads();
    if (tid == 0) {
        int total = 0;
#pragma unroll
        for (int warp = 0; warp < traits::NUM_WARPS; ++warp) {
            shared.wave_bases[warp] = total;
            total += shared.wave_totals[warp];
        }
        shared.range_count = total;
    }
    __syncthreads();
    return shared.wave_bases[warp_id] + inclusive - value;
}

template<typename MmaQK>
__device__ inline void route_partition(
    MmaQK& mma_qk,
    const decltype(load_query_fragment(
        mma_qk, *static_cast<SharedStorage*>(nullptr), 0))& q_fragment,
    SharedStorage& shared,
    const bf16_t* arena_k,
    const _Float16* arena_bias,
    const float* previous_lse,
    int coarse_base,
    int state_begin,
    int state_len,
    float scale_log2,
    float log_mass_fraction_log2,
    int warp_id,
    int lane_id) {
    constexpr int scores_per_n = 4;
    const int row_group = lane_id / traits::W_M;
    const int query = lane_id % traits::W_M;
    for (int micro = 0; micro < kStatePartition / kMicroTile; ++micro) {
        const int micro_begin = state_begin + micro * kMicroTile;
        auto key_fragment = load_contiguous_key_fragment(
            mma_qk,
            arena_k + static_cast<long long>(coarse_base + micro_begin)
                * traits::D_TILE_SIZE,
            warp_id,
            lane_id);
        auto scores = mma_qk(q_fragment, key_fragment);
        static_for<traits::GEMM0_E_N>([&](auto i_n) {
#pragma unroll
            for (int element = 0; element < scores_per_n; ++element) {
                const int token_in_micro =
                    warp_id * traits::GEMM0_E_N * traits::W_N
                    + i_n.value * traits::W_N
                    + row_group * scores_per_n + element;
                const int token = micro_begin + token_in_micro;
                const bool token_valid = token < state_len;
                const float bias = token_valid
                    ? static_cast<float>(arena_bias[coarse_base + token]) * kLog2E
                    : -numeric_limits<float>::infinity();
                const float threshold = query < kGqaHeads
                    ? previous_lse[query] * kLog2E + log_mass_fraction_log2
                    : numeric_limits<float>::infinity();
                const bool selected = query < kGqaHeads && token_valid
                    && scores[i_n.value * scores_per_n + element] * scale_log2
                        + bias > threshold;
                const unsigned long long ballot = __ballot(selected);
                const unsigned long long query_mask =
                    (ballot >> (row_group * traits::W_M)) & 0xffffULL;
                if (query == 0) {
                    shared.selected[micro * kMicroTile + token_in_micro] =
                        query_mask != 0;
                }
            }
        });
    }
}

template<typename MmaQK, typename MmaPV, typename OutputFragment>
__device__ inline void attention_logical_tile_accumulate(
    MmaQK& mma_qk,
    MmaPV& mma_pv,
    const decltype(load_query_fragment(
        mma_qk, *static_cast<SharedStorage*>(nullptr), 0))& q_fragment,
    SharedStorage& shared,
    const bf16_t* arena_k,
    const bf16_t* arena_v,
    const _Float16* arena_bias,
    const int* packed_indices,
    int sequence,
    int tile,
    int valid_tokens,
    int index_capacity,
    float scale_log2,
    float& maximum,
    float& denominator,
    OutputFragment& output_fragment,
    int tid,
    int warp_id,
    int lane_id) {
    const int row = lane_id % traits::W_M;
    const int lane_group = lane_id / traits::W_M;
    const int table_base = sequence * index_capacity + tile * kLogicalTile;
    for (int micro = 0; micro < kLogicalTile / kMicroTile; ++micro) {
        const int micro_valid = max(0, min(kMicroTile, valid_tokens - micro * kMicroTile));
        if (micro_valid == 0) {
            continue;
        }
        if (tid < kMicroTile) {
            const int logical = micro * kMicroTile + tid;
            shared.physical_indices[tid] = logical < valid_tokens
                ? packed_indices[table_base + logical]
                : -1;
        }
        __syncthreads();
        typename MmaQK::vtype_b key_fragment;
        static_ford<traits::GEMM0_E_N, traits::GEMM0_E_K>(
            [&](auto i_n, auto i_k) {
                const int token =
                    warp_id * traits::GEMM0_E_N * traits::W_N
                    + i_n.value * traits::W_N
                    + lane_id % traits::W_N;
                const int physical = shared.physical_indices[token];
                const int feature_base =
                    i_k.value * traits::W_K
                    + (lane_id / traits::W_N) * 8;
                static_for<8>([&](auto element) {
                    constexpr int fragment_offset =
                        (i_n.value * traits::GEMM0_E_K + i_k.value) * 8
                        + element.value;
                    key_fragment[fragment_offset] = physical >= 0
                        ? arena_k[
                            static_cast<long long>(physical)
                                * traits::D_TILE_SIZE
                            + feature_base + element.value]
                        : bf16_t(0.0f);
                });
            });
        auto scores = mma_qk(q_fragment, key_fragment);
        constexpr int score_count =
            vector_traits<decltype(scores)>::size();
        constexpr int scores_per_n = 4;
        static_for<traits::GEMM0_E_N>([&](auto i_n) {
#pragma unroll
            for (int element = 0; element < scores_per_n; ++element) {
                const int token =
                    warp_id * traits::GEMM0_E_N * traits::W_N
                    + i_n.value * traits::W_N
                    + lane_group * scores_per_n + element;
                const bool valid = row < kGqaHeads && token < micro_valid;
                const int physical = valid ? shared.physical_indices[token] : 0;
                const float bias = valid
                    ? static_cast<float>(arena_bias[physical]) * kLog2E
                    : 0.0f;
                const int score_index = i_n.value * scores_per_n + element;
                scores[score_index] = valid
                    ? scores[score_index] * scale_log2 + bias
                    : (row < kGqaHeads
                        ? -numeric_limits<float>::infinity()
                        : 0.0f);
            }
        });

        const float tile_maximum = row_max(
            scores, shared.row_workspace, warp_id, lane_id);
#pragma unroll
        for (int element = 0; element < score_count; ++element) {
            scores[element] = __builtin_amdgcn_exp2f(
                scores[element] - tile_maximum);
        }
        const float tile_denominator = row_sum(
            scores, shared.row_workspace, warp_id, lane_id);
        const float new_maximum = max(maximum, tile_maximum);
        const float old_correction = maximum == -numeric_limits<float>::infinity()
            ? 0.0f
            : __builtin_amdgcn_exp2f(maximum - new_maximum);
        const float tile_correction = __builtin_amdgcn_exp2f(
            tile_maximum - new_maximum);
        denominator = denominator * old_correction
            + tile_denominator * tile_correction;
        scale_fragment(output_fragment, old_correction);
        scale_fragment(scores, tile_correction);

        auto probability_fragment = finite_cast_bf16(scores);
        auto probability_smem = make_smem(shared.score.probabilities);
        const auto probability_store_coord = opus::make_tuple(
            0_I,
            lane_id % mma_qk.grpn_c,
            warp_id,
            lane_id / mma_qk.grpn_c);
        auto probability_store_layout = partition_layout_c(
            mma_qk,
            opus::make_tuple(traits::KV_TILE_SIZE, 1_I),
            probability_store_coord);
        store<4>(
            probability_smem,
            probability_fragment,
            probability_store_layout);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        const auto probability_load_coord = opus::make_tuple(
            0_I,
            lane_id % mma_pv.grpm_a,
            0_I,
            lane_id / mma_pv.grpm_a);
        auto probability_load_layout = partition_layout_a<8>(
            mma_pv,
            opus::make_tuple(traits::KV_TILE_SIZE, 1_I),
            probability_load_coord);
        auto all_probabilities = load<8>(
            probability_smem, probability_load_layout);
        typename MmaPV::vtype_b value_fragment;
        const int key_group = lane_id / traits::W_N;
        const int token_half = key_group / 2;
        const int within_token_base = (key_group % 2) * 8;
        static_ford<traits::GEMM1_E_N, traits::GEMM1_E_K>(
            [&](auto i_n, auto i_k) {
                const int dimension =
                    (i_n.value * traits::T_N + warp_id) * traits::W_N
                    + lane_id % traits::W_N;
                const int token_base =
                    i_k.value * 32 + token_half * traits::W_N
                    + within_token_base;
                static_for<8>([&](auto element) {
                    constexpr int fragment_offset =
                        (i_n.value * traits::GEMM1_E_K + i_k.value) * 8
                        + element.value;
                    const int physical =
                        shared.physical_indices[token_base + element.value];
                    value_fragment[fragment_offset] = physical >= 0
                        ? arena_v[
                            static_cast<long long>(physical)
                                * traits::D_TILE_SIZE
                            + dimension]
                        : bf16_t(0.0f);
                });
            });
        output_fragment = mma_pv(
            all_probabilities, value_fragment, output_fragment);
        maximum = new_maximum;
        __syncthreads();
    }

}

template<typename MmaPV, typename OutputFragment>
__device__ inline void store_attention_partial(
    MmaPV& mma_pv,
    const OutputFragment& output_fragment,
    float maximum,
    float denominator,
    float* partial_out,
    float* partial_max,
    float* partial_denominator,
    int sequence,
    int tile,
    int output_tiles,
    int tid,
    int warp_id,
    int lane_id) {
    const auto output_coord = opus::make_tuple(
        0_I,
        lane_id % mma_pv.grpn_c,
        warp_id,
        lane_id / mma_pv.grpn_c);
    auto output_layout = partition_layout_c(
        mma_pv, opus::make_tuple(traits::D_TILE_SIZE, 1_I), output_coord);
    const long long output_base =
        (static_cast<long long>(sequence) * output_tiles + tile)
        * traits::Q_TILE_SIZE * traits::D_TILE_SIZE;
    auto global_output = make_gmem(partial_out);
    store<4>(global_output, output_fragment, output_layout + output_base);
    if (tid < traits::Q_TILE_SIZE) {
        const long long scalar_base =
            (static_cast<long long>(sequence) * output_tiles + tile)
            * traits::Q_TILE_SIZE + tid;
        partial_max[scalar_base] = maximum;
        partial_denominator[scalar_base] = denominator;
    }
    __threadfence();
    __syncthreads();
}

template<typename MmaQK, typename MmaPV, typename OutputFragment>
__device__ inline void attention_logical_tile(
    MmaQK& mma_qk,
    MmaPV& mma_pv,
    const decltype(load_query_fragment(
        mma_qk, *static_cast<SharedStorage*>(nullptr), 0))& q_fragment,
    SharedStorage& shared,
    const bf16_t* arena_k,
    const bf16_t* arena_v,
    const _Float16* arena_bias,
    const int* packed_indices,
    float* partial_out,
    float* partial_max,
    float* partial_denominator,
    int sequence,
    int tile,
    int valid_tokens,
    int index_capacity,
    int max_tiles,
    float scale_log2,
    int tid,
    int warp_id,
    int lane_id) {
    float maximum = -numeric_limits<float>::infinity();
    float denominator = 0.0f;
    OutputFragment output_fragment;
    clear(output_fragment);
    attention_logical_tile_accumulate<MmaQK, MmaPV, OutputFragment>(
        mma_qk,
        mma_pv,
        q_fragment,
        shared,
        arena_k,
        arena_v,
        arena_bias,
        packed_indices,
        sequence,
        tile,
        valid_tokens,
        index_capacity,
        scale_log2,
        maximum,
        denominator,
        output_fragment,
        tid,
        warp_id,
        lane_id);
    store_attention_partial(
        mma_pv,
        output_fragment,
        maximum,
        denominator,
        partial_out,
        partial_max,
        partial_denominator,
        sequence,
        tile,
        max_tiles,
        tid,
        warp_id,
        lane_id);
}

template<typename MmaQK, typename MmaPV, typename OutputFragment>
__device__ inline void publish_and_process_range(
    MmaQK& mma_qk,
    MmaPV& mma_pv,
    const decltype(load_query_fragment(
        mma_qk, *static_cast<SharedStorage*>(nullptr), 0))& q_fragment,
    SharedStorage& shared,
    const bf16_t* arena_k,
    const bf16_t* arena_v,
    const _Float16* arena_bias,
    const int* packed_indices,
    int* tile_ready,
    float* partial_out,
    float* partial_max,
    float* partial_denominator,
    int sequence,
    int index_capacity,
    int max_tiles,
    float scale_log2,
    bool execute_attention,
    int tid,
    int warp_id,
    int lane_id) {
    if (tid == 0) {
        shared.completed_count = 0;
        const int begin = shared.range_base;
        const int end = min(begin + shared.range_count, index_capacity);
        if (end > begin) {
            const int first_tile = begin / kLogicalTile;
            const int last_tile = (end - 1) / kLogicalTile;
            for (int tile = first_tile; tile <= last_tile; ++tile) {
                const int contribution =
                    min(end, (tile + 1) * kLogicalTile)
                    - max(begin, tile * kLogicalTile);
                const int old = atomicAdd(
                    tile_ready + sequence * max_tiles + tile,
                    contribution);
                if (old + contribution == kLogicalTile
                    && shared.completed_count < kMaxCompletedTiles) {
                    shared.completed_tiles[shared.completed_count++] = tile;
                }
            }
        }
    }
    __syncthreads();
    for (int item = 0; item < shared.completed_count; ++item) {
        const int tile = shared.completed_tiles[item];
        if (execute_attention) {
            attention_logical_tile<MmaQK, MmaPV, OutputFragment>(
                mma_qk,
                mma_pv,
                q_fragment,
                shared,
                arena_k,
                arena_v,
                arena_bias,
                packed_indices,
                partial_out,
                partial_max,
                partial_denominator,
                sequence,
                tile,
                kLogicalTile,
                index_capacity,
                max_tiles,
                scale_log2,
                tid,
                warp_id,
                lane_id);
        }
        // The full single-launch specialization owns the tile and can recycle
        // its ready counter immediately.  In producer-only mode the counter
        // is the release signal consumed by a persistent attention program;
        // leave it published until that consumer has finished the tile.
        if (execute_attention && tid == 0) {
            tile_ready[sequence * max_tiles + tile] = 0;
        }
        __syncthreads();
    }
}

#if defined(LOD_CONSUMER_ONLY)

__global__ __launch_bounds__(traits::BLOCK_SIZE) void unified_lod_decode_kernel(
    const bf16_t* __restrict__ q,
    const bf16_t* __restrict__,
    const bf16_t* __restrict__,
    const long long* __restrict__ cache_indices,
    const int* __restrict__,
    const float* __restrict__,
    const int* __restrict__,
    const int* __restrict__,
    const int* __restrict__,
    const int* __restrict__,
    bf16_t* __restrict__ arena_k,
    bf16_t* __restrict__ arena_v,
    _Float16* __restrict__ arena_bias,
    float* __restrict__,
    int* __restrict__ packed_indices,
    int* __restrict__ stream_counts,
    int* __restrict__,
    int* __restrict__ producer_done,
    int* __restrict__ tile_ready,
    int* __restrict__,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_denominator,
    bf16_t* __restrict__,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int,
    int state_len,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int index_capacity,
    int max_tiles,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int consumer_count,
    float scale,
    float) {
    const int sequence = static_cast<int>(blockIdx.x);
    const int consumer = static_cast<int>(blockIdx.y);
    const int logical_batch = sequence / kv_heads;
    const int kv_head = sequence - logical_batch * kv_heads;
    if (logical_batch >= batch || consumer >= consumer_count) {
        return;
    }
    int cache_batch = static_cast<int>(cache_indices[logical_batch]);
    const bool cache_valid = cache_batch >= 0 && cache_batch < cache_batches;
    cache_batch = cache_valid ? cache_batch : 0;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / traits::WARP_SIZE;
    const int lane_id = tid % traits::WARP_SIZE;
    const int producer_count = (state_len + kStatePartition - 1)
        / kStatePartition + 1;

    __shared__ SharedStorage shared;
    for (int element = tid;
         element < traits::Q_TILE_SIZE * traits::D_TILE_SIZE;
         element += traits::BLOCK_SIZE) {
        const int query = element / traits::D_TILE_SIZE;
        const int dimension = element - query * traits::D_TILE_SIZE;
        shared.q[element] = cache_valid && query < kGqaHeads
            ? q[(static_cast<long long>(logical_batch) * query_heads
                    + kv_head * kGqaHeads + query)
                    * traits::D_TILE_SIZE
                + dimension]
            : bf16_t(0.0f);
    }
    __syncthreads();

    auto mma_qk = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<traits::GEMM0_E_M, traits::GEMM0_E_N, traits::GEMM0_E_K>{},
        seq<traits::T_M, traits::T_N, traits::T_K>{},
        seq<traits::W_M, traits::W_N, traits::W_K>{},
        mfma_adaptor_swap_ab{});
    auto mma_pv = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<traits::GEMM1_E_M, traits::GEMM1_E_N, traits::GEMM1_E_K>{},
        seq<traits::T_M, traits::T_N, traits::T_K>{},
        seq<traits::W_M, traits::W_N, traits::W_K>{},
        mfma_adaptor_swap_ab{});
    using MmaQK = decltype(mma_qk);
    using MmaPV = decltype(mma_pv);
    auto q_fragment = load_query_fragment(mma_qk, shared, lane_id);
    using OutputFragment = typename decltype(mma_pv)::vtype_c;
    const float scale_log2 = scale * kLog2E;
    float maximum = -numeric_limits<float>::infinity();
    float denominator = 0.0f;
    OutputFragment output_fragment;
    clear(output_fragment);

    for (int tile = consumer; tile < max_tiles; tile += consumer_count) {
        if (tid == 0) {
            shared.is_last = 0;
        }
        __syncthreads();
        while (true) {
            if (tid == 0) {
                const int ready = atomicAdd(
                    tile_ready + sequence * max_tiles + tile, 0);
                const int done = atomicAdd(producer_done + sequence, 0);
                if (ready >= kLogicalTile || done >= producer_count) {
                    shared.is_last = done >= producer_count;
                    shared.total_length = min(
                        atomicAdd(stream_counts + sequence, 0), index_capacity);
                    shared.range_count = ready;
                } else {
                    shared.range_count = -1;
                }
            }
            __syncthreads();
            if (shared.range_count >= 0) {
                break;
            }
        }
        const int logical_begin = tile * kLogicalTile;
        if (shared.is_last && logical_begin >= shared.total_length) {
            break;
        }
        const int valid_tokens = shared.is_last
            ? min(kLogicalTile, shared.total_length - logical_begin)
            : kLogicalTile;
        attention_logical_tile_accumulate<MmaQK, MmaPV, OutputFragment>(
            mma_qk,
            mma_pv,
            q_fragment,
            shared,
            arena_k,
            arena_v,
            arena_bias,
            packed_indices,
            sequence,
            tile,
            valid_tokens,
            index_capacity,
            scale_log2,
            maximum,
            denominator,
            output_fragment,
            tid,
            warp_id,
            lane_id);
        if (tid == 0) {
            atomicExch(tile_ready + sequence * max_tiles + tile, 0);
        }
        __syncthreads();
    }
    store_attention_partial(
        mma_pv,
        output_fragment,
        maximum,
        denominator,
        partial_out,
        partial_max,
        partial_denominator,
        sequence,
        consumer,
        consumer_count,
        tid,
        warp_id,
        lane_id);
}

#else

__global__ __launch_bounds__(traits::BLOCK_SIZE) void unified_lod_decode_kernel(
    const bf16_t* __restrict__ q,
    const bf16_t* __restrict__ new_k,
    const bf16_t* __restrict__ new_v,
    const long long* __restrict__ cache_indices,
    const int* __restrict__ local_lens,
    const float* __restrict__ counts,
    const int* __restrict__ slot_pages,
    const int* __restrict__ directory_values,
    const int* __restrict__ slot_lengths,
    const int* __restrict__ page_indices,
    bf16_t* __restrict__ arena_k,
    bf16_t* __restrict__ arena_v,
    _Float16* __restrict__ arena_bias,
    float* __restrict__ previous_total_lse,
    int* __restrict__ packed_indices,
    int* __restrict__ stream_counts,
    int* __restrict__ opened_counts,
    int* __restrict__ producer_done,
    int* __restrict__ tile_ready,
    int* __restrict__ overflow_flags,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_denominator,
    bf16_t* __restrict__ output,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    int local_capacity,
    int local_limit,
    int sink_capacity,
    int sink_len,
    int leaf_capacity,
    int page_capacity,
    int directory_capacity,
    int root_capacity,
    int index_capacity,
    int max_tiles,
    int protected_len,
    int max_leaf_tokens,
    int open_capacity,
    int leaf_offset,
    int local_offset,
    int sink_offset,
    int coarse_offset,
    int execution_mode,
    float scale,
    float log_mass_fraction) {
    const int sequence = static_cast<int>(blockIdx.x);
    const int producer = static_cast<int>(blockIdx.y);
    const int logical_batch = sequence / kv_heads;
    const int kv_head = sequence - logical_batch * kv_heads;
    if (logical_batch >= batch) {
        return;
    }
    int cache_batch = static_cast<int>(cache_indices[logical_batch]);
    const bool cache_valid = cache_batch >= 0 && cache_batch < cache_batches;
    cache_batch = cache_valid ? cache_batch : 0;
    const int kv_row = cache_batch * kv_heads + kv_head;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / traits::WARP_SIZE;
    const int lane_id = tid % traits::WARP_SIZE;
    const int active_partitions =
        (state_len + kStatePartition - 1) / kStatePartition;
    const int producer_count = active_partitions + 1;
    if (producer >= producer_count) {
        return;
    }

    __shared__ SharedStorage shared;
    for (int element = tid;
         element < traits::Q_TILE_SIZE * traits::D_TILE_SIZE;
         element += traits::BLOCK_SIZE) {
        const int query = element / traits::D_TILE_SIZE;
        const int dimension = element - query * traits::D_TILE_SIZE;
        shared.q[element] = cache_valid && query < kGqaHeads
            ? q[(static_cast<long long>(logical_batch) * query_heads
                    + kv_head * kGqaHeads + query)
                    * traits::D_TILE_SIZE
                + dimension]
            : bf16_t(0.0f);
    }
    __syncthreads();

    auto mma_qk = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<traits::GEMM0_E_M, traits::GEMM0_E_N, traits::GEMM0_E_K>{},
        seq<traits::T_M, traits::T_N, traits::T_K>{},
        seq<traits::W_M, traits::W_N, traits::W_K>{},
        mfma_adaptor_swap_ab{});
    auto mma_pv = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<traits::GEMM1_E_M, traits::GEMM1_E_N, traits::GEMM1_E_K>{},
        seq<traits::T_M, traits::T_N, traits::T_K>{},
        seq<traits::W_M, traits::W_N, traits::W_K>{},
        mfma_adaptor_swap_ab{});
    using MmaQK = decltype(mma_qk);
    using MmaPV = decltype(mma_pv);
    auto q_fragment = load_query_fragment(mma_qk, shared, lane_id);
    using OutputFragment = typename decltype(mma_pv)::vtype_c;
    const float scale_log2 = scale * kLog2E;

    if (producer < active_partitions) {
        const int state_begin = producer * kStatePartition;
        float retained_lse[kGqaHeads];
#pragma unroll
        for (int query = 0; query < kGqaHeads; ++query) {
            retained_lse[query] = previous_total_lse[
                static_cast<long long>(cache_batch) * query_heads
                + kv_head * kGqaHeads + query];
        }
        route_partition(
            mma_qk,
            q_fragment,
            shared,
            arena_k,
            arena_bias,
            retained_lse,
            coarse_offset + kv_row * state_capacity,
            state_begin,
            state_len,
            scale_log2,
            log_mass_fraction * kLog2E,
            warp_id,
            lane_id);
        __syncthreads();

        const int slot = state_begin + tid;
        const bool thread_slot = tid < kStatePartition && slot < state_len;
        const float count = thread_slot
            ? counts[(static_cast<long long>(cache_batch) * kv_heads + kv_head)
                    * state_capacity
                + slot]
            : 0.0f;
        const int leaf_count = thread_slot
            ? slot_lengths[(static_cast<long long>(kv_row) * state_capacity) + slot]
            : 0;
        const bool eligible = thread_slot && slot >= protected_len
            && count > 0.0f
            && (max_leaf_tokens <= 0 || leaf_count < max_leaf_tokens);
        const bool selected = eligible && shared.selected[tid] != 0;
        const unsigned long long selected_ballot = __ballot(selected);
        const unsigned long long preceding_mask = lane_id == 0
            ? 0ULL
            : ((1ULL << lane_id) - 1ULL);
        const int selected_prefix_wave =
            __popcll(selected_ballot & preceding_mask);
        const int selected_wave_count = __popcll(selected_ballot);
        if (lane_id == 0) {
            shared.wave_totals[warp_id] = selected_wave_count;
        }
        __syncthreads();
        if (tid == 0) {
            int total = 0;
#pragma unroll
            for (int warp = 0; warp < traits::NUM_WARPS; ++warp) {
                shared.wave_bases[warp] = total;
                total += shared.wave_totals[warp];
            }
            shared.open_total = total;
            shared.open_base = atomicAdd(opened_counts + sequence, total);
        }
        __syncthreads();
        const int selected_rank =
            shared.wave_bases[warp_id] + selected_prefix_wave;
        const bool accepted = selected
            && shared.open_base + selected_rank < open_capacity;
        const int output_count = thread_slot
            ? (accepted ? leaf_count : 1)
            : 0;
        const int output_prefix = exclusive_scan_256(
            output_count, shared, tid, warp_id, lane_id);
        if (tid == 0) {
            shared.range_base = atomicAdd(
                stream_counts + sequence, shared.range_count);
            if (shared.range_base + shared.range_count > index_capacity) {
                atomicExch(overflow_flags + sequence, 1);
            }
        }
        __syncthreads();

        const int destination = shared.range_base + output_prefix;
        if (thread_slot && destination < index_capacity) {
            if (!accepted) {
                packed_indices[
                    static_cast<long long>(sequence) * index_capacity
                    + destination] = coarse_offset + kv_row * state_capacity + slot;
            } else {
                for (int leaf = 0; leaf < leaf_count; ++leaf) {
                    const int page_ordinal = leaf / 16;
                    const int within_page = leaf % 16;
                    const int directory_ordinal = page_ordinal / 64;
                    const int directory_offset = page_ordinal % 64;
                    int physical = -1;
                    if (directory_ordinal < root_capacity) {
                        const int directory_id = slot_pages[
                            (static_cast<long long>(kv_row) * state_capacity + slot)
                                * root_capacity
                            + directory_ordinal];
                        if (directory_id >= 0 && directory_id < directory_capacity) {
                            const int page_id = directory_values[
                                (static_cast<long long>(kv_row) * directory_capacity
                                    + directory_id)
                                    * 64
                                + directory_offset];
                            if (page_id >= 0 && page_id < page_capacity) {
                                const int leaf_index = page_indices[
                                    (static_cast<long long>(kv_row) * page_capacity
                                        + page_id)
                                        * 16
                                    + within_page];
                                if (leaf_index >= 0 && leaf_index < leaf_capacity) {
                                    physical = leaf_offset + kv_row * leaf_capacity
                                        + leaf_index;
                                }
                            }
                        }
                    }
                    if (destination + leaf < index_capacity) {
                        packed_indices[
                            static_cast<long long>(sequence) * index_capacity
                            + destination + leaf] = physical;
                    }
                }
            }
        }
    } else {
        const int active_local = min(local_lens[cache_batch], local_limit);
        const int local_and_new = active_local + 1;
        const int total = cache_valid ? local_and_new + sink_len : 0;
        if (tid < traits::D_TILE_SIZE && cache_valid) {
            const int local_physical =
                local_offset + kv_row * local_capacity + active_local;
            arena_k[static_cast<long long>(local_physical) * traits::D_TILE_SIZE + tid]
                = new_k[(static_cast<long long>(logical_batch) * kv_heads + kv_head)
                        * traits::D_TILE_SIZE
                    + tid];
            arena_v[static_cast<long long>(local_physical) * traits::D_TILE_SIZE + tid]
                = new_v[(static_cast<long long>(logical_batch) * kv_heads + kv_head)
                        * traits::D_TILE_SIZE
                    + tid];
            if (tid == 0) {
                arena_bias[local_physical] = _Float16(0.0f);
            }
        }
        __syncthreads();
        if (tid == 0) {
            shared.range_count = total;
            shared.range_base = atomicAdd(stream_counts + sequence, total);
            if (shared.range_base + total > index_capacity) {
                atomicExch(overflow_flags + sequence, 1);
            }
        }
        __syncthreads();
        for (int item = tid; item < total; item += traits::BLOCK_SIZE) {
            int physical;
            if (item < local_and_new) {
                physical = local_offset + kv_row * local_capacity + item;
            } else {
                physical = sink_offset + kv_row * sink_capacity
                    + item - local_and_new;
            }
            if (shared.range_base + item < index_capacity) {
                packed_indices[
                    static_cast<long long>(sequence) * index_capacity
                    + shared.range_base + item] = physical;
            }
        }
    }

    __threadfence();
    __syncthreads();
    publish_and_process_range<MmaQK, MmaPV, OutputFragment>(
        mma_qk,
        mma_pv,
        q_fragment,
        shared,
        arena_k,
        arena_v,
        arena_bias,
        packed_indices,
        tile_ready,
        partial_out,
        partial_max,
        partial_denominator,
        sequence,
        index_capacity,
        max_tiles,
        scale_log2,
        kExecuteAttention,
        tid,
        warp_id,
        lane_id);

    __threadfence();
    __syncthreads();
    if (tid == 0) {
            const int old = atomicAdd(producer_done + sequence, 1);
        shared.is_last = old + 1 == producer_count;
        if (shared.is_last) {
            shared.total_length = min(stream_counts[sequence], index_capacity);
        }
    }
    __syncthreads();
    if (!shared.is_last) {
        return;
    }

    if constexpr (!kExecuteAttention) {
        // producer_done == producer_count is the completion signal for the
        // external consumers.  stream_counts contains the final tail length,
        // and tile_ready contains either 256 (complete) or the published tail
        // contribution.  The consumer/reducer owns all counter resets.
        return;
    }

    const int remainder = shared.total_length % kLogicalTile;
    if (remainder != 0 && overflow_flags[sequence] == 0) {
        const int tail_tile = shared.total_length / kLogicalTile;
        attention_logical_tile<MmaQK, MmaPV, OutputFragment>(
            mma_qk,
            mma_pv,
            q_fragment,
            shared,
            arena_k,
            arena_v,
            arena_bias,
            packed_indices,
            partial_out,
            partial_max,
            partial_denominator,
            sequence,
            tail_tile,
            remainder,
            index_capacity,
            max_tiles,
            scale_log2,
            tid,
            warp_id,
            lane_id);
        if (tid == 0) {
            tile_ready[sequence * max_tiles + tail_tile] = 0;
        }
        __syncthreads();
    }

    const int tile_count =
        (shared.total_length + kLogicalTile - 1) / kLogicalTile;
    if (tid < kGqaHeads) {
        float maximum = -numeric_limits<float>::infinity();
        for (int tile = 0; tile < tile_count; ++tile) {
            maximum = max(
                maximum,
                partial_max[
                    (static_cast<long long>(sequence) * max_tiles + tile)
                        * traits::Q_TILE_SIZE
                    + tid]);
        }
        float denominator = 0.0f;
        for (int tile = 0; tile < tile_count; ++tile) {
            const long long scalar =
                (static_cast<long long>(sequence) * max_tiles + tile)
                    * traits::Q_TILE_SIZE
                + tid;
            denominator += partial_denominator[scalar]
                * __builtin_amdgcn_exp2f(partial_max[scalar] - maximum);
        }
        shared.row_workspace[tid] = maximum;
        shared.row_workspace[kGqaHeads + tid] = denominator;
        previous_total_lse[
            static_cast<long long>(cache_batch) * query_heads
            + kv_head * kGqaHeads + tid]
            = (maximum + __builtin_amdgcn_logf(denominator)) * kLn2;
    }
    __syncthreads();
    for (int element = tid;
         element < kGqaHeads * traits::D_TILE_SIZE;
         element += traits::BLOCK_SIZE) {
        const int query = element / traits::D_TILE_SIZE;
        const int dimension = element - query * traits::D_TILE_SIZE;
        const float maximum = shared.row_workspace[query];
        const float denominator = shared.row_workspace[kGqaHeads + query];
        float numerator = 0.0f;
        for (int tile = 0; tile < tile_count; ++tile) {
            const long long scalar =
                (static_cast<long long>(sequence) * max_tiles + tile)
                    * traits::Q_TILE_SIZE
                + query;
            const long long vector = scalar * traits::D_TILE_SIZE + dimension;
            numerator += partial_out[vector]
                * __builtin_amdgcn_exp2f(partial_max[scalar] - maximum);
        }
        output[
            (static_cast<long long>(logical_batch) * query_heads
                + kv_head * kGqaHeads + query)
                * traits::D_TILE_SIZE
            + dimension] = finite_fp32_to_bf16(numerator / denominator);
    }
    __syncthreads();
    if (tid == 0) {
        stream_counts[sequence] = 0;
        opened_counts[sequence] = 0;
        producer_done[sequence] = 0;
        overflow_flags[sequence] = 0;
    }
}

#endif // LOD_CONSUMER_ONLY

} // namespace

#endif // gfx942

#else // host pass

#include "opus/opus.hpp"
#include "opus/hip_minimal.hpp"

namespace {

__global__ void unified_lod_decode_kernel(
    const opus::bf16_t*, const opus::bf16_t*, const opus::bf16_t*,
    const long long*, const int*, const float*, const int*, const int*,
    const int*, const int*, opus::bf16_t*, opus::bf16_t*, _Float16*, float*,
    int*, int*, int*, int*, int*, int*, float*, float*, float*, opus::bf16_t*,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, float, float) {}

} // namespace

#endif

extern "C" int launch_unified_lod_decode(
    const void* q,
    const void* new_k,
    const void* new_v,
    const void* cache_indices,
    const void* local_lens,
    const void* counts,
    const void* slot_pages,
    const void* directory_values,
    const void* slot_lengths,
    const void* page_indices,
    void* arena_k,
    void* arena_v,
    void* arena_bias,
    void* previous_total_lse,
    void* packed_indices,
    void* stream_counts,
    void* opened_counts,
    void* producer_done,
    void* tile_ready,
    void* overflow_flags,
    void* partial_out,
    void* partial_max,
    void* partial_denominator,
    void* output,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    int local_capacity,
    int local_limit,
    int sink_capacity,
    int sink_len,
    int leaf_capacity,
    int page_capacity,
    int directory_capacity,
    int root_capacity,
    int index_capacity,
    int max_tiles,
    int protected_len,
    int max_leaf_tokens,
    int open_capacity,
    int leaf_offset,
    int local_offset,
    int sink_offset,
    int coarse_offset,
    int execution_mode,
    float scale,
    float log_mass_fraction,
    void* stream) {
    if (query_heads != kv_heads * 4 || state_len < 0
        || state_len > state_capacity || index_capacity < 1 || max_tiles < 1) {
        return 1;
    }
    const int sequences = batch * kv_heads;
#if defined(LOD_CONSUMER_ONLY)
    const int producers = execution_mode;
#else
    const int producers = (state_len + 255) / 256 + 1;
#endif
    hipLaunchKernelGGL(
        unified_lod_decode_kernel,
        dim3(static_cast<unsigned int>(sequences), static_cast<unsigned int>(producers)),
        dim3(256),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const opus::bf16_t*>(q),
        static_cast<const opus::bf16_t*>(new_k),
        static_cast<const opus::bf16_t*>(new_v),
        static_cast<const long long*>(cache_indices),
        static_cast<const int*>(local_lens),
        static_cast<const float*>(counts),
        static_cast<const int*>(slot_pages),
        static_cast<const int*>(directory_values),
        static_cast<const int*>(slot_lengths),
        static_cast<const int*>(page_indices),
        static_cast<opus::bf16_t*>(arena_k),
        static_cast<opus::bf16_t*>(arena_v),
        static_cast<_Float16*>(arena_bias),
        static_cast<float*>(previous_total_lse),
        static_cast<int*>(packed_indices),
        static_cast<int*>(stream_counts),
        static_cast<int*>(opened_counts),
        static_cast<int*>(producer_done),
        static_cast<int*>(tile_ready),
        static_cast<int*>(overflow_flags),
        static_cast<float*>(partial_out),
        static_cast<float*>(partial_max),
        static_cast<float*>(partial_denominator),
        static_cast<opus::bf16_t*>(output),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        state_len,
        local_capacity,
        local_limit,
        sink_capacity,
        sink_len,
        leaf_capacity,
        page_capacity,
        directory_capacity,
        root_capacity,
        index_capacity,
        max_tiles,
        protected_len,
        max_leaf_tokens,
        open_capacity,
        leaf_offset,
        local_offset,
        sink_offset,
        coarse_offset,
        execution_mode,
        scale,
        log_mass_fraction);
    return static_cast<int>(hipGetLastError());
}

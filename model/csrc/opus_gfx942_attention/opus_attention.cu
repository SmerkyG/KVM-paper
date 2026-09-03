// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// Adapted from ROCm/AITER pa_sparse_prefill_opus.h.
// Restricted gfx942 performance probe: BF16, M=16, K<=64, D=128.

#ifdef __HIP_DEVICE_COMPILE__

#include "opus/opus.hpp"

#if defined(__gfx942__) || defined(__gfx9_4_generic__)

using namespace opus;

struct attention_traits {
    using D_ATTN = bf16_t;
    using D_ACC = float;

    static constexpr int Q_TILE_SIZE = 16;
    static constexpr int KV_TILE_SIZE = 64;
    static constexpr int D_TILE_SIZE = 128;
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

__device__ inline bf16_t finite_fp32_to_bf16(float value)
{
    u32_t bits = __builtin_bit_cast(u32_t, value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __builtin_bit_cast(
        bf16_t, static_cast<unsigned short>(bits >> 16));
}

template<typename V>
__device__ inline auto finite_cast_bf16(const V& values)
{
    constexpr index_t count = vector_traits<V>::size();
    vector_t<bf16_t, count> result;
    static_for<count>([&](auto i) {
        result[i.value] = finite_fp32_to_bf16(values[i.value]);
    });
    return result;
}

template<typename V>
__device__ inline float attention_row_max(
    const V& scores,
    float* shared_max,
    int warp_id,
    int lane_id)
{
    constexpr index_t score_count = vector_traits<V>::size();
    float row_max = -numeric_limits<float>::infinity();
    static_for<score_count>([&](auto i) {
        row_max = max(row_max, scores[i.value]);
    });

    row_max = max(row_max, shfl(row_max, lane_id ^ 32));
    row_max = max(row_max, shfl(row_max, lane_id ^ 16));

    const int row = lane_id % attention_traits::W_M;
    shared_max[row * attention_traits::T_N + warp_id] = row_max;
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    auto smem = make_smem(shared_max);
    auto warp_max = load<attention_traits::T_N>(
        smem, row * attention_traits::T_N);
    static_for<attention_traits::T_N>([&](auto i) {
        row_max = max(row_max, warp_max[i.value]);
    });
    return row_max;
}

template<typename V>
__device__ inline float attention_row_sum(
    const V& probabilities,
    float* shared_sum,
    int warp_id,
    int lane_id)
{
    constexpr index_t score_count = vector_traits<V>::size();
    float row_sum = 0.0f;
    static_for<score_count>([&](auto i) {
        row_sum += probabilities[i.value];
    });

    row_sum += shfl(row_sum, lane_id ^ 32);
    row_sum += shfl(row_sum, lane_id ^ 16);

    const int row = lane_id % attention_traits::W_M;
    shared_sum[row * attention_traits::T_N + warp_id] = row_sum;
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    auto smem = make_smem(shared_sum);
    auto warp_sum = load<attention_traits::T_N>(
        smem, row * attention_traits::T_N);
    row_sum = 0.0f;
    static_for<attention_traits::T_N>([&](auto i) {
        row_sum += warp_sum[i.value];
    });
    return row_sum;
}

template<typename V>
__device__ inline void mask_scores(V& scores, int valid_k, int warp_id, int lane_id)
{
    using T = attention_traits;
    constexpr int elems_per_wave_tile = T::W_M * T::W_N / T::WARP_SIZE;
    constexpr int pack = 4;
    constexpr int repeats = elems_per_wave_tile / pack;
    constexpr int repeat_stride = (T::WARP_SIZE / T::W_M) * pack;
    const int last_valid = valid_k - 1;
    const int key_start = warp_id * T::GEMM0_E_N * T::W_N;
    const int lane_group = lane_id / T::W_M;
    const float negative_infinity = -numeric_limits<float>::infinity();

    static_for<T::GEMM0_E_N>([&](auto n) {
        constexpr int base = n.value * elems_per_wave_tile;
        const int key = key_start + n.value * T::W_N + lane_group * pack;
        const int remaining = last_valid - key;
        static_for<repeats>([&](auto repeat) {
            constexpr int repeat_base = base + repeat.value * pack;
            constexpr int threshold_base = repeat.value * repeat_stride;
            static_for<pack>([&](auto element) {
                constexpr int index = repeat_base + element.value;
                constexpr int threshold = threshold_base + element.value;
                scores[index] = remaining < threshold
                    ? negative_infinity
                    : scores[index];
            });
        });
    });
}

__global__ void opus_gfx942_qk_debug_kernel(
    const bf16_t* __restrict__ q,
    const bf16_t* __restrict__ k,
    float* __restrict__ output)
{
    using T = attention_traits;
    const int expert = block_id_x();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(
        thread_id_x() / T::WARP_SIZE);
    auto mma_qk = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    const auto q_coord = make_tuple(
        0_I,
        lane_id % mma_qk.grpm_a,
        0_I,
        lane_id / mma_qk.grpm_a);
    const auto k_coord = make_tuple(
        warp_id,
        lane_id % mma_qk.grpn_b,
        0_I,
        lane_id / mma_qk.grpn_b);
    auto q_layout = partition_layout_a<8>(
        mma_qk, make_tuple(T::D_TILE_SIZE, 1_I), q_coord);
    auto k_layout = partition_layout_b<8>(
        mma_qk, make_tuple(T::D_TILE_SIZE, 1_I), k_coord);
    auto global_q = make_gmem(
        q + expert * T::Q_TILE_SIZE * T::D_TILE_SIZE,
        T::Q_TILE_SIZE * T::D_TILE_SIZE * sizeof(bf16_t));
    auto global_k = make_gmem(
        k + expert * T::KV_TILE_SIZE * T::D_TILE_SIZE,
        T::KV_TILE_SIZE * T::D_TILE_SIZE * sizeof(bf16_t));
    auto scores = mma_qk(load<8>(global_q, q_layout), load<8>(global_k, k_layout));
    const auto output_coord = make_tuple(
        0_I,
        lane_id % mma_qk.grpn_c,
        warp_id,
        lane_id / mma_qk.grpn_c);
    auto output_layout = partition_layout_c(
        mma_qk, make_tuple(T::KV_TILE_SIZE, 1_I), output_coord);
    auto global_output = make_gmem(
        output + expert * T::Q_TILE_SIZE * T::KV_TILE_SIZE,
        T::Q_TILE_SIZE * T::KV_TILE_SIZE * sizeof(float));
    store<4>(global_output, scores, output_layout);
}

__global__ void opus_gfx942_pv_debug_kernel(
    const bf16_t* __restrict__ probability,
    const bf16_t* __restrict__ v,
    float* __restrict__ output)
{
    using T = attention_traits;
    const int expert = block_id_x();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(
        thread_id_x() / T::WARP_SIZE);
    auto mma_pv = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    const auto probability_coord = make_tuple(
        0_I,
        lane_id % mma_pv.grpm_a,
        0_I,
        lane_id / mma_pv.grpm_a);
    auto probability_layout = partition_layout_a<8>(
        mma_pv, make_tuple(T::KV_TILE_SIZE, 1_I), probability_coord);
    auto global_probability = make_gmem(
        probability + expert * T::Q_TILE_SIZE * T::KV_TILE_SIZE,
        T::Q_TILE_SIZE * T::KV_TILE_SIZE * sizeof(bf16_t));
    auto probability_fragment = load<8>(global_probability, probability_layout);
    const auto v_coord = make_tuple(
        warp_id,
        lane_id % mma_pv.grpn_b,
        0_I,
        lane_id / mma_pv.grpn_b);
    auto v_layout = partition_layout_b<1>(
        mma_pv, make_tuple(1_I, T::D_TILE_SIZE), v_coord);
    auto global_v = make_gmem(
        v + expert * T::KV_TILE_SIZE * T::D_TILE_SIZE,
        T::KV_TILE_SIZE * T::D_TILE_SIZE * sizeof(bf16_t));
    auto v_fragment = load<1>(global_v, v_layout);
    auto output_fragment = mma_pv(probability_fragment, v_fragment);
    const auto output_coord = make_tuple(
        0_I,
        lane_id % mma_pv.grpn_c,
        warp_id,
        lane_id / mma_pv.grpn_c);
    auto output_layout = partition_layout_c(
        mma_pv, make_tuple(T::D_TILE_SIZE, 1_I), output_coord);
    auto global_output = make_gmem(
        output + expert * T::Q_TILE_SIZE * T::D_TILE_SIZE,
        T::Q_TILE_SIZE * T::D_TILE_SIZE * sizeof(float));
    store<4>(global_output, output_fragment, output_layout);
}

__global__ void opus_gfx942_softmax_debug_kernel(
    const bf16_t* __restrict__ q,
    const bf16_t* __restrict__ k,
    const int* __restrict__ lengths,
    float* __restrict__ output,
    float scale)
{
    using T = attention_traits;
    const int expert = block_id_x();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(
        thread_id_x() / T::WARP_SIZE);
    __shared__ float shared_max[T::W_M * T::T_N];
    __shared__ float shared_sum[T::W_M * T::T_N];
    auto mma_qk = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    const auto q_coord = make_tuple(
        0_I, lane_id % mma_qk.grpm_a, 0_I, lane_id / mma_qk.grpm_a);
    const auto k_coord = make_tuple(
        warp_id, lane_id % mma_qk.grpn_b, 0_I, lane_id / mma_qk.grpn_b);
    auto q_layout = partition_layout_a<8>(
        mma_qk, make_tuple(T::D_TILE_SIZE, 1_I), q_coord);
    auto k_layout = partition_layout_b<8>(
        mma_qk, make_tuple(T::D_TILE_SIZE, 1_I), k_coord);
    auto global_q = make_gmem(
        q + expert * T::Q_TILE_SIZE * T::D_TILE_SIZE,
        T::Q_TILE_SIZE * T::D_TILE_SIZE * sizeof(bf16_t));
    auto global_k = make_gmem(
        k + expert * T::KV_TILE_SIZE * T::D_TILE_SIZE,
        T::KV_TILE_SIZE * T::D_TILE_SIZE * sizeof(bf16_t));
    auto scores = mma_qk(load<8>(global_q, q_layout), load<8>(global_k, k_layout));
    constexpr index_t score_count = vector_traits<decltype(scores)>::size();
    constexpr float LOG2_E = 1.4426950408889634f;
    static_for<score_count>([&](auto i) { scores[i.value] *= scale * LOG2_E; });
    mask_scores(scores, lengths[expert], warp_id, lane_id);
    const float row_max = attention_row_max(
        scores, shared_max, warp_id, lane_id);
    static_for<score_count>([&](auto i) {
        scores[i.value] = __builtin_amdgcn_exp2f(scores[i.value] - row_max);
    });
    const float denominator = attention_row_sum(
        scores, shared_sum, warp_id, lane_id);
    static_for<score_count>([&](auto i) { scores[i.value] /= denominator; });
    const auto output_coord = make_tuple(
        0_I, lane_id % mma_qk.grpn_c, warp_id, lane_id / mma_qk.grpn_c);
    auto output_layout = partition_layout_c(
        mma_qk, make_tuple(T::KV_TILE_SIZE, 1_I), output_coord);
    auto global_output = make_gmem(
        output + expert * T::Q_TILE_SIZE * T::KV_TILE_SIZE,
        T::Q_TILE_SIZE * T::KV_TILE_SIZE * sizeof(float));
    store<4>(global_output, scores, output_layout);
}

__global__ void opus_gfx942_attention_kernel(
    const bf16_t* __restrict__ q,
    const bf16_t* __restrict__ k,
    const bf16_t* __restrict__ v,
    const int* __restrict__ lengths,
    bf16_t* __restrict__ output,
    float scale)
{
    using T = attention_traits;
    const int expert = block_id_x();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(
        thread_id_x() / T::WARP_SIZE);
    const int valid_k = lengths[expert];

    __shared__ float shared_max[T::W_M * T::T_N];
    __shared__ float shared_sum[T::W_M * T::T_N];
    __shared__ bf16_t shared_probabilities[
        T::T_N * T::W_M * T::W_N];

    auto mma_qk = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    auto mma_pv = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    const auto q_coord = make_tuple(
        0_I,
        lane_id % mma_qk.grpm_a,
        0_I,
        lane_id / mma_qk.grpm_a);
    const auto k_coord = make_tuple(
        warp_id,
        lane_id % mma_qk.grpn_b,
        0_I,
        lane_id / mma_qk.grpn_b);
    auto q_layout = partition_layout_a<8>(
        mma_qk, make_tuple(T::D_TILE_SIZE, 1_I), q_coord);
    auto k_layout = partition_layout_b<8>(
        mma_qk, make_tuple(T::D_TILE_SIZE, 1_I), k_coord);

    auto global_q = make_gmem(
        q + expert * T::Q_TILE_SIZE * T::D_TILE_SIZE,
        T::Q_TILE_SIZE * T::D_TILE_SIZE * sizeof(bf16_t));
    auto global_k = make_gmem(
        k + expert * T::KV_TILE_SIZE * T::D_TILE_SIZE,
        T::KV_TILE_SIZE * T::D_TILE_SIZE * sizeof(bf16_t));
    auto q_fragment = load<8>(global_q, q_layout);
    auto k_fragment = load<8>(global_k, k_layout);
    auto scores = mma_qk(q_fragment, k_fragment);

    constexpr index_t score_count = vector_traits<decltype(scores)>::size();
    constexpr float LOG2_E = 1.4426950408889634f;
    const float score_scale = scale * LOG2_E;
    static_for<score_count>([&](auto i) {
        scores[i.value] *= score_scale;
    });
    mask_scores(scores, valid_k, warp_id, lane_id);

    const float row_max = attention_row_max(
        scores, shared_max, warp_id, lane_id);
    static_for<score_count>([&](auto i) {
        scores[i.value] = __builtin_amdgcn_exp2f(scores[i.value] - row_max);
    });
    const float denominator = attention_row_sum(
        scores, shared_sum, warp_id, lane_id);

    auto probability_fragment = finite_cast_bf16(scores);
    auto probability_smem = make_smem(shared_probabilities);
    const auto probability_store_coord = make_tuple(
        0_I,
        lane_id % mma_qk.grpn_c,
        warp_id,
        lane_id / mma_qk.grpn_c);
    auto probability_store_layout = partition_layout_c(
        mma_qk,
        make_tuple(T::KV_TILE_SIZE, 1_I),
        probability_store_coord);
    store<4>(
        probability_smem, probability_fragment, probability_store_layout);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    const auto probability_load_coord = make_tuple(
        0_I,
        lane_id % mma_pv.grpm_a,
        0_I,
        lane_id / mma_pv.grpm_a);
    auto probability_load_layout = partition_layout_a<8>(
        mma_pv,
        make_tuple(T::KV_TILE_SIZE, 1_I),
        probability_load_coord);
    auto all_probabilities = load<8>(
        probability_smem, probability_load_layout);

    // mma_pv computes P @ V. Its B operand is logically V^T [D, K],
    // represented as a strided view over row-major V [K, D]. Scalar packing
    // avoids pretending the K dimension is contiguous on gfx942.
    const auto v_coord = make_tuple(
        warp_id,
        lane_id % mma_pv.grpn_b,
        0_I,
        lane_id / mma_pv.grpn_b);
    auto v_layout = partition_layout_b<1>(
        mma_pv, make_tuple(1_I, T::D_TILE_SIZE), v_coord);
    auto global_v = make_gmem(
        v + expert * T::KV_TILE_SIZE * T::D_TILE_SIZE,
        T::KV_TILE_SIZE * T::D_TILE_SIZE * sizeof(bf16_t));
    typename decltype(mma_pv)::vtype_b v_fragment = load<1>(
        global_v, v_layout);

    typename decltype(mma_pv)::vtype_c output_fragment;
    clear(output_fragment);
    output_fragment = mma_pv(
        all_probabilities, v_fragment, output_fragment);
    const float inverse_denominator = 1.0f / denominator;
    static_for<vector_traits<decltype(output_fragment)>::size()>([&](auto i) {
        output_fragment[i.value] *= inverse_denominator;
    });

    const auto output_coord = make_tuple(
        0_I,
        lane_id % mma_pv.grpn_c,
        warp_id,
        lane_id / mma_pv.grpn_c);
    auto output_layout = partition_layout_c(
        mma_pv, make_tuple(T::D_TILE_SIZE, 1_I), output_coord);
    auto global_output = make_gmem(
        output + expert * T::Q_TILE_SIZE * T::D_TILE_SIZE,
        T::Q_TILE_SIZE * T::D_TILE_SIZE * sizeof(bf16_t));
    auto output_bf16 = finite_cast_bf16(output_fragment);
    store<4>(global_output, output_bf16, output_layout);
}

// Short-bucket LOD probe. One workgroup handles one routed expert block with
// up to 16 queries and 64 keys held in four independent 16-token pages.
// ROUTE_COUNT is fixed to the production value so integer division folds to a
// multiply/shift sequence rather than becoming a dynamic divide.
__global__ void opus_gfx942_paged_attention_kernel(
    const bf16_t* __restrict__ q,
    const long long* __restrict__ packed_route_row,
    const int* __restrict__ block_expert,
    const int* __restrict__ block_starts,
    const bf16_t* __restrict__ page_k,
    const bf16_t* __restrict__ page_v,
    const int* __restrict__ slot_pages,
    const int* __restrict__ slot_lengths,
    const int* __restrict__ q_lengths,
    const int* __restrict__ cu_q,
    const long long* __restrict__ expert_kv_row,
    const long long* __restrict__ expert_slot,
    bf16_t* __restrict__ output,
    float* __restrict__ lse,
    int page_capacity,
    int state_capacity,
    int inline_pages,
    float scale)
{
    using T = attention_traits;
    constexpr int ROUTE_COUNT = 3;
    constexpr float LOG2_E = 1.4426950408889634f;
    constexpr float LN_2 = 0.6931471805599453f;

    const int program = block_id_x();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(
        thread_id_x() / T::WARP_SIZE);
    auto global_block_expert = make_gmem(block_expert);
    const int expert = load(global_block_expert, program)[0];
    auto global_block_starts = make_gmem(block_starts);
    const int query_block = program - load(global_block_starts, expert)[0];
    auto global_q_lengths = make_gmem(q_lengths);
    const int query_count = load(global_q_lengths, expert)[0];
    const int row = lane_id % T::W_M;
    const int query_offset = query_block * T::Q_TILE_SIZE + row;
    const bool valid_query = query_offset < query_count;
    auto global_cu_q = make_gmem(cu_q);
    const int packed_begin = load(global_cu_q, expert)[0];
    auto global_packed_route_row = make_gmem(packed_route_row);
    const long long route_row = valid_query
        ? load(global_packed_route_row, packed_begin + query_offset)[0]
        : 0;
    const long long query_row = route_row / ROUTE_COUNT;

    auto global_expert_kv_row = make_gmem(expert_kv_row);
    auto global_expert_slot = make_gmem(expert_slot);
    const long long kv_row = load(global_expert_kv_row, expert)[0];
    const long long slot = load(global_expert_slot, expert)[0];
    auto global_slot_lengths = make_gmem(slot_lengths);
    const long long slot_index = kv_row * state_capacity + slot;
    const int valid_k = load(global_slot_lengths, slot_index)[0];
    auto global_slot_pages = make_gmem(slot_pages);
    auto page_ids = load<4>(
        global_slot_pages, slot_index * inline_pages);

    __shared__ float shared_max[T::W_M * T::T_N];
    __shared__ float shared_sum[T::W_M * T::T_N];
    __shared__ bf16_t shared_probabilities[
        T::T_N * T::W_M * T::W_N];

    auto mma_qk = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    auto mma_pv = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    const auto q_coord = make_tuple(
        0_I,
        lane_id % mma_qk.grpm_a,
        0_I,
        lane_id / mma_qk.grpm_a);
    auto q_layout = partition_layout_a<8>(
        mma_qk, make_tuple(0_I, 1_I), q_coord);
    auto global_q = make_gmem(q);
    auto q_fragment = load<8>(
        global_q, q_layout + query_row * T::D_TILE_SIZE);
    if (!valid_query) {
        clear(q_fragment);
    }

    const bool valid_page = warp_id * T::W_N < valid_k;
    const int k_page_id = valid_page ? page_ids[warp_id] : 0;
    const long long k_page_offset =
        (kv_row * page_capacity + k_page_id)
        * T::W_N * T::D_TILE_SIZE;
    const auto k_coord = make_tuple(
        0_I,
        lane_id % mma_qk.grpn_b,
        0_I,
        lane_id / mma_qk.grpn_b);
    auto k_layout = partition_layout_b<8>(
        mma_qk, make_tuple(T::D_TILE_SIZE, 1_I), k_coord);
    auto global_k = make_gmem(page_k);
    auto k_fragment = load<8>(global_k, k_layout + k_page_offset);
    if (!valid_page) {
        clear(k_fragment);
    }
    auto scores = mma_qk(q_fragment, k_fragment);
    constexpr index_t score_count = vector_traits<decltype(scores)>::size();
    static_for<score_count>([&](auto i) {
        scores[i.value] *= scale * LOG2_E;
    });
    mask_scores(scores, valid_k, warp_id, lane_id);

    const float row_max = attention_row_max(
        scores, shared_max, warp_id, lane_id);
    static_for<score_count>([&](auto i) {
        scores[i.value] = __builtin_amdgcn_exp2f(scores[i.value] - row_max);
    });
    const float denominator = attention_row_sum(
        scores, shared_sum, warp_id, lane_id);

    auto probability_fragment = finite_cast_bf16(scores);
    auto probability_smem = make_smem(shared_probabilities);
    const auto probability_store_coord = make_tuple(
        0_I,
        lane_id % mma_qk.grpn_c,
        warp_id,
        lane_id / mma_qk.grpn_c);
    auto probability_store_layout = partition_layout_c(
        mma_qk,
        make_tuple(T::KV_TILE_SIZE, 1_I),
        probability_store_coord);
    store<4>(
        probability_smem, probability_fragment, probability_store_layout);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    const auto probability_load_coord = make_tuple(
        0_I,
        lane_id % mma_pv.grpm_a,
        0_I,
        lane_id / mma_pv.grpm_a);
    auto probability_load_layout = partition_layout_a<8>(
        mma_pv,
        make_tuple(T::KV_TILE_SIZE, 1_I),
        probability_load_coord);
    auto all_probabilities = load<8>(
        probability_smem, probability_load_layout);

    // The B operand is logical V^T [D, K]. gfx942 lacks gfx950's
    // ds_read_b64_tr instruction, so gather its scalar lanes directly from the
    // four pages. Adjacent lanes still form coalesced 32-byte transactions.
    auto global_v = make_gmem(page_v);
    typename decltype(mma_pv)::vtype_b v_fragment;
    const int key_group = lane_id / T::W_N;
    const int page_half = key_group / 2;
    const int within_page_base = (key_group % 2) * 8;
    const int first_page_ordinal = page_half;
    const int second_page_ordinal = 2 + page_half;
    const int first_page_id = first_page_ordinal * T::W_N < valid_k
        ? page_ids[first_page_ordinal]
        : 0;
    const int second_page_id = second_page_ordinal * T::W_N < valid_k
        ? page_ids[second_page_ordinal]
        : 0;
    static_ford<T::GEMM1_E_N, T::GEMM1_E_K>([&](auto i_n, auto i_k) {
        const int dimension =
            (i_n.value * T::T_N + warp_id) * T::W_N
            + lane_id % T::W_N;
        const int page_id = i_k.value == 0 ? first_page_id : second_page_id;
        const long long page_base =
            (kv_row * page_capacity + page_id)
            * T::W_N * T::D_TILE_SIZE;
        static_for<8>([&](auto element) {
            constexpr int fragment_offset =
                (i_n.value * T::GEMM1_E_K + i_k.value) * 8
                + element.value;
            const int within_page = within_page_base + element.value;
            v_fragment[fragment_offset] = load(
                global_v,
                page_base + within_page * T::D_TILE_SIZE + dimension)[0];
        });
    });

    auto output_fragment = mma_pv(all_probabilities, v_fragment);
    const float inverse_denominator = 1.0f / denominator;
    static_for<vector_traits<decltype(output_fragment)>::size()>([&](auto i) {
        output_fragment[i.value] *= inverse_denominator;
    });
    const auto output_coord = make_tuple(
        0_I,
        lane_id % mma_pv.grpn_c,
        warp_id,
        lane_id / mma_pv.grpn_c);
    auto output_layout = partition_layout_c(
        mma_pv, make_tuple(0_I, 1_I), output_coord);
    output_layout += route_row * T::D_TILE_SIZE;
    auto global_output = make_gmem(output);
    auto output_bf16 = finite_cast_bf16(output_fragment);
    store_if<4>(
        global_output,
        [&](auto...) { return valid_query; },
        output_bf16,
        output_layout);
    if (lane_id < T::W_M && valid_query) {
        auto global_lse = make_gmem(lse);
        const float natural_lse =
            (row_max + __builtin_amdgcn_logf(denominator)) * LN_2;
        store(global_lse, natural_lse, route_row);
    }
}

#endif  // gfx942

#else  // host pass

#include "opus/opus.hpp"
#include "opus/hip_minimal.hpp"

__global__ void opus_gfx942_attention_kernel(
    const opus::bf16_t*,
    const opus::bf16_t*,
    const opus::bf16_t*,
    const int*,
    opus::bf16_t*,
    float)
{}

__global__ void opus_gfx942_paged_attention_kernel(
    const opus::bf16_t*,
    const long long*,
    const int*,
    const int*,
    const opus::bf16_t*,
    const opus::bf16_t*,
    const int*,
    const int*,
    const int*,
    const int*,
    const long long*,
    const long long*,
    opus::bf16_t*,
    float*,
    int,
    int,
    int,
    float)
{}

__global__ void opus_gfx942_qk_debug_kernel(
    const opus::bf16_t*,
    const opus::bf16_t*,
    float*)
{}

__global__ void opus_gfx942_pv_debug_kernel(
    const opus::bf16_t*,
    const opus::bf16_t*,
    float*)
{}

__global__ void opus_gfx942_softmax_debug_kernel(
    const opus::bf16_t*,
    const opus::bf16_t*,
    const int*,
    float*,
    float)
{}

extern "C" int launch_opus_gfx942_attention(
    const void* q,
    const void* k,
    const void* v,
    const void* lengths,
    void* output,
    int experts,
    float scale,
    void* stream)
{
    hipLaunchKernelGGL(
        opus_gfx942_attention_kernel,
        dim3(static_cast<unsigned int>(experts)),
        dim3(256),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const opus::bf16_t*>(q),
        static_cast<const opus::bf16_t*>(k),
        static_cast<const opus::bf16_t*>(v),
        static_cast<const int*>(lengths),
        static_cast<opus::bf16_t*>(output),
        scale);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_opus_gfx942_qk_debug(
    const void* q,
    const void* k,
    void* output,
    int experts,
    void* stream)
{
    hipLaunchKernelGGL(
        opus_gfx942_qk_debug_kernel,
        dim3(static_cast<unsigned int>(experts)),
        dim3(256),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const opus::bf16_t*>(q),
        static_cast<const opus::bf16_t*>(k),
        static_cast<float*>(output));
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_opus_gfx942_pv_debug(
    const void* probability,
    const void* v,
    void* output,
    int experts,
    void* stream)
{
    hipLaunchKernelGGL(
        opus_gfx942_pv_debug_kernel,
        dim3(static_cast<unsigned int>(experts)),
        dim3(256),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const opus::bf16_t*>(probability),
        static_cast<const opus::bf16_t*>(v),
        static_cast<float*>(output));
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_opus_gfx942_softmax_debug(
    const void* q,
    const void* k,
    const void* lengths,
    void* output,
    int experts,
    float scale,
    void* stream)
{
    hipLaunchKernelGGL(
        opus_gfx942_softmax_debug_kernel,
        dim3(static_cast<unsigned int>(experts)),
        dim3(256),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const opus::bf16_t*>(q),
        static_cast<const opus::bf16_t*>(k),
        static_cast<const int*>(lengths),
        static_cast<float*>(output),
        scale);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_opus_gfx942_paged_attention(
    const void* q,
    const void* packed_route_row,
    const void* block_expert,
    const void* block_starts,
    const void* page_k,
    const void* page_v,
    const void* slot_pages,
    const void* slot_lengths,
    const void* q_lengths,
    const void* cu_q,
    const void* expert_kv_row,
    const void* expert_slot,
    void* output,
    void* lse,
    int programs,
    int page_capacity,
    int state_capacity,
    int inline_pages,
    float scale,
    void* stream)
{
    hipLaunchKernelGGL(
        opus_gfx942_paged_attention_kernel,
        dim3(static_cast<unsigned int>(programs)),
        dim3(256),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const opus::bf16_t*>(q),
        static_cast<const long long*>(packed_route_row),
        static_cast<const int*>(block_expert),
        static_cast<const int*>(block_starts),
        static_cast<const opus::bf16_t*>(page_k),
        static_cast<const opus::bf16_t*>(page_v),
        static_cast<const int*>(slot_pages),
        static_cast<const int*>(slot_lengths),
        static_cast<const int*>(q_lengths),
        static_cast<const int*>(cu_q),
        static_cast<const long long*>(expert_kv_row),
        static_cast<const long long*>(expert_slot),
        static_cast<opus::bf16_t*>(output),
        static_cast<float*>(lse),
        page_capacity,
        state_capacity,
        inline_pages,
        scale);
    return static_cast<int>(hipGetLastError());
}

#endif

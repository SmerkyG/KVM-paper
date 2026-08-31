#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

using bf16 = hip_bfloat16;

constexpr int kHeadDim = 256;
constexpr int kPageSize = 16;
constexpr int kRouteCount = 8;
constexpr int kWaveSize = 64;

__device__ __forceinline__ float wave_sum(float value) {
    value += __shfl_down(value, 32, kWaveSize);
    value += __shfl_down(value, 16, kWaveSize);
    value += __shfl_down(value, 8, kWaveSize);
    value += __shfl_down(value, 4, kWaveSize);
    value += __shfl_down(value, 2, kWaveSize);
    value += __shfl_down(value, 1, kWaveSize);
    return value;
}

__device__ __forceinline__ float wave_max(float value) {
    value = fmaxf(value, __shfl_down(value, 32, kWaveSize));
    value = fmaxf(value, __shfl_down(value, 16, kWaveSize));
    value = fmaxf(value, __shfl_down(value, 8, kWaveSize));
    value = fmaxf(value, __shfl_down(value, 4, kWaveSize));
    value = fmaxf(value, __shfl_down(value, 2, kWaveSize));
    value = fmaxf(value, __shfl_down(value, 1, kWaveSize));
    return value;
}

__device__ __forceinline__ int quantize_int8(float value, float inverse_scale) {
    return max(-127, min(127, static_cast<int>(nearbyintf(value * inverse_scale))));
}

__device__ __forceinline__ int pack_int8x4(
    int value0, int value1, int value2, int value3) {
    const uint32_t packed =
        static_cast<uint8_t>(static_cast<int8_t>(value0))
        | (static_cast<uint32_t>(
            static_cast<uint8_t>(static_cast<int8_t>(value1))) << 8)
        | (static_cast<uint32_t>(
            static_cast<uint8_t>(static_cast<int8_t>(value2))) << 16)
        | (static_cast<uint32_t>(
            static_cast<uint8_t>(static_cast<int8_t>(value3))) << 24);
    return static_cast<int>(packed);
}

union SharedKeyStorage {
    bf16 exact[kPageSize * kHeadDim];
    int8_t quantized[kPageSize * kHeadDim];
};

union SharedValueStorage {
    bf16 exact[kPageSize * kHeadDim];
    int8_t quantized[kPageSize * kHeadDim];
};

__global__ __launch_bounds__(kWaveSize) void quantize_decode_queries_kernel(
    const bf16* __restrict__ q,
    int* __restrict__ quantized_q,
    float* __restrict__ query_scales,
    int query_rows) {
    const int query_row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (query_row >= query_rows) {
        return;
    }
    float query_fragment[4];
    float local_maximum = 0.0f;
    #pragma unroll
    for (int element = 0; element < 4; ++element) {
        query_fragment[element] = static_cast<float>(
            q[query_row * kHeadDim + lane * 4 + element]);
        local_maximum = fmaxf(local_maximum, fabsf(query_fragment[element]));
    }
    const float query_maximum = __shfl(
        wave_max(local_maximum), 0, kWaveSize);
    const float query_scale = fmaxf(query_maximum / 127.0f, 1.0e-20f);
    const float inverse_query_scale = 1.0f / query_scale;
    quantized_q[query_row * kWaveSize + lane] = pack_int8x4(
        quantize_int8(query_fragment[0], inverse_query_scale),
        quantize_int8(query_fragment[1], inverse_query_scale),
        quantize_int8(query_fragment[2], inverse_query_scale),
        quantize_int8(query_fragment[3], inverse_query_scale));
    if (lane == 0) {
        query_scales[query_row] = query_scale;
    }
}

template <
    int RouteSplits,
    bool AdaptiveSplits,
    bool Int8Storage,
    int TotalGqaHeads,
    int BlockGqaHeads,
    int MtpSteps>
__global__ __launch_bounds__(BlockGqaHeads * MtpSteps * kWaveSize, 1)
void gqa_cooperative_decode_kernel(
    const bf16* __restrict__ q,
    const int* __restrict__ quantized_q,
    const float* __restrict__ query_scales,
    const int64_t* __restrict__ cache_indices,
    const void* __restrict__ page_k,
    const void* __restrict__ page_v,
    const bf16* __restrict__ page_k_scales,
    const bf16* __restrict__ page_v_scales,
    const int* __restrict__ page_indices,
    const int* __restrict__ slot_pages,
    const int* __restrict__ directory_values,
    const int* __restrict__ slot_lengths,
    const int64_t* __restrict__ top_slots,
    float* __restrict__ partial_out,
    float* __restrict__ partial_lse,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int page_capacity,
    int leaf_capacity,
    int state_capacity,
    int inline_pages,
    int directory_capacity,
    int page_lookup_mode,
    int indexed,
    int64_t top_batch_stride,
    int64_t top_head_stride,
    float scale_log2) {
    constexpr int kHeadGroups = TotalGqaHeads / BlockGqaHeads;
    constexpr int kQueryRows = BlockGqaHeads * MtpSteps;
    const int request_kv_group = static_cast<int>(blockIdx.x);
    const int candidate = static_cast<int>(blockIdx.y);
    const int route_split = static_cast<int>(blockIdx.z);
    const int request_kv = request_kv_group / kHeadGroups;
    const int head_group = request_kv_group - request_kv * kHeadGroups;
    const int request = request_kv / kv_heads;
    const int kv_head = request_kv - request * kv_heads;
    const int first_batch = request * MtpSteps;
    int cache_batch = static_cast<int>(cache_indices[first_batch]);
    const bool cache_valid = cache_batch >= 0 && cache_batch < cache_batches;
    cache_batch = cache_valid ? cache_batch : 0;
    const int cache_batch_kv = cache_batch * kv_heads + kv_head;
    const int tid = static_cast<int>(threadIdx.x);
    const int head = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    constexpr bool kInt8Pv = Int8Storage;
    constexpr bool kPackProbabilities = Int8Storage && RouteSplits == 8;

    __shared__ SharedKeyStorage shared_k;
    __shared__ SharedValueStorage shared_v;
    __shared__ bf16 shared_k_scales[kPageSize];
    __shared__ bf16 shared_v_scales[kPageSize];
    __shared__ int shared_leaf_indices[kPageSize];
    __shared__ float shared_probabilities[kQueryRows * kPageSize];
    __shared__ int8_t shared_probability_codes[kQueryRows * kPageSize];
    __shared__ float shared_correction[kQueryRows];
    __shared__ float shared_maximum[kQueryRows];
    __shared__ float shared_denominator[kQueryRows];
    __shared__ int shared_slot;
    __shared__ int shared_key_count;
    __shared__ int shared_selected_mask;
    __shared__ int shared_selected_rank[kQueryRows];
    __shared__ int shared_leader;

    if (tid == 0) {
        const int candidate_row = candidate / kRouteCount;
        const int candidate_rank = candidate % kRouteCount;
        const int candidate_step = candidate_row / BlockGqaHeads;
        const int candidate_head =
            candidate_row - candidate_step * BlockGqaHeads;
        const int query_head_base = kv_head * TotalGqaHeads
            + head_group * BlockGqaHeads;
        const int64_t slot = top_slots[
            static_cast<int64_t>(first_batch + candidate_step) * top_batch_stride
            + static_cast<int64_t>(query_head_base + candidate_head)
                * top_head_stride
            + candidate_rank];
        bool leader = cache_valid && slot >= 0 && slot < state_capacity;
        for (int earlier = 0; earlier < candidate && leader; ++earlier) {
            const int earlier_row = earlier / kRouteCount;
            const int earlier_rank = earlier % kRouteCount;
            const int earlier_step = earlier_row / BlockGqaHeads;
            const int earlier_head =
                earlier_row - earlier_step * BlockGqaHeads;
            const int64_t earlier_slot = top_slots[
                static_cast<int64_t>(first_batch + earlier_step)
                    * top_batch_stride
                + static_cast<int64_t>(query_head_base + earlier_head)
                    * top_head_stride
                + earlier_rank];
            leader = earlier_slot != slot;
        }
        int selected_mask = 0;
        for (int selected_row = 0; selected_row < kQueryRows; ++selected_row) {
            const int selected_step = selected_row / BlockGqaHeads;
            const int selected_head =
                selected_row - selected_step * BlockGqaHeads;
            shared_selected_rank[selected_row] = 0;
            for (int rank = 0; rank < kRouteCount; ++rank) {
                const int64_t selected_slot = top_slots[
                    static_cast<int64_t>(first_batch + selected_step)
                        * top_batch_stride
                    + static_cast<int64_t>(query_head_base + selected_head)
                        * top_head_stride
                    + rank];
                if (selected_slot == slot && slot >= 0) {
                    selected_mask |= 1 << selected_row;
                    shared_selected_rank[selected_row] = rank;
                }
            }
        }
        shared_leader = leader ? 1 : 0;
        shared_slot = leader ? static_cast<int>(slot) : 0;
        shared_selected_mask = selected_mask;
        shared_key_count = leader
            ? slot_lengths[cache_batch_kv * state_capacity + static_cast<int>(slot)]
            : 0;
    }
    __syncthreads();
    if (!shared_leader) {
        return;
    }

    const bool selected = (shared_selected_mask & (1 << head)) != 0;
    const int query_step = head / BlockGqaHeads;
    const int query_group_head = head - query_step * BlockGqaHeads;
    const int query_row = (first_batch + query_step) * query_heads
        + kv_head * TotalGqaHeads + head_group * BlockGqaHeads
        + query_group_head;
    float query_fragment[4];
    int packed_query = 0;
    float query_scale = 1.0f;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!Int8Storage) {
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            query_fragment[element] = static_cast<float>(
                q[query_row * kHeadDim + lane + element * kWaveSize]);
        }
    }
    if (Int8Storage) {
        packed_query = quantized_q[query_row * kWaveSize + lane];
        query_scale = query_scales[query_row];
    }
    if (lane == 0) {
        shared_maximum[head] = -INFINITY;
        shared_denominator[head] = 0.0f;
    }
    __syncthreads();

    const int page_count = (shared_key_count + kPageSize - 1) / kPageSize;
    // Batch size supplies occupancy but does not shorten a selected route's
    // serial tail. This is next_pow2(ceil(page_count / 4)), clamped to 1--32:
    // target at most four pages per workgroup without over-splitting short
    // centroids. The enclosing launch retains RouteSplits blocks, but inactive
    // blocks return before touching partial output.
    const int desired_route_splits = AdaptiveSplits
        ? (
            page_count <= 4 ? 1
            : (page_count <= 8 ? 2
            : (page_count <= 16 ? 4
            : (page_count <= 32 ? 8
            : (page_count <= 64 ? 16 : 32))))
        )
        : RouteSplits;
    const int active_route_splits = desired_route_splits < RouteSplits
        ? desired_route_splits
        : RouteSplits;
    if (route_split >= active_route_splits) {
        return;
    }
    for (
        int page_ordinal = route_split;
        page_ordinal < page_count;
        page_ordinal += active_route_splits
    ) {
        int page_id = -1;
        if (lane == 0) {
            const int slot_index = cache_batch_kv * state_capacity + shared_slot;
            if (page_lookup_mode == -1) {
                const int root_ordinal = page_ordinal / 64;
                const int directory_offset = page_ordinal % 64;
                if (root_ordinal < inline_pages) {
                    const int directory_id = slot_pages[
                        slot_index * inline_pages + root_ordinal];
                    if (directory_id >= 0 && directory_id < directory_capacity) {
                        page_id = directory_values[
                            (cache_batch_kv * directory_capacity + directory_id) * 64
                            + directory_offset];
                    }
                }
            } else if (page_ordinal < inline_pages) {
                page_id = slot_pages[slot_index * inline_pages + page_ordinal];
            }
        }
        page_id = __shfl(page_id, 0, kWaveSize);

        if (page_id < 0 || page_id >= page_capacity) {
            continue;
        }
        if (indexed && tid < kPageSize) {
            shared_leaf_indices[tid] = page_indices[
                (static_cast<int64_t>(cache_batch_kv) * page_capacity + page_id)
                    * kPageSize
                + tid];
        }
        __syncthreads();
        if (Int8Storage && tid < kPageSize) {
            const int token = tid;
            const int leaf_index = indexed ? shared_leaf_indices[token] : 0;
            const bool valid_leaf = !indexed
                || (leaf_index >= 0 && leaf_index < leaf_capacity);
            const int safe_leaf = valid_leaf ? leaf_index : 0;
            const int64_t storage_token = indexed
                ? static_cast<int64_t>(cache_batch_kv) * leaf_capacity + safe_leaf
                : (static_cast<int64_t>(cache_batch_kv) * page_capacity + page_id)
                    * kPageSize + token;
            shared_k_scales[token] = valid_leaf
                ? page_k_scales[storage_token] : bf16(0.0f);
            shared_v_scales[token] = valid_leaf
                ? page_v_scales[storage_token] : bf16(0.0f);
        }
        __syncthreads();
        for (
            int index = tid;
            index < kPageSize * kHeadDim;
            index += kQueryRows * kWaveSize
        ) {
            const int token = index / kHeadDim;
            const int dimension = index - token * kHeadDim;
            const int leaf_index = indexed ? shared_leaf_indices[token] : 0;
            if (indexed) {
                const bool valid_leaf = leaf_index >= 0 && leaf_index < leaf_capacity;
                const int safe_leaf = valid_leaf ? leaf_index : 0;
                const int64_t offset =
                    (static_cast<int64_t>(cache_batch_kv) * leaf_capacity + safe_leaf)
                        * kHeadDim
                    + dimension;
                if (Int8Storage) {
                    const int8_t* quant_k = static_cast<const int8_t*>(page_k);
                    const int8_t* quant_v = static_cast<const int8_t*>(page_v);
                    shared_k.quantized[index] = valid_leaf ? quant_k[offset] : 0;
                    if (kInt8Pv) {
                        shared_v.quantized[index] = valid_leaf ? quant_v[offset] : 0;
                    } else {
                        shared_v.exact[index] = valid_leaf
                            ? bf16(
                                static_cast<float>(quant_v[offset])
                                * static_cast<float>(shared_v_scales[token]))
                            : bf16(0.0f);
                    }
                } else {
                    const bf16* exact_k = static_cast<const bf16*>(page_k);
                    const bf16* exact_v = static_cast<const bf16*>(page_v);
                    shared_k.exact[index] = valid_leaf
                        ? exact_k[offset] : bf16(0.0f);
                    shared_v.exact[index] = valid_leaf
                        ? exact_v[offset] : bf16(0.0f);
                }
            } else {
                const int64_t offset =
                    ((static_cast<int64_t>(cache_batch_kv) * page_capacity + page_id)
                        * kPageSize + token)
                        * kHeadDim
                    + dimension;
                if (Int8Storage) {
                    const int8_t* quant_k = static_cast<const int8_t*>(page_k);
                    const int8_t* quant_v = static_cast<const int8_t*>(page_v);
                    shared_k.quantized[index] = quant_k[offset];
                    if (kInt8Pv) {
                        shared_v.quantized[index] = quant_v[offset];
                    } else {
                        shared_v.exact[index] = bf16(
                            static_cast<float>(quant_v[offset])
                            * static_cast<float>(shared_v_scales[token]));
                    }
                } else {
                    const bf16* exact_k = static_cast<const bf16*>(page_k);
                    const bf16* exact_v = static_cast<const bf16*>(page_v);
                    shared_k.exact[index] = exact_k[offset];
                    shared_v.exact[index] = exact_v[offset];
                }
            }
        }
        __syncthreads();

        float probability_scale = 1.0f;
        if (selected) {
            float score_fragment[kPageSize];
            #pragma unroll
            for (int token = 0; token < kPageSize; ++token) {
                float score = 0.0f;
                if (Int8Storage) {
                    const int dimension = lane * 4;
                    const int packed_key = pack_int8x4(
                        shared_k.quantized[token * kHeadDim + dimension],
                        shared_k.quantized[token * kHeadDim + dimension + 1],
                        shared_k.quantized[token * kHeadDim + dimension + 2],
                        shared_k.quantized[token * kHeadDim + dimension + 3]);
                    const int partial_score = __builtin_amdgcn_sdot4(
                        packed_query, packed_key, 0, false);
                    score = static_cast<float>(partial_score)
                        * query_scale
                        * static_cast<float>(shared_k_scales[token]);
                } else {
                    #pragma unroll
                    for (int element = 0; element < 4; ++element) {
                        const int dimension = lane + element * kWaveSize;
                        score += query_fragment[element] * static_cast<float>(
                            shared_k.exact[token * kHeadDim + dimension]);
                    }
                }
                score = wave_sum(score);
                if (lane == 0) {
                    const int logical_token = page_ordinal * kPageSize + token;
                    score_fragment[token] =
                        logical_token < shared_key_count
                        ? score * scale_log2
                        : -INFINITY;
                }
            }
            if (lane == 0) {
                float block_maximum = -INFINITY;
                #pragma unroll
                for (int token = 0; token < kPageSize; ++token) {
                    block_maximum = fmaxf(block_maximum, score_fragment[token]);
                }
                const float old_maximum = shared_maximum[head];
                const float new_maximum = fmaxf(old_maximum, block_maximum);
                const float correction = exp2f(old_maximum - new_maximum);
                float block_denominator = 0.0f;
                #pragma unroll
                for (int token = 0; token < kPageSize; ++token) {
                    const int logical_token = page_ordinal * kPageSize + token;
                    const float probability = logical_token < shared_key_count
                        ? exp2f(score_fragment[token] - new_maximum)
                        : 0.0f;
                    shared_probabilities[head * kPageSize + token] = probability;
                    block_denominator += probability;
                }
                shared_correction[head] = correction;
                shared_maximum[head] = new_maximum;
                shared_denominator[head] =
                    shared_denominator[head] * correction + block_denominator;
            }
            __builtin_amdgcn_wave_barrier();
            if (kInt8Pv) {
                const bool probability_lane = lane < kPageSize;
                const float scaled_probability = probability_lane
                    ? shared_probabilities[head * kPageSize + lane]
                        * static_cast<float>(shared_v_scales[lane])
                    : 0.0f;
                const float maximum_scaled_probability = __shfl(
                    wave_max(fabsf(scaled_probability)), 0, kWaveSize);
                probability_scale = fmaxf(
                    maximum_scaled_probability / 127.0f, 1.0e-20f);
                if (probability_lane) {
                    shared_probability_codes[head * kPageSize + lane] =
                        static_cast<int8_t>(quantize_int8(
                            scaled_probability, 1.0f / probability_scale));
                }
                __builtin_amdgcn_wave_barrier();
            }
        }

        if (selected) {
            const float correction = shared_correction[head];
            int packed_probabilities[kPageSize / 4];
            if (kPackProbabilities) {
                // Keeping four packed words live pays off for the dense
                // eight-split launch. At 16/32 splits it lowers occupancy, so
                // those specializations repack while traversing each output.
                #pragma unroll
                for (int token = 0; token < kPageSize; token += 4) {
                    packed_probabilities[token / 4] = pack_int8x4(
                        shared_probability_codes[
                            head * kPageSize + token],
                        shared_probability_codes[
                            head * kPageSize + token + 1],
                        shared_probability_codes[
                            head * kPageSize + token + 2],
                        shared_probability_codes[
                            head * kPageSize + token + 3]);
                }
            }
            #pragma unroll
            for (int element = 0; element < 4; ++element) {
                const int dimension = lane + element * kWaveSize;
                float update;
                if (kInt8Pv) {
                    int quantized_update = 0;
                    #pragma unroll
                    for (int token = 0; token < kPageSize; token += 4) {
                        const int packed_probability = kPackProbabilities
                            ? packed_probabilities[token / 4]
                            : pack_int8x4(
                                shared_probability_codes[
                                    head * kPageSize + token],
                                shared_probability_codes[
                                    head * kPageSize + token + 1],
                                shared_probability_codes[
                                    head * kPageSize + token + 2],
                                shared_probability_codes[
                                    head * kPageSize + token + 3]);
                        const int packed_value = pack_int8x4(
                            shared_v.quantized[
                                token * kHeadDim + dimension],
                            shared_v.quantized[
                                (token + 1) * kHeadDim + dimension],
                            shared_v.quantized[
                                (token + 2) * kHeadDim + dimension],
                            shared_v.quantized[
                                (token + 3) * kHeadDim + dimension]);
                        quantized_update = __builtin_amdgcn_sdot4(
                            packed_probability,
                            packed_value,
                            quantized_update,
                            false);
                    }
                    update = static_cast<float>(quantized_update)
                        * probability_scale;
                } else {
                    update = 0.0f;
                    #pragma unroll
                    for (int token = 0; token < kPageSize; ++token) {
                        update += shared_probabilities[head * kPageSize + token]
                            * static_cast<float>(
                                shared_v.exact[token * kHeadDim + dimension]);
                    }
                }
                accumulator[element] = accumulator[element] * correction + update;
            }
        }
        __syncthreads();
    }

    if (selected) {
        const int partial_row =
            (query_row * kRouteCount + shared_selected_rank[head])
            * RouteSplits + route_split;
        const float denominator = shared_denominator[head];
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            partial_out[
                partial_row * kHeadDim + lane + element * kWaveSize]
                = denominator > 0.0f ? accumulator[element] / denominator : 0.0f;
        }
        if (lane == 0) {
            partial_lse[partial_row] = denominator > 0.0f
                ? (shared_maximum[head] + log2f(denominator))
                    * 0.6931471805599453f
                : -INFINITY;
        }
    }
}

template <int RoutePartitions, int PageSplits>
__global__ __launch_bounds__(256, 1)
void mtp2_gqa2_aggregate_decode_kernel(
    const bf16* __restrict__ q,
    const int64_t* __restrict__ cache_indices,
    const bf16* __restrict__ page_k,
    const bf16* __restrict__ page_v,
    const int* __restrict__ page_indices,
    const int* __restrict__ slot_pages,
    const int* __restrict__ directory_values,
    const int* __restrict__ slot_lengths,
    const int64_t* __restrict__ top_slots,
    float* __restrict__ partial_out,
    float* __restrict__ partial_lse,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int page_capacity,
    int leaf_capacity,
    int state_capacity,
    int inline_pages,
    int directory_capacity,
    int page_lookup_mode,
    int indexed,
    int64_t top_batch_stride,
    int64_t top_head_stride,
    float scale_log2) {
    constexpr int kTotalGqaHeads = 6;
    constexpr int kBlockGqaHeads = 2;
    constexpr int kMtpSteps = 2;
    constexpr int kQueryRows = kBlockGqaHeads * kMtpSteps;
    constexpr int kHeadGroups = kTotalGqaHeads / kBlockGqaHeads;
    constexpr int kPartialSplits = RoutePartitions * PageSplits;
    const int request_kv_group = static_cast<int>(blockIdx.x);
    const int route_partition = static_cast<int>(blockIdx.y);
    const int page_split = static_cast<int>(blockIdx.z);
    const int request_kv = request_kv_group / kHeadGroups;
    const int head_group = request_kv_group - request_kv * kHeadGroups;
    const int request = request_kv / kv_heads;
    const int kv_head = request_kv - request * kv_heads;
    const int first_batch = request * kMtpSteps;
    int cache_batch = static_cast<int>(cache_indices[first_batch]);
    const bool cache_valid = cache_batch >= 0 && cache_batch < cache_batches;
    cache_batch = cache_valid ? cache_batch : 0;
    const int cache_batch_kv = cache_batch * kv_heads + kv_head;
    const int tid = static_cast<int>(threadIdx.x);
    const int row = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int query_step = row / kBlockGqaHeads;
    const int query_group_head = row - query_step * kBlockGqaHeads;
    const int query_head_base = kv_head * kTotalGqaHeads
        + head_group * kBlockGqaHeads;
    const int query_row = (first_batch + query_step) * query_heads
        + query_head_base + query_group_head;

    __shared__ SharedKeyStorage shared_k;
    __shared__ SharedValueStorage shared_v;
    __shared__ int shared_leaf_indices[kPageSize];
    __shared__ float shared_probabilities[kQueryRows * kPageSize];
    __shared__ float shared_correction[kQueryRows];
    __shared__ float shared_maximum[kQueryRows];
    __shared__ float shared_denominator[kQueryRows];
    __shared__ int shared_slot;
    __shared__ int shared_key_count;
    __shared__ int shared_selected_mask;
    __shared__ int shared_leader;

    float query_fragment[4];
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int element = 0; element < 4; ++element) {
        query_fragment[element] = static_cast<float>(
            q[query_row * kHeadDim + lane + element * kWaveSize]);
    }
    if (lane == 0) {
        shared_maximum[row] = -INFINITY;
        shared_denominator[row] = 0.0f;
    }
    __syncthreads();

    for (
        int candidate = route_partition;
        candidate < kQueryRows * kRouteCount;
        candidate += RoutePartitions
    ) {
        if (tid == 0) {
            const int candidate_row = candidate / kRouteCount;
            const int candidate_rank = candidate % kRouteCount;
            const int candidate_step = candidate_row / kBlockGqaHeads;
            const int candidate_head =
                candidate_row - candidate_step * kBlockGqaHeads;
            const int64_t slot = top_slots[
                static_cast<int64_t>(first_batch + candidate_step)
                    * top_batch_stride
                + static_cast<int64_t>(query_head_base + candidate_head)
                    * top_head_stride
                + candidate_rank];
            bool leader = cache_valid && slot >= 0 && slot < state_capacity;
            for (int earlier = 0; earlier < candidate && leader; ++earlier) {
                const int earlier_row = earlier / kRouteCount;
                const int earlier_rank = earlier % kRouteCount;
                const int earlier_step = earlier_row / kBlockGqaHeads;
                const int earlier_head =
                    earlier_row - earlier_step * kBlockGqaHeads;
                const int64_t earlier_slot = top_slots[
                    static_cast<int64_t>(first_batch + earlier_step)
                        * top_batch_stride
                    + static_cast<int64_t>(query_head_base + earlier_head)
                        * top_head_stride
                    + earlier_rank];
                leader = earlier_slot != slot;
            }
            int selected_mask = 0;
            for (int selected_row = 0; selected_row < kQueryRows; ++selected_row) {
                const int selected_step = selected_row / kBlockGqaHeads;
                const int selected_head =
                    selected_row - selected_step * kBlockGqaHeads;
                for (int rank = 0; rank < kRouteCount; ++rank) {
                    const int64_t selected_slot = top_slots[
                        static_cast<int64_t>(first_batch + selected_step)
                            * top_batch_stride
                        + static_cast<int64_t>(query_head_base + selected_head)
                            * top_head_stride
                        + rank];
                    if (selected_slot == slot && slot >= 0) {
                        selected_mask |= 1 << selected_row;
                    }
                }
            }
            shared_leader = leader ? 1 : 0;
            shared_slot = leader ? static_cast<int>(slot) : 0;
            shared_selected_mask = selected_mask;
            shared_key_count = leader
                ? slot_lengths[
                    cache_batch_kv * state_capacity + static_cast<int>(slot)]
                : 0;
        }
        __syncthreads();
        if (shared_leader) {
            const bool selected =
                (shared_selected_mask & (1 << row)) != 0;
            const int page_count =
                (shared_key_count + kPageSize - 1) / kPageSize;
            for (
                int page_ordinal = page_split;
                page_ordinal < page_count;
                page_ordinal += PageSplits
            ) {
                int page_id = -1;
                if (lane == 0) {
                    const int slot_index =
                        cache_batch_kv * state_capacity + shared_slot;
                    if (page_lookup_mode == -1) {
                        const int root_ordinal = page_ordinal / 64;
                        const int directory_offset = page_ordinal % 64;
                        if (root_ordinal < inline_pages) {
                            const int directory_id = slot_pages[
                                slot_index * inline_pages + root_ordinal];
                            if (
                                directory_id >= 0
                                && directory_id < directory_capacity
                            ) {
                                page_id = directory_values[
                                    (
                                        cache_batch_kv * directory_capacity
                                        + directory_id
                                    ) * 64 + directory_offset];
                            }
                        }
                    } else if (page_ordinal < inline_pages) {
                        page_id = slot_pages[
                            slot_index * inline_pages + page_ordinal];
                    }
                }
                page_id = __shfl(page_id, 0, kWaveSize);
                if (page_id < 0 || page_id >= page_capacity) {
                    continue;
                }
                if (indexed && tid < kPageSize) {
                    shared_leaf_indices[tid] = page_indices[
                        (
                            static_cast<int64_t>(cache_batch_kv)
                                * page_capacity
                            + page_id
                        ) * kPageSize + tid];
                }
                __syncthreads();
                for (
                    int index = tid;
                    index < kPageSize * kHeadDim;
                    index += kQueryRows * kWaveSize
                ) {
                    const int token = index / kHeadDim;
                    const int dimension = index - token * kHeadDim;
                    const int leaf_index = indexed
                        ? shared_leaf_indices[token] : 0;
                    const bool valid_leaf = !indexed
                        || (leaf_index >= 0 && leaf_index < leaf_capacity);
                    const int safe_leaf = valid_leaf ? leaf_index : 0;
                    const int64_t storage_token = indexed
                        ? static_cast<int64_t>(cache_batch_kv) * leaf_capacity
                            + safe_leaf
                        : (
                            static_cast<int64_t>(cache_batch_kv) * page_capacity
                            + page_id
                        ) * kPageSize + token;
                    const int64_t offset = storage_token * kHeadDim + dimension;
                    shared_k.exact[index] = valid_leaf
                        ? page_k[offset] : bf16(0.0f);
                    shared_v.exact[index] = valid_leaf
                        ? page_v[offset] : bf16(0.0f);
                }
                __syncthreads();
                if (selected) {
                    float score_fragment[kPageSize];
                    #pragma unroll
                    for (int token = 0; token < kPageSize; ++token) {
                        float score = 0.0f;
                        #pragma unroll
                        for (int element = 0; element < 4; ++element) {
                            const int dimension = lane + element * kWaveSize;
                            score += query_fragment[element]
                                * static_cast<float>(shared_k.exact[
                                    token * kHeadDim + dimension]);
                        }
                        score = wave_sum(score);
                        if (lane == 0) {
                            const int logical_token =
                                page_ordinal * kPageSize + token;
                            score_fragment[token] =
                                logical_token < shared_key_count
                                ? score * scale_log2 : -INFINITY;
                        }
                    }
                    if (lane == 0) {
                        float block_maximum = -INFINITY;
                        #pragma unroll
                        for (int token = 0; token < kPageSize; ++token) {
                            block_maximum = fmaxf(
                                block_maximum, score_fragment[token]);
                        }
                        const float old_maximum = shared_maximum[row];
                        const float new_maximum =
                            fmaxf(old_maximum, block_maximum);
                        const float correction =
                            exp2f(old_maximum - new_maximum);
                        float block_denominator = 0.0f;
                        #pragma unroll
                        for (int token = 0; token < kPageSize; ++token) {
                            const int logical_token =
                                page_ordinal * kPageSize + token;
                            const float probability =
                                logical_token < shared_key_count
                                ? exp2f(score_fragment[token] - new_maximum)
                                : 0.0f;
                            shared_probabilities[row * kPageSize + token] =
                                probability;
                            block_denominator += probability;
                        }
                        shared_correction[row] = correction;
                        shared_maximum[row] = new_maximum;
                        shared_denominator[row] =
                            shared_denominator[row] * correction
                            + block_denominator;
                    }
                    __builtin_amdgcn_wave_barrier();
                    const float correction = shared_correction[row];
                    #pragma unroll
                    for (int element = 0; element < 4; ++element) {
                        const int dimension = lane + element * kWaveSize;
                        float update = 0.0f;
                        #pragma unroll
                        for (int token = 0; token < kPageSize; ++token) {
                            update += shared_probabilities[
                                row * kPageSize + token]
                                * static_cast<float>(shared_v.exact[
                                    token * kHeadDim + dimension]);
                        }
                        accumulator[element] =
                            accumulator[element] * correction + update;
                    }
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }

    const int partial_split = route_partition * PageSplits + page_split;
    const int partial_row = query_row * kPartialSplits + partial_split;
    const float denominator = shared_denominator[row];
    #pragma unroll
    for (int element = 0; element < 4; ++element) {
        partial_out[partial_row * kHeadDim + lane + element * kWaveSize] =
            denominator > 0.0f ? accumulator[element] / denominator : 0.0f;
    }
    if (lane == 0) {
        partial_lse[partial_row] = denominator > 0.0f
            ? (shared_maximum[row] + log2f(denominator))
                * 0.6931471805599453f
            : -INFINITY;
    }
}

}  // namespace

extern "C" int launch_gqa_cooperative_decode(
    const void* q,
    void* quantized_q,
    void* query_scales,
    const void* cache_indices,
    const void* page_k,
    const void* page_v,
    const void* page_k_scales,
    const void* page_v_scales,
    const void* page_indices,
    const void* slot_pages,
    const void* directory_values,
    const void* slot_lengths,
    const void* top_slots,
    void* partial_out,
    void* partial_lse,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int page_capacity,
    int leaf_capacity,
    int state_capacity,
    int inline_pages,
    int directory_capacity,
    int page_lookup_mode,
    int route_splits,
    int adaptive_splits,
    int indexed,
    int int8_storage,
    int speculative_steps,
    int gqa_head_group_size,
    int aggregate_routes,
    long long top_batch_stride,
    long long top_head_stride,
    float scale_log2,
    void* stream) {
    if (speculative_steps != 1 && speculative_steps != 2) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (batch % speculative_steps != 0) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const int gqa_heads = query_heads / kv_heads;
    if (
        query_heads != kv_heads * gqa_heads
        || !(
            (speculative_steps == 1 && gqa_heads == 4)
            || (speculative_steps == 2 && gqa_heads == 6)
        )
    ) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (
        aggregate_routes
        && (
            speculative_steps != 2
            || gqa_heads != 6
            || gqa_head_group_size != 2
            || route_splits != 8
            || int8_storage
        )
    ) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (
        (speculative_steps == 1 && gqa_head_group_size != 4)
        || (
            speculative_steps == 2
            && gqa_head_group_size != 2
            && gqa_head_group_size != 3
            && gqa_head_group_size != 6
        )
    ) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (
        route_splits != 4 && route_splits != 8
        && route_splits != 16 && route_splits != 32
    ) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (int8_storage) {
        const int query_rows = batch * query_heads;
        hipLaunchKernelGGL(
            quantize_decode_queries_kernel,
            dim3(static_cast<unsigned int>(query_rows)),
            dim3(kWaveSize),
            0,
            hip_stream,
            static_cast<const bf16*>(q),
            static_cast<int*>(quantized_q),
            static_cast<float*>(query_scales),
            query_rows);
        const hipError_t quantize_error = hipGetLastError();
        if (quantize_error != hipSuccess) {
            return static_cast<int>(quantize_error);
        }
    }
    if (aggregate_routes) {
        const dim3 aggregate_grid(
            static_cast<unsigned int>((batch / 2) * kv_heads * 3),
            8,
            2);
        hipLaunchKernelGGL(
            (mtp2_gqa2_aggregate_decode_kernel<8, 2>),
            aggregate_grid,
            dim3(256),
            0,
            hip_stream,
            static_cast<const bf16*>(q),
            static_cast<const int64_t*>(cache_indices),
            static_cast<const bf16*>(page_k),
            static_cast<const bf16*>(page_v),
            static_cast<const int*>(page_indices),
            static_cast<const int*>(slot_pages),
            static_cast<const int*>(directory_values),
            static_cast<const int*>(slot_lengths),
            static_cast<const int64_t*>(top_slots),
            static_cast<float*>(partial_out),
            static_cast<float*>(partial_lse),
            query_heads,
            kv_heads,
            cache_batches,
            page_capacity,
            leaf_capacity,
            state_capacity,
            inline_pages,
            directory_capacity,
            page_lookup_mode,
            indexed,
            static_cast<int64_t>(top_batch_stride),
            static_cast<int64_t>(top_head_stride),
            scale_log2);
        return static_cast<int>(hipGetLastError());
    }
    const dim3 grid(
        static_cast<unsigned int>(
            (batch / speculative_steps) * kv_heads
            * (gqa_heads / gqa_head_group_size)),
        static_cast<unsigned int>(
            speculative_steps * gqa_head_group_size * kRouteCount),
        static_cast<unsigned int>(route_splits));
#define LAUNCH_GQA_COOPERATIVE(                                             \
    RouteSplits, AdaptiveSplits, Int8Storage, TotalGqaHeads,                 \
    BlockGqaHeads, MtpSteps)                                                 \
    hipLaunchKernelGGL(                                                       \
        (gqa_cooperative_decode_kernel<                                      \
            RouteSplits, AdaptiveSplits, Int8Storage, TotalGqaHeads,         \
            BlockGqaHeads, MtpSteps>),                                       \
        grid,                                                                 \
        dim3(BlockGqaHeads * MtpSteps * kWaveSize),                          \
        0,                                                                    \
        hip_stream,                                                           \
        static_cast<const bf16*>(q),                                          \
        static_cast<const int*>(quantized_q),                                 \
        static_cast<const float*>(query_scales),                              \
        static_cast<const int64_t*>(cache_indices),                           \
        page_k,                                                               \
        page_v,                                                               \
        static_cast<const bf16*>(page_k_scales),                              \
        static_cast<const bf16*>(page_v_scales),                              \
        static_cast<const int*>(page_indices),                                \
        static_cast<const int*>(slot_pages),                                  \
        static_cast<const int*>(directory_values),                            \
        static_cast<const int*>(slot_lengths),                                \
        static_cast<const int64_t*>(top_slots),                               \
        static_cast<float*>(partial_out),                                     \
        static_cast<float*>(partial_lse),                                     \
        query_heads,                                                          \
        kv_heads,                                                             \
        cache_batches,                                                        \
        page_capacity,                                                        \
        leaf_capacity,                                                        \
        state_capacity,                                                       \
        inline_pages,                                                         \
        directory_capacity,                                                   \
        page_lookup_mode,                                                     \
        indexed,                                                              \
        static_cast<int64_t>(top_batch_stride),                               \
        static_cast<int64_t>(top_head_stride),                                \
        scale_log2)
#define LAUNCH_SELECTED(RouteSplits, AdaptiveSplits, Int8Storage)            \
    do {                                                                      \
        if (speculative_steps == 1) {                                         \
            LAUNCH_GQA_COOPERATIVE(                                           \
                RouteSplits, AdaptiveSplits, Int8Storage, 4, 4, 1);          \
        } else if (gqa_head_group_size == 2) {                               \
            LAUNCH_GQA_COOPERATIVE(                                           \
                RouteSplits, AdaptiveSplits, Int8Storage, 6, 2, 2);          \
        } else if (gqa_head_group_size == 3) {                               \
            LAUNCH_GQA_COOPERATIVE(                                           \
                RouteSplits, AdaptiveSplits, Int8Storage, 6, 3, 2);          \
        } else {                                                              \
            LAUNCH_GQA_COOPERATIVE(                                           \
                RouteSplits, AdaptiveSplits, Int8Storage, 6, 6, 2);          \
        }                                                                     \
    } while (false)
    if (adaptive_splits && int8_storage) {
        switch (route_splits) {
            case 8:
                LAUNCH_SELECTED(8, true, true);
                break;
            case 16:
                LAUNCH_SELECTED(16, true, true);
                break;
            case 32:
                LAUNCH_SELECTED(32, true, true);
                break;
            default:
                return static_cast<int>(hipErrorInvalidValue);
        }
    } else if (adaptive_splits) {
        switch (route_splits) {
            case 8:
                LAUNCH_SELECTED(8, true, false);
                break;
            case 16:
                LAUNCH_SELECTED(16, true, false);
                break;
            case 32:
                LAUNCH_SELECTED(32, true, false);
                break;
            default:
                return static_cast<int>(hipErrorInvalidValue);
        }
    } else if (int8_storage) {
        switch (route_splits) {
            case 4:
                LAUNCH_SELECTED(4, false, true);
                break;
            case 8:
                LAUNCH_SELECTED(8, false, true);
                break;
            case 16:
                LAUNCH_SELECTED(16, false, true);
                break;
            case 32:
                LAUNCH_SELECTED(32, false, true);
                break;
        }
    } else {
        switch (route_splits) {
            case 4:
                LAUNCH_SELECTED(4, false, false);
                break;
            case 8:
                LAUNCH_SELECTED(8, false, false);
                break;
            case 16:
                LAUNCH_SELECTED(16, false, false);
                break;
            case 32:
                LAUNCH_SELECTED(32, false, false);
                break;
        }
    }
#undef LAUNCH_SELECTED
#undef LAUNCH_GQA_COOPERATIVE
    return static_cast<int>(hipGetLastError());
}

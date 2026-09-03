#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocprim/warp/warp_sort.hpp>

#include <cmath>
#include <cstdint>

namespace {

using bf16 = hip_bfloat16;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

struct bit16x8 {
    bit16x4 xy[2];
};

constexpr int kHeadDim = 128;
constexpr int kGqaHeads = 16;
constexpr int kPartitionTokens = 256;
constexpr int kThreads = 256;
constexpr int kWaveSize = 64;
constexpr int kWarps = kThreads / kWaveSize;
constexpr int kTokensPerWarp = kPartitionTokens / kWarps;
constexpr int kTokenTilesPerWarp = kTokensPerWarp / 16;
constexpr int kFeatureTiles = kHeadDim / 32;
constexpr int kRouteCount = 8;
constexpr int kItemsPerProducer = kPartitionTokens / kGqaHeads;

__device__ __forceinline__ floatx4 mfma_bf16_16x16x16(
    const bit16x4& a,
    const bit16x4& b,
    const floatx4& c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ bool route_better(
    float left_score,
    int left_index,
    float right_score,
    int right_index) {
    return left_score > right_score
        || (left_score == right_score && left_index >= 0
            && (right_index < 0 || left_index < right_index));
}

__device__ __forceinline__ void insert_route_candidate(
    float (&scores)[kRouteCount],
    int (&indices)[kRouteCount],
    float score,
    int index) {
    if (!route_better(score, index, scores[kRouteCount - 1], indices[kRouteCount - 1])) {
        return;
    }
    int position = kRouteCount - 1;
#pragma unroll
    for (int rank = kRouteCount - 2; rank >= 0; --rank) {
        if (route_better(score, index, scores[rank], indices[rank])) {
            scores[rank + 1] = scores[rank];
            indices[rank + 1] = indices[rank];
            position = rank;
        }
    }
    scores[position] = score;
    indices[position] = index;
}

// This is the D=128, GQA=16 QK half of AITER's gfx942 paged-attention
// schedule.  One block owns 256 centroids for one (batch, KV-head) pair.
// Four waves each scan 64 centroids, while every K fragment participates in
// an MFMA against all sixteen query heads.  Unlike attention, the kernel
// deliberately stops after QK and materializes the corrected routing scores.
template <bool EmitCandidates, bool EmitLse, bool EmitPredictedUnion>
__global__ __launch_bounds__(kThreads) void gqa16_coarse_score_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ state_k,
    const float* __restrict__ counts,
    const int64_t* __restrict__ cache_indices,
    float* __restrict__ scores,
    float* __restrict__ candidate_scores,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ partial_lse,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    int64_t state_batch_stride,
    int64_t state_head_stride,
    int64_t state_token_stride,
    int64_t count_batch_stride,
    int64_t count_head_stride,
    int64_t count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    int max_segments,
    float scale,
    const float* __restrict__ predicted_thresholds,
    int* __restrict__ seen_stamps,
    const int* __restrict__ sequence_epochs,
    int* __restrict__ union_counts,
    int* __restrict__ union_slots,
    int union_capacity) {
    const int batch_kv = static_cast<int>(blockIdx.x);
    const int partition = static_cast<int>(blockIdx.y);
    const int logical_batch = batch_kv / kv_heads;
    const int kv_head = batch_kv - logical_batch * kv_heads;
    if (logical_batch >= batch) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int warpid = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int lane16 = lane % 16;
    const int row = lane / 16;
    const int partition_begin = partition * kPartitionTokens;

    __shared__ bit16x4 shared_q[kFeatureTiles][4][kGqaHeads][4];
    __shared__ float shared_inverse_count[kPartitionTokens];
    __shared__ float shared_log_count[kPartitionTokens];
    __shared__ float shared_candidate_scores[
        kGqaHeads * kGqaHeads * kItemsPerProducer];
    __shared__ int shared_candidate_indices[
        kGqaHeads * kGqaHeads * kItemsPerProducer];
    __shared__ float shared_lse_maximum[kGqaHeads * kGqaHeads];
    __shared__ float shared_lse_denominator[kGqaHeads * kGqaHeads];
    __shared__ int shared_union_flags[kPartitionTokens];
    __shared__ int shared_union_wave_counts[kWarps];
    __shared__ int shared_union_wave_bases[kWarps];
    __shared__ int shared_union_block_base;
    __shared__ int shared_cache_batch;

    if (tid == 0) {
        int cache_batch = static_cast<int>(cache_indices[logical_batch]);
        shared_cache_batch =
            cache_batch >= 0 && cache_batch < cache_batches ? cache_batch : 0;
    }

    const int local_token = tid;
    const int global_token = partition_begin + local_token;
    if (global_token < state_len) {
        const int cache_batch = static_cast<int>(cache_indices[logical_batch]);
        const bool valid_cache = cache_batch >= 0 && cache_batch < cache_batches;
        const int safe_cache_batch = valid_cache ? cache_batch : 0;
        const int64_t count_offset =
            static_cast<int64_t>(safe_cache_batch) * count_batch_stride
            + static_cast<int64_t>(kv_head) * count_head_stride
            + static_cast<int64_t>(global_token) * count_token_stride;
        const float count = valid_cache ? counts[count_offset] : 0.0f;
        const bool valid = count > 0.0f;
        shared_inverse_count[local_token] = valid ? 1.0f / count : 0.0f;
        shared_log_count[local_token] = valid ? logf(count) : -INFINITY;
    } else {
        shared_inverse_count[local_token] = 0.0f;
        shared_log_count[local_token] = -INFINITY;
    }

    // Each wave loads four query heads.  Across its four rows, the wave loads
    // 16 bytes from each query for every lane16; the shared transpose then
    // presents the A/B fragments expected by mfma_f32_16x16x16bf16_1k.
    const int loaded_query_head = kv_head * kGqaHeads + 4 * warpid + row;
    const int query_dimension = lane16 * 8;
    if (loaded_query_head < query_heads) {
        const bf16* query_ptr =
            q + static_cast<int64_t>(logical_batch * query_heads + loaded_query_head)
                * kHeadDim
            + query_dimension;
        const bit16x8 query_bits =
            *reinterpret_cast<const bit16x8*>(query_ptr);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int feature_group = lane16 * 2 + half;
            const int feature_tile = feature_group / 8;
            const int feature_row = (feature_group / 2) % 4;
            const int feature_lane = feature_group % 2;
            shared_q[feature_tile][feature_row][4 * warpid + row][feature_lane]
                = query_bits.xy[half];
        }
    }
    __syncthreads();

    bit16x8 query_fragments[kFeatureTiles];
#pragma unroll
    for (int feature_tile = 0; feature_tile < kFeatureTiles; ++feature_tile) {
        // The final dimension is laid out as two adjacent bit16x4 values.  Use
        // an aligned 16-byte load, matching AITER's Qlocal construction.
        query_fragments[feature_tile] = *reinterpret_cast<const bit16x8*>(
            &shared_q[feature_tile][row][lane16][0]);
    }

    const int cache_batch = shared_cache_batch;
    const bf16* key_head =
        state_k + static_cast<int64_t>(cache_batch) * state_batch_stride
        + static_cast<int64_t>(kv_head) * state_head_stride;

    float local_scores[kRouteCount];
    int local_indices[kRouteCount];
    float local_partition_scores[kItemsPerProducer];
    int local_partition_indices[kItemsPerProducer];
#pragma unroll
    for (int rank = 0; rank < kRouteCount; ++rank) {
        local_scores[rank] = -INFINITY;
        local_indices[rank] = -1;
    }
#pragma unroll
    for (int item = 0; item < kItemsPerProducer; ++item) {
        local_partition_scores[item] = -INFINITY;
        local_partition_indices[item] = -1;
    }

#pragma unroll
    for (int token_tile = 0; token_tile < kTokenTilesPerWarp; ++token_tile) {
        const int key_token =
            partition_begin + warpid * kTokensPerWarp + token_tile * 16 + lane16;
        bit16x8 key_fragments[kFeatureTiles];
#pragma unroll
        for (int feature_tile = 0; feature_tile < kFeatureTiles; ++feature_tile) {
            const int feature = feature_tile * 32 + row * 8;
            if (key_token < state_len) {
                const bf16* key_ptr =
                    key_head + static_cast<int64_t>(key_token) * state_token_stride
                    + feature;
                key_fragments[feature_tile] =
                    *reinterpret_cast<const bit16x8*>(key_ptr);
            } else {
                key_fragments[feature_tile].xy[0] = {0, 0, 0, 0};
                key_fragments[feature_tile].xy[1] = {0, 0, 0, 0};
            }
        }

        floatx4 dot = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int feature_tile = 0; feature_tile < kFeatureTiles; ++feature_tile) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                dot = mfma_bf16_16x16x16(
                    key_fragments[feature_tile].xy[half],
                    query_fragments[feature_tile].xy[half],
                    dot);
            }
        }

#pragma unroll
        for (int element = 0; element < 4; ++element) {
            const int token_in_partition =
                warpid * kTokensPerWarp + token_tile * 16 + row * 4 + element;
            const int token = partition_begin + token_in_partition;
            bool selected = false;
            if (token < state_len) {
                float score = dot[element] * scale
                    * shared_inverse_count[token_in_partition]
                    + shared_log_count[token_in_partition];
                const bool route_valid =
                    token >= protected_len
                    && (max_leaf_tokens <= 0
                        || shared_inverse_count[token_in_partition]
                            > 1.0f / static_cast<float>(max_leaf_tokens));
                if (!route_valid) {
                    score = -INFINITY;
                }
                const int query_head = kv_head * kGqaHeads + lane16;
                if constexpr (EmitPredictedUnion) {
                    const int row_index = logical_batch * query_heads + query_head;
                    selected = route_valid && score > predicted_thresholds[row_index];
                }
                const int item = token_tile * 4 + element;
                local_partition_scores[item] = score;
                local_partition_indices[item] = token;
                if constexpr (!EmitCandidates && !EmitPredictedUnion) {
                    scores[
                        static_cast<int64_t>(logical_batch * query_heads + query_head)
                            * state_capacity
                        + token] = score;
                }
            }
            if constexpr (EmitPredictedUnion) {
                const unsigned long long ballot = __ballot(selected);
                const unsigned long long subgroup =
                    (ballot >> (row * kGqaHeads)) & 0xffffULL;
                if (lane16 == 0) {
                    shared_union_flags[token_in_partition] =
                        subgroup != 0 && token < state_len;
                }
            }
        }
    }

    if constexpr (EmitPredictedUnion) {
        // Reserve output space once per 256-centroid block rather than once
        // per selected centroid. This keeps the current-query union fully
        // parallel without turning a broad GQA union into an atomic hotspot.
        __syncthreads();
        const bool union_selected = shared_union_flags[tid] != 0;
        const unsigned long long ballot = __ballot(union_selected);
        const unsigned long long preceding_mask =
            lane == 0 ? 0ULL : ((1ULL << lane) - 1ULL);
        const int lane_prefix = __popcll(ballot & preceding_mask);
        const int wave_count = __popcll(ballot);
        if (lane == 0) {
            shared_union_wave_counts[warpid] = wave_count;
        }
        __syncthreads();
        if (tid == 0) {
            int total = 0;
#pragma unroll
            for (int current_wave = 0; current_wave < kWarps; ++current_wave) {
                shared_union_wave_bases[current_wave] = total;
                total += shared_union_wave_counts[current_wave];
            }
            const int sequence = logical_batch * kv_heads + kv_head;
            shared_union_block_base = atomicAdd(union_counts + sequence, total);
        }
        __syncthreads();
        if (union_selected) {
            const int sequence = logical_batch * kv_heads + kv_head;
            const int token = partition_begin + tid;
            seen_stamps[
                static_cast<int64_t>(sequence) * state_capacity + token]
                = sequence_epochs[sequence];
            const int destination = shared_union_block_base
                + shared_union_wave_bases[warpid] + lane_prefix;
            if (destination < union_capacity) {
                union_slots[
                    static_cast<int64_t>(sequence) * union_capacity + destination]
                    = token;
            }
        }
    }

    if constexpr (EmitLse) {
        float local_maximum = -INFINITY;
#pragma unroll
        for (int item = 0; item < kItemsPerProducer; ++item) {
            local_maximum = fmaxf(local_maximum, local_partition_scores[item]);
        }
        float local_denominator = 0.0f;
        // A producer can own no live/eligible centroids (especially in the
        // final, partially populated state partition).  Treat that as an
        // empty softmax contribution instead of evaluating -inf - -inf.
        if (local_maximum != -INFINITY) {
#pragma unroll
            for (int item = 0; item < kItemsPerProducer; ++item) {
                local_denominator +=
                    expf(local_partition_scores[item] - local_maximum);
            }
        }
        const int query = lane16;
        const int producer = warpid * 4 + row;
        const int lse_offset = query * kGqaHeads + producer;
        shared_lse_maximum[lse_offset] = local_maximum;
        shared_lse_denominator[lse_offset] = local_denominator;
        __syncthreads();

        const int reduce_query = tid / kGqaHeads;
        const int reduce_producer = tid % kGqaHeads;
        if (reduce_producer == 0) {
            float maximum = -INFINITY;
            float denominator = 0.0f;
#pragma unroll
            for (int source = 0; source < kGqaHeads; ++source) {
                const int source_offset = reduce_query * kGqaHeads + source;
                const float source_maximum = shared_lse_maximum[source_offset];
                const float source_denominator =
                    shared_lse_denominator[source_offset];
                if (source_denominator > 0.0f) {
                    if (denominator == 0.0f) {
                        maximum = source_maximum;
                        denominator = source_denominator;
                    } else {
                        const float new_maximum = fmaxf(maximum, source_maximum);
                        denominator = denominator * expf(maximum - new_maximum)
                            + source_denominator
                                * expf(source_maximum - new_maximum);
                        maximum = new_maximum;
                    }
                }
            }
            const int query_head = kv_head * kGqaHeads + reduce_query;
            partial_lse[
                static_cast<int64_t>(logical_batch * query_heads + query_head)
                    * max_segments
                + partition] = denominator > 0.0f
                    ? maximum + logf(denominator)
                    : -INFINITY;
        }
    }

    if constexpr (EmitCandidates) {
        const int query = lane16;
        const int producer = warpid * 4 + row;
        const int shared_base =
            (query * kGqaHeads + producer) * kItemsPerProducer;
#pragma unroll
        for (int item = 0; item < kItemsPerProducer; ++item) {
            shared_candidate_scores[shared_base + item] = local_partition_scores[item];
            shared_candidate_indices[shared_base + item] = local_partition_indices[item];
        }
        __syncthreads();

        // Remap the block to sixteen contiguous 16-lane subgroups, one per
        // query head.  A four-step merge reduces the sixteen producer lists.
        const int reduce_query = tid / kGqaHeads;
        const int reduce_producer = tid % kGqaHeads;
        const int reduce_base =
            (reduce_query * kGqaHeads + reduce_producer) * kItemsPerProducer;
#pragma unroll
        for (int item = 0; item < kItemsPerProducer; ++item) {
            local_partition_scores[item] = shared_candidate_scores[reduce_base + item];
            local_partition_indices[item] = shared_candidate_indices[reduce_base + item];
        }
        rocprim::warp_sort<float, kGqaHeads, int> sorter;
        sorter.sort(
            local_partition_scores,
            local_partition_indices,
            rocprim::greater<float>());
        if (reduce_producer == 0) {
            const int query_head = kv_head * kGqaHeads + reduce_query;
            const int64_t output_base =
                (static_cast<int64_t>(logical_batch * query_heads + query_head)
                    * max_segments
                    + partition)
                * kRouteCount;
#pragma unroll
            for (int rank = 0; rank < kRouteCount; ++rank) {
                candidate_scores[output_base + rank] = local_partition_scores[rank];
                candidate_indices[output_base + rank] = local_partition_indices[rank];
            }
        }
    }
}

__global__ __launch_bounds__(kWaveSize) void init_mass_union_kernel(
    int* __restrict__ sequence_epochs,
    int* __restrict__ union_counts,
    int* __restrict__ union_token_counts,
    int sequences) {
    const int sequence = static_cast<int>(blockIdx.x) * kWaveSize
        + static_cast<int>(threadIdx.x);
    if (sequence < sequences) {
        sequence_epochs[sequence] += 1;
        union_counts[sequence] = 0;
        union_token_counts[sequence] = 0;
    }
}

__global__ __launch_bounds__(kWaveSize) void reduce_route_top8_kernel(
    const float* __restrict__ candidate_scores,
    const int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ top_slots,
    float* __restrict__ top_scores,
    int query_rows,
    int active_segments,
    int max_segments) {
    const int tid = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.x) * 4 + tid / 16;
    const int producer = tid % 16;
    float local_scores[kRouteCount];
    int local_indices[kRouteCount];
#pragma unroll
    for (int rank = 0; rank < kRouteCount; ++rank) {
        local_scores[rank] = -INFINITY;
        local_indices[rank] = -1;
    }
    if (row < query_rows && producer < active_segments) {
        const int segment = producer;
            const int64_t base =
                (static_cast<int64_t>(row) * max_segments + segment) * kRouteCount;
#pragma unroll
            for (int rank = 0; rank < kRouteCount; ++rank) {
                local_scores[rank] = candidate_scores[base + rank];
                local_indices[rank] = static_cast<int>(candidate_indices[base + rank]);
            }
    }
    rocprim::warp_sort<float, 16, int> sorter;
    sorter.sort(local_scores, local_indices, rocprim::greater<float>());
    if (row < query_rows && producer == 0) {
#pragma unroll
        for (int rank = 0; rank < kRouteCount; ++rank) {
            const int64_t output = static_cast<int64_t>(row) * kRouteCount + rank;
            top_slots[output] = local_indices[rank];
            top_scores[output] = local_scores[rank];
        }
    }
}

__global__ __launch_bounds__(kWaveSize) void reduce_partition_lse_kernel(
    const float* __restrict__ partial_lse,
    float* __restrict__ full_lse,
    int* __restrict__ sequence_epochs,
    int* __restrict__ union_counts,
    int* __restrict__ union_token_counts,
    int query_rows,
    int query_heads,
    int kv_heads,
    int active_segments,
    int max_segments) {
    const int tid = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.x) * 4 + tid / 16;
    const int producer = tid % 16;
    if (row < query_rows && producer == 0) {
        float maximum = -INFINITY;
        float denominator = 0.0f;
        for (int segment = 0; segment < active_segments; ++segment) {
            const float segment_lse = partial_lse[
                static_cast<int64_t>(row) * max_segments + segment];
            if (segment_lse != -INFINITY) {
                if (denominator == 0.0f) {
                    maximum = segment_lse;
                    denominator = 1.0f;
                } else {
                    const float new_maximum = fmaxf(maximum, segment_lse);
                    denominator = denominator * expf(maximum - new_maximum)
                        + expf(segment_lse - new_maximum);
                    maximum = new_maximum;
                }
            }
        }
        full_lse[row] = denominator > 0.0f
            ? maximum + logf(denominator)
            : -INFINITY;
        const int batch = row / query_heads;
        const int query_head = row - batch * query_heads;
        if ((query_head % kGqaHeads) == 0) {
            const int sequence = batch * kv_heads + query_head / kGqaHeads;
            sequence_epochs[sequence] += 1;
            union_counts[sequence] = 0;
            union_token_counts[sequence] = 0;
        }
    }
}

__global__ __launch_bounds__(kThreads) void mass_cutoff_union_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ full_lse,
    const float* __restrict__ counts,
    const int64_t* __restrict__ cache_indices,
    int* __restrict__ seen_stamps,
    const int* __restrict__ sequence_epochs,
    int* __restrict__ union_counts,
    int* __restrict__ union_slots,
    int batch,
    int query_heads,
    int kv_heads,
    int state_capacity,
    int state_len,
    int union_capacity,
    int64_t count_batch_stride,
    int64_t count_head_stride,
    int64_t count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    float log_mass_fraction) {
    const int sequence = static_cast<int>(blockIdx.x);
    const int partition = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int token = partition * kPartitionTokens + tid;
    const int logical_batch = sequence / kv_heads;
    const int kv_head = sequence - logical_batch * kv_heads;
    bool selected = false;
    if (logical_batch < batch && token < state_len) {
        const int64_t cache_batch = cache_indices[logical_batch];
        const float count = counts[
            cache_batch * count_batch_stride
            + static_cast<int64_t>(kv_head) * count_head_stride
            + static_cast<int64_t>(token) * count_token_stride];
        const bool route_eligible = token >= protected_len && count > 0.0f
            && (max_leaf_tokens <= 0 || count < max_leaf_tokens);
#pragma unroll
        for (int head = 0; head < kGqaHeads; ++head) {
            const int row =
                logical_batch * query_heads + kv_head * kGqaHeads + head;
            const float score = scores[
                static_cast<int64_t>(row) * state_capacity + token];
            selected = route_eligible
                && (selected || score > full_lse[row] + log_mass_fraction);
        }
    }
    if (selected) {
        seen_stamps[
            static_cast<int64_t>(sequence) * state_capacity + token]
            = sequence_epochs[sequence];
    }

    const int lane = tid % kWaveSize;
    const int wave = tid / kWaveSize;
    const unsigned long long ballot = __ballot(selected);
    const unsigned long long preceding_mask =
        lane == 0 ? 0ULL : ((1ULL << lane) - 1ULL);
    const int lane_prefix = __popcll(ballot & preceding_mask);
    const int wave_count = __popcll(ballot);
    __shared__ int shared_wave_counts[kWarps];
    __shared__ int shared_wave_bases[kWarps];
    __shared__ int shared_block_base;
    if (lane == 0) {
        shared_wave_counts[wave] = wave_count;
    }
    __syncthreads();
    if (tid == 0) {
        int total = 0;
#pragma unroll
        for (int current_wave = 0; current_wave < kWarps; ++current_wave) {
            shared_wave_bases[current_wave] = total;
            total += shared_wave_counts[current_wave];
        }
        shared_block_base = atomicAdd(union_counts + sequence, total);
    }
    __syncthreads();
    if (selected) {
        const int destination =
            shared_block_base + shared_wave_bases[wave] + lane_prefix;
        if (destination < union_capacity) {
            union_slots[
                static_cast<int64_t>(sequence) * union_capacity + destination]
                = token;
        }
    }
}

}  // namespace

extern "C" int launch_gqa16_coarse_score(
    const void* q,
    const void* state_k,
    const void* counts,
    const void* cache_indices,
    void* scores,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    long long state_batch_stride,
    long long state_head_stride,
    long long state_token_stride,
    long long count_batch_stride,
    long long count_head_stride,
    long long count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    float scale,
    void* stream) {
    if (query_heads != kv_heads * kGqaHeads || state_len < 0
        || state_len > state_capacity) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const dim3 grid(
        static_cast<unsigned int>(batch * kv_heads),
        static_cast<unsigned int>((state_len + kPartitionTokens - 1) / kPartitionTokens));
    hipLaunchKernelGGL(
        (gqa16_coarse_score_kernel<false, false, false>),
        grid,
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const bf16*>(q),
        static_cast<const bf16*>(state_k),
        static_cast<const float*>(counts),
        static_cast<const int64_t*>(cache_indices),
        static_cast<float*>(scores),
        static_cast<float*>(scores),
        static_cast<int64_t*>(nullptr),
        static_cast<float*>(nullptr),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        state_len,
        state_batch_stride,
        state_head_stride,
        state_token_stride,
        count_batch_stride,
        count_head_stride,
        count_token_stride,
        protected_len,
        max_leaf_tokens,
        1,
        scale,
        static_cast<const float*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<const int*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<int*>(nullptr),
        0);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_gqa16_coarse_candidates(
    const void* q,
    const void* state_k,
    const void* counts,
    const void* cache_indices,
    void* candidate_scores,
    void* candidate_indices,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    long long state_batch_stride,
    long long state_head_stride,
    long long state_token_stride,
    long long count_batch_stride,
    long long count_head_stride,
    long long count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    int max_segments,
    float scale,
    void* stream) {
    if (query_heads != kv_heads * kGqaHeads || state_len < 0
        || state_len > state_capacity || max_segments < 1) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const int active_segments =
        (state_len + kPartitionTokens - 1) / kPartitionTokens;
    if (active_segments > max_segments) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const dim3 grid(
        static_cast<unsigned int>(batch * kv_heads),
        static_cast<unsigned int>(active_segments));
    hipLaunchKernelGGL(
        (gqa16_coarse_score_kernel<true, false, false>),
        grid,
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const bf16*>(q),
        static_cast<const bf16*>(state_k),
        static_cast<const float*>(counts),
        static_cast<const int64_t*>(cache_indices),
        static_cast<float*>(candidate_scores),
        static_cast<float*>(candidate_scores),
        static_cast<int64_t*>(candidate_indices),
        static_cast<float*>(nullptr),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        state_len,
        state_batch_stride,
        state_head_stride,
        state_token_stride,
        count_batch_stride,
        count_head_stride,
        count_token_stride,
        protected_len,
        max_leaf_tokens,
        max_segments,
        scale,
        static_cast<const float*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<const int*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<int*>(nullptr),
        0);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_reduce_route_top8(
    const void* candidate_scores,
    const void* candidate_indices,
    void* top_slots,
    void* top_scores,
    int query_rows,
    int active_segments,
    int max_segments,
    void* stream) {
    if (query_rows < 0 || active_segments < 0 || active_segments > max_segments
        || active_segments > 16) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        reduce_route_top8_kernel,
        dim3(static_cast<unsigned int>((query_rows + 3) / 4)),
        dim3(kWaveSize),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const float*>(candidate_scores),
        static_cast<const int64_t*>(candidate_indices),
        static_cast<int64_t*>(top_slots),
        static_cast<float*>(top_scores),
        query_rows,
        active_segments,
        max_segments);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_gqa16_coarse_scores_lse(
    const void* q,
    const void* state_k,
    const void* counts,
    const void* cache_indices,
    void* scores,
    void* partial_lse,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    long long state_batch_stride,
    long long state_head_stride,
    long long state_token_stride,
    long long count_batch_stride,
    long long count_head_stride,
    long long count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    int max_segments,
    float scale,
    void* stream) {
    if (query_heads != kv_heads * kGqaHeads || state_len < 0
        || state_len > state_capacity) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const int active_segments =
        (state_len + kPartitionTokens - 1) / kPartitionTokens;
    if (active_segments > max_segments) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (gqa16_coarse_score_kernel<false, true, false>),
        dim3(
            static_cast<unsigned int>(batch * kv_heads),
            static_cast<unsigned int>(active_segments)),
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const bf16*>(q),
        static_cast<const bf16*>(state_k),
        static_cast<const float*>(counts),
        static_cast<const int64_t*>(cache_indices),
        static_cast<float*>(scores),
        static_cast<float*>(scores),
        static_cast<int64_t*>(nullptr),
        static_cast<float*>(partial_lse),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        state_len,
        state_batch_stride,
        state_head_stride,
        state_token_stride,
        count_batch_stride,
        count_head_stride,
        count_token_stride,
        protected_len,
        max_leaf_tokens,
        max_segments,
        scale,
        static_cast<const float*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<const int*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<int*>(nullptr),
        0);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_reduce_partition_lse(
    const void* partial_lse,
    void* full_lse,
    void* sequence_epochs,
    void* union_counts,
    void* union_token_counts,
    int query_rows,
    int query_heads,
    int kv_heads,
    int active_segments,
    int max_segments,
    void* stream) {
    if (active_segments < 1 || active_segments > max_segments) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        reduce_partition_lse_kernel,
        dim3(static_cast<unsigned int>((query_rows + 3) / 4)),
        dim3(kWaveSize),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const float*>(partial_lse),
        static_cast<float*>(full_lse),
        static_cast<int*>(sequence_epochs),
        static_cast<int*>(union_counts),
        static_cast<int*>(union_token_counts),
        query_rows,
        query_heads,
        kv_heads,
        active_segments,
        max_segments);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_init_mass_union(
    void* sequence_epochs,
    void* union_counts,
    void* union_token_counts,
    int sequences,
    void* stream) {
    if (sequences < 0) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        init_mass_union_kernel,
        dim3(static_cast<unsigned int>((sequences + kWaveSize - 1) / kWaveSize)),
        dim3(kWaveSize),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<int*>(sequence_epochs),
        static_cast<int*>(union_counts),
        static_cast<int*>(union_token_counts),
        sequences);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_gqa16_predicted_mass_union(
    const void* q,
    const void* state_k,
    const void* counts,
    const void* cache_indices,
    const void* predicted_thresholds,
    void* partial_lse,
    void* seen_stamps,
    const void* sequence_epochs,
    void* union_counts,
    void* union_slots,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    long long state_batch_stride,
    long long state_head_stride,
    long long state_token_stride,
    long long count_batch_stride,
    long long count_head_stride,
    long long count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    int max_segments,
    int union_capacity,
    float scale,
    void* stream) {
    if (query_heads != kv_heads * kGqaHeads || state_len < 0
        || state_len > state_capacity || union_capacity < 1) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const int active_segments =
        (state_len + kPartitionTokens - 1) / kPartitionTokens;
    if (active_segments > max_segments) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (gqa16_coarse_score_kernel<false, true, true>),
        dim3(
            static_cast<unsigned int>(batch * kv_heads),
            static_cast<unsigned int>(active_segments)),
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const bf16*>(q),
        static_cast<const bf16*>(state_k),
        static_cast<const float*>(counts),
        static_cast<const int64_t*>(cache_indices),
        static_cast<float*>(partial_lse),
        static_cast<float*>(partial_lse),
        static_cast<int64_t*>(nullptr),
        static_cast<float*>(partial_lse),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        state_len,
        state_batch_stride,
        state_head_stride,
        state_token_stride,
        count_batch_stride,
        count_head_stride,
        count_token_stride,
        protected_len,
        max_leaf_tokens,
        max_segments,
        scale,
        static_cast<const float*>(predicted_thresholds),
        static_cast<int*>(seen_stamps),
        static_cast<const int*>(sequence_epochs),
        static_cast<int*>(union_counts),
        static_cast<int*>(union_slots),
        union_capacity);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_gqa16_predicted_mass_union_no_lse(
    const void* q,
    const void* state_k,
    const void* counts,
    const void* cache_indices,
    const void* predicted_thresholds,
    void* partial_lse,
    void* seen_stamps,
    const void* sequence_epochs,
    void* union_counts,
    void* union_slots,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    long long state_batch_stride,
    long long state_head_stride,
    long long state_token_stride,
    long long count_batch_stride,
    long long count_head_stride,
    long long count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    int max_segments,
    int union_capacity,
    float scale,
    void* stream) {
    if (query_heads != kv_heads * kGqaHeads || state_len < 0
        || state_len > state_capacity || union_capacity < 1) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const int active_segments =
        (state_len + kPartitionTokens - 1) / kPartitionTokens;
    if (active_segments > max_segments) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        (gqa16_coarse_score_kernel<false, false, true>),
        dim3(
            static_cast<unsigned int>(batch * kv_heads),
            static_cast<unsigned int>(active_segments)),
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const bf16*>(q),
        static_cast<const bf16*>(state_k),
        static_cast<const float*>(counts),
        static_cast<const int64_t*>(cache_indices),
        static_cast<float*>(partial_lse),
        static_cast<float*>(partial_lse),
        static_cast<int64_t*>(nullptr),
        static_cast<float*>(partial_lse),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        state_len,
        state_batch_stride,
        state_head_stride,
        state_token_stride,
        count_batch_stride,
        count_head_stride,
        count_token_stride,
        protected_len,
        max_leaf_tokens,
        max_segments,
        scale,
        static_cast<const float*>(predicted_thresholds),
        static_cast<int*>(seen_stamps),
        static_cast<const int*>(sequence_epochs),
        static_cast<int*>(union_counts),
        static_cast<int*>(union_slots),
        union_capacity);
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_mass_cutoff_union(
    const void* scores,
    const void* full_lse,
    const void* counts,
    const void* cache_indices,
    void* seen_stamps,
    const void* sequence_epochs,
    void* union_counts,
    void* union_slots,
    int batch,
    int query_heads,
    int kv_heads,
    int state_capacity,
    int state_len,
    int union_capacity,
    long long count_batch_stride,
    long long count_head_stride,
    long long count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    float log_mass_fraction,
    void* stream) {
    if (query_heads != kv_heads * kGqaHeads || union_capacity < 1) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        mass_cutoff_union_kernel,
        dim3(
            static_cast<unsigned int>(batch * kv_heads),
            static_cast<unsigned int>((state_len + kPartitionTokens - 1)
                / kPartitionTokens)),
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        static_cast<const float*>(scores),
        static_cast<const float*>(full_lse),
        static_cast<const float*>(counts),
        static_cast<const int64_t*>(cache_indices),
        static_cast<int*>(seen_stamps),
        static_cast<const int*>(sequence_epochs),
        static_cast<int*>(union_counts),
        static_cast<int*>(union_slots),
        batch,
        query_heads,
        kv_heads,
        state_capacity,
        state_len,
        union_capacity,
        count_batch_stride,
        count_head_stride,
        count_token_stride,
        protected_len,
        max_leaf_tokens,
        log_mass_fraction);
    return static_cast<int>(hipGetLastError());
}

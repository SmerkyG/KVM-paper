#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocprim/warp/warp_sort.hpp>

#include <cmath>
#include <cstdint>

namespace {

using bf16 = hip_bfloat16;
using bit16x2 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;

constexpr int kHeadDim = 256;
constexpr int kGqaHeads = 4;
constexpr int kCentroidsPerBlock = 32;
constexpr int kThreads = 256;
constexpr int kWaveSize = 64;
constexpr int kWarps = kThreads / kWaveSize;
constexpr int kCentroidsPerWarp = kCentroidsPerBlock / kWarps;
constexpr int kCentroidSubgroup = kWaveSize / kCentroidsPerWarp;
constexpr int kDimensionsPerLane = kHeadDim / kCentroidSubgroup;
constexpr int kRouteCount = 8;

static_assert(kWarps == kGqaHeads);
static_assert(kCentroidsPerWarp == 8);
static_assert(kCentroidSubgroup == 8);
static_assert(kDimensionsPerLane == 32);

// Qwen3.8-27B uses D=256/GQA=6. Eight scoring waves let sixteen lanes
// cooperate on each of the same 32 centroids, while six waves later sort one
// query each. All eight waves also participate in fixed-mask preparation.
constexpr int kGqa6Heads = 6;
constexpr int kGqa6Threads = 512;
constexpr int kGqa6Warps = kGqa6Threads / kWaveSize;
constexpr int kGqa6ScoringWarps = kGqa6Warps;
constexpr int kGqa6CentroidsPerWarp =
    kCentroidsPerBlock / kGqa6ScoringWarps;
constexpr int kGqa6CentroidSubgroup =
    kWaveSize / kGqa6CentroidsPerWarp;
constexpr int kGqa6DimensionsPerLane =
    kHeadDim / kGqa6CentroidSubgroup;
static_assert(kGqa6Warps == 8);
static_assert(kGqa6CentroidsPerWarp == 4);
static_assert(kGqa6CentroidSubgroup == 16);
static_assert(kGqa6DimensionsPerLane == 16);

struct FixedPrepareArguments {
    const int32_t* local_lens;
    const int32_t* fixed_lengths;
    int32_t* context_lens;
    int32_t* launch_lens;
    const bf16* new_k;
    const bf16* new_v;
    bf16* arena_k;
    bf16* arena_v;
    int32_t* execution_marker;
    int32_t* previous_cache_rows;
    int32_t* previous_counts;
    int32_t* previous_slots;
    const int32_t* fixed_slot_offsets;
    uint8_t* active_mask;
    uint8_t* active_blocks;
    int64_t new_k_batch_stride;
    int64_t new_k_head_stride;
    int64_t new_v_batch_stride;
    int64_t new_v_head_stride;
    int64_t slot_offset_stride;
    int64_t mask_stride;
    int64_t block_stride;
    int union_capacity;
    int local_offset;
    int local_capacity;
    int local_limit;
    int sink_len;
    int leaf_begin;
    int mask_capacity;
    int tile_size;
    bool include_new;
    bool separate_local_sink;
};

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

// A score-only D=256/GQA=4 route kernel for low-row decode. One workgroup owns
// 32 centroids for one (batch, KV-head) pair. Each 8-lane subgroup owns one
// centroid, loads its K vector once, and accumulates all four query scores.
// Queries are staged in LDS so increasing vector parallelism does not multiply
// global query traffic. After a block-wide transpose through LDS, four waves
// independently emit the exact top eight candidates for their query head.
template <bool MeanBeforeDot, bool FixedPrepare>
__global__ __launch_bounds__(kThreads) void centroid_major_route_score_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ state_k,
    const float* __restrict__ counts,
    const int64_t* __restrict__ cache_indices,
    float* __restrict__ candidate_scores,
    int64_t* __restrict__ candidate_indices,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    int max_groups,
    int64_t state_batch_stride,
    int64_t state_head_stride,
    int64_t state_token_stride,
    int64_t count_batch_stride,
    int64_t count_head_stride,
    int64_t count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    float scale,
    FixedPrepareArguments fixed) {
    const int batch_kv = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(blockIdx.y);
    const int logical_batch = batch_kv / kv_heads;
    const int kv_head = batch_kv - logical_batch * kv_heads;
    if (logical_batch >= batch) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int warpid = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int centroid_in_warp = lane / kCentroidSubgroup;
    const int centroid_lane = lane % kCentroidSubgroup;
    const int centroid_in_block = warpid * kCentroidsPerWarp + centroid_in_warp;
    const int centroid = group * kCentroidsPerBlock + centroid_in_block;

    __shared__ bf16 shared_q[kGqaHeads][kHeadDim];
    __shared__ float shared_scores[kGqaHeads][kCentroidsPerBlock];
    __shared__ int shared_cache_batch;

    if (tid == 0) {
        const int cache_batch = static_cast<int>(cache_indices[logical_batch]);
        shared_cache_batch =
            cache_batch >= 0 && cache_batch < cache_batches ? cache_batch : 0;
    }

    // Four naturally coalesced 512-byte loads stage the whole GQA query group.
#pragma unroll
    for (int query = 0; query < kGqaHeads; ++query) {
        const int query_head = kv_head * kGqaHeads + query;
        shared_q[query][tid] = q[
            static_cast<int64_t>(logical_batch * query_heads + query_head)
                * kHeadDim
            + tid];
    }
    __syncthreads();

    const int cache_batch = shared_cache_batch;

    if constexpr (FixedPrepare) {
        const int sequence = batch_kv;
        const int physical_sequence = cache_batch * kv_heads + kv_head;
        const int active_local = min(
            fixed.local_lens[cache_batch], fixed.local_limit)
            + static_cast<int>(fixed.include_new);
        const int metadata_index = group * kThreads + tid;

        if (metadata_index < fixed.local_limit) {
            fixed.active_mask[
                static_cast<int64_t>(sequence) * fixed.mask_stride
                + metadata_index] = !fixed.separate_local_sink
                && metadata_index < active_local;
        }
        if (metadata_index < state_capacity) {
            fixed.active_mask[
                static_cast<int64_t>(sequence) * fixed.mask_stride
                + fixed.local_limit + fixed.sink_len + metadata_index] = 1;
        }
        const int prefix_blocks =
            (fixed.leaf_begin + fixed.tile_size - 1) / fixed.tile_size;
        if (metadata_index < prefix_blocks) {
            fixed.active_blocks[
                static_cast<int64_t>(sequence) * fixed.block_stride
                + metadata_index] = !fixed.separate_local_sink
                || (metadata_index * fixed.tile_size + fixed.tile_size
                    > fixed.local_limit + fixed.sink_len);
        }

        if (group == 0) {
            if (tid < fixed.sink_len) {
                fixed.active_mask[
                    static_cast<int64_t>(sequence) * fixed.mask_stride
                    + fixed.local_limit + tid] =
                    !fixed.separate_local_sink;
            }
            if (tid == 0) {
                const int fixed_length = fixed.fixed_lengths[physical_sequence];
                fixed.context_lens[sequence] = fixed_length;
                fixed.launch_lens[sequence] = max(fixed_length, 1);
                if (sequence == 0) {
                    *fixed.execution_marker = 2;
                }
            }
            if (fixed.include_new && !fixed.separate_local_sink
                && tid < kHeadDim) {
                const int64_t physical_local = fixed.local_offset
                    + static_cast<int64_t>(physical_sequence)
                        * fixed.local_capacity
                    + active_local - 1;
                fixed.arena_k[physical_local * kHeadDim + tid] = fixed.new_k[
                    static_cast<int64_t>(logical_batch)
                        * fixed.new_k_batch_stride
                    + static_cast<int64_t>(kv_head) * fixed.new_k_head_stride
                    + tid];
                fixed.arena_v[physical_local * kHeadDim + tid] = fixed.new_v[
                    static_cast<int64_t>(logical_batch)
                        * fixed.new_v_batch_stride
                    + static_cast<int64_t>(kv_head) * fixed.new_v_head_stride
                    + tid];
            }
        }

        // One score block owns one retained route. Its 256 threads clear that
        // posting list cooperatively while other blocks score independently.
        const int previous_count = fixed.previous_counts[sequence];
        if (group < previous_count) {
            const int previous_slot = fixed.previous_slots[
                static_cast<int64_t>(sequence) * fixed.union_capacity + group];
            const int previous_cache_batch = fixed.previous_cache_rows[sequence];
            if (previous_slot >= 0 && previous_slot < state_capacity
                && previous_cache_batch >= 0) {
                const int64_t previous_offset_base =
                    static_cast<int64_t>(previous_cache_batch * kv_heads + kv_head)
                    * fixed.slot_offset_stride;
                const int previous_start = fixed.fixed_slot_offsets[
                    previous_offset_base + previous_slot];
                const int previous_stop = fixed.fixed_slot_offsets[
                    previous_offset_base + previous_slot + 1];
                const int previous_leaf_count = previous_stop - previous_start;
                for (int offset = tid; offset < previous_leaf_count;
                     offset += kThreads) {
                    const int logical_token =
                        fixed.leaf_begin + previous_start + offset;
                    if (logical_token < fixed.mask_capacity) {
                        fixed.active_mask[
                            static_cast<int64_t>(sequence) * fixed.mask_stride
                            + logical_token] = 0;
                    }
                }
                const int first_block =
                    (fixed.leaf_begin + previous_start) / fixed.tile_size;
                const int last_block =
                    (fixed.leaf_begin + previous_stop + fixed.tile_size - 1)
                    / fixed.tile_size;
                const int block_capacity =
                    (fixed.mask_capacity + fixed.tile_size - 1) / fixed.tile_size;
                for (int logical_block = first_block + tid;
                     logical_block < last_block;
                     logical_block += kThreads) {
                    if (logical_block < block_capacity
                        && logical_block * fixed.tile_size >= fixed.leaf_begin) {
                        fixed.active_blocks[
                            static_cast<int64_t>(sequence) * fixed.block_stride
                            + logical_block] = 0;
                    }
                }
            }
        }
    }

    const bool in_range = centroid < state_len;
    const bf16* key = state_k
        + static_cast<int64_t>(cache_batch) * state_batch_stride
        + static_cast<int64_t>(kv_head) * state_head_stride
        + static_cast<int64_t>(in_range ? centroid : 0) * state_token_stride;
    const int feature_begin = centroid_lane * kDimensionsPerLane;

    float count = 0.0f;
    if (centroid_lane == 0 && in_range) {
        count = counts[
            static_cast<int64_t>(cache_batch) * count_batch_stride
            + static_cast<int64_t>(kv_head) * count_head_stride
            + static_cast<int64_t>(centroid) * count_token_stride];
    }
    count = __shfl(count, 0, kCentroidSubgroup);
    const bool valid = in_range && count > 0.0f
        && centroid >= protected_len
        && (max_leaf_tokens <= 0
            || count < static_cast<float>(max_leaf_tokens));
    const float inverse_count = valid ? 1.0f / count : 0.0f;

    float dots[kGqaHeads] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int feature = 0; feature < kDimensionsPerLane; feature += 2) {
        bit16x2 key_bits = {0, 0};
        if (in_range) {
            key_bits = *reinterpret_cast<const bit16x2*>(
                key + feature_begin + feature);
        }
        float key0 = bf16_bits_to_float(key_bits[0]);
        float key1 = bf16_bits_to_float(key_bits[1]);
        if constexpr (MeanBeforeDot) {
            // Production routing forms the mean in FP32 and rounds each
            // component back to the BF16 K dtype before tl.dot.
            key0 = static_cast<float>(bf16(key0 * inverse_count));
            key1 = static_cast<float>(bf16(key1 * inverse_count));
        }
#pragma unroll
        for (int query = 0; query < kGqaHeads; ++query) {
            const float query0 = static_cast<float>(
                shared_q[query][feature_begin + feature]);
            const float query1 = static_cast<float>(
                shared_q[query][feature_begin + feature + 1]);
            dots[query] = fmaf(key0, query0, dots[query]);
            dots[query] = fmaf(key1, query1, dots[query]);
        }
    }

    // Each eight-lane centroid subgroup reduces independently inside its wave.
#pragma unroll
    for (int offset = kCentroidSubgroup / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int query = 0; query < kGqaHeads; ++query) {
            dots[query] += __shfl_down(dots[query], offset, kCentroidSubgroup);
        }
    }

    if (centroid_lane == 0) {
        const float multiplier = MeanBeforeDot ? scale : scale * inverse_count;
        const float bias = valid ? logf(count) : -INFINITY;
#pragma unroll
        for (int query = 0; query < kGqaHeads; ++query) {
            shared_scores[query][centroid_in_block] =
                valid ? dots[query] * multiplier + bias : -INFINITY;
        }
    }
    __syncthreads();

    // Reuse one wave per query to sort this block's 32 scores. Inactive lanes
    // carry -inf, so the first eight lanes hold the exact block candidates.
    const int query = warpid;
    float score = lane < kCentroidsPerBlock
        ? shared_scores[query][lane]
        : -INFINITY;
    int index = lane < kCentroidsPerBlock
        ? group * kCentroidsPerBlock + lane
        : -1;
    rocprim::warp_sort<float, kWaveSize, int> sorter;
    sorter.sort(score, index, rocprim::greater<float>());
    if (lane < kRouteCount) {
        const int query_head = kv_head * kGqaHeads + query;
        const int64_t output =
            (static_cast<int64_t>(logical_batch * query_heads + query_head)
                * max_groups
                + group)
                * kRouteCount
            + lane;
        candidate_scores[output] = score;
        candidate_indices[output] = index;
    }
}

// D=256/GQA=6 specialization for Qwen3.8-27B. One 512-thread workgroup
// owns 32 centroids and loads each K component exactly once. Six query scores
// are accumulated while the K fragment remains in registers. Eight waves score
// four centroids apiece, then six waves emit one query's block top-eight.
template <bool MeanBeforeDot, bool FixedPrepare>
__global__ __launch_bounds__(kGqa6Threads) void
centroid_major_gqa6_route_score_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ state_k,
    const float* __restrict__ counts,
    const int64_t* __restrict__ cache_indices,
    float* __restrict__ candidate_scores,
    int64_t* __restrict__ candidate_indices,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    int max_groups,
    int64_t state_batch_stride,
    int64_t state_head_stride,
    int64_t state_token_stride,
    int64_t count_batch_stride,
    int64_t count_head_stride,
    int64_t count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    float scale,
    FixedPrepareArguments fixed) {
    const int batch_kv = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(blockIdx.y);
    const int logical_batch = batch_kv / kv_heads;
    const int kv_head = batch_kv - logical_batch * kv_heads;
    if (logical_batch >= batch) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int warpid = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    __shared__ bf16 shared_q[kGqa6Heads][kHeadDim];
    __shared__ float shared_scores[kGqa6Heads][kCentroidsPerBlock];
    __shared__ int shared_cache_batch;

    if (tid == 0) {
        const int cache_batch = static_cast<int>(cache_indices[logical_batch]);
        shared_cache_batch =
            cache_batch >= 0 && cache_batch < cache_batches ? cache_batch : 0;
    }
    if (tid < kHeadDim) {
#pragma unroll
        for (int query = 0; query < kGqa6Heads; ++query) {
            const int query_head = kv_head * kGqa6Heads + query;
            shared_q[query][tid] = q[
                static_cast<int64_t>(logical_batch * query_heads + query_head)
                    * kHeadDim
                + tid];
        }
    }
    __syncthreads();

    const int cache_batch = shared_cache_batch;

    if constexpr (FixedPrepare) {
        const int sequence = batch_kv;
        const int physical_sequence = cache_batch * kv_heads + kv_head;
        const int active_local = min(
            fixed.local_lens[cache_batch], fixed.local_limit)
            + static_cast<int>(fixed.include_new);
        const int metadata_index = group * kGqa6Threads + tid;

        if (metadata_index < fixed.local_limit) {
            fixed.active_mask[
                static_cast<int64_t>(sequence) * fixed.mask_stride
                + metadata_index] = !fixed.separate_local_sink
                && metadata_index < active_local;
        }
        if (metadata_index < state_capacity) {
            fixed.active_mask[
                static_cast<int64_t>(sequence) * fixed.mask_stride
                + fixed.local_limit + fixed.sink_len + metadata_index] = 1;
        }
        const int prefix_blocks =
            (fixed.leaf_begin + fixed.tile_size - 1) / fixed.tile_size;
        if (metadata_index < prefix_blocks) {
            fixed.active_blocks[
                static_cast<int64_t>(sequence) * fixed.block_stride
                + metadata_index] = !fixed.separate_local_sink
                || (metadata_index * fixed.tile_size + fixed.tile_size
                    > fixed.local_limit + fixed.sink_len);
        }

        if (group == 0) {
            if (tid < fixed.sink_len) {
                fixed.active_mask[
                    static_cast<int64_t>(sequence) * fixed.mask_stride
                    + fixed.local_limit + tid] =
                    !fixed.separate_local_sink;
            }
            if (tid == 0) {
                const int fixed_length = fixed.fixed_lengths[physical_sequence];
                fixed.context_lens[sequence] = fixed_length;
                fixed.launch_lens[sequence] = max(fixed_length, 1);
                if (sequence == 0) {
                    *fixed.execution_marker = 2;
                }
            }
            if (fixed.include_new && !fixed.separate_local_sink
                && tid < kHeadDim) {
                const int64_t physical_local = fixed.local_offset
                    + static_cast<int64_t>(physical_sequence)
                        * fixed.local_capacity
                    + active_local - 1;
                fixed.arena_k[physical_local * kHeadDim + tid] = fixed.new_k[
                    static_cast<int64_t>(logical_batch)
                        * fixed.new_k_batch_stride
                    + static_cast<int64_t>(kv_head) * fixed.new_k_head_stride
                    + tid];
                fixed.arena_v[physical_local * kHeadDim + tid] = fixed.new_v[
                    static_cast<int64_t>(logical_batch)
                        * fixed.new_v_batch_stride
                    + static_cast<int64_t>(kv_head) * fixed.new_v_head_stride
                    + tid];
            }
        }

        const int previous_count = fixed.previous_counts[sequence];
        if (group < previous_count) {
            const int previous_slot = fixed.previous_slots[
                static_cast<int64_t>(sequence) * fixed.union_capacity + group];
            const int previous_cache_batch = fixed.previous_cache_rows[sequence];
            if (previous_slot >= 0 && previous_slot < state_capacity
                && previous_cache_batch >= 0) {
                const int64_t previous_offset_base =
                    static_cast<int64_t>(
                        previous_cache_batch * kv_heads + kv_head)
                    * fixed.slot_offset_stride;
                const int previous_start = fixed.fixed_slot_offsets[
                    previous_offset_base + previous_slot];
                const int previous_stop = fixed.fixed_slot_offsets[
                    previous_offset_base + previous_slot + 1];
                const int previous_leaf_count = previous_stop - previous_start;
                for (int offset = tid; offset < previous_leaf_count;
                     offset += kGqa6Threads) {
                    const int logical_token =
                        fixed.leaf_begin + previous_start + offset;
                    if (logical_token < fixed.mask_capacity) {
                        fixed.active_mask[
                            static_cast<int64_t>(sequence) * fixed.mask_stride
                            + logical_token] = 0;
                    }
                }
                const int first_block =
                    (fixed.leaf_begin + previous_start) / fixed.tile_size;
                const int last_block =
                    (fixed.leaf_begin + previous_stop + fixed.tile_size - 1)
                    / fixed.tile_size;
                const int block_capacity =
                    (fixed.mask_capacity + fixed.tile_size - 1)
                    / fixed.tile_size;
                for (int logical_block = first_block + tid;
                     logical_block < last_block;
                     logical_block += kGqa6Threads) {
                    if (logical_block < block_capacity
                        && logical_block * fixed.tile_size >= fixed.leaf_begin) {
                        fixed.active_blocks[
                            static_cast<int64_t>(sequence) * fixed.block_stride
                            + logical_block] = 0;
                    }
                }
            }
        }
    }

    if (warpid < kGqa6ScoringWarps) {
        const int centroid_in_warp = lane / kGqa6CentroidSubgroup;
        const int centroid_lane = lane % kGqa6CentroidSubgroup;
        const int centroid_in_block =
            warpid * kGqa6CentroidsPerWarp + centroid_in_warp;
        const int centroid = group * kCentroidsPerBlock + centroid_in_block;
        const bool in_range = centroid < state_len;
        const bf16* key = state_k
            + static_cast<int64_t>(cache_batch) * state_batch_stride
            + static_cast<int64_t>(kv_head) * state_head_stride
            + static_cast<int64_t>(in_range ? centroid : 0)
                * state_token_stride;
        const int feature_begin = centroid_lane * kGqa6DimensionsPerLane;

        float count = 0.0f;
        if (centroid_lane == 0 && in_range) {
            count = counts[
                static_cast<int64_t>(cache_batch) * count_batch_stride
                + static_cast<int64_t>(kv_head) * count_head_stride
                + static_cast<int64_t>(centroid) * count_token_stride];
        }
        count = __shfl(count, 0, kGqa6CentroidSubgroup);
        const bool valid = in_range && count > 0.0f
            && centroid >= protected_len
            && (max_leaf_tokens <= 0
                || count < static_cast<float>(max_leaf_tokens));
        const float inverse_count = valid ? 1.0f / count : 0.0f;

        float dots[kGqa6Heads] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int feature = 0; feature < kGqa6DimensionsPerLane;
             feature += 2) {
            bit16x2 key_bits = {0, 0};
            if (in_range) {
                key_bits = *reinterpret_cast<const bit16x2*>(
                    key + feature_begin + feature);
            }
            float key0 = bf16_bits_to_float(key_bits[0]);
            float key1 = bf16_bits_to_float(key_bits[1]);
            if constexpr (MeanBeforeDot) {
                key0 = static_cast<float>(bf16(key0 * inverse_count));
                key1 = static_cast<float>(bf16(key1 * inverse_count));
            }
#pragma unroll
            for (int query = 0; query < kGqa6Heads; ++query) {
                const float query0 = static_cast<float>(
                    shared_q[query][feature_begin + feature]);
                const float query1 = static_cast<float>(
                    shared_q[query][feature_begin + feature + 1]);
                dots[query] = fmaf(key0, query0, dots[query]);
                dots[query] = fmaf(key1, query1, dots[query]);
            }
        }

#pragma unroll
        for (int offset = kGqa6CentroidSubgroup / 2; offset > 0;
             offset /= 2) {
#pragma unroll
            for (int query = 0; query < kGqa6Heads; ++query) {
                dots[query] +=
                    __shfl_down(dots[query], offset, kGqa6CentroidSubgroup);
            }
        }

        if (centroid_lane == 0) {
            const float multiplier =
                MeanBeforeDot ? scale : scale * inverse_count;
            const float bias = valid ? logf(count) : -INFINITY;
#pragma unroll
            for (int query = 0; query < kGqa6Heads; ++query) {
                shared_scores[query][centroid_in_block] =
                    valid ? dots[query] * multiplier + bias : -INFINITY;
            }
        }
    }
    __syncthreads();

    if (warpid < kGqa6Heads) {
        const int query = warpid;
        float score = lane < kCentroidsPerBlock
            ? shared_scores[query][lane]
            : -INFINITY;
        int index = lane < kCentroidsPerBlock
            ? group * kCentroidsPerBlock + lane
            : -1;
        rocprim::warp_sort<float, kWaveSize, int> sorter;
        sorter.sort(score, index, rocprim::greater<float>());
        if (lane < kRouteCount) {
            const int query_head = kv_head * kGqa6Heads + query;
            const int64_t output =
                (static_cast<int64_t>(
                    logical_batch * query_heads + query_head)
                    * max_groups
                    + group)
                    * kRouteCount
                + lane;
            candidate_scores[output] = score;
            candidate_indices[output] = index;
        }
    }
}

// Controlled alternative mapping: each wave owns one of the four GQA query
// heads and scores all 32 centroids. The workgroup first loads and normalizes
// the complete K tile once, transposed into LDS so the 32 centroids accessed by
// a wave lie next to one another. Two lanes cooperate on each centroid. This
// trades the centroid-major kernel's register reuse of K across four queries
// for a shorter two-lane reduction and fully coalesced global K loads.
template <bool MeanBeforeDot>
__global__ __launch_bounds__(kThreads) void query_wave_route_score_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ state_k,
    const float* __restrict__ counts,
    const int64_t* __restrict__ cache_indices,
    float* __restrict__ candidate_scores,
    int64_t* __restrict__ candidate_indices,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    int max_groups,
    int64_t state_batch_stride,
    int64_t state_head_stride,
    int64_t state_token_stride,
    int64_t count_batch_stride,
    int64_t count_head_stride,
    int64_t count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    float scale) {
    const int batch_kv = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(blockIdx.y);
    const int logical_batch = batch_kv / kv_heads;
    const int kv_head = batch_kv - logical_batch * kv_heads;
    if (logical_batch >= batch) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int query = wave;
    const int centroid_in_block = lane / 2;
    const int centroid_half = lane % 2;
    const int centroid = group * kCentroidsPerBlock + centroid_in_block;

    int cache_batch = 0;
    if (lane == 0) {
        const int candidate = static_cast<int>(cache_indices[logical_batch]);
        cache_batch = candidate >= 0 && candidate < cache_batches ? candidate : 0;
    }
    cache_batch = __shfl(cache_batch, 0);

    __shared__ bf16 shared_q[kGqaHeads][kHeadDim];
    // Feature-major storage makes a wave's simultaneous centroid reads
    // contiguous in LDS instead of creating a 32-way row-stride bank conflict.
    // The extra column breaks the 64-byte feature stride that would otherwise
    // collapse a wave's transposed stores onto the same small set of LDS banks.
    __shared__ bf16 shared_k[kHeadDim][kCentroidsPerBlock + 1];
    __shared__ float shared_inverse_counts[kCentroidsPerBlock];
    __shared__ float shared_biases[kCentroidsPerBlock];

#pragma unroll
    for (int query_index = 0; query_index < kGqaHeads; ++query_index) {
        const int query_head = kv_head * kGqaHeads + query_index;
        shared_q[query_index][tid] = q[
            static_cast<int64_t>(logical_batch * query_heads + query_head)
                * kHeadDim
            + tid];
    }

    if (tid < kCentroidsPerBlock) {
        const int count_centroid = group * kCentroidsPerBlock + tid;
        float count = 0.0f;
        if (count_centroid < state_len) {
            count = counts[
                static_cast<int64_t>(cache_batch) * count_batch_stride
                + static_cast<int64_t>(kv_head) * count_head_stride
                + static_cast<int64_t>(count_centroid) * count_token_stride];
        }
        const bool valid = count_centroid < state_len && count > 0.0f
            && count_centroid >= protected_len
            && (max_leaf_tokens <= 0
                || count < static_cast<float>(max_leaf_tokens));
        shared_inverse_counts[tid] = valid ? 1.0f / count : 0.0f;
        shared_biases[tid] = valid ? logf(count) : -INFINITY;
    }
    __syncthreads();

    // Each iteration assigns one centroid row to all 256 threads, yielding a
    // coalesced 512-byte global load. The transposed LDS write is consumed by
    // all four query waves after the barrier.
    for (int linear = tid;
         linear < kCentroidsPerBlock * kHeadDim;
         linear += kThreads) {
        const int key_centroid_in_block = linear / kHeadDim;
        const int feature = linear - key_centroid_in_block * kHeadDim;
        const int key_centroid =
            group * kCentroidsPerBlock + key_centroid_in_block;
        bf16 key_value = bf16(0.0f);
        const float inverse_count =
            shared_inverse_counts[key_centroid_in_block];
        if (key_centroid < state_len && inverse_count > 0.0f) {
            const bf16 value = state_k[
                static_cast<int64_t>(cache_batch) * state_batch_stride
                + static_cast<int64_t>(kv_head) * state_head_stride
                + static_cast<int64_t>(key_centroid) * state_token_stride
                + feature];
            if constexpr (MeanBeforeDot) {
                key_value = bf16(static_cast<float>(value) * inverse_count);
            } else {
                key_value = value;
            }
        }
        shared_k[feature][key_centroid_in_block] = key_value;
    }
    __syncthreads();

    // Interleave the two lanes' dimensions. Splitting them into [0,128) and
    // [128,256) maps both lanes onto the same LDS banks. Four independent
    // accumulators also hide the long scalar FMA dependency chain.
    float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int offset = 0; offset < kHeadDim / 2; offset += 4) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int feature = 2 * (offset + item) + centroid_half;
            const float key_value =
                static_cast<float>(shared_k[feature][centroid_in_block]);
            const float query_value =
                static_cast<float>(shared_q[query][feature]);
            partial[item] = fmaf(key_value, query_value, partial[item]);
        }
    }
    float dot = (partial[0] + partial[1]) + (partial[2] + partial[3]);
    dot += __shfl_down(dot, 1, 2);

    const bool score_lane = centroid_half == 0;
    const float inverse_count = shared_inverse_counts[centroid_in_block];
    const bool valid = score_lane && inverse_count > 0.0f;
    const float multiplier = MeanBeforeDot ? scale : scale * inverse_count;
    float score = valid
        ? dot * multiplier + shared_biases[centroid_in_block]
        : -INFINITY;
    int index = valid ? centroid : -1;

    rocprim::warp_sort<float, kWaveSize, int> sorter;
    sorter.sort(score, index, rocprim::greater<float>());
    if (lane < kRouteCount) {
        const int query_head = kv_head * kGqaHeads + query;
        const int64_t output =
            (static_cast<int64_t>(logical_batch * query_heads + query_head)
                * max_groups
                + group)
                * kRouteCount
            + lane;
        candidate_scores[output] = score;
        candidate_indices[output] = index;
    }
}

}  // namespace

extern "C" int launch_centroid_major_route_score(
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
    int max_groups,
    long long state_batch_stride,
    long long state_head_stride,
    long long state_token_stride,
    long long count_batch_stride,
    long long count_head_stride,
    long long count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    int mean_before_dot,
    float scale,
    void* stream) {
    const bool gqa4 = kv_heads > 0 && query_heads == kv_heads * kGqaHeads;
    const bool gqa6 = kv_heads > 0 && query_heads == kv_heads * kGqa6Heads;
    if ((!gqa4 && !gqa6) || state_len < 0
        || state_len > state_capacity || max_groups < 1) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const int active_groups =
        (state_len + kCentroidsPerBlock - 1) / kCentroidsPerBlock;
    if (active_groups > max_groups) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const dim3 grid(
        static_cast<unsigned int>(batch * kv_heads),
        static_cast<unsigned int>(active_groups));
#define LAUNCH_CENTROID_MAJOR(MEAN_BEFORE_DOT)                               \
    hipLaunchKernelGGL(                                                     \
        (centroid_major_route_score_kernel<MEAN_BEFORE_DOT, false>),       \
        grid,                                                              \
        dim3(kThreads),                                                    \
        0,                                                                 \
        static_cast<hipStream_t>(stream),                                  \
        static_cast<const bf16*>(q),                                       \
        static_cast<const bf16*>(state_k),                                 \
        static_cast<const float*>(counts),                                 \
        static_cast<const int64_t*>(cache_indices),                        \
        static_cast<float*>(candidate_scores),                             \
        static_cast<int64_t*>(candidate_indices),                          \
        batch,                                                             \
        query_heads,                                                       \
        kv_heads,                                                          \
        cache_batches,                                                     \
        state_capacity,                                                    \
        state_len,                                                         \
        max_groups,                                                        \
        state_batch_stride,                                                \
        state_head_stride,                                                 \
        state_token_stride,                                                \
        count_batch_stride,                                                \
        count_head_stride,                                                 \
        count_token_stride,                                                \
        protected_len,                                                     \
        max_leaf_tokens,                                                   \
        scale,                                                             \
        FixedPrepareArguments{})
    if (gqa4 && mean_before_dot) {
        LAUNCH_CENTROID_MAJOR(true);
    } else if (gqa4) {
        LAUNCH_CENTROID_MAJOR(false);
    } else if (mean_before_dot) {
        hipLaunchKernelGGL(
            (centroid_major_gqa6_route_score_kernel<true, false>),
            grid,
            dim3(kGqa6Threads),
            0,
            static_cast<hipStream_t>(stream),
            static_cast<const bf16*>(q),
            static_cast<const bf16*>(state_k),
            static_cast<const float*>(counts),
            static_cast<const int64_t*>(cache_indices),
            static_cast<float*>(candidate_scores),
            static_cast<int64_t*>(candidate_indices),
            batch,
            query_heads,
            kv_heads,
            cache_batches,
            state_capacity,
            state_len,
            max_groups,
            state_batch_stride,
            state_head_stride,
            state_token_stride,
            count_batch_stride,
            count_head_stride,
            count_token_stride,
            protected_len,
            max_leaf_tokens,
            scale,
            FixedPrepareArguments{});
    } else {
        hipLaunchKernelGGL(
            (centroid_major_gqa6_route_score_kernel<false, false>),
            grid,
            dim3(kGqa6Threads),
            0,
            static_cast<hipStream_t>(stream),
            static_cast<const bf16*>(q),
            static_cast<const bf16*>(state_k),
            static_cast<const float*>(counts),
            static_cast<const int64_t*>(cache_indices),
            static_cast<float*>(candidate_scores),
            static_cast<int64_t*>(candidate_indices),
            batch,
            query_heads,
            kv_heads,
            cache_batches,
            state_capacity,
            state_len,
            max_groups,
            state_batch_stride,
            state_head_stride,
            state_token_stride,
            count_batch_stride,
            count_head_stride,
            count_token_stride,
            protected_len,
            max_leaf_tokens,
            scale,
            FixedPrepareArguments{});
    }
#undef LAUNCH_CENTROID_MAJOR
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_query_wave_route_score(
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
    int max_groups,
    long long state_batch_stride,
    long long state_head_stride,
    long long state_token_stride,
    long long count_batch_stride,
    long long count_head_stride,
    long long count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    int mean_before_dot,
    float scale,
    void* stream) {
    if (query_heads != kv_heads * kGqaHeads || state_len < 0
        || state_len > state_capacity || max_groups < 1) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const int active_groups =
        (state_len + kCentroidsPerBlock - 1) / kCentroidsPerBlock;
    if (active_groups > max_groups) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const dim3 grid(
        static_cast<unsigned int>(batch * kv_heads),
        static_cast<unsigned int>(active_groups));
#define LAUNCH_QUERY_WAVE(MEAN_BEFORE_DOT)                                  \
    hipLaunchKernelGGL(                                                     \
        (query_wave_route_score_kernel<MEAN_BEFORE_DOT>),                  \
        grid,                                                              \
        dim3(kThreads),                                                    \
        0,                                                                 \
        static_cast<hipStream_t>(stream),                                  \
        static_cast<const bf16*>(q),                                       \
        static_cast<const bf16*>(state_k),                                 \
        static_cast<const float*>(counts),                                 \
        static_cast<const int64_t*>(cache_indices),                        \
        static_cast<float*>(candidate_scores),                             \
        static_cast<int64_t*>(candidate_indices),                          \
        batch,                                                             \
        query_heads,                                                       \
        kv_heads,                                                          \
        cache_batches,                                                     \
        state_capacity,                                                    \
        state_len,                                                         \
        max_groups,                                                        \
        state_batch_stride,                                                \
        state_head_stride,                                                 \
        state_token_stride,                                                \
        count_batch_stride,                                                \
        count_head_stride,                                                 \
        count_token_stride,                                                \
        protected_len,                                                     \
        max_leaf_tokens,                                                   \
        scale)
    if (mean_before_dot) {
        LAUNCH_QUERY_WAVE(true);
    } else {
        LAUNCH_QUERY_WAVE(false);
    }
#undef LAUNCH_QUERY_WAVE
    return static_cast<int>(hipGetLastError());
}

extern "C" int launch_centroid_major_route_score_fixed_prepare(
    const void* q,
    const void* state_k,
    const void* counts,
    const void* cache_indices,
    void* candidate_scores,
    void* candidate_indices,
    const void* local_lens,
    const void* fixed_lengths,
    void* context_lens,
    void* launch_lens,
    const void* new_k,
    const void* new_v,
    void* arena_k,
    void* arena_v,
    void* execution_marker,
    void* previous_cache_rows,
    void* previous_counts,
    void* previous_slots,
    const void* fixed_slot_offsets,
    void* active_mask,
    void* active_blocks,
    int batch,
    int query_heads,
    int kv_heads,
    int cache_batches,
    int state_capacity,
    int state_len,
    int max_groups,
    long long state_batch_stride,
    long long state_head_stride,
    long long state_token_stride,
    long long count_batch_stride,
    long long count_head_stride,
    long long count_token_stride,
    int protected_len,
    int max_leaf_tokens,
    int mean_before_dot,
    long long new_k_batch_stride,
    long long new_k_head_stride,
    long long new_v_batch_stride,
    long long new_v_head_stride,
    long long slot_offset_stride,
    long long mask_stride,
    long long block_stride,
    int union_capacity,
    int local_offset,
    int local_capacity,
    int local_limit,
    int sink_len,
    int leaf_begin,
    int mask_capacity,
    int tile_size,
    int include_new,
    int separate_local_sink,
    float scale,
    void* stream) {
    const bool gqa4 = kv_heads > 0 && query_heads == kv_heads * kGqaHeads;
    const bool gqa6 = kv_heads > 0 && query_heads == kv_heads * kGqa6Heads;
    if ((!gqa4 && !gqa6) || state_len < 0
        || state_len > state_capacity || max_groups < 1 || tile_size <= 0
        || union_capacity <= 0 || mask_capacity <= 0) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const int active_groups =
        (state_len + kCentroidsPerBlock - 1) / kCentroidsPerBlock;
    if (active_groups > max_groups) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const FixedPrepareArguments fixed{
        static_cast<const int32_t*>(local_lens),
        static_cast<const int32_t*>(fixed_lengths),
        static_cast<int32_t*>(context_lens),
        static_cast<int32_t*>(launch_lens),
        static_cast<const bf16*>(new_k),
        static_cast<const bf16*>(new_v),
        static_cast<bf16*>(arena_k),
        static_cast<bf16*>(arena_v),
        static_cast<int32_t*>(execution_marker),
        static_cast<int32_t*>(previous_cache_rows),
        static_cast<int32_t*>(previous_counts),
        static_cast<int32_t*>(previous_slots),
        static_cast<const int32_t*>(fixed_slot_offsets),
        static_cast<uint8_t*>(active_mask),
        static_cast<uint8_t*>(active_blocks),
        new_k_batch_stride,
        new_k_head_stride,
        new_v_batch_stride,
        new_v_head_stride,
        slot_offset_stride,
        mask_stride,
        block_stride,
        union_capacity,
        local_offset,
        local_capacity,
        local_limit,
        sink_len,
        leaf_begin,
        mask_capacity,
        tile_size,
        include_new != 0,
        separate_local_sink != 0,
    };
    const dim3 grid(
        static_cast<unsigned int>(batch * kv_heads),
        static_cast<unsigned int>(active_groups));
#define LAUNCH_CENTROID_MAJOR_FIXED(MEAN_BEFORE_DOT)                         \
    hipLaunchKernelGGL(                                                     \
        (centroid_major_route_score_kernel<MEAN_BEFORE_DOT, true>),        \
        grid,                                                              \
        dim3(kThreads),                                                    \
        0,                                                                 \
        static_cast<hipStream_t>(stream),                                  \
        static_cast<const bf16*>(q),                                       \
        static_cast<const bf16*>(state_k),                                 \
        static_cast<const float*>(counts),                                 \
        static_cast<const int64_t*>(cache_indices),                        \
        static_cast<float*>(candidate_scores),                             \
        static_cast<int64_t*>(candidate_indices),                          \
        batch,                                                             \
        query_heads,                                                       \
        kv_heads,                                                          \
        cache_batches,                                                     \
        state_capacity,                                                    \
        state_len,                                                         \
        max_groups,                                                        \
        state_batch_stride,                                                \
        state_head_stride,                                                 \
        state_token_stride,                                                \
        count_batch_stride,                                                \
        count_head_stride,                                                 \
        count_token_stride,                                                \
        protected_len,                                                     \
        max_leaf_tokens,                                                   \
        scale,                                                             \
        fixed)
    if (gqa4 && mean_before_dot) {
        LAUNCH_CENTROID_MAJOR_FIXED(true);
    } else if (gqa4) {
        LAUNCH_CENTROID_MAJOR_FIXED(false);
    } else if (mean_before_dot) {
        hipLaunchKernelGGL(
            (centroid_major_gqa6_route_score_kernel<true, true>),
            grid,
            dim3(kGqa6Threads),
            0,
            static_cast<hipStream_t>(stream),
            static_cast<const bf16*>(q),
            static_cast<const bf16*>(state_k),
            static_cast<const float*>(counts),
            static_cast<const int64_t*>(cache_indices),
            static_cast<float*>(candidate_scores),
            static_cast<int64_t*>(candidate_indices),
            batch,
            query_heads,
            kv_heads,
            cache_batches,
            state_capacity,
            state_len,
            max_groups,
            state_batch_stride,
            state_head_stride,
            state_token_stride,
            count_batch_stride,
            count_head_stride,
            count_token_stride,
            protected_len,
            max_leaf_tokens,
            scale,
            fixed);
    } else {
        hipLaunchKernelGGL(
            (centroid_major_gqa6_route_score_kernel<false, true>),
            grid,
            dim3(kGqa6Threads),
            0,
            static_cast<hipStream_t>(stream),
            static_cast<const bf16*>(q),
            static_cast<const bf16*>(state_k),
            static_cast<const float*>(counts),
            static_cast<const int64_t*>(cache_indices),
            static_cast<float*>(candidate_scores),
            static_cast<int64_t*>(candidate_indices),
            batch,
            query_heads,
            kv_heads,
            cache_batches,
            state_capacity,
            state_len,
            max_groups,
            state_batch_stride,
            state_head_stride,
            state_token_stride,
            count_batch_stride,
            count_head_stride,
            count_token_stride,
            protected_len,
            max_leaf_tokens,
            scale,
            fixed);
    }
#undef LAUNCH_CENTROID_MAJOR_FIXED
    return static_cast<int>(hipGetLastError());
}

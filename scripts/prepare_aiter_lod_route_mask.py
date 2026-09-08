#!/usr/bin/env python3
"""Build a source overlay that adds compact LOD attention modes to AITER.

The overlay keeps every unmodified AITER/CK file as a symlink and writes regular
copies of only the six files changed or materialized below.  It is therefore safe to point
``AITER_META_DIR`` at the result without modifying the shared AITER install.

The patched page-size-one SGLang batch-prefill ABI reuses metadata arguments
that are otherwise unused in that mode:

The leaf-only ABI uses a tiny ``int32[1, 4]`` ``block_table`` marker and an
``int32[num_indexed_k]`` ``seqlen_k`` whose low 16 bits identify the queries
that selected each indexed token.

The fused high/low-detail ABI instead supplies ``block_table`` as a compact
per-sequence geometry table.  ``bias`` is a one-row FP16 table indexed by
physical page and ``seqlen_k`` contains membership words for the centroid
prefix and leaf union.  Each logical KV sequence is an implicit contiguous
centroid prefix followed by its indexed leaf union.
Selected centroids are replaced by their leaves in the same online softmax;
there is no materialized Q-by-centroid score tensor and no second attention
launch.

The physical ``kv_page_indices`` remain ordinary page-size-one indices.  The
CK kernel applies the route predicate to its existing 64x128/128x128 score
tile after QK MFMA and before online softmax.  Calls that do not have the two
LOD metadata tensors retain the original instruction path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def materialize(text: str) -> str:
    """Turn a source symlink into a regular overlay file without changing it."""
    return text


def patch_torch_interface(text: str) -> str:
    old = """        if(block_table_.has_value())
        {
            auto block_table = block_table_.value();
            CHECK_DEVICE(block_table);
            TORCH_CHECK(block_table.scalar_type() == at::kInt,
                        \"block_table must be int32\");
            TORCH_CHECK(block_table.dim() == 2, \"block_table must be 2d\");
            TORCH_CHECK(block_table.size(0) == batch_size,
                        \"block_table first dim must match batch_size\");
            TORCH_CHECK(block_table.stride(-1) == 1,
                        \"block_table must have contiguous last dimension\");
            TORCH_CHECK(seqlen_k_.has_value(),
                        \"block_table requires seqlen_k for per-batch lengths\");

            auto seqlen_k = seqlen_k_.value();
            CHECK_DEVICE(seqlen_k);
            TORCH_CHECK(seqlen_k.scalar_type() == at::kInt,
                        \"seqlen_k must be int32\");
            TORCH_CHECK(seqlen_k.dim() == 1, \"seqlen_k must be 1d\");
            TORCH_CHECK(seqlen_k.size(0) == batch_size,
                        \"seqlen_k must have shape [batch_size]\");

            args.kv_page_indices = block_table.data_ptr();
            args.batch_stride_block_table = block_table.stride(0);
            args.seqlen_k_ptr = seqlen_k.data_ptr();
            args.kv_lookup_table =
                ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D;
        }
"""
    new = """        bool is_lod_hilo = false;
        if(block_table_.has_value())
        {
            auto block_table = block_table_.value();
            CHECK_DEVICE(block_table);
            TORCH_CHECK(block_table.scalar_type() == at::kInt,
                        \"block_table must be int32\");
            TORCH_CHECK(block_table.dim() == 2, \"block_table must be 2d\");
            TORCH_CHECK(block_table.stride(-1) == 1,
                        \"block_table must have contiguous last dimension\");
            TORCH_CHECK(seqlen_k_.has_value(),
                        \"block_table requires seqlen_k metadata\");

            auto seqlen_k = seqlen_k_.value();
            CHECK_DEVICE(seqlen_k);
            TORCH_CHECK(seqlen_k.scalar_type() == at::kInt,
                        \"seqlen_k must be int32\");
            TORCH_CHECK(seqlen_k.dim() == 1, \"seqlen_k must be 1d\");
            TORCH_CHECK(seqlen_k.stride(-1) == 1,
                        \"seqlen_k must be contiguous\");

            const bool is_lod_route_mask =
                page_block_size == 1 && block_table.size(0) == 1 &&
                block_table.size(1) == 4 &&
                seqlen_k.numel() == kv_page_indices.numel();
            is_lod_hilo =
                page_block_size == 1 && block_table.size(0) == batch_size + 1 &&
                block_table.size(1) == 4 &&
                seqlen_k.numel() >= kv_page_indices.numel() && bias_.has_value();
            if(is_lod_route_mask)
            {
                // Page-size-one SGLang does not consume either field. Reuse
                // them as the ABI marker and indexed-token query masks.
                args.kv_last_page_lens = block_table.data_ptr();
                args.seqlen_k_ptr = seqlen_k.data_ptr();
                args.batch_stride_block_table = block_table.stride(0);
            }
            else if(is_lod_hilo)
            {
                // The negative stride selects the fused high/low-detail ABI.
                // block_table owns the metadata and bias owns one FP16 scalar
                // per physical page; neither is interpreted by stock AITER.
                args.kv_last_page_lens = block_table.data_ptr();
                args.seqlen_k_ptr = seqlen_k.data_ptr();
                args.batch_stride_block_table = -block_table.stride(0);
            }
            else
            {
                TORCH_CHECK(block_table.size(0) == batch_size,
                            \"block_table first dim must match batch_size\");
                TORCH_CHECK(seqlen_k.size(0) == batch_size,
                            \"seqlen_k must have shape [batch_size]\");
                args.kv_page_indices = block_table.data_ptr();
                args.batch_stride_block_table = block_table.stride(0);
                args.seqlen_k_ptr = seqlen_k.data_ptr();
                args.kv_lookup_table =
                    ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D;
            }
        }
        if(is_lod_hilo)
        {
            // Keep bias_ptr available to the LOD specialization, but dispatch
            // a no-bias stock instance so it does not construct a Q-by-K bias
            // window. The patched score stage loads by physical page instead.
            bias_type = bias_enum::no_bias;
        }
"""
    return replace_once(text, old, new, label="torch interface metadata dispatch")


def patch_fmha_driver(text: str) -> str:
    old = """            return PageTableKargs{reinterpret_cast<const int32_t*>(args.kv_indptr),
                                  reinterpret_cast<const int32_t*>(args.kv_page_indices),
                                  reinterpret_cast<const int32_t*>(args.kv_last_page_lens)};
"""
    new = """            const bool has_lod_route_mask =
                args.page_block_size == 1 && args.seqlen_k_ptr != nullptr &&
                args.batch_stride_block_table == 4;
            const bool has_lod_hilo =
                args.page_block_size == 1 && args.seqlen_k_ptr != nullptr &&
                args.batch_stride_block_table < 0;
            return PageTableKargs{
                reinterpret_cast<const int32_t*>(args.kv_indptr),
                reinterpret_cast<const int32_t*>(args.kv_page_indices),
                (has_lod_route_mask || has_lod_hilo) ? nullptr
                                   : reinterpret_cast<const int32_t*>(args.kv_last_page_lens),
                (has_lod_route_mask || has_lod_hilo)
                    ? reinterpret_cast<const int32_t*>(args.kv_last_page_lens)
                    : nullptr,
                (has_lod_route_mask || has_lod_hilo)
                    ? reinterpret_cast<const int32_t*>(args.seqlen_k_ptr)
                    : nullptr,
                (has_lod_route_mask || has_lod_hilo)
                    ? args.batch_stride_block_table
                    : 0,
                has_lod_hilo ? args.bias_ptr : nullptr};
"""
    return replace_once(text, old, new, label="SGLang page-table construction")


def patch_fmha_kernel(text: str) -> str:
    old = """    struct SglangPageTableKargs
    {
        const int32_t* kv_indptr;
        const int32_t* kv_page_indices;
        const int32_t* kv_last_page_lens;
    };
"""
    new = """    struct SglangPageTableKargs
    {
        const int32_t* kv_indptr;
        const int32_t* kv_page_indices;
        const int32_t* kv_last_page_lens;
        const int32_t* lod_query_slots;
        const int32_t* lod_kv_slots;
        ck_tile::index_t lod_route_count;
        const void* lod_key_bias;
    };
"""
    text = replace_once(text, old, new, label="SGLang page-table kargs")

    old = """        long_index_t batch_offset_q       = 0;
        long_index_t batch_offset_bias    = 0;
"""
    new = """        long_index_t batch_offset_q       = 0;
        long_index_t query_start          = 0;
        long_index_t batch_offset_bias    = 0;
"""
    text = replace_once(text, old, new, label="query-start declaration")

    old = """        // WA i_batch capture structure binding before c++20
        const index_t seqlen_k = [&, i_batch_ = i_batch]() {
"""
    new = """        const index_t lod_route_stride = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return kargs.page_table.lod_route_count < 0
                           ? -kargs.page_table.lod_route_count
                           : kargs.page_table.lod_route_count;
            else
                return index_t{0};
        }();
        const bool lod_hilo = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return kargs.page_table.lod_route_count < 0;
            else
                return false;
        }();
        const index_t lod_metadata_offset = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return lod_hilo ? kargs.page_table.lod_query_slots[3] : index_t{0};
            else
                return index_t{0};
        }();
        const index_t lod_state_len = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return lod_hilo
                           ? kargs.page_table
                                 .lod_query_slots[lod_metadata_offset +
                                                  i_batch * lod_route_stride]
                           : index_t{0};
            else
                return index_t{0};
        }();
        const index_t lod_centroid_base = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return lod_hilo
                           ? kargs.page_table
                                 .lod_query_slots[lod_metadata_offset +
                                                  i_batch * lod_route_stride + 1]
                           : index_t{0};
            else
                return index_t{0};
        }();
        const index_t lod_sparse_mask_base = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return lod_hilo
                           ? kargs.page_table
                                 .lod_query_slots[lod_metadata_offset +
                                                  i_batch * lod_route_stride + 2]
                           : index_t{0};
            else
                return index_t{0};
        }();
        const index_t lod_leaf_mask_base = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return lod_hilo
                           ? kargs.page_table
                                 .lod_query_slots[lod_metadata_offset +
                                                  i_batch * lod_route_stride + 3]
                           : index_t{0};
            else
                return index_t{0};
        }();
        // WA i_batch capture structure binding before c++20
        const index_t seqlen_k = [&, i_batch_ = i_batch]() {
"""
    text = replace_once(text, old, new, label="fused LOD sequence metadata")

    old = """                return num_page_blocks > 0
                           ? static_cast<index_t>((num_page_blocks - 1) * kargs.page_block_size +
                                                  last_page_len)
                           : 0;
"""
    new = """                if(lod_hilo)
                    return lod_state_len + num_page_blocks;
                return num_page_blocks > 0
                           ? static_cast<index_t>((num_page_blocks - 1) * kargs.page_block_size +
                                                  last_page_len)
                           : 0;
"""
    text = replace_once(text, old, new, label="fused LOD logical KV length")
    text = replace_once(
        text,
        """            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
""",
        """            query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
""",
        label="query-start assignment",
    )

    old = """        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
"""
    new = """        const int32_t* lod_query_slots = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return lod_hilo
                           ? kargs.page_table.lod_kv_slots + lod_sparse_mask_base
                           : (kargs.page_table.lod_query_slots != nullptr
                                  ? kargs.page_table.lod_query_slots +
                                        query_start * lod_route_stride
                                  : nullptr);
            else
                return static_cast<const int32_t*>(nullptr);
        }();
        const int32_t* lod_kv_slots = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return kargs.page_table.lod_kv_slots != nullptr
                           ? kargs.page_table.lod_kv_slots +
                                 (lod_hilo ? lod_leaf_mask_base
                                           : kargs.page_table.kv_indptr[i_batch])
                           : nullptr;
            else
                return static_cast<const int32_t*>(nullptr);
        }();
        const index_t lod_route_count = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                return lod_hilo ? kargs.page_table.lod_query_slots[1]
                                : lod_route_stride;
            else
                return index_t{0};
        }();

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
"""
    text = replace_once(text, old, new, label="per-sequence LOD pointers")

    old = """                                      dropout,
                                      sink_value,
                                      max_page_table_idx);
"""
    new = """                                      dropout,
                                      sink_value,
                                      max_page_table_idx,
                                      lod_query_slots,
                                      lod_kv_slots,
                                      lod_route_count,
                                      kargs.seqlen_q,
                                      lod_state_len,
                                      lod_centroid_base,
                                      [&]() {
                                          if constexpr(kKVLookupTable ==
                                              BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
                                              return reinterpret_cast<const ck_tile::half_t*>(
                                                  kargs.page_table.lod_key_bias);
                                          else
                                              return static_cast<const ck_tile::half_t*>(nullptr);
                                      }());
"""
    text = replace_once(text, old, new, label="no-scale pipeline call")

    old = """        const index_t max_page_table_idx =
            kargs.seqlen_k > 0 ? (kargs.seqlen_k - 1) / kPageBlockSize : 0;
"""
    new = """        const index_t max_page_table_idx = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
            {
                if(lod_hilo)
                {
                    const index_t leaf_count =
                        kargs.page_table.kv_indptr[i_batch + 1] -
                        kargs.page_table.kv_indptr[i_batch];
                    return leaf_count > 0 ? leaf_count - 1 : index_t{0};
                }
            }
            return kargs.seqlen_k > 0 ? (kargs.seqlen_k - 1) / kPageBlockSize
                                      : index_t{0};
        }();
"""
    # Only the no-quantization branch uses the compact LOD metadata for now.
    return replace_once(text, old, new, label="fused LOD leaf table bound")


def patch_fmha_pipeline(text: str) -> str:
    old = """                                        IndexArrayType& physical_pages,
                                        index_t max_page_table_idx)
"""
    new = """                                        IndexArrayType& physical_pages,
                                        index_t max_page_table_idx,
                                        index_t lod_state_len = 0,
                                        index_t lod_centroid_base = 0)
"""
    text = replace_once(text, old, new, label="fused LOD physical-page signature")

    old = """            const index_t page_id =
                ck_tile::min(global_token_idx >> kLog2PageSize, max_page_table_idx);
            physical_pages[k0] = page_idx[page_id];
"""
    new = """            if(lod_state_len > 0 && global_token_idx < lod_state_len)
            {
                physical_pages[k0] = lod_centroid_base + global_token_idx;
            }
            else
            {
                const index_t leaf_token_idx = global_token_idx - lod_state_len;
                const index_t page_id =
                    ck_tile::min(leaf_token_idx >> kLog2PageSize, max_page_table_idx);
                physical_pages[k0] = page_idx[page_id];
            }
"""
    text = replace_once(text, old, new, label="fused LOD K physical-page lookup")

    old = """                physical_pages[k0] = page_idx[ck_tile::min(global_token_idx, max_page_table_idx)];
"""
    new = """                physical_pages[k0] =
                    lod_state_len > 0 && global_token_idx < lod_state_len
                        ? lod_centroid_base + global_token_idx
                        : page_idx[ck_tile::min(global_token_idx - lod_state_len,
                                                max_page_table_idx)];
"""
    text = replace_once(text, old, new, label="fused LOD V physical-page lookup")

    old = """template <typename Problem_,
          typename Policy_ = BlockFmhaBatchPrefillPipelineQRKSVSAsyncDefaultPolicy>
struct BlockFmhaBatchPrefillPipelineQRKSVSAsync
"""
    new = """template <typename Problem_,
          typename Policy_ = BlockFmhaBatchPrefillPipelineQRKSVSAsyncDefaultPolicy,
          bool kHasLodRouteMask_ = false>
struct BlockFmhaBatchPrefillPipelineQRKSVSAsync
"""
    text = replace_once(text, old, new, label="compile-time LOD pipeline flag")

    old = """    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
"""
    new = """    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    static constexpr bool kHasLodRouteMask = kHasLodRouteMask_;
"""
    text = replace_once(text, old, new, label="compile-time LOD pipeline constant")

    old = """    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }
"""
    # The route-mask specialization reuses the K tile after QK consumes it, so
    # it retains precisely the ordinary kernel's shared-memory allocation.
    text = replace_once(text, old, old, label="route-mask LDS allocation")

    old = """        index_t current_seq_k = kv_load_start;

        // Load physical pages first, then compute offsets.
"""
    new = """        index_t current_seq_k = kv_load_start;

        // Preload one compact membership word per lane while QK is pending.
        // Once QK has consumed K LDS, the same storage is safe to reuse for
        // the score predicate, preserving the ordinary kernel's LDS size.
        auto* lod_query_mask_lds = reinterpret_cast<uint32_t*>(smem_ptr);
        auto* lod_key_bias_lds = reinterpret_cast<float*>(lod_query_mask_lds + kN0);
        auto load_lod_query_mask = [&](index_t tile_start) {
            uint32_t membership = 0;
            if constexpr(kHasLodRouteMask)
            {
                const index_t tid = get_thread_local_1d_id();
                if(tid < kN0)
                {
                    const index_t col = tile_start + tid;
                    if(col < lod_state_len)
                    {
                        index_t first = 0;
                        index_t count = lod_route_count;
                        static_for<0, 8, 1>{}([&](auto) {
                            if(count > 0)
                            {
                                const index_t step = count / 2;
                                const index_t middle = first + step;
                                if(lod_query_slots[middle] < col)
                                {
                                    first = middle + 1;
                                    count -= step + 1;
                                }
                                else
                                {
                                    count = step;
                                }
                            }
                        });
                        if(first < lod_route_count && lod_query_slots[first] == col)
                            membership = static_cast<uint32_t>(
                                lod_query_slots[lod_route_count + first]);
                    }
                    else
                    {
                        const index_t leaf_col = col - lod_state_len;
                        membership = leaf_col <= max_page_table_idx
                                         ? static_cast<uint32_t>(lod_kv_slots[leaf_col])
                                         : uint32_t{0};
                    }
                }
            }
            return membership;
        };
        auto load_lod_key_bias = [&](index_t tile_start) {
            float key_bias = 0.0f;
            if constexpr(kHasLodRouteMask)
            {
                const index_t tid = get_thread_local_1d_id();
                const index_t col = tile_start + tid;
                if(tid < kN0 && lod_state_len > 0 && col < lod_state_len)
                    // The fast softmax runs in base 2 and scale_s already
                    // contains log2(e).  Convert the stored natural-log
                    // multiplicity into the equivalent raw-score offset.
                    key_bias = type_convert<float>(
                                   lod_key_bias_ptr[lod_centroid_base + col]) *
                               ck_tile::log2e_v<> / scale_s;
            }
            return key_bias;
        };
        auto lod_query_mask_reg = load_lod_query_mask(current_seq_k);
        auto lod_key_bias_reg = load_lod_key_bias(current_seq_k);

        // Load physical pages first, then compute offsets.
"""
    text = replace_once(text, old, new, label="route-mask LDS staging")

    old = """                current_seq_k += k_advance;
                // move K tile windows
"""
    new = """                current_seq_k += k_advance;
                lod_query_mask_reg = load_lod_query_mask(current_seq_k);
                lod_key_bias_reg = load_lod_key_bias(current_seq_k);
                // move K tile windows
"""
    text = replace_once(text, old, new, label="next route-mask LDS tile")

    old = """               const float* k_descale_ptr             = nullptr,
               const float* v_descale_ptr             = nullptr,
               index_t nblock_stride_kv_block_descale = 0,
               index_t nhead_stride_kv_block_descale  = 0) const
"""
    new = """               const float* k_descale_ptr             = nullptr,
               const float* v_descale_ptr             = nullptr,
               index_t nblock_stride_kv_block_descale = 0,
               index_t nhead_stride_kv_block_descale  = 0,
               const index_t* lod_query_slots         = nullptr,
               const index_t* lod_kv_slots            = nullptr,
               index_t lod_route_count                = 0,
               index_t lod_query_count                = 0,
               index_t lod_state_len                  = 0,
               index_t lod_centroid_base              = 0,
               const half_t* lod_key_bias_ptr         = nullptr) const
"""
    text = replace_once(text, old, new, label="main pipeline signature")

    old = """            page_idx, k_coord, current_seq_k, k_physical_pages, max_page_table_idx);
"""
    new = """            page_idx,
            k_coord,
            current_seq_k,
            k_physical_pages,
            max_page_table_idx,
            lod_state_len,
            lod_centroid_base);
"""
    if text.count(old) != 2:
        raise RuntimeError(
            "fused LOD K physical-page calls: expected two source matches, "
            f"found {text.count(old)}"
        )
    text = text.replace(old, new)

    old = """                        page_idx, v_coord, current_seq_k, v_physical_pages_k2, max_page_table_idx);
"""
    new = """                        page_idx,
                        v_coord,
                        current_seq_k,
                        v_physical_pages_k2,
                        max_page_table_idx,
                        lod_state_len,
                        lod_centroid_base);
"""
    text = replace_once(text, old, new, label="fused LOD outer V physical-page call")

    old = """                    page_idx, v_coord, current_seq_k, v_physical_pages, max_page_table_idx);
"""
    new = """                    page_idx,
                    v_coord,
                    current_seq_k,
                    v_physical_pages,
                    max_page_table_idx,
                    lod_state_len,
                    lod_centroid_base);
"""
    text = replace_once(text, old, new, label="fused LOD V physical-page call")

    old = """                }

                const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
"""
    new = """                }

                if constexpr(kHasLodRouteMask)
                {
                    // All waves must finish reading the K tile before its LDS
                    // bytes become the compact membership staging buffer.
                    __builtin_amdgcn_s_barrier();
                    const index_t tid = get_thread_local_1d_id();
                    if(tid < kN0)
                    {
                        lod_query_mask_lds[tid] = lod_query_mask_reg;
                        lod_key_bias_lds[tid] = lod_key_bias_reg;
                    }
                    __builtin_amdgcn_s_barrier();

                    // Walk columns outside rows so every thread reuses one LDS
                    // membership load across all of its score rows.
                    constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                    using RowSpan = remove_cvref_t<decltype(s_spans[number<0>{}])>;
                    using ColSpan = remove_cvref_t<decltype(s_spans[number<1>{}])>;
                    constexpr auto idx0_zero = detail::make_tile_distributed_index(
                        typename uniform_sequence_gen<RowSpan::Impl::size(), 0>::type{});
                    constexpr auto idx1_zero = detail::make_tile_distributed_index(
                        typename uniform_sequence_gen<ColSpan::Impl::size(), 0>::type{});
                    const auto row_base_tile_idx = get_x_indices_from_distributed_indices(
                        s_acc.get_tile_distribution(), make_tuple(idx0_zero, idx1_zero));
                    const auto row_base =
                        q_origin.at(number<0>{}) + row_base_tile_idx.at(number<0>{});
                    if(row_base < lod_query_count && row_base < 16)
                    {
                        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                            const auto col_tile_idx = get_x_indices_from_distributed_indices(
                                s_acc.get_tile_distribution(), make_tuple(idx0_zero, idx1));
                            const uint32_t query_membership =
                                lod_query_mask_lds[col_tile_idx.at(number<1>{})];
                            const auto logical_col =
                                current_seq_k + col_tile_idx.at(number<1>{});
                            const auto key_bias =
                                lod_key_bias_lds[col_tile_idx.at(number<1>{})];
                            sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                const auto tile_idx = get_x_indices_from_distributed_indices(
                                    s_acc.get_tile_distribution(), i_j_idx);
                                const auto row =
                                    q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                const bool valid_row =
                                    row < lod_query_count && row < 16;
                                const bool selected =
                                    valid_row && ((query_membership >> row) & 1u) != 0;
                                const bool active = logical_col < lod_state_len
                                                        ? !selected
                                                        : selected;
                                if(valid_row && !active)
                                {
                                    s_acc(i_j_idx) =
                                        -numeric<SMPLComputeDataType>::infinity();
                                }
                                else if(valid_row && logical_col < lod_state_len)
                                {
                                    s_acc(i_j_idx) += key_bias;
                                }
                            });
                        });
                    }
                }

                const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
"""
    text = replace_once(text, old, new, label="score-tile route predicate")

    # A route mask can legitimately leave one query row with no exact leaves.
    # Reuse CK's all-masked-row handling so exp(-inf - -inf) never produces NaN
    # and the final exact contribution is O=0, LSE=-inf.
    old = """                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 FmhaMask::IsMasking)
"""
    new = """                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 FmhaMask::IsMasking || kHasLodRouteMask)
"""
    text = replace_once(text, old, new, label="route-mask validated row max")

    old = """                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
"""
    new = """                if constexpr(FmhaMask::IsMasking || kHasLodRouteMask)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
"""
    text = replace_once(text, old, new, label="route-mask empty row output")

    old = """               DropoutType& dropout,
               float sink_v,
               const index_t max_page_table_idx) const
    {
        return operator()(q_dram_block_window_tmp,
"""
    new = """               DropoutType& dropout,
               float sink_v,
               const index_t max_page_table_idx,
               const index_t* lod_query_slots = nullptr,
               const index_t* lod_kv_slots = nullptr,
               index_t lod_route_count = 0,
               index_t lod_query_count = 0,
               index_t lod_state_len = 0,
               index_t lod_centroid_base = 0,
               const half_t* lod_key_bias_ptr = nullptr) const
    {
        return operator()(q_dram_block_window_tmp,
"""
    text = replace_once(text, old, new, label="no-scale convenience signature")

    old = """                          dropout,
                          sink_v,
                          max_page_table_idx);
"""
    new = """                          dropout,
                          sink_v,
                          max_page_table_idx,
                          nullptr,
                          nullptr,
                          0,
                          0,
                          lod_query_slots,
                          lod_kv_slots,
                          lod_route_count,
                          lod_query_count,
                          lod_state_len,
                          lod_centroid_base,
                          lod_key_bias_ptr);
"""
    return replace_once(text, old, new, label="no-scale convenience forwarding")


def patch_batch_prefill_codegen(text: str) -> str:
    old = """using fmha_pipeline_{F_idx} = {F_pipeline}<
    fmha_pipeline_problem_{F_idx}>;
"""
    new = """using fmha_pipeline_{F_idx} = {F_pipeline}<
    fmha_pipeline_problem_{F_idx}>;

using fmha_lod_pipeline_{F_idx} = {F_pipeline}<
    fmha_pipeline_problem_{F_idx}, typename fmha_pipeline_{F_idx}::Policy, true>;
"""
    text = replace_once(text, old, new, label="masked pipeline codegen alias")

    old = """using fmha_kernel_{F_idx} =
    ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel<fmha_pipeline_{F_idx}, fmha_epilogue_{F_idx}>;
"""
    new = """using fmha_kernel_{F_idx} =
    ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel<fmha_pipeline_{F_idx}, fmha_epilogue_{F_idx}>;

using fmha_lod_kernel_{F_idx} =
    ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel<fmha_lod_pipeline_{F_idx}, fmha_epilogue_{F_idx}>;
"""
    text = replace_once(text, old, new, label="masked kernel codegen alias")

    old = """{{
    using k_ = fmha_kernel_{F_idx};
    if(s.log_level_ > 0)
        std::cout << ", {F_kname}" << std::flush;
    auto [kargs, grids] = fmha_batch_prefill_create_kargs_and_grids<k_>(a);
"""
    new = """{{
    if constexpr(({F_page_size} == 1) &&
                 ({F_kv_lookup_table} == ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D))
    {{
        if(a.seqlen_k_ptr != nullptr &&
           (a.batch_stride_block_table == 4 || a.batch_stride_block_table < 0))
        {{
            using k_lod_ = fmha_lod_kernel_{F_idx};
            if(s.log_level_ > 0)
                std::cout << ", {F_kname}_lod_route_mask" << std::flush;
            auto [kargs, grids] = fmha_batch_prefill_create_kargs_and_grids<k_lod_>(a);
            const dim3 blocks = k_lod_::BlockSize();
            constexpr ck_tile::index_t kBlockPerCu = k_lod_::kBlockPerCu;
            return ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(k_lod_{{}}, grids, blocks, 0, kargs));
        }}
    }}
    using k_ = fmha_kernel_{F_idx};
    if(s.log_level_ > 0)
        std::cout << ", {F_kname}" << std::flush;
    auto [kargs, grids] = fmha_batch_prefill_create_kargs_and_grids<k_>(a);
"""
    return replace_once(text, old, new, label="masked kernel launch dispatch")


PATCHES = {
    # Python resolves an executed script symlink to its installed source tree,
    # which would make this entry point import the unpatched codegen module.
    Path(
        "3rdparty/composable_kernel/example/ck_tile/01_fmha/generate.py"
    ): materialize,
    Path("csrc/py_itfs_ck/mha_batch_prefill_kernels.cu"): patch_torch_interface,
    Path("3rdparty/composable_kernel/example/ck_tile/01_fmha/fmha_fwd.hpp"): patch_fmha_driver,
    Path(
        "3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/"
        "fmha_batch_prefill_kernel.hpp"
    ): patch_fmha_kernel,
    Path(
        "3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/"
        "block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp"
    ): patch_fmha_pipeline,
    Path(
        "3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/"
        "fmha_batch_prefill.py"
    ): patch_batch_prefill_codegen,
}


def symlink_file(source: str, destination: str) -> str:
    os.symlink(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source_meta.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"overlay already exists: {output}")
        shutil.rmtree(output)
    # Never inherit cached codegen bytecode from the source installation: this
    # overlay intentionally patches one generator module, and stale pyc files
    # can otherwise silently emit the unpatched C++ instance bodies.
    shutil.copytree(
        source,
        output,
        copy_function=symlink_file,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    for relative, transform in PATCHES.items():
        source_file = source / relative
        output_file = output / relative
        original = source_file.read_text()
        patched = transform(original)
        output_file.unlink()
        output_file.write_text(patched)
        print(relative)


if __name__ == "__main__":
    main()

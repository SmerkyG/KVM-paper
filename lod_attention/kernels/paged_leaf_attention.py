"""Public compatibility façade for paged LoD attention kernels.

The implementation is split by responsibility so cache maintenance, prefill,
routing, and decode kernels can be understood independently.
"""

from .paged_cache import (
    append_quantized_virtual_paged_kv,
    append_virtual_paged_kv,
    quantize_page_summaries_int8,
    quantize_virtual_paged_kv,
    rehash_overflow_pages,
)
from .paged_decode import fused_decode_paged_lod_attention
from .paged_decode_buffers import (
    advance_decode_cache_lengths,
    new_fused_decode_buffers,
    prepare_speculative_decode_kv,
)
from .paged_decode_kernels import (
    materialize_page1_coarse_means,
    materialize_page1_fixed_indices,
)
from .paged_prefill import (
    paged_leaf_attention,
    query_major_indexed_residual_page_attention,
    query_major_residual_page_attention,
)
from .paged_routing import materialized_state_route_gqa

__all__ = [
    "advance_decode_cache_lengths",
    "append_quantized_virtual_paged_kv",
    "append_virtual_paged_kv",
    "fused_decode_paged_lod_attention",
    "materialize_page1_coarse_means",
    "materialize_page1_fixed_indices",
    "materialized_state_route_gqa",
    "new_fused_decode_buffers",
    "paged_leaf_attention",
    "prepare_speculative_decode_kv",
    "query_major_indexed_residual_page_attention",
    "query_major_residual_page_attention",
    "quantize_page_summaries_int8",
    "quantize_virtual_paged_kv",
    "rehash_overflow_pages",
]

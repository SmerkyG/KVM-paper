"""AITER PA-v1 partition kernel without its standalone output reduction."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import torch
from jinja2 import Template


def _compile(gqa_ratio: int, head_size: int, partition_size: int):
    from csrc.cpp_itfs.utils import AITER_CORE_DIR, compile_template_op

    template_path = Path(__file__).parent / "source" / "aiter_pa_stage1.cpp.jinja"
    return compile_template_op(
        Template(template_path.read_text()),
        "lod_pa_stage1",
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_kernels.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_v1.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_common.cuh",
            f"{AITER_CORE_DIR}/csrc/include",
            f"{AITER_CORE_DIR}/csrc/include/ck_tile/",
        ],
        gqa_ratio=gqa_ratio,
        head_size=head_size,
        partition_size=partition_size,
    )


def paged_attention_stage1(
    workspace: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: int,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    partition_size: int = 256,
) -> None:
    """Write PA-v1 partition maxima, sums, and normalized partial outputs."""
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    if query.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16:
        raise ValueError("LOD PA stage one currently supports BF16 Q/K/V only")
    if value_cache.dtype != key_cache.dtype or key_cache.ndim != 4:
        raise ValueError("LOD PA stage-one cache geometry is incompatible")
    if query.ndim != 3 or int(key_cache.size(1)) != 1 or int(key_cache.size(2)) != 1:
        raise ValueError("LOD PA stage one requires one page-size-one KV head")
    num_seqs, num_heads, head_size = query.shape
    if tuple(block_tables.shape) != (num_seqs, max_context_len):
        raise ValueError("LOD PA stage-one block table has incompatible geometry")
    if tuple(context_lens.shape) != (num_seqs,):
        raise ValueError("LOD PA stage-one context lengths are incompatible")
    max_num_partitions = math.ceil(max_context_len / partition_size)
    required_bytes = num_seqs * num_heads * max_num_partitions * (
        8 + head_size * query.element_size()
    )
    if workspace.numel() * workspace.element_size() < required_bytes:
        raise ValueError("LOD PA stage-one workspace is too small")

    func = _compile(num_heads, head_size, partition_size)
    (
        workspace_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        scale_value,
        max_blocks,
        max_partitions,
        sequences,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        kv_seq_stride,
        stream,
    ) = torch_to_c_types(
        workspace,
        query,
        key_cache,
        value_cache,
        scale,
        max_context_len,
        max_num_partitions,
        num_seqs,
        query.stride(0),
        key_cache.stride(0),
        key_cache.stride(2),
        key_cache.stride(1),
        torch.cuda.current_stream(query.device),
    )
    func(
        workspace_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        ctypes.cast(block_tables.data_ptr(), ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(context_lens.data_ptr(), ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(k_scale.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(v_scale.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        scale_value,
        max_blocks,
        max_partitions,
        sequences,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        kv_seq_stride,
        stream,
    )


__all__ = ["paged_attention_stage1"]

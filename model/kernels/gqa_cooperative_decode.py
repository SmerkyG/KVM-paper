"""Lazy loader for gfx942 shared-page cooperative decode leaf kernels."""

from __future__ import annotations

import ctypes
import fcntl
import functools
import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import torch


_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_FUNCTION = None


@functools.lru_cache(maxsize=None)
def gqa_cooperative_decode_available(device_index: int) -> bool:
    if torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(device_index)
    architecture = str(getattr(properties, "gcnArchName", ""))
    return architecture.split(":", 1)[0] == "gfx942"


def _build_library() -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "csrc/gqa_cooperative_decode/gqa_cooperative_decode.cu"
    )
    build_dir = Path(tempfile.gettempdir()) / "lod_gqa_cooperative_decode"
    build_dir.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library = build_dir / f"libgqa_cooperative_decode_{source_digest}.so"
    if not library.exists():
        lock_path = build_dir / f".{source_digest}.lock"
        with lock_path.open("w") as build_lock:
            fcntl.flock(build_lock.fileno(), fcntl.LOCK_EX)
            if not library.exists():
                rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
                temporary_library = library.with_name(
                    f".{library.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                try:
                    subprocess.run(
                        [
                            str(rocm / "bin/hipcc"),
                            "--offload-arch=gfx942",
                            "-O3",
                            "-fPIC",
                            "-shared",
                            str(source),
                            "-o",
                            str(temporary_library),
                        ],
                        check=True,
                    )
                    os.replace(temporary_library, library)
                finally:
                    temporary_library.unlink(missing_ok=True)
    return library


def _function():
    global _LIBRARY, _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION
    with _LOCK:
        if _FUNCTION is not None:
            return _FUNCTION
        library = ctypes.CDLL(str(_build_library()))
        function = library.launch_gqa_cooperative_decode
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int] * 17 + [
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        _LIBRARY = library
        _FUNCTION = function
        return function


def gqa_cooperative_decode(
    q: torch.Tensor,
    cache_indices: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    directory_values: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    quantized_q_scratch: torch.Tensor | None = None,
    query_scale_scratch: torch.Tensor | None = None,
    page_indices: torch.Tensor | None = None,
    page_k_scales: torch.Tensor | None = None,
    page_v_scales: torch.Tensor | None = None,
    scale_log2: float,
    page_lookup_mode: int,
    route_splits: int,
    adaptive_splits: bool = False,
    speculative_steps: int = 1,
    gqa_head_group_size: int | None = None,
    aggregate_routes: bool = False,
) -> None:
    """Launch a shared-page H=256, top-eight gfx942 leaf kernel."""
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(page_k.size(1))
    if speculative_steps not in (1, 2) or batch % speculative_steps:
        raise ValueError("HIP cooperative decode has invalid speculative geometry")
    kv_group_size = query_heads // kv_heads
    if gqa_head_group_size is None:
        gqa_head_group_size = kv_group_size
    indexed = page_indices is not None
    page_shape = page_indices if indexed else page_k
    supported_geometry = (speculative_steps, kv_group_size) in {(1, 4), (2, 6)}
    if (
        query_len != 1
        or head_dim != 256
        or query_heads != kv_heads * kv_group_size
        or not supported_geometry
    ):
        raise ValueError(
            "HIP cooperative decode requires H=256 and either GQA4 decode "
            "or two-position GQA6 verification"
        )
    invalid_subgroup = (
        speculative_steps == 1 and gqa_head_group_size != 4
    ) or (
        speculative_steps == 2 and gqa_head_group_size not in (2, 3, 6)
    )
    if kv_group_size % gqa_head_group_size or invalid_subgroup:
        raise ValueError("HIP cooperative decode has an unsupported GQA subgroup")
    if page_lookup_mode not in {-1, 0}:
        raise ValueError("HIP cooperative decode supports direct page directories")
    if route_splits not in {4, 8, 16, 32}:
        raise ValueError("HIP cooperative decode requires 4, 8, 16, or 32 splits")
    int8_storage = page_k.dtype == torch.int8 or page_v.dtype == torch.int8
    if q.dtype != torch.bfloat16:
        raise ValueError("HIP cooperative decode requires BF16 queries")
    if int8_storage:
        if page_k.dtype != torch.int8 or page_v.dtype != torch.int8:
            raise ValueError("HIP cooperative INT8 decode requires INT8 K and V")
        if page_k_scales is None or page_v_scales is None:
            raise ValueError("HIP cooperative INT8 decode requires K/V scales")
        if tuple(page_k_scales.shape) != tuple(page_k.shape[:-1]):
            raise ValueError("HIP cooperative INT8 K scales have the wrong shape")
        if tuple(page_v_scales.shape) != tuple(page_v.shape[:-1]):
            raise ValueError("HIP cooperative INT8 V scales have the wrong shape")
        if (
            page_k_scales.dtype != torch.bfloat16
            or page_v_scales.dtype != torch.bfloat16
        ):
            raise ValueError("HIP cooperative INT8 decode requires BF16 scales")
        required_query_bytes = batch * query_heads * head_dim
        if quantized_q_scratch is None or query_scale_scratch is None:
            raise ValueError("HIP cooperative INT8 decode requires query scratch")
        if quantized_q_scratch.numel() * quantized_q_scratch.element_size() < (
            required_query_bytes
        ):
            raise ValueError("HIP cooperative INT8 query scratch is too small")
        if (
            query_scale_scratch.dtype != torch.float32
            or query_scale_scratch.numel() < batch * query_heads
        ):
            raise ValueError("HIP cooperative INT8 query-scale scratch is invalid")
    elif page_k.dtype != torch.bfloat16 or page_v.dtype != torch.bfloat16:
        raise ValueError("HIP cooperative decode requires BF16 or INT8 K/V")
    if quantized_q_scratch is None:
        quantized_q_scratch = partial_out
    if query_scale_scratch is None:
        query_scale_scratch = partial_lse
    if not all(
        tensor.is_cuda
        for tensor in (
            q,
            quantized_q_scratch,
            query_scale_scratch,
            cache_indices,
            page_k,
            page_v,
            *(tuple() if page_k_scales is None else (page_k_scales,)),
            *(tuple() if page_v_scales is None else (page_v_scales,)),
            *(tuple() if page_indices is None else (page_indices,)),
            slot_pages,
            directory_values,
            slot_lengths,
            top_slots,
            partial_out,
            partial_lse,
        )
    ):
        raise ValueError("HIP cooperative decode tensors must be on the GPU")
    expected_out = (
        (batch, query_heads, 32, head_dim)
        if aggregate_routes
        else (batch, query_heads, 8, route_splits, head_dim)
    )
    expected_lse = (
        (batch, query_heads, 32)
        if aggregate_routes
        else (batch, query_heads, 8, route_splits)
    )
    if tuple(partial_out.shape) != expected_out:
        raise ValueError("HIP cooperative output workspace has the wrong shape")
    if tuple(partial_lse.shape) != expected_lse:
        raise ValueError("HIP cooperative LSE workspace has the wrong shape")
    error = _function()(
        q.data_ptr(),
        quantized_q_scratch.data_ptr(),
        query_scale_scratch.data_ptr(),
        cache_indices.data_ptr(),
        page_k.data_ptr(),
        page_v.data_ptr(),
        (page_k_scales if page_k_scales is not None else page_k).data_ptr(),
        (page_v_scales if page_v_scales is not None else page_v).data_ptr(),
        (page_indices if page_indices is not None else page_k).data_ptr(),
        slot_pages.data_ptr(),
        directory_values.data_ptr(),
        slot_lengths.data_ptr(),
        top_slots.data_ptr(),
        partial_out.data_ptr(),
        partial_lse.data_ptr(),
        batch,
        query_heads,
        kv_heads,
        int(page_k.size(0)),
        int(page_shape.size(2)),
        int(page_k.size(2)) if indexed else 1,
        int(slot_pages.size(2)),
        int(slot_pages.size(3)),
        int(directory_values.size(2)),
        page_lookup_mode,
        route_splits,
        int(adaptive_splits),
        int(indexed),
        int(int8_storage),
        speculative_steps,
        gqa_head_group_size,
        int(aggregate_routes),
        top_slots.stride(0),
        top_slots.stride(1),
        scale_log2,
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"HIP cooperative decode launch failed with error {error}")

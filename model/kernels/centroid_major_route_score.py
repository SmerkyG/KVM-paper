"""Lazy loader for the gfx942 D=256/GQA=4-or-6 centroid-major route scorer."""

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
_QUERY_WAVE_FUNCTION = None
_FIXED_FUNCTION = None


@functools.lru_cache(maxsize=None)
def centroid_major_route_score_available(device_index: int) -> bool:
    if torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(device_index)
    architecture = str(getattr(properties, "gcnArchName", ""))
    return architecture.split(":", 1)[0] == "gfx942"


def _build_library() -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "csrc/centroid_major_route_score/centroid_major_route_score.cu"
    )
    build_dir = Path(tempfile.gettempdir()) / "lod_centroid_major_route_score"
    build_dir.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library = build_dir / f"libcentroid_major_route_score_{source_digest}.so"
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
        function = library.launch_centroid_major_route_score
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 7 + [
            ctypes.c_longlong
        ] * 6 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
        _LIBRARY = library
        _FUNCTION = function
        return function


def _fixed_function():
    global _LIBRARY, _FIXED_FUNCTION
    if _FIXED_FUNCTION is not None:
        return _FIXED_FUNCTION
    with _LOCK:
        if _FIXED_FUNCTION is not None:
            return _FIXED_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_centroid_major_route_score_fixed_prepare
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 21 + [ctypes.c_int] * 7 + [
            ctypes.c_longlong
        ] * 6 + [ctypes.c_int] * 3 + [ctypes.c_longlong] * 7 + [
            ctypes.c_int
        ] * 10 + [ctypes.c_float, ctypes.c_void_p]
        _LIBRARY = library
        _FIXED_FUNCTION = function
        return function


def _query_wave_function():
    global _LIBRARY, _QUERY_WAVE_FUNCTION
    if _QUERY_WAVE_FUNCTION is not None:
        return _QUERY_WAVE_FUNCTION
    with _LOCK:
        if _QUERY_WAVE_FUNCTION is not None:
            return _QUERY_WAVE_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_query_wave_route_score
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 7 + [
            ctypes.c_longlong
        ] * 6 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
        _LIBRARY = library
        _QUERY_WAVE_FUNCTION = function
        return function


def centroid_major_route_score(
    q: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    cache_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    state_len: int,
    protected_len: int = 0,
    max_leaf_tokens: int = 0,
    mean_before_dot: bool = True,
    scale: float,
) -> None:
    """Emit eight candidates per 32-centroid D=256/GQA=4-or-6 partition."""
    batch, query_heads, query_len, head_dim = q.shape
    cache_batches, kv_heads, state_capacity, key_dim = state_k.shape
    if query_len != 1 or head_dim != 256 or key_dim != 256:
        raise ValueError("centroid-major route score requires decode D=256")
    if query_heads not in (kv_heads * 4, kv_heads * 6):
        raise ValueError("centroid-major route score requires GQA=4 or GQA=6")
    if q.dtype != torch.bfloat16 or state_k.dtype != torch.bfloat16:
        raise TypeError("centroid-major route score requires BF16 Q and K sums")
    if counts.dtype != torch.float32 or cache_indices.dtype != torch.int64:
        raise TypeError("centroid-major route score requires FP32 counts and INT64 indices")
    if candidate_scores.dtype != torch.float32:
        raise TypeError("centroid-major candidate scores must be FP32")
    if candidate_indices.dtype != torch.int64:
        raise TypeError("centroid-major candidate indices must be INT64")
    if candidate_scores.shape != candidate_indices.shape:
        raise ValueError("centroid-major candidate buffers must have matching shapes")
    if (
        candidate_scores.ndim != 3
        or candidate_scores.shape[0] != batch * query_heads
        or candidate_scores.shape[2] != 8
    ):
        raise ValueError("candidate buffers must have shape [B*QH, groups, 8]")
    if not q.is_contiguous():
        raise ValueError("centroid-major route score requires contiguous Q")
    if not all(
        tensor.is_cuda
        for tensor in (
            q,
            state_k,
            counts,
            cache_indices,
            candidate_scores,
            candidate_indices,
        )
    ):
        raise ValueError("centroid-major route score tensors must be on the GPU")
    max_groups = int(candidate_scores.size(1))
    error = _function()(
        q.data_ptr(),
        state_k.data_ptr(),
        counts.data_ptr(),
        cache_indices.data_ptr(),
        candidate_scores.data_ptr(),
        candidate_indices.data_ptr(),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        int(state_len),
        max_groups,
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        int(protected_len),
        int(max_leaf_tokens),
        int(mean_before_dot),
        float(scale),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"centroid-major route-score launch failed with error {error}")


def query_wave_route_score(
    q: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    cache_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    state_len: int,
    protected_len: int = 0,
    max_leaf_tokens: int = 0,
    mean_before_dot: bool = True,
    scale: float,
) -> None:
    """Emit block candidates with one wave assigned to each GQA query."""
    batch, query_heads, query_len, head_dim = q.shape
    cache_batches, kv_heads, state_capacity, key_dim = state_k.shape
    if query_len != 1 or head_dim != 256 or key_dim != 256:
        raise ValueError("query-wave route score requires decode D=256")
    if query_heads != kv_heads * 4:
        raise ValueError("query-wave route score requires GQA=4")
    if q.dtype != torch.bfloat16 or state_k.dtype != torch.bfloat16:
        raise TypeError("query-wave route score requires BF16 Q and K sums")
    if counts.dtype != torch.float32 or cache_indices.dtype != torch.int64:
        raise TypeError("query-wave route score requires FP32 counts and INT64 indices")
    if candidate_scores.dtype != torch.float32:
        raise TypeError("query-wave candidate scores must be FP32")
    if candidate_indices.dtype != torch.int64:
        raise TypeError("query-wave candidate indices must be INT64")
    if candidate_scores.shape != candidate_indices.shape:
        raise ValueError("query-wave candidate buffers must have matching shapes")
    if (
        candidate_scores.ndim != 3
        or candidate_scores.shape[0] != batch * query_heads
        or candidate_scores.shape[2] != 8
    ):
        raise ValueError("candidate buffers must have shape [B*QH, groups, 8]")
    if not q.is_contiguous():
        raise ValueError("query-wave route score requires contiguous Q")
    if not all(
        tensor.is_cuda
        for tensor in (
            q,
            state_k,
            counts,
            cache_indices,
            candidate_scores,
            candidate_indices,
        )
    ):
        raise ValueError("query-wave route score tensors must be on the GPU")
    max_groups = int(candidate_scores.size(1))
    error = _query_wave_function()(
        q.data_ptr(),
        state_k.data_ptr(),
        counts.data_ptr(),
        cache_indices.data_ptr(),
        candidate_scores.data_ptr(),
        candidate_indices.data_ptr(),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        int(state_len),
        max_groups,
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        int(protected_len),
        int(max_leaf_tokens),
        int(mean_before_dot),
        float(scale),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"query-wave route-score launch failed with error {error}")


def centroid_major_route_score_fixed_prepare(
    q: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    cache_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    local_lens: torch.Tensor,
    fixed_lengths: torch.Tensor,
    context_lens: torch.Tensor,
    launch_lens: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    arena_k: torch.Tensor,
    arena_v: torch.Tensor,
    execution_marker: torch.Tensor,
    previous_cache_rows: torch.Tensor,
    previous_counts: torch.Tensor,
    previous_slots: torch.Tensor,
    fixed_slot_offsets: torch.Tensor,
    active_mask: torch.Tensor,
    active_blocks: torch.Tensor,
    *,
    state_len: int,
    protected_len: int = 0,
    max_leaf_tokens: int = 0,
    mean_before_dot: bool = True,
    union_capacity: int,
    local_offset: int,
    local_capacity: int,
    local_limit: int,
    sink_len: int,
    leaf_begin: int,
    mask_capacity: int,
    tile_size: int,
    include_new: bool,
    separate_local_sink: bool,
    scale: float,
) -> None:
    """Score routes while maintaining the persistent fixed attention mask."""
    batch, query_heads, query_len, head_dim = q.shape
    cache_batches, kv_heads, state_capacity, key_dim = state_k.shape
    if query_len != 1 or head_dim != 256 or key_dim != 256:
        raise ValueError("centroid-major fixed preparation requires decode D=256")
    if query_heads not in (kv_heads * 4, kv_heads * 6):
        raise ValueError(
            "centroid-major fixed preparation requires GQA=4 or GQA=6"
        )
    if q.dtype != torch.bfloat16 or state_k.dtype != torch.bfloat16:
        raise TypeError("centroid-major fixed preparation requires BF16 Q and K")
    if counts.dtype != torch.float32 or cache_indices.dtype != torch.int64:
        raise TypeError("centroid-major fixed preparation has invalid state metadata")
    if candidate_scores.dtype != torch.float32 or candidate_indices.dtype != torch.int64:
        raise TypeError("centroid-major fixed candidate buffers have invalid dtypes")
    if candidate_scores.shape != candidate_indices.shape:
        raise ValueError("centroid-major fixed candidate buffers do not match")
    if candidate_scores.ndim != 3 or candidate_scores.shape[2] != 8:
        raise ValueError("fixed candidate buffers must have shape [B*QH, groups, 8]")
    int32_tensors = (
        local_lens,
        fixed_lengths,
        context_lens,
        launch_lens,
        execution_marker,
        previous_cache_rows,
        previous_counts,
        previous_slots,
        fixed_slot_offsets,
    )
    if any(tensor.dtype != torch.int32 for tensor in int32_tensors):
        raise TypeError("centroid-major fixed preparation requires INT32 metadata")
    if active_mask.dtype != torch.uint8 or active_blocks.dtype != torch.uint8:
        raise TypeError("centroid-major fixed masks must be UINT8")
    if new_k.dtype != torch.bfloat16 or new_v.dtype != torch.bfloat16:
        raise TypeError("centroid-major fixed preparation requires BF16 new K/V")
    if arena_k.dtype != torch.bfloat16 or arena_v.dtype != torch.bfloat16:
        raise TypeError("centroid-major fixed preparation requires a BF16 arena")
    tensors = (
        q,
        state_k,
        counts,
        cache_indices,
        candidate_scores,
        candidate_indices,
        *int32_tensors,
        new_k,
        new_v,
        arena_k,
        arena_v,
        active_mask,
        active_blocks,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("centroid-major fixed preparation tensors must be on GPU")
    max_groups = int(candidate_scores.size(1))
    error = _fixed_function()(
        q.data_ptr(),
        state_k.data_ptr(),
        counts.data_ptr(),
        cache_indices.data_ptr(),
        candidate_scores.data_ptr(),
        candidate_indices.data_ptr(),
        local_lens.data_ptr(),
        fixed_lengths.data_ptr(),
        context_lens.data_ptr(),
        launch_lens.data_ptr(),
        new_k.data_ptr(),
        new_v.data_ptr(),
        arena_k.data_ptr(),
        arena_v.data_ptr(),
        execution_marker.data_ptr(),
        previous_cache_rows.data_ptr(),
        previous_counts.data_ptr(),
        previous_slots.data_ptr(),
        fixed_slot_offsets.data_ptr(),
        active_mask.data_ptr(),
        active_blocks.data_ptr(),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        int(state_len),
        max_groups,
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        int(protected_len),
        int(max_leaf_tokens),
        int(mean_before_dot),
        new_k.stride(0),
        new_k.stride(1),
        new_v.stride(0),
        new_v.stride(1),
        fixed_slot_offsets.stride(1),
        active_mask.stride(0),
        active_blocks.stride(0),
        int(union_capacity),
        int(local_offset),
        int(local_capacity),
        int(local_limit),
        int(sink_len),
        int(leaf_begin),
        int(mask_capacity),
        int(tile_size),
        int(include_new),
        int(separate_local_sink),
        float(scale),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(
            f"centroid-major fixed preparation launch failed with error {error}"
        )


__all__ = [
    "centroid_major_route_score",
    "centroid_major_route_score_fixed_prepare",
    "centroid_major_route_score_available",
    "query_wave_route_score",
]

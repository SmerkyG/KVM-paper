"""Lazy loader for the gfx942 D=128/GQA=16 coarse score control."""

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
_CANDIDATE_FUNCTION = None
_REDUCE_FUNCTION = None
_SCORES_LSE_FUNCTION = None
_REDUCE_LSE_FUNCTION = None
_MASS_UNION_FUNCTION = None
_INIT_MASS_UNION_FUNCTION = None
_PREDICTED_MASS_UNION_FUNCTION = None
_PREDICTED_MASS_UNION_NO_LSE_FUNCTION = None


@functools.lru_cache(maxsize=None)
def gqa16_coarse_score_available(device_index: int) -> bool:
    if torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(device_index)
    architecture = str(getattr(properties, "gcnArchName", ""))
    return architecture.split(":", 1)[0] == "gfx942"


def _build_library() -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "csrc/gqa16_coarse_score/gqa16_coarse_score.cu"
    )
    build_dir = Path(tempfile.gettempdir()) / "lod_gqa16_coarse_score"
    build_dir.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library = build_dir / f"libgqa16_coarse_score_{source_digest}.so"
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
        function = library.launch_gqa16_coarse_score
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6 + [
            ctypes.c_longlong
        ] * 6 + [ctypes.c_int] * 2 + [ctypes.c_float, ctypes.c_void_p]
        _LIBRARY = library
        _FUNCTION = function
        return function


def _candidate_function():
    global _LIBRARY, _CANDIDATE_FUNCTION
    if _CANDIDATE_FUNCTION is not None:
        return _CANDIDATE_FUNCTION
    with _LOCK:
        if _CANDIDATE_FUNCTION is not None:
            return _CANDIDATE_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_gqa16_coarse_candidates
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [
            ctypes.c_longlong
        ] * 6 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
        _LIBRARY = library
        _CANDIDATE_FUNCTION = function
        return function


def _reduce_function():
    global _LIBRARY, _REDUCE_FUNCTION
    if _REDUCE_FUNCTION is not None:
        return _REDUCE_FUNCTION
    with _LOCK:
        if _REDUCE_FUNCTION is not None:
            return _REDUCE_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_reduce_route_top8
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [
            ctypes.c_void_p
        ]
        _LIBRARY = library
        _REDUCE_FUNCTION = function
        return function


def _scores_lse_function():
    global _LIBRARY, _SCORES_LSE_FUNCTION
    if _SCORES_LSE_FUNCTION is not None:
        return _SCORES_LSE_FUNCTION
    with _LOCK:
        if _SCORES_LSE_FUNCTION is not None:
            return _SCORES_LSE_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_gqa16_coarse_scores_lse
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [
            ctypes.c_longlong
        ] * 6 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
        _LIBRARY = library
        _SCORES_LSE_FUNCTION = function
        return function


def _reduce_lse_function():
    global _LIBRARY, _REDUCE_LSE_FUNCTION
    if _REDUCE_LSE_FUNCTION is not None:
        return _REDUCE_LSE_FUNCTION
    with _LOCK:
        if _REDUCE_LSE_FUNCTION is not None:
            return _REDUCE_LSE_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_reduce_partition_lse
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [
            ctypes.c_void_p
        ]
        _LIBRARY = library
        _REDUCE_LSE_FUNCTION = function
        return function


def _mass_union_function():
    global _LIBRARY, _MASS_UNION_FUNCTION
    if _MASS_UNION_FUNCTION is not None:
        return _MASS_UNION_FUNCTION
    with _LOCK:
        if _MASS_UNION_FUNCTION is not None:
            return _MASS_UNION_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_mass_cutoff_union
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 6 + [
            ctypes.c_longlong
        ] * 3 + [ctypes.c_int] * 2 + [ctypes.c_float, ctypes.c_void_p]
        _LIBRARY = library
        _MASS_UNION_FUNCTION = function
        return function


def _init_mass_union_function():
    global _LIBRARY, _INIT_MASS_UNION_FUNCTION
    if _INIT_MASS_UNION_FUNCTION is not None:
        return _INIT_MASS_UNION_FUNCTION
    with _LOCK:
        if _INIT_MASS_UNION_FUNCTION is not None:
            return _INIT_MASS_UNION_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_init_mass_union
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 3 + [
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _LIBRARY = library
        _INIT_MASS_UNION_FUNCTION = function
        return function


def _predicted_mass_union_function():
    global _LIBRARY, _PREDICTED_MASS_UNION_FUNCTION
    if _PREDICTED_MASS_UNION_FUNCTION is not None:
        return _PREDICTED_MASS_UNION_FUNCTION
    with _LOCK:
        if _PREDICTED_MASS_UNION_FUNCTION is not None:
            return _PREDICTED_MASS_UNION_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_gqa16_predicted_mass_union
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 6 + [
            ctypes.c_longlong
        ] * 6 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        _LIBRARY = library
        _PREDICTED_MASS_UNION_FUNCTION = function
        return function


def _predicted_mass_union_no_lse_function():
    global _LIBRARY, _PREDICTED_MASS_UNION_NO_LSE_FUNCTION
    if _PREDICTED_MASS_UNION_NO_LSE_FUNCTION is not None:
        return _PREDICTED_MASS_UNION_NO_LSE_FUNCTION
    with _LOCK:
        if _PREDICTED_MASS_UNION_NO_LSE_FUNCTION is not None:
            return _PREDICTED_MASS_UNION_NO_LSE_FUNCTION
        library = _LIBRARY or ctypes.CDLL(str(_build_library()))
        function = library.launch_gqa16_predicted_mass_union_no_lse
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 6 + [
            ctypes.c_longlong
        ] * 6 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        _LIBRARY = library
        _PREDICTED_MASS_UNION_NO_LSE_FUNCTION = function
        return function


def gqa16_coarse_score(
    q: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    cache_indices: torch.Tensor,
    scores: torch.Tensor,
    *,
    state_len: int,
    protected_len: int = 0,
    max_leaf_tokens: int = 0,
    scale: float,
) -> None:
    """Materialize corrected D=128/GQA16 routing scores with HIP MFMA."""
    batch, query_heads, query_len, head_dim = q.shape
    cache_batches, kv_heads, state_capacity, key_dim = state_k.shape
    if query_len != 1 or head_dim != 128 or key_dim != 128:
        raise ValueError("HIP coarse score requires decode D=128")
    if query_heads != kv_heads * 16:
        raise ValueError("HIP coarse score requires GQA=16")
    if q.dtype != torch.bfloat16 or state_k.dtype != torch.bfloat16:
        raise TypeError("HIP coarse score requires BF16 Q and K sums")
    if counts.dtype != torch.float32:
        raise TypeError("HIP coarse score requires FP32 counts")
    if cache_indices.dtype != torch.int64:
        raise TypeError("HIP coarse score requires INT64 cache indices")
    if scores.dtype != torch.float32:
        raise TypeError("HIP coarse score requires FP32 output scores")
    if tuple(scores.shape) != (batch, query_heads, 1, state_capacity):
        raise ValueError("HIP coarse score output has the wrong shape")
    if not q.is_contiguous() or not scores.is_contiguous():
        raise ValueError("HIP coarse score requires contiguous Q and output")
    if not all(tensor.is_cuda for tensor in (q, state_k, counts, cache_indices, scores)):
        raise ValueError("HIP coarse score tensors must be on the GPU")
    error = _function()(
        q.data_ptr(),
        state_k.data_ptr(),
        counts.data_ptr(),
        cache_indices.data_ptr(),
        scores.data_ptr(),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        int(state_len),
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        int(protected_len),
        int(max_leaf_tokens),
        float(scale),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"HIP coarse score launch failed with error {error}")


def gqa16_coarse_candidates(
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
    scale: float,
) -> None:
    """Emit eight routing candidates per 256-centroid HIP partition."""
    batch, query_heads, query_len, head_dim = q.shape
    cache_batches, kv_heads, state_capacity, key_dim = state_k.shape
    if query_len != 1 or head_dim != 128 or key_dim != 128:
        raise ValueError("HIP coarse candidates require decode D=128")
    if query_heads != kv_heads * 16:
        raise ValueError("HIP coarse candidates require GQA=16")
    if q.dtype != torch.bfloat16 or state_k.dtype != torch.bfloat16:
        raise TypeError("HIP coarse candidates require BF16 Q and K sums")
    if counts.dtype != torch.float32 or cache_indices.dtype != torch.int64:
        raise TypeError("HIP coarse candidates require FP32 counts and INT64 indices")
    if candidate_scores.dtype != torch.float32 or candidate_indices.dtype != torch.int64:
        raise TypeError("HIP candidate buffers require FP32 scores and INT64 indices")
    if candidate_scores.shape != candidate_indices.shape:
        raise ValueError("HIP candidate score/index buffers must have matching shapes")
    if (
        candidate_scores.ndim != 3
        or candidate_scores.shape[0] != batch * query_heads
        or candidate_scores.shape[2] != 8
    ):
        raise ValueError("HIP candidate buffers must have shape [B*QH, segments, 8]")
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
        raise ValueError("HIP coarse candidate tensors must be on the GPU")
    max_segments = int(candidate_scores.size(1))
    error = _candidate_function()(
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
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        int(protected_len),
        int(max_leaf_tokens),
        max_segments,
        float(scale),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"HIP coarse candidate launch failed with error {error}")


def reduce_route_top8(
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    top_slots: torch.Tensor,
    top_scores: torch.Tensor,
    *,
    active_segments: int,
) -> None:
    """Reduce HIP partition candidates to a global top eight per query row."""
    if candidate_scores.shape != candidate_indices.shape:
        raise ValueError("HIP route candidate buffers do not match")
    if candidate_scores.ndim != 3 or candidate_scores.size(2) != 8:
        raise ValueError("HIP route candidates must have shape [rows, segments, 8]")
    rows, max_segments, _ = candidate_scores.shape
    if top_slots.numel() != rows * 8 or top_scores.numel() != rows * 8:
        raise ValueError("HIP route top-eight outputs have the wrong size")
    if candidate_scores.dtype != torch.float32 or top_scores.dtype != torch.float32:
        raise TypeError("HIP route scores must be FP32")
    if candidate_indices.dtype != torch.int64 or top_slots.dtype != torch.int64:
        raise TypeError("HIP route indices must be INT64")
    error = _reduce_function()(
        candidate_scores.data_ptr(),
        candidate_indices.data_ptr(),
        top_slots.data_ptr(),
        top_scores.data_ptr(),
        rows,
        int(active_segments),
        max_segments,
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"HIP route top-eight reduction failed with error {error}")


def gqa16_coarse_scores_lse(
    q: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    cache_indices: torch.Tensor,
    scores: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    state_len: int,
    protected_len: int = 0,
    max_leaf_tokens: int = 0,
    scale: float,
) -> None:
    """Materialize scores and one LSE value per 256-centroid partition."""
    batch, query_heads, query_len, head_dim = q.shape
    cache_batches, kv_heads, state_capacity, key_dim = state_k.shape
    if query_len != 1 or head_dim != 128 or key_dim != 128:
        raise ValueError("HIP score/LSE requires decode D=128")
    if query_heads != kv_heads * 16:
        raise ValueError("HIP score/LSE requires GQA=16")
    if scores.shape != (batch, query_heads, 1, state_capacity):
        raise ValueError("HIP score table has the wrong shape")
    if partial_lse.ndim != 2 or partial_lse.size(0) != batch * query_heads:
        raise ValueError("HIP partial LSE must have shape [B*QH, segments]")
    if q.dtype != torch.bfloat16 or state_k.dtype != torch.bfloat16:
        raise TypeError("HIP score/LSE requires BF16 Q and state K")
    if counts.dtype != torch.float32 or scores.dtype != torch.float32 or partial_lse.dtype != torch.float32:
        raise TypeError("HIP score/LSE buffers must be FP32")
    max_segments = int(partial_lse.size(1))
    error = _scores_lse_function()(
        q.data_ptr(),
        state_k.data_ptr(),
        counts.data_ptr(),
        cache_indices.data_ptr(),
        scores.data_ptr(),
        partial_lse.data_ptr(),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        int(state_len),
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        int(protected_len),
        int(max_leaf_tokens),
        max_segments,
        float(scale),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"HIP score/LSE launch failed with error {error}")


def reduce_partition_lse(
    partial_lse: torch.Tensor,
    full_lse: torch.Tensor,
    sequence_epochs: torch.Tensor,
    union_counts: torch.Tensor,
    union_token_counts: torch.Tensor,
    *,
    query_heads: int,
    kv_heads: int,
    active_segments: int,
) -> None:
    """Reduce partition LSEs and initialize the mass-union workspace."""
    rows, max_segments = partial_lse.shape
    if full_lse.numel() != rows:
        raise ValueError("HIP full LSE output has the wrong size")
    if not all(tensor.dtype == torch.float32 for tensor in (partial_lse, full_lse)):
        raise TypeError("HIP LSE buffers must be FP32")
    if not all(
        tensor.dtype == torch.int32
        for tensor in (sequence_epochs, union_counts, union_token_counts)
    ):
        raise TypeError("HIP mass-union counters must be INT32")
    error = _reduce_lse_function()(
        partial_lse.data_ptr(),
        full_lse.data_ptr(),
        sequence_epochs.data_ptr(),
        union_counts.data_ptr(),
        union_token_counts.data_ptr(),
        rows,
        int(query_heads),
        int(kv_heads),
        int(active_segments),
        max_segments,
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"HIP partition LSE reduction failed with error {error}")


def mass_cutoff_union(
    scores: torch.Tensor,
    full_lse: torch.Tensor,
    counts: torch.Tensor,
    cache_indices: torch.Tensor,
    seen_stamps: torch.Tensor,
    sequence_epochs: torch.Tensor,
    union_counts: torch.Tensor,
    union_slots: torch.Tensor,
    *,
    state_len: int,
    mass_fraction: float,
    protected_len: int = 0,
    max_leaf_tokens: int = 0,
) -> None:
    """Compact the GQA union of centroids above a per-head mass cutoff."""
    import math

    batch, query_heads, query_len, state_capacity = scores.shape
    if query_len != 1 or query_heads % 16:
        raise ValueError("HIP mass union requires decode GQA=16 scores")
    kv_heads = query_heads // 16
    sequences = batch * kv_heads
    if seen_stamps.shape != (sequences, state_capacity):
        raise ValueError("HIP mass-union stamp buffer has the wrong shape")
    if union_slots.ndim != 2 or union_slots.size(0) != sequences:
        raise ValueError("HIP mass-union slot buffer has the wrong shape")
    if full_lse.numel() != batch * query_heads:
        raise ValueError("HIP mass-union LSE buffer has the wrong size")
    if not 0.0 < mass_fraction < 1.0:
        raise ValueError("mass fraction must be in (0, 1)")
    error = _mass_union_function()(
        scores.data_ptr(),
        full_lse.data_ptr(),
        counts.data_ptr(),
        cache_indices.data_ptr(),
        seen_stamps.data_ptr(),
        sequence_epochs.data_ptr(),
        union_counts.data_ptr(),
        union_slots.data_ptr(),
        batch,
        query_heads,
        kv_heads,
        state_capacity,
        int(state_len),
        int(union_slots.size(1)),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        int(protected_len),
        int(max_leaf_tokens),
        float(math.log(mass_fraction)),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"HIP mass-cutoff union launch failed with error {error}")


def init_mass_union(
    sequence_epochs: torch.Tensor,
    union_counts: torch.Tensor,
    union_token_counts: torch.Tensor,
) -> None:
    """Advance stamps and clear route/list counters without reducing LSE."""
    if not (
        sequence_epochs.shape == union_counts.shape == union_token_counts.shape
    ):
        raise ValueError("HIP mass-union counter shapes do not match")
    if not all(
        tensor.dtype == torch.int32
        for tensor in (sequence_epochs, union_counts, union_token_counts)
    ):
        raise TypeError("HIP mass-union counters must be INT32")
    error = _init_mass_union_function()(
        sequence_epochs.data_ptr(),
        union_counts.data_ptr(),
        union_token_counts.data_ptr(),
        int(sequence_epochs.numel()),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"HIP mass-union init launch failed with error {error}")


def gqa16_predicted_mass_union(
    q: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    cache_indices: torch.Tensor,
    predicted_thresholds: torch.Tensor,
    partial_lse: torch.Tensor,
    seen_stamps: torch.Tensor,
    sequence_epochs: torch.Tensor,
    union_counts: torch.Tensor,
    union_slots: torch.Tensor,
    *,
    state_len: int,
    protected_len: int = 0,
    max_leaf_tokens: int = 0,
    scale: float,
    emit_lse: bool = True,
) -> None:
    """Score current centroids and route against a retained absolute cutoff.

    The current routes are not lagged. The only prediction is the per-head
    absolute score threshold, normally the preceding token's LSE plus
    ``log(mass_fraction)``. Current partition LSEs are emitted so a later or
    overlapped refresh can update that threshold.
    """
    batch, query_heads, query_len, head_dim = q.shape
    cache_batches, kv_heads, state_capacity, key_dim = state_k.shape
    if query_len != 1 or head_dim != 128 or key_dim != 128:
        raise ValueError("HIP predicted mass routing requires decode D=128")
    if query_heads != kv_heads * 16:
        raise ValueError("HIP predicted mass routing requires GQA=16")
    if tuple(predicted_thresholds.shape) != (batch, query_heads, 1):
        raise ValueError("predicted mass thresholds have the wrong shape")
    if partial_lse.ndim != 2 or partial_lse.size(0) != batch * query_heads:
        raise ValueError("predicted mass partial LSE has the wrong shape")
    sequences = batch * kv_heads
    if seen_stamps.shape != (sequences, state_capacity):
        raise ValueError("predicted mass stamp buffer has the wrong shape")
    if union_slots.ndim != 2 or union_slots.size(0) != sequences:
        raise ValueError("predicted mass union slots have the wrong shape")
    if not all(
        tensor.dtype == torch.int32
        for tensor in (seen_stamps, sequence_epochs, union_counts, union_slots)
    ):
        raise TypeError("predicted mass union metadata must be INT32")
    function = (
        _predicted_mass_union_function()
        if emit_lse
        else _predicted_mass_union_no_lse_function()
    )
    error = function(
        q.data_ptr(),
        state_k.data_ptr(),
        counts.data_ptr(),
        cache_indices.data_ptr(),
        predicted_thresholds.data_ptr(),
        partial_lse.data_ptr(),
        seen_stamps.data_ptr(),
        sequence_epochs.data_ptr(),
        union_counts.data_ptr(),
        union_slots.data_ptr(),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        int(state_len),
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        int(protected_len),
        int(max_leaf_tokens),
        int(partial_lse.size(1)),
        int(union_slots.size(1)),
        float(scale),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"HIP predicted mass route launch failed with error {error}")

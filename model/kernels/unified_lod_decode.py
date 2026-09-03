"""gfx942 retained-mass LOD decode prototypes.

The universal variant performs routing, cross-centroid list packing, indexed
attention, reduction, and retained-LSE refresh in one resource allocation. The
producer/consumer variant instead publishes N=128 tiles from a route-only HIP
kernel to bounded attention-only HIP consumers on another stream. Both first
specializations deliberately match Qwen3.5-0.8B: BF16, D=256, GQA=4, and the
two-level page directory.
"""

from __future__ import annotations

import ctypes
import fcntl
import functools
import hashlib
import math
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import torch


_LOCK = threading.Lock()
_LIBRARIES: dict[str, ctypes.CDLL] = {}
_FUNCTIONS: dict[str, object] = {}


@functools.lru_cache(maxsize=None)
def unified_lod_decode_available(device_index: int) -> bool:
    if torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(device_index)
    architecture = str(getattr(properties, "gcnArchName", ""))
    return architecture.split(":", 1)[0] == "gfx942"


@functools.lru_cache(maxsize=None)
def _build_library(variant: str = "full") -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "csrc/unified_lod_decode/unified_lod_decode.cu"
    )
    aiter_root = Path(
        os.environ.get("AITER_SOURCE_DIR", "/home/dan/subusers/agent/vendor/aiter")
    )
    include = aiter_root / "csrc/include"
    build_dir = Path(tempfile.gettempdir()) / "lod_unified_lod_decode"
    build_dir.mkdir(parents=True, exist_ok=True)
    if variant not in ("full", "route", "consumer"):
        raise ValueError(f"unknown unified LOD HIP variant: {variant}")
    digest = hashlib.sha256(
        source.read_bytes() + b"\0" + variant.encode("ascii")
    ).hexdigest()[:16]
    library = build_dir / f"libunified_lod_decode_{digest}_{variant}.so"
    if not library.exists():
        lock_path = build_dir / f".{digest}.lock"
        with lock_path.open("w") as build_lock:
            fcntl.flock(build_lock.fileno(), fcntl.LOCK_EX)
            if not library.exists():
                rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
                temporary = library.with_name(
                    f".{library.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                try:
                    command = [
                            str(rocm / "bin/hipcc"),
                            "--offload-arch=gfx942",
                            "-O3",
                            "-fPIC",
                            "-shared",
                            f"-I{include}",
                    ]
                    if variant == "route":
                        command.append("-DLOD_ROUTE_ONLY=1")
                    elif variant == "consumer":
                        command.append("-DLOD_CONSUMER_ONLY=1")
                    command.extend(
                        [
                            str(source),
                            "-o",
                            str(temporary),
                        ]
                    )
                    subprocess.run(command, check=True)
                    os.replace(temporary, library)
                finally:
                    temporary.unlink(missing_ok=True)
    return library


def _function(variant: str = "full"):
    if variant in _FUNCTIONS:
        return _FUNCTIONS[variant]
    with _LOCK:
        if variant in _FUNCTIONS:
            return _FUNCTIONS[variant]
        library = ctypes.CDLL(str(_build_library(variant)))
        function = library.launch_unified_lod_decode
        function.restype = ctypes.c_int
        function.argtypes = (
            [ctypes.c_void_p] * 24
            + [ctypes.c_int] * 24
            + [ctypes.c_float] * 2
            + [ctypes.c_void_p]
        )
        _LIBRARIES[variant] = library
        _FUNCTIONS[variant] = function
        return function


def new_unified_lod_decode_buffers(
    *,
    sequences: int,
    index_capacity: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate graph-stable scratch for the N=256 fused launch."""
    max_tiles = math.ceil(index_capacity / 256)
    return {
        "unified_stream_counts": torch.zeros(
            sequences, dtype=torch.int32, device=device
        ),
        "unified_opened_counts": torch.zeros(
            sequences, dtype=torch.int32, device=device
        ),
        "unified_producer_done": torch.zeros(
            sequences, dtype=torch.int32, device=device
        ),
        "unified_tile_ready": torch.zeros(
            sequences, max_tiles, dtype=torch.int32, device=device
        ),
        "unified_overflow_flags": torch.zeros(
            sequences, dtype=torch.int32, device=device
        ),
        "unified_packed_indices": torch.empty(
            sequences, index_capacity, dtype=torch.int32, device=device
        ),
        # Padded M=16 is intentional: it preserves the OPUS/AITER MFMA layout
        # while only the first four rows are reduced into model output.
        "unified_partial_out": torch.empty(
            sequences,
            max_tiles,
            16,
            256,
            dtype=torch.float32,
            device=device,
        ),
        "unified_partial_max": torch.empty(
            sequences, max_tiles, 16, dtype=torch.float32, device=device
        ),
        "unified_partial_denominator": torch.empty(
            sequences, max_tiles, 16, dtype=torch.float32, device=device
        ),
    }


def new_producer_consumer_lod_decode_buffers(
    *,
    sequences: int,
    index_capacity: int,
    query_heads_per_kv: int,
    consumer_segments: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate graph-stable route queues and bounded-consumer scratch."""
    if query_heads_per_kv != 4:
        raise ValueError("producer/consumer prototype currently requires GQA=4")
    max_tiles = math.ceil(index_capacity / 128)
    # Route-only compilation ignores the partials; the attention-only
    # consumers write one padded M=16 online-softmax partial per consumer.
    return {
        "unified_stream_counts": torch.zeros(
            sequences, dtype=torch.int32, device=device
        ),
        "unified_opened_counts": torch.zeros(
            sequences, dtype=torch.int32, device=device
        ),
        "unified_producer_done": torch.zeros(
            sequences, dtype=torch.int32, device=device
        ),
        "unified_tile_ready": torch.zeros(
            sequences, max_tiles, dtype=torch.int32, device=device
        ),
        "unified_overflow_flags": torch.zeros(
            sequences, dtype=torch.int32, device=device
        ),
        "unified_packed_indices": torch.empty(
            sequences, index_capacity, dtype=torch.int32, device=device
        ),
        "unified_partial_out": torch.empty(
            sequences,
            consumer_segments,
            16,
            256,
            dtype=torch.float32,
            device=device,
        ),
        "unified_partial_max": torch.empty(
            sequences,
            consumer_segments,
            16,
            dtype=torch.float32,
            device=device,
        ),
        "unified_partial_denominator": torch.empty(
            sequences,
            consumer_segments,
            16,
            dtype=torch.float32,
            device=device,
        ),
    }


def unified_lod_decode(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_indices: torch.Tensor,
    local_lens: torch.Tensor,
    counts: torch.Tensor,
    slot_pages: torch.Tensor,
    directory_values: torch.Tensor,
    slot_lengths: torch.Tensor,
    page_indices: torch.Tensor,
    arena_k: torch.Tensor,
    arena_v: torch.Tensor,
    arena_bias: torch.Tensor,
    previous_total_lse: torch.Tensor,
    output: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    state_len: int,
    local_limit: int,
    sink_len: int,
    protected_len: int,
    max_leaf_tokens: int,
    open_capacity: int,
    leaf_offset: int,
    local_offset: int,
    sink_offset: int,
    coarse_offset: int,
    scale: float,
    mass_fraction: float,
    execution_mode: int = 0,
    kernel_variant: str | None = None,
) -> None:
    """Launch the Qwen-geometry unified LOD decoder."""
    batch, query_heads, query_len, head_dim = q.shape
    cache_batches, kv_heads, state_capacity, count_width = counts.shape
    if query_len != 1 or head_dim != 256 or query_heads != kv_heads * 4:
        raise ValueError("unified HIP decode requires D=256 and GQA=4")
    if count_width != 1:
        raise ValueError("unified HIP decode requires scalar centroid counts")
    if q.dtype != torch.bfloat16 or new_k.dtype != q.dtype or new_v.dtype != q.dtype:
        raise TypeError("unified HIP decode requires BF16 Q/K/V")
    if arena_k.dtype != q.dtype or arena_v.dtype != q.dtype:
        raise TypeError("unified HIP arena must be BF16")
    if arena_bias.dtype != torch.float16:
        raise TypeError("unified HIP arena bias must be FP16")
    if cache_indices.dtype != torch.int64:
        raise TypeError("unified HIP cache indices must be INT64")
    if local_lens.dtype != torch.int32:
        raise TypeError("unified HIP local lengths must be INT32")
    if counts.dtype != torch.float32 or previous_total_lse.dtype != torch.float32:
        raise TypeError("unified HIP counts and retained LSE must be FP32")
    for tensor in (slot_pages, directory_values, slot_lengths, page_indices):
        if tensor.dtype != torch.int32:
            raise TypeError("unified HIP page metadata must be INT32")
    if tuple(new_k.shape) != (batch, kv_heads, head_dim) or tuple(new_v.shape) != (
        batch,
        kv_heads,
        head_dim,
    ):
        raise ValueError("unified HIP current K/V geometry is invalid")
    if tuple(output.shape) != tuple(q.shape):
        raise ValueError("unified HIP output must match Q")
    if slot_pages.ndim != 4 or directory_values.ndim != 4:
        raise ValueError("unified HIP requires the two-level page directory")
    if page_indices.ndim != 4 or page_indices.size(-1) != 16:
        raise ValueError("unified HIP requires page-size-16 leaf indices")

    required = (
        "unified_stream_counts",
        "unified_opened_counts",
        "unified_producer_done",
        "unified_tile_ready",
        "unified_overflow_flags",
        "unified_packed_indices",
        "unified_partial_out",
        "unified_partial_max",
        "unified_partial_denominator",
    )
    if not all(name in buffers for name in required):
        raise ValueError("unified HIP decode scratch is incomplete")
    sequences = batch * kv_heads
    kv_rows = cache_batches * kv_heads
    leaf_capacity = (local_offset - leaf_offset) // kv_rows
    local_capacity = (sink_offset - local_offset) // kv_rows
    sink_capacity = (coarse_offset - sink_offset) // kv_rows
    if min(leaf_capacity, local_capacity, sink_capacity) < 0:
        raise ValueError("unified HIP arena offsets are not monotonic")
    packed = buffers["unified_packed_indices"][:sequences]
    ready = buffers["unified_tile_ready"][:sequences]
    partial_out = buffers["unified_partial_out"][:sequences]
    partial_max = buffers["unified_partial_max"][:sequences]
    partial_denominator = buffers["unified_partial_denominator"][:sequences]
    index_capacity = int(packed.size(1))
    max_tiles = int(ready.size(1))
    if kernel_variant is None:
        kernel_variant = "route" if execution_mode != 0 else "full"
    logical_tile = 128 if kernel_variant in ("route", "consumer") else 256
    if max_tiles != math.ceil(index_capacity / logical_tile):
        raise ValueError(
            f"unified HIP N={logical_tile} scratch geometry is inconsistent"
        )
    error = _function(kernel_variant)(
        q.data_ptr(),
        new_k.data_ptr(),
        new_v.data_ptr(),
        cache_indices.data_ptr(),
        local_lens.data_ptr(),
        counts.data_ptr(),
        slot_pages.data_ptr(),
        directory_values.data_ptr(),
        slot_lengths.data_ptr(),
        page_indices.data_ptr(),
        arena_k.data_ptr(),
        arena_v.data_ptr(),
        arena_bias.data_ptr(),
        previous_total_lse.data_ptr(),
        packed.data_ptr(),
        buffers["unified_stream_counts"].data_ptr(),
        buffers["unified_opened_counts"].data_ptr(),
        buffers["unified_producer_done"].data_ptr(),
        ready.data_ptr(),
        buffers["unified_overflow_flags"].data_ptr(),
        partial_out.data_ptr(),
        partial_max.data_ptr(),
        partial_denominator.data_ptr(),
        output.data_ptr(),
        batch,
        query_heads,
        kv_heads,
        cache_batches,
        state_capacity,
        int(state_len),
        int(local_capacity),
        int(local_limit),
        int(sink_capacity),
        int(sink_len),
        int(leaf_capacity),
        int(page_indices.size(2)),
        int(directory_values.size(2)),
        int(slot_pages.size(3)),
        index_capacity,
        max_tiles,
        int(protected_len),
        int(max_leaf_tokens),
        int(open_capacity),
        int(leaf_offset),
        int(local_offset),
        int(sink_offset),
        int(coarse_offset),
        int(execution_mode),
        float(scale),
        float(math.log(mass_fraction)),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"unified HIP decode launch failed with error {error}")


def producer_consumer_lod_decode(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_indices: torch.Tensor,
    local_lens: torch.Tensor,
    counts: torch.Tensor,
    slot_pages: torch.Tensor,
    directory_values: torch.Tensor,
    slot_lengths: torch.Tensor,
    page_indices: torch.Tensor,
    arena_k: torch.Tensor,
    arena_v: torch.Tensor,
    arena_bias: torch.Tensor,
    previous_total_lse: torch.Tensor,
    output: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    producer_stream: torch.cuda.Stream,
    producer_ready_event: torch.cuda.Event,
    *,
    state_len: int,
    local_limit: int,
    sink_len: int,
    protected_len: int,
    max_leaf_tokens: int,
    open_capacity: int,
    leaf_offset: int,
    local_offset: int,
    sink_offset: int,
    coarse_offset: int,
    scale: float,
    mass_fraction: float,
    consumer_segments: int,
) -> None:
    """Overlap route/list production with bounded AITER-shaped consumers."""
    from model.kernels.aiter_page1_attention import (
        reduce_page1_hip_consumers,
    )

    batch, query_heads, query_len, head_dim = q.shape
    cache_batches, kv_heads, state_capacity, _ = counts.shape
    if query_len != 1 or head_dim != 256 or query_heads != kv_heads * 4:
        raise ValueError("producer/consumer decode requires D=256 and GQA=4")
    sequences = batch * kv_heads
    packed = buffers["unified_packed_indices"][:sequences]
    ready = buffers["unified_tile_ready"][:sequences]

    # Record all current-stream producers of Q/new-K/V and the recycled queue,
    # then let the route kernel run independently of the persistent consumers.
    current_stream = torch.cuda.current_stream()
    producer_ready_event.record(current_stream)
    producer_stream.wait_event(producer_ready_event)
    with torch.cuda.stream(producer_stream):
        unified_lod_decode(
            q,
            new_k,
            new_v,
            cache_indices,
            local_lens,
            counts,
            slot_pages,
            directory_values,
            slot_lengths,
            page_indices,
            arena_k,
            arena_v,
            arena_bias,
            previous_total_lse,
            output,
            buffers,
            state_len=state_len,
            local_limit=local_limit,
            sink_len=sink_len,
            protected_len=protected_len,
            max_leaf_tokens=max_leaf_tokens,
            open_capacity=open_capacity,
            leaf_offset=leaf_offset,
            local_offset=local_offset,
            sink_offset=sink_offset,
            coarse_offset=coarse_offset,
            scale=scale,
            mass_fraction=mass_fraction,
            execution_mode=1,
        )

    unified_lod_decode(
        q,
        new_k,
        new_v,
        cache_indices,
        local_lens,
        counts,
        slot_pages,
        directory_values,
        slot_lengths,
        page_indices,
        arena_k,
        arena_v,
        arena_bias,
        previous_total_lse,
        output,
        buffers,
        state_len=state_len,
        local_limit=local_limit,
        sink_len=sink_len,
        protected_len=protected_len,
        max_leaf_tokens=max_leaf_tokens,
        open_capacity=open_capacity,
        leaf_offset=leaf_offset,
        local_offset=local_offset,
        sink_offset=sink_offset,
        coarse_offset=coarse_offset,
        scale=scale,
        mass_fraction=mass_fraction,
        execution_mode=consumer_segments,
        kernel_variant="consumer",
    )
    reduce_page1_hip_consumers[(sequences, 4)](
        output,
        previous_total_lse,
        buffers["unified_partial_out"],
        buffers["unified_partial_max"],
        buffers["unified_partial_denominator"],
        cache_indices,
        buffers["unified_stream_counts"],
        buffers["unified_opened_counts"],
        buffers["unified_producer_done"],
        buffers["unified_overflow_flags"],
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_size=256,
        num_consumers=consumer_segments,
        reduce_consumers=1 << math.ceil(math.log2(consumer_segments)),
        num_warps=2,
        waves_per_eu=2,
        num_stages=1,
    )

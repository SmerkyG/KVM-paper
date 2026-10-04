"""Idle-prefill allocator retention, with host-only diagnostic counters.

This module deliberately has no vLLM dependency so policy tests and offline
benchmark metadata do not have to initialize a serving backend.
"""

from __future__ import annotations

import os

import torch


_PREFILL_ALLOCATOR_AUDIT = {
    "calls": 0, "retained": 0, "reclaimed": 0, "minimum_free_bytes_at_check": None,
}


def _reclaim_prefill_allocator(device: torch.device) -> None:
    """Retain already-idle warm allocations with bounded physical headroom.

    This never retains staging tensors or skips completion fences. The
    counters reuse the existing memory-pressure check; no extra GPU event,
    allocation or synchronization is added to the timed path.
    """
    _PREFILL_ALLOCATOR_AUDIT["calls"] += 1
    if os.environ.get("LOD_KIMI_REUSE_PREFILL_ALLOCATOR") == "1":
        # Development-only pressure control. Eight GiB remains the default;
        # the full resident model can test a smaller positive reserve without
        # ever making reclamation unconditional or removing completion fences.
        minimum_free_gib = int(os.environ.get("LOD_KIMI_PREFILL_MIN_FREE_GIB", "8"))
        if not 1 <= minimum_free_gib <= 64:
            raise ValueError("LOD_KIMI_PREFILL_MIN_FREE_GIB must be in [1, 64]")
        free_bytes, _ = torch.cuda.mem_get_info(device)
        minimum = _PREFILL_ALLOCATOR_AUDIT["minimum_free_bytes_at_check"]
        _PREFILL_ALLOCATOR_AUDIT["minimum_free_bytes_at_check"] = (
            free_bytes if minimum is None else min(minimum, free_bytes))
        if free_bytes >= minimum_free_gib * 1024**3:
            _PREFILL_ALLOCATOR_AUDIT["retained"] += 1
            return
    _PREFILL_ALLOCATOR_AUDIT["reclaimed"] += 1
    torch.cuda.empty_cache()

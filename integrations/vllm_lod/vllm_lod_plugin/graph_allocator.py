"""Keep expandable eager memory without making graph collectives copy in.

HIP IPC cannot export expandable-segment pointers. AITER consequently
disables registered graph communication when that allocator mode is global.
Use ordinary allocations only while initializing IPC communicators and
capturing graphs; restore the exact caller configuration for eager prefill
and large persistent caches. Nothing changes on graph replay.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import logging
import threading

import torch


logger = logging.getLogger(__name__)
_allocator_lock = threading.RLock()


@contextmanager
def ipc_allocator():
    """Temporarily prohibit VMM allocations, without changing other settings.

    This is a startup/capture scope, not a per-token toggle or cache migration.
    Existing expandable allocations remain intact. Nested communicator scopes
    restore correctly, including if initialization/capture raises.
    """
    with _allocator_lock:
        settings = torch.cuda.memory._snapshot()["allocator_settings"]
        if not settings.get("expandable_segments", False):
            yield
            return
        original = settings["PYTORCH_CUDA_ALLOC_CONF"]
        ordinary = original + ("," if original else "") + "expandable_segments:False"
        setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
        if setter is None:
            setter = torch.cuda.memory._set_allocator_settings
        setter(ordinary)
        try:
            yield
        finally:
            setter(original)


def install_ipc_graph_allocator() -> None:
    """Patch only AITER's classic HIP-IPC transport; retain other safety guards."""
    try:
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    except ImportError:
        return
    if getattr(CustomAllreduce, "_lod_ipc_graph_allocator", False):
        return
    original_init = CustomAllreduce.__init__
    original_capture = CustomAllreduce.capture

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        with ipc_allocator():
            original_init(self, *args, **kwargs)
        # AITER itself still rejects gfx1250 registration, unsupported groups,
        # or an explicitly disabled registered path. Never force its flag.

    @contextmanager
    @wraps(original_capture)
    def capture(self, *args, **kwargs):
        use_ipc = (
            not self.disabled
            and not getattr(self, "_use_vmm", False)
            and getattr(self, "enable_register_for_capturing", False)
        )
        if use_ipc:
            with ipc_allocator(), original_capture(self, *args, **kwargs):
                yield
        else:
            with original_capture(self, *args, **kwargs):
                yield

    CustomAllreduce.__init__ = initialize
    CustomAllreduce.capture = capture
    CustomAllreduce._lod_ipc_graph_allocator = True
    logger.info("Registered-IPC graph allocations isolated from expandable eager memory")

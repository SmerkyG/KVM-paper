"""IPC-compatible capture allocations must not disable eager expansion."""

import pytest
import torch
from contextlib import contextmanager
import sys
from types import SimpleNamespace

from vllm_lod_plugin.graph_allocator import ipc_allocator


@pytest.mark.parametrize("raises", [False, True])
def test_ipc_allocator_restores_nested_scope_and_other_settings(monkeypatch, raises):
    original = "max_split_size_mb:128,roundup_power2_divisions:[256:1,>:2],expandable_segments:True"
    state = {"expandable_segments": True, "PYTORCH_CUDA_ALLOC_CONF": original}
    changes = []
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda: {"allocator_settings": state})

    def setter(config):
        changes.append(config)
        state.update(PYTORCH_CUDA_ALLOC_CONF=config,
            expandable_segments=config.endswith("expandable_segments:True"))

    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", setter, raising=False)
    try:
        with ipc_allocator():
            assert not state["expandable_segments"]
            with ipc_allocator():
                assert not state["expandable_segments"]
            assert not state["expandable_segments"]
            if raises:
                raise ValueError("capture failed")
    except ValueError:
        assert raises
    assert changes == [original + ",expandable_segments:False", original]
    assert state["expandable_segments"]


def test_ordinary_allocator_has_no_toggle(monkeypatch):
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda: {
        "allocator_settings": {"expandable_segments": False}})
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings",
        lambda _: pytest.fail("unnecessary setting change"), raising=False)
    with ipc_allocator():
        pass


@pytest.mark.parametrize("allowed", [True, False])
def test_transport_capture_scope_respects_aiter_safety_guard(monkeypatch, allowed):
    from vllm_lod_plugin import graph_allocator

    state = {"expandable_segments": True, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda: {"allocator_settings": state})
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", lambda config: state.update(
        expandable_segments=config.endswith("True"), PYTORCH_CUDA_ALLOC_CONF=config), raising=False)

    class FakeAR:
        def __init__(self):
            assert not state["expandable_segments"]
            self.disabled, self._use_vmm = False, False
            # Stand in for an unsupported architecture or explicit opt-out.
            self.enable_register_for_capturing = allowed

        @contextmanager
        def capture(self):
            assert state["expandable_segments"] == (not allowed)
            try:
                yield
            finally:
                # IPC registration happens on exit, still inside our scope.
                assert state["expandable_segments"] == (not allowed)

    monkeypatch.setitem(sys.modules, "aiter.dist.device_communicators.custom_all_reduce",
        SimpleNamespace(CustomAllreduce=FakeAR))
    graph_allocator.install_ipc_graph_allocator()
    graph_allocator.install_ipc_graph_allocator()  # idempotent
    comm = FakeAR()
    assert state["expandable_segments"]
    assert comm.enable_register_for_capturing == allowed
    with comm.capture():
        pass
    assert state["expandable_segments"]

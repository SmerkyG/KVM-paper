"""CPU adapter-contract tests; real GPU graphs/streams are checked by the probe."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from benchmarks.kimi_k3_moe_front_probe import reference_front


@pytest.fixture
def front_module(monkeypatch):
    # CPU tests never compile the GPU kernel or require the AITER installation.
    path = Path(__file__).resolve().parents[1] / "benchmarks/experimental/kimi_moe_front.py"
    spec = importlib.util.spec_from_file_location("_test_moe_front", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def project(x, weight, scratch, shared, router, latent):
        values = reference_front(x, weight, shared.size(1), router.size(1))
        for target, value in zip((shared, router, latent), values, strict=True):
            target.copy_(value)
        return shared, router, latent
    monkeypatch.setattr(module, "merged_front", project)
    def routed_project(x, weight, scratch, router, latent):
        _, r, v = reference_front(x, weight, 0, router.size(1))
        router.copy_(r)
        latent.copy_(v)
        return router, latent
    monkeypatch.setattr(module, "merged_routed_front", routed_project)
    return module


@pytest.fixture
def adapter(front_module):
    return front_module.DecodeFront


class Linear(torch.nn.Linear):
    def __init__(self, n, m):
        super().__init__(n, m, bias=False, dtype=torch.bfloat16)
    def forward(self, x):
        return super().forward(x), None


class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj, self.down_proj = Linear(4, 4), Linear(2, 4)
        self.act_fn = SimpleNamespace(beta=4.0, linear_beta=25.0)
        self.native_calls = 0
    def forward(self, x):
        self.native_calls += 1
        shared = reference_front(x, self.gate_up_proj.weight, 2, 0)[0]
        return self.down_proj(shared)[0]


class SharedQueue:
    def __init__(self, shared):
        self._layer, self.value, self.enqueues, self.can_overlap = shared, None, 0, True
    def maybe_forward_async(self, hidden):
        if not self.can_overlap:
            return False
        assert self.value is None
        self.enqueues += 1
        self.value = self._layer(hidden)
        return True
    @property
    def output(self):
        result, self.value = self.value, None
        return result


class Experts(torch.nn.Module):
    def __init__(self, shared, down):
        super().__init__()
        self.shared_experts = SharedQueue(shared)
        self.down, self.up = down, Linear(3, 4)
        self.enable_dbo = False
        self.preprojected_calls = 0
    def forward(self, hidden_states, router_logits, shared_experts_input=None):
        assert router_logits.size(0) == hidden_states.size(0)
        if shared_experts_input is None:
            original = hidden_states
            hidden_states = self.down(hidden_states)[0]
        else:
            self.preprojected_calls += 1
            original = shared_experts_input
            assert original.size(-1) == 4  # Native tail must retain original width.
        overlapping = self.shared_experts.maybe_forward_async(original)
        shared = self.shared_experts.output if overlapping else self.shared_experts._layer(original)
        return self.up(hidden_states)[0]+shared


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_experts, self.gate = Shared(), Linear(4, 2)
        self.routed_expert_down_proj = Linear(4, 3)
        self.experts = Experts(self.shared_experts, self.routed_expert_down_proj)
    def forward(self, hidden):
        return self.experts(hidden_states=hidden, router_logits=self.gate(hidden)[0])


def test_decode_front_skips_duplicate_projections_and_preserves_tail(adapter):
    torch.manual_seed(7)
    layer = Layer()
    x = torch.randn(2, 4).bfloat16()
    expected = layer(x)
    front = adapter(layer, 2)
    front.enabled = True
    pointers = [t.data_ptr() for t in (front.projection, front.shared, front.router, front.latent)]
    torch.testing.assert_close(layer(x), expected, rtol=0.02, atol=0.01)
    assert layer.experts.preprojected_calls == 1
    assert layer.shared_experts.native_calls == 1
    assert pointers == [t.data_ptr() for t in (front.projection, front.shared, front.router, front.latent)]
    # Larger prefill and disabled decode must still use the native front.
    layer(torch.randn(5, 4).bfloat16())
    front.enabled = False
    layer(x)
    assert layer.experts.preprojected_calls == 1
    assert layer.shared_experts.native_calls == 3


def test_fp32_router_is_not_silently_downcast(adapter):
    layer = Layer()
    layer.gate.weight = torch.nn.Parameter(layer.gate.weight.float())
    with pytest.raises(ValueError, match="no router downcast"):
        adapter(layer)


def test_routed_only_merge_keeps_native_shared_front(adapter):
    torch.manual_seed(7)
    layer = Layer()
    x = torch.randn(2, 4).bfloat16()
    expected = layer(x)
    front = adapter(layer, 2, kind="routed")
    front.enabled = True
    torch.testing.assert_close(layer(x), expected, rtol=0.02, atol=0.01)
    assert layer.experts.preprojected_calls == 1
    assert layer.shared_experts.native_calls == 2
    assert front.packed.size(0) == 5  # Router plus latent only, no shared rows.


def test_early_shared_front_reuses_native_events_and_math(front_module):
    layer = Layer()
    x = torch.randn(2, 4).bfloat16()
    expected = layer(x)
    front = front_module.EarlySharedFront(layer, 2)
    front.enabled = front.record_capture = True
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    assert layer.experts.shared_experts.enqueues == 2  # Once per invocation, never twice.
    assert front.capture_enqueues == 1 and front.extra_bytes == 0 and not front.pending
    layer(torch.randn(5, 4).bfloat16())  # Prefill remains native.
    assert front.capture_enqueues == 1
    layer.experts.shared_experts.can_overlap = False  # Native internally-overlapped fallback.
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    assert front.capture_enqueues == 1 and not front.pending


def test_unsupported_stream_or_activation_contract_fails(adapter):
    layer = Layer()
    layer.experts.enable_dbo = True
    with pytest.raises(ValueError, match="no DBO"):
        adapter(layer)
    layer.experts.enable_dbo = False
    layer.shared_experts.act_fn.beta = 1.0
    with pytest.raises(ValueError, match="SiTU"):
        adapter(layer)


def test_graph_switch_uses_two_distinct_captures(monkeypatch):
    from benchmarks.kimi_k3_moe_front_decode import select_front
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    native_graph, merged_graph = object(), object()
    manager = SimpleNamespace(graphs={8: native_graph},
        run_fullgraph=lambda desc: desc, captured_token_counts=lambda: [8])
    captures = []
    def capture():
        captures.append(True)
        manager.graphs = {8: merged_graph}
    fronts = [SimpleNamespace(enabled=False, kind="routed")]
    worker = SimpleNamespace(rank=0, _lod_moe_fronts=fronts,
        _lod_moe_graphs={"native": dict(manager.graphs)},
        model_runner=SimpleNamespace(cudagraph_manager=manager, capture_model=capture))
    select_front(worker, "merged")
    assert fronts[0].enabled and manager.graphs[8] is merged_graph
    select_front(worker, "native")
    assert not fronts[0].enabled and manager.graphs[8] is native_graph
    select_front(worker, "merged")
    assert len(captures) == 1 and manager.graphs[8] is merged_graph


def test_global_decode_update_count_is_not_divided_by_batch_or_dcp():
    from benchmarks.kimi_k3_moe_front_decode import expected_decode_updates
    assert expected_decode_updates(16384, 1026) == 4
    assert expected_decode_updates(131072, 1026) == 4
    assert expected_decode_updates(4096, 258) == 1


def test_graph_canary_proves_private_scratch_writes_without_profiler():
    from benchmarks.kimi_k3_moe_front_decode import arm_graph_kernel_check, graph_kernel_check
    fronts = [SimpleNamespace(router=torch.empty(2, 3), kind="routed") for _ in range(2)]
    def replay(desc):
        if desc == "merged":
            for front in fronts:
                front.router.fill_(1.0)
    manager = SimpleNamespace(run_fullgraph=replay)
    worker = SimpleNamespace(rank=7, _lod_moe_fronts=fronts,
                             model_runner=SimpleNamespace(cudagraph_manager=manager))
    arm_graph_kernel_check(worker)
    # Prefill writes do not count: poison just before the first actual graph.
    for front in fronts:
        front.router.fill_(123.0)
    with pytest.raises(RuntimeError, match="did not observe"):
        graph_kernel_check(worker)
    manager.run_fullgraph("native")
    assert manager.run_fullgraph is replay  # No persistent/per-step hook.
    assert graph_kernel_check(worker)["written_front_layers"] == 0
    arm_graph_kernel_check(worker)
    manager.run_fullgraph("merged")
    assert graph_kernel_check(worker)["written_front_layers"] == 2
    arm_graph_kernel_check(worker)
    manager.run_fullgraph("native")
    fronts[0].router[0].fill_(1.0)
    with pytest.raises(RuntimeError, match="only part"):
        graph_kernel_check(worker)

from types import SimpleNamespace

import pytest
import torch

from vllm_lod_plugin import prefill_collectives


@pytest.mark.parametrize("reuse,free_gib,expected", [(False, 100, True), (True, 100, False), (True, 7, True)])
def test_allocator_reuse_keeps_memory_pressure_reclamation(monkeypatch, reuse, free_gib, expected):
    pytest.importorskip("vllm")
    from vllm_lod_plugin import runtime

    calls = []
    monkeypatch.setenv("LOD_KIMI_REUSE_PREFILL_ALLOCATOR", "1" if reuse else "0")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_gib * 1024**3, 256 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("reclaim"))
    runtime._reclaim_prefill_allocator(torch.device("cuda:0"))
    assert bool(calls) is expected


@pytest.mark.parametrize("layers", [8, 16, 24])
def test_layer_shares_reassemble_in_original_order(layers):
    original = list(range(layers))
    shares = [original[prefill_collectives.construction_layer_slice(layers, 8, rank)]
              for rank in range(8)]
    assert len({len(share) for share in shares}) == 1
    assert [layer for share in shares for layer in share] == original
    with pytest.raises(ValueError, match="evenly"):
        prefill_collectives.construction_layer_slice(layers + 1, 8, 0)


def test_construction_uses_a_separate_communicator(monkeypatch):
    calls = []
    process_group = object()
    coordinator = SimpleNamespace(
        world_size=2,
        make_sibling_device_group=lambda **kwargs: (
            calls.append(kwargs) or process_group
        ),
    )
    group = prefill_collectives.PrefillConstructionGroup(coordinator)

    def gather(output, source, **kwargs):
        assert kwargs == {"group": process_group, "async_op": True}
        assert source.is_contiguous()
        output.copy_(torch.cat((source, source + 1), dim=0))
        return SimpleNamespace(wait=lambda: calls.append("wait"))

    monkeypatch.setattr(prefill_collectives.dist, "all_gather_into_tensor", gather)
    source = torch.arange(12).reshape(3, 4).T
    output = group.all_gather(source)
    assert torch.equal(output, torch.cat((source, source + 1), dim=0))
    assert calls == [{"group_desc": "lod-prefill-construction"}, "wait"]
    with pytest.raises(ValueError, match="dimension zero"):
        group.all_gather(source, dim=1)

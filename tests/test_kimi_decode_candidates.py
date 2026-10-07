"""Candidates preserve 16-head KV sets, masks and captured step semantics."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gfx942 decode candidate GPU checks")
@pytest.mark.parametrize("heads,work", [(7, 4), (17, 8), (96, 4), (96, 8)])
def test_head_work_subdivision_keeps_attention_and_lse(heads, work, monkeypatch):
    from benchmarks import kimi_gluon_lod_decode_probe as probe
    from benchmarks.experimental.kimi_decode_work_heads import absorbed_mla_lod_decode_gfx942

    monkeypatch.setattr("sys.argv", ["probe", "--batch-size", "2", "--heads", str(heads),
        "--coarse", "65", "--local", "3", "--exact-pages", "3", "--splits", "32",
        "--head-tiled-metadata"])
    monkeypatch.setattr(probe, "_time_ms", lambda *args, **kw: 0.)
    def run(*args, **kwargs):
        return absorbed_mla_lod_decode_gfx942(*args, **kwargs, work_heads=work)
    monkeypatch.setattr(probe, "absorbed_mla_lod_decode_gfx942", run)
    # The probe uses an independent FP32 effective-attention and LSE reference.
    probe.main()


def test_rejected_subdivision_has_no_serving_dispatch_option():
    import inspect
    from lod_attention.kernels.kimi_gluon_decode import absorbed_mla_lod_decode_gfx942
    from lod_attention.kernels.paged_decode import fused_decode_paged_lod_attention

    assert "work_heads" not in inspect.signature(absorbed_mla_lod_decode_gfx942).parameters
    signature = inspect.signature(fused_decode_paged_lod_attention)
    assert signature.parameters["gqa_union_fuse_compact_route"].default is False
    assert "kimi_decode_work_heads" not in signature.parameters


def test_live_oracle_restores_replay_and_does_not_advance_cache(monkeypatch):
    from types import SimpleNamespace as NS
    from benchmarks.kimi_k3_decode_union_fixture import arm_live_union_check, live_union_check
    from lod_attention.kernels import paged_decode

    q = torch.zeros(1, 96, 1, 576)
    buffers = dict(kimi_gluon_final_lse=torch.zeros(1, 96),
                   gqa_union_counts=torch.ones(6, dtype=torch.int32),
                   gqa_union_slots=torch.zeros(6, 128, dtype=torch.int32))
    flags = []
    def attention(*args, **kwargs):
        assert kwargs["new_k"] is kwargs["new_v"] is None
        assert kwargs["store_new_kv"] is kwargs["advance_local_lens"] is False
        flags.append(kwargs["gqa_union_fuse_compact_route"])
        return torch.zeros(1, 96, 1, 512)
    monkeypatch.setattr(paged_decode, "fused_decode_paged_lod_attention", attention)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    replayed = []
    original = lambda desc: replayed.append(desc) or "output"
    manager = NS(run_fullgraph=original)
    kwargs = dict(buffers=buffers, new_k=torch.ones(1), new_v=torch.ones(1),
                  store_new_kv=True, advance_local_lens=False)
    worker = NS(rank=0, model_runner=NS(cudagraph_manager=manager,
        model_state=NS(_vllm_lod_runtime=NS(pools={"layer": object()}))),
        _kimi_union_fixture_calls=[((q,), kwargs), ((q,), kwargs)])
    arm_live_union_check(worker)
    assert manager.run_fullgraph("first") == "output"
    assert manager.run_fullgraph is original
    assert manager.run_fullgraph("timed") == "output"
    assert flags == [False, True]
    assert replayed == ["first", "timed"]
    assert len(live_union_check(worker)["layers"]) == 1
    # Stored captured arguments are unchanged; only private oracle kwargs
    # were changed, so the real graph still inserts each subsequent token.
    assert worker._kimi_union_fixture_calls[0][1]["store_new_kv"] is True

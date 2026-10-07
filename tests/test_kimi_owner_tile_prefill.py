"""CPU checks for the isolated prefill tile comparison's dispatch/corpus."""

import sys
from types import SimpleNamespace

import pytest

from benchmarks.kimi_k3_owner_tile_prefill import configure_tile_environment, frozen_prompts, set_tile


def test_head_group_trial_changes_worker_scratch_bound_only(monkeypatch):
    import os
    from benchmarks.kimi_k3_owner_tile_prefill import set_head_group

    environment = {"LOD_KIMI_COARSE_QUERY_TILE": "128"}
    monkeypatch.setattr(os, "environ", environment)
    worker = SimpleNamespace(rank=7)
    for heads in (6, 12):
        assert set_head_group(worker, heads) == {"rank": 7, "head_group": heads}
        assert environment == {"LOD_KIMI_COARSE_QUERY_TILE": "128",
                               "LOD_KIMI_OWNER_PREFILL_HEAD_GROUP": str(heads)}


def test_prefill_cohort_is_admitted_before_any_owner_slice(monkeypatch):
    import os
    environment = dict(os.environ)
    monkeypatch.setattr(os, "environ", environment)
    configure_tile_environment(6)
    assert environment["LOD_BENCHMARK_ADMISSION_COHORT"] == "8"
    # max_tokens=1 has no decode window to synchronize.
    assert environment["LOD_BENCHMARK_SYNCHRONIZED_DECODE"] == "0"
    assert environment["LOD_KIMI_OWNER_PREFILL_HEAD_GROUP"] == "6"
    assert environment["LOD_KIMI_SUBTILE64"] == "score"


def test_frozen_prompt_stream_matches_existing_speed_panel():
    documents = [[row * 10 + j for j in range(3)] for row in range(8)]
    prompts = frozen_prompts(documents, 8)
    assert len(prompts) == 8
    for row, prompt in enumerate(prompts):
        assert prompt["prompt_token_ids"] == (documents[row] * 3)[:8]


@pytest.mark.parametrize("documents", [[], [[1]] * 7, [[1]] * 7 + [[]]])
def test_reject_insufficient_or_empty_corpus(documents):
    with pytest.raises(ValueError):
        frozen_prompts(documents, 8)


def test_only_query_tile_changes_and_instrumentation_can_be_removed(monkeypatch):
    calls = []
    def original(**kwargs):
        calls.append(kwargs)
        return lambda *args, **kw: "result"
    module = SimpleNamespace(serving_subtile_factory=original)
    monkeypatch.setitem(sys.modules, "benchmarks.kimi_k3_subtile_route", module)
    worker = SimpleNamespace(rank=3)
    set_tile(worker, 64, audit=True)
    function = module.serving_subtile_factory(query_tile=64, score_only=True, reuse_max=False)
    q = SimpleNamespace(size=lambda dim: {1: 2048, 2: 12}[dim])
    k = SimpleNamespace(size=lambda dim: 4096)
    assert function(q, k) == "result"
    assert worker._lod_tile_calls == {"queries=2048,heads=12,states=4096": 1}
    assert calls == [dict(query_tile=64, score_only=True, reuse_max=False)]
    with pytest.raises(AssertionError, match="wrong current-path"):
        module.serving_subtile_factory(query_tile=64, score_only=True, reuse_max=True)
    set_tile(worker, 128)
    assert module.serving_subtile_factory is original


def test_aiter_torch_registration_names_include_every_specialization():
    from benchmarks.kimi_k3_subtile_route import subtile_operator_name

    names = {subtile_operator_name(score, reuse, tile_n, query_tile)
             for score in (False, True) for reuse in (False, True)
             for tile_n in (32, 64) for query_tile in (64, 128)}
    assert len(names) == 16
    assert subtile_operator_name(True, False, 64, 64) != subtile_operator_name(True, False, 64, 128)


def test_factories_register_distinct_operators_before_aiter_guard(monkeypatch):
    from benchmarks.kimi_k3_subtile_route import subtile_factory, subtile_operator_name

    registered = {}
    def compile_ops(*args, **kwargs):
        assert kwargs["fc_name"] == "mha_fwd"
        def register(function):
            # Reproduce the guard's reuse-by-name behavior, without a GPU.
            return registered.setdefault(function.__name__, function)
        return register
    monkeypatch.setitem(sys.modules, "aiter.jit.core", SimpleNamespace(
        compile_ops=compile_ops, get_args_of_build=lambda *_: {}))
    monkeypatch.setitem(sys.modules, "aiter.ops.mha", SimpleNamespace(cmdGenFunc_mha_fwd=lambda *_: {}))
    first = subtile_factory([], score_only=True, query_tile=128)
    second = subtile_factory([], score_only=True, query_tile=64)
    assert first is not second
    assert first.__name__ == subtile_operator_name(True, False, 64, 128)
    assert second.__name__ == subtile_operator_name(True, False, 64, 64)

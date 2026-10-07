import pytest
from types import SimpleNamespace as NS

from benchmarks.kimi_k3_prefill_sweep import owner_prefill_layout, reference_trace_prefix, timed_sweep_generate
from benchmarks.prolong import token_digest


def test_startup_oom_is_logged_but_never_becomes_a_timing(tmp_path):
    import json
    from benchmarks.kimi_k3_prefill_sweep import create_sweep_engine

    args = NS(output=tmp_path / "capacity.json", mode="two-tier", batch_size=8,
        checkpoint="trained", kv_cache_memory_bytes=1 << 30, lengths=[1044480])
    kwargs = dict(max_model_len=1045522)
    def out_of_memory(**config):
        assert config == kwargs
        raise RuntimeError("out of memory during profiling")
    with pytest.raises(RuntimeError, match="out of memory"):
        create_sweep_engine(out_of_memory, kwargs, args)
    result = json.loads(args.output.read_text())
    assert result["measurement_status"] == "failed"
    assert result["current_phase"] == {"phase": "engine_initialization"}
    assert result["measurements"] == {}
    assert result["planned_lengths"] == [1044480]
    assert create_sweep_engine(lambda **config: config, kwargs, args) == kwargs


@pytest.mark.parametrize("explicit_moe_chunk", [None, "8192"])
def test_owner_setup_preserves_moe_bound_without_shrinking_attention_budget(monkeypatch, explicit_moe_chunk):
    import os
    from benchmarks.kimi_k3_prefill_sweep import configure_owner_prefill_environment

    monkeypatch.setenv("LOD_KIMI_OWNER_QUERY_CHUNK", "2048")
    monkeypatch.delenv("LOD_KIMI_OWNER_MOE_CHUNK", raising=False)
    if explicit_moe_chunk is not None:
        monkeypatch.setenv("LOD_KIMI_OWNER_MOE_CHUNK", explicit_moe_chunk)
    assert configure_owner_prefill_environment(8) == (2048, 16392)
    assert os.environ["LOD_KIMI_OWNER_MOE_CHUNK"] == (explicit_moe_chunk or "16392")
    assert os.environ["LOD_BENCHMARK_PREFILL_COHORT"] == "8"


def test_capacity_diagnostic_uses_global_counts_and_local_leaf_storage():
    import torch
    from benchmarks.kimi_k3_prefill_sweep import centroid_leaf_stats

    pool = NS(is_absorbed_mla=True, engine=NS(max_open_centroid_leaves=1024),
        state=dict(counts=torch.tensor([[[[1024.], [1025.], [8000.], [0.]]]]),
                   page_cache=dict(slot_lengths=torch.tensor([[[128, 128, 1000, 0]]]))))
    parent = NS(owner_decode_pool=pool)
    worker = NS(model_runner=NS(model_state=NS(_vllm_lod_runtime=NS(pools={"mla": parent}))))
    result = centroid_leaf_stats(worker)["mla"]
    assert result["closed_centroids"] == 2
    assert result["local_leaf_count"] == 1256
    assert result["local_leaves_in_closed_centroids"] == 1128
    assert result["largest_global_centroid"] == 8000


def test_ranked_attention_audit_records_physical_cached_means_and_live_splits(monkeypatch):
    import torch
    import benchmarks.prolong as prolong
    from benchmarks.kimi_k3_prefill_sweep import ranked_attention_audit

    monkeypatch.setattr(prolong, "audit_worker_attention_mode", lambda worker: {})
    pool = NS(kv_heads=1, engine=NS(), dcp_decode_buffers={1: {}}, state=dict(
        state_k=torch.empty(1, 1, 17, 576), page_cache=dict(
            unified_page1_k=torch.empty(31, 576), unified_page1_coarse_offset=13)))
    worker = NS(rank=0, model_runner=NS(_vllm_lod_runtime=NS(pools={"mla": pool})))
    result = ranked_attention_audit(worker)["decode_geometry"]["mla"]
    assert result["cached_centroid_mean_routing"] is True
    assert result["splits_by_live_batch"] == {"1": 32}
    audit = ranked_attention_audit(worker)
    assert audit["owner_shared_construction_scope"] is True
    assert audit["update_score_workspace_follows_overflow"] is True


@pytest.mark.parametrize("runtime_on_state", [True, False])
@pytest.mark.parametrize("request_owner", [True, False])
def test_leaf_memory_observer_captures_before_cleanup_and_leaves_no_timing_hook(runtime_on_state, request_owner):
    import torch
    from benchmarks.kimi_k3_prefill_sweep import arm_warmup_leaf_stats, finish_warmup_leaf_stats

    class Pool:
        is_absorbed_mla = True
        engine = NS(max_open_centroid_leaves=1024)

        def __init__(self):
            self.state = dict(counts=torch.zeros(1, 1, 2, 1),
                page_cache=dict(slot_lengths=torch.zeros(1, 1, 2, dtype=torch.int32)))

        def reset(self, slot):
            self.state["counts"].zero_()
            self.state["page_cache"]["slot_lengths"].zero_()
            return slot

        def _reset_range(self, start, stop):
            return self.reset(start)

    pool = Pool()
    class Parent:
        owner_decode_pool = pool

        def reset(self, slot):
            return pool.reset(slot)

        def _reset_range(self, start, stop):
            return pool._reset_range(start, stop)

    lifecycle = Parent() if request_owner else pool
    runtime = NS(pools={"mla": lifecycle})
    runner = NS(model_state=NS(_vllm_lod_runtime=runtime)) if runtime_on_state else NS(
        model_state=NS(), _vllm_lod_runtime=runtime)
    worker = NS(model_runner=runner)
    arm_warmup_leaf_stats(worker)
    assert lifecycle.reset(0) == 0
    assert worker._kimi_warmup_leaf_stats == {}
    pool.state["counts"][0, 0, :, 0] = torch.tensor([1025., 10.])
    pool.state["page_cache"]["slot_lengths"][0, 0] = torch.tensor([128, 10])
    lifecycle._reset_range(0, 1)
    saved = finish_warmup_leaf_stats(worker)["mla"]
    assert saved["global_member_count"] == 1035
    assert saved["local_leaves_in_closed_centroids"] == 128
    assert "reset" not in vars(pool) and "_reset_range" not in vars(pool)
    assert "reset" not in vars(lifecycle) and "_reset_range" not in vars(lifecycle)
    lifecycle.reset(0)
    assert worker._kimi_warmup_leaf_stats["mla"] == saved


def test_fit_test_can_replay_two_tokens_from_verified_long_trace():
    tokens = list(range(17))
    record = {"trace_tokens": 13, "trace_token_sha256": token_digest(tokens[4:])}
    assert reference_trace_prefix(tokens, record, length=4, output_tokens=2) == [4, 5]


def test_short_replay_still_rejects_tampered_archived_suffix():
    tokens = list(range(17))
    record = {"trace_tokens": 13, "trace_token_sha256": token_digest(tokens[4:])}
    tokens[-1] = 99
    with pytest.raises(RuntimeError, match="trace mismatch"):
        reference_trace_prefix(tokens, record, length=4, output_tokens=2)


def test_short_replay_cannot_exceed_archived_continuation():
    with pytest.raises(ValueError, match="exceeds"):
        reference_trace_prefix([1, 2], {"trace_tokens": 1}, length=1, output_tokens=2)


def test_canonical_four_update_trace_adds_one_real_token_after_verifying_old_dense_panel():
    tokens = list(range(1031))
    record = dict(trace_tokens=1025, trace_token_sha256=token_digest(tokens[4:1029]))
    assert reference_trace_prefix(tokens, record, length=4, output_tokens=1026) == tokens[4:1030]
    tokens[1028] = -1
    with pytest.raises(RuntimeError, match="trace mismatch"):
        reference_trace_prefix(tokens, record, length=4, output_tokens=1026)


def test_canonical_extension_rejects_missing_source_token_and_arbitrary_extra_steps():
    tokens = list(range(1029))
    record = dict(trace_tokens=1025, trace_token_sha256=token_digest(tokens[4:]))
    with pytest.raises(ValueError, match="frozen source stream"):
        reference_trace_prefix(tokens, record, length=4, output_tokens=1026)
    with pytest.raises(ValueError, match="exceeds"):
        reference_trace_prefix(tokens, record, length=4, output_tokens=1027)


def test_owner_scheduler_can_use_eight_2k_slices_without_changing_logical_16k(monkeypatch):
    monkeypatch.setenv("LOD_KIMI_OWNER_QUERY_CHUNK", "2048")
    assert owner_prefill_layout(8) == (2048, 16392)
    monkeypatch.delenv("LOD_KIMI_OWNER_QUERY_CHUNK")
    assert owner_prefill_layout(8) == (16384, 131080)


@pytest.mark.parametrize("chunk", ["0", "2000", "32768", "-1"])
def test_owner_scheduler_rejects_slices_that_cross_logical_blocks(monkeypatch, chunk):
    monkeypatch.setenv("LOD_KIMI_OWNER_QUERY_CHUNK", chunk)
    with pytest.raises(ValueError, match="divisor of 16384"):
        owner_prefill_layout(8)


def fake_generation(*, last_times=(20., 20.), preemptions=(0, 0), cached=(0, 0)):
    outputs = [NS(outputs=[NS(token_ids=[7, 8])], num_cached_tokens=cache,
                  metrics=NS(scheduled_ts=0., first_token_ts=first,
                             last_token_ts=last, num_preemptions=preemption))
               for first, last, preemption, cache in
               zip((1., 5.), last_times, preemptions, cached, strict=True)]
    return NS(generate=lambda *args, **kwargs: outputs)


def test_serial_prefill_first_samples_are_not_decode_desynchronization():
    result = timed_sweep_generate(fake_generation(), [{}, {}], NS(max_tokens=2),
                                  synchronized_decode=True)
    assert result[1:3] == (5., 15.)
    assert result[-1]["first_token_spread_seconds"] == 4.
    assert result[-1]["all_requests_live_overlap_seconds"] == 15.


def test_split_decode_cohort_is_still_rejected(monkeypatch):
    monkeypatch.setenv("LOD_BENCHMARK_SYNCHRONIZED_DECODE", "0")
    with pytest.raises(RuntimeError, match="synchronized decode cohort"):
        timed_sweep_generate(fake_generation(last_times=(19., 20.)), [{}, {}],
                             NS(max_tokens=2), synchronized_decode=True)


@pytest.mark.parametrize("kwargs,message", [
    ({"preemptions": (0, 1)}, "preempted"),
    ({"cached": (1, 0)}, "prefix-cached"),
])
def test_sweep_uses_canonical_request_validity_checks(kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        timed_sweep_generate(fake_generation(**kwargs), [{}, {}], NS(max_tokens=2),
                             synchronized_decode=True)

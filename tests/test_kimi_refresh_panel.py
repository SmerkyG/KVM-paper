from benchmarks.kimi_k3_refresh_panel import plans


def test_short_owner_reserves_native_cache_not_a_second_full_latent_archive():
    assert plans("two-tier", 8, 262144) == [
        ("short", [16384, 32768, 65536, 131072, 262144], 1, False)]


def test_long_b1_shards_tokens_without_changing_global_contexts():
    assert plans("two-tier", 1, 1044480)[1] == (
        "long", [524288, 1044480], 1, True)
    assert plans("two-tier", 8, 1044480)[1:] == [
        ("long-512k", [524288], 1, False),
        ("long-1020k", [1044480], 1, False)]


def test_dense_b8_native_cache_scales_with_the_live_cohort():
    assert plans("full", 8, 524288)[1] == ("long", [262144, 524288], 17, False)
    assert plans("full", 8, 1044480)[1] == ("long", [262144, 524288, 1044480], 31, False)


def test_short_panel_does_not_invent_an_unrequested_long_context():
    assert plans("full", 1, 65536) == [("short", [16384, 32768, 65536], 1, False)]


def test_capacity_environment_does_not_change_attention_or_update_schedule():
    from benchmarks.kimi_k3_refresh_panel import lod_memory_environment

    assert lod_memory_environment(8, "short", False) == {"HSA_NO_SCRATCH_RECLAIM": "0"}
    assert lod_memory_environment(1, "long", True) == {
        "HSA_NO_SCRATCH_RECLAIM": "0", "LOD_KIMI_DCP_SHARDED_LEAVES": "1"}
    assert lod_memory_environment(8, "long-1020k", False) == {
        "HSA_NO_SCRATCH_RECLAIM": "0", "LOD_KIMI_OWNER_PREFILL_HEAD_GROUP": "2",
        "LOD_KIMI_OWNER_MOE_CHUNK": "1024", "LOD_KIMI_COMPACT_PAGE_DIRECTORY": "1",
        "LOD_KIMI_OWNER_SHARD_RESIDUAL": "1", "LOD_KIMI_LOCAL_PREFILL_HEAD_GROUP": "8",
        "LOD_KIMI_COARSE_PREFILL_HEAD_GROUP": "16", "HSA_SCRATCH_SINGLE_LIMIT_ASYNC": "268435456"}
    assert lod_memory_environment(8, "long-512k", False) == {
        "HSA_NO_SCRATCH_RECLAIM": "0", "LOD_KIMI_OWNER_PREFILL_HEAD_GROUP": "4",
        "LOD_KIMI_OWNER_MOE_CHUNK": "4096", "LOD_KIMI_COMPACT_PAGE_DIRECTORY": "1",
        "LOD_KIMI_OWNER_SHARD_RESIDUAL": "1"}


def test_resumption_reuses_audited_points_even_if_next_warmup_failed(tmp_path):
    import json
    from tests.test_kimi_current_timings import fixture
    from benchmarks.kimi_k3_refresh_panel import completed_lengths

    data, point = fixture()
    data.update(measurement_status="failed", measurements={"16384": point,
        "32768": dict(measurement_status="warming")})
    (tmp_path / "oct7-current-lod-b8-short.json").write_text(json.dumps(data))
    assert completed_lengths("two-tier", 8, tmp_path) == {16384}
    assert completed_lengths("full", 8, tmp_path) == set()


def test_second_resumption_cannot_overwrite_first_resumptions_points(tmp_path):
    from benchmarks.kimi_k3_refresh_panel import remaining_output

    base = tmp_path / "oct7-current-lod-b8-short.json"
    assert remaining_output(base, [65536]) == base
    base.touch()
    first = remaining_output(base, [32768, 65536])
    assert first.name == "oct7-current-lod-b8-short-remaining-64k.json"
    first.touch()
    assert remaining_output(base, [65536]).name == "oct7-current-lod-b8-short-remaining-64k-retry2.json"


def test_failed_predecessor_does_not_start_a_new_engine(tmp_path):
    import pytest
    from benchmarks.kimi_k3_refresh_panel import wait_for_preceding

    log = tmp_path / "job.log"
    log.write_text("==> cluster-run completed: status=finished exit_code=1\n")
    with pytest.raises(RuntimeError, match="preceding engine"):
        wait_for_preceding(log)
    log.write_text("==> cluster-run completed: status=finished exit_code=0\n")
    wait_for_preceding(log)

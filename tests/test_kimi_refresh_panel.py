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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lod_attention._config import (
    EXACT_DECODE_LIMIT,
    LODConfig,
    LODMode,
    ModelFamily,
    PagedLODConfig,
    PREFIX_CACHE_LOCAL_WINDOW,
    ROUTE_COUNT,
    kernel_config,
    model_family,
)
from lod_attention._profile import configure_engine


@pytest.mark.parametrize("routing_normalization", ["query", "none"])
@pytest.mark.parametrize("inverse_mass", [False, True])
@pytest.mark.parametrize("route_count", [4, 8])
def test_aiter_prefill_reuses_fused_coarse_result(
    monkeypatch: pytest.MonkeyPatch,
    routing_normalization: str,
    inverse_mass: bool,
    route_count: int,
) -> None:
    from lod_attention._core import TritonLODAttentionCore
    from lod_attention.kernels import aiter_prefill_attention

    engine = TritonLODAttentionCore()
    engine.num_key_value_groups = 2
    engine.scaling = 0.5
    engine.two_level_topk = route_count
    engine.prefill_two_level_topk = route_count
    engine.separate_sink_cache = True
    engine.routing_normalization = routing_normalization
    engine.prefill_aiter_route_coarse = True
    engine.split_prefill_local_attention = True
    engine.mla_state_key_normalization = "none"
    engine.inverse_coherence_mass = inverse_mass

    q = torch.randn(1, 4, 3, 4)
    state_k = torch.randn(1, 2, 8, 4)
    state_v = torch.randn_like(state_k)
    counts = torch.randint(1, 5, (1, 2, 8, 1)).float()
    key_norm_sums = counts.clone()
    expected_routes = torch.zeros(1, 4, 3, route_count, dtype=torch.long)
    expected_coarse = object()
    expected_route_head_counts = torch.zeros(4 * 8, dtype=torch.int32)
    expected_route_offsets = torch.zeros_like(expected_routes, dtype=torch.int32)

    def fake_aiter(
        passed_q: torch.Tensor,
        passed_k: torch.Tensor,
        passed_v: torch.Tensor,
        passed_counts: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, object, torch.Tensor, torch.Tensor]:
        torch.testing.assert_close(passed_q, q)
        assert passed_k.data_ptr() == state_k.data_ptr()
        assert passed_v.data_ptr() == state_v.data_ptr()
        assert passed_counts.data_ptr() == counts.data_ptr()
        passed_norms = kwargs.pop("key_norm_sums")
        if inverse_mass:
            assert isinstance(passed_norms, torch.Tensor)
            assert passed_norms.data_ptr() == key_norm_sums.data_ptr()
        else:
            assert passed_norms is None
        assert kwargs == {
            "route_count": route_count,
            "state_len": 8,
            "kv_group_size": 2,
            "scale": 0.5,
            "normalize_route_query": routing_normalization == "query",
            "exact_mass_coverage": None,
            "local_lse": None,
            "sink_k": None,
            "buffers": None,
        }
        return (
            expected_routes,
            expected_coarse,
            expected_route_head_counts,
            expected_route_offsets,
        )

    monkeypatch.setattr(
        aiter_prefill_attention,
        "aiter_prefill_route_coarse_attention",
        fake_aiter,
    )
    routes = engine._route_top_slots(
        q,
        state_k,
        state_v,
        counts,
        key_norm_sums=key_norm_sums,
        state_len=8,
        state_capacity=8,
    )
    assert routes is expected_routes
    assert engine._lod_prefill_route_head_counts is expected_route_head_counts
    assert engine._lod_prefill_route_offsets is expected_route_offsets
    assert engine._lod_prefill_aiter_coarse is expected_coarse


def test_disabling_qwen_coherence_changes_only_assignment_geometry() -> None:
    from lod_attention._core import TritonLODAttentionCore

    engine = TritonLODAttentionCore()
    engine.state_clustering_normalization = "none"
    engine.state_clustering_centroid_rescale = "coherence"
    key = torch.tensor([[[[1.0, 1.0]]]])
    radial_rms = torch.tensor([[[[2.0]]]])
    engine.state_clustering_centroid_rescale_scope = "assignment"
    baseline_assignment = engine._state_clustering_key(
        key, role="centroid", radial_rms=radial_rms, purpose="assignment"
    )
    baseline_append = engine._state_clustering_key(
        key, role="centroid", radial_rms=radial_rms, purpose="append"
    )
    engine.state_clustering_centroid_rescale_scope = "none"
    ablated_assignment = engine._state_clustering_key(
        key, role="centroid", radial_rms=radial_rms, purpose="assignment"
    )
    ablated_append = engine._state_clustering_key(
        key, role="centroid", radial_rms=radial_rms, purpose="append"
    )
    torch.testing.assert_close(baseline_assignment, key / 2)
    torch.testing.assert_close(ablated_assignment, baseline_append)
    torch.testing.assert_close(ablated_append, baseline_append)
    assert engine._streaming_state_geometry() == "spherical"


def test_public_modes_map_to_the_three_cache_organizations() -> None:
    two = kernel_config(LODMode.TWO_TIER)
    bf16 = kernel_config(LODMode.THREE_TIER_BF16)
    int4 = kernel_config(LODMode.THREE_TIER_INT4)

    assert isinstance(two, LODConfig) and not isinstance(two, PagedLODConfig)
    assert isinstance(bf16, PagedLODConfig) and bf16.kv_bits == 0
    assert isinstance(int4, PagedLODConfig) and int4.kv_bits == 4
    assert {two.max_routes, bf16.max_routes, int4.max_routes} == {ROUTE_COUNT}


def test_mode_parser_rejects_old_configuration_names() -> None:
    with pytest.raises(ValueError, match="two-tier"):
        LODMode.parse("recursive-int8")


def test_only_the_base_and_prefix_rollback_windows_are_valid() -> None:
    assert LODConfig(local_window=PREFIX_CACHE_LOCAL_WINDOW).local_window == 1_024
    with pytest.raises(ValueError, match="prefix rollback window"):
        LODConfig(local_window=768)


def test_only_qwen38_and_k2_are_recognized() -> None:
    qwen_text = SimpleNamespace(
        model_type="qwen3_5_text",
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
    )
    qwen = SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        get_text_config=lambda decoder=True: qwen_text,
    )
    k2 = SimpleNamespace(
        model_type="k2_horizon",
        architectures=["K2HorizonForCausalLM"],
    )
    unsupported = SimpleNamespace(model_type="llama", architectures=[])

    assert model_family(qwen) is ModelFamily.QWEN38
    assert model_family(qwen_text) is ModelFamily.QWEN38
    assert model_family(k2) is ModelFamily.K2
    with pytest.raises(ValueError, match="only Qwen3.8 and K2 Horizon"):
        model_family(unsupported)


@pytest.mark.parametrize(
    ("family", "heads", "kv_heads", "head_dim"),
    [
        (ModelFamily.QWEN38, 24, 4, 256),
        (ModelFamily.K2, 64, 8, 128),
    ],
)
def test_profile_fixes_top_eight_for_both_families(
    family: ModelFamily, heads: int, kv_heads: int, head_dim: int
) -> None:
    engine = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
        ),
        head_dim=head_dim,
    )
    configure_engine(
        engine,
        family=family,
        mode=LODMode.THREE_TIER_INT4,
        request_capacity=131_072,
        has_query_norm=family is ModelFamily.QWEN38,
        has_key_norm=family is ModelFamily.QWEN38,
    )
    assert engine.two_level_topk == ROUTE_COUNT
    assert engine.prefill_two_level_topk == ROUTE_COUNT
    assert engine.recursive_prefill_all_leaves is True
    assert engine.recursive_state_route_backend == "fused"
    assert engine.separate_sink_cache is True
    assert engine.prefill_aiter_route_coarse is True
    assert engine.decode_state_update_len == 256
    assert engine.leaf_inline_pages_per_slot == 32
    assert engine.leaf_block_m == (256 if family is ModelFamily.K2 else 32)
    assert engine.leaf_block_n == 16
    assert engine.leaf_num_warps == (4 if family is ModelFamily.K2 else 2)
    configure_engine(
        engine,
        family=family,
        mode=LODMode.THREE_TIER_BF16,
        request_capacity=131_072,
        has_query_norm=family is ModelFamily.QWEN38,
        has_key_norm=family is ModelFamily.QWEN38,
    )
    assert engine.leaf_block_m == (128 if family is ModelFamily.K2 else 32)
    assert engine.leaf_block_n == (32 if family is ModelFamily.K2 else 16)


@pytest.mark.parametrize(
    ("mode", "exact_limit"),
    [
        (LODMode.TWO_TIER, EXACT_DECODE_LIMIT),
        (LODMode.THREE_TIER_BF16, EXACT_DECODE_LIMIT),
        (LODMode.THREE_TIER_INT4, EXACT_DECODE_LIMIT),
    ],
)
def test_modes_use_exact_short_decode(
    mode: LODMode, exact_limit: int
) -> None:
    engine = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=24,
            num_key_value_heads=4,
        ),
        head_dim=256,
    )
    configure_engine(
        engine,
        family=ModelFamily.QWEN38,
        mode=mode,
        request_capacity=131_072,
        has_query_norm=True,
        has_key_norm=True,
    )
    assert engine.exact_decode_limit == exact_limit


def test_materialized_maxsim_batch_chunk_bounds_dense_workspace() -> None:
    from lod_attention.kernels.lod_kernels import (
        _materialized_maxsim_batch_chunk,
    )

    geometry = {
        "batch": 16,
        "kv_heads": 8,
        "overflow_len": 16_384,
        "state_len": 4_096,
        "element_size": 2,
    }
    gib = 1 << 30
    assert (
        _materialized_maxsim_batch_chunk(
            **geometry, score_fields=1, free_bytes=64 * gib
        )
        == 16
    )
    assert (
        _materialized_maxsim_batch_chunk(
            **geometry, score_fields=1, free_bytes=8 * gib
        )
        == 4
    )
    assert (
        _materialized_maxsim_batch_chunk(
            **geometry, score_fields=1, free_bytes=3 * gib
        )
        == 2
    )
    assert (
        _materialized_maxsim_batch_chunk(
            **geometry, score_fields=2, free_bytes=8 * gib
        )
        == 2
    )


def test_direct_int4_finalization_releases_bf16_construction_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lod_attention._core as core

    engine = core.TritonLODAttentionCore()
    engine.leaf_quant_scale_mode = "l2"
    engine.leaf_append_quant_scale_mode = "l2"
    engine.leaf_quant_group_size = 2
    engine.leaf_quant_token_group_size = 2
    engine.page_summary_quant_bits = 8
    engine.page_summary_scale_mode = "l2"

    leaf_k = torch.randn(1, 1, 4, 4)
    leaf_v = torch.randn_like(leaf_k)
    destination = {
        "leaf_k": torch.empty(1, 1, 1, 4),
        "leaf_v": torch.empty(1, 1, 1, 4),
        "quantized_leaf_k": torch.empty(1, 1, 4, 2, dtype=torch.uint8),
        "quantized_leaf_v": torch.empty(1, 1, 4, 2, dtype=torch.uint8),
        "quantized_page_sum_k": torch.empty(1, 1, 2, 4, dtype=torch.int8),
        "quantized_page_sum_v": torch.empty(1, 1, 2, 4, dtype=torch.int8),
        "page_sum_k_scales": torch.empty(1, 1, 2, 2),
        "page_sum_v_scales": torch.empty(1, 1, 2, 2),
        "page_sum_k": torch.empty(1, 1, 1, 4),
        "page_sum_v": torch.empty(1, 1, 1, 4),
    }
    page_cache: dict[str, object] = {
        "leaf_k": leaf_k,
        "leaf_v": leaf_v,
        "page_indices": torch.zeros(1, 1, 2, 2, dtype=torch.int32),
        "page_sum_k": torch.randn(1, 1, 2, 4),
        "page_sum_v": torch.randn(1, 1, 2, 4),
        "page_counts": torch.ones(1, 1, 2, dtype=torch.int32),
        "quantized_leaf_k": destination["quantized_leaf_k"],
        "quantized_leaf_v": destination["quantized_leaf_v"],
        "page_k_scales": torch.empty(1, 1, 2, 4),
        "page_v_scales": torch.empty(1, 1, 2, 4),
        "page_quantized_counts": torch.zeros(1, 1, 2, dtype=torch.int32),
        "leaf_quant_bits": 4,
        "quantization_finalized": False,
    }
    quantize_calls = 0

    def fake_quantize(*args: object, **kwargs: object) -> None:
        nonlocal quantize_calls
        quantize_calls += 1
        assert kwargs["quant_group_size"] == 2
        assert kwargs["quant_token_group_size"] == 2
        assert kwargs["quant_bits"] == 4
        assert kwargs["optimize_scale"] is True
        args[6].fill_(3)
        args[7].fill_(5)

    def fake_summaries(
        page_sum_k: torch.Tensor,
        page_sum_v: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert kwargs == {"quant_group_size": 2, "optimize_scale": True}
        return (
            torch.full_like(page_sum_k, 7, dtype=torch.int8),
            torch.full_like(page_sum_v, 9, dtype=torch.int8),
            torch.full((1, 1, 2, 2), 0.5),
            torch.full((1, 1, 2, 2), 0.25),
        )

    monkeypatch.setattr(core, "quantize_virtual_paged_kv", fake_quantize)
    monkeypatch.setattr(core, "quantize_page_summaries_int8", fake_summaries)
    engine._finalize_virtual_page_quantization(
        page_cache,
        destination_page=destination,
    )

    assert quantize_calls == 1
    assert page_cache["quantization_finalized"] is True
    assert page_cache["summary_quantization_finalized"] is True
    assert page_cache["leaf_k"].data_ptr() == destination["leaf_k"].data_ptr()
    assert page_cache["leaf_v"].data_ptr() == destination["leaf_v"].data_ptr()
    assert page_cache["page_sum_k"].data_ptr() == destination["page_sum_k"].data_ptr()
    assert page_cache["page_sum_v"].data_ptr() == destination["page_sum_v"].data_ptr()
    assert bool(destination["quantized_leaf_k"].eq(3).all())
    assert bool(destination["quantized_leaf_v"].eq(5).all())
    assert bool(destination["quantized_page_sum_k"].eq(7).all())
    assert bool(destination["quantized_page_sum_v"].eq(9).all())

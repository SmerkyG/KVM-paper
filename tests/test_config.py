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
def test_aiter_prefill_reuses_fused_coarse_result(
    monkeypatch: pytest.MonkeyPatch,
    routing_normalization: str,
) -> None:
    from lod_attention._core import TritonLODAttentionCore
    from lod_attention.kernels import aiter_prefill_attention

    engine = TritonLODAttentionCore()
    engine.num_key_value_groups = 2
    engine.scaling = 0.5
    engine.two_level_topk = 4
    engine.prefill_two_level_topk = 4
    engine.separate_sink_cache = True
    engine.routing_normalization = routing_normalization
    engine.prefill_aiter_route_coarse = True
    engine.split_prefill_local_attention = True
    engine.mla_state_key_normalization = "none"

    q = torch.randn(1, 4, 3, 4)
    state_k = torch.randn(1, 2, 8, 4)
    state_v = torch.randn_like(state_k)
    counts = torch.randint(1, 5, (1, 2, 8, 1)).float()
    expected_routes = torch.zeros(1, 4, 3, 4, dtype=torch.long)
    expected_output = torch.randn_like(q)
    expected_lse = torch.randn(1, 4, 3)

    def fake_aiter(
        passed_q: torch.Tensor,
        mean_k: torch.Tensor,
        passed_v: torch.Tensor,
        passed_counts: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.testing.assert_close(passed_q, q)
        torch.testing.assert_close(mean_k, state_k / counts)
        assert passed_v.data_ptr() == state_v.data_ptr()
        assert passed_counts.data_ptr() == counts.data_ptr()
        assert kwargs == {
            "state_len": 8,
            "kv_group_size": 2,
            "scale": 0.5,
            "normalize_route_query": routing_normalization == "query",
        }
        return expected_routes, expected_output, expected_lse

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
        state_len=8,
        state_capacity=8,
    )
    assert routes is expected_routes

    def fail_route(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("fused coarse result was not reused")

    monkeypatch.setattr(engine, "_state_route_logits", fail_route)
    output, lse = engine._coarse_attention(
        q,
        q[:, :2],
        q[:, :2],
        state_k,
        state_v,
        counts,
        routes,
        state_len=8,
        state_capacity=8,
        include_local=False,
    )
    assert output is expected_output
    assert lse is expected_lse
    assert not hasattr(engine, "_lod_prefill_fused_coarse")


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
def test_profile_fixes_top_four_for_both_families(
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
    assert engine.separate_sink_cache is True
    assert engine.prefill_aiter_route_coarse is True
    assert engine.leaf_block_m == (64 if family is ModelFamily.K2 else 32)
    assert engine.leaf_num_warps == (4 if family is ModelFamily.K2 else 2)


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

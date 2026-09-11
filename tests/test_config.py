from __future__ import annotations

from types import SimpleNamespace

import pytest

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
        has_query_norm=True,
        has_key_norm=True,
    )
    assert engine.two_level_topk == ROUTE_COUNT
    assert engine.prefill_two_level_topk == ROUTE_COUNT
    assert engine.recursive_prefill_all_leaves is True
    assert engine.separate_sink_cache is True


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

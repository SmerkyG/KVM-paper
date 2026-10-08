from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from lod_attention import LODMode, PytorchLODAttention, PytorchLODConfig
from lod_attention._config import (
    CHUNK_SIZE,
    EXACT_DECODE_LIMIT,
    LOCAL_WINDOW,
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    ROUTE_COUNT,
    LODConfig,
    ModelFamily,
)
from lod_attention._hf_backend import HFLODCache, HFLODCacheLayer, HFLODSettings


def _dense_causal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    query_heads = int(query.size(1))
    groups = query_heads // int(key.size(1))
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))
    length = int(query.size(2))
    position = torch.arange(length)
    visible = position.unsqueeze(0) <= position.unsqueeze(1)
    visible = visible.view(1, 1, length, length)
    if valid_starts is not None:
        visible = visible & (
            position.view(1, 1, 1, length) >= valid_starts.view(-1, 1, 1, 1)
        )
    score = score.masked_fill(~visible, -torch.inf)
    probability = torch.softmax(score, dim=-1)
    if valid_starts is not None:
        query_valid = position.view(1, 1, length, 1) >= valid_starts.view(-1, 1, 1, 1)
        probability = torch.where(query_valid, probability, 0)
    return torch.matmul(probability, value)


def _small_config() -> PytorchLODConfig:
    return PytorchLODConfig(
        chunk_size=2,
        local_window=4,
        prefill_block_size=4,
        prefill_lookback=1,
        state_growth_factor=8.0,
        state_min_size=2,
        route_count=8,
        max_region_size=1_024,
        page_size=2,
        exact_decode_limit=0,
    )


def test_reference_defaults_match_the_release_methodology() -> None:
    config = PytorchLODConfig()
    assert config.chunk_size == CHUNK_SIZE == 256
    assert config.local_window == LOCAL_WINDOW == 512
    assert config.prefill_block_size == PREFILL_CHUNK_SIZE == 16_384
    assert config.prefill_lookback == CHUNK_SIZE
    assert config.state_growth_factor == 16.0
    assert config.route_count == ROUTE_COUNT == 8
    assert config.max_region_size == 1_024
    assert config.page_size == PAGE_SIZE == 16
    assert config.exact_decode_limit == EXACT_DECODE_LIMIT == 2_048


def test_two_tier_matches_dense_when_every_region_is_open() -> None:
    torch.manual_seed(1)
    query = torch.randn(1, 2, 8, 4)
    key = torch.randn(1, 1, 8, 4)
    value = torch.randn(1, 1, 8, 3)
    engine = PytorchLODAttention(
        _small_config(),
        mode=LODMode.TWO_TIER,
        normalized_keys=False,
        normalize_routing_query=True,
    )

    output, cache = engine(query, key, value, use_cache=True)

    assert cache is not None
    torch.testing.assert_close(output, _dense_causal(query, key, value))
    torch.testing.assert_close(cache.sink_key, key[..., :1, :])
    assert cache.state.size <= math.floor(8.0 * math.sqrt(8))
    assert cache.coverage == 3

    next_query = torch.randn(1, 2, 1, 4)
    next_key = torch.randn(1, 1, 1, 4)
    next_value = torch.randn(1, 1, 1, 3)
    decode, next_cache = engine(
        next_query,
        next_key,
        next_value,
        cache=cache,
        use_cache=True,
    )
    assert next_cache is not None and next_cache.total_length == 9
    full_query = torch.cat((query, next_query), dim=2)
    full_key = torch.cat((key, next_key), dim=2)
    full_value = torch.cat((value, next_value), dim=2)
    expected = _dense_causal(full_query, full_key, full_value)[..., -1:, :]
    torch.testing.assert_close(decode, expected)


def test_left_padding_uses_first_real_token_as_separate_sink() -> None:
    torch.manual_seed(2)
    query = torch.randn(2, 2, 8, 4)
    key = torch.randn(2, 1, 8, 4)
    value = torch.randn(2, 1, 8, 3)
    valid_starts = torch.tensor([1, 2])
    engine = PytorchLODAttention(replace(_small_config(), exact_decode_limit=32))

    output, cache = engine(
        query,
        key,
        value,
        use_cache=True,
        prefill_valid_starts=valid_starts,
    )

    assert cache is not None
    torch.testing.assert_close(cache.sink_key[0], key[0, :, 1:2])
    torch.testing.assert_close(cache.sink_key[1], key[1, :, 2:3])
    assert cache.owner[0, :, 0].eq(-1).all()
    assert cache.owner[1, :, :2].eq(-1).all()
    expected = _dense_causal(query, key, value, valid_starts=valid_starts)
    torch.testing.assert_close(output, expected)

    next_query = torch.randn(2, 2, 1, 4)
    next_key = torch.randn(2, 1, 1, 4)
    next_value = torch.randn(2, 1, 1, 3)
    decode, _ = engine(
        next_query,
        next_key,
        next_value,
        cache=cache,
        use_cache=True,
    )
    expected_decode = _dense_causal(
        torch.cat((query, next_query), dim=2),
        torch.cat((key, next_key), dim=2),
        torch.cat((value, next_value), dim=2),
        valid_starts=valid_starts,
    )[..., -1:, :]
    torch.testing.assert_close(decode, expected_decode)


def test_two_tier_replaces_only_the_selected_summary() -> None:
    from lod_attention.pytorch_engine import PytorchLODCache, PytorchLODState

    key = torch.tensor([[[[3.0], [2.0], [-1.0], [-2.0]]]])
    value = torch.tensor([[[[8.0], [4.0], [2.0], [0.0]]]])
    owner = torch.tensor([[[0, 0, 1, 1]]])
    state = PytorchLODState(
        key_sum=torch.tensor([[[[5.0], [-3.0]]]]),
        value_sum=torch.tensor([[[[12.0], [2.0]]]]),
        count=torch.tensor([[[2.0, 2.0]]]),
        key_rms_sum=torch.tensor([[[5.0, 3.0]]]),
    )
    cache = PytorchLODCache(
        state=state,
        owner=owner,
        archive_key=key,
        archive_value=value,
        archive_valid=torch.ones(1, 4, dtype=torch.bool),
        coverage=4,
        sink_key=torch.tensor([[[[-4.0]]]]),
        sink_value=torch.tensor([[[[3.0]]]]),
        total_length=6,
    )
    engine = PytorchLODAttention(
        replace(_small_config(), route_count=1),
        normalize_routing_query=False,
    )
    query = torch.ones(1, 1, 1, 1)
    local_key = torch.tensor([[[[-3.0]]]])
    local_value = torch.tensor([[[[5.0]]]])

    result = engine._lod_attention(
        query,
        local_key,
        local_value,
        cache,
        scale=1.0,
        query_offset=0,
        full_regions=True,
    )

    # Region 0 is exact; region 1 contributes its mean with log population.
    frontier_score = torch.tensor([3.0, 2.0, -1.5 + math.log(2), -3.0, -4.0])
    frontier_value = torch.tensor([[8.0], [4.0], [1.0], [5.0], [3.0]])
    expected = (frontier_score.softmax(dim=0).unsqueeze(-1) * frontier_value).sum(dim=0)
    torch.testing.assert_close(result.output[0, 0, 0], expected)
    assert result.routes.item() == 0


def test_region_cap_filters_after_ranking_without_substitution() -> None:
    from lod_attention.pytorch_engine import PytorchLODState

    state = PytorchLODState(
        key_sum=torch.tensor([[[[10.0], [1.0]]]]),
        value_sum=torch.tensor([[[[5.0], [1.0]]]]),
        count=torch.tensor([[[5.0, 1.0]]]),
        key_rms_sum=torch.tensor([[[5.0, 1.0]]]),
    )
    engine = PytorchLODAttention(
        replace(_small_config(), route_count=1, max_region_size=4),
        normalize_routing_query=False,
    )

    _, _, routes = engine._routes(torch.ones(1, 1, 1, 1), state, scale=1.0)

    # Region 0 wins the uncapped ranking but is too large to refine. The
    # lower-ranked region 1 must not be substituted; both remain coarse.
    assert routes.item() == -1


@pytest.mark.parametrize("remote_length", [2, 4, 6])
def test_recursive_mode_uses_two_exact_pages_and_disjoint_residual(remote_length) -> None:
    from lod_attention.pytorch_engine import PytorchLODCache, PytorchLODState

    key = torch.tensor([[[[3.0], [2.0], [1.0], [0.0], [-1.0], [-2.0]]]])[..., :remote_length, :]
    value = torch.tensor([[[[8.0], [4.0], [2.0], [1.0], [6.0], [0.0]]]])[..., :remote_length, :]
    state = PytorchLODState(
        key_sum=key.sum(dim=2, keepdim=True),
        value_sum=value.sum(dim=2, keepdim=True),
        count=torch.tensor([[[float(remote_length)]]]),
        key_rms_sum=key.abs().sum(dim=2),
    )
    cache = PytorchLODCache(
        state=state,
        owner=torch.zeros(1, 1, remote_length, dtype=torch.long),
        archive_key=key,
        archive_value=value,
        archive_valid=torch.ones(1, remote_length, dtype=torch.bool),
        coverage=remote_length,
        sink_key=torch.tensor([[[[-2.0]]]]),
        sink_value=torch.tensor([[[[3.0]]]]),
        total_length=remote_length + 2,
    )
    engine = PytorchLODAttention(
        PytorchLODConfig(
            chunk_size=2,
            local_window=4,
            prefill_block_size=4,
            prefill_lookback=1,
            state_growth_factor=1.0,
            state_min_size=1,
            route_count=1,
            max_region_size=1_024,
            page_size=2,
            exact_decode_limit=0,
        ),
        mode=LODMode.THREE_TIER_BF16,
        normalize_routing_query=False,
    )
    query = torch.ones(1, 1, 1, 1)
    local_key = torch.tensor([[[[-1.0]]]])
    local_value = torch.tensor([[[[5.0]]]])

    result = engine._lod_attention(
        query,
        local_key,
        local_value,
        cache,
        scale=1.0,
        query_offset=0,
        full_regions=False,
    )

    # Both leading pages are exact. Any third page is one disjoint residual.
    opened = min(remote_length, 4)
    remote_scores = [3.0, 2.0, 1.0, 0.0][:opened]
    remote_values = [8.0, 4.0, 2.0, 1.0][:opened]
    if remote_length == 6:
        remote_scores.append(-1.5 + math.log(2))
        remote_values.append(3.0)
    frontier_score = torch.tensor(remote_scores + [-1.0, -2.0])
    frontier_value = torch.tensor(remote_values + [5.0, 3.0]).unsqueeze(-1)
    expected = (frontier_score.softmax(dim=0).unsqueeze(-1) * frontier_value).sum(dim=0)
    torch.testing.assert_close(result.output[0, 0, 0], expected)


def test_int4_is_explicitly_not_a_reference_engine_mode() -> None:
    with pytest.raises(ValueError, match="BF16 frontiers"):
        PytorchLODAttention(mode=LODMode.THREE_TIER_INT4)


def test_hf_cache_runs_prefill_and_decode_with_the_pytorch_engine() -> None:
    module = nn.Module()
    module.layer_idx = 0
    module.scaling = 0.5
    settings = HFLODSettings(
        config=LODConfig(),
        family=ModelFamily.QWEN38,
        mode=LODMode.TWO_TIER,
        request_capacity=4_096,
        has_query_norm=True,
        has_key_norm=True,
        implementation="pytorch",
    )
    module._hf_lod_settings = settings
    module._hf_lod_active_cache_layer = None
    layer = HFLODCacheLayer(module, settings)
    outer_cache = HFLODCache([layer])

    query = torch.randn(1, 2, 4, 4)
    key = torch.randn(1, 1, 4, 4)
    value = torch.randn(1, 1, 4, 3)
    staged_key, staged_value = layer.update(key, value)
    assert staged_key is key and staged_value is value
    output = layer.consume(
        module,
        query,
        key,
        value,
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        scale=0.5,
    )
    assert tuple(output.shape) == (1, 2, 4, 3)
    assert layer.total_length == 4
    assert layer.lod_cache is not None

    next_query = torch.randn(1, 2, 1, 4)
    next_key = torch.randn(1, 1, 1, 4)
    next_value = torch.randn(1, 1, 1, 3)
    layer.update(next_key, next_value)
    decode = layer.consume(
        module,
        next_query,
        next_key,
        next_value,
        attention_mask=None,
        scale=0.5,
    )
    assert tuple(decode.shape) == (1, 2, 1, 3)
    assert layer.total_length == 5
    assert outer_cache.get_seq_length() == 5

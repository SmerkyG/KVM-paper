from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from lod_attention._config import LODMode, ModelFamily
from lod_attention._profile import configure_engine
from lod_attention.kernels.paged_leaf_attention import (
    dequantize_owned_virtual_paged_keys,
)
from lod_attention.kernels.aiter_mla_prefill_attention import expand_kimi_leaf_kv
import vllm_lod_plugin.models.kimi_k3 as kimi_k3
from vllm_lod_plugin.models.kimi_k3 import (
    absorb_query,
    pack_latent_record,
)
from vllm_lod_plugin.pool import VLLMLayerLODPool


def _dcp_schedule_fixture(rank: int = 0) -> VLLMLayerLODPool:
    pool = VLLMLayerLODPool.__new__(VLLMLayerLODPool)
    pool.dcp_world_size = 8
    pool.dcp_rank = rank
    pool.dcp_interleave_size = 1
    pool.engine = SimpleNamespace(
        state_growth_factor=16.0,
        state_min_len=256,
        chunk_len=256,
        local_len=512,
        prefill_chunk_len=16_384,
        prefill_local_len=16_640,
        prefill_state_update_len=16_384,
        decode_state_update_len=256,
        decode_cache_headroom=256,
    )
    pool._dcp_global_state_growth_factor = 16.0
    pool._dcp_global_state_min_len = 256
    pool._dcp_global_lengths = {
        name: int(getattr(pool.engine, name))
        for name in (
            "chunk_len",
            "local_len",
            "prefill_chunk_len",
            "prefill_local_len",
            "prefill_state_update_len",
            "decode_state_update_len",
            "decode_cache_headroom",
        )
    }
    return pool


def test_dcp_local_state_schedule_scales_every_token_count() -> None:
    pool = _dcp_schedule_fixture()
    with pool._dcp_local_state_schedule():
        assert pool.engine.state_growth_factor == 16.0 / math.sqrt(8)
        assert pool.engine.state_min_len == 32
        assert (
            pool.engine.chunk_len,
            pool.engine.local_len,
            pool.engine.prefill_chunk_len,
            pool.engine.prefill_local_len,
            pool.engine.prefill_state_update_len,
            pool.engine.decode_state_update_len,
            pool.engine.decode_cache_headroom,
        ) == (32, 64, 2_048, 2_080, 2_048, 32, 32)

    assert pool.engine.state_growth_factor == 16.0
    assert pool.engine.state_min_len == 256
    assert (
        pool.engine.chunk_len,
        pool.engine.local_len,
        pool.engine.prefill_chunk_len,
        pool.engine.prefill_local_len,
        pool.engine.prefill_state_update_len,
        pool.engine.decode_state_update_len,
        pool.engine.decode_cache_headroom,
    ) == (256, 512, 16_384, 16_640, 16_384, 256, 256)


def test_dcp_catch_up_uses_rank_local_share_of_global_update() -> None:
    # The trigger is the same global per-request boundary on every rank.  At
    # DCP8 each rank then archives its 32 owned leaves: 256 global leaves in
    # aggregate.  Batch size never enters this calculation, so B8 performs
    # 8 * 256 source-token work at the aligned update.
    for rank in range(8):
        pool = _dcp_schedule_fixture(rank)
        pool.dcp_sharded = [True]
        pool.local_capacity = 96
        pool.metadata = [{"coverage": 8_160, "dcp_global_coverage": 65_280}]

        recent, target = pool._catch_up_target(0, 65_791)
        assert recent in (63, 64)
        assert target == 8_160

        recent, target = pool._catch_up_target(0, 65_792)
        assert recent == 64
        assert target == 8_192


def test_dcp_prefill_boundary_is_global_not_rank_local() -> None:
    # Immediately before the next global boundary, ranks own unequal local
    # lengths.  None may advance early merely because its local count rounded
    # up to another 32-token block.
    expected = []
    for rank in range(8):
        pool = _dcp_schedule_fixture(rank)
        expected.append(pool._dcp_local_length(65_280))
        assert pool._dcp_global_decode_coverage(65_791) == 65_280
        assert pool._dcp_global_decode_coverage(65_792) == 65_536
    assert expected == [8_160] * 8


def test_dcp_batch_does_not_divide_the_per_request_cadence() -> None:
    local_work = 0
    for rank in range(8):
        pool = _dcp_schedule_fixture(rank)
        pool.dcp_sharded = [True] * 8
        pool.local_capacity = 96
        pool.metadata = [
            {"coverage": 8_160, "dcp_global_coverage": 65_280}
            for _ in range(8)
        ]
        for slot in range(8):
            _recent, target = pool._catch_up_target(slot, 65_792)
            local_work += target - int(pool.metadata[slot]["coverage"])
    assert local_work == 8 * 256


def test_full_k3_profile_uses_launchable_unmasked_mla_geometry() -> None:
    engine = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=96, num_key_value_heads=1),
        head_dim=576,
    )
    configure_engine(
        engine,
        family=ModelFamily.KIMI_K3,
        mode=LODMode.THREE_TIER_BF16,
        request_capacity=16_640,
        has_query_norm=False,
        has_key_norm=True,
    )
    assert (engine.leaf_block_m, engine.leaf_block_n) == (32, 16)
    assert engine.prefill_aiter_route_coarse is True


def test_absorbed_latent_logits_equal_expanded_mla_logits() -> None:
    torch.manual_seed(7)
    tokens, heads, nope, latent_dim, direct = 11, 8, 64, 128, 32
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)
    w_uk_t = torch.randn(heads, nope, latent_dim)

    absorbed = absorb_query(query, w_uk_t, nope_dim=nope)
    key = pack_latent_record(latent, direct_key)
    value = key[..., :latent_dim]

    expanded_nope = torch.einsum("tl,hpl->thp", latent, w_uk_t)
    expanded_key = torch.cat(
        (expanded_nope, direct_key.expand(-1, heads, -1)), dim=-1
    )
    expanded_logits = torch.einsum("thd,shd->hts", query, expanded_key)
    latent_logits = torch.einsum("thd,shd->hts", absorbed, key)

    # The two equivalent expressions use different contraction orders, so
    # float32 roundoff grows slightly with the latent width.
    torch.testing.assert_close(latent_logits, expanded_logits, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(value, latent.unsqueeze(1))
    assert value.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()


def test_projected_prefill_defers_absorbed_query_materialization(monkeypatch) -> None:
    # Query-deferred projected prefill is a full-K3 specialization.  The small
    # K3-for-All geometry intentionally uses the generic absorbed-query path.
    tokens, heads, nope, latent_dim, direct, value_dim = 5, 8, 128, 512, 64, 128
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)
    w_uk_t = torch.randn(heads, nope, latent_dim)
    w_uv = torch.randn(heads, latent_dim, value_dim)
    observed: dict[str, object] = {}

    class Pool:
        direct_prefill_plan = ((0, 0, tokens, 0),)
        decode_enabled = False

        def direct_prefill(
            self,
            absorbed_shape_carrier,
            key,
            value,
            output,
            **kwargs,
        ):
            observed["carrier_stride"] = absorbed_shape_carrier.stride()
            observed["key_shape"] = tuple(key.shape)
            observed["value_aliases_key"] = (
                value.untyped_storage().data_ptr()
                == key.untyped_storage().data_ptr()
            )
            observed.update(kwargs)
            output.fill_(3)
            return output

    layer = SimpleNamespace(
        _vllm_lod_pool=Pool(),
        W_UK_T=w_uk_t,
        W_UV=w_uv,
        qk_nope_head_dim=nope,
        num_heads=heads,
        v_head_dim=value_dim,
    )

    def unexpected_absorption(*_args, **_kwargs):
        raise AssertionError("projected prefill must not materialize absorbed Q")

    monkeypatch.setattr(kimi_k3, "absorb_query", unexpected_absorption)
    result = kimi_k3._run_lod_mla(
        layer,
        query,
        latent,
        direct_key,
        (tokens, heads * value_dim),
        None,
    )

    assert result.shape == (tokens, heads * value_dim)
    assert torch.all(result == 3)
    assert observed["carrier_stride"][1] == 0
    assert observed["key_shape"] == (tokens, 1, latent_dim + direct)
    assert observed["value_aliases_key"] is True
    assert observed["mla_query"] is query
    assert observed["mla_w_uk_t"] is w_uk_t
    assert observed["mla_w_uv"] is w_uv
    assert observed["defer_mla_query_absorption"] is True


def test_graph_warmup_rows_do_not_enter_smaller_decode_pool() -> None:
    tokens, heads, nope, latent_dim, direct = 16, 2, 4, 8, 2
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)

    class Pool:
        direct_prefill_plan = None
        decode_enabled = True
        max_requests = 1

        def decode(self, *_args, **_kwargs):
            raise AssertionError("synthetic graph rows must not enter decode")

    def copy_output(attention_output, output):
        output.copy_(attention_output.reshape(tokens, heads * latent_dim))

    layer = SimpleNamespace(
        _vllm_lod_pool=Pool(),
        W_UK_T=torch.randn(heads, nope, latent_dim),
        qk_nope_head_dim=nope,
        num_heads=heads,
        v_head_dim=latent_dim,
        _v_up_proj=copy_output,
    )
    result = kimi_k3._run_lod_mla(
        layer,
        query,
        latent,
        direct_key,
        (tokens, heads * latent_dim),
        None,
    )
    assert torch.count_nonzero(result) == 0


def test_latent_centroid_is_exact_expanded_kv_centroid() -> None:
    torch.manual_seed(11)
    count, heads, nope, latent_dim, direct, value_dim = 13, 8, 64, 128, 32, 64
    latent = torch.randn(count, latent_dim)
    direct_key = torch.randn(count, direct)
    w_uk_t = torch.randn(heads, nope, latent_dim)
    w_uv = torch.randn(heads, latent_dim, value_dim)
    query = torch.randn(heads, nope + direct)

    q_absorbed = absorb_query(query.unsqueeze(0), w_uk_t, nope_dim=nope)[0]
    summed_key = torch.cat((latent.sum(0), direct_key.sum(0)))
    latent_score = torch.einsum("hd,d->h", q_absorbed, summed_key)

    expanded_key_sum = torch.cat(
        (
            torch.einsum("l,hpl->hp", latent.sum(0), w_uk_t),
            direct_key.sum(0).expand(heads, -1),
        ),
        dim=-1,
    )
    expanded_score = torch.einsum("hd,hd->h", query, expanded_key_sum)
    torch.testing.assert_close(latent_score, expanded_score, atol=3e-5, rtol=3e-5)

    latent_mean = latent.mean(0).expand(heads, -1)
    projected_mean = torch.einsum("hl,hlv->hv", latent_mean, w_uv)
    expanded_values = torch.einsum("tl,hlv->thv", latent, w_uv)
    torch.testing.assert_close(
        projected_mean,
        expanded_values.mean(0),
        atol=2e-5 * math.sqrt(count),
        rtol=2e-5,
    )


def test_multihead_leaf_expansion_writes_projected_key_prefix() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(13)
    tokens, heads = 19, 12
    latent_key = torch.randn(
        1, 1, tokens, 576, dtype=torch.bfloat16, device="cuda"
    )
    w_uk_t = torch.randn(
        heads, 128, 512, dtype=torch.bfloat16, device="cuda"
    )
    w_uv = torch.randn(heads, 512, 128, dtype=torch.bfloat16, device="cuda")
    expanded_k, expanded_v = expand_kimi_leaf_kv(latent_key, w_uk_t, w_uv)

    latent = latent_key[0, 0, :, :512].float()
    expected_nope = torch.einsum("tl,hpl->htp", latent, w_uk_t.float())
    expected_direct = latent_key[0, 0, :, 512:].expand(heads, -1, -1)
    expected_v = torch.einsum("tl,hlv->htv", latent, w_uv.float())
    torch.testing.assert_close(
        expanded_k[0, ..., :128].float(), expected_nope, atol=0.5, rtol=0.02
    )
    torch.testing.assert_close(expanded_k[0, ..., 128:], expected_direct)
    torch.testing.assert_close(
        expanded_v[0].float(), expected_v, atol=0.5, rtol=0.02
    )


def test_count_channel_equals_centroid_mass_bias() -> None:
    """A padded dot-product channel exactly represents ``+ log(count)``."""
    torch.manual_seed(19)
    queries, states, heads = 7, 11, 3
    scale = 192**-0.5
    query = torch.randn(queries, heads, 192)
    key = torch.randn(states, heads, 192)
    counts = torch.randint(1, 128, (states,), dtype=torch.int64).float()

    expected = torch.einsum("qhd,khd->hqk", query, key) * scale
    expected += counts.log()[None, None, :]

    augmented_query = torch.zeros(queries, heads, 256)
    augmented_key = torch.zeros(states, heads, 256)
    augmented_query[..., :192] = query
    augmented_key[..., :192] = key
    augmented_query[..., 192] = 1.0 / scale
    augmented_key[..., 192] = counts.log()[:, None]
    actual = torch.einsum(
        "qhd,khd->hqk", augmented_query, augmented_key
    ) * scale
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_dcp_int4_page_dequantization_restores_owned_records() -> None:
    if not torch.cuda.is_available():
        return
    batch, heads, pages, page_size, head_dim = 1, 2, 2, 16, 8
    leaves = pages * page_size
    page_indices = (
        torch.arange(leaves, dtype=torch.int32, device="cuda")
        .view(batch, 1, pages, page_size)
        .expand(batch, heads, pages, page_size)
        .contiguous()
    )
    page_counts = torch.full(
        (batch, heads, pages), page_size, dtype=torch.int32, device="cuda"
    )
    next_page = torch.full(
        (batch, heads), pages, dtype=torch.int32, device="cuda"
    )
    # Low nibbles reconstruct to the page centroid; high nibbles add one.
    packed_keys = torch.full(
        (batch, heads, leaves, head_dim // 2),
        0x98,
        dtype=torch.uint8,
        device="cuda",
    )
    page_scales = torch.ones(
        batch, heads, pages, head_dim // 4, dtype=torch.bfloat16, device="cuda"
    )
    quantized_sums = torch.full(
        (batch, heads, pages, head_dim),
        32,
        dtype=torch.int8,
        device="cuda",
    )
    summary_scales = torch.full(
        (batch, heads, pages, head_dim // 4),
        0.5,
        dtype=torch.bfloat16,
        device="cuda",
    )

    for rank in range(2):
        actual = dequantize_owned_virtual_paged_keys(
            page_indices,
            page_counts,
            next_page,
            packed_keys,
            page_scales,
            quantized_sums,
            summary_scales,
            source_slot=0,
            sink_len=0,
            local_length=leaves // 2,
            dcp_rank=rank,
            dcp_world_size=2,
            dcp_interleave_size=4,
        )
        expected = torch.empty_like(actual)
        expected[..., 0::2] = 1.0
        expected[..., 1::2] = 2.0
        torch.testing.assert_close(actual, expected)

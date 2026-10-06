"""Six 16-head owners preserve native TP8 head ordering and LoD arithmetic."""

from types import SimpleNamespace

import pytest
import torch

from vllm_lod_plugin.models.kimi_k3_head_prefill import exchange_heads, head_transfers


def test_transfers_cover_all_heads_once_with_two_sources_per_owner():
    transfers = head_transfers()
    assert len(transfers) == 12
    covered = []
    for owner in range(6):
        parts = [part for part in transfers if part[0] == owner]
        assert len(parts) == 2 and sum(part[-1] for part in parts) == 16
        for _, source, local, owner_local, width in parts:
            assert source * 12 + local == owner * 16 + owner_local
            covered.extend(range(source * 12 + local, source * 12 + local + width))
    assert covered == list(range(96))


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("returning", [False, True])
def test_regroup_and_return_exact_head_order(rank, returning):
    tokens, channels = 3, 7
    full = torch.arange(tokens * 96 * channels).reshape(tokens, 96, channels).float()
    transfers = head_transfers()
    packets = [(source, owner, owner_offset, source_offset, width)
               if returning else (owner, source, source_offset, owner_offset, width)
               for owner, source, source_offset, owner_offset, width in transfers]
    received = iter(part for part in packets if part[0] == rank and part[1] != rank)
    sent = iter(part for part in packets if part[1] == rank and part[0] != rank)

    def recv(target, peer):
        dest, source, local, _, width = next(received)
        assert source == peer and dest == rank and target.is_contiguous()
        start = source * (16 if returning else 12) + local
        target.copy_(full[:, start:start + width])

    def send(packet, peer):
        dest, source, local, _, width = next(sent)
        assert dest == peer and source == rank and packet.is_contiguous()
        start = source * (16 if returning else 12) + local
        torch.testing.assert_close(packet, full[:, start:start + width], atol=0, rtol=0)

    group = SimpleNamespace(rank_in_group=rank, world_size=8,
        device_communicator=SimpleNamespace(pynccl_comm=SimpleNamespace(disabled=False,
            group_start=lambda: None, group_end=lambda: None, recv=recv, send=send)))
    width = (16 if rank < 6 else 0) if returning else 12
    first = rank * (16 if returning else 12)
    source = full[:, first:first + width].clone()
    result = exchange_heads(group, source, returning=returning)
    output_width = 12 if returning else (16 if rank < 6 else 0)
    output_first = rank * (12 if returning else 16)
    torch.testing.assert_close(result, full[:, output_first:output_first + output_width], atol=0, rtol=0)
    assert next(received, None) is None and next(sent, None) is None
    buffers = group._lod_owner_transport
    old_pointers = {name: tensor.data_ptr() for name, tensor in buffers.items()}
    # Grouping scratch must be reused for identical shapes across MLA layers.
    received = iter(part for part in packets if part[0] == rank and part[1] != rank)
    sent = iter(part for part in packets if part[1] == rank and part[0] != rank)
    again = exchange_heads(group, source, returning=returning)
    torch.testing.assert_close(again, result, atol=0, rtol=0)
    assert {name: tensor.data_ptr() for name, tensor in buffers.items()} == old_pointers


def test_six_owner_audit_rejects_missing_or_wrong_head_ranges():
    from benchmarks.kimi_k3_prefill_sweep import validate_owner_prefill_audits

    audits = [dict(rank=rank, owned_query_sizes={16384: 1} if rank < 6 else {},
        head_owner_ranges=[(rank * 16, (rank + 1) * 16)] if rank < 6 else [])
        for rank in range(8)]
    assert validate_owner_prefill_audits(audits, row_chunk=16384, world_size=8,
        batch_size=1, head_owners=True) == set(range(6))
    for audit in audits[:6]:
        audit["head_owner_last_states"] = {"layer": dict(total_len=32768,
            coverage=32512, state_len=512, prefill_update_len=16384, query_heads=16)}
    assert validate_owner_prefill_audits(audits, row_chunk=16384, world_size=8,
        batch_size=1, head_owners=True, length=32768) == set(range(6))
    with pytest.raises(RuntimeError, match="history/cadence"):
        validate_owner_prefill_audits(audits, row_chunk=16384, world_size=8,
            batch_size=1, head_owners=True, length=131072)
    audits[5]["head_owner_last_states"]["layer"]["state_len"] += 1
    with pytest.raises(RuntimeError, match="different global"):
        validate_owner_prefill_audits(audits, row_chunk=16384, world_size=8,
            batch_size=1, head_owners=True, length=32768)
    audits[0]["head_owner_ranges"] = [(0, 12)]
    with pytest.raises(RuntimeError, match="incorrect head range"):
        validate_owner_prefill_audits(audits, row_chunk=16384, world_size=8,
            batch_size=1, head_owners=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AITER K3 GPU equivalence")
def test_six_head_groups_match_native_twelve_head_groups():
    from lod_attention._config import LODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine
    from vllm_lod_plugin.models.kimi_k3_request_prefill import attend_slice

    torch.manual_seed(73)
    def rand(shape, scale):
        return (torch.randn(shape, device="cuda") * scale).bfloat16()
    key = rand((16512, 1, 576), .2)
    query = rand((128, 96, 192), .1)
    uk, uv = rand((96, 128, 512), .03), rand((96, 512, 128), .03)

    def engine(heads):
        result = KernelTwoLevelLODAttention(LODConfig(state_clustering_normalization="cosine"),
            query_heads=heads, key_value_heads=1, scale=.072)
        result.head_dim = 576
        configure_engine(result, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
            request_capacity=32768, has_query_norm=True, has_key_norm=False)
        result._lod_kimi_reduce_prefill_routes = True
        return result

    outputs = {}
    states = []
    for heads in (12, 16):
        result = []
        for first in range(0, 96, heads):
            eng = engine(heads)
            initial = key[:16384].permute(1, 0, 2).unsqueeze(0)
            cache = eng.build_cache_from_bf16(initial, initial[..., :512], finalize_cache_for_decode=False)
            pool = SimpleNamespace(engine=eng, _kimi_request_owner_rows={
                0: dict(cache=cache, parts=[], total=16384)})
            result.append(attend_slice(pool, 0, query[:, first:first + heads], key[16384:],
                uk[first:first + heads], uv[first:first + heads], previous=16384,
                prompt=16512, head_group_limit=heads))
            state = pool._kimi_request_owner_rows[0]["cache"].state
            states.append({name: state[name] for name in ("state_k", "state_v", "counts")})
            assert state["total_len"] == 16512 and state["coverage"] == 16512 - 256
        outputs[heads] = torch.cat(result, dim=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[16], outputs[12], atol=5e-4, rtol=.05)
    for state in states[1:]:
        for name, expected in states[0].items():
            torch.testing.assert_close(state[name], expected, atol=0, rtol=0)

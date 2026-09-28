from __future__ import annotations

import pytest
import torch

from lod_attention._config import LODConfig
from lod_attention._engines import KernelTwoLevelLODAttention
from lod_attention._hf_left_padding import (
    build_padding_plan,
    chunk_align_padding_plan,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_chunk_aligned_left_padding_preserves_each_rows_sink() -> None:
    device = torch.device("cuda")
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        device=device,
    )
    plan = chunk_align_padding_plan(
        build_padding_plan(attention_mask, batch_size=2, sequence_length=4),
        chunk_size=4,
    )
    assert len(plan.groups) == 1
    assert plan.groups[0].valid_starts == (1, 2)

    engine = KernelTwoLevelLODAttention(
        LODConfig(state_clustering_policy="manual"),
        query_heads=2,
        key_value_heads=1,
        scale=0.5,
    )
    engine.chunk_len = 4
    engine.local_len = 4
    engine.prefill_chunk_len = 4
    engine.prefill_local_len = 4
    engine.prefill_state_update_len = 4
    engine.prefill_exact_first_chunk = True
    engine.separate_sink_cache = True
    engine.virtual_page_storage = True

    query = torch.randn(2, 2, 4, 4, dtype=torch.bfloat16, device=device)
    key = torch.arange(2 * 4 * 4, dtype=torch.float32, device=device).reshape(
        2, 1, 4, 4
    )
    key = key.to(torch.bfloat16)
    value = (key + 100).clone()
    output, cache = engine(
        query,
        key,
        value,
        use_cache=True,
        prefill_valid_starts=torch.tensor([1, 2], device=device),
    )

    assert cache is not None
    assert tuple(output.shape) == tuple(query.shape)
    torch.testing.assert_close(cache.state["sink_k"][0], key[0, :, 1:2])
    torch.testing.assert_close(cache.state["sink_k"][1], key[1, :, 2:3])
    torch.testing.assert_close(cache.state["sink_v"][0], value[0, :, 1:2])
    torch.testing.assert_close(cache.state["sink_v"][1], value[1, :, 2:3])

    torch.testing.assert_close(cache.state["state_k"][0, :, :2], key[0, :, 2:])
    torch.testing.assert_close(cache.state["state_v"][0, :, :2], value[0, :, 2:])
    torch.testing.assert_close(cache.state["state_k"][1, :, :1], key[1, :, 3:])
    torch.testing.assert_close(cache.state["state_v"][1, :, :1], value[1, :, 3:])
    assert isinstance(cache.state["page_cache"], dict)
    assert cache.total_length == 4

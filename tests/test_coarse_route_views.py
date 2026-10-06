import pytest
import torch

from lod_attention.kernels._coarse_route_views import coarse_route_mean_view


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("stored,virtual,dim", [(1, 6, 576), (2, 2, 128), (1, 1, 256)])
def test_cached_mean_view_keeps_physical_storage_and_head_aliases(batch, stored, virtual, dim):
    slots, offset = 17, 13
    state = torch.zeros(batch, stored, slots, dim).expand(batch, virtual, slots, dim)
    rows = batch * stored * slots
    # Exactly one extra arena row, not enough for six independently stored
    # virtual heads. Poison the prefix so wrong offsets cannot pass silently.
    arena = torch.full((offset + rows + 1, dim), float("nan"))
    expected = torch.arange(rows * dim).view(batch, stored, slots, dim).float()
    arena[offset:offset+rows].copy_(expected.flatten(0, 2))
    result = coarse_route_mean_view(state, arena, offset, stored)
    torch.testing.assert_close(result, expected.expand_as(state))
    assert result.untyped_storage().data_ptr() == arena.untyped_storage().data_ptr()
    if virtual != stored:
        assert result.stride(1) == 0
    # Rows/channels are shared without a gather or copy, including mutations.
    arena[offset+slots-1, -1] = -123
    assert result[0, 0, slots-1, -1] == -123
    if virtual != stored:
        assert result[0, -1, slots-1, -1] == -123


@pytest.mark.parametrize("fault", ["short", "negative", "dim", "dtype", "layout", "heads", "independent"])
def test_unavailable_cached_mean_view_retains_sum_route(fault):
    state = torch.empty(8, 1, 17, 576).expand(8, 6, 17, 576)
    arena = torch.empty(8 * 17, 576)
    offset, physical = 0, 1
    if fault == "short":
        arena = arena[:-1]
    elif fault == "negative":
        offset = -1
    elif fault == "dim":
        arena = arena[:, :-1]
    elif fault == "dtype":
        arena = arena.double()
    elif fault == "layout":
        arena = torch.empty(576, 8*17).t()
    elif fault == "heads":
        physical = 2
    elif fault == "independent":
        state = state.clone()
    assert coarse_route_mean_view(state, arena, offset, physical) is None

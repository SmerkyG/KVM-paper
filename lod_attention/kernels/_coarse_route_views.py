"""Allocation-free views of the already refreshed centroid-mean cache."""

import torch


def coarse_route_mean_view(state: torch.Tensor, arena: torch.Tensor,
                           offset: int, physical_heads: int | None = None) -> torch.Tensor | None:
    """Alias one physical MLA KV head across virtual 16-query-head tiles.

    Ordinary GQA keeps its existing head layout. MLA's expanded head axis has
    stride zero, not six independently stored copies. Use physical storage to
    bound the view; interpreting its virtual shape as storage reads past the
    coarse region. Unavailable/incompatible cached means retain sum routing.
    """
    if state.ndim != 4 or arena.ndim != 2:
        return None
    batch, heads, slots, dim = state.shape
    stored_heads = heads if physical_heads is None else int(physical_heads)
    if stored_heads != heads and stored_heads != 1:
        return None
    if stored_heads != heads and state.stride(1) != 0:
        return None
    rows = batch * stored_heads * slots
    if (stored_heads <= 0 or offset < 0 or offset + rows > arena.size(0)
            or arena.size(1) != dim or arena.dtype != state.dtype
            or arena.device != state.device or not arena.is_contiguous()):
        return None
    view = arena.narrow(0, offset, rows).view(batch, stored_heads, slots, dim)
    return view.expand_as(state) if stored_heads != heads else view


__all__ = ["coarse_route_mean_view"]

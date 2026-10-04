"""Isolated communication for experimental asynchronous cache construction."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def construction_layer_slice(layer_count: int, world_size: int, rank: int) -> slice:
    """Contiguous equal work shares preserve rank-major gather ordering."""
    if world_size < 1 or layer_count < 1 or layer_count % world_size:
        raise ValueError("construction layer group must divide evenly across ranks")
    if not 0 <= rank < world_size:
        raise ValueError("construction rank is outside its group")
    share = layer_count // world_size
    return slice(rank * share, (rank + 1) * share)


class PrefillConstructionGroup:
    """A separate communicator with the foreground DCP group's membership.

    Construct collectively during worker initialization, never lazily inside
    a background stream. Its gathers run in submission order on the cache
    construction stream, independently of foreground TP/DCP collectives.
    """

    def __init__(self, dcp_group: Any) -> None:
        self.world_size = int(dcp_group.world_size)
        self.process_group = dcp_group.make_sibling_device_group(
            group_desc="lod-prefill-construction"
        )

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if dim != 0:
            raise ValueError("construction gathers concatenate only dimension zero")
        source = tensor.contiguous()
        output = source.new_empty(
            self.world_size * int(source.size(0)), *source.shape[1:]
        )
        work = dist.all_gather_into_tensor(
            output, source, group=self.process_group, async_op=True,
        )
        # NCCL/RCCL wait orders the construction stream after the collective;
        # it does not insert a wait on the foreground transformer stream.
        work.wait()
        return output

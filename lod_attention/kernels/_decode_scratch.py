"""Named, fixed-address temporary buffers for sequential attention layers.

Only intermediates overwritten by every decode call may be shared. Final
outputs, LSEs returned to the caller, epochs/stamps and incremental masks
remain per-layer. A registry belongs to one model runner/execution stream;
concurrent microbatches must use separate registries (or no sharing).
"""

from __future__ import annotations

import torch


TRANSIENT_DECODE_BUFFERS = frozenset({
    "partial_out", "partial_lse",
    "route_candidate_scores", "route_candidate_indices",
    "route_group_out", "route_group_lse", "route_state_scores",
    "route_top_slots", "route_top_scores",
    "distributed_route_packed",
    "coarse_out", "coarse_lse", "route_local_out", "route_local_lse",
    "gqa_union_slots", "gqa_union_destinations", "gqa_union_token_indices",
    "gqa_union_hip_out", "gqa_union_hip_lse",
    "gqa_union_hip_segment_out", "gqa_union_hip_exp_sums",
    "gqa_union_hip_max_logits", "kimi_gluon_partial", "kimi_gluon_partial_lse",
})


def can_share_decode_scratch(parallel_config, speculative_tokens: int) -> bool:
    """Only sequential, non-speculative attention calls may alias scratch."""
    return (speculative_tokens == 0
            and not bool(getattr(parallel_config, "use_ubatching", False))
            and not bool(getattr(parallel_config, "enable_dbo", False))
            and int(getattr(parallel_config, "ubatch_size", 0)) <= 1)


def empty_decode_tensor(name: str, *shape: int, dtype: torch.dtype,
                        device: torch.device,
                        shared_scratch: dict | None = None) -> torch.Tensor:
    """Allocate once per name/geometry; never share persistent/output state."""
    if shared_scratch is None or name not in TRANSIENT_DECODE_BUFFERS:
        return torch.empty(*shape, dtype=dtype, device=device)
    # Different TP/DCP head layouts or capacities receive distinct storage.
    key = (name, tuple(shape), dtype, torch.device(device))
    tensor = shared_scratch.get(key)
    if tensor is None:
        tensor = torch.empty(*shape, dtype=dtype, device=device)
        shared_scratch[key] = tensor
    return tensor

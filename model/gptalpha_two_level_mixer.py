"""Inference-time two-level approximation for pretrained GPTAlpha2.

State slots store sums of the pretrained model's ordinary partial-RoPE keys
and values.  Coarse attention uses arithmetic means with an exact count bias;
the top routed slots are replaced by their original KV leaves.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .kvm_mixer import (
    _all_idx,
    _gather_by_idx,
    _split_append_merge_idx_by_maxsim,
)
from .kvm_two_level_mixer import SequenceMixer as TwoLevelSequenceMixer


class SequenceMixer(TwoLevelSequenceMixer):
    """Mass-corrected top-k LOD attention over ordinary GPTAlpha2 keys."""

    def __init__(self, config, layer_idx: int):
        if config.kvm_use_head_temps:
            raise ValueError("GPTAlpha2 two-level attention has no head temperatures")
        super().__init__(config, layer_idx)
        # The checkpoint's learned ln_k is already applied by calc_qkv before
        # partial RoPE.  There is no additional state-specific normalizer.
        self.ln_s_k = nn.Identity()

    def _prepare_state_update_k(self, k_block: torch.Tensor) -> torch.Tensor:
        # Keep the pretrained GPTAlpha2 key exactly: partial RoPE included.
        return k_block

    @staticmethod
    def _mean_state_key(
        s_k: torch.Tensor, state_counts: torch.Tensor
    ) -> torch.Tensor:
        return s_k / state_counts.to(s_k.dtype).clamp_min(1)

    def _update_state_and_owners(
        self,
        s_k: torch.Tensor,
        s_v: torch.Tensor,
        state_counts: torch.Tensor,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        *,
        ctx_len: int,
        available_context: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        overflow_len = int(overflow_k.size(2))
        if overflow_len == 0:
            empty = torch.empty(
                *overflow_k.shape[:-1], device=overflow_k.device, dtype=torch.long
            )
            return s_k, s_v, state_counts, empty

        state_k = self._prepare_state_update_k(overflow_k)
        current_state_len = int(s_k.size(2))
        desired_state_len = self._desired_state_len(
            ctx_len, available_context, current_state_len
        )
        n_append = min(
            max(desired_state_len - current_state_len, 0), overflow_len
        )
        owners = torch.full(
            state_k.shape[:-1], -1, device=state_k.device, dtype=torch.long
        )

        if n_append:
            append_idx, merge_idx = _split_append_merge_idx_by_maxsim(
                state_k,
                n_append,
                self._mean_state_key(s_k.detach(), state_counts),
            )
            append_k = _gather_by_idx(state_k, append_idx)
            append_v = _gather_by_idx(overflow_v, append_idx)
            append_counts = torch.ones_like(
                append_v[..., :1], dtype=torch.float32
            )
            s_k = torch.cat((s_k, append_k), dim=2)
            s_v = torch.cat((s_v, append_v), dim=2)
            state_counts = torch.cat((state_counts, append_counts), dim=2)
            append_slots = torch.arange(
                current_state_len,
                current_state_len + n_append,
                device=state_k.device,
                dtype=torch.long,
            ).view(1, 1, n_append).expand_as(append_idx)
            owners.scatter_(2, append_idx, append_slots)
            merge_k = _gather_by_idx(state_k, merge_idx)
            merge_v = _gather_by_idx(overflow_v, merge_idx)
        else:
            merge_idx = _all_idx(state_k, overflow_len)
            merge_k = state_k
            merge_v = overflow_v

        if int(merge_k.size(2)) == 0:
            return s_k, s_v, state_counts, owners

        protected_slots = min(self.sink_len, int(s_k.size(2)))
        if int(s_k.size(2)) <= protected_slots:
            raise AssertionError(
                "GPTAlpha2 two-level attention requires a non-sink merge slot"
            )
        with torch.no_grad():
            route_logits = torch.matmul(
                merge_k,
                self._mean_state_key(
                    s_k.detach(), state_counts
                ).transpose(-1, -2),
            )
            route_logits[..., :protected_slots] = float("-inf")
            destination = route_logits.argmax(dim=-1)
            assignment = F.one_hot(
                destination, num_classes=int(s_k.size(2))
            ).to(merge_k.dtype)

        assignment_t = assignment.transpose(-1, -2)
        s_k = s_k + torch.matmul(assignment_t, merge_k)
        s_v = s_v + torch.matmul(assignment_t.to(merge_v.dtype), merge_v)
        state_counts = state_counts + assignment_t.float().sum(
            dim=-1, keepdim=True
        )
        owners.scatter_(2, merge_idx, destination)
        return s_k, s_v, state_counts, owners

    def _route_top_slots(
        self,
        q: torch.Tensor,
        s_k: torch.Tensor,
        state_counts: torch.Tensor,
    ) -> torch.Tensor:
        route_count = min(self.two_level_topk, int(s_k.size(2)))
        with torch.no_grad():
            state_k = self._repeat_kv_for_query_heads(
                self._mean_state_key(s_k.detach(), state_counts)
            )
            logits = torch.matmul(q.detach(), state_k.transpose(-1, -2))
            logits = logits * (1.0 / math.sqrt(float(self.d_qk_head)))
            log_counts = state_counts.detach().clamp_min(1).log()
            log_counts = self._repeat_kv_for_query_heads(
                log_counts
            ).transpose(-1, -2)
            logits = logits + log_counts
            return logits.topk(route_count, dim=-1, sorted=False).indices

    def _coarse_attention(
        self,
        q: torch.Tensor,
        bswa_k: torch.Tensor,
        bswa_v: torch.Tensor,
        s_k: torch.Tensor,
        s_v: torch.Tensor,
        state_counts: torch.Tensor,
        top_slots: torch.Tensor,
        *,
        state_capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super()._coarse_attention(
            q,
            bswa_k,
            bswa_v,
            self._mean_state_key(s_k, state_counts),
            s_v,
            state_counts,
            top_slots,
            state_capacity=state_capacity,
        )

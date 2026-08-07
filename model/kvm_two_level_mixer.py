"""Mass-conserving two-level KVM attention.

The ordinary KVM state treats each compressed slot as one attention entry.  In
this variant a slot represents a partition of exact historical KV leaves.  A
query expands the leaves owned by its top-k slots and replaces those slots with
exact attention; every other slot contributes a count-corrected coarse
summary.  The two branches are combined with their log-sum-exp statistics.

This is intentionally a pure-PyTorch training implementation.  FlexAttention
provides the coarse attention output and a differentiable LSE.  Exact leaves
are gathered through per-slot posting lists, so the implementation never
materializes a query-by-history mask.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.attention.flex_attention import AuxRequest

from utils.flex_attention import separately_compiled_flex_attention

from .kvm_mixer import (
    SequenceMixer as KVMSequenceMixer,
    _all_idx,
    _gather_by_idx,
    _split_append_merge_idx_by_maxsim,
)


def _pad_state(x: torch.Tensor, capacity: int) -> torch.Tensor:
    missing = int(capacity) - int(x.size(2))
    if missing < 0:
        raise ValueError("two-level KVM state exceeded its prefill capacity")
    if missing == 0:
        return x
    return F.pad(x, (0, 0, 0, missing))


def _merge_lse_branches(
    coarse_output: torch.Tensor,
    coarse_lse: torch.Tensor,
    exact_output: torch.Tensor,
    exact_lse: torch.Tensor,
) -> torch.Tensor:
    """Renormalize two independently normalized attention branches."""
    branch_lse = torch.stack((coarse_lse, exact_lse), dim=-1).float()
    weights = torch.softmax(branch_lse, dim=-1).to(coarse_output.dtype)
    return (
        coarse_output * weights[..., 0].unsqueeze(-1)
        + exact_output * weights[..., 1].unsqueeze(-1)
    )


class _PackedAttentionWithLSE(Function):
    """Packed FlashAttention whose LSE remains part of the autograd graph."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_lengths,
        k_lengths,
        max_q,
        max_k,
        scale,
    ):
        out, padded_lse, rng_state, unused, _ = (
            torch.ops.aten._flash_attention_forward(
                q,
                k,
                v,
                cu_q,
                cu_k,
                int(max_q),
                int(max_k),
                0.0,
                False,
                False,
                scale=float(scale),
            )
        )
        expert = torch.repeat_interleave(
            torch.arange(
                int(q_lengths.numel()), device=q.device, dtype=torch.long
            ),
            q_lengths,
        )
        q_offset = torch.arange(int(q.size(0)), device=q.device) - cu_q[:-1].long()[
            expert
        ]
        packed_lse = padded_lse[expert, 0, q_offset]
        ctx.max_q = int(max_q)
        ctx.max_k = int(max_k)
        ctx.scale = float(scale)
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            padded_lse,
            cu_q,
            cu_k,
            rng_state,
            unused,
        )
        return out, packed_lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):  # type: ignore[override]
        (
            q,
            k,
            v,
            out,
            padded_lse,
            cu_q,
            cu_k,
            rng_state,
            unused,
        ) = ctx.saved_tensors
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_out.contiguous(),
            q,
            k,
            v,
            out,
            padded_lse,
            cu_q,
            cu_k,
            ctx.max_q,
            ctx.max_k,
            0.0,
            False,
            rng_state,
            unused,
            scale=ctx.scale,
        )

        # FlashAttention's ordinary backward consumes grad_out but not an LSE
        # gradient.  Avoid expanding every query-key pair for that term:
        #
        #   dLSE_i/dq_i = scale * sum_j p_ij k_j
        #   dLSE_i/dk_j = scale * sum_i p_ij q_i
        #
        # A packed attention pass with v=k gives the first expectation.  Its
        # value gradient, seeded with scale * grad_lse_i * q_i, gives the
        # second expression exactly.
        if grad_lse is not None:
            key_mean, key_lse, key_rng_state, key_unused, _ = (
                torch.ops.aten._flash_attention_forward(
                    q,
                    k,
                    k,
                    cu_q,
                    cu_k,
                    ctx.max_q,
                    ctx.max_k,
                    0.0,
                    False,
                    False,
                    scale=ctx.scale,
                )
            )
            scaled_grad_lse = (
                grad_lse.to(q.dtype).view(-1, 1, 1) * ctx.scale
            )
            key_value_grad = (scaled_grad_lse * q).contiguous()
            _, _, dk_lse = torch.ops.aten._flash_attention_backward(
                key_value_grad,
                q,
                k,
                k,
                key_mean,
                key_lse,
                cu_q,
                cu_k,
                ctx.max_q,
                ctx.max_k,
                0.0,
                False,
                key_rng_state,
                key_unused,
                scale=ctx.scale,
            )
            dq = dq + key_mean * scaled_grad_lse
            dk = dk + dk_lse

        return dq, dk, dv, None, None, None, None, None, None, None


def _expert_leaf_attention(
    q: torch.Tensor,
    exact_k: torch.Tensor,
    exact_v: torch.Tensor,
    owners: torch.Tensor,
    state_counts: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    kv_group_size: int,
    head_temperature: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MoE-style dispatch of routed queries to singly-owned KV leaf sets."""
    batch, query_heads, query_len, head_dim = q.shape
    state_len = int(state_counts.size(2))
    route_count = int(top_slots.size(-1))
    history_len = int(owners.size(2))
    if int(exact_v.size(-1)) != head_dim:
        raise ValueError("packed exact attention requires equal QK and V head sizes")

    with torch.no_grad():
        if bool((owners < 0).any().item()):
            raise AssertionError("two-level KVM found an unowned compressed leaf")
        counts = state_counts.squeeze(-1).round().to(torch.long)
        if not bool((counts.sum(-1) == history_len).all().item()):
            raise AssertionError(
                "two-level KVM leaf counts disagree with ownership"
            )
        starts = counts.cumsum(-1) - counts
        sorted_positions = owners.argsort(dim=-1, stable=False)
        counts = counts.repeat_interleave(kv_group_size, dim=1).flatten(0, 1)
        starts = starts.repeat_interleave(kv_group_size, dim=1).flatten(0, 1)
        sorted_positions = sorted_positions.repeat_interleave(
            kv_group_size, dim=1
        ).flatten(0, 1)

        rows = batch * query_heads * query_len
        query_row = torch.arange(rows, device=q.device, dtype=torch.long)
        query_row = query_row.unsqueeze(-1).expand(rows, route_count).reshape(-1)
        bh_for_row = torch.div(query_row, query_len, rounding_mode="floor")
        route_slot = top_slots.reshape(-1)
        expert_id = bh_for_row * state_len + route_slot
        order = expert_id.argsort(stable=False)
        sorted_expert = expert_id[order]
        unique_expert, q_lengths = torch.unique_consecutive(
            sorted_expert, return_counts=True
        )
        packed_query_row = query_row[order]
        expert_bh = torch.div(unique_expert, state_len, rounding_mode="floor")
        expert_slot = unique_expert % state_len
        k_lengths = counts[expert_bh, expert_slot]
        if bool((k_lengths <= 0).any().item()):
            raise AssertionError("a routed state expert owns no leaves")

        cu_q = F.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int32)
        cu_k = F.pad(k_lengths.cumsum(0), (1, 0)).to(torch.int32)
        max_q = int(q_lengths.max().item())
        max_k = int(k_lengths.max().item())
        expert_for_leaf = torch.repeat_interleave(
            torch.arange(
                int(k_lengths.numel()), device=q.device, dtype=torch.long
            ),
            k_lengths,
        )
        leaf_begin = (k_lengths.cumsum(0) - k_lengths)[expert_for_leaf]
        leaf_offset = torch.arange(int(k_lengths.sum().item()), device=q.device)
        leaf_offset = leaf_offset - leaf_begin
        leaf_bh = expert_bh[expert_for_leaf]
        posting_rank = starts[leaf_bh, expert_slot[expert_for_leaf]] + leaf_offset
        leaf_position = sorted_positions[leaf_bh, posting_rank]
        inverse_order = order.argsort(stable=False)

    q_flat = q.reshape(rows, head_dim)
    packed_q = q_flat.index_select(0, packed_query_row).unsqueeze(1)
    exact_k = exact_k[..., :history_len, :].repeat_interleave(
        kv_group_size, dim=1
    ).flatten(0, 1)
    exact_v = exact_v[..., :history_len, :].repeat_interleave(
        kv_group_size, dim=1
    ).flatten(0, 1)
    packed_k = exact_k[leaf_bh, leaf_position]
    packed_v = exact_v[leaf_bh, leaf_position]
    expert_head = expert_bh % query_heads
    packed_temperature = head_temperature[expert_head[expert_for_leaf]]
    packed_k = (packed_k * packed_temperature.view(-1, 1)).unsqueeze(1)
    packed_v = packed_v.unsqueeze(1)

    packed_out, packed_lse = _PackedAttentionWithLSE.apply(
        packed_q,
        packed_k,
        packed_v,
        cu_q,
        cu_k,
        q_lengths,
        k_lengths,
        max_q,
        max_k,
        scale,
    )
    route_out = packed_out.squeeze(1).index_select(0, inverse_order)
    route_lse = packed_lse.index_select(0, inverse_order)
    route_out = route_out.reshape(rows, route_count, int(packed_v.size(-1)))
    route_lse = route_lse.reshape(rows, route_count)
    route_weight = torch.softmax(route_lse.float(), dim=-1).to(route_out.dtype)
    exact_out = (route_out * route_weight.unsqueeze(-1)).sum(dim=1)
    exact_lse = torch.logsumexp(route_lse.float(), dim=-1)
    return (
        exact_out.reshape(batch, query_heads, query_len, -1),
        exact_lse.reshape(batch, query_heads, query_len),
    )


class SequenceMixer(KVMSequenceMixer):
    """Count-corrected KVM state plus exact leaves from top routed slots."""

    _CACHE_OWNERS = "kvm_two_level_owners"
    _CACHE_EXACT_K = "kvm_two_level_exact_k"
    _CACHE_EXACT_V = "kvm_two_level_exact_v"

    def __init__(self, config, layer_idx: int):
        if config.kvm_use_merge_gate_keys or config.kvm_use_merge_gate_values:
            raise ValueError(
                "kvm_two_level_mixer requires merge gates to be disabled"
            )
        if config.kvm_use_vlens:
            raise ValueError(
                "kvm_two_level_mixer uses s_vlen as exact counts; set "
                "kvm_use_vlens=0"
            )
        super().__init__(config, layer_idx)
        self.two_level_topk = 8

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
        """Ungated state update returning one owner slot for every leaf."""
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
                state_k, n_append, self.ln_s_k(s_k.detach())
            )
            append_k = _gather_by_idx(state_k, append_idx)
            append_v = _gather_by_idx(overflow_v, append_idx)
            append_counts = torch.ones_like(append_v[..., :1], dtype=torch.float32)
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
                "two-level KVM requires a non-sink state slot for merging"
            )
        with torch.no_grad():
            route_logits = torch.matmul(
                merge_k, self.ln_s_k(s_k.detach()).transpose(-1, -2)
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
            state_k = self._repeat_kv_for_query_heads(self.ln_s_k(s_k.detach()))
            logits = torch.matmul(q.detach(), state_k.transpose(-1, -2))
            logits = logits * (1.0 / math.sqrt(float(self.d_qk_head)))
            if self.config.kvm_use_head_temps:
                logits = logits * self.state_head_temp.detach().view(1, -1, 1, 1)
            log_counts = state_counts.detach().clamp_min(1).log()
            log_counts = self._repeat_kv_for_query_heads(log_counts).transpose(-1, -2)
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
        padded_k = _pad_state(self.ln_s_k(s_k), state_capacity)
        state_mean_v = s_v / state_counts.to(s_v.dtype).clamp_min(1)
        padded_v = _pad_state(state_mean_v, state_capacity)
        padded_k = self._repeat_kv_for_query_heads(padded_k)
        padded_v = self._repeat_kv_for_query_heads(padded_v)
        bswa_k = self._repeat_kv_for_query_heads(bswa_k)
        bswa_v = self._repeat_kv_for_query_heads(bswa_v)
        if self.config.kvm_use_head_temps:
            state_temperature = self.state_head_temp.view(1, -1, 1, 1)
            front_temperature = self.front_head_temp.view(1, -1, 1, 1)
            padded_k = padded_k * state_temperature
            bswa_k = bswa_k * front_temperature
        coarse_k = torch.cat((padded_k, bswa_k), dim=2)
        coarse_v = torch.cat((padded_v, bswa_v), dim=2)

        padded_counts = _pad_state(state_counts, state_capacity)
        state_log_counts = torch.where(
            padded_counts > 0,
            padded_counts.log(),
            torch.full_like(padded_counts, float("-inf")),
        ).squeeze(-1)
        state_log_counts = self._repeat_kv_for_query_heads(
            state_log_counts.unsqueeze(-1)
        ).squeeze(-1)
        query_len = int(q.size(2))
        local_len = int(bswa_k.size(2))
        local_offset = local_len - query_len
        state_bias = state_log_counts.unsqueeze(2).expand(
            -1, -1, query_len, -1
        ).clone()
        state_bias.scatter_(-1, top_slots, float("-inf"))
        query_index = torch.arange(query_len, device=q.device).unsqueeze(-1)
        local_index = torch.arange(local_len, device=q.device).unsqueeze(0)
        local_visible = local_index <= query_index + local_offset
        local_bias = torch.zeros(
            query_len, local_len, device=q.device, dtype=state_bias.dtype
        ).masked_fill(~local_visible, float("-inf"))
        local_bias = local_bias.view(1, 1, query_len, local_len).expand(
            int(q.size(0)), int(q.size(1)), query_len, local_len
        )
        attention_bias = torch.cat((state_bias, local_bias), dim=-1)

        # Capturing one additive-bias tensor also avoids a PyTorch 2.9 ROCm
        # FlexAttention tracing bug triggered by score_mod closures with
        # multiple tensor captures.
        def score_mod(score, batch, head, q_idx, kv_idx):
            return score + attention_bias[batch, head, q_idx, kv_idx]

        output, aux = separately_compiled_flex_attention(
            q,
            coarse_k,
            coarse_v,
            score_mod=score_mod,
            enable_gqa=False,
            scale=1.0 / math.sqrt(float(self.d_qk_head)),
            return_aux=AuxRequest(lse=True),
        )
        if aux.lse is None:
            raise RuntimeError("FlexAttention did not return coarse branch LSE")
        return output, aux.lse

    def _two_level_attention(
        self,
        q: torch.Tensor,
        bswa_k: torch.Tensor,
        bswa_v: torch.Tensor,
        s_k: torch.Tensor,
        s_v: torch.Tensor,
        state_counts: torch.Tensor,
        owners: torch.Tensor,
        exact_k: torch.Tensor,
        exact_v: torch.Tensor,
        *,
        state_capacity: int,
    ) -> torch.Tensor:
        top_slots = self._route_top_slots(q, s_k, state_counts)
        if self.config.kvm_use_head_temps:
            state_temperature = self.state_head_temp
        else:
            state_temperature = q.new_ones(self.num_attention_heads)
        exact_output, exact_lse = _expert_leaf_attention(
            q,
            exact_k,
            exact_v,
            owners,
            state_counts,
            top_slots,
            kv_group_size=self.kv_group_size,
            head_temperature=state_temperature,
            scale=1.0 / math.sqrt(float(self.d_qk_head)),
        )
        coarse_output, coarse_lse = self._coarse_attention(
            q,
            bswa_k,
            bswa_v,
            s_k,
            s_v,
            state_counts,
            top_slots,
            state_capacity=state_capacity,
        )
        return _merge_lse_branches(
            coarse_output, coarse_lse, exact_output, exact_lse
        )

    @torch.compiler.disable
    def forward_prefill(
        self,
        q,
        k,
        v,
        merge_gate,
        v_first,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        del merge_gate, v_first, position_embeddings, attention_mask, kwargs
        batch_size, _, prefill_len, _ = q.size()
        bswa_len = self.n_bswa_chunks * self.chunk_len
        front_bswa_len = min(prefill_len, bswa_len)
        if self.config.kvm_use_head_temps:
            front_temperature = self.front_head_temp.view(
                1, self.num_attention_heads, 1, 1
            )
        else:
            front_temperature = 1.0
        outputs = [
            self._sdpa_with_repeated_kv(
                q[..., :front_bswa_len, :],
                k[..., :front_bswa_len, :],
                v[..., :front_bswa_len, :],
                key_temperature=front_temperature,
                is_causal=True,
            )
        ]

        initial_len = min(prefill_len, self.chunk_len)
        s_k = self._prepare_state_update_k(k[..., :initial_len, :])
        s_v = v[..., :initial_len, :]
        state_counts = torch.ones_like(s_v[..., :1], dtype=torch.float32)
        owners = torch.arange(
            initial_len, device=k.device, dtype=torch.long
        ).view(1, 1, initial_len).expand(
            batch_size, self.num_key_value_heads, initial_len
        )
        state_coverage_len = initial_len
        state_capacity = self._desired_state_len(
            prefill_len, prefill_len, initial_len
        )

        # Exact keys use precisely the same state-style transformation as an
        # individual KVM slot, and remain differentiable during training.
        exact_k = self.ln_s_k(self._prepare_state_update_k(k))
        exact_v = v

        for query_begin in range(front_bswa_len, prefill_len, self.chunk_len):
            query_end = min(prefill_len, query_begin + self.chunk_len)
            bswa_begin = self._bswa_begin_for_total_len(query_end)
            if bswa_begin != state_coverage_len:
                raise AssertionError(
                    "two-level KVM state coverage drifted from its leaf archive"
                )
            outputs.append(
                self._two_level_attention(
                    q[..., query_begin:query_end, :],
                    k[..., bswa_begin:query_end, :],
                    v[..., bswa_begin:query_end, :],
                    s_k,
                    s_v,
                    state_counts,
                    owners,
                    exact_k,
                    exact_v,
                    state_capacity=state_capacity,
                )
            )

            if self.training and query_end >= prefill_len:
                break
            next_bswa_begin = self._bswa_begin_for_total_len(
                min(prefill_len, query_end + self.chunk_len)
            )
            if next_bswa_begin > bswa_begin:
                s_k, s_v, state_counts, new_owners = self._update_state_and_owners(
                    s_k,
                    s_v,
                    state_counts,
                    k[..., bswa_begin:next_bswa_begin, :],
                    v[..., bswa_begin:next_bswa_begin, :],
                    ctx_len=query_end,
                    available_context=next_bswa_begin,
                )
                owners = torch.cat((owners, new_owners), dim=2)
                state_coverage_len = next_bswa_begin

        y = torch.cat(outputs, dim=-2)
        y = y.transpose(1, 2).contiguous().view(batch_size, prefill_len, -1)
        y = self.c_proj(y)

        if past_key_values is not None:
            bswa_begin = self._bswa_begin_for_total_len(prefill_len)
            expected_coverage = max(initial_len, bswa_begin)
            if state_coverage_len != expected_coverage:
                raise AssertionError(
                    "two-level KVM prefill cache coverage drifted"
                )
            past_key_values.update(
                self.layer_idx,
                offset=prefill_len,
                states_dict={
                    self._CACHE_S_K: s_k.detach(),
                    self._CACHE_S_V: s_v.detach(),
                    self._CACHE_S_VLEN: state_counts.detach(),
                    self._CACHE_STATE_COVERAGE_LEN: state_coverage_len,
                    self._CACHE_BSWA_BEGIN: bswa_begin,
                    self._CACHE_BSWA_K: k[..., bswa_begin:, :].detach(),
                    self._CACHE_BSWA_V: v[..., bswa_begin:, :].detach(),
                    self._CACHE_OWNERS: owners.detach(),
                    self._CACHE_EXACT_K: exact_k.detach(),
                    self._CACHE_EXACT_V: exact_v.detach(),
                },
            )
        return y

    @torch.compiler.disable
    def forward_single(
        self,
        q,
        k,
        v,
        merge_gate,
        v_first,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        del merge_gate, v_first, position_embeddings, attention_mask, kwargs
        if past_key_values is None:
            raise ValueError("two-level KVM cached decode requires a cache")
        if int(q.size(2)) != 1:
            raise AssertionError("two-level KVM cached decode expects one token")

        cache = past_key_values.get_states(self.layer_idx)
        past_len = past_key_values.get_seq_length(self.layer_idx)
        s_k = cache[self._CACHE_S_K]
        s_v = cache[self._CACHE_S_V]
        state_counts = cache[self._CACHE_S_VLEN]
        owners = cache[self._CACHE_OWNERS]
        old_bswa_k = cache[self._CACHE_BSWA_K]
        old_bswa_v = cache[self._CACHE_BSWA_V]
        old_bswa_begin = int(cache[self._CACHE_BSWA_BEGIN])
        state_coverage = int(cache[self._CACHE_STATE_COVERAGE_LEN])
        old_exact_k = cache[self._CACHE_EXACT_K]
        old_exact_v = cache[self._CACHE_EXACT_V]

        full_bswa_k = torch.cat((old_bswa_k, k), dim=2)
        full_bswa_v = torch.cat((old_bswa_v, v), dim=2)
        current_exact_k = self.ln_s_k(self._prepare_state_update_k(k))
        full_exact_k = torch.cat((old_exact_k, current_exact_k), dim=2)
        full_exact_v = torch.cat((old_exact_v, v), dim=2)

        total_len = past_len + 1
        direct_target = min(total_len, self.chunk_len)
        new_bswa_begin = self._bswa_begin_for_total_len(total_len)
        target_coverage = max(direct_target, new_bswa_begin)

        if state_coverage < direct_target:
            relative_begin = max(state_coverage - old_bswa_begin, 0)
            relative_end = max(direct_target - old_bswa_begin, 0)
            direct_k = self._prepare_state_update_k(
                full_bswa_k[..., relative_begin:relative_end, :]
            )
            direct_v = full_bswa_v[..., relative_begin:relative_end, :]
            direct_len = int(direct_k.size(2))
            first_slot = int(s_k.size(2))
            s_k = torch.cat((s_k, direct_k), dim=2)
            s_v = torch.cat((s_v, direct_v), dim=2)
            state_counts = torch.cat(
                (
                    state_counts,
                    torch.ones_like(direct_v[..., :1], dtype=torch.float32),
                ),
                dim=2,
            )
            direct_owners = torch.arange(
                first_slot,
                first_slot + direct_len,
                device=q.device,
                dtype=torch.long,
            ).view(1, 1, direct_len).expand(
                int(q.size(0)), self.num_key_value_heads, direct_len
            )
            owners = torch.cat((owners, direct_owners), dim=2)
            state_coverage = direct_target

        if state_coverage < target_coverage:
            relative_begin = max(state_coverage - old_bswa_begin, 0)
            relative_end = max(target_coverage - old_bswa_begin, 0)
            overflow_k = full_bswa_k[..., relative_begin:relative_end, :]
            overflow_v = full_bswa_v[..., relative_begin:relative_end, :]
            if int(overflow_k.size(2)) != self.chunk_len:
                raise AssertionError(
                    "two-level KVM decode can overflow only one complete chunk"
                )
            s_k, s_v, state_counts, new_owners = self._update_state_and_owners(
                s_k,
                s_v,
                state_counts,
                overflow_k,
                overflow_v,
                ctx_len=(self.n_bswa_chunks * self.chunk_len) + state_coverage,
                available_context=state_coverage + self.chunk_len,
            )
            owners = torch.cat((owners, new_owners), dim=2)
            state_coverage = target_coverage
        if state_coverage != target_coverage:
            raise AssertionError("two-level KVM decode state coverage drifted")

        relative_begin = max(new_bswa_begin - old_bswa_begin, 0)
        relative_end = max(total_len - old_bswa_begin, 0)
        new_bswa_k = full_bswa_k[..., relative_begin:relative_end, :]
        new_bswa_v = full_bswa_v[..., relative_begin:relative_end, :]
        if new_bswa_begin == 0:
            if self.config.kvm_use_head_temps:
                front_temperature = self.front_head_temp.view(
                    1, self.num_attention_heads, 1, 1
                )
            else:
                front_temperature = 1.0
            out = self._sdpa_with_repeated_kv(
                q,
                new_bswa_k,
                new_bswa_v,
                key_temperature=front_temperature,
                is_causal=False,
            )
        else:
            out = self._two_level_attention(
                q,
                new_bswa_k,
                new_bswa_v,
                s_k,
                s_v,
                state_counts,
                owners,
                full_exact_k,
                full_exact_v,
                state_capacity=int(s_k.size(2)),
            )

        output = out.transpose(1, 2).contiguous().view(int(q.size(0)), 1, -1)
        output = self.c_proj(output)
        past_key_values.update(
            self.layer_idx,
            offset=1,
            states_dict={
                self._CACHE_S_K: s_k.detach(),
                self._CACHE_S_V: s_v.detach(),
                self._CACHE_S_VLEN: state_counts.detach(),
                self._CACHE_STATE_COVERAGE_LEN: state_coverage,
                self._CACHE_BSWA_BEGIN: new_bswa_begin,
                self._CACHE_BSWA_K: new_bswa_k.detach(),
                self._CACHE_BSWA_V: new_bswa_v.detach(),
                self._CACHE_OWNERS: owners.detach(),
                self._CACHE_EXACT_K: full_exact_k.detach(),
                self._CACHE_EXACT_V: full_exact_v.detach(),
            },
        )
        return output

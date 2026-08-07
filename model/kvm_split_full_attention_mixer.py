"""Full-attention teacher for mass-conserving two-level KVM.

The causal history is partitioned exactly as KVM prefill partitions it.  Keys
inside the chunk-aligned BSWA field retain RoPE, while older keys use the same
RoPE-zeroed, state-normalized representation as singleton KVM leaves.  Both
fields are exact and are combined with their log-sum-exp statistics.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask

from utils.flex_attention import separately_compiled_flex_attention
from utils.opt import set_label

from .rwkv7_backbone import StatesDictCache, apply_rotary_embeddings
from .value_residual_mixin import ValueResidualMixin


def _merge_attention_branches(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    remote_output: torch.Tensor,
    remote_lse: torch.Tensor,
) -> torch.Tensor:
    branch_lse = torch.stack((local_lse, remote_lse), dim=-1).float()
    weights = torch.softmax(branch_lse, dim=-1).to(local_output.dtype)
    return (
        local_output * weights[..., 0].unsqueeze(-1)
        + remote_output * weights[..., 1].unsqueeze(-1)
    )


class SequenceMixer(ValueResidualMixin):
    """Exact split attention whose remote field can later be replaced by KVM."""

    _CACHE_LOCAL_K = "kvm_split_full_local_k"
    _CACHE_REMOTE_K = "kvm_split_full_remote_k"
    _CACHE_V = "kvm_split_full_v"

    def __init__(self, config, layer_idx: int):
        super().__init__(
            config,
            layer_idx,
            config.kvm_value_residual_mode,
            config.kvm_token_shift_mode,
            qk_norm=True,
        )
        self.rope_partial_dim = (
            config.rope_partial_dim
            if config.rope_partial_dim > 0
            else self.d_qk_head
        )
        self.chunk_len = int(config.chunk_len)
        self.bswa_len = int(config.n_bswa_chunks) * self.chunk_len
        if self.bswa_len < self.chunk_len:
            raise ValueError("split full attention requires at least one BSWA chunk")

        self.ln_s_k = set_label("scalars", nn.LayerNorm(self.d_qk_head))
        if config.kvm_use_head_temps:
            self.front_head_temp = set_label(
                "scalars", nn.Parameter(torch.ones(config.num_attention_heads))
            )
            self.state_head_temp = set_label(
                "scalars", nn.Parameter(torch.ones(config.num_attention_heads))
            )
        self._split_mask_cache: dict[tuple[str, int], tuple[object, object]] = {}

    def _state_leaf_k(self, k: torch.Tensor) -> torch.Tensor:
        zeroed = torch.cat(
            (
                torch.zeros_like(k[..., : self.rope_partial_dim]),
                k[..., self.rope_partial_dim :],
            ),
            dim=-1,
        )
        # This matches ln_s_k(_prepare_state_update_k(k)) in the two-level
        # mixer: a singleton leaf receives LayerNorm before entering state and
        # again when used as an attention key.
        return self.ln_s_k(self.ln_s_k(zeroed))

    def _bswa_begin_for_total_len(self, total_len: int) -> int:
        chunk_end = math.ceil(total_len / self.chunk_len) * self.chunk_len
        return max(chunk_end - self.bswa_len, 0)

    def _split_masks(self, seq_len: int, device: torch.device):
        # Round the internal attention shape so variable-length evaluation
        # prompts reuse one compiled FlexAttention graph per chunk boundary.
        # Extra keys are strictly after every real query and remain masked.
        seq_len = math.ceil(seq_len / self.chunk_len) * self.chunk_len
        device_key = (
            device.type if device.index is None else f"{device.type}:{device.index}"
        )
        cache_key = (device_key, seq_len)
        cached = self._split_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        chunk_len = self.chunk_len
        bswa_len = self.bswa_len

        def local_mask(batch, head, q_idx, kv_idx):
            del batch, head
            chunk_end = ((q_idx // chunk_len) + 1) * chunk_len
            begin = torch.where(
                q_idx < bswa_len,
                q_idx.new_zeros(()),
                chunk_end - bswa_len,
            )
            return (kv_idx >= begin) & (kv_idx <= q_idx)

        local = create_block_mask(
            local_mask,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            BLOCK_SIZE=128,
        )

        remote_query_len = max(seq_len - bswa_len, 0)
        remote_key_len = min(
            seq_len,
            math.ceil(remote_query_len / chunk_len) * chunk_len,
        )

        def remote_mask(batch, head, q_idx, kv_idx):
            del batch, head
            remote_chunk_end = ((q_idx // chunk_len) + 1) * chunk_len
            return kv_idx < remote_chunk_end

        remote = None
        if remote_query_len:
            remote = create_block_mask(
                remote_mask,
                B=None,
                H=None,
                Q_LEN=remote_query_len,
                KV_LEN=remote_key_len,
                device=device,
                BLOCK_SIZE=128,
            )
        self._split_mask_cache[cache_key] = (local, remote)
        return local, remote

    def get_first_layer_kwargs(self, x0, x, input_ids=None, **kwargs):
        del input_ids
        result = super().get_first_layer_kwargs(x0=x0, x=x, **kwargs)
        if int(x.size(1)) > 1:
            local, remote = self._split_masks(int(x.size(1)), x.device)
            result["split_local_block_mask"] = local
            result["split_remote_block_mask"] = remote
        return result

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.kv_group_size == 1:
            return x
        return x.repeat_interleave(self.kv_group_size, dim=1)

    def _split_prefill_attention(
        self,
        q: torch.Tensor,
        local_k: torch.Tensor,
        remote_k: torch.Tensor,
        v: torch.Tensor,
        local_block_mask,
        remote_block_mask,
    ) -> torch.Tensor:
        seq_len = int(q.size(2))
        attention_seq_len = math.ceil(seq_len / self.chunk_len) * self.chunk_len
        pad_len = attention_seq_len - seq_len
        if pad_len:
            q_for_attention = F.pad(q, (0, 0, 0, pad_len))
            local_k_for_attention = F.pad(local_k, (0, 0, 0, pad_len))
            v_for_local_attention = F.pad(v, (0, 0, 0, pad_len))
        else:
            q_for_attention = q
            local_k_for_attention = local_k
            v_for_local_attention = v
        q_for_attention = q_for_attention.contiguous()
        local_k_for_attention = local_k_for_attention.contiguous()
        v_for_local_attention = v_for_local_attention.contiguous()

        scale = 1.0 / math.sqrt(float(self.d_qk_head))
        local_output, local_aux = separately_compiled_flex_attention(
            q_for_attention,
            local_k_for_attention,
            v_for_local_attention,
            block_mask=local_block_mask,
            scale=scale,
            return_aux=AuxRequest(lse=True),
        )
        if local_aux.lse is None:
            raise RuntimeError("local FlexAttention did not return LSE")

        remote_query_len = attention_seq_len - self.bswa_len
        if remote_query_len <= 0:
            return local_output[..., :seq_len, :]
        remote_key_len = min(
            seq_len,
            math.ceil(remote_query_len / self.chunk_len) * self.chunk_len,
        )
        remote_output, remote_aux = separately_compiled_flex_attention(
            q_for_attention[..., self.bswa_len :, :],
            remote_k[..., :remote_key_len, :],
            v[..., :remote_key_len, :],
            block_mask=remote_block_mask,
            scale=scale,
            return_aux=AuxRequest(lse=True),
        )
        if remote_aux.lse is None:
            raise RuntimeError("remote FlexAttention did not return LSE")
        merged_tail = _merge_attention_branches(
            local_output[..., self.bswa_len :, :],
            local_aux.lse[..., self.bswa_len :],
            remote_output,
            remote_aux.lse,
        )
        return torch.cat(
            (local_output[..., : self.bswa_len, :], merged_tail), dim=2
        )[..., :seq_len, :]

    def forward(
        self,
        x,
        v_first,
        position_embeddings,
        attention_mask=None,
        past_key_values: StatesDictCache | None = None,
        split_local_block_mask=None,
        split_remote_block_mask=None,
        **kwargs,
    ):
        del attention_mask, kwargs
        batch_size, seq_len, _ = x.shape
        q, k, v = self.calc_qkv(x, v_first, past_key_values)
        q = apply_rotary_embeddings(q, position_embeddings).transpose(1, 2)
        local_k = apply_rotary_embeddings(k, position_embeddings).transpose(1, 2)
        remote_k = self._state_leaf_k(
            apply_rotary_embeddings(k, position_embeddings)
        ).transpose(1, 2)
        v = v.transpose(1, 2)
        local_k = self._repeat_kv(local_k)
        remote_k = self._repeat_kv(remote_k)
        v = self._repeat_kv(v)

        if self.config.kvm_use_head_temps:
            local_k = local_k * self.front_head_temp.view(1, -1, 1, 1)
            remote_k = remote_k * self.state_head_temp.view(1, -1, 1, 1)

        prior_len = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else 0
        )
        if prior_len:
            if seq_len != 1:
                raise AssertionError("cached split attention expects one token")
            cache = past_key_values.get_states(self.layer_idx)
            all_local_k = torch.cat((cache[self._CACHE_LOCAL_K], local_k), dim=2)
            all_remote_k = torch.cat((cache[self._CACHE_REMOTE_K], remote_k), dim=2)
            all_v = torch.cat((cache[self._CACHE_V], v), dim=2)
            total_len = int(all_v.size(2))
            local_begin = self._bswa_begin_for_total_len(total_len)
            combined_k = torch.cat(
                (
                    all_remote_k[..., :local_begin, :],
                    all_local_k[..., local_begin:, :],
                ),
                dim=2,
            )
            combined_v = torch.cat(
                (all_v[..., :local_begin, :], all_v[..., local_begin:, :]), dim=2
            )
            output = F.scaled_dot_product_attention(q, combined_k, combined_v)
        else:
            if split_local_block_mask is None:
                split_local_block_mask, split_remote_block_mask = self._split_masks(
                    seq_len, x.device
                )
            output = self._split_prefill_attention(
                q,
                local_k,
                remote_k,
                v,
                split_local_block_mask,
                split_remote_block_mask,
            )
            all_local_k, all_remote_k, all_v = local_k, remote_k, v

        if past_key_values is not None:
            past_key_values.update(
                self.layer_idx,
                offset=seq_len,
                states_dict={
                    self._CACHE_LOCAL_K: all_local_k.detach(),
                    self._CACHE_REMOTE_K: all_remote_k.detach(),
                    self._CACHE_V: all_v.detach(),
                },
            )

        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        return self.c_proj(output)

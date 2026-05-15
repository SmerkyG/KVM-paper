import math
from numpy import s_
from typing import Any
import torch, torch.nn.functional as F
from torch import cat, scatter, zeros_like
from torch import nn, Tensor
from torch.nn.functional import normalize, elu, pad, scaled_dot_product_attention

try:
    import helion  # pyright: ignore[reportMissingImports]
    import helion.language as hl  # pyright: ignore[reportMissingImports]
    import logging

    logging.getLogger().setLevel(logging.WARNING)
except Exception:
    helion = None
    hl = None

from utils.grad_cp import separately_compiled_flex_attention
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    AuxRequest,
    AuxOutput,
    BlockMask,
)
from model.kvm_decode_attn import kvm_generate_state_banks, calc_initial_state

_HELION_AVAILABLE = helion is not None and hl is not None


def _flex_attn(*args, **kwargs):
    return separately_compiled_flex_attention(*args, **kwargs)


def _causal_mask_after_state(chunk_len: int, bswa_len: int, device):
    q_len = chunk_len
    kv_len = chunk_len + bswa_len
    q_offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device)[:, None]
    k_idx = torch.arange(kv_len, device=device)[None, :]
    return k_idx <= q_idx + q_offset


def _same_block_mask_mod(chunk_len: int, sink_len: int, q_len: int, kv_len: int):
    def mask_mod(b, h, q_idx, kv_idx):
        return kv_idx // chunk_len == q_idx // chunk_len

    return mask_mod


def _bswa_sink_mask_mod(
    n_bswa_chunks: int, chunk_len: int, sink_len: int, q_len: int, kv_len: int
):
    q_offset = kv_len - q_len

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = kv_idx <= q_idx
        bswa_mask = (
            kv_idx // chunk_len >= (q_idx + q_offset) // chunk_len - n_bswa_chunks + 1
        )
        sink_mask = kv_idx < sink_len
        return (sink_mask | bswa_mask) & causal_mask

    return mask_mod


def _block_mask(mod_fn, q_len: int, kv_len: int, device: torch.device):
    return create_block_mask(
        mod_fn(q_len, kv_len),
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        BLOCK_SIZE=128,
    )


def _normalize_lse(lse: Tensor) -> Tensor:
    if lse.dim() == 4:
        return lse.squeeze(-1)
    return lse


def _require_helion() -> None:
    if not _HELION_AVAILABLE:
        raise RuntimeError(
            "Helion is not available. Install `helion` in a supported PyTorch 2.9+ Linux/CUDA environment "
            "before calling the Helion KVM attention path."
        )


if _HELION_AVAILABLE:

    @helion.kernel(static_shapes=False, autotune_effort="quick")
    def _kvm_state_attention_with_updates_helion_kernel_simple(
        q_tail: torch.Tensor,
        init_k: torch.Tensor,
        init_v: torch.Tensor,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        s_vlen: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        state_temperature: torch.Tensor,
        state_positions: torch.Tensor,
        sink_len: int,
        ln_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_rows = q_tail.size(0)
        tail_len = q_tail.size(1)
        chunk_len = hl.specialize(init_k.size(1))
        head_dim = hl.specialize(q_tail.size(2))
        value_dim = hl.specialize(init_v.size(2))

        assert tail_len == overflow_k.size(1) == overflow_v.size(1)
        assert chunk_len == s_vlen.size(1) == state_positions.size(0)
        assert head_dim == init_k.size(2) == overflow_k.size(2)
        assert value_dim == init_v.size(2) == overflow_v.size(2)

        out = torch.empty(
            [batch_rows, tail_len, value_dim], dtype=init_v.dtype, device=init_v.device
        )
        lse = torch.empty(
            [batch_rows, tail_len], dtype=torch.float32, device=q_tail.device
        )
        final_s_k = torch.empty_like(init_k)
        final_s_v = torch.empty_like(init_v)
        scale = 1.0 / math.sqrt(float(head_dim))

        for tile_b in hl.tile(batch_rows):
            s_k = init_k[tile_b, :, :]
            s_v = init_v[tile_b, :, :]
            tile_state_temperature = state_temperature[tile_b, :, :]
            tile_s_vlen = s_vlen[tile_b, :, :]
            tile_ln_weight = ln_weight[:].float()
            tile_ln_bias = ln_bias[:].float()
            tile_state_positions = state_positions[:]
            tile_sink_mask = tile_state_positions >= sink_len
            tile_state_temperature_dtype = tile_state_temperature.to(s_k.dtype)
            tile_s_vlen_dtype = tile_s_vlen.to(s_v.dtype)

            for tile_chunk in hl.tile(tail_len, block_size=chunk_len):
                a_q = q_tail[tile_b, tile_chunk, :].float()

                s_k_float = s_k.float()
                s_k_mean = torch.mean(s_k_float, dim=-1, keepdim=True)
                s_k_centered = s_k_float - s_k_mean
                s_k_var = torch.mean(s_k_centered * s_k_centered, dim=-1, keepdim=True)
                s_k_norm = s_k_centered * torch.rsqrt(s_k_var + ln_eps)
                s_k_norm = s_k_norm * tile_ln_weight + tile_ln_bias
                s_k_norm_dtype = s_k_norm.to(s_k.dtype)
                s_k_attn = (s_k_norm_dtype * tile_state_temperature_dtype).float()

                s_v_float = s_v.float()
                s_v_norm = torch.sqrt(
                    torch.sum(s_v_float * s_v_float, dim=-1, keepdim=True)
                )
                s_v_attn = (s_v_float / s_v_norm.clamp_min(1e-12)).to(s_v.dtype)
                s_v_attn = (s_v_attn * tile_s_vlen_dtype).float()

                scores = torch.bmm(a_q, s_k_attn.transpose(1, 2)) * scale
                row_max = torch.amax(scores, dim=-1)
                probs = torch.exp(scores - row_max[:, :, None])
                row_sum = torch.sum(probs, dim=-1)
                out_block = torch.bmm(probs, s_v_attn) / row_sum[:, :, None]
                lse_block = row_max + torch.log(row_sum)

                out[tile_b, tile_chunk, :] = out_block.to(out.dtype)
                lse[tile_b, tile_chunk] = lse_block

                o_k = overflow_k[tile_b, tile_chunk, :]
                sim = torch.bmm(o_k, s_k_norm_dtype.transpose(1, 2))
                sim = torch.where(tile_sink_mask[None, None, :], sim, float("-inf"))
                best_s_idx = torch.argmax(sim, dim=-1, keepdim=True)
                sim_max = (tile_state_positions[None, None, :] == best_s_idx).to(
                    o_k.dtype
                )

                s_k = s_k + torch.bmm(sim_max.transpose(1, 2), o_k)

                o_v = overflow_v[tile_b, tile_chunk, :]
                s_v = s_v + torch.bmm(sim_max.transpose(1, 2), o_v)

            final_s_k[tile_b, :, :] = s_k
            final_s_v[tile_b, :, :] = s_v

        return final_s_k, final_s_v, out, lse

    # @helion.kernel(static_shapes=False, autotune_effort="quick")
    # @helion.kernel(config=helion.Config(block_sizes=[1], indexing=['pointer', 'pointer', 'block_ptr', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], load_eviction_policies=['', '', '', '', '', '', '', '', '', ''], matrix_instr_nonkdim=32, num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, False], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[], waves_per_eu=1), static_shapes=False)
    # @helion.kernel(config=helion.Config(block_sizes=[256, 16, 256, 16, 128, 16, 128, 16], indexing=['pointer', 'pointer', 'pointer', 'block_ptr', 'pointer', 'block_ptr', 'pointer', 'pointer', 'block_ptr', 'pointer', 'pointer', 'block_ptr', 'pointer', 'block_ptr', 'pointer', 'pointer', 'block_ptr', 'pointer', 'block_ptr', 'pointer', 'pointer', 'pointer', 'block_ptr', 'pointer', 'pointer'], load_eviction_policies=['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], matrix_instr_nonkdim=16, num_sm_multiplier=1, num_stages=2, num_warps=8, pid_type='persistent_interleaved', range_flattens=[None, False, False, None, None, None, None, None], range_multi_buffers=[False, None, None, None, None, None, None, True], range_num_stages=[0, 0, 0, 0, 1, 0, 2, 0], range_unroll_factors=[0, 0, 0, 0, 1, 0, 0, 0], range_warp_specializes=[], waves_per_eu=2), static_shapes=False)
    @helion.kernel(
        config=helion.Config(
            block_sizes=[256, 16, 256, 16, 256, 32, 256, 32],
            indexing=[
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
            ],
            load_eviction_policies=[
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            matrix_instr_nonkdim=16,
            num_sm_multiplier=1,
            num_stages=2,
            num_warps=8,
            pid_type="flat",
            range_flattens=[
                None,
                False,
                False,
                None,
                None,
                None,
                True,
                None,
                None,
                None,
            ],
            range_multi_buffers=[
                None,
                None,
                None,
                None,
                False,
                None,
                None,
                None,
                False,
                None,
            ],
            range_num_stages=[0, 0, 0, 0, 0, 0, 2, 0, 2, 0],
            range_unroll_factors=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            range_warp_specializes=[],
            waves_per_eu=2,
        ),
        static_shapes=False,
    )
    def _kvm_state_attention_with_updates_helion_kernel(
        q_tail: torch.Tensor,
        init_k: torch.Tensor,
        init_v: torch.Tensor,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        s_vlen: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        state_temperature: torch.Tensor,
        state_positions: torch.Tensor,
        sink_len: int,
        ln_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_rows = q_tail.size(0)
        tail_len = q_tail.size(1)
        chunk_len = hl.specialize(init_k.size(1))
        head_dim = hl.specialize(q_tail.size(2))
        value_dim = hl.specialize(init_v.size(2))
        q_block_len = hl.specialize(32)
        state_block_len = hl.specialize(16)
        overflow_block_len = hl.specialize(32)

        assert tail_len == overflow_k.size(1) == overflow_v.size(1)
        assert chunk_len == s_vlen.size(1) == state_positions.size(0)
        assert head_dim == init_k.size(2) == overflow_k.size(2)
        assert value_dim == init_v.size(2) == overflow_v.size(2)

        out = torch.empty(
            [batch_rows, tail_len, value_dim], dtype=init_v.dtype, device=init_v.device
        )
        lse = torch.empty(
            [batch_rows, tail_len], dtype=torch.float32, device=q_tail.device
        )
        state_k = torch.empty_like(init_k)
        state_v = torch.empty_like(init_v)
        best_idx_buf = torch.empty(
            [batch_rows, tail_len], dtype=state_positions.dtype, device=q_tail.device
        )
        scale = 1.0 / math.sqrt(float(head_dim))

        for tile_b in hl.tile(batch_rows, block_size=1):
            tile_state_temperature = state_temperature[tile_b, :, :]
            tile_ln_weight = ln_weight[:].float()
            tile_ln_bias = ln_bias[:].float()
            tile_state_temperature_dtype = tile_state_temperature.to(init_k.dtype)

            state_k[tile_b, :chunk_len, :] = init_k[tile_b, :chunk_len, :]
            state_v[tile_b, :chunk_len, :] = init_v[tile_b, :chunk_len, :]

            for tile_chunk in hl.tile(tail_len, block_size=chunk_len):
                for tile_q in hl.tile(
                    tile_chunk.begin, tile_chunk.end
                ):  # , block_size=q_block_len):
                    a_q = q_tail[tile_b, tile_q, :].float()
                    acc = hl.zeros([tile_b, tile_q, value_dim], dtype=torch.float32)
                    row_max = hl.full(
                        [tile_b, tile_q], float("-inf"), dtype=torch.float32
                    )
                    row_sum = hl.zeros([tile_b, tile_q], dtype=torch.float32)

                    for tile_state in hl.tile(
                        chunk_len
                    ):  # , block_size=state_block_len):
                        s_k_block = state_k[tile_b, tile_state, :]
                        s_v_block = state_v[tile_b, tile_state, :]
                        tile_s_vlen_block = s_vlen[tile_b, tile_state, :].to(
                            s_v_block.dtype
                        )

                        s_k_float = s_k_block.float()
                        s_k_mean = torch.mean(s_k_float, dim=-1, keepdim=True)
                        s_k_centered = s_k_float - s_k_mean
                        s_k_var = torch.mean(
                            s_k_centered * s_k_centered, dim=-1, keepdim=True
                        )
                        s_k_norm = s_k_centered * torch.rsqrt(s_k_var + ln_eps)
                        s_k_norm = s_k_norm * tile_ln_weight + tile_ln_bias
                        s_k_norm_dtype = s_k_norm.to(s_k_block.dtype)
                        s_k_attn = (
                            s_k_norm_dtype * tile_state_temperature_dtype
                        ).float()

                        s_v_float = s_v_block.float()
                        s_v_norm = torch.sqrt(
                            torch.sum(s_v_float * s_v_float, dim=-1, keepdim=True)
                        )
                        s_v_attn = (s_v_float / s_v_norm.clamp_min(1e-12)).to(
                            s_v_block.dtype
                        )
                        s_v_attn = (s_v_attn * tile_s_vlen_block).float()

                        scores = torch.bmm(a_q, s_k_attn.transpose(1, 2)) * scale
                        tile_row_max = torch.amax(scores, dim=-1)
                        next_row_max = torch.maximum(row_max, tile_row_max)
                        prev_scale = torch.exp(row_max - next_row_max)
                        probs = torch.exp(scores - next_row_max[:, :, None])
                        row_sum = row_sum * prev_scale + torch.sum(probs, dim=-1)
                        acc = acc * prev_scale[:, :, None] + torch.bmm(probs, s_v_attn)
                        row_max = next_row_max

                    out[tile_b, tile_q, :] = (acc / row_sum[:, :, None]).to(out.dtype)
                    lse[tile_b, tile_q] = row_max + torch.log(row_sum)

                for tile_o in hl.tile(
                    tile_chunk.begin, tile_chunk.end
                ):  # , block_size=overflow_block_len):
                    o_k = overflow_k[tile_b, tile_o, :]
                    best_score = hl.full(
                        [tile_b, tile_o], float("-inf"), dtype=torch.float32
                    )
                    best_idx = hl.full(
                        [tile_b, tile_o], -1, dtype=state_positions.dtype
                    )

                    for tile_state in hl.tile(
                        chunk_len
                    ):  # , block_size=state_block_len):
                        s_k_block = state_k[tile_b, tile_state, :]
                        tile_state_positions = state_positions[tile_state]
                        tile_sink_mask = tile_state_positions >= sink_len

                        s_k_float = s_k_block.float()
                        s_k_mean = torch.mean(s_k_float, dim=-1, keepdim=True)
                        s_k_centered = s_k_float - s_k_mean
                        s_k_var = torch.mean(
                            s_k_centered * s_k_centered, dim=-1, keepdim=True
                        )
                        s_k_norm = s_k_centered * torch.rsqrt(s_k_var + ln_eps)
                        s_k_norm = s_k_norm * tile_ln_weight + tile_ln_bias
                        s_k_norm_dtype = s_k_norm.to(s_k_block.dtype)

                        sim = torch.bmm(o_k, s_k_norm_dtype.transpose(1, 2)).float()
                        sim = torch.where(
                            tile_sink_mask[None, None, :], sim, float("-inf")
                        )
                        tile_best_score = torch.amax(sim, dim=-1)
                        tile_best_rel = torch.argmax(sim, dim=-1)
                        tile_best_idx = tile_best_rel + tile_state.begin
                        better = tile_best_score > best_score
                        best_score = torch.where(better, tile_best_score, best_score)
                        best_idx = torch.where(better, tile_best_idx, best_idx)

                    best_idx_buf[tile_b, tile_o] = best_idx

                for tile_state in hl.tile(chunk_len):  # , block_size=state_block_len):
                    s_k_block = state_k[tile_b, tile_state, :]
                    tile_state_positions = state_positions[tile_state]
                    delta_k = hl.zeros(
                        [tile_b, tile_state, head_dim], dtype=torch.float32
                    )

                    for tile_o in hl.tile(
                        tile_chunk.begin, tile_chunk.end
                    ):  # , block_size=overflow_block_len):
                        o_k = overflow_k[tile_b, tile_o, :]
                        best_idx = best_idx_buf[tile_b, tile_o]
                        sim_max = (
                            tile_state_positions[None, None, :] == best_idx[:, :, None]
                        ).float()
                        delta_k = delta_k + torch.bmm(
                            sim_max.transpose(1, 2), o_k.float()
                        )

                    state_k[tile_b, tile_state, :] = (s_k_block.float() + delta_k).to(
                        s_k_block.dtype
                    )

                for tile_state in hl.tile(chunk_len):  # , block_size=state_block_len):
                    s_v_block = state_v[tile_b, tile_state, :]
                    tile_state_positions = state_positions[tile_state]
                    delta_v = hl.zeros(
                        [tile_b, tile_state, value_dim], dtype=torch.float32
                    )

                    for tile_o in hl.tile(
                        tile_chunk.begin, tile_chunk.end
                    ):  # , block_size=overflow_block_len):
                        o_v = overflow_v[tile_b, tile_o, :]
                        best_idx = best_idx_buf[tile_b, tile_o]
                        sim_max = (
                            tile_state_positions[None, None, :] == best_idx[:, :, None]
                        ).float()
                        delta_v = delta_v + torch.bmm(
                            sim_max.transpose(1, 2), o_v.float()
                        )

                    state_v[tile_b, tile_state, :] = (s_v_block.float() + delta_v).to(
                        s_v_block.dtype
                    )

        return state_k, state_v, out, lse

# class KVMCore(nn.Module):
#     def __init__(self, config, layer_idx: int):
#         super().__init__(config, layer_idx, config.kvm_value_residual_mode, config.kvm_token_shift_mode, qk_norm=True)

#         self.c_q = set_label("matrix_params", nn.Linear(self.hidden_size, self.hidden_size, bias=False))
#         self.c_k = set_label("matrix_params", nn.Linear(self.hidden_size, self.hidden_size, bias=False))
#         self.c_v = set_label("matrix_params", nn.Linear(self.hidden_size, self.hidden_size, bias=False))

#         # FIXME - this works for our current configs but won't if kvm is the alt layer
#         self.rope_partial_dim = config.rope_partial_dim if config.rope_partial_dim > 0 else self.head_dim

#         self.ln_d_k = set_label("scalars", nn.LayerNorm(self.head_dim))
#         self.merge_gate = set_label("matrix_params", nn.Linear(config.hidden_size, self.num_attention_heads, bias=False))
#         self.front_head_temp = set_label("scalars", nn.Parameter(torch.ones(config.num_attention_heads)))
#         self.state_head_temp = set_label("scalars", nn.Parameter(torch.ones(config.num_attention_heads)))

#     def prepare_activations(x):
#         g = 1 + elu(x @ self.W_merge_gate)

#         # remove rope and normalize
#         k_norope_normalized = cat([torch.zeros_like(k[..., :rope_partial_dim]), k[..., -rope_partial_dim:]], dim=-1)
#         k_norope_normalized = self.layernorm_s_k(k_norope_normalized)

#         return q, k, v, g, k_norope_normalized


def calc_k_norope_normalized_gated(k, rope_partial_dim, layernorm_s_k, g):
    k_norope = cat(
        [torch.zeros_like(k[..., :rope_partial_dim]), k[..., -rope_partial_dim:]],
        dim=-1,
    )
    k_norope_normalized = layernorm_s_k(k_norope)
    k_norope_normalized_gated = k_norope_normalized * g
    return k_norope_normalized_gated


def state_update(
    v,
    k_norope_normalized_gated,
    s_k,
    s_v,
    overflow_begin,
    chunk_len,
    sink_len,
    layernorm_s_k,
):
    # identify overflow chunk of tokens to merge into (or append to) the state
    o_k = k_norope_normalized_gated[:, :, overflow_begin : overflow_begin + chunk_len]
    o_v = v[:, :, overflow_begin : overflow_begin + chunk_len]

    # note: some tokens out of these will be appended, split and append
    # to be done as explained in the main text

    # obtain normalized state keys
    s_k_norm = layernorm_s_k(s_k)

    # find the most similar key in state for each incoming key to merge
    sim = o_k @ s_k_norm.mT
    sim[..., 0:sink_len] = float("-inf")
    best_sim, best_s_idx = sim.max(dim=-1, keepdim=True)
    sim_max = scatter(zeros_like(sim), -1, best_s_idx, torch.ones_like(sim))

    # update state by adding the most similar keys and their values, gated by the merge gate
    s_k = s_k + sim_max.mT @ o_k
    s_v = s_v + sim_max.mT @ o_v

    return s_k, s_v


def kvm_alternate_states_and_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_gated: Tensor,
    k_norope_normalized_gated: Tensor,
    s_vlen: Tensor,
    bswa_len: Tensor,
    chunk_len: Tensor,
    sink_len: Tensor,
    layernorm_s_k: nn.LayerNorm,
    state_temperature: Tensor,
    bswa_temperature: Tensor,
    causal_mask_after_state: Tensor,
):
    assert bswa_len > chunk_len
    state_attentions_results = [
        scaled_dot_product_attention(
            q[:, :, :bswa_len],
            k[:, :, :bswa_len] * bswa_temperature,
            v[:, :, :bswa_len],
            is_causal=True,
        )
    ]
    s_k, s_v = calc_initial_state(k_norope_normalized_gated, v_gated, chunk_len)

    for bswa_end in range(bswa_len + chunk_len, k.size(-2) + 1, chunk_len):
        bswa_begin = bswa_end - bswa_len

        # calculate attention across the newly updated state and BSWA window
        a_q = q[:, :, bswa_end - chunk_len : bswa_end]
        s_k_attn = layernorm_s_k(s_k) * state_temperature
        bswa_k = k[:, :, bswa_begin:bswa_end] * bswa_temperature
        s_v_attn = (normalize(s_v.float(), dim=-1) * s_vlen).to(s_v.dtype)
        bswa_v = v[:, :, bswa_begin:bswa_end]
        a_k = cat([s_k_attn, bswa_k], dim=-2)
        a_v = cat([s_v_attn, bswa_v], dim=-2)
        state_attentions_results.append(
            scaled_dot_product_attention(
                a_q, a_k, a_v, attn_mask=causal_mask_after_state
            )
        )

        overflow_begin = bswa_begin
        s_k, s_v = state_update(
            v_gated,
            k_norope_normalized_gated,
            s_k,
            s_v,
            overflow_begin,
            chunk_len,
            sink_len,
            layernorm_s_k,
        )
    return s_k, s_v, cat(state_attentions_results, dim=-2)


def kvm_states_first(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_gated: Tensor,
    k_norope_normalized_gated: Tensor,
    s_vlen: Tensor,
    bswa_len: Tensor,
    chunk_len: Tensor,
    sink_len: Tensor,
    layernorm_s_k: nn.LayerNorm,
    state_temperature: Tensor,
    bswa_temperature: Tensor,
    same_block_mask: BlockMask,
    bswa_mask: BlockMask,
):
    assert bswa_len > chunk_len
    s_k, s_v = calc_initial_state(k_norope_normalized_gated, v_gated, chunk_len)
    s_k_attns = []
    s_v_attns = []
    for bswa_end in range(bswa_len + chunk_len, k.size(-2) + 1, chunk_len):
        bswa_begin = bswa_end - bswa_len

        # FIXME - in the future consider moving state_temperature into the layernorm or applying to lse
        s_k_attns.append(layernorm_s_k(s_k) * state_temperature)
        s_v_attns.append((normalize(s_v.float(), dim=-1) * s_vlen).to(s_v.dtype))

        overflow_begin = bswa_begin
        s_k, s_v = state_update(
            v_gated,
            k_norope_normalized_gated,
            s_k,
            s_v,
            overflow_begin,
            chunk_len,
            sink_len,
            layernorm_s_k,
        )

    a_q = q[:, :, bswa_len:]
    s_k_attn = cat(s_k_attns, dim=-2)
    s_v_attn = cat(s_v_attns, dim=-2)
    state_att_result: Any = _flex_attn(
        a_q,
        s_k_attn,
        s_v_attn,
        block_mask=same_block_mask,
        return_aux=AuxRequest(lse=True),
    )
    state_att_out, state_aux_out = state_att_result

    # calculate BSWA attention across all tokens
    # FIXME - in the future consider moving bswa_temperature into the layernorm or applying to lse, or removing it in favor of only state_temperature
    bswa_result: Any = _flex_attn(
        q,
        k * bswa_temperature,
        v,
        block_mask=bswa_mask,
        return_aux=AuxRequest(lse=True),
    )
    bswa_out, bswa_aux_out = bswa_result

    # combine BSWA and state attention outputs
    split_point = bswa_len
    state_lse = _normalize_lse(state_aux_out.lse)
    bswa_lse = _normalize_lse(bswa_aux_out.lse)[:, :, split_point:]
    joint_lse = torch.logaddexp(state_lse, bswa_lse)
    state_weight = torch.exp(state_lse - joint_lse)
    bswa_weight = torch.exp(bswa_lse - joint_lse)
    out = torch.cat(
        [
            bswa_out[:, :, :split_point],
            state_att_out * state_weight.unsqueeze(-1)
            + bswa_out[:, :, split_point:] * bswa_weight.unsqueeze(-1),
        ],
        dim=-2,
    )

    return s_k, s_v, out


def kvm_states_first_triton_states(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_gated: Tensor,
    k_norope_normalized_gated: Tensor,
    s_vlen: Tensor,
    bswa_len: Tensor,
    chunk_len: Tensor,
    sink_len: Tensor,
    layernorm_s_k: nn.LayerNorm,
    state_temperature: Tensor,
    bswa_temperature: Tensor,
    same_block_mask: BlockMask,
    bswa_mask: BlockMask,
):
    assert bswa_len > chunk_len
    ln_weight = layernorm_s_k.weight
    ln_bias = layernorm_s_k.bias
    if ln_weight is None:
        ln_weight = torch.ones(q.size(-1), device=q.device, dtype=q.dtype)
    if ln_bias is None:
        ln_bias = torch.zeros(q.size(-1), device=q.device, dtype=q.dtype)

    s_k, s_v, s_k_attn, s_v_attn = kvm_generate_state_banks(
        k_norope_normalized_gated,
        v_gated,
        s_vlen,
        ln_weight,
        ln_bias,
        state_temperature.reshape(-1).to(q.dtype),
        int(bswa_len),
        int(chunk_len),
        int(sink_len),
        q.dtype,
        layernorm_s_k.eps,
    )

    a_q = q[:, :, bswa_len:]
    state_att_result: Any = _flex_attn(
        a_q,
        s_k_attn,
        s_v_attn,
        block_mask=same_block_mask,
        return_aux=AuxRequest(lse=True),
    )
    state_att_out, state_aux_out = state_att_result

    bswa_result: Any = _flex_attn(
        q,
        k * bswa_temperature,
        v,
        block_mask=bswa_mask,
        return_aux=AuxRequest(lse=True),
    )
    bswa_out, bswa_aux_out = bswa_result

    split_point = bswa_len
    state_lse = _normalize_lse(state_aux_out.lse)
    bswa_lse = _normalize_lse(bswa_aux_out.lse)[:, :, split_point:]
    joint_lse = torch.logaddexp(state_lse, bswa_lse)
    state_weight = torch.exp(state_lse - joint_lse)
    bswa_weight = torch.exp(bswa_lse - joint_lse)
    out = torch.cat(
        [
            bswa_out[:, :, :split_point],
            state_att_out * state_weight.unsqueeze(-1)
            + bswa_out[:, :, split_point:] * bswa_weight.unsqueeze(-1),
        ],
        dim=-2,
    )

    return s_k, s_v, out


def kvm_states_first_flex_last_inner(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_gated: Tensor,
    k_norope_normalized_gated: Tensor,
    s_vlen: Tensor,
    bswa_len: Tensor,
    chunk_len: Tensor,
    sink_len: Tensor,
    layernorm_s_k: nn.LayerNorm,
    state_temperature: Tensor,
    bswa_temperature: Tensor,
):
    assert bswa_len > chunk_len
    s_k, s_v = calc_initial_state(k_norope_normalized_gated, v_gated, chunk_len)
    state_attentions_results = []
    # Example: the first post-front query chunk attends against the initial state chunk 0 and BSWA chunks 1-3. Later iterations compact one new overflow chunk into state before attention.
    for bswa_end in range(bswa_len + chunk_len, k.size(-2) + 1, chunk_len):
        bswa_begin = bswa_end - bswa_len

        a_q = q[:, :, bswa_end - chunk_len : bswa_end]
        s_k_attn = (
            layernorm_s_k(s_k) * state_temperature
        )  # FIXME - in the future consider moving state_temperature into the layernorm or applying to lse
        s_v_attn = (normalize(s_v.float(), dim=-1) * s_vlen).to(s_v.dtype)
        state_attentions_results.append(
            _flex_attn(
                a_q,
                s_k_attn,
                s_v_attn,
                block_mask=None,
                return_aux=AuxRequest(lse=True),
            )
        )

        overflow_begin = bswa_begin
        s_k, s_v = state_update(
            v_gated,
            k_norope_normalized_gated,
            s_k,
            s_v,
            overflow_begin,
            chunk_len,
            sink_len,
            layernorm_s_k,
        )

    state_att_out = cat([out for out, _ in state_attentions_results], dim=-2)
    state_lse_out = cat(
        [_normalize_lse(aux.lse) for _, aux in state_attentions_results], dim=-1
    )

    return s_k, s_v, state_att_out, state_lse_out


def kvm_states_first_helion_last_inner(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_gated: Tensor,
    k_norope_normalized_gated: Tensor,
    s_vlen: Tensor,
    bswa_len: Tensor,
    chunk_len: Tensor,
    sink_len: Tensor,
    layernorm_s_k: nn.LayerNorm,
    state_temperature: Tensor,
    bswa_temperature: Tensor,
):
    del k, bswa_temperature
    _require_helion()
    assert bswa_len > chunk_len

    if q.device.type != "cuda":
        raise RuntimeError(
            "kvm_states_first_helion_last_inner requires CUDA tensor inputs"
        )

    chunk_len_int = int(chunk_len)
    bswa_len_int = int(bswa_len)
    sink_len_int = int(sink_len)
    tail_len = int(q.size(-2) - bswa_len_int)

    if tail_len < 0:
        raise ValueError(
            f"bswa_len ({bswa_len_int}) must not exceed sequence length ({q.size(-2)})"
        )
    if tail_len % chunk_len_int != 0:
        raise ValueError(
            f"tail length ({tail_len}) must be divisible by chunk_len ({chunk_len_int})"
        )

    if tail_len == 0:
        s_k, s_v = calc_initial_state(k_norope_normalized_gated, v_gated, chunk_len_int)
        return (
            s_k,
            s_v,
            v.new_empty(v.size(0), v.size(1), 0, v.size(-1)),
            q.new_empty(q.size(0), q.size(1), 0),
        )

    batch_size, num_attention_heads, _, head_dim = q.shape
    value_dim = int(v.size(-1))
    batch_rows = batch_size * num_attention_heads

    ln_weight = layernorm_s_k.weight
    ln_bias = layernorm_s_k.bias
    if ln_weight is None:
        ln_weight = torch.ones(head_dim, device=q.device, dtype=q.dtype)
    if ln_bias is None:
        ln_bias = torch.zeros(head_dim, device=q.device, dtype=q.dtype)

    init_s_k = k_norope_normalized_gated[:, :, :chunk_len_int]
    init_s_v = v_gated[:, :, :chunk_len_int]
    q_tail = q[:, :, bswa_len_int:]
    overflow_k = k_norope_normalized_gated[
        :, :, chunk_len_int : chunk_len_int + tail_len
    ]
    overflow_v = v_gated[:, :, chunk_len_int : chunk_len_int + tail_len]

    s_vlen_expanded = torch.broadcast_to(
        s_vlen, (batch_size, num_attention_heads, chunk_len_int, s_vlen.size(-1))
    )
    state_temperature_expanded = torch.broadcast_to(
        state_temperature, (batch_size, num_attention_heads, 1, 1)
    )
    state_positions = torch.arange(chunk_len_int, device=q.device, dtype=torch.int64)

    s_k_flat, s_v_flat, state_att_out, state_lse_out = (
        _kvm_state_attention_with_updates_helion_kernel(
            q_tail.contiguous().reshape(batch_rows, tail_len, head_dim),
            init_s_k.contiguous().reshape(batch_rows, chunk_len_int, head_dim),
            init_s_v.contiguous().reshape(batch_rows, chunk_len_int, value_dim),
            overflow_k.contiguous().reshape(batch_rows, tail_len, head_dim),
            overflow_v.contiguous().reshape(batch_rows, tail_len, value_dim),
            s_vlen_expanded.contiguous().reshape(
                batch_rows, chunk_len_int, s_vlen.size(-1)
            ),
            ln_weight,
            ln_bias,
            state_temperature_expanded.contiguous().reshape(batch_rows, 1, 1),
            state_positions,
            sink_len_int,
            float(layernorm_s_k.eps),
        )
    )

    s_k = s_k_flat.reshape(batch_size, num_attention_heads, chunk_len_int, head_dim)
    s_v = s_v_flat.reshape(batch_size, num_attention_heads, chunk_len_int, value_dim)
    state_att_out = state_att_out.reshape(batch_size, num_attention_heads, tail_len, value_dim)
    state_lse_out = state_lse_out.reshape(q.size(0), q.size(1), tail_len)

    return s_k, s_v, state_att_out, state_lse_out


def kvm_states_first_flex_last(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_gated: Tensor,
    k_norope_normalized_gated: Tensor,
    s_vlen: Tensor,
    bswa_len: Tensor,
    chunk_len: Tensor,
    sink_len: Tensor,
    layernorm_s_k: nn.LayerNorm,
    state_temperature: Tensor,
    bswa_temperature: Tensor,
    bswa_mask: BlockMask,
):
    s_k, s_v, state_att_out, state_lse_out = kvm_states_first_flex_last_inner(
        q,
        k,
        v,
        v_gated,
        k_norope_normalized_gated,
        s_vlen,
        bswa_len,
        chunk_len,
        sink_len,
        layernorm_s_k,
        state_temperature,
        bswa_temperature,
        bswa_mask,
    )

    # calculate BSWA attention across all tokens
    # FIXME - in the future consider moving bswa_temperature into the layernorm or applying to lse, or removing it in favor of only state_temperature
    bswa_result: Any = _flex_attn(
        q,
        k * bswa_temperature,
        v,
        block_mask=bswa_mask,
        return_aux=AuxRequest(lse=True),
    )
    bswa_out, bswa_aux_out = bswa_result

    # combine BSWA and state attention outputs
    split_point = bswa_len
    bswa_lse = _normalize_lse(bswa_aux_out.lse)[:, :, split_point:]
    joint_lse = torch.logaddexp(state_lse_out, bswa_lse)
    state_weight = torch.exp(state_lse_out - joint_lse)
    bswa_weight = torch.exp(bswa_lse - joint_lse)
    out = torch.cat(
        [
            bswa_out[:, :, :split_point],
            state_att_out * state_weight.unsqueeze(-1)
            + bswa_out[:, :, split_point:] * bswa_weight.unsqueeze(-1),
        ],
        dim=-2,
    )

    return s_k, s_v, out


def kvm_states_first_helion_last(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_gated: Tensor,
    k_norope_normalized_gated: Tensor,
    s_vlen: Tensor,
    bswa_len: Tensor,
    chunk_len: Tensor,
    sink_len: Tensor,
    layernorm_s_k: nn.LayerNorm,
    state_temperature: Tensor,
    bswa_temperature: Tensor,
    bswa_mask: BlockMask,
):
    s_k, s_v, state_att_out, state_lse_out = kvm_states_first_helion_last_inner(
        q,
        k,
        v,
        v_gated,
        k_norope_normalized_gated,
        s_vlen,
        bswa_len,
        chunk_len,
        sink_len,
        layernorm_s_k,
        state_temperature,
        bswa_temperature,
        bswa_mask,
    )

    bswa_result: Any = _flex_attn(
        q,
        k * bswa_temperature,
        v,
        block_mask=bswa_mask,
        return_aux=AuxRequest(lse=True),
    )
    bswa_out, bswa_aux_out = bswa_result

    split_point = bswa_len
    bswa_lse = _normalize_lse(bswa_aux_out.lse)[:, :, split_point:]
    joint_lse = torch.logaddexp(state_lse_out, bswa_lse)
    state_weight = torch.exp(state_lse_out - joint_lse)
    bswa_weight = torch.exp(bswa_lse - joint_lse)
    out = torch.cat(
        [
            bswa_out[:, :, :split_point],
            state_att_out * state_weight.unsqueeze(-1)
            + bswa_out[:, :, split_point:] * bswa_weight.unsqueeze(-1),
        ],
        dim=-2,
    )

    return s_k, s_v, out


def kvm_fake_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_gated: Tensor,
    k_norope_normalized_gated: Tensor,
    s_vlen: Tensor,
    bswa_len: Tensor,
    chunk_len: Tensor,
    sink_len: Tensor,
    layernorm_s_k: nn.LayerNorm,
    state_temperature: Tensor,
    bswa_temperature: Tensor,
    bswa_mask: BlockMask = None,
):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
    )


def kvm_states_first_flash_loop_flex_last(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_gated: Tensor,
    k_norope_normalized_gated: Tensor,
    s_vlen: Tensor,
    bswa_len: Tensor,
    chunk_len: Tensor,
    sink_len: Tensor,
    layernorm_s_k: nn.LayerNorm,
    state_temperature: Tensor,
    bswa_temperature: Tensor,
    bswa_mask: BlockMask,
):
    assert bswa_len > chunk_len
    s_k, s_v = calc_initial_state(k_norope_normalized_gated, v_gated, chunk_len)

    tail_len = int(q.size(-2) - int(bswa_len))
    output_dtype = torch.promote_types(q.dtype, v.dtype)
    compute_dtype = (
        torch.float32
        if output_dtype in {torch.float16, torch.bfloat16}
        else output_dtype
    )
    value_dim = int(v.size(-1))
    scale = q.size(-1) ** -0.5

    state_att_out = torch.empty(
        *q.shape[:-2], tail_len, value_dim, device=q.device, dtype=output_dtype
    )
    state_lse_out = torch.empty(
        *q.shape[:-2], tail_len, device=q.device, dtype=compute_dtype
    )

    for bswa_end in range(bswa_len + chunk_len, k.size(-2) + 1, chunk_len):
        bswa_begin = bswa_end - bswa_len
        tail_begin = bswa_begin - chunk_len
        tail_end = tail_begin + chunk_len

        a_q = q[:, :, bswa_end - chunk_len : bswa_end, :].to(compute_dtype) * scale
        s_k_attn = (layernorm_s_k(s_k) * state_temperature).to(compute_dtype)
        s_v_attn = (normalize(s_v.float(), dim=-1) * s_vlen).to(compute_dtype)
        scores = torch.matmul(a_q, s_k_attn.transpose(-1, -2))

        row_max = scores.amax(dim=-1)
        safe_row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )
        probs = torch.where(
            torch.isfinite(scores),
            torch.exp(scores - safe_row_max.unsqueeze(-1)),
            torch.zeros_like(scores),
        )
        row_sum = probs.sum(dim=-1)
        out_block = torch.where(
            row_sum.unsqueeze(-1) > 0,
            torch.matmul(probs, s_v_attn) / row_sum.unsqueeze(-1),
            torch.zeros(
                *probs.shape[:-1], value_dim, device=q.device, dtype=compute_dtype
            ),
        )
        lse_block = torch.where(
            row_sum > 0,
            safe_row_max + torch.log(row_sum),
            torch.full_like(row_sum, float("-inf")),
        )

        state_att_out[:, :, tail_begin:tail_end] = out_block.to(output_dtype)
        state_lse_out[:, :, tail_begin:tail_end] = lse_block

        overflow_begin = bswa_begin
        s_k, s_v = state_update(
            v_gated,
            k_norope_normalized_gated,
            s_k,
            s_v,
            overflow_begin,
            chunk_len,
            sink_len,
            layernorm_s_k,
        )

    bswa_result: Any = _flex_attn(
        q,
        k * bswa_temperature,
        v,
        block_mask=bswa_mask,
        return_aux=AuxRequest(lse=True),
    )
    bswa_out, bswa_aux_out = bswa_result

    split_point = bswa_len
    bswa_lse = _normalize_lse(bswa_aux_out.lse)[:, :, split_point:]
    joint_lse = torch.logaddexp(state_lse_out, bswa_lse)
    state_weight = torch.exp(state_lse_out - joint_lse)
    bswa_weight = torch.exp(bswa_lse - joint_lse)
    out = torch.cat(
        [
            bswa_out[:, :, :split_point],
            state_att_out * state_weight.unsqueeze(-1)
            + bswa_out[:, :, split_point:] * bswa_weight.unsqueeze(-1),
        ],
        dim=-2,
    )

    return s_k, s_v, out

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash and DFlash2 support for the vLLM revision used by the LOD benchmarks.

The benchmark environment already contains vLLM's DFlash model runner, but it
predates the Gemma4 fixes and the small DFlash2 model/path-selector additions.
Keep the compatibility code in the plugin so full-attention and CUSTOM-attention
runs load exactly the same drafter without modifying the installed vLLM package.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.config import CacheConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models import qwen3_dflash as _dflash_model
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32, tl_rand64
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}"
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Any,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    """DFlash2 model adapted to vLLM before ``decoder_layer_cls`` existed."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        previous = _dflash_model.DFlashQwen3DecoderLayer
        _dflash_model.DFlashQwen3DecoderLayer = DFlash2Qwen3DecoderLayer
        try:
            super().__init__(
                vllm_config=vllm_config,
                start_layer_id=start_layer_id,
                prefix=prefix,
            )
        finally:
            _dflash_model.DFlashQwen3DecoderLayer = previous
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DFlash2 head adapted to vLLM before ``model_cls`` existed."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        previous = _dflash_model.DFlashQwen3Model
        _dflash_model.DFlashQwen3Model = DFlash2Qwen3Model
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _dflash_model.DFlashQwen3Model = previous
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processor = self.candidate_logits_processor
        logits = processor._apply_head(self.lm_head, hidden_states, None)
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        top_k = self.model.candidate_selector.top_k
        values, ids = torch.topk(logits, top_k, dim=-1, sorted=False)
        ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index
        if self.lm_head.tp_size > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = torch.topk(values, top_k, dim=-1, sorted=False)
            ids = ids.gather(-1, selected)
        values = values.float()
        if processor.scale != 1.0:
            values = values * processor.scale
        if processor.soft_cap is not None:
            values = torch.tanh(values / processor.soft_cap) * processor.soft_cap
        return ids, values


class DFlashGemmaCompatQwen3Model(DFlashQwen3Model):
    """Original DFlash draft model with Gemma4 target embedding semantics."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        target_config = vllm_config.model_config.hf_text_config
        self.input_embedding_scale = (
            float(target_config.hidden_size) ** 0.5
            if str(getattr(target_config, "model_type", "")).startswith("gemma4")
            else 1.0
        )
        # DFlash operates on text/query tokens whose context K/V is prewritten.
        # It must not inherit the Gemma4 target's multimodal-prefix masking.
        for layer in self.layers:
            layer.self_attn.attn.use_mm_prefix = False

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlashGemmaCompatForCausalLM(DFlashQwen3ForCausalLM):
    """Original DFlash head with Gemma4 normalization and logit soft-capping."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        previous = _dflash_model.DFlashQwen3Model
        _dflash_model.DFlashQwen3Model = DFlashGemmaCompatQwen3Model
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _dflash_model.DFlashQwen3Model = previous
        softcap = float(getattr(self.config, "final_logit_softcapping", 0.0) or 0.0)
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=float(getattr(self.config, "logit_scale", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )


@triton.jit
def _prepare_dflash_inputs_gemma_compat_kernel(
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    temperature_ptr,
    seeds_ptr,
    block_table_ptr,
    block_table_stride,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Prepare compact DFlash rows without retaining rejected target suffixes."""
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    # A fully rejected target row can occur during batch verification.  Use
    # the scheduled row's first position as the next-position anchor then.
    fallback_last_pos = tl.load(target_positions_ptr + ctx_start) - 1
    last_valid_pos_idx = tl.maximum(valid_ctx_end - 1, ctx_start)
    last_valid_pos = tl.load(
        target_positions_ptr + last_valid_pos_idx,
        mask=num_valid_ctx > 0,
        other=fallback_last_pos,
    )
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    ctx_pos_idx = ctx_start + tl.where(is_valid_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    ctx_block_num = tl.minimum(ctx_pos // block_size, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(
        out_context_positions_ptr + ctx_start + j,
        tl.where(is_valid_ctx, ctx_pos, 0),
        mask=is_ctx,
    )
    tl.store(
        out_context_slot_mapping_ptr + ctx_start + j,
        tl.where(is_valid_ctx, ctx_slot, PAD_SLOT_ID),
        mask=is_ctx,
    )

    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
    q_block_num = tl.minimum(query_pos // block_size, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_slot = q_block_id * block_size + (query_pos % block_size)
    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    tl.store(
        out_query_positions_ptr + query_idx,
        tl.minimum(query_pos, max_model_len - 1),
        mask=is_query,
    )
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len),
        )
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
        if req_idx == num_reqs - 1:
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


_TL_RAND_MIN = tl.constexpr(4.6566127342e-10) if HAS_TRITON else 4.6566127342e-10
_DRAFT_NOISE_SALT = tl.constexpr(1 << 30) if HAS_TRITON else (1 << 30)


@triton.jit
def _gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    USE_FP64: tl.constexpr,
):
    if temp != 0.0:
        logits = logits / temp
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        gumbel_seed = tl.randint(seed, pos + _DRAFT_NOISE_SALT)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            u = tl.maximum(u, _TL_RAND_MIN)
            noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(mask, logits + noise, float("-inf"))
    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )
        sample_pos = tl.load(sample_pos_ptr + flat) - 1
        _, index = _gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            sample_pos,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            USE_FP64=USE_FP64,
        )
        tl.store(realized_scores_ptr + candidate_base + offsets, scores, mask=mask & valid)
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )
        if self.draft_logits is not None:
            self.draft_logits.fill_(-float("inf"))

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        block_k = triton.next_power_of_2(self.selector_top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        assert self.draft_logits is not None
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            self.draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            self.draft_logits.stride(0),
            self.draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self._sample_path(candidate_ids, scores, num_reqs)
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)


def _is_dflash2(vllm_config: VllmConfig) -> bool:
    config = vllm_config.speculative_config
    if config is None or config.method != "dflash":
        return False
    return "DFlash2DraftModel" in config.draft_model_config.architectures


def register_dflash2_compat() -> None:
    """Register compatible DFlash models/speculators, idempotently."""
    from vllm import ModelRegistry
    from vllm.v1.worker.gpu import spec_decode

    ModelRegistry.register_model(
        "DFlashDraftModel",
        "vllm_lod_plugin.dflash2_compat:DFlashGemmaCompatForCausalLM",
    )
    ModelRegistry.register_model(
        "DFlash2DraftModel",
        "vllm_lod_plugin.dflash2_compat:DFlash2Qwen3ForCausalLM",
    )
    # The pinned V2 runner writes rejected target tokens back into the draft
    # cache and does not handle a fully rejected row safely. Replace only the
    # Triton preparation kernel; the surrounding speculator and fixed buffer
    # layout stay unchanged.
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator

    dflash_speculator._prepare_dflash_inputs_kernel = (
        _prepare_dflash_inputs_gemma_compat_kernel
    )
    if getattr(spec_decode, "_vllm_lod_dflash2_installed", False):
        return
    original = spec_decode.init_speculator

    def init_speculator(vllm_config: VllmConfig, device: torch.device):
        if _is_dflash2(vllm_config):
            return DFlash2Speculator(vllm_config, device)
        return original(vllm_config, device)

    spec_decode.init_speculator = init_speculator
    # model_runner imports the factory by value. Patch it too when already loaded;
    # otherwise its later import observes the replacement above.
    import sys

    runner = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if runner is not None:
        runner.init_speculator = init_speculator
    spec_decode._vllm_lod_dflash2_installed = True


EntryClass = DFlash2Qwen3ForCausalLM

__all__ = [
    "DFlashGemmaCompatForCausalLM",
    "DFlash2Qwen3ForCausalLM",
    "DFlash2Speculator",
    "register_dflash2_compat",
]

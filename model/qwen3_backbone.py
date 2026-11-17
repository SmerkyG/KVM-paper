"""Qwen3 blocks with the paper's partial RoPE and pre-shifted-label trainer.

RMSNorm, SwiGLU, GQA, QK RMSNorm, and initialization use Transformers' Qwen3
implementation. Only the rotary application is replaced to match GPTAlpha-2's
64-of-128-channel positional encoding. The shared loss helper deliberately
does not shift labels a second time.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from utils.defer import defer
from utils.opt import set_label
from .rwkv7_backbone import (
    MixerConfig, RotaryEmbedding, RWKV7BackboneForCausalLM, apply_rotary_embeddings,
)


class PartialRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rotary = RotaryEmbedding(
            full_dim=config.d_qk_head or config.hidden_size // config.num_attention_heads,
            partial_dim=config.rope_partial_dim,
            base=config.rope_theta,
        )

    def forward(self, hidden_states, position_ids):
        return self.rotary(position_ids)


class PartialQwen3Attention(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            setattr(self, name, getattr(original, name))
        for name in ("config", "layer_idx", "head_dim", "num_key_value_groups",
                     "scaling", "attention_dropout", "is_causal", "sliding_window"):
            setattr(self, name, getattr(original, name))

    def forward(
        self, hidden_states, position_embeddings, attention_mask=None,
        past_key_values=None, cache_position=None, **kwargs,
    ):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shape))
        k = self.k_norm(self.k_proj(hidden_states).view(shape))
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q = apply_rotary_embeddings(q, position_embeddings).transpose(1, 2)
        k = apply_rotary_embeddings(k, position_embeddings).transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(
                k, v, self.layer_idx, {"cache_position": cache_position}
            )
        output, _ = sdpa_attention_forward(
            self, q, k, v, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )
        return self.o_proj(output.reshape(*hidden_states.shape[:-1], -1)), None


class PartialQwen3DecoderLayer(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.attention_type = original.attention_type
        self.self_attn = PartialQwen3Attention(original.self_attn)
        self.mlp = original.mlp
        self.input_layernorm = original.input_layernorm
        self.post_attention_layernorm = original.post_attention_layernorm

    @defer(torch.compile)
    def forward(
        self, hidden_states, attention_mask=None, position_embeddings=None,
        past_key_values=None, cache_position=None, **kwargs,
    ):
        attention, _ = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
            cache_position=cache_position,
        )
        hidden_states = hidden_states + attention
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3BackboneForCausalLM(RWKV7BackboneForCausalLM):
    """Trainer-compatible, randomly initialized Qwen3-style full-attention LM."""

    config_class = MixerConfig

    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        if config.use_tokenshift_att or config.use_tokenshift_ffn or config.use_value_residual:
            raise ValueError("The Qwen control does not use token shift or value residuals")
        head_dim = config.d_qk_head or config.hidden_size // config.num_attention_heads
        if not 0 < config.rope_partial_dim <= head_dim or config.rope_partial_dim % 2:
            raise ValueError("Qwen partial RoPE must have a positive even width <= head_dim")
        qwen_config = Qwen3Config(
            vocab_size=config.vocab_size, hidden_size=config.hidden_size,
            intermediate_size=config.qwen_intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads or config.num_attention_heads,
            head_dim=head_dim, max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.qwen_rms_norm_eps,
            initializer_range=config.qwen_initializer_range,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
            tie_word_embeddings=False, use_cache=False, attention_dropout=0.0,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
        )
        qwen_config._attn_implementation = "sdpa"
        self.transformer = Qwen3Model(qwen_config)
        self.transformer.rotary_emb = PartialRotaryEmbedding(config)
        self.transformer.layers = nn.ModuleList(
            [PartialQwen3DecoderLayer(layer) for layer in self.transformer.layers]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.qwen_initializer_range)
        self.post_init()
        for name, parameter in self.named_parameters():
            if parameter.ndim < 2:
                label = "scalars"
            elif name == "transformer.embed_tokens.weight":
                label = "wte_embed"
            elif name == "lm_head.weight":
                label = "lm_head"
            else:
                label = "matrix_params"
            set_label(label, parameter)

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
        logits_to_keep=0, return_logits=True, **kwargs,
    ):
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("The Wave 1 Qwen run uses the shared cram packing, not varlen packing")
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=(labels is None) if use_cache is None else use_cache,
        )
        return self._compute_loss_and_logits(
            outputs, labels=labels, logits_to_keep=logits_to_keep,
            return_logits=return_logits,
        )


Qwen3BackboneForCausalLM.register_for_auto_class("AutoModelForCausalLM")

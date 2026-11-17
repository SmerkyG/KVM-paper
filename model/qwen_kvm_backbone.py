"""Qwen control with KVM in place of full attention."""

from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from utils.defer import defer
from .qwen3_backbone import Qwen3BackboneForCausalLM
from model.kvm_mixer import SequenceMixer
from model.statesdictcache import StatesDictCache


class QwenKVMCache(StatesDictCache):
    # The mixer owns masking; do not generate a dense full-history HF mask.
    is_compileable = False

    def get_seq_length(self, layer_idx=0):
        return super().get_seq_length(layer_idx)


class QwenKVMAttention(nn.Module):
    def __init__(self, config, original, mixer_class):
        super().__init__()
        self.mixer = mixer_class(config, original.layer_idx)
        # Keep Qwen's initial projections, head geometry and Q/K RMSNorm.
        self.mixer.c_q = original.q_proj
        self.mixer.c_k = original.k_proj
        self.mixer.c_v = original.v_proj
        self.mixer.c_proj = original.o_proj
        self.mixer.ln_q = original.q_norm
        self.mixer.ln_k = original.k_norm

    def forward(
        self, hidden_states, position_embeddings, attention_mask=None,
        past_key_values=None, cache_position=None, **kwargs,
    ):
        if past_key_values is not None and not self.training:
            attention = self.mixer.forward_inference
        else:
            attention = self.mixer
        return attention(
            hidden_states, v_first=None, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
        ), None


class QwenKVMDecoderLayer(nn.Module):
    def __init__(self, config, original, mixer_class):
        super().__init__()
        self.self_attn = QwenKVMAttention(config, original.self_attn, mixer_class)
        self.mlp = original.mlp
        self.input_layernorm = original.input_layernorm
        self.post_attention_layernorm = original.post_attention_layernorm

    def forward(self, hidden_states, position_embeddings, past_key_values=None):
        hidden_states = self.forward_attention(hidden_states, position_embeddings, past_key_values)
        return self.forward_mlp(hidden_states)

    # Keep KVM dispatch outside Inductor. Its mixer owns the attention backend,
    # and the FFN and loss compile independently.
    def forward_attention(self, hidden_states, position_embeddings, past_key_values=None):
        attention, _ = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_values=past_key_values,
        )
        return hidden_states + attention

    @defer(torch.compile)
    def forward_mlp(self, hidden_states):
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class QwenKVMBackboneForCausalLM(Qwen3BackboneForCausalLM):
    def __init__(self, config):
        if config.token_mixer != "kvm":
            raise ValueError("Q1 requires a KVM mixer")
        if config.token_mixer_class_path is None:
            mixer_class = SequenceMixer
        elif config.token_mixer_class_path == "model.kvm_triton_mixer.SequenceMixer":
            from model.kvm_triton_mixer import SequenceMixer as TritonSequenceMixer

            mixer_class = TritonSequenceMixer
        else:
            raise ValueError(f"Unsupported Q1 mixer: {config.token_mixer_class_path!r}")
        super().__init__(config)
        # Finish initializing all common parameters before adding KVM-specific ones.
        self.transformer.layers = nn.ModuleList([
            QwenKVMDecoderLayer(config, layer, mixer_class) for layer in self.transformer.layers
        ])

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
        cache_position=None, logits_to_keep=0, return_logits=True, **kwargs,
    ):
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("Q1 uses the shared cram packing")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
            raise NotImplementedError("KVM expects unpadded prompts")
        if use_cache is None:
            use_cache = labels is None
        if use_cache:
            if past_key_values is None:
                past_key_values = QwenKVMCache()
            elif not isinstance(past_key_values, StatesDictCache):
                raise TypeError("Q1 requires a KVM state cache")
        elif past_key_values is not None:
            raise ValueError("A KVM state cache requires use_cache=True")

        outputs = BaseModelOutputWithPast(
            last_hidden_state=self.forward_hidden_states(
                input_ids, inputs_embeds=inputs_embeds, position_ids=position_ids,
                past_key_values=past_key_values, cache_position=cache_position,
            ),
            past_key_values=past_key_values,
        )
        return self._compute_loss_and_logits(
            outputs, labels=labels, logits_to_keep=logits_to_keep,
            return_logits=return_logits,
        )

    def forward_hidden_states(
        self, input_ids, *, inputs_embeds=None, position_ids=None,
        past_key_values=None, cache_position=None,
    ):
        hidden_states = self.transformer.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            if cache_position is None:
                offset = past_key_values.get_seq_length(0) if past_key_values is not None else 0
                cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device) + offset
            position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.transformer.rotary_emb(hidden_states, position_ids)
        for layer in self.transformer.layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )
        return self.transformer.norm(hidden_states)


QwenKVMBackboneForCausalLM.register_for_auto_class("AutoModelForCausalLM")

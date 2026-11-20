"""Run the native Qwen3 projections and Q/K norms in RetroInfer's runtime."""
import gc

import flashinfer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from model_hub.LLM import LLM
from model_hub.llama import LlamaLayer, LlamaModel


class Qwen3Layer(LlamaLayer):
    def init_layer(self, layer):
        self.q_norm = layer.self_attn.q_norm.weight.detach().to(self.device)
        self.k_norm = layer.self_attn.k_norm.weight.detach().to(self.device)
        self.q_norm_eps = layer.self_attn.q_norm.variance_epsilon
        self.k_norm_eps = layer.self_attn.k_norm.variance_epsilon
        super().init_layer(layer)


class RetroQwen3Model(LlamaModel):
    """Single-GPU Qwen3-8B with per-head Q/K normalization before RoPE."""
    lod_has_qk_norm = True

    def __init__(self, model_name, max_length, dtype, device_map, tokenizer=None, *, rope_scaling=None):
        LLM.__init__(self, model_name, max_length, dtype, device_map)
        if device_map != 'cuda:0':
            raise ValueError('Qwen3 benchmark supports one GPU')
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.config = Qwen3Config.from_pretrained(model_name)
        if rope_scaling is not None:
            from benchmarks.qwen3_rope import configure_rope
            configure_rope(self.config, rope_scaling)
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.config.head_dim
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError('unsupported Qwen3 attention geometry')
        self.max_position_embeddings = self.config.max_position_embeddings
        self.vocab_size = self.config.vocab_size
        self.eos_tokens = [self.config.eos_token_id]
        self.init_model()

    def init_model(self):
        hf = Qwen3ForCausalLM.from_pretrained(self.model_name, config=self.config, torch_dtype=self.dtype)
        self.num_gpus = 1
        self.layer_mapping = {str(i): self.device_map for i in range(self.num_layers)}
        self.embed_tokens = hf.model.embed_tokens.weight.detach().to(self.device_map)
        self.lm_head = hf.lm_head.weight.detach().to(self.device_map)
        self.norm_weight = hf.model.norm.weight.detach().to(self.device_map)
        self.norm_variance_epsilon = hf.model.norm.variance_epsilon
        self.position_ids = torch.arange(self.max_length, device=self.device_map)
        self.inv_freq = hf.model.rotary_emb.inv_freq.detach().to(self.device_map)
        self.attention_scaling = hf.model.rotary_emb.attention_scaling
        cos, sin = self._set_cos_sin_cache()
        self.cos_sin_cache = torch.cat((cos, sin), dim=-1)
        self.layers = []
        for idx, hf_layer in enumerate(hf.model.layers):
            layer = Qwen3Layer(idx, self.device_map)
            layer.init_layer(hf_layer)
            self.layers.append(layer)
            hf.model.layers[idx] = None
        del hf, self.inv_freq, cos, sin
        gc.collect()
        torch.cuda.empty_cache()

    def wqkv(self, hidden_states, layer):
        qkv = F.linear(hidden_states, layer.wqkv)
        q, k, v = qkv.split((self.hidden_size, self.num_key_value_heads * self.head_dim,
                             self.num_key_value_heads * self.head_dim), dim=-1)
        q_shape, k_shape = q.shape, k.shape
        q = flashinfer.rmsnorm(q.reshape(-1, self.head_dim).contiguous(), layer.q_norm, layer.q_norm_eps).view(q_shape)
        k = flashinfer.rmsnorm(k.reshape(-1, self.head_dim).contiguous(), layer.k_norm, layer.k_norm_eps).view(k_shape)
        return q, k, v

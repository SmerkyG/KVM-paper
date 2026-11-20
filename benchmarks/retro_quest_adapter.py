"""Independent QUEST-style BF16 attention in RetroInfer's Llama runtime.

Uses the same 16-token page bounds and selection kernels as our experimental
vLLM implementation, with native GQA, dense first two layers, and full-model
decode graph capture. This is not the authors' released QUEST implementation.
"""
import importlib.util
import math
import types
from pathlib import Path

import torch
import triton
import triton.language as tl


_kernel_path = (Path(__file__).resolve().parents[1] / 'integrations' / 'vllm_lod'
                / 'vllm_lod_plugin' / 'quest_kernels.py')
_spec = importlib.util.spec_from_file_location('retro_quest_kernels', _kernel_path)
_kernels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kernels)


@triton.jit
def _append(K, V, Cache, Slot, S0: tl.constexpr, S1: tl.constexpr,
            S2: tl.constexpr, S3: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0)
    d = tl.arange(0, D)
    slot = tl.load(Slot)
    base = (slot // 16) * S0 + h * S1 + (slot % 16) * S2 + d * S3
    tl.store(Cache + base, tl.load(K + h * D + d))
    tl.store(Cache + base + D * S3, tl.load(V + h * D + d))


class RetroQuestCache:
    def __init__(self, model, budget_tokens):
        if model.batch_size != 1 or model.num_gpus != 1:
            raise NotImplementedError('QUEST adapter supports batch one on one GPU')
        self.context = 0
        self.layer_num = model.num_layers
        self.heads = model.num_heads
        self.kv_heads = model.num_key_value_heads
        self.dim = model.head_dim
        self.device = torch.device(model.device_map)
        self.capacity = ((model.input_length + model.max_new_length + 15) // 16) * 16
        self.pages = self.capacity // 16
        self.budget = budget_tokens
        self.table = torch.arange(self.pages, device=self.device, dtype=torch.int32)[None, :]
        self.slot = torch.zeros(1, device=self.device, dtype=torch.int64)
        self.lengths = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.graph_position = torch.zeros((1, 1), device=self.device, dtype=torch.int64)
        self.graph_mode = False
        self.graph = None
        self.dense_k = [torch.empty((1, self.capacity, self.kv_heads, self.dim),
                                    device=self.device, dtype=model.dtype) for _ in range(2)]
        self.dense_v = [torch.empty_like(k) for k in self.dense_k]
        self.page_cache = [None] * 2 + [
            torch.empty((self.pages, 16, self.kv_heads, 2 * self.dim),
                        device=self.device, dtype=model.dtype).transpose(1, 2)
            for _ in range(self.layer_num - 2)]
        self.bounds = [None] * 2 + [
            torch.empty((self.pages, self.kv_heads, 2, self.dim),
                        device=self.device, dtype=model.dtype)
            for _ in range(self.layer_num - 2)]

    def prefill_update_kv_cache(self, q, k, v, layer_idx, start_bdx):
        if start_bdx != 0 or k.shape[0] != 1:
            raise ValueError('QUEST requires a single unpadded prompt')
        length = k.shape[1]
        if layer_idx < 2:
            self.dense_k[layer_idx][:, :length].copy_(k)
            self.dense_v[layer_idx][:, :length].copy_(v)
        else:
            cache = self.page_cache[layer_idx]
            bounds = self.bounds[layer_idx]
            full, remainder = divmod(length, 16)
            if full:
                keys = k[0, :full * 16].view(full, 16, self.kv_heads, self.dim)
                vals = v[0, :full * 16].view(full, 16, self.kv_heads, self.dim)
                cache[:full, :, :, :self.dim].copy_(keys.transpose(1, 2))
                cache[:full, :, :, self.dim:].copy_(vals.transpose(1, 2))
                bounds[:full, :, 0].copy_(keys.amin(dim=1))
                bounds[:full, :, 1].copy_(keys.amax(dim=1))
            if remainder:
                keys = k[0, full * 16:]
                vals = v[0, full * 16:]
                cache[full, :, :remainder, :self.dim].copy_(keys.transpose(0, 1))
                cache[full, :, :remainder, self.dim:].copy_(vals.transpose(0, 1))
                bounds[full, :, 0].copy_(keys.amin(dim=0))
                bounds[full, :, 1].copy_(keys.amax(dim=0))
        if layer_idx == self.layer_num - 1:
            self.context += length
        return k, v

    def sync(self, layer_idx, start_bdx):
        pass

    def decode_update_kv_cache(self, k, v, layer_idx):
        if layer_idx == 0 and not self.graph_mode:
            self._set_position(self.context)
        if layer_idx == self.layer_num - 1:
            self.context += 1
        return k, v

    def decode_attention(self, q, k, v, layer_idx):
        if layer_idx < 2:
            from flash_attn import flash_attn_with_kvcache
            self.dense_k[layer_idx].index_copy_(1, self.slot, k)
            self.dense_v[layer_idx].index_copy_(1, self.slot, v)
            return flash_attn_with_kvcache(q, self.dense_k[layer_idx],
                self.dense_v[layer_idx], cache_seqlens=self.lengths)
        cache = self.page_cache[layer_idx]
        _append[(self.kv_heads,)](k, v, cache, self.slot, *cache.stride(), self.dim,
                                  num_warps=4)
        _kernels.update_bounds(cache, self.slot, self.bounds[layer_idx])
        query = q.reshape(1, self.heads, self.dim)
        out = torch.empty_like(query)
        _kernels.quest_attention(query, cache, self.table, self.lengths,
            self.bounds[layer_idx], out, self.budget, 1.0 / math.sqrt(self.dim))
        return out.reshape_as(q)

    def _set_position(self, position):
        self.slot.fill_(position)
        self.lengths.fill_(position + 1)
        self.graph_position.fill_(position)

    def capture_decode(self, model, initial_token):
        previous = self.context
        self.graph_mode = True
        self.graph_input = initial_token.clone()
        self._set_position(previous)
        model.sampling(model.decode_forward(inputs_ids=self.graph_input))
        self.context = previous
        torch.cuda.synchronize()
        self._set_position(previous)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.graph_logits = model.decode_forward(inputs_ids=self.graph_input)
            self.graph_output = model.sampling(self.graph_logits)
        self.context = previous

    def graph_step(self, token):
        self._set_position(self.context)
        self.graph_input.copy_(token)
        self.graph.replay()
        self.context += 1
        return self.graph_output

    def checked_graph_step(self, model, token):
        previous = self.context
        graph_token = self.graph_step(token)
        graph_logits = self.graph_logits.clone()
        self.context = previous
        self._set_position(previous)
        eager_logits = model.decode_forward(inputs_ids=token)
        difference = (graph_logits - eager_logits).abs()
        return graph_token, dict(max_abs=float(difference.max()),
            mean_abs=float(difference.mean()), graph_argmax=int(graph_logits.argmax()),
            eager_argmax=int(eager_logits.argmax()))


def attach_quest(model, budget_tokens):
    original_position = model.position_embedd

    def init_cache(this, valid_start, config):
        if len(valid_start) != 1 or int(valid_start[0]) != 0:
            raise ValueError('QUEST requires one unpadded request')
        this.kv_cache = None
        this.kv_cache = RetroQuestCache(this, budget_tokens)

    def decode(this, q, k, v, layer_idx):
        return this.kv_cache.decode_attention(q, k, v, layer_idx)

    def move(this):
        torch.cuda.empty_cache()

    def position(this, q, k):
        if this.kv_cache.graph_mode:
            return this.apply_rotary_pos_emb(q, k, this.kv_cache.graph_position)
        return original_position(q, k)

    model.init_kv_cache = types.MethodType(init_cache, model)
    model.decode_attention = types.MethodType(decode, model)
    model.move = types.MethodType(move, model)
    model.position_embedd = types.MethodType(position, model)

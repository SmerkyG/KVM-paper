"""Independent author Squeezed controllers in a batched RetroInfer model.

Each request has its own fixed-prefix state; released author kernel launches
run sequentially for the requests inside a whole-model CUDA graph.
"""
import math
from types import MethodType, SimpleNamespace
import torch
from flash_attn import flash_attn_with_kvcache
from benchmarks.retro_squeezed_adapter import RetroSqueezedCache, sparse_attention


class RetroSqueezedBatchCache:
    def __init__(self, model, index, percentile):
        self.batch = model.batch_size
        self.layer_num = model.num_layers
        self.context = 0
        self.graph_mode = False
        self.graph_position = torch.zeros((self.batch, 1), dtype=torch.int64, device=model.device_map)
        single = SimpleNamespace(batch_size=1, num_gpus=model.num_gpus, dtype=model.dtype,
            input_length=model.input_length, max_new_length=model.max_new_length,
            num_layers=model.num_layers, num_heads=model.num_heads,
            num_key_value_heads=model.num_key_value_heads, head_dim=model.head_dim,
            device_map=model.device_map)
        self.requests = [RetroSqueezedCache(single, index, percentile) for _ in range(self.batch)]
        self.block_m = self.requests[0].block_m
        self.num_warps = self.requests[0].num_warps
        self.selection_counts = None

    def prefill_update_kv_cache(self, q, k, v, layer_idx, start_bdx):
        self.requests[start_bdx].prefill_update_kv_cache(q, k, v, layer_idx, 0)
        if layer_idx == self.layer_num - 1 and start_bdx == self.batch - 1:
            self.context += k.shape[1]
        return k, v

    def sync(self, layer_idx, start_bdx):
        pass

    def decode_update_kv_cache(self, k, v, layer_idx):
        if layer_idx == self.layer_num - 1:
            self.context += 1
        return k, v

    def decode_attention(self, q, k, v, layer_idx):
        return torch.cat([request.decode_attention(q[i:i+1], k[i:i+1], v[i:i+1], layer_idx)
                          for i, request in enumerate(self.requests)], dim=0)

    def _set_position(self, previous):
        self.graph_position.fill_(previous)
        for request in self.requests:
            request._set_position(previous)

    def capture_decode(self, model, token):
        previous = self.context
        self.graph_mode = True
        self.graph_input = token.clone()
        self._set_position(previous)
        model.sampling(model.decode_forward(inputs_ids=self.graph_input))
        self.context = previous
        torch.cuda.synchronize()
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
        return graph_token, dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
            graph_argmax=int(graph_logits.argmax()), eager_argmax=int(eager_logits.argmax()),
            graph_argmax_by_request=graph_logits.argmax(dim=-1).flatten().tolist(),
            eager_argmax_by_request=eager_logits.argmax(dim=-1).flatten().tolist())


def attach_squeezed_batch(model, index, threshold_percentile):
    original_position = model.position_embedd
    def init_cache(this, valid_start, config):
        if len(valid_start) != this.batch_size or any(int(x) != 0 for x in valid_start):
            raise ValueError('Squeezed batch adapter requires unpadded equal-length requests')
        this.kv_cache = RetroSqueezedBatchCache(this, index, threshold_percentile)
    def decode(this, q, k, v, layer_idx):
        return this.kv_cache.decode_attention(q, k, v, layer_idx)
    def position(this, q, k):
        if this.kv_cache.graph_mode:
            return this.apply_rotary_pos_emb(q, k, this.kv_cache.graph_position)
        return original_position(q, k)
    def move(this):
        torch.cuda.empty_cache()
    model.init_kv_cache = MethodType(init_cache, model)
    model.decode_attention = MethodType(decode, model)
    model.position_embedd = MethodType(position, model)
    model.move = MethodType(move, model)


class PackedSqueezedBatchCache(RetroSqueezedCache):
    """Run the released kernels over flattened request/head axes."""

    def __init__(self, model, index, percentile):
        self.batch = model.batch_size
        self.request_heads = model.num_heads
        self.request_kv_heads = model.num_key_value_heads
        # Flattened request-major heads have the same GQA ratio. Set up the
        # author's selector and sparse buffers once for the full batch.
        flat = SimpleNamespace(
            batch_size=1, num_gpus=model.num_gpus, dtype=model.dtype, device_map=model.device_map,
            input_length=model.input_length, max_new_length=model.max_new_length,
            num_layers=model.num_layers, num_heads=self.batch * model.num_heads,
            num_key_value_heads=self.batch * model.num_key_value_heads, head_dim=model.head_dim,
        )
        super().__init__(flat, index, percentile)
        self.graph_position = self.graph_position.expand(self.batch, 1).clone()
        if self.routing_position is not None:
            self.routing_position = self.routing_position.expand(self.batch, 1).clone()
        self.dynamic_len = self.dynamic_len.expand(self.batch).clone()
        shape = (self.batch, self.max_new, self.request_kv_heads, self.dim)
        for layer in range(self.layer_num):
            self.dynamic_k[layer] = torch.empty(shape, dtype=model.dtype, device=self.device)
            self.dynamic_v[layer] = torch.empty_like(self.dynamic_k[layer])

    def install_prefix(self, layers):
        for layer, (k, v, _, _) in enumerate(layers):
            shape = (self.batch, self.request_kv_heads, self.prefix, self.dim)
            self.fixed_k[layer] = k.to(self.device).expand(shape).contiguous().view(
                1, self.kv_heads, self.prefix, self.dim)
            self.fixed_v[layer] = v.to(self.device).expand(shape).contiguous().view_as(self.fixed_k[layer])
        self.context = self.prefix

    def sparse(self, q, k, v, out, m, l):
        return sparse_attention(
            q, k, v, out, m, l, self.indices, self.head_start, self.head_count,
            self.partial, self.partial_l, self.partial_m, self.scale,
            block_m=self.block_m, num_warps=self.num_warps, fixed_blocks_per_head=self.slots_per_head,
        )

    def decode_attention(self, q, k, v, layer_idx):
        self.dynamic_k[layer_idx].index_copy_(1, self.dynamic_index, k)
        self.dynamic_v[layer_idx].index_copy_(1, self.dynamic_index, v)
        head_shape = (1, self.heads, 1, self.dim)
        q_heads = q.reshape(head_shape)
        routing_q = self.current_window_q.reshape(head_shape) if self.routing_position is not None else q_heads
        self.select(routing_q, layer_idx)
        local, lse = flash_attn_with_kvcache(
            q, self.dynamic_k[layer_idx], self.dynamic_v[layer_idx],
            cache_seqlens=self.dynamic_len, return_softmax_lse=True,
        )
        out = local.reshape(head_shape).float().contiguous()
        m = (lse.reshape(1, self.heads, 1) * math.log2(math.e)).contiguous()
        return self.sparse(q_heads, self.fixed_k[layer_idx], self.fixed_v[layer_idx],
                           out, m, torch.ones_like(m)).reshape_as(q).to(q.dtype)

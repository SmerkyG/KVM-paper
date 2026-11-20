"""Run independent author QUEST controllers inside one batched model graph.

The released CUDA operators are single-request, so the requests' author kernel
calls run sequentially within each layer. This is a batching adaptation, not a
native-batched QUEST kernel, and is labeled as such in benchmark results.
"""
from types import SimpleNamespace, MethodType
import torch
from benchmarks.retro_quest_author_adapter import RetroAuthorQuestCache


class RetroAuthorQuestBatchCache:
    def __init__(self, model, budget, shared_kv):
        if model.num_gpus != 1 or model.dtype != torch.float16:
            raise ValueError('author QUEST requires FP16 on one GPU')
        self.batch = model.batch_size
        self.layer_num = model.num_layers
        self.context = 0
        self.prompt_length = model.input_length
        self.graph_mode = False
        self.graph_position = torch.zeros((self.batch, 1), dtype=torch.int64, device=model.device_map)
        single = SimpleNamespace(batch_size=1, num_gpus=model.num_gpus, dtype=model.dtype,
            input_length=model.input_length, max_new_length=model.max_new_length,
            num_layers=model.num_layers, num_heads=model.num_heads,
            num_key_value_heads=model.num_key_value_heads, head_dim=model.head_dim,
            device_map=model.device_map)
        self.requests = [RetroAuthorQuestCache(single, budget, shared_kv=shared_kv)
                         for _ in range(self.batch)]

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
            request.context = previous
            request._set_position(previous)

    def capture_decode(self, model, token):
        for request in self.requests:
            request._initialize_decode()
        previous = self.context
        self.graph_mode = True
        self.graph_input = token.clone()
        self._set_position(previous)
        for request in self.requests:
            request.phase = 1
        model.sampling(model.decode_forward(inputs_ids=self.graph_input))
        self.context = previous
        torch.cuda.synchronize()
        self.graphs, self.graph_logits_variants, self.graph_outputs = [], [], []
        pool = None
        for phase in range(1, 17):
            for request in self.requests:
                request.phase = phase
            self._set_position(previous)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                logits = model.decode_forward(inputs_ids=self.graph_input)
                output = model.sampling(logits)
            if pool is None:
                pool = graph.pool()
            self.graphs.append(graph)
            self.graph_logits_variants.append(logits)
            self.graph_outputs.append(output)
            self.context = previous

    def graph_step(self, token):
        if self.context > self.prompt_length and self.context % 16 == 0:
            for request in self.requests:
                request.context = self.context
                request._archive_completed_page()
        self._set_position(self.context)
        phase = self.context % 16 + 1
        self.graph_input.copy_(token)
        self.graphs[phase - 1].replay()
        self.context += 1
        self.graph_logits = self.graph_logits_variants[phase - 1]
        return self.graph_outputs[phase - 1]

    def checked_graph_step(self, model, token):
        previous = self.context
        graph_token = self.graph_step(token)
        graph_logits = self.graph_logits.clone()
        self.context = previous
        self._set_position(previous)
        phase = previous % 16 + 1
        for request in self.requests:
            request.phase = phase
        eager_logits = model.decode_forward(inputs_ids=token)
        difference = (graph_logits - eager_logits).abs()
        return graph_token, dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
            graph_argmax=int(graph_logits.argmax()), eager_argmax=int(eager_logits.argmax()),
            graph_argmax_by_request=graph_logits.argmax(dim=-1).flatten().tolist(),
            eager_argmax_by_request=eager_logits.argmax(dim=-1).flatten().tolist())


def attach_author_quest_batch(model, budget, shared_kv=False):
    original_position = model.position_embedd
    def init_cache(this, valid_start, config):
        if len(valid_start) != this.batch_size or any(int(x) != 0 for x in valid_start):
            raise ValueError('QUEST batch adapter requires unpadded equal-length requests')
        this.kv_cache = RetroAuthorQuestBatchCache(this, budget, shared_kv)
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

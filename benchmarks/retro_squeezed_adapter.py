"""Released Squeezed Attention kernels in RetroInfer's batch-one Llama.

Cluster metadata is precomputed offline and loaded before timed prefill. The
Triton lookup and sparse kernels share their inputs across GQA query heads.
Per-query-head thresholds and selected keys are retained. Decode uses fixed-size graph
buffers; the released sparse kernel's per-head block segments are padded to
fixed capacity, while its reduction still reads only the selected blocks.
"""
import math
import os
import types

import torch
import triton
from flash_attn import flash_attn_with_kvcache
import squeezed_shared_kernels as author


def centroid_scores(q, c, count, labels, scale, out, scores, threshold, *, block_m=16, num_warps=4):
    heads = q.size(1)
    author._fwd_centroid_simple_kernel_qk[(1, heads, 1)](
        q, c, count, labels, scale, out, scores, threshold,
        *q.stride(), *c.stride(), *out.stride(), *scores.stride(),
        1, heads, 1, labels.size(2), c.size(2),
        BLOCK_M=block_m, BLOCK_N=128, BLOCK_DMODEL=q.size(-1),
        GQA_GROUP=heads // c.size(1), num_warps=num_warps)
    return scores


def sparse_attention(q, k, v, out, m, l, indices, head_start, head_count, partial, partial_l, partial_m,
                     scale, *, block_m=16, num_warps=4, fixed_blocks_per_head=0):
    heads, dim = q.size(1), q.size(-1)
    index_group = heads // indices.size(1)
    author._fwd_kernel_qk[(1, partial.size(1), 1)](
        q, k, v, indices, scale, partial,
        *q.stride(), *k.stride(), *v.stride(), *indices.stride(), *partial.stride(),
        head_start, head_count, partial_l, partial_m,
        1, heads, 1, k.size(2), indices.size(2),
        NUMHEADS=heads, BLOCK_M=block_m, BLOCK_N=128, BLOCK_KV=2048, BLOCK_DMODEL=dim,
        GQA_GROUP=heads // k.size(1), GQA_INDEX_GROUP=index_group, num_warps=num_warps,
        **({'FIXED_BLOCKS_PER_HEAD': fixed_blocks_per_head} if fixed_blocks_per_head else {}))
    author._reduce_kernel_qk[(1, heads, 1)](
        partial, out, *partial.stride(), *out.stride(), head_start, head_count, partial_l, partial_m, l, m, 1,
        BLOCK_M=block_m, BLOCK_N=128, BLOCK_KV=2048, BLOCK_DMODEL=dim,
        GQA_INDEX_GROUP=index_group, num_warps=num_warps)
    return out


class RetroSqueezedCache:
    def __init__(self, model, index, threshold_percentile):
        if model.batch_size != 1 or model.num_gpus != 1 or model.dtype != torch.bfloat16:
            raise ValueError('Squeezed adapter requires batch-one single-GPU BF16')
        self.layer_num = model.num_layers
        self.heads = model.num_heads
        self.kv_heads = model.num_key_value_heads
        self.group = self.heads // self.kv_heads
        self.dim = model.head_dim
        self.device = torch.device(model.device_map)
        self.prefix = model.input_length
        self.max_new = model.max_new_length
        self.context = 0
        self.threshold = float(index['threshold'][threshold_percentile])
        self.obs = int(index['observation_window'])
        position = index.get('routing_position')
        self.routing_position = (torch.full((1, 1), position, dtype=torch.int64, device=self.device)
                                 if position is not None else None)
        self.graph_mode = False
        self.graph_position = torch.zeros((1, 1), dtype=torch.int64, device=self.device)
        self.dynamic_index = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.dynamic_len = torch.ones(1, dtype=torch.int32, device=self.device)
        self.fixed_k = [None] * self.layer_num
        self.fixed_v = [None] * self.layer_num
        self.dynamic_k = [torch.empty((1, self.max_new, self.kv_heads, self.dim), dtype=model.dtype, device=self.device)
                          for _ in range(self.layer_num)]
        self.dynamic_v = [torch.empty_like(k) for k in self.dynamic_k]
        self.centroids, self.labels, self.cluster_counts = [], [], []
        for layer in range(self.layer_num):
            centroids = index['centroids'][layer].to(device=self.device, dtype=model.dtype).contiguous()
            labels = index['labels'][layer].to(device=self.device, dtype=torch.int64).contiguous()
            index_heads = centroids.size(1) if centroids.ndim == 4 else 0
            # Older indexes and explicitly per-query-head indexes remain valid.
            if (centroids.ndim != 4 or centroids.shape[0] != 1 or centroids.shape[-1] != self.dim
                    or index_heads not in (self.kv_heads, self.heads)
                    or labels.shape != (1, index_heads, self.prefix - self.obs)):
                raise ValueError(f'index shape mismatch at layer {layer}: {centroids.shape}, {labels.shape}')
            k = centroids.shape[2]
            # The sentinel also permits exact sink tokens in a reused index.
            count = torch.zeros((1, index_heads, k + 1), dtype=torch.int32, device=self.device)
            count.scatter_add_(2, labels, torch.ones_like(labels, dtype=torch.int32))
            obs_labels = torch.full((1, index_heads, self.obs), k, dtype=torch.int64, device=self.device)
            self.centroids.append(centroids)
            self.labels.append(torch.cat((labels, obs_labels), dim=2))
            self.cluster_counts.append(count[:, :, :k].contiguous())
        self.clusters = self.centroids[0].shape[2]
        if any(x.shape[2] != self.clusters for x in self.centroids):
            raise ValueError('all layers must use the same cluster count')
        # The author's reduction reads only ceil(selected_count / 2048) slots.
        self.slots_per_head = triton.cdiv(self.prefix, 2048)
        total_slots = self.heads * self.slots_per_head
        self.head_start = torch.arange(self.heads, dtype=torch.int32, device=self.device) * self.slots_per_head
        self.head_count = torch.empty((1, self.heads), dtype=torch.int32, device=self.device)
        self.selection_counts = (torch.empty((self.layer_num, self.heads), dtype=torch.int32, device=self.device)
                                 if os.environ.get('SQUEEZED_REPORT_SELECTION') == '1' else None)
        self.indices = torch.empty((1, self.heads, self.prefix), dtype=torch.int64, device=self.device)
        self.stable_tiles = (torch.empty((self.heads, triton.cdiv(self.prefix, 4096)), dtype=torch.int32,
                                         device=self.device)
                             if os.environ.get('SQUEEZED_STABLE_PACK') == '1' else None)
        self.centroid_out = torch.empty((1, self.heads, self.clusters, 1), dtype=torch.float32, device=self.device)
        self.scores = torch.empty((1, self.heads, self.clusters + 1, 1), dtype=torch.float32, device=self.device)
        self.scores[:, :, -1, :].fill_(1.0)
        self.partial = torch.empty((1, total_slots, 1, self.dim), dtype=model.dtype, device=self.device)
        self.partial_l = torch.empty((total_slots, 1), dtype=torch.float32, device=self.device)
        self.partial_m = torch.empty_like(self.partial_l)
        self.scale = 1.0 / math.sqrt(self.dim)
        # Decode has one query; a 16-row launch avoids the authors' 128-row
        # prefill-oriented tile while retaining their kernel computations.
        self.block_m = int(os.environ.get('SQUEEZED_BLOCK_M', '16'))
        self.num_warps = int(os.environ.get('SQUEEZED_NUM_WARPS', '4'))
        if self.block_m not in (16, 32, 64, 128) or self.num_warps not in (4, 8):
            raise ValueError('unsupported Squeezed Triton launch geometry')

    def prefill_update_kv_cache(self, q, k, v, layer_idx, start_bdx):
        if start_bdx != 0 or k.shape[0] != 1 or k.shape[1] != self.prefix:
            raise ValueError('Squeezed adapter requires one unpadded, complete prefix')
        fixed_k, fixed_v = k.transpose(1, 2), v.transpose(1, 2)
        if self.routing_position is not None:
            # The graph keeps its fixed prefix in the worker's own allocations.
            fixed_k = fixed_k.clone(memory_format=torch.contiguous_format)
            fixed_v = fixed_v.clone(memory_format=torch.contiguous_format)
        self.fixed_k[layer_idx] = fixed_k.contiguous()
        self.fixed_v[layer_idx] = fixed_v.contiguous()
        if layer_idx == self.layer_num - 1:
            self.context += k.shape[1]
        return k, v

    def sync(self, layer_idx, start_bdx):
        pass

    def decode_update_kv_cache(self, k, v, layer_idx):
        if layer_idx == self.layer_num - 1:
            self.context += 1
        return k, v

    def select(self, q, layer_idx):
        c = self.centroids[layer_idx]
        labels = self.labels[layer_idx]
        count = self.cluster_counts[layer_idx]
        index_heads = c.shape[1]
        lookup_group = self.heads // index_heads
        centroid_scores(q, c, count, labels, self.scale, self.centroid_out, self.scores, self.threshold,
                        block_m=self.block_m, num_warps=self.num_warps)
        scores = self.scores.view(1, index_heads, lookup_group, self.clusters + 1)
        labels_by_query = labels.unsqueeze(2).expand(-1, -1, lookup_group, -1)
        mask = (scores.gather(3, labels_by_query) > self.threshold).view(1, self.heads, self.prefix)
        if self.selection_counts is not None:
            self.head_count = self.selection_counts[layer_idx:layer_idx + 1]
        if self.stable_tiles is not None:
            from benchmarks.squeezed_stable_pack import compact
            compact(mask, self.indices, self.head_count, self.stable_tiles)
        else:
            self.head_count.zero_()
            author.compact_indices_kernel[(1, self.heads, triton.cdiv(self.prefix, 4096))](
                mask, self.indices, self.head_count, self.prefix, self.heads,
                BLOCK_SIZE=4096, num_warps=4)
        return mask

    def sparse(self, q, k, v, out, m, l):
        return sparse_attention(q, k, v, out, m, l, self.indices, self.head_start, self.head_count,
                                self.partial, self.partial_l, self.partial_m, self.scale,
                                block_m=self.block_m, num_warps=self.num_warps)

    def decode_attention(self, q, k, v, layer_idx):
        self.dynamic_k[layer_idx].index_copy_(1, self.dynamic_index, k)
        self.dynamic_v[layer_idx].index_copy_(1, self.dynamic_index, v)
        q_heads = q.transpose(1, 2).contiguous()
        routing_q = self.current_window_q.view_as(q).transpose(1, 2).contiguous() if self.routing_position is not None else q_heads
        self.select(routing_q, layer_idx)
        # The dynamic generated-token field remains exact and uses native GQA.
        local, lse = flash_attn_with_kvcache(q, self.dynamic_k[layer_idx], self.dynamic_v[layer_idx],
                                             cache_seqlens=self.dynamic_len, return_softmax_lse=True)
        out = local.transpose(1, 2).float().contiguous()
        # The author's merge uses base-2 max and sum; a normalized local result
        # can be represented by (M=log2(Z), L=1).
        m = (lse * math.log2(math.e)).contiguous()
        l = torch.ones_like(m)
        result = self.sparse(q_heads, self.fixed_k[layer_idx], self.fixed_v[layer_idx], out, m, l)
        return result.transpose(1, 2).to(q.dtype)

    def _set_position(self, previous):
        offset = previous - self.prefix
        self.graph_position.fill_(previous)
        self.dynamic_index.fill_(offset)
        self.dynamic_len.fill_(offset + 1)

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
            graph_argmax=int(graph_logits.argmax()), eager_argmax=int(eager_logits.argmax()))


def attach_squeezed(model, index, threshold_percentile, *, packed_batch=False):
    original_position = model.position_embedd
    cache_type = RetroSqueezedCache
    if packed_batch:
        from benchmarks.retro_squeezed_batch_adapter import PackedSqueezedBatchCache
        cache_type = PackedSqueezedBatchCache

    def init_cache(this, valid_start, config):
        if len(valid_start) != this.batch_size or any(int(start) != 0 for start in valid_start):
            raise ValueError('Squeezed adapter requires unpadded equal-length requests')
        this.kv_cache = None
        this.kv_cache = cache_type(this, index, threshold_percentile)

    def decode(this, q, k, v, layer_idx):
        return this.kv_cache.decode_attention(q, k, v, layer_idx)

    def position(this, q, k):
        cache = this.kv_cache
        if cache.graph_mode:
            if cache.routing_position is not None:
                cache.current_window_q, _ = this.apply_rotary_pos_emb(q.clone(), k.clone(), cache.routing_position)
            return this.apply_rotary_pos_emb(q, k, cache.graph_position)
        return original_position(q, k)

    def move(this):
        torch.cuda.empty_cache()

    model.init_kv_cache = types.MethodType(init_cache, model)
    model.decode_attention = types.MethodType(decode, model)
    model.position_embedd = types.MethodType(position, model)
    model.move = types.MethodType(move, model)

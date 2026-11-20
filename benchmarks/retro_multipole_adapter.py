"""Multipole Attention in RetroInfer's Llama runtime, batch-one BF16.

Uses the released single-level kernels and blockwise online k-means.  Fixed
capacity buffers make the whole model graph replayable as clusters are updated.
The selection policy and windowed RoPE follow the authors' Qwen integration.
"""
import math
import os
import time
import types

import torch
import triton
from flash_attn import flash_attn_with_kvcache
from multipoleattention import kernels as author
from multipoleattention.clustering import run_clustering_online, run_clustering_online_update
from multipoleattention.kernel_wrappers import centroid_replacement_kernel
from multipoleattention.kmeans_ops_sequential import block_kmeans, sequential_kmeans_append


class RetroMultipoleCache:
    def __init__(self, model, *, budget=512, replacement=True, interval=128, tokens_per_centroid=16,
                 fixed_centroids=None, squeezed_kernels=False):
        if model.batch_size != 1 or model.num_gpus != 1 or model.dtype != torch.bfloat16:
            raise ValueError('Multipole requires batch-one single-GPU BF16')
        self.model = model
        self.layer_num = model.num_layers
        self.heads = model.num_heads
        self.kv_heads = model.num_key_value_heads
        self.group = self.heads // self.kv_heads
        self.dim = model.head_dim
        self.device = torch.device(model.device_map)
        self.prefix = model.input_length
        self.capacity = model.input_length + model.max_new_length
        self.interval = interval
        if tokens_per_centroid not in (8,16):raise ValueError('Multipole harness supports 1:8 or 1:16 centroids')
        self.tokens_per_centroid=tokens_per_centroid
        self.sink = 10
        self.context = 0
        self.clustered = ((self.prefix - self.sink) // interval - 1) * interval
        if self.clustered <= 0 or interval % 16:
            raise ValueError('prefix must exceed two intervals; interval must be divisible by 16')
        self.fixed_centroids = fixed_centroids
        if fixed_centroids is None:
            self.max_clusters = (self.capacity - self.sink) // tokens_per_centroid
        else:
            self.max_clusters = fixed_centroids
            self.fixed_blocks = self._fixed_block_layout(self.clustered)
        if budget <= 0:
            raise ValueError("Multipole token budget must be positive")
        self.budget = budget
        self.replacement = replacement
        if squeezed_kernels and replacement:
            raise ValueError('Squeezed kernels support the exact-only subset')
        self.squeezed_kernels = squeezed_kernels
        self.profile_decode = False
        self.scale = self.dim ** -0.5
        self.graph_mode = False
        self.graph_position = torch.zeros((1, 1), dtype=torch.int64, device=self.device)
        self.fixed_position = torch.full((1, 1), 2048, dtype=torch.int64, device=self.device)
        self.all_index = torch.zeros(1, dtype=torch.long, device=self.device)
        self.local_index = torch.zeros(1, dtype=torch.long, device=self.device)
        self.local_len = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.active_length = torch.tensor(self.clustered, dtype=torch.int32, device=self.device)
        self.raw_k, self.k, self.v, self.local_k, self.local_v = [], [], [], [], []
        self.centroids, self.value_centroids, self.counts, self.labels = [], [], [], []
        self.local_capacity = self.sink + 2 * interval + 1
        for _ in range(self.layer_num):
            shape = (1, self.kv_heads, self.capacity, self.dim)
            self.raw_k.append(torch.empty(shape, dtype=model.dtype, device=self.device))
            self.k.append(torch.empty(shape, dtype=model.dtype, device=self.device))
            self.v.append(torch.empty(shape, dtype=model.dtype, device=self.device))
            local_shape = (1, self.local_capacity, self.kv_heads, self.dim)
            self.local_k.append(torch.empty(local_shape, dtype=model.dtype, device=self.device))
            self.local_v.append(torch.empty(local_shape, dtype=model.dtype, device=self.device))
            self.centroids.append(torch.zeros((1, self.kv_heads, self.max_clusters, self.dim), dtype=model.dtype, device=self.device))
            self.value_centroids.append(
                torch.zeros((1, self.kv_heads, self.max_clusters, self.dim), dtype=torch.float32, device=self.device)
                if self.replacement else None)
            self.counts.append(torch.zeros((1, self.kv_heads, self.max_clusters), dtype=torch.int64, device=self.device))
            self.labels.append(torch.zeros((1, self.kv_heads, self.capacity - self.sink), dtype=torch.int64, device=self.device))
        self.valid_positions = torch.arange(self.capacity - self.sink, device=self.device)
        # The authors' strict whole-cluster cutoff selects at most budget
        # tokens per KV head. Reserve only that many sparse blocks; the
        # unused slots are still masked by each head's actual count.
        self.slots_per_head = max(1, triton.cdiv(min(budget, self.capacity - self.sink), 2048))
        sparse_heads, sparse_rows = (self.heads, 1) if squeezed_kernels else (self.kv_heads, self.group)
        self.head_start = torch.arange(sparse_heads, dtype=torch.int32, device=self.device) * self.slots_per_head
        self.head_count = torch.empty((1, self.kv_heads), dtype=torch.int32, device=self.device)
        self.indices = torch.empty((1, self.kv_heads, self.capacity-self.sink), dtype=torch.int64, device=self.device)
        slots = sparse_heads * self.slots_per_head
        self.partial = torch.empty((1, slots, sparse_rows, self.dim), dtype=model.dtype, device=self.device)
        self.partial_l = torch.empty((slots, sparse_rows), dtype=torch.float32, device=self.device)
        self.partial_m = torch.empty_like(self.partial_l)
        self.cluster_state = [None] * self.layer_num
        self.update_count = 0
        self.update_times = []

    @staticmethod
    def _final_block_start(length):
        # The author's online path keeps a final window of up to 12,288 tokens.
        return max(0, (length - 4096) // 8192) * 8192

    def _fixed_block_layout(self, length):
        boundaries = [*range(0, self._final_block_start(length) + 1, 8192), length]
        blocks = [(start, end, start * self.fixed_centroids // length, end * self.fixed_centroids // length)
                  for start, end in zip(boundaries, boundaries[1:])]
        if any(not 0 < hi - lo <= end - start for start, end, lo, hi in blocks):
            raise ValueError('fixed centroid count must allocate between one centroid and one per token in each block')
        return blocks

    def _cluster_fixed(self, data):
        centers, labels, counts = [], [], []
        for start, end, lo, hi in self.fixed_blocks:
            center, count, label = block_kmeans(data[0, :, start:end], hi - lo)
            centers.append(center)
            labels.append(label + lo)
            counts.append(count)
        return tuple(torch.cat(parts, dim=1).unsqueeze(0) for parts in (centers, labels, counts))

    def _refine_fixed(self, data, center, labels, counts):
        start, _, lo, _ = self.fixed_blocks[-1]
        tail, count, label = sequential_kmeans_append(
            data_new=data[0, :, -self.interval:], centroids_old=center[0, :, lo:],
            counts_old=counts[0, :, lo:], refine_data=data[0, :, start:], k_extra=0,
        )
        return (torch.cat((center[:, :, :lo], tail.unsqueeze(0)), dim=2),
                torch.cat((labels[:, :, :start], (label + lo).unsqueeze(0)), dim=2),
                torch.cat((counts[:, :, :lo], count.unsqueeze(0)), dim=2))

    def _install_clusters(self, layer, center, labels, author_counts):
        # K-means returns labels from a final assignment after its returned
        # counts were computed. Recompute counts from those labels so exact and
        # approximate regions partition the cache even when assignments move.
        count = torch.zeros_like(author_counts, dtype=torch.int64)
        count.scatter_add_(2, labels, torch.ones_like(labels))
        n = center.shape[2]
        self.centroids[layer].zero_(); self.centroids[layer][:, :, :n].copy_(center)
        if self.replacement:
            values = self.v[layer][:, :, self.sink:self.sink+self.clustered]
            sums = torch.zeros_like(center, dtype=torch.float32)
            sums.scatter_add_(2, labels.unsqueeze(-1).expand_as(values), values.float())
            value_center = sums / count.clamp_min(1).unsqueeze(-1)
            self.value_centroids[layer].zero_(); self.value_centroids[layer][:, :, :n].copy_(value_center)
        self.counts[layer].zero_(); self.counts[layer][:, :, :n].copy_(count)
        self.labels[layer].zero_(); self.labels[layer][:, :, :self.clustered].copy_(labels)
        self.cluster_state[layer] = (center, labels, count)

    def prefill_update_kv_cache(self, q, k, v, layer_idx, start_bdx, *, cluster_state=None):
        if start_bdx != 0 or k.shape[1] != self.prefix:
            raise ValueError('requires a complete unpadded prefix')
        self.raw_k[layer_idx][:, :, :self.prefix].copy_(self.current_raw_k.view(1, self.prefix, self.kv_heads, self.dim).transpose(1, 2))
        self.k[layer_idx][:, :, :self.prefix].copy_(k.transpose(1, 2))
        self.v[layer_idx][:, :, :self.prefix].copy_(v.transpose(1, 2))
        tail = self.sink + self.clustered
        local_k = torch.cat((k[:, :self.sink], k[:, tail:]), dim=1)
        local_v = torch.cat((v[:, :self.sink], v[:, tail:]), dim=1)
        self.local_k[layer_idx][:, :local_k.shape[1]].copy_(local_k)
        self.local_v[layer_idx][:, :local_v.shape[1]].copy_(local_v)
        data = self.raw_k[layer_idx][:, :, self.sink:tail].contiguous()
        if cluster_state is not None:
            state = cluster_state
        elif self.fixed_centroids is None:
            centers, labels, counts, _ = run_clustering_online(
                data, [self.clustered // self.tokens_per_centroid], 1, rank=0)
            state = centers[0], labels[0], counts[0]
        else:
            state = self._cluster_fixed(data)
        self._install_clusters(layer_idx, *state)
        if layer_idx == self.layer_num - 1:
            self.context += k.shape[1]
        return k, v

    def sync(self, layer_idx, start_bdx):
        pass

    def decode_update_kv_cache(self, k, v, layer_idx):
        if layer_idx == self.layer_num - 1:
            self.context += 1
        return k, v

    def _centroid_scores(self, q, c, labels, counts, *, squeezed=None):
        squeezed = self.squeezed_kernels if squeezed is None else squeezed
        n = c.size(2)
        if squeezed:
            from benchmarks.retro_squeezed_adapter import centroid_scores

            scores = torch.empty((1, self.heads, n, 1), device=self.device, dtype=torch.float32)
            centroid_scores(q.view(1, self.heads, 1, self.dim), c, counts, labels,
                            self.scale, scores, scores, 0.0)
            score = scores.view(1, self.kv_heads, self.group, n).sum(dim=2)
            qk = None
        elif os.environ.get('MULTIPOLE_SPLIT_LOOKUP') == '1':
            from benchmarks.multipole_split_lookup import split_centroid_scores
            score, qk = split_centroid_scores(q, c, counts, labels, self.scale)
        else:
            out = torch.empty((1, self.kv_heads, n, self.group), device=self.device, dtype=torch.float32)
            score = torch.empty((1, self.kv_heads, n, 1), device=self.device, dtype=torch.float32)
            qk = torch.zeros((1, self.kv_heads, self.group, n), device=self.device, dtype=torch.bfloat16)
            author._fwd_centroid_simple_kernel_qk[(1, self.kv_heads, 1)](
                q, c, qk, counts, labels, self.scale, out, score,
                *q.stride(), *c.stride(), *out.stride(), *score.stride(), *qk.stride(),
                1, self.kv_heads, self.group, labels.shape[2], n,
                BLOCK_M=16, BLOCK_N=128, BLOCK_DMODEL=self.dim, num_warps=4)
            score = score.squeeze(-1)
        return score.masked_fill(counts == 0, -float('inf')), qk

    def _select(self, score, labels, counts):
        sorted_score, order = score.sort(dim=2, descending=True)
        cumulative = counts.gather(2, order).cumsum(dim=2)
        exceed = cumulative > self.budget
        first = exceed.to(torch.int32).argmax(dim=2)
        first = torch.where(exceed.any(dim=2), first, torch.full_like(first, self.max_clusters - 1))
        threshold = sorted_score.gather(2, first.unsqueeze(-1))
        selected = (score > threshold) & (counts > 0)
        token_mask = selected.gather(2, labels) & (self.valid_positions < self.active_length)
        self.head_count.zero_()
        author.compact_indices_kernel[(1, self.kv_heads, triton.cdiv(labels.shape[2], 4096))](
            token_mask, self.indices, self.head_count, labels.shape[2], self.kv_heads,
            BLOCK_SIZE=4096, num_warps=4)
        if os.environ.get('MULTIPOLE_STABLE_PACK')=='1':
            # Quality-check option: preserve the selected set, but stabilize the
            # authors' atomic compaction order before BF16 sparse reductions.
            width=min(self.budget,self.indices.shape[2])
            lane=torch.arange(width,device=self.device)[None,None,:]
            packed=torch.where(lane<self.head_count.unsqueeze(-1),self.indices[:,:,:width],self.indices.shape[2])
            self.indices[:,:,:width].copy_(packed.sort(dim=-1).values)
        return selected

    def _lookup(self, q, layer):
        c, labels, counts = self.centroids[layer], self.labels[layer], self.counts[layer]
        score, qk = self._centroid_scores(q, c, labels, counts)
        return self._select(score, labels, counts), qk

    def _sparse(self, q, layer, out, m, l):
        k = self.k[layer][:, :, self.sink:]
        v = self.v[layer][:, :, self.sink:]
        if self.squeezed_kernels:
            from benchmarks.retro_squeezed_adapter import sparse_attention

            shape = (1, self.heads, 1, self.dim)
            sparse_attention(q.view(shape), k, v, out.view(shape), m, l, self.indices,
                             self.head_start, self.head_count, self.partial, self.partial_l,
                             self.partial_m, self.scale)
            return out
        slots = self.kv_heads * self.slots_per_head
        author._fwd_kernel_qk_gen[(1, slots)](
            q, k, v, self.indices, self.scale, self.partial,
            *q.stride(), *k.stride(), *v.stride(), *self.indices.stride(), *self.partial.stride(),
            self.head_start, self.head_count, self.partial_l, self.partial_m,
            1, self.kv_heads, self.group, k.shape[2], self.indices.shape[2], self.kv_heads,
            NUMHEADS=self.kv_heads, BLOCK_M=16, BLOCK_N=128, BLOCK_KV=2048,
            BLOCK_DMODEL=self.dim, num_warps=4)
        author._reduce_kernel_qk_gen[(1, self.kv_heads, 1)](
            self.partial, out, *self.partial.stride(), *out.stride(), self.head_start,
            self.head_count, self.partial_l, self.partial_m, l, m, self.group, self.kv_heads,
            BLOCK_M=16, BLOCK_N=128, BLOCK_KV=2048, BLOCK_DMODEL=self.dim, num_warps=4)
        return out

    def decode_attention(self, q, k, v, layer_idx):
        raw = self.current_raw_k.view(1, 1, self.kv_heads, self.dim).transpose(1, 2)
        self.raw_k[layer_idx].index_copy_(2, self.all_index, raw)
        self.k[layer_idx].index_copy_(2, self.all_index, k.transpose(1, 2))
        self.v[layer_idx].index_copy_(2, self.all_index, v.transpose(1, 2))
        self.local_k[layer_idx].index_copy_(1, self.local_index, k)
        self.local_v[layer_idx].index_copy_(1, self.local_index, v)
        q_window = self.current_window_q.view(1, self.kv_heads, self.group, self.dim)
        selected, qk = self._lookup(q_window, layer_idx)
        local, lse = flash_attn_with_kvcache(q, self.local_k[layer_idx], self.local_v[layer_idx],
                                              cache_seqlens=self.local_len, return_softmax_lse=True)
        out = local.view(1, self.kv_heads, self.group, self.dim).float().contiguous()
        m = (lse.view(1, self.kv_heads, self.group) * math.log2(math.e)).contiguous()
        l = torch.ones_like(m)
        if self.replacement:
            out, m, l = centroid_replacement_kernel(q_window, self.value_centroids[layer_idx],
                ~selected, self.counts[layer_idx], self.scale, qk.float(), out, m, l)
        q_full = q.view(1, self.kv_heads, self.group, self.dim)
        if (self.profile_decode and layer_idx == self.layer_num - 1
                and torch.cuda.is_current_stream_capturing()):
            self.profile_inputs = (q_full, q_window, local, lse)
        return self._sparse(q_full, layer_idx, out, m, l).view_as(q).to(q.dtype)

    def _update_clusters(self):
        profiling = os.environ.get('MULTIPOLE_PROFILE_UPDATES') == '1'
        if profiling:
            torch.cuda.synchronize()
            update_start = time.perf_counter()
        previous = self.clustered
        self.clustered += self.interval
        rebuild_fixed = (self.fixed_centroids is not None
                         and self._final_block_start(self.clustered) != self.fixed_blocks[-1][0])
        if rebuild_fixed:
            self.fixed_blocks = self._fixed_block_layout(self.clustered)
        for layer in range(self.layer_num):
            center, labels, counts = self.cluster_state[layer]
            data = self.raw_k[layer][:, :, self.sink:self.sink+self.clustered].contiguous()
            if self.fixed_centroids is None:
                centers, label_list, count_list, _ = run_clustering_online_update(
                    data, [center], [labels], [counts], [self.interval // self.tokens_per_centroid],
                    [self.interval], 1, rank=0)
                state = centers[0], label_list[0], count_list[0]
            elif rebuild_fixed:
                state = self._cluster_fixed(data)
            else:
                state = self._refine_fixed(data, center, labels, counts)
            self._install_clusters(layer, *state)
            old_len = self.context - self.sink - previous
            # Keep the sink and shift the remaining local window into place.
            self.local_k[layer][:, self.sink:self.sink+old_len-self.interval].copy_(
                self.local_k[layer][:, self.sink+self.interval:self.sink+old_len].clone())
            self.local_v[layer][:, self.sink:self.sink+old_len-self.interval].copy_(
                self.local_v[layer][:, self.sink+self.interval:self.sink+old_len].clone())
        self.active_length.fill_(self.clustered)
        self.update_count += 1
        if profiling:
            torch.cuda.synchronize()
            self.update_times.append(time.perf_counter() - update_start)

    def _set_position(self, previous):
        self.graph_position.fill_(previous)
        self.all_index.fill_(previous)
        local_position = previous - self.clustered
        self.local_index.fill_(local_position)
        self.local_len.fill_(local_position + 1)

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
        if self.context + 1 - self.sink - self.clustered >= 2 * self.interval:
            self._update_clusters()
        self._set_position(self.context)
        self.graph_input.copy_(token)
        self.graph.replay()
        self.context += 1
        return self.graph_output

    def checked_graph_step(self, model, token):
        # Update first, so graph and eager see the identical index.
        if self.context + 1 - self.sink - self.clustered >= 2 * self.interval:
            self._update_clusters()
        previous = self.context
        graph_token = self.graph_step(token)
        graph_logits = self.graph_logits.clone()
        self.context = previous
        self._set_position(previous)
        eager_logits = model.decode_forward(inputs_ids=token)
        difference = (graph_logits - eager_logits).abs()
        stats = dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                     graph_argmax=int(graph_logits.argmax()), eager_argmax=int(eager_logits.argmax()))
        if os.environ.get('MULTIPOLE_CHECK_REPEAT') == '1':
            self.context = previous
            self._set_position(previous)
            repeated_logits = model.decode_forward(inputs_ids=token)
            repeat_diff = (eager_logits - repeated_logits).abs()
            stats.update(eager_repeat_max_abs=float(repeat_diff.max()),
                         eager_repeat_mean_abs=float(repeat_diff.mean()),
                         eager_repeat_argmax=int(repeated_logits.argmax()))
        return graph_token, stats


def attach_multipole(model, *, budget=512, replacement=True, tokens_per_centroid=16,
                     fixed_centroids=None, squeezed_kernels=False):
    original_position = model.position_embedd

    def init_cache(this, valid_start, config):
        if len(valid_start) != 1 or int(valid_start[0]) != 0:
            raise ValueError('requires one unpadded request')
        this.kv_cache = None
        this.kv_cache = RetroMultipoleCache(this, budget=budget, replacement=replacement,
                                           tokens_per_centroid=tokens_per_centroid, fixed_centroids=fixed_centroids,
                                           squeezed_kernels=squeezed_kernels)

    def decode(this, q, k, v, layer_idx):
        return this.kv_cache.decode_attention(q, k, v, layer_idx)

    def position(this, q, k):
        cache = this.kv_cache
        cache.current_raw_k = k.clone()
        if cache.graph_mode:
            # Authors cluster unrotated keys and route using query at position 2048.
            cache.current_window_q, _ = this.apply_rotary_pos_emb(q.clone(), k.clone(), cache.fixed_position)
            return this.apply_rotary_pos_emb(q, k, cache.graph_position)
        return original_position(q, k)

    def move(this):
        torch.cuda.empty_cache()

    model.init_kv_cache = types.MethodType(init_cache, model)
    model.decode_attention = types.MethodType(decode, model)
    model.position_embedd = types.MethodType(position, model)
    model.move = types.MethodType(move, model)

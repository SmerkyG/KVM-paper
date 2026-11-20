"""Multipole Attention in RetroInfer's Llama runtime, synchronous equal-length batches, BF16.

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
from benchmarks.retro_multipole_adapter import RetroMultipoleCache


class RetroMultipoleBatchCache:
    _final_block_start = staticmethod(RetroMultipoleCache._final_block_start)
    _fixed_block_layout = RetroMultipoleCache._fixed_block_layout
    _cluster_fixed = RetroMultipoleCache._cluster_fixed
    _refine_fixed = RetroMultipoleCache._refine_fixed
    _select = RetroMultipoleCache._select

    def __init__(self, model, *, budget=512, replacement=True, interval=128, fixed_centroids=None):
        if model.batch_size < 1 or model.num_gpus != 1 or model.dtype != torch.bfloat16:
            raise ValueError('Multipole requires a single-GPU BF16 batch')
        self.batch = model.batch_size
        self.request_kv_heads = model.num_key_value_heads
        self.model = model
        self.layer_num = model.num_layers
        self.heads = self.batch * model.num_heads
        self.kv_heads = self.batch * model.num_key_value_heads
        self.group = self.heads // self.kv_heads
        self.dim = model.head_dim
        self.device = torch.device(model.device_map)
        self.prefix = model.input_length
        self.capacity = model.input_length + model.max_new_length
        self.interval = interval
        self.sink = 10
        self.context = 0
        self.clustered = ((self.prefix - self.sink) // interval - 1) * interval
        if self.clustered <= 0 or interval % 16:
            raise ValueError('prefix must exceed two intervals; interval must be divisible by 16')
        self.fixed_centroids = fixed_centroids
        self.max_clusters = fixed_centroids if fixed_centroids is not None else (self.capacity - self.sink) // 16
        if fixed_centroids is not None:
            self.fixed_blocks = self._fixed_block_layout(self.clustered)
        if budget <= 0:
            raise ValueError("Multipole token budget must be positive")
        self.budget = budget
        self.replacement = replacement
        self.scale = self.dim ** -0.5
        self.graph_mode = False
        self.graph_position = torch.zeros((self.batch, 1), dtype=torch.int64, device=self.device)
        self.fixed_position = torch.full((self.batch, 1), 2048, dtype=torch.int64, device=self.device)
        self.all_index = torch.zeros(1, dtype=torch.long, device=self.device)
        self.local_index = torch.zeros(1, dtype=torch.long, device=self.device)
        self.local_len = torch.zeros(self.batch, dtype=torch.int32, device=self.device)
        self.active_length = torch.tensor(self.clustered, dtype=torch.int32, device=self.device)
        self.raw_k, self.k, self.v, self.local_k, self.local_v = [], [], [], [], []
        self.centroids, self.value_centroids, self.counts, self.labels = [], [], [], []
        self.local_capacity = self.sink + 2 * interval + 1
        for _ in range(self.layer_num):
            shape = (1, self.kv_heads, self.capacity, self.dim)
            self.raw_k.append(torch.empty(shape, dtype=model.dtype, device=self.device))
            self.k.append(torch.empty(shape, dtype=model.dtype, device=self.device))
            self.v.append(torch.empty(shape, dtype=model.dtype, device=self.device))
            local_shape = (self.batch, self.local_capacity, self.request_kv_heads, self.dim)
            self.local_k.append(torch.empty(local_shape, dtype=model.dtype, device=self.device))
            self.local_v.append(torch.empty(local_shape, dtype=model.dtype, device=self.device))
            self.centroids.append(torch.zeros((1, self.kv_heads, self.max_clusters, self.dim), dtype=model.dtype, device=self.device))
            self.value_centroids.append(torch.zeros((1, self.kv_heads, self.max_clusters, self.dim), dtype=torch.float32, device=self.device))
            self.counts.append(torch.zeros((1, self.kv_heads, self.max_clusters), dtype=torch.int64, device=self.device))
            self.labels.append(torch.zeros((1, self.kv_heads, self.capacity - self.sink), dtype=torch.int64, device=self.device))
        self.valid_positions = torch.arange(self.capacity - self.sink, device=self.device)
        # The authors' strict whole-cluster cutoff selects at most budget
        # tokens per KV head. Reserve only that many sparse blocks; the
        # unused slots are still masked by each head's actual count.
        self.slots_per_head = max(1, triton.cdiv(min(budget, self.capacity - self.sink), 2048))
        self.head_start = torch.arange(self.kv_heads, dtype=torch.int32, device=self.device) * self.slots_per_head
        self.head_count = torch.empty((1, self.kv_heads), dtype=torch.int32, device=self.device)
        self.indices = torch.empty((1, self.kv_heads, self.capacity-self.sink), dtype=torch.int64, device=self.device)
        slots = self.kv_heads * self.slots_per_head
        self.partial = torch.empty((1, slots, self.group, self.dim), dtype=model.dtype, device=self.device)
        self.partial_l = torch.empty((slots, self.group), dtype=torch.float32, device=self.device)
        self.partial_m = torch.empty_like(self.partial_l)
        self.cluster_state = [None] * self.layer_num
        self.update_count = 0
        self.update_times = []
        self.update_stage_times = []

    def _install_clusters(self, layer, center, labels, author_counts, head_start=0):
        # K-means returns labels from a final assignment after its returned
        # counts were computed. Recompute counts from those labels so exact and
        # approximate regions partition the cache even when assignments move.
        count = torch.zeros_like(author_counts, dtype=torch.int64)
        count.scatter_add_(2, labels, torch.ones_like(labels))
        head_end = head_start + center.shape[1]
        values = self.v[layer][:, head_start:head_end, self.sink:self.sink+self.clustered]
        sums = torch.zeros_like(center, dtype=torch.float32)
        sums.scatter_add_(2, labels.unsqueeze(-1).expand_as(values), values.float())
        value_center = sums / count.clamp_min(1).unsqueeze(-1)
        n = center.shape[2]
        self.centroids[layer][:, head_start:head_end].zero_()
        self.centroids[layer][:, head_start:head_end, :n].copy_(center)
        self.value_centroids[layer][:, head_start:head_end].zero_()
        self.value_centroids[layer][:, head_start:head_end, :n].copy_(value_center)
        self.counts[layer][:, head_start:head_end].zero_()
        self.counts[layer][:, head_start:head_end, :n].copy_(count)
        self.labels[layer][:, head_start:head_end].zero_()
        self.labels[layer][:, head_start:head_end, :self.clustered].copy_(labels)

    def prefill_update_kv_cache(self, q, k, v, layer_idx, start_bdx):
        if k.shape[0] != 1 or k.shape[1] != self.prefix or not 0 <= start_bdx < self.batch:
            raise ValueError('requires unpadded full prefixes, prefilled one request at a time')
        first = start_bdx * self.request_kv_heads
        last = first + self.request_kv_heads
        self.raw_k[layer_idx][:, first:last, :self.prefix].copy_(self.current_raw_k.view(1, self.prefix, self.request_kv_heads, self.dim).transpose(1, 2))
        self.k[layer_idx][:, first:last, :self.prefix].copy_(k.transpose(1, 2))
        self.v[layer_idx][:, first:last, :self.prefix].copy_(v.transpose(1, 2))
        tail = self.sink + self.clustered
        local_k = torch.cat((k[:, :self.sink], k[:, tail:]), dim=1)
        local_v = torch.cat((v[:, :self.sink], v[:, tail:]), dim=1)
        self.local_k[layer_idx][start_bdx:start_bdx+1, :local_k.shape[1]].copy_(local_k)
        self.local_v[layer_idx][start_bdx:start_bdx+1, :local_v.shape[1]].copy_(local_v)
        data = self.raw_k[layer_idx][:, first:last, self.sink:tail].contiguous()
        if self.fixed_centroids is None:
            centers, labels, counts, _ = run_clustering_online(data, [self.clustered // 16], 1, rank=0)
            state = centers[0], labels[0], counts[0]
        else:
            state = self._cluster_fixed(data)
        self._install_clusters(layer_idx, *state, head_start=first)
        if layer_idx == self.layer_num - 1 and start_bdx + 1 == self.batch:
            self.context += k.shape[1]
        return k, v

    def install_repeated_prefix(self, layers):
        """Build the common initial index once, then copy to independent requests."""
        if self.fixed_centroids is None:
            raise ValueError('repeated-prefix installation requires a fixed centroid budget')
        tail = self.sink + self.clustered
        for layer, (k, v, _, raw) in enumerate(layers):
            k, v, raw = k.to(self.device), v.to(self.device), raw.to(self.device)
            head_shape = (self.batch, self.request_kv_heads, self.capacity, self.dim)
            for destination, prefix in ((self.raw_k[layer], raw), (self.k[layer], k), (self.v[layer], v)):
                destination.view(head_shape)[:, :, :self.prefix].copy_(prefix)
            self.local_k[layer][:, :self.prefix-self.clustered].copy_(torch.cat(
                (k[:, :, :self.sink], k[:, :, tail:]), dim=2).transpose(1, 2))
            self.local_v[layer][:, :self.prefix-self.clustered].copy_(torch.cat(
                (v[:, :, :self.sink], v[:, :, tail:]), dim=2).transpose(1, 2))
            centers, labels, counts = self._cluster_fixed(raw[:, :, self.sink:tail].contiguous())
            self._install_clusters(layer, centers.repeat(1, self.batch, 1, 1),
                                   labels.repeat(1, self.batch, 1), counts.repeat(1, self.batch, 1))
        self.context = self.prefix

    def sync(self, layer_idx, start_bdx):
        pass

    def decode_update_kv_cache(self, k, v, layer_idx):
        if layer_idx == self.layer_num - 1:
            self.context += 1
        return k, v

    def _lookup(self, q, layer):
        c, labels, counts = self.centroids[layer], self.labels[layer], self.counts[layer]
        if os.environ.get('MULTIPOLE_SPLIT_LOOKUP') == '1':
            from benchmarks.multipole_split_lookup import split_centroid_scores
            score, qk = split_centroid_scores(q, c, counts, labels, self.scale)
            score = score.masked_fill(counts == 0, -float('inf'))
        else:
            out = torch.empty((1, self.kv_heads, self.max_clusters, self.group), device=self.device, dtype=torch.float32)
            score = torch.empty((1, self.kv_heads, self.max_clusters, 1), device=self.device, dtype=torch.float32)
            qk = torch.zeros((1, self.kv_heads, self.group, self.max_clusters), device=self.device, dtype=torch.bfloat16)
            author._fwd_centroid_simple_kernel_qk[(1, self.kv_heads, 1)](
                q, c, qk, counts, labels, self.scale, out, score,
                *q.stride(), *c.stride(), *out.stride(), *score.stride(), *qk.stride(),
                1, self.kv_heads, self.group, labels.shape[2], self.max_clusters,
                BLOCK_M=16, BLOCK_N=128, BLOCK_DMODEL=self.dim, num_warps=4)
            score = score.squeeze(-1).masked_fill(counts == 0, -float('inf'))
        return self._select(score, labels, counts), qk

    def _sparse(self, q, layer, out, m, l):
        k = self.k[layer][:, :, self.sink:]
        v = self.v[layer][:, :, self.sink:]
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
        raw = self.current_raw_k.view(1, self.kv_heads, 1, self.dim)
        self.raw_k[layer_idx].index_copy_(2, self.all_index, raw)
        self.k[layer_idx].index_copy_(2, self.all_index, k.reshape(1, self.kv_heads, 1, self.dim))
        self.v[layer_idx].index_copy_(2, self.all_index, v.reshape(1, self.kv_heads, 1, self.dim))
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
        q_full = q.contiguous().view(1, self.kv_heads, self.group, self.dim)
        return self._sparse(q_full, layer_idx, out, m, l).view_as(q).to(q.dtype)

    def _update_clusters(self):
        profiling = os.environ.get('MULTIPOLE_PROFILE_UPDATES') == '1'
        stage_profiling = os.environ.get('MULTIPOLE_PROFILE_STAGES') == '1'
        if profiling:
            torch.cuda.synchronize()
            update_start = time.perf_counter()
        if stage_profiling:
            torch.cuda.synchronize()
            stage_start = time.perf_counter()
            stages = {}

            def finish_stage(name):
                nonlocal stage_start
                torch.cuda.synchronize()
                now = time.perf_counter()
                stages[name] = stages.get(name, 0.0) + now - stage_start
                stage_start = now
        previous = self.clustered
        self.clustered += self.interval
        rebuild_fixed = (self.fixed_centroids is not None
                         and self._final_block_start(self.clustered) != self.fixed_blocks[-1][0])
        if rebuild_fixed:
            self.fixed_blocks = self._fixed_block_layout(self.clustered)
        for layer in range(self.layer_num):
            old_clusters = self.fixed_centroids or previous // 16
            center = self.centroids[layer][:, :, :old_clusters].contiguous()
            labels = self.labels[layer][:, :, :previous].contiguous()
            counts = self.counts[layer][:, :, :old_clusters].contiguous()
            data = self.raw_k[layer][:, :, self.sink:self.sink+self.clustered].contiguous()
            if stage_profiling:
                finish_stage('prepare')
            if self.fixed_centroids is None:
                centers, label_list, count_list, _ = run_clustering_online_update(
                    data, [center], [labels], [counts], [self.interval // 16], [self.interval], 1, rank=0)
                state = centers[0], label_list[0], count_list[0]
            elif rebuild_fixed:
                state = self._cluster_fixed(data)
            else:
                state = self._refine_fixed(data, center, labels, counts)
            if stage_profiling:
                finish_stage('cluster_update')
            self._install_clusters(layer, *state)
            if stage_profiling:
                finish_stage('install_index_and_recompute_value_centroids')
            old_len = self.context - self.sink - previous
            # Keep the sink and shift the remaining local window into place.
            self.local_k[layer][:, self.sink:self.sink+old_len-self.interval].copy_(
                self.local_k[layer][:, self.sink+self.interval:self.sink+old_len].clone())
            self.local_v[layer][:, self.sink:self.sink+old_len-self.interval].copy_(
                self.local_v[layer][:, self.sink+self.interval:self.sink+old_len].clone())
        self.active_length.fill_(self.clustered)
        if stage_profiling:
            finish_stage('shift_local_and_metadata')
            self.update_stage_times.append(stages)
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
                     graph_argmax=int(graph_logits.argmax()), eager_argmax=int(eager_logits.argmax()),
                     graph_argmax_by_request=graph_logits.argmax(dim=-1).flatten().tolist(),
                     eager_argmax_by_request=eager_logits.argmax(dim=-1).flatten().tolist())
        if os.environ.get('MULTIPOLE_CHECK_REPEAT') == '1':
            self.context = previous
            self._set_position(previous)
            repeated_logits = model.decode_forward(inputs_ids=token)
            repeat_diff = (eager_logits - repeated_logits).abs()
            stats.update(eager_repeat_max_abs=float(repeat_diff.max()),
                         eager_repeat_mean_abs=float(repeat_diff.mean()),
                         eager_repeat_argmax=int(repeated_logits.argmax()))
        return graph_token, stats


def attach_multipole_batch(model, *, budget=512, replacement=True, fixed_centroids=None):
    original_position = model.position_embedd

    def init_cache(this, valid_start, config):
        if len(valid_start) != this.batch_size or any(int(x) != 0 for x in valid_start):
            raise ValueError('requires unpadded equal-length requests')
        this.kv_cache = None
        this.kv_cache = RetroMultipoleBatchCache(
            this, budget=budget, replacement=replacement, fixed_centroids=fixed_centroids)

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

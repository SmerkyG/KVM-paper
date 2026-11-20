"""Whole-model CUDA graph adapter for RetroInfer's dense and GPU-only paths.

RetroInfer's released GPU-only cache has no graph implementation. The adapter
keeps its kernels and selection policy, replacing the Python-indexed append with
an equivalent device-indexed append so one captured graph can handle each
successive token. After each author index update it recaptures the model graph;
both the update and recapture are part of timed decode.
"""
import types
import torch


class NativeDecodeGraph:
    def __init__(self, model, token, attention):
        self.model = model
        self.cache = model.kv_cache
        self.attention = attention
        self.original_position = model.position_embedd
        self.position = torch.full((model.batch_size, 1), self.cache.context, dtype=torch.int64, device=token.device)
        self.input = token.clone()

        def position_embedd(this, q, k):
            return this.apply_rotary_pos_emb(q, k, this._graph_position)
        model._graph_position = self.position
        model.position_embedd = types.MethodType(position_embedd, model)
        if attention == 'RetroInfer':
            self._install_retro_cache()
        elif attention != 'Full_Flash_Attn':
            raise ValueError('native graph supports dense or GPU-only RetroInfer')

        self.index_update_count = 0
        self._capture()

    def _capture(self):
        model = self.model
        attention = self.attention
        context = self.cache.context
        static = self.cache.static_pattern_total if attention == 'RetroInfer' else None
        self._set_dynamic_metadata()
        model.sampling(model.decode_forward(inputs_ids=self.input))
        if attention == 'Full_Flash_Attn':
            for lengths in self.cache.valid_length_dict.values():
                lengths.sub_(1)
        else:
            self.cache.static_pattern_total = static
        self.cache.context = context
        torch.cuda.synchronize()
        self._set_dynamic_metadata()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = model.decode_forward(inputs_ids=self.input)
            self.output = model.sampling(self.logits)
        self.cache.context = context
        if attention == 'RetroInfer':
            self.cache.static_pattern_total = static

    def _install_retro_cache(self):
        from retroinfer_kernels import batch_gemm_softmax, gather_copy_vectors, gather_copy_cluster_and_concat_fuse
        from weighted_flash_decoding import weighted_flash_decoding
        cache = self.cache
        if not cache.index_built:
            raise NotImplementedError('graph pilot requires the sparse GPU index')
        self.append_index = torch.empty(1, dtype=torch.int64, device=self.input.device)
        self.original_update = cache.decode_update_kv_cache
        self.original_attn = cache.attn_func

        def graph_update(this, key, value, layer_idx):
            this.new_key_states = key
            this.new_value_states = value
            if layer_idx == this.layer_num - 1:
                this.context += 1
                this.static_pattern_total += 1
            return None, None

        def graph_sparse(this, queries, layer_idx, static_len):
            # All layers append at the same device-resident current position.
            this.steady_zone_keys[layer_idx].index_copy_(2, self.append_index, this.new_key_states.transpose(1, 2))
            this.steady_zone_values[layer_idx].index_copy_(2, self.append_index, this.new_value_states.transpose(1, 2))
            this.new_key_states = None
            this.new_value_states = None
            batch_gemm_softmax(queries, this.centroids[layer_idx], this.gemm_o, this.norm, this.sum,
                this.softmax_o, this.batch_groups, this.group_size, this.n_centroids, this.head_dim,
                this.RSQRT_DIM, 0)
            torch.sum(this.softmax_o, dim=1, out=this.dist)
            this.dist.masked_fill_(this.centroids_mask[layer_idx], this.DTYPE_MIN)
            torch.topk(this.dist, this.max_compute_cluster_num, dim=-1, largest=True, sorted=True,
                       out=(this.cV, this.cI))
            if this.es_cluster_num > 0:
                gather_copy_vectors(this.centroids[layer_idx], this.es_centroids,
                    this.value_sum[layer_idx], this.es_value_sum, this.cluster_size[layer_idx],
                    this.es_cluster_size, this.cI, this.batch_groups, this.n_centroids,
                    this.es_cluster_num, this.max_compute_cluster_num, this.nprobe, this.es_cluster_num)
                es_out, es_lse = weighted_flash_decoding(
                    queries.view(this.batch_groups, 1, this.group_size, this.head_dim),
                    this.es_centroids, this.es_value_sum, this.es_cluster_size,
                    previous_out=None, previous_lse=None, return_softmax_lse=True)
            else:
                es_out, es_lse = None, None
            gather_copy_cluster_and_concat_fuse(
                this.steady_zone_keys[layer_idx], this.list_keys[layer_idx], this.execution_buffer_keys,
                this.steady_zone_values[layer_idx], this.list_values[layer_idx], this.execution_buffer_values,
                this.cluster_size_cumsum[layer_idx], this.cI, this.valid_lengths,
                this.batch_groups, this.static_stride, this.list_stride, this.execution_stride,
                this.nprobe, this.nprobe_tensor, this.static_len_tensor)
            out = weighted_flash_decoding(
                queries.view(this.batch_groups, 1, this.group_size, this.head_dim),
                this.execution_buffer_keys, this.execution_buffer_values,
                previous_out=es_out, previous_lse=es_lse, cache_seqlens=this.valid_lengths,
                return_softmax_lse=False)
            return out.view(this.batch_size, 1, this.num_heads, this.head_dim)

        cache.decode_update_kv_cache = types.MethodType(graph_update, cache)
        cache.attn_func = types.MethodType(graph_sparse, cache)

    def _set_dynamic_metadata(self):
        self.position.fill_(self.cache.context)
        if self.attention == 'RetroInfer':
            self.append_index.fill_(self.cache.static_pattern_total)
            self.cache.static_len_tensor.fill_(self.cache.static_pattern_total + 1)

    def step(self, token):
        self._update_if_needed()
        self._set_dynamic_metadata()
        self.input.copy_(token)
        self.graph.replay()
        self.cache.context += 1
        if self.attention == 'RetroInfer':
            self.cache.static_pattern_total += 1
        return self.output

    def _update_if_needed(self):
        if self.attention == 'RetroInfer' and self.cache.static_pattern_total == self.cache.THRESHOLD:
            # The author's update replaces tensors and changes captured shapes.
            # Release the old graph before rebuilding to reduce peak memory.
            del self.graph, self.logits, self.output
            self.cache._update_kv_cache()
            self.index_update_count += 1
            self._capture()

    def checked_step(self, token):
        """Compare captured output to the author's eager code on the same cache."""
        # Update once before taking the shared cache snapshot. Otherwise the
        # subsequent eager check would incorrectly rebuild the index a second time.
        self._update_if_needed()
        previous = self.cache.context
        static = self.cache.static_pattern_total if self.attention == 'RetroInfer' else None
        graph_token = self.step(token)
        graph_logits = self.logits.clone()
        self.cache.context = previous
        if self.attention == 'Full_Flash_Attn':
            for lengths in self.cache.valid_length_dict.values():
                lengths.sub_(1)
        else:
            self.cache.static_pattern_total = static
            patched_update, patched_attn = self.cache.decode_update_kv_cache, self.cache.attn_func
            self.cache.decode_update_kv_cache = self.original_update
            self.cache.attn_func = self.original_attn
        eager_logits = self.model.decode_forward(inputs_ids=token)
        if self.attention == 'RetroInfer':
            self.cache.decode_update_kv_cache, self.cache.attn_func = patched_update, patched_attn
        difference = (graph_logits - eager_logits).abs()
        stats = dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                     graph_argmax=int(graph_logits.argmax()), eager_argmax=int(eager_logits.argmax()),
                     graph_argmax_by_request=graph_logits.argmax(dim=-1).flatten().tolist(),
                     eager_argmax_by_request=eager_logits.argmax(dim=-1).flatten().tolist())
        return graph_token, stats

    def close(self):
        self.model.position_embedd = self.original_position
        del self.model._graph_position

"""LoD attention and cache adapter for RetroInfer's Llama model.

This benchmark-only adapter preserves RetroInfer's model, projections, rotary
encoding, feed-forward layers and sampling. Its per-layer LoD pool uses the
same Llama geometry and three-tier top-two settings as the vLLM backend.
"""
from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path


def install_lod_modules():
    # RetroInfer's transformers version predates AttentionInterface. LoD's
    # projection-free engine does not require the Hugging Face integration;
    # import it as a namespace package without executing the public HF facade.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'integrations' / 'vllm_lod'))
    if 'lod_attention' not in sys.modules:
        package = types.ModuleType('lod_attention')
        package.__path__ = [str(root / 'lod_attention')]
        sys.modules['lod_attention'] = package

    import torch
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from lod_attention.kernels import cuda_prefill_attention

    def local_flash(q, k, v, *, scale, output_buffer=None, return_lse=True):
        # RetroInfer ships FlashAttention 2 rather than vLLM's FlashAttention 3.
        # Use its native GQA kernel and expose its log-sum-exp for LoD's merge.
        out, lse, _, _ = _flash_attn_forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            0.0, scale, True, -1, -1, 0.0, None, False,
        )
        out = out.transpose(1, 2)
        if output_buffer is not None:
            output_buffer.copy_(out)
            out = output_buffer
        if not return_lse:
            lse = torch.empty(0, device=q.device, dtype=torch.float32)
        return out, lse

    cuda_prefill_attention.cuda_local_attention = local_flash


class RetroLODCache:
    """One persistent LoD pool per RetroInfer model layer, batch one."""
    def __init__(self, model, *, mode):
        import torch
        from vllm_lod_plugin.config import VLLMLODSettings
        from vllm_lod_plugin.pool import VLLMLayerLODPool

        if model.batch_size != 1 or model.num_gpus != 1:
            raise NotImplementedError('LoD adapter currently supports one request and one GPU')
        self.mode = mode
        threshold = os.environ.get('LOD_H100_THRESHOLD_MULTIPLIER')
        self.threshold_multiplier = float(threshold) if threshold else None
        if os.environ.get('LOD_H100_PAGE_THRESHOLD') is not None and (mode != 'three-tier-bf16' or self.threshold_multiplier is None):
            raise ValueError('page thresholds require three-tier BF16 threshold decode')
        self.threshold_rule = os.environ.get('LOD_H100_THRESHOLD_RULE', 'region')
        self.threshold_entry_bound = int(os.environ.get('LOD_H100_ENTRY_BOUND', '16'))
        self.threshold_work_bound = int(os.environ.get('LOD_H100_ENTRY_WORK_BOUND', str(self.threshold_entry_bound)))
        if self.threshold_entry_bound < 1:
            raise ValueError('entry bound must be positive')
        if self.threshold_work_bound < 1:
            raise ValueError('work bound must be positive')
        if self.threshold_rule not in ('region', 'entry'):
            raise ValueError('unknown H100 threshold rule')
        if self.threshold_rule == 'entry' and self.threshold_multiplier is None:
            raise ValueError('entry threshold requires enabling threshold decode')
        if self.threshold_multiplier is not None and (not math.isfinite(self.threshold_multiplier) or self.threshold_multiplier <= 0):
            raise ValueError('threshold experiment requires a finite positive multiplier')
        if self.threshold_multiplier is not None and mode != 'two-tier' and self.threshold_rule != 'region':
            raise ValueError('three-tier threshold decode requires per-region routing')
        self.context = 0
        self.layer_num = model.num_layers
        self.graph_mode = False
        self.graph_capturing = False
        self.graph = None
        capacity = max(512, model.input_length + model.max_new_length)
        device = torch.device(model.device_map)
        indices = torch.zeros(1, device=device, dtype=torch.int32)
        settings = VLLMLODSettings.production(mode=mode, pool_size=1,
                                              request_capacity=capacity)
        layer = types.SimpleNamespace(
            num_heads=model.num_heads, num_kv_heads=model.num_key_value_heads,
            head_size=model.head_dim, head_size_v=model.head_dim,
            impl=types.SimpleNamespace(scale=1.0 / math.sqrt(model.head_dim)),
        )
        self.pools = [VLLMLayerLODPool(layer, settings=settings, max_requests=1,
                      request_capacity=capacity, active_indices=indices,
                      has_query_norm=getattr(model, 'lod_has_qk_norm', False),
                      has_key_norm=getattr(model, 'lod_has_qk_norm', False),
                      dtype=model.dtype, device=device) for _ in range(model.num_layers)]
        if self.threshold_multiplier is not None and mode != 'two-tier':
            from benchmarks.h100_threshold_three_tier import install_cached_means
            if os.environ.get('LOD_H100_THREE_TIER_CACHED_MEANS', '1') == '1':
                for pool in self.pools:
                    install_cached_means(pool)
        prefill_rule = os.environ.get('LOD_H100_PREFILL_RULE')
        if prefill_rule:
            from benchmarks.h100_threshold_prefill import install, validate_cache_mode
            validate_cache_mode(mode, prefill_rule)
            # Batch-one RetroInfer executes prefill layers sequentially on its
            # model stream. Share scratch between layers rather than retaining
            # one potentially large selected-route buffer per model layer.
            shared_prefill_buffers = {} if os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION') in ('fused', 'fast', 'closed') else None
            for pool in self.pools:
                if shared_prefill_buffers is not None:
                    pool.engine._lod_prefill_attention_buffers = shared_prefill_buffers
                install(pool.engine, rule=prefill_rule,
                        multiplier=float(os.environ.get('LOD_H100_PREFILL_MULTIPLIER', '16')),
                        entry_bound=int(os.environ.get('LOD_H100_PREFILL_ENTRY_BOUND', '16')),
                        work_bound=int(os.environ.get('LOD_H100_PREFILL_WORK_BOUND',
                                                      os.environ.get('LOD_H100_PREFILL_ENTRY_BOUND', '16'))),
                        tile_queries=int(os.environ.get('LOD_H100_PREFILL_TILE_QUERIES', '512')))
        if os.environ.get('LOD_H100_TOP8_QUERY_BACKEND') == '1':
            if prefill_rule or mode != 'two-tier':
                raise ValueError('top-eight query backend requires ordinary two-tier routing')
            from benchmarks.h100_top8_query_backend import install
            shared_top8_buffers = {}
            for pool in self.pools:
                install(pool.engine, shared_top8_buffers)
        if os.environ.get('LOD_H100_OVERLAP_AUDIT') == '1' and not prefill_rule:
            from benchmarks.h100_prefill_overlap import install_top8
            for pool in self.pools:
                install_top8(pool.engine)
        self.catch_up_events = 0
        if os.environ.get('LOD_TRACK_UPDATES') == '1':
            for pool in self.pools:
                finish = pool._finish_single_catch_up
                def tracked_finish(slot, row, finish=finish):
                    finish(slot, row)
                    self.catch_up_events += 1
                pool._finish_single_catch_up = tracked_finish
        # Benchmark-only quality ablation; retain the published defaults in
        # production. This pool is Llama BF16 three-tier and has separate,
        # non-arena sink storage, so it can hold the four exact prefix KVs.
        self.native_route_four_sink = os.environ.get('LOD_RETRO_NATIVE_ROUTE_FOUR_SINK') == '1'
        if self.native_route_four_sink:
            if mode != 'three-tier-bf16' or model.num_key_value_heads != 8 or model.num_heads != 32:
                raise ValueError('native-route/four-sink ablation requires Llama three-tier BF16')
            for pool in self.pools:
                if pool.state['page_cache'].get('unified_page1_k') is not None:
                    raise RuntimeError('diagnostic requires separate sink storage')
                pool.engine.sink_len = 4
                pool.engine.routing_normalization = 'none'
                shape = (1, pool.kv_heads, 4, pool.head_dim)
                pool.state['sink_k'] = torch.empty(shape, dtype=pool.dtype, device=pool.device)
                pool.state['sink_v'] = torch.empty_like(pool.state['sink_k'])
        # H100, batch one, 32K--64K: smaller route tiles with one warp
        # reduce end-to-end decode time for both recursive cache formats.
        # Retain the general production geometry outside the tested regime.
        if (torch.cuda.get_device_capability(device) == (9, 0)
                and 32768 <= model.input_length <= 65536):
            for pool in self.pools:
                pool.engine.decode_route_group_size = 32
                pool.engine.decode_route_num_warps = 1
        # Benchmark-only route geometry overrides, applied before decode
        # scratch allocation. The cluster-run record retains these settings.
        for pool in self.pools:
            for env, attr in (
                ('LOD_RETRO_ROUTE_GROUP', 'decode_route_group_size'),
                ('LOD_RETRO_ROUTE_WARPS', 'decode_route_num_warps'),
                ('LOD_RETRO_ROUTE_REDUCE_WARPS', 'decode_route_reduce_num_warps'),
                ('LOD_RETRO_PAGE_BLOCK', 'decode_block_n'),
            ):
                if env in os.environ:
                    setattr(pool.engine, attr, int(os.environ[env]))

    def prefill_update_kv_cache(self, q, k, v, layer_idx, start_bdx):
        if start_bdx != 0 or k.shape[0] != 1:
            raise ValueError('LoD adapter supports one unpadded request')
        if layer_idx == self.layer_num - 1:
            self.context += k.shape[1]
        return k, v

    def sync(self, layer_idx, start_bdx):
        pass

    def decode_update_kv_cache(self, k, v, layer_idx):
        if layer_idx == self.layer_num - 1:
            self.context += 1
        return k, v

    def prefill_attention(self, q, k, v, layer_idx):
        import torch
        from lod_attention._config import PREFILL_CHUNK_SIZE
        length = q.shape[1]
        pool = self.pools[layer_idx]
        out = torch.empty_like(q).reshape(length, q.shape[2], q.shape[3])
        q_flat = q.reshape(length, q.shape[2], q.shape[3])
        k_flat = k.reshape(length, k.shape[2], k.shape[3])
        v_flat = v.reshape(length, v.shape[2], v.shape[3])
        for begin in range(0, length, PREFILL_CHUNK_SIZE):
            end = min(begin + PREFILL_CHUNK_SIZE, length)
            pool.direct_prefill_plan = ((0, begin, end, begin),)
            pool.direct_prefill_prompt_lengths = {0: length}
            pool.direct_prefill(q_flat, k_flat, v_flat, out)
        if self.native_route_four_sink:
            expected_k = k[:, :4].transpose(1, 2)
            expected_v = v[:, :4].transpose(1, 2)
            if not torch.equal(pool.state['sink_k'], expected_k) or not torch.equal(pool.state['sink_v'], expected_v):
                raise RuntimeError(f'four-sink KV mismatch in layer {layer_idx}')
            if pool.engine.sink_len != 4 or pool.engine.routing_normalization != 'none':
                raise RuntimeError('four-sink native-route settings were not applied')
            archived = int(pool.leaf_lens[0].item())
            local = int(pool.local_lens[0].item())
            if archived + local + 4 != length:
                raise RuntimeError(f'sink/archive/local partition mismatch: {archived}+{local}+4 != {length}')
            counts = pool.state['counts'][0, :, :, 0].sum(-1)
            if not torch.all(counts == archived):
                raise RuntimeError('centroid counts do not cover the non-sink archive')
        return out.reshape_as(q)

    def decode_attention(self, q, k, v, layer_idx):
        import torch
        pool = self.pools[layer_idx]
        previous = self.context - int(layer_idx == self.layer_num - 1)
        if not self.graph_capturing:
            pool.catch_up(0, previous)
        metadata = types.SimpleNamespace(num_actual_tokens=1, max_seq_len=previous+1)
        out = torch.empty_like(q).reshape(1, q.shape[2], q.shape[3])
        if self.threshold_multiplier is None:
            pool.decode(q.reshape(1, q.shape[2], q.shape[3]),
                        k.reshape(1, k.shape[2], k.shape[3]),
                        v.reshape(1, v.shape[2], v.shape[3]), metadata, out)
        else:
            if self.mode == 'two-tier':
                from benchmarks.h100_threshold_decode import threshold_attention
            else:
                from benchmarks.h100_threshold_three_tier import threshold_attention
            threshold_attention(pool,q,k,v,out,self.threshold_multiplier,
                                tile=int(os.environ.get('LOD_H100_THRESHOLD_TILE','64')),
                                splits=int(os.environ.get('LOD_H100_THRESHOLD_SPLITS','64')),
                                rule=self.threshold_rule,entry_bound=self.threshold_entry_bound,
                                work_bound=self.threshold_work_bound)
        return out.reshape_as(q)

    def capture_decode(self, model, initial_token):
        """Capture the entire one-token model step with persistent device state.

        A real warmup initializes per-layer scratch before capture. Roll back
        that single KV append; the captured step itself is recorded but does
        not execute until replay. Dynamic cache catch-up stays outside graphs.
        """
        import torch
        if self.context <= 2048:
            raise ValueError('whole-step graph comparison requires a routed context above 2048 tokens')
        previous = self.context
        for pool in self.pools:
            pool.catch_up(0, previous)
        self.graph_mode = True
        self.graph_capturing = True
        self.graph_position = torch.tensor([[previous]], dtype=torch.int64,
                                           device=initial_token.device)
        self.graph_input = initial_token.clone()
        # Warm up all graph operations outside the capture stream.
        model.sampling(model.decode_forward(inputs_ids=self.graph_input))
        for pool in self.pools:
            pool.local_lens.sub_(1)
        self.context = previous
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.graph_logits = model.decode_forward(inputs_ids=self.graph_input)
            self.graph_output = model.sampling(self.graph_logits)
        # Python-side context advanced during graph recording; GPU work has
        # not executed and the captured KV append is still pending.
        self.context = previous
        self.graph_capturing = False

    def graph_step(self, token):
        if self.graph is None:
            raise RuntimeError('LoD decode graph is not initialized')
        for pool in self.pools:
            pool.catch_up(0, self.context)
        self.graph_position.fill_(self.context)
        self.graph_input.copy_(token)
        self.graph.replay()
        self.context += 1
        return self.graph_output

    def checked_graph_step(self, model, token):
        """Compare graph and eager on the same cache state, leaving one append."""
        import torch
        previous = self.context
        graph_token = self.graph_step(token)
        graph_logits = self.graph_logits.clone()
        self.context = previous
        for pool in self.pools:
            pool.local_lens.sub_(1)
        # Optionally compare the tuned graph to the original route geometry
        # on the identical cache, rather than comparing separate prefills.
        engines = [pool.engine for pool in self.pools]
        attrs = ('decode_route_group_size', 'decode_route_num_warps',
                 'decode_route_reduce_num_warps', 'decode_block_n')
        saved = [tuple(getattr(engine, attr) for attr in attrs) for engine in engines]
        decode_implementation = os.environ.get('LOD_H100_THRESHOLD_IMPLEMENTATION')
        threshold_reference = os.environ.get('LOD_H100_DECODE_CHECK_REFERENCE') == '1'
        try:
            if threshold_reference:
                os.environ['LOD_H100_THRESHOLD_IMPLEMENTATION'] = 'legacy' if self.mode == 'two-tier' else 'three-tier-reference'
            if os.environ.get('LOD_RETRO_CHECK_REFERENCE') == '1':
                for engine in engines:
                    for attr, value in zip(attrs, (64, 2, 4, 16)):
                        setattr(engine, attr, value)
            eager_logits = model.decode_forward(inputs_ids=token)
        finally:
            if threshold_reference:
                if decode_implementation is None:
                    os.environ.pop('LOD_H100_THRESHOLD_IMPLEMENTATION', None)
                else:
                    os.environ['LOD_H100_THRESHOLD_IMPLEMENTATION'] = decode_implementation
            for engine, values in zip(engines, saved):
                for attr, value in zip(attrs, values):
                    setattr(engine, attr, value)
        difference = (graph_logits - eager_logits).abs()
        stats = dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                     graph_argmax=int(graph_logits.argmax()),
                     eager_argmax=int(eager_logits.argmax()))
        if threshold_reference:
            stats['reference_implementation'] = 'legacy' if self.mode == 'two-tier' else 'three-tier-reference'
            if not bool(torch.isfinite(difference).all().item()) or stats['graph_argmax'] != stats['eager_argmax']:
                raise RuntimeError('optimized threshold graph differs from legacy next-token choice: ' + str(stats))
        return graph_token, stats


def attach_lod(model, mode, *, dense_prefill=False):
    """Install the adapter on an instance without modifying the author source."""
    install_lod_modules()
    original_position = model.position_embedd

    def init_cache(this, valid_start, config):
        if len(valid_start) != 1 or int(valid_start[0]) != 0:
            raise ValueError('LoD adapter requires an unpadded batch of one')
        this.kv_cache = None
        this.kv_cache = RetroLODCache(this, mode=mode)

    def prefill(this, q, k, v, layer_idx):
        output = this.kv_cache.prefill_attention(q, k, v, layer_idx)
        if dense_prefill:
            # Build the identical LoD cache but use dense attention output to
            # isolate how sparse prefill affects later layers and decoding.
            from attn_hub import full_prefill_attn
            return full_prefill_attn(q, k, v, causal=True)
        return output

    def decode(this, q, k, v, layer_idx):
        return this.kv_cache.decode_attention(q, k, v, layer_idx)

    def move(this):
        # LoD cache construction is performed during each prefill layer.
        import torch
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    def position(this, q, k):
        if this.kv_cache.graph_mode:
            return this.apply_rotary_pos_emb(q, k, this.kv_cache.graph_position)
        return original_position(q, k)

    model.init_kv_cache = types.MethodType(init_cache, model)
    model.prefill_attention = types.MethodType(prefill, model)
    model.decode_attention = types.MethodType(decode, model)
    model.move = types.MethodType(move, model)
    model.position_embedd = types.MethodType(position, model)


def attach_lod_sparse_prefill_dense_decode(model, mode):
    """Keep native exact KV and decode, using the LoD prefill calculation."""
    install_lod_modules()
    original_init = model.init_kv_cache

    def init_cache(this, valid_start, config):
        if len(valid_start) != 1 or int(valid_start[0]) != 0:
            raise ValueError('diagnostic requires an unpadded batch of one')
        # Native prefill_update_kv_cache retains every original K/V produced
        # by the sparse-prefill model, for its usual full-attention decoder.
        original_init(valid_start, config)
        this._lod_prefill_cache = RetroLODCache(this, mode=mode)

    def prefill(this, q, k, v, layer_idx):
        import torch
        output = this._lod_prefill_cache.prefill_attention(q, k, v, layer_idx)
        # This is a diagnostic, so check every stored K and V rather than
        # assuming the native asynchronous cache copies completed correctly.
        this.kv_cache.sync(layer_idx, 0)
        length = k.shape[1]
        stored_k = this.kv_cache.key_cache[layer_idx][:, :length].to(k.device)
        stored_v = this.kv_cache.value_cache[layer_idx][:, :length].to(v.device)
        if not torch.equal(stored_k, k) or not torch.equal(stored_v, v):
            raise RuntimeError(f'dense KV cache differs at layer {layer_idx}')
        return output

    model.init_kv_cache = types.MethodType(init_cache, model)
    model.prefill_attention = types.MethodType(prefill, model)

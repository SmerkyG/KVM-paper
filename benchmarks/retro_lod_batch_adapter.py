"""LoD attention and cache adapter for RetroInfer's Llama model.

This benchmark-only adapter preserves RetroInfer's model, projections, rotary
encoding, feed-forward layers and sampling. Its per-layer LoD pool uses the
same Llama geometry and three-tier top-two settings as the vLLM backend.
"""
from __future__ import annotations

from functools import partial
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


class RetroLODBatchCache:
    """One persistent LoD pool per model layer, synchronous batches."""
    def __init__(self, model, *, mode):
        import torch
        from vllm_lod_plugin.config import VLLMLODSettings
        from vllm_lod_plugin.pool import VLLMLayerLODPool

        if model.batch_size < 1 or model.num_gpus != 1:
            raise NotImplementedError('LoD adapter requires a batch on one GPU')
        self.batch = model.batch_size
        threshold = os.environ.get('LOD_H100_THRESHOLD_MULTIPLIER')
        self.threshold_multiplier = float(threshold) if threshold else None
        threshold_impl = os.environ.get('LOD_H100_THRESHOLD_IMPLEMENTATION')
        supported_impls = ('materialized',) if mode == 'two-tier' else ('three-tier', 'three-tier-reference')
        if self.threshold_multiplier is not None and (
            not math.isfinite(self.threshold_multiplier) or self.threshold_multiplier <= 0
            or os.environ.get('LOD_H100_THRESHOLD_RULE', 'region') != 'region'
            or threshold_impl not in supported_impls
        ):
            raise ValueError('batch threshold decode requires a positive per-region threshold and a matching kernel')
        device = torch.device(model.device_map)
        self.threshold_tile = int(os.environ.get('LOD_H100_THRESHOLD_TILE', '64'))
        self.threshold_splits = int(os.environ.get('LOD_H100_THRESHOLD_SPLITS', '32' if mode == 'two-tier' else '64'))
        self.threshold_launch = None
        self.two_tier_optimized = None
        self.two_tier_kernel_options = None
        self.threshold_attention = None
        if self.threshold_multiplier is not None:
            batch_eight_32k = self.batch == 8 and model.input_length == 32768
            tuned_h100 = (
                batch_eight_32k and self.threshold_multiplier == 16 and model.dtype == torch.bfloat16
                and model.model_name == 'meta-llama/Llama-3.1-8B-Instruct'
                and torch.cuda.get_device_capability(device) == (9, 0)
            )
            if mode == 'two-tier':
                from benchmarks.h100_threshold_decode_materialized import threshold_attention_h100_materialized

                self.two_tier_optimized = os.environ.get('LOD_H100_TWO_TIER_OPTIMIZED', '1' if tuned_h100 else '0') == '1'
                if self.two_tier_optimized:
                    self.threshold_tile = int(os.environ.get('LOD_H100_THRESHOLD_TILE', '128' if tuned_h100 else '64'))
                tuned_two_tier = tuned_h100 and self.two_tier_optimized
                self.two_tier_kernel_options = dict(
                    entry_block=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK', '128' if tuned_two_tier else '64')),
                    entry_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS', '4' if tuned_two_tier else '1')),
                    merge_block=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK', '32' if tuned_two_tier else '16')) or 16,
                    precise=os.environ.get('LOD_AMD_THRESHOLD_PRECISE', '1' if tuned_two_tier else '0') == '1',
                )
                self.threshold_attention = partial(threshold_attention_h100_materialized,
                                                   optimized=self.two_tier_optimized,
                                                   **self.two_tier_kernel_options)
            else:
                from benchmarks.h100_threshold_three_tier import (
                    threshold_attention, threshold_launch_options, use_tuned_adaptive_pages,
                )

                precision = 'INT4' if mode == 'three-tier-int4' else 'BF16'
                tuned_fixed = (
                    tuned_h100 and threshold_impl == 'three-tier'
                    and os.environ.get('LOD_H100_THREE_TIER_PAGES', '2') == '2'
                    and os.environ.get('LOD_H100_PAGE_THRESHOLD') is None
                    and os.environ.get('LOD_H100_THREE_TIER_CACHED_MEANS', '1') == '1'
                    and os.environ.get(f'LOD_H100_{precision}_TUNED_DEFAULTS', '1') == '1'
                )
                tuned_mode = mode if tuned_fixed else None
                if mode == 'three-tier-bf16' and use_tuned_adaptive_pages(device):
                    tuned_mode = 'three-tier-adaptive-bf16'
                self.threshold_launch = threshold_launch_options(
                    tile=self.threshold_tile, splits=self.threshold_splits,
                    page_block_n=16 if batch_eight_32k and mode == 'three-tier-bf16' else None,
                    tuned_mode=tuned_mode,
                )
                self.threshold_attention = partial(threshold_attention, launch_options=self.threshold_launch)
        self.current_slot = 0
        self.context = 0
        self.layer_num = model.num_layers
        self.graph_mode = False
        self.graph_capturing = False
        self.graph = None
        capacity = max(512, model.input_length + model.max_new_length)
        indices = torch.arange(self.batch, device=device, dtype=torch.int32)
        settings = VLLMLODSettings.production(mode=mode, pool_size=self.batch,
                                              request_capacity=capacity)
        layer = types.SimpleNamespace(
            num_heads=model.num_heads, num_kv_heads=model.num_key_value_heads,
            head_size=model.head_dim, head_size_v=model.head_dim,
            impl=types.SimpleNamespace(scale=1.0 / math.sqrt(model.head_dim)),
        )
        self.pools = [VLLMLayerLODPool(layer, settings=settings, max_requests=self.batch,
                      request_capacity=capacity, active_indices=indices,
                      dtype=model.dtype, device=device) for _ in range(model.num_layers)]
        if (self.threshold_multiplier is not None and threshold_impl == 'three-tier'
                and os.environ.get('LOD_H100_THREE_TIER_CACHED_MEANS', '1') == '1'):
            from benchmarks.h100_threshold_three_tier import install_cached_means
            for pool in self.pools:
                install_cached_means(pool)
        if prefill_rule := os.environ.get('LOD_H100_PREFILL_RULE'):
            if prefill_rule != 'region':
                raise ValueError('batch threshold prefill requires per-region routing')
            from benchmarks.h100_threshold_prefill import install, validate_cache_mode
            validate_cache_mode(mode, prefill_rule)

            shared_buffers = {}
            for pool in self.pools:
                pool.engine._lod_prefill_attention_buffers = shared_buffers
                install(pool.engine, rule=prefill_rule,
                        multiplier=float(os.environ.get('LOD_H100_PREFILL_MULTIPLIER', '16')),
                        entry_bound=16, work_bound=16,
                        tile_queries=int(os.environ.get('LOD_H100_PREFILL_TILE_QUERIES', '512')))
        # H100, batch one, 32K--64K: smaller route tiles with one warp
        # reduce end-to-end decode time for both recursive cache formats.
        # Retain the general production geometry outside the tested regime.
        if (self.batch == 1 and torch.cuda.get_device_capability(device) == (9, 0)
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
        if k.shape[0] != 1 or not 0 <= start_bdx < self.batch:
            raise ValueError('LoD batch prefill requires unpadded single-request chunks')
        self.current_slot = start_bdx
        if layer_idx == self.layer_num - 1 and start_bdx + 1 == self.batch:
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
            pool.direct_prefill_plan = ((self.current_slot, begin, end, begin),)
            pool.direct_prefill_prompt_lengths = {self.current_slot: length}
            pool.direct_prefill(q_flat, k_flat, v_flat, out)
        return out.reshape_as(q)

    def decode_attention(self, q, k, v, layer_idx):
        import torch
        pool = self.pools[layer_idx]
        previous = self.context - int(layer_idx == self.layer_num - 1)
        if not self.graph_capturing:
            pool.catch_up_many([(slot, previous) for slot in range(self.batch)])
        if self.threshold_attention is not None:
            out = torch.empty_like(q, memory_format=torch.contiguous_format)
            return self.threshold_attention(
                pool, q.contiguous(), k.contiguous(), v.contiguous(), out, self.threshold_multiplier,
                tile=self.threshold_tile, splits=self.threshold_splits,
            )
        metadata = types.SimpleNamespace(num_actual_tokens=self.batch, max_seq_len=previous+1)
        out = torch.empty_like(q).reshape(self.batch, q.shape[2], q.shape[3])
        pool.decode(q.reshape(self.batch, q.shape[2], q.shape[3]),
                    k.reshape(self.batch, k.shape[2], k.shape[3]),
                    v.reshape(self.batch, v.shape[2], v.shape[3]), metadata, out)
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
        if self.threshold_multiplier is not None:
            from lod_attention.kernels.lod_kernels import new_state_delta_buffers

            # Layers update sequentially on the same stream, and the fused
            # apply kernel clears the touched entries before reuse.
            for pool in self.pools:
                for name in ('_lod_state_update_buffers', '_lod_prefill_attention_buffers'):
                    if hasattr(pool.engine, name):
                        delattr(pool.engine, name)
            first_pool = self.pools[0]
            if first_pool.engine.fused_state_update or first_pool.engine.auto_fused_state_update:
                deltas = new_state_delta_buffers(first_pool.state['state_k'], first_pool.state['state_v'],
                                                 first_pool.state_capacity)
                for pool in self.pools:
                    pool.engine._lod_state_update_buffers = deltas
        previous = self.context
        for pool in self.pools:
            pool.catch_up_many([(slot, previous) for slot in range(self.batch)])
        self.graph_mode = True
        self.graph_capturing = True
        self.graph_position = torch.full((self.batch, 1), previous, dtype=torch.int64,
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
            pool.catch_up_many([(slot, self.context) for slot in range(self.batch)])
        self.graph_position.fill_(self.context)
        self.graph_input.copy_(token)
        self.graph.replay()
        self.context += 1
        return self.graph_output

    def checked_graph_step(self, model, token):
        """Compare graph and eager on the same cache state, leaving one append."""
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
        try:
            if os.environ.get('LOD_RETRO_CHECK_REFERENCE') == '1':
                for engine in engines:
                    for attr, value in zip(attrs, (64, 2, 4, 16)):
                        setattr(engine, attr, value)
            eager_logits = model.decode_forward(inputs_ids=token)
        finally:
            for engine, values in zip(engines, saved):
                for attr, value in zip(attrs, values):
                    setattr(engine, attr, value)
        difference = (graph_logits - eager_logits).abs()
        stats = dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                     graph_argmax=int(graph_logits.argmax()),
                     eager_argmax=int(eager_logits.argmax()),
                     graph_argmax_by_request=graph_logits.argmax(dim=-1).flatten().tolist(),
                     eager_argmax_by_request=eager_logits.argmax(dim=-1).flatten().tolist())
        return graph_token, stats


def attach_lod_batch(model, mode):
    """Install the adapter on an instance without modifying the author source."""
    install_lod_modules()
    original_position = model.position_embedd

    def init_cache(this, valid_start, config):
        if len(valid_start) != this.batch_size or any(int(x) != 0 for x in valid_start):
            raise ValueError('LoD adapter requires unpadded equal-length requests')
        this.kv_cache = None
        this.kv_cache = RetroLODBatchCache(this, mode=mode)

    def prefill(this, q, k, v, layer_idx):
        return this.kv_cache.prefill_attention(q, k, v, layer_idx)

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

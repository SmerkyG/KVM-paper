"""Experimental QUEST-style BF16 backend for controlled H100 comparison.

This is a separate implementation, using vLLM's paged GQA KV cache, not the
released QUEST CUDA kernels. Restrict it to the single-request benchmark.
"""
import os
import re

import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl

from .quest_kernels import update_bounds, quest_attention


class QuestMetadataBuilder(FlashAttentionBackend.get_builder_cls()):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH


class QuestAttentionImpl(FlashAttentionImpl):
    def fused_rope_kvcache_supported(self):
        return False

    def fused_qk_norm_rope_kvcache_supported(self):
        return False

    def _dense_layer(self, layer):
        match = re.search(r"\.layers\.(\d+)\.", layer.layer_name)
        if not match:
            raise RuntimeError(f"Cannot identify decoder layer: {layer.layer_name}")
        return int(match.group(1)) < 2

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        if self._dense_layer(layer) or not kv_cache.numel():
            return
        if kv_cache.dtype != torch.bfloat16 or kv_cache.shape[2] != 16:
            raise ValueError("QUEST benchmark requires BF16 KV and 16-token pages")
        shape = (kv_cache.shape[0], kv_cache.shape[1], 2, self.head_size)
        if not hasattr(self, "quest_bounds"):
            self.quest_bounds = torch.empty(shape, device=kv_cache.device, dtype=kv_cache.dtype)
        if tuple(self.quest_bounds.shape) != shape:
            raise RuntimeError("KV cache shape changed after QUEST bounds allocation")
        update_bounds(kv_cache, slot_mapping, self.quest_bounds)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if (attn_metadata is None or attn_metadata.max_query_len != 1
                or self._dense_layer(layer)):
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("quantized attention output is unsupported")
        if query.shape[0] != 1 or attn_metadata.num_actual_tokens != 1:
            raise NotImplementedError("QUEST comparison currently supports batch one decode only")
        if not hasattr(self, "quest_bounds"):
            raise RuntimeError("QUEST page bounds were not initialized")
        quest_attention(query, kv_cache, attn_metadata.block_table,
                        attn_metadata.seq_lens, self.quest_bounds, output,
                        int(os.environ["QUEST_BUDGET_TOKENS"]), self.scale)
        return output


class QuestAttentionBackend(FlashAttentionBackend):
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return QuestAttentionImpl

    @staticmethod
    def get_builder_cls():
        return QuestMetadataBuilder

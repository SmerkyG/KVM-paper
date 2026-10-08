"""Benchmark-only exact all-history GLM MLA, preserving the daemon weights.

Use GLMDenseWorker explicitly; native/LoD serving never imports this module.
Only per-layer constructor configs change. The global HF config stays intact
so the existing native daemon fingerprint and FP8 tensors are reused.
"""
from __future__ import annotations

import copy
import inspect

import torch


def is_dense_ablation(module):
    return any(getattr(child, "_vllm_lod_glm53_dense_ablation", False)
               for child in module.modules())


def unused_dense_indexer(model, name):
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part not in ("indexer", "indexer_rope_emb"):
            continue
        parent = model
        for component in parts[:index]:
            parent = getattr(parent, component, None)
            if not isinstance(parent, torch.nn.Module):
                return False
        return (is_dense_ablation(parent)
                and getattr(parent, "indexer", None) is None
                and getattr(parent, part, None) is None)
    return False


def dense_metadata(model, metadata):
    modules = dict(model.named_modules())
    return {path: {name: value for name, value in values.items()
                   if not (name in ("is_sparse", "use_sparse", "is_v32",
                                    "prefill_backend", "q_pad_num_heads")
                           and path in modules and is_dense_ablation(modules[path]))}
            for path, values in metadata.items()}


def install_dense_ablation():
    from vllm.models.glm5next.common.attention import Glm5NextMLAAttention
    from vllm_lod_plugin import weight_cache_loader as loader

    cls = Glm5NextMLAAttention
    original = cls.__init__
    if getattr(original, "_glm53_dense_ablation", False):
        return
    signature = inspect.signature(original)

    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        config = copy.copy(bound.arguments["config"])
        assert config.model_type == "glm5_next_text"
        config.index_topk = None
        bound.arguments["config"] = config
        bound.arguments["topk_indices_buffer"] = None
        original(*bound.args, **bound.kwargs)
        for module in self.modules():
            module._vllm_lod_glm53_dense_ablation = True
        assert self.indexer is None and not self.is_v32
        assert not self.mla_attn.is_sparse
        assert not self.mla_attn.mla_attn.use_sparse
        assert not self.mla_attn.mla_attn.impl.is_sparse

    initialize._glm53_dense_ablation = True
    cls.__init__ = initialize
    original_unused = loader._unused_glm_indexer
    original_metadata = loader._apply_module_metadata
    loader._unused_glm_indexer = lambda model, name: (
        original_unused(model, name) or unused_dense_indexer(model, name))
    loader._apply_module_metadata = lambda model, metadata: original_metadata(
        model, dense_metadata(model, metadata))


# Imported only in the image's vLLM worker process, never by CPU helper tests.
def __getattr__(name):
    if name != "GLMDenseWorker":
        raise AttributeError(name)
    from vllm.v1.worker.gpu_worker import Worker

    class GLMDenseWorker(Worker):
        def __init__(self, *args, **kwargs):
            install_dense_ablation()
            super().__init__(*args, **kwargs)

    return GLMDenseWorker

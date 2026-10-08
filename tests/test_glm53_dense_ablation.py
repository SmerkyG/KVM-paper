"""The dense ablation must not weaken normal serving's weight import checks."""
import torch

from benchmarks._glm53_dense_attention import dense_metadata, unused_dense_indexer


def test_ablation_skips_only_its_intentionally_absent_indexer():
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.indexer = None
    model.attn.indexer_rope_emb = None
    assert not unused_dense_indexer(model, "attn.indexer.weight")
    model.attn._vllm_lod_glm53_dense_ablation = True
    assert unused_dense_indexer(model, "attn.indexer.weight")
    assert unused_dense_indexer(model, "attn.indexer_rope_emb.cos_sin_cache")
    assert not unused_dense_indexer(model, "attn.kv_b_proj.weight")
    assert not unused_dense_indexer(model, "missing.indexer.weight")
    model.attn.indexer = torch.nn.Module()
    assert not unused_dense_indexer(model, "attn.indexer_rope_emb.cos_sin_cache")


def test_ablation_metadata_retains_local_dispatch_but_imports_weight_layout():
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn._vllm_lod_glm53_dense_ablation = True
    model.other = torch.nn.Module()
    metadata = {"attn": {"is_sparse": True, "use_sparse": True,
                         "is_v32": True, "is_weight_shuffled": True,
                         "prefill_backend": None, "q_pad_num_heads": 16},
                "other": {"is_sparse": True}}
    assert dense_metadata(model, metadata) == {
        "attn": {"is_weight_shuffled": True}, "other": {"is_sparse": True}}
    assert metadata["attn"]["is_sparse"]  # original export is untouched

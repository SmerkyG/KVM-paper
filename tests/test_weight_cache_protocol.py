from __future__ import annotations

import os
import socket
from collections import namedtuple
from types import SimpleNamespace

import pytest

from vllm_lod_plugin.weight_cache_protocol import (
    WeightCacheFingerprint,
    cache_namespace,
    receive_message,
    send_message,
    socket_path,
    stable_hash,
)
from vllm_lod_plugin.weight_cache_daemon import _simple_metadata
from vllm_lod_plugin.weight_cache_loader import (
    _apply_module_metadata,
    _restore_kimi_lod_projection_views,
    _attention_weight_layout_hash,
    _weight_layout_hash,
    _unused_glm_indexer,
    _restore_fp8_moe_runtime,
)


def _fingerprint(**overrides: object) -> WeightCacheFingerprint:
    values: dict[str, object] = {
        "model": "example/model",
        "revision": "main",
        "architectures": ("ExampleForCausalLM",),
        "model_hash": "model-hash",
        "dtype": "torch.bfloat16",
        "quantization": "mxfp4",
        "quantization_hash": "quant-hash",
        "attention_hash": "attention-hash",
        "tp_size": 8,
        "tp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
        "dp_size": 1,
        "expert_parallel": True,
        "torch_version": "test",
        "vllm_version": "test",
        "device_arch": "gfx942",
    }
    values.update(overrides)
    return WeightCacheFingerprint(**values)  # type: ignore[arg-type]


def test_fingerprint_round_trip_and_exact_mismatch() -> None:
    fingerprint = _fingerprint()
    assert WeightCacheFingerprint.from_dict(fingerprint.to_dict()) == fingerprint
    assert fingerprint.mismatch(_fingerprint()) == {}
    assert fingerprint.mismatch(_fingerprint(tp_rank=1)) == {
        "tp_rank": (0, 1)
    }


def test_stable_hash_is_order_independent() -> None:
    assert stable_hash({"a": 1, "b": [2, 3]}) == stable_hash(
        {"b": [2, 3], "a": 1}
    )


def test_cache_paths_are_local_short_and_validated() -> None:
    root = f"/tmp/wcp-{os.getpid()}"
    namespace = cache_namespace(root, "experiment")
    assert namespace.name == "experiment"
    endpoint = socket_path(root, "x" * 200, "gpu-uuid", "group")
    assert len(os.fsencode(endpoint)) <= 100
    with pytest.raises(ValueError):
        cache_namespace(root, "../escape")


def test_socket_protocol_round_trip() -> None:
    sender, receiver = socket.socketpair()
    try:
        payload = {"type": "fetch", "values": [1, "two", None]}
        send_message(sender, payload)
        assert receive_message(receiver) == payload
    finally:
        sender.close()
        receiver.close()


def test_metadata_export_keeps_structured_runtime_objects_local() -> None:
    group_shape = namedtuple("GroupShape", ("row", "col"))
    assert _simple_metadata((1, 2)) == (1, 2)
    assert _simple_metadata([1, 2]) == [1, 2]
    assert _simple_metadata(group_shape(1, 32)) is _simple_metadata.missing


@pytest.mark.parametrize("client_lod", [False, True])
def test_weight_import_keeps_serving_attention_flags_local(client_lod) -> None:
    import torch

    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn._vllm_lod_absorbed_mla = client_lod
    _apply_module_metadata(model, {"attn": {
        "_vllm_lod_absorbed_mla": not client_lod,
        "_vllm_lod_backend_installed": True,
        "is_weight_shuffled": True}})
    assert model.attn._vllm_lod_absorbed_mla is client_lod
    assert not hasattr(model.attn, "_vllm_lod_backend_installed")
    assert model.attn.is_weight_shuffled is True


def test_lod_canonical_views_reuse_dense_daemon_original_kv_storage() -> None:
    import torch

    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    attn = model.attn
    attn._vllm_lod_absorbed_mla = True
    attn.num_heads, attn.qk_nope_head_dim = 2, 3
    attn.v_head_dim, attn.kv_lora_rank = 4, 5
    attn.kv_b_proj = torch.nn.Linear(5, 14, bias=False, dtype=torch.bfloat16)
    attn.is_aiter_triton_fp8_bmm_enabled = True
    attn.is_aiter_triton_fp4_bmm_enabled = False
    original = attn.kv_b_proj.weight
    before = original.detach().clone()
    _restore_kimi_lod_projection_views(model)
    assert tuple(attn.W_UK_T.shape) == (2, 3, 5)
    assert tuple(attn.W_UV.shape) == (2, 5, 4)
    assert attn.W_UK_T.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert attn.W_UV.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert torch.equal(attn.W_UK_T, before.view(2, 7, 5)[:, :3])
    assert torch.equal(attn.W_UV, before.view(2, 7, 5)[:, 3:].transpose(1, 2))
    assert torch.equal(original, before)
    assert not attn.is_aiter_triton_fp8_bmm_enabled
    saved = attn.W_UK_T
    _restore_kimi_lod_projection_views(model)
    assert attn.W_UK_T is saved


@pytest.mark.parametrize("native_prefix", [False, True])
def test_glm_ipc_preserves_client_sparse_dispatch(native_prefix) -> None:
    import torch

    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.is_v32 = native_prefix
    model.attn.core = torch.nn.Module()
    model.attn.core.is_sparse = native_prefix
    model.attn.core.inner = torch.nn.Module()
    model.attn.core.inner._vllm_lod_glm53 = True
    model.attn.core.inner.use_sparse = native_prefix
    model.native = torch.nn.Module()
    model.native.is_sparse = False
    _apply_module_metadata(model, {
        "attn": {"is_v32": not native_prefix},
        "attn.core": {"is_sparse": not native_prefix},
        "attn.core.inner": {"use_sparse": not native_prefix},
        "native": {"is_sparse": True},
    })
    assert model.attn.is_v32 is native_prefix
    assert model.attn.core.is_sparse is native_prefix
    assert model.attn.core.inner.use_sparse is native_prefix
    assert model.native.is_sparse is True


def test_glm_lod_skips_only_its_absent_native_indexer() -> None:
    import torch

    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.indexer = None
    model.attn.indexer_rope_emb = None
    model.attn.core = torch.nn.Module()
    model.attn.core._vllm_lod_glm53 = True
    assert _unused_glm_indexer(model, "attn.indexer.wk.weight")
    assert _unused_glm_indexer(model, "attn.indexer_rope_emb.cos_sin_cache")
    assert not _unused_glm_indexer(model, "attn.q_proj.weight")
    assert not _unused_glm_indexer(model, "missing.indexer.wk.weight")
    _apply_module_metadata(model, {"attn.indexer.wk": {"unused": True}})
    _apply_module_metadata(model, {"attn.indexer_rope_emb": {"unused": True}})
    with pytest.raises(RuntimeError, match="missing module"):
        _apply_module_metadata(model, {"attn.missing": {"invalid": True}})
    model.attn.core._vllm_lod_glm53 = False
    assert not _unused_glm_indexer(model, "attn.indexer.wk.weight")
    assert not _unused_glm_indexer(model, "attn.indexer_rope_emb.cos_sin_cache")
    model.attn.core._vllm_lod_glm53 = True
    model.attn.indexer = torch.nn.Module()
    assert not _unused_glm_indexer(model, "attn.indexer_rope_emb.cos_sin_cache")


def test_fp8_ipc_rebuild_does_not_convert_shared_weights() -> None:
    import torch

    class Method:
        def __init__(self):
            self.moe_kernel = None
            self.calls = 0

        def _init_moe_kernel(self, layer):
            self.calls += 1
            self.moe_kernel = object()

        def process_weights_after_loading(self, layer):
            raise AssertionError("must not reconvert daemon-owned weights")

    model = torch.nn.Linear(4, 4, bias=False)
    original = model.weight.detach().clone()
    model.quant_method = Method()
    assert _restore_fp8_moe_runtime(model, Method) == 1
    assert _restore_fp8_moe_runtime(model, Method) == 0
    assert model.quant_method.calls == 1
    assert torch.equal(model.weight, original)


def test_weight_layout_hash_ignores_runtime_context_capacity() -> None:
    class FakeHFConfig:
        def __init__(self, hidden_size: int) -> None:
            self.hidden_size = hidden_size

        def to_dict(self) -> dict[str, object]:
            return {
                "architectures": ["ExampleForCausalLM"],
                "hidden_size": self.hidden_size,
            }

    def model_config(max_model_len: int, hidden_size: int = 1024) -> object:
        return SimpleNamespace(
            hf_config=FakeHFConfig(hidden_size),
            model_impl="vllm",
            convert="auto",
            multimodal_config=SimpleNamespace(language_model_only=True),
            max_model_len=max_model_len,
        )

    assert _weight_layout_hash(model_config(8_192)) == _weight_layout_hash(
        model_config(262_144)
    )
    assert _weight_layout_hash(model_config(8_192)) != _weight_layout_hash(
        model_config(8_192, hidden_size=2048)
    )


def test_kimi_dcp_weight_layout_is_shared_across_dense_and_lod() -> None:
    def config(backend: str, dcp_size: int) -> object:
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="kimi_linear")
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp_size
            ),
            attention_config=SimpleNamespace(backend=backend),
        )

    dense_dcp = _attention_weight_layout_hash(config("TRITON_MLA", 8))
    lod_dcp = _attention_weight_layout_hash(config("CUSTOM", 8))
    assert dense_dcp == lod_dcp == "kimi-canonical-mla-v1"
    assert _attention_weight_layout_hash(config("TRITON_MLA", 1)) != lod_dcp

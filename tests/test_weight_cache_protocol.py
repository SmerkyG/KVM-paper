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
    _attention_weight_layout_hash,
    _weight_layout_hash,
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

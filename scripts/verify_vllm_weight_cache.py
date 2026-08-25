#!/usr/bin/env python3
"""Fast CPU checks for the out-of-tree vLLM weight-cache integration."""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations" / "vllm_lod"))

from vllm_lod_plugin.weight_cache_daemon import (
    ResidentGroup,
    WeightCacheBroker,
    WeightCacheServer,
)
from vllm_lod_plugin.weight_cache_loader import (
    _as_bool,
    _install_tensor,
    _remove_unexported_meta_tensors,
    ensure_broker_running,
)
from vllm_lod_plugin.weight_cache_protocol import (
    WeightCacheFingerprint,
    control_socket_path,
    receive_message,
    send_message,
    socket_path,
)


def fingerprint() -> WeightCacheFingerprint:
    return WeightCacheFingerprint(
        model="model",
        revision="revision",
        architectures=("ToyForCausalLM",),
        model_hash="model-hash",
        dtype="torch.bfloat16",
        quantization="",
        quantization_hash="quant-hash",
        tp_size=2,
        tp_rank=1,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        expert_parallel=False,
        torch_version="torch-version",
        vllm_version="vllm-version",
        device_arch="gfx942",
    )


def check_protocol() -> None:
    expected = fingerprint()
    assert WeightCacheFingerprint.from_dict(expected.to_dict()) == expected
    changed = WeightCacheFingerprint.from_dict(expected.to_dict() | {"tp_rank": 0})
    assert expected.mismatch(changed) == {"tp_rank": (1, 0)}

    left, right = socket.socketpair()
    payload = {"fingerprint": expected.to_dict(), "bytes": b"ipc-handle"}
    sender = threading.Thread(target=send_message, args=(left, payload))
    sender.start()
    assert receive_message(right) == payload
    sender.join()
    left.close()
    right.close()

    with tempfile.TemporaryDirectory() as temp:
        path = socket_path(temp, "cache", "GPU-uuid")
        assert path.parent.name == "cache"
        assert path.name.startswith("gpu-")
        assert path != socket_path(temp, "cache", "GPU-uuid", "other-model")
        assert control_socket_path(temp, "cache").name == "control.sock"
        assert len(str(path)) < 108

    assert _as_bool(True, name="value")
    assert _as_bool("yes", name="value")
    assert not _as_bool("0", name="value")
    try:
        _as_bool("maybe", name="value")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid boolean was accepted")


def check_automatic_broker_start() -> None:
    with tempfile.TemporaryDirectory() as temp:
        endpoint = ensure_broker_running(
            temp,
            "automatic",
            auto_start=True,
            timeout=10.0,
        )
        assert endpoint == control_socket_path(temp, "automatic")
        assert endpoint.is_socket()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(endpoint))
            send_message(client, {"type": "stop"})
            assert receive_message(client)["status"] == "ok"
        deadline = time.monotonic() + 5.0
        while endpoint.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not endpoint.exists()


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False, device="meta")
        self.register_buffer("obsolete", torch.empty(1, device="meta"))
        self.scale = torch.empty(1, device="meta")


def check_binding() -> None:
    model = Toy()
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    scale = torch.tensor([0.5])
    _install_tensor(model, "proj.weight", weight, "parameter")
    _install_tensor(model, "scale", scale, "attribute")
    _remove_unexported_meta_tensors(model, {"proj.weight", "scale"})
    assert model.proj.weight.data_ptr() == weight.data_ptr()
    assert model.scale.data_ptr() == scale.data_ptr()
    assert "obsolete" not in model._buffers
    assert model.proj(weight.new_ones(1, 3)).tolist() == [[3.0, 12.0]]

    server = WeightCacheServer(model, fingerprint(), Path("unused.sock"))
    server.export()
    assert server.entries["proj.weight"]["transport"] == "cpu"
    assert server.entries["scale"]["transport"] == "cpu"
    assert server.resident_bytes == 0


class FakeProcess:
    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        return None

    def kill(self) -> None:
        self.alive = False


def resident_group(key: str, process: FakeProcess) -> ResidentGroup:
    return ResidentGroup(
        key=key,
        description=key,
        processes=[process],
        endpoints={"gpu": f"/{key}.sock"},
        resident_bytes={"gpu": 60},
        allocation_bytes={"gpu": 80},
        total_memory={"gpu": 100},
    )


def check_lru_eviction() -> None:
    with tempfile.TemporaryDirectory() as temp:
        broker = WeightCacheBroker(
            cache_dir=temp,
            cache_id="cache",
            max_cache_fraction=1.0,
            max_cache_gb_per_gpu=None,
            load_timeout=1.0,
            force=False,
        )
        old_process = FakeProcess()
        new_process = FakeProcess()
        broker.groups = {
            "old": resident_group("old", old_process),
            "new": resident_group("new", new_process),
        }
        broker._enforce_budget("new")
        assert set(broker.groups) == {"new"}
        assert not old_process.alive
        assert new_process.alive

        active = resident_group("active", FakeProcess())
        active.client_pids.add(os.getpid())
        broker.groups["active"] = active
        broker._enforce_budget("new")
        assert "active" in broker.groups


def main() -> None:
    check_protocol()
    check_automatic_broker_start()
    check_binding()
    check_lru_eviction()
    print("vLLM weight-cache protocol and zero-copy binding checks passed")


if __name__ == "__main__":
    main()

"""Small, local-only protocol for the vLLM GPU weight cache.

The tensor payloads are CUDA/HIP IPC reconstruction arguments produced by
PyTorch.  The exporting process must remain alive while any client uses them.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import socket
import stat
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_MESSAGE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class WeightCacheFingerprint:
    model: str
    revision: str
    architectures: tuple[str, ...]
    model_hash: str
    dtype: str
    quantization: str
    quantization_hash: str
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    expert_parallel: bool
    torch_version: str
    vllm_version: str
    device_arch: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["architectures"] = list(self.architectures)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightCacheFingerprint:
        values = dict(data)
        values["architectures"] = tuple(values.get("architectures", ()))
        return cls(**values)

    def mismatch(self, other: WeightCacheFingerprint) -> dict[str, tuple[Any, Any]]:
        mine = self.to_dict()
        theirs = other.to_dict()
        return {
            key: (mine.get(key), theirs.get(key))
            for key in mine.keys() | theirs.keys()
            if mine.get(key) != theirs.get(key)
        }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                key: _jsonable(item)
                for key, item in sorted(vars(value).items())
                if not key.startswith("_")
                and isinstance(
                    item, (type(None), bool, int, float, str, list, tuple, dict)
                )
            },
        }
    return f"{type(value).__module__}.{type(value).__qualname__}"


def stable_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_model_name(model: str) -> str:
    path = Path(model).expanduser()
    return str(path.resolve()) if path.exists() else model


def default_cache_dir() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(runtime_dir) if runtime_dir else Path("/tmp")
    return root / f"vllm-weight-cache-{os.getuid()}"


def cache_namespace(cache_dir: str | os.PathLike[str] | None, cache_id: str) -> Path:
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cache_id) or cache_id in {".", ".."}:
        raise ValueError(
            "Weight-cache id must contain only letters, digits, '.', '_', and '-'"
        )
    return root / cache_id


def control_socket_path(
    cache_dir: str | os.PathLike[str] | None, cache_id: str
) -> Path:
    return cache_namespace(cache_dir, cache_id) / "control.sock"


def socket_path(
    cache_dir: str | os.PathLike[str] | None,
    cache_id: str,
    device_uuid: str,
    group_key: str = "preloaded",
) -> Path:
    # Unix-domain paths are normally limited to 108 bytes. Hash the UUID and
    # model group into one flat filename rather than spending path bytes on a
    # nested model directory.
    digest = hashlib.sha256(
        f"{cache_id}\0{group_key}\0{device_uuid}".encode()
    ).hexdigest()[:24]
    return cache_namespace(cache_dir, cache_id) / f"gpu-{digest}.sock"


def ready_path(
    cache_dir: str | os.PathLike[str] | None,
    cache_id: str,
    device_uuid: str,
    group_key: str = "preloaded",
) -> Path:
    return socket_path(cache_dir, cache_id, device_uuid, group_key).with_suffix(
        ".ready.json"
    )


def prepare_namespace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def validate_owned_socket(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"Refusing to use non-owned Unix socket: {path}")
    return True


def send_message(sock: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Weight-cache message is {len(payload)} bytes; limit is "
            f"{MAX_MESSAGE_BYTES} bytes"
        )
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Weight-cache peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_message(sock: socket.socket) -> Any:
    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Weight-cache peer sent {size} bytes; limit is {MAX_MESSAGE_BYTES} bytes"
        )
    # The socket is owner-only and validate_owned_socket rejects replacement by
    # another uid. This protocol is intentionally local and must not be exposed
    # over TCP because the payload uses pickle for PyTorch IPC metadata.
    return pickle.loads(_recv_exact(sock, size))


def read_ready(path: Path) -> dict[str, Any] | None:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(f"Refusing to read non-owned ready file: {path}")
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def clean_stale_endpoint(sock_path: Path, *, force: bool = False) -> None:
    state_path = sock_path.with_suffix(".ready.json")
    ready = read_ready(state_path)
    pid = int(ready.get("pid", 0)) if ready else 0
    if pid > 0 and pid_alive(pid):
        if not force:
            raise RuntimeError(
                f"A weight-cache daemon is already alive at {sock_path} (pid={pid})"
            )
        os.kill(pid, 15)
    for path in (sock_path, state_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

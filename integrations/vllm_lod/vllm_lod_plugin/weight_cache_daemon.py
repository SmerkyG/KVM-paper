"""Persistent vLLM GPU-weight cache using CUDA/HIP IPC.

The daemon loads the normal, final TP/PP shard once. Fresh vLLM workers use
``--load-format ipc_cache`` to create only model structure on ``meta`` and map
these tensors without rereading or reprocessing the checkpoint.

This design is adapted from SGLang's Weight Cache Daemon (Apache-2.0):
https://github.com/sgl-project/sglang/pull/27139
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import pickle
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

from .weight_cache_protocol import (
    WeightCacheFingerprint,
    clean_stale_endpoint,
    control_socket_path,
    pid_alive,
    prepare_namespace,
    receive_message,
    send_message,
    socket_path,
    stable_hash,
    validate_owned_socket,
)

logger = logging.getLogger(__name__)

_MODULE_INTERNAL_ATTRS = {
    "training",
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_modules",
    "_compiled_call_impl",
}


def _simple_metadata(value: Any, *, depth: int = 0) -> Any:
    """Return safe, small post-load metadata or a private sentinel."""
    import torch

    missing = _simple_metadata.missing
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype)):
        return value
    if depth >= 3:
        return missing
    if isinstance(value, (tuple, list)) and len(value) <= 256:
        items = [_simple_metadata(item, depth=depth + 1) for item in value]
        if any(item is missing for item in items):
            return missing
        return type(value)(items)
    if isinstance(value, dict) and len(value) <= 256:
        result = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                return missing
            simple = _simple_metadata(item, depth=depth + 1)
            if simple is missing:
                return missing
            result[key] = simple
        return result
    return missing


_simple_metadata.missing = object()


class WeightCacheServer:
    def __init__(
        self,
        model: Any,
        fingerprint: WeightCacheFingerprint,
        endpoint: Path,
    ) -> None:
        self.model = model
        self.fingerprint = fingerprint
        self.endpoint = endpoint
        self.ready = endpoint.with_suffix(".ready.json")
        self.entries: dict[str, dict[str, Any]] = {}
        self.module_metadata: dict[str, dict[str, Any]] = {}
        self.resident_bytes = 0
        self._storage_ids: set[int] = set()
        self._running = True

    @staticmethod
    def _tensor_payload(tensor: Any) -> dict[str, Any]:
        if tensor.device.type == "cuda":
            from torch.multiprocessing.reductions import (
                rebuild_cuda_tensor,
                reduce_tensor,
            )

            rebuild, args = reduce_tensor(tensor.detach())
            if rebuild is not rebuild_cuda_tensor:
                raise RuntimeError(f"Unexpected reducer {rebuild} for CUDA/HIP tensor")
            return {"transport": "cuda_ipc", "ipc_args": args}
        if tensor.device.type == "cpu":
            # Loaded GPU models occasionally retain small CPU lookup tensors.
            # They are copied in the control manifest; model weights still use
            # zero-copy CUDA/HIP IPC and dominate both size and startup time.
            return {"transport": "cpu", "tensor": tensor.detach().clone()}
        raise RuntimeError(
            f"Cannot export tensor on {tensor.device}; expected CUDA/HIP or CPU"
        )

    def export(self) -> None:
        import torch

        self.entries.clear()
        self.module_metadata.clear()
        self.resident_bytes = 0
        self._storage_ids.clear()
        for module_name, module in self.model.named_modules():
            prefix = f"{module_name}." if module_name else ""
            for leaf, tensor in module._parameters.items():
                if tensor is not None:
                    self._add_tensor(prefix + leaf, tensor, "parameter")
            for leaf, tensor in module._buffers.items():
                if tensor is not None:
                    self._add_tensor(
                        prefix + leaf,
                        tensor,
                        "buffer",
                        persistent=leaf not in module._non_persistent_buffers_set,
                    )

            registered = (
                set(module._parameters) | set(module._buffers) | set(module._modules)
            )
            metadata: dict[str, Any] = {}
            for leaf, value in vars(module).items():
                if leaf in registered or leaf in _MODULE_INTERNAL_ATTRS:
                    continue
                if isinstance(value, torch.Tensor):
                    self._add_tensor(prefix + leaf, value, "attribute")
                    continue
                simple = _simple_metadata(value)
                if simple is not _simple_metadata.missing:
                    metadata[leaf] = simple
            if metadata:
                self.module_metadata[module_name] = metadata

        logger.info(
            "Exported %d parameters, buffers, and tensor attributes",
            len(self.entries),
        )

    def _add_tensor(
        self,
        name: str,
        tensor: Any,
        kind: str,
        *,
        persistent: bool = True,
    ) -> None:
        storage = tensor.untyped_storage()
        storage_id = int(storage._cdata)
        if tensor.device.type == "cuda" and storage_id not in self._storage_ids:
            self._storage_ids.add(storage_id)
            self.resident_bytes += int(storage.nbytes())
        self.entries[name] = self._tensor_payload(tensor) | {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "kind": kind,
            "persistent": persistent,
        }

    def serve(self, *, force: bool = False, on_ready: Any = None) -> None:
        prepare_namespace(self.endpoint.parent)
        clean_stale_endpoint(self.endpoint, force=force)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            listener.bind(str(self.endpoint))
        finally:
            os.umask(old_umask)
        listener.listen(16)
        listener.settimeout(1.0)
        self.ready.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "socket": str(self.endpoint),
                    "fingerprint": self.fingerprint.to_dict(),
                    "tensors": len(self.entries),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.chmod(self.ready, 0o600)
        logger.info("Weight cache ready at %s", self.endpoint)
        if on_ready is not None:
            on_ready()

        def stop(_signum: int, _frame: Any) -> None:
            self._running = False

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            while self._running:
                try:
                    conn, _ = listener.accept()
                except TimeoutError:
                    continue
                with conn:
                    conn.settimeout(30.0)
                    try:
                        self._handle(conn)
                    except Exception:
                        logger.exception("Weight-cache client exchange failed")
        finally:
            listener.close()
            for path in (self.endpoint, self.ready):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _handle(self, conn: socket.socket) -> None:
        request = receive_message(conn)
        kind = request.get("type")
        if kind == "ping":
            send_message(conn, {"status": "ok", "pid": os.getpid()})
            return
        if kind == "stop":
            send_message(conn, {"status": "ok"})
            self._running = False
            return
        if kind != "fetch":
            send_message(conn, {"status": "error", "message": "unknown request"})
            return

        expected = WeightCacheFingerprint.from_dict(request["fingerprint"])
        mismatch = self.fingerprint.mismatch(expected)
        if mismatch:
            send_message(
                conn,
                {
                    "status": "mismatch",
                    "message": f"weight-cache fingerprint mismatch: {mismatch}",
                    "fingerprint": self.fingerprint.to_dict(),
                },
            )
            return
        send_message(
            conn,
            {
                "status": "ok",
                "pid": os.getpid(),
                "fingerprint": self.fingerprint.to_dict(),
                "entries": self.entries,
                "module_metadata": self.module_metadata,
            },
        )


def _assert_ipc_allocator() -> None:
    for variable in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        for allocation_field in os.environ.get(variable, "").split(","):
            key, _, value = allocation_field.partition(":")
            if key.strip() == "expandable_segments" and value.strip().lower() == "true":
                raise RuntimeError(
                    f"{variable}=...expandable_segments:True is incompatible "
                    "with PyTorch CUDA/HIP IPC. Disable it for the daemon."
                )


def _run_rank(
    config_pickle: bytes,
    backing_load_format: str,
    backing_loader_extra_config: dict[str, Any],
    rank: int,
    init_method: str,
    cache_dir: str | None,
    cache_id: str,
    group_key: str,
    parent_pid: int,
    result_queue: Any,
) -> None:
    worker = None
    try:
        _assert_ipc_allocator()
        import torch
        from vllm.config import set_current_vllm_config
        from vllm.plugins import load_general_plugins
        from vllm.v1.worker.gpu_worker import Worker

        load_general_plugins()
        config = pickle.loads(config_pickle)
        config.load_config.load_format = backing_load_format
        config.load_config.model_loader_extra_config = backing_loader_extra_config
        torch.set_num_threads(1)
        # Establish the minimal device context, then measure all persistent
        # memory added by the retained loader (weights, NCCL, and workspaces).
        free_before_worker, _ = torch.cuda.mem_get_info(rank)
        with set_current_vllm_config(config):
            worker = Worker(
                vllm_config=config,
                local_rank=rank,
                rank=rank,
                distributed_init_method=init_method,
                is_driver_worker=rank == 0,
            )
            worker.init_device()

        from .weight_cache_loader import build_fingerprint, current_device_uuid

        device_uuid = current_device_uuid()
        # Quantized loaders may mutate their QuantizationConfig while creating
        # runtime methods and scales. Cache identity describes the immutable
        # client request, so capture it before weight loading/post-processing.
        fingerprint = build_fingerprint(config)
        endpoint = socket_path(
            cache_dir,
            cache_id,
            device_uuid,
            group_key=group_key,
        )
        prepare_namespace(endpoint.parent)
        clean_stale_endpoint(endpoint)
        with set_current_vllm_config(config):
            worker.load_model()
        torch.accelerator.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        model = worker.model_runner.get_model()
        server = WeightCacheServer(model, fingerprint, endpoint)
        server.export()
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize()
        free_after_export, _ = torch.cuda.mem_get_info(
            torch.accelerator.current_device_index()
        )
        allocation_bytes = max(
            server.resident_bytes,
            int(free_before_worker - free_after_export),
        )

        def parent_watch() -> None:
            while server._running:
                time.sleep(2.0)
                if os.getppid() != parent_pid:
                    logger.error("Weight-cache broker exited; stopping rank %d", rank)
                    server._running = False
                    return

        threading.Thread(target=parent_watch, daemon=True).start()

        def report_ready() -> None:
            props = torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            )
            result_queue.put(
                {
                    "status": "ready",
                    "rank": rank,
                    "device_uuid": device_uuid,
                    "socket": str(endpoint),
                    "resident_bytes": server.resident_bytes,
                    "allocation_bytes": allocation_bytes,
                    "total_memory": int(props.total_memory),
                }
            )

        server.serve(on_ready=report_ready)
    except BaseException as error:
        result_queue.put(
            {
                "status": "error",
                "rank": rank,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        if worker is not None:
            worker.shutdown()


def _free_tcp_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{probe.getsockname()[1]}"


@dataclass
class ResidentGroup:
    key: str
    description: str
    processes: list[Any]
    endpoints: dict[str, str]
    resident_bytes: dict[str, int]
    allocation_bytes: dict[str, int]
    total_memory: dict[str, int]
    client_pids: set[int] = field(default_factory=set)
    last_used: float = field(default_factory=time.monotonic)

    def active(self) -> bool:
        self.client_pids = {pid for pid in self.client_pids if pid_alive(pid)}
        return bool(self.client_pids)

    def healthy(self) -> bool:
        return bool(self.processes) and all(proc.is_alive() for proc in self.processes)


class WeightCacheBroker:
    """GPU-light broker that loads exact vLLM shards on the first request."""

    def __init__(
        self,
        *,
        cache_dir: str | None,
        cache_id: str,
        max_cache_fraction: float,
        max_cache_gb_per_gpu: float | None,
        load_timeout: float,
        force: bool,
    ) -> None:
        if not 0 < max_cache_fraction <= 1:
            raise ValueError("max_cache_fraction must be in (0, 1]")
        self.cache_dir = cache_dir
        self.cache_id = cache_id
        self.max_cache_fraction = max_cache_fraction
        self.max_cache_bytes = (
            int(max_cache_gb_per_gpu * 1024**3)
            if max_cache_gb_per_gpu is not None
            else None
        )
        self.load_timeout = load_timeout
        self.force = force
        self.endpoint = control_socket_path(cache_dir, cache_id)
        self.ready = self.endpoint.with_suffix(".ready.json")
        self.groups: dict[str, ResidentGroup] = {}
        self._running = True

    @staticmethod
    def _group_key(request: dict[str, Any]) -> str:
        fingerprint = dict(request["fingerprint"])
        fingerprint.pop("tp_rank", None)
        fingerprint.pop("pp_rank", None)
        factors = {
            "fingerprint": fingerprint,
            "backing_load_format": request["backing_load_format"],
            "backing_loader_extra_config": request["backing_loader_extra_config"],
        }
        return stable_hash(factors)[:24]

    def _start_group(self, request: dict[str, Any], group_key: str) -> ResidentGroup:
        config_pickle = request["vllm_config_pickle"]
        # Read only the simple topology from the authenticated client's
        # fingerprint. The rank children unpickle the exact VllmConfig.
        first = WeightCacheFingerprint.from_dict(request["fingerprint"])
        world_size = first.tp_size * first.pp_size
        if first.dp_size != 1:
            raise RuntimeError("The weight-cache broker currently supports DP=1")
        visible = next(
            (
                os.environ[name]
                for name in (
                    "ROCR_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                )
                if os.environ.get(name)
            ),
            None,
        )
        if visible is not None and world_size > len(visible.split(",")):
            raise RuntimeError(
                f"Requested TP*PP={world_size}, but only {visible} is visible"
            )

        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        init_method = _free_tcp_endpoint()
        parent_pid = os.getpid()
        processes = [
            context.Process(
                target=_run_rank,
                args=(
                    config_pickle,
                    request["backing_load_format"],
                    request["backing_loader_extra_config"],
                    rank,
                    init_method,
                    self.cache_dir,
                    self.cache_id,
                    group_key,
                    parent_pid,
                    result_queue,
                ),
                name=f"vllm-weight-cache-{group_key}-rank-{rank}",
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()

        ready: dict[int, dict[str, Any]] = {}
        deadline = time.monotonic() + self.load_timeout
        try:
            while len(ready) < world_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Only {len(ready)}/{world_size} cache ranks became ready"
                    )
                try:
                    message = result_queue.get(timeout=min(5.0, remaining))
                except Empty:
                    failed = [
                        (process.name, process.exitcode)
                        for process in processes
                        if not process.is_alive()
                    ]
                    if failed:
                        raise RuntimeError(
                            f"Weight-cache rank exited before ready: {failed}"
                        )
                    continue
                if message["status"] == "error":
                    raise RuntimeError(
                        f"Weight-cache rank {message['rank']} failed: "
                        f"{message['error']}"
                    )
                ready[int(message["rank"])] = message
        except BaseException:
            self._stop_processes(processes)
            raise

        description = (
            f"{first.model} dtype={first.dtype} quant={first.quantization or 'none'} "
            f"tp={first.tp_size} pp={first.pp_size}"
        )
        return ResidentGroup(
            key=group_key,
            description=description,
            processes=processes,
            endpoints={msg["device_uuid"]: msg["socket"] for msg in ready.values()},
            resident_bytes={
                msg["device_uuid"]: int(msg["resident_bytes"]) for msg in ready.values()
            },
            allocation_bytes={
                msg["device_uuid"]: int(msg["allocation_bytes"])
                for msg in ready.values()
            },
            total_memory={
                msg["device_uuid"]: int(msg["total_memory"]) for msg in ready.values()
            },
        )

    @staticmethod
    def _stop_processes(processes: list[Any]) -> None:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.kill()

    def _evict(self, key: str, reason: str) -> None:
        group = self.groups.pop(key)
        if group.active():
            self.groups[key] = group
            raise RuntimeError(
                f"Refusing to evict active weight cache {group.description}"
            )
        logger.info("Evicting %s (%s)", group.description, reason)
        self._stop_processes(group.processes)

    def _sweep(self) -> None:
        for key, group in list(self.groups.items()):
            group.active()
            if not group.healthy():
                logger.warning("Removing unhealthy cache group %s", group.description)
                self._stop_processes(group.processes)
                self.groups.pop(key, None)

    def _budget_for(self, group: ResidentGroup, uuid: str) -> int:
        if self.max_cache_bytes is not None:
            return self.max_cache_bytes
        return int(group.total_memory[uuid] * self.max_cache_fraction)

    def _enforce_budget(self, protected_key: str) -> None:
        while True:
            usage: dict[str, int] = {}
            limits: dict[str, int] = {}
            for group in self.groups.values():
                for uuid, size in group.allocation_bytes.items():
                    usage[uuid] = usage.get(uuid, 0) + size
                    limits[uuid] = self._budget_for(group, uuid)
            if not any(usage[uuid] > limits[uuid] for uuid in usage):
                return
            candidates = sorted(
                (
                    group
                    for key, group in self.groups.items()
                    if key != protected_key and not group.active()
                ),
                key=lambda group: group.last_used,
            )
            if not candidates:
                logger.warning(
                    "Weight cache exceeds its budget but no inactive model can "
                    "be evicted; retaining the requested model"
                )
                return
            self._evict(candidates[0].key, "LRU GPU budget")

    def _ensure(self, request: dict[str, Any]) -> dict[str, Any]:
        self._sweep()
        key = self._group_key(request)
        device_uuid = request["device_uuid"]
        before_bytes = sum(
            group.allocation_bytes.get(device_uuid, 0) for group in self.groups.values()
        )
        group = self.groups.get(key)
        cache_hit = group is not None
        if group is None:
            logger.info("Cache miss for %s; loading just in time", key)
            try:
                group = self._start_group(request, key)
            except RuntimeError as error:
                message = str(error).lower()
                inactive = [
                    group for group in self.groups.values() if not group.active()
                ]
                if "memory" not in message or not inactive:
                    raise
                logger.warning("Load ran out of memory; evicting inactive caches")
                for old in sorted(inactive, key=lambda item: item.last_used):
                    self._evict(old.key, "retry after load OOM")
                group = self._start_group(request, key)
            self.groups[key] = group
            self._enforce_budget(key)
        else:
            logger.info("Cache hit for %s", group.description)

        group.client_pids.add(int(request["client_pid"]))
        group.last_used = time.monotonic()
        endpoint = group.endpoints.get(device_uuid)
        if endpoint is None:
            raise RuntimeError(
                f"Requested GPU {device_uuid} is not in the cache daemon's GPU set; "
                f"available GPUs: {sorted(group.endpoints)}"
            )
        after_bytes = sum(
            item.allocation_bytes.get(device_uuid, 0) for item in self.groups.values()
        )
        resident_bytes = group.resident_bytes[device_uuid]
        allocation_bytes = group.allocation_bytes[device_uuid]
        return {
            "status": "ok",
            "group_key": key,
            "socket": endpoint,
            "cache_hit": cache_hit,
            "resident_bytes": resident_bytes,
            "allocation_bytes": allocation_bytes,
            # vLLM naturally observes cache allocations made after its initial
            # snapshot. Only the pre-existing portion needs explicit memory
            # accounting in the fresh worker.
            "profile_correction_bytes": allocation_bytes - (after_bytes - before_bytes),
        }

    def _status(self) -> dict[str, Any]:
        self._sweep()
        return {
            "status": "ok",
            "pid": os.getpid(),
            "groups": [
                {
                    "key": group.key,
                    "description": group.description,
                    "active_client_pids": sorted(
                        pid for pid in group.client_pids if pid_alive(pid)
                    ),
                    "resident_bytes": group.resident_bytes,
                    "allocation_bytes": group.allocation_bytes,
                    "endpoints": group.endpoints,
                }
                for group in self.groups.values()
            ],
        }

    def serve(self) -> None:
        prepare_namespace(self.endpoint.parent)
        clean_stale_endpoint(self.endpoint, force=self.force)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            listener.bind(str(self.endpoint))
        finally:
            os.umask(old_umask)
        listener.listen(128)
        listener.settimeout(1.0)
        self.ready.write_text(
            json.dumps({"pid": os.getpid(), "socket": str(self.endpoint)}) + "\n"
        )
        os.chmod(self.ready, 0o600)
        logger.info(
            "Weight-cache broker ready at %s; models load on first request",
            self.endpoint,
        )

        def stop(_signum: int, _frame: Any) -> None:
            self._running = False

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            while self._running:
                try:
                    conn, _ = listener.accept()
                except TimeoutError:
                    self._sweep()
                    continue
                with conn:
                    conn.settimeout(self.load_timeout + 30.0)
                    try:
                        request = receive_message(conn)
                        kind = request.get("type")
                        if kind == "ensure":
                            response = self._ensure(request)
                        elif kind in {"ping", "status"}:
                            response = self._status()
                        elif kind == "stop":
                            response = {"status": "ok"}
                            self._running = False
                        else:
                            response = {
                                "status": "error",
                                "message": f"unknown request type {kind!r}",
                            }
                    except Exception as error:
                        logger.exception("Weight-cache broker request failed")
                        response = {
                            "status": "error",
                            "message": f"{type(error).__name__}: {error}",
                        }
                    send_message(conn, response)
        finally:
            listener.close()
            for group in list(self.groups.values()):
                self._stop_processes(group.processes)
            for path in (self.endpoint, self.ready):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def _control(action: str, cache_dir: str | None, cache_id: str) -> int:
    endpoint = control_socket_path(cache_dir, cache_id)
    if not validate_owned_socket(endpoint):
        print(f"No weight-cache broker found at {endpoint}")
        return 1
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(10.0)
            client.connect(str(endpoint))
            send_message(client, {"type": action})
            response = receive_message(client)
        print(json.dumps(response, indent=2, sort_keys=True))
        return int(response.get("status") != "ok")
    except Exception as error:  # noqa: BLE001 - status must report every failure
        print(f"{endpoint}: ERROR {error}", file=sys.stderr)
        return 1


def _detach() -> None:
    """Reparent the broker outside vLLM's worker process tree."""
    first = os.fork()
    if first > 0:
        os._exit(0)
    os.setsid()
    second = os.fork()
    if second > 0:
        os._exit(0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", choices=("serve", "status", "stop"), default="serve"
    )
    parser.add_argument("--cache-dir", default=os.environ.get("VLLM_WEIGHT_CACHE_DIR"))
    parser.add_argument(
        "--cache-id", default=os.environ.get("VLLM_WEIGHT_CACHE_ID", "default")
    )
    parser.add_argument("--max-cache-fraction", type=float, default=0.60)
    parser.add_argument("--max-cache-gb-per-gpu", type=float)
    parser.add_argument("--load-timeout", type=float, default=1800.0)
    parser.add_argument("--detach", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.action in {"status", "stop"}:
        raise SystemExit(_control(args.action, args.cache_dir, args.cache_id))
    if args.detach:
        _detach()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    WeightCacheBroker(
        cache_dir=args.cache_dir,
        cache_id=args.cache_id,
        max_cache_fraction=args.max_cache_fraction,
        max_cache_gb_per_gpu=args.max_cache_gb_per_gpu,
        load_timeout=args.load_timeout,
        force=args.force,
    ).serve()


if __name__ == "__main__":
    main()

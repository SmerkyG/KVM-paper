"""Zero-copy vLLM model loader backed by a persistent GPU weight daemon."""

from __future__ import annotations

import copy
import fcntl
import logging
import math
import os
import pickle
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cloudpickle
import torch
from torch import nn

from .weight_cache_protocol import (
    WeightCacheFingerprint,
    cache_namespace,
    canonical_model_name,
    control_socket_path,
    pid_alive,
    prepare_namespace,
    read_ready,
    receive_message,
    send_message,
    stable_hash,
    validate_owned_socket,
)

logger = logging.getLogger(__name__)

_REGISTERED = False
_MEMORY_HOOKS_INSTALLED = False
_LIVENESS_POLL_SECONDS = 5.0
_AUTO_STARTED_BROKERS: list[subprocess.Popen[bytes]] = []


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _broker_is_ready(endpoint: Path) -> bool:
    if not validate_owned_socket(endpoint):
        return False
    ready = read_ready(endpoint.with_suffix(".ready.json"))
    return ready is not None and pid_alive(int(ready.get("pid", 0)))


def ensure_broker_running(
    cache_dir: str | None,
    cache_id: str,
    *,
    auto_start: bool,
    timeout: float,
) -> Path:
    """Return a live broker endpoint, starting one exactly once if needed."""
    endpoint = control_socket_path(cache_dir, cache_id)
    if _broker_is_ready(endpoint):
        return endpoint
    if not auto_start:
        raise RuntimeError(
            f"No vLLM weight-cache broker is running at {endpoint}. Start "
            f"`vllm-weight-cache --cache-id {cache_id}` or enable automatic startup."
        )

    namespace = cache_namespace(cache_dir, cache_id)
    prepare_namespace(namespace)
    lock_path = namespace / "startup.lock"
    lock_fd = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(lock_fd, "r+") as lock_file:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if _broker_is_ready(endpoint):
            return endpoint

        log_path = namespace / "broker.log"
        log_fd = os.open(
            log_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        command = [
            sys.executable,
            "-m",
            "vllm_lod_plugin.weight_cache_daemon",
            "serve",
            "--cache-id",
            cache_id,
            "--detach",
        ]
        if cache_dir:
            command.extend(("--cache-dir", str(cache_dir)))
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parents[1])
        current_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root
            if not current_pythonpath
            else package_root + os.pathsep + current_pythonpath
        )
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                close_fds=True,
                env=environment,
            )
        finally:
            os.close(log_fd)
        _AUTO_STARTED_BROKERS.append(process)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _broker_is_ready(endpoint):
                logger.info(
                    "Auto-started vLLM weight-cache broker pid=%d at %s; log=%s",
                    process.pid,
                    endpoint,
                    log_path,
                )
                return endpoint
            returncode = process.poll()
            if returncode is not None and returncode != 0:
                break
            time.sleep(0.05)

        returncode = process.poll()
        detail = ""
        try:
            detail = log_path.read_text(errors="replace")[-4000:]
        except OSError:
            pass
        raise RuntimeError(
            f"Automatic weight-cache broker startup failed at {endpoint} "
            f"(pid={process.pid}, exit={returncode}, log={log_path}).\n{detail}"
        )


def current_device_uuid(device_index: int | None = None) -> str:
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    return str(torch.cuda.get_device_properties(device_index).uuid)


def _device_arch(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    arch = getattr(props, "gcnArchName", None)
    if arch:
        return str(arch).split(":", 1)[0]
    capability = getattr(props, "major", None), getattr(props, "minor", None)
    if capability[0] is not None:
        return f"{capability[0]}.{capability[1]}"
    return str(getattr(props, "name", ""))


def build_fingerprint(vllm_config: Any) -> WeightCacheFingerprint:
    import vllm
    from vllm.distributed import (
        get_pp_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    model_config = vllm_config.model_config
    parallel = vllm_config.parallel_config
    hf_config = model_config.hf_config
    device_index = torch.accelerator.current_device_index()
    quant_config = vllm_config.quant_config
    if quant_config is None:
        quant_config = getattr(hf_config, "quantization_config", None)
    return WeightCacheFingerprint(
        model=canonical_model_name(model_config.model),
        revision=str(model_config.revision or ""),
        architectures=tuple(getattr(hf_config, "architectures", None) or ()),
        model_hash=str(model_config.compute_hash()),
        dtype=str(model_config.dtype),
        quantization=str(model_config.quantization or ""),
        quantization_hash=stable_hash(quant_config),
        tp_size=int(get_tensor_model_parallel_world_size()),
        tp_rank=int(get_tensor_model_parallel_rank()),
        pp_size=int(parallel.pipeline_parallel_size),
        pp_rank=int(get_pp_group().rank_in_group),
        dp_size=int(parallel.data_parallel_size),
        expert_parallel=bool(parallel.enable_expert_parallel),
        torch_version=str(torch.__version__),
        vllm_version=str(vllm.__version__),
        device_arch=_device_arch(device_index),
    )


def _module_for_name(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    module: nn.Module = model
    for part in parts[:-1]:
        child = getattr(module, part)
        if not isinstance(child, nn.Module):
            raise TypeError(
                f"Weight-cache path {dotted_name!r} crosses non-module {part!r}"
            )
        module = child
    return module, parts[-1]


def _copy_tensor_metadata(source: torch.Tensor, target: torch.Tensor) -> None:
    for key, value in source.__dict__.items():
        # Loader callables and module references come from the freshly-created
        # model. Only overwrite simple post-load facts supplied by the daemon.
        if value is None or isinstance(value, (bool, int, float, str, tuple, list)):
            target.__dict__[key] = value


def _install_tensor(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    kind: str,
    *,
    persistent: bool = True,
) -> None:
    module, leaf = _module_for_name(model, name)
    old_param = module._parameters.get(leaf)
    old_buffer = module._buffers.get(leaf)

    if kind == "parameter":
        if old_param is not None:
            old_type = type(old_param)
            old_attrs = old_param.__dict__.copy()
            try:
                tensor.__class__ = old_type
                tensor.__dict__ = old_attrs
                replacement = tensor
            except (TypeError, RuntimeError):
                replacement = nn.Parameter(tensor, requires_grad=False)
                _copy_tensor_metadata(old_param, replacement)
        else:
            replacement = nn.Parameter(tensor, requires_grad=False)
        if leaf in module._buffers:
            del module._buffers[leaf]
        module._parameters[leaf] = replacement
        return

    if kind == "buffer":
        if leaf in module._parameters:
            del module._parameters[leaf]
        if old_buffer is not None:
            _copy_tensor_metadata(old_buffer, tensor)
        module._buffers[leaf] = tensor
        if persistent:
            module._non_persistent_buffers_set.discard(leaf)
        else:
            module._non_persistent_buffers_set.add(leaf)
        return

    if kind == "attribute":
        if leaf in module._parameters:
            del module._parameters[leaf]
        if leaf in module._buffers:
            del module._buffers[leaf]
            module._non_persistent_buffers_set.discard(leaf)
        object.__setattr__(module, leaf, tensor)
        return

    raise RuntimeError(f"Unknown weight-cache tensor kind {kind!r} for {name}")


def _remove_unexported_meta_tensors(model: nn.Module, exported: set[str]) -> None:
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for leaf, tensor in list(module._parameters.items()):
            name = prefix + leaf
            if tensor is not None and tensor.is_meta and name not in exported:
                del module._parameters[leaf]
        for leaf, tensor in list(module._buffers.items()):
            name = prefix + leaf
            if tensor is not None and tensor.is_meta and name not in exported:
                del module._buffers[leaf]
                module._non_persistent_buffers_set.discard(leaf)


def _remaining_meta_tensors(model: nn.Module) -> list[str]:
    names = [name for name, value in model.named_parameters() if value.is_meta]
    names.extend(name for name, value in model.named_buffers() if value.is_meta)
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for leaf, value in vars(module).items():
            if isinstance(value, torch.Tensor) and value.is_meta:
                names.append(prefix + leaf)
    return sorted(set(names))


def _apply_module_metadata(
    model: nn.Module, metadata: dict[str, dict[str, Any]]
) -> None:
    modules = dict(model.named_modules())
    modules[""] = model
    for module_name, values in metadata.items():
        module = modules.get(module_name)
        if module is None:
            raise RuntimeError(
                f"Weight-cache daemon exported missing module {module_name!r}"
            )
        for name, value in values.items():
            if name not in module._parameters and name not in module._buffers:
                setattr(module, name, value)


def _restore_daemon_runtime_objects(model: nn.Module) -> None:
    """Rebuild small non-tensor kernel objects omitted from CUDA IPC.

    The daemon exports weights *after* vLLM's post-load conversion. Calling
    ``process_weights_after_loading`` again in the client would therefore
    shuffle or quantize those shared tensors a second time. Most runtime state
    is either present on a freshly initialized module or represented by an
    exported tensor, but the ROCm unquantized MoE path constructs a Python
    ``moe_kernel`` object only during post-load processing. Recreate just that
    object around the already-converted daemon-owned weights.
    """

    try:
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            make_unquantized_moe_kernel,
        )
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
    except ImportError:
        return

    restored = 0
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, UnquantizedFusedMoEMethod):
            continue
        if method.moe_kernel is not None:
            continue
        if method.experts_cls is None:
            raise RuntimeError(
                "Weight-cache client cannot restore an unquantized MoE "
                "kernel without its selected experts implementation"
            )
        quant_config = method.get_fused_moe_quant_config(module)
        if quant_config is None:
            raise RuntimeError(
                "Weight-cache client could not reconstruct unquantized MoE "
                "kernel configuration"
            )
        method.moe_quant_config = quant_config
        method.moe_kernel = make_unquantized_moe_kernel(
            quant_config=quant_config,
            moe_config=method.moe,
            backend=method.unquantized_backend,
            experts_cls=method.experts_cls,
            routing_tables=module._expert_routing_tables(),
        )
        restored += 1
    if restored:
        logger.info(
            "Rebuilt %d daemon-backed unquantized MoE runtime kernels", restored
        )


class IPCWeightCacheModelLoader:
    """vLLM loader that replaces meta tensors with daemon-owned IPC mappings."""

    def __init__(self, load_config: Any):
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader

        # Runtime subclassing would obscure loader registration errors. Keep an
        # explicit compatibility check while inheriting is installed below.
        if not isinstance(self, BaseModelLoader):
            raise TypeError("IPCWeightCacheModelLoader registration is incomplete")
        self.load_config = load_config
        extra = load_config.model_loader_extra_config or {}
        if not isinstance(extra, dict):
            raise TypeError("ipc_cache model_loader_extra_config must be a JSON object")
        allowed = {
            "auto_start",
            "cache_dir",
            "cache_id",
            "connect_timeout",
            "broker_timeout",
            "backing_load_format",
            "backing_loader_extra_config",
        }
        unexpected = set(extra) - allowed
        if unexpected:
            raise ValueError(f"Unknown ipc_cache options: {sorted(unexpected)}")
        self.cache_dir = extra.get("cache_dir") or os.environ.get(
            "VLLM_WEIGHT_CACHE_DIR"
        )
        self.cache_id = str(
            extra.get("cache_id") or os.environ.get("VLLM_WEIGHT_CACHE_ID") or "default"
        )
        auto_start = extra.get(
            "auto_start", os.environ.get("VLLM_WEIGHT_CACHE_AUTO_START", "1")
        )
        self.auto_start = _as_bool(auto_start, name="auto_start")
        self.connect_timeout = float(extra.get("connect_timeout", 30.0))
        self.broker_timeout = float(extra.get("broker_timeout", 1800.0))
        self.backing_load_format = str(extra.get("backing_load_format", "auto"))
        self.backing_loader_extra_config = extra.get("backing_loader_extra_config", {})
        if not isinstance(self.backing_loader_extra_config, dict):
            raise TypeError("backing_loader_extra_config must be a JSON object")

    def download_model(self, model_config: Any) -> None:
        return None

    def load_weights(self, model: nn.Module, model_config: Any) -> None:
        raise NotImplementedError("ipc_cache binds complete post-load model state")

    def load_model(
        self, vllm_config: Any, model_config: Any, prefix: str = ""
    ) -> nn.Module:
        # Parallel drafters are nested model loads: vLLM deliberately passes a
        # draft ModelConfig while retaining the target VllmConfig (the draft
        # constructor uses its speculative_config to share target weights).
        # A daemon Worker can only load vllm_config.model_config as its primary
        # model, so caching this nested call under the target fingerprint would
        # return the target module tree. Keep the large target daemon-backed
        # and use vLLM's ordinary loader for the much smaller nested drafter.
        if model_config is not vllm_config.model_config:
            from vllm.model_executor.model_loader.default_loader import (
                DefaultModelLoader,
            )

            load_config = copy.copy(vllm_config.load_config)
            load_config.load_format = "auto"
            load_config.model_loader_extra_config = {}
            return DefaultModelLoader(load_config).load_model(
                vllm_config=vllm_config,
                model_config=model_config,
                prefix=prefix,
            )
        from torch.multiprocessing.reductions import rebuild_cuda_tensor
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.utils.torch_utils import set_default_torch_dtype

        started = time.perf_counter()
        device_index = torch.accelerator.current_device_index()
        expected = build_fingerprint(vllm_config)
        endpoint, cache_info = self._ensure_resident(
            vllm_config,
            expected,
            current_device_uuid(device_index),
        )
        if not validate_owned_socket(endpoint):
            raise RuntimeError(
                f"Weight-cache broker returned an unavailable shard socket: {endpoint}"
            )

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.connect_timeout)
            client.connect(str(endpoint))
            send_message(
                client,
                {"type": "fetch", "fingerprint": expected.to_dict()},
            )
            response = receive_message(client)
        if response.get("status") != "ok":
            raise RuntimeError(
                "Weight-cache daemon rejected this engine: "
                f"{response.get('message', response)}"
            )

        with (
            set_default_torch_dtype(model_config.dtype),
            torch.device("meta"),
            set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix),
        ):
            model = initialize_model(
                vllm_config=vllm_config,
                model_config=model_config,
                prefix=prefix,
            )

        entries: dict[str, dict[str, Any]] = response["entries"]
        imported: list[torch.Tensor] = []
        for name, entry in entries.items():
            if entry["transport"] == "cuda_ipc":
                args = list(entry["ipc_args"])
                # PyTorch reduce_tensor's CUDA/HIP reconstruction tuple stores
                # the producer's logical device index here. The physical GPU
                # is the same (the socket is UUID-keyed), but visibility may
                # renumber it.
                args[6] = device_index
                tensor = rebuild_cuda_tensor(*args)
                imported.append(tensor)
            elif entry["transport"] == "cpu":
                tensor = entry["tensor"]
            else:
                raise RuntimeError(
                    f"Unknown weight-cache transport for {name}: {entry['transport']!r}"
                )
            if tuple(tensor.shape) != tuple(entry["shape"]):
                raise RuntimeError(
                    f"IPC reconstruction changed shape for {name}: "
                    f"{tuple(tensor.shape)} != {tuple(entry['shape'])}"
                )
            _install_tensor(
                model,
                name,
                tensor,
                entry["kind"],
                persistent=bool(entry.get("persistent", True)),
            )

        _remove_unexported_meta_tensors(model, set(entries))
        _apply_module_metadata(model, response.get("module_metadata", {}))
        remaining = _remaining_meta_tensors(model)
        if remaining:
            raise RuntimeError(
                "Weight-cache mapping left tensors on the meta device: "
                + ", ".join(remaining[:20])
            )
        _restore_daemon_runtime_objects(model)

        model._vllm_weight_cache_imports = imported
        model._vllm_weight_cache_endpoint = str(endpoint)
        model._vllm_weight_cache_resident_bytes = int(cache_info["resident_bytes"])
        model._vllm_weight_cache_profile_correction_bytes = int(
            cache_info["profile_correction_bytes"]
        )
        self._watch_daemon(int(response["pid"]), endpoint)
        logger.info(
            "Mapped %d daemon-owned tensors from %s in %.3fs",
            len(entries),
            endpoint,
            time.perf_counter() - started,
        )
        return model.eval()

    def _ensure_resident(
        self,
        vllm_config: Any,
        fingerprint: WeightCacheFingerprint,
        device_uuid: str,
    ) -> tuple[Path, dict[str, Any]]:
        control = ensure_broker_running(
            self.cache_dir,
            self.cache_id,
            auto_start=self.auto_start,
            timeout=self.connect_timeout,
        )
        # A draft model is loaded after the target has populated this field
        # with live attention modules.  Those modules are neither part of the
        # model-construction configuration nor picklable (some contain RLocks).
        # Send the daemon a shallowly detached compilation config just as it
        # would have seen during the initial target load.
        daemon_config = copy.copy(vllm_config)
        compilation_config = copy.copy(vllm_config.compilation_config)
        compilation_config.static_forward_context = {}
        daemon_config.compilation_config = compilation_config
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.broker_timeout)
            client.connect(str(control))
            send_message(
                client,
                {
                    "type": "ensure",
                    "client_pid": os.getpid(),
                    "device_uuid": device_uuid,
                    "fingerprint": fingerprint.to_dict(),
                    # ModelConfig permits callable hf_overrides. Benchmark and
                    # embedding applications commonly define those callbacks
                    # in __main__, which ordinary pickle can serialize by name
                    # but the daemon's spawned rank cannot import. Cloudpickle
                    # embeds such callbacks while remaining readable by the
                    # daemon's ordinary pickle.loads.
                    "vllm_config_pickle": cloudpickle.dumps(
                        daemon_config, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                    "backing_load_format": self.backing_load_format,
                    "backing_loader_extra_config": self.backing_loader_extra_config,
                },
            )
            response = receive_message(client)
        if response.get("status") != "ok":
            raise RuntimeError(
                "Weight-cache broker could not make the model resident: "
                f"{response.get('message', response)}"
            )
        endpoint = Path(response["socket"])
        namespace = cache_namespace(self.cache_dir, self.cache_id).resolve()
        if namespace not in endpoint.resolve().parents:
            raise RuntimeError(
                f"Weight-cache broker returned an endpoint outside {namespace}: "
                f"{endpoint}"
            )
        return endpoint, response

    @staticmethod
    def _watch_daemon(pid: int, endpoint: Path) -> None:
        def watch() -> None:
            while True:
                time.sleep(_LIVENESS_POLL_SECONDS)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    logger.critical(
                        "Weight-cache daemon %d for %s exited; mapped model "
                        "storage is no longer safe. Terminating this worker.",
                        pid,
                        endpoint,
                    )
                    os.kill(os.getpid(), signal.SIGKILL)
                    return
                except PermissionError:
                    pass

        threading.Thread(
            target=watch,
            name="vllm-weight-cache-watchdog",
            daemon=True,
        ).start()


def _uses_ipc_cache(worker: Any) -> bool:
    return str(worker.load_config.load_format).lower() == "ipc_cache"


def install_weight_cache_memory_hooks() -> None:
    """Teach vLLM's memory profiler about daemon-owned model allocations.

    A warm daemon allocation predates the worker's initial memory snapshot, so
    upstream vLLM classifies it as another process and both rejects ordinary
    utilization settings and omits the weights from the KV budget. Keep the
    upstream methods intact, but scope those two corrections to ``ipc_cache``.
    """

    global _MEMORY_HOOKS_INSTALLED
    if _MEMORY_HOOKS_INSTALLED:
        return

    from vllm.v1.worker.gpu_worker import Worker

    original_init_device = Worker.init_device
    original_determine_available_memory = Worker.determine_available_memory

    def init_device(worker: Any) -> None:
        if not _uses_ipc_cache(worker):
            original_init_device(worker)
            return

        requested_utilization = worker.cache_config.gpu_memory_utilization
        # The exact requested byte count is restored immediately below. A tiny
        # temporary value only bypasses the upstream free-memory check, which
        # cannot know that the resident daemon allocation belongs to this
        # engine and will be mapped zero-copy.
        worker.cache_config.gpu_memory_utilization = min(requested_utilization, 0.001)
        try:
            original_init_device(worker)
        finally:
            worker.cache_config.gpu_memory_utilization = requested_utilization
        worker.requested_memory = math.ceil(
            worker.init_snapshot.total_memory * requested_utilization
        )

    def determine_available_memory(worker: Any) -> int:
        rebase_bytes = 0
        correction = 0
        if _uses_ipc_cache(worker):
            model = worker.model_runner.get_model()
            correction = int(
                getattr(model, "_vllm_weight_cache_profile_correction_bytes", 0)
            )
            current_free, _ = torch.accelerator.get_memory_info(worker.device)
            initial_free = int(worker.init_snapshot.free_memory)
            if current_free > initial_free:
                # A cold cache miss can evict an older inactive daemon model
                # after vLLM takes its initial snapshot.  That legitimately
                # increases free memory and otherwise trips vLLM's guard
                # against unrelated processes changing allocation during the
                # profile.  The broker's correction is the exact expected
                # pre-existing allocation, so only rebase a release covered
                # by that correction; a larger change remains an error.
                rebase_bytes = int(current_free - initial_free)
                if rebase_bytes <= correction:
                    worker.init_snapshot.free_memory = int(current_free)
                    worker.init_snapshot.cuda_memory = int(
                        worker.init_snapshot.total_memory - current_free
                    )
                    worker.init_snapshot.non_torch_memory = int(
                        worker.init_snapshot.cuda_memory
                        - worker.init_snapshot.torch_memory
                    )
                    logger.info(
                        "Rebased vLLM's memory profile by %.3f GiB after "
                        "the weight-cache broker evicted an inactive model",
                        rebase_bytes / 1024**3,
                    )
                else:
                    rebase_bytes = 0
        available = int(original_determine_available_memory(worker))
        if (
            not _uses_ipc_cache(worker)
            or worker.cache_config.kv_cache_memory_bytes is not None
        ):
            return available

        # Rebasing made the broker's expected release part of the baseline;
        # remove the same bytes from its explicit correction to avoid counting
        # the evicted allocation twice.
        correction -= rebase_bytes
        if correction == 0:
            return available
        corrected = available - correction
        if corrected <= 0:
            raise RuntimeError(
                "The daemon-backed model leaves no memory for vLLM's KV cache "
                "at the requested gpu_memory_utilization; stop active engines "
                "or lower the cache daemon's retained-model budget"
            )
        worker.available_kv_cache_memory_bytes = corrected
        worker.total_consumed += correction
        logger.info(
            "Accounted for %.3f GiB of daemon-owned model memory that was "
            "resident before the vLLM worker memory snapshot",
            correction / 1024**3,
        )
        return corrected

    Worker.init_device = init_device
    Worker.determine_available_memory = determine_available_memory
    _MEMORY_HOOKS_INSTALLED = True


def register_weight_cache_loader() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    # Keep the implementation importable in CPU-only protocol tests while
    # satisfying vLLM's explicit BaseModelLoader subclass contract at runtime.
    loader_cls = type(
        "IPCWeightCacheModelLoader",
        (IPCWeightCacheModelLoader, BaseModelLoader),
        {},
    )
    register_model_loader("ipc_cache")(loader_cls)
    install_weight_cache_memory_hooks()
    _REGISTERED = True


def register() -> None:
    """General-plugin entry point for weight-cache-only vLLM runs."""
    from .dflash2_compat import register_dflash2_compat
    from .model_compat import register_k2_horizon

    register_dflash2_compat()
    register_k2_horizon()
    register_weight_cache_loader()


__all__ = [
    "IPCWeightCacheModelLoader",
    "build_fingerprint",
    "cache_namespace",
    "current_device_uuid",
    "install_weight_cache_memory_hooks",
    "register",
    "register_weight_cache_loader",
]

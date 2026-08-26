"""Deterministic teardown for vLLM offline engines used by local runners.

vLLM's ``LLM`` facade does not expose a public shutdown method. Its
multiprocess engine normally closes through a finalizer, but an unhandled
exception can keep the facade reachable while Python waits for the EngineCore
child. Register the underlying client's idempotent shutdown as soon as an
``LLM`` has initialized, and also make it available to runner ``finally``
blocks so failures cannot leave cluster jobs or GPU allocations behind.
"""

from __future__ import annotations

import atexit
import logging
import threading
from collections.abc import Callable
from typing import TypeVar


logger = logging.getLogger(__name__)
LLMType = TypeVar("LLMType")


class _ShutdownOnce:
    def __init__(self, shutdown: Callable[[], None]) -> None:
        self._shutdown = shutdown
        self._lock = threading.Lock()
        self._done = False

    def __call__(self) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        try:
            self._shutdown()
        except BaseException:
            # Cleanup must not replace the benchmark or validation exception
            # that caused this path to run.
            logger.exception("vLLM EngineCore shutdown failed during runner teardown")


_registry_lock = threading.Lock()
_registered: list[_ShutdownOnce] = []


def register_llm_shutdown(llm: LLMType) -> LLMType:
    """Register deterministic shutdown for an initialized ``vllm.LLM``."""

    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if not callable(shutdown):
        raise RuntimeError("vLLM LLM has no callable EngineCore shutdown method")
    callback = _ShutdownOnce(shutdown)
    with _registry_lock:
        _registered.append(callback)
    atexit.register(callback)
    return llm


def shutdown_registered_llms() -> None:
    """Close all registered engines in reverse construction order."""

    with _registry_lock:
        callbacks = list(reversed(_registered))
        _registered.clear()
    for callback in callbacks:
        callback()
        atexit.unregister(callback)


__all__ = ["register_llm_shutdown", "shutdown_registered_llms"]

"""Keep engine-factory environment defaults local to each test."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_kimi_layout_defaults(monkeypatch):
    from benchmarks import _vllm

    monkeypatch.setattr(_vllm, "_automatic_kimi_owner_environment", {})
    # The factory sets these directly rather than through pytest's monkeypatch.
    # Register their original states for teardown, like independent processes.
    for name in (
        "LOD_KIMI_REQUEST_OWNER_PREFILL", "LOD_KIMI_REQUEST_OWNER_DECODE",
        "LOD_KIMI_OWNER_QUERY_CHUNK", "LOD_BENCHMARK_PREFILL_COHORT",
        "LOD_KIMI_OWNER_MOE_CHUNK", "LOD_KIMI_OWNER_REUSE_TRANSPORT",
        "LOD_KIMI_OWNER_SHARD_RESIDUAL", "LOD_KIMI_OWNER_POOL_BACKED_PREFILL",
        "LOD_KIMI_OWNER_PREFILL_HEAD_GROUP", "LOD_KIMI_OWNER_PRESSURE_CHECK",
    ):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)

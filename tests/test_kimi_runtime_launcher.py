"""Check the launcher's compiler paths without starting or copying the image."""

import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "benchmarks/run_kimi_k3_v10_direct.sh"
CACHES = (
    "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR",
    "VLLM_CACHE_ROOT", "AITER_JIT_DIR", "FLYDSL_RUNTIME_CACHE_DIR",
    "FLYDSL_AUTOTUNE_CACHE_DIR",
)


def cache_environment(overrides=None):
    script = LAUNCHER.read_text()
    exports = script[script.index("export TRITON_CACHE_DIR="):script.index("mkdir -p")]
    # Isolate shell startup and multiline environment values from the host.
    env = {"PATH": os.defpath}
    env.update(overrides or {})
    values = " ".join(f'"${{{name}}}"' for name in CACHES)
    result = subprocess.check_output(
        ["bash", "--noprofile", "--norc", "-c", exports + "\nprintf '%s\\0' " + values],
        env=env, text=True,
    )
    return dict(zip(CACHES, result.split("\0")[:-1], strict=True))


def test_compilation_defaults_are_local():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    env = cache_environment()
    assert all(env[name].startswith("/tmp/dan-agent/") for name in CACHES)
    assert env["FLYDSL_RUNTIME_CACHE_DIR"] == env["AITER_JIT_DIR"] + "/flydsl_cache"


@pytest.mark.parametrize("name", CACHES)
def test_explicit_compilation_cache_override_is_preserved(name, tmp_path):
    path = str(tmp_path / name)
    env = cache_environment({name: path})
    assert env[name] == path
    if name == "AITER_JIT_DIR":
        assert env["FLYDSL_RUNTIME_CACHE_DIR"] == path + "/flydsl_cache"

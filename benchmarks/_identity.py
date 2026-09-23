"""Reproducible source and runtime identity for benchmark artifacts."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    "benchmarks",
    "integrations/vllm_lod",
    "lod_attention",
)
SOURCE_FILES = ("pyproject.toml", "uv.lock")
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hip",
    ".hpp",
    ".patch",
    ".py",
    ".toml",
}
PACKAGE_NAMES = ("aiter", "torch", "transformers", "triton", "vllm")


def _source_files(root: Path) -> list[Path]:
    files = [root / relative for relative in SOURCE_FILES]
    for relative in SOURCE_ROOTS:
        directory = root / relative
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def source_identity(root: Path = ROOT) -> dict[str, Any]:
    """Hash the exact local sources that can affect the benchmark result."""
    digest = hashlib.sha256()
    files = _source_files(root)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"benchmark identity source is missing: {missing[0]}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "source_sha256": digest.hexdigest(),
        "source_file_count": len(files),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def benchmark_identity(root: Path = ROOT) -> dict[str, Any]:
    """Return the strict identity required for real-minus-dummy pairing."""
    return {
        "schema": 1,
        "source": source_identity(root),
        "runtime": {
            "python": platform.python_version(),
            "packages": {name: _package_version(name) for name in PACKAGE_NAMES},
        },
    }


__all__ = ["benchmark_identity", "source_identity"]

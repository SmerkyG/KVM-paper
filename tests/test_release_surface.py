from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ENV = {
    "VLLM_LOD_MODE",
    "VLLM_LOD_POOL_SIZE",
    "VLLM_LOD_MAX_CONTEXT",
}


def test_no_research_tuning_environment_surface() -> None:
    roots = (ROOT / "lod_attention", ROOT / "integrations" / "vllm_lod")
    names: set[str] = set()
    for root in roots:
        for path in root.rglob("*.py"):
            names.update(
                re.findall(r"(?:VLLM_LOD|LOD_DEV)_[A-Z0-9_]+", path.read_text())
            )
    assert names == PUBLIC_ENV


def test_removed_model_and_weight_cache_adapters_stay_removed() -> None:
    plugin = ROOT / "integrations" / "vllm_lod" / "vllm_lod_plugin"
    assert not (plugin / "muse_glimmer.py").exists()
    assert not (plugin / "weight_cache_daemon.py").exists()
    assert not (plugin / "weight_cache_loader.py").exists()
    assert not (plugin / "weight_cache_protocol.py").exists()


def test_dflash2_is_isolated_from_the_attention_engine() -> None:
    dflash = (
        ROOT
        / "integrations"
        / "vllm_lod"
        / "vllm_lod_plugin"
        / "models"
        / "dflash2.py"
    )
    assert dflash.exists()
    for path in (ROOT / "lod_attention").rglob("*.py"):
        assert "DFlash" not in path.read_text()

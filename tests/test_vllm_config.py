from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from lod_attention._config import LODMode, ModelFamily


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations" / "vllm_lod"))
config_module = importlib.import_module("vllm_lod_plugin.config")
VLLMLODSettings = config_module.VLLMLODSettings
validate_production_scheduler = config_module.validate_production_scheduler


def test_environment_exposes_only_mode_and_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_LOD_MODE", "three-tier-int4")
    monkeypatch.setenv("VLLM_LOD_POOL_SIZE", "12")
    monkeypatch.setenv("VLLM_LOD_MAX_CONTEXT", "65536")
    settings = VLLMLODSettings.from_environment()
    assert settings.mode is LODMode.THREE_TIER_INT4
    assert settings.pool_size == 12
    assert settings.request_capacity == 65_536


def test_removed_tuning_flag_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_LOD_OPEN_COUNT", "8")
    with pytest.raises(ValueError, match="has no tuning flags"):
        VLLMLODSettings.from_environment()


def test_family_resolution_changes_launch_geometry_not_policy() -> None:
    base = VLLMLODSettings.production(mode="two-tier")
    qwen = base.for_family(ModelFamily.QWEN38)
    k2 = base.for_family(ModelFamily.K2)
    assert qwen.decode_gqa_fixed_mask_aiter is True
    assert k2.decode_gqa_fixed_mask_aiter is False
    assert qwen.mode is k2.mode is LODMode.TWO_TIER


def test_scheduler_cannot_slice_the_production_prefill() -> None:
    with pytest.raises(RuntimeError, match="max-num-batched-tokens"):
        validate_production_scheduler(
            max_model_len=65_536,
            max_num_batched_tokens=8_192,
            long_prefill_token_threshold=0,
        )
    validate_production_scheduler(
        max_model_len=65_536,
        max_num_batched_tokens=16_384,
        long_prefill_token_threshold=0,
    )


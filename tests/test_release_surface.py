from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ENV = {
    "VLLM_LOD_MODE",
    "VLLM_LOD_POOL_SIZE",
    "VLLM_LOD_MAX_CONTEXT",
}


def test_exact_decode_is_limited_to_two_thousand_tokens() -> None:
    from lod_attention._config import (
        EXACT_DECODE_LIMIT,
        PREFILL_CHUNK_SIZE,
    )

    assert EXACT_DECODE_LIMIT == 2_048
    assert PREFILL_CHUNK_SIZE == 16_384


def test_no_research_tuning_environment_surface() -> None:
    roots = (ROOT / "lod_attention", ROOT / "integrations" / "vllm_lod")
    names: set[str] = set()
    for root in roots:
        for path in root.rglob("*.py"):
            names.update(
                re.findall(
                    r"(?:VLLM_LOD|LOD_DEV|LOD_RELEASE_TEST)_[A-Z0-9_]+",
                    path.read_text(),
                )
            )
    assert names == PUBLIC_ENV


def test_removed_model_and_weight_cache_adapters_stay_removed() -> None:
    plugin = ROOT / "integrations" / "vllm_lod" / "vllm_lod_plugin"
    assert not (plugin / "muse_glimmer.py").exists()
    assert not (plugin / "weight_cache_daemon.py").exists()
    assert not (plugin / "weight_cache_loader.py").exists()
    assert not (plugin / "weight_cache_protocol.py").exists()


def test_experimental_kernel_families_stay_removed() -> None:
    kernels = ROOT / "lod_attention" / "kernels"
    csrc = ROOT / "lod_attention" / "csrc"
    for name in (
        "centroid_major_route_score.py",
        "gqa16_coarse_score.py",
    ):
        assert not (kernels / name).exists()
    assert (kernels / "aiter_prefill_attention.py").exists()
    assert not (csrc / "centroid_major_route_score").exists()
    assert not (csrc / "gqa16_coarse_score").exists()


def test_aiter_route_workspace_is_tight_for_k2_and_safe_for_qwen() -> None:
    patch = (
        ROOT
        / "integrations"
        / "vllm_lod"
        / "patches"
        / "aiter-mha-prefill-route4.patch"
    ).read_text()
    assert "head_size_q == 128 ? 128 : 64" in patch
    assert "D=256 can dispatch either a 64- or 128-key CK tile" in patch
    assert "kQKHeaddim == 128 ? index_t{128} : index_t{64}" in patch
    assert "variant_params.route_seqlen_k, route_storage_tile" in patch
    assert "elif receipt in (100, 101)" in patch
    assert 'receipt == 101 and dtype == "bf16"' in patch
    assert "coarse_score + (query_rms - 1.0f) * bias_value" in patch

    source = (
        ROOT / "lod_attention" / "kernels" / "aiter_prefill_attention.py"
    ).read_text()
    assert 'replace("--receipt 100", "--receipt 101")' in source
    assert 'revision = "_d128w8_mulnorm_v1"' in source


def test_aiter_state_preparation_pads_non_power_of_two_gqa() -> None:
    source = (
        ROOT / "lod_attention" / "kernels" / "aiter_prefill_attention.py"
    ).read_text()
    assert "group = tl.arange(0, BLOCK_G)" in source
    assert "mask=valid_group" in source
    launch = source.split(
        "_prepare_aiter_state_kernel[(batch * kv_heads * dispatch_state_len,)](",
        maxsplit=1,
    )[1].split("\n    )", maxsplit=1)[0]
    assert "BLOCK_G=triton.next_power_of_2(kv_group_size)" in launch


def test_paged_kernel_entrypoint_stays_a_small_facade() -> None:
    facade = ROOT / "lod_attention" / "kernels" / "paged_leaf_attention.py"
    assert len(facade.read_text().splitlines()) < 100


def test_dflash2_is_isolated_from_the_attention_engine() -> None:
    dflash = (
        ROOT / "integrations" / "vllm_lod" / "vllm_lod_plugin" / "models" / "dflash2.py"
    )
    assert dflash.exists()
    for path in (ROOT / "lod_attention").rglob("*.py"):
        assert "DFlash" not in path.read_text()


def test_cross_layer_prefill_supports_every_release_cache_mode() -> None:
    plugin = ROOT / "integrations" / "vllm_lod" / "vllm_lod_plugin"
    pool = (plugin / "pool.py").read_text()
    runtime = (plugin / "runtime.py").read_text()

    assert pool.count("self.settings.levels in (2, 3)") >= 2
    assert pool.count("self.settings.kv_bits in (0, 4)") >= 2
    assert "staged_leaves: tuple[torch.Tensor, torch.Tensor] | None" in pool
    assert "_initial_prefill_sources" in runtime
    assert "_cached_prefill_sources" in runtime
    assert "tensor.record_stream(stream)" in runtime
    cached_builder = runtime.split(
        "def _build_cached_prefill_across_layers", 1
    )[1].split("def _catch_up_decode_rows", 1)[0]
    assert '"overflow_safe_until"' not in cached_builder

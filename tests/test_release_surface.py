from __future__ import annotations

import ast
from collections import Counter
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ENV = {
    "VLLM_LOD_ENABLED",
    "VLLM_LOD_MODE",
    "VLLM_LOD_POOL_SIZE",
    "VLLM_LOD_MAX_CONTEXT",
}

# vLLM finds these through fully qualified names or plugin registration rather
# than ordinary Python references.
DYNAMIC_ENTRY_POINTS = {
    "LODAttentionBackend",
    "LODChunkAlignedScheduler",
    "K2HorizonForCausalLM",
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


def test_removed_model_adapters_stay_removed() -> None:
    plugin = ROOT / "integrations" / "vllm_lod" / "vllm_lod_plugin"
    assert not (plugin / "muse_glimmer.py").exists()


def test_weight_cache_release_surface_is_complete() -> None:
    plugin = ROOT / "integrations" / "vllm_lod" / "vllm_lod_plugin"
    for name in (
        "weight_cache_daemon.py",
        "weight_cache_loader.py",
        "weight_cache_protocol.py",
    ):
        assert (plugin / name).is_file()


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


def test_release_python_symbols_have_a_consumer() -> None:
    roots = (
        ROOT / "lod_attention",
        ROOT / "integrations" / "vllm_lod" / "vllm_lod_plugin",
        ROOT / "benchmarks",
        ROOT / "examples",
    )
    references: Counter[str] = Counter()
    definitions: list[tuple[Path, int, str]] = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    references[node.id] += 1
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    references[node.attr] += 1
                elif isinstance(node, ast.ImportFrom):
                    for imported in node.names:
                        references[imported.name] += 1
            definitions.extend(
                (path, node.lineno, node.name)
                for node in tree.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            )

    unreachable = [
        f"{path.relative_to(ROOT)}:{line}: {name}"
        for path, line, name in definitions
        if not name.startswith("__")
        and name not in DYNAMIC_ENTRY_POINTS
        and references[name] == 0
    ]
    assert unreachable == []


def test_removed_release_ablation_surfaces_stay_removed() -> None:
    roots = (ROOT / "lod_attention", ROOT / "benchmarks", ROOT / "tests")
    source = "\n".join(
        path.read_text()
        for root in roots
        for path in root.rglob("*.py")
        if path != Path(__file__)
    )
    for vestige in (
        "compact_top4_candidates",
        "EXACT_TOP4",
        "FLOAT_TOP4",
        "prefill_exact_mass_coverage",
        "prefill_route_key_spread",
        "prefill_route_exclude_singleton",
    ):
        assert vestige not in source
    assert not (
        ROOT / "lod_attention" / "kernels" / "gqa_cooperative_decode.py"
    ).exists()
    assert not (
        ROOT
        / "lod_attention"
        / "csrc"
        / "gqa_cooperative_decode"
        / "gqa_cooperative_decode.cu"
    ).exists()


def test_aiter_route_workspace_is_tight_for_k2_and_safe_for_qwen() -> None:
    patch = (
        ROOT
        / "integrations"
        / "vllm_lod"
        / "patches"
        / "aiter-mha-prefill-route8.patch"
    ).read_text()
    assert "(head_size_q == 128 || head_size_q == 192) ? 128 : 64" in patch
    assert "release D=128 and Kimi MLA" in patch
    assert "D=256 can dispatch either a 64- or 128-key CK tile" in patch
    assert "(kQKHeaddim == 128 || kQKHeaddim == 192)" in patch
    assert "variant_params.route_seqlen_k, route_storage_tile" in patch
    assert "elif receipt in (100, 101)" in patch
    assert 'receipt == 101 and dtype == "bf16"' in patch
    assert "coarse_score + (query_rms - 1.0f) * bias_value" in patch
    # The route-only kernel shares the query tensor with the concurrent coarse
    # pass as its ABI-required output argument.  It must never run the normal
    # output epilogue, or the two streams race and routing becomes nondeterministic.
    assert patch.count("if(!kargs.route_only)") >= 2
    assert patch.count(
        "EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);"
    ) >= 4

    source = (
        ROOT / "lod_attention" / "kernels" / "aiter_prefill_attention.py"
    ).read_text()
    assert 'replace("--receipt 100", "--receipt 101")' in source
    assert 'revision = "_d128w8_mulnorm_v1"' in source
    assert '"_tile128v2_noepilogue" if head_dim == 192' in source

    kimi_source = (
        ROOT / "lod_attention" / "kernels" / "aiter_mla_prefill_attention.py"
    ).read_text()
    assert "fused_route_coarse = True" in kimi_source
    assert "LOD_KIMI_FUSED_ROUTE_COARSE" not in kimi_source


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
    cached_builder = runtime.split("def _build_cached_prefill_across_layers", 1)[
        1
    ].split("def _catch_up_decode_rows", 1)[0]
    assert '"overflow_safe_until"' not in cached_builder
    assert "cross-layer cached construction requires B=1" not in cached_builder
    assert "row_begin = group_row * row_count" in cached_builder
    assert "state_k=packed_k[row_begin:row_end]" in cached_builder
    assert "allow_cross_layer_cached=len(groups) == 1" in pool
    assert "self._batched_dcp_shadow(slots)" in pool

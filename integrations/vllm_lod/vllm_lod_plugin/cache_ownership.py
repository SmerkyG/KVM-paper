"""vLLM cache-spec hooks for authoritative LOD storage."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from .metadata_cache import (
    LODMetadataOnlyFullAttentionManager,
    LODMetadataOnlyFullAttentionSpec,
    is_metadata_only_spec,
)

logger = logging.getLogger(__name__)


def _install_attention_spec_hook() -> None:
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    if getattr(Attention, "_vllm_lod_cache_spec_installed", False):
        return
    original = Attention.get_kv_cache_spec

    def get_kv_cache_spec(self: Any, vllm_config: Any) -> Any:
        spec = original(self, vllm_config)
        impl = getattr(self, "impl", None)
        if not bool(getattr(impl, "lod_eligible", False)):
            return spec
        if not isinstance(spec, FullAttentionSpec):
            raise RuntimeError(
                "LOD-eligible custom attention produced a non-full cache spec "
                f"({type(spec).__name__}); refusing to fall back to native K/V"
            )
        if int(self.head_size) != int(self.head_size_v):
            raise NotImplementedError(
                "authoritative LOD cache requires equal key and value widths"
            )
        self._vllm_lod_external_kv_cache = True
        # Keep the original full-attention topology visible to the scheduler in
        # every model.  Its manager maintains logical block/hash metadata while
        # core sizing and worker initialization allocate no native K/V storage.
        self._vllm_lod_external_scheduler_cache = True
        return LODMetadataOnlyFullAttentionSpec(
            **{
                field: getattr(spec, field)
                for field in LODMetadataOnlyFullAttentionSpec.__dataclass_fields__
            }
        )

    Attention.get_kv_cache_spec = get_kv_cache_spec
    Attention._vllm_lod_cache_spec_installed = True


def _install_metadata_spec_manager_hook() -> None:
    """Dispatch the LOD spec to its virtual manager without mutating vLLM."""
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    if getattr(KVCacheSpecRegistry, "_vllm_lod_manager_installed", False):
        return
    original = KVCacheSpecRegistry.get_manager_class.__func__

    def get_manager_class(cls: Any, spec: Any) -> Any:
        if isinstance(spec, LODMetadataOnlyFullAttentionSpec):
            return LODMetadataOnlyFullAttentionManager
        return original(cls, spec)

    KVCacheSpecRegistry.get_manager_class = classmethod(get_manager_class)
    KVCacheSpecRegistry._vllm_lod_manager_installed = True


def _physical_groups(groups: list[Any]) -> list[Any]:
    """Return cache groups that actually own worker-side GPU pages."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    physical = []
    for group in groups:
        spec = group.kv_cache_spec
        if is_metadata_only_spec(spec):
            continue
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec_names = set(spec.kv_cache_specs)
            group_names = set(group.layer_names)
            if spec_names != group_names:
                raise RuntimeError(
                    "KV cache group layer names do not match its per-layer specs: "
                    f"group_only={sorted(group_names - spec_names)}, "
                    f"spec_only={sorted(spec_names - group_names)}"
                )
            kept = {
                name: inner
                for name, inner in spec.kv_cache_specs.items()
                if not isinstance(inner, LODMetadataOnlyFullAttentionSpec)
            }
            if not kept:
                continue
            spec = UniformTypeKVCacheSpecs(
                block_size=spec.block_size, kv_cache_specs=kept
            )
            names = [name for name in group.layer_names if name in kept]
            physical.append(replace(group, layer_names=names, kv_cache_spec=spec))
        else:
            physical.append(group)
    return physical


def _metadata_layer_groups(groups: list[Any]) -> dict[str, int]:
    """Map every allocation-free LOD layer to its scheduler group.

    This is deliberately derived from the cache specs, independently from the
    model-layer marker. Comparing both views makes hook/order regressions fail
    at startup instead of silently allocating or omitting a native cache.
    """
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    result: dict[str, int] = {}
    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        if isinstance(spec, LODMetadataOnlyFullAttentionSpec):
            names = list(group.layer_names)
        elif isinstance(spec, UniformTypeKVCacheSpecs):
            spec_names = set(spec.kv_cache_specs)
            group_names = set(group.layer_names)
            if spec_names != group_names:
                raise RuntimeError(
                    "KV cache group layer names do not match its per-layer specs: "
                    f"group_only={sorted(group_names - spec_names)}, "
                    f"spec_only={sorted(spec_names - group_names)}"
                )
            names = [
                name
                for name, inner in spec.kv_cache_specs.items()
                if isinstance(inner, LODMetadataOnlyFullAttentionSpec)
            ]
        else:
            names = []
        for name in names:
            if name in result:
                raise RuntimeError(
                    f"metadata-only LOD layer {name!r} occurs in multiple groups"
                )
            result[name] = group_id
    return result


def _assert_no_external_tensors(
    kv_cache_config: Any, external_layer_names: set[str]
) -> None:
    leaked = {
        name
        for tensor in kv_cache_config.kv_cache_tensors
        for name in tensor.shared_by
        if name in external_layer_names
    }
    if leaked:
        raise RuntimeError(
            "externally owned LOD layers leaked into native GPU K/V tensors: "
            f"{sorted(leaked)}"
        )


def _install_core_cache_sizing_hook() -> None:
    """Exclude scheduler-only groups from GPU byte and capacity accounting."""
    import vllm.v1.core.kv_cache_utils as utils
    from vllm.v1.kv_cache_interface import KVCacheConfig

    if getattr(utils, "_vllm_lod_metadata_sizing_installed", False):
        return
    original_config = utils.get_kv_cache_config_from_groups
    original_pool_bytes = utils._pool_bytes_per_block
    original_concurrency = utils.get_max_concurrency_for_kv_cache_config

    def get_config(
        vllm_config: Any,
        groups: list[Any],
        available_memory: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # vLLM renamed this parameter from ``available`` to
        # ``available_memory``.  It is passed by keyword while profiling the
        # minimal CUDA-graph cache, so keep the hook compatible with both API
        # spellings.  Forward the value positionally because that also works
        # with both upstream signatures.
        if available_memory is None and "available" in kwargs:
            available_memory = int(kwargs.pop("available"))
        if available_memory is None:
            raise TypeError("missing required cache-sizing memory argument")
        external = set(_metadata_layer_groups(groups))
        physical = _physical_groups(groups)
        config = original_config(
            vllm_config, physical, available_memory, *args, **kwargs
        )
        if not physical:
            config.num_blocks = _metadata_pool_size()
        config.kv_cache_groups = groups
        _assert_no_external_tensors(config, external)
        return config

    def pool_bytes(vllm_config: Any, groups: list[Any]) -> int:
        physical = _physical_groups(groups)
        return original_pool_bytes(vllm_config, physical) if physical else 1

    def concurrency(vllm_config: Any, config: Any) -> float:
        physical = _physical_groups(config.kv_cache_groups)
        if not physical:
            return original_concurrency(vllm_config, config)
        physical_config = KVCacheConfig(
            num_blocks=config.num_blocks,
            kv_cache_tensors=config.kv_cache_tensors,
            kv_cache_groups=physical,
        )
        return original_concurrency(vllm_config, physical_config)

    utils.get_kv_cache_config_from_groups = get_config
    utils._pool_bytes_per_block = pool_bytes
    utils.get_max_concurrency_for_kv_cache_config = concurrency
    utils._vllm_lod_metadata_sizing_installed = True


def _metadata_pool_size() -> int:
    import os

    return max(1, int(os.getenv("VLLM_LOD_POOL_SIZE", "8")))


def _externalize_worker_cache_config(
    kv_cache_config: Any, external_layer_names: set[str]
) -> dict[str, int]:
    """Remove scheduler-only LOD storage from a worker-side config copy."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    logical_groups = _metadata_layer_groups(kv_cache_config.kv_cache_groups)
    actual = set(logical_groups)
    if actual != external_layer_names:
        raise RuntimeError(
            "external LOD ownership marker/spec mismatch: "
            f"missing_metadata_specs={sorted(external_layer_names - actual)}, "
            f"unmarked_metadata_specs={sorted(actual - external_layer_names)}"
        )
    # The core sizing hook must already have excluded these layers. Silently
    # deleting a tensor here would hide a regression and could leave capacity
    # accounting or packed offsets wrong, so reject it instead.
    _assert_no_external_tensors(kv_cache_config, external_layer_names)
    groups = []
    for group in kv_cache_config.kv_cache_groups:
        kept = [
            name
            for name in group.layer_names
            if name not in external_layer_names
        ]
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            kept_specs = {
                name: inner
                for name, inner in spec.kv_cache_specs.items()
                if name not in external_layer_names
            }
            if kept_specs:
                spec = UniformTypeKVCacheSpecs(
                    block_size=spec.block_size,
                    kv_cache_specs=kept_specs,
                )
        groups.append(replace(group, layer_names=kept, kv_cache_spec=spec))
    kv_cache_config.kv_cache_groups = groups
    _assert_no_external_tensors(kv_cache_config, external_layer_names)
    return logical_groups


def _install_external_worker_group_hook() -> None:
    """Translate scheduler-visible external groups into worker-only metadata.

    vLLM already has a worker-only group path for encoder-only attention: it
    builds normal attention metadata while excluding the group from block
    tables and physical K/V allocation. Reuse that contract for externally
    owned LOD layers instead of advertising a degenerate cache geometry.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        return

    if getattr(GPUModelRunner, "_vllm_lod_external_group_installed", False):
        return
    original = GPUModelRunner.may_add_encoder_only_layers_to_kv_cache_config

    def add_runner_only_layers(self: Any) -> None:
        from collections import defaultdict

        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.kv_cache_interface import (
            EncoderOnlyAttentionSpec,
            KVCacheGroupSpec,
        )

        original(self)
        external_specs: dict[Any, list[str]] = defaultdict(list)
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        external_layers = {
            layer_name: layer
            for layer_name, layer in layers.items()
            if bool(getattr(layer, "_vllm_lod_external_kv_cache", False))
        }
        logical_groups = _externalize_worker_cache_config(
            self.kv_cache_config, set(external_layers)
        )
        if set(logical_groups) != set(external_layers):
            raise RuntimeError("not every external LOD layer retained a logical group")
        for layer_name, layer in external_layers.items():
            spec = EncoderOnlyAttentionSpec(
                block_size=int(self.vllm_config.cache_config.block_size),
                num_kv_heads=int(layer.num_kv_heads),
                head_size=int(layer.head_size),
                dtype=layer.kv_cache_torch_dtype,
            )
            external_specs[spec].append(layer_name)
            self.runner_only_attn_layers.add(layer_name)
            if layer_name in logical_groups:
                layer._vllm_lod_external_scheduler_group = logical_groups[layer_name]

        for spec, layer_names in external_specs.items():
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )
        if external_specs:
            self._vllm_lod_external_layer_names = frozenset(
                name for names in external_specs.values() for name in names
            )
            logger.info(
                "Authoritative LOD cache-ownership invariant active for %d layers: "
                "scheduler metadata retained, native GPU K/V forbidden",
                len(self._vllm_lod_external_layer_names),
            )

    GPUModelRunner.may_add_encoder_only_layers_to_kv_cache_config = (
        add_runner_only_layers
    )
    GPUModelRunner._vllm_lod_external_group_installed = True


def _install_v2_external_metadata_hook() -> None:
    """Attach external layers to V2 metadata without cache/block ownership."""
    try:
        import vllm.v1.worker.gpu.model_runner as model_runner
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
        from vllm.v1.worker.utils import AttentionGroup
    except ImportError:
        return

    if getattr(model_runner, "_vllm_lod_external_metadata_installed", False):
        return
    original = model_runner.init_attn_backend

    def init_attn_backend(
        kv_cache_config: Any,
        vllm_config: Any,
        device: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        layers = get_layers_from_vllm_config(vllm_config, Attention)
        external = {
            name: layer
            for name, layer in layers.items()
            if bool(getattr(layer, "_vllm_lod_external_kv_cache", False))
        }
        logical_groups: dict[str, int] = {}
        if external:
            logical_groups = _externalize_worker_cache_config(
                kv_cache_config, set(external)
            )
            if set(logical_groups) != set(external):
                raise RuntimeError(
                    "not every external LOD layer retained a logical group"
                )
        attn_groups, cg_support, kernel_block_sizes = original(
            kv_cache_config, vllm_config, device, *args, **kwargs
        )
        if not external:
            return attn_groups, cg_support, kernel_block_sizes
        if not attn_groups:
            raise RuntimeError("V2 did not create the retained metadata group")

        # The common metadata (query starts, sequence lengths, and padding) is
        # identical across groups. Attach to the first group only as a vehicle
        # for that metadata. Do not add these names to KVCacheGroupSpec: that is
        # what keeps them out of allocation, block tables, and slot mappings.
        metadata_group_id = next(
            (group_id for group_id, groups in enumerate(attn_groups) if groups),
            0,
        )
        by_spec: dict[Any, list[str]] = {}
        for layer_name, layer in external.items():
            spec = EncoderOnlyAttentionSpec(
                block_size=int(vllm_config.cache_config.block_size),
                num_kv_heads=int(layer.num_kv_heads),
                head_size=int(layer.head_size),
                dtype=layer.kv_cache_torch_dtype,
            )
            by_spec.setdefault(spec, []).append(layer_name)
            layer._vllm_lod_external_metadata_group = metadata_group_id
            if layer_name in logical_groups:
                layer._vllm_lod_external_scheduler_group = logical_groups[layer_name]

        for spec, layer_names in by_spec.items():
            backend = external[layer_names[0]].get_attn_backend()
            group = AttentionGroup(
                backend=backend,
                layer_names=layer_names,
                kv_cache_spec=spec,
                kv_cache_group_id=metadata_group_id,
            )
            group.create_metadata_builders(
                vllm_config=vllm_config,
                device=device,
                kernel_block_size=None,
                num_metadata_builders=1,
            )
            builder = group.get_metadata_builder(0)
            support = builder.get_cudagraph_support(vllm_config, spec)
            cg_support = cg_support.narrow(support, backend.__name__)
            attn_groups[metadata_group_id].append(group)

        logger.info(
            "Authoritative LOD cache-ownership invariant active for %d V2 layers: "
            "metadata group %d, native GPU K/V forbidden",
            len(external),
            metadata_group_id,
        )
        return attn_groups, cg_support, kernel_block_sizes

    # GPUModelRunner imported the helper into its module namespace, so replace
    # that binding rather than only the source module's attribute.
    model_runner.init_attn_backend = init_attn_backend
    model_runner._vllm_lod_external_metadata_installed = True


def _install_v2_external_kv_init_hook() -> None:
    """Exclude worker-only LOD metadata groups from physical KV reshape."""
    try:
        import vllm.v1.worker.gpu.model_runner as model_runner
    except ImportError:
        return

    if getattr(model_runner, "_vllm_lod_external_kv_init_installed", False):
        return
    original = model_runner.init_kv_cache

    def init_kv_cache(
        runner_kv_caches: Any,
        forward_context: dict[str, Any],
        kv_cache_config: Any,
        attn_groups: list[list[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        external = {
            name
            for name, layer in forward_context.items()
            if bool(getattr(layer, "_vllm_lod_external_kv_cache", False))
        }
        if not external:
            return original(
                runner_kv_caches,
                forward_context,
                kv_cache_config,
                attn_groups,
                *args,
                **kwargs,
            )
        _assert_no_external_tensors(kv_cache_config, external)
        for groups in attn_groups:
            for group in groups:
                names = set(group.layer_names)
                overlap = names & external
                if overlap and overlap != names:
                    raise RuntimeError(
                        "an attention metadata group mixes external and native "
                        f"layers: {sorted(names)}"
                    )
        physical_groups = [
            [
                group
                for group in groups
                if not any(name in external for name in group.layer_names)
            ]
            for groups in attn_groups
        ]
        return original(
            runner_kv_caches,
            forward_context,
            kv_cache_config,
            physical_groups,
            *args,
            **kwargs,
        )

    model_runner.init_kv_cache = init_kv_cache
    model_runner._vllm_lod_external_kv_init_installed = True


def _assert_cache_ownership_hooks_installed() -> None:
    """Fail plugin registration if this vLLM version bypasses the invariant."""
    import vllm.v1.core.kv_cache_utils as utils
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    missing = []
    if not getattr(Attention, "_vllm_lod_cache_spec_installed", False):
        missing.append("Attention.get_kv_cache_spec")
    if not getattr(utils, "_vllm_lod_metadata_sizing_installed", False):
        missing.append("core cache sizing")
    if not getattr(KVCacheSpecRegistry, "_vllm_lod_manager_installed", False):
        missing.append("scheduler metadata manager")

    worker_installed = False
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        worker_installed |= bool(
            getattr(GPUModelRunner, "_vllm_lod_external_group_installed", False)
        )
    except ImportError:
        pass
    try:
        import vllm.v1.worker.gpu.model_runner as model_runner

        worker_installed |= bool(
            getattr(
                model_runner,
                "_vllm_lod_external_metadata_installed",
                False,
            )
            and getattr(
                model_runner,
                "_vllm_lod_external_kv_init_installed",
                False,
            )
        )
    except ImportError:
        pass
    if not worker_installed:
        missing.append("worker metadata/KV initialization")
    if missing:
        raise RuntimeError(
            "authoritative LOD cache ownership is not installed for: "
            + ", ".join(missing)
        )


def install_cache_ownership_hooks() -> None:
    _install_metadata_spec_manager_hook()
    _install_core_cache_sizing_hook()
    _install_attention_spec_hook()
    _install_external_worker_group_hook()
    _install_v2_external_metadata_hook()
    _install_v2_external_kv_init_hook()
    _assert_cache_ownership_hooks_installed()


__all__ = ["install_cache_ownership_hooks"]

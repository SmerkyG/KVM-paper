"""Native-prefix indexer allocation without a duplicate main MLA history.

vLLM's GLM packing aliases KDA state into a native MLA allocation. LoD owns
that MLA history, so keep the small indexer/tail and KDA groups in contiguous
layer-outer slabs of one backing allocation. Native addressing survives.
"""

from dataclasses import fields, replace


def install_native_prefix_cache_groups():
    import vllm.v1.core.kv_cache_utils as utils
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig, MLAAttentionSpec, MambaSpec, UniformTypeKVCacheSpecs,
    )
    from ..metadata_cache import LODMetadataOnlyFullAttentionSpec

    original_groups = utils._get_kv_cache_groups_glm5_next
    if getattr(original_groups, "_lod_native_prefix", False):
        return
    original_config = utils.get_kv_cache_config_from_groups
    original_bytes = utils._pool_bytes_per_block

    def enabled(config):
        from ..config import lod_enabled
        return (lod_enabled() and
                config.model_config.hf_text_config.model_type == "glm5_next_text" and
                getattr(config.model_config.hf_text_config, "lod_native_prefix", False))

    def groups(config, specs):
        metadata = {name: spec for name, spec in specs.items()
                    if isinstance(spec, LODMetadataOnlyFullAttentionSpec)}
        if not enabled(config) or not metadata:
            return original_groups(config, specs)
        # Ask the original helper for its native indexer/tail/KDA grouping and
        # page alignment, then remove the allocation-only MLA stand-in.
        native = dict(specs)
        for name, spec in metadata.items():
            native[name] = MLAAttentionSpec(**{
                field.name: getattr(spec, field.name)
                for field in fields(MLAAttentionSpec) if hasattr(spec, field.name)
            })
        result = original_groups(config, native)
        if result is None:
            raise RuntimeError("GLM native-prefix cache grouping is unsupported")
        physical = []
        for group in result:
            kept = [name for name in group.layer_names if name not in metadata]
            if not kept:
                continue
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = replace(spec, kv_cache_specs={name: spec.kv_cache_specs[name]
                                                     for name in kept})
            physical.append(replace(group, layer_names=kept, kv_cache_spec=spec))
        logical = utils.create_kv_cache_group_specs(metadata, [list(metadata)])
        return logical + physical

    def bytes_per_block(cache_groups):
        return sum(original_bytes([group]) for group in cache_groups)

    def pool_bytes(cache_groups):
        from ..config import lod_enabled
        # Core sizing does not always run inside a current-config context.
        # Pooled MLA indexing plus KDA uniquely identifies this GLM layout.
        pooled_indexer = any(
            isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) and any(
                isinstance(spec, MLAAttentionSpec) and spec.tokens_per_state > 1
                for spec in group.kv_cache_spec.kv_cache_specs.values())
            for group in cache_groups)
        kda = any(isinstance(group.kv_cache_spec, MambaSpec) for group in cache_groups)
        return (bytes_per_block(cache_groups) if lod_enabled() and pooled_indexer and kda
                else original_bytes(cache_groups))

    def config(config, cache_groups, available_memory):
        if not enabled(config):
            return original_config(config, cache_groups, available_memory)
        if not cache_groups:
            return original_config(config, cache_groups, available_memory)
        # A shared block-ID pool, but independent per-group contiguous slabs.
        # Divide the budget so every group has exactly the same block capacity.
        count = available_memory // bytes_per_block(cache_groups)
        total_bytes = count * bytes_per_block(cache_groups)
        tensors = []
        byte_offset = 0
        for group in cache_groups:
            group_bytes = count * original_bytes([group])
            allocated = original_config(config, [group],
                                        group_bytes)
            if allocated.num_blocks != count:
                raise RuntimeError("GLM prefix group block capacities disagree")
            # vLLM 0.30 allocates one shared byte buffer. Keep each native
            # group's strides, shifting just its base into that buffer.
            tensors.extend(replace(tensor, size=total_bytes,
                                   offset=tensor.offset + byte_offset)
                           for tensor in allocated.kv_cache_tensors)
            byte_offset += group_bytes
        assert byte_offset == total_bytes
        return KVCacheConfig(num_blocks=count, kv_cache_tensors=tensors,
            kv_cache_groups=cache_groups,
            prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval)

    groups._lod_native_prefix = True
    utils._get_kv_cache_groups_glm5_next = groups
    utils.get_kv_cache_config_from_groups = config
    utils._pool_bytes_per_block = pool_bytes

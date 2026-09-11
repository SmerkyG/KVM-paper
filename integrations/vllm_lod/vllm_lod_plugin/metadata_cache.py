"""Scheduler-only full-attention geometry for externally owned LOD K/V.

The scheduler must continue to see a full-attention group for hybrid cache
coordination and prefix hashes.  The corresponding K/V, however, lives in the
LOD pool rather than vLLM's paged cache.  These classes preserve the logical
block table while consuming neither GPU pages nor IDs from the shared physical
``BlockPool``.
"""

from __future__ import annotations

import itertools
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    KVCacheBlock,
    resolve_block_hashes,
)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.request import Request


def _lod_pool_size() -> int:
    return max(1, int(os.getenv("VLLM_LOD_POOL_SIZE", "8")))


@dataclass(frozen=True, kw_only=True)
class LODMetadataOnlyFullAttentionSpec(FullAttentionSpec):
    """A real logical full-attention group with no worker-side K/V tensor."""

    def max_memory_usage_bytes(self, vllm_config: Any) -> int:
        # One scheduler metadata row per request.  Physical sizing removes this
        # group, while all-global capacity reporting uses this one-page unit.
        del vllm_config
        return self.page_size_bytes

    @classmethod
    def merge(cls, specs: list[Any]) -> Any:
        # Never accidentally absorb a native full-attention layer into an
        # externally owned group merely because its tensor geometry matches.
        if not specs or not all(type(spec) is cls for spec in specs):
            # vLLM treats ValueError/AssertionError as the public signal that
            # two otherwise same-shaped cache specs need distinct groups.
            # DFlash2 adds native draft-attention specs alongside the target
            # model's metadata-only LOD specs, so this distinction matters.
            raise ValueError(
                "metadata-only LOD cache specs cannot be merged with native "
                "K/V cache specs"
            )
        return super().merge(specs)


class _VirtualPrefixStore:
    """Small CPU-only LRU of semantic LOD rows represented by hash sentinels."""

    def __init__(self, capacity: int, *, namespace: int) -> None:
        # Sentinels never enter a physical BlockPool or worker block table. Give
        # each one a distinct negative ID anyway: KVCacheBlock is a value-equal
        # dataclass, and vLLM compares cached-block lists while reconciling
        # hybrid groups. Reusing ID zero made unrelated semantic rows compare
        # equal and could manufacture a false 64K prefix hit.
        base = (namespace + 1) << 32
        self.sentinels = [
            KVCacheBlock(-(base + index + 1)) for index in range(capacity)
        ]
        self.free: deque[KVCacheBlock] = deque(self.sentinels)
        self.hash_to_sentinel: dict[Any, KVCacheBlock] = {}
        # Dict insertion order is the chronological block order for the one
        # semantic sequence represented by a sentinel.  Unlike native paged
        # K/V, one LOD row cannot splice blocks from several cached requests.
        self.hashes_by_identity: dict[int, dict[Any, None]] = {}

    def _forget(self, sentinel: KVCacheBlock) -> None:
        for block_hash in self.hashes_by_identity.pop(id(sentinel), ()):
            if self.hash_to_sentinel.get(block_hash) is sentinel:
                self.hash_to_sentinel.pop(block_hash, None)

    def retain_prefix(self, sentinel: KVCacheBlock, blocks: int) -> None:
        """Drop hashes beyond the semantic prefix being resumed."""
        owned = self.hashes_by_identity.get(id(sentinel))
        if owned is None:
            if blocks:
                raise RuntimeError("a cached LOD sentinel has no owned hashes")
            return
        stale = tuple(itertools.islice(owned, blocks, None))
        for block_hash in stale:
            owned.pop(block_hash)
            if self.hash_to_sentinel.get(block_hash) is sentinel:
                self.hash_to_sentinel.pop(block_hash, None)

    def acquire(
        self,
        preferred: KVCacheBlock | None = None,
        *,
        retained_blocks: int | None = None,
    ) -> KVCacheBlock:
        if preferred is not None and preferred.ref_cnt == 0:
            # Keep removal explicitly identity-based. KVCacheBlock is a
            # value-equal dataclass, while this store owns object lifetimes.
            for index, candidate in enumerate(self.free):
                if candidate is preferred:
                    del self.free[index]
                    if retained_blocks is not None:
                        self.retain_prefix(preferred, retained_blocks)
                    preferred.ref_cnt = 1
                    return preferred
            raise RuntimeError("an idle LOD prefix sentinel is not in the free queue")
        if not self.free:
            raise RuntimeError(
                "scheduler-only LOD prefix rows are exhausted; increase "
                "VLLM_LOD_POOL_SIZE or lower max concurrency"
            )
        sentinel = self.free.popleft()
        self._forget(sentinel)
        assert sentinel.ref_cnt == 0
        sentinel.ref_cnt = 1
        return sentinel

    def release(self, sentinel: KVCacheBlock) -> None:
        sentinel.ref_cnt -= 1
        assert sentinel.ref_cnt >= 0
        if sentinel.ref_cnt == 0:
            # Cached sentinels join the tail, yielding the same release-order
            # LRU behavior as vLLM's ordinary prefix block pool.
            self.free.append(sentinel)

    def insert(self, block_hash: Any, sentinel: KVCacheBlock) -> None:
        previous = self.hash_to_sentinel.get(block_hash)
        if previous is sentinel:
            return
        if previous is not None:
            self.hashes_by_identity.get(id(previous), {}).pop(block_hash, None)
        self.hash_to_sentinel[block_hash] = sentinel
        self.hashes_by_identity.setdefault(id(sentinel), {})[block_hash] = None


def _stores(block_pool: BlockPool) -> dict[int, _VirtualPrefixStore]:
    stores = getattr(block_pool, "_vllm_lod_metadata_prefix_stores", None)
    if stores is None:
        stores = {}
        setattr(block_pool, "_vllm_lod_metadata_prefix_stores", stores)
    return stores


class LODMetadataOnlyFullAttentionManager(FullAttentionManager):
    """Full-attention scheduling using one virtual semantic row per request."""

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def __init__(self, kv_cache_spec: KVCacheSpec, **kwargs: Any) -> None:
        assert isinstance(kv_cache_spec, LODMetadataOnlyFullAttentionSpec)
        super().__init__(kv_cache_spec, **kwargs)
        stores = _stores(self.block_pool)
        self.store = stores.setdefault(
            self.kv_cache_group_id,
            _VirtualPrefixStore(
                _lod_pool_size(), namespace=self.kv_cache_group_id
            ),
        )
        self.req_to_sentinel: dict[str, KVCacheBlock] = {}

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del (
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        return 0

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert request_id not in self.req_to_sentinel
        preferred = new_computed_blocks[0] if new_computed_blocks else None
        if preferred is not None and any(
            block is not preferred for block in new_computed_blocks
        ):
            raise RuntimeError(
                "a metadata-only LOD prefix hit spans multiple semantic rows"
            )
        sentinel = self.store.acquire(
            preferred,
            retained_blocks=len(new_computed_blocks) if preferred is not None else None,
        )
        self.req_to_sentinel[request_id] = sentinel
        logical_blocks = cdiv(
            num_local_computed_tokens + num_external_computed_tokens,
            self.block_size,
        )
        self.req_to_blocks[request_id].extend([self._null_block] * logical_blocks)
        self.num_cached_block[request_id] = logical_blocks

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        del request_id, num_local_computed_tokens, num_external_computed_tokens

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        del num_tokens_main_model
        if request_id not in self.req_to_sentinel:
            self.req_to_sentinel[request_id] = self.store.acquire()
        req_blocks = self.req_to_blocks[request_id]
        required = cdiv(num_tokens, self.block_size)
        added = max(required - len(req_blocks), 0)
        if added:
            req_blocks.extend([self._null_block] * added)
        # Preserve the normal logical block-table append contract.  ID zero is
        # safe even if a worker happens to serialize this allocation-free group.
        return [self._null_block] * added

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        del retention_interval
        sentinel = self.req_to_sentinel.get(request.request_id)
        if sentinel is None:
            return
        num_cached = self.num_cached_block.get(request.request_id, 0)
        num_full = num_tokens // self.block_size
        if num_cached >= num_full:
            return
        block_hashes = resolve_block_hashes(
            request.block_hashes,
            self.block_pool.hash_block_size,
            self.block_size,
        )
        for block_hash in itertools.islice(block_hashes, num_cached, num_full):
            self.store.insert(block_hash, sentinel)
        self.num_cached_block[request.request_id] = num_full

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self.req_to_blocks.pop(request_id, None)
        self.num_cached_block.pop(request_id, None)
        sentinel = self.req_to_sentinel.pop(request_id, None)
        if sentinel is not None:
            self.store.release(sentinel)
        # Virtual metadata never enters BlockPool.free_blocks().
        return []

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        del running_request_id
        # Disable cascade-attention inference for this virtual group.
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        del pcp_world_size
        assert isinstance(kv_cache_spec, LODMetadataOnlyFullAttentionSpec)
        block_size = kv_cache_spec.block_size * dcp_world_size
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            block_size,
            alignment_tokens=alignment_tokens,
        )
        stores = _stores(block_pool)
        hits: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in kv_cache_group_ids
        )
        owners: list[KVCacheBlock | None] = [None] * len(kv_cache_group_ids)
        for block_hash in itertools.islice(block_hashes, max_length // block_size):
            sentinels = [
                stores[group_id].hash_to_sentinel.get(block_hash)
                for group_id in kv_cache_group_ids
            ]
            if any(
                sentinel is None or sentinel.ref_cnt != 0
                for sentinel in sentinels
            ):
                break
            if any(
                owner is not None and sentinel is not owner
                for owner, sentinel in zip(owners, sentinels, strict=True)
            ):
                break
            for index, (group_hits, sentinel) in enumerate(
                zip(hits, sentinels, strict=True)
            ):
                assert sentinel is not None
                if owners[index] is None:
                    owners[index] = sentinel
                group_hits.append(sentinel)
        hit_length = len(hits[0]) * block_size
        if drop_eagle_block and hit_length:
            hit_length -= block_size
            for group_hits in hits:
                group_hits.pop()
        hit_length -= hit_length % alignment_tokens
        keep = cdiv(hit_length, block_size)
        for group_hits in hits:
            del group_hits[keep:]
        return hits, hit_length


def is_metadata_only_spec(spec: KVCacheSpec) -> bool:
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    if isinstance(spec, LODMetadataOnlyFullAttentionSpec):
        return True
    return (
        isinstance(spec, UniformTypeKVCacheSpecs)
        and bool(spec.kv_cache_specs)
        and all(
            isinstance(inner, LODMetadataOnlyFullAttentionSpec)
            for inner in spec.kv_cache_specs.values()
        )
    )


__all__ = [
    "LODMetadataOnlyFullAttentionManager",
    "LODMetadataOnlyFullAttentionSpec",
    "is_metadata_only_spec",
]

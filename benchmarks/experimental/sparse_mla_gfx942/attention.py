"""Minimal BF16 gfx942 interface to AITER's new 512+64 sparse MLA kernel.

Experimental, not installed in the production engine. Cache rows contain a
512-d latent (both K and V) followed by 64 K-only channels. Each query has a
CSR list of cache-row indices; the list is shared by its query heads. Optional
FP32 ``key_log_count`` adds natural-log mass AFTER scaling QK. Use zero for
exact leaves/local keys and log(count) for centroid means. LSE is natural log.
"""

def sparse_mla(q, cache, indptr, indices, *, scale, key_log_count=None, splits=32,
               has_invalid=False):
    import torch
    import triton
    from .kernel import _sparse_mla, _sparse_mla_reduce

    if q.ndim != 3 or q.shape[-1] != 576 or cache.ndim != 2 or cache.shape[-1] != 576:
        raise ValueError("expected Q [C,H,576] and cache [N,576]")
    if q.dtype != torch.bfloat16 or cache.dtype != q.dtype or q.device != cache.device or not q.is_cuda:
        raise ValueError("this probe supports same-device BF16 CUDA/ROCm inputs only")
    if q.stride(-1) != 1 or cache.stride(-1) != 1:
        raise ValueError("channel dimension must be contiguous")
    queries, heads, _ = q.shape
    if min(queries, heads, cache.shape[0], splits) < 1:
        raise ValueError("positive geometry/split count required")
    for name, value, shape in (("indptr", indptr, (queries + 1,)), ("indices", indices, (indices.numel(),))):
        if value.shape != shape or value.dtype != torch.int32 or value.device != q.device or not value.is_contiguous():
            raise ValueError(f"{name} must be same-device contiguous int32 {shape}")
    if key_log_count is not None and (key_log_count.shape != (cache.shape[0],)
            or key_log_count.dtype != torch.float32 or key_log_count.device != q.device
            or not key_log_count.is_contiguous()):
        raise ValueError("key_log_count must be same-device contiguous FP32 [N]")
    if "gfx942" not in torch.cuda.get_device_properties(q.device).gcnArchName:
        raise ValueError("this isolated backport is validated only on gfx942")

    block_m = 16 if heads >= 16 else max(8, triton.next_power_of_2(heads))
    grid_splits = triton.next_power_of_2(splits)
    out = torch.empty((queries, heads, 512), device=q.device, dtype=q.dtype)
    lse = torch.empty((queries, heads), device=q.device, dtype=torch.float32)
    if splits > 1:
        part_m = torch.empty((queries, grid_splits, heads), device=q.device, dtype=torch.float32)
        part_l = torch.empty_like(part_m)
        part_acc = torch.empty((queries, grid_splits, heads, 512), device=q.device, dtype=torch.float32)
        ms0, mss = part_m.stride()[:2]
        as0, ass, ash = part_acc.stride()[:3]
    else:
        part_m = part_l = part_acc = out
        ms0 = mss = as0 = ass = ash = 0
    # Buffer-load offsets are signed 32-bit byte offsets. Respect oversized
    # caches rather than making the capacity/live-length addressing mistake.
    buffer_load = (cache.shape[0] * cache.stride(0) * cache.element_size()) < 2**31
    index_buffer = indices.numel() * indices.element_size() < 2**31
    _sparse_mla[(queries, grid_splits, triton.cdiv(heads, block_m))](
        q, cache, cache, indices, indptr, cache, cache, indices, indptr,
        lse, out, part_m, part_l, part_acc, None, None,
        float(scale), q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        cache.stride(0), cache.stride(0), cache.shape[0], cache.shape[0],
        ms0, mss, as0, ass, ash, heads,
        HAS_EXTRA=False, HAS_SINK=False, MAIN_FMT="bf16", EXTRA_FMT="bf16",
        MAIN_BLOCK_SIZE=1, EXTRA_BLOCK_SIZE=1, CS0_ALIGN=1,
        NOPE_DIM=512, ROPE_DIM=64, HEAD_SIZE=512, ROPE_SEPARATE=True,
        BLOCK_M=block_m, BLOCK_K=32, num_splits=splits, SPLIT_K=splits > 1,
        HEAD_ALIGNED=heads % block_m == 0, NOPE_CHUNK=8, CHUNK_AXIS=0,
        PART_STORE_CACHE="", UNI_TILE=True, GRID_ORDER="qsh", Q_CACHE="",
        main_num_splits=splits, ADAPTIVE_SPLITS=splits > 1, DEQ="none",
        MAIN_USE_BUFFER_LOAD=buffer_load, EXTRA_USE_BUFFER_LOAD=buffer_load,
        IDX_BUFFER_LOAD=index_buffer, HAS_INVALID=has_invalid,
        lse_ptr=lse, HAS_LSE=True, GATHER_CACHE="", ASYNC_LDS=False,
        key_bias_ptr=key_log_count if key_log_count is not None else cache,
        HAS_KEY_BIAS=key_log_count is not None,
        num_warps=4, waves_per_eu=1)
    if splits > 1:
        _sparse_mla_reduce[(queries, heads)](
            part_m, part_l, part_acc, lse, out, out.stride(0), out.stride(1),
            ms0, mss, as0, ass, ash, heads,
            HAS_SINK=False, HEAD_SIZE=512, BLOCK_M=1, NUM_SPLITS=grid_splits,
            HEAD_ALIGNED=True, ADAPTIVE_SPLITS=True, lse_ptr=lse, HAS_LSE=True,
            num_warps=min(4, grid_splits))
    return out, lse

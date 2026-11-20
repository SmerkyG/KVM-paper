// Call the existing split-KV attention and merge with persistent GPU metadata.
template <bool full>
void decode_planned_impl(torch::Tensor q, torch::Tensor out, torch::Tensor kv,
                         torch::Tensor indices, torch::Tensor indptr, torch::Tensor merge_indptr,
                         torch::Tensor batch_map, torch::Tensor scratch) {
    const int splits = batch_map.numel();
    const int kv_heads = kv.size(3);
    const int qo_heads = q.size(1);
    const int head_dim = q.size(2);
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && kv.scalar_type() == torch::kFloat16 &&
                kv.size(2) == 1 && head_dim == 128 && qo_heads == 4 * kv_heads,
                "expected FP16 NHD page-one Llama GQA");
    TORCH_CHECK(indptr.numel() == splits + 2 && merge_indptr.numel() == 2 &&
                scratch.numel() >= splits * qo_heads * (head_dim * sizeof(half) + sizeof(float)),
                "invalid persistent split-KV buffers");
    using Page = paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t, full>;
    Page page(kv_heads, 1, head_dim, splits, full ? 0 : indices.size(1), 1, -1,
              static_cast<half*>(kv.data_ptr()), indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>());
    kv_partition_info_t<int32_t> partition;
    partition.batch_size_before_partition = 1;
    partition.chunk_indptr = merge_indptr.data_ptr<int32_t>();
    partition.batch_idx_map = batch_map.data_ptr<int32_t>();
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto status = BatchDecodeWithPagedKVCacheDispatched<
        4, 128, PageStorage::kIndices, QKVLayout::kNHD, RotaryMode::kNone, half, half, int32_t, full>(
            static_cast<half*>(q.data_ptr()), page, partition, static_cast<half*>(out.data_ptr()),
            static_cast<half*>(scratch.data_ptr()), nullptr, 1.f, 1.e4f, stream);
    TORCH_CHECK(status == cudaSuccess, "ClusterKV planned decode failed: ", cudaGetErrorString(status));
}

void decode_planned(torch::Tensor q, torch::Tensor out, torch::Tensor kv,
                    torch::Tensor indices, torch::Tensor indptr, torch::Tensor merge_indptr,
                    torch::Tensor batch_map, torch::Tensor scratch) {
    if (indices.dim() == 1) {
        decode_planned_impl<true>(q, out, kv, indices, indptr, merge_indptr, batch_map, scratch);
    } else {
        decode_planned_impl<false>(q, out, kv, indices, indptr, merge_indptr, batch_map, scratch);
    }
}

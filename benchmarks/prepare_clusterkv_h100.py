"""Correct public ClusterKV clustering and selection for the 400-cluster setting."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    root = args.checkout
    edits = {
        'kernel/cluster/src/cluster.cu': [
            ('const int block_size = 256;', 'const int block_size = 512;\n'
             '\tTORCH_CHECK(nlist <= block_size, "ClusterKV lookup supports at most 512 centroids");'),
            ('get_neigh_c_kernel<<<grid_size, block_size>>>(',
             'get_neigh_c_kernel<<<grid_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>('),
            ('<<<gridSize, blockSize, shmemUse>>>(',
             '<<<gridSize, blockSize, shmemUse, c10::cuda::getCurrentCUDAStream()>>>('),
        ],
        'kernel/cluster/include/km_include/cluster.cuh': [
            ('#include <cub/cub.cuh>', '#include <cub/cub.cuh>\n#include <math_constants.h>'),
            ('int scan_output;\n            BlockScan(cub_temp_storage).InclusiveSum(thread_data, scan_output);',
             'int scan_output, tile_sum;\n'
             '            BlockScan(cub_temp_storage).InclusiveSum(thread_data, scan_output, tile_sum);'),
            ("            // The last thread in the block holds the sum of the tile (in scan_output)\n"
             "            // Broadcast this sum to all threads to update the running_prefix for the next tile.\n"
             "            int tile_sum = __shfl_sync(0xFFFFFFFF, scan_output, block_size - 1);\n", ""),
            ('            atomicAdd(&count[label], 1);',
             '            if (count_only || threadIdx.x == 0) atomicAdd(&count[label], 1);'),
            ('__shared__ DistT shared_dist[256];  // Adjust the size as needed\n'
             '    __shared__ int shared_indices[256];',
             '__shared__ DistT shared_dist[512];\n    __shared__ int shared_indices[512];'),
            ('std::numeric_limits<DistT>::min()', 'static_cast<DistT>(-CUDART_INF_F)'),
            ('size <= nlist; size *= 2)', 'size <= blockDim.x; size *= 2)'),
            ('if (j > idx && j < nlist)', 'if (j > idx)'),
            ('                if (shared_dist[idx] < shared_dist[j]) {',
             '                const bool descending = (idx & size) == 0;\n'
             '                if (descending ? shared_dist[idx] < shared_dist[j]\n'
             '                               : shared_dist[idx] > shared_dist[j]) {'),
            ('if (idx < nlist) {\n        DistT dist_val = 0;',
             'if (idx < nlist && cluster_size[kv_head_idx * nlist + idx] >= 0) {\n        DistT dist_val = 0;'),
            ('neigh_c_size[head_idx * nlist + idx] = cluster_size[kv_head_idx * nlist + t];',
             'neigh_c_size[head_idx * nlist + idx] = t >= 0 ? cluster_size[kv_head_idx * nlist + t] : 0;'),
            ('            const int cid = neigh_c[head * nlist + c];\n'
             '            const int ck_ind_end = cluster_size_ps[kv_head * nlist + cid];\n'
             '            const int c_size = neigh_c_size[head * nlist + c];// or cluster_size[head*nlist + cid]',
             '            const int c_size = neigh_c_size[head * nlist + c];\n'
             '            if (c_size == 0) continue;\n'
             '            const int cid = neigh_c[head * nlist + c];\n'
             '            const int ck_ind_end = cluster_size_ps[kv_head * nlist + cid];'),
        ],
        'kernel/cluster/include/km_include/decode_attn.cuh': [
            ('\tconst uint32_t kv_head_idx =\n\t\tblockIdx.y; // Use kv_head_idx as another dimention in page indices.\n'
             '\tconst uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;\n'
             '\tconst uint32_t num_qo_heads = gridDim.y * bdy;',
             '\tconst uint32_t num_qo_heads = gridDim.y * bdy;\n'
             '\tconst uint32_t qo_head_idx = blockIdx.y * bdy + threadIdx.y;\n'
             '\tconst uint32_t kv_head_idx = qo_head_idx / (num_qo_heads / paged_kv.num_heads);'),
            ('\t\t\tkv_head_idx,\n\t\t\t0,\n\t\t\tlast_indptr);',
             '\t\t\tkv_head_idx,\n\t\t\tqo_head_idx,\n\t\t\t0,\n\t\t\tlast_indptr);'),
            ('\t\t\t\t\t\tkv_head_idx,\n\t\t\t\t\t\t0,\n\t\t\t\t\t\tlast_indptr);',
             '\t\t\t\t\t\tkv_head_idx,\n\t\t\t\t\t\tqo_head_idx,\n\t\t\t\t\t\t0,\n\t\t\t\t\t\tlast_indptr);'),
            ('\tconstexpr uint32_t bdy = GROUP_SIZE;\n\tconstexpr uint32_t num_threads = std::max(128U, bdx * bdy);\n'
             '\tconstexpr uint32_t bdz = num_threads / (bdx * bdy);\n'
             '\tconstexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;',
             '\t// Sparse retrieval has separate selected indices per query head.\n'
             '\tconstexpr uint32_t bdy = full ? GROUP_SIZE : 1U;\n'
             '\tconstexpr uint32_t num_threads = std::max(128U, bdx * bdy);\n'
             '\tconstexpr uint32_t bdz = num_threads / (bdx * bdy);\n'
             '\tconstexpr uint32_t tile_size_per_bdx = bdy == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;\n'
             '\tconst uint32_t attention_heads = num_qo_heads / bdy;'),
            ('dim3 nblks(batch_size, num_kv_heads);', 'dim3 nblks(batch_size, attention_heads);', 2),
        ],
        'kernel/cluster/include/km_include/page.cuh': [
            ('return data + get_k_elem_offset(last_page_idx, head_idx, 0, feat_idx);',
             'const IdType tail = last_page_idx < 0 ? indptr[batch_size + 1] : last_page_idx;\n'
             '\t\t\t\treturn data + get_k_elem_offset(tail, head_idx, 0, feat_idx);'),
            ('protective_get_k_ptr_page1(IdType page_iter,\n'
             '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t uint32_t head_idx,',
             'protective_get_k_ptr_page1(IdType page_iter,\n'
             '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t uint32_t head_idx,\n'
             '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t uint32_t qo_head_idx,'),
            ('__ldg(indices + page_iter + head_idx * page_budget),\n'
             '\t\t\t\t\t\t\t\t\t\t\t\thead_idx,\n'
             '\t\t\t\t\t\t\t\t\t\t\t\t0,',
             '__ldg(indices + page_iter + qo_head_idx * page_budget),\n'
             '\t\t\t\t\t\t\t\t\t\t\t\thead_idx,\n'
             '\t\t\t\t\t\t\t\t\t\t\t\t0,'),
        ],
        'kernel/cluster/src/km_ops.h': [
            ('void search_indices(', 'void decode_planned(torch::Tensor q, torch::Tensor out, torch::Tensor kv,\n'
             '                    torch::Tensor indices, torch::Tensor indptr, torch::Tensor merge_indptr,\n'
             '                    torch::Tensor batch_map, torch::Tensor scratch);\n\nvoid search_indices('),
        ],
        'kernel/cluster/src/km_ops.cu': [
            ('\tm.def("search_indices",', '\tm.def("decode_planned", &decode_planned);\n\tm.def("search_indices",'),
        ],
        'clusterkv/clusterkv_utils/clusterkv_controller.py': [
            ('\t\t\tif layer_idx == 2:\n\t\t\t\tself.neigh_c = torch.cat(',
             '\t\t\tif layer_idx == self.num_layers - 1:\n\t\t\t\tself.neigh_c = torch.cat('),
            ('self.infer_token_budget + cur_win_size],',
             'self.infer_token_budget + cur_win_size\n'
             '\t\t\t\t\t\t\t\t\t\t\t\t\t\t- (0 if self.offload else 1 if updateTensor else 2)],'),
        ],
        'clusterkv/clusterkv_utils/__init__.py': [
            ('controller.sink_indices, topk_indices, controller.cur_win_indices',
             'controller.sink_indices, topk_indices, controller.cur_win_indices[:, :-1]'),
        ],
    }
    for name, replacements in edits.items():
        path = root / name
        before = text = path.read_text()
        for old, new, *occurrences in replacements:
            count = occurrences[0] if occurrences else 1
            if new and text.count(new) == count:
                continue
            if text.count(old) == count:
                text = text.replace(old, new)
            elif new or old in text:
                raise RuntimeError(f'{name}: expected one occurrence of {old!r}')
        if text != before:
            path.write_text(text)

    source = root / 'kernel/cluster/src/approx_attn.cu'
    marker = '\n// H100 device-planned ClusterKV decode\n'
    graph_decode = Path(__file__).with_name('clusterkv_graph_decode.cuh').read_text()
    before = source.read_text()
    after = before.split(marker, 1)[0].rstrip() + '\n' + marker + graph_decode
    if after != before:
        source.write_text(after)


if __name__ == '__main__':
    main()

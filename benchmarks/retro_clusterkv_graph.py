"""Device-resident split-KV metadata for the released ClusterKV decode kernels."""
import torch
import triton
import triton.language as tl

from clusterkv import _clusterkv_knl as kernels
from clusterkv.clusterkv_utils import build_cluster


@triton.jit
def _decode_plan(position, dense, sparse, prompt: tl.constexpr, window: tl.constexpr,
                 retrieved: tl.constexpr, dense_chunks: tl.constexpr, sparse_chunks: tl.constexpr,
                 block: tl.constexpr):
    slot = tl.arange(0, block)
    current = tl.load(position).to(tl.int32)
    recent = (current - prompt) % window
    sparse_past = retrieved + recent
    tl.store(dense + slot, tl.minimum(slot, dense_chunks) * current // dense_chunks
             + (slot == dense_chunks + 1), slot <= dense_chunks + 1)
    tl.store(sparse + slot, tl.minimum(slot, sparse_chunks) * sparse_past // sparse_chunks
             + (slot == sparse_chunks + 1), slot <= sparse_chunks + 1)
    tl.store(dense + dense_chunks + 2, current)
    tl.store(sparse + sparse_chunks + 2, current)


@triton.jit
def _decode_indices(selected, indices, position, prompt: tl.constexpr, window: tl.constexpr,
                    sink: tl.constexpr, budget: tl.constexpr, stride: tl.constexpr, block: tl.constexpr):
    head = tl.program_id(0)
    col = tl.program_id(1) * block + tl.arange(0, block)
    current = tl.load(position).to(tl.int32)
    begin = current - (current - prompt) % window
    remote = tl.load(selected + head * budget + col - sink, (col >= sink) & (col < sink + budget), 0)
    index = tl.where(col < sink, col, tl.where(col < sink + budget, remote + sink,
                                              begin + col - sink - budget))
    tl.store(indices + head * stride + col, index, col < stride)


class SplitKV:
    def __init__(self, chunks, heads, dim, device):
        # One extra chunk attends to the current token, using the author's
        # dedicated tail block. Store its physical page after the prefix sums.
        splits = chunks + 1
        self.chunks = chunks
        self.indptr = torch.empty(splits + 2, dtype=torch.int32, device=device)
        self.merge = torch.tensor([0, splits], dtype=torch.int32, device=device)
        self.batch_map = torch.zeros(splits, dtype=torch.int32, device=device)
        self.scratch = torch.empty(splits * heads * (dim * 2 + 4), dtype=torch.uint8, device=device)

    def attention(self, q, out, kv, indices):
        kernels.decode_planned(q, out, kv, indices, self.indptr, self.merge, self.batch_map, self.scratch)


class ClusterKVDeviceDecode:
    def __init__(self, ctrl, budget):
        self.ctrl = ctrl
        self.device = ctrl.kv_cache[0].device
        self.prompt = ctrl.prompt_len
        self.budget = budget - 1
        self.position = torch.empty((1, 1), dtype=torch.int64, device=self.device)
        self.input = torch.empty_like(self.position)
        self.dense = SplitKV(triton.cdiv(ctrl.max_seq_len, 256), ctrl.num_heads, ctrl.head_dim, self.device)
        self.sparse = SplitKV(triton.cdiv(ctrl.sink + self.budget + ctrl.window - 1, 128),
                              ctrl.num_heads, ctrl.head_dim, self.device)
        stride = ctrl.sink + self.budget + ctrl.window - 1
        self.indices = torch.empty((ctrl.num_heads, stride), dtype=torch.int32, device=self.device)
        self.full_indices = torch.empty(0, dtype=torch.int32, device=self.device)
        self.graph = None
        if not ctrl.full:
            self.centers, self.sizes, self.prefixes, self.keys = [], [], [], []
            for layer in range(2, ctrl.num_layers):
                center = ctrl.centroids[layer]
                size = ctrl.cluster_size[layer]
                prefix = ctrl.cluster_size_ps[layer]
                keys = ctrl.cluster_key_indices[layer]
                shape = (ctrl.num_kv_heads, 512)
                self.centers.append(center.new_zeros((*shape, ctrl.head_dim)))
                self.sizes.append(size.new_full(shape, -1))
                self.prefixes.append(prefix.new_zeros(shape))
                self.keys.append(keys.new_empty((ctrl.num_kv_heads, ctrl.max_seq_len - ctrl.sink)))
                self.centers[-1][:, :ctrl.nlist].copy_(center)
                self.sizes[-1][:, :ctrl.nlist].copy_(size)
                self.prefixes[-1][:, :ctrl.nlist].copy_(prefix)
                self.keys[-1][:, :keys.shape[1]].copy_(keys)
            self.near = torch.empty((ctrl.num_heads, 512), dtype=torch.int32, device=self.device)
            self.near_sizes = torch.empty_like(self.near)
            self.selected = ctrl.sel_token_indices
            self.niter = ctrl.niter
            self.max_seq_len = ctrl.max_seq_len
            self.get_labels = ctrl.get_labels

    def prepare(self):
        ctrl = self.ctrl
        _decode_plan[(1,)](self.position, self.dense.indptr, self.sparse.indptr,
                           self.prompt, ctrl.window, ctrl.sink + self.budget,
                           self.dense.chunks, self.sparse.chunks,
                           triton.next_power_of_2(max(self.dense.chunks, self.sparse.chunks) + 3))

    def append(self, k, v, layer):
        kv = self.ctrl.kv_cache[layer]
        pos = self.position.view(1)
        kv[:, 0, 0].index_copy_(0, pos, k[0])
        kv[:, 1, 0].index_copy_(0, pos, v[0])

    def attention(self, q, layer):
        ctrl = self.ctrl
        q = q[0].contiguous()
        out = torch.empty_like(q)
        if layer < 2 or ctrl.full:
            self.dense.attention(q, out, ctrl.kv_cache[layer], self.full_indices)
        else:
            i = layer - 2
            kernels.get_neigh_c(q, self.centers[i], self.sizes[i], self.near, self.near_sizes)
            kernels.get_sel_indices(self.near, self.near_sizes, self.prefixes[i], self.keys[i], self.selected)
            stride = self.indices.shape[1]
            _decode_indices[(ctrl.num_heads, triton.cdiv(stride, 256))](
                self.selected, self.indices, self.position, self.prompt, ctrl.window, ctrl.sink,
                self.budget, stride, 256)
            self.sparse.attention(q, out, ctrl.kv_cache[layer], self.indices)
        return out.unsqueeze(0)

    def publish_window(self, context):
        ctrl = self.ctrl
        end = context - ctrl.sink
        self.key_start = end - ctrl.window
        self.centroid_start = ctrl.nlist + ((context - self.prompt) // ctrl.window - 1) * ctrl.window_nlist
        stream = torch.cuda.current_stream()
        for layer in range(2, ctrl.num_layers):
            keys = ctrl.kv_cache[layer][context - ctrl.window:context, 0, 0]
            build_cluster(self, layer, keys, self.key_start, ctrl.window_nlist, stream)

    def update_metadata(self, layer, centers, sizes, prefixes, keys):
        i = layer - 2
        start = self.centroid_start
        end = start + centers.shape[1]
        self.centers[i][:, start:end].copy_(centers)
        self.sizes[i][:, start:end].copy_(sizes)
        self.prefixes[i][:, start:end].copy_(prefixes + self.key_start)
        self.keys[i][:, self.key_start:self.key_start + keys.shape[1]].copy_(keys)

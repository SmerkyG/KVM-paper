"""Check Multipole sparse addressing beyond the signed 32-bit element limit."""
import torch
from multipoleattention import kernels as author

torch.manual_seed(81)
torch.set_grad_enabled(False)
heads, group, dim, capacity, selected = 128, 4, 128, 132624, 64
assert (heads - 1) * capacity * dim > 2**31
device = 'cuda'
q = torch.randn((1, heads, group, dim), device=device, dtype=torch.bfloat16)
k = torch.empty((1, heads, capacity, dim), device=device, dtype=torch.bfloat16)
v = torch.empty_like(k)
selected_k = torch.randn((1, heads, selected, dim), device=device, dtype=torch.bfloat16)
selected_v = torch.randn_like(selected_k)
k[:, :, :selected].copy_(selected_k)
v[:, :, :selected].copy_(selected_v)
indices = torch.arange(selected, device=device, dtype=torch.int64).view(1, 1, -1).expand(1, heads, -1).contiguous()
starts = torch.arange(heads, device=device, dtype=torch.int32)
counts = torch.full((1, heads), selected, device=device, dtype=torch.int32)
partial = torch.empty((1, heads, group, dim), device=device, dtype=torch.bfloat16)
partial_l = torch.empty((heads, group), device=device, dtype=torch.float32)
partial_m = torch.empty_like(partial_l)
out = torch.zeros((1, heads, group, dim), device=device, dtype=torch.float32)
denom = torch.zeros((1, heads, group), device=device, dtype=torch.float32)
maximum = torch.full_like(denom, -float('inf'))
scale = dim ** -0.5
author._fwd_kernel_qk_gen[(1, heads)](
    q, k, v, indices, scale, partial,
    *q.stride(), *k.stride(), *v.stride(), *indices.stride(), *partial.stride(),
    starts, counts, partial_l, partial_m, 1, heads, group, capacity, selected, heads,
    NUMHEADS=heads, BLOCK_M=16, BLOCK_N=128, BLOCK_KV=2048,
    BLOCK_DMODEL=dim, num_warps=4)
author._reduce_kernel_qk_gen[(1, heads, 1)](
    partial, out, *partial.stride(), *out.stride(), starts, counts, partial_l,
    partial_m, denom, maximum, group, heads, BLOCK_M=16, BLOCK_N=128,
    BLOCK_KV=2048, BLOCK_DMODEL=dim, num_warps=4)
expected = torch.matmul(
    (torch.matmul(q.float(), selected_k.float().transpose(-1, -2)) * scale).softmax(-1),
    selected_v.float(),
)
torch.testing.assert_close(out, expected, atol=0.025, rtol=0.04)
print('PASS: all 128 heads, including offsets beyond 2^31 elements, match dense reference', flush=True)

"""Check the production state-merge CUDA kernel against scatter-add."""
import argparse
import torch

from benchmarks.retro_lod_adapter import install_lod_modules

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--repeats', type=int, default=10)
    a = p.parse_args()
    install_lod_modules()
    from lod_attention.kernels.lod_kernels import new_state_delta_buffers, merge_state_in_place, _accumulate_state_deltas_kernel
    torch.manual_seed(105)
    b, h, s, t, d = a.batch, 8, 1536, 256, 128
    k = torch.randn((b,h,s,d), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    counts = torch.ones((b,h,s,1), device='cuda', dtype=torch.float32)
    mk = torch.randn((b,h,t,d), device='cuda', dtype=torch.bfloat16)
    mv = torch.randn_like(mk)
    dest = torch.randint(s, (b,h,t), device='cuda')
    idx = torch.arange(t, device='cuda').view(1,1,-1).expand(b,h,-1).contiguous()
    owners = torch.empty_like(idx)
    expected_v = v.float().scatter_add(2, dest.unsqueeze(-1).expand_as(mv), mv.float())
    expected_k = k.float().scatter_add(2, dest.unsqueeze(-1).expand_as(mk), mk.float())
    buffers = new_state_delta_buffers(k, v, s)
    mc = torch.ones((b,h,t), device='cuda')
    _accumulate_state_deltas_kernel[(b*h, 64)](
        mk, mv, mc, mc, idx, dest, owners, buffers['delta_k'], buffers['delta_v'],
        buffers['delta_counts'], buffers['touched'], counts,
        mk.stride(1), mv.stride(1), mc.stride(1), mc.stride(1), owners.stride(1),
        buffers['delta_k'].stride(1), buffers['delta_v'].stride(1), buffers['touched'].stride(1),
        counts.stride(1), counts.stride(2), TOKENS=t, TOKEN_BLOCK=4,
        HEAD_DIM=d, VALUE_DIM=d, HEAD_BLOCK_DIM=d, VALUE_BLOCK_DIM=d,
        HAS_KEY_NORMS=False, INDIRECT_SOURCE=False, num_warps=8)
    torch.cuda.synchronize()
    print('accumulate key/value max error',
          float((buffers['delta_k'] - (expected_k - k.float())).abs().max()),
          float((buffers['delta_v'] - (expected_v - v.float())).abs().max()), flush=True)
    buffers = new_state_delta_buffers(k, v, s)
    for i in range(a.repeats):
        sk, sv, c = k.clone(), v.clone(), counts.clone()
        merge_state_in_place(sk, sv, c, mk, mv, None, idx, dest, owners, buffers, active_slots=s)
        torch.cuda.synchronize()
        diff = (sv.float()-expected_v).abs()
        bad = int(((diff > .05) & (diff > expected_v.abs() * .01)).sum().item())
        print(f'iteration={i} max_abs={float(diff.max()):.6f} bad_values={bad}', flush=True)
        print(f'key_max_abs={float((sk.float()-expected_k).abs().max()):.6f}', flush=True)
        if bad or not torch.equal(owners, dest):
            print('bad counts by dim', ((diff > .05) & (diff > expected_v.abs() * .01)).sum(dim=(0,1,2)).tolist(), flush=True)
            position = torch.nonzero(diff == diff.max())[0].tolist()
            batch, head, slot, dim = position
            assigned = torch.nonzero(dest[batch,head] == slot).flatten()
            print(f'index={position} old={v[tuple(position)].item()} got={sv[tuple(position)].item()} '
                  f'expected={expected_v[tuple(position)].item()} '
                  f'inputs={mv[batch,head,assigned,dim].tolist()} count={c[batch,head,slot,0].item()}', flush=True)
            raise RuntimeError('state merge disagrees with reference')

if __name__ == '__main__':
    main()

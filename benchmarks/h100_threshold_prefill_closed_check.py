"""Independent H100 closed-prefill frontier, dispatch, and graph checks."""
import argparse
import json
import os
from pathlib import Path
import types
import sys

import torch
from benchmarks.retro_lod_adapter import RetroLODCache, install_lod_modules


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=('triton', 'blas'), default='triton')
    p.add_argument('--kernel-only', action='store_true', help='Use kernels without RetroInfer compiled dependencies')
    a = p.parse_args()
    if a.kernel_only:
        root = Path(__file__).resolve().parents[1]
        package = types.ModuleType('lod_attention')
        package.__path__ = [str(root / 'lod_attention')]
        sys.modules['lod_attention'] = package
        sys.path.insert(0, str(root / 'integrations/vllm_lod'))
    else:
        install_lod_modules()
    from benchmarks.h100_threshold_prefill_closed import route_coarse, exact_attention, threshold_prefill_closed
    from lod_attention.kernels.lod_kernels import merge_attention_branches_with_sink
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(129)
    os.environ['LOD_H100_CLOSED_MATMUL'] = a.backend
    model = types.SimpleNamespace(batch_size=1, num_gpus=1, input_length=16384, max_new_length=32,
        num_layers=1, num_heads=32, num_key_value_heads=8, head_dim=128, device_map='cuda:0', dtype=torch.bfloat16)
    engine = RetroLODCache(model, mode='two-tier').pools[0].engine
    cases = [('region', 1., 16, 16, False, False), ('region', 16., 16, 16, False, False),
             ('entry', 16., 16, 16, False, False), ('entry', 16., 32, 1, False, False),
             ('region', 16., 16, 16, True, False), ('entry', 16., 16, 16, True, False),
             ('region', 1., 16, 16, False, True)]
    rows = []
    for rule, c, bound, work, empty, stress in cases:
        sizes = torch.tensor(([1] * 257) if stress else ([0, 1, 3, 7, 29, 1024, 1025, 13] * 17)[:129], device='cuda')
        s, tokens = sizes.numel(), int(sizes.sum())
        owner = torch.repeat_interleave(torch.arange(s, device='cuda'), sizes)
        leafk = torch.randn((1, 8, tokens, 128), device='cuda', dtype=torch.bfloat16)
        leafv = torch.randn_like(leafk)
        if stress:
            leafk.zero_()
            leafk[:, :, -1] = -4
        sk, sv = torch.zeros((1, 8, s, 128), device='cuda', dtype=torch.bfloat16), torch.zeros((1, 8, s, 128), device='cuda', dtype=torch.bfloat16)
        for slot in range(s):
            sk[:, :, slot] = leafk[:, :, owner == slot].float().sum(2).bfloat16()
            sv[:, :, slot] = leafv[:, :, owner == slot].float().sum(2).bfloat16()
        counts = sizes.float()[None, None, :, None].expand(1, 8, -1, -1).contiguous()
        page = engine._new_page_cache(leafk, leafv, owner[None, None, :].expand(1, 8, -1).contiguous(),
            state_capacity=s, sequence_capacity=tokens + 256, virtual_k=leafk, virtual_v=leafv)
        if empty:
            counts.zero_()
        q = torch.randn((1, 32, 67, 128), device='cuda', dtype=torch.bfloat16)
        if stress:
            q.fill_(1)
        lk = torch.randn((1, 8, 5, 128), device='cuda', dtype=torch.bfloat16)
        lv, sinkk, sinkv = torch.randn_like(lk), torch.randn((1, 8, 1, 128), device='cuda', dtype=torch.bfloat16), torch.randn((1, 8, 1, 128), device='cuda', dtype=torch.bfloat16)
        native_counts = counts[0, :, :, 0]
        divisor = counts.bfloat16().float().clamp_min(1)
        mk = (sk.float() / divisor).bfloat16().repeat_interleave(4, 1).float()
        mv = (sv.float() / divisor).bfloat16().repeat_interleave(4, 1).float()
        dot = q.float() @ mk.transpose(-1, -2) * engine.scaling
        size = native_counts.repeat_interleave(4, 0)[None, :, None, :]
        valid = size > 0
        route = (dot / q.float().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt() + size.clamp_min(1).log()).masked_fill(~valid, -torch.inf)
        if rule == 'region':
            expected = valid & (size <= 1024) & (route.softmax(-1) > c / valid.sum(-1, keepdim=True).clamp_min(1))
        else:
            remote = size.sum(-1, keepdim=True).long()
            cut = route.logsumexp(-1, keepdim=True) - __import__('math').log(bound) - .5 * remote.clamp_min(1).float().log() + 1.e-6
            candidate = valid & (route - size.clamp_min(1).log() > cut)
            expected = candidate & ((candidate * size.long()).cumsum(-1).square() < work * work * remote)
        native = (dot + size.clamp_min(1).log()).masked_fill(~valid | expected, -torch.inf)
        local_scores = q.float() @ lk.repeat_interleave(4, 1).float().transpose(-1, -2) * engine.scaling
        local_lse = local_scores.logsumexp(-1)
        local_out = (local_scores.softmax(-1) @ lv.repeat_interleave(4, 1).float()).bfloat16()
        leaf_logits = q.float() @ leafk.repeat_interleave(4, 1).float().transpose(-1, -2) * engine.scaling
        leaf_logits.masked_fill_(~expected[..., owner], -torch.inf)
        sink_logits = q.float() @ sinkk.repeat_interleave(4, 1).float().transpose(-1, -2) * engine.scaling
        logits = torch.cat((native, leaf_logits, local_scores, sink_logits), -1)
        values = torch.cat((mv, leafv.repeat_interleave(4, 1).float(), lv.repeat_interleave(4, 1).float(), sinkv.repeat_interleave(4, 1).float()), 2)
        reference = logits.softmax(-1) @ values
        buffers = {}
        routed = route_coarse(q, sk, sv, counts, state_len=s, scale=engine.scaling, group=4,
            rule=rule, multiplier=c, entry_bound=bound, work_bound=work, leaf_capacity=tokens, buffers=buffers)
        slots, number = routed[:2]
        mask = torch.arange(slots.size(-1), device='cuda') < number[..., None]
        actual = torch.zeros_like(expected, dtype=torch.int32)
        actual.scatter_add_(-1, torch.where(mask, slots, 0), mask.int())
        mismatches = int((actual.bool() != expected).sum())
        if mismatches:
            raise AssertionError(f'selections differ: {mismatches}')
        heads = torch.arange(32, device='cuda')[None, :, None, None]
        if routed[-1] is not None:
            address = ((heads * s + torch.where(mask, slots, 0)) * 67 + routed[-1])[mask]
            if address.numel() != torch.unique(address).numel():
                raise AssertionError('expert packing contains duplicate query offsets')
            if not torch.equal(routed[-2].view(32, s), actual[0].sum(1).int()):
                raise AssertionError('expert counts differ from the selected frontier')
            limit = routed[-2][heads * s + torch.where(mask, slots, 0)]
            if not bool(((routed[-1] >= 0) & (routed[-1] < limit))[mask].all()):
                raise AssertionError('expert packing offsets are out of bounds')
        exact, el = exact_attention(engine, q, page, s, routed, buffers)
        out = merge_attention_branches_with_sink(q, sinkk, sinkv, routed[3], routed[4], exact, el,
            local_out, local_lse, kv_group_size=4, scale=engine.scaling)
        torch.testing.assert_close(out.float(), reference, atol=.02, rtol=.03)
        if stress and int(number.max()) <= 128:
            raise AssertionError('uncapped reduction fallback was not exercised')
        os.environ['LOD_H100_CLOSED_TILE'] = '33'
        wrapped = threshold_prefill_closed(engine, q, lk, lv, sk, sv, counts, None, leafk, leafv,
            state_len=s, state_capacity=s, page_cache=page, local_branch=(local_out, local_lse),
            sink_k=sinkk, sink_v=sinkv, rule=rule, multiplier=c, entry_bound=bound, work_bound=work, tile_queries=33)
        torch.testing.assert_close(wrapped.float(), reference, atol=.02, rtol=.03)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = route_coarse(q, sk, sv, counts, state_len=s, scale=engine.scaling, group=4,
                rule=rule, multiplier=c, entry_bound=bound, work_bound=work, leaf_capacity=tokens, buffers=buffers)
            eo, ee = exact_attention(engine, q, page, s, captured, buffers)
        graph.replay()
        torch.cuda.synchronize()
        replayed = merge_attention_branches_with_sink(q, sinkk, sinkv, captured[3], captured[4], eo, ee,
            local_out, local_lse, kv_group_size=4, scale=engine.scaling)
        torch.testing.assert_close(replayed.float(), reference, atol=.02, rtol=.03)
        row = dict(rule=rule, multiplier=c, entry_bound=bound, work_bound=work, empty=empty, stress=stress,
            selection_mismatches=mismatches, max_regions=int(number.max()), max_abs=float((out.float() - reference).abs().max()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.output.write_text(json.dumps(dict(stage='complete', torch=torch.__version__, backend=a.backend, records=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()

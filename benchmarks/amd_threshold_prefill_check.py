"""Check threshold AMD prefill against an independent PyTorch frontier."""
import argparse
import json
import os
from pathlib import Path

import torch

from benchmarks.amd_threshold_three_tier_check import fixture
from lod_attention.kernels.threshold_prefill import threshold_prefill_route_coarse_attention
from lod_attention.kernels.paged_prefill import paged_leaf_attention
from lod_attention.kernels.aiter_prefill_attention import merge_aiter_prefill_refinement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('triton', 'blas', 'stream'), default='triton')
    parser.add_argument('--queries', type=int, default=67)
    parser.add_argument('--rule', choices=('region', 'entry'), default='region')
    parser.add_argument('--reducer', choices=('bucket', 'tc'), default='bucket')
    args = parser.parse_args()
    os.environ['LOD_AMD_PREFILL_MATMUL'] = args.backend
    os.environ['LOD_AMD_PREFILL_RULE'] = args.rule
    torch.set_grad_enabled(False)
    records = []
    for dim, group in [(256, 6), (128, 8)]:
        pool, *_ = fixture(dim, group, False, 913)
        st, page = pool.state, pool.state['page_cache']
        for c, normalize, empty, stress in [(1., True, False, False), (4., True, False, False), (16., True, False, False),
                                    (100., True, False, False), (4., False, False, False), (16., True, True, False),
                                    (1., True, False, True)]:
            if stress:
                pool,*_=fixture(dim,group,False,913,slots=257)
                st,page=pool.state,pool.state['page_cache']
            q = (torch.randn((1, 2*group, args.queries, dim), device='cuda') * 3).bfloat16()
            counts = st['counts'].clone()
            if empty:
                counts.zero_()
            if stress:
                counts.masked_fill_(counts>32,0)
            os.environ['LOD_AMD_PREFILL_THRESHOLD'] = str(c)
            slots, coarse, hc, offsets = threshold_prefill_route_coarse_attention(q, st['state_k'], st['state_v'], counts,
                state_len=pool.state_capacity, kv_group_size=group, scale=dim**-.5,
                normalize_route_query=normalize)
            sizes = counts[0, :, :, 0]
            mk = (st['state_k'][0].float() / sizes.clamp_min(1)[..., None]).bfloat16().float()
            mv = (st['state_v'][0].float() / sizes.clamp_min(1)[..., None]).bfloat16().float()
            query = q[0].reshape(2, group, args.queries, dim).float()
            dot = torch.einsum('hgtd,hsd->hgts', query, mk) * dim**-.5
            valid = sizes > 0
            native = dot + sizes.clamp_min(1).log()[:, None, None, :]
            route = dot / query.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt() if normalize else dot
            route = (route + sizes.clamp_min(1).log()[:, None, None, :]).masked_fill(~valid[:, None, None], -torch.inf)
            expected_mask = ((route.softmax(-1) > c / valid.sum(-1).clamp_min(1)[:, None, None, None]) &
                             valid[:, None, None] & (sizes[:, None, None] <= 1024))
            if args.rule == 'entry':
                remote = sizes.sum(-1)[:, None, None, None].to(torch.int64)
                limit = 1. / (16 * remote.clamp_min(1).sqrt())
                candidate = valid[:, None, None] & (route.softmax(-1) / sizes.clamp_min(1)[:, None, None] > limit)
                prefix = (candidate * sizes[:, None, None].long()).cumsum(-1)
                work_bound = int(os.environ.get('LOD_AMD_PREFILL_WORK_BOUND', '16'))
                expected_mask = candidate & (prefix.square() < work_bound**2 * remote)
                opened = (expected_mask * sizes[:, None, None].long()).sum(-1, keepdim=True)
                assert bool(((remote == 0) | (opened.square() < work_bound**2 * remote)).all())
            if stress and args.rule == 'region' and int(expected_mask.sum(-1).max())<=64:
                raise AssertionError('stress fixture did not exercise the uncapped reduction fallback')
            actual = torch.zeros((*slots.shape[:-1], pool.state_capacity), device='cuda', dtype=torch.int32)
            actual.scatter_add_(-1, slots.clamp_min(0).long(), (slots >= 0).int())
            if not torch.equal(actual[0].reshape_as(expected_mask).bool(), expected_mask):
                raise AssertionError('threshold selections differ from independent reference')
            closed = native.masked_fill(~(valid[:, None, None] & ~expected_mask), -torch.inf)
            expected_lse = closed.logsumexp(-1).reshape_as(coarse.lse_0)
            torch.testing.assert_close(coarse.lse_0, expected_lse, atol=2e-4, rtol=2e-5)
            expected_coarse = torch.einsum('hgts,hsd->hgtd', closed.softmax(-1).nan_to_num(), mv).reshape_as(q)
            torch.testing.assert_close(coarse.output_0.float(), expected_coarse, atol=.004, rtol=.015)
            # Exact leaves, local, and sink must partition the frontier once.
            direct = os.environ.get('LOD_AMD_PREFILL_TEST_DIRECT', '')
            reference = coarse.lse_0 if direct.startswith('atomic') else None
            if direct == 'atomic-repair':
                reference = torch.full_like(coarse.lse_0, 128.0)
            exact, lse = paged_leaf_attention(q, page['leaf_k'], page['leaf_v'], page['slot_pages'],
                page['overflow_page_keys'], page['overflow_page_values'], page['overflow_used'],
                page['slot_lengths'], slots, page_indices=page['page_indices'], kv_group_size=group,
                active_slots=pool.state_capacity, scale=dim**-.5, block_m=16, block_n=16, num_warps=2,
                route_counts=coarse.route_counts, exact_row_starts=coarse.exact_row_starts, persistent_workers=7,route_head_counts=hc,route_offsets=offsets,
                variable_reduce_backend=args.reducer, reduce_routes=not coarse.fused_exact_merge,
                exact_reference=reference, query_major=direct=='query',
                exact_query_group=int(os.environ.get('LOD_AMD_PREFILL_DIRECT_QUERY_GROUP', '1')),
                exact_query_order=os.environ.get('LOD_AMD_PREFILL_DIRECT_QUERY_ORDER', 'adjacent'))
            local = torch.randn_like(q)
            local_lse = torch.randn(q.shape[:-1], device='cuda')
            output = merge_aiter_prefill_refinement(q, st['sink_k'], st['sink_v'], coarse, slots,
                exact, lse, local, local_lse, kv_group_size=group, scale=dim**-.5)
            max_error = 0.
            for h in range(2):
                ids = page['page_indices'][0, h].long()
                owners = torch.full((page['leaf_k'].size(2),), -1, device='cuda', dtype=torch.long)
                # The fixture's randomized page tables make storage order irrelevant.
                for s in range(pool.state_capacity):
                    n = int(page['slot_lengths'][0, h, s])
                    pages = []
                    for ordinal in range((n+15)//16):
                        if ordinal < 32:
                            pages.append(int(page['slot_pages'][0,h,s,ordinal]))
                        else:
                            key = s * 65536 + ordinal
                            at = (page['overflow_page_keys'][0,h] == key).nonzero()[0,0]
                            pages.append(int(page['overflow_page_values'][0,h,at]))
                    if pages:
                        leafids = ids[pages].flatten(); leafids = leafids[leafids >= 0]
                        owners[leafids] = s
                leaf_logits = torch.einsum('gtd,nd->gtn', query[h], page['leaf_k'][0,h].float()) * dim**-.5
                leaf_logits.masked_fill_(~expected_mask[h].index_select(-1, owners), -torch.inf)
                sink_logits = torch.einsum('gtd,nd->gtn', query[h], st['sink_k'][0,h].float()) * dim**-.5
                logits = torch.cat((closed[h], leaf_logits, local_lse[0,h*group:(h+1)*group,:,None], sink_logits), -1)
                weight = logits.softmax(-1)
                expected = torch.einsum('gts,sd->gtd', weight[..., :pool.state_capacity], mv[h])
                start = pool.state_capacity
                end = start + page['leaf_k'].size(2)
                expected += torch.einsum('gtn,nd->gtd', weight[...,start:end], page['leaf_v'][0,h].float())
                expected += weight[...,end:end+1] * local[0,h*group:(h+1)*group].float()
                expected += torch.einsum('gtn,nd->gtd', weight[...,end+1:], st['sink_v'][0,h].float())
                observed = output[0,h*group:(h+1)*group].float()
                torch.testing.assert_close(observed, expected, atol=.012, rtol=.025)
                max_error = max(max_error, float((observed-expected).abs().max()))
            # Same GPU-only selection and attention path must be graph safe.
            graph = torch.cuda.CUDAGraph()
            debug=os.environ.pop('LOD_AMD_PREFILL_DEBUG',None)
            with torch.cuda.graph(graph):
                threshold_prefill_route_coarse_attention(q, st['state_k'], st['state_v'], counts,
                    state_len=pool.state_capacity, kv_group_size=group, scale=dim**-.5,
                    normalize_route_query=normalize)
            graph.replay(); torch.cuda.synchronize()
            if debug is not None:os.environ['LOD_AMD_PREFILL_DEBUG']=debug
            row = dict(dim=dim, group=group, multiplier=c, rule=args.rule, reducer=args.reducer,
                       normalized=normalize, empty=empty, stress=stress,
                       mean_regions=float(expected_mask.sum(-1).float().mean()), max_abs=max_error)
            records.append(row); print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(stage='complete', checks=records), indent=2)+'\n')


if __name__ == '__main__':
    main()

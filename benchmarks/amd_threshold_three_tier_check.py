"""Independent GPU frontier checks for variable-region three-tier decode."""
import argparse
import functools
import json
import math
import os
from pathlib import Path
import types

import torch

def hash_index(key, capacity):
    x = key & 0xffffffff
    x = ((x ^ (x >> 16)) * 0x7feb352d) & 0xffffffff
    x = ((x ^ (x >> 15)) * 0x846ca68b) & 0xffffffff
    return (x ^ (x >> 16)) & (capacity - 1)


def fixture(dimension, group, quantized, seed, slots=67, boost_region=None):
    torch.manual_seed(seed)
    device = 'cuda'
    kv = 2
    base = [0, 1, 7, 16, 17, 31, 32, 33, 511, 513, 1024, 1025]
    base += [1 + (i * 13) % 37 for i in range(slots - len(base))]
    sizes = [base, list(reversed(base))]
    length = sum(base)
    capacity = sum(math.ceil(n / 16) for n in base)
    indices = torch.full((1, kv, capacity, 16), -1, dtype=torch.int32)
    table = torch.full((1, kv, slots, 32), -1, dtype=torch.int32)
    keys = torch.full((1, kv, 4096), -1, dtype=torch.int32)
    values = torch.full_like(keys, -1)
    counts = torch.zeros((1, kv, capacity), dtype=torch.int32)
    owner = torch.zeros((kv, length), dtype=torch.int64)
    owned = [[[] for _ in range(slots)] for _ in range(kv)]
    for h in range(kv):
        permutation = torch.randperm(length)
        pages = torch.randperm(capacity).tolist()
        begin = page_next = 0
        for slot, n in enumerate(sizes[h]):
            owner[h, permutation[begin:begin+n]] = slot
            for ordinal, start in enumerate(range(begin, begin+n, 16)):
                ids = permutation[start:min(start+16, begin+n)]
                page = pages[page_next]
                page_next += 1
                owned[h][slot].append(page)
                indices[0, h, page, :len(ids)] = ids.int()
                counts[0, h, page] = len(ids)
                if ordinal < 32:
                    table[0, h, slot, ordinal] = page
                else:
                    key = slot * 65536 + ordinal
                    at = hash_index(key, 4096)
                    while int(keys[0, h, at]) >= 0:
                        at = (at + 1) & 4095
                    keys[0, h, at] = key
                    values[0, h, at] = page
            begin += n
    indices, table, keys, values, counts, owner = (x.to(device) for x in (indices, table, keys, values, counts, owner))
    lk = torch.randn((1, kv, length, dimension), device=device, dtype=torch.bfloat16)
    lv = torch.randn_like(lk)
    if boost_region is not None:
        lk[0, 0, owner[0] == boost_region, 0] += 16
    hh = torch.arange(kv, device=device)[:, None, None]
    mask = indices[0] >= 0
    sk = (lk[0][hh, indices[0].clamp_min(0).long()].float() * mask[..., None]).sum(2).to(torch.bfloat16)[None]
    sv = (lv[0][hh, indices[0].clamp_min(0).long()].float() * mask[..., None]).sum(2).to(torch.bfloat16)[None]
    state_k = torch.zeros((kv, slots, dimension), device=device).scatter_add_(1, owner[..., None].expand(-1, -1, dimension), lk[0].float()).to(torch.bfloat16)[None]
    state_v = torch.zeros((kv, slots, dimension), device=device).scatter_add_(1, owner[..., None].expand(-1, -1, dimension), lv[0].float()).to(torch.bfloat16)[None]
    page = dict(slot_pages=table, overflow_page_keys=keys, overflow_page_values=values,
                overflow_used=torch.ones((), device=device, dtype=torch.int32), slot_lengths=torch.tensor([sizes], device=device, dtype=torch.int32),
                leaf_k=lk, leaf_v=lv, page_indices=indices, page_sum_k=sk, page_sum_v=sv, page_counts=counts, leaf_quant_bits=0)
    effective_k, effective_v = sk.float(), sv.float()
    exact_k, exact_v = lk.float(), lv.float()
    if quantized:
        from lod_attention.kernels.paged_cache import quantize_page_summaries_int8, quantize_virtual_paged_kv
        qsk, qsv, ssk, ssv = quantize_page_summaries_int8(sk, sv, quant_group_size=4)
        pk = torch.empty((1, kv, length, dimension//2), device=device, dtype=torch.uint8)
        pv = torch.empty_like(pk)
        kscale = torch.empty((1, kv, capacity, dimension//4), device=device, dtype=torch.bfloat16)
        vscale = torch.empty_like(kscale)
        qc = torch.zeros_like(counts)
        quantize_virtual_paged_kv(lk, lv, indices, sk, sv, counts, pk, pv, kscale, vscale, qc)
        page.update(leaf_quant_bits=4, quantized_leaf_k=pk, quantized_leaf_v=pv, page_k_scales=kscale,
                    page_v_scales=vscale, page_quantized_counts=qc, quantized_page_sum_k=qsk,
                    quantized_page_sum_v=qsv, page_sum_k_scales=ssk, page_sum_v_scales=ssv)
        effective_k = qsk.float() * ssk.float().repeat_interleave(4, -1)
        effective_v = qsv.float() * ssv.float().repeat_interleave(4, -1)
        exact_k, exact_v = torch.empty_like(lk, dtype=torch.float32), torch.empty_like(lv, dtype=torch.float32)
        for h in range(kv):
            for page_id in range(capacity):
                ids = indices[0, h, page_id]
                ids = ids[ids >= 0].long()
                for packed, scale, sums, dest in ((pk, kscale, effective_k, exact_k), (pv, vscale, effective_v, exact_v)):
                    code = packed[0, h, ids].int()
                    code = torch.stack((code & 15, (code >> 4) & 15), -1).flatten(-2).float() - 8
                    dest[0, h, ids] = code * scale[0, h, page_id].float().repeat_interleave(4) + sums[0, h, page_id] / len(ids)
    state = dict(state_k=state_k, state_v=state_v, counts=page['slot_lengths'].float()[..., None], page_cache=page,
                 recent_k=torch.randn((1, kv, 128, dimension), device=device, dtype=torch.bfloat16),
                 recent_v=torch.randn((1, kv, 128, dimension), device=device, dtype=torch.bfloat16),
                 sink_k=torch.randn((1, kv, 1, dimension), device=device, dtype=torch.bfloat16),
                 sink_v=torch.randn((1, kv, 1, dimension), device=device, dtype=torch.bfloat16))
    pool = types.SimpleNamespace(settings=types.SimpleNamespace(levels=3, quant_group_size=4 if quantized else 32),
            dtype=torch.bfloat16, kv_heads=kv, state_capacity=slots, state=state, decode_local_limit=64,
            engine=types.SimpleNamespace(scaling=dimension**-.5, leaf_hash_probes=8, leaf_block_n=16 if quantized or dimension==256 else 32),
            state_lens=torch.tensor([slots], device=device, dtype=torch.int32), local_lens=torch.tensor([41], device=device, dtype=torch.int32))
    q = torch.randn((1, 1, kv*group, dimension), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 1, kv, dimension), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return pool, q, k, v, owned, effective_k, effective_v, exact_k, exact_v


def check(dimension, group, quantized, multiplier, workers, seed, empty=False, ties=False,
          round_counts_bf16=False, cached=False, pages_per_slot=2, page_multiplier=None, page_rule='page',page_query='native',page_residual='region'):
    from benchmarks.amd_threshold_three_tier import adaptive_page_attention, threshold_attention
    pool, q, k, v, owned, pk, pv, lk, lv = fixture(dimension, group, quantized, seed)
    if ties:
        q.zero_()
    if empty:
        pool.state_lens.zero_()
    H, S = pool.kv_heads, pool.state_capacity
    st = pool.state
    query = q[0, 0].reshape(H, group, dimension).float()
    count = st['counts'][0, :, :, 0]
    valid = (count > 0) & (torch.arange(S, device=q.device)[None] < pool.state_lens)
    divisor = count.to(torch.bfloat16).float() if round_counts_bf16 else count
    mk = (st['state_k'][0].float() / divisor.clamp_min(1)[..., None]).to(torch.bfloat16).float()
    mv = (st['state_v'][0].float() / divisor.clamp_min(1)[..., None]).to(torch.bfloat16).float()
    normalized = query * query.square().mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
    route = (torch.einsum('hgd,hsd->hgs', normalized, mk) * dimension**-.5 + count.clamp_min(1).log()[:, None]).masked_fill(~valid[:, None], -torch.inf)
    selected = (route.softmax(-1) > multiplier / valid.sum(-1).clamp_min(1)[:, None, None]) & valid[:, None] & (count[:, None] <= 1024)
    expected = torch.empty_like(query)
    expected_page_stats = torch.zeros((H,group,4),device=q.device,dtype=torch.int32)
    for h in range(H):
        for g in range(group):
            scores = []
            values = []
            for slot in range(S):
                if not bool(valid[h, slot]):
                    continue
                if not bool(selected[h, g, slot]):
                    scores.append((query[h, g] @ mk[h, slot] * dimension**-.5 + count[h, slot].log())[None])
                    values.append(mv[h, slot][None])
                    continue
                pages = owned[h][slot]
                pc = st['page_cache']['page_counts'][0, h, pages].float()
                page_q = normalized[h,g] if page_multiplier is not None and page_query=='normalized' else query[h,g]
                page_score = (pk[0, h, pages] / pc[:, None]) @ page_q * dimension**-.5 + pc.log()
                order = sorted(range(len(pages)), key=lambda i: (-float(page_score[i]), i//pool.engine.leaf_block_n, -pages[i]))[:pages_per_slot]
                picked = [pages[i] for i in order]
                if page_multiplier is not None:
                    cutoff = page_multiplier * pc / count[h,slot] * (1.+1.e-6) if page_rule=='entry' else page_multiplier / len(pages)
                    keep = page_score.softmax(-1) > cutoff
                    picked = pages if len(pages)==1 else [p for p,yes in zip(pages,keep.tolist()) if yes]
                ids = st['page_cache']['page_indices'][0, h, picked].flatten()
                ids = ids[ids >= 0].long()
                expected_page_stats[h,g] += torch.tensor([1,len(pages),len(picked),len(ids)],device=q.device,dtype=torch.int32)
                scores.append(lk[0, h, ids] @ query[h, g] * dimension**-.5)
                values.append(lv[0, h, ids])
                left = count[h, slot] - len(ids)
                if float(left) > 0:
                    if page_residual=='pages':
                        remaining=[p for p in pages if p not in picked]
                        residual_k=pk[0,h,remaining].sum(0)/left
                        residual_v=pv[0,h,remaining].sum(0)/left
                    else:
                        residual_k = (st['state_k'][0, h, slot].float() - pk[0, h, picked].sum(0)) / left
                        residual_v = (st['state_v'][0, h, slot].float() - pv[0, h, picked].sum(0)) / left
                    scores.append((residual_k @ query[h, g] * dimension**-.5 + left.log())[None])
                    values.append(residual_v[None])
            local_k = torch.cat((st['recent_k'][0, h, :41].float(), st['sink_k'][0, h].float(), k[0, 0, h][None].float()))
            local_v = torch.cat((st['recent_v'][0, h, :41].float(), st['sink_v'][0, h].float(), v[0, 0, h][None].float()))
            scores.append(local_k @ query[h, g] * dimension**-.5)
            values.append(local_v)
            expected[h, g] = torch.cat(scores).softmax(0) @ torch.cat(values)
    out = torch.empty_like(q)
    os.environ['LOD_AMD_THRESHOLD_PAGE_WORKERS'] = str(workers)
    options = dict(round_counts_bf16=round_counts_bf16, pages_per_slot=pages_per_slot,
                   cached_means=(mk.to(torch.bfloat16), mv.to(torch.bfloat16)) if cached else None)
    if page_multiplier is not None:
        import functools
        from benchmarks.h100_threshold_pages import page_threshold_attention
        if torch.version.hip:
            options['page_attention'] = adaptive_page_attention(pool, q, workers, page_multiplier)
            stats = pool._amd_page_threshold_stats
        else:
            stats = torch.empty((H*group,workers,4),device=q.device,dtype=torch.int32)
            options['page_attention'] = functools.partial(page_threshold_attention,
                multiplier=page_multiplier,stats_buffer=stats,rule=page_rule,query_score=page_query,residual=page_residual)
    threshold_attention(pool, q, k, v, out, multiplier, **options)
    torch.cuda.synchronize()
    got = out[0, 0].reshape(H, group, dimension).float()
    actual_mask = pool._threshold_scratch['allowed'].reshape(H, group, S).bool()
    if not torch.equal(actual_mask, selected):
        raise RuntimeError(f'selected region mismatch: {int((actual_mask != selected).sum())}')
    torch.testing.assert_close(got, expected, rtol=.008, atol=.008)
    if page_multiplier is not None and not torch.equal(stats.sum(1).reshape(H,group,4).int(),expected_page_stats):
        raise RuntimeError('page threshold work counters disagree with the independent frontier')
    if int(pool.local_lens.item()) != 42 or not torch.equal(st['recent_k'][0, :, 41], k[0, 0]) or not torch.equal(st['recent_v'][0, :, 41], v[0, 0]):
        raise RuntimeError('current KV append or length update mismatch')
    if seed == 2:
        frozen = out.clone()
        pool.local_lens.fill_(41)
        def invoke():
            threshold_attention(pool, q, k, v, out, multiplier, **options)
            pool.local_lens.sub_(1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(4):
                invoke()
        for _ in range(8):
            graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(out, frozen) or int(pool.local_lens.item()) != 41:
            raise RuntimeError('graph replay changed the fixed-cache frontier or length')
    return dict(dimension=dimension, group=group, kv_bits=4 if quantized else 0, multiplier=multiplier,
                workers=workers, empty=empty, ties=ties, max_abs=float((got-expected).abs().max()),
                mean_opened_regions=float(selected.sum(-1).float().mean()), selection_mismatches=0,
                cached_means=cached, round_counts_bf16=round_counts_bf16, pages_per_slot=pages_per_slot,
                page_multiplier=page_multiplier,page_rule=page_rule,page_query=page_query,page_residual=page_residual)


def check_batch(quantized, page_multiplier=None, page_rule='page', page_query='native', page_residual='region'):
    from benchmarks.amd_threshold_three_tier import threshold_attention

    # Force routed regions from one page up to the 64-page eligibility limit.
    factored_scales = quantized and os.environ.get('LOD_H100_INT4_FACTOR_PAGE_SCALES') == '1'
    fixed_kernel = page_multiplier is None and (os.environ.get('LOD_H100_FIXED_PAGE_SCAN') is not None
        or factored_scales or os.environ.get('LOD_H100_THREE_TIER_GQA_LOCAL') == '1')
    adaptive_kernel = page_multiplier is not None and 'LOD_H100_ADAPTIVE_EXACT_PAGES' in os.environ
    if fixed_kernel or adaptive_kernel:
        regions = (7, 6, 7, 8, 9, 10, 2, 5)
    else:
        regions = (7 if page_multiplier is None else 9,) * 8
    fixtures = [fixture(128, 4, quantized, seed, boost_region=region)[:4]
                for seed, region in enumerate(regions)]
    pools, queries, keys, values = zip(*fixtures, strict=True)
    for i, pool in enumerate(pools):
        pool.state_lens.fill_(0 if i == 0 else 60 + i)
        pool.local_lens.fill_(41 + i)

    batch = types.SimpleNamespace(**vars(pools[0]))
    batch.state = {name: torch.cat([pool.state[name] for pool in pools])
                   for name in batch.state if name != 'page_cache'}
    pages = [pool.state['page_cache'] for pool in pools]
    batch.state['page_cache'] = {
        name: torch.cat([page[name] for page in pages]) if torch.is_tensor(value) and value.ndim else value
        for name, value in pages[0].items()
    }
    batch.state_lens = torch.cat([pool.state_lens for pool in pools])
    batch.local_lens = torch.cat([pool.local_lens for pool in pools])
    q, k, v = (torch.cat(parts) for parts in (queries, keys, values))
    actual, expected = torch.empty_like(q), torch.empty_like(q)

    def decoder(pool, *, batched=False):
        count = pool.state['counts'].to(torch.bfloat16).float().clamp_min(1)
        means = tuple((pool.state[name].float() / count).to(torch.bfloat16)
                      for name in ('state_k', 'state_v'))
        workers = int(os.environ.get('LOD_H100_THREE_TIER_WORKERS', '32')) if batched and (fixed_kernel or adaptive_kernel) else (
            64 if page_multiplier is None else 32)
        page_attention = None
        if page_multiplier is not None:
            from benchmarks.h100_threshold_pages import page_threshold_attention
            from benchmarks.h100_threshold_three_tier import adaptive_page_options
            stats = torch.empty((pool.state_lens.numel() * q.size(2), workers, 4),
                                device=q.device, dtype=torch.int32)
            pool._check_page_stats = stats.view(-1, q.size(2), workers, 4)
            page_attention = functools.partial(
                page_threshold_attention, multiplier=page_multiplier, stats_buffer=stats,
                rule=page_rule, query_score=page_query, residual=page_residual,
                page_threshold_warps=int(os.environ.get('LOD_H100_PAGE_THRESHOLD_WARPS', '2')) if batched else 2,
                **(adaptive_page_options() if batched else {}))
        elif batched and not quantized and 'LOD_H100_FIXED_PAGE_SCAN' in os.environ:
            from benchmarks.h100_threshold_fixed_pages import fixed_two_page_attention
            page_attention = functools.partial(fixed_two_page_attention,
                                               small_scan=int(os.environ['LOD_H100_FIXED_PAGE_SCAN']))
        elif batched and factored_scales:
            from lod_attention.kernels.paged_prefill import query_major_indexed_residual_page_attention
            page_attention = functools.partial(query_major_indexed_residual_page_attention, factor_int4_page_scales=True)

        def attention(q, k, v, out):
            page_warps = int(os.environ.get('LOD_H100_THREE_TIER_PAGE_WARPS', '1')) if batched and fixed_kernel else (
                1 if quantized else 2)
            tile = int(os.environ.get('LOD_H100_THRESHOLD_TILE', '64')) if batched and (fixed_kernel or adaptive_kernel) else 64
            route_warps = int(os.environ.get('LOD_H100_THREE_TIER_ROUTE_WARPS', '4')) if batched and fixed_kernel else 4
            threshold_attention(pool, q, k, v, out, 16, tile=tile, workers=workers, page_warps=page_warps,
                                precise=True, cached_means=means, round_counts_bf16=True,
                                page_attention=page_attention,
                                page_block_n=(int(os.environ['LOD_H100_THREE_TIER_PAGE_BLOCK'])
                                    if batched and 'LOD_H100_THREE_TIER_PAGE_BLOCK' in os.environ else None),
                                trim_idle_merge=batched and os.environ.get('LOD_H100_THREE_TIER_TRIM_MERGE') == '1',
                                materialize_cutoffs=batched and os.environ.get('LOD_H100_THREE_TIER_MATERIALIZE_CUTOFFS') == '1',
                                gqa_local=batched and os.environ.get('LOD_H100_THREE_TIER_GQA_LOCAL') == '1',
                                route_warps=route_warps,
                                fused_pack=batched and os.environ.get('LOD_H100_THREE_TIER_FUSED_PACK') == '1',
                                local_tile=int(os.environ['LOD_H100_THREE_TIER_LOCAL_TILE']) if batched and 'LOD_H100_THREE_TIER_LOCAL_TILE' in os.environ else None)
        return attention

    batched = decoder(batch, batched=True)
    singles = [decoder(pool) for pool in pools]
    initial = batch.local_lens.clone()
    max_abs = 0.0
    batched(q, k, v, actual)
    batch.local_lens.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        batched(q, k, v, actual)
    for _ in range(3):
        for inp in (q, k, v):
            inp.normal_()
        q[1:, 0, 0, 0] = 16
        graph.replay()
        selected = batch._threshold_scratch['allowed'].view(8, q.size(2), batch.state_capacity)
        assert all(selected[i, 0, region] for i, region in enumerate(regions) if i)
        assert not selected[0].any()
        if page_multiplier is not None:
            if adaptive_kernel:
                assert torch.any(batch._check_page_stats[1:, 0, :, 2].sum(-1) > 2)
            else:
                assert torch.all(batch._check_page_stats[1:, 0, :, 2].sum(-1) > 2)
        for i, (pool, single) in enumerate(zip(pools, singles, strict=True)):
            row = slice(i, i + 1)
            single(q[row], k[row], v[row], expected[row])
            torch.testing.assert_close(selected[i], pool._threshold_scratch['allowed'], rtol=0, atol=0)
            torch.testing.assert_close(batch.local_lens[row], pool.local_lens, rtol=0, atol=0)
            for name in ('recent_k', 'recent_v'):
                torch.testing.assert_close(batch.state[name][row], pool.state[name], rtol=0, atol=0)
            if page_multiplier is not None:
                torch.testing.assert_close(batch._check_page_stats[row].sum(2), pool._check_page_stats.sum(2), rtol=0, atol=0)
        max_abs = max(float((actual.float() - expected.float()).abs().max()), max_abs)
        torch.testing.assert_close(actual, expected, rtol=.008, atol=.008)
    return dict(batch_size=8, kv_bits=4 if quantized else 0, graph_replays=3, max_abs=max_abs,
                page_multiplier=page_multiplier, page_rule=page_rule, page_query=page_query,
                page_residual=page_residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--h100', action='store_true', help='Llama GQA with H100 centroid arithmetic')
    parser.add_argument('--batch-check', action='store_true', help='Compare a CUDA graph with eight independent rows')
    parser.add_argument('--pages', type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument('--page-threshold',type=float)
    parser.add_argument('--page-threshold-rule',choices=('page','entry'),default='page')
    parser.add_argument('--page-query',choices=('native','normalized'),default='native')
    parser.add_argument('--page-residual',choices=('region','pages'),default='region')
    args = parser.parse_args()
    if args.page_threshold is not None and not args.h100:
        if (args.page_threshold_rule, args.page_residual) != ('page', 'region'):
            parser.error('AMD adaptive check uses the per-page cutoff and region residual')
        args.page_query = 'normalized'
    if args.batch_check and (not args.h100 or args.pages != 2):
        parser.error('--batch-check requires --h100 and the default fixed page budget')
    if args.h100:
        from benchmarks.retro_lod_adapter import install_lod_modules
        install_lod_modules()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [(0.01, 1, 0, False, False), (4, 3, 1, False, False), (16, 64, 2, False, False),
             (4096, 8, 3, False, False), (0.1, 3, 4, False, True), (16, 8, 5, True, False)]
    rows = []
    geometries = ((128, 4),) if args.h100 else ((128, 8), (256, 6))
    for dimension, group in (() if args.batch_check else geometries):
        for quantized in ((False,) if args.page_threshold is not None else (False, True)):
            for c, workers, seed, empty, ties in cases:
                for cached in ((False, True) if args.h100 else (False,)):
                    row = check(dimension, group, quantized, c, workers, seed, empty, ties,
                                round_counts_bf16=args.h100, cached=cached, pages_per_slot=args.pages,
                                page_multiplier=args.page_threshold,page_rule=args.page_threshold_rule,page_query=args.page_query,page_residual=args.page_residual)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    if args.batch_check:
        for quantized in ((False,) if args.page_threshold is not None else (False, True)):
            row = check_batch(quantized, args.page_threshold, args.page_threshold_rule,
                              args.page_query, args.page_residual)
            rows.append(row)
            print(json.dumps(row), flush=True)
    from benchmarks._identity import benchmark_identity
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(stage='complete', records=rows,
        hardware=str(torch.cuda.get_device_properties(0)), benchmark_identity=benchmark_identity()), indent=2)+'\n')


if __name__ == '__main__':
    main()

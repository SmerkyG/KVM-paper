"""Compare frozen LoD exact routes with AITER BF16 varlen attention."""

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph

from lod_attention.kernels.aiter_prefill_attention import merge_aiter_prefill_refinement
from lod_attention.kernels.paged_prefill import (
    _materialize_prefill_regions, _pack_expert_routes, paged_leaf_attention,
)
from lod_attention.kernels.threshold_prefill import threshold_prefill_route_coarse_attention


@triton.jit
def _gather_queries(Q, ExpertOrder, BucketOrder, PackedQ, Destinations, Starts,
                    N: tl.constexpr, D: tl.constexpr, W: tl.constexpr, M: tl.constexpr):
    i = tl.program_id(0) * M + tl.arange(0, M)
    packed = tl.load(BucketOrder + i, mask=i < N, other=0)
    route = tl.load(ExpertOrder + packed, mask=i < N, other=0)
    row = route // W
    dest = tl.load(Starts + row, mask=i < N, other=0) + route % W
    d = tl.arange(0, D)
    query = tl.load(Q + row[:, None].to(tl.int64) * D + d[None, :], mask=i[:, None] < N, other=0)
    tl.store(PackedQ + i[:, None].to(tl.int64) * D + d[None, :], query, mask=i[:, None] < N)
    tl.store(Destinations + i, dest, mask=i < N)


@triton.jit
def _scatter_leaf_outputs(Out, LSE, Destinations, NativeOut, NativeLSE,
                          N: tl.constexpr, D: tl.constexpr, M: tl.constexpr):
    i = tl.program_id(0) * M + tl.arange(0, M)
    dest = tl.load(Destinations + i, mask=i < N, other=0).to(tl.int64)
    d = tl.arange(0, D)
    out = tl.load(Out + i[:, None].to(tl.int64) * D + d[None, :], mask=i[:, None] < N, other=0)
    lse = tl.load(LSE + i, mask=i < N, other=0)
    tl.store(NativeOut + dest[:, None] * D + d[None, :], out, mask=i[:, None] < N)
    tl.store(NativeLSE + dest, lse, mask=i < N)


def benchmark(args):
    from aiter.ops.mha import flash_attn_varlen_func, mha_varlen_fwd

    snapshot = torch.load(args.snapshot, map_location='cuda', weights_only=False)
    data = snapshot['data']
    page = data['pages']
    q = data['q'][..., :args.queries, :].contiguous()
    b, h, t, d = q.shape
    kv = snapshot['kv_heads']
    group = h // kv
    regions = data['s']
    experts = b * kv * regions
    engine = snapshot['engine']
    block_m, block_n = engine['leaf_block_m'], engine['leaf_block_n']
    scale = float(engine['scaling'])
    probes = -1 if page.get('paged_page_directory') else (8 if page.get('overflow_active') else 0)
    state_capacity = page['slot_lengths'].size(-1)
    page_capacity, page_size = page['page_indices'].shape[-2:]
    leaf_capacity = page['leaf_k'].size(2)
    slots, coarse, hc, offsets = threshold_prefill_route_coarse_attention(
        q, data['k'], data['v'], data['c'], state_len=regions, kv_group_size=group,
        scale=scale, normalize_route_query=True)
    width = slots.size(-1)
    buffers = {}
    route_buffers = {}
    zero = torch.zeros_like(q)
    empty_lse = torch.full(q.shape[:-1], -torch.inf, device=q.device)

    def merge(out, lse):
        return merge_aiter_prefill_refinement(
            q, data['sink_k'], data['sink_v'], coarse, slots, out, lse, zero, empty_lse,
            kv_group_size=group, scale=scale)

    def native():
        return paged_leaf_attention(
            q, page['leaf_k'], page['leaf_v'], page['slot_pages'], page['overflow_page_keys'],
            page['overflow_page_values'], page['overflow_used'], page['slot_lengths'], slots,
            page_indices=page['page_indices'], kv_group_size=group, active_slots=regions, scale=scale,
            block_m=block_m, block_n=block_n, num_warps=engine['leaf_num_warps'], hash_probes=probes,
            route_head_counts=hc, route_offsets=offsets, route_counts=coarse.route_counts,
            exact_row_starts=coarse.exact_row_starts, reduce_routes=False, persistent_workers=608,
            buffers=buffers)

    def route_order():
        return _pack_expert_routes(
            slots, active_slots=regions, kv_heads=kv, kv_group_size=group, expert_count=experts,
            block_m=block_m, head_counts=hc, route_offsets=offsets, buffers=route_buffers,
            live_prefix_counts=coarse.route_counts)

    order, counts, starts, *_ = route_order()
    lengths = page['slot_lengths'][..., :regions].contiguous().view(-1)
    starts_k = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    lengths_host = lengths.tolist()
    counts_host = counts.tolist()
    qs = starts.tolist()
    ks = starts_k.tolist()
    total_k, total_q = ks[-1], qs[-1]
    key = q.new_empty(total_k, 1, d)
    value = torch.empty_like(key)
    valid = torch.empty(total_k, device=q.device, dtype=torch.uint8)

    def gather_kv():
        _materialize_prefill_regions[(experts,)](
            page['leaf_k'], page['leaf_v'], page['page_indices'], page['slot_pages'],
            page['overflow_page_keys'], page['overflow_page_values'], page['overflow_used'],
            page['slot_lengths'], counts, starts_k, key, value, valid, regions, d, state_capacity,
            leaf_capacity, page_capacity, page['slot_pages'].size(3),
            page['overflow_page_values'].size(2), probes, page_size, 32, num_warps=4)

    native_out, native_lse = native()
    reference = merge(native_out, native_lse)
    gather_kv()
    shape = (b, h, t, width)
    aiter_out = torch.empty((*shape, d), device=q.device, dtype=q.dtype)
    aiter_lse = torch.empty(shape, device=q.device, dtype=torch.float32)
    control = {
        'native_ms': do_bench_cudagraph(native, rep=args.rep),
        'native_with_merge_ms': do_bench_cudagraph(lambda: merge(*native()), rep=args.rep),
    }
    print(json.dumps(control), flush=True)
    result = dict(layer=snapshot['layer'], queries=t, all_exact_routes=total_q,
                  used_regions=sum(c > 0 for c in counts_host), **control, runs=[])

    for scheme in args.schemes.split(','):
        # Only the *frozen experiment* uses data-dependent host lists. Capture
        # query/region movement and the common dispatch within the timed path.
        buckets = {}
        for region, (nq, nk) in enumerate(zip(counts_host, lengths_host)):
            if nq == 0:
                continue
            if nk <= 0:
                raise RuntimeError('opened empty region')
            tile = nq if scheme == 'length' else int(scheme)
            for first in range(0, nq, tile):
                count = min(tile, nq - first)
                key_bucket = (count - 1).bit_length() if scheme == 'length' else 0
                buckets.setdefault(key_bucket, []).append((region, first, count, nk))

        tasks = []
        for groups in buckets.values():
            n_q = max(count for _, _, count, _ in groups)
            n_k = max(nk for _, _, _, nk in groups)
            q_ids = torch.cat([torch.arange(qs[r]+first, qs[r]+first+count, dtype=torch.int32)
                               for r, first, count, _ in groups]).to(q.device)
            k_ids = torch.cat([torch.arange(ks[r], ks[r]+nk, dtype=torch.long) for r, _, _, nk in groups]).to(q.device)
            if not valid.index_select(0, k_ids).all().item():
                raise RuntimeError('frozen region contains missing keys')
            q_cu = torch.tensor([0] + [n for _, _, n, _ in groups], dtype=torch.int32, device=q.device).cumsum(0, dtype=torch.int32)
            k_cu = torch.tensor([0] + [n for _, _, _, n in groups], dtype=torch.int32, device=q.device).cumsum(0, dtype=torch.int32)
            p_q = q.new_empty(q_ids.numel(), 1, d)
            p_k = q.new_empty(k_ids.numel(), 1, d)
            p_v = torch.empty_like(p_k)
            dest = torch.empty_like(q_ids)
            task_out = torch.empty_like(p_q)
            tasks.append((q_ids, k_ids, q_cu, k_cu, n_q, n_k, p_q, p_k, p_v, dest, task_out))

        def gather(route_indices):
            for q_ids, k_ids, _, _, _, _, p_q, p_k, p_v, dest, _ in tasks:
                size = q_ids.numel()
                _gather_queries[(triton.cdiv(size, 32),)](
                    q, route_indices, q_ids, p_q, dest, coarse.exact_row_starts,
                    size, d, width, 32, num_warps=4)
                torch.index_select(key, 0, k_ids, out=p_k)
                torch.index_select(value, 0, k_ids, out=p_v)

        def attention(scatter=False):
            for q_ids, _, q_cu, k_cu, nq, nk, p_q, p_k, p_v, dest, out in tasks:
                if args.backend == 'ck':
                    result, lse, *_ = mha_varlen_fwd(
                        p_q, p_k, p_v, q_cu, k_cu, nq, nk, 0, 0., scale, 0.,
                        False, False, -1, -1, 0, True, False, out=out)
                else:
                    result, lse = flash_attn_varlen_func(
                        p_q, p_k, p_v, q_cu, k_cu, nq, nk,
                        softmax_scale=scale, causal=False, return_lse=True, out=out)
                if scatter:
                    size = q_ids.numel()
                    _scatter_leaf_outputs[(triton.cdiv(size, 32),)](
                        result, lse, dest, aiter_out, aiter_lse, size, d, 32, num_warps=4)

        def full():
            current_order, *_ = route_order()
            gather_kv()
            gather(current_order)
            attention(scatter=True)

        full()
        output = merge(aiter_out, aiter_lse)
        # Compare only actual refinement outputs. The reserved tails are uninitialized.
        native_values = native_out.view(-1, d)[:total_q]
        other_values = aiter_out.view(-1, d)[:total_q]
        native_mass = native_lse.view(-1)[:total_q]
        other_mass = aiter_lse.view(-1)[:total_q]
        row = dict(
            scheme=scheme, buckets=len(tasks), attention_groups=sum(x[2].numel()-1 for x in tasks),
            packed_k_rows=sum(x[1].numel() for x in tasks),
            native_values_bitwise=torch.equal(native_values, other_values),
            native_lse_bitwise=torch.equal(native_mass, other_mass),
            lse_max_difference=float((native_mass-other_mass).abs().max()),
            values_max_difference=float((native_values.float()-other_values.float()).abs().max()),
            merge_max_difference=float((reference.float()-output.float()).abs().max()),
            merge_bitwise=torch.equal(reference, output),
        )
        row.update(aiter_only_ms=do_bench_cudagraph(attention, rep=args.rep),
                   full_ms=do_bench_cudagraph(full, rep=args.rep),
                   full_with_merge_ms=do_bench_cudagraph(lambda: (full(), merge(aiter_out, aiter_lse)), rep=args.rep))
        print(json.dumps(row), flush=True)
        result['runs'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=4096)
    parser.add_argument('--schemes', default='length,256,512')
    parser.add_argument('--rep', type=int, default=45)
    parser.add_argument('--backend', choices=('auto', 'ck'), default='auto')
    args = parser.parse_args()
    result = benchmark(args)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

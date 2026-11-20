"""Check experimental threshold prefill against an independent dense frontier."""
import argparse
import json
import math
import os
import types
from pathlib import Path

import torch

from benchmarks.retro_lod_adapter import RetroLODCache, install_lod_modules
from benchmarks.h100_threshold_prefill import threshold_prefill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=3)
    parser.add_argument('--tile-queries', type=int, default=2)
    parser.add_argument('--slots', type=int, default=12)
    parser.add_argument('--entry-only', action='store_true', help='Skip the equal-region case for bounded-only implementations')
    parser.add_argument('--compare-reuse', action='store_true', help='Compare baseline and reused-score frontiers on identical synthetic states')
    args = parser.parse_args()
    if args.queries < 1 or args.tile_queries < 1 or args.slots < 1:
        raise ValueError('query, tile, and slot counts must be positive')
    install_lod_modules()
    from benchmarks.h100_threshold_prefill_fused import threshold_prefill_fused, _sample_selected
    torch.set_grad_enabled(False)
    torch.manual_seed(117)
    model = types.SimpleNamespace(batch_size=1, num_gpus=1, input_length=1024,
        max_new_length=32, num_layers=1, num_heads=32, num_key_value_heads=8,
        head_dim=128, device_map='cuda:0', dtype=torch.bfloat16)
    pool = RetroLODCache(model, mode='two-tier').pools[0]
    engine = pool.engine
    device = 'cuda'
    pattern = torch.tensor([3, 13, 30, 2, 9, 5, 19, 7, 4, 8, 6, 11], device=device)
    sizes = pattern.repeat((args.slots + pattern.numel() - 1) // pattern.numel())[:args.slots]
    tokens, slots = int(sizes.sum()), int(sizes.numel())
    owner = torch.repeat_interleave(torch.arange(slots, device=device), sizes)
    owners = owner[None, None, :].expand(1, 8, -1).contiguous()
    leaf_k = torch.randn((1, 8, tokens, 128), dtype=torch.bfloat16, device=device)
    leaf_v = torch.randn_like(leaf_k)
    page = engine._new_page_cache(leaf_k, leaf_v, owners, state_capacity=slots,
                                  sequence_capacity=tokens + 256, virtual_k=leaf_k,
                                  virtual_v=leaf_v)
    state_k = torch.zeros((1, 8, slots, 128), device=device, dtype=torch.bfloat16)
    state_v = torch.zeros_like(state_k)
    for slot in range(slots):
        state_k[:, :, slot] = leaf_k[:, :, owner == slot].float().sum(2).to(torch.bfloat16)
        state_v[:, :, slot] = leaf_v[:, :, owner == slot].float().sum(2).to(torch.bfloat16)
    counts = sizes.float()[None, None, :, None].expand(1, 8, -1, -1).contiguous()
    q = torch.randn((1, 32, args.queries, 128), device=device, dtype=torch.bfloat16)
    local_k = torch.randn((1, 8, 5, 128), device=device, dtype=torch.bfloat16)
    local_v = torch.randn_like(local_k)
    sink_k = torch.randn((1, 8, 1, 128), device=device, dtype=torch.bfloat16)
    sink_v = torch.randn_like(sink_k)
    qf = q.float(); lk = local_k.repeat_interleave(4, 1).float(); lv = local_v.repeat_interleave(4, 1).float()
    local_scores = torch.matmul(qf, lk.transpose(-1, -2)) * engine.scaling
    local_lse = torch.logsumexp(local_scores, -1)
    local_out = torch.matmul(local_scores.softmax(-1), lv).to(q.dtype)
    sk = sink_k.repeat_interleave(4, 1).float(); sv = sink_v.repeat_interleave(4, 1).float()
    mean_k = (state_k.float() / counts.to(torch.bfloat16).float()).to(torch.bfloat16).repeat_interleave(4, 1).float()
    mean_v = (state_v.float() / counts.to(torch.bfloat16).float()).to(torch.bfloat16).repeat_interleave(4, 1).float()
    route_q = (qf * torch.rsqrt(qf.square().mean(-1, keepdim=True).clamp_min(1e-12))).to(q.dtype).float()
    native = torch.matmul(q, mean_k.to(q.dtype).transpose(-1, -2)).float() * engine.scaling + sizes.float().log()
    route = torch.matmul(route_q.to(q.dtype), mean_k.to(q.dtype).transpose(-1, -2)).float() * engine.scaling + sizes.float().log()
    p = route.softmax(-1)
    leaves_k = leaf_k.repeat_interleave(4, 1).float(); leaves_v = leaf_v.repeat_interleave(4, 1).float()
    leaf_scores = torch.matmul(qf, leaves_k.transpose(-1, -2)) * engine.scaling
    sink_scores = torch.matmul(qf, sk.transpose(-1, -2)) * engine.scaling
    results = []
    cases = [('region', 2.0, 16, 16), ('entry', 16.0, 16, 16), ('entry', 16.0, 32, 1)]
    max_regions = int(os.environ.get('LOD_H100_PREFILL_MAX_REGIONS', '0')) or None
    if args.entry_only:
        cases = [case for case in cases if case[0] == 'entry']
    for rule, multiple, bound, cap in cases:
        if rule == 'region':
            selected = p > multiple / slots
        else:
            selected = p / sizes > 1 / (bound * math.sqrt(tokens))
            cumul = (selected * sizes).cumsum(-1)
            clipped = selected & ~(cumul.square() < cap * cap * tokens)
            if cap == 1 and not bool(clipped.any()):
                raise RuntimeError('synthetic strict-cap case did not exercise clipping')
            selected &= cumul.square() < cap * cap * tokens
        if max_regions:
            if cap == 16 and not bool((selected.sum(-1) > max_regions).any()):
                raise RuntimeError('synthetic cap case did not exercise more than eight selected regions')
            selected &= selected.to(torch.int32).cumsum(-1) <= max_regions
        leaf_selected = selected[..., owner]
        frontier = torch.cat((native.masked_fill(selected, -torch.inf),
                              leaf_scores.masked_fill(~leaf_selected, -torch.inf),
                              local_scores, sink_scores), dim=-1)
        values = torch.cat((mean_v, leaves_v, lv, sv), dim=2)
        reference = torch.matmul(frontier.softmax(-1), values)
        cases_to_run = [('reference', threshold_prefill), ('fused', threshold_prefill_fused)]
        if args.compare_reuse:
            cases_to_run.append(('reuse', threshold_prefill_fused))
        baseline_out = baseline_selected = None
        for label, implementation in cases_to_run:
            if args.compare_reuse:
                if label == 'reuse':
                    os.environ['LOD_H100_PREFILL_REUSE_SCORES'] = '1'
                else:
                    os.environ.pop('LOD_H100_PREFILL_REUSE_SCORES', None)
            out = implementation(engine, q, local_k, local_v, state_k, state_v, counts,
                None, leaf_k, leaf_v, state_len=slots, state_capacity=slots, page_cache=page,
                local_branch=(local_out, local_lse), sink_k=sink_k, sink_v=sink_v,
                rule=rule, multiplier=multiple, entry_bound=bound, work_bound=cap, tile_queries=args.tile_queries,
                max_regions=max_regions)
            error = (out.float() - reference).abs()
            max_error = float(error.max())
            if max_error > 0.025 and label != 'reuse':
                raise RuntimeError(f'frontier mismatch for {label} {rule}: {max_error}')
            mismatches = 0
            if label in ('fused', 'reuse'):
                tail = (args.queries - 1) % args.tile_queries + 1
                if os.environ.get('LOD_H100_PREFILL_PACK_DIRECT') == '1' or os.environ.get('LOD_H100_PREFILL_GROUPED8') == '1':
                    buffers = engine._lod_prefill_attention_buffers
                    counts_actual = buffers['threshold_route_counts'][:32 * tail].view(32, tail)
                    leaf_capacity = int(page['leaf_k'].size(2))
                    max_entries = max(1, min(leaf_capacity, math.isqrt(cap * cap * leaf_capacity) + 1))
                    max_routes = max(1, min(slots, max_entries))
                    stored_width = max_regions or 1 << (max_routes - 1).bit_length()
                    stored_slots = buffers['threshold_slots'][:32 * tail * stored_width]
                    actual = _sample_selected(buffers['threshold_selected'], stored_slots, counts_actual,
                                              bitmask=False, direct_pack=True, h=32, qlen=tail, n=tail,
                                              state_len=slots, width=stored_width)
                elif os.environ.get('LOD_H100_PREFILL_BITMASK') == '1':
                    stored = engine._lod_prefill_attention_buffers['threshold_selected_bits']
                    byte_count = (slots + 7) // 8
                    indices = torch.arange(slots, device=device)
                    actual = ((stored[:32 * tail * byte_count].reshape(1, 32, tail, byte_count).to(torch.int32)
                               .index_select(-1, indices // 8) >> (indices % 8)) & 1).bool()
                else:
                    stored = engine._lod_prefill_attention_buffers['threshold_selected']
                    actual = stored[:32 * tail * slots].reshape(1, 32, tail, slots).bool()
                mismatches = int((actual != selected[..., -tail:, :]).sum().item())
                if mismatches and label != 'reuse':
                    if os.environ.get('LOD_H100_PREFILL_PACK_DIRECT') == '1' or os.environ.get('LOD_H100_PREFILL_GROUPED8') == '1':
                        pos = (actual != selected[..., -tail:, :]).nonzero()[0].tolist()
                        b, head, row, _ = pos
                        print({'first_mismatch': pos, 'count': int(counts_actual[head, row]),
                               'expected_slots': torch.where(selected[b, head, -tail + row])[0].tolist(),
                               'actual_slots': torch.where(actual[b, head, row])[0].tolist(),
                               'stored_slots': stored_slots.view(32, tail, stored_width)[head, row, :int(counts_actual[head, row])].tolist()}, flush=True)
                    raise RuntimeError(f'fused selection differs on {mismatches} query-region pairs')
                if label == 'fused':
                    baseline_out = out.clone()
                    baseline_selected = actual.clone()
                else:
                    changed = int((actual != baseline_selected).sum().item())
                    difference = float((out.float() - baseline_out.float()).abs().max().item())
                    results.append(dict(implementation='reuse_vs_fused', rule=rule,
                                        entry_bound=bound, work_bound=cap,
                                        changed_regions=changed, query_region_pairs=actual.numel(),
                                        max_abs=difference))
            results.append(dict(implementation=label, rule=rule, multiplier=multiple,
                                entry_bound=bound, work_bound=cap, max_abs=max_error,
                                mean_abs=float(error.mean()), selection_mismatches=mismatches))
    args.output.write_text(json.dumps({'stage': 'complete', 'records': results}, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()

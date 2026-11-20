"""Audit dense attention vs LoD frontiers on the first wrong RULER token.

Runs the original LoD model eagerly through its recorded first divergence,
while computing dense and pure PyTorch LoD attention on each layer's actual
query and cache. It reports per-head mass, retrieval and output errors.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, os.environ.get('RETROINFER_ROOT', '/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model, load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_run import save


def reconstruct(pool, length):
    from lod_attention.pytorch_engine import PytorchLODState, PytorchLODCache
    from lod_attention.kernels.paged_cache import stable_owner_ranks
    state, page = pool.state, pool.state['page_cache']
    h, t, n = pool.kv_heads, int(pool.leaf_lens[0].item()), int(pool.state_lens[0].item())
    counts = state['counts'][:, :, :n, 0]
    owners = torch.full((h, pool.page_capacity), -1, device=pool.device, dtype=torch.long)
    ordinal = torch.full_like(owners, -1)
    roots = page['slot_pages'][0, :, :n].long()
    if page.get('paged_page_directory', False):
        hh, slot, root = (roots >= 0).nonzero(as_tuple=True)
        directory = page['overflow_page_values'][0, hh, roots[hh, slot, root]].long()
        rr, entry = (directory >= 0).nonzero(as_tuple=True)
        owners[hh[rr], directory[rr, entry]] = slot[rr]
        ordinal[hh[rr], directory[rr, entry]] = root[rr] * 64 + entry
    else:
        hh, slot, ords = (roots >= 0).nonzero(as_tuple=True)
        owners[hh, roots[hh, slot, ords]] = slot
        ordinal[hh, roots[hh, slot, ords]] = ords
        keys, vals = page['overflow_page_keys'][0].long(), page['overflow_page_values'][0].long()
        hh, idx = ((keys >= 0) & (vals >= 0)).nonzero(as_tuple=True)
        owners[hh, vals[hh, idx]] = keys[hh, idx] // 65536
        ordinal[hh, vals[hh, idx]] = keys[hh, idx] % 65536
    indices = page['page_indices'][0].long()
    valid = (indices >= 0) & (owners[:, :, None] >= 0)
    hh, pp, j = valid.nonzero(as_tuple=True)
    leaf = indices[hh, pp, j]
    if torch.any((leaf < 0) | (leaf >= t)) or leaf.numel() != h*t:
        raise RuntimeError('page directory does not cover the archive')
    leaf_owner = torch.full((h,t), -1, device=pool.device, dtype=torch.long)
    leaf_page = torch.full_like(leaf_owner, -1)
    leaf_owner[hh,leaf] = owners[hh,pp]
    leaf_page[hh,leaf] = pp
    if torch.any(leaf_owner < 0):
        raise RuntimeError('missing or duplicate leaf')
    actual = torch.zeros_like(counts[0]).scatter_add_(1, leaf_owner, torch.ones_like(leaf_owner,dtype=counts.dtype))
    if not torch.equal(actual, counts[0]):
        raise RuntimeError('page and centroid counts disagree')
    ranks = stable_owner_ranks(leaf_owner[None])[0]
    if not torch.equal(ranks[hh,leaf]//16, ordinal[hh,pp]) or not torch.equal(ranks[hh,leaf]%16, j):
        raise RuntimeError('page order differs from reference')
    cache = PytorchLODCache(PytorchLODState(state['state_k'][:,:,:n], state['state_v'][:,:,:n],
        counts, torch.zeros_like(counts)), leaf_owner[None], page['leaf_k'][:,:,:t],
        page['leaf_v'][:,:,:t], torch.ones(1,t,dtype=torch.bool,device=pool.device), t,
        state['sink_k'],state['sink_v'],length)
    return cache, owners, ordinal, leaf_page


def audit(pool, q, optimized, gold_positions, context):
    from lod_attention.pytorch_engine import PytorchLODAttention
    ref, page_owner, page_ordinal, leaf_page = reconstruct(pool, context)
    engine = PytorchLODAttention(mode='three-tier-bf16')
    scale = 1.0 / math.sqrt(pool.head_dim)
    Q = q.transpose(1,2)
    H, G, D = pool.kv_heads, pool.query_heads // pool.kv_heads, pool.head_dim
    query = Q[0,:,0].reshape(H,G,D).float()
    nlocal = int(pool.local_lens[0].item())
    local_k,local_v = pool.state['recent_k'][0,:,:nlocal],pool.state['recent_v'][0,:,:nlocal]
    sink_k,sink_v = ref.sink_key[0], ref.sink_value[0]
    leaf_k,leaf_v = ref.archive_key[0],ref.archive_value[0]
    leaf_logits = torch.einsum('hgd,htd->hgt',query,leaf_k.float())*scale
    local_logits = torch.einsum('hgd,htd->hgt',query,local_k.float())*scale
    sink_logits = torch.einsum('hgd,htd->hgt',query,sink_k.float())*scale
    logits = torch.cat((leaf_logits,local_logits,sink_logits),-1)
    probs = torch.softmax(logits,dim=-1)
    nl = leaf_k.shape[1]
    leaf_p,local_p,sink_p = probs[...,:nl],probs[...,nl:nl+nlocal],probs[...,nl+nlocal:]
    dense = (torch.einsum('hgt,htd->hgd',leaf_p,leaf_v.float()) +
             torch.einsum('hgt,htd->hgd',local_p,local_v.float()) +
             torch.einsum('hgt,htd->hgd',sink_p,sink_v.float())).reshape(H*G,D)
    _,_,route = engine._routes(Q,ref.state,scale=scale)
    route = route[0,:,0].reshape(H,G,-1)
    owner = ref.owner[0]
    remote_selected = (owner[:,None,:,None] == route[:,:,None,:]).any(-1) & (owner[:,None,:] >= 0)
    page_mask = None
    if pool.settings.levels == 3:
        page = pool.state['page_cache']
        ids = page['page_indices'][0].long()
        valid = ids >= 0
        hh = torch.arange(H,device=pool.device)[:,None,None]
        keys = leaf_k[hh,ids.clamp_min(0)].float()
        keys = (keys * valid[...,None]).sum(2)
        page_counts = valid.sum(-1)
        scores = torch.einsum('hgd,hpd->hgp', query, keys) * scale / page_counts.clamp_min(1)[:,None,:]
        scores = scores + page_counts.clamp_min(1).float().log()[:,None,:]
        scores = scores[:, :, None, :].expand(-1,-1,route.size(-1),-1)
        eligible = (page_owner[:,None,None,:] == route[:,:,:,None]) & (route[:,:,:,None]>=0) & (page_counts[:,None,None,:]>0)
        scored = scores.masked_fill(~eligible, -torch.inf)
        # Sort by region-local page order first for stable ties, matching the reference.
        order = torch.argsort(page_ordinal.masked_fill(page_ordinal < 0, 2**30),dim=-1,stable=True)
        order = order[:,None,None,:].expand_as(scored)
        ordered = scored.gather(-1,order)
        chosen_order = torch.argsort(ordered,dim=-1,descending=True,stable=True)[...,:2]
        chosen = order.gather(-1,chosen_order)
        chosen_valid = ordered.gather(-1,chosen_order).isfinite()
        page_mask = ((leaf_page[:,None,:,None,None] == chosen[:,:,None,:,:]) & chosen_valid[:,:,None,:,:]).any((-1,-2))
        if torch.any(page_mask & ~remote_selected):
            raise RuntimeError('page outside selected region')
    local_ref = pool.state['recent_k'][:,:,:nlocal]
    local_val_ref = pool.state['recent_v'][:,:,:nlocal]
    two = engine._lod_attention(Q,local_ref,local_val_ref,ref,scale=scale,
                query_offset=nlocal-1,full_regions=True).output[0,:,0].float()
    if pool.settings.levels == 3:
        three = engine._lod_attention(Q,local_ref,local_val_ref,ref,scale=scale,
                  query_offset=nlocal-1,full_regions=False).output[0,:,0].float()
    else:
        three = None
    opt = optimized[0,0].float()
    denom = dense.norm(dim=-1).clamp_min(1e-12)
    err_opt = (opt-dense).norm(dim=-1)/denom
    err_two = (two-dense).norm(dim=-1)/denom
    ref_diff = (opt-(three if three is not None else two)).abs()
    err_three = (three-dense).norm(dim=-1)/denom if three is not None else None
    exact_region_mass = (leaf_p * remote_selected).sum(-1)
    page_mass = (leaf_p*page_mask).sum(-1) if page_mask is not None else None
    # Four mutually exclusive dense-mass categories, summed per head.
    remote_mass = leaf_p.sum(-1)
    local_mass = local_p.sum(-1)
    sink_mass = sink_p.sum(-1)
    topids = probs.topk(10,dim=-1).indices
    included_region = torch.cat((remote_selected,torch.ones_like(local_p,dtype=torch.bool),
                                  torch.ones_like(sink_p,dtype=torch.bool)),dim=-1)
    top_region = included_region.gather(-1,topids).float().mean(-1)
    top_page = None
    if page_mask is not None:
        included_page = torch.cat((page_mask,torch.ones_like(local_p,dtype=torch.bool),
                                    torch.ones_like(sink_p,dtype=torch.bool)),dim=-1)
        top_page = included_page.gather(-1,topids).float().mean(-1)
    gold_leaf = [p - sink_k.size(1) for p in gold_positions if sink_k.size(1) <= p < sink_k.size(1)+nl]
    gold_local = [p - sink_k.size(1)-nl for p in gold_positions if sink_k.size(1)+nl <= p < sink_k.size(1)+nl+nlocal]
    gold_sink = [p for p in gold_positions if p < sink_k.size(1)]
    goldmass = torch.zeros((H,G),device=pool.device)
    gold_region_mass = torch.zeros_like(goldmass)
    gold_page_mass = torch.zeros_like(goldmass)
    if gold_leaf:
        goldmass += leaf_p[...,gold_leaf].sum(-1)
        gold_region_mass += (leaf_p[...,gold_leaf]*remote_selected[...,gold_leaf]).sum(-1)
        if page_mask is not None:
            gold_page_mass += (leaf_p[...,gold_leaf]*page_mask[...,gold_leaf]).sum(-1)
    if gold_local:
        mass = local_p[...,gold_local].sum(-1); goldmass+=mass; gold_region_mass+=mass; gold_page_mass+=mass
    if gold_sink:
        mass = sink_p[...,gold_sink].sum(-1); goldmass+=mass; gold_region_mass+=mass; gold_page_mass+=mass
    if not gold_leaf or gold_local or gold_sink:
        raise RuntimeError('gold token detail requires all answer tokens in the remote archive')
    # For each head, locate the individual answer token with the highest
    # dense attention, then inspect its region's routing rank and mass.
    gold_idx = torch.tensor(gold_leaf, device=pool.device, dtype=torch.long)
    gold_pos = gold_idx[leaf_p[...,gold_idx].argmax(-1)]
    best_gold_mass = leaf_p.gather(-1,gold_pos[...,None]).squeeze(-1)
    gold_slot = owner[:,None,:].expand(-1,G,-1).gather(-1,gold_pos[...,None]).squeeze(-1)
    count = ref.state.count[0].float()
    mean_key = ref.state.mean_key[0].float()
    native = torch.einsum('hgd,hsd->hgs',query,mean_key)*scale + count.clamp_min(1).log()[:,None,:]
    query_normalized = query * torch.rsqrt(query.square().mean(-1,keepdim=True).clamp_min(1e-12))
    normalized = torch.einsum('hgd,hsd->hgs',query_normalized,mean_key)*scale + count.clamp_min(1).log()[:,None,:]
    native = native.masked_fill(count[:,None,:] <= 0,-torch.inf)
    normalized = normalized.masked_fill(count[:,None,:] <= 0,-torch.inf)
    rank_native = (native > native.gather(-1,gold_slot[...,None])).sum(-1)+1
    rank_normalized = (normalized > normalized.gather(-1,gold_slot[...,None])).sum(-1)+1
    true_region_mass = torch.zeros(H,G,ref.state.size,device=pool.device).scatter_add_(
        -1,owner[:,None,:].expand(-1,G,-1),leaf_p)
    gold_region_true_mass = true_region_mass.gather(-1,gold_slot[...,None]).squeeze(-1)
    log_z = torch.logsumexp(logits,-1)
    gold_region_coarse_mass = (native.gather(-1,gold_slot[...,None]).squeeze(-1)-log_z).exp()
    gold_region_count = count[:,None,:].expand(-1,G,-1).gather(-1,gold_slot[...,None]).squeeze(-1)
    # Oracle diagnostic: open the region containing the highest-mass answer
    # token, replacing the eighth route only if that region was not selected.
    # This uses answer identities and dense scores, and is not a deployable policy.
    forced_routes = route.clone()
    already = (route == gold_slot[...,None]).any(-1)
    can_open = gold_region_count <= engine.config.max_region_size
    forced_routes[:,:,-1] = torch.where(already | ~can_open, forced_routes[:,:,-1], gold_slot)
    base_routes = engine._routes
    def oracle_routes(query_arg,state_arg,*,scale):
        coarse, mean_value, _ = base_routes(query_arg,state_arg,scale=scale)
        return coarse, mean_value, forced_routes.reshape(1,H*G,1,-1)
    engine._routes = oracle_routes
    forced = engine._lod_attention(Q,local_ref,local_val_ref,ref,scale=scale,
        query_offset=nlocal-1,full_regions=True).output[0,:,0].float()
    forced_error = (forced-dense).norm(dim=-1)/denom
    def items(x): return x.reshape(-1).tolist()
    out = {'remote_mass':items(remote_mass), 'local_mass':items(local_mass), 'sink_mass':items(sink_mass),
           'selected_region_mass':items(exact_region_mass), 'top10_region_fraction':items(top_region),
           'gold_mass':items(goldmass), 'gold_region_mass':items(gold_region_mass),
           'top_gold_token_mass':items(best_gold_mass), 'top_gold_token_position':items(gold_pos+int(sink_k.size(1))),
           'top_gold_region_count':items(gold_region_count), 'top_gold_region_rank_normalized':items(rank_normalized),
           'top_gold_region_rank_native':items(rank_native), 'top_gold_region_true_mass':items(gold_region_true_mass),
           'top_gold_region_coarse_mass':items(gold_region_coarse_mass),
           'optimized_relative_l2':items(err_opt), 'two_tier_relative_l2':items(err_two),
           'oracle_open_gold_region_relative_l2':items(forced_error),
           'reference_vs_optimized_max_abs':float(ref_diff.max()),
           'reference_vs_optimized_mean_abs':float(ref_diff.mean()),
           'num_centroids':ref.state.size, 'remote_tokens':nl,'local_tokens':nlocal,
           'sink_tokens':int(sink_k.size(1))}
    if three is not None:
        out.update(selected_page_mass=items(page_mass), top10_page_fraction=items(top_page),
                   gold_page_mass=items(gold_page_mass), three_tier_relative_l2=items(err_three))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cohort','dense','prior','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=('two-tier','three-tier-bf16'),required=True)
    p.add_argument('--rows',type=int,nargs='+',default=[78,126,302,447])
    a=p.parse_args()
    C=json.loads(a.cohort.read_text()); D=json.loads(a.dense.read_text()); P=json.loads(a.prior.read_text())
    if P['mode'] != a.mode: raise ValueError('prior mode mismatch')
    dense={(r['task'],r['source_row']):r for r in D['records']}
    prior={(r['task'],r['source_row']):r for r in P['records']}
    samples=[next(s for s in C['samples'] if s['task']=='niah_multikey_3' and s['source_row']==row) for row in a.rows]
    torch.set_grad_enabled(False)
    name='meta-llama/Llama-3.1-8B-Instruct'
    tok=load_tokenizer(name); model=load_model(name,65536,torch.bfloat16,'cuda:0',tok);attach_lod(model,a.mode)
    result={'stage':'running','mode':a.mode,'model':name,'rows':a.rows,'records':[]}
    save(a.output,result)
    for sample in samples:
        start=time.monotonic(); task,row=sample['task'],sample['source_row']
        orig,base=prior[(task,row)],dense[(task,row)]
        if orig['score'] != 0 or base['score'] != 1: raise ValueError('expected LoD failure and dense success')
        divergence=next((i for i,(x,y) in enumerate(zip(orig['output_token_ids'],base['output_token_ids'])) if x!=y),None)
        if divergence is None or divergence==0: raise ValueError('no eligible decode divergence')
        prompt=sample['prompt']; encoded=tok(prompt,return_tensors='pt',return_offsets_mapping=True)
        offsets=encoded.pop('offset_mapping')[0].tolist(); inputs=encoded['input_ids'].to('cuda'); n=inputs.shape[1]
        digest=hashlib.sha256(struct.pack(f'<{n}I',*inputs[0].tolist())).hexdigest()
        if digest != orig['input_token_sha256'] or digest != base['input_token_sha256']: raise RuntimeError('input differs')
        answer=sample['answers'][0]; spans=[]; pos=0
        while (pos:=prompt.find(answer,pos))>=0:
            spans.append((pos,pos+len(answer)));pos+=len(answer)
        gold=[i for i,(lo,hi) in enumerate(offsets) if hi>lo and any(lo<end and hi>begin for begin,end in spans)]
        if not gold: raise RuntimeError('answer UUID not found in prompt token offsets')
        model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=n
        model.max_new_length=128;model.prefill_method='full'
        model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,n,'Full_Flash_Attn'))
        token=model.sampling(model.prefill_forward(inputs_ids=inputs));model.move()
        if token.item()!=orig['output_token_ids'][0]: raise RuntimeError('prefill token mismatch')
        cache=model.kv_cache; orig_decode=cache.decode_attention; layer_audit=[]; enabled=False
        def decode(this,q,k,v,layer_idx):
            out=orig_decode(q,k,v,layer_idx)
            if enabled:
                x=audit(this.pools[layer_idx],q,out,gold,this.context)
                x['layer']=layer_idx;layer_audit.append(x)
                print(json.dumps({'row':row,'mode':a.mode,'layer':layer_idx,'max_error':x['reference_vs_optimized_max_abs']}),flush=True)
            return out
        cache.decode_attention=types.MethodType(decode,cache)
        for i in range(1,divergence+1):
            enabled=(i==divergence)
            token=model.sampling(model.decode_forward(inputs_ids=token))
            if token.item()!=orig['output_token_ids'][i]:
                raise RuntimeError(f'eager does not reproduce prior graph token at {row}/{i}: {token.item()} vs {orig["output_token_ids"][i]}')
        if len(layer_audit)!=32: raise RuntimeError('did not collect all layers')
        result['records'].append({'row':row,'input_tokens':n,'divergence_token_index':divergence,
            'dense_next_token':base['output_token_ids'][divergence], 'lod_next_token':int(token.item()),
            'gold_token_positions':gold,'gold_occurrences':len(spans), 'layers':layer_audit,
            'elapsed_s':round(time.monotonic()-start,2)})
        save(a.output,result); del model.kv_cache;torch.cuda.empty_cache()
        print(f'completed row {row} ({a.mode})',flush=True)
    result['stage']='complete';save(a.output,result)

if __name__=='__main__': main()

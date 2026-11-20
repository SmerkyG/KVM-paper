"""Trace a three-tier failure on identical queries, regions and BF16 KV states.

This deliberately slow diagnostic measures dense mass lost at page refinement,
checks the actual kernel against a PyTorch frontier, and evaluates alternative
page scores/budgets without changing the replayed model output. Oracle page
scores are marked explicitly; they read every leaf and are not speed candidates.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import types

import numpy as np
import torch
sys.path.insert(0, os.environ.get('RETROINFER_ROOT', '/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model, load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_run import save
from benchmarks.retro_ruler_three_tier_mean_regions import build_page_reference


def pick_pages(score, slots, eligible, regions, budget):
    """Stable physical-ID tie break, matching a one-block serving page scan."""
    H, G, P = score.shape
    ids = torch.arange(P, device=score.device)[None, None].expand(H, G, -1)
    owner = slots[:, None].expand(-1, G, -1)
    remaining = score.masked_fill(~eligible, -torch.inf)
    selected = torch.zeros_like(eligible)
    for _ in range(budget):
        best = torch.full((H, G, regions), -torch.inf, device=score.device).scatter_reduce_(
            -1, owner, remaining, reduce='amax', include_self=True)
        tied = (remaining == best.gather(-1, owner)) & remaining.isfinite()
        first = torch.full_like(best, -1, dtype=torch.long).scatter_reduce_(
            -1, owner, ids.masked_fill(~tied, -1), reduce='amax', include_self=True)
        chosen = tied & (ids == first.gather(-1, owner))
        selected |= chosen
        remaining.masked_fill_(chosen, -torch.inf)
    return selected


def inspect(pool, q, optimized, gold, context, layer):
    cache, slots, ordinals, leaf_page, pk, pv, pc, region_k, region_v = build_page_reference(pool, context)
    H, G, D = pool.kv_heads, pool.query_heads // pool.kv_heads, pool.head_dim
    query = q[0, 0].reshape(H, G, D).float()
    scale = pool.engine.scaling
    n = cache.state.size
    page = pool.state['page_cache']
    count = cache.state.count[0].float()
    # Match the existing H100 BF16 count division exactly.
    mk = (cache.state.key_sum[0].float()/count.to(torch.bfloat16).float().clamp_min(1)[..., None]).to(torch.bfloat16).float()
    mv = (cache.state.value_sum[0].float()/count.to(torch.bfloat16).float().clamp_min(1)[..., None]).to(torch.bfloat16).float()
    native = (torch.einsum('hgd,hsd->hgs', query, mk)*scale + count.clamp_min(1).log()[:, None]).masked_fill(count[:, None] <= 0, -torch.inf)
    opened = pool._threshold_scratch['allowed'].view(H, G, -1)[..., :n].bool()
    owner = slots[:, None].expand(-1, G, -1)
    eligible = (pc[:, None] > 0) & opened.gather(-1, owner)
    stored_k, stored_v = page['page_sum_k'][0].float(), page['page_sum_v'][0].float()
    page_score = torch.einsum('hgd,hpd->hgp', query, stored_k)*scale/pc.clamp_min(1)[:, None] + pc.clamp_min(1).log()[:, None]
    leaf_k, leaf_v = cache.archive_key[0].float(), cache.archive_value[0].float()
    leaf_logits = torch.einsum('hgd,htd->hgt', query, leaf_k)*scale
    nl = int(pool.local_lens[0].item())
    local_k = torch.cat((pool.state['recent_k'][0, :, :nl], pool.state['sink_k'][0]), 1).float()
    local_v = torch.cat((pool.state['recent_v'][0, :, :nl], pool.state['sink_v'][0]), 1).float()
    local_logits = torch.einsum('hgd,htd->hgt', query, local_k)*scale
    dense_logits = torch.cat((leaf_logits, local_logits), -1)
    dense_p = dense_logits.softmax(-1)
    t = leaf_k.shape[1]
    dense = torch.einsum('hgt,htd->hgd', dense_p[..., :t], leaf_v) + torch.einsum('hgt,htd->hgd', dense_p[..., t:], local_v)
    gold_idx = torch.tensor([i-pool.state['sink_k'].shape[2] for i in gold], device=q.device)
    if not bool(((gold_idx >= 0) & (gold_idx < t)).all()):
        raise RuntimeError('expected remote answer tokens')
    leaf_owner = cache.owner[0][:, None].expand(-1, G, -1)
    region_mask = opened.gather(-1, leaf_owner)
    leafpage = leaf_page[:, None].expand(-1, G, -1)
    # Practical bound score uses per-channel extrema; exact scores are oracle diagnostics.
    idx = page['page_indices'][0].long()
    good = (idx >= 0) & (pc[:, :, None] > 0)
    hh = torch.arange(H, device=q.device)[:, None, None]
    keys = leaf_k[hh, idx.clamp_min(0)]
    kmin = keys.masked_fill(~good[..., None], torch.inf).amin(2)
    kmax = keys.masked_fill(~good[..., None], -torch.inf).amax(2)
    bound = (torch.einsum('hgd,hpd->hgp', query.clamp_min(0), kmax.nan_to_num()) + torch.einsum('hgd,hpd->hgp', query.clamp_max(0), kmin.nan_to_num())) * scale
    page_max = torch.full_like(page_score, -torch.inf).scatter_reduce_(-1, leafpage, leaf_logits, reduce='amax', include_self=True)
    page_exp = torch.zeros_like(page_score).scatter_add_(-1, leafpage, (leaf_logits-page_max.gather(-1,leafpage)).exp())
    exact_mass = page_max + page_exp.clamp_min(1e-30).log()
    mean_keys = pk / pc.clamp_min(1)[..., None]
    outlier_idx = (keys-mean_keys[:,:,None]).square().sum(-1).masked_fill(~good,-torch.inf).argmax(-1)
    norm_idx = keys.square().sum(-1).masked_fill(~good,-torch.inf).argmax(-1)
    def representative(indices):
        return keys.gather(2,indices[:,:,None,None].expand(-1,-1,1,D)).squeeze(2)
    outlier_key = representative(outlier_idx)
    opposite_idx = (keys-outlier_key[:,:,None]).square().sum(-1).masked_fill(~good,-torch.inf).argmax(-1)
    outlier_score = torch.einsum('hgd,hpd->hgp',query,outlier_key)*scale
    norm_score = torch.einsum('hgd,hpd->hgp',query,representative(norm_idx))*scale
    pair_score = torch.maximum(outlier_score,torch.einsum('hgd,hpd->hgp',query,representative(opposite_idx))*scale)
    # A diagonal covariance correction is another summary-only diagnostic.
    variance = ((keys-mean_keys[:,:,None]).square()*good[...,None]).sum(2)/pc.clamp_min(1)[...,None]
    variance_score = page_score + .5*torch.einsum('hgd,hpd->hgp',query.square(),variance)*scale**2
    pgscore=page_score.masked_fill(pc[:,None]<=0,-torch.inf)
    pgmax=torch.full_like(native,-torch.inf).scatter_reduce_(-1,owner,pgscore,reduce='amax',include_self=True)
    pgweight=(pgscore-pgmax.gather(-1,owner)).exp().nan_to_num(0.)
    pgz=torch.zeros_like(native).scatter_add_(-1,owner,pgweight)
    pgprob=pgweight/pgz.gather(-1,owner).clamp_min(1.e-30)
    npages=((count+15)//16)[:,None].expand(-1,G,-1).gather(-1,owner)
    pagevariants=[]
    for rule in ('page','entry'):
        for c in (.25,.5,1.):
            cutoff=c*pc[:,None]/count[:,None].expand(-1,G,-1).gather(-1,owner).clamp_min(1)*(1.+1.e-6) if rule=='entry' else c/npages.clamp_min(1)
            mask=eligible&((pgprob>cutoff)|(npages==1))
            pagevariants.append((f'{rule}-{c:g}',mask,0,False))

    def frontier(selected, full=False, exact_sums=False):
        if full:
            mask = region_mask
            summary = native.masked_fill(opened, -torch.inf)
            sums_k, sums_v = cache.state.key_sum[0].float(), cache.state.value_sum[0].float()
            removed_v = torch.zeros((H,G,n,D), device=q.device)
            left = count[:,None].expand(-1,G,-1)
        else:
            mask = selected.gather(-1, leafpage)
            nc = torch.zeros_like(native).scatter_add_(-1, owner, selected*pc[:,None])
            left = count[:,None]-nc
            pkeys, pvalues = (pk,pv) if exact_sums else (stored_k,stored_v)
            sums_k,sums_v = (region_k,region_v) if exact_sums else (cache.state.key_sum[0].float(),cache.state.value_sum[0].float())
            removed_dot = torch.zeros_like(native).scatter_add_(-1,owner,selected*torch.einsum('hgd,hpd->hgp',query,pkeys)*scale)
            residual_logits = (torch.einsum('hgd,hsd->hgs',query,sums_k)*scale-removed_dot)/left.clamp_min(1)+left.clamp_min(1).log()
            summary = torch.where(opened,residual_logits,native).masked_fill(opened & (left <= 0),-torch.inf)
            # Aggregate removed values weighted after computing summary probabilities below.
        logits = torch.cat((summary, leaf_logits.masked_fill(~mask,-torch.inf),local_logits),-1)
        prob = logits.softmax(-1)
        sp = prob[...,:n]
        if full:
            output = torch.einsum('hgs,hsd->hgd',sp,mv)
        else:
            rw = sp*opened/left.clamp_min(1)
            output = (torch.einsum('hgs,hsd->hgd',sp*(~opened),mv) + torch.einsum('hgs,hsd->hgd',rw,sums_v) - torch.einsum('hgp,hpd->hgd',rw.gather(-1,owner)*selected,pvalues))
        output += torch.einsum('hgt,htd->hgd',prob[...,n:n+t],leaf_v) + torch.einsum('hgt,htd->hgd',prob[...,n+t:],local_v)
        return output,mask

    result = {'layer':layer,'opened_regions':opened.sum(-1).flatten().tolist(),
              'opened_region_entries':region_mask.sum(-1).flatten().tolist(),
              'gold_dense_mass':dense_p[...,:t][...,gold_idx].sum(-1).flatten().tolist(),
              'gold_region_mass':(dense_p[...,:t]*region_mask)[...,gold_idx].sum(-1).flatten().tolist(),
              'variants':{},'actual_policy':('top2' if os.environ.get('LOD_H100_PAGE_THRESHOLD') is None else
                  f"{os.environ.get('LOD_H100_PAGE_THRESHOLD_RULE','page')}-{float(os.environ['LOD_H100_PAGE_THRESHOLD']):g}")}
    best_gold = gold_idx[dense_p[...,:t][...,gold_idx].argmax(-1)]
    best_page = leafpage.gather(-1,best_gold[...,None]).squeeze(-1)
    best_region = leaf_owner.gather(-1,best_gold[...,None]).squeeze(-1)
    def page_rank(score):
        value = score.gather(-1,best_page[...,None])
        return (((score > value) & (owner == best_region[...,None]) & (pc[:,None] > 0)).sum(-1)+1).flatten().tolist()
    result['best_answer_token'] = dict(
        position=(best_gold+pool.state['sink_k'].shape[2]).flatten().tolist(),
        region_count=count[:,None].expand(-1,G,-1).gather(-1,best_region[...,None]).flatten().tolist(),
        region_opened=opened.gather(-1,best_region[...,None]).flatten().tolist(),
        mean_page_rank=page_rank(page_score), oracle_page_rank=page_rank(exact_mass),
        representatives_page_rank=page_rank(pair_score), variance_page_rank=page_rank(variance_score))
    goldpc=pc[:,None].expand(-1,G,-1).gather(-1,best_page[...,None]).squeeze(-1)
    goldpr=pgprob.gather(-1,best_page[...,None]).squeeze(-1)
    goldnr=count[:,None].expand(-1,G,-1).gather(-1,best_region[...,None]).squeeze(-1)
    result['best_answer_token'].update(page_entries=goldpc.flatten().tolist(),
        page_to_average_mass=(goldpr*((goldnr+15)//16)).flatten().tolist(),
        entry_to_average_mass=(goldpr*goldnr/goldpc.clamp_min(1)).flatten().tolist())
    denom = dense.norm(dim=-1).clamp_min(1e-12)
    for name, scorer, budget, exact_sums in [('top2',page_score,2,False),('top2-exact-sums',page_score,2,True),('top4',page_score,4,False),('top8',page_score,8,False),('bound2',bound,2,False),('outlier2',outlier_score,2,False),('norm2',norm_score,2,False),('representatives2',pair_score,2,False),('variance2',variance_score,2,False),('oracle-mass2',exact_mass,2,False),('oracle-max2',page_max,2,False),('two-tier',page_score,2,False),*pagevariants]:
        selected = scorer if budget==0 else pick_pages(scorer,slots,eligible,n,budget)
        value,mask = frontier(selected,full=name=='two-tier',exact_sums=exact_sums)
        v = {'relative_l2':((value-dense).norm(dim=-1)/denom).flatten().tolist(),
             'exact_entries':mask.sum(-1).flatten().tolist(),
             'exact_dense_mass':(dense_p[...,:t]*mask).sum(-1).flatten().tolist(),
             'exact_gold_mass':(dense_p[...,:t]*mask)[...,gold_idx].sum(-1).flatten().tolist()}
        if name==result['actual_policy']:
            torch.testing.assert_close(value,optimized.reshape(H,G,D).float(),rtol=.008,atol=.008)
            delta=(value-optimized.reshape(H,G,D).float()).abs()
            v.update(kernel_max_abs=float(delta.max()),kernel_mean_abs=float(delta.mean()))
        result['variants'][name]=v
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','prior','dense','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rows',nargs='+',type=int,default=[59])
    a=p.parse_args(); torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    cohort=json.loads(a.cohort.read_text()); prior=json.loads(a.prior.read_text()); dense=json.loads(a.dense.read_text())
    tok=load_tokenizer('meta-llama/Llama-3.1-8B-Instruct')
    model=load_model('meta-llama/Llama-3.1-8B-Instruct',65536,torch.bfloat16,'cuda:0',tok);attach_lod(model,'three-tier-bf16')
    result={'stage':'running','scope':'same-query BF16 page-frontier diagnostic; no latency claims','records':[]};save(a.output,result)
    for row in a.rows:
        started=time.monotonic();sample=next(s for s in cohort['samples'] if s['task']=='niah_multikey_3' and s['source_row']==row)
        old=next(s for s in prior['records'] if s['task']==sample['task'] and s['source_row']==row)
        base=next(s for s in dense['records'] if s['task']==sample['task'] and s['source_row']==row)
        divergence=next(i for i,(x,y) in enumerate(zip(old['output_token_ids'],base['output_token_ids'])) if x!=y)
        enc=tok(sample['prompt'],return_tensors='pt',return_offsets_mapping=True);offsets=enc.pop('offset_mapping')[0].tolist();inputs=enc['input_ids'].to('cuda');n=inputs.shape[1]
        digest=hashlib.sha256(struct.pack(f'<{n}I',*inputs[0].tolist())).hexdigest();assert digest==old['input_token_sha256']==base['input_token_sha256']
        spans=[];pos=0;answer=sample['answers'][0]
        while (pos:=sample['prompt'].find(answer,pos))>=0:spans.append((pos,pos+len(answer)));pos+=len(answer)
        gold=[i for i,(lo,hi) in enumerate(offsets) if hi>lo and any(lo<end and hi>begin for begin,end in spans)]
        assert gold and divergence>0
        model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=n;model.max_new_length=128;model.prefill_method='full'
        model.init_kv_cache(np.zeros(1,dtype=int),generate_config('meta-llama/Llama-3.1-8B-Instruct',n,'Full_Flash_Attn'))
        token=model.sampling(model.prefill_forward(inputs_ids=inputs));model.move();assert int(token.item())==old['output_token_ids'][0]
        cache=model.kv_cache;original=cache.decode_attention;layers=[];enabled=False
        def decode(this,q,k,v,layer_idx):
            out=original(q,k,v,layer_idx)
            if enabled:
                layers.append(inspect(this.pools[layer_idx],q,out,gold,this.context,layer_idx))
                print(json.dumps({'row':row,'layer':layer_idx,'kernel_max_abs':layers[-1]['variants'][layers[-1]['actual_policy']]['kernel_max_abs']}),flush=True)
            return out
        cache.decode_attention=types.MethodType(decode,cache)
        for step in range(1,divergence+1):
            enabled=step==divergence;token=model.sampling(model.decode_forward(inputs_ids=token))
            assert int(token.item())==old['output_token_ids'][step],(step,int(token.item()),old['output_token_ids'][step])
        result['records'].append({'source_row':row,'divergence':divergence,'gold_positions':gold,'layers':layers,'elapsed_s':time.monotonic()-started});save(a.output,result)
        del model.kv_cache;gc.collect();torch.cuda.empty_cache()
    result['stage']='complete';save(a.output,result)


if __name__=='__main__':main()

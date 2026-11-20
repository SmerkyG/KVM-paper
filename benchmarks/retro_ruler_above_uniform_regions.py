"""RULER two-tier diagnostic opening regions above uniform centroid mass.

The remote region distribution is computed from native-query centroid logits
and counts. Exactly opened regions retain the existing 1,024-entry cap; every
unopened region remains a centroid summary. Decode is an eager FP32 PyTorch
reference path, not an optimized latency implementation.
"""
import argparse
import gc
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
sys.path.insert(0,os.environ.get('RETROINFER_ROOT','/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model,load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_run import save,score
from benchmarks.retro_ruler_frontier_audit import reconstruct


def adaptive_decode(pool,q,cache,threshold,check_heads,routing_score):
    H,G,D=pool.kv_heads,pool.query_heads//pool.kv_heads,pool.head_dim
    query=q[0,0].reshape(H,G,D).float();scale=1/math.sqrt(D)
    state=cache.state;counts=state.count[0].float();valid=counts>0
    keys=state.mean_key[0].float();values=state.mean_value[0].float()
    coarse=torch.einsum('hgd,hsd->hgs',query,keys)*scale+counts.clamp_min(1).log()[:,None,:]
    coarse=coarse.masked_fill(~valid[:,None,:],-torch.inf)
    if routing_score=='normalized-query':
        normalized=query*torch.rsqrt(query.square().mean(-1,keepdim=True).clamp_min(1e-12))
        route_logits=torch.einsum('hgd,hsd->hgs',normalized,keys)*scale+counts.clamp_min(1).log()[:,None,:]
        route_logits=route_logits.masked_fill(~valid[:,None,:],-torch.inf)
    else:
        route_logits=coarse
    sorted_logits,indices=route_logits.sort(dim=-1,descending=True,stable=True)
    prob=torch.softmax(sorted_logits,dim=-1)
    prefix=prob.cumsum(-1)
    # Include the crossing region and exclude zero-probability padding.
    valid_count=valid.sum(-1)[:,None,None]
    candidates=(prob > valid_count.reciprocal()) & torch.isfinite(sorted_logits)
    size=counts[:,None,:].expand(-1,G,-1).gather(-1,indices)
    eligible=size<=1024
    selected=candidates & eligible
    opened=torch.zeros_like(coarse,dtype=torch.bool).scatter_(-1,indices,selected)
    remote_mass=(prob*selected).sum(-1)
    nregions=selected.sum(-1); candidate_count=candidates.sum(-1)
    leaf_count=(size*selected).sum(-1)
    coarse_logits=coarse.masked_fill(opened,-torch.inf)
    owner=cache.owner[0];leaf_mask=opened.gather(-1,owner[:,None,:].expand(-1,G,-1))
    leaf_k=cache.archive_key[0].float();leaf_v=cache.archive_value[0].float()
    leaf_logits=torch.einsum('hgd,htd->hgt',query,leaf_k)*scale
    selected_leaf_logits=leaf_logits.masked_fill(~leaf_mask,-torch.inf)
    nlocal=int(pool.local_lens[0].item())
    local_k=pool.state['recent_k'][0,:,:nlocal].float();local_v=pool.state['recent_v'][0,:,:nlocal].float()
    sink_k=cache.sink_key[0].float();sink_v=cache.sink_value[0].float()
    local_logits=torch.einsum('hgd,htd->hgt',query,local_k)*scale
    sink_logits=torch.einsum('hgd,htd->hgt',query,sink_k)*scale
    all_logits=torch.cat((coarse_logits,selected_leaf_logits,local_logits,sink_logits),-1)
    p=torch.softmax(all_logits,dim=-1);n=state.size;t=leaf_k.size(1)
    output=(torch.einsum('hgs,hsd->hgd',p[...,:n],values)+
            torch.einsum('hgt,htd->hgd',p[...,n:n+t],leaf_v)+
            torch.einsum('hgt,htd->hgd',p[...,n+t:n+t+nlocal],local_v)+
            torch.einsum('hgt,htd->hgd',p[...,n+t+nlocal:],sink_v))
    checks=[]
    for h in check_heads:
        kh,gh=divmod(h,G)
        scores=route_logits[kh,gh].tolist();amounts=counts[kh].tolist()
        ranking=sorted((x for x in range(n) if amounts[x]>0),key=lambda x:(-scores[x],x))
        maximum=scores[ranking[0]];weights=[math.exp(scores[i]-maximum) for i in ranking]
        total=sum(weights);acc=0.0;chosen=[]
        for slot,w in zip(ranking,weights):
            if w/total <= 1.0/len(ranking): break
            if amounts[slot]<=1024:chosen.append(slot)
        got=opened[kh,gh].nonzero(as_tuple=True)[0].tolist()
        if set(chosen)!=set(got):raise RuntimeError(f'adaptive top-p selection mismatch head {h}')
        # Independent explicit single-head frontier: gather only open entries,
        # closed means, and exact local/sink and evaluate one shared softmax.
        leaf_indices=leaf_mask[kh,gh].nonzero(as_tuple=True)[0]
        closed=(~opened[kh,gh] & valid[kh]).nonzero(as_tuple=True)[0]
        explicit_scores=torch.cat((coarse[kh,gh,closed],leaf_logits[kh,gh,leaf_indices],
                                   local_logits[kh,gh],sink_logits[kh,gh]))
        explicit_values=torch.cat((values[kh,closed],leaf_v[kh,leaf_indices],local_v[kh],sink_v[kh]),0)
        explicit=torch.softmax(explicit_scores,0)@explicit_values
        difference=(explicit-output[kh,gh]).abs()
        maxerr=float(difference.max());meanerr=float(difference.mean())
        if not math.isfinite(maxerr) or maxerr>0.001:raise RuntimeError(f'frontier reference mismatch {maxerr}')
        checks.append({'head':h,'opened':len(chosen),'max_abs':maxerr,'mean_abs':meanerr})
    metrics=torch.stack((nregions.float(),leaf_count.float(),remote_mass,
                         (candidate_count-nregions).float()),dim=-1).reshape(H*G,4)
    return output.reshape_as(q).to(q.dtype),metrics,checks


def install(model,threshold,check,routing_score):
    cache=model.kv_cache;orig=cache.decode_attention;cache._refs={};cache._checks=[]
    device=cache.pools[0].device
    cache._sums=torch.zeros((32,32,4),dtype=torch.float64,device=device)
    cache._mins=torch.full((32,32),float('inf'),device=device)
    cache._maxs=torch.zeros((32,32),device=device)
    cache._trace_step=0
    def decode(this,q,k,v,layer_idx):
        pool=this.pools[layer_idx];orig(q,k,v,layer_idx)
        signature=(int(pool.leaf_lens[0].item()),int(pool.state_lens[0].item()))
        if layer_idx not in this._refs or this._refs[layer_idx][0]!=signature:
            this._refs[layer_idx]=(signature,reconstruct(pool,this.context)[0])
        heads=(0,15,31) if check and this._trace_step==0 and layer_idx in (0,15,31) else ()
        result,metrics,checks=adaptive_decode(pool,q,this._refs[layer_idx][1],threshold,heads,routing_score)
        this._sums[layer_idx]+=metrics.double()
        this._mins[layer_idx]=torch.minimum(this._mins[layer_idx],metrics[:,0])
        this._maxs[layer_idx]=torch.maximum(this._maxs[layer_idx],metrics[:,0])
        this._checks.extend({'layer':layer_idx,**x} for x in checks)
        return result
    cache.decode_attention=types.MethodType(decode,cache)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cohort','prior','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--top-p',type=float,default=0.70)
    p.add_argument('--routing-score',choices=('native-query','normalized-query'),default='native-query')
    p.add_argument('--limit',type=int,default=32)
    p.add_argument('--rows',type=int,nargs='+')
    p.add_argument('--check-reference',action='store_true')
    a=p.parse_args()
    if not 0<a.top_p<=1:raise ValueError('top-p must be in (0,1]')
    if os.environ.get('LOD_RETRO_NATIVE_ROUTE_FOUR_SINK')=='1':raise ValueError('uses original sink configuration')
    cohort=json.loads(a.cohort.read_text());prior=json.loads(a.prior.read_text())
    if prior['mode']!='two-tier':raise ValueError('requires original two-tier run')
    originals={(r['task'],r['source_row']):r for r in prior['records']}
    samples=[s for s in cohort['samples'] if s['task']=='niah_multikey_3' and (a.rows is None or s['source_row'] in a.rows)][:a.limit]
    if a.rows is not None and len(samples)!=len(a.rows):raise ValueError('missing requested rows')
    torch.set_grad_enabled(False);name='meta-llama/Llama-3.1-8B-Instruct'
    tok=load_tokenizer(name);model=load_model(name,65536,torch.bfloat16,'cuda:0',tok);attach_lod(model,'two-tier')
    result={'stage':'running','model':name,'mode':'two-tier','selection':'probability > 1 / valid region count',
        'routing':a.routing_score+' centroid scores normalized over remote regions only; open region when its probability exceeds 1 / valid region count',
        'routing_score':a.routing_score,
        'oversize':'regions larger than 1024 entries are not opened and retain summaries',
        'method':'FP32 PyTorch decode; optimized two-tier sparse prefill and cache update',
        'generation':'greedy, stop on first end-of-turn',
        'cohort_sha256':hashlib.sha256(a.cohort.read_bytes()).hexdigest(),'records':[]}
    save(a.output,result)
    for sample in samples:
        start=time.monotonic();row=sample['source_row'];old=originals[(sample['task'],row)]
        inputs=tok(sample['prompt'],return_tensors='pt')['input_ids'].to('cuda');n=inputs.shape[1]
        digest=hashlib.sha256(struct.pack(f'<{n}I',*inputs[0].tolist())).hexdigest()
        if digest!=old['input_token_sha256']:raise RuntimeError('input mismatch')
        model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=n
        model.max_new_length=128;model.prefill_method='full'
        model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,n,'Full_Flash_Attn'))
        token=model.sampling(model.prefill_forward(inputs_ids=inputs))
        if int(token.item())!=old['output_token_ids'][0]:raise RuntimeError('prefill first-token mismatch')
        model.move();install(model,a.top_p,check=a.check_reference and len(result['records'])==0,routing_score=a.routing_score)
        generated=[int(token.item())]
        for step in range(127):
            model.kv_cache._trace_step=step
            token=model.sampling(model.decode_forward(inputs_ids=token));generated.append(int(token.item()))
            if int(token.item()) in (tok.eos_token_id,tok.convert_tokens_to_ids('<|eot_id|>')):break
        text=tok.decode(generated,skip_special_tokens=True)
        steps=len(generated)-1
        record={'task':sample['task'],'row':row,'input_tokens':n,'input_token_sha256':digest,'answers':sample['answers'],
            'prediction':text,'score':score(sample['task'],text,sample['answers']),'output_token_ids':generated,
            'decode_steps':steps,'routing_sums':model.kv_cache._sums.cpu().tolist(),
            'routing_mins':model.kv_cache._mins.cpu().tolist(),'routing_maxs':model.kv_cache._maxs.cpu().tolist(),
            'reference_checks':model.kv_cache._checks,'elapsed_s':round(time.monotonic()-start,2)}
        result['records'].append(record);save(a.output,result)
        print(json.dumps({'row':row,'score':record['score'],'mean_regions':sum(sum(x[0] for x in l) for l in record['routing_sums'])/(steps*1024),
                          'elapsed_s':record['elapsed_s']}),flush=True)
        del model.kv_cache,inputs,token;gc.collect();torch.cuda.empty_cache()
    result['stage']='complete';save(a.output,result)
if __name__=='__main__':main()

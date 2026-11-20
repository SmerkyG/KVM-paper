"""RULER three-tier top-two-page diagnostic with sort-free centroid threshold.

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


def build_page_reference(pool, context):
    ref,page_owner,page_ordinal,leaf_page=reconstruct(pool,context)
    page=pool.state['page_cache']; indices=page['page_indices'][0].long()
    H,P,W=indices.shape; valid=(indices>=0)&(page_owner[:,:,None]>=0)
    hh=torch.arange(H,device=pool.device)[:,None,None]
    leaf_k=ref.archive_key[0].float();leaf_v=ref.archive_value[0].float()
    pk=(leaf_k[hh,indices.clamp_min(0)]*valid[...,None]).sum(2)
    pv=(leaf_v[hh,indices.clamp_min(0)]*valid[...,None]).sum(2)
    count=valid.sum(-1).float(); slots=page_owner.clamp_min(0)
    n=ref.state.size
    represented=torch.zeros((H,n),device=pool.device).scatter_add_(1,slots,count)
    if not torch.equal(represented,ref.state.count[0].float()):raise RuntimeError('page counts disagree with region counts')
    # The readable recursive reference computes residual sums from the actual
    # BF16 leaves, rather than from the possibly rounded stored region sums.
    D=pool.head_dim; owner=ref.owner[0]
    region_k=torch.zeros((H,n,D),device=pool.device).scatter_add_(1,owner[:,:,None].expand(-1,-1,D),leaf_k)
    region_v=torch.zeros_like(region_k).scatter_add_(1,owner[:,:,None].expand(-1,-1,D),leaf_v)
    return ref,slots,page_ordinal,leaf_page,pk,pv,count,region_k,region_v


def adaptive_decode(pool,q,data,threshold,check_heads,routing_score):
    cache,pslot,pord,leaf_page,pk,pv,pc,region_k,region_v=data
    H,G,D=pool.kv_heads,pool.query_heads//pool.kv_heads,pool.head_dim
    query=q[0,0].reshape(H,G,D).float(); scale=1/math.sqrt(D)
    state=cache.state;counts=state.count[0].float();valid=counts>0
    keys=state.mean_key[0].float();values=state.mean_value[0].float(); n=state.size
    coarse=torch.einsum('hgd,hsd->hgs',query,keys)*scale+counts.clamp_min(1).log()[:,None,:]
    coarse=coarse.masked_fill(~valid[:,None,:],-torch.inf)
    if routing_score=='normalized-query':
        norm=query*torch.rsqrt(query.square().mean(-1,keepdim=True).clamp_min(1e-12))
        route=(torch.einsum('hgd,hsd->hgs',norm,keys)*scale+counts.clamp_min(1).log()[:,None,:]).masked_fill(~valid[:,None,:],-torch.inf)
    else: route=coarse
    probs=route.softmax(-1); candidates=(probs>threshold/valid.sum(-1)[:,None,None])&valid[:,None,:]
    opened=candidates&(counts[:,None,:]<=1024)
    page_slot=pslot[:,None,:].expand(-1,G,-1)
    page_ok=(pc[:,None,:]>0)&opened.gather(-1,page_slot)
    # Score pages from the actual keys, as in the readable top-two reference.
    page_dot=torch.einsum('hgd,hpd->hgp',query,pk)*scale
    page_scores=page_dot/pc.clamp_min(1)[:,None,:]+pc.clamp_min(1).log()[:,None,:]
    page_scores=page_scores.masked_fill(~page_ok,-torch.inf)
    selected=torch.zeros_like(page_scores,dtype=torch.bool)
    big=torch.iinfo(torch.int64).max
    for _ in range(2):
        mx=torch.full_like(coarse,-torch.inf).scatter_reduce_(-1,page_slot,page_scores,reduce='amax',include_self=True)
        ties=(page_scores==mx.gather(-1,page_slot))&page_scores.isfinite()
        ords=pord[:,None,:].expand(-1,G,-1)
        first=torch.full_like(coarse,big,dtype=torch.int64).scatter_reduce_(-1,page_slot,torch.where(ties,ords,big),reduce='amin',include_self=True)
        pick=ties&(ords==first.gather(-1,page_slot));selected|=pick
        page_scores=page_scores.masked_fill(pick,-torch.inf)
    selected_float=selected.float()
    exact_per_region=torch.zeros_like(coarse).scatter_add_(-1,page_slot,selected_float*pc[:,None,:])
    removed_dot=torch.zeros_like(coarse).scatter_add_(-1,page_slot,selected_float*page_dot)
    residual_count=(counts[:,None,:]-exact_per_region).clamp_min(0)
    region_dot=torch.einsum('hgd,hsd->hgs',query,region_k)*scale
    residual_logits=(region_dot-removed_dot)/residual_count.clamp_min(1)+residual_count.clamp_min(1).log()
    summary_logits=torch.where(opened,residual_logits,coarse).masked_fill(opened & (residual_count<=0),-torch.inf)
    owner=cache.owner[0];leaf_k=cache.archive_key[0].float();leaf_v=cache.archive_value[0].float()
    leaf_mask=selected.gather(-1,leaf_page[:,None,:].expand(-1,G,-1))
    leaf_logits=torch.einsum('hgd,htd->hgt',query,leaf_k)*scale
    nlocal=int(pool.local_lens[0].item()); local_k=pool.state['recent_k'][0,:,:nlocal].float();local_v=pool.state['recent_v'][0,:,:nlocal].float()
    sink_k=cache.sink_key[0].float();sink_v=cache.sink_value[0].float()
    local_logits=torch.einsum('hgd,htd->hgt',query,local_k)*scale
    sink_logits=torch.einsum('hgd,htd->hgt',query,sink_k)*scale
    logits=torch.cat((summary_logits,leaf_logits.masked_fill(~leaf_mask,-torch.inf),local_logits,sink_logits),-1)
    p=logits.softmax(-1);t=leaf_k.size(1);summary_p=p[...,:n]
    # Closed regions use the original centroid value; selected regions use the
    # residual sum after removing the selected page values.
    residual_weight=summary_p*opened/residual_count.clamp_min(1)
    page_weight=residual_weight.gather(-1,page_slot)*selected_float
    output=(torch.einsum('hgs,hsd->hgd',summary_p*(~opened),values)+
            torch.einsum('hgs,hsd->hgd',residual_weight,region_v)-
            torch.einsum('hgp,hpd->hgd',page_weight,pv)+
            torch.einsum('hgt,htd->hgd',p[...,n:n+t],leaf_v)+
            torch.einsum('hgt,htd->hgd',p[...,n+t:n+t+nlocal],local_v)+
            torch.einsum('hgt,htd->hgd',p[...,n+t+nlocal:],sink_v))
    checks=[]
    for h in check_heads:
        kh,gh=divmod(h,G)
        scores=route[kh,gh].tolist(); amounts=counts[kh].tolist()
        ranking=[i for i,x in enumerate(amounts) if x>0]; maximum=max(scores[i] for i in ranking)
        w={i:math.exp(scores[i]-maximum) for i in ranking};z=sum(w.values())
        expected={i for i in ranking if w[i]/z>threshold/len(ranking) and amounts[i]<=1024}
        if expected!=set(opened[kh,gh].nonzero(as_tuple=True)[0].tolist()):raise RuntimeError('region threshold mismatch')
        ids=pslot[kh].tolist(); ord_list=pord[kh].tolist(); sizes=pc[kh].tolist()
        raw_dot=page_dot[kh,gh].tolist(); selected_ids=set(selected[kh,gh].nonzero(as_tuple=True)[0].tolist());expected_pages=set()
        for slot in expected:
            pages=[i for i in range(len(ids)) if ids[i]==slot and sizes[i]>0]
            pages.sort(key=lambda i:(-(raw_dot[i]/sizes[i]+math.log(sizes[i])),ord_list[i]))
            expected_pages.update(pages[:2])
        if expected_pages!=selected_ids:raise RuntimeError('top-two page mismatch')
        # Independent single-head frontier and residual arithmetic.
        picked=leaf_mask[kh,gh].nonzero(as_tuple=True)[0]
        closed=(~opened[kh,gh]&valid[kh]).nonzero(as_tuple=True)[0]
        resid=(opened[kh,gh]&(residual_count[kh,gh]>0)).nonzero(as_tuple=True)[0]
        # Rebuild residuals directly from the unopened leaves, independently
        # of the page sums and subtraction used by the vectorized path.
        kept=~leaf_mask[kh,gh]; leaf_slot=owner[kh,kept]
        independent_k=torch.zeros_like(region_k[kh]).index_add_(0,leaf_slot,leaf_k[kh,kept])
        independent_v=torch.zeros_like(region_v[kh]).index_add_(0,leaf_slot,leaf_v[kh,kept])
        residual_v=independent_v[resid]/residual_count[kh,gh,resid,None]
        independent_scores=(independent_k[resid]@query[kh,gh])*scale/residual_count[kh,gh,resid]+residual_count[kh,gh,resid].log()
        if resid.numel() and float((independent_scores-residual_logits[kh,gh,resid]).abs().max())>0.001:raise RuntimeError('residual logit mismatch')
        explicit_scores=torch.cat((coarse[kh,gh,closed],independent_scores,leaf_logits[kh,gh,picked],local_logits[kh,gh],sink_logits[kh,gh]))
        explicit_values=torch.cat((values[kh,closed],residual_v,leaf_v[kh,picked],local_v[kh],sink_v[kh]),0)
        explicit=explicit_scores.softmax(0)@explicit_values
        maxerr=float((explicit-output[kh,gh]).abs().max())
        if not math.isfinite(maxerr) or maxerr>0.001:raise RuntimeError(f'frontier mismatch {maxerr}')
        checks.append({'head':h,'opened_regions':len(expected),'opened_pages':len(expected_pages),'max_abs':maxerr})
    metrics=torch.stack((opened.sum(-1).float(),leaf_mask.sum(-1).float(),(probs*opened).sum(-1),(candidates.sum(-1)-opened.sum(-1)).float(),selected.sum(-1).float()),-1).reshape(H*G,5)
    return output.reshape_as(q).to(q.dtype),metrics,checks



def install(model,threshold,check,routing_score):
    cache=model.kv_cache;orig=cache.decode_attention;cache._refs={};cache._checks=[]
    device=cache.pools[0].device
    cache._sums=torch.zeros((32,32,5),dtype=torch.float64,device=device)
    cache._mins=torch.full((32,32),float('inf'),device=device)
    cache._maxs=torch.zeros((32,32),device=device)
    cache._trace_step=0
    def decode(this,q,k,v,layer_idx):
        pool=this.pools[layer_idx];orig(q,k,v,layer_idx)
        signature=(int(pool.leaf_lens[0].item()),int(pool.state_lens[0].item()))
        if layer_idx not in this._refs or this._refs[layer_idx][0]!=signature:
            this._refs[layer_idx]=(signature,build_page_reference(pool,this.context))
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
    p.add_argument('--multiplier',type=float,default=4.0)
    p.add_argument('--routing-score',choices=('native-query','normalized-query'),default='native-query')
    p.add_argument('--limit',type=int,default=32)
    p.add_argument('--rows',type=int,nargs='+')
    p.add_argument('--check-reference',action='store_true')
    a=p.parse_args()
    if a.multiplier<=0:raise ValueError('multiplier must be positive')
    if os.environ.get('LOD_RETRO_NATIVE_ROUTE_FOUR_SINK')=='1':raise ValueError('uses original sink configuration')
    cohort=json.loads(a.cohort.read_text());prior=json.loads(a.prior.read_text())
    if prior['mode']!='three-tier-bf16':raise ValueError('requires original three-tier BF16 run')
    originals={(r['task'],r['source_row']):r for r in prior['records']}
    samples=[s for s in cohort['samples'] if s['task']=='niah_multikey_3' and (a.rows is None or s['source_row'] in a.rows)][:a.limit]
    if a.rows is not None and len(samples)!=len(a.rows):raise ValueError('missing requested rows')
    torch.set_grad_enabled(False);name='meta-llama/Llama-3.1-8B-Instruct'
    tok=load_tokenizer(name);model=load_model(name,65536,torch.bfloat16,'cuda:0',tok);attach_lod(model,'three-tier-bf16')
    result={'stage':'running','model':name,'mode':'three-tier-bf16','selection':'probability > multiplier / valid region count','multiplier':a.multiplier,
        'routing':a.routing_score+' centroid scores normalized over remote regions only; open region when its probability exceeds multiplier / valid region count',
        'routing_score':a.routing_score,
        'oversize':'regions larger than 1024 entries are not opened and retain summaries',
        'method':'FP32 PyTorch two-page recursive decode; optimized three-tier BF16 sparse prefill and cache update',
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
        model.move();install(model,a.multiplier,check=a.check_reference and len(result['records'])==0,routing_score=a.routing_score)
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

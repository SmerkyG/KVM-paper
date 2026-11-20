"""Check and graph-time page thresholds against fixed page budgets on real KV.

Attention-call timing includes region routing, coarse attention, page selection,
exact leaves, residual, local/sink attention, append and identical length rollback.
Full-model timings with amortized updates are separate experiments.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import types
import numpy as np
import torch
sys.path.insert(0,os.environ.get('RETROINFER_ROOT','/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model,load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_three_tier_mean_regions import build_page_reference


def reference(pool,q,data,c,budget=None,rule='page',query_score='native',residual_kind='region'):
    cache,slot,ordinal,leaf_page,pk,pv,pc,rk,rv=data
    H,G,D=8,4,128;n=cache.state.size;scale=pool.engine.scaling
    query=q[0,0].reshape(H,G,D).float();count=cache.state.count[0].float()
    divisor=count.to(torch.bfloat16).float().clamp_min(1)[...,None]
    mk=(cache.state.key_sum[0].float()/divisor).to(torch.bfloat16).float()
    mv=(cache.state.value_sum[0].float()/divisor).to(torch.bfloat16).float()
    coarse=(torch.einsum('hgd,hsd->hgs',query,mk)*scale+count.clamp_min(1).log()[:,None]).masked_fill(count[:,None]<=0,-torch.inf)
    opened=pool._threshold_scratch['allowed'].reshape(H,G,-1)[...,:n].bool()
    owner=slot[:,None].expand(-1,G,-1)
    ok=(pc[:,None]>0)&opened.gather(-1,owner)
    sk=pool.state['page_cache']['page_sum_k'][0].float();sv=pool.state['page_cache']['page_sum_v'][0].float()
    score=torch.einsum('hgd,hpd->hgp',query,sk)*scale/pc.clamp_min(1)[:,None]+pc.clamp_min(1).log()[:,None]
    if c is not None and query_score=='normalized':
        score=(score-pc.clamp_min(1).log()[:,None])*query.square().mean(-1,keepdim=True).clamp_min(1.e-12).rsqrt()+pc.clamp_min(1).log()[:,None]
    score=score.masked_fill(~ok,-torch.inf)
    best=torch.full_like(coarse,-torch.inf).scatter_reduce_(-1,owner,score,reduce='amax',include_self=True)
    weight=(score-best.gather(-1,owner)).exp().nan_to_num(0.)
    z=torch.zeros_like(coarse).scatter_add_(-1,owner,weight)
    np=((count+15)//16)[:,None].expand(-1,G,-1)
    if c is None:
        from benchmarks.h100_three_tier_diagnose import pick_pages
        selected=pick_pages(score,slot,ok,n,budget)
    else:
        cutoff=(c*z.gather(-1,owner)*pc[:,None]/count[:,None].expand(-1,G,-1).gather(-1,owner).clamp_min(1)*(1.+1.e-6)
                if rule=='entry' else c*z.gather(-1,owner)/np.gather(-1,owner).clamp_min(1))
        selected=ok&((weight>cutoff)|(np.gather(-1,owner)==1))
    removed_n=torch.zeros_like(coarse).scatter_add_(-1,owner,selected*pc[:,None])
    left=count[:,None]-removed_n
    removed_dot=torch.zeros_like(coarse).scatter_add_(-1,owner,selected*torch.einsum('hgd,hpd->hgp',query,sk)*scale)
    if residual_kind=='pages' and c is not None:
        remainder=ok&~selected
        remaining_dot=torch.zeros_like(coarse).scatter_add_(-1,owner,remainder*torch.einsum('hgd,hpd->hgp',query,sk)*scale)
        residual=remaining_dot/left.clamp_min(1)+left.clamp_min(1).log()
    else:
        residual=(torch.einsum('hgd,hsd->hgs',query,cache.state.key_sum[0].float())*scale-removed_dot)/left.clamp_min(1)+left.clamp_min(1).log()
    summary=torch.where(opened,residual,coarse).masked_fill(opened&(left<=0),-torch.inf)
    leafmask=selected.gather(-1,leaf_page[:,None].expand(-1,G,-1))
    lk,lv=cache.archive_key[0].float(),cache.archive_value[0].float()
    ll=torch.einsum('hgd,htd->hgt',query,lk)*scale
    # Each measured call rolled back its length, but left the current KV written.
    nl=int(pool.local_lens[0].item())+1
    local_k=torch.cat((pool.state['recent_k'][0,:,:nl],pool.state['sink_k'][0]),1).float()
    local_v=torch.cat((pool.state['recent_v'][0,:,:nl],pool.state['sink_v'][0]),1).float()
    local=torch.einsum('hgd,htd->hgt',query,local_k)*scale
    probs=torch.cat((summary,ll.masked_fill(~leafmask,-torch.inf),local),-1).softmax(-1)
    sp=probs[...,:n];rw=sp*opened/left.clamp_min(1);t=lk.shape[1]
    residual_value=(torch.einsum('hgp,hpd->hgd',rw.gather(-1,owner)*remainder,sv)
                    if residual_kind=='pages' and c is not None else
                    torch.einsum('hgs,hsd->hgd',rw,cache.state.value_sum[0].float())-
                    torch.einsum('hgp,hpd->hgd',rw.gather(-1,owner)*selected,sv))
    output=(torch.einsum('hgs,hsd->hgd',sp*(~opened),mv)+residual_value+
        torch.einsum('hgt,htd->hgd',probs[...,n:n+t],lv)+
        torch.einsum('hgt,htd->hgd',probs[...,n+t:],local_v))
    stats=torch.stack((opened.sum(-1),ok.sum(-1),selected.sum(-1),leafmask.sum(-1)),-1).reshape(32,4)
    return output.reshape(1,1,32,128),stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--length',type=int,choices=(32768,65536),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rule',choices=('page','entry'),default='page');p.add_argument('--query',choices=('native','normalized'),default='native');p.add_argument('--multipliers',type=float,nargs='+',default=[.25,.5,1.]);p.add_argument('--residual',choices=('region','pages'),default='region');p.add_argument('--compare-residuals',action='store_true');a=p.parse_args()
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    region_multiplier=float(os.environ.get('LOD_H100_THRESHOLD_MULTIPLIER','16'))
    os.environ['LOD_H100_THRESHOLD_MULTIPLIER']=str(region_multiplier);os.environ['LOD_H100_THRESHOLD_IMPLEMENTATION']='three-tier'
    os.environ['LOD_H100_PAGE_THRESHOLD_RULE']=a.rule
    os.environ['LOD_H100_PAGE_QUERY']=a.query
    os.environ.pop('LOD_H100_PAGE_THRESHOLD',None);os.environ['LOD_H100_THREE_TIER_PAGES']='2'
    name='meta-llama/Llama-3.1-8B-Instruct';tok=load_tokenizer(name);model=load_model(name,a.length+2048,torch.bfloat16,'cuda:0',tok);attach_lod(model,'three-tier-bf16')
    ids=json.loads(a.manifest.read_text())['contexts'][str(a.length)]['prompts'][0]['prompt_token_ids'];inputs=torch.tensor([ids],device='cuda')
    model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=a.length;model.max_new_length=2048;model.prefill_method='full'
    model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,a.length,'Full_Flash_Attn'))
    token=model.sampling(model.prefill_forward(inputs_ids=inputs));model.move();cache=model.kv_cache;original=cache.decode_attention;saved={}
    def capture(this,q,k,v,layer):
        result=original(q,k,v,layer)
        if layer in (0,15,31):saved[layer]=(q.clone(),k.clone(),v.clone(),this.pools[layer]);this.pools[layer].local_lens.sub_(1)
        return result
    cache.decode_attention=types.MethodType(capture,cache);model.sampling(model.decode_forward(inputs_ids=token));torch.cuda.synchronize()
    from benchmarks.h100_threshold_three_tier import threshold_attention
    configs=[dict(pages=pages,c=None,workers=64,warps=2 if a.length==32768 else 1,residual='region') for pages in (2,4)]
    residuals=('region','pages') if a.compare_residuals else (a.residual,)
    configs += [dict(pages=None,c=c,workers=w,warps=warps,residual=residual) for c in a.multipliers for w in (32,64) for warps in (2,4) for residual in residuals]
    records=[]
    for layer,(q,k,v,pool) in saved.items():
        data=build_page_reference(pool,cache.context);refs={};out=torch.empty_like(q)
        for cfg in configs:
            os.environ['LOD_H100_THREE_TIER_WORKERS']=str(cfg['workers']);os.environ['LOD_H100_THREE_TIER_PAGE_WARPS']=str(cfg['warps'])
            os.environ['LOD_H100_PAGE_THRESHOLD_WARPS']=str(cfg['warps'])
            os.environ['LOD_H100_PAGE_RESIDUAL']=cfg['residual']
            if cfg['c'] is None:os.environ.pop('LOD_H100_PAGE_THRESHOLD',None);os.environ['LOD_H100_THREE_TIER_PAGES']=str(cfg['pages'])
            else:os.environ['LOD_H100_PAGE_THRESHOLD']=str(cfg['c'])
            def invoke():
                threshold_attention(pool,q,k,v,out,region_multiplier,tile=64 if a.length==32768 else 128,splits=cfg['workers']);pool.local_lens.sub_(1)
            invoke();torch.cuda.synchronize()
            key=cfg['c'],cfg['pages'],cfg['residual']
            if key not in refs:refs[key]=reference(pool,q,data,cfg['c'],cfg['pages'],a.rule,a.query,cfg['residual'])
            expected,expected_stats=refs[key];torch.testing.assert_close(out.float(),expected,rtol=.008,atol=.008)
            error=float((out.float()-expected).abs().max());stats=expected_stats.float().mean(0).tolist()
            if cfg['c'] is not None:
                actual=pool._h100_page_threshold_stats.sum(1)
                if not torch.equal(actual.long(),expected_stats):raise RuntimeError('real-query page counts disagree with reference')
                stats=actual.float().mean(0).tolist()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):invoke()
            for _ in range(4):graph.replay()
            times=[]
            for _ in range(3):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                for _ in range(40):graph.replay()
                end.record();end.synchronize();times.append(begin.elapsed_time(end)*1000/320)
            row=dict(layer=layer,**cfg,microseconds=statistics.median(times),repetitions_us=times,reference_max_abs=error,work=stats,
                     query_rms=q.float().square().mean(-1).sqrt().flatten().tolist())
            records.append(row);print(json.dumps(row),flush=True);a.output.write_text(json.dumps(dict(stage='running',length=a.length,region_multiplier=region_multiplier,page_threshold_rule=a.rule,page_query=a.query,records=records),indent=2)+'\n');del graph
        del data,refs
    a.output.write_text(json.dumps(dict(stage='complete',length=a.length,region_multiplier=region_multiplier,page_threshold_rule=a.rule,page_query=a.query,records=records),indent=2)+'\n')


if __name__=='__main__':main()

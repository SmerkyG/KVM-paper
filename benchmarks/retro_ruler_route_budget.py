"""Paired two-tier RULER quality diagnostic with configurable region count.

Uses optimized sparse prefill and cache updates, and an eager FP32/PyTorch
frontier computation during decode. This is an accuracy probe, not a timing
benchmark or a production kernel.
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
sys.path.insert(0, os.environ.get('RETROINFER_ROOT','/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model,load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_run import save,score
from benchmarks.retro_ruler_frontier_audit import reconstruct


def reference_decode(pool, q, cache, route_count):
    """Use torch to score exactly opened leaves and closed-region summaries."""
    H,G,D=pool.kv_heads,pool.query_heads//pool.kv_heads,pool.head_dim
    query=q[0,0].reshape(H,G,D).float(); scale=1/math.sqrt(D)
    state=cache.state;counts=state.count[0].float();valid=counts>0
    means=state.mean_key[0].float();values=state.mean_value[0].float()
    logits=torch.einsum('hgd,hsd->hgs',query,means)*scale+counts.clamp_min(1).log()[:,None,:]
    normalized=query*torch.rsqrt(query.square().mean(-1,keepdim=True).clamp_min(1e-12))
    route_score=torch.einsum('hgd,hsd->hgs',normalized,means)*scale+counts.clamp_min(1).log()[:,None,:]
    route_score=route_score.masked_fill(~valid[:,None,:],-torch.inf)
    indices=route_score.topk(min(route_count,state.size),dim=-1).indices
    active=valid[:,None,:].expand(-1,G,-1).gather(-1,indices) & (counts[:,None,:].expand(-1,G,-1).gather(-1,indices)<=1024)
    opened=torch.zeros_like(logits,dtype=torch.bool).scatter_(-1,indices,active)
    coarse_logits=logits.masked_fill(opened|~valid[:,None,:],-torch.inf)
    owner=cache.owner[0];mask=opened.gather(-1,owner[:,None,:].expand(-1,G,-1))
    leaf_k=cache.archive_key[0].float();leaf_v=cache.archive_value[0].float()
    leaf_logits=torch.einsum('hgd,htd->hgt',query,leaf_k)*scale
    leaf_logits=leaf_logits.masked_fill(~mask,-torch.inf)
    nlocal=int(pool.local_lens[0].item())
    local_k=pool.state['recent_k'][0,:,:nlocal].float();local_v=pool.state['recent_v'][0,:,:nlocal].float()
    sink_k=cache.sink_key[0].float();sink_v=cache.sink_value[0].float()
    local_logits=torch.einsum('hgd,htd->hgt',query,local_k)*scale
    sink_logits=torch.einsum('hgd,htd->hgt',query,sink_k)*scale
    all_logits=torch.cat((coarse_logits,leaf_logits,local_logits,sink_logits),dim=-1)
    p=torch.softmax(all_logits,dim=-1)
    n=state.size;t=leaf_k.size(1)
    result=(torch.einsum('hgs,hsd->hgd',p[...,:n],values)+
            torch.einsum('hgt,htd->hgd',p[...,n:n+t],leaf_v)+
            torch.einsum('hgt,htd->hgd',p[...,n+t:n+t+nlocal],local_v)+
            torch.einsum('hgt,htd->hgd',p[...,n+t+nlocal:],sink_v))
    return result.reshape_as(q).to(q.dtype)


def install_reference(model,route_count,*,check):
    cache=model.kv_cache; orig=cache.decode_attention; cache._route_refs={};cache._route_checks=[]
    def decode(this,q,k,v,layer_idx):
        pool=this.pools[layer_idx];optimized=orig(q,k,v,layer_idx)
        signature=(int(pool.leaf_lens[0].item()),int(pool.state_lens[0].item()))
        if layer_idx not in this._route_refs or this._route_refs[layer_idx][0]!=signature:
            this._route_refs[layer_idx]=(signature,reconstruct(pool,this.context)[0])
        ref=this._route_refs[layer_idx][1]
        result=reference_decode(pool,q,ref,route_count)
        if check and len(this._route_checks)<32:
            from lod_attention.pytorch_engine import PytorchLODAttention,PytorchLODConfig
            config=PytorchLODConfig(route_count=route_count)
            engine=PytorchLODAttention(config,mode='two-tier')
            local_len=int(pool.local_lens[0].item())
            direct=engine._lod_attention(q.transpose(1,2),pool.state['recent_k'][:,:,:local_len],
                pool.state['recent_v'][:,:,:local_len],ref,scale=1/math.sqrt(pool.head_dim),
                query_offset=local_len-1,full_regions=True).output.transpose(1,2)
            difference=(direct.float()-result.float()).abs()
            a={'layer':layer_idx,'max_abs':float(difference.max()),'mean_abs':float(difference.mean())}
            if route_count==8:
                difference=(optimized.float()-result.float()).abs()
                a.update(optimized_max_abs=float(difference.max()),optimized_mean_abs=float(difference.mean()))
            if not math.isfinite(a['max_abs']) or a['max_abs']>0.2 or a['mean_abs']>0.01:
                raise RuntimeError(f'FP32 frontier differs from reference in layer {layer_idx}: {a}')
            this._route_checks.append(a)
        return result
    cache.decode_attention=types.MethodType(decode,cache)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cohort','prior','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--routes',type=int,choices=(8,36),required=True)
    p.add_argument('--limit',type=int,default=32)
    p.add_argument('--rows',type=int,nargs='+')
    p.add_argument('--check-reference',action='store_true')
    a=p.parse_args()
    if os.environ.get('LOD_RETRO_NATIVE_ROUTE_FOUR_SINK')=='1':raise ValueError('requires original routing and sink settings')
    C=json.loads(a.cohort.read_text());P=json.loads(a.prior.read_text())
    if P['mode']!='two-tier':raise ValueError('requires original two-tier run')
    originals={(r['task'],r['source_row']):r for r in P['records']}
    samples=[s for s in C['samples'] if s['task']=='niah_multikey_3' and (a.rows is None or s['source_row'] in a.rows)][:a.limit]
    if a.rows is not None and len(samples)!=len(a.rows):raise ValueError('missing requested rows')
    torch.set_grad_enabled(False);name='meta-llama/Llama-3.1-8B-Instruct'
    tok=load_tokenizer(name);model=load_model(name,65536,torch.bfloat16,'cuda:0',tok);attach_lod(model,'two-tier')
    result={'stage':'running','model':name,'mode':'two-tier','routes':a.routes,
            'method':'FP32 PyTorch frontier, optimized two-tier sparse prefill and cache update',
            'generation':'greedy, stop on first end-of-turn',
            'cohort_sha256':hashlib.sha256(a.cohort.read_bytes()).hexdigest(),'records':[]}
    save(a.output,result)
    for sample in samples:
        begin=time.monotonic(); row=sample['source_row']; old=originals[(sample['task'],row)]
        inputs=tok(sample['prompt'],return_tensors='pt')['input_ids'].to('cuda');n=inputs.shape[1]
        digest=hashlib.sha256(struct.pack(f'<{n}I',*inputs[0].tolist())).hexdigest()
        if digest!=old['input_token_sha256']:raise RuntimeError('input mismatch')
        model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=n
        model.max_new_length=128;model.prefill_method='full'
        model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,n,'Full_Flash_Attn'))
        token=model.sampling(model.prefill_forward(inputs_ids=inputs))
        if int(token.item())!=old['output_token_ids'][0]:raise RuntimeError('prefill first token mismatch')
        model.move();install_reference(model,a.routes,check=a.check_reference and len(result['records'])==0)
        generated=[int(token.item())]
        for step in range(127):
            token=model.sampling(model.decode_forward(inputs_ids=token));generated.append(int(token.item()))
            if int(token.item()) in (tok.eos_token_id,tok.convert_tokens_to_ids('<|eot_id|>')):break
        text=tok.decode(generated,skip_special_tokens=True)
        record={'task':sample['task'],'row':row,'input_tokens':n,'input_token_sha256':digest,'answers':sample['answers'],
                'prediction':text,'score':score(sample['task'],text,sample['answers']),
                'output_token_ids':generated,'reference_checks':model.kv_cache._route_checks,
                'elapsed_s':round(time.monotonic()-begin,2)}
        result['records'].append(record);save(a.output,result)
        print(json.dumps({k:record[k] for k in ('row','score','elapsed_s')}),flush=True)
        del model.kv_cache;del inputs,token;gc.collect();torch.cuda.empty_cache()
    result['stage']='complete';save(a.output,result)
if __name__=='__main__':main()

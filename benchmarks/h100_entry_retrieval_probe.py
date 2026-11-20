"""Inspect a missed answer's regions at the first differing decode step."""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import types
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,os.environ.get('RETROINFER_ROOT','/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model,load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_frontier_audit import reconstruct
from benchmarks.retro_ruler_run import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','entry','region','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--row',type=int,required=True);a=p.parse_args()
    if os.environ.get('LOD_H100_THRESHOLD_RULE')!='entry' or not os.environ.get('LOD_H100_THRESHOLD_MULTIPLIER'):
        p.error('enable entry threshold decode')
    cohort=json.loads(a.cohort.read_text());entry=json.loads(a.entry.read_text());region=json.loads(a.region.read_text())
    if entry['stage']!='complete' or region['stage']!='complete':raise ValueError('both comparison runs must be complete')
    sample=next(x for x in cohort['samples'] if x['task']==a.task and x['source_row']==a.row)
    candidate=next(x for x in entry['records'] if x['task']==a.task and x['source_row']==a.row)
    baseline=next(x for x in region['records'] if x['task']==a.task and x['source_row']==a.row)
    target=next(i for i,(x,y) in enumerate(zip(candidate['output_token_ids'],baseline['output_token_ids'])) if x!=y)
    if target<1:raise ValueError('first divergence is before decode')
    torch.set_grad_enabled(False);name='meta-llama/Llama-3.1-8B-Instruct';tok=load_tokenizer(name)
    encoded=tok(sample['prompt'],return_tensors='pt',return_offsets_mapping=True)
    offsets=encoded.pop('offset_mapping')[0].tolist();inputs=encoded['input_ids'].to('cuda');n=inputs.shape[1]
    digest=hashlib.sha256(struct.pack(f'<{n}I',*inputs[0].tolist())).hexdigest()
    if digest!=candidate['input_token_sha256'] or digest!=baseline['input_token_sha256']:raise ValueError('prompt tokens differ')
    spans=[]
    for answer in sample['answers']:
        position=0
        while (position:=sample['prompt'].find(answer,position))>=0:
            spans.append((position,position+len(answer)));position+=len(answer)
    gold=[i for i,(lo,hi) in enumerate(offsets) if hi>lo and any(lo<end and hi>start for start,end in spans)]
    if not gold:raise ValueError('answer does not appear in prompt')
    model=load_model(name,65536,torch.bfloat16,'cuda:0',tok);attach_lod(model,'two-tier')
    model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=n;model.max_new_length=128;model.prefill_method='full'
    model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,n,'Full_Flash_Attn'))
    token=model.sampling(model.prefill_forward(inputs_ids=inputs));model.move()
    if int(token.item())!=candidate['output_token_ids'][0]:raise RuntimeError('prefill token does not reproduce entry run')
    cache=model.kv_cache;original=cache.decode_attention;rows=[];state={'step':-1}
    def trace(this,q,k,v,layer):
        output=original(q,k,v,layer)
        if state['step']!=target-1:return output
        pool=this.pools[layer];ref,_,_,_=reconstruct(pool,this.context)
        H,G,D=pool.kv_heads,pool.query_heads//pool.kv_heads,pool.head_dim
        leaf=ref.archive_key[0].float();owner=ref.owner[0].long();T=leaf.shape[1];sink=int(pool.engine.sink_len)
        ids=torch.tensor([position-sink for position in gold if sink<=position<sink+T],device=q.device,dtype=torch.long)
        if not len(ids):raise RuntimeError('no gold tokens in archived KV')
        query=q[0,0].reshape(H,G,D).float();scale=1/math.sqrt(D)
        local_len=int(pool.local_lens[0].item());local=pool.state['recent_k'][0,:,:local_len].float();sinks=ref.sink_key[0].float()
        logits=torch.cat((torch.einsum('hgd,htd->hgt',query,leaf),torch.einsum('hgd,htd->hgt',query,local),torch.einsum('hgd,htd->hgt',query,sinks)),dim=-1)*scale
        gold_prob=logits.softmax(-1)[...,:T].index_select(-1,ids)
        b=pool._threshold_scratch;S=pool.state_capacity;route=b['route'].reshape(H,G,S).float();prob=route.softmax(-1)
        counts=pool.state['counts'][0,:,:,0].float();lengths=pool.state['page_cache']['slot_lengths'][0]
        region_open=(prob>16/(counts.gt(0).sum(-1)[:,None,None]))&(lengths[:,None,:]<=1024)&(counts[:,None,:]>0)
        selected=b['selected'].reshape(H*G,S).long().clamp(0,S-1);selected_count=b['count'].long()
        selected_valid=torch.arange(S,device=q.device)[None,:]<selected_count[:,None]
        opened=torch.zeros((H*G,S),device=q.device,dtype=torch.int32).scatter_add_(1,selected,selected_valid.int()).reshape(H,G,S)>0
        gold_owner=owner.index_select(-1,ids)[:,None,:].expand(-1,G,-1)
        entry_mask=opened.gather(-1,gold_owner);region_mask=region_open.gather(-1,gold_owner)
        mass=gold_prob.sum(-1);entry_mass=(gold_prob*entry_mask).sum(-1);region_mass=(gold_prob*region_mask).sum(-1)
        for head in range(H*G):
            kh,gh=divmod(head,G)
            rows.append({'layer':layer,'head':head,'gold_exact_mass':float(mass[kh,gh]),
                         'gold_mass_open_entry':float(entry_mass[kh,gh]),'gold_mass_open_region':float(region_mass[kh,gh])})
        return output
    cache.decode_attention=types.MethodType(trace,cache)
    generated=[int(token.item())]
    for step in range(target):
        state['step']=step;token=model.sampling(model.decode_forward(inputs_ids=token));generated.append(int(token.item()))
        if generated[-1]!=candidate['output_token_ids'][step+1]:
            raise RuntimeError(f'entry run did not reproduce at output token {step+1}')
    if len(rows)!=32*32:raise RuntimeError('missing layer/head audit')
    result={'stage':'complete','task':a.task,'source_row':a.row,'input_token_sha256':digest,'answer_token_positions':gold,
            'first_differing_output_token':target,'entry_output_prefix':generated,
            'scope':'same entry-run queries at the first differing decode step; dense exact attention calculated on that cache; region rule simulated on the same queries',
            'heads':rows}
    save(a.output,result);print(json.dumps({'stage':'complete','task':a.task,'row':a.row,'divergence':target,'head_rows':len(rows)}),flush=True)

if __name__=='__main__':main()

"""Trace answer-region retention in the native-query 70% reference run."""
import argparse, hashlib, json, math, os, struct, sys, time, types
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.environ.get('RETROINFER_ROOT', '/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model, load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_run import save
from benchmarks.retro_ruler_frontier_audit import audit
from benchmarks.retro_ruler_adaptive_regions import install

def p70_selection(pool, q, ref, audit_result):
    h, g, d = pool.kv_heads, pool.query_heads // pool.kv_heads, pool.head_dim
    query = q[0,0].reshape(h,g,d).float()
    counts = ref.state.count[0].float()
    logits = torch.einsum('hgd,hsd->hgs',query,ref.state.mean_key[0].float())/math.sqrt(d)
    logits += counts.clamp_min(1).log()[:,None,:]
    logits = logits.masked_fill(counts[:,None,:] <= 0, -torch.inf)
    sorted_logits, indices = logits.sort(dim=-1,descending=True,stable=True)
    p = sorted_logits.softmax(-1)
    selected = (p.cumsum(-1)-p < 0.7) & torch.isfinite(sorted_logits)
    selected &= counts[:,None,:].expand(-1,g,-1).gather(-1,indices) <= 1024
    num = selected.sum(-1).reshape(-1).tolist()
    rank = audit_result['top_gold_region_rank_native']
    # Sizes are <= 1024 in these traces; retain explicit checking.
    if bool((counts > 1024).any()): raise RuntimeError('oversize region present')
    return {'p70_open_regions':num,'p70_top_gold_region_open':[rank[i] <= num[i] for i in range(h*g)],
            'p70_top_region_estimated_remote_mass':p[...,0].reshape(-1).tolist()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cohort','dense','prior','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=('two-tier','three-tier-bf16'),required=True)
    p.add_argument('--rows',type=int,nargs='+',default=[78,126,302,447])
    a=p.parse_args()
    C=json.loads(a.cohort.read_text()); D=json.loads(a.dense.read_text()); P=json.loads(a.prior.read_text())
    if P['mode'] != a.mode: raise ValueError('prior mode mismatch')
    dense={(r['task'],r['source_row']):r for r in D['records']}
    prior={(r['task'],r['row']):r for r in P['records']}
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
        token=model.sampling(model.prefill_forward(inputs_ids=inputs));model.move(); install(model,0.7,False,'native-query')
        if token.item()!=orig['output_token_ids'][0]: raise RuntimeError('prefill token mismatch')
        cache=model.kv_cache; orig_decode=cache.decode_attention; layer_audit=[]; enabled=False
        def decode(this,q,k,v,layer_idx):
            out=orig_decode(q,k,v,layer_idx)
            if enabled:
                x=audit(this.pools[layer_idx],q,out,gold,this.context)
                x.update(p70_selection(this.pools[layer_idx], q, this._refs[layer_idx][1], x))
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

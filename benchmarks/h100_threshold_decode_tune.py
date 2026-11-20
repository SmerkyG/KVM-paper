"""Compare uncapped threshold decode on identical real layer queries and caches.

CUDA graphs cover the complete attention call and the same local-length rollback
for every method. Fixed-cache timings exclude model layers and cache catch-up;
full-model amortized measurements must be reported separately.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
import torch

sys.path.insert(0,os.environ.get('RETROINFER_ROOT','/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model,load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.h100_threshold_decode import threshold_attention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--length',type=int,default=32768)
    p.add_argument('--layers',type=int,nargs='+',default=[0,15,31])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_grad_enabled(False)
    os.environ.pop('LOD_H100_THRESHOLD_MULTIPLIER',None)
    name='meta-llama/Llama-3.1-8B-Instruct';tok=load_tokenizer(name)
    model=load_model(name,a.length+1024,torch.bfloat16,'cuda:0',tok);attach_lod(model,'two-tier')
    ids=json.loads(a.manifest.read_text())['contexts'][str(a.length)]['prompts'][0]['prompt_token_ids']
    inputs=torch.tensor([ids],device='cuda')
    model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=a.length
    model.max_new_length=1024;model.prefill_method='full'
    model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,a.length,'Full_Flash_Attn'))
    token=model.sampling(model.prefill_forward(inputs_ids=inputs));model.move()
    cache=model.kv_cache;original=cache.decode_attention;saved={}
    def capture(this,q,k,v,layer):
        result=original(q,k,v,layer)
        if layer in a.layers:
            saved[layer]=(q.clone(),k.clone(),v.clone(),this.pools[layer])
            this.pools[layer].local_lens.sub_(1)
        return result
    cache.decode_attention=types.MethodType(capture,cache)
    model.sampling(model.decode_forward(inputs_ids=token));torch.cuda.synchronize()
    records=[]
    for layer in a.layers:
        q,k,v,pool=saved[layer];out=torch.empty_like(q)
        for rule in ('region','entry'):
            expected=None;expected_mask=None
            configs=[('legacy',64,32),('legacy',64,64),('packed',64,32),('packed',64,64)]
            configs += [(impl,64,splits) for impl in ('indexed','combined') for splits in (8,16,32,64)]
            for impl,tile,splits in configs:
                os.environ['LOD_H100_THRESHOLD_IMPLEMENTATION']=impl
                def invoke():
                    threshold_attention(pool,q,k,v,out,16.,tile=tile,splits=splits,rule=rule,entry_bound=16)
                    pool.local_lens.sub_(1)
                invoke();torch.cuda.synchronize()
                result=out.clone();scratch=pool._threshold_scratch
                counts=scratch['count'].clone();selected=scratch['selected'].clone()
                mask=torch.arange(pool.state_capacity,device=q.device)[None,:]<counts[:,None]
                encoded=torch.where(mask,selected,-1)
                if expected is None:expected=result;expected_mask=encoded
                if not torch.equal(encoded,expected_mask):raise RuntimeError('selected regions differ from legacy')
                delta=(result.float()-expected.float()).abs();maximum=float(delta.max().item())
                if not bool(torch.isfinite(result).all().item()) or maximum>0.02:
                    raise RuntimeError(f'output differs from legacy: {impl} {rule} {maximum}')
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):invoke()
                for _ in range(4):graph.replay()
                torch.cuda.synchronize();measurements=[]
                for _ in range(3):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                    for _ in range(40):graph.replay()
                    end.record();torch.cuda.synchronize()
                    measurements.append(begin.elapsed_time(end)*1000/(40*16))
                sizes=pool.state['page_cache']['slot_lengths'][0].repeat_interleave(4,0)
                opened=sizes.gather(1,encoded.clamp_min(0).long())*mask
                row=dict(layer=layer,rule=rule,implementation=impl,tile=tile,splits=splits,
                         microseconds=float(np.median(measurements)),repetitions_us=measurements,
                         selection_mismatches=0,max_abs_vs_legacy=maximum,
                         mean_abs_vs_legacy=float(delta.mean().item()),
                         mean_regions=float(counts.float().mean().item()),
                         mean_entries=float(opened.sum(-1).float().mean().item()),
                         local_entries=int(pool.local_lens.item()))
                records.append(row);print(json.dumps(row),flush=True)
                a.output.write_text(json.dumps(dict(stage='running',length=a.length,records=records),indent=2)+'\n')
                del graph
    a.output.write_text(json.dumps(dict(stage='complete',length=a.length,records=records,
                        scope='fixed real query and cache; full attention call plus identical local-length rollback; no cache catch-up'),indent=2)+'\n')


if __name__=='__main__':main()

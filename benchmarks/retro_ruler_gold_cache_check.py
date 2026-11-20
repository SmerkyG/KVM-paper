"""Verify that prompt answer-token offsets identify the same cached KV entries."""
import argparse, hashlib, json, os, struct, sys, types
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.environ.get('RETROINFER_ROOT','/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model,load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_run import save

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('cohort','prior','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--mode',choices=('two-tier','three-tier-bf16'),required=True)
 p.add_argument('--rows',type=int,nargs='+',default=[78,126,302,447]);a=p.parse_args()
 c=json.loads(a.cohort.read_text()); d=json.loads(a.prior.read_text())
 if d['mode']!=a.mode:raise ValueError('mode mismatch')
 old={(r['task'],r['source_row']):r for r in d['records']}
 torch.set_grad_enabled(False); name='meta-llama/Llama-3.1-8B-Instruct';tok=load_tokenizer(name)
 model=load_model(name,65536,torch.bfloat16,'cuda:0',tok);attach_lod(model,a.mode)
 original_prefill=model.prefill_attention
 output={'stage':'running','mode':a.mode,'records':[]};save(a.output,output)
 for row in a.rows:
  sample=next(s for s in c['samples'] if s['task']=='niah_multikey_3' and s['source_row']==row)
  encoded=tok(sample['prompt'],return_tensors='pt',return_offsets_mapping=True)
  offsets=encoded.pop('offset_mapping')[0].tolist();inputs=encoded['input_ids'].to('cuda');n=inputs.shape[1]
  digest=hashlib.sha256(struct.pack(f'<{n}I',*inputs[0].tolist())).hexdigest()
  if digest!=old[('niah_multikey_3',row)]['input_token_sha256']:raise RuntimeError('input token mismatch')
  text=sample['prompt']; answer=sample['answers'][0];spans=[];pos=0
  while (pos:=text.find(answer,pos))>=0:spans.append((pos,pos+len(answer)));pos+=len(answer)
  gold=[i for i,(lo,hi) in enumerate(offsets) if hi>lo and any(lo<end and hi>begin for begin,end in spans)]
  if not gold:raise RuntimeError('no answer token offsets')
  model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=n;model.max_new_length=128;model.prefill_method='full'
  model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,n,'Full_Flash_Attn'))
  verified=[]
  def checked_prefill(this,q,k,v,layer_idx):
   out=original_prefill(q,k,v,layer_idx);pool=this.kv_cache.pools[layer_idx];sink=pool.engine.sink_len
   if min(gold)<sink or max(gold)>=sink+int(pool.leaf_lens[0].item()):raise RuntimeError('gold token not in archived field')
   expected_k=k[:,gold].transpose(1,2);expected_v=v[:,gold].transpose(1,2)
   page=pool.state['page_cache'];ids=[i-sink for i in gold]
   if not torch.equal(page['leaf_k'][:,:,ids],expected_k) or not torch.equal(page['leaf_v'][:,:,ids],expected_v):
    raise RuntimeError(f'gold K/V mismatch layer {layer_idx}')
   verified.append(layer_idx);return out
  model.prefill_attention=types.MethodType(checked_prefill,model)
  token=model.sampling(model.prefill_forward(inputs_ids=inputs))
  if int(token.item())!=old[('niah_multikey_3',row)]['output_token_ids'][0] or len(verified)!=32:raise RuntimeError('prior prefill mismatch')
  output['records'].append({'row':row,'token_positions':gold,'layers_verified':len(verified)})
  save(a.output,output);print(f'row={row} mode={a.mode} layers={len(verified)} tokens={len(gold)} verified',flush=True)
  del model.kv_cache;torch.cuda.empty_cache()
 output['stage']='complete';save(a.output,output)
if __name__=='__main__':main()

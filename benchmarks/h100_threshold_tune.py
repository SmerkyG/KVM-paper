"""Tune threshold decode kernels on one real Llama layer's prefilled cache."""
import argparse,json,os,sys,types
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,os.environ.get('RETROINFER_ROOT','/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model,load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.h100_threshold_decode import threshold_attention

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--length',type=int,default=32768);p.add_argument('--layer',type=int,default=15);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_grad_enabled(False);name='meta-llama/Llama-3.1-8B-Instruct';tok=load_tokenizer(name);model=load_model(name,a.length+1024,torch.bfloat16,'cuda:0',tok);attach_lod(model,'two-tier')
 ids=json.loads(a.manifest.read_text())['contexts'][str(a.length)]['prompts'][0]['prompt_token_ids'];inp=torch.tensor([ids],device='cuda');model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=a.length;model.max_new_length=1024;model.prefill_method='full'
 model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,a.length,'Full_Flash_Attn'));token=model.sampling(model.prefill_forward(inputs_ids=inp));model.move();cache=model.kv_cache;old=cache.decode_attention;saved={}
 def capture(this,q,k,v,layer):
  out=old(q,k,v,layer)
  if layer==a.layer:
   saved.update(q=q.clone(),k=k.clone(),v=v.clone(),pool=this.pools[layer]);this.pools[layer].local_lens.sub_(1)
  return out
 cache.decode_attention=types.MethodType(capture,cache);model.sampling(model.decode_forward(inputs_ids=token));torch.cuda.synchronize()
 pool=saved['pool'];q,k,v=(saved[x] for x in ('q','k','v'));out=torch.empty_like(q);results=[];expected=None
 def run_one(kind,tile,splits):
  if kind=='top8':
   metadata=types.SimpleNamespace(num_actual_tokens=1,max_seq_len=a.length+1)
   def invoke():pool.decode(q.reshape(1,32,128),k.reshape(1,8,128),v.reshape(1,8,128),metadata,out.reshape(1,32,128))
  else:
   def invoke():threshold_attention(pool,q,k,v,out,16,tile=tile,splits=splits)
  invoke();pool.local_lens.sub_(1);torch.cuda.synchronize();sample=out.clone()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
   for _ in range(16):invoke();pool.local_lens.sub_(1)
  for _ in range(4):graph.replay()
  torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
  for _ in range(80):graph.replay()
  end.record();torch.cuda.synchronize();micros=start.elapsed_time(end)*1000/(80*16)
  return micros,sample
 for kind,tile,splits in [('top8',0,0)]+[('threshold',tile,splits) for tile in (32,64,128) for splits in (8,16,32,64)]:
  t,output=run_one(kind,tile,splits)
  if kind=='threshold' and expected is None:expected=output
  diff=float((output.float()-expected.float()).abs().max()) if kind=='threshold' else None
  row={'kind':kind,'tile':tile,'splits':splits,'microseconds':t,'max_abs_vs_first_threshold':diff};results.append(row);print(json.dumps(row),flush=True)
 a.output.write_text(json.dumps({'stage':'complete','model':name,'length':a.length,'layer':a.layer,'replays':80,'steps_per_replay':16,'records':results},indent=2)+'\n')
if __name__=='__main__':main()

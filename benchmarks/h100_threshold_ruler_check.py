"""Run BF16 CUDA-graph threshold decode on the paired RULER cohort."""
import argparse,gc,hashlib,json,os,struct,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,os.environ.get('RETROINFER_ROOT','/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model,load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_run import save,score

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('cohort','prior','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--multiplier',type=float,default=16);p.add_argument('--limit',type=int,default=32);a=p.parse_args()
 os.environ['LOD_H100_THRESHOLD_MULTIPLIER']=str(a.multiplier)
 torch.set_grad_enabled(False);C=json.loads(a.cohort.read_text());P=json.loads(a.prior.read_text())
 originals={(r['task'],r['source_row']):r for r in P['records']};samples=[s for s in C['samples'] if s['task']=='niah_multikey_3'][:a.limit]
 name='meta-llama/Llama-3.1-8B-Instruct';tok=load_tokenizer(name);model=load_model(name,65536,torch.bfloat16,'cuda:0',tok);attach_lod(model,'two-tier')
 result={'stage':'running','model':name,'mode':'two-tier','multiplier':a.multiplier,'routing_score':'normalized-query','method':'BF16 H100 experimental sort-free threshold CUDA graph, optimized two-tier sparse prefill',
 'generation':'greedy, stop on first end-of-turn','cohort_sha256':hashlib.sha256(a.cohort.read_bytes()).hexdigest(),'records':[]};save(a.output,result)
 for sample in samples:
  t=time.monotonic();row=sample['source_row'];old=originals[(sample['task'],row)];inputs=tok(sample['prompt'],return_tensors='pt')['input_ids'].to('cuda');n=inputs.shape[1]
  digest=hashlib.sha256(struct.pack(f'<{n}I',*inputs[0].tolist())).hexdigest()
  if digest!=old['input_token_sha256']:raise RuntimeError('input mismatch')
  model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=n;model.max_new_length=128;model.prefill_method='full'
  model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,n,'Full_Flash_Attn'))
  token=model.sampling(model.prefill_forward(inputs_ids=inputs));model.move()
  if int(token.item())!=old['output_token_ids'][0]:raise RuntimeError('prefill first-token mismatch')
  generated=[int(token.item())];model.kv_cache.capture_decode(model,token)
  for step in range(127):
   token=model.kv_cache.graph_step(token);generated.append(int(token.item()))
   if int(token.item()) in (tok.eos_token_id,tok.convert_tokens_to_ids('<|eot_id|>')):break
  text=tok.decode(generated,skip_special_tokens=True)
  record={'task':sample['task'],'row':row,'input_tokens':n,'input_token_sha256':digest,'answers':sample['answers'],
    'prediction':text,'score':score(sample['task'],text,sample['answers']),'output_token_ids':generated,'elapsed_s':round(time.monotonic()-t,2)}
  result['records'].append(record);save(a.output,result);print(json.dumps({'row':row,'score':record['score'],'elapsed_s':record['elapsed_s']}),flush=True)
  del model.kv_cache,inputs,token;gc.collect();torch.cuda.empty_cache()
 result['stage']='complete';save(a.output,result)
if __name__=='__main__':main()

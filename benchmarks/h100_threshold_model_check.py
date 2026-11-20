"""Check H100 threshold decode on real model queries and prefill caches."""
import argparse,json,math,os,sys,types
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,os.environ.get('RETROINFER_ROOT','/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model,load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.h100_threshold_decode import threshold_attention
from benchmarks.retro_ruler_frontier_audit import reconstruct
from benchmarks.retro_ruler_mean_multiple_regions import adaptive_decode

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',type=Path,required=True);p.add_argument('--row',type=int,default=4);p.add_argument('--multiplier',type=float,default=16);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 os.environ.pop('LOD_H100_THRESHOLD_MULTIPLIER',None)
 torch.set_grad_enabled(False);dataset=json.loads(a.cohort.read_text());sample=next(x for x in dataset['samples'] if x['task']=='niah_multikey_3' and x['source_row']==a.row)
 name='meta-llama/Llama-3.1-8B-Instruct';tok=load_tokenizer(name);model=load_model(name,65536,torch.bfloat16,'cuda:0',tok);attach_lod(model,'two-tier')
 inp=tok(sample['prompt'],return_tensors='pt')['input_ids'].to('cuda');n=inp.shape[1]
 model.attention_type='LoD';model.batch_size=model.prefill_bsz=1;model.input_length=n;model.max_new_length=32;model.prefill_method='full'
 model.init_kv_cache(np.zeros(1,dtype=int),generate_config(name,n,'Full_Flash_Attn'))
 token=model.sampling(model.prefill_forward(inputs_ids=inp));model.move();cache=model.kv_cache;old=cache.decode_attention;records=[]
 def check(this,q,k,v,layer):
  pool=this.pools[layer];baseline=old(q,k,v,layer);pool.local_lens.sub_(1)
  out=torch.empty((1,32,128),dtype=q.dtype,device=q.device);threshold_attention(pool,q,k,v,out,a.multiplier)
  ref=reconstruct(pool,this.context)[0]; expected,_,_=adaptive_decode(pool,q,ref,a.multiplier,(),'normalized-query')
  Q=q[0,0].reshape(8,4,128).float();cnt=ref.state.count[0].float();valid=cnt>0
  Qn=Q*torch.rsqrt(Q.square().mean(-1,keepdim=True).clamp_min(1e-12))
  score=(torch.einsum('hgd,hsd->hgs',Qn,ref.state.mean_key[0].float())/math.sqrt(128)+cnt.clamp_min(1).log()[:,None,:]).masked_fill(~valid[:,None,:],-torch.inf)
  mask=(score.softmax(-1)>a.multiplier/valid.sum(-1)[:,None,None])&valid[:,None,:]&(cnt[:,None,:]<=1024)
  b=pool._threshold_scratch;found=torch.zeros((32,pool.state_capacity),dtype=torch.bool,device=q.device)
  for head in range(32):
   size=int(b['count'][head]); inds=b['selected'][head,:size].long()
   if size!=len(inds.unique()):raise RuntimeError('duplicate region in selection')
   found[head,inds]=True
  actual=found[:,:ref.state.size].reshape(8,4,-1); mismatch=int((actual!=mask).sum());extra=int(found[:,ref.state.size:].sum())
  delta=(actual!=mask); cutoff=torch.logsumexp(score,-1,keepdim=True)+torch.log(torch.tensor(a.multiplier,device=q.device)/valid.sum(-1)[:,None,None])
  margin=float((score-cutoff).abs()[delta].max()) if mismatch else 0.0
  kernel_route=b['route'][:,:ref.state.size].reshape_as(score); score_diff=float((kernel_route[valid[:,None,:].expand_as(score)]-score[valid[:,None,:].expand_as(score)]).abs().max())
  diff=(out.reshape_as(q).float()-expected.float()).abs();maxabs=float(diff.max());meanabs=float(diff.mean())
  rows={'layer':layer,'selection_mismatches':mismatch,'outside_active':extra,'max_mismatch_margin':margin,'max_route_logit_difference':score_diff,'max_abs':maxabs,'mean_abs':meanabs,'regions_mean':float(mask.sum(-1).float().mean())}
  records.append(rows);print(json.dumps(rows),flush=True)
  if extra or maxabs>0.03 or (mismatch and margin>0.01):raise RuntimeError('model reference mismatch '+str(rows))
  return out.reshape_as(q)
 cache.decode_attention=types.MethodType(check,cache)
 model.sampling(model.decode_forward(inputs_ids=token));torch.cuda.synchronize()
 if len(records)!=32:raise RuntimeError('missing layers')
 a.output.write_text(json.dumps({'stage':'complete','model':name,'row':a.row,'multiplier':a.multiplier,'records':records},indent=2)+'\n')
if __name__=='__main__':main()

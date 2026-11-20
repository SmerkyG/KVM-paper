"""Check threshold decode against independent dense frontier on H100."""
import argparse,json,math,types
from pathlib import Path
import torch
from benchmarks.h100_threshold_decode import threshold_attention, _scan_groups

def run(seed,c,slots,oversize=False,max_size=False,rule='region',force_large=False,entry_bound=16,work_bound=None):
 if work_bound is None:work_bound=entry_bound
 torch.manual_seed(seed);dev='cuda';H,G,D=8,4,128;loc,cap=41,128;pcap=slots*4+64; leafcap=slots*36+1024
 counts=torch.randint(1,34,(H,slots),device=dev,dtype=torch.int32)
 if oversize:counts[0,0]=1025
 if max_size:counts[0,0]=1024
 lk=torch.randn((1,H,leafcap,D),device=dev,dtype=torch.bfloat16);lv=torch.randn_like(lk)
 if force_large:lk[0,0,:1025]=0.5
 pi=torch.full((1,H,pcap,16),-1,device=dev,dtype=torch.int32);roots=torch.full((1,H,slots,2),-1,device=dev,dtype=torch.int32)
 dirs=torch.full((1,H,pcap,64),-1,device=dev,dtype=torch.int32);owner=torch.full((H,leafcap),-1,device=dev,dtype=torch.int64)
 sk=torch.zeros((1,H,slots,D),device=dev,dtype=torch.bfloat16);sv=torch.zeros_like(sk)
 counts_cpu=counts.cpu().tolist()
 for h in range(H):
  pos=0;page=0
  for s in range(slots):
   n=counts_cpu[h][s];roots[0,h,s,0]=s
   if n>1024:roots[0,h,s,1]=slots+s
   for p in range(math.ceil(n/16)):
    take=min(16,n-p*16);dirs[0,h,s+(p//64)*slots,p%64]=page
    pi[0,h,page,:take]=torch.arange(pos+p*16,pos+p*16+take,device=dev,dtype=torch.int32);page+=1
   owner[h,pos:pos+n]=s;sk[0,h,s]=lk[0,h,pos:pos+n].float().sum(0).to(torch.bfloat16);sv[0,h,s]=lv[0,h,pos:pos+n].float().sum(0).to(torch.bfloat16);pos+=n
 state={'state_k':sk,'state_v':sv,'counts':counts.float()[None,:,:,None],
  'recent_k':torch.randn((1,H,cap,D),device=dev,dtype=torch.bfloat16),'recent_v':torch.randn((1,H,cap,D),device=dev,dtype=torch.bfloat16),
  'sink_k':torch.randn((1,H,1,D),device=dev,dtype=torch.bfloat16),'sink_v':torch.randn((1,H,1,D),device=dev,dtype=torch.bfloat16),
  'page_cache':{'paged_page_directory':True,'slot_pages':roots,'overflow_page_values':dirs,'page_indices':pi,'leaf_k':lk,'leaf_v':lv,'slot_lengths':counts[None]}}
 pool=types.SimpleNamespace(kv_heads=H,settings=types.SimpleNamespace(levels=2),state_capacity=slots,dtype=torch.bfloat16,state=state,device=dev,decode_local_limit=64,engine=types.SimpleNamespace(scaling=1/math.sqrt(D)),state_lens=torch.tensor([slots],device=dev,dtype=torch.int32),local_lens=torch.tensor([loc],device=dev,dtype=torch.int32))
 q=torch.randn((1,1,H*G,D),device=dev,dtype=torch.bfloat16);k=torch.randn((1,1,H,D),device=dev,dtype=torch.bfloat16);v=torch.randn_like(k); out=torch.empty_like(q)
 if force_large:q[0,0,:G]=1.
 # Independent PyTorch frontier; region means intentionally use BF16 rounding.
 Q=q[0,0].reshape(H,G,D).float(); count=counts.float();divisor=count.to(torch.bfloat16).float();mk=(sk[0].float()/divisor[:,:,None]).to(torch.bfloat16).float();mv=(sv[0].float()/divisor[:,:,None]).to(torch.bfloat16).float()
 coarse=torch.einsum('hgd,hsd->hgs',Q,mk)/math.sqrt(D)+count.log()[:,None,:]
 Qnorm=Q*torch.rsqrt(Q.square().mean(-1,keepdim=True).clamp_min(1e-12));route=torch.einsum('hgd,hsd->hgs',Qnorm,mk)/math.sqrt(D)+count.log()[:,None,:]
 probs=route.softmax(-1)
 if rule=='entry':
  selected=probs/count[:,None,:]>1/(entry_bound*count.sum(-1).sqrt()[:,None,None])
  if work_bound<entry_bound:
   cumulative=(selected*count[:,None,:]).cumsum(-1)
   if work_bound==1 and not bool((cumulative[...,-1].square()>=count.sum(-1)[:,None]).any()):raise RuntimeError('guard fallback was not exercised')
   selected=selected&(cumulative.square()<work_bound*work_bound*count.sum(-1)[:,None,None])
 else:selected=(probs>c/slots)&(count[:,None,:]<=1024)
 mask=selected.gather(-1,owner.clamp_min(0)[:,None,:].expand(-1,G,-1))&(owner[:,None,:]>=0)
 leafs=torch.einsum('hgd,htd->hgt',Q,lk[0].float())/math.sqrt(D); local_s=torch.einsum('hgd,htd->hgt',Q,state['recent_k'][0,:,:loc].float())/math.sqrt(D)
 sink_s=torch.einsum('hgd,htd->hgt',Q,state['sink_k'][0].float())/math.sqrt(D);new_s=torch.einsum('hgd,hd->hg',Q,k[0,0].float())/math.sqrt(D)
 scores=torch.cat((coarse.masked_fill(selected,-torch.inf),leafs.masked_fill(~mask,-torch.inf),local_s,sink_s,new_s[:,:,None]),-1);p=scores.softmax(-1)
 ref=(torch.einsum('hgs,hsd->hgd',p[...,:slots],mv)+torch.einsum('hgt,htd->hgd',p[...,slots:slots+leafcap],lv[0].float())+torch.einsum('hgt,htd->hgd',p[...,slots+leafcap:slots+leafcap+loc],state['recent_v'][0,:,:loc].float())+torch.einsum('hgt,htd->hgd',p[...,slots+leafcap+loc:slots+leafcap+loc+1],state['sink_v'][0].float())+p[...,-1,None]*v[0,0,:,None,:].float())
 threshold_attention(pool,q,k,v,out,c,rule=rule,entry_bound=entry_bound,work_bound=work_bound);torch.cuda.synchronize()
 dif=(out[0,0].float().reshape(H,G,D)-ref).abs();scratch=pool._threshold_scratch;got=torch.zeros_like(selected)
 if scratch.get('mask_only'):
  got=scratch['allowed'].view_as(selected).bool()
 else:
  for h in range(H*G):
   count_h=int(scratch['count'][h]);idx=scratch['selected'][h,:count_h].long();got.reshape(H*G,slots)[h,idx]=True
 mismatches=int((got!=selected).sum());maxerr=float(dif.max());meanerr=float(dif.mean())
 if mismatches or maxerr>0.02:raise RuntimeError(f'frontier mismatch {mismatches=} {maxerr=} {meanerr=}')
 entries=(selected*count[:,None,:]).sum(-1);bound=work_bound*count.sum(-1).sqrt()[:,None]
 if rule=='entry' and not bool(torch.all(entries<bound)):raise RuntimeError('opened entries violate the per-head bound')
 if force_large and not bool(selected[0,:,0].all()):raise RuntimeError('large region was not opened')
 if int(pool.local_lens.item())!=loc+1 or not torch.equal(state['recent_k'][0,:,loc],k[0,0]) or not torch.equal(state['recent_v'][0,:,loc],v[0,0]):raise RuntimeError('local append mismatch')
 return {'seed':seed,'multiplier':c,'rule':rule,'entry_bound':entry_bound,'work_bound':work_bound,'slots':slots,'oversize':oversize,'max_size':max_size,'force_large':force_large,'selection_mismatches':mismatches,'max_abs':maxerr,'mean_abs':meanerr,'mean_selected':float(selected.sum(-1).float().mean()),'max_opened_entries':int(entries.max()),'max_bound_fraction':float((entries/bound).max())}

def scan_check():
 for n in (1,3,31,32,33,77,513):
  counts=torch.randint(0,65,(32,n),device='cuda',dtype=torch.int32);pages=torch.randint(0,128,(32,n),device='cuda',dtype=torch.int32)
  offsets=torch.empty_like(counts);page_offsets=torch.empty_like(pages);total=torch.empty(32,device='cuda',dtype=torch.int32);total_pages=torch.empty_like(total)
  _scan_groups[(32,)](counts,pages,offsets,page_offsets,total,total_pages,n,num_warps=1)
  if not torch.equal(offsets,counts.cumsum(1).int()-counts) or not torch.equal(page_offsets,pages.cumsum(1).int()-pages) or not torch.equal(total,counts.sum(1).int()) or not torch.equal(total_pages,pages.sum(1).int()):raise RuntimeError(f'group scan mismatch {n}')
 return [1,3,31,32,33,77,513]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_grad_enabled(False)
 scan_sizes=scan_check()
 cases=[(2,0.1,67,False,False),(3,4,67,False,False),(4,16,127,False,False),(5,32,127,False,False),(6,0.1,67,True,False),(7,0.1,129,False,True),(8,16,127,False,False,'entry'),(9,16,256,True,False,'entry',True),(10,16,127,False,False,'entry',False,24),(11,16,127,False,False,'entry',False,32),(11,16,127,False,False,'entry',False,32,1)]
 rows=[run(*case) for case in cases]
 a.output.write_text(json.dumps({'stage':'complete','scan_sizes':scan_sizes,'records':rows},indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()

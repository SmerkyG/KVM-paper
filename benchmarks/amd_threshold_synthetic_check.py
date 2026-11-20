"""Check AMD threshold decode against an independent PyTorch frontier."""
import argparse,json,math,os,types
from pathlib import Path
import torch
from benchmarks.amd_threshold_decode import threshold_attention

def run(seed,c,slots,oversize=False,max_size=False,dimension=128,group=8,union=False,rule='region',force_large=False,entry_bound=16,work_bound=None):
 if work_bound is None:work_bound=entry_bound
 torch.manual_seed(seed);dev='cuda';H,G,D=2,group,dimension;loc,cap=41,128;pcap=slots*4+64; leafcap=slots*36+1024
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
 # Independent PyTorch frontier; match the AMD route's FP32 count divisor.
 Q=q[0,0].reshape(H,G,D).float(); count=counts.float();mk=(sk[0].float()/count[:,:,None]).to(torch.bfloat16).float();mv=(sv[0].float()/count[:,:,None]).to(torch.bfloat16).float()
 coarse=torch.einsum('hgd,hsd->hgs',Q,mk)/math.sqrt(D)+count.log()[:,None,:]
 Qnorm=Q*torch.rsqrt(Q.square().mean(-1,keepdim=True).clamp_min(1e-12));route=torch.einsum('hgd,hsd->hgs',Qnorm,mk)/math.sqrt(D)+count.log()[:,None,:]
 if rule=='entry':
  selected=route.softmax(-1)/count[:,None,:]>1/(entry_bound*count.sum(-1).sqrt()[:,None,None])
  prefix=(selected.to(torch.int64)*counts[:,None,:]).cumsum(-1)
  selected&=prefix.square()<work_bound*work_bound*counts.sum(-1)[:,None,None]
 else:selected=(route.softmax(-1)>c/slots)&(count[:,None,:]<=1024)
 mask=selected.gather(-1,owner.clamp_min(0)[:,None,:].expand(-1,G,-1))&(owner[:,None,:]>=0)
 leafs=torch.einsum('hgd,htd->hgt',Q,lk[0].float())/math.sqrt(D); local_s=torch.einsum('hgd,htd->hgt',Q,state['recent_k'][0,:,:loc].float())/math.sqrt(D)
 sink_s=torch.einsum('hgd,htd->hgt',Q,state['sink_k'][0].float())/math.sqrt(D);new_s=torch.einsum('hgd,hd->hg',Q,k[0,0].float())/math.sqrt(D)
 scores=torch.cat((coarse.masked_fill(selected,-torch.inf),leafs.masked_fill(~mask,-torch.inf),local_s,sink_s,new_s[:,:,None]),-1);p=scores.softmax(-1)
 ref=(torch.einsum('hgs,hsd->hgd',p[...,:slots],mv)+torch.einsum('hgt,htd->hgd',p[...,slots:slots+leafcap],lv[0].float())+torch.einsum('hgt,htd->hgd',p[...,slots+leafcap:slots+leafcap+loc],state['recent_v'][0,:,:loc].float())+torch.einsum('hgt,htd->hgd',p[...,slots+leafcap+loc:slots+leafcap+loc+1],state['sink_v'][0].float())+p[...,-1,None]*v[0,0,:,None,:].float())
 if union:
  from benchmarks.amd_threshold_union import threshold_attention as invoke
 else:invoke=threshold_attention
 config={'tile':int(os.environ.get('LOD_AMD_THRESHOLD_TILE','64')),'splits':int(os.environ.get('LOD_AMD_THRESHOLD_SPLITS','64'))}
 invoke(pool,q,k,v,out,c,**config,**({'rule':rule,'entry_bound':entry_bound,'work_bound':work_bound} if not union else {}));torch.cuda.synchronize()
 dif=(out[0,0].float().reshape(H,G,D)-ref).abs();got=torch.zeros_like(selected)
 if union:
  scratch=pool._threshold_union_scratch
  for h in range(H):
   n=int(scratch['count'][h]);idx=scratch['selected'][h,:n].long();bits=scratch['selected_masks'][h,:n]
   for g in range(G):got[h,g,idx]=(bits&(1<<g))!=0
 else:
  scratch=pool._threshold_scratch
  from benchmarks.amd_threshold_decode import selected_region_mask
  got=selected_region_mask(pool).reshape(H,G,slots)
 mismatches=int((got!=selected).sum());maxerr=float(dif.max());meanerr=float(dif.mean())
 if mismatches or maxerr>0.02:raise RuntimeError(f'frontier mismatch {mismatches=} {maxerr=} {meanerr=}')
 entries=(selected*count[:,None,:]).sum(-1);bound=work_bound*count.sum(-1).sqrt()[:,None]
 if rule=='entry' and not bool((entries<bound).all()):raise RuntimeError('work bound violated')
 if force_large and not bool(selected[0,:,0].all()):raise RuntimeError('large region not opened')
 if int(pool.local_lens.item())!=loc+1 or not torch.equal(state['recent_k'][0,:,loc],k[0,0]) or not torch.equal(state['recent_v'][0,:,loc],v[0,0]):raise RuntimeError('local append mismatch')
 if seed in (2,3,4):
  expected = out.clone()
  for _ in range(32):
   pool.local_lens.fill_(loc)
   invoke(pool,q,k,v,out,c,**config,**({'rule':rule,'entry_bound':entry_bound,'work_bound':work_bound} if not union else {}))
   if not torch.equal(out,expected):raise RuntimeError('fixed-cache threshold output changed across repeated steps')
 return {'seed':seed,'multiplier':c,'rule':rule,'entry_bound':entry_bound,'work_bound':work_bound,'slots':slots,'oversize':oversize,'max_size':max_size,'dimension':D,'group':G,'selection_mismatches':mismatches,'max_abs':maxerr,'mean_abs':meanerr,'mean_selected':float(selected.sum(-1).float().mean())}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--union',action='store_true');a=p.parse_args();torch.set_grad_enabled(False)
 cases=[(2,0.1,67,False,False),(3,4,67,False,False),(4,16,127,False,False),(5,32,127,False,False),(6,0.1,67,True,False),(7,0.1,129,False,True)]
 rows=[run(*case,dimension=dimension,group=group,union=a.union) for dimension,group in ((128,8),(256,6)) for case in cases]
 if not a.union and os.environ.get('LOD_AMD_THRESHOLD_IMPLEMENTATION') not in ('frontier','flat','tiled','ranked','cached','balanced','active','shared','materialized','table'):
  rows += [run(8,16,127,dimension=dimension,group=group,rule='entry') for dimension,group in ((128,8),(256,6))]
  rows += [run(9,16,256,oversize=True,dimension=dimension,group=group,rule='entry',force_large=True) for dimension,group in ((128,8),(256,6))]
  rows += [run(11,16,127,dimension=dimension,group=group,rule='entry',entry_bound=32,work_bound=1) for dimension,group in ((128,8),(256,6))]
 from benchmarks._identity import benchmark_identity
 a.output.write_text(json.dumps({'stage':'complete','implementation':os.environ.get('LOD_AMD_THRESHOLD_IMPLEMENTATION','legacy'),'hardware':str(torch.cuda.get_device_properties(0)),'benchmark_identity':benchmark_identity(),'records':rows},indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()

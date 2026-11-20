"""Check fused route merging against PyTorch, including cancellation and >2^31 indexing."""
import argparse
import json
from pathlib import Path
import torch
import triton


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks.h100_threshold_prefill_merge import merge_routes,repair_routes
    torch.set_grad_enabled(False);torch.manual_seed(2719)
    h,kv,g,t,s,d,w=32,8,4,193,73,128,32
    q=torch.randn((1,h,t,d),device='cuda',dtype=torch.bfloat16)
    v=torch.randn((1,kv,s,d),device='cuda',dtype=q.dtype)
    k=torch.zeros_like(v);size=torch.randint(1,17,(1,kv,s,1),device='cuda',dtype=torch.int32)
    raw=torch.randn((h,t,s),device='cuda')
    counts=torch.tensor([0,1,5,31],device='cuda',dtype=torch.int32).repeat((h*t+3)//4)[:h*t].view(h,t)
    slots=torch.arange(w,device='cuda',dtype=torch.int64).view(1,1,1,w).expand(1,h,t,w).contiguous()
    raw[:,:,0]=torch.where((torch.arange(t,device='cuda')[None,:]%3==0)&(counts>0),60.,raw[:,:,0])
    valid=torch.arange(w,device='cuda')[None,None,:]<counts[:,:,None]
    sizes=size[0,:,:,0].repeat_interleave(g,0).float()
    logits=raw+sizes[:,None,:].log()
    fl=torch.logsumexp(logits,-1);full=torch.matmul(logits.softmax(-1),v[0].repeat_interleave(g,0).float())
    exact_lse=(logits[:,:,:w]+torch.randn_like(logits[:,:,:w])*.2).masked_fill(~valid,-torch.inf)
    exact=torch.randn((h,t,w,d),device='cuda',dtype=q.dtype)
    lo=torch.randn_like(q);ll=torch.randn((1,h,t),device='cuda')
    sk=torch.randn((1,kv,2,d),device='cuda',dtype=q.dtype);sv=torch.randn_like(sk)
    closed=torch.zeros((h,t,s),device='cuda',dtype=torch.bool)
    closed[:,:,:w]=valid
    closed_lse=torch.logsumexp(logits.masked_fill(closed,-torch.inf),-1)
    sl=torch.matmul(q[0].float(),sk[0].repeat_interleave(g,0).float().transpose(-1,-2))*d**-.5
    maximum=torch.maximum(torch.maximum(closed_lse,exact_lse.max(-1).values),torch.maximum(ll[0],sl.max(-1).values))
    cw=torch.exp(logits.masked_fill(closed,-torch.inf)-maximum[:,:,None])
    ew=torch.exp(exact_lse-maximum[:,:,None]);lw=torch.exp(ll[0]-maximum);sw=torch.exp(sl-maximum[:,:,None])
    num=torch.matmul(cw,v[0].repeat_interleave(g,0).float())+(ew[:,:,:,None]*exact.float()).sum(2)
    num+=lw[:,:,None]*lo[0].float()+torch.matmul(sw,sv[0].repeat_interleave(g,0).float())
    den=cw.sum(-1)+ew.sum(-1)+lw+sw.sum(-1)
    reference=num/den[:,:,None]
    bits=torch.zeros((h,t,triton.cdiv(s,8)),device='cuda',dtype=torch.uint8)
    for i in range(w):bits[:,:,i//8]|=(valid[:,:,i].to(torch.uint8)<<(i%8))
    records=[]
    for m in (1,2,4):
        out=torch.empty_like(q);flags=torch.empty((h,t),device='cuda',dtype=torch.uint8)
        geom=(t,s,s,g,d,w,d**-.5,2,lo.stride(1),lo.stride(2),ll.stride(1),ll.stride(2),out.stride(1),out.stride(2))
        merge_routes[(h,triton.cdiv(t,m))](q,raw,v,size,slots,counts,full,fl,exact,exact_lse,lo,ll,sk,sv,out,flags,
            *geom,m,16,w,.0001,num_warps=4)
        fallbacks=int(flags.sum())
        assert fallbacks>0,'fixture must exercise cancellation repair'
        repair_routes[(triton.cdiv(h*t,128),)](q,raw,k,v,size,slots,counts,full,fl,exact,exact_lse,lo,ll,sk,sv,out,flags,bits,
            *geom,.0001,h*t,num_warps=4)
        error=float((out[0].float()-reference).abs().max())
        assert torch.isfinite(out).all() and error<=.04,error
        records.append(dict(query_tile=m,max_abs_error=error,repaired_queries=fallbacks))
    del exact,exact_lse
    # A sparse tail query addresses a route value beyond signed 32-bit range.
    t,w=4096,256;rows=h*t
    q=torch.zeros((1,h,t,d),device='cuda',dtype=torch.bfloat16)
    raw=torch.zeros((h,t,s),device='cuda');raw[-1,-1,0]=60.
    v=torch.zeros((1,kv,s,d),device='cuda',dtype=q.dtype);v[:,-1,0]=1
    counts=torch.zeros((h,t),device='cuda',dtype=torch.int32);counts[-1,-1]=1
    slots=torch.zeros((1,h,t,w),device='cuda',dtype=torch.int64)
    size=torch.ones((1,kv,s,1),device='cuda',dtype=torch.int32)
    full=torch.zeros((1,h,t,d),device='cuda');full[:,-1,-1]=1
    fl=torch.zeros((h,t),device='cuda');fl[-1,-1]=60.
    exact=torch.empty((rows*w,d),device='cuda',dtype=q.dtype)
    exact[(rows-1)*w]=1
    exact_lse=torch.full((rows*w,),-torch.inf,device='cuda');exact_lse[(rows-1)*w]=60.
    lo=torch.zeros_like(q);ll=torch.zeros((1,h,t),device='cuda')
    sk=torch.zeros((1,kv,2,d),device='cuda',dtype=q.dtype);sv=torch.zeros_like(sk)
    out=torch.zeros_like(q);flags=torch.zeros((h,t),device='cuda',dtype=torch.uint8);flags[-1,-1]=1
    bits=torch.zeros((h,t,triton.cdiv(s,8)),device='cuda',dtype=torch.uint8);bits[-1,-1,0]=1
    repair_routes[(triton.cdiv(rows,128),)](q,raw,v,v,size,slots,counts,full,fl,exact,exact_lse,lo,ll,sk,sv,out,flags,bits,
        t,s,s,g,d,w,d**-.5,2,lo.stride(1),lo.stride(2),ll.stride(1),ll.stride(2),out.stride(1),out.stride(2),.0001,rows,num_warps=4)
    error=float((out[0,-1,-1].float()-1.).abs().max());assert error==0,error
    records.append(dict(large_route_element_offset=(rows-1)*w*d,max_abs_error=error))
    a.output.write_text(json.dumps(dict(stage='complete',cases=records),indent=2)+'\n');print(records)


if __name__=='__main__':main()

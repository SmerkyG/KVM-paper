"""Compare region mean routing with QUEST-style key bounds on fixed LoD queries.

Analysis only: region bounds are recomputed from all stored BF16 keys in FP32,
not maintained by a running implementation. The original LoD model trajectory
and first-divergence positions are replayed unchanged.
"""
import math
import torch
import benchmarks.retro_ruler_frontier_audit as runner


def region_probe(pool,q,optimized,gold_positions,context):
    from lod_attention.pytorch_engine import PytorchLODAttention
    ref,_,_,_=runner.reconstruct(pool,context)
    Q=q.transpose(1,2);H,G,D=pool.kv_heads,pool.query_heads//pool.kv_heads,pool.head_dim
    query=Q[0,:,0].reshape(H,G,D).float(); scale=1/math.sqrt(D)
    leaf_k=ref.archive_key[0].float(); owner=ref.owner[0]; H,T,_=leaf_k.shape
    n=ref.state.size; count=ref.state.count[0].float()
    kindex=owner[...,None].expand(-1,-1,D)
    shape=(H,n,D)
    minimum=torch.full(shape,float('inf'),device=q.device).scatter_reduce_(1,kindex,leaf_k,reduce='amin',include_self=True)
    maximum=torch.full(shape,float('-inf'),device=q.device).scatter_reduce_(1,kindex,leaf_k,reduce='amax',include_self=True)
    valid=count>0
    minimum=torch.where(valid[...,None],minimum,0.0)
    maximum=torch.where(valid[...,None],maximum,0.0)
    upper=(torch.einsum('hgd,hsd->hgs',query.clamp_min(0),maximum)+
           torch.einsum('hgd,hsd->hgs',query.clamp_max(0),minimum))*scale
    upper=upper.masked_fill(~valid[:,None,:],-torch.inf)
    leaf_logits=torch.einsum('hgd,htd->hgt',query,leaf_k)*scale
    nlocal=int(pool.local_lens[0].item())
    local_logits=torch.einsum('hgd,htd->hgt',query,pool.state['recent_k'][0,:,:nlocal].float())*scale
    sink_logits=torch.einsum('hgd,htd->hgt',query,ref.sink_key[0].float())*scale
    all_logits=torch.cat((leaf_logits,local_logits,sink_logits),-1)
    leaf_p=torch.softmax(all_logits,-1)[...,:T]
    owner_expanded=owner[:,None,:].expand(-1,G,-1)
    true_region_logits=torch.full((H,G,n),float('-inf'),device=q.device).scatter_reduce_(
        -1,owner_expanded,leaf_logits,reduce='amax',include_self=True)
    if torch.any((upper-true_region_logits).masked_fill(~valid[:,None,:],0)<-0.0002):
        raise RuntimeError('channel bound below true max score')
    true_region_mass=torch.zeros((H,G,n),device=q.device).scatter_add_(-1,owner_expanded,leaf_p)
    sink_len=int(ref.sink_key.size(2))
    gold=[x-sink_len for x in gold_positions if sink_len<=x<sink_len+T]
    if len(gold)!=len(gold_positions):raise RuntimeError('answer not wholly remote')
    ix=torch.tensor(gold,device=q.device,dtype=torch.long)
    gold_p=leaf_p[...,ix];gold_w=torch.zeros_like(leaf_p);gold_w[...,ix]=gold_p
    gold_region_mass=torch.zeros_like(true_region_mass).scatter_add_(-1,owner_expanded,gold_w)
    top_gold_pos=ix[gold_p.argmax(-1)]
    top_gold_slot=owner_expanded.gather(-1,top_gold_pos[...,None]).squeeze(-1)
    nrm=query*torch.rsqrt(query.square().mean(-1,keepdim=True).clamp_min(1e-12))
    mean=ref.state.mean_key[0].float()
    route=(torch.einsum('hgd,hsd->hgs',nrm,mean)*scale + count.clamp_min(1).log()[:,None,:]).masked_fill(~valid[:,None,:],-torch.inf)
    native=(torch.einsum('hgd,hsd->hgs',query,mean)*scale + count.clamp_min(1).log()[:,None,:]).masked_fill(~valid[:,None,:],-torch.inf)
    upper_mass=upper+count.clamp_min(1).log()[:,None,:]
    def ranks(x):return (x>x.gather(-1,top_gold_slot[...,None])).sum(-1)+1
    def selection(x):
        idx=x.topk(8,dim=-1).indices
        allowed=count[:,None,:].expand(-1,G,-1).gather(-1,idx)<=1024
        gathered=true_region_mass.gather(-1,idx)*allowed
        ggold=gold_region_mass.gather(-1,idx)*allowed
        # A dense-mass threshold measures wasted opened regions on this query.
        tiny=((gathered<1e-4)&allowed).sum(-1)
        return gathered.sum(-1),ggold.sum(-1),tiny
    base_mass,base_gold,base_tiny=selection(route)
    bound_mass,bound_gold,bound_tiny=selection(upper)
    ubmass_mass,ubmass_gold,ubmass_tiny=selection(upper_mass)
    rmax=true_region_logits.gather(-1,top_gold_slot[...,None]).squeeze(-1)
    total_logits=torch.zeros_like(true_region_logits).scatter_add_(-1,owner_expanded,leaf_logits)
    total_squares=torch.zeros_like(true_region_logits).scatter_add_(-1,owner_expanded,leaf_logits.square())
    all_count=count[:,None,:].expand(-1,G,-1).clamp_min(1)
    true_mean=total_logits/all_count
    true_std=(total_squares/all_count-true_mean.square()).clamp_min(0).sqrt()
    gold_mean_true=true_mean.gather(-1,top_gold_slot[...,None]).squeeze(-1)
    gold_std=true_std.gather(-1,top_gold_slot[...,None]).squeeze(-1)
    gold_min_all=torch.full_like(true_region_logits,float('inf')).scatter_reduce_(
        -1,owner_expanded,leaf_logits,reduce='amin',include_self=True)
    gold_min=gold_min_all.gather(-1,top_gold_slot[...,None]).squeeze(-1)
    query_length=query.norm(dim=-1)
    gold_key=leaf_k[:,None,:,:].expand(-1,G,-1,-1).gather(-2,top_gold_pos[...,None,None].expand(-1,-1,1,D)).squeeze(-2)
    region_key=mean[:,None,:,:].expand(-1,G,-1,-1).gather(-2,top_gold_slot[...,None,None].expand(-1,-1,1,D)).squeeze(-2)
    gold_key_norm=gold_key.norm(dim=-1);region_key_norm=region_key.norm(dim=-1)
    gold_key_to_centroid_cos=(gold_key*region_key).sum(-1)/(gold_key_norm*region_key_norm).clamp_min(1e-12)
    b=upper.gather(-1,top_gold_slot[...,None]).squeeze(-1)
    mean_logits=torch.einsum('hgd,hsd->hgs',query,mean)*scale
    mean_gold=mean_logits.gather(-1,top_gold_slot[...,None]).squeeze(-1)
    winning_score=leaf_logits.gather(-1,top_gold_pos[...,None]).squeeze(-1)
    size=count[:,None,:].expand(-1,G,-1).gather(-1,top_gold_slot[...,None]).squeeze(-1)
    true_mass=true_region_mass.gather(-1,top_gold_slot[...,None]).squeeze(-1)
    gold_total=gold_p.sum(-1)
    # Verify the reference route and output are consistent with this kernel.
    refengine=PytorchLODAttention(mode='two-tier')
    refout=refengine._lod_attention(Q,pool.state['recent_k'][:,:,:nlocal],pool.state['recent_v'][:,:,:nlocal],
        ref,scale=scale,query_offset=nlocal-1,full_regions=True).output.transpose(1,2)
    diff=(optimized.float()-refout.float()).abs()
    def items(x):return x.reshape(-1).tolist()
    return {'reference_vs_optimized_max_abs':float(diff.max()),
      'reference_vs_optimized_mean_abs':float(diff.mean()), 'centroids':n,
      'gold_mass':items(gold_total),'gold_region_true_mass':items(true_mass),
      'gold_token_mass':items(gold_p.max(-1).values),'gold_region_size':items(size),
      'gold_token_logit':items(winning_score),'gold_region_mean_logit':items(mean_gold),
      'gold_region_exact_mean_logit':items(gold_mean_true),'gold_region_logit_std':items(gold_std),
      'gold_region_min_logit':items(gold_min),'gold_key_norm':items(gold_key_norm),
      'gold_centroid_norm':items(region_key_norm),'query_norm':items(query_length),
      'gold_key_centroid_cos':items(gold_key_to_centroid_cos),
      'gold_region_max_logit':items(rmax),'gold_region_upper_logit':items(b),
      'gold_rank_current':items(ranks(route)),'gold_rank_native':items(ranks(native)),
      'gold_rank_bound':items(ranks(upper)),'gold_rank_bound_plus_log_count':items(ranks(upper_mass)),
      'current_true_mass_captured':items(base_mass),'bound_true_mass_captured':items(bound_mass),
      'bound_plus_count_true_mass_captured':items(ubmass_mass),
      'current_gold_mass_captured':items(base_gold),'bound_gold_mass_captured':items(bound_gold),
      'bound_plus_count_gold_mass_captured':items(ubmass_gold),
      'current_low_mass_regions':items(base_tiny),'bound_low_mass_regions':items(bound_tiny),
      'bound_plus_count_low_mass_regions':items(ubmass_tiny)}


if __name__=='__main__':
    runner.audit=region_probe
    runner.main()

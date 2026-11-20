"""AMD two-tier BF16 threshold decode with fused packing and compact exact work.

Uses the tested H100 compact traversal, but retains AMD's native score producer,
FP32 count division, coarse attention and append logic. Kernel launch settings
are tunable independently. This experiment leaves serving defaults unchanged.

Enable with LOD_AMD_THRESHOLD_MULTIPLIER=16 and
LOD_AMD_THRESHOLD_IMPLEMENTATION=indexed. The tested MI325X region-rule
settings use 64 splits, ENTRY_BLOCK=32 and ENTRY_WARPS=2 (all prefixed with
LOD_AMD_THRESHOLD_). See results/amd-threshold-fast/REPORT.md for scope.
"""
import os
import torch
import triton
from benchmarks.h100_threshold_decode_fast import _threshold_pack, _compact_leaves
from benchmarks.amd_threshold_decode import _scores, _coarse_select_gqa, _expand_pages, _leaves, _local, _merge_append


def threshold_attention_fast(pool,q,k,v,out,multiplier,*,tile=64,splits=64,rule='region',
                             entry_bound=16,work_bound=None,implementation='combined'):
    if implementation not in ('packed','indexed','combined','union','union-combined'):
        raise ValueError('unknown optimized threshold decode implementation')
    if work_bound is None:work_bound=entry_bound
    if rule not in ('region','entry') or multiplier<=0 or not isinstance(entry_bound,int) or entry_bound<1 or not isinstance(work_bound,int) or work_bound<1:
        raise ValueError('invalid threshold configuration')
    if q.ndim!=4 or q.shape[:2]!=(1,1) or pool.settings.levels!=2 or pool.dtype!=torch.bfloat16:
        raise ValueError('optimized AMD threshold decode requires batch-one two-tier BF16')
    page=pool.state['page_cache']
    if not page.get('paged_page_directory'):raise ValueError('requires directory cache')
    S=pool.state_capacity;H=int(q.size(2));KV=pool.kv_heads;G=H//KV;D=int(q.size(-1))
    if H!=KV*G or (D,G) not in ((128,8),(256,6)):
        raise ValueError("optimized AMD threshold geometry must be K2 or Qwen3.8")
    localcap=int(pool.state['recent_k'].size(2));limit=int(pool.decode_local_limit);sink=int(pool.state['sink_k'].size(2))
    groups=triton.cdiv(S,tile);combined=implementation in ('combined','union-combined');union=implementation.startswith('union')
    localgroups=0 if combined else triton.cdiv(limit+1+sink,tile)
    spec=('fast',S,tile,splits,localgroups,union)
    b=getattr(pool,'_threshold_fast_scratch',None)
    if b is None or b['spec']!=spec:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        b=dict(spec=spec,native=f(H,S),route=f(H,S),threshold=f(H),allowed=i(H,S),
               selected=i(H,S),count=i(H),offsets=i(H,S),entry_ends=i(H,S),total_entries=i(H),
               total_pages=i(H),saved=i(1),co=f(H,groups,D),cl=f(H,groups),eo=f(H,splits,D),el=f(H,splits),
               lo=f(H,max(1,localgroups),D),ll=f(H,max(1,localgroups)))
        if implementation=='packed':b['descriptors']=i(H,int(page['page_indices'].size(2)))
        if union:b.update(union_selected=i(KV,S),union_masks=i(KV,S),union_ends=i(KV,S),union_count=i(KV),union_pages=i(KV))
        pool._threshold_fast_scratch=b
    elif implementation=='packed' and 'descriptors' not in b:
        b['descriptors']=torch.empty((H,int(page['page_indices'].size(2))),device=q.device,dtype=torch.int32)
    pool._threshold_scratch=b
    st=pool.state;scale=float(pool.engine.scaling)
    profiling=os.environ.get('LOD_AMD_DECODE_PROFILE')=='1'
    if profiling:
        first=torch.cuda.Event(enable_timing=True);first.record()
    def mark(name):
        nonlocal first
        if profiling:
            last=torch.cuda.Event(enable_timing=True);last.record()
            pool._threshold_decode_events.setdefault(name,[]).append((first,last));first=last
    if profiling and not hasattr(pool,'_threshold_decode_events'):pool._threshold_decode_events={}
    _scores[(KV,groups)](q,st['state_k'],st['counts'],pool.state_lens,pool.local_lens,b['native'],b['route'],b['saved'],S,KV,G,D,scale,tile,num_warps=4)
    mark('scores')
    _threshold_pack[(H,)](b['route'],page['slot_lengths'],st['counts'],b['threshold'],b['allowed'],b['selected'],b['count'],
        b['offsets'],b['total_pages'],b['entry_ends'],b['total_entries'],S,G,float(multiplier),rule=='entry',
        entry_bound,work_bound,triton.next_power_of_2(S),implementation=='packed',num_warps=4)
    mark('select_pack')
    _coarse_select_gqa[(KV,groups)](b['native'],b['route'],b['threshold'],b['allowed'],st['counts'],st['state_v'],
        page['slot_lengths'],b['selected'],b['offsets'],b['count'],b['total_pages'],b['co'],b['cl'],
        S,G,D,tile,groups,1024,True,False,union,num_warps=4)
    mark('coarse')
    pagecap=int(page['page_indices'].size(2));roots=int(page['slot_pages'].size(3))
    if union:
        from benchmarks.amd_threshold_compact_union import union_scan,union_leaves
        union_scan[(KV,)](b['offsets'],page['slot_lengths'],b['union_selected'],b['union_masks'],b['union_ends'],
                         b['union_count'],b['union_pages'],S,triton.next_power_of_2(S),num_warps=4)
        mark('union_pack')
        union_leaves[(KV,splits)](q,b['union_selected'],b['union_masks'],b['union_ends'],b['union_count'],b['union_pages'],
            page['slot_lengths'],page['slot_pages'],page['overflow_page_values'],page['page_indices'],page['leaf_k'],page['leaf_v'],
            st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,b['saved'],b['eo'],b['el'],
            S,G,D,roots,pagecap,int(page['leaf_k'].size(2)),splits,scale,int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','64')),
            localcap,limit,sink,combined,num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','4')))
    elif implementation=='packed':
        _expand_pages[(H,splits)](b['selected'],b['count'],page['slot_lengths'],page['slot_pages'],page['overflow_page_values'],
            b['offsets'],b['descriptors'],S,G,roots,pagecap,splits,num_warps=2)
        _leaves[(H,splits)](q,b['descriptors'],b['total_pages'],page['page_indices'],page['leaf_k'],page['leaf_v'],b['eo'],b['el'],
            KV,G,D,pagecap,int(page['leaf_k'].size(2)),scale,splits,num_warps=4)
    else:
        block=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','32'))
        _compact_leaves[(H,splits)](q,b['selected'],b['count'],b['entry_ends'],b['total_entries'],page['slot_pages'],
            page['overflow_page_values'],page['page_indices'],page['leaf_k'],page['leaf_v'],st['recent_k'],st['recent_v'],
            st['sink_k'],st['sink_v'],k,v,b['saved'],b['eo'],b['el'],S,G,D,roots,pagecap,int(page['leaf_k'].size(2)),
            splits,scale,block,localcap,limit,sink,combined,num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','4')))
    mark('exact')
    if not combined:
        _local[(H,localgroups)](q,st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,b['saved'],b['lo'],b['ll'],
            KV,G,D,localcap,limit,sink,scale,tile,localgroups,num_warps=4)
    mark('local')
    merge_block=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK','0'))
    args=(b['co'],b['cl'],b['eo'],b['el'],b['lo'],b['ll'],k,v,st['recent_k'],st['recent_v'],pool.local_lens,
          b['saved'],out,D,G,KV,localcap,groups,splits,localgroups,triton.next_power_of_2(groups+splits+localgroups))
    if merge_block:
        from benchmarks.amd_threshold_merge import merge_split_d
        merge_split_d[(H,triton.cdiv(D,merge_block))](*args,merge_block,num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_WARPS','4')))
    else:_merge_append[(H,)](*args,num_warps=4)
    mark('merge_append')
    return out
